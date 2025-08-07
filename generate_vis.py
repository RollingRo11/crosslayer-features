#!/usr/bin/env python3
"""
Direct generator for crosscoder visualization.
"""

import os
import sys
import torch
import json
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
import einops
import argparse
import random
from sophia import SophiaG

# Add local sae_vis to path FIRST
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sae_vis'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import crosscoder
from crosscoder.crosscoder import Crosscoder

# Import NNsight
import nnsight
from nnsight import LanguageModel

# Import datasets
from datasets import load_dataset

# Direct imports from local files
from sae_vis.data_config_classes import CrosscoderVisConfig, CrosscoderVisLayoutConfig
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_storing_fns import CrosscoderVisData


def load_latest_checkpoint(device=None):
    """Load the most recent crosscoder checkpoint."""
    saves_dir = Path("crosscoder/saves")

    # Find all checkpoints
    checkpoints = list(saves_dir.glob("version_*/[0-9]*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {saves_dir}")

    # Sort by version and checkpoint number
    def sort_key(path):
        version = int(path.parent.name.split('_')[1])
        checkpoint = int(path.stem)
        return (version, checkpoint)

    checkpoints.sort(key=sort_key)
    latest = checkpoints[-1]

    print(f"Loading checkpoint: {latest}")

    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load checkpoint with appropriate device mapping
    map_location = device if device == "cpu" else None
    checkpoint = torch.load(latest, map_location=map_location)

    # Load config
    config_file = latest.parent / f"{latest.stem}_cfg.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        # Convert dtype string
        if 'dtype' in cfg and isinstance(cfg['dtype'], str):
            cfg['dtype'] = getattr(torch, cfg['dtype'].split('.')[-1])
    else:
        # Default config
        cfg = {
            "model_name": "gpt2",
            "ae_dim": 4096,
            "device": device,
            "dtype": torch.float32,
            "drop_bos": True,
        }

    # Use detected/specified device
    cfg["device"] = device

    # Create crosscoder
    crosscoder = Crosscoder(cfg)

    # Load weights to the correct device
    crosscoder.W_enc.data = checkpoint['W_enc'].to(device)
    crosscoder.W_dec.data = checkpoint['W_dec'].to(device)
    crosscoder.b_enc.data = checkpoint['b_enc'].to(device)
    crosscoder.b_dec.data = checkpoint['b_dec'].to(device)

    # Convert crosscoder to float32 to avoid dtype mismatches with model activations
    crosscoder = crosscoder.to(torch.float32)

    crosscoder.eval()
    return crosscoder, cfg


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate crosscoder visualization dashboard")

    parser.add_argument(
        "--num_features",
        type=int,
        default=3,
        help="Number of features to visualize (default: 3)"
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Select features randomly instead of in numerical order"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for token processing (default: 8)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="crosscoder_dashboard_new.html",
        help="Output filename for the dashboard (default: crosscoder_dashboard_new.html)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print("Starting crosscoder visualization generation...")
    print(f"Configuration: {args.num_features} features, {'random' if args.random else 'sequential'} selection, batch_size={args.batch_size}")

    # Auto-detect device for consistent usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Auto-detected device: {device}")

    # Load crosscoder with consistent device
    crosscoder, crosscoder_cfg = load_latest_checkpoint(device=device)
    print(f"Loaded crosscoder with {crosscoder.ae_dim} features")

    # Load model - use the same model that crosscoder was trained on
    model_name = crosscoder_cfg.get("model_name", "gpt2")
    print(f"Loading model: {model_name}")

    # Map model names to actual model identifiers
    model_name_mapping = {
        "gpt2": "gpt2",
        "pythia": "EleutherAI/pythia-2.8b-deduped",
        "pythia7b": "EleutherAI/pythia-6.9b-deduped",
        "pythia160m": "EleutherAI/pythia-160m-deduped",
        "pythia410m": "EleutherAI/pythia-410m-deduped",
        "pythia1b": "EleutherAI/pythia-1b-deduped",
        "pythia1.4b": "EleutherAI/pythia-1.4b-deduped",
        "pythia2.8b": "EleutherAI/pythia-2.8b-deduped",
    }

    actual_model_name = model_name_mapping.get(model_name, model_name)
    # Use consistent device mapping
    if device == "cuda":
        model = nnsight.LanguageModel(actual_model_name, device_map="cuda")
    else:
        model = nnsight.LanguageModel(actual_model_name, device_map="cpu")

    # Verify model dimensions match crosscoder expectations
    from sae_vis.model_utils import get_hidden_size
    model_hidden_size = get_hidden_size(model)
    expected_hidden_size = crosscoder.W_enc.shape[1]  # d_model dimension

    if model_hidden_size != expected_hidden_size:
        raise ValueError(
            f"Model dimension mismatch: model has {model_hidden_size} dimensions "
            f"but crosscoder expects {expected_hidden_size} dimensions. "
            f"Make sure you're using the same model that the crosscoder was trained on."
        )

    print(f"Model dimension verification passed: {model_hidden_size}")

    # Load some data - using Alpaca dataset for more diverse content
    print("Loading data...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

    # Tokenize sequences - use longer sequences for proper windowing
    tokens = []
    max_sequences = 50  # More sequences for better sampling
    seq_len = 512  # Much longer sequences to allow windowing

    # Simplified collection - just take first N sequences that meet length requirement
    all_examples = []

    for i, example in enumerate(dataset):
        if len(all_examples) >= max_sequences:
            break

        # Alpaca dataset has 'instruction', 'input', and 'output' fields
        text_parts = []
        if 'instruction' in example and example['instruction']:
            text_parts.append(example['instruction'])
        if 'input' in example and example['input']:
            text_parts.append(example['input'])
        if 'output' in example and example['output']:
            text_parts.append(example['output'])

        # Combine into single text
        text = ' '.join(text_parts).strip()

        # Remove newlines and extra whitespace
        text = ' '.join(text.split())

        # Skip short texts - we want substantial content for windowing
        if len(text) < 800:  # Much longer minimum text length
            continue

        all_examples.append(text)

    sampled_texts = all_examples

    # Tokenization with endoftext filtering
    for text in sampled_texts:
        token_ids = model.tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True, padding="max_length")

        # Remove endoftext tokens (50256 for GPT-2)
        input_ids = token_ids['input_ids'].squeeze(0)  # Remove batch dimension
        endoftext_token_id = 50256

        # Filter out endoftext tokens and pad/truncate to desired length
        filtered_ids = input_ids[input_ids != endoftext_token_id]

        # If we have too few tokens after filtering, pad with a neutral token (e.g., space token)
        if len(filtered_ids) < seq_len:
            # Use space token (220) for padding instead of endoftext
            space_token_id = 220  # Space token in GPT-2
            padding_needed = seq_len - len(filtered_ids)
            padding = torch.full((padding_needed,), space_token_id, dtype=filtered_ids.dtype)
            filtered_ids = torch.cat([filtered_ids, padding])
        else:
            # Truncate if too long
            filtered_ids = filtered_ids[:seq_len]

        tokens.append(filtered_ids.unsqueeze(0))  # Add batch dimension back

    tokens = torch.cat(tokens, dim=0)
    print(f"Tokenized {len(tokens)} sequences of length {seq_len}")

    # Generate feature list based on arguments
    if args.random:
        # Select random features from available range
        available_features = list(range(crosscoder.ae_dim))
        selected_features = random.sample(available_features, min(args.num_features, len(available_features)))
        selected_features.sort()  # Sort for consistent display
        print(f"Randomly selected features: {selected_features}")
    else:
        # Select features in numerical order
        selected_features = list(range(min(args.num_features, crosscoder.ae_dim)))
        print(f"Sequential features: {selected_features}")

    # Create config with user-specified parameters
    config = CrosscoderVisConfig(
        features=selected_features,
        minibatch_size_features=16,
        minibatch_size_tokens=args.batch_size,
        verbose=True,  # Enable progress reporting
    )

    # Use the default layout which includes ActsHistogramConfig, and just modify the sequence config
    from sae_vis.data_config_classes import SeqMultiGroupConfig, CrossLayerTrajectoryConfig

    # Configure sequence display to match reference implementation approach
    # Reduce verbose output and focus on high-quality examples
    config.feature_centric_layout.seq_cfg = SeqMultiGroupConfig(
        top_acts_group_size=15,  # Fewer, higher quality top activation examples
        n_quantiles=5,  # Show 5 quantile intervals for better distribution
        quantile_group_size=6,  # Fewer examples per quantile to reduce clutter
        buffer=(8, 8),  # Smaller buffer for more focused context
        compute_buffer=True,  # Enable proper buffer computation
        top_logits_hoverdata=5,  # Show top 5 logits in hover
    )

    # Add cross-layer trajectory visualization
    config.feature_centric_layout.cross_layer_trajectory_cfg = CrossLayerTrajectoryConfig(
        n_sequences=20,  # Show trajectories for 20 token sequences
        height=400,
        normalize=True,  # Normalize to [0, 1] like in the reference image
        show_mean=True,
    )

    # Get feature data
    print(f"🔄 Step 1/2: Generating feature data...")
    print(f"   └── Processing {len(tokens)} sequences with {len(selected_features)} features using {args.batch_size} batch size")
    try:
        vis_data = get_feature_data(
            model=model,
            crosscoder=crosscoder,
            tokens=tokens,
            cfg=config,
        )

        # Save visualization
        print(f"🔄 Step 2/2: Saving visualization to {args.output}...")
        vis_data.save_feature_centric_vis(
            filename=args.output,
            verbose=True,
        )

        print(f"✅ Done! Open {args.output} in a browser.")

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
