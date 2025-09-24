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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sae_vis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crosscoder.crosscoder import Crosscoder

import nnsight
from nnsight import LanguageModel

from datasets import load_dataset

from sae_vis.data_config_classes import CrosscoderVisConfig, CrosscoderVisLayoutConfig
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_storing_fns import CrosscoderVisData


def load_latest_checkpoint(device=None):
    """Load the most recent crosscoder checkpoint."""
    saves_dir = Path("crosscoder/saves")

    checkpoints = list(saves_dir.glob("version_*/[0-9]*.pt"))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {saves_dir}")

    # Sort by version and checkpoint number
    def sort_key(path):
        version = int(path.parent.name.split("_")[1])
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
        if "dtype" in cfg and isinstance(cfg["dtype"], str):
            cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
    else:
        # Default config
        cfg = {
            "model_name": "gpt2",
            "ae_dim": 4096,
            "device": device,
            "dtype": torch.float32,
            "drop_bos": True,
        }

    # Add missing required fields with defaults
    if "dec_init_norm" not in cfg:
        cfg["dec_init_norm"] = 0.05
    if "seed" not in cfg:
        cfg["seed"] = 42

    # Use detected/specified device
    cfg["device"] = device

    # Load the model first - needed for Crosscoder initialization
    model_name = cfg.get("model_name", "gpt2")
    print(f"Loading model for crosscoder: {model_name}")
    model_name_mapping = {
        "gpt2": "gpt2",
        "pythia": "EleutherAI/pythia-2.8b-deduped",
        "pythia7b": "EleutherAI/pythia-6.9b-deduped",
        "pythia160m": "EleutherAI/pythia-160m-deduped",
        "pythia410m": "EleutherAI/pythia-410m-deduped",
        "pythia1b": "EleutherAI/pythia-1b-deduped",
        "pythia1.4b": "EleutherAI/pythia-1.4b-deduped",
        "pythia2.8b": "EleutherAI/pythia-2.8b-deduped",
        "gemma3-4b": "google/gemma-2-9b",
        "qwen3-4b": "Qwen/Qwen2.5-3B",
        "gemma2-2b": "google/gemma-2-2b",
    }

    actual_model_name = model_name_mapping.get(model_name, model_name)
    if device == "cuda":
        model = nnsight.LanguageModel(actual_model_name, device_map="cuda")
    else:
        model = nnsight.LanguageModel(actual_model_name, device_map="cpu")

    # Create crosscoder with both config and model
    crosscoder = Crosscoder(cfg, model)

    # Load weights to the correct device
    crosscoder.W_enc.data = checkpoint["W_enc"].to(device)
    crosscoder.W_dec.data = checkpoint["W_dec"].to(device)
    crosscoder.b_enc.data = checkpoint["b_enc"].to(device)
    crosscoder.b_dec.data = checkpoint["b_dec"].to(device)

    crosscoder.eval()
    return crosscoder, cfg, model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate crosscoder visualization dashboard with large-scale data"
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=10,
        help="Number of features to visualize (default: 10)",
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Select features randomly instead of in numerical order",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for token processing (default: 16)",
    )

    parser.add_argument(
        "--num_sequences",
        type=int,
        default=5000,
        help="Number of sequences to process (default: 5000)",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Length of each sequence (default: 1024, max: 1024 for GPT-2, 2048 for Pythia)",
    )

    parser.add_argument(
        "--top_acts_per_feature",
        type=int,
        default=100,
        help="Number of top activating sequences to show per feature (default: 100)",
    )

    parser.add_argument(
        "--quantile_examples",
        type=int,
        default=20,
        help="Number of examples per quantile group (default: 20)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="crosscoder_dashboard.html",
        help="Output filename for the dashboard (default: crosscoder_dashboard.html)",
    )

    parser.add_argument(
        "--use_cached_data",
        action="store_true",
        help="Use cached token data if available",
    )

    parser.add_argument(
        "--cache_file",
        type=str,
        default="cached_tokens.pt",
        help="Cache file for token data (default: cached_tokens.pt)",
    )

    return parser.parse_args()


def load_diverse_data(
    model, num_sequences=5000, seq_len=1024, cache_file=None, use_cache=False
):
    """Load a large, diverse dataset for better feature interpretation."""

    # Check for cached data
    if use_cache and cache_file and Path(cache_file).exists():
        print(f"Loading cached tokens from {cache_file}...")
        tokens = torch.load(cache_file)
        print(f"Loaded {len(tokens)} cached sequences")
        return tokens

    print(f"Loading {num_sequences} diverse text sequences of length {seq_len}...")

    # Set up dataset cache directory
    PROJECT_ROOT = Path(__file__).parent
    DATASET_CACHE_DIR = PROJECT_ROOT / "data" / "hf_datasets_cache"
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_texts = []
    tokens = []

    # Try multiple data sources for diversity
    data_sources = [
        ("monology/pile-uncopyrighted", "train"),
        ("wikitext", "wikitext-103-raw-v1", "train"),
        ("openwebtext", "plain_text", "train"),
    ]

    texts_per_source = num_sequences // len(data_sources) + 100  # Extra buffer

    for source_info in data_sources:
        if len(all_texts) >= num_sequences:
            break

        try:
            if len(source_info) == 2:
                dataset_name, split = source_info
                config = None
            else:
                dataset_name, config, split = source_info

            print(f"Loading from {dataset_name}...")

            # Load dataset with streaming for memory efficiency
            dataset = (
                load_dataset(
                    dataset_name,
                    config,
                    split=split,
                    streaming=True,
                    cache_dir=str(DATASET_CACHE_DIR),
                )
                if config
                else load_dataset(
                    dataset_name,
                    split=split,
                    streaming=True,
                    cache_dir=str(DATASET_CACHE_DIR),
                )
            )

            collected = 0
            for i, example in enumerate(dataset):
                if collected >= texts_per_source:
                    break

                # Get text field (different datasets use different field names)
                text = example.get("text", example.get("content", ""))
                if not text:
                    continue

                # Clean and normalize
                text = " ".join(text.split())

                # Filter for reasonable length texts
                word_count = len(text.split())
                if 100 < word_count < 10000:
                    all_texts.append(text)
                    collected += 1

                # Progress indicator
                if collected % 100 == 0 and collected > 0:
                    print(f"  Collected {collected} texts from {dataset_name}")

        except Exception as e:
            print(f"Could not load {dataset_name}: {e}")
            continue

    print(f"Collected {len(all_texts)} diverse text samples")

    # If we don't have enough texts, use the Pile as primary source
    if len(all_texts) < num_sequences // 2:
        print("Loading more texts from The Pile...")
        try:
            dataset = load_dataset(
                "monology/pile-uncopyrighted",
                split="train",
                streaming=True,
                cache_dir=str(DATASET_CACHE_DIR),
            )

            for i, example in enumerate(dataset):
                if len(all_texts) >= num_sequences:
                    break

                text = example.get("text", "")
                if not text:
                    continue

                # Clean and normalize
                text = " ".join(text.split())

                # Filter for reasonable length
                if 200 < len(text.split()) < 5000:
                    all_texts.append(text)

                # Progress indicator
                if len(all_texts) % 100 == 0:
                    print(f"  Collected {len(all_texts)} texts...")

        except Exception as e:
            print(f"Could not load additional Pile data: {e}")

    # Convert texts to tokens with sliding windows
    print(f"Tokenizing {num_sequences} sequences of length {seq_len}...")

    window_stride = seq_len // 2  # 50% overlap

    with tqdm(total=num_sequences, desc="Tokenizing") as pbar:
        for text in all_texts:
            if len(tokens) >= num_sequences:
                break

            # Tokenize the full text
            token_ids = model.tokenizer(
                text, return_tensors="pt", truncation=False, add_special_tokens=False
            )
            input_ids = token_ids["input_ids"].squeeze(0)

            # Skip if too short
            if len(input_ids) < seq_len:
                # Pad if needed
                padding_needed = seq_len - len(input_ids)
                padding = torch.full(
                    (padding_needed,),
                    model.tokenizer.pad_token_id or 0,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, padding])
                tokens.append(input_ids.unsqueeze(0))
                pbar.update(1)
            else:
                # Use sliding windows for longer texts
                num_windows = min(
                    (len(input_ids) - seq_len) // window_stride + 1, 5
                )  # Max 5 windows per text
                for i in range(num_windows):
                    if len(tokens) >= num_sequences:
                        break
                    start_idx = i * window_stride
                    window = input_ids[start_idx : start_idx + seq_len]
                    tokens.append(window.unsqueeze(0))
                    pbar.update(1)

    # Concatenate all tokens
    tokens = torch.cat(tokens[:num_sequences], dim=0)
    print(f"Created {len(tokens)} sequences of length {seq_len}")

    # Save cache if requested
    if cache_file:
        print(f"Saving token cache to {cache_file}...")
        torch.save(tokens, cache_file)

    return tokens


def main():
    args = parse_args()
    print("=" * 80)
    print("CROSSCODER VISUALIZATION WITH LARGE-SCALE DATA")
    print("=" * 80)
    print(f"Configuration:")
    print(
        f"  - Features: {args.num_features} ({'random' if args.random else 'sequential'})"
    )
    print(f"  - Sequences: {args.num_sequences} x {args.seq_len} tokens")
    print(f"  - Top acts per feature: {args.top_acts_per_feature}")
    print(f"  - Examples per quantile: {args.quantile_examples}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Output: {args.output}")
    print("=" * 80)

    # Auto-detect device for consistent usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Auto-detected device: {device}")

    # Load crosscoder with consistent device
    crosscoder, crosscoder_cfg, model = load_latest_checkpoint(device=device)
    print(f"Loaded crosscoder with {crosscoder.ae_dim} features")

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

    # Load large diverse dataset
    tokens = load_diverse_data(
        model,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        cache_file=args.cache_file if args.use_cached_data else None,
        use_cache=args.use_cached_data,
    )

    # Generate feature list based on arguments
    if args.random:
        # Select random features from available range
        available_features = list(range(crosscoder.ae_dim))
        selected_features = random.sample(
            available_features, min(args.num_features, len(available_features))
        )
        selected_features.sort()  # Sort for consistent display
        print(f"Randomly selected features: {selected_features}")
    else:
        # Select features in numerical order
        selected_features = list(range(min(args.num_features, crosscoder.ae_dim)))
        print(f"Sequential features: {selected_features}")

    # Create config with enhanced parameters for large-scale analysis
    config = CrosscoderVisConfig(
        features=selected_features,
        minibatch_size_features=32,  # Larger feature batch for efficiency
        minibatch_size_tokens=args.batch_size,
        verbose=True,  # Enable progress reporting
    )

    # Configure sequence display for large-scale analysis
    from sae_vis.data_config_classes import (
        SeqMultiGroupConfig,
        CrossLayerTrajectoryConfig,
    )

    # Much more data for better interpretation
    config.feature_centric_layout.seq_cfg = SeqMultiGroupConfig(
        top_acts_group_size=args.top_acts_per_feature,  # Many more top examples
        n_quantiles=10,  # Keep 10 quantile groups for good distribution view
        quantile_group_size=args.quantile_examples,  # More examples per quantile
        buffer=(32, 32),  # Larger context window for 1024 token sequences
        compute_buffer=True,  # Enable proper buffer computation
        top_logits_hoverdata=10,  # Show top 10 logits in hover
    )

    # Add cross-layer trajectory visualization
    config.feature_centric_layout.cross_layer_trajectory_cfg = (
        CrossLayerTrajectoryConfig(
            n_sequences=1,  # Not used for decoder norms (always single trajectory)
            height=400,
            normalize=True,  # Normalize decoder norms to [0, 1]
            show_mean=True,
        )
    )

    # Get feature data with progress tracking
    print(
        f"\n🔄 Processing {len(tokens)} sequences with {len(selected_features)} features..."
    )
    print(f"   This will take longer due to the large amount of data being processed.")
    print(f"   The results will be much more interpretable!\n")
    try:
        vis_data = get_feature_data(
            model=model,
            crosscoder=crosscoder,
            tokens=tokens,
            cfg=config,
        )

        # Save visualization
        print(f"\n💾 Saving visualization to {args.output}...")
        vis_data.save_feature_centric_vis(
            filename=args.output,
            verbose=True,
        )

        print(
            f"\n✅ Success! Open {args.output} in a browser to explore your features."
        )
        print(
            f"   With {args.num_sequences} sequences analyzed, you should see much clearer patterns!"
        )

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
