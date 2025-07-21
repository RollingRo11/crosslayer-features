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


def load_latest_checkpoint():
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
    
    # Load checkpoint
    checkpoint = torch.load(latest, map_location='cpu')
    
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
            "ae_dim": 4000,
            "device": "cpu",
            "dtype": torch.float32,
            "drop_bos": True,
        }
    
    # Force CPU
    cfg["device"] = "cpu"
    
    # Create crosscoder
    crosscoder = Crosscoder(cfg)
    
    # Load weights
    crosscoder.W_enc.data = checkpoint['W_enc']
    crosscoder.W_dec.data = checkpoint['W_dec']
    crosscoder.b_enc.data = checkpoint['b_enc']
    crosscoder.b_dec.data = checkpoint['b_dec']
    
    crosscoder.eval()
    return crosscoder


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
    
    # Load crosscoder
    crosscoder = load_latest_checkpoint()
    print(f"Loaded crosscoder with {crosscoder.ae_dim} features")
    
    # Load model
    print("Loading GPT-2 model...")
    model = LanguageModel("gpt2", device_map="cpu")
    
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
    )
    
    # Use the default layout which includes ActsHistogramConfig, and just modify the sequence config
    from sae_vis.data_config_classes import SeqMultiGroupConfig
    
    # The default layout already includes ActsHistogramConfig, LogitsTableConfig, and LogitsHistogramConfig
    # We just need to modify the sequence configuration to show fewer examples
    config.feature_centric_layout.seq_cfg = SeqMultiGroupConfig(
        top_acts_group_size=25,  # Show more top activation examples
        n_quantiles=3,  # Show 3 quantile intervals  
        quantile_group_size=8,  # Show 8 examples per quantile
        buffer=(15, 15),  # Show 15 tokens before and after peak activation
        compute_buffer=True,  # Enable proper buffer computation
    )
    
    # Get feature data
    print("Generating feature data...")
    try:
        vis_data = get_feature_data(
            model=model,
            crosscoder=crosscoder,
            tokens=tokens,
            cfg=config,
        )
        
        # Save visualization
        print(f"Saving to {args.output}...")
        vis_data.save_feature_centric_vis(
            filename=args.output,
            verbose=True,
        )
        
        print(f"Done! Open {args.output} in a browser.")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()