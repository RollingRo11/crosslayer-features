#!/usr/bin/env python3
"""
Simple generator for crosscoder visualization dashboard.
"""

import os
import sys
import torch
import json
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sae_vis'))

# Import crosscoder
from crosscoder.crosscoder import Crosscoder

# Import NNsight
import nnsight
from nnsight import LanguageModel

# Import datasets
from datasets import load_dataset

# Now import our modified sae_vis after setting up paths
from sae_vis.data_config_classes import (
    CrosscoderVisConfig,
    CrosscoderVisLayoutConfig,
    Column,
    FeatureTablesConfig,
    ActsHistogramConfig,
    LogitsTableConfig,
    LogitsHistogramConfig,
    SeqMultiGroupConfig,
)
from sae_vis.data_storing_fns import CrosscoderVisData
from sae_vis.data_fetching_fns import get_feature_data


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


def main():
    print("Starting crosscoder visualization generation...")
    
    # Load crosscoder
    crosscoder = load_latest_checkpoint()
    print(f"Loaded crosscoder with {crosscoder.ae_dim} features")
    
    # Load model
    print("Loading GPT-2 model...")
    model = LanguageModel("gpt2", device_map="cpu")
    
    # Load some data
    print("Loading data...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # Tokenize sequences
    tokens = []
    for i, example in enumerate(dataset):
        if i >= 50:  # Just 50 sequences for demo
            break
        text = example['text']
        if len(text) < 100:
            continue
        token_ids = model.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        tokens.append(token_ids['input_ids'])
    
    tokens = torch.cat(tokens, dim=0)
    print(f"Tokenized {len(tokens)} sequences")
    
    # Create config
    config = CrosscoderVisConfig(
        features=list(range(10)),  # First 10 features
        minibatch_size_features=32,
        minibatch_size_tokens=8,
        verbose=True,
    )
    
    # Get feature data
    print("Generating feature data...")
    vis_data = get_feature_data(
        model=model,
        crosscoder=crosscoder,
        tokens=tokens,
        cfg=config,
    )
    
    # Save visualization
    output_file = "crosscoder_dashboard_new.html"
    print(f"Saving to {output_file}...")
    vis_data.save_feature_centric_vis(
        filename=output_file,
        verbose=True,
    )
    
    print(f"Done! Open {output_file} in a browser.")


if __name__ == "__main__":
    main()