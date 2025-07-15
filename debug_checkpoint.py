#!/usr/bin/env python3
"""Debug script to check checkpoint shapes."""

import torch
import json
from pathlib import Path

def debug_checkpoint(checkpoint_path):
    """Debug a checkpoint to understand tensor shapes."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg_path = checkpoint_path.parent / f"{checkpoint_path.stem}_cfg.json"
    
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    print(f"Config: {cfg}")
    print()
    
    print("Checkpoint tensor shapes:")
    for key, tensor in checkpoint.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
        else:
            print(f"  {key}: {type(tensor)}")
    
    print()
    print("Expected shapes for crosscoder:")
    print(f"  W_enc: ({cfg.get('n_layer', 12)}, {cfg.get('resid_dim', 768)}, {cfg['ae_dim']})")
    print(f"  W_dec: ({cfg['ae_dim']}, {cfg.get('n_layer', 12)}, {cfg.get('resid_dim', 768)})")
    print(f"  b_enc: ({cfg['ae_dim']},)")
    print(f"  b_dec: ({cfg.get('n_layer', 12)}, {cfg.get('resid_dim', 768)})")

if __name__ == "__main__":
    debug_checkpoint(Path("crosscoder/saves/version_2/1.pt"))