#!/usr/bin/env python3
"""
Dashboard generator for crosscoder visualization using NNsight.
This script loads a saved crosscoder model and generates a visualization dashboard.
"""

import json
import torch
import argparse
from pathlib import Path
from nnsight import LanguageModel
from datasets import load_dataset

# Import the adapted crosscoder-vis modules
import sys
sys.path.append(str(Path(__file__).parent / "crosscoder-vis"))

from sae_vis.data_config_classes import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import CrossCoderConfig, CrossCoder

def load_crosscoder_from_checkpoint(checkpoint_path: Path):
    """Load a crosscoder model from a saved checkpoint."""
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load the config
    cfg_path = checkpoint_path.parent / f"{checkpoint_path.stem}_cfg.json"
    with open(cfg_path, 'r') as f:
        saved_cfg = json.load(f)
    
    # Convert dtype string back to torch dtype
    if isinstance(saved_cfg['dtype'], str):
        saved_cfg['dtype'] = getattr(torch, saved_cfg['dtype'].split('.')[-1])
    
    # Create crosscoder config
    crosscoder_cfg = CrossCoderConfig(
        d_in=saved_cfg.get('resid_dim', 768),  # Default for GPT-2
        d_hidden=saved_cfg['ae_dim']
    )
    
    # Create crosscoder model
    crosscoder = CrossCoder(crosscoder_cfg)
    
    # Load the weights
    crosscoder.W_enc.data = checkpoint['W_enc']
    crosscoder.W_dec.data = checkpoint['W_dec']
    crosscoder.b_enc.data = checkpoint['b_enc']
    crosscoder.b_dec.data = checkpoint['b_dec']
    
    return crosscoder, saved_cfg

def get_sample_tokens(model: LanguageModel, n_samples: int = 100, context_length: int = 256):
    """Get sample tokens from the dataset for visualization."""
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)
    
    tokens = []
    count = 0
    
    for item in dataset:
        if count >= n_samples:
            break
            
        text = item['text']
        if len(text.strip()) < 50:
            continue
            
        token_ids = model.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=context_length,
            truncation=True,
            padding="max_length"
        )
        
        tokens.append(token_ids)
        count += 1
    
    return torch.cat(tokens, dim=0)

def create_dashboard(
    checkpoint_path: Path,
    output_path: Path,
    n_features: int = 50,
    n_samples: int = 100,
    hook_point: str = "blocks.6.hook_resid_post"
):
    """Create a dashboard from a crosscoder checkpoint."""
    
    print(f"Loading crosscoder from {checkpoint_path}")
    crosscoder, saved_cfg = load_crosscoder_from_checkpoint(checkpoint_path)
    
    print(f"Loading model: {saved_cfg['model_name']}")
    model = LanguageModel(saved_cfg['model_name'], device_map="auto")
    
    print(f"Getting sample tokens (n_samples={n_samples})")
    tokens = get_sample_tokens(model, n_samples=n_samples)
    
    print("Creating visualization config")
    vis_cfg = SaeVisConfig(
        hook_point=hook_point,
        features=list(range(min(n_features, crosscoder.cfg.d_hidden))),
        minibatch_size_features=10,
        minibatch_size_tokens=50,
        verbose=True
    )
    
    print("Generating visualization data")
    # Create dual models (using same model for both A and B for simplicity)
    sae_vis_data = SaeVisData.create(
        encoder=crosscoder,
        model_A=model,
        model_B=model,
        tokens=tokens,
        cfg=vis_cfg
    )
    
    print(f"Saving dashboard to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sae_vis_data.save_feature_centric_vis(output_path)
    
    print(f"Dashboard saved successfully to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate crosscoder visualization dashboard")
    parser.add_argument("checkpoint", type=Path, help="Path to the crosscoder checkpoint (.pt file)")
    parser.add_argument("-o", "--output", type=Path, default="dashboard.html", help="Output HTML file path")
    parser.add_argument("-n", "--n-features", type=int, default=50, help="Number of features to visualize")
    parser.add_argument("-s", "--n-samples", type=int, default=100, help="Number of text samples to use")
    parser.add_argument("--hook-point", type=str, default="blocks.6.hook_resid_post", help="Hook point for feature extraction")
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file {args.checkpoint} does not exist")
        return 1
    
    if not args.checkpoint.suffix == '.pt':
        print(f"Error: Checkpoint file must have .pt extension")
        return 1
    
    try:
        create_dashboard(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            n_features=args.n_features,
            n_samples=args.n_samples,
            hook_point=args.hook_point
        )
        return 0
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())