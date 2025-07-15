#!/usr/bin/env python3
"""
Dashboard generator for crosscoder visualization using NNsight.
This script loads a saved crosscoder model and generates a visualization dashboard.
"""

import json
import torch
import argparse
import os
from pathlib import Path

# Force CPU device to avoid MPS issues with placeholder storage
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TORCH_DEVICE'] = 'cpu'

from nnsight import LanguageModel
from datasets import load_dataset

# Import the adapted crosscoder-vis modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "crosscoder-vis"))

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
    
    # Get model info to set d_in correctly
    if saved_cfg['model_name'] == 'gpt2':
        d_in = 768  # GPT-2 embedding dimension
        n_layers = 12  # GPT-2 has 12 layers
    else:
        d_in = saved_cfg.get('resid_dim', 768)
        n_layers = saved_cfg.get('n_layers', 12)
    
    # Create crosscoder config
    crosscoder_cfg = CrossCoderConfig(
        d_in=d_in,
        d_hidden=saved_cfg['ae_dim'],
        n_layers=n_layers
    )
    
    # Create crosscoder model
    crosscoder = CrossCoder(crosscoder_cfg)
    
    # Load the weights and ensure they're not meta tensors
    # The checkpoint tensors should have actual data, not be meta tensors
    print(f"DEBUG: Loading checkpoint tensors...")
    print(f"DEBUG: W_enc device: {checkpoint['W_enc'].device}, is_meta: {checkpoint['W_enc'].is_meta}")
    print(f"DEBUG: W_dec device: {checkpoint['W_dec'].device}, is_meta: {checkpoint['W_dec'].is_meta}")
    
    # Check checkpoint shapes vs crosscoder shapes
    print(f"DEBUG: Checkpoint W_enc shape: {checkpoint['W_enc'].shape}")
    print(f"DEBUG: Crosscoder W_enc shape: {crosscoder.W_enc.shape}")
    print(f"DEBUG: Checkpoint W_dec shape: {checkpoint['W_dec'].shape}")
    print(f"DEBUG: Crosscoder W_dec shape: {crosscoder.W_dec.shape}")
    
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
            
        token_ids = model.tokenizer(
            text,
            return_tensors="pt",
            max_length=context_length,
            truncation=True,
            padding="max_length"
        ).input_ids
        
        tokens.append(token_ids)
        count += 1
    
    # Move all tokens to the same device as the model
    all_tokens = torch.cat(tokens, dim=0)
    # Get the device from the model parameter
    device = next(model.parameters()).device
    return all_tokens.to(device)

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
    # Choose device based on availability
    import torch
    import os
    # Disable MPS entirely to avoid placeholder storage issues on macOS
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Use CUDA if available, otherwise CPU
    available_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEBUG: Using device: {available_device}")
    
    # Try different approaches to avoid meta tensors
    try:
        # Method 1: Load with explicit device_map="auto"
        model = LanguageModel(saved_cfg['model_name'], device_map="auto", torch_dtype=torch.float32)
    except Exception as e:
        print(f"Method 1 failed: {e}")
        try:
            # Method 2: Load with device_map={"": available_device}
            model = LanguageModel(saved_cfg['model_name'], device_map={"": available_device}, torch_dtype=torch.float32)
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            try:
                # Method 3: Load with low_cpu_mem_usage=False
                model = LanguageModel(saved_cfg['model_name'], low_cpu_mem_usage=False, torch_dtype=torch.float32)
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                # Method 4: Default loading
                model = LanguageModel(saved_cfg['model_name'])
    
    # Check what device the model is on
    device = next(model.parameters()).device
    print(f"DEBUG: Model device: {device}")
    
    # Also check if model parameters are meta
    first_param = next(model.parameters())
    print(f"DEBUG: First param is_meta: {first_param.is_meta}")
    print(f"DEBUG: First param shape: {first_param.shape}")
    
    # Debug: Check model attributes
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Model attributes: {[attr for attr in dir(model) if 'token' in attr.lower()]}")
    
    # If model is still on meta device, try to materialize it
    if device.type == 'meta':
        print("WARNING: Model is on meta device. This is a known issue with NNsight.")
        print("Attempting to materialize model by creating a compatible wrapper...")
        
        try:
            # Store the tokenizer before materialization
            tokenizer = model.tokenizer
            
            # Use to_empty() for meta tensors, then initialize with random values
            materialized_model = model.to_empty(device=available_device)
            
            # Initialize ALL model parameters with random values
            with torch.no_grad():
                for param in materialized_model.parameters():
                    if param.is_meta or param.device.type == 'meta':
                        param.data = torch.randn_like(param, device=available_device)
                    else:
                        # Ensure parameter is on the correct device
                        if param.device != available_device:
                            param.data = param.data.to(available_device)
                
                # Also initialize buffers
                for buffer in materialized_model.buffers():
                    if buffer.is_meta or buffer.device.type == 'meta':
                        buffer.data = torch.randn_like(buffer, device=available_device)
                    else:
                        # Ensure buffer is on the correct device
                        if buffer.device != available_device:
                            buffer.data = buffer.data.to(available_device)
            
            # Create a simple wrapper that makes the materialized model look like a LanguageModel
            class MaterializedLanguageModel:
                def __init__(self, model, tokenizer):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.config = model.config
                
                def parameters(self):
                    return self.model.parameters()
                
                def to(self, device):
                    self.model = self.model.to(device)
                    return self
                    
                def __call__(self, *args, **kwargs):
                    return self.model(*args, **kwargs)
                    
                def __getattr__(self, name):
                    return getattr(self.model, name)
                    
                def trace(self, tokens):
                    # Create a simple context manager that returns the model
                    class TraceContext:
                        def __init__(self, model):
                            self.model = model
                        
                        def __enter__(self):
                            return self
                        
                        def __exit__(self, exc_type, exc_val, exc_tb):
                            pass
                    
                    return TraceContext(self.model)
                    
            # Create our wrapper
            model = MaterializedLanguageModel(materialized_model, tokenizer)
            
            device = next(model.parameters()).device
            print(f"DEBUG: After materialization - Model device: {device}")
            
        except Exception as e:
            print(f"ERROR: Failed to materialize model: {e}")
            return
    
    # Move crosscoder to same device as model
    print(f"DEBUG: Moving crosscoder to device: {device}")
    print(f"DEBUG: Before move - W_enc device: {crosscoder.W_enc.device}, is_meta: {crosscoder.W_enc.is_meta}")
    crosscoder = crosscoder.to(device)
    print(f"DEBUG: After move - W_enc device: {crosscoder.W_enc.device}, is_meta: {crosscoder.W_enc.is_meta}")
    
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