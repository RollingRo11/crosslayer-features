#!/usr/bin/env python3
"""
Generate a crosscoder visualization dashboard from a checkpoint file.
This script loads the most recent crosscoder checkpoint and generates an SAE-vis style dashboard.
"""

import os
import sys
import torch
import glob
from pathlib import Path
from typing import Optional
import einops
import numpy as np
from tqdm import tqdm

# Add the current directory and sae_vis to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'sae_vis', 'sae_vis'))

import nnsight
from nnsight import LanguageModel
from crosscoder.crosscoder import Crosscoder
from data_config_classes import (
    CrosscoderVisConfig,
    CrosscoderVisLayoutConfig,
    Column,
    FeatureTablesConfig,
    ActsHistogramConfig,
    LogitsTableConfig,
    LogitsHistogramConfig,
    SeqMultiGroupConfig,
)
from data_storing_fns import CrosscoderVisData
from data_fetching_fns import ActivationCache, parse_feature_data
from data_storing_fns import save_feature_centric_vis, save_prompt_centric_vis
from utils_fns import RollingCorrCoef
from model_fns import load_crosscoder_checkpoint, tokenize_dataset, to_resid_dir


def get_latest_checkpoint(saves_dir: str = "crosscoder/saves") -> str:
    """Find the most recent checkpoint file."""
    # Find all .pt files
    pattern = os.path.join(saves_dir, "**/*.pt")
    checkpoint_files = glob.glob(pattern, recursive=True)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {saves_dir}")

    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint


def run_model_with_cache(
    model: LanguageModel,
    crosscoder: Crosscoder,
    tokens: torch.Tensor,
    feature_indices: list[int],
) -> tuple[ActivationCache, torch.Tensor]:
    """Run the model and collect activations for visualization."""
    cache = ActivationCache()
    device = next(crosscoder.parameters()).device
    tokens = tokens.to(device)

    # Process in smaller batches to avoid OOM
    batch_size = 16
    all_crosscoder_acts = []
    all_resid_acts = []

    for i in tqdm(range(0, len(tokens), batch_size), desc="Processing batches"):
        batch_tokens = tokens[i:i+batch_size]

        with torch.no_grad():
            # Run model with nnsight to get residual stream activations
            with model.trace(batch_tokens, validate=False):
                # Collect residual stream activations from all layers
                resid_acts = []
                for layer_idx in range(crosscoder.num_layers):
                    if layer_idx == 0:
                        # First layer: after embeddings
                        resid = model.transformer.wte(batch_tokens) + model.transformer.wpe(torch.arange(batch_tokens.shape[1], device=device))
                    else:
                        # Other layers: after previous layer
                        resid = model.transformer.h[layer_idx-1].output[0]
                    resid_acts.append(resid.value)

                # Stack residual activations: [batch, seq, n_layers, d_model]
                stacked_resid = torch.stack(resid_acts, dim=2)

                # Get final logits
                final_resid = model.transformer.h[-1].output[0]
                logits = model.lm_head(model.transformer.ln_f(final_resid))
                final_logits = logits.value

        # Apply crosscoder to get features
        crosscoder_acts = crosscoder.encode(stacked_resid)

        all_crosscoder_acts.append(crosscoder_acts)
        all_resid_acts.append(stacked_resid)

    # Concatenate all batches
    all_crosscoder_acts = torch.cat(all_crosscoder_acts, dim=0)
    all_resid_acts = torch.cat(all_resid_acts, dim=0)

    # Store in cache
    cache["crosscoder_acts"] = all_crosscoder_acts
    cache["resid_acts"] = all_resid_acts

    return cache, final_logits


def get_feature_directions(
    crosscoder: Crosscoder,
    feature_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the decoder directions for specified features."""
    # Get decoder weights for the specified features
    # W_dec shape: [ae_dim, n_layers, d_model]
    feature_dirs = crosscoder.W_dec[feature_indices]  # [n_features, n_layers, d_model]

    # For input directions, we can use the encoder weights
    # W_enc shape: [n_layers, d_model, ae_dim]
    feature_dirs_input = crosscoder.W_enc[:, :, feature_indices]  # [n_layers, d_model, n_features]
    feature_dirs_input = einops.rearrange(feature_dirs_input, "n_layers d_model n_features -> n_features n_layers d_model")

    return feature_dirs, feature_dirs_input


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "roneneldan/TinyStories"  # Small dataset for demos
    num_features_to_vis = 10  # Number of features to visualize
    num_tokens = 5000  # Number of tokens to process
    seq_len = 128  # Sequence length

    print("Finding latest checkpoint...")
    checkpoint_path = get_latest_checkpoint()
    print(f"Using checkpoint: {checkpoint_path}")

    print("Loading crosscoder and model...")
    crosscoder, model = load_crosscoder_checkpoint(checkpoint_path, device)

    print("Tokenizing dataset...")
    tokens = tokenize_dataset(dataset_name, model, seq_len, num_samples=num_tokens // seq_len)

    # Select random features to visualize
    total_features = crosscoder.ae_dim
    feature_indices = list(np.random.choice(total_features, num_features_to_vis, replace=False))
    print(f"Visualizing features: {feature_indices}")

    print("Running model and collecting activations...")
    cache, logits = run_model_with_cache(model, crosscoder, tokens, feature_indices)

    print("Getting feature directions...")
    feature_resid_dir, feature_resid_dir_input = get_feature_directions(crosscoder, feature_indices)

    # For crosscoders, we need to handle the multi-layer aspect
    # Let's focus on the final layer for the output direction
    feature_out_dir = feature_resid_dir[:, -1, :]  # [n_features, d_model]

    # Create visualization config
    vis_config = CrosscoderVisConfig(
        features=feature_indices,
        minibatch_size_features=32,
        minibatch_size_tokens=16,
        feature_centric_layout=CrosscoderVisLayoutConfig(
            columns=[
                Column(FeatureTablesConfig(n_rows=5)),
                Column(
                    ActsHistogramConfig(n_bins=50),
                    LogitsTableConfig(n_rows=10),
                    LogitsHistogramConfig(n_bins=50),
                ),
                Column(SeqMultiGroupConfig(
                    n_quantiles=7,
                    top_acts_group_size=20,
                    quantile_group_size=10,
                )),
            ],
            height=750,
        ),
        prompt_centric_layout=CrosscoderVisLayoutConfig(
            columns=[
                Column(
                    SeqMultiGroupConfig(
                        n_quantiles=0,
                        top_acts_group_size=5,
                    ),
                    width=600,
                ),
            ],
            height=400,
        ),
    )

    print("Parsing feature data...")
    vis_data, time_logs = parse_feature_data(
        model=model,
        cfg=vis_config,
        crosscoder=crosscoder,
        crosscoder_B=None,
        tokens=tokens,
        feature_indices=feature_indices,
        feature_resid_dir=feature_resid_dir,
        feature_resid_dir_input=feature_resid_dir_input,
        cache=cache,
        feature_out_dir=feature_out_dir,
        target_logits=logits if logits.shape[-1] < 1000 else None,
    )

    # Save visualizations
    output_dir = Path("crosscoder_vis_outputs")
    output_dir.mkdir(exist_ok=True)

    print("Saving feature-centric visualizations...")
    feature_vis_path = output_dir / "feature_vis.html"
    save_feature_centric_vis(
        data=vis_data,
        filename=str(feature_vis_path),
    )

    print("Saving prompt-centric visualization...")
    prompt_vis_path = output_dir / "prompt_vis.html"
    prompt_tokens = tokens[0:1]  # Just use first sequence for prompt-centric view
    save_prompt_centric_vis(
        data=vis_data,
        filename=str(prompt_vis_path),
        prompts=[model.tokenizer.decode(prompt_tokens[0])],
        tokens=prompt_tokens,
    )

    print(f"\nVisualization saved to:")
    print(f"  - Feature-centric: {feature_vis_path}")
    print(f"  - Prompt-centric: {prompt_vis_path}")
    print("\nOpen these HTML files in a web browser to view the visualizations.")


if __name__ == "__main__":
    main()
