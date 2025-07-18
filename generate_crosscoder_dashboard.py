#!/usr/bin/env python3
"""
Generate Crosscoder Dashboard

This script loads the latest crosscoder checkpoint and generates a full SAE-VIS style dashboard
using the modified SAE-VIS codebase that works with NNsight and crosscoders.

Usage:
    python generate_crosscoder_dashboard.py
"""

import sys
from pathlib import Path

# Add the modified SAE-VIS to the path
sys.path.insert(0, str(Path("newcrosscodervis/sae_vis")))

from sae_vis.model_fns import load_crosscoder_and_data, get_model_activations
from sae_vis.data_config_classes import CrosscoderVisConfig, SaeVisLayoutConfig
from sae_vis.data_fetching_fns import get_feature_data
from sae_vis.data_storing_fns import CrosscoderVisData
import torch
import numpy as np
from typing import Optional


def generate_crosscoder_dashboard(
    output_path: str = "crosscoder_dashboard.html",
    n_features: int = 20,
    seq_len: int = 1024,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
    verbose: bool = True
):
    """
    Generate a crosscoder dashboard using the modified SAE-VIS codebase.

    Args:
        output_path: Path to save the HTML dashboard
        n_features: Number of features to visualize
        seq_len: Sequence length for tokenization
        checkpoint_path: Path to crosscoder checkpoint (auto-detected if None)
        config_path: Path to crosscoder config (auto-detected if None)
        verbose: Whether to show progress information
    """
    print("=== Crosscoder Dashboard Generator ===")
    print(f"Generating dashboard with {n_features} features...")

    # Load crosscoder and data
    print("Loading crosscoder model and data...")
    crosscoder, model, tokens = load_crosscoder_and_data(
        seq_len=seq_len,
        device="cpu",  # Force CPU to avoid MPS issues
        checkpoint_path=checkpoint_path,
        config_path=config_path
    )

    # Select features to visualize
    if n_features > crosscoder.ae_dim:
        print(f"Requested {n_features} features, but crosscoder only has {crosscoder.ae_dim}. Using all features.")
        n_features = crosscoder.ae_dim

    # Select random features for visualization
    # np.random.seed(42)  # For reproducibility
    selected_features = np.random.choice(crosscoder.ae_dim, n_features, replace=False).tolist()

    # Create configuration
    cfg = CrosscoderVisConfig(
        features=selected_features,
        minibatch_size_features=min(50, n_features),  # Smaller batches for memory efficiency
        minibatch_size_tokens=8,  # Small token batches for memory efficiency
        seed=42,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        feature_centric_layout=SaeVisLayoutConfig.default_feature_centric_layout(),
    )

    print(f"Selected features: {selected_features}")
    print(f"Processing {len(tokens)} sequences...")

    # Generate feature data
    print("Generating feature visualization data...")
    crosscoder_vis_data = get_feature_data(
        crosscoder=crosscoder,
        model=model,
        tokens=tokens,
        cfg=cfg,
        verbose=verbose,
        clear_memory_between_batches=True
    )

    # Save the dashboard
    print(f"Saving dashboard to {output_path}...")
    crosscoder_vis_data.save_feature_centric_vis(
        filename=output_path,
        verbose=verbose
    )

    print(f"\n=== Dashboard Generated Successfully ===")
    print(f"Open {output_path} in your browser to view the dashboard")
    print(f"\nFeatures:")
    print(f"- Interactive feature visualization with {n_features} features")
    print(f"- Activation histograms and distributions")
    print(f"- Top activating sequences with token highlighting")
    print(f"- Logit tables showing most affected tokens")
    print(f"- Feature correlation tables")
    print(f"- Cross-layer feature analysis")

    return output_path


def main():
    """Main function"""
    try:
        output_path = generate_crosscoder_dashboard(
            output_path="crosscoder_sae_vis_dashboard.html",
            n_features=20,  # Reasonable number for visualization
            seq_len=1024,    # Shorter sequences for faster processing
            verbose=True
        )

        print(f"\n✅ Success! Dashboard saved to: {output_path}")
        print("\nThis dashboard includes:")
        print("• Feature-centric visualization (like the image you showed)")
        print("• Neuron alignment tables")
        print("• Activation histograms")
        print("• Top activating sequences")
        print("• Logit tables")
        print("• Cross-layer feature analysis")

    except Exception as e:
        print(f"\n❌ Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
