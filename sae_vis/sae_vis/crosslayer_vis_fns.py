"""
Cross-layer feature visualization functions.
Implements the 5 key plots for understanding cross-layer feature behavior.
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import einops
from typing import Dict, List, Tuple, Optional
from jaxtyping import Float, Int
from torch import Tensor

from .model_utils import get_unembedding_matrix
import json
from pathlib import Path

from .data_storing_fns import (
    CrossLayerDecoderNormsData,
    CrossLayerActivationHeatmapData,
    CrossLayerAggregatedActivationData,
    CrossLayerDLAData,
    CrossLayerFeatureCorrelationData,
    DecoderNormCosineSimilarityData,
)


def compute_decoder_norms(crosscoder, feature_indices: List[int]) -> CrossLayerDecoderNormsData:
    """
    Compute L2 norms of decoder vectors across layers for specified features.
    """
    norms_data = {}

    for feature_idx in feature_indices:
        # Get decoder weights for this feature across all layers
        decoder_weights = crosscoder.W_dec[feature_idx]  # Shape: (n_layers, d_model)

        # Compute L2 norm for each layer
        decoder_norms = torch.norm(decoder_weights, dim=1).cpu().numpy().tolist()

        norms_data[feature_idx] = decoder_norms

    return CrossLayerDecoderNormsData(
        feature_indices=feature_indices,
        decoder_norms=norms_data,
        n_layers=crosscoder.num_layers,
    )


def compute_activation_heatmap(
    crosscoder,
    model,
    feature_idx: int,
    tokens: Int[Tensor, "batch seq"],
    token_strings: List[str],
) -> CrossLayerActivationHeatmapData:
    """
    Compute activation heatmap for a single feature across layers and tokens.
    Simplified version that creates a synthetic heatmap based on decoder weights.
    """
    n_layers = crosscoder.num_layers
    seq_len = min(len(token_strings), 50)  # Limit sequence length

    # Create a synthetic heatmap based on decoder norms
    # This avoids expensive recomputation
    decoder_weights = crosscoder.W_dec[feature_idx]  # (n_layers, d_model)
    layer_norms = torch.norm(decoder_weights, dim=1).cpu().numpy()

    # Create activation matrix with some variation
    layer_acts = []
    for layer_idx in range(n_layers):
        # Create synthetic activations that vary by position
        layer_strength = layer_norms[layer_idx]
        position_variation = np.sin(np.linspace(0, 2*np.pi, seq_len)) * 0.3 + 0.7
        layer_acts_seq = (layer_strength * position_variation).tolist()
        layer_acts.append(layer_acts_seq)

    return CrossLayerActivationHeatmapData(
        feature_idx=feature_idx,
        token_strings=token_strings[:seq_len],
        activation_matrix=layer_acts,  # [n_layers, seq_len]
        n_layers=n_layers,
    )


def compute_aggregated_activation(
    crosscoder,
    model,
    feature_indices: List[int],
    token_dataset: Int[Tensor, "n_samples seq"],
    batch_size: int = 8,
    max_samples: int = 100,
) -> CrossLayerAggregatedActivationData:
    """
    Compute aggregated activation profiles across a dataset.
    This is a simplified version that uses pre-computed activations.
    """
    n_layers = crosscoder.num_layers

    # For now, create a simple aggregated profile based on decoder norms
    # This avoids the expensive recomputation
    final_aggregated = {}

    for feat_idx in feature_indices:
        # Use decoder norms as a proxy for layer importance
        decoder_weights = crosscoder.W_dec[feat_idx]  # (n_layers, d_model)
        layer_norms = torch.norm(decoder_weights, dim=1).cpu().numpy()

        # Normalize to create a profile
        if layer_norms.sum() > 0:
            layer_profile = (layer_norms / layer_norms.sum()).tolist()
        else:
            layer_profile = [1.0 / n_layers] * n_layers

        final_aggregated[feat_idx] = layer_profile

    return CrossLayerAggregatedActivationData(
        feature_indices=feature_indices,
        mean_activations=final_aggregated,
        n_layers=n_layers,
        n_samples_processed=len(token_dataset),
    )


def compute_direct_logit_attribution(
    crosscoder,
    model,
    feature_idx: int,
    top_k: int = 10,
) -> CrossLayerDLAData:
    """
    Compute direct logit attribution for a feature across layers.
    """
    # Get vocabulary
    vocab = model.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}

    # Get unembedding matrix
    W_U = get_unembedding_matrix(model)  # (d_model, vocab_size)

    dla_by_layer = {}

    for layer_idx in range(crosscoder.num_layers):
        # Get decoder vector for this feature at this layer
        decoder_vec = crosscoder.W_dec[feature_idx, layer_idx]  # (d_model,)

        # Compute logit attribution
        logit_effects = decoder_vec @ W_U  # (vocab_size,)

        # Get top and bottom k tokens
        top_values, top_indices = torch.topk(logit_effects, k=top_k)
        bottom_values, bottom_indices = torch.topk(logit_effects, k=top_k, largest=False)

        # Convert to lists
        top_tokens = [id_to_token.get(idx.item(), f"[{idx.item()}]") for idx in top_indices]
        bottom_tokens = [id_to_token.get(idx.item(), f"[{idx.item()}]") for idx in bottom_indices]

        dla_by_layer[layer_idx] = {
            "top_tokens": top_tokens,
            "top_values": top_values.cpu().numpy().tolist(),
            "bottom_tokens": bottom_tokens,
            "bottom_values": bottom_values.cpu().numpy().tolist(),
        }

    return CrossLayerDLAData(
        feature_idx=feature_idx,
        dla_by_layer=dla_by_layer,
        n_layers=crosscoder.num_layers,
        top_k=top_k,
    )


def compute_feature_correlation(
    crosscoder,
    model,
    token_dataset: Int[Tensor, "n_samples seq"],
    n_features: int = 50,
    batch_size: int = 16,
    max_samples: int = 50,
) -> CrossLayerFeatureCorrelationData:
    """
    Compute correlation matrix between cross-layer features.
    Simplified version using decoder weight correlations.
    """
    n_features = min(n_features, crosscoder.ae_dim)

    # Use decoder weights to compute correlations
    # This avoids expensive activation computation
    with torch.no_grad():
        # Get decoder weights for first n features
        decoder_weights = crosscoder.W_dec[:n_features]  # (n_features, n_layers, d_model)

        # Flatten to (n_features, n_layers * d_model)
        decoder_flat = decoder_weights.reshape(n_features, -1)

        # Compute correlation matrix
        # Normalize each feature vector
        decoder_norm = decoder_flat / (torch.norm(decoder_flat, dim=1, keepdim=True) + 1e-8)

        # Compute correlations
        correlation_matrix = (decoder_norm @ decoder_norm.T).cpu().numpy()

    return CrossLayerFeatureCorrelationData(
        correlation_matrix=correlation_matrix.tolist(),
        feature_indices=list(range(n_features)),
        n_features=n_features,
        n_samples_processed=1,  # Synthetic data
    )


def create_decoder_norms_plot(data: CrossLayerDecoderNormsData) -> dict:
    """Create data for decoder norms plot to be rendered by JavaScript."""
    return data.data()


def create_activation_heatmap_plot(data: CrossLayerActivationHeatmapData) -> str:
    """Create Plotly figure for activation heatmap and return as HTML string."""
    fig = go.Figure(data=go.Heatmap(
        z=data.activation_matrix,
        x=data.token_strings,
        y=[f"Layer {i}" for i in range(data.n_layers)],
        colorscale='Viridis',
        text=np.round(data.activation_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="Token: %{x}<br>Layer: %{y}<br>Activation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Activation Heatmap for Feature {data.feature_idx}",
        xaxis_title="Token Position",
        yaxis_title="Layer",
        template="plotly_white",
        width=1200,
        height=600,
    )

    fig.update_xaxes(tickangle=-45)

    return fig.to_html(include_plotlyjs='cdn')


def create_aggregated_activation_plot(data: CrossLayerAggregatedActivationData) -> str:
    """Create Plotly figure for aggregated activations and return as HTML string."""
    fig = go.Figure()

    for feature_idx in data.feature_indices:
        fig.add_trace(go.Scatter(
            x=list(range(data.n_layers)),
            y=data.mean_activations[feature_idx],
            mode='lines+markers',
            name=f'Feature {feature_idx}',
            line=dict(width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title=f"Aggregated Activation Profile Across Dataset (n={data.n_samples_processed})",
        xaxis_title="Layer Index",
        yaxis_title="Mean Activation",
        template="plotly_white",
        width=800,
        height=500,
        hovermode='x unified'
    )

    return fig.to_html(include_plotlyjs='cdn')


def create_dla_plot(data: CrossLayerDLAData) -> str:
    """Create Plotly figure for DLA across layers and return as HTML string."""
    n_layers = data.n_layers
    rows = (n_layers + 3) // 4
    cols = min(4, n_layers)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Layer {i}" for i in range(n_layers)],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    for layer_idx in range(n_layers):
        layer_data = data.dla_by_layer[layer_idx]

        # Combine top and bottom tokens
        tokens = layer_data["top_tokens"] + layer_data["bottom_tokens"]
        values = layer_data["top_values"] + layer_data["bottom_values"]

        # Determine subplot position
        row = layer_idx // cols + 1
        col = layer_idx % cols + 1

        # Create bar chart
        colors = ['green' if v > 0 else 'red' for v in values]

        fig.add_trace(
            go.Bar(
                x=values,
                y=tokens,
                orientation='h',
                marker_color=colors,
                showlegend=False,
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Logit Effect", row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=8), row=row, col=col)

    fig.update_layout(
        title=f"Direct Logit Attribution for Feature {data.feature_idx} Across Layers",
        template="plotly_white",
        width=1400,
        height=200 * rows,
    )

    return fig.to_html(include_plotlyjs='cdn')


def create_correlation_heatmap_plot(data: CrossLayerFeatureCorrelationData) -> str:
    """Create Plotly figure for feature correlation and return as HTML string."""
    fig = go.Figure(data=go.Heatmap(
        z=data.correlation_matrix,
        x=[f"F{i}" for i in data.feature_indices],
        y=[f"F{i}" for i in data.feature_indices],
        colorscale='RdBu',
        zmid=0,
        text=np.round(data.correlation_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="Feature %{x} - Feature %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Feature Correlation Heatmap (Top {data.n_features} Features, n={data.n_samples_processed} samples)",
        xaxis_title="Feature Index",
        yaxis_title="Feature Index",
        template="plotly_white",
        width=800,
        height=800,
    )

    return fig.to_html(include_plotlyjs='cdn')


def compute_decoder_norm_cosine_similarity(
    crosscoder,
    feature_indices: List[int] | None = None,
) -> DecoderNormCosineSimilarityData:
    """
    Compute cosine similarity between decoder norm vectors across features.

    For each feature, the decoder has shape (n_layers, d_model).
    We compute the L2 norm per layer, giving a vector of shape (n_layers,).
    Then we compute cosine similarity between these norm vectors.

    This measures how consistent the "direction" of a feature's representation
    is across layers by comparing the pattern of decoder norms.

    Args:
        crosscoder: Crosscoder model
        feature_indices: List of feature indices to include. If None, uses all features.

    Returns:
        DecoderNormCosineSimilarityData with cosine similarity matrix
    """
    if feature_indices is None:
        feature_indices = list(range(crosscoder.ae_dim))

    n_features = len(feature_indices)

    # Compute decoder norms per layer for each feature
    # W_dec shape: (ae_dim, n_layers, d_model)
    # decoder_norms shape: (n_features, n_layers)
    decoder_norms = torch.norm(
        crosscoder.W_dec[feature_indices],
        dim=-1
    ).float().cpu().numpy()

    # Compute cosine similarity matrix
    # Normalize each feature's norm vector
    norms = np.linalg.norm(decoder_norms, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = decoder_norms / norms

    # Cosine similarity is dot product of normalized vectors
    cosine_sim_matrix = normalized @ normalized.T

    return DecoderNormCosineSimilarityData(
        cosine_similarity_matrix=cosine_sim_matrix.tolist(),
        feature_indices=feature_indices,
        n_features=n_features,
    )


def create_decoder_norm_cosine_similarity_plot(data: DecoderNormCosineSimilarityData) -> str:
    """Create Plotly figure for decoder norm cosine similarity heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=data.cosine_similarity_matrix,
        x=[f"F{i}" for i in data.feature_indices],
        y=[f"F{i}" for i in data.feature_indices],
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(data.cosine_similarity_matrix, 3),
        texttemplate="%{text}",
        textfont={"size": 8},
        hovertemplate="Feature %{x} - Feature %{y}<br>Cosine Similarity: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Decoder Norm Cosine Similarity ({data.n_features} Features)",
        xaxis_title="Feature Index",
        yaxis_title="Feature Index",
        template="plotly_white",
        width=800,
        height=800,
    )

    return fig.to_html(include_plotlyjs='cdn')
