# Crosscoder Visualization with NNsight

This is an adapted version of the [sae-vis](https://github.com/ckkissane/sae_vis/tree/crosscoder-vis) library that has been modified to work with NNsight instead of TransformerLens.

## Overview

The crosscoder-vis library provides interactive HTML visualizations for analyzing crosscoder (cross-layer sparse autoencoder) features. This adapted version integrates with NNsight for model interaction and supports the crosscoder architecture used in this project.

## Key Changes from Original

1. **NNsight Integration**: Replaced TransformerLens with NNsight for model loading and activation extraction
2. **Crosscoder Support**: Adapted to work with the crosscoder architecture (cross-layer sparse autoencoders)
3. **Simplified Interface**: Streamlined to work with saved model checkpoints

## Installation

The required dependencies should already be installed if you're using this project. The main dependencies are:

- `nnsight`: For model interaction
- `torch`: For deep learning operations
- `datasets`: For loading text data
- `jaxtyping`: For type annotations

## Usage

### Basic Usage

```python
python generate_dashboard.py path/to/checkpoint.pt -o dashboard.html
```

### Advanced Usage

```python
python generate_dashboard.py path/to/checkpoint.pt \
    -o dashboard.html \
    -n 50 \
    -s 100 \
    --hook-point blocks.6.hook_resid_post
```

### Parameters

- `checkpoint`: Path to the crosscoder checkpoint (.pt file)
- `-o, --output`: Output HTML file path (default: dashboard.html)
- `-n, --n-features`: Number of features to visualize (default: 50)
- `-s, --n-samples`: Number of text samples to use (default: 100)
- `--hook-point`: Hook point for feature extraction (default: blocks.6.hook_resid_post)

### Example Usage Script

Run the example script to generate a dashboard from the most recent checkpoint:

```python
python example_usage.py
```

## Dashboard Features

The generated dashboard provides:

1. **Feature Tables**: Show which neurons/features are most correlated with each crosscoder feature
2. **Activation Histograms**: Display the distribution of feature activations
3. **Logit Effects**: Show how features affect token predictions
4. **Sequence Visualization**: Interactive sequences showing where features activate

## File Structure

```
crosscoder-vis/
├── sae_vis/
│   ├── data_config_classes.py    # Configuration classes
│   ├── data_fetching_fns.py      # Data collection functions (adapted for NNsight)
│   ├── data_storing_fns.py       # Data storage and HTML generation
│   ├── model_fns.py              # Model wrapper classes (NNsightWrapper)
│   ├── utils_fns.py              # Utility functions
│   ├── html_fns.py               # HTML generation utilities
│   ├── html/                     # HTML templates
│   ├── css/                      # CSS styling
│   └── js/                       # JavaScript for interactivity
generate_dashboard.py              # Main dashboard generator script
example_usage.py                   # Example usage script
```

## Key Adaptations

### NNsightWrapper Class

The `NNsightWrapper` class in `model_fns.py` provides a standardized interface for NNsight models:

```python
class NNsightWrapper(nn.Module):
    def __init__(self, model: LanguageModel, hook_point: str):
        # Initialize with NNsight model and hook point
        
    def forward(self, tokens, return_logits=True):
        # Extract activations using NNsight tracing
        with self.model.trace(tokens) as tracer:
            # Get activations based on hook_point
            activation = self.model.transformer.h[layer].output[0].save()
            residual = self.model.transformer.h[-1].output[0].save()
```

### Crosscoder Integration

The adapted code works with the crosscoder architecture:

- **Cross-layer weights**: `W_enc` has shape `(n_layers, d_model, d_hidden)`
- **Cross-layer decoding**: `W_dec` has shape `(d_hidden, n_layers, d_model)`
- **Dual model support**: Can visualize features across different model variants

## Troubleshooting

1. **Memory Issues**: Reduce `n_features` or `n_samples` if you encounter out-of-memory errors
2. **Slow Generation**: The dashboard generation can be slow for large numbers of features. Start with smaller values.
3. **Checkpoint Loading**: Ensure the checkpoint file has a corresponding `_cfg.json` file in the same directory

## Performance Tips

- Start with a small number of features (e.g., 10-20) for faster generation
- Use fewer text samples for initial testing
- The dashboard is interactive, so you can explore different features once generated

## Output

The generated HTML dashboard can be opened in any modern web browser and provides:

- Interactive feature exploration
- Hover effects for detailed information
- Sortable tables and histograms
- Responsive design for different screen sizes

## Notes

This adaptation maintains the core functionality of the original sae-vis library while making it compatible with NNsight and the crosscoder architecture. The visualizations provide the same rich insights into feature behavior and interactions.