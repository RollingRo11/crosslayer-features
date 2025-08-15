"""
Utility functions to handle different model architectures.
"""
import torch
from typing import Union, Any


def detect_model_type(model) -> str:
    """
    Detect the model type based on its attributes.

    Args:
        model: The model object (either raw model or LanguageModel wrapper)

    Returns:
        str: Model type ('gpt2', 'gpt_neox', 'gemma', 'qwen', 'llama', or 'unknown')
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    # Check for GPT-2 style models
    if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'h'):
        return 'gpt2'

    # Check for GPTNeoX style models
    if hasattr(actual_model, 'gpt_neox') and hasattr(actual_model.gpt_neox, 'layers'):
        return 'gpt_neox'

    # Check for Gemma models
    if hasattr(actual_model, 'model') and hasattr(actual_model.model, 'layers'):
        # Check if it's specifically a Gemma model
        if hasattr(actual_model, 'config') and 'gemma' in str(type(actual_model.config)).lower():
            return 'gemma'
        # Check if it's specifically a Qwen model  
        elif hasattr(actual_model, 'config') and 'qwen' in str(type(actual_model.config)).lower():
            return 'qwen'
        # General LLaMA-style architecture (includes Gemma/Qwen as fallback)
        else:
            return 'llama'

    return 'unknown'


def get_model_layers(model) -> Any:
    """
    Get the layers container for the model.

    Args:
        model: The model object

    Returns:
        The layers container (e.g., transformer.h or gpt_neox.layers)
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    model_type = detect_model_type(model)

    if model_type == 'gpt2':
        return actual_model.transformer.h
    elif model_type == 'gpt_neox':
        return actual_model.gpt_neox.layers
    elif model_type in ['llama', 'gemma', 'qwen']:
        return actual_model.model.layers
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_embedding_matrix(model) -> torch.Tensor:
    """
    Get the embedding matrix for the model.

    Args:
        model: The model object

    Returns:
        torch.Tensor: The embedding matrix
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    model_type = detect_model_type(model)

    if model_type == 'gpt2':
        return actual_model.transformer.wte.weight
    elif model_type == 'gpt_neox':
        return actual_model.gpt_neox.embed_in.weight
    elif model_type in ['llama', 'gemma', 'qwen']:
        return actual_model.model.embed_tokens.weight
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_unembedding_matrix(model) -> torch.Tensor:
    """
    Get the unembedding matrix (output projection) for the model.

    Args:
        model: The model object

    Returns:
        torch.Tensor: The unembedding matrix (transposed for logit computation)
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    # Most models have lm_head for the final projection
    if hasattr(actual_model, 'lm_head'):
        return actual_model.lm_head.weight.T  # (d_model, vocab_size)
    elif hasattr(actual_model, 'embed_out'):
        return actual_model.embed_out.weight.T
    else:
        # Fallback: try to use tied embeddings
        embedding_matrix = get_embedding_matrix(model)
        return embedding_matrix.T


def get_layer_output(model, layer_idx: int, trace_context):
    """
    Get the output of a specific layer during model tracing.

    Args:
        model: The model object (LanguageModel)
        layer_idx: Index of the layer
        trace_context: The tracing context from model.trace()

    Returns:
        The layer output tensor that can be saved
    """
    model_type = detect_model_type(model)

    if model_type == 'gpt2':
        return model.transformer.h[layer_idx].output[0].save()
    elif model_type == 'gpt_neox':
        return model.gpt_neox.layers[layer_idx].output[0].save()
    elif model_type in ['llama', 'gemma', 'qwen']:
        return model.model.layers[layer_idx].output[0].save()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_num_layers(model) -> int:
    """
    Get the number of layers in the model.

    Args:
        model: The model object

    Returns:
        int: Number of layers
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    # Try config first
    if hasattr(actual_model, 'config'):
        config = actual_model.config
        if hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            return config.n_layer
        elif hasattr(config, 'num_layers'):
            return config.num_layers

    # Fallback: count layers directly
    layers = get_model_layers(model)
    return len(layers)


def get_hidden_size(model) -> int:
    """
    Get the hidden size (d_model) of the model.

    Args:
        model: The model object

    Returns:
        int: Hidden size
    """
    # Handle LanguageModel wrapper
    if hasattr(model, '_model'):
        actual_model = model._model
    else:
        actual_model = model

    # Try config first
    if hasattr(actual_model, 'config'):
        config = actual_model.config
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        elif hasattr(config, 'd_model'):
            return config.d_model
        elif hasattr(config, 'n_embd'):
            return config.n_embd

    # Fallback: infer from embedding matrix
    embedding_matrix = get_embedding_matrix(model)
    return embedding_matrix.shape[1]
