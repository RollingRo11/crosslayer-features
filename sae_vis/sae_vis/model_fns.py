import einops
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from jaxtyping import Float
from torch import Tensor
import nnsight
from nnsight import LanguageModel

from .utils_fns import VocabType
from .model_utils import get_unembedding_matrix
import sys
sys.path.append('..')
from crosscoder.crosscoder import Crosscoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_resid_dir(
    dir: Float[Tensor, "feats d"],
    crosscoder: "Crosscoder",
    model: LanguageModel,
    input: bool = False,
):
    """
    For crosscoders, the feature directions are already in the residual stream space,
    so this is essentially an identity function. The crosscoder operates across layers
    rather than within specific components.
    """
    # For crosscoders, directions are already in residual stream space
    return dir


def resid_final_pre_layernorm_to_logits(x: Tensor, model: LanguageModel):
    # Access model's final layer norm and unembedding through NNsight
    config = model.config.to_dict()
    eps = config.get('layer_norm_epsilon', 1e-5)

    x = x - x.mean(-1, keepdim=True)  # [batch, pos, length]
    scale = x.pow(2).mean(-1, keepdim=True) + eps
    x_normalized = x / scale

    # Get unembedding weights through model
    with torch.no_grad():
        W_U = get_unembedding_matrix(model)
        return x_normalized @ W_U


def load_othello_vocab() -> dict[VocabType, dict[int, str]]:
    """
    Returns vocab dicts (embedding and unembedding) for OthelloGPT, i.e. token_id -> token_str.

    This means ["pass"] + ["A0", "A1", ..., "H7"].

    If probes=True, then this is actually the board squares (including middle ones)
    """

    all_squares = [r + c for r in "ABCDEFGH" for c in "01234567"]
    legal_squares = [sq for sq in all_squares if sq not in ["D3", "D4", "E3", "E4"]]

    vocab_dict_probes = {
        token_id: str_token for token_id, str_token in enumerate(all_squares)
    }
    vocab_dict = {
        token_id: str_token
        for token_id, str_token in enumerate(["pass"] + legal_squares)
    }
    return {
        "embed": vocab_dict,
        "unembed": vocab_dict,
        "probes": vocab_dict_probes,
    }


def load_crosscoder_checkpoint(checkpoint_path: str, device: str) -> tuple[Crosscoder, LanguageModel]:
    """
    Load a crosscoder checkpoint and the associated model.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config from checkpoint
    cc_config = checkpoint['cc_config']
    cc_config['device'] = device

    # Create crosscoder instance
    crosscoder = Crosscoder(cc_config)

    # Load state dict
    crosscoder.load_state_dict(checkpoint['cc_state_dict'])

    # Get the model (already loaded in crosscoder)
    model = crosscoder.model

    return crosscoder, model


def tokenize_dataset(dataset_name: str, model: LanguageModel, seq_len: int, num_samples: int = 10000):
    """Tokenize a dataset for use with the crosscoder visualization."""
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Get tokenizer
    tokenizer = model.tokenizer

    # Tokenize samples
    all_tokens = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break

        # Get text from sample
        if 'text' in sample:
            text = sample['text']
        elif 'content' in sample:
            text = sample['content']
        else:
            # Try to find any text field
            text = str(list(sample.values())[0])

        # Tokenize
        tokens = tokenizer.encode(text, max_length=seq_len, truncation=True)
        if len(tokens) == seq_len:
            all_tokens.append(tokens)

    # Convert to tensor
    return torch.tensor(all_tokens, dtype=torch.long)
