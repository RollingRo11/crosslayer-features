import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class cc_config:
    # crosscoder config:
    model: str = "gpt2"
    ae_dim: int = 2**15
    model_batch: int = 32
    init_norm: float = 0.08

    # Train
    optim: str = "AdamW"
    lr: float = 5e-5
    steps: int = 50000
    batch_size: int = 2048

    # wandb
    log_interval: int = 100
    save_interval: int = 20000

    # other
    dtype = torch.bfloat16
    device: str = "cuda"
    seed: int = 11
    verbose: bool = True

    # anthropic jan 2025 update config:
    is_jan: bool = False  # are we using the Anthropic Jan 2025 update?
    l_s: float = 10
    l_p: float = 3e-6
    c: float = 4.0


class Crosscoder_Model(nn.Module):
    def __init__(self, cfg: cc_config):
        self.cfg: cc_config = cfg
        self.model: str = cfg.model
        self.ae_dim: int = cfg.ae_dim

        if cfg.model == "gpt2":
            self.model_embd = 768
        if cfg.model == "gemma2-2b":
            self.model_embd = 2048

        n, m = 0, 0

        if self.cfg.is_jan:
            n = self.model_embd
            m = self.cfg.ae_dim

        self.W_dec =



class Buffer(nn.Module):
