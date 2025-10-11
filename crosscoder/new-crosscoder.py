# rewriting to be less bulky

from pyarrow import SparseCOOTensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
from typing import NamedTuple
from nnsight import LanguageModel
import einops
from datasets import load_dataset
from pathlib import Path
import sys

from crosscoder.crosscoder import LossOutput

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

    # buffer
    buffer_mult: int = 8

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

class LossOut(NamedTuple):
    loss: torch.Tensor
    sparsity_loss: torch.Tensor
    recon_loss: torch.Tensor

class Crosscoder_Model(nn.Module):
    def __init__(self, cfg: cc_config):
        super().__init__()
        self.cfg: cc_config = cfg
        self.model_name: str = cfg.model
        self.ae_dim: int = cfg.ae_dim
        self.dtype = cfg.dtype

        if cfg.model == "gpt2":
            self.model_cfg = self.model.config.to_dict()  # type: ignore
            self.resid: int = 768
            self.model: LanguageModel = LanguageModel(
                "openai-community/gpt2", device_map="auto"
            )
            self.num_layers: int = self.model_cfg["n_layer"]
        if cfg.model == "gemma2-2b":
            self.model_cfg = self.model.config.to_dict()  # type: ignore
            self.model_embd = 2048

        self.num_layers = self.model_cfg["n_layer"]
        n, m = 0, 0

        self.W_enc = nn.Parameter(
            torch.empty(self.num_layers, self.resid, self.ae_dim, dtype=self.dtype)
        )

        self.W_dec = nn.Parameter(
            torch.empty(self.ae_dim, self.num_layers, self.resid, dtype=self.dtype)
        )

        n = self.model_embd
        m = self.cfg.ae_dim
        bound = 1.0 / math.sqrt(n)
        scale = n / m
        torch.nn.init.uniform_(self.W_dec, -bound, bound)

        self.W_enc.data = (
            einops.rearrange(
                self.W_dec.data.clone(),
                "ae_dim num_layers resid -> n_layers resid ae_dim",
            )
            * scale
        )

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.model_embd), dtype=self.dtype)
        )

    def encode(self, x):
        x_enc = einops.einsum(
            x, self.W_enc, "... n_layers resid, n_layers resid ae_dim -> ... ae_dim"
        )

        acts = F.relu(x_enc + self.b_enc)

        return acts

    def decode(self, acts):
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "... ae_dim, ae_dim n_layers resid -> ... n_layers resid",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    def get_loss(self, x):
        acts = self.encode(x).float()
        reconstructed = self.decode(acts).float()

        squared_diff = (x - reconstructed).pow(2).sum(dim=(-2, -1)).mean()

        decoder_norms = self.W_dec.norm(dim=-1)
        decoder_norms_summed = decoder_norms.sum(dim=1)

        reg_term = (acts * decoder_norms_summed).sum(dim=-1).mean()

        loss = squared_diff + reg_term

        return LossOut(loss=loss, sparsity_loss=reg_term, recon_loss=squared_diff)








class Buffer:
    def __init__(self, cfg: cc_config, model: LanguageModel):
        self.model = model
        self.cfg = cfg
        self.modelcfg = self.model.config.to_dict()
        self.num_layers = self.modelcfg["n_layer"]
        self.resid = self.modelcfg["n_embd"]
        self.context = 1024

        self.buffer_size = self.cfg.batch_size * self.cfg.buffer_mult
        self.buffer = torch.zeros(
            (self.buffer_size, self.num_layers, self.resid),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        self.pointer = 0

        self.dataset = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )
        self.dataset_iter = iter(self.dataset)

        self.refresh()

    def get_tokens(self, n_samples):
        """Get tokenized batch"""
        tokens = []
        for _ in range(n_samples):
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                item = next(self.dataset_iter)

            text = item["text"]
            if len(text.strip()) < 50:
                continue

            toks = self.model.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.context,
                truncation=True,
                padding="max_length",
            )
            tokens.append(toks)

        return torch.cat(tokens, dim=0)

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer...")
        self.pointer = 0
        n_samples = self.buffer_size // self.context

        tokens = self.get_tokens(n_samples).to(self.cfg.device)

        with self.model.trace(tokens) as tracer:
            layer_acts = []
            for i in range(self.num_layers):
                acts = get_layer_output(self.model, i, tracer)
                layer_acts.append(acts)

        all_acts = torch.stack(layer_acts, dim=2)

        all_acts = all_acts.reshape(-1, self.num_layers, self.resid)
        self.buffer[: len(all_acts)] = all_acts[: self.buffer_size]

        perm = torch.randperm(self.buffer_size, device=self.cfg.device)
        self.buffer = self.buffer[perm]

    @torch.no_grad()
    def next(self):
        """Get next batch"""
        if self.pointer + self.cfg.batch_size > self.buffer_size:
            self.refresh()

        batch = self.buffer[self.pointer : self.pointer + self.cfg.batch_size]
        self.pointer += self.cfg.batch_size
        return batch


class Trainer:
    def __init__(self, cfg):
