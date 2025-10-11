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
import wandb


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
            self.resid: int = 768
            self.num_layers: int = 12
        elif cfg.model == "gemma2-2b":
            self.resid = 2048
            self.num_layers = 32
        else:
            raise ValueError(f"Model {cfg.model} not supported")

        self.W_enc = nn.Parameter(
            torch.empty(self.num_layers, self.resid, self.ae_dim, dtype=self.dtype)
        )

        self.W_dec = nn.Parameter(
            torch.empty(self.ae_dim, self.num_layers, self.resid, dtype=self.dtype)
        )

        n = self.resid
        m = self.cfg.ae_dim
        bound = 1.0 / math.sqrt(n)
        scale = n / m
        torch.nn.init.uniform_(self.W_dec, -bound, bound)

        self.W_enc.data = (
            einops.rearrange(
                self.W_dec.data.clone(),
                "ae_dim num_layers resid -> num_layers resid ae_dim",
            )
            * scale
        )

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.resid), dtype=self.dtype)
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

    def get_loss(self, x, debug=False):
        acts = self.encode(x)
        reconstructed = self.decode(acts)

        if debug:
            print(f"Input x shape: {x.shape}")
            print(f"Acts shape: {acts.shape}")
            print(f"Reconstructed shape: {reconstructed.shape}")

        x_float = x.float()
        reconstructed_float = reconstructed.float()
        acts_float = acts.float()

        squared_diff = (x_float - reconstructed_float).pow(2).sum(dim=(-2, -1)).mean()

        decoder_norms = self.W_dec.norm(dim=-1)
        decoder_norms_summed = decoder_norms.sum(dim=1)

        reg_term = (acts_float * decoder_norms_summed).sum(dim=-1).mean()

        if debug:
            print(f"Squared diff (recon loss): {squared_diff.item():.4f}")
            print(f"Reg term (sparsity loss): {reg_term.item():.4f}")
            print(f"decoder_norms shape: {decoder_norms.shape}")
            print(f"decoder_norms_summed shape: {decoder_norms_summed.shape}")

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

        self.layer_means = None
        self.layer_stds = None

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

        with self.model.trace(tokens):
            layer_acts = [
                self.model.transformer.h[i].output[0].save()
                for i in range(self.num_layers)
            ]

        all_acts = torch.stack(layer_acts, dim=2)

        all_acts = all_acts.reshape(-1, self.num_layers, self.resid)

        if self.layer_means is None:
            self.layer_means = all_acts.mean(dim=0, keepdim=True)
            self.layer_stds = all_acts.std(dim=0, keepdim=True) + 1e-8
            print(
                f"  Mean range: [{self.layer_means.min():.4f}, {self.layer_means.max():.4f}]"
            )
            print(
                f"  Std range: [{self.layer_stds.min():.4f}, {self.layer_stds.max():.4f}]"
            )

        all_acts = (all_acts - self.layer_means) / self.layer_stds

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
    def __init__(self, cfg: cc_config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        print("Initializing crosscoder model...")
        self.crosscoder = Crosscoder_Model(cfg).to(cfg.device)

        print(f"Loading {cfg.model}...")
        if cfg.model == "gpt2":
            self.lm = LanguageModel("openai-community/gpt2", device_map=cfg.device)
        elif cfg.model == "gemma2-2b":
            self.lm = LanguageModel("google/gemma-2-2b", device_map=cfg.device)
        else:
            raise ValueError(f"Model {cfg.model} not supported")

        print("Initializing buffer...")
        self.buffer = Buffer(cfg, self.lm)

        if cfg.optim == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.crosscoder.parameters(), lr=cfg.lr, betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Optimizer {cfg.optim} not supported")

        # Initialize wandb
        wandb.init(
            project="crosscoder",
            config=cfg.__dict__,
            name=f"{cfg.model}_ae{cfg.ae_dim}",
        )

        self.step = 0

    def calculate_sparsity(self, acts: torch.Tensor) -> float:
        """Calculate L0 sparsity (percentage of active features)"""
        active = (acts > 0).float().sum(dim=-1).mean()
        return active.item()

    def train_step(self):
        self.crosscoder.train()

        batch = self.buffer.next()  # [batch_size, num_layers, resid]

        debug = self.step == 0
        loss_out = self.crosscoder.get_loss(batch, debug=debug)

        self.optimizer.zero_grad()
        loss_out.loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acts = self.crosscoder.encode(batch)
            sparsity = self.calculate_sparsity(acts)

        return {
            "loss": loss_out.loss.item(),
            "recon_loss": loss_out.recon_loss.item(),
            "sparsity_loss": loss_out.sparsity_loss.item(),
            "l0_sparsity": sparsity,
        }

    def train(self):
        print(f"Starting training for {self.cfg.steps} steps...")

        for step in range(self.cfg.steps):
            self.step = step

            metrics = self.train_step()

            if step % self.cfg.log_interval == 0:
                wandb.log(metrics, step=step)

                if self.cfg.verbose:
                    print(
                        f"Step {step}/{self.cfg.steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Recon: {metrics['recon_loss']:.4f} | "
                        f"Sparsity Loss: {metrics['sparsity_loss']:.4f} | "
                        f"L0: {metrics['l0_sparsity']:.1f}"
                    )

            if step % self.cfg.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        print("Training complete!")
        wandb.finish()

    def save_checkpoint(self, step: int):
        save_dir = Path("checkpoints")
        save_dir.mkdir(exist_ok=True)

        checkpoint_path = save_dir / f"crosscoder_step_{step}.pt"

        torch.save(
            {
                "step": step,
                "model_state_dict": self.crosscoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            checkpoint_path,
        )

        print(f"Saved checkpoint to {checkpoint_path}")
