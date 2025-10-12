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
from torch.nn.utils import clip_grad_norm_


@dataclass
class cc_config:
    # crosscoder config:
    model: str = "gpt2"
    ae_dim: int = 2**15
    model_batch: int = 128
    init_norm: float = 0.008

    # Train
    optim: str = "AdamW"
    lr: float = 5e-5
    steps: int = 50000
    batch_size: int = 2048
    warmup_steps: int = 5000
    l1_coeff: float = 0.8

    # wandb
    log_interval: int = 100
    save_interval: int = 5000

    # buffer
    buffer_mult: int = 64

    # other
    dtype = torch.float32
    device: str = "cuda"
    seed: int = 721

    # anthropic jan 2025 update config:
    l_s: float = 10
    l_p: float = 3e-6
    c: float = 4.0


class LossOut(NamedTuple):
    loss: torch.Tensor
    sparsity_loss: torch.Tensor
    recon_loss: torch.Tensor


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        threshold = log_threshold.exp()
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output

        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None


class RectangleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


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

        torch.nn.init.normal_(self.W_dec, std=0.1)
        self.W_dec.data = (
            self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True) * cfg.init_norm
        )

        effective_fan_in = self.num_layers * self.resid
        k_bound = math.sqrt(2.0) * math.sqrt(3.0 / effective_fan_in)
        torch.nn.init.uniform_(self.W_enc, -k_bound, k_bound)

        self.b_enc = nn.Parameter(torch.zeros(self.ae_dim, dtype=self.dtype))
        self.b_dec = nn.Parameter(
            torch.zeros((self.num_layers, self.resid), dtype=self.dtype)
        )

        self.log_threshold = nn.Parameter(
            torch.full((self.ae_dim,), 0.1, dtype=self.dtype)
        )

    def encode(self, x):
        x_enc = einops.einsum(
            x, self.W_enc, "... n_layers resid, n_layers resid ae_dim -> ... ae_dim"
        )

        preacts = x_enc + self.b_enc
        acts = JumpReLUFunction.apply(preacts, self.log_threshold, 2.0)

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

        if self.layer_stds is None:
            # Only calculate standard deviation, not the mean
            self.layer_stds = all_acts.std(dim=0, keepdim=True) + 1e-8
            print(
                f"  Std range: [{self.layer_stds.min():.4f}, {self.layer_stds.max():.4f}]"
            )

        all_acts = all_acts / self.layer_stds

        self.buffer[: len(all_acts)] = all_acts[: self.buffer_size]

        perm = torch.randperm(self.buffer_size, device=self.cfg.device)
        self.buffer = self.buffer[perm]

    @torch.no_grad()
    def next(self):
        if self.pointer + self.cfg.batch_size > self.buffer_size:
            self.refresh()

        batch = self.buffer[self.pointer : self.pointer + self.cfg.batch_size]
        self.pointer += self.cfg.batch_size
        return batch


class Trainer:
    def __init__(self, cfg: cc_config):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        self.run_dir = self._get_next_run_dir()
        print(f"Saving checkpoints to: {self.run_dir}")

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
                self.crosscoder.parameters(),
                lr=cfg.lr,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Optimizer {cfg.optim} not supported")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: self._get_lr_multiplier(step)
        )
        self.steps = cfg.steps

        wandb.init(
            project="crosscoder",
            config=cfg.__dict__,
            name=f"{cfg.model}_ae{cfg.ae_dim}",
        )
        wandb.watch(self.crosscoder, log="all", log_freq=self.cfg.log_interval)

        self.step = 0

    def _get_next_run_dir(self) -> Path:
        """Find the next available run_n directory"""
        checkpoints_dir = Path("./checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)

        # Find all existing run directories
        existing_runs = [
            d
            for d in checkpoints_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]

        if not existing_runs:
            run_num = 0
        else:
            run_numbers = []
            for run_dir in existing_runs:
                try:
                    num = int(run_dir.name.split("_")[1])
                    run_numbers.append(num)
                except (IndexError, ValueError):
                    continue

            run_num = max(run_numbers) + 1 if run_numbers else 0

        run_dir = checkpoints_dir / f"run_{run_num}"
        run_dir.mkdir(exist_ok=True)

        return run_dir

    def _get_lr_multiplier(self, step: int) -> float:
        if step < self.cfg.warmup_steps:
            return step / self.cfg.warmup_steps
        return 1.0

    def get_l1_coeff(self):
        if self.step < 0.05 * self.steps:
            return self.cfg.l1_coeff * self.step / (0.05 * self.steps)
        else:
            return self.cfg.l1_coeff

    def calculate_sparsity(self, acts: torch.Tensor):
        active = (acts > 0).float().sum(dim=-1).mean()
        return active.item()

    def calculate_dead_features(self, acts: torch.Tensor):
        num_dead = (acts.sum(dim=0) == 0).sum().item()
        return num_dead

    def train_step(self):
        self.crosscoder.train()

        batch = self.buffer.next()  # [batch_size, num_layers, resid]

        debug = self.step == 0
        loss_out = self.crosscoder.get_loss(batch, debug=debug)
        loss = loss_out.recon_loss + self.get_l1_coeff() * loss_out.sparsity_loss

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            acts = self.crosscoder.encode(batch)
            sparsity = self.calculate_sparsity(acts)
            dead_features = self.calculate_dead_features(acts)

        return {
            "losses/loss": loss.item(),
            "losses/recon_loss": loss_out.recon_loss.item(),
            "losses/sparsity_loss": loss_out.sparsity_loss.item(),
            "stats/l0_sparsity": sparsity,
            "dead_features": dead_features,
            "hyperparams/lr": self.scheduler.get_last_lr()[0],
            "hyperparams/l1_coeff": self.get_l1_coeff(),
            "stats/W_dec_norm": self.crosscoder.W_dec.norm(),
        }

    def train(self):
        print(f"Starting training for {self.cfg.steps} steps...")

        for step in range(self.cfg.steps):
            self.step = step

            metrics = self.train_step()

            if step % self.cfg.log_interval == 0:
                wandb.log(metrics, step=step)

                (
                    print(
                        f"Step {step}/{self.cfg.steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Recon: {metrics['recon_loss']:.4f} | "
                        f"Sparsity Loss: {metrics['sparsity_loss']:.4f} | "
                        f"L0: {metrics['l0_sparsity']:.1f} | "
                        f"Dead Features: {metrics['dead_features']:.1f}"
                    ),
                )

            if step % self.cfg.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        print("Training complete!")
        wandb.finish()

    def save_checkpoint(self, step: int):
        print("Saving checkpoint...")
        checkpoint_path = self.run_dir / f"crosscoder_step_{step}.pt"

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
