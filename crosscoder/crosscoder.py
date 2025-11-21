import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple

import einops
import torch
import torch.nn as nn
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import LanguageModel
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

import wandb

load_dotenv()


@dataclass
class cc_config:
    # crosscoder config:
    model: str = "gemma2-2b"
    ae_dim: int = 2**15
    model_batch: int = 4
    init_norm: float = 0.008

    # Train
    optim: str = "AdamW"
    lr: float = 2e-4
    steps: int = 50000
    batch_size: int = 4096
    warmup_steps: int = 5000
    l1_coeff: float = 0.8
    gradient_accumulation_steps: int = 1

    # wandb
    log_interval: int = 100
    save_interval: int = 10000

    # buffer
    buffer_mult: int = 128  # Total tokens = batch_size * buffer_mult
    context_length: int = 256  # Reduced context for efficiency

    # other
    dtype = torch.bfloat16
    device: str = "cuda"
    seed: int = 721

    # anthropic jan 2025 update config:
    l_s: float = 10
    l_p: float = 0.002
    c: float = 4.0


class LossOut(NamedTuple):
    loss: torch.Tensor
    sparsity_loss: torch.Tensor
    recon_loss: torch.Tensor
    preact_loss: torch.Tensor


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        threshold = log_threshold.exp()
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).to(x.dtype) * grad_output

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
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

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
        self.dtype = cfg.dtype

        if cfg.model == "gpt2":
            self.resid: int = 768
            self.num_layers: int = 12
        elif cfg.model == "gemma2-2b":
            self.resid = 2304
            self.num_layers = 26
        else:
            raise ValueError(f"Model {cfg.model} not supported")

        self.ae_dim = cfg.ae_dim

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
            torch.full((self.ae_dim,), math.log(0.1), dtype=self.dtype)
        )

    def encode(self, x, return_preacts=False):
        x_enc = einops.einsum(
            x, self.W_enc, "... n_layers resid, n_layers resid ae_dim -> ... ae_dim"
        )

        preacts = x_enc + self.b_enc
        acts = JumpReLUFunction.apply(preacts, self.log_threshold, 0.1)

        if return_preacts:
            return acts, preacts
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
        acts, preacts = self.encode(x, return_preacts=True)
        reconstructed = self.decode(acts)

        x_float = x.float()
        reconstructed_float = reconstructed.float()
        squared_diff = (x_float - reconstructed_float).pow(2).sum(dim=(-2, -1)).mean()

        decoder_norms = self.W_dec.norm(dim=-1)  # [ae_dim, n_layers]
        decoder_norms_summed = decoder_norms.sum(dim=1)  # [ae_dim]

        reg_term = (acts.float() * decoder_norms_summed).sum(dim=-1).mean()

        thresholds = self.log_threshold.exp()
        preact_loss_per_feature = F.relu(thresholds - preacts)
        preact_loss = (
            (preact_loss_per_feature * decoder_norms_summed).sum(dim=-1).mean()
        )

        loss = (
            squared_diff + (self.cfg.l1_coeff * reg_term) + (self.cfg.l_p * preact_loss)
        )

        return LossOut(
            loss=loss,
            sparsity_loss=reg_term,
            recon_loss=squared_diff,
            preact_loss=preact_loss,
        )


class Buffer:
    def __init__(self, cfg: cc_config, model: LanguageModel):
        self.model = model
        self.cfg = cfg
        self.modelcfg = self.model.config.to_dict()

        if cfg.model == "gpt2":
            self.num_layers = self.modelcfg["n_layer"]
            self.resid = self.modelcfg["n_embd"]
        elif cfg.model == "gemma2-2b":
            self.num_layers = self.modelcfg["num_hidden_layers"]
            self.resid = self.modelcfg["hidden_size"]

        self.context = cfg.context_length

        self.total_tokens = self.cfg.batch_size * self.cfg.buffer_mult

        self.buffer = torch.zeros(
            (self.total_tokens, self.num_layers, self.resid),
            dtype=self.cfg.dtype,
            device="cpu",
        )
        self.pointer = 0

        self.dataset = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )
        self.dataset_iter = iter(self.dataset)

        self.layer_stds = None
        self.refresh()

    def get_batch_data(self, n_samples):
        tokens_list = []
        masks_list = []

        while len(tokens_list) < n_samples:
            try:
                item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                item = next(self.dataset_iter)

            text = item["text"]
            if len(text.strip()) < 50:
                continue

            out = self.model.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.context,
                truncation=True,
                padding="max_length",
            )

            tokens_list.append(out["input_ids"])
            masks_list.append(out["attention_mask"])

        return torch.cat(tokens_list, dim=0), torch.cat(masks_list, dim=0)

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer...")
        self.pointer = 0

        tokens_collected = 0

        while tokens_collected < self.total_tokens:
            current_batch_size = self.cfg.model_batch

            input_ids, attention_mask = self.get_batch_data(current_batch_size)
            input_ids = input_ids.to(self.cfg.device)
            attention_mask = attention_mask.to(self.cfg.device)

            with self.model.trace(input_ids):
                if self.cfg.model == "gpt2":
                    layer_acts = [
                        self.model.transformer.h[i].output[0].save()
                        for i in range(self.num_layers)
                    ]
                elif self.cfg.model == "gemma2-2b":
                    layer_acts = [
                        self.model.model.layers[i].output[0].save()
                        for i in range(self.num_layers)
                    ]

            all_acts = torch.stack(layer_acts, dim=2)

            valid_mask = attention_mask.bool().view(-1)

            flat_acts = all_acts.view(-1, self.num_layers, self.resid)

            valid_acts = flat_acts[valid_mask]

            if self.layer_stds is None:
                # Normalize across the batch dimension (0)
                self.layer_stds = valid_acts.std(dim=0, keepdim=True) + 1e-8
                self.layer_stds = self.layer_stds.to(self.cfg.dtype)
                print(
                    f"Calculated STD. Range: {self.layer_stds.min():.4f} - {self.layer_stds.max():.4f}"
                )

            valid_acts = valid_acts / self.layer_stds

            remaining_space = self.total_tokens - tokens_collected
            take_n = min(remaining_space, valid_acts.shape[0])

            self.buffer[tokens_collected : tokens_collected + take_n] = valid_acts[
                :take_n
            ].cpu()

            tokens_collected += take_n
            print(f"Buffer fill: {tokens_collected}/{self.total_tokens}", end="\r")

        print("\nBuffer refreshed.")

        perm = torch.randperm(self.total_tokens)
        self.buffer = self.buffer[perm]

    @torch.no_grad()
    def next(self):
        if self.pointer + self.cfg.batch_size > self.total_tokens:
            self.refresh()

        batch = self.buffer[self.pointer : self.pointer + self.cfg.batch_size]
        self.pointer += self.cfg.batch_size
        return batch.to(self.cfg.device)


class Trainer:
    def __init__(self, cfg: cc_config):
        self.cfg: cc_config = cfg
        torch.manual_seed(cfg.seed)

        self.run_dir = self._get_next_run_dir()
        print(f"Saving checkpoints to: {self.run_dir}")

        print("Initializing crosscoder model...")
        self.crosscoder = Crosscoder_Model(cfg).to(cfg.device)

        print(f"Loading {cfg.model}...")
        if cfg.model == "gpt2":
            self.lm = LanguageModel("openai-community/gpt2", device_map=cfg.device)
        elif cfg.model == "gemma2-2b":
            self.lm = LanguageModel(
                "google/gemma-2-2b",
                device_map=cfg.device,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )

        if self.lm.tokenizer.pad_token is None:
            self.lm.tokenizer.pad_token = self.lm.tokenizer.eos_token

        print("Initializing buffer...")
        self.buffer = Buffer(cfg, self.lm)

        self.optimizer = torch.optim.AdamW(
            self.crosscoder.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: self._get_lr_multiplier(step)
        )

        # Wandb setup
        wandb.init(
            project="crosscoder",
            config=cfg.__dict__,
            name=f"{cfg.model}_ae{cfg.ae_dim}",
        )

        self.step = 0
        self.accumulation_counter = 0

    def _get_next_run_dir(self) -> Path:
        checkpoints_dir = Path("./checkpoints")
        checkpoints_dir.mkdir(exist_ok=True)
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
            return (step + 1) / self.cfg.warmup_steps
        return 1.0

    def train_step(self):
        self.crosscoder.train()

        batch = self.buffer.next()  # [batch_size, num_layers, resid]

        if self.accumulation_counter == 0:
            self.optimizer.zero_grad()

        loss_out = self.crosscoder.get_loss(batch)

        scaled_loss = loss_out.loss / self.cfg.gradient_accumulation_steps
        scaled_loss.backward()

        self.accumulation_counter += 1

        metrics = {
            "losses/loss": loss_out.loss.item(),
            "losses/recon_loss": loss_out.recon_loss.item(),
            "losses/sparsity_loss": loss_out.sparsity_loss.item(),
            "losses/preact_loss": loss_out.preact_loss.item(),
        }

        if self.accumulation_counter >= self.cfg.gradient_accumulation_steps:
            clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.accumulation_counter = 0

            if self.step % 10 == 0:
                with torch.no_grad():
                    acts = self.crosscoder.encode(batch)
                    metrics["stats/l0_sparsity"] = (
                        (acts > 0).float().sum(dim=-1).mean().item()
                    )
                    metrics["stats/dead_features"] = (acts.sum(dim=0) == 0).sum().item()
                    metrics["hyperparams/lr"] = self.scheduler.get_last_lr()[0]
                    metrics["stats/W_dec_norm"] = self.crosscoder.W_dec.norm().item()

        return metrics

    def train(self):
        print(f"Starting training for {self.cfg.steps} steps...")

        for step in range(self.cfg.steps):
            self.step = step
            metrics = self.train_step()

            if step % self.cfg.log_interval == 0 and self.accumulation_counter == 0:
                wandb.log(metrics, step=step)
                print(
                    f"Step {step} | Loss: {metrics['losses/loss']:.4f} | Recon: {metrics['losses/recon_loss']:.4f}"
                )

            if step % self.cfg.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        wandb.finish()

    def save_checkpoint(self, step: int):
        checkpoint_path = self.run_dir / f"crosscoder_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.crosscoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
                "layer_stds": self.buffer.layer_stds,  # Important to save normalization stats
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    cfg = cc_config()
    trainer = Trainer(cfg)
    trainer.train()
