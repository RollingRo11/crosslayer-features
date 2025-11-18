import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

import einops
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import LanguageModel
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler

import wandb

load_dotenv()


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
    save_interval: int = 10000

    # buffer
    buffer_mult: int = 64

    # other
    dtype = torch.float32
    device: str = "cuda"
    seed: int = 721

    # anthropic jan 2025 update config:
    l_s: float = 10
    l_p: float = 0.002
    c: float = 4.0

    # Multi-GPU / Distributed training parameters
    world_size: int = 1  # Total number of GPUs
    rank: int = 0  # Global rank of this process
    local_rank: int = 0  # Local rank on this node
    backend: str = "nccl"  # Distributed backend (nccl for GPU)
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    use_amp: bool = True  # Use automatic mixed precision
    fsdp_cpu_offload: bool = False  # Offload to CPU to save memory
    fsdp_min_num_params: int = 1e6  # Minimum parameters for FSDP wrapping


class LossOut(NamedTuple):
    loss: torch.Tensor
    sparsity_loss: torch.Tensor
    recon_loss: torch.Tensor
    preact_loss: torch.Tensor


# ============================================================================
# Distributed Utility Functions
# ============================================================================


def setup_distributed(cfg: cc_config) -> None:
    """Initialize distributed training environment"""
    # Get environment variables set by torchrun
    cfg.rank = int(os.environ.get("RANK", 0))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))

    if cfg.world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=cfg.backend,
            init_method="env://",
        )

        # Set device for this process
        torch.cuda.set_device(cfg.local_rank)
        cfg.device = f"cuda:{cfg.local_rank}"

        if is_main_process(cfg):
            print(f"Initialized distributed training with {cfg.world_size} GPUs")
            print(f"Backend: {cfg.backend}")
    else:
        if is_main_process(cfg):
            print("Running in single-GPU mode")


def cleanup_distributed(cfg: cc_config) -> None:
    """Clean up distributed training"""
    if cfg.world_size > 1:
        dist.destroy_process_group()


def is_main_process(cfg: cc_config) -> bool:
    """Check if this is the main process (rank 0)"""
    return cfg.rank == 0


def gather_metrics(tensor: torch.Tensor, cfg: cc_config) -> torch.Tensor:
    """Aggregate a metric tensor across all GPUs"""
    if cfg.world_size <= 1:
        return tensor

    if not tensor.is_cuda:
        tensor = tensor.cuda()

    # Average across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def barrier(cfg: cc_config) -> None:
    """Synchronization barrier across all processes"""
    if cfg.world_size > 1:
        dist.barrier()


# ============================================================================
# Custom Activation Functions (unchanged)
# ============================================================================


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


# ============================================================================
# Crosscoder Model (unchanged architecture, FSDP-compatible)
# ============================================================================


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
            self.resid = 2304
            self.num_layers = 26
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
        x_enc = einops.einsum(
            x, self.W_enc, "... n_layers resid, n_layers resid ae_dim -> ... ae_dim"
        )

        preacts = x_enc + self.b_enc

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

        thresholds = self.log_threshold.exp()
        preact_loss_per_feature = F.relu(
            thresholds - preacts
        )  # Shape: [batch_size, ae_dim]

        # This is the corrected line: sum across the feature dimension (-1), THEN average over the batch
        preact_loss = (
            (preact_loss_per_feature * decoder_norms_summed).sum(dim=-1).mean()
        )

        loss = squared_diff + reg_term

        return LossOut(
            loss=loss,
            sparsity_loss=reg_term,
            recon_loss=squared_diff,
            preact_loss=preact_loss,
        )


# ============================================================================
# Distributed Buffer
# ============================================================================


class Buffer:
    def __init__(self, cfg: cc_config, model: LanguageModel):
        self.model = model
        self.cfg = cfg
        self.modelcfg = self.model.config.to_dict()

        # Handle different config keys for different models
        if cfg.model == "gpt2":
            self.num_layers = self.modelcfg["n_layer"]
            self.resid = self.modelcfg["n_embd"]
        elif cfg.model == "gemma2-2b":
            self.num_layers = self.modelcfg["num_hidden_layers"]
            self.resid = self.modelcfg["hidden_size"]
        else:
            raise ValueError(f"Model {cfg.model} not supported")

        self.context = 1024

        # Each rank gets its own buffer
        self.buffer_size = self.cfg.batch_size * self.cfg.buffer_mult
        self.buffer = torch.zeros(
            (self.buffer_size, self.num_layers, self.resid),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )
        self.pointer = 0

        # Load dataset with rank-specific shuffling for diversity
        self.dataset = load_dataset(
            "monology/pile-uncopyrighted", split="train", streaming=True
        )

        # Skip ahead in dataset based on rank to get different data
        if cfg.world_size > 1:
            self.dataset = self.dataset.shuffle(seed=cfg.seed + cfg.rank, buffer_size=10000)

        self.dataset_iter = iter(self.dataset)

        # Normalization statistics shared across all ranks
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
        if is_main_process(self.cfg):
            print("Refreshing buffer...")

        self.pointer = 0
        n_samples = self.buffer_size // self.context

        tokens = self.get_tokens(n_samples).to(self.cfg.device)

        with self.model.trace(tokens):
            # Handle different model architectures
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
            else:
                raise ValueError(f"Model {self.cfg.model} not supported")

        all_acts = torch.stack(layer_acts, dim=2)

        all_acts = all_acts.reshape(-1, self.num_layers, self.resid)

        # Calculate normalization stats and synchronize across ranks
        if self.layer_stds is None:
            # Only calculate standard deviation, not the mean
            local_std = all_acts.std(dim=0, keepdim=True) + 1e-8

            # Synchronize normalization statistics across all ranks
            if self.cfg.world_size > 1:
                # Gather std from all ranks and average
                std_tensor = local_std.clone()
                dist.all_reduce(std_tensor, op=dist.ReduceOp.AVG)
                self.layer_stds = std_tensor
            else:
                self.layer_stds = local_std

            if is_main_process(self.cfg):
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


# ============================================================================
# Multi-GPU Trainer with FSDP
# ============================================================================


class Trainer:
    def __init__(self, cfg: cc_config):
        self.cfg: cc_config = cfg

        # Setup distributed training
        setup_distributed(cfg)

        # Set random seed (different per rank for data diversity)
        torch.manual_seed(cfg.seed + cfg.rank)

        if is_main_process(cfg):
            self.run_dir = self._get_next_run_dir()
            print(f"Saving checkpoints to: {self.run_dir}")
        else:
            self.run_dir = None

        # Ensure all ranks have the same run_dir
        if cfg.world_size > 1:
            # Broadcast run_dir from rank 0
            if is_main_process(cfg):
                run_dir_str = str(self.run_dir)
            else:
                run_dir_str = ""

            # Simple broadcast using object list
            run_dir_list = [run_dir_str] if is_main_process(cfg) else [None]
            if cfg.world_size > 1:
                dist.broadcast_object_list(run_dir_list, src=0)

            if not is_main_process(cfg):
                self.run_dir = Path(run_dir_list[0])

        if is_main_process(cfg):
            print("Initializing crosscoder model...")

        # Create model on CPU first for FSDP
        self.crosscoder = Crosscoder_Model(cfg)

        # Configure mixed precision for FSDP
        if cfg.use_amp:
            mp_policy = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float16,  # Use fp16 for gradient reduction
                buffer_dtype=torch.float32,
            )
        else:
            mp_policy = None

        # Wrap model with FSDP for distributed training
        if cfg.world_size > 1:
            # Auto wrap policy based on parameter count
            auto_wrap_policy = size_based_auto_wrap_policy(
                min_num_params=cfg.fsdp_min_num_params
            )

            self.crosscoder = FSDP(
                self.crosscoder,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                device_id=cfg.local_rank,
                cpu_offload=cfg.fsdp_cpu_offload,
            )
            if is_main_process(cfg):
                print(f"Model wrapped with FSDP (FULL_SHARD strategy)")
        else:
            # Single GPU mode
            self.crosscoder = self.crosscoder.to(cfg.device)
            if is_main_process(cfg):
                print("Model loaded on single GPU")

        if is_main_process(cfg):
            print(f"Loading {cfg.model}...")

        # Load language model - keep on local device
        if cfg.model == "gpt2":
            self.lm = LanguageModel("openai-community/gpt2", device_map=cfg.device)
        elif cfg.model == "gemma2-2b":
            self.lm = LanguageModel("google/gemma-2-2b", device_map=cfg.device)
        else:
            raise ValueError(f"Model {cfg.model} not supported")

        if is_main_process(cfg):
            print("Initializing buffer...")
        self.buffer = Buffer(cfg, self.lm)

        # Setup optimizer
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

        # Setup gradient scaler for mixed precision
        if cfg.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Initialize wandb only on main process
        if is_main_process(cfg):
            wandb.init(
                project="crosscoder-multigpu",
                config=cfg.__dict__,
                name=f"{cfg.model}_ae{cfg.ae_dim}_gpu{cfg.world_size}",
            )
            # Note: wandb.watch doesn't work well with FSDP, so we skip it

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
        return active

    def calculate_dead_features(self, acts: torch.Tensor):
        num_dead = (acts.sum(dim=0) == 0).sum()
        return num_dead

    def train_step(self):
        self.crosscoder.train()

        # Initialize accumulation variables
        total_loss = 0.0
        total_recon_loss = 0.0
        total_sparsity_loss = 0.0
        total_preact_loss = 0.0

        # Zero gradients at the start of accumulation
        self.optimizer.zero_grad()

        # Gradient accumulation loop
        for accum_step in range(self.cfg.gradient_accumulation_steps):
            batch = self.buffer.next()  # [batch_size, num_layers, resid]

            # Use automatic mixed precision if enabled
            if self.cfg.use_amp:
                with autocast():
                    debug = self.step == 0 and accum_step == 0
                    loss_out = self.crosscoder.get_loss(batch, debug=debug)
                    loss = (
                        loss_out.recon_loss
                        + self.get_l1_coeff() * loss_out.sparsity_loss
                        + self.cfg.l_p * loss_out.preact_loss
                    )
                    # Scale loss by accumulation steps
                    loss = loss / self.cfg.gradient_accumulation_steps

                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                debug = self.step == 0 and accum_step == 0
                loss_out = self.crosscoder.get_loss(batch, debug=debug)
                loss = (
                    loss_out.recon_loss
                    + self.get_l1_coeff() * loss_out.sparsity_loss
                    + self.cfg.l_p * loss_out.preact_loss
                )
                # Scale loss by accumulation steps
                loss = loss / self.cfg.gradient_accumulation_steps

                loss.backward()

            # Accumulate losses for logging
            total_loss += loss.item()
            total_recon_loss += loss_out.recon_loss.item() / self.cfg.gradient_accumulation_steps
            total_sparsity_loss += loss_out.sparsity_loss.item() / self.cfg.gradient_accumulation_steps
            total_preact_loss += loss_out.preact_loss.item() / self.cfg.gradient_accumulation_steps

        # Gradient clipping and optimizer step
        if self.cfg.use_amp:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            clip_grad_norm_(self.crosscoder.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()

        # Calculate metrics on the last batch
        with torch.no_grad():
            acts = self.crosscoder.encode(batch)
            sparsity = self.calculate_sparsity(acts)
            dead_features = self.calculate_dead_features(acts)

            # Get W_dec norm (handle FSDP access)
            if self.cfg.world_size > 1:
                # For FSDP, we need to access the wrapped module
                w_dec_norm = self.crosscoder.module.W_dec.norm()
            else:
                w_dec_norm = self.crosscoder.W_dec.norm()

        # Aggregate metrics across all GPUs
        if self.cfg.world_size > 1:
            # Create tensors for aggregation
            metrics_tensor = torch.tensor(
                [total_loss, total_recon_loss, total_sparsity_loss,
                 total_preact_loss, sparsity, float(dead_features), w_dec_norm],
                device=self.cfg.device
            )

            # Average metrics across ranks
            gather_metrics(metrics_tensor, self.cfg)

            # Unpack aggregated metrics
            (total_loss, total_recon_loss, total_sparsity_loss,
             total_preact_loss, sparsity, dead_features, w_dec_norm) = metrics_tensor.tolist()

        return {
            "losses/loss": total_loss,
            "losses/recon_loss": total_recon_loss,
            "losses/sparsity_loss": total_sparsity_loss,
            "losses/preact_loss": self.cfg.l_p * total_preact_loss,
            "stats/l0_sparsity": sparsity,
            "stats/dead_features": dead_features,
            "hyperparams/lr": self.scheduler.get_last_lr()[0],
            "hyperparams/l1_coeff": self.get_l1_coeff(),
            "stats/W_dec_norm": w_dec_norm,
        }

    def train(self):
        if is_main_process(self.cfg):
            print(f"Starting training for {self.cfg.steps} steps...")
            print(f"Effective batch size: {self.cfg.batch_size * self.cfg.gradient_accumulation_steps * self.cfg.world_size}")

        for step in range(self.cfg.steps):
            self.step = step

            metrics = self.train_step()

            # Log only from main process
            if step % self.cfg.log_interval == 0 and is_main_process(self.cfg):
                wandb.log(metrics, step=step)

                print(
                    f"Step {step}/{self.cfg.steps} | "
                    f"Loss: {metrics['losses/loss']:.4f} | "
                    f"Recon: {metrics['losses/recon_loss']:.4f} | "
                    f"Preact Loss: {metrics['losses/preact_loss']:.4f} | "
                    f"Sparsity Loss: {metrics['losses/sparsity_loss']:.4f} | "
                    f"L0: {metrics['stats/l0_sparsity']:.1f} | "
                    f"Dead Features: {metrics['stats/dead_features']:.1f}"
                )

            # Save checkpoint from main process
            if step % self.cfg.save_interval == 0 and step > 0:
                # Ensure all ranks are synchronized before saving
                barrier(self.cfg)
                self.save_checkpoint(step)
                barrier(self.cfg)

        if is_main_process(self.cfg):
            print("Training complete!")
            wandb.finish()

        # Cleanup distributed training
        cleanup_distributed(self.cfg)

    def save_checkpoint(self, step: int):
        """Save checkpoint - only main process saves"""
        if not is_main_process(self.cfg):
            return

        print("Saving checkpoint...")
        checkpoint_path = self.run_dir / f"crosscoder_step_{step}.pt"

        # For FSDP, gather full state dict on rank 0
        if self.cfg.world_size > 1:
            # Use FSDP's state_dict with full_state_dict option
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.crosscoder, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = self.crosscoder.state_dict()
        else:
            model_state = self.crosscoder.state_dict()

        torch.save(
            {
                "step": step,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg,
            },
            checkpoint_path,
        )

        print(f"Saved checkpoint to {checkpoint_path}")
