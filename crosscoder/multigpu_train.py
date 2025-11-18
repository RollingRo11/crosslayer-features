import argparse
import torch
from pathlib import Path
from multigpucrosscoder import cc_config, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Multi-GPU Crosscoder model with FSDP")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["gpt2", "gemma2-2b"],
        help="Model to train on",
    )
    parser.add_argument(
        "--ae_dim",
        type=int,
        default=None,
        help="Autoencoder dimension",
    )
    parser.add_argument("--model_batch", type=int, default=None, help="Model batch size")

    # Training config
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Training batch size per GPU"
    )
    parser.add_argument("--optim", type=str, default=None, help="Optimizer")
    parser.add_argument(
        "--warmup_steps", type=int, default=None, help="Number of warmup steps"
    )
    parser.add_argument(
        "--l1_coeff", type=float, default=None, help="L1 sparsity coefficient"
    )
    parser.add_argument(
        "--l_s", type=float, default=None, help="Sparsity loss coefficient"
    )
    parser.add_argument(
        "--init_norm", type=float, default=None, help="Decoder initialization norm"
    )
    parser.add_argument(
        "--l_p", type=float, default=None, help="Preactivation loss coefficient"
    )

    # Logging and saving
    parser.add_argument(
        "--log_interval", type=int, default=None, help="Steps between logging"
    )
    parser.add_argument(
        "--save_interval", type=int, default=None, help="Steps between checkpoints"
    )

    # Buffer config
    parser.add_argument(
        "--buffer_mult", type=int, default=None, help="Buffer size multiplier"
    )

    # Multi-GPU / Distributed config
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=None,
        help="Enable automatic mixed precision training (default: True)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision training",
    )
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        default=None,
        help="Enable CPU offloading for FSDP (saves memory)",
    )
    parser.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=None,
        help="Minimum parameters for FSDP wrapping (default: 1e6)",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (will be overridden by distributed setup)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config with defaults from cc_config, then override with provided args
    cfg = cc_config()

    # Only override values that were explicitly provided
    if args.model is not None:
        cfg.model = args.model
    if args.ae_dim is not None:
        cfg.ae_dim = args.ae_dim
    if args.model_batch is not None:
        cfg.model_batch = args.model_batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.optim is not None:
        cfg.optim = args.optim
    if args.log_interval is not None:
        cfg.log_interval = args.log_interval
    if args.save_interval is not None:
        cfg.save_interval = args.save_interval
    if args.buffer_mult is not None:
        cfg.buffer_mult = args.buffer_mult
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.warmup_steps is not None:
        cfg.warmup_steps = args.warmup_steps
    if args.l1_coeff is not None:
        cfg.l1_coeff = args.l1_coeff
    if args.l_s is not None:
        cfg.l_s = args.l_s
    if args.init_norm is not None:
        cfg.init_norm = args.init_norm
    if args.l_p is not None:
        cfg.l_p = args.l_p

    # Multi-GPU specific config
    if args.gradient_accumulation_steps is not None:
        cfg.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.no_amp:
        cfg.use_amp = False
    elif args.use_amp is not None:
        cfg.use_amp = args.use_amp
    if args.fsdp_cpu_offload is not None:
        cfg.fsdp_cpu_offload = args.fsdp_cpu_offload
    if args.fsdp_min_num_params is not None:
        cfg.fsdp_min_num_params = args.fsdp_min_num_params

    print("=" * 60)
    print("Training Multi-GPU Crosscoder with configuration:")
    print("=" * 60)
    print(f"Model: {cfg.model}")
    print(f"AE dimension: {cfg.ae_dim}")
    print(f"Batch size per GPU: {cfg.batch_size}")
    print(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Steps: {cfg.steps}")
    print(f"Mixed precision (AMP): {cfg.use_amp}")
    print(f"FSDP CPU offload: {cfg.fsdp_cpu_offload}")
    print("=" * 60)
    print("Note: Distributed training parameters (world_size, rank) will be")
    print("      automatically set from environment variables by torchrun.")
    print("=" * 60)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
