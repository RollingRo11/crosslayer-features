import argparse
import torch
import sys
from pathlib import Path
import importlib.util

# Import from file with hyphen in name
spec = importlib.util.spec_from_file_location(
    "new_crosscoder", Path(__file__).parent / "new-crosscoder.py"
)
new_crosscoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new_crosscoder)

cc_config = new_crosscoder.cc_config
Trainer = new_crosscoder.Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Crosscoder model")

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gemma2-2b"],
        help="Model to train on",
    )
    parser.add_argument(
        "--ae_dim",
        type=int,
        default=2**15,
        help="Autoencoder dimension (default: 32768)",
    )
    parser.add_argument("--model_batch", type=int, default=32, help="Model batch size")

    # Training config
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=50000, help="Number of training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2048, help="Training batch size"
    )
    parser.add_argument("--optim", type=str, default="AdamW", help="Optimizer")

    # Logging and saving
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Steps between logging"
    )
    parser.add_argument(
        "--save_interval", type=int, default=20000, help="Steps between checkpoints"
    )

    # Buffer config
    parser.add_argument(
        "--buffer_mult", type=int, default=8, help="Buffer size multiplier"
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument(
        "--no_verbose", action="store_true", help="Disable verbose logging"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create config from arguments
    cfg = cc_config(
        model=args.model,
        ae_dim=args.ae_dim,
        model_batch=args.model_batch,
        lr=args.lr,
        steps=args.steps,
        batch_size=args.batch_size,
        optim=args.optim,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        buffer_mult=args.buffer_mult,
        device=args.device,
        seed=args.seed,
        verbose=not args.no_verbose,
    )

    print("=" * 60)
    print("Training Crosscoder with configuration:")
    print("=" * 60)
    print(f"Model: {cfg.model}")
    print(f"AE dimension: {cfg.ae_dim}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Steps: {cfg.steps}")
    print(f"Device: {cfg.device}")
    print("=" * 60)

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
