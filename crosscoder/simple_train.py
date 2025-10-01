import torch
import argparse
from pathlib import Path
from crosscoder import Trainer, cc_config, SAVE_DIR, WANDB_DIR


def find_latest_checkpoint(version_dir=None):
    """Find the latest checkpoint in the most recent version directory"""
    if version_dir is None:
        # Find the most recent version directory
        version_dirs = [d for d in SAVE_DIR.iterdir() if d.is_dir() and d.name.startswith("version_")]
        if not version_dirs:
            return None
        version_dir = max(version_dirs, key=lambda d: int(d.name.split("_")[1]))

    # Look for latest.pt symlink first
    latest_path = version_dir / "latest.pt"
    if latest_path.exists():
        return latest_path

    # Otherwise find the most recent checkpoint file
    checkpoints = list(version_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))


def main():
    parser = argparse.ArgumentParser(description="Train Crosscoder with optional resume")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically resume from the latest checkpoint if available"
    )
    args = parser.parse_args()

    config = cc_config.copy()

    print("--- CROSSCODER TRAINING ---")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"AE dimension: {config['ae_dim']}")
    print(f"Total steps: {config['total_steps']:,}")
    print(f"Buffer Multiplier: {config['buffer_mult']}")
    print(f"enc/dec init norm: {config['dec_init_norm']}")
    print()

    # Ensure directories exist
    SAVE_DIR.mkdir(exist_ok=True)
    WANDB_DIR.mkdir(exist_ok=True)

    # Determine checkpoint to resume from
    resume_from = None
    if args.resume_from:
        resume_from = Path(args.resume_from)
        if not resume_from.exists():
            print(f"Warning: Checkpoint {resume_from} not found. Starting fresh.")
            resume_from = None
        else:
            print(f"Resuming from: {resume_from}")
    elif args.auto_resume:
        resume_from = find_latest_checkpoint()
        if resume_from:
            print(f"Auto-resuming from: {resume_from}")
        else:
            print("No checkpoint found. Starting fresh.")

    try:
        print("Initializing trainer...")
        trainer = Trainer(config, use_wandb=True, resume_from=resume_from)

        if resume_from is None:
            try:
                initial_analysis = trainer.analyze()
                print(
                    f"Initial dead features: {initial_analysis['dead_features']}/{initial_analysis['total_features']}"
                )
            except Exception as e:
                print(f"Initial analysis failed: {e}")

        trainer.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()

        try:
            final_analysis = trainer.analyze()
            print(f"Final mean sparsity: {final_analysis['mean_sparsity']:.3f}")
            print(
                f"Final dead features: {final_analysis['dead_features']}/{final_analysis['total_features']}"
            )
        except Exception as e:
            print(f"Final analysis failed: {e}")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()

        try:
            trainer.save_checkpoint()
            print("Checkpoint saved despite error")
        except:
            print("Could not save checkpoint")

    print("Training completed!")


if __name__ == "__main__":
    main()
