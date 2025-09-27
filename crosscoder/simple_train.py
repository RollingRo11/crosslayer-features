import torch
from pathlib import Path
from crosscoder import Trainer, cc_config, SAVE_DIR, WANDB_DIR


def main():
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

    try:
        print("Initializing trainer...")
        trainer = Trainer(config, use_wandb=True)

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
        trainer.save()

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
            trainer.save()
            print("Checkpoint saved despite error")
        except:
            print("Could not save checkpoint")

    print("Training completed!")


if __name__ == "__main__":
    main()
