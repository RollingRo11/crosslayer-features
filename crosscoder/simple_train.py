import torch
from pathlib import Path
from crosscoder import Trainer, cc_config, SAVE_DIR

def main():
    config = cc_config.copy()

    print("--- CROSSCODER TRAINING ---")
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"AE dimension: {config['ae_dim']}")
    print(f"Total tokens: {config['num_tokens']:,}")
    print(f"Save directory: {SAVE_DIR}")
    print()

    SAVE_DIR.mkdir(exist_ok=True)

    try:
        print("Initializing trainer...")
        trainer = Trainer(config, use_wandb=True)

        print("Starting training...")
        trainer.train()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save()

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
