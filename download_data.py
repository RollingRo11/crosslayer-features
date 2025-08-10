from datasets import load_dataset
from pathlib import Path

# Use your existing cache directory structure
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_CACHE_DIR = PROJECT_ROOT / "data" / "hf_datasets_cache"
DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Download the dataset
print("Downloading Pile-uncopyrighted dataset...")
dataset = load_dataset(
   'monology/pile-uncopyrighted',
   split='train',
   cache_dir=str(DATASET_CACHE_DIR)
)
print(f"Download complete! Dataset cached to: {DATASET_CACHE_DIR}")
print(f"Number of examples: {len(dataset):,}")
