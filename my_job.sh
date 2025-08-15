#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=crosscoder-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=job_output_%j.txt
#SBATCH --error=job_error_%j.txt
#SBATCH --mail-user=$USER@northeastern.edu
#SBATCH --mail-type=ALL

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Run the training script with uv
uv run crosscoder/simple_train.py

echo "Job completed at: $(date)"
