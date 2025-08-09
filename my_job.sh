#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=model-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --time=08:00:00
#SBATCH --mail-user=$USER@northeastern.edu
#SBATCH --mail-type=ALL

uv run crosscoder/simple_train.py
