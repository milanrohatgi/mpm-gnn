#!/bin/bash
#SBATCH --job-name=particle_gnn
#SBATCH --time=7-00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --partition=rondror


eval "$(micromamba shell hook --shell bash)"

micromamba activate rhomax2


wandb login 1c39c30d32736a2f6134d770c83d5c167971b82d

python train.py
