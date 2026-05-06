#!/bin/bash
#SBATCH --job-name=train_manifold
#SBATCH --output=train_statistical_manifold_output.log
#SBATCH --time=08:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python train_statistical_manifold_multimodal.py > train_statistical_manifold_output.txt 2>&1
