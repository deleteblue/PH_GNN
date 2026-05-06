#!/bin/bash
#SBATCH --job-name=ood_spectral
#SBATCH --time=08:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python train_ood_spectral.py > train_ood_spectral_output.txt 2>&1
