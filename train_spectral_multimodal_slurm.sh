#!/bin/bash
#SBATCH --job-name=train_spectral
#SBATCH --output=train_spectral_multimodal_output.log
#SBATCH --time=24:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python train_spectral_multimodal.py > train_spectral_multimodal_output.txt 2>&1
