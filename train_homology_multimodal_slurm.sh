#!/bin/bash
#SBATCH --job-name=train_homology
#SBATCH --output=train_homology_multimodal_output.log
#SBATCH --time=16:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python train_homology_multimodal.py > train_homology_multimodal_output.txt 2>&1
