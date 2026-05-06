#!/bin/bash
#SBATCH --job-name=train_multimodal_v2
#SBATCH --output=train_full_multimodal_v2_output.log
#SBATCH --time=12:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python train_full_multimodal_v2.py > train_full_multimodal_v2_output.txt 2>&1
