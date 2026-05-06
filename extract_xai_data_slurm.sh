#!/bin/bash
#SBATCH --job-name=extract_xai
#SBATCH --output=extract_xai_output.log
#SBATCH --time=4:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python extract_xai_data.py
