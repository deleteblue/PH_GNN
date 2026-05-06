#!/bin/bash
#SBATCH --job-name=fasta_based_esm
#SBATCH --output=fasta_based_esm_output.log
#SBATCH --time=6:00:00
#SBATCH --partition=EM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32GB

module load anaconda3
source activate py371

python fasta_based_esm_generator.py > fasta_based_esm_output.txt 2>&1