#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 3:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=ALL

# FASTA-only generation for non_ion_toxin dataset (2,547 proteins)
apptainer exec claude_env.sif python3 colabfold_all_datasets.py fasta_only > fasta_generation.log 2>&1
