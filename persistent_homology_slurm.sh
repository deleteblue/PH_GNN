#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 1:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mail-type=ALL
#SBATCH -J persistent_homology

apptainer exec claude_env.sif python persistent_homology_analysis.py > persistent_homology_output.txt 2>&1