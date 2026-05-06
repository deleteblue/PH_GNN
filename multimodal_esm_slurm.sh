#!/bin/bash
#SBATCH -N 1
#SBATCH -p EM
#SBATCH -t 6:00:00
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=80G
#SBATCH --mail-type=ALL

apptainer exec claude_env.sif /opt/conda/bin/python multimodal_esm_maker.py > output_multimodal_esm.txt 2>&1