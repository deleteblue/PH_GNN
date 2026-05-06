#!/bin/bash
#SBATCH --job-name=tsne_homology
#SBATCH --output=tsne_homology_slurm.log
#SBATCH --time=04:00:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python tsne_homology.py > tsne_homology_output.txt 2>&1
