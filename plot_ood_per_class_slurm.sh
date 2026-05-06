#!/bin/bash
#SBATCH --job-name=plot_ood_per_class
#SBATCH --time=00:30:00
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15GB
#SBATCH --mail-type=ALL

module load anaconda3
source activate py371

python plot_ood_per_class.py > plot_ood_per_class_output.txt 2>&1
