#!/bin/bash
sbatch <<EOT
#SBATCH --job-name=slicing
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --account=def-bengioy_gpu
#SBATCH --mem=64G
#SBATCH --time=00:07:00
#SBATCH --output /home/sankarak/logs/slicing-%j.out
#SBATCH --error /home/sankarak/logs/slicing-%j.err

module load singularity/3.5
singularity exec --bind /scratch/sankarak/data/ /scratch/sankarak/images/glaciers.sif python3 slice.py $1

EOT