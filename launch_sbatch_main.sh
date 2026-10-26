#!/bin/bash
#SBATCH --job-name=long-asyh-rl
#SBATCH --output=logs/asyh-rl_%j.out
#SBATCH --error=logs/asyh-rl_%j.err
#SBATCH --time=99:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=long


source /home/gsergei/asyh/bin/activate

python3 main_rl_optimization.py 
