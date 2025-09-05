#!/bin/bash
#SBATCH --job-name=plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dkovacevic/MHM_project/start_scripts/slurm_outputs/%x-%j.out

# Activate conda
source /home/dkovacevic/miniconda3/bin/activate denis_env  # use your env here

cd ..
cd training

# Run our code
echo "-------- PYTHON OUTPUT ----------"
accelerate launch plotting.py
echo "---------------------------------"

conda deactivate
