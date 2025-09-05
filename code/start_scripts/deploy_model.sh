#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=100:00:00
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dkovacevic/MHM_project/start_scripts/slurm_outputs/%x-%j.out

# Activate conda
source /home/dkovacevic/miniconda3/bin/activate denis_env  # use your env here

cd ..
cd training

# Run our code
echo "-------- PYTHON OUTPUT ----------"
accelerate launch train.py
echo "---------------------------------"

conda deactivate
