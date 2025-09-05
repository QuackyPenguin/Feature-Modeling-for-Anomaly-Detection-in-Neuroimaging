#!/bin/bash
#SBATCH --job-name=gaus_fit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=320G
#SBATCH --time=180:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/dkovacevic/MHM_project/start_scripts/slurm_outputs/%x-%j.out

# Activate conda
source /home/dkovacevic/miniconda3/bin/activate denis_env  # use your env here

cd ..
cd training

# Run our code
echo "-------- PYTHON OUTPUT ----------"
accelerate launch gaussian_fitting.py
echo "---------------------------------"

conda deactivate
