#!/bin/bash
#SBATCH --job-name=create_LMDB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=/home/dkovacevic/MHM_project/start_scripts/slurm_outputs/%x-%j.out

# Activate conda
source /home/dkovacevic/miniconda3/bin/activate denis_env  # use your env here

cd ..
cd utils

# Run our code
echo "-------- PYTHON OUTPUT ----------"
python create_LMDB.py -d $1 -t $2 -dim $3
echo "---------------------------------"

# Deactivate environment again
conda deactivate
