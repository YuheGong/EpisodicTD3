#!/bin/bash
#SBATCH -p gpu_8
# #SBATCH -A
#SBATCH -J promptd3

# Please use the complete path details :
#SBATCH -D ./
#SBATCH -o ./slurm/out_%A_%a.log
#SBATCH -e ./slurm/err_%A_%a.log

# Cluster Settings
#SBATCH -n 1         # Number of tasks
#SBATCH -c 1  # Number of cores per task
#SBATCH -t 20:0:00             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

#SBATCH --gres gpu:1
# -------------------------------

# Activate the virtualenv / conda environment

# Export Pythonpath


# Additional Instructions from CONFIG.yml


python simple.py 

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE
