#!/bin/bash
#SBATCH -p nrdlc2_gpu-l40s
#SBATCH -p alldlc_gpu-rtx2080 
#SBATCH --mem 48000 # memory pool for all cores (4GB)
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 60 # number of cores
#SBATCH --mail-user=distelza@cs.uni-freiburg.de 
#SBATCH --mail-type=ALL
#SBATCH --job-name=puffer_drive_train
#SBATCH --output=scripts/slurm/logs/job_%j.out  # Standard output log (%j is job ID)
#SBATCH --error=scripts/slurm/logs/job_%j.err   # Standard error log


source ../../.venv/bin/activate
puffer train puffer_drive "$@"
