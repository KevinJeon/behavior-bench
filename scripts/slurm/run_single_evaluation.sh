#!/bin/bash
#SBATCH -p nrdlc2_gpu-l40s
#SBATCH --mem 96000 # memory pool for all cores (4GB)
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16 # number of cores
#SBATCH --mail-user=distelza@cs.uni-freiburg.de 
#SBATCH --mail-type=ALL
#SBATCH --job-name=puffer_drive_train
#SBATCH --output=/work/dlclarge1/distelza-data/puffer_exps/log/job_%j.out  # Standard output log (%j is job ID)
#SBATCH --error=/work/dlclarge1/distelza-data/puffer_exps/log/job_%j.err   # Standard error log


source ../../.venv/bin/activate
python $HOME/pufferdrive/pufferlib/ocean/benchmark/eval.py "$@"
