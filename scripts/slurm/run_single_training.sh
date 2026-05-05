#!/bin/bash
#SBATCH -p nrdlc2_gpu-l40s
#SBATCH --mem 48000
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 60
#SBATCH --mail-user=distelza@cs.uni-freiburg.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=ppo_train
#SBATCH --output=/work/dlclarge1/distelza-data/puffer_exps/log/ppo_train_%j.out
#SBATCH --error=/work/dlclarge1/distelza-data/puffer_exps/log/ppo_train_%j.err

source $HOME/pufferdrive/.venv/bin/activate
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:?Set DRIVE_BINARIES_DATA_ROOT}"
cd $HOME/pufferdrive

puffer train puffer_drive --wandb "$@"
