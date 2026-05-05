#!/bin/bash
#SBATCH -p nrdlc2_gpu-l40s
#SBATCH --mem 48000
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mail-user=distelza@cs.uni-freiburg.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=wosac_eval
#SBATCH --output=/work/dlclarge1/distelza-data/puffer_exps/log/wosac_%j.out
#SBATCH --error=/work/dlclarge1/distelza-data/puffer_exps/log/wosac_%j.err


source ../../.venv/bin/activate
export DRIVE_BINARIES_DATA_ROOT="${DRIVE_BINARIES_DATA_ROOT:?Set DRIVE_BINARIES_DATA_ROOT}"

python $HOME/pufferdrive/pufferlib/ocean/benchmark/eval_realism.py "$@"
