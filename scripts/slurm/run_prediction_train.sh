#!/bin/bash
#SBATCH -p nrdlc2_gpu-h200
#SBATCH --mem 96000
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mail-user=distelza@cs.uni-freiburg.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=smart_7m
#SBATCH --output=/work/dlclarge1/distelza-data/puffer_exps/log/smart_7m_%j.out
#SBATCH --error=/work/dlclarge1/distelza-data/puffer_exps/log/smart_7m_%j.err

source $HOME/pufferdrive/.venv/bin/activate
cd $HOME/pufferdrive

CONFIG=${CONFIG:-pufferlib/config/prediction/smart_1m_slurm.ini}

# Train
# python -m pufferlib.prediction.puffer_prediction pretrain \
#     --config "$CONFIG" "$@"

# Resume training from checkpoint (uncomment to use)
python -m pufferlib.prediction.puffer_prediction pretrain \
    --config "$CONFIG" \
    --resume /work/dlclarge1/distelza-data/puffer_exps/prediction/checkpoints_1m/epoch_052.pt \
    --wandb-run-id m060qbuy
    "$@"
