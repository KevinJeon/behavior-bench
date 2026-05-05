#!/bin/bash
#SBATCH -p nrdlc2_gpu-l40s
#SBATCH --mem 400000
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:8
#SBATCH -c 120
#SBATCH --mail-user=distelza@cs.uni-freiburg.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=gigaflow_8gpu
#SBATCH --output=/work/dlclarge1/distelza-data/log/gigaflow_8gpu_%j.out
#SBATCH --error=/work/dlclarge1/distelza-data/log/gigaflow_8gpu_%j.err

source $HOME/pufferdrive/.venv/bin/activate
cd $HOME/pufferdrive

export DRIVE_BINARIES_DATA_ROOT=/work/dlclarge2/distelza-gpudrive/binariesv3
export PUFFER_DISABLE_VIDEO=1

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_conditioning.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name gigaflow_conditioning_jerk \
#   "$@"

torchrun \
  --nproc-per-node=8 \
  --master-port=29521 \
  -m pufferlib.pufferl train puffer_drive \
  --config pufferlib/config/ocean/drive_gigaflow_simple.ini \
  --wandb \
  --wandb-project puffer \
  --train.data-dir /work/dlclarge1/distelza-data/experiments \
  --train.name simple_reward_jerk \
  "$@"

  # --train.learning-rate 0.0005 \
  # --train.gamma 0.9985 \

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_conditioning_gf.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name gigaflow_conditioning_adv_transformer \
#   "$@"


# RESUME TRAINING
# RUN_ID=yxku2gty
# DATA_DIR="/work/dlclarge1/distelza-data/experiments"
# EXP_DIR="$DATA_DIR/puffer_drive_${RUN_ID}"
# LATEST_CKPT=$(ls "$EXP_DIR"/model_puffer_drive_*.pt 2>/dev/null | sort -V | tail -1)


# if [ -z "$LATEST_CKPT" ]; then
#   echo "Error: No checkpoint found in $EXP_DIR"
#   exit 1
# fi

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_conditioning_paper.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name gigaflow_conditioning_adv_6m_mlp_cont \
#   --load-model-path "$LATEST_CKPT" \
#   --train.data-dir "$DATA_DIR" \
#   --wandb-project puffer \
#   "$@"

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_complex.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name complex_reward \
#   "$@"

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_paper_classic.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name paper_classic_no_cond \
#   "$@"

# Small-net (Drive/DriveConditioned, 256-hidden MLP + LSTM, ~600K params) — classic dynamics

# export CUDA_LAUNCH_BLOCKING=1
# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_conditioning_classic.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name conditioning \
#   "$@"
  
  
  # --load-model-path /work/dlclarge1/distelza-data/experiments/puffer_drive_ig5riusz/model_puffer_drive_002000.pt \

# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_classic_simple.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name simple_reward \
#   "$@"


  # --env.reward-offroad-collision -1.0 \
  # --env.reward-vehicle-collision -1.0 \
  # --env.reward-goal-post-respawn 0.1 \
  # --env.goal-target-distance 10 \


# torchrun \
#   --nproc-per-node=8 \
#   --master-port=29521 \
#   -m pufferlib.pufferl train puffer_drive \
#   --config pufferlib/config/ocean/drive_gigaflow_classic.ini \
#   --wandb \
#   --wandb-project puffer \
#   --train.data-dir /work/dlclarge1/distelza-data/experiments \
#   --train.name complex_reward_cont \
#   --load-model-path /work/dlclarge1/distelza-data/experiments/puffer_drive_xmdvjlpp/model_puffer_drive_024000.pt \
#   "$@"

