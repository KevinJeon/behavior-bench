#!/bin/bash
# Submit GameFormer self-play training on Waymo maps
#
# Usage: bash scripts/slurm/run_waymo_gameformer_selfplay.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

NUPLAN_DATA=/work/dlclarge2/distelza-gpudrive/gpudrive_nuplan_binaries_178k

echo "Submitting nuPlan GameFormer self-play training..."

sbatch \
  --job-name=nuplan_selfplay \
  --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_gameformer_selfplay_%j.out" \
  --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_gameformer_selfplay_%j.err" \
  --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
  "$SCRIPT_DIR/run_single_training.sh" \
  --config pufferlib/config/ocean/drive_nuplan.ini \
  --wandb-project puffer \
  --policy-name DriveGameFormer \
  --train.data-dir "$EXP_DIR" \
  --train.name nuplan_gameformer_selfplay \
  --train.minibatch-size 16384 \
  --train.max-minibatch-size 16384 \

echo "Job submitted. Check status with: squeue -u \$USER"
