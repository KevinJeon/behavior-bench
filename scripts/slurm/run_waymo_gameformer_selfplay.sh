#!/bin/bash
# Submit GameFormer self-play training on Waymo maps
#
# Usage: bash scripts/slurm/run_waymo_gameformer_selfplay.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

WAYMO_DATA=${WAYMO_DATA:-/work/dlclarge2/distelza-gpudrive/binariesv3}

echo "Submitting Waymo GameFormer self-play training..."

sbatch \
  --job-name=waymo_gameformer_selfplay \
  --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_gameformer_selfplay_%j.out" \
  --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_gameformer_selfplay_%j.err" \
  --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
  "$SCRIPT_DIR/run_single_training.sh" \
  --policy-name DriveGameFormer \
  --wandb-project puffer \
  --train.data-dir "$EXP_DIR" \
  --train.name waymo_gameformer_selfplay \
  --train.minibatch-size 16384 \
  --train.max-minibatch-size 16384 \

echo "Job submitted. Check status with: squeue -u \$USER"
