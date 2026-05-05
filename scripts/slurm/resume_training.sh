#!/bin/bash
# Resume a training run from its latest checkpoint.
#
# Usage:
#   bash scripts/slurm/resume_training.sh <run_id> [extra args...]
#
# The script finds the latest model checkpoint in experiments/puffer_drive_<run_id>/
# and submits a SLURM job that loads those weights.
#
# All original training args must be re-specified as extra args, since the
# checkpoint only stores model weights (not config).
#
# Examples:
#   # Resume waymo selfplay run pvp093uj
#   bash scripts/slurm/resume_training.sh pvp093uj \
#     --env.collision-shrink 1.0 \
#     --wandb-project puffer \
#     --train.name waymo_selfplay_colshrink1_resumed

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [ $# -lt 1 ]; then
  echo "Usage: $0 <run_id> [extra training args...]"
  echo ""
  echo "Example:"
  echo "  $0 bh4l4jx1 --env.collision-shrink 1.0 --wandb-project puffer --train.name my_resumed_run"
  exit 1
fi

RUN_ID="$1"
shift

DATA_DIR="/work/dlclarge1/distelza-data/puffer_exps"
EXP_DIR="$DATA_DIR/puffer_drive_${RUN_ID}"

if [ ! -d "$EXP_DIR" ]; then
  echo "Error: Experiment directory not found: $EXP_DIR"
  exit 1
fi

# Find the latest checkpoint (highest epoch number)
LATEST_CKPT=$(ls "$EXP_DIR"/model_puffer_drive_*.pt 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_CKPT" ]; then
  echo "Error: No checkpoint found in $EXP_DIR"
  exit 1
fi

echo "Resuming from: $LATEST_CKPT"
echo "Resuming wandb run: $RUN_ID"

WAYMO_DATA=${WAYMO_DATA:-/work/dlclarge2/distelza-gpudrive/binariesv3}
NUPLAN_DATA=/work/dlclarge2/distelza-gpudrive/gpudrive_nuplan_binaries_178k

sbatch \
  --job-name="resume_${RUN_ID}" \
  --output="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.out" \
  --error="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.err" \
  --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
  "$SCRIPT_DIR/run_single_training.sh" \
  --config pufferlib/config/ocean/drive.ini \
  --load-model-path "$LATEST_CKPT" \
  --load-id "$RUN_ID" \
  --train.data-dir "$DATA_DIR" \
  --wandb-project puffer \
  --train.total-timesteps 20_000_000_000
  "$@"

# sbatch \
#   --job-name="resume_${RUN_ID}" \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --config pufferlib/config/ocean/drive_nuplan.ini \
#   --load-model-path "$LATEST_CKPT" \
#   --load-id "$RUN_ID" \
#   --train.data-dir "$DATA_DIR" \
#   --wandb-project puffer \
#   "$@"

  
# sbatch \
#   --job-name="resume_${RUN_ID}" \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/resume_${RUN_ID}_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --config pufferlib/config/ocean/drive_nuplan.ini \
#   --env.max-controlled-agents 1 \
#   --env.num-agents 64 \
#   --train.batch-size 32768 \
#   --train.minibatch-size 8192 \
#   --train.max-minibatch-size 8192 \
#   --load-model-path "$LATEST_CKPT" \
#   --load-id "$RUN_ID" \
#   --train.data-dir "$DATA_DIR" \
#   --wandb-project puffer \
#   "$@"
