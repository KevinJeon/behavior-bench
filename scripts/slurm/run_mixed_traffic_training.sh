#!/bin/bash
# Submit mixed traffic training jobs on the SLURM cluster
#
# Usage: bash scripts/slurm/run_mixed_traffic_training.sh
#
# Jobs:
#   Waymo:
#     1. Waymo mixed traffic (50% PPO, 25% IDM random vel, 25% Expert)
#   nuPlan:
#     2. nuPlan mixed traffic (50% PPO, 25% IDM random vel, 25% Expert)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

WAYMO_DATA=${WAYMO_DATA:-/work/dlclarge2/distelza-gpudrive/binariesv3}
NUPLAN_DATA=${NUPLAN_DATA:-/work/dlclarge2/distelza-gpudrive/gpudrive_nuplan_binaries_178k}

echo "Submitting mixed traffic training jobs..."
echo ""

# =============================================================================
# Waymo
# =============================================================================

echo "1: Waymo mixed traffic (50% PPO, 25% IDM random, 25% Expert)"
# sbatch \
#   --job-name=waymo_mixed_traffic \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_mixed_traffic_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_mixed_traffic_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --env.traffic-mix "ppo:0.6,idm:0.2,expert:0.2" \
#   --env.idm-random-velocity True \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name waymo_mixed_traffic

# =============================================================================
# nuPlan
# =============================================================================

echo "2: nuPlan mixed traffic (50% PPO, 25% IDM random, 25% Expert)"
sbatch \
  --job-name=nuplan_mixed_traffic \
  --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_mixed_traffic_%j.out" \
  --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_mixed_traffic_%j.err" \
  --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
  "$SCRIPT_DIR/run_single_training.sh" \
  --config pufferlib/config/ocean/drive_nuplan.ini \
  --env.traffic-mix "ppo:0.4,idm:0.4,expert:0.2" \
  --env.idm-random-velocity True \
  --wandb-project puffer \
  --train.data-dir "$EXP_DIR" \
  --train.name nuplan_mixed_traffic

echo ""
echo "Jobs submitted. Check status with: squeue -u \$USER"
