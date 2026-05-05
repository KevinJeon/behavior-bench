#!/bin/bash
# Submit Waymo mixed traffic training jobs with different traffic fractions
#
# Usage: bash scripts/slurm/run_mixed_traffic_waymo.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

WAYMO_DATA=${WAYMO_DATA:-/work/dlclarge2/distelza-gpudrive/binariesv3}

# Define fraction configurations: "ppo_frac idm_frac expert_frac"
FRACTIONS=(
  "0.4 0.4 0.2"
  "0.1 0.9 0"
)

echo "Submitting Waymo mixed traffic training jobs..."
echo ""

for frac in "${FRACTIONS[@]}"; do
  read -r PPO IDM EXPERT <<< "$frac"
  NAME="waymo_mixed_ppo${PPO}_idm${IDM}_expert${EXPERT}"
  TRAFFIC_MIX="ppo:${PPO},idm:${IDM},expert:${EXPERT}"

  echo "Submitting: $NAME (traffic-mix: $TRAFFIC_MIX)"
  sbatch \
    --job-name="$NAME" \
    --output="/work/dlclarge1/distelza-data/puffer_exps/log/${NAME}_%j.out" \
    --error="/work/dlclarge1/distelza-data/puffer_exps/log/${NAME}_%j.err" \
    --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
    "$SCRIPT_DIR/run_single_training.sh" \
    --env.traffic-mix "$TRAFFIC_MIX" \
    --env.idm-random-velocity True \
    --wandb-project puffer \
    --train.data-dir "$EXP_DIR" \
    --train.name "$NAME"
done

echo ""
echo "Jobs submitted. Check status with: squeue -u \$USER"
