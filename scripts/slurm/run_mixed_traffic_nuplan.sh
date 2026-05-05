#!/bin/bash
# Submit nuPlan mixed traffic training jobs with different traffic fractions
#
# Usage: bash scripts/slurm/run_mixed_traffic_nuplan.sh

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

NUPLAN_DATA=${NUPLAN_DATA:-/work/dlclarge2/distelza-gpudrive/gpudrive_nuplan_binaries_178k}

# Define fraction configurations: "ppo_frac idm_frac expert_frac"
FRACTIONS=(
  #"0 1.0 0"
  # "0.4 0.4 0.2"
  #"0.1 0.9 0"
  # "0.1 0.9 0"
  # "0.2 0.8 0"
  # "0.4 0.6 0"
  "0.5 0.5 0"
  "0.9 0.1 0"
  "0.8 0.2 0"
  "0.3 0.8 0"
  "0.1 0.9 0"
  # "0.7 0.5 0"
  # "0.8 0.2 0"
  # "0.9 0.1 0"
)

echo "Submitting nuPlan mixed traffic training jobs..."
echo ""

for frac in "${FRACTIONS[@]}"; do
  read -r PPO IDM EXPERT <<< "$frac"
  NAME="nuplan_mixed_ppo${PPO}_idm${IDM}_expert${EXPERT}"
  TRAFFIC_MIX="ppo:${PPO},idm:${IDM},expert:${EXPERT}"

  echo "Submitting: $NAME (traffic-mix: $TRAFFIC_MIX)"
  sbatch \
    --job-name="$NAME" \
    --output="/work/dlclarge1/distelza-data/puffer_exps/log/${NAME}_%j.out" \
    --error="/work/dlclarge1/distelza-data/puffer_exps/log/${NAME}_%j.err" \
    --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
    "$SCRIPT_DIR/run_single_training.sh" \
    --config pufferlib/config/ocean/drive_nuplan.ini \
    --env.mix-traffic True \
    --env.ppo-fraction ${PPO} \
    --env.idm-fraction ${IDM} \
    --env.expert-fraction ${EXPERT} \
    --env.idm-random-velocity True \
    --wandb-project puffer \
    --train.data-dir "$EXP_DIR" \
    --train.name "$NAME" \
    --train.total-timesteps 2_000_000_000 \
    --train.batch-size auto \
    --train.minibatch-size auto \
    --train.max-minibatch-size auto
done

echo ""
echo "Jobs submitted. Check status with: squeue -u \$USER"
