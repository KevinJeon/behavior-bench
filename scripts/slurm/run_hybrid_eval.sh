#!/bin/bash

PPO_WEIGHTS="${PPO_WEIGHTS:?Set PPO_WEIGHTS}"
ENSEMBLE_WEIGHTS="${ENSEMBLE_WEIGHTS:?Set ENSEMBLE_WEIGHTS}"
SMART_TRAFFIC_WEIGHTS="${SMART_WEIGHTS:?Set SMART_WEIGHTS}"
SPLITS=(pufferinter pufferrandom)
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}/hybrid_eval"

# Extra args passed at invocation are forwarded to every submit call
# Usage: ./run_hybrid_eval.sh [extra args...]
EXTRA_ARGS=("$@")

submit () {
  local split="$1"
  local traffic="$2"
  shift 2
  local extra_args=("$@")

  args=(
    --eval.split "$split"
    --output-dir "$OUTPUT_DIR"
    --planner.type hybrid
    --planner.hybrid.ppo-weights-path "$PPO_WEIGHTS"
    --planner.hybrid.ensemble-weights "$ENSEMBLE_WEIGHTS"
    --planner.hybrid.epistemic-threshold 0.02
    --planner.hybrid.pdm-min-steps 20
    --planner.hybrid.switch-mode epistemic
  )

  # traffic controller
  args+=(--traffic.type "$traffic")
  if [[ "$traffic" == "ppo" ]]; then
    args+=(--traffic.ppo.weights-path "$PPO_WEIGHTS")
  elif [[ "$traffic" == "smart" ]]; then
    args+=(--traffic.smart.weights-path "$SMART_TRAFFIC_WEIGHTS")
  fi

  # Extra args
  args+=("${extra_args[@]}")
  args+=("${EXTRA_ARGS[@]}")

  JOB_NAME="eval_hybrid_vs_${traffic}_${split}"
  echo "Submitting: ${JOB_NAME}"
  sbatch --job-name="$JOB_NAME" run_single_evaluation.sh "${args[@]}"
}

# ------------------------------------------------------------
# Hybrid planner vs all traffic agents: IDM, PPO, SMART, EXPERT
# ------------------------------------------------------------

TRAFFIC_TYPES=(idm ppo smart expert)

for split in "${SPLITS[@]}"; do
  for traffic in "${TRAFFIC_TYPES[@]}"; do
    submit "$split" "$traffic"
  done
done
