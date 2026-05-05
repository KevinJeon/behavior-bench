#!/bin/bash
# Submit script: launches one SLURM job per (switch_mode, threshold) combination.
# Sweeps epistemic uncertainty and value variance thresholds for the hybrid planner.
# Usage: bash scripts/slurm/run_hybrid_uncertainty_sweep.sh [extra args for eval.py]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${PUFFER_EXP_ROOT:-experiments}/hybrid_uncertainty_sweep"

PPO_WEIGHTS="${PPO_WEIGHTS:?Set PPO_WEIGHTS}"
ENSEMBLE_WEIGHTS="${ENSEMBLE_WEIGHTS:?Set ENSEMBLE_WEIGHTS}"

SPLITS=(pufferinter pufferrandom)
TRAFFIC_TYPES=(idm)
PDM_MIN_STEPS=20

# Threshold grid: 0.01 to 0.15
THRESHOLDS=(0.01 0.02 0.03 0.05 0.07 0.10 0.15)

# Two switch modes to sweep
SWITCH_MODES=(epistemic value_variance)

EXTRA_ARGS=("$@")

for split in "${SPLITS[@]}"; do
  for traffic in "${TRAFFIC_TYPES[@]}"; do
    for MODE in "${SWITCH_MODES[@]}"; do
      for THRESH in "${THRESHOLDS[@]}"; do
        JOB_NAME="hybrid_${MODE}_t${THRESH}_${traffic}_${split}"
        OUT_DIR="${OUTPUT_BASE}/${MODE}/threshold_${THRESH}/${traffic}_${split}"

        args=(
          --eval.split "$split"
          --output-dir "$OUT_DIR"
          --planner.type hybrid
          --planner.hybrid.ppo-weights-path "$PPO_WEIGHTS"
          --planner.hybrid.ensemble-weights "$ENSEMBLE_WEIGHTS"
          --planner.hybrid.switch-mode "$MODE"
          --planner.hybrid.pdm-min-steps "$PDM_MIN_STEPS"
          --traffic.type "$traffic"
        )

        # Set the right threshold flag based on mode
        if [[ "$MODE" == "epistemic" ]]; then
          args+=(--planner.hybrid.epistemic-threshold "$THRESH")
        elif [[ "$MODE" == "value_variance" ]]; then
          args+=(--planner.hybrid.value-variance-threshold "$THRESH")
        fi

        args+=("${EXTRA_ARGS[@]}")

        sbatch \
          --job-name="$JOB_NAME" \
          --output="/work/dlclarge1/distelza-data/puffer_exps/log/${JOB_NAME}_%j.out" \
          --error="/work/dlclarge1/distelza-data/puffer_exps/log/${JOB_NAME}_%j.err" \
          "$SCRIPT_DIR/run_single_evaluation.sh" \
          "${args[@]}"

        echo "Submitted: mode=${MODE}, threshold=${THRESH}, traffic=${traffic}, split=${split} -> $JOB_NAME"
      done
    done
  done
done

TOTAL=$(( ${#SPLITS[@]} * ${#TRAFFIC_TYPES[@]} * ${#SWITCH_MODES[@]} * ${#THRESHOLDS[@]} ))
echo "All ${TOTAL} jobs submitted."
