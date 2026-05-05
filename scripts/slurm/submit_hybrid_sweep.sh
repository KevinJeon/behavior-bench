#!/bin/bash

# Submits hybrid eval jobs sweeping over switch modes and thresholds.
# Usage: ./submit_hybrid_sweep.sh [extra args...]
# Example: ./submit_hybrid_sweep.sh --map-ids 0-199

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

PDM_MIN_STEPS=20

# --- PPO-only baseline ---
echo "Submitting: ppo_only"
sbatch --job-name="hybrid_ppo_only" "$SCRIPT_DIR/run_hybrid_eval.sh" \
  --planner.hybrid.force-ppo true \
  --planner.hybrid.pdm-min-steps "$PDM_MIN_STEPS" \
  "$@"

# --- PDM-only baseline ---
echo "Submitting: pdm_only"
sbatch --job-name="hybrid_pdm_only" "$SCRIPT_DIR/run_hybrid_eval.sh" \
  --planner.hybrid.force-pdm true \
  --planner.hybrid.pdm-min-steps "$PDM_MIN_STEPS" \
  "$@"

# --- Epistemic uncertainty switching ---
for THRESH in 0.02 0.04 0.06 0.08 0.10 0.15 0.20; do
  NAME="epistemic_t${THRESH}_m${PDM_MIN_STEPS}"
  echo "Submitting: $NAME"
  sbatch --job-name="hybrid_${NAME}" "$SCRIPT_DIR/run_hybrid_eval.sh" \
    --planner.hybrid.switch-mode epistemic \
    --planner.hybrid.epistemic-threshold "$THRESH" \
    --planner.hybrid.pdm-min-steps "$PDM_MIN_STEPS" \
    "$@"
done

# --- Value variance switching ---
for THRESH in 0.01 0.02 0.05 0.10 0.15 0.20; do
  NAME="valvar_t${THRESH}_m${PDM_MIN_STEPS}"
  echo "Submitting: $NAME"
  sbatch --job-name="hybrid_${NAME}" "$SCRIPT_DIR/run_hybrid_eval.sh" \
    --planner.hybrid.switch-mode value_variance \
    --planner.hybrid.value-variance-threshold "$THRESH" \
    --planner.hybrid.pdm-min-steps "$PDM_MIN_STEPS" \
    "$@"
done
