#!/bin/bash
# Submit script: launches one SLURM job per (threshold, lookahead) combination
# using run_hybrid_eval.sh as the base job script.
# Usage: bash scripts/slurm/run_hybrid_threshold_sweep.sh [extra args for eval.py]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${PUFFER_EXP_ROOT:-experiments}/hybrid_threshold_sweep"

THRESHOLDS=(0.02 0.04 0.06 0.08 0.10 0.15 0.20)
LOOKAHEAD_STEPS=(0 3 5 10)

for LOOK in "${LOOKAHEAD_STEPS[@]}"; do
    for THRESH in "${THRESHOLDS[@]}"; do
        JOB_NAME="hybrid_l${LOOK}_t${THRESH}"
        OUT_DIR="${OUTPUT_BASE}/lookahead_${LOOK}/threshold_${THRESH}"

        sbatch \
            --job-name="$JOB_NAME" \
            --output="/work/dlclarge1/distelza-data/puffer_exps/log/${JOB_NAME}_%j.out" \
            --error="/work/dlclarge1/distelza-data/puffer_exps/log/${JOB_NAME}_%j.err" \
            "$SCRIPT_DIR/run_hybrid_eval.sh" \
            --planner.hybrid.epistemic-threshold "$THRESH" \
            --planner.hybrid.lookahead-steps "$LOOK" \
            --output-dir "$OUT_DIR" \
            "$@"

        echo "Submitted: lookahead=${LOOK}, threshold=${THRESH} -> $JOB_NAME"
    done
done

echo "All ${#LOOKAHEAD_STEPS[@]}x${#THRESHOLDS[@]} = $(( ${#LOOKAHEAD_STEPS[@]} * ${#THRESHOLDS[@]} )) jobs submitted."
