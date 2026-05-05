#!/bin/bash

# Submits two hybrid eval jobs in parallel: one vs IDM, one vs PPO
# Usage: ./submit_hybrid_eval.sh [extra args...]
# Example: ./submit_hybrid_eval.sh --map-ids 0-100

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OTHER_WEIGHTS="${PPO_WEIGHTS:?Set PPO_WEIGHTS}"

echo "Submitting: hybrid_vs_idm"
sbatch --job-name="hybrid_vs_idm" "$SCRIPT_DIR/run_hybrid_eval.sh" --traffic.type idm "$@"

echo "Submitting: hybrid_vs_ppo"
sbatch --job-name="hybrid_vs_ppo" "$SCRIPT_DIR/run_hybrid_eval.sh" --traffic.type ppo --traffic.ppo.weights-path "$OTHER_WEIGHTS" "$@"
