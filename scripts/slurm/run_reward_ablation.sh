#!/bin/bash
# Reward ablation study.
#
# For each creward.* coefficient, sweep across its training distribution
# (per Table A2) while holding the other nine at the "aggressive" baseline
# below. Traffic is always IDM, so any score difference is attributable to
# how the conditioning vector reshapes the ego policy's behaviour.
#
# Usage:
#   ./run_reward_ablation.sh [extra args forwarded to eval.py...]
# Env overrides:
#   PARTITION       slurm partition (default: nrdlc2_gpu-l40s)
#   SPLIT           eval split (default: pufferinter)
#   PLANNER_TYPE    conditioned_paper | conditioned_aggr | conditioned_normal | ...
#                   (default: conditioned_paper -> jerk dynamics)
#   PLANNER_WEIGHTS override checkpoint (default: paper checkpoint z7rheqpi)

PARTITION="${PARTITION:-nrdlc2_gpu-l40s}"
SPLIT="${SPLIT:-pufferinter}"
PLANNER_TYPE="${PLANNER_TYPE:-conditioned_paper}"
PLANNER_WEIGHTS="${PLANNER_WEIGHTS:-/work/dlclarge1/distelza-data/experiments/puffer_drive_z7rheqpi/model_puffer_drive_019000.pt}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}/reward_ablation_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Reward ablation output root: $OUTPUT_DIR"
echo "Planner: $PLANNER_TYPE  (weights: $PLANNER_WEIGHTS)"
echo "Traffic: idm   Split: $SPLIT"

EXTRA_ARGS=("$@")

# argparse converts every '_' in the flag to '-', so the planner section
# name needs the same treatment when used in CLI flags.
ptype_cli="${PLANNER_TYPE//_/-}"

# Aggressive-profile baseline (matches [planner.conditioned_aggr]).
declare -A BASE
BASE[delta_goal]=0.0
BASE[alpha_collision]=0.2
BASE[alpha_boundary]=0.2
BASE[alpha_comfort]=0.0
BASE[alpha_l_align]=0.001
BASE[alpha_vel_align]=0.1
BASE[alpha_l_center]=0.00025
BASE[alpha_center_bias]=0.0
BASE[alpha_reverse]=0.0005
BASE[goal_speed]=30.0

CREWARD_FIELDS=(
  delta_goal
  alpha_collision
  alpha_boundary
  alpha_comfort
  alpha_l_align
  alpha_vel_align
  alpha_l_center
  alpha_center_bias
  alpha_reverse
  goal_speed
)

# Per-field sweep grids covering the training U(...) ranges from Table A2.
# delta_goal:        U(2, 12)               (0 -> use env goal radius)
# alpha_collision:   U(0, 3)
# alpha_boundary:    U(0, 3)
# alpha_comfort:     U(0, 0.1)
# alpha_l_align:     U(2.5e-4, 2.5e-2)
# alpha_vel_align:   U(0, 1)
# alpha_l_center:    U(2.5e-4, 7.5e-3)
# alpha_center_bias: U(-0.5, 0.5)
# alpha_reverse:     U(2.5e-4, 7.5e-3)
# goal_speed:        U(3, 30)
declare -A SWEEPS
# SWEEPS[delta_goal]="2.0 6.0 12.0"
# SWEEPS[alpha_collision]="0.0 1.5 3.0"
# SWEEPS[alpha_boundary]="0.0 1.5 3.0"
# SWEEPS[alpha_comfort]="0.025 0.05 0.1"
# SWEEPS[alpha_l_align]="0.0025 0.0127 0.025"
SWEEPS[alpha_vel_align]="0.0 0.5 1.0"
SWEEPS[alpha_l_center]="0.002 0.00388 0.0075"
SWEEPS[alpha_center_bias]="-0.5 0.25 0.5"
SWEEPS[alpha_reverse]="0.002 0.00388 0.0075"
SWEEPS[goal_speed]="3.0 15.0 30.0"

submit_run () {
  local sweep_field="$1"
  local sweep_value="$2"
  local args=()

  args+=(--eval.split "$SPLIT" --output-dir "$OUTPUT_DIR")
  # conditioned_paper uses the jerk-dynamics ego obs; the conditioned_*
  # variants stay on classic.
  if [[ "$PLANNER_TYPE" == "conditioned_paper" ]]; then
    args+=(--eval.dynamics-model jerk)
  fi

  args+=(--planner.type "$PLANNER_TYPE")
  if [[ -n "$PLANNER_WEIGHTS" ]]; then
    args+=("--planner.${ptype_cli}.weights-path" "$PLANNER_WEIGHTS")
  fi

  args+=(--traffic.type idm)

  for field in "${CREWARD_FIELDS[@]}"; do
    local value="${BASE[$field]}"
    if [[ "$field" == "$sweep_field" ]]; then
      value="$sweep_value"
    fi
    local cli_field="${field//_/-}"
    args+=("--planner.${ptype_cli}.creward.${cli_field}" "$value")
  done

  args+=("${EXTRA_ARGS[@]}")

  local job_name="abl_${sweep_field}_${sweep_value}"
  echo "Submitting: ${job_name}"
  sbatch --partition="$PARTITION" --job-name="$job_name" run_single_evaluation.sh "${args[@]}"
}

# Baseline: all fields at the aggressive-profile values.
submit_run "baseline" "base"

# One-at-a-time sweeps.
for field in "${CREWARD_FIELDS[@]}"; do
  for v in ${SWEEPS[$field]}; do
    submit_run "$field" "$v"
  done
done
