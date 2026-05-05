#!/bin/bash
# Three-profile comparison — DriveConditioned (classic dynamics) evaluated
# under the Aggressive / Normal / Cautious creward conditioning vectors.
# Same network weights for all three runs; only the per-agent creward
# vector changes. Traffic is always IDM.
#
# Profile values follow the paper Table A2 with the following overrides:
#   - Normal:    alpha_comfort = 0.002 (was 0.05),  goal_speed = 10 (was 20)
#   - Cautious:  alpha_comfort = 0.005 (was 0.1),   goal_speed = 3  (was 5)
#
# Usage:
#   ./run_reward_ablation_normal.sh [extra args forwarded to eval.py...]
# Env overrides:
#   PARTITION       slurm partition (default: nrdlc2_gpu-l40s)
#   SPLIT           eval split (default: pufferinter)
#   PLANNER_TYPE    planner section used for the CLI flags
#                   (default: conditioned_normal — any conditioned_* with the
#                   same weights works since we override all creward fields)
#   PLANNER_WEIGHTS override checkpoint
#                   (default: ukkwspio:024000)

PARTITION="${PARTITION:-nrdlc2_gpu-l40s}"
SPLIT="${SPLIT:-pufferinter}"
PLANNER_TYPE="${PLANNER_TYPE:-conditioned_normal}"
PLANNER_WEIGHTS="${PLANNER_WEIGHTS:-/work/dlclarge1/distelza-data/experiments/puffer_drive_ukkwspio/model_puffer_drive_024000.pt}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}/three_profiles_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Three-profile output root: $OUTPUT_DIR"
echo "Planner: $PLANNER_TYPE  (weights: $PLANNER_WEIGHTS)"
echo "Traffic: idm   Split: $SPLIT"

EXTRA_ARGS=("$@")

# argparse converts '_' in flags to '-'; same applies to the planner section.
ptype_cli="${PLANNER_TYPE//_/-}"

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

# Profiles below follow Table A2. Modified entries marked with "← override".
declare -A PROFILE_AGGR
PROFILE_AGGR[delta_goal]=0.0
PROFILE_AGGR[alpha_collision]=0.2
PROFILE_AGGR[alpha_boundary]=0.2
PROFILE_AGGR[alpha_comfort]=0.0
PROFILE_AGGR[alpha_l_align]=0.001
PROFILE_AGGR[alpha_vel_align]=0.1
PROFILE_AGGR[alpha_l_center]=0.00025
PROFILE_AGGR[alpha_center_bias]=0.0
PROFILE_AGGR[alpha_reverse]=0.00025
PROFILE_AGGR[goal_speed]=30.0

declare -A PROFILE_NORM
PROFILE_NORM[delta_goal]=0.0
PROFILE_NORM[alpha_collision]=3.0
PROFILE_NORM[alpha_boundary]=3.0
PROFILE_NORM[alpha_comfort]=0.002          # ← override (table: 0.05)
PROFILE_NORM[alpha_l_align]=0.015
PROFILE_NORM[alpha_vel_align]=0.5
PROFILE_NORM[alpha_l_center]=0.0038
PROFILE_NORM[alpha_center_bias]=0.0
PROFILE_NORM[alpha_reverse]=0.0038
PROFILE_NORM[goal_speed]=10.0              # ← override (table: 20)

declare -A PROFILE_CAUT
PROFILE_CAUT[delta_goal]=0.0
PROFILE_CAUT[alpha_collision]=3.0
PROFILE_CAUT[alpha_boundary]=3.0
PROFILE_CAUT[alpha_comfort]=0.005          # ← override (table: 0.1)
PROFILE_CAUT[alpha_l_align]=0.025
PROFILE_CAUT[alpha_vel_align]=1.0
PROFILE_CAUT[alpha_l_center]=0.0075
PROFILE_CAUT[alpha_center_bias]=0.5
PROFILE_CAUT[alpha_reverse]=0.0075
PROFILE_CAUT[goal_speed]=3.0               # ← override (table: 5)

submit_profile () {
  local profile_name="$1"
  local -n profile="$2"   # nameref to associative array
  local args=()

  args+=(--eval.split "$SPLIT" --output-dir "$OUTPUT_DIR")
  if [[ "$PLANNER_TYPE" == "conditioned_paper" ]]; then
    args+=(--eval.dynamics-model jerk)
  fi

  args+=(--planner.type "$PLANNER_TYPE")
  if [[ -n "$PLANNER_WEIGHTS" ]]; then
    args+=("--planner.${ptype_cli}.weights-path" "$PLANNER_WEIGHTS")
  fi

  args+=(--traffic.type idm)

  for field in "${CREWARD_FIELDS[@]}"; do
    local cli_field="${field//_/-}"
    args+=("--planner.${ptype_cli}.creward.${cli_field}" "${profile[$field]}")
  done

  args+=("${EXTRA_ARGS[@]}")

  local job_name="profile_${profile_name}"
  echo "Submitting: ${job_name}"
  sbatch --partition="$PARTITION" --job-name="$job_name" run_single_evaluation.sh "${args[@]}"
}

submit_profile "aggr" PROFILE_AGGR
submit_profile "norm" PROFILE_NORM
submit_profile "caut" PROFILE_CAUT
