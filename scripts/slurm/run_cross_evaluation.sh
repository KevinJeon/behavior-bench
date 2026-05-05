#!/bin/bash
# Single source of truth for all planner / traffic weight paths.
# Keep weights here only — evaluation.ini's weights_path entries are blank.
PPO_WEIGHTS="/work/dlclarge1/distelza-data/experiments/puffer_drive_m9crl211/model_puffer_drive_020000.pt"
SMART_WEIGHTS="/work/dlclarge1/distelza-data/experiments/SMART_epoch_030.pt"
SMART_TRAFFIC_WEIGHTS="$SMART_WEIGHTS"
CONDITIONED_WEIGHTS="/work/dlclarge1/distelza-data/experiments/puffer_drive_ukkwspio/model_puffer_drive_024000.pt"

SPLITS=(pufferinter pufferrandom)
# SPLITS=(pufferinter)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}/cross_eval_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "Cross-eval output root: $OUTPUT_DIR"

# Extra args passed at invocation are forwarded to every submit call
# Usage: ./run_cross_evaluation.sh [extra args...]
EXTRA_ARGS=("$@")

# Append the right --planner.<type>.weights-path for the chosen ego planner.
# Modifies the caller's `args` array (bash function vars are global by default).
# Note: load_eval_config registers CLI flags with underscores replaced by
# dashes (registry.py: fmt.replace("_", "-")), so e.g. the section
# [planner.conditioned_normal] is reachable as --planner.conditioned-normal.*.
add_ego_weights() {
  local ego="$1"
  local cli="${ego//_/-}"
  case "$ego" in
    ppo)
      args+=(--planner.ppo.weights-path "$PPO_WEIGHTS")
      ;;
    smart)
      args+=(--planner.smart.weights-path "$SMART_WEIGHTS")
      ;;
    hybrid)
      args+=(--planner.hybrid.ppo-weights-path "$PPO_WEIGHTS")
      ;;
    conditioned_mix|conditioned_aggr|conditioned_normal|conditioned_caut)
      args+=(--planner."$cli".weights-path "$CONDITIONED_WEIGHTS")
      ;;
  esac
}

# Append the right --traffic.<type>.weights-path for the chosen traffic type.
add_traffic_weights() {
  local traffic="$1"
  local cli="${traffic//_/-}"
  case "$traffic" in
    ppo)
      args+=(--traffic.ppo.weights-path "$PPO_WEIGHTS")
      ;;
    smart)
      args+=(--traffic.smart.weights-path "$SMART_TRAFFIC_WEIGHTS")
      ;;
    conditioned_mix|conditioned_aggr|conditioned_normal|conditioned_caut)
      args+=(--traffic."$cli".weights-path "$CONDITIONED_WEIGHTS")
      ;;
  esac
}

submit () {
  local ego="$1"
  local other="$2"
  shift 2
  local extra_ego_args=("$@")

  args=(--eval.split "$SPLIT" --output-dir "$OUTPUT_DIR")

  # Ego planner + its weights
  args+=(--planner.type "$ego")
  add_ego_weights "$ego"

  # Hybrid: PPO-rollout fallback configuration. When PDM has no viable
  # proposal, run a beam search over PPO actions (constant-velocity others)
  # and pick the beam with the best outcome × behavior score.
  if [[ "$ego" == "hybrid" ]]; then
    args+=(--planner.hybrid.ppo-mode rollout)
    args+=(--planner.hybrid.ppo-rollout-strategy beam_search)
    args+=(--planner.hybrid.ppo-rollout-beam-width 4)
    args+=(--planner.hybrid.ppo-rollout-branch-factor 4)
    args+=(--planner.hybrid.ppo-rollout-horizon 10)
  fi

  # Traffic controller + its weights
  args+=(--traffic.type "$other")
  add_traffic_weights "$other"

  # Extra ego-planner args (e.g. CEM variants)
  args+=("${extra_ego_args[@]}")

  # Append extra args from command line
  args+=("${EXTRA_ARGS[@]}")

  JOB_NAME="eval_${ego}_vs_${other}"
  echo "Submitting: ${JOB_NAME}  ${extra_ego_args[*]}"
  sbatch --partition="$PARTITION" --job-name="$JOB_NAME" run_single_evaluation.sh "${args[@]}"
}

# ------------------------------------------------------------
# Planners:
#   IDM, PDM, PPO, SMART, hybrid,
#   conditioned_mix / conditioned_aggr / conditioned_normal / conditioned_caut
# Traffic: IDM, PPO, SMART, EXPERT, conditioned_*
# ------------------------------------------------------------

PLANNER_TYPES=(hybrid)
TRAFFIC_TYPES=(conditioned_normal)
# TRAFFIC_TYPES=(idm expert smart ppo conditioned_mix conditioned_aggr conditioned_normal conditioned_caut)
# TRAFFIC_TYPES=(conditioned_mix)

for SPLIT in "${SPLITS[@]}"; do
  for ego in "${PLANNER_TYPES[@]}"; do
    for traffic in "${TRAFFIC_TYPES[@]}"; do
      submit "$ego" "$traffic"
    done
  done
done
