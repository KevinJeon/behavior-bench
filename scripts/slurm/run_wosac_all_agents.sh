#!/bin/bash
# Submit WOSAC realism evaluation for all traffic agents on pufferinter & pufferrandom.
# Usage: ./run_wosac_all_agents.sh [extra args forwarded to all jobs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_wosac_single.sh"

PPO_WEIGHTS="/home/distelza/pufferdrive/weights/puffer_drive_pvp093uj.pt"
# SMART_WEIGHTS="${SMART_WEIGHTS:?Set SMART_WEIGHTS}"
CONDITIONED_WEIGHTS="/work/dlclarge1/distelza-data/experiments/puffer_drive_ukkwspio/model_puffer_drive_024000.pt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${PUFFER_EXP_ROOT:-experiments}/wosac_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"
echo "WOSAC output root: $OUTPUT_DIR"

EXTRA_ARGS=("$@")

submit () {
  local job_name="$1"
  shift
  echo "Submitting: ${job_name}"
  sbatch --job-name="$job_name" "$WORKER" "$@" "${EXTRA_ARGS[@]}"
}

# SPLITS=(pufferinter pufferrandom)
SPLITS=(pufferrandom)

for split in "${SPLITS[@]}"; do

  # # SMART open-loop
  # submit "wosac_smart_ol_${split}" \
  #   --planner.type smart \
  #   --realism.eval-mode open_loop \
  #   --planner.smart.weights-path "$SMART_WEIGHTS" \
  #   --realism.split "$split" \
  #   --map-ids all \
  #   --output-dir "$OUTPUT_DIR"

  # # SMART closed-loop
  # submit "wosac_smart_cl_${split}" \
  #   --planner.type smart \
  #   --realism.eval-mode closed_loop \
  #   --planner.smart.weights-path "$SMART_WEIGHTS" \
  #   --realism.split "$split" \
  #   --map-ids all \
  #   --output-dir "$OUTPUT_DIR"

  # PPO closed-loop
#   submit "wosac_ppo_${split}" \
#     --planner.type ppo \
#     --realism.eval-mode closed_loop \
#     --planner.ppo.weights-path "$PPO_WEIGHTS" \
#     --realism.split "$split" \
#     --map-ids all \
#     --output-dir "$OUTPUT_DIR"
#     # --realism.viz True \
#     # --realism.viz-maps 10 \

#   # IDM closed-loop
#   submit "wosac_idm_${split}" \
#     --planner.type idm \
#     --realism.eval-mode closed_loop \
#     --realism.split "$split" \
#     --map-ids all \
#     --output-dir "$OUTPUT_DIR"

#   # Random baseline
#   submit "wosac_random_${split}" \
#     --planner.type random \
#     --realism.eval-mode closed_loop \
#     --realism.split "$split" \
#     --map-ids all \
#     --output-dir "$OUTPUT_DIR"

#   # Ground-truth (sanity check)
#   submit "wosac_gt_${split}" \
#     --planner.type idm \
#     --realism.eval-mode ground_truth \
#     --realism.split "$split" \
#     --map-ids all \
#     --output-dir "$OUTPUT_DIR"

  # DriveConditioned variants (Aggr/Normal/Caut/Mix profiles).
  # The matching [planner.conditioned_<variant>] section in evaluation.ini
  # supplies the per-variant creward.* fields; all agents use that single
  # profile (WOSAC controls all agents with the same policy).
  for variant in conditioned_normal; do
  #for variant in conditioned_aggr conditioned_norm conditioned_caut conditioned_mix; do
    flag_name="${variant//_/-}"  # CLI flag uses hyphens; type value keeps underscores
    submit "wosac_${variant}_${split}" \
      --planner.type "$variant" \
      --realism.eval-mode closed_loop \
      --planner.${flag_name}.weights-path "$CONDITIONED_WEIGHTS" \
      --realism.split "$split" \
      --map-ids all \
      --output-dir "$OUTPUT_DIR"
  done

done

echo "Submitted 28 jobs. Track with: squeue -u \$USER"
