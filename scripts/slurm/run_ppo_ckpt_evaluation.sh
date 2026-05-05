#!/bin/bash
# PPO-vs-PPO self-play evaluation across 4 training checkpoints from k1yx2van.
# Each row evaluates ego=ckpt_X against traffic=ckpt_X at that training step.
# k1yx2van trained with batch_size=524288 on 8 GPUs (DDP) → 1 update = 4.19e6
# agent_steps. Final agent_step ~1.05e11 (update 25000).
#
# Requested levels vs closest available checkpoint:
#   10^9   → no ckpt that early (min available is 4.19e9 at update 1000)
#   10^10  → update  2000  (8.39e9)
#   5×10^10 → update 12000 (5.03e10)
#   10^11  → update 24000  (1.01e11)
# We include update 1000 as a stand-in for the 10^9 level.

CKPT_DIRS=(
  # "simple:/work/dlclarge1/distelza-data/experiments/puffer_drive_k1yx2van"
  "complex:/work/dlclarge1/distelza-data/experiments/puffer_drive_xmdvjlpp"
)

# (label, file). Labels appear in the SLURM job name. Steps shown for reference.
# 8 log-uniform updates over the available range 1000..50000 (xmdvjlpp).
CKPTS=(
  "4e9:model_puffer_drive_001000.pt"    # 4.19e9
  "8e9:model_puffer_drive_002000.pt"    # 8.39e9
  "1e10:model_puffer_drive_003000.pt"   # 1.26e10
  "2e10:model_puffer_drive_005000.pt"   # 2.10e10
  "4e10:model_puffer_drive_009000.pt"   # 3.77e10
  "7e10:model_puffer_drive_016000.pt"   # 6.71e10
  "1e11:model_puffer_drive_029000.pt"   # 1.22e11
  "2e11:model_puffer_drive_050000.pt"   # 2.10e11
)

SPLITS=(pufferinter)
TRAFFIC_TYPES=(idm)
TRAFFIC_TYPES=(ppo idm)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${PUFFER_EXP_ROOT:-experiments}/ppo_ckpt_eval_${TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"

EXTRA_ARGS=("$@")

submit_pair () {
  local model_label="$1"   # e.g. simple, complex
  local ckpt_dir="$2"
  local label="$3"         # step-label, e.g. 5e10
  local ckpt="$4"          # filename, e.g. model_puffer_drive_015000.pt
  local traffic="$5"       # ppo | idm
  local split="$6"
  local weights="$ckpt_dir/$ckpt"

  if [[ ! -f "$weights" ]]; then
    echo "MISSING: $weights"; return
  fi

  # Per-checkpoint output dir: <model_label>_<ckpt_basename>
  local ckpt_basename="${ckpt%.pt}"
  local ckpt_out="$OUTPUT_ROOT/${model_label}_${ckpt_basename}"
  mkdir -p "$ckpt_out"

  args=(
    --eval.split "$split"
    --output-dir "$ckpt_out"
    --planner.type ppo
    --planner.ppo.weights-path "$weights"
    --traffic.type "$traffic"
  )
  if [[ "$traffic" == "ppo" ]]; then
    args+=(--traffic.ppo.weights-path "$weights")
  fi
  args+=("${EXTRA_ARGS[@]}")

  local job_name="eval_${model_label}_${label}_vs_${traffic}_${split}"
  echo "Submitting: ${job_name} -> ${ckpt_out}"
  sbatch --job-name="$job_name" run_single_evaluation.sh "${args[@]}"
}

for dir_entry in "${CKPT_DIRS[@]}"; do
  model_label="${dir_entry%%:*}"
  ckpt_dir="${dir_entry#*:}"
  for SPLIT in "${SPLITS[@]}"; do
    for entry in "${CKPTS[@]}"; do
      label="${entry%%:*}"
      ckpt="${entry##*:}"
      for traffic in "${TRAFFIC_TYPES[@]}"; do
        submit_pair "$model_label" "$ckpt_dir" "$label" "$ckpt" "$traffic" "$SPLIT"
      done
    done
  done
done
