#!/bin/bash
# Submit training jobs in parallel on the SLURM cluster
#
# Usage: bash scripts/slurm/run_all_training.sh
#
# Jobs:
#   Waymo:
#     1. Waymo PPO Self-play
#     2. Waymo PPO vs IDM (1 PPO, N-1 IDM at 10 m/s)
#     3. Waymo PPO mixed (50% PPO, 50% IDM at 10 m/s)
#   nuPlan:
#     4. nuPlan PPO Self-play
#     5. nuPlan PPO vs IDM (1 PPO, N-1 IDM at 15 m/s)
#     6. nuPlan PPO mixed (50% PPO, 50% IDM at 15 m/s)

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
EXP_DIR="/work/dlclarge1/distelza-data/puffer_exps"

WAYMO_DATA=${WAYMO_DATA:-/work/dlclarge2/distelza-gpudrive/binariesv3}

echo "Submitting training jobs..."
echo ""

# =============================================================================
# Waymo
# =============================================================================

# 1. Waymo PPO Self-play
# echo "1: Waymo PPO Self-play (all vehicles)"
# sbatch \
#   --job-name=waymo_selfplay \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_selfplay_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_selfplay_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name waymo_selfplay

# 2. Waymo PPO vs Expert (1 PPO ego, rest expert replay)
# echo "2: Waymo PPO vs Expert (1 PPO agent, N-1 expert replay)"
# sbatch \
#   --job-name=waymo_ppo_vs_expert \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_ppo_vs_expert_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_ppo_vs_expert_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --env.max-controlled-agents 1 \
#   --env.num-agents 64 \
#   --train.batch-size 32768 \
#   --train.minibatch-size 8192 \
#   --train.max-minibatch-size 8192 \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name waymo_ppo_vs_expert

# 3. Waymo PPO vs IDM (1 PPO ego, rest IDM at 10 m/s)
# echo "3: Waymo PPO vs IDM (1 PPO agent, N-1 IDM at 10 m/s)"
# sbatch \
#   --job-name=waymo_ppo_vs_idm \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_ppo_vs_idm_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_ppo_vs_idm_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --env.num-agents 64 \
#   --env.idm-others True \
#   --env.max-controlled-agents 1 \
#   --env.idm-target-velocity 10.0 \
#   --train.batch-size 32768 \
#   --train.minibatch-size 8192 \
#   --train.max-minibatch-size 8192 \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name waymo_ppo_vs_idm10

# # 3. Waymo PPO mixed (50% PPO, 50% IDM at 10 m/s)
# echo "3: Waymo PPO mixed (50% PPO, 50% IDM at 10 m/s)"
# sbatch \
#   --job-name=waymo_mixed_idm50 \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_mixed_idm50_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/waymo_mixed_idm50_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$WAYMO_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --env.traffic-mix "ppo:0.5,idm_10:0.5" \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name waymo_mixed_ppo50_idm50

# # =============================================================================
# # nuPlan
# # =============================================================================

NUPLAN_DATA=/work/dlclarge2/distelza-gpudrive/gpudrive_nuplan_binaries_178k

# 4. nuPlan PPO Self-play
# echo "4: nuPlan PPO Self-play (all vehicles)"
sbatch \
  --job-name=nuplan_selfplay \
  --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_selfplay_%j.out" \
  --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_selfplay_%j.err" \
  --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
  "$SCRIPT_DIR/run_single_training.sh" \
  --config pufferlib/config/ocean/drive_nuplan.ini \
  --wandb-project puffer \
  --train.data-dir "$EXP_DIR" \
  --train.name nuplan_selfplay \
  --train.total-timesteps 20_000_000_000 \

# 5. nuPlan PPO vs Expert (1 PPO ego, rest expert replay)
# echo "5: nuPlan PPO vs Expert (1 PPO agent, N-1 expert replay)"
# sbatch \
#   --job-name=nuplan_ppo_vs_expert \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_ppo_vs_expert_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_ppo_vs_expert_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --config pufferlib/config/ocean/drive_nuplan.ini \
#   --env.max-controlled-agents 1 \
#   --env.num-agents 64 \
#   --train.batch-size 32768 \
#   --train.minibatch-size 8192 \
#   --train.max-minibatch-size 8192 \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name nuplan_ppo_vs_expert

# # 6. nuPlan PPO vs IDM (1 PPO ego, rest IDM at 15 m/s)
# echo "5: nuPlan PPO vs IDM (1 PPO agent, N-1 IDM at 15 m/s)"
# sbatch \
#   --job-name=nuplan_ppo_vs_idm \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_ppo_vs_idm_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_ppo_vs_idm_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --config pufferlib/config/ocean/drive_nuplan_sdc.ini \
#   --env.idm-others True \
#   --env.max-controlled-agents 1 \
#   --env.idm-target-velocity 15.0 \
#   --env.collision-behavior 0 \
#   --env.offroad-behavior 0 \
#   --env.goal-behavior 4 \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name nuplan_ppo_vs_idm15

# # 6. nuPlan PPO mixed (50% PPO, 50% IDM at 15 m/s)
# echo "6: nuPlan PPO mixed (50% PPO, 50% IDM at 15 m/s)"
# sbatch \
#   --job-name=nuplan_mixed_idm50 \
#   --output="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_mixed_idm50_%j.out" \
#   --error="/work/dlclarge1/distelza-data/puffer_exps/log/nuplan_mixed_idm50_%j.err" \
#   --export=ALL,DRIVE_BINARIES_DATA_ROOT=$NUPLAN_DATA \
#   "$SCRIPT_DIR/run_single_training.sh" \
#   --config pufferlib/config/ocean/drive_nuplan.ini \
#   --env.traffic-mix "ppo:0.5,idm_15:0.5" \
#   --env.collision-behavior 0 \
#   --env.offroad-behavior 0 \
#   --env.goal-behavior 4 \
#   --wandb-project puffer \
#   --train.data-dir "$EXP_DIR" \
#   --train.name nuplan_mixed_ppo50_idm50

# echo ""
# echo "Jobs submitted. Check status with: squeue -u \$USER"
