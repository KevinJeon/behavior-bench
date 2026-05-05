# PufferDrive Evaluation Framework

This document describes the config-driven evaluation framework for mixing and matching ego planners and traffic controllers in the PufferDrive simulator. The framework evaluates ego planners against traffic agents on Waymo Open Motion Dataset scenarios.

**Entry point:** `pufferlib/ocean/benchmark/eval.py`
**Default config:** `pufferlib/config/evaluation.ini`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Configuration System](#configuration-system)
4. [Available Planners](#available-planners)
5. [Execution Examples](#execution-examples)
6. [Map Selection](#map-selection)
7. [Metrics](#metrics)
8. [Collision Classification](#collision-classification)
9. [Uncertainty Estimation](#uncertainty-estimation)
10. [Visualization](#visualization)
11. [Output Structure](#output-structure)
12. [Adding a New Planner](#adding-a-new-planner)
13. [Running Tests](#running-tests)
14. [Architecture Overview](#architecture-overview)

---

## Pre-trained Weights

The repository includes pre-trained weights for PPO and SMART planners in the `weights/` directory:

| File | Description |
|------|-------------|
| `weights/ppo_self_play.pt` | PPO policy trained with self-play on WOMD (2.4 MB) |
| `weights/smart_1M_epoch_029.pt` | SMART prediction model, 1M parameters, 29 epochs (14 MB) |

These weights are used in the evaluation examples below and can be used as baselines for comparison.

---

## Quick Start

```bash
# Set the required environment variable
export DRIVE_BINARIES_DATA_ROOT=/path/to/binaries

# Run with defaults (PDM ego vs IDM traffic on the pufferhard split)
python pufferlib/ocean/benchmark/eval.py --map-ids 0-10

# PPO ego vs IDM traffic (using provided weights)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path weights/ppo_self_play.pt \
    --map-ids 0-50

# SMART ego vs IDM traffic (using provided weights)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path weights/smart_1M_epoch_029.pt \
    --map-ids 0-50

# PPO ego vs SMART traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path weights/ppo_self_play.pt \
    --traffic.type smart \
    --traffic.smart.weights-path weights/smart_1M_epoch_029.pt \
    --map-ids 0-50
```

---

## Prerequisites

1. **Environment variable:** `DRIVE_BINARIES_DATA_ROOT` must point to the root directory containing Waymo binary scenario files. The script will exit immediately if this is not set.

2. **C extensions:** The PufferDrive simulator uses compiled C extensions. If you modify the C code, rebuild with:
   ```bash
   python setup.py build_ext --inplace
   ```

3. **Model weights** (optional): PPO, SMART, and Hybrid planners require pre-trained model weights. The default paths are configured in `evaluation.ini` but can be overridden via CLI.

4. **GPU** (optional): PPO, SMART, and Hybrid planners default to CUDA. Override with `--planner.<type>.device cpu` if no GPU is available.

---

## Configuration System

### INI File Structure

The default configuration lives at `pufferlib/config/evaluation.ini`. It is organized into sections:

```ini
[env]                          # Drive C environment settings
action_type = continuous
episode_length = 91
goal_behavior = 3              # 0=respawn, 1=generate_new, 2=stop, 3=remove
termination_mode = 1           # 0=episode_length only, 1=all agents done
collision_behavior = 2         # 0=ignore, 1=stop, 2=remove
offroad_behavior = 2           # 0=ignore, 1=stop, 2=remove
dt = 0.1
collision_shrink = 0.7
reward_vehicle_collision = -0.5
reward_offroad_collision = -0.5

[eval]                         # General evaluation settings
split = pufferhard
episode_length = 91
action_type = continuous
viz = False                    # Scene visualization (per-step PNGs + GIF)
planner_viz = False            # Planner-specific visualization (PDM proposals, etc.)
goal_behavior = 3
termination_mode = 1
collision_behavior = 2
offroad_behavior = 2
compute_uncertainty = False    # Epistemic/aleatoric uncertainty estimation
ensemble_weight_paths =        # Comma-separated paths to ensemble weights

[planner]                      # Ego planner selection
type = pdm                     # pdm|ppo|idm|smart|hybrid|conditioned_aggr|conditioned_normal|conditioned_caut|constant_velocity

[planner.pdm]                  # PDM-specific parameters
horizon = 40
proposal_other = constant_velocity
velocity_fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
lateral_offsets = (-1.0, 0.0, 1.0)
max_velocity = 25.0
min_gap = 1.0
headway_time = 1.5
accel_max = 1.5
decel_max = 3.0

[planner.ppo]                  # PPO-specific parameters
weights_path = /path/to/weights.pt
device = cuda
ensemble_weights =

[planner.idm]                  # IDM-specific parameters
target_velocity = 10.0
min_gap = 1.0
headway_time = 1.5
accel_max = 1.0
decel_max = 3.0

[planner.smart]                # SMART-specific parameters
weights_path = /path/to/smart_weights.pt
device = cuda
temperature = 1.0
greedy = True
repredict_interval = 5

[planner.hybrid]               # Hybrid PPO+PDM parameters
ppo_weights_path =
pdm_min_steps = 1
force_ppo = false
force_pdm = false
switch_mode = pdm_score        # default: switch to PDM when its rollout score beats PPO's
lookahead_steps = 0
device = cuda
; --- PPO mode: how PPO acts when selected ---
ppo_mode = rollout             # single_step | rollout (default)
ppo_rollout_strategy = beam_search   # beam_search is the default rollout strategy
ppo_rollout_top_k = 8          # candidate actions sampled per step from PPO logits
ppo_rollout_beam_width = 4     # beams kept per expansion
ppo_rollout_branch_factor = 4  # branches per beam
ppo_rollout_horizon = 10       # rollout length (steps)
ppo_rollout_w_cmf = 0.333      # comfort weight in rollout score
ppo_rollout_w_align = 0.333    # lane-alignment weight
ppo_rollout_w_ctr = 0.333      # center-bias weight
ppo_rollout_lane_dist_scale = 2.0

[planner.constant_velocity]    # No parameters

[planner.conditioned_aggr]     # Reward-conditioned PPO (aggressive profile)
weights_path = /path/to/conditioned_ppo.pt
device = cuda
creward.alpha_collision = 0.2  # 9 creward.* fields define the reward profile
creward.alpha_boundary = 0.2
creward.alpha_comfort = 0.0
creward.alpha_l_align = 0.001
creward.alpha_vel_align = 0.1
creward.alpha_l_center = 0.00025
creward.alpha_center_bias = 0.0
creward.alpha_reverse = 0.0005
creward.goal_speed = 30.0

[planner.conditioned_normal]   # Reward-conditioned PPO (normal profile)
weights_path = /path/to/conditioned_ppo.pt
device = cuda
; ... same creward.* fields, normal-driving values

[planner.conditioned_caut]     # Reward-conditioned PPO (cautious profile)
weights_path = /path/to/conditioned_ppo.pt
device = cuda
; ... same creward.* fields, cautious-driving values

[traffic]                      # Traffic controller selection
type = idm                     # pdm|ppo|idm|smart|expert|conditioned_mix|conditioned_aggr|conditioned_normal|conditioned_caut|constant_velocity

[traffic.pdm]                  # Traffic PDM parameters
horizon = 40
proposal_other = constant_velocity

[traffic.ppo]                  # Traffic PPO parameters
weights_path = /path/to/weights.pt
device = cuda

[traffic.idm]                  # Traffic IDM parameters
target_velocity = 10.0
min_gap = 1.0
headway_time = 1.5
accel_max = 1.0
decel_max = 3.0

[traffic.smart]                # Traffic SMART parameters
weights_path = /path/to/smart_weights.pt
device = cuda
temperature = 1.0
greedy = True
repredict_interval = 5

[traffic.expert]               # Expert replay (no parameters)

[traffic.constant_velocity]    # No parameters

[traffic.conditioned_mix]      # Mix of aggr/normal/caut profiles per agent
weights_path = /path/to/conditioned_ppo.pt
device = cuda
creward_profiles = [
    (0.2, 0.2, 0.0,  0.001, 0.1, 0.00025, 0.0, 0.0005, 30.0),  # aggressive
    (3.0, 3.0, 0.05, 0.015, 0.5, 0.0038,  0.0, 0.0038, 20.0),  # normal
    (3.0, 3.0, 0.1,  0.025, 1.0, 0.0075,  0.5, 0.0075,  5.0),  # cautious
    ]

[traffic.conditioned_aggr]     # All traffic agents use the aggressive profile
weights_path = /path/to/conditioned_ppo.pt
device = cuda
creward_profiles = [(0.2, 0.2, 0.0, 0.001, 0.1, 0.00025, 0.0, 0.0005, 30.0)]

[traffic.conditioned_normal]   # All traffic agents use the normal profile
weights_path = /path/to/conditioned_ppo.pt
device = cuda
creward_profiles = [(3.0, 3.0, 0.05, 0.015, 0.5, 0.0038, 0.0, 0.0038, 20.0)]

[traffic.conditioned_caut]     # All traffic agents use the cautious profile
weights_path = /path/to/conditioned_ppo.pt
device = cuda
creward_profiles = [(3.0, 3.0, 0.1, 0.025, 1.0, 0.0075, 0.5, 0.0075, 5.0)]
```

### CLI Override System

Any parameter in the INI file can be overridden from the command line using dot-notation. The general pattern is:

```
--section.key value
```

Underscores in key names are converted to hyphens on the CLI:

```bash
# Override planner type
--planner.type ppo

# Override a nested planner parameter
--planner.ppo.weights-path /path/to/weights.pt

# Override traffic controller type and parameters
--traffic.type smart
--traffic.smart.weights-path /path/to/smart.pt
--traffic.smart.temperature 0.5

# Override eval settings
--eval.viz True
--eval.split validation
--eval.episode-length 50
```

### Additional CLI-only Arguments

These arguments are not in the INI file and are only available on the command line:

| Argument | Default | Description |
|----------|---------|-------------|
| `--map-ids` | None (all maps in split) | Map selection: `all`, `0-100` (range), `0,5,10` (list) |
| `--output-dir` | None (auto: `experiments/<timestamp>_<uuid>`) | Custom output directory |
| `--config` | None | Path to a custom INI file layered on top of defaults |

### Custom Config Files

You can layer a custom INI file on top of the defaults. CLI arguments still take priority:

```bash
python pufferlib/ocean/benchmark/eval.py \
    --config my_experiment.ini \
    --map-ids 0-50
```

---

## Available Planners

### PDM (Predictive Driver Model)

Proposal-based planning. Generates multiple trajectory candidates using IDM with different target velocities and lateral offsets, evaluates each candidate against a cost function, and selects the best trajectory.

- **Available as:** ego planner, traffic controller
- **Key parameters:**
  - `horizon` (int, default 40): Planning horizon in timesteps
  - `velocity_fractions` (tuple, default (0.2, 0.4, 0.6, 0.8, 1.0)): Fractions of max_velocity to use as IDM targets
  - `lateral_offsets` (tuple, default (-1.0, 0.0, 1.0)): Lateral offset candidates in meters
  - `max_velocity` (float, default 25.0): Maximum velocity in m/s (fractions are multiplied by this)
  - `min_gap` (float, default 1.0): Minimum gap to lead vehicle in meters
  - `headway_time` (float, default 1.5): Time headway in seconds
  - `accel_max` (float, default 1.5): Maximum acceleration in m/s^2
  - `decel_max` (float, default 3.0): Maximum deceleration in m/s^2
  - `proposal_other` (str, default "constant_velocity"): Planner to predict other agents during proposal evaluation

### PPO (Proximal Policy Optimization)

Pre-trained reinforcement learning policy with LSTM. Requires a checkpoint file containing the trained model weights.

- **Available as:** ego planner, traffic controller
- **Key parameters:**
  - `weights_path` (str): Path to the trained model checkpoint (.pt file)
  - `device` (str, default "cuda"): Device for inference (cuda or cpu)
  - `ensemble_weights` (str): Comma-separated paths for ensemble uncertainty

### SMART (Scalable Motion prediction with Autoregressive Trajectory generation)

Autoregressive trajectory prediction model. Predicts future trajectories using motion tokens. Can operate in greedy or sampling mode.

- **Available as:** ego planner, traffic controller
- **Key parameters:**
  - `weights_path` (str): Path to trained SMART checkpoint (.pt file)
  - `device` (str, default "cuda"): Device for inference
  - `temperature` (float, default 1.0): Sampling temperature (higher = more diverse)
  - `greedy` (bool, default True): Use greedy decoding (deterministic)
  - `repredict_interval` (int, default 5): Re-run prediction every N steps

### IDM (Intelligent Driver Model)

Classical rule-based car-following model. Uses lane connectivity for route chaining. Does not require any model weights. Good baseline and default traffic controller.

- **Available as:** ego planner, traffic controller
- **Key parameters:**
  - `target_velocity` (float, default 10.0): Desired velocity in m/s
  - `min_gap` (float, default 1.0): Minimum gap to lead vehicle in meters
  - `headway_time` (float, default 1.5): Time headway in seconds
  - `accel_max` (float, default 1.0): Maximum acceleration in m/s^2
  - `decel_max` (float, default 3.0): Maximum deceleration in m/s^2

### Hybrid (PPO + PDM with PDM-score Switching and Beam-Search Rollout)

Combines a PPO policy with the PDM proposal-based planner. By default, PPO acts in **rollout mode** with **beam-search** strategy: at each step, the top-K actions sampled from the PPO logits are expanded into short batched rollouts, scored with a comfort × lane-alignment × center-bias objective, and the best beam wins. The hybrid switches between PPO and PDM using `switch_mode = pdm_score` (default), which compares PDM's rollout score against PPO's and picks the better trajectory.

- **Available as:** ego planner only
- **Key parameters:**
  - `ppo_weights_path` (str): Path to PPO checkpoint
  - `switch_mode` (str, default `"pdm_score"`): Switching criterion. Default compares PDM's rollout score with PPO's
  - `pdm_min_steps` (int, default 1): Minimum consecutive PDM steps after switching
  - `force_ppo` (bool, default false): Force PPO only (disables switching)
  - `force_pdm` (bool, default false): Force PDM only (disables PPO)
  - `lookahead_steps` (int, default 0): Steps to look ahead for switching decision
  - `device` (str, default "cuda"): Device for neural network inference

  **PPO rollout parameters (defaults):**
  - `ppo_mode` (str, default `"rollout"`): `single_step` for a one-shot PPO action, `rollout` for batched beam-search (default)
  - `ppo_rollout_strategy` (str, default `"beam_search"`): Rollout expansion strategy
  - `ppo_rollout_top_k` (int, default 8): Number of candidate actions sampled from the PPO logits per step
  - `ppo_rollout_beam_width` (int, default 4): Beams kept after each expansion
  - `ppo_rollout_branch_factor` (int, default 4): Branches grown from each beam
  - `ppo_rollout_horizon` (int, default 10): Rollout length in steps
  - `ppo_rollout_w_cmf` / `ppo_rollout_w_align` / `ppo_rollout_w_ctr` (float, default 0.333 each): Score weights for comfort, lane alignment and center bias
  - `ppo_rollout_lane_dist_scale` (float, default 2.0): Scale used when normalizing lane distance in the score

### Conditioned PPO Variants (`conditioned_aggr`, `conditioned_normal`, `conditioned_caut`, `conditioned_mix`)

Reward-conditioned PPO policy (`DriveConditioned`). The base architecture matches PPO but appends a 9-dim reward-conditioning (creward) vector to the observation. Each variant points to the same trained checkpoint but feeds a different creward profile that biases driving style:

- `conditioned_aggr`: aggressive (low collision/boundary penalties, high goal speed)
- `conditioned_normal`: normal driving (moderate penalties, default goal speed)
- `conditioned_caut`: cautious (high penalties, comfort-aware, low goal speed)
- `conditioned_mix` (**traffic only**): rotates per-agent through multiple profiles, producing heterogeneous traffic

When a conditioned variant is in the mix, the evaluator auto-enables `reward_conditioning=1`.

- **Available as:**
  - ego planner: `conditioned_aggr`, `conditioned_normal`, `conditioned_caut`
  - traffic controller: all four variants (incl. `conditioned_mix`)
- **Key parameters:**
  - `weights_path` (str): Path to the trained DriveConditioned checkpoint (.pt file)
  - `device` (str, default "cuda"): Device for inference
  - **Ego variants** use scalar `creward.<name>` fields (9 values: `delta_goal`, `alpha_collision`, `alpha_boundary`, `alpha_comfort`, `alpha_l_align`, `alpha_vel_align`, `alpha_l_center`, `alpha_center_bias`, `alpha_reverse`, plus `goal_speed`)
  - **Traffic variants** use `creward_profiles`: a list of 9-tuples `(alpha_collision, alpha_boundary, alpha_comfort, alpha_l_align, alpha_vel_align, alpha_l_center, alpha_center_bias, alpha_reverse, goal_speed)`. Multiple tuples (as in `conditioned_mix`) are distributed across agents.

### Expert (Ground Truth Replay)

Replays ground truth trajectories from the Waymo dataset. Provides an upper bound for traffic realism.

- **Available as:** traffic controller only
- **Key parameters:** None

### Constant Velocity

Simple baseline that maintains the current velocity with zero steering. Useful as a lower bound for comparison.

- **Available as:** ego planner, traffic controller
- **Key parameters:** None

---

## Execution Examples

### Basic Usage

```bash
# Default: PDM ego vs IDM traffic on pufferhard split
python pufferlib/ocean/benchmark/eval.py --map-ids 0-10

# Run on all maps in the split
python pufferlib/ocean/benchmark/eval.py --map-ids all

# Run on specific maps
python pufferlib/ocean/benchmark/eval.py --map-ids 0,5,10,20,50
```

### PDM Ego Planner Configurations

```bash
# PDM with default parameters vs IDM traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --traffic.type idm \
    --map-ids 0-100

# PDM with custom velocity fractions and lateral offsets
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --planner.pdm.velocity-fractions "(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)" \
    --planner.pdm.lateral-offsets "(-2.0, -1.0, 0.0, 1.0, 2.0)" \
    --planner.pdm.horizon 60 \
    --map-ids 0-50

# PDM with aggressive driving parameters
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --planner.pdm.max-velocity 30.0 \
    --planner.pdm.accel-max 3.0 \
    --planner.pdm.decel-max 5.0 \
    --planner.pdm.headway-time 0.8 \
    --map-ids 0-50

# PDM with conservative driving parameters
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --planner.pdm.max-velocity 15.0 \
    --planner.pdm.accel-max 1.0 \
    --planner.pdm.decel-max 2.0 \
    --planner.pdm.headway-time 2.5 \
    --planner.pdm.min-gap 3.0 \
    --map-ids 0-50

# PDM ego vs Expert traffic (ground truth replay)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --traffic.type expert \
    --map-ids 0-100

# PDM ego vs Constant Velocity traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --traffic.type constant-velocity \
    --map-ids 0-50

# PDM ego vs PDM traffic (both agents use PDM)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --traffic.type pdm \
    --traffic.pdm.horizon 20 \
    --map-ids 0-50
```

### PPO Ego Planner Configurations

```bash
# PPO ego vs IDM traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --traffic.type idm \
    --map-ids 0-100

# PPO ego vs Expert traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --traffic.type expert \
    --map-ids 0-100

# PPO ego vs SMART traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --traffic.type smart \
    --traffic.smart.weights-path /path/to/smart_weights.pt \
    --map-ids 0-50

# PPO ego on CPU
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --planner.ppo.device cpu \
    --map-ids 0-10

# PPO ego vs PPO traffic (self-play evaluation)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ego_model.pt \
    --traffic.type ppo \
    --traffic.ppo.weights-path /path/to/traffic_model.pt \
    --map-ids 0-100
```

### SMART Planner Configurations

```bash
# SMART ego vs IDM traffic (greedy decoding)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path /path/to/smart_weights.pt \
    --planner.smart.greedy True \
    --traffic.type idm \
    --map-ids 0-100

# SMART ego with sampling (stochastic)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path /path/to/smart_weights.pt \
    --planner.smart.greedy False \
    --planner.smart.temperature 0.8 \
    --traffic.type idm \
    --map-ids 0-50

# SMART ego with frequent re-prediction
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path /path/to/smart_weights.pt \
    --planner.smart.repredict-interval 1 \
    --traffic.type idm \
    --map-ids 0-50

# SMART as traffic controller (PDM ego vs SMART traffic)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --traffic.type smart \
    --traffic.smart.weights-path /path/to/smart_weights.pt \
    --traffic.smart.temperature 1.0 \
    --map-ids 0-100

# SMART ego vs SMART traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path /path/to/ego_smart.pt \
    --traffic.type smart \
    --traffic.smart.weights-path /path/to/traffic_smart.pt \
    --map-ids 0-50

# SMART ego vs Expert traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type smart \
    --planner.smart.weights-path /path/to/smart_weights.pt \
    --traffic.type expert \
    --map-ids 0-100
```

### IDM Ego Planner Configurations

```bash
# IDM ego vs IDM traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type idm \
    --traffic.type idm \
    --map-ids 0-100

# IDM ego with custom parameters vs Expert traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type idm \
    --planner.idm.target-velocity 15.0 \
    --planner.idm.min-gap 2.0 \
    --planner.idm.headway-time 2.0 \
    --planner.idm.accel-max 2.0 \
    --planner.idm.decel-max 4.0 \
    --traffic.type expert \
    --map-ids 0-100

# IDM ego vs Constant Velocity traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type idm \
    --traffic.type constant-velocity \
    --map-ids 0-50
```

### Hybrid Planner Configurations

```bash
# Hybrid (PPO + PDM) — default: PPO beam-search rollout + pdm_score switching, vs IDM traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --traffic.type idm \
    --map-ids 0-100

# Hybrid with a wider beam and longer rollout horizon
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --planner.hybrid.ppo-rollout-beam-width 8 \
    --planner.hybrid.ppo-rollout-branch-factor 6 \
    --planner.hybrid.ppo-rollout-horizon 20 \
    --traffic.type idm \
    --map-ids 0-50

# Hybrid in single-step PPO mode (no rollout) — falls back to plain PPO action
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --planner.hybrid.ppo-mode single_step \
    --traffic.type idm \
    --map-ids 0-100

# Hybrid forced to PPO only (for ablation, still uses beam-search rollout)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --planner.hybrid.force-ppo true \
    --traffic.type idm \
    --map-ids 0-100

# Hybrid forced to PDM only (for ablation)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.force-pdm true \
    --traffic.type idm \
    --map-ids 0-100

# Hybrid with minimum PDM steps after switching, vs Expert traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --planner.hybrid.pdm-min-steps 5 \
    --traffic.type expert \
    --map-ids 0-100

# Hybrid with custom rollout score weights (boost comfort, downweight lane alignment)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type hybrid \
    --planner.hybrid.ppo-weights-path /path/to/ppo_model.pt \
    --planner.hybrid.ppo-rollout-w-cmf 0.6 \
    --planner.hybrid.ppo-rollout-w-align 0.2 \
    --planner.hybrid.ppo-rollout-w-ctr 0.2 \
    --traffic.type idm \
    --map-ids 0-50
```

### Conditioned PPO Configurations

```bash
# Conditioned PPO ego (aggressive profile) vs IDM traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type conditioned_aggr \
    --planner.conditioned-aggr.weights-path weights/conditioned_ppo.pt \
    --traffic.type idm \
    --map-ids 0-100

# Conditioned PPO ego (cautious profile) vs Mixed conditioned traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type conditioned_caut \
    --planner.conditioned-caut.weights-path weights/conditioned_ppo.pt \
    --traffic.type conditioned_mix \
    --traffic.conditioned-mix.weights-path weights/conditioned_ppo.pt \
    --map-ids 0-100

# Conditioned (normal) ego vs Conditioned (aggressive) traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type conditioned_normal \
    --planner.conditioned-normal.weights-path weights/conditioned_ppo.pt \
    --traffic.type conditioned_aggr \
    --traffic.conditioned-aggr.weights-path weights/conditioned_ppo.pt \
    --map-ids 0-100

# PPO ego vs Conditioned mix traffic (heterogeneous traffic styles)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path weights/ppo_self_play.pt \
    --traffic.type conditioned_mix \
    --traffic.conditioned-mix.weights-path weights/conditioned_ppo.pt \
    --map-ids 0-50
```

### Constant Velocity Ego Planner

```bash
# Constant velocity ego vs IDM traffic (baseline)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type constant-velocity \
    --traffic.type idm \
    --map-ids 0-100

# Constant velocity ego vs Expert traffic
python pufferlib/ocean/benchmark/eval.py \
    --planner.type constant-velocity \
    --traffic.type expert \
    --map-ids 0-100
```

### Visualization Examples

```bash
# Scene visualization (road, agents, trajectories per step, creates GIF)
python pufferlib/ocean/benchmark/eval.py \
    --eval.viz True \
    --map-ids 5

# Planner-specific visualization (PDM proposal trajectories, costs, etc.)
python pufferlib/ocean/benchmark/eval.py \
    --planner.type pdm \
    --eval.viz True \
    --eval.planner-viz True \
    --map-ids 0-5

# Both visualization modes with PPO ego
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/weights.pt \
    --eval.viz True \
    --eval.planner-viz True \
    --map-ids 0,3,7
```

### Split and Output Examples

```bash
# Evaluate on the validation split
python pufferlib/ocean/benchmark/eval.py \
    --eval.split validation \
    --map-ids 0-50

# Evaluate on the training split
python pufferlib/ocean/benchmark/eval.py \
    --eval.split training \
    --map-ids 0-200

# Evaluate on the testing split
python pufferlib/ocean/benchmark/eval.py \
    --eval.split testing \
    --map-ids all

# Custom output directory
python pufferlib/ocean/benchmark/eval.py \
    --output-dir /path/to/my/results \
    --map-ids 0-100

# Use a custom config file with CLI overrides on top
python pufferlib/ocean/benchmark/eval.py \
    --config experiments/my_config.ini \
    --planner.pdm.horizon 60 \
    --map-ids 0-50
```

### Environment Parameter Overrides

```bash
# Shorter episodes
python pufferlib/ocean/benchmark/eval.py \
    --eval.episode-length 50 \
    --map-ids 0-100

# Ignore collisions (agents pass through each other)
python pufferlib/ocean/benchmark/eval.py \
    --eval.collision-behavior 0 \
    --map-ids 0-100

# Stop on collision instead of removing agent
python pufferlib/ocean/benchmark/eval.py \
    --eval.collision-behavior 1 \
    --map-ids 0-100

# Change goal behavior to stop at goal (instead of removing agent)
python pufferlib/ocean/benchmark/eval.py \
    --eval.goal-behavior 2 \
    --map-ids 0-100
```

### Uncertainty Estimation

```bash
# Enable uncertainty estimation with PPO ensemble
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --eval.compute-uncertainty True \
    --eval.ensemble-weight-paths "/path/to/w1.pt,/path/to/w2.pt,/path/to/w3.pt" \
    --map-ids 0-50

# Uncertainty is auto-enabled when ensemble weights are provided
python pufferlib/ocean/benchmark/eval.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_model.pt \
    --planner.ppo.ensemble-weights "/path/to/w1.pt,/path/to/w2.pt,/path/to/w3.pt" \
    --map-ids 0-50
```

---

## Map Selection

Maps are selected using the `--map-ids` argument:

| Syntax | Meaning | Example |
|--------|---------|---------|
| `all` | All `.bin` files in the split directory | `--map-ids all` |
| `start-end` | Inclusive range | `--map-ids 0-100` |
| `a,b,c` | Specific map indices | `--map-ids 0,5,10,42` |
| (omitted) | All maps in the split | (no `--map-ids` flag) |

The split directory is resolved as `$DRIVE_BINARIES_DATA_ROOT/<split>/` where `<split>` is the value of `--eval.split` (default: `pufferhard`).

Each map corresponds to a `.bin` file in the split directory. The framework reads `manifest.csv` from the split directory to determine which agent in each scenario is the ego agent.

---

## Metrics

The evaluator collects the following per-map metrics (defined in `pufferlib/evaluation/metrics.py`):

| Metric | Description |
|--------|-------------|
| `total_reward` | Cumulative reward over the episode |
| `episode_return` | Episode return from environment info |
| `score` | Composite score from environment |
| `num_steps` | Number of steps the ego agent was active |
| `goal_reached` | Whether the ego agent reached its goal |
| `final_goal_distance` | Distance to goal at episode end |
| `collision_rate` | Percentage of steps where a collision occurred |
| `at_fault_collision_rate` | Percentage of steps with at-fault collisions |
| `offroad_rate` | Percentage of steps the ego was off-road |
| `lane_alignment_rate` | Percentage of steps with good lane alignment |
| `planning_time_ms` | Total planning time (milliseconds) |
| `avg_planning_time_ms` | Average planning time per step |
| `total_time_s` | Total wall-clock time for the map |
| `mean_aleatoric` | Mean aleatoric uncertainty (if computed) |
| `mean_epistemic` | Mean epistemic uncertainty (if computed) |
| `mean_value_variance` | Mean value variance (if computed) |

Metrics are aggregated across maps into a summary with mean, standard deviation, min, and max for each numeric metric.

---

## Collision Classification

Collisions are classified following the nuPlan/PDM scorer logic (implemented in `pufferlib/evaluation/collision_classifier.py`). Each collision is categorized into one of five types:

| Type | Description | At Fault? |
|------|-------------|-----------|
| `STOPPED_EGO_COLLISION` | Ego is stationary when hit | No |
| `STOPPED_TRACK_COLLISION` | Ego hits a stationary agent | Yes |
| `ACTIVE_FRONT_COLLISION` | Ego's front bumper hits another agent | Yes |
| `ACTIVE_REAR_COLLISION` | Another agent hits ego from behind | No |
| `ACTIVE_LATERAL_COLLISION` | Side collision | Yes (if ego is in multiple lanes) |

The stopped-speed threshold is 0.05 m/s. Collision snapshots (positions, velocities, headings at time of collision) are saved to `collision_snapshots.json` in the output directory for post-hoc analysis.

---

## Uncertainty Estimation

The framework supports epistemic and aleatoric uncertainty estimation using policy ensembles (implemented in `pufferlib/evaluation/uncertainty.py`).

**Aleatoric uncertainty** captures inherent noise in the environment (irreducible). It is estimated from the variance of the action distribution output by a single policy.

**Epistemic uncertainty** captures model uncertainty (reducible with more data). It is estimated from the disagreement between multiple independently trained models (ensemble).

Uncertainty is automatically enabled when ensemble weight paths are provided. The output includes:
- Per-step uncertainty values saved in metrics
- Correlation plots between uncertainty and reward saved to the output directory
- Log messages reporting Pearson correlation coefficients

---

## Visualization

### Scene Visualization (`--eval.viz True`)

Renders a bird's-eye view of the scene at each timestep, showing:
- Road geometry (lanes, edges, crosswalks)
- Agent bounding boxes (ego highlighted)
- Goal position
- Agent trajectories

Each map gets a subdirectory with per-step PNG images and an animated GIF:

```
map_000/
  step_000.png
  step_001.png
  ...
  episode.gif
```

### Planner Visualization (`--eval.planner-viz True`)

Renders planner-specific debug information. For PDM, this includes:
- All trajectory proposals (colored by cost)
- Selected trajectory (highlighted)
- Cost breakdown per proposal

This is most useful for debugging planner behavior on specific scenarios.

---

## Output Structure

Each evaluation run creates a timestamped output directory:

```
experiments/<YYYYMMDD_HHMMSS>_<6-char-uuid>/
    config.json              # Full configuration snapshot (INI defaults + CLI overrides)
    eval.log                 # Timestamped log file
    per_map.csv              # Per-map metrics table
    summary.json             # Aggregated statistics (mean, std, min, max)
    collision_snapshots.json # Collision event details (if any collisions occurred)
    map_000/                 # Per-map visualization (if --eval.viz True)
        step_000.png
        step_001.png
        ...
        episode.gif
    map_001/
        ...
```

The `config.json` file captures the exact configuration used, making every run reproducible. To rerun with the same settings, inspect the JSON and reconstruct the CLI arguments.

The `summary.json` contains aggregated metrics:
```json
{
    "num_maps": 100,
    "reward": {"mean": 12.5, "std": 3.2, "min": -5.0, "max": 25.0},
    "collision_rate": {"mean": 0.05, "std": 0.03, ...},
    "goal_reached_rate": 0.85,
    "total_time_s": 120.5,
    ...
}
```

---

## Adding a New Planner

To add a new planner to the evaluation framework, follow these five steps:

### Step 1: Implement the Planner

Create a new file `pufferlib/planning/<your_planner>.py` that extends `BasePlanner`:

```python
"""Your planner description."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from pufferlib.planning.base import BasePlanner, PlanResult


@dataclass
class YourPlannerConfig:
    """Configuration for YourPlanner."""
    param_a: float = 1.0
    param_b: int = 10


class YourPlanner(BasePlanner):
    """Your planner implementation."""

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: YourPlannerConfig,
    ):
        action_dim = len(action_lb)
        super().__init__(
            horizon=config.param_b,
            action_dim=action_dim,
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config

    def plan(
        self,
        current_step: int = 0,
        obs: Optional[np.ndarray] = None,
        extract_trajectories: bool = False,
    ) -> np.ndarray:
        """Plan next action for the agent.

        Must return an action array of shape (action_dim,).
        """
        # Your planning logic here
        action = np.zeros(self.action_dim, dtype=np.float32)
        return action

    def plot(self, ax, state, axis_limits=None):
        """Plot planner-specific visualization."""
        # Optional: add planner debug visualization
        pass

    @property
    def population_size(self) -> int:
        """Number of candidate sequences evaluated per iteration."""
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        """Return True if your planner generates trajectory proposals."""
        return False

    def reset(self):
        """Reset internal state between episodes."""
        pass
```

### Step 2: Register in the Planner Registry

Edit `pufferlib/planning/registry.py` and add your planner in three places:

**A. Add to `_get_planner_class()`:**

```python
def _get_planner_class(planner_type: str):
    # ... existing entries ...
    elif planner_type == "your_planner":
        from pufferlib.planning.your_planner import YourPlanner, YourPlannerConfig
        return YourPlanner, YourPlannerConfig
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")
```

**B. Add a config builder function:**

```python
def _build_your_planner_config(cfg: dict):
    from pufferlib.planning.your_planner import YourPlannerConfig
    return YourPlannerConfig(
        param_a=float(cfg.get("param_a", 1.0)),
        param_b=int(cfg.get("param_b", 10)),
    )
```

**C. Add to `create_ego_planner()` and `create_traffic_controller()`:**

```python
def create_ego_planner(config, env, action_config, traffic_controller=None, ego_agent_idx=0):
    # ... existing entries ...
    elif planner_type == "your_planner":
        yp_cfg = _build_your_planner_config(type_cfg)
        return cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                   config=yp_cfg)

def create_traffic_controller(config, env, action_config, ego_agent_idx=0):
    # ... existing entries ...
    elif traffic_type == "your_planner":
        yp_cfg = _build_your_planner_config(type_cfg)
        return cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                   config=yp_cfg)
```

### Step 3: Add Default Configuration

Add sections to `pufferlib/config/evaluation.ini`:

```ini
[planner.your_planner]
param_a = 1.0
param_b = 10

[traffic.your_planner]
param_a = 1.0
param_b = 10
```

### Step 4: Export from `__init__.py`

Edit `pufferlib/planning/__init__.py`:

```python
from .your_planner import YourPlanner, YourPlannerConfig

__all__ = [
    # ... existing exports ...
    "YourPlanner", "YourPlannerConfig",
]
```

### Step 5: Add Integration Test

Add test cases to `tests/test_eval_planners.py`:

```python
def test_your_planner_vs_idm(self):
    """YourPlanner ego with IDM traffic."""
    config = _build_config("your_planner", "idm", ego_kwargs={"param_a": 2.0})
    summary = _run_evaluation(config)
    self._assert_valid_summary(summary)

def test_pdm_vs_your_planner_traffic(self):
    """PDM ego with YourPlanner traffic."""
    config = _build_config("pdm", "your_planner", ego_kwargs={"horizon": 5})
    summary = _run_evaluation(config)
    self._assert_valid_summary(summary)
```

After completing all five steps, your planner is fully integrated and can be used via:

```bash
python pufferlib/ocean/benchmark/eval.py \
    --planner.type your-planner \
    --planner.your-planner.param-a 2.0 \
    --map-ids 0-10
```

---

## Running Tests

### Unit and Integration Tests

The test suite is at `tests/test_eval_planners.py`. It tests config loading (no data needed) and planner combinations (requires data).

```bash
# Run all tests (skips weight-dependent and data-dependent tests automatically)
pytest tests/test_eval_planners.py -v

# Run only config tests (no data or weights needed)
pytest tests/test_eval_planners.py -v -k "TestEvalConfig"

# Run planner integration tests (requires DRIVE_BINARIES_DATA_ROOT)
pytest tests/test_eval_planners.py -v -k "TestPlannerCombinations"

# Run a specific test
pytest tests/test_eval_planners.py -v -k "test_pdm_vs_idm"

# Stop on first failure
pytest tests/test_eval_planners.py -v -x
```

### Benchmark-Specific Tests

Additional tests exist in the benchmark directory:

```bash
# Map metrics tests
pytest pufferlib/ocean/benchmark/test_map_metrics.py -v

# Time-to-collision tests
pytest pufferlib/ocean/benchmark/test_ttc.py -v

# Road edge detection tests
pytest pufferlib/ocean/benchmark/test_road_edges.py -v

# Geometry utility tests
pytest pufferlib/ocean/benchmark/test_geometry.py -v
```

### Environment Variable for Tests

Tests that require scenario data check for `DRIVE_BINARIES_DATA_ROOT` and skip gracefully if it is not set. Tests that require model weights (PPO, SMART) check for the weight files and skip if not found. Set `PPO_WEIGHTS_PATH` to enable PPO tests:

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/waymo/binaries
export PPO_WEIGHTS_PATH=/path/to/ppo_model.pt
pytest tests/test_eval_planners.py -v
```

---

## Architecture Overview

```
pufferlib/
  config/
    evaluation.ini              # Default configuration
  planning/
    base.py                     # BasePlanner ABC and PlanResult dataclass
    registry.py                 # Config loading, planner creation factories
    __init__.py                 # Public exports
    pdm.py                      # PDM planner
    policy.py                   # PPO planner
    idm.py                      # IDM planner
    smart.py                    # SMART planner
    hybrid.py                   # Hybrid PPO+PDM planner
    expert.py                   # Expert replay planner
    constant_velocity.py        # Constant velocity baseline
  evaluation/
    config.py                   # EvaluatorConfig, ActionConfig dataclasses
    evaluator.py                # Evaluator class (runs maps, manages planners)
    metrics.py                  # MapMetrics, MetricsWriter (CSV/JSON output)
    collision_classifier.py     # At-fault collision classification
    uncertainty.py              # Ensemble uncertainty estimation
  ocean/
    benchmark/
      eval.py                   # Main entry point (CLI)
tests/
  test_eval_planners.py         # Integration tests for all planner combinations
```

The data flow is:

1. `eval.py` loads config via `registry.load_eval_config()` (INI + CLI overrides).
2. It creates planner factory functions (`partial(create_ego_planner, config)` and `partial(create_traffic_controller, config)`).
3. The `Evaluator` iterates over maps, creating a Drive environment and planners for each.
4. For each timestep, the ego planner and traffic controller produce actions, which are fed to the environment.
5. Metrics are collected per-map and aggregated into the summary.
6. Results are written to the output directory as CSV, JSON, and optional visualizations.

Evaluation is deterministic: random seeds for Python, NumPy, and PyTorch are fixed at 42 at startup.
