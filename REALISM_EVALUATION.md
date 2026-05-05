# WOSAC Realism Evaluation

This document describes the realism evaluation system implemented in
`pufferlib/ocean/benchmark/eval_realism.py`. The evaluation measures how well a
driving planner reproduces the distributional properties of real human driving
behavior, using the **Waymo Open Sim Agents Challenge (WOSAC)** metric suite.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Design Decision: All Agents Use the Same Planner](#key-design-decision-all-agents-use-the-same-planner)
3. [Environment Variables](#environment-variables)
4. [Configuration](#configuration)
   - [Config File (realism.ini)](#config-file-realismini)
   - [CLI Overrides](#cli-overrides)
5. [How It Works](#how-it-works)
   - [Closed-Loop Evaluation](#closed-loop-evaluation)
   - [Open-Loop Evaluation](#open-loop-evaluation)
   - [Ground-Truth Evaluation](#ground-truth-evaluation)
6. [WOSAC Metrics](#wosac-metrics)
   - [Metric Table](#metric-table)
   - [Meta-Score Calculation](#meta-score-calculation)
   - [Metric Buckets](#metric-buckets)
   - [Additional Metrics (Not in Meta-Score)](#additional-metrics-not-in-meta-score)
7. [Available Planners](#available-planners)
   - [IDM (Intelligent Driver Model)](#idm-intelligent-driver-model)
   - [SMART](#smart)
   - [PPO](#ppo)
   - [PPO NoGoal](#ppo-nogoal)
   - [PDM (Planning-based Driving Model)](#pdm-planning-based-driving-model)
   - [Constant Velocity](#constant-velocity)
   - [Random](#random)
8. [Output Files](#output-files)
9. [Execution Examples](#execution-examples)
10. [Visualization](#visualization)
11. [Agent Removal Detection](#agent-removal-detection)
12. [Internals and Data Flow](#internals-and-data-flow)

---

## Overview

The realism evaluation answers the question: *"Does this planner make agents
drive like real humans?"*

It does this by replaying Waymo driving scenarios inside the PufferDrive
simulator. For each scenario, the planner controls every agent for multiple
stochastic rollouts. The resulting trajectory distributions are compared against
recorded ground-truth human trajectories using nine WOSAC metrics that cover
kinematics, interactions with other agents, and map compliance.

The evaluation runs at **10 Hz** (matching the Waymo dataset). A default
episode is **91 steps = 9.1 seconds** of driving. The first 10 steps replay
ground-truth data to initialize the scene, and the remaining **81 steps
(8.1 seconds)** are simulated by the planner.

---

## Key Design Decision: All Agents Use the Same Planner

Unlike many autonomous driving benchmarks that distinguish between an "ego"
vehicle and background "traffic" agents, this evaluation controls **ALL agents
in every scene with the same planner**. There is no ego/traffic distinction.
This means the planner must produce realistic behavior not just for one vehicle,
but for every participant in the scene simultaneously.

---

## Environment Variables

### DRIVE_BINARIES_DATA_ROOT (required)

Points to the root directory containing the Waymo binary scenario files. The
evaluation will refuse to start if this variable is not set.

```bash
export DRIVE_BINARIES_DATA_ROOT=/path/to/waymo/binaries
```

The directory structure under this root is expected to be:

```
$DRIVE_BINARIES_DATA_ROOT/
  training/
    map_000000.bin
    map_000001.bin
    ...
  validation/
    ...
```

### PUFFER_EXP_ROOT (optional)

Controls where experiment output directories are created. Defaults to
`experiments/` in the current working directory.

```bash
export PUFFER_EXP_ROOT=/my/experiments
```

---

## Configuration

### Config File (realism.ini)

The default configuration lives at `pufferlib/config/realism.ini`. It is
organized into INI sections:

```ini
[realism]
split = training            ; Dataset split: training or validation
num_rollouts = 32           ; Number of stochastic rollouts per scenario
init_steps = 10             ; Steps of ground-truth replay before simulation
episode_length = 91         ; Total episode length (init_steps + sim_steps)
device = cuda               ; Device for evaluation (cuda or cpu)
viz = False                 ; Enable per-step visualization
viz_maps = 5                ; Number of maps to visualize
eval_mode = open_loop       ; Evaluation mode: closed_loop, open_loop, ground_truth

[planner]
type = idm                  ; Which planner to use

[planner.idm]
target_velocity = 15.0      ; Target speed in m/s

[planner.smart]
weights_path = /path/to/smart_epoch_030.pt
device = cuda
temperature = 1.0           ; Sampling temperature (higher = more diverse)
greedy = False              ; If True, use argmax decoding
repredict_interval = 5      ; Re-run inference every N steps (closed-loop)

[planner.ppo]
weights_path =              ; Path to PPO checkpoint
device = cuda
stochastic = False          ; If True, sample from policy; if False, use argmax
temperature = 1.0           ; Temperature scaling for logits

[planner.ppo_nogoal]
weights_path =              ; Path to PPO NoGoal checkpoint
device = cuda
stochastic = False
temperature = 1.0

[planner.pdm]
horizon = 40                ; Planning horizon in steps
proposal_other = constant_velocity  ; How to predict other agents
velocity_fractions = (0.2, 0.4, 0.6, 0.8, 1.0)
lateral_offsets = (-1.0, 0.0, 1.0)

[planner.constant_velocity]
; No parameters. Applies zero actions (agents coast at current velocity).

[planner.random]
; No parameters. Generates kinematic random walk trajectories.
```

### CLI Overrides

Every config key can be overridden on the command line using **dot-notation**
with dashes instead of underscores. The format is `--section.key value`:

```bash
# Override planner type
--planner.type smart

# Override a planner-specific parameter
--planner.smart.temperature 0.5

# Override realism section parameters
--realism.num-rollouts 64
--realism.episode-length 91
--realism.viz True
```

There are also three special CLI-only arguments:

| Argument       | Description                                              |
|----------------|----------------------------------------------------------|
| `--map-ids`    | Which maps to evaluate (see examples below)              |
| `--output-dir` | Override the output directory                            |
| `--config`     | Path to a custom `.ini` file that overrides the default  |

The `--map-ids` argument accepts three formats:

- **`all`** (or omit entirely) -- evaluate every `.bin` file in the split directory
- **Range**: `0-19` -- maps 0 through 19 inclusive
- **Comma-separated**: `0,5,10,42` -- specific map IDs

---

## How It Works

### Closed-Loop Evaluation

This is the standard WOSAC evaluation mode (`eval_mode = closed_loop`).
The planner interacts with the simulator step-by-step.

For each batch of maps:

1. **Create the Drive environment** in WOSAC mode (`control_mode="control_wosac"`,
   `init_mode="create_all_valid"`, `use_all_maps=True`). All requested maps are
   loaded simultaneously.

2. **Collect ground truth** -- the environment provides GT trajectories, agent
   state (length, width), and road edge polylines.

3. **For each rollout** (default 32):
   - Reset the environment. The first `init_steps` (default 10) time steps
     replay ground-truth positions to warm up the scene.
   - Create/reset the planner for all agents.
   - For each of the remaining `sim_steps` (default 81) steps:
     - Record the current global agent state (x, y, z, heading, id).
     - Feed observations to the planner and receive actions.
     - Step the environment with those actions.
   - After the rollout, detect agents that were removed by the simulator
     (indicated by a position jump > 5000 units) and mark their subsequent
     timesteps as invalid.

4. **Compute WOSAC metrics** via the `WOSACEvaluator` class.

### Open-Loop Evaluation

Open-loop mode (`eval_mode = open_loop`) is used exclusively with the SMART
planner. Instead of stepping the simulator, the model produces all future
positions in a single forward pass:

1. Load scenarios via the `WaymoBinaryDataset` pipeline (same data
   preprocessing as SMART training).
2. Match environment agents to dataset agents by comparing positions at step 10
   using a KD-tree (tolerance: 0.1m).
3. Run batched SMART inference (chunked in groups of 50 scenarios) to predict
   80 future timesteps.
4. Map predictions back to environment agent coordinates (un-centering from SDC
   frame to world frame).
5. Unmatched agents hold their last known position.

In this mode, `init_steps` is forced to 0 so the GT covers all 91 timesteps
(the model uses steps 0-10 as history internally).

### Ground-Truth Evaluation

A sanity-check mode (`eval_mode = ground_truth`) that uses the ground-truth
trajectories as the simulated output. The meta-score should be approximately
1.0. This is useful for verifying that the metric pipeline is working correctly.

---

## WOSAC Metrics

The WOSAC framework evaluates realism using nine metrics that compare the
distributions of simulated trajectories against ground-truth human driving data.
Each metric produces a **likelihood score in the range [0, 1]**, where 1 means
the simulated distribution perfectly matches the ground truth.

### Metric Table

| Metric                      | Weight | Category    | Description                                                                 |
|-----------------------------|--------|-------------|-----------------------------------------------------------------------------|
| `linear_speed`              | 0.05   | Kinematic   | Distribution of agent speeds (m/s), computed via central differences        |
| `linear_acceleration`       | 0.05   | Kinematic   | Longitudinal acceleration (m/s^2), computed via central differences         |
| `angular_speed`             | 0.05   | Kinematic   | Yaw rate (rad/s), computed via central differences                          |
| `angular_acceleration`      | 0.05   | Kinematic   | Angular acceleration (rad/s^2), computed via central differences            |
| `distance_to_nearest_object`| 0.10   | Interaction | Signed distance to nearest agent bounding box                              |
| `time_to_collision`         | 0.10   | Interaction | Estimated time-to-collision (TTC) with nearest agent                        |
| `collision_indication`      | 0.25   | Interaction | Binary: whether a collision occurred                                        |
| `distance_to_road_edge`     | 0.10   | Map         | Signed distance to nearest road edge polyline                               |
| `offroad_indication`        | 0.25   | Map         | Binary: whether the agent went off-road                                     |

### Meta-Score Calculation

The meta-score is a weighted average of the nine likelihood scores:

```
meta_score = sum(weight_i * likelihood_i) / sum(weight_i)
```

Where `weight_i` and `likelihood_i` correspond to the rows in the table above.
Since the weights sum to 1.0, this simplifies to:

```
meta_score = sum(weight_i * likelihood_i)
```

**Crucially, `collision_indication` (0.25) and `offroad_indication` (0.25)
together account for 50% of the meta-score.** A planner that avoids collisions
and stays on the road is already halfway to a good score.

### Metric Buckets

The WOSAC challenge groups the nine metrics into three buckets for reporting:

| Bucket        | Metrics                                                                            |
|---------------|------------------------------------------------------------------------------------|
| **Kinematic** | `linear_speed`, `linear_acceleration`, `angular_speed`, `angular_acceleration`     |
| **Interactive**| `distance_to_nearest_object`, `collision_indication`, `time_to_collision`          |
| **Map-based** | `distance_to_road_edge`, `offroad_indication`                                      |

Each bucket score is computed as the weighted average of its constituent metrics
(using the WOSAC weights). The evaluation logs both the Waymo-weighted bucket
score and an unweighted (equal-weight) bucket score.

### Statistical Methods

- **Histogram metrics** (speed, acceleration, distances, TTC): Simulated and GT
  values are binned into histograms. Laplace (additive) smoothing is applied to
  avoid zero-probability bins. The likelihood is computed from the histogram
  overlap. The histogram parameters (min, max, number of bins, pseudocount) are
  loaded from `pufferlib/ocean/benchmark/wosac.ini`.

- **Bernoulli metrics** (`collision_indication`, `offroad_indication`): These
  are binary events. The likelihood is computed using a Bernoulli model
  comparing simulated and GT collision/offroad rates.

### Additional Metrics (Not in Meta-Score)

| Metric   | Description                                                        |
|----------|--------------------------------------------------------------------|
| `ADE`    | Average Displacement Error -- mean L2 distance between simulated and GT positions across all agents and timesteps |
| `minADE` | Minimum ADE across all rollouts for each agent (best-of-N)        |

These are reported for diagnostic purposes but do not contribute to the
meta-score.

---

## Available Planners

### IDM (Intelligent Driver Model)

A rule-based car-following model. The IDM planner uses the C++ environment's
built-in IDM logic. It actually returns neutral (zero) actions because the
IDM behavior is handled internally by the environment when an `IDMPlanner`
is registered.

**Config:**
```ini
[planner.idm]
target_velocity = 15.0   ; Target speed in m/s
```

### SMART

A learned autoregressive motion prediction model. In closed-loop mode, it
re-predicts trajectories every `repredict_interval` steps. In open-loop mode,
it produces a single-shot 80-step prediction. Supports batched inference across
all maps simultaneously via `BatchSMARTController`.

**Config:**
```ini
[planner.smart]
weights_path = /path/to/smart_epoch_030.pt
device = cuda
temperature = 1.0         ; Sampling temperature
greedy = False            ; Argmax decoding if True
repredict_interval = 5    ; Steps between re-predictions (closed-loop only)
```

### PPO

A learned reinforcement-learning policy with an LSTM backbone. Uses a discrete
action space (7 acceleration bins x 13 steering bins = 91 discrete actions)
that is converted to continuous (accel, steer) pairs for the simulator.

**Config:**
```ini
[planner.ppo]
weights_path = /path/to/ppo_checkpoint.pt
device = cuda
stochastic = False        ; Sample from policy (True) or argmax (False)
temperature = 1.0         ; Temperature scaling for logits
```

**Action mapping:**
- Acceleration: [-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0] m/s^2 (normalized by max)
- Steering: 13 values linearly spaced from -1.0 to 1.0

### PPO NoGoal

Identical to PPO but uses the `DriveNoGoal` policy architecture, which ignores
goal features in the observation. Useful for evaluating policies trained without
goal conditioning.

**Config:**
```ini
[planner.ppo_nogoal]
weights_path = /path/to/ppo_nogoal_checkpoint.pt
device = cuda
stochastic = False
temperature = 1.0
```

### PDM (Planning-based Driving Model)

A sampling-based planner that generates trajectory proposals by varying velocity
fractions and lateral offsets along a reference path, then scores them.

**Important:** PDM is a single-agent planner. Only agent 0 is actively
controlled; all other agents receive neutral (zero) actions. A warning is logged
when PDM is selected.

**Config:**
```ini
[planner.pdm]
horizon = 40                                    ; Planning horizon in steps
proposal_other = constant_velocity              ; How to predict other agents
velocity_fractions = (0.2, 0.4, 0.6, 0.8, 1.0) ; Speed profile candidates
lateral_offsets = (-1.0, 0.0, 1.0)              ; Lateral offset candidates (meters)
```

### Constant Velocity

A trivial baseline that applies zero actions at every step. Agents coast at
whatever velocity they had at the end of the ground-truth initialization
period. No configuration parameters.

### Random

A kinematic random baseline (matching WOSAC 2023). At each timestep, samples
`(dx, dy, d_heading)` from `N(mu=1.0, sigma=0.1)` and propagates positions
in each agent's local coordinate frame. This planner does **not** use the
simulator at all -- trajectories are generated purely from the initial GT state.

---

## Output Files

Each evaluation run creates a timestamped directory:

```
experiments/<YYYYMMDD_HHMMSS>_wosac_<planner_type>/
  wosac_summary.json    -- Aggregate scores (meta-score, ADE, per-metric, per-bucket)
  wosac_results.csv     -- Per-scenario breakdown of all metrics
  config.json           -- Full configuration used for this run
  realism.log           -- Complete log output
  viz/                  -- (if viz=True) Per-scenario trajectory plots
    scenario_0.png
    scenario_1.png
    ...
  map_000/              -- (if viz=True) Per-step visualization frames and GIFs
    rollout_000/
      step_000.png
      step_001.png
      ...
    rollout_000.gif
    ...
```

### wosac_summary.json

```json
{
  "planner_type": "smart",
  "realism_meta_score": 0.6234,
  "ade": 3.45,
  "min_ade": 2.10,
  "num_maps": 20,
  "total_agents": 487,
  "num_rollouts": 32,
  "likelihood_linear_speed": 0.82,
  "likelihood_linear_acceleration": 0.79,
  "likelihood_angular_speed": 0.85,
  "likelihood_angular_acceleration": 0.81,
  "likelihood_distance_to_nearest_object": 0.54,
  "likelihood_time_to_collision": 0.61,
  "likelihood_collision_indication": 0.42,
  "likelihood_distance_to_road_edge": 0.70,
  "likelihood_offroad_indication": 0.55,
  "bucket_kinematic_waymo": 0.8175,
  "bucket_kinematic_puffer": 0.8175,
  "bucket_interactive_waymo": 0.4778,
  "bucket_interactive_puffer": 0.5233,
  "bucket_map_based_waymo": 0.5857,
  "bucket_map_based_puffer": 0.6250
}
```

### wosac_results.csv

One row per scenario, indexed by scenario ID. Columns include all nine
likelihood metrics, `realism_meta_score`, `ade`, `min_ade`, and `num_agents`.

---

## Execution Examples

All examples assume `DRIVE_BINARIES_DATA_ROOT` is set.

### Basic: Evaluate IDM on all training maps

```bash
python pufferlib/ocean/benchmark/eval_realism.py
```

Uses all defaults from `realism.ini`: IDM planner, training split, 32 rollouts,
all maps.

### Evaluate IDM on a range of maps

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type idm \
    --map-ids 0-19
```

### Evaluate SMART (closed-loop)

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --realism.eval-mode closed_loop \
    --planner.smart.weights-path /path/to/smart_epoch_030.pt \
    --planner.smart.temperature 0.5 \
    --map-ids 0-99
```

### Evaluate SMART (open-loop, single-shot prediction)

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --realism.eval-mode open_loop \
    --planner.smart.weights-path /path/to/smart_epoch_030.pt \
    --planner.smart.greedy False \
    --map-ids 0-99
```

In open-loop mode, `init_steps` is forced to 0 and the model handles the
history/future split internally.

### Evaluate SMART with greedy decoding (deterministic)

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --planner.smart.greedy True \
    --map-ids 0-19
```

Note: in open-loop mode, the first rollout is always greedy regardless of this
setting. Subsequent rollouts use sampling with the configured temperature.

### Evaluate PPO policy

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type ppo \
    --planner.ppo.weights-path /path/to/ppo_checkpoint.pt \
    --planner.ppo.device cuda \
    --planner.ppo.temperature 1.0 \
    --map-ids 0-19
```

### Evaluate PPO NoGoal policy

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type ppo_nogoal \
    --planner.ppo-nogoal.weights-path /path/to/ppo_nogoal_checkpoint.pt \
    --map-ids 0-49
```

### Evaluate PDM planner

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type pdm \
    --planner.pdm.horizon 40 \
    --map-ids 0-9
```

Remember: PDM only controls agent 0. Other agents get neutral actions.

### Evaluate constant velocity baseline

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type constant_velocity \
    --map-ids 0-19
```

### Evaluate random baseline

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type random \
    --map-ids 0-99
```

### Ground-truth sanity check (should yield meta-score near 1.0)

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --realism.eval-mode ground_truth \
    --map-ids 0-19
```

### Evaluate specific maps by ID

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --map-ids 0,5,10,42,99
```

### Custom number of rollouts

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type idm \
    --realism.num-rollouts 64 \
    --map-ids 0-9
```

### Use a custom output directory

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --output-dir /path/to/results/realism \
    --map-ids 0-19
```

The run will be saved under `/path/to/results/realism/<timestamp>_wosac_smart/`.

### Use a custom config file

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --config my_custom_realism.ini \
    --planner.type smart \
    --map-ids 0-19
```

The custom INI file is layered on top of the default `realism.ini`. CLI
arguments take precedence over both.

### Enable visualization

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --realism.viz True \
    --realism.viz-maps 3 \
    --realism.num-rollouts 4 \
    --map-ids 0-4
```

This generates per-step PNG frames and animated GIFs for the first 3 maps
across all rollouts, plus per-scenario trajectory comparison plots.

### Run on CPU

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type idm \
    --realism.device cpu \
    --map-ids 0-9
```

### Full evaluation on validation split

```bash
python pufferlib/ocean/benchmark/eval_realism.py \
    --planner.type smart \
    --realism.split validation \
    --realism.num-rollouts 32 \
    --planner.smart.weights-path /path/to/smart_epoch_030.pt \
    --map-ids all
```

---

## Visualization

When `viz=True`, two types of visualizations are generated:

### Per-Step Frames and GIFs

For the first `viz_maps` maps, every simulation step is rendered as a PNG
showing the full simulator state (road geometry, agent bounding boxes,
positions). For SMART, predicted trajectory overlays are drawn in red. After
each rollout completes, the frames are assembled into an animated GIF at 10 FPS.

Output location: `<output_dir>/map_NNN/rollout_NNN/step_NNN.png` and
`<output_dir>/map_NNN/rollout_NNN.gif`.

### Per-Scenario Trajectory Plots

After metric computation, the evaluator generates bird's-eye trajectory plots
comparing GT (green) against simulated (blue) trajectories. Road edges are
drawn in black. Agent bounding boxes are shown at the initial position. Each
plot is annotated with the scenario ID, agent count, and ADE.

Output location: `<output_dir>/viz/scenario_N.png`.

---

## Agent Removal Detection

In the WOSAC evaluation, collisions and offroad events are **ignored** (`collision_behavior=0`, `offroad_behavior=0`) so that agents continue driving regardless. However, agents that reach their goal receive a new goal (`goal_behavior=1`), and agents may still be removed by the simulator if they leave the scene boundary. Removed agents are teleported to a sentinel position (x < -9000). The evaluation detects this by checking for position jumps exceeding 5000 units between consecutive timesteps. When a removal is detected, all subsequent timesteps for that agent in that rollout are marked as invalid in the `sim_valid` array and excluded from metric computation.

---

## Internals and Data Flow

### Data Shapes

All trajectory data uses the shape convention
`(num_agents, num_rollouts, num_steps)`:

| Array          | Shape                          | Description                          |
|----------------|--------------------------------|--------------------------------------|
| `gt["x"]`      | (N, 1, 91)                    | Ground-truth x positions             |
| `gt["y"]`      | (N, 1, 91)                    | Ground-truth y positions             |
| `gt["heading"]` | (N, 1, 91)                   | Ground-truth headings (radians)      |
| `gt["valid"]`  | (N, 1, 91)                    | Ground-truth validity mask           |
| `gt["id"]`     | (N, 1)                        | Agent IDs                            |
| `sim["x"]`     | (N, 32, 81)                   | Simulated x positions (32 rollouts)  |
| `sim_valid`    | (N, 32, 81)                   | Simulated validity mask              |

Where N is the total number of agents across all maps.

### Batched vs. Sequential Execution

- **SMART and PPO**: Use batched inference across all maps simultaneously.
  SMART uses `BatchSMARTController`; PPO runs a single forward pass for all
  agents. This is significantly faster than per-map sequential execution.

- **IDM, PDM, Constant Velocity**: Use a sequential path where per-map
  planner instances are created and actions are computed per-map, then
  concatenated before stepping the environment.

- **Random**: Does not use the simulator at all. Trajectories are generated
  purely from initial GT state using kinematic propagation.

### Environment Configuration

The Drive environment is created with these key settings for WOSAC evaluation:

| Parameter                | Value              | Purpose                                          |
|--------------------------|--------------------|--------------------------------------------------|
| `control_mode`           | `"control_wosac"`  | WOSAC-specific control mode                      |
| `init_mode`              | `"create_all_valid"`| Initialize all valid agents                     |
| `init_steps`             | 10 (closed-loop)   | Ground-truth replay steps                        |
| `episode_length`         | 91                 | Total episode length (9.1s at 10 Hz)             |
| `action_type`            | `"continuous"`      | Continuous accel/steer actions                   |
| `use_all_maps`           | `True`             | Load all requested maps deterministically        |
| `max_controlled_agents`  | -1                 | Control all agents (no limit)                    |
| `goal_behavior`          | 1                  | Goal-reaching enabled                            |
| `collision_behavior`     | 0                  | No special collision handling                    |
| `offroad_behavior`       | 0                  | No special offroad handling                      |

### Determinism

The evaluation sets a fixed random seed (42) for Python `random`, NumPy, and
PyTorch (both CPU and CUDA) at startup to ensure reproducible results. The
`use_all_maps=True` flag ensures maps are loaded in a deterministic order.
