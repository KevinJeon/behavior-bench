# nuPlan Integration

PufferDrive integrates with Motional's nuPlan devkit to evaluate trained driving policies using nuPlan's closed-loop simulation and metrics suite. This allows benchmarking PufferDrive agents on nuPlan scenarios alongside other planning approaches.

---

## How It Works

At each simulation step, the integration performs the following:

1. **Observation Conversion** - The `ObservationBuilder` converts nuPlan's native data representation (ego state, detected agents, map information) into PufferDrive's flat 1120-float observation vector.

2. **Policy Inference** - The PPO policy (with LSTM hidden state) processes the observation and produces a continuous action (steering, acceleration).

3. **Bicycle Dynamics** - The `BicycleDynamics` module applies the action to the current ego state using a kinematic bicycle model. For the remaining trajectory points beyond the first action, constant-velocity extrapolation is used.

4. **Trajectory Output** - The result is an 80-point `InterpolatedTrajectory` returned to nuPlan's simulation engine, which advances the scenario and computes metrics.

---

## Integration Files

All integration code lives in the `pufferlib/nuplan_integration/` directory:

| File | Description |
|---|---|
| `planner.py` | Implements nuPlan's `AbstractPlanner` interface for PufferDrive PPO policies |
| `smart_planner.py` | Implements nuPlan's `AbstractPlanner` interface for SMART prediction-based planning |
| `observation_builder.py` | Converts nuPlan observations to PufferDrive's 1120-float observation format |
| `bicycle_dynamics.py` | Kinematic bicycle model for applying actions to ego state |
| `trajectory_filler.py` | Builds the 80-point `InterpolatedTrajectory` that nuPlan expects |
| `nuplan_to_binary.py` | Utility to convert nuPlan scenario data into PufferDrive's binary format |
| `pdm_planner.py` | PDM planner for nuPlan: proposal-based planning with IDM speed profiles |
| `pdm_ppo_planner.py` | Hybrid PDM+PPO planner with uncertainty-based switching |

---

## Prerequisites & Installation

The nuPlan integration is **optional** and only needed if you want to evaluate PufferDrive planners on nuPlan scenarios. It is not required for standard training or WOMD-based evaluation.

### 1. Install the nuPlan Devkit

The [nuPlan devkit](https://github.com/motional/nuplan-devkit) provides the simulation engine, scenario database, and metrics framework.

```bash
# Clone the devkit
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

# Install (editable mode recommended)
pip install -e .

# Set the devkit root
export NUPLAN_DEVKIT_ROOT=$(pwd)
```

See the [nuPlan devkit README](https://github.com/motional/nuplan-devkit#readme) for detailed installation instructions and troubleshooting.

### 2. Download nuPlan Data

nuPlan scenarios and maps must be downloaded separately. Follow the [official data download instructions](https://www.nuscenes.org/nuplan#download) to obtain:

- **Scenario database** (`.db` files) - contains logged trajectories and annotations
- **Map data** - vector maps for each city (Las Vegas, Boston, Pittsburgh, Singapore)

After downloading, set the environment variables:

```bash
export NUPLAN_DATA_ROOT=/path/to/nuplan/dataset
export NUPLAN_MAPS_ROOT=/path/to/nuplan/dataset/maps
```

### 3. (Optional) Install ScenarioMax

[ScenarioMax](https://github.com/valeoai/ScenarioMax) can convert nuPlan scenarios to PufferDrive's binary format, enabling training on nuPlan data:

```bash
git clone https://github.com/valeoai/ScenarioMax.git
cd ScenarioMax
pip install -e .
```

See [BENCHMARK.md](BENCHMARK.md#d-creating-binaries-from-external-data-sources) for details on data conversion.

### 4. Register Planner Configs

Copy PufferDrive's planner YAML configs into the nuPlan devkit so Hydra can discover them:

```bash
cp pufferlib/nuplan_integration/config/simulation/planner/*.yaml \
   $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/config/simulation/planner/
```

This registers all PufferDrive planners: `pufferdrive_planner`, `smart_planner`, `pdm_nuplan_planner`, and `pdm_ppo_planner`.

---

## Usage

### Running the PufferDrive Planner

Set the path to your trained PufferDrive checkpoint and invoke nuPlan's simulation runner:

```bash
export PUFFERDRIVE_WEIGHTS_PATH=/path/to/checkpoint.pt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=pufferdrive_planner \
    scenario_filter=val14_split \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

This runs closed-loop simulation with non-reactive agents (other vehicles replay their logged trajectories) on nuPlan's val14 split using the PufferDrive planner.

### Running with Reactive Agents

To test against reactive agents (other vehicles respond to ego):

```bash
export PUFFERDRIVE_WEIGHTS_PATH=/path/to/checkpoint.pt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_reactive_agents \
    planner=pufferdrive_planner \
    scenario_filter=val14_split \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

### Running the SMART Planner on nuPlan

The SMART prediction model can also be used as a nuPlan planner:

```bash
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=smart_planner \
    planner.smart_planner.weights_path=/path/to/smart_weights.pt \
    scenario_filter=val14_split \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

### Running the PDM Planner on nuPlan

The PDM (Predictive Driver Model) planner generates trajectory proposals by combining lateral path offsets with IDM speed profiles at different target velocities, then selects the best-scoring proposal:

```bash
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=pdm_nuplan_planner \
    scenario_filter=val14 \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.experiments]"
```

### Running the PDM+PPO Hybrid Planner

The hybrid planner combines PPO with PDM fallback. It always runs the PPO policy (to keep the LSTM state updated), computes epistemic uncertainty from an ensemble of PPO checkpoints, and switches to PDM when uncertainty exceeds a threshold:

```bash
export PUFFERDRIVE_WEIGHTS_PATH=/path/to/checkpoint.pt
export PUFFERDRIVE_ENSEMBLE_WEIGHTS="/path/to/w1.pt,/path/to/w2.pt,/path/to/w3.pt"

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=pdm_ppo_planner \
    scenario_filter=val14 \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.config.simulation, pkg://nuplan.planning.script.experiments]"
```

Without ensemble weights, the hybrid planner will always use PPO (uncertainty is 0). To enable switching, provide 2+ ensemble checkpoint paths via `PUFFERDRIVE_ENSEMBLE_WEIGHTS`.

### Running on a Specific Scenario

To run simulation on a single scenario for debugging:

```bash
export PUFFERDRIVE_WEIGHTS_PATH=/path/to/checkpoint.pt

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=pufferdrive_planner \
    scenario_filter=one_of_each_scenario_type \
    scenario_filter.limit_total_scenarios=1 \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]"
```

### interPlan Evaluation

[interPlan](https://github.com/mh0797/interPlan) is a closed-loop planning benchmark built on top of nuPlan that focuses on challenging interactive scenarios. It modifies original nuPlan scenarios by increasing traffic density and complexity, and adjusting navigation targets to require lane changes. The benchmark contains 335 scenarios across 8 categories (jaywalkers, overtaking, nudging, lane changes, accident sites, construction zones, etc.).

interPlan uses its **own simulation runner** (not nuPlan's default) and its own scenario filters.

**Install interPlan:**

```bash
git clone https://github.com/mh0797/interPlan.git
cd interPlan
pip install -e .
```

**Run PufferDrive on interPlan:**

```bash
export PUFFERDRIVE_WEIGHTS_PATH=/path/to/checkpoint.pt

# Run on the official 80-scenario split (10 per category)
python interplan/planning/script/run_simulation.py \
    +simulation=default_interplan_benchmark \
    planner=pufferdrive_planner \
    scenario_filter=interplan10 \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://interplan.planning.script.config.common, pkg://interplan.planning.script.config.simulation, pkg://interplan.planning.script.experiments]"

# Run on all 335 scenarios
python interplan/planning/script/run_simulation.py \
    +simulation=default_interplan_benchmark \
    planner=pufferdrive_planner \
    scenario_filter=benchmark_scenarios \
    scenario_builder=nuplan \
    hydra.searchpath="[pkg://pufferlib.nuplan_integration.config, pkg://interplan.planning.script.config.common, pkg://interplan.planning.script.config.simulation, pkg://interplan.planning.script.experiments]"
```

See the [interPlan repository](https://github.com/mh0797/interPlan) for the full benchmark setup, metric definitions, and baseline results.

### Converting nuPlan Data to Binary Format

If you want to convert nuPlan scenarios into PufferDrive's binary format (for training or analysis):

```bash
python pufferlib/nuplan_integration/nuplan_to_binary.py \
    --scenario-builder nuplan \
    --scenario-filter val14_split \
    --output-dir /path/to/nuplan_binaries
```

---

## Configuration

The planner is configured via Hydra YAML files in `pufferlib/nuplan_integration/config/`.

### PufferDrive Planner Config

| Parameter | Default | Description |
|---|---|---|
| `weights_path` | (from env) | Path to the trained PPO checkpoint (.pt file) |
| `input_size` | 64 | Input embedding size (must match training config) |
| `hidden_size` | 256 | LSTM/MLP hidden size (must match training config) |
| `device` | cuda | Device for inference (cuda or cpu) |
| `stochastic` | false | Whether to sample actions stochastically or use the mode |
| `trajectory_steps` | 80 | Number of trajectory points returned to nuPlan |
| `trajectory_dt` | 0.1 | Time delta between trajectory points in seconds |

The `input_size` and `hidden_size` must exactly match the values used during training, otherwise the checkpoint will fail to load.

---