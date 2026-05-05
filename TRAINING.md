# PufferRL Training

PufferDrive uses PufferLib for reinforcement learning training with Proximal Policy Optimization (PPO). This document covers training, evaluation, configuration, observation space, and sweep functionality.

---

## Basic Training

Train a PufferDrive agent from scratch:

```bash
puffer train puffer_drive
```

Override any config parameter from the command line:

```bash
puffer train puffer_drive --train.learning-rate 0.001
```

Resume training from a saved checkpoint:

```bash
puffer train puffer_drive --load-model-path experiments/puffer_drive_20240101/model.pt
```

Run evaluation with a trained model (no training, inference only):

```bash
puffer eval puffer_drive --load-model-path model.pt
```

---

## Configuration

PufferDrive uses a two-level INI config system. The base defaults are defined in one file, and drive-specific overrides are layered on top.

### Config Files

- `pufferlib/config/default.ini` - Base defaults shared across all PufferLib environments.
- `pufferlib/config/ocean/drive.ini` - PufferDrive-specific overrides for environment, training, policy, and vectorization parameters.

Command-line arguments override both config files. The precedence order is: command-line > drive.ini > default.ini.

### Environment Parameters `[env]`

These control the driving simulation itself.

| Parameter | Default | Description |
|---|---|---|
| `num_agents` | 1024 | Number of agents simulated in parallel across all maps |
| `action_type` | continuous | Action space type (continuous or discrete) |
| `dynamics_model` | classic | Vehicle dynamics model used for stepping |
| `dt` | 0.1 | Simulation timestep in seconds |
| `episode_length` | 91 | Number of steps per episode |
| `num_maps` | 80000 | Number of scenario maps loaded for training |
| `split` | training | Which data split to use (training, validation, testing) |
| `init_steps` | 0 | Number of initial warmup steps before training begins |
| `control_mode` | control_vehicles | What entities the policy controls |
| `init_mode` | create_all_valid | How agents are initialized in scenarios |

### Reward Parameters `[env]`

| Parameter | Default | Description |
|---|---|---|
| `reward_vehicle_collision` | -0.5 | Penalty applied when ego collides with another vehicle |
| `reward_offroad_collision` | -0.5 | Penalty applied when ego drives off the road |
| `reward_goal` | 1.0 | Reward for reaching the goal position |
| `goal_behavior` | 3 | Goal handling mode (controls what happens at goal) |
| `goal_radius` | 2.0 | Distance threshold in meters to consider goal reached |
| `collision_behavior` | 2 | What happens on vehicle collision (e.g., stop, continue) |
| `offroad_behavior` | 2 | What happens on offroad collision |

### PPO Training Parameters `[train]`

| Parameter | Default | Description |
|---|---|---|
| `total_timesteps` | 2000000000 | Total environment steps for the entire training run (2 billion) |
| `learning_rate` | 0.003 | PPO learning rate |
| `batch_size` | 524288 | Number of transitions per PPO update |
| `minibatch_size` | 32768 | Minibatch size for gradient updates within each PPO epoch |
| `bptt_horizon` | 32 | Backpropagation-through-time horizon for LSTM unrolling |
| `gamma` | 0.98 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_coef` | 0.2 | PPO clipping coefficient |
| `vf_coef` | 2.0 | Value function loss coefficient |
| `ent_coef` | 0.005 | Entropy bonus coefficient (encourages exploration) |
| `optimizer` | muon | Optimizer choice (muon is a momentum-based optimizer) |
| `checkpoint_interval` | 1000 | Save a checkpoint every N PPO updates |

### Vectorization Parameters `[vec]`

| Parameter | Default | Description |
|---|---|---|
| `num_workers` | 16 | Number of parallel environment worker processes |
| `num_envs` | 16 | Number of environments per worker |

Total parallel environments = `num_workers` x `num_envs` = 256 by default. Each environment runs `num_agents` agents, so the effective parallelism is 256 x 1024 = 262,144 agents.

### Policy Parameters `[policy]`

| Parameter | Default | Description |
|---|---|---|
| `input_size` | 64 | Dimensionality of the input embedding layer |
| `hidden_size` | 256 | Hidden size of the LSTM and MLP layers |

---

## Observation Space

Each agent receives a flat observation vector of **1120 floats**, structured as follows:

### Ego Features (7 floats)

Information about the agent's own state:

- Position relative to goal (x, y offsets)
- Speed (scalar)
- Heading angle
- Current steering angle
- Current acceleration

### Partner Features (217 floats)

Information about other nearby agents (7 features per agent, up to 31 agents):

- Relative position (x, y)
- Relative speed
- Relative heading
- Size information
- Additional state features

Agents beyond the 31-agent limit are not observed. If fewer than 31 partners exist, the remaining slots are zero-padded.

### Road Features (896 floats)

Information about nearby road segments (7 features per segment, up to 128 segments):

- Segment position relative to ego (x, y)
- Segment orientation
- Lane type / boundary information
- Additional road geometry features

Road segments beyond 128 are not included. If fewer than 128 segments are nearby, the remaining slots are zero-padded.

---

## Hyperparameter Sweeps

PufferLib supports built-in sweep functionality for hyperparameter search:

```bash
puffer sweep puffer_drive
```

This launches multiple training runs with varied hyperparameters. Sweep configurations can be customized in the config files or via command-line overrides.

To sweep over specific parameters, you can define sweep ranges in the config or pass them explicitly. Refer to PufferLib's sweep documentation for details on defining custom sweep spaces.

---

## Evaluation Modes

PufferDrive supports several evaluation modes beyond standard policy rollouts.

### Standard Evaluation

Run the trained policy in the environment and collect metrics (collision rate, goal completion, offroad rate):

```bash
puffer eval puffer_drive --load-model-path model.pt
```

### WOSAC Realism Evaluation

Evaluate the policy using the Waymo Open Sim Agents Challenge (WOSAC) realism metrics. This measures how realistic the generated trajectories are compared to human driving:

```bash
puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path model.pt
```

WOSAC evaluation computes metrics such as kinematic feasibility, map compliance, collision avoidance, and trajectory realism against logged human trajectories.

### Human Replay Evaluation

Compare the policy's behavior against recorded human driving trajectories. This mode replays human actions and measures divergence:

```bash
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path model.pt
```

This is useful for measuring the gap between learned and expert driving behavior on the same scenarios.

---

## Tips and Common Workflows

**Start with a small run to verify setup:**

```bash
puffer train puffer_drive --train.total-timesteps 1000000 --env.num-maps 100
```

This trains for 1M steps on 100 maps, which completes quickly and verifies that data loading, training, and checkpointing all work.

**Monitor training with WandB:**

PufferLib integrates with Weights & Biases for experiment tracking. Training metrics (reward, loss, collision rate) are logged automatically if WandB is configured.

**Adjust parallelism for your hardware:**

On a machine with fewer CPU cores, reduce `num_workers`. On machines with less GPU memory, reduce `batch_size` and `minibatch_size` proportionally:

```bash
puffer train puffer_drive --vec.num-workers 4 --vec.num-envs 4 --train.batch-size 131072 --train.minibatch-size 8192
```

**Train on a specific data subset:**

```bash
puffer train puffer_drive --env.split validation --env.num-maps 1000
```
