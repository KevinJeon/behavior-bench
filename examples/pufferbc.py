# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Behavioral Cloning for DriveNoGoal+LSTM.

Pre-trains the student network (no goal features) on expert trajectory data.
Two phases:
  1. Preprocess: Collect observations from Drive env (expert replay) + compute
     GT discrete actions via inverse dynamics. Cache as .pt files.
  2. Train: Load cached data, train DriveNoGoal+LSTM with cross-entropy loss.

Usage:
    # Preprocess (compute obs + GT actions, save to cache)
    DRIVE_BINARIES_DATA_ROOT=/path/to/binaries python examples/pufferbc.py preprocess \
        --output-dir experiments/bc_cache --num-maps 79000

    # Train on cached data (no goal)
    python examples/pufferbc.py train --cache-dir experiments/bc_cache --epochs 30 --wandb

    # Train on cached data (with goal features)
    python examples/pufferbc.py train --goal --cache-dir experiments/bc_cache --epochs 30 --wandb
"""

import os
import sys
import math
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inverse dynamics for 7x13 = 91 discrete action space (classic dynamics)
# ---------------------------------------------------------------------------

ACCEL_VALUES = np.array([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0], dtype=np.float64)
STEER_VALUES = np.array([-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0,
                          0.167, 0.333, 0.5, 0.667, 0.833, 1.0], dtype=np.float64)
NUM_ACCEL = len(ACCEL_VALUES)   # 7
NUM_STEER = len(STEER_VALUES)   # 13
NUM_ACTIONS = NUM_ACCEL * NUM_STEER  # 91

# Pre-computed grids for vectorized simulation
_ACC_GRID = np.repeat(ACCEL_VALUES, NUM_STEER)   # (91,)
_STEER_GRID = np.tile(STEER_VALUES, NUM_ACCEL)    # (91,)
_BETA_GRID = np.tanh(0.5 * np.tan(_STEER_GRID))
_TAN_STEER_GRID = np.tan(_STEER_GRID)
_COS_BETA_GRID = np.cos(_BETA_GRID)


def simulate_all_actions(x, y, heading, signed_speed, vehicle_length, dt=0.1):
    """Simulate all 91 actions for one step from a given state.

    Uses the classic bicycle model matching drive.h:1658-1718.
    Wheelbase = vehicle_length (matching drive.h:1700).

    Returns:
        np.ndarray (91, 4): [new_x, new_y, new_heading, new_speed]
    """
    wheelbase = max(float(vehicle_length), 0.1)

    cx = np.full(NUM_ACTIONS, x, dtype=np.float64)
    cy = np.full(NUM_ACTIONS, y, dtype=np.float64)
    ch = np.full(NUM_ACTIONS, heading, dtype=np.float64)
    cs = np.full(NUM_ACTIONS, signed_speed, dtype=np.float64)

    cs = cs + _ACC_GRID * dt
    cs = np.clip(cs, -10.0, 20.0)
    yaw_rate = (cs * _COS_BETA_GRID * _TAN_STEER_GRID) / wheelbase
    vx = cs * np.cos(ch + _BETA_GRID)
    vy = cs * np.sin(ch + _BETA_GRID)
    cx = cx + vx * dt
    cy = cy + vy * dt
    ch = ch + yaw_rate * dt

    return np.stack([cx, cy, ch, cs], axis=-1).astype(np.float32)


def compute_gt_actions_batch(traj_x, traj_y, traj_heading, traj_vx, traj_vy,
                             traj_valid, vehicle_lengths, dt=0.1):
    """Compute GT discrete actions for ALL agents at ALL timesteps (fully vectorized).

    Args:
        traj_x: (N, T) float arrays for N agents, T timesteps
        traj_y, traj_heading, traj_vx, traj_vy, traj_valid: same shape
        vehicle_lengths: (N,) float

    Returns:
        actions: (N, T-1) int64
        valid: (N, T-1) bool
    """
    N, T = traj_x.shape
    actions = np.zeros((N, T - 1), dtype=np.int64)
    valid_out = np.zeros((N, T - 1), dtype=bool)

    # Valid mask: both current and next step valid
    v = traj_valid.astype(bool)
    step_valid = v[:, :-1] & v[:, 1:]  # (N, T-1)

    if not step_valid.any():
        return actions, valid_out

    # Gather valid (agent, time) pairs
    ai, ti = np.where(step_valid)  # indices into (N, T-1)
    M = len(ai)

    x = traj_x[ai, ti].astype(np.float64)
    y = traj_y[ai, ti].astype(np.float64)
    h = traj_heading[ai, ti].astype(np.float64)
    vx = traj_vx[ai, ti].astype(np.float64)
    vy = traj_vy[ai, ti].astype(np.float64)
    wl = np.maximum(vehicle_lengths[ai].astype(np.float64), 0.1)

    gt_x = traj_x[ai, ti + 1].astype(np.float64)
    gt_y = traj_y[ai, ti + 1].astype(np.float64)
    gt_h = traj_heading[ai, ti + 1].astype(np.float64)

    # Signed speed
    speed_mag = np.sqrt(vx**2 + vy**2)
    v_dot_h = vx * np.cos(h) + vy * np.sin(h)
    signed_speed = np.copysign(speed_mag, v_dot_h)

    # Simulate all 91 actions for all M states: (M, 91)
    acc = _ACC_GRID[None, :]       # (1, 91)
    beta = _BETA_GRID[None, :]     # (1, 91)
    tan_s = _TAN_STEER_GRID[None, :]
    cos_b = _COS_BETA_GRID[None, :]

    new_speed = np.clip(signed_speed[:, None] + acc * dt, -10.0, 20.0)  # (M, 91)
    yaw_rate = (new_speed * cos_b * tan_s) / wl[:, None]
    pred_vx = new_speed * np.cos(h[:, None] + beta)
    pred_vy = new_speed * np.sin(h[:, None] + beta)
    pred_x = x[:, None] + pred_vx * dt
    pred_y = y[:, None] + pred_vy * dt
    pred_h = h[:, None] + yaw_rate * dt

    # Error
    pos_err = (pred_x - gt_x[:, None])**2 + (pred_y - gt_y[:, None])**2
    h_diff = pred_h - gt_h[:, None]
    h_err = np.arctan2(np.sin(h_diff), np.cos(h_diff))**2
    total_err = pos_err + 0.1 * h_err

    best = np.argmin(total_err, axis=1)  # (M,)
    actions[ai, ti] = best
    valid_out[ai, ti] = True

    return actions, valid_out


# ---------------------------------------------------------------------------
# Preprocessing: collect obs from Drive env + compute GT actions
# ---------------------------------------------------------------------------

def _worker_process_map(task):
    """Worker function for multiprocessing pool."""
    map_id, bin_path, data_root, split, out_path = task
    try:
        data = _process_single_map(map_id, bin_path, data_root, split)
        if data is not None:
            torch.save(data, out_path)
            return "ok"
        return "empty"
    except Exception as e:
        return f"error: {e}"


def preprocess(args):
    """Preprocess expert data: collect observations and compute GT actions."""
    import multiprocessing as mp

    data_root = args.data_root or os.environ.get("DRIVE_BINARIES_DATA_ROOT")
    if not data_root:
        raise ValueError("Set --data-root or DRIVE_BINARIES_DATA_ROOT")

    split = args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_maps = args.num_maps
    num_workers = args.num_workers

    split_dir = Path(data_root) / split
    available = len(list(split_dir.glob("*.bin")))
    num_maps = min(num_maps, available)

    # Build task list (skip already cached)
    tasks = []
    skipped_existing = 0
    for map_id in range(num_maps):
        out_path = output_dir / f"map_{map_id:06d}.pt"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        bin_path = split_dir / f"map_{map_id:06d}.bin"
        if not bin_path.exists():
            continue
        tasks.append((map_id, str(bin_path), data_root, split, str(out_path)))

    print(f"Processing {len(tasks)} maps with {num_workers} workers "
          f"({skipped_existing} already cached)")

    if not tasks:
        print("Nothing to do.")
        return

    t_start = time.time()
    processed = 0
    errors = 0

    with mp.Pool(num_workers) as pool:
        for result in pool.imap_unordered(_worker_process_map, tasks, chunksize=16):
            processed += 1
            if result.startswith("error"):
                errors += 1
            if processed % 500 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                remaining = (len(tasks) - processed) / max(rate, 1e-6)
                print(f"  {processed}/{len(tasks)} ({rate:.1f} maps/s, "
                      f"~{remaining/60:.0f}min remaining, {errors} errors)")

    elapsed = time.time() - t_start
    print(f"Done: {processed} processed, {errors} errors in {elapsed:.0f}s")


def _process_single_map(map_id, bin_path, data_root, split):
    """Process a single map using C binding directly (faster than Drive wrapper)."""
    from pufferlib.prediction.binary_reader import read_binary_scenario
    from pufferlib.ocean.drive import binding

    # Read binary for vehicle lengths
    scenario = read_binary_scenario(bin_path)
    objects = scenario["objects"]
    if len(objects) == 0:
        return None

    # Use C binding directly — avoids os.listdir() overhead in Drive.__init__
    agent_offsets, map_ids, num_envs = binding.shared(
        num_agents=512, num_maps=1, split=split, data_root=data_root,
        init_mode=0, control_mode=0, init_steps=10,
        max_controlled_agents=-1, goal_behavior=3, goal_target_distance=30.0,
        use_all_maps=False, map_id=map_id,
    )
    num_agents = agent_offsets[-1]
    if num_agents == 0:
        return None

    obs_dim = 1120  # EGO(7) + PARTNER(31*7) + ROAD(128*7)
    episode_length = 91
    init_steps = 10
    T = episode_length - init_steps - 1  # 80 usable steps

    # Allocate buffers
    observations = np.zeros((num_agents, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((num_agents,), dtype=np.int32)
    rewards = np.zeros(num_agents, dtype=np.float32)
    terminals = np.zeros(num_agents, dtype=np.int8)
    truncations = np.zeros(num_agents, dtype=np.int8)

    env_id = binding.env_init(
        observations, actions_buf, rewards, terminals, truncations,
        42,  # seed (positional)
        action_type=0, human_agent_idx=0,
        reward_vehicle_collision=0.0, reward_offroad_collision=0.0,
        reward_speed_limit=0.0,
        reward_goal=0.0, reward_goal_post_respawn=0.0,
        goal_radius=2.0, goal_speed=100.0, goal_behavior=3,
        goal_target_distance=30.0,
        collision_behavior=0, offroad_behavior=0,
        dt=0.1, episode_length=episode_length, termination_mode=0,
        max_controlled_agents=-1, idm_others=0,
        map_id=map_ids[0], max_agents=num_agents,
        ini_file="pufferlib/config/ocean/drive.ini",
        init_steps=init_steps, init_mode=0, control_mode=0,
        data_root=data_root, split=split, include_global_state=0,
    )
    c_envs = binding.vectorize(env_id)
    binding.vec_reset(c_envs, 0)

    # Set all agents to expert replay (MOVEMENT_EXPERT=2)
    all_indices = list(range(num_agents))
    binding.vec_set_movement_mode(c_envs, all_indices, 2)

    # Collect observations over T steps
    all_obs = np.zeros((num_agents, T, obs_dim), dtype=np.float32)
    all_obs[:, 0] = observations.copy()

    for t in range(1, T):
        actions_buf[:] = 0
        binding.vec_step(c_envs)
        all_obs[:, t] = observations.copy()

    # Get GT trajectories (aligned with agent ordering)
    gt_steps = episode_length - init_steps
    gt_x = np.zeros((num_agents, gt_steps), dtype=np.float32)
    gt_y = np.zeros((num_agents, gt_steps), dtype=np.float32)
    gt_z = np.zeros((num_agents, gt_steps), dtype=np.float32)
    gt_heading = np.zeros((num_agents, gt_steps), dtype=np.float32)
    gt_valid_arr = np.zeros((num_agents, gt_steps), dtype=np.int32)
    gt_id = np.zeros(num_agents, dtype=np.int32)
    gt_scenario_id = np.zeros(num_agents, dtype=np.int32)
    gt_is_vehicle = np.zeros(num_agents, dtype=np.int32)

    binding.vec_get_global_ground_truth_trajectories(
        c_envs, gt_x, gt_y, gt_z, gt_heading,
        gt_valid_arr, gt_id, gt_scenario_id, gt_is_vehicle,
    )
    binding.vec_close(c_envs)

    # Compute velocities from position differences
    dt = 0.1
    gt_vx = np.zeros_like(gt_x)
    gt_vy = np.zeros_like(gt_y)
    gt_vx[:, :-1] = (gt_x[:, 1:] - gt_x[:, :-1]) / dt
    gt_vy[:, :-1] = (gt_y[:, 1:] - gt_y[:, :-1]) / dt
    gt_vx[:, -1] = gt_vx[:, -2] if gt_steps > 1 else 0
    gt_vy[:, -1] = gt_vy[:, -2] if gt_steps > 1 else 0

    default_length = 4.5

    # Build vehicle lengths array
    vehicle_lengths = np.full(num_agents, default_length, dtype=np.float64)
    for i in range(min(num_agents, len(objects))):
        vehicle_lengths[i] = max(objects[i].get("length", default_length), 0.1)

    # Compute GT actions via vectorized inverse dynamics (all agents × all timesteps)
    gt_steps_act = min(gt_steps, T + 1)  # need T+1 states for T actions
    all_actions_raw, all_valid_raw = compute_gt_actions_batch(
        gt_x[:, :gt_steps_act], gt_y[:, :gt_steps_act],
        gt_heading[:, :gt_steps_act],
        gt_vx[:, :gt_steps_act], gt_vy[:, :gt_steps_act],
        gt_valid_arr[:, :gt_steps_act],
        vehicle_lengths, dt=dt,
    )
    # Mask out non-vehicles
    for i in range(num_agents):
        if not gt_is_vehicle[i]:
            all_valid_raw[i, :] = False

    all_actions = np.zeros((num_agents, T), dtype=np.int64)
    all_valid = np.zeros((num_agents, T), dtype=bool)
    copy_t = min(all_actions_raw.shape[1], T)
    all_actions[:, :copy_t] = all_actions_raw[:, :copy_t]
    all_valid[:, :copy_t] = all_valid_raw[:, :copy_t]

    # Filter to valid agents that actually move (>2m displacement over episode)
    agent_mask = all_valid.any(axis=1)
    for i in range(num_agents):
        if not agent_mask[i]:
            continue
        valid_steps = gt_valid_arr[i].astype(bool)
        if valid_steps.sum() < 2:
            agent_mask[i] = False
            continue
        valid_idx = np.where(valid_steps)[0]
        displacement = np.sqrt(
            (gt_x[i, valid_idx[-1]] - gt_x[i, valid_idx[0]])**2 +
            (gt_y[i, valid_idx[-1]] - gt_y[i, valid_idx[0]])**2
        )
        if displacement < 2.0:
            agent_mask[i] = False

    if not agent_mask.any():
        return None

    return {
        "observations": torch.from_numpy(all_obs[agent_mask].astype(np.float16)),
        "actions": torch.from_numpy(all_actions[agent_mask]),
        "valid": torch.from_numpy(all_valid[agent_mask]),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """Dataset of cached BC data. Each item = one agent trajectory."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.files = sorted(self.cache_dir.glob("map_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No cached data in {cache_dir}")

        # Build index: (file_idx, agent_idx) for each sample
        self.index = []
        for fi, fpath in enumerate(self.files):
            data = torch.load(fpath, weights_only=True)
            num_agents = data["actions"].shape[0]
            for ai in range(num_agents):
                self.index.append((fi, ai))

        print(f"BCDataset: {len(self.files)} maps, {len(self.index)} agent trajectories")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, ai = self.index[idx]
        data = torch.load(self.files[fi], weights_only=True)
        return {
            "observations": data["observations"][ai].float(),  # (T, obs_dim)
            "actions": data["actions"][ai],                      # (T,)
            "valid": data["valid"][ai],                          # (T,)
        }


# ---------------------------------------------------------------------------
# WOSAC evaluation helper
# ---------------------------------------------------------------------------

def setup_wosac_eval(data_root, device, num_maps=20, num_rollouts=32,
                     eval_data_root=None, eval_split="training"):
    """Create eval env and evaluator for WOSAC closed-loop evaluation.

    Uses continuous action env + manual discrete→continuous conversion,
    matching eval_realism.py PPO evaluation setup.
    """
    import tempfile
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    # Create temp ini with action_type=continuous for eval.
    # The C backend reads action_type, dynamics_model, rewards, dt, episode_length,
    # collision_behavior, offroad_behavior from the ini (NOT from Python kwargs).
    eval_ini = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    eval_ini.write(
        "[env]\n"
        "action_type = continuous\n"
        "dynamics_model = classic\n"
        "dt = 0.1\n"
        "episode_length = 91\n"
        "collision_behavior = 0\n"
        "offroad_behavior = 0\n"
        "reward_vehicle_collision = 0.0\n"
        "reward_offroad_collision = 0.0\n"
        "reward_speed_limit = 0.0\n"
        "reward_goal = 0.0\n"
        "reward_goal_post_respawn = 0.0\n"
        "termination_mode = 1\n"
    )
    eval_ini.close()
    print(f"Eval ini file: {eval_ini.name}")

    env = Drive(
        data_root=eval_data_root or data_root,
        split=eval_split,
        episode_length=91,
        action_type="continuous",
        dynamics_model="classic",
        num_maps=num_maps,
        num_agents=num_maps * 50,
        control_mode="control_agents",
        init_mode="create_all_valid",
        init_steps=10,
        goal_behavior=2,
        goal_radius=2.0,
        collision_behavior=0,
        offroad_behavior=0,
        max_controlled_agents=-1,
        use_all_maps=True,
        ini_file=eval_ini.name,
    )

    eval_config = {
        "eval": {
            "wosac_init_steps": 10,
            "wosac_num_rollouts": num_rollouts,
        },
        "train": {
            "device": str(device),
            "use_rnn": False,
        },
    }
    evaluator = WOSACEvaluator(eval_config)
    gt_trajectories = env.get_ground_truth_trajectories()

    print(f"WOSAC eval: {len(np.unique(gt_trajectories['scenario_id']))} scenarios, "
          f"{np.sum(gt_trajectories['id'] >= 0)} agents, {num_rollouts} rollouts")

    return env, evaluator, gt_trajectories


# Discrete→continuous action conversion (matching eval_realism.py)
_ACCEL_VALS = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
_STEER_VALS = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
_NUM_STEER = len(_STEER_VALS)
_MAX_ACCEL = np.max(np.abs(_ACCEL_VALS))


def run_wosac_eval(model, env, evaluator, eval_config, gt_trajectories):
    """Run WOSAC closed-loop evaluation with discrete→continuous conversion."""
    import pufferlib.pytorch

    device = eval_config["train"]["device"]
    num_rollouts = evaluator.num_rollouts
    sim_steps = evaluator.sim_steps
    num_agents = env.num_agents

    sim = {
        "x": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "y": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "z": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "heading": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "id": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.int32),
    }

    has_lstm = hasattr(model, 'cell')
    if has_lstm:
        h_size = model.hidden_size

    model.eval()
    with torch.no_grad():
        for r in range(num_rollouts):
            print(f"\rCollecting rollout {r + 1}/{num_rollouts}...", end="", flush=True)
            obs, _ = env.reset()

            if has_lstm:
                lstm_h = torch.zeros(num_agents, h_size, device=device)
                lstm_c = torch.zeros(num_agents, h_size, device=device)

            for t in range(sim_steps):
                agent_state = env.get_global_agent_state()
                sim["x"][:, r, t] = agent_state["x"][:num_agents]
                sim["y"][:, r, t] = agent_state["y"][:num_agents]
                sim["z"][:, r, t] = agent_state["z"][:num_agents]
                sim["heading"][:, r, t] = agent_state["heading"][:num_agents]
                sim["id"][:, r, t] = agent_state["id"][:num_agents]

                ob_tensor = torch.as_tensor(obs[:num_agents]).float().to(device)

                if has_lstm:
                    state = {"lstm_h": lstm_h, "lstm_c": lstm_c}
                    logits, _ = model.forward_eval(ob_tensor, state)
                    lstm_h = state["lstm_h"]
                    lstm_c = state["lstm_c"]
                else:
                    logits, _ = model.forward_eval(ob_tensor, {})

                action, _, _ = pufferlib.pytorch.sample_logits(logits)

                # Discrete → continuous conversion (matching eval_realism.py)
                flat_idx = action.cpu().numpy().flatten()
                accel_idx = flat_idx // _NUM_STEER
                steer_idx = flat_idx % _NUM_STEER
                accel = _ACCEL_VALS[accel_idx] / _MAX_ACCEL
                steer = _STEER_VALS[steer_idx]
                cont_actions = np.stack([accel, steer], axis=-1).astype(np.float32)

                obs, _, _, _, _ = env.step(cont_actions[:num_agents])
        print()
    model.train()

    agent_state = env.get_global_agent_state()
    road_edge_polylines = env.get_road_edge_polylines()
    results = evaluator.compute_metrics(
        gt_trajectories, sim, agent_state, road_edge_polylines,
        aggregate_results=True)
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Train Drive/DriveNoGoal MLP with behavioral cloning (open-loop, no env)."""
    from pufferlib.ocean.torch import Drive as DrivePolicy, DriveNoGoal
    from pufferlib.ocean.drive.drive import Drive

    device = torch.device(args.device)

    # Create a minimal env just for model construction
    data_root = args.data_root or os.environ.get("DRIVE_BINARIES_DATA_ROOT")
    if not data_root:
        raise ValueError("Set --data-root or DRIVE_BINARIES_DATA_ROOT")
    dummy_env = Drive(
        data_root=data_root,
        split="training",
        episode_length=91,
        action_type="discrete",
        dynamics_model="classic",
        num_maps=100,
        num_agents=32,
    )

    # Build model
    PolicyClass = DrivePolicy if args.goal else DriveNoGoal
    base_policy = PolicyClass(
        dummy_env,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
    )
    if args.lstm:
        import pufferlib.models
        model = pufferlib.models.LSTMWrapper(
            dummy_env, base_policy,
            input_size=args.hidden_size, hidden_size=args.hidden_size,
        ).to(device)
    else:
        model = base_policy.to(device)
    dummy_env.close()

    # Setup WOSAC evaluation
    wosac_eval = None
    if args.eval_interval > 0:
        eval_env, evaluator, gt_traj = setup_wosac_eval(
            data_root, device, num_maps=args.eval_num_maps,
            num_rollouts=args.wosac_num_rollouts,
            eval_data_root=args.eval_data_root,
            eval_split=args.eval_split)
        eval_config = {
            "train": {"device": str(device), "use_rnn": False},
            "eval": {"wosac_init_steps": 10, "wosac_num_rollouts": args.wosac_num_rollouts},
        }
        wosac_eval = (eval_env, evaluator, eval_config, gt_traj)

    # Dataset and dataloader
    dataset = BCDataset(args.cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    # Wandb
    if args.wandb:
        import wandb
        arch = "bc-lstm" if args.lstm else "bc-mlp"
        wandb.init(project="pufferlib", name=f"{arch}-{args.run_name}" if args.run_name else arch,
                   config=vars(args))

    print(f"Training: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  {len(dataset)} samples, {len(dataloader)} batches/epoch")

    best_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Run WOSAC eval with untrained policy (baseline)
    if wosac_eval is not None:
        eval_env, evaluator, eval_config, gt_traj = wosac_eval
        print("Running WOSAC evaluation (untrained baseline)...")
        try:
            results = run_wosac_eval(model, eval_env, evaluator, eval_config, gt_traj)
            meta_score = results.get("realism_meta_score", 0)
            ade = results.get("ade", 0)
            min_ade = results.get("min_ade", 0)
            print(f"  WOSAC baseline: meta_score={meta_score:.4f}  ade={ade:.4f}  "
                  f"min_ade={min_ade:.4f}")
            if args.wandb:
                import wandb
                wandb.log({
                    "epoch": 0,
                    "eval/wosac_realism_meta_score": meta_score,
                    "eval/wosac_ade": ade,
                    "eval/wosac_min_ade": min_ade,
                })
        except Exception as e:
            print(f"  WOSAC baseline evaluation failed: {e}")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        for batch in dataloader:
            obs = batch["observations"].to(device)      # (B, T, obs_dim)
            actions = batch["actions"].to(device)        # (B, T)
            valid = batch["valid"].to(device)            # (B, T)
            B, T, obs_dim = obs.shape

            if args.lstm:
                # LSTM mode: process full trajectories sequentially
                # Reset LSTM where previous step was invalid
                trunc_or_term = torch.zeros(B, T, device=device)
                trunc_or_term[:, 1:] = (~valid[:, :-1].bool()).float()
                trunc_or_term[:, 0] = 1.0  # always reset at trajectory start

                # LSTMWrapper needs state["action"] with shape (B, T, action_dim)
                action_dim = 1  # discrete actions
                state = {
                    "action": torch.zeros(B, T, action_dim, device=device),
                    "lstm_h": None,
                    "lstm_c": None,
                }
                logits, _ = model(obs, state, trunc_or_term)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                # logits: (B*T, action_dim) from LSTMWrapper
                actions_flat = actions.reshape(B * T)
                valid_flat = valid.reshape(B * T).bool()

                if not valid_flat.any():
                    continue

                loss = F.cross_entropy(
                    logits[valid_flat], actions_flat[valid_flat],
                    label_smoothing=args.label_smoothing)

                pred = logits[valid_flat].argmax(dim=-1)
                acc = (pred == actions_flat[valid_flat]).float().mean().item()
            else:
                # MLP mode: flatten and shuffle individual frames
                flat_obs = obs.reshape(B * T, obs_dim)
                flat_actions = actions.reshape(B * T)
                flat_valid = valid.reshape(B * T)

                mask = flat_valid.bool()
                if not mask.any():
                    continue

                valid_obs = flat_obs[mask]
                valid_actions = flat_actions[mask]

                # Shuffle to break temporal adjacency within batch
                perm = torch.randperm(valid_obs.shape[0], device=valid_obs.device)
                valid_obs = valid_obs[perm]
                valid_actions = valid_actions[perm]

                logits, _ = model(valid_obs)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]

                loss = F.cross_entropy(logits, valid_actions,
                                       label_smoothing=args.label_smoothing)

                pred = logits.argmax(dim=-1)
                acc = (pred == valid_actions).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_steps += 1

        if epoch_steps == 0:
            continue

        avg_loss = epoch_loss / epoch_steps
        avg_acc = epoch_acc / epoch_steps
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_epoch

        print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
              f"acc={avg_acc:.3f}  lr={lr:.6f}  time={elapsed:.1f}s")

        if args.wandb:
            import wandb
            wandb.log({
                "epoch": epoch + 1,
                "bc/loss": avg_loss,
                "bc/accuracy": avg_acc,
                "bc/learning_rate": lr,
            })

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = save_dir / "bc_best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best model to {ckpt_path}")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = save_dir / f"bc_epoch_{epoch+1:04d}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # WOSAC evaluation
        if wosac_eval is not None and (epoch + 1) % args.eval_interval == 0:
            eval_env, evaluator, eval_config, gt_traj = wosac_eval
            print(f"  Running WOSAC evaluation...")
            t_eval = time.time()
            try:
                results = run_wosac_eval(model, eval_env, evaluator, eval_config,
                                         gt_traj)
                eval_elapsed = time.time() - t_eval
                meta_score = results.get("realism_meta_score", 0)
                ade = results.get("ade", 0)
                min_ade = results.get("min_ade", 0)
                print(f"  WOSAC: meta_score={meta_score:.4f}  ade={ade:.4f}  "
                      f"min_ade={min_ade:.4f}  time={eval_elapsed:.1f}s")
                if args.wandb:
                    import wandb
                    wandb.log({
                        "epoch": epoch + 1,
                        "eval/wosac_realism_meta_score": meta_score,
                        "eval/wosac_ade": ade,
                        "eval/wosac_min_ade": min_ade,
                    })
            except Exception as e:
                print(f"  WOSAC evaluation failed: {e}")

    # Save final
    torch.save(model.state_dict(), save_dir / "bc_final.pt")
    print(f"Training complete. Best loss: {best_loss:.4f}")

    if wosac_eval is not None:
        wosac_eval[0].close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral Cloning for DriveNoGoal+LSTM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Preprocess command
    pp = subparsers.add_parser("preprocess", help="Preprocess expert data")
    pp.add_argument("--data-root", type=str, default=None)
    pp.add_argument("--split", type=str, default="training")
    pp.add_argument("--output-dir", type=str, default="experiments/bc_cache")
    pp.add_argument("--num-maps", type=int, default=79000)
    pp.add_argument("--num-workers", type=int, default=80)
    pp.add_argument("--overwrite", action="store_true")

    # Train command
    tr = subparsers.add_parser("train", help="Train on cached data")
    tr.add_argument("--cache-dir", type=str, default="experiments/bc_cache")
    tr.add_argument("--data-root", type=str, default=None)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--input-size", type=int, default=64)
    tr.add_argument("--hidden-size", type=int, default=256)
    tr.add_argument("--max-grad-norm", type=float, default=1.0)
    tr.add_argument("--device", type=str, default="cuda")
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--save-dir", type=str, default="experiments/bc_checkpoints")
    tr.add_argument("--save-interval", type=int, default=5)
    tr.add_argument("--goal", action="store_true", help="Use goal-aware Drive policy instead of DriveNoGoal")
    tr.add_argument("--lstm", action="store_true", help="Wrap policy in LSTMWrapper and train on sequences")
    tr.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for cross-entropy loss")
    tr.add_argument("--eval-interval", type=int, default=1, help="Run WOSAC eval every N epochs (0 to disable)")
    tr.add_argument("--eval-num-maps", type=int, default=20, help="Number of maps for WOSAC evaluation")
    tr.add_argument("--eval-data-root", type=str, default=None, help="Data root for WOSAC eval (default: same as --data-root)")
    tr.add_argument("--eval-split", type=str, default="training", help="Split for WOSAC eval dataset")
    tr.add_argument("--wosac-num-rollouts", type=int, default=32, help="Number of rollouts per WOSAC scenario")
    tr.add_argument("--wandb", action="store_true")
    tr.add_argument("--run-name", type=str, default=None)

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess(args)
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    main()
