# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Prepare BC training data with token labels for token-based behavioral cloning.

For each map: runs Drive env in expert replay mode, collects observations at
shift=2 boundaries, computes GT token labels via trajectory tokenizer.

Usage:
    DRIVE_BINARIES_DATA_ROOT=/path/to/binaries \
    python scripts/prepare_bc_token_data.py \
        --codebook codebook_200_shift2.pkl \
        --output-dir /path/to/bc_token_data \
        --num-maps 79000 --num-workers 80 --shift 2
"""

import os
import sys
import argparse
import pickle
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

# Global codebook loaded once per worker
_CODEBOOK_POLY = None
_SHIFT = 2


def _init_worker(codebook_path, shift):
    """Initialize worker process with shared codebook."""
    global _CODEBOOK_POLY, _SHIFT
    _SHIFT = shift
    with open(codebook_path, 'rb') as f:
        cb = pickle.load(f)
    _CODEBOOK_POLY = cb['token']['veh']


def _process_single_map(task):
    """Process a single map: collect observations + compute token labels."""
    map_id, data_root, split, out_path = task
    try:
        return _process_map_impl(map_id, data_root, split, out_path)
    except Exception as e:
        return f"error[{map_id}]: {e}"


def _process_map_impl(map_id, data_root, split, out_path):
    import os
    from pufferlib.ocean.drive import binding
    from pufferlib.prediction.trajectory_tokenizer import compute_token_data

    # Skip if already cached
    if os.path.exists(out_path):
        return "cached"

    global _CODEBOOK_POLY, _SHIFT
    shift = _SHIFT

    # Use C binding directly (same as pufferbc.py)
    agent_offsets, map_ids, num_envs = binding.shared(
        num_agents=512, num_maps=1, split=split, data_root=data_root,
        init_mode=0, control_mode=0, init_steps=10,
        max_controlled_agents=-1, goal_behavior=3, goal_target_distance=30.0,
        use_all_maps=False, map_id=map_id,
    )
    num_agents = agent_offsets[-1]
    if num_agents == 0:
        return "empty"

    obs_dim = 1120
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
        42,
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

    # Set all agents to expert replay
    all_indices = list(range(num_agents))
    binding.vec_set_movement_mode(c_envs, all_indices, 2)

    # Collect observations over T steps
    all_obs = np.zeros((num_agents, T, obs_dim), dtype=np.float32)
    all_obs[:, 0] = observations.copy()

    for t in range(1, T):
        actions_buf[:] = 0
        binding.vec_step(c_envs)
        all_obs[:, t] = observations.copy()

    # Get GT trajectories
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

    # Compute token labels using trajectory tokenizer
    positions = np.stack([gt_x, gt_y], axis=-1)  # (N, gt_steps, 2)
    token_data = compute_token_data(
        positions, gt_heading, gt_valid_arr.astype(bool),
        shift=shift, codebook=_CODEBOOK_POLY,
    )
    token_idx = token_data['token_idx']  # (N, num_tokens)
    num_tokens = token_idx.shape[1]

    # Subsample observations at shift boundaries
    boundary_times = list(range(0, T, shift))[:num_tokens]
    if len(boundary_times) < num_tokens:
        boundary_times.extend([T - 1] * (num_tokens - len(boundary_times)))
    obs_at_boundaries = all_obs[:, boundary_times, :]  # (N, num_tokens, obs_dim)

    # Valid mask per token: both start and end of token must be valid + must be vehicle
    valid_token = np.ones((num_agents, num_tokens), dtype=bool)
    for i in range(num_tokens):
        t_start = i * shift
        t_end = min((i + 1) * shift, gt_steps - 1)
        valid_token[:, i] = (
            gt_valid_arr[:, t_start].astype(bool) &
            gt_valid_arr[:, t_end].astype(bool) &
            gt_is_vehicle.astype(bool)
        )

    # Filter agents: must be vehicle with >2m displacement
    agent_mask = valid_token.any(axis=1)
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
        return "empty"

    data = {
        "observations": torch.from_numpy(obs_at_boundaries[agent_mask].astype(np.float16)),
        "token_labels": token_idx[agent_mask],
        "valid": torch.from_numpy(valid_token[agent_mask]),
    }
    torch.save(data, out_path)
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Prepare BC token training data")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--codebook", type=str, required=True,
                        help="Path to codebook pickle (from extract_tokens.py)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shift", type=int, default=2)
    parser.add_argument("--num-maps", type=int, default=79000)
    parser.add_argument("--num-workers", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root or os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    if not data_root:
        raise ValueError("Set --data-root or DRIVE_BINARIES_DATA_ROOT")

    split_dir = Path(data_root) / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task list directly from map_id range.
    # Skip exists() checks on NFS (too slow for 460k files) — workers handle missing files.
    tasks = []
    for map_id in range(args.num_maps):
        out_path = str(output_dir / f"map_{map_id:06d}.pt")
        tasks.append((map_id, data_root, args.split, out_path))
    skipped = 0

    print(f"Processing {len(tasks)} maps ({skipped} already cached), "
          f"output: {output_dir}")

    if not tasks:
        print("Nothing to do.")
        return

    t_start = time.time()
    processed = 0
    ok_count = 0
    empty_count = 0
    cached_count = 0
    error_count = 0

    with mp.Pool(args.num_workers, initializer=_init_worker,
                 initargs=(args.codebook, args.shift)) as pool:
        for result in pool.imap_unordered(_process_single_map, tasks, chunksize=16):
            processed += 1
            if result == "ok":
                ok_count += 1
            elif result == "empty":
                empty_count += 1
            elif result == "cached":
                cached_count += 1
            else:
                error_count += 1
                if error_count <= 5:
                    print(f"  {result}")

            if processed % 1000 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                remaining = (len(tasks) - processed) / max(rate, 1e-6)
                print(f"  {processed}/{len(tasks)} ({rate:.1f}/s, "
                      f"~{remaining/60:.0f}min remaining, "
                      f"{ok_count} ok, {empty_count} empty, {error_count} errors)")

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s: {ok_count} ok, {empty_count} empty, "
          f"{cached_count} cached, {error_count} errors")


if __name__ == "__main__":
    main()
