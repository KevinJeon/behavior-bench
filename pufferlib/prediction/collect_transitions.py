# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Collect transitions from RL policy rollouts for world model fine-tuning.

Loads a trained policy checkpoint, runs it in the Drive environment,
and saves scene-level transitions (all agents' states + actions) as .pt files.

Usage:
    python -m pufferlib.prediction.collect_transitions \
        --policy-ckpt experiments/rl/policy.pt \
        --env-config config/ocean/drive.ini \
        --save-dir experiments/transitions \
        --num-episodes 1000
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from pufferlib.prediction.inverse_dynamics import compute_gt_actions_for_agent
from pufferlib.prediction.map_tokenizer import tokenize_roads, load_map_codebook


def collect_transitions(args):
    """Collect transitions from policy rollouts."""
    # Lazy imports to avoid requiring pufferlib env setup unless needed
    import pufferlib
    import pufferlib.vector
    from pufferlib.ocean.drive.drive import Drive

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    map_codebook = load_map_codebook()

    # Create environment
    print(f"Creating environment from config: {args.env_config}")
    env = Drive.from_config(args.env_config)

    # Load policy
    policy = None
    if args.policy_ckpt and os.path.exists(args.policy_ckpt):
        print(f"Loading policy from: {args.policy_ckpt}")
        ckpt = torch.load(args.policy_ckpt, map_location='cpu', weights_only=False)
        # The policy loading depends on how your policies are saved
        # This is a placeholder - adapt to your specific policy format
        if 'model_state_dict' in ckpt:
            from pufferlib.ocean.torch import Policy
            policy = Policy(env)
            policy.load_state_dict(ckpt['model_state_dict'])
            policy.eval()
        else:
            policy = ckpt
    else:
        print("No policy checkpoint provided, using random actions")

    num_saved = 0
    dt = float(getattr(env, 'dt', 0.1))

    for episode_idx in range(args.num_episodes):
        obs, info = env.reset()
        episode_states = []
        done = False
        step = 0

        while not done:
            # Get action from policy or random
            if policy is not None:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                    action = policy(obs_tensor).squeeze(0).numpy()
            else:
                action = env.action_space.sample()

            # Record full scene state BEFORE stepping
            state = env.get_global_agent_state()
            episode_states.append({
                'x': np.array(state['x'], dtype=np.float32),
                'y': np.array(state['y'], dtype=np.float32),
                'z': np.array(state.get('z', np.zeros_like(state['x'])), dtype=np.float32),
                'heading': np.array(state['heading'], dtype=np.float32),
                'id': np.array(state['id'], dtype=np.int32),
                'length': np.array(state['length'], dtype=np.float32),
                'width': np.array(state['width'], dtype=np.float32),
                'step': step,
            })

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        if len(episode_states) < 3:
            continue

        # Convert episode states to trajectories and save
        data = _process_episode(episode_states, env, map_codebook, dt)
        if data is not None:
            save_path = save_dir / f'episode_{num_saved:06d}.pt'
            torch.save(data, save_path)
            num_saved += 1

            if num_saved % 100 == 0:
                print(f"Saved {num_saved}/{args.num_episodes} episodes")

    print(f"Done. Saved {num_saved} episodes to {save_dir}")


def _process_episode(episode_states, env, map_codebook, dt=0.1):
    """Convert episode states list to HeteroData."""
    T = len(episode_states)
    if T < 3:
        return None

    # Get agent IDs from first frame
    agent_ids = episode_states[0]['id']
    N = len(agent_ids)
    if N == 0:
        return None

    # Build trajectory arrays
    position = torch.zeros(N, T, 2)
    heading = torch.zeros(N, T)
    valid_mask = torch.ones(N, T, dtype=torch.bool)

    for t, state in enumerate(episode_states):
        # Match agents by ID across timesteps
        for i, aid in enumerate(agent_ids):
            idx_in_state = np.where(state['id'] == aid)[0]
            if len(idx_in_state) > 0:
                j = idx_in_state[0]
                position[i, t, 0] = state['x'][j]
                position[i, t, 1] = state['y'][j]
                heading[i, t] = state['heading'][j]
            else:
                valid_mask[i, t] = False

    # Compute velocities from position differences
    velocity = torch.zeros(N, T, 2)
    velocity[:, 1:] = (position[:, 1:] - position[:, :-1]) / dt

    # Compute GT actions via inverse dynamics
    gt_actions = torch.zeros(N, T, dtype=torch.long)
    action_valid = torch.zeros(N, T, dtype=torch.bool)

    for i in range(N):
        length_val = float(episode_states[0]['length'][i])
        actions, av = compute_gt_actions_for_agent(
            position[i, :, 0].numpy(),
            position[i, :, 1].numpy(),
            heading[i].numpy(),
            velocity[i, :, 0].numpy(),
            velocity[i, :, 1].numpy(),
            valid_mask[i].numpy().astype(np.int32),
            vehicle_length=max(length_val, 0.1),
            dt=dt)
        gt_actions[i, :len(actions)] = torch.from_numpy(actions)
        action_valid[i, :len(av)] = torch.from_numpy(av)

    # Get road data
    try:
        road_data = env.get_road_edge_polylines()
        roads = _road_data_to_list(road_data)
    except Exception:
        roads = []

    map_data = tokenize_roads(roads, map_codebook)

    # Agent shape info
    shape = torch.zeros(N, T, 3)
    for i in range(N):
        shape[i, :, 0] = episode_states[0]['width'][i]
        shape[i, :, 1] = episode_states[0]['length'][i]

    # Map raw entity types (1=veh, 2=ped, 3=cyc) to SMART indices (0, 1, 2)
    raw_types = episode_states[0].get('type', np.ones(N, dtype=np.int32))
    agent_type = torch.from_numpy(np.clip(raw_types - 1, 0, 2)).long()

    # Build HeteroData
    data = HeteroData()
    num_hist = min(11, T)

    data['agent'] = {}
    data['agent']['position'] = position
    data['agent']['heading'] = heading
    data['agent']['velocity'] = velocity
    data['agent']['valid_mask'] = valid_mask
    data['agent']['type'] = agent_type
    data['agent']['shape'] = shape
    data['agent']['category'] = torch.full((N,), 3, dtype=torch.long)
    data['agent']['num_nodes'] = N
    data['agent']['av_index'] = 0
    data['agent']['token_pos'] = position
    data['agent']['token_heading'] = heading
    data['agent']['token_idx'] = gt_actions
    data['agent']['agent_valid_mask'] = valid_mask & action_valid

    for key, val in map_data['pt_token'].items():
        data['pt_token'][key] = val

    return data


def _road_data_to_list(road_data):
    """Convert road data from env.get_road_edge_polylines() to list format."""
    roads = []
    if 'x' not in road_data or len(road_data['x']) == 0:
        return roads

    x_all = np.array(road_data['x'])
    y_all = np.array(road_data['y'])
    lengths = np.array(road_data['lengths'])

    offset = 0
    for i, length in enumerate(lengths):
        roads.append({
            'type': 6,  # road edge
            'id': i,
            'polyline_x': x_all[offset:offset + length],
            'polyline_y': y_all[offset:offset + length],
            'polyline_z': np.zeros(length, dtype=np.float32),
            'exit_lanes': [],
        })
        offset += length

    return roads


def main():
    parser = argparse.ArgumentParser(
        description='Collect transitions from RL policy rollouts')
    parser.add_argument('--policy-ckpt', type=str, default='',
                        help='Path to trained policy checkpoint')
    parser.add_argument('--env-config', type=str,
                        default='config/ocean/drive.ini',
                        help='Drive environment config file')
    parser.add_argument('--save-dir', type=str,
                        default='experiments/transitions',
                        help='Directory to save transition .pt files')
    parser.add_argument('--num-episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    args = parser.parse_args()
    collect_transitions(args)


if __name__ == '__main__':
    main()
