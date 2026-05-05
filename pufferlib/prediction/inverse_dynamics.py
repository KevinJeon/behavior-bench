# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Inverse dynamics: derive discrete actions from Waymo trajectories.

Maps observed state transitions to the closest discrete action in the
32 acceleration x 32 steering = 1024 action space (SMART 7M vocab size).

Supports segment-level matching (shift=5): find the best single action
applied for `shift` consecutive steps that best matches the GT trajectory.

Uses the classic bicycle model from drive.h:1642-1679.
"""

import math
import numpy as np
from typing import List, Dict

# Action grid: 32 acc x 32 steer = 1024 tokens (SMART 7M)
ACCELERATION_VALUES = np.linspace(-4.0, 4.0, 32, dtype=np.float32)
STEERING_VALUES = np.linspace(-1.0, 1.0, 32, dtype=np.float32)

NUM_ACC = len(ACCELERATION_VALUES)    # 32
NUM_STEER = len(STEERING_VALUES)      # 32
NUM_ACTIONS = NUM_ACC * NUM_STEER     # 1024

# Pre-computed grids for vectorized simulation (float64 for precision)
_ACC_GRID = np.repeat(ACCELERATION_VALUES, NUM_STEER).astype(np.float64)
_STEER_GRID = np.tile(STEERING_VALUES, NUM_ACC).astype(np.float64)
_BETA_GRID = np.tanh(0.5 * np.tan(_STEER_GRID))
_TAN_STEER_GRID = np.tan(_STEER_GRID)
_COS_BETA_GRID = np.cos(_BETA_GRID)


def bicycle_dynamics_step(x, y, heading, signed_speed, acceleration, steering,
                          vehicle_length, dt=0.1):
    """Single step of the classic bicycle model (drive.h:1642-1679)."""
    new_signed_speed = signed_speed + acceleration * dt
    wheelbase = 0.6 * vehicle_length
    if wheelbase < 0.1:
        wheelbase = 0.1

    beta = math.tanh(0.5 * math.tan(steering))
    yaw_rate = (new_signed_speed * math.cos(beta) * math.tan(steering)) / wheelbase
    new_vx = new_signed_speed * math.cos(heading + beta)
    new_vy = new_signed_speed * math.sin(heading + beta)
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_heading = heading + yaw_rate * dt
    return new_x, new_y, new_heading, new_signed_speed


def _simulate_all_actions(x, y, heading, signed_speed, vehicle_length,
                          num_steps=1, dt=0.1):
    """Vectorized: simulate all 2048 actions for num_steps sub-steps.

    Returns:
        np.ndarray of shape (2048, 4): [new_x, new_y, new_heading, new_speed].
    """
    wheelbase = max(0.6 * vehicle_length, 0.1)

    cx = np.full(NUM_ACTIONS, x, dtype=np.float64)
    cy = np.full(NUM_ACTIONS, y, dtype=np.float64)
    ch = np.full(NUM_ACTIONS, heading, dtype=np.float64)
    cs = np.full(NUM_ACTIONS, signed_speed, dtype=np.float64)

    for _ in range(num_steps):
        cs = cs + _ACC_GRID * dt
        yaw_rate = (cs * _COS_BETA_GRID * _TAN_STEER_GRID) / wheelbase
        vx = cs * np.cos(ch + _BETA_GRID)
        vy = cs * np.sin(ch + _BETA_GRID)
        cx = cx + vx * dt
        cy = cy + vy * dt
        ch = ch + yaw_rate * dt

    return np.stack([cx, cy, ch, cs], axis=-1).astype(np.float32)


def compute_gt_actions_for_agent(traj_x, traj_y, traj_heading, traj_vx, traj_vy,
                                 traj_valid, vehicle_length, shift=5, dt=0.1):
    """Compute GT discrete actions per segment for a single agent.

    For each segment [k*shift, (k+1)*shift), simulates all 2048 actions
    for `shift` sub-steps and picks the one that best matches the GT
    position at step (k+1)*shift.

    Args:
        traj_x, traj_y: Position arrays of shape (T,).
        traj_heading: Heading array of shape (T,).
        traj_vx, traj_vy: Velocity arrays of shape (T,).
        traj_valid: Validity flags of shape (T,).
        vehicle_length: Vehicle length scalar.
        shift: Number of sub-steps per segment (token shift).
        dt: Timestep.

    Returns:
        actions: np.ndarray of shape (num_tokens,) with action indices [0..2047].
        valid: np.ndarray of shape (num_tokens,) boolean.
    """
    T = len(traj_x)
    token_steps = list(range(0, T, shift))
    num_tokens = len(token_steps)

    actions = np.zeros(num_tokens, dtype=np.int64)
    valid = np.zeros(num_tokens, dtype=bool)

    for k in range(num_tokens - 1):
        t_start = token_steps[k]
        t_end = token_steps[k + 1]

        if t_end >= T or not (traj_valid[t_start] and traj_valid[t_end]):
            continue

        valid[k] = True

        x = float(traj_x[t_start])
        y = float(traj_y[t_start])
        h = float(traj_heading[t_start])
        speed = float(math.sqrt(traj_vx[t_start]**2 + traj_vy[t_start]**2))
        speed_dir = math.cos(h) * traj_vx[t_start] + math.sin(h) * traj_vy[t_start]
        if speed_dir < 0:
            speed = -speed

        num_sub = t_end - t_start
        predictions = _simulate_all_actions(x, y, h, speed, vehicle_length, num_sub, dt)

        gt_x = float(traj_x[t_end])
        gt_y = float(traj_y[t_end])
        gt_h = float(traj_heading[t_end])

        pos_error = (predictions[:, 0] - gt_x)**2 + (predictions[:, 1] - gt_y)**2
        heading_diff = predictions[:, 2] - gt_h
        heading_error = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))**2
        total_error = pos_error + 0.1 * heading_error
        actions[k] = np.argmin(total_error)

    return actions, valid


def compute_gt_actions(objects: List[Dict], shift: int = 5,
                       dt: float = 0.1) -> List[Dict]:
    """Compute GT discrete actions for all agents in a scenario.

    Returns:
        List of dicts with 'actions' (num_tokens,) and 'valid' (num_tokens,).
    """
    results = []
    for obj in objects:
        vl = max(obj['length'], 0.1)
        actions, valid = compute_gt_actions_for_agent(
            obj['traj_x'], obj['traj_y'], obj['traj_heading'],
            obj['traj_vx'], obj['traj_vy'], obj['traj_valid'],
            vehicle_length=vl, shift=shift, dt=dt)
        results.append({'actions': actions, 'valid': valid})
    return results


def action_index_to_values(action_idx):
    """Convert action index [0..2047] to (acceleration, steering) values."""
    acc_idx = action_idx // NUM_STEER
    steer_idx = action_idx % NUM_STEER
    return ACCELERATION_VALUES[acc_idx], STEERING_VALUES[steer_idx]
