# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Decode SMART motion tokens to (acceleration, steering) action sequences.

Each SMART token represents a short trajectory segment (shift steps at dt=0.1s).
The codebook entry traj['veh'][token_idx] has shape (shift+1, 3) = (x, y, heading)
in LOCAL frame relative to the agent's state at the token boundary:
  waypoint 0 = boundary state (displacement ≈ 0)
  waypoints 1..shift = predicted positions at t+1..shift steps

For shift=5 (2Hz): 6 waypoints, 0.5s per token.
For shift=2 (5Hz): 3 waypoints, 0.2s per token.

The P-controller converts world-space waypoints to (accel, steer) actions,
replicating SMARTPlanner._positions_to_actions logic.
"""

import numpy as np

DT = 0.1        # timestep in seconds
MAX_ACCEL = 4.0  # m/s^2, matches action space
K_STEER = 2.0   # proportional gain for steering


def decode_token_to_actions(
    token_idx: int,
    cur_pos: np.ndarray,
    cur_heading: float,
    cur_speed: float,
    traj_codebook: np.ndarray,
) -> np.ndarray:
    """Decode a motion token to (accel, steer) action pairs via P-controller.

    Args:
        token_idx: int in [0, K)
        cur_pos: (2,) world position (x, y) at the token boundary
        cur_heading: float, world heading in radians at the boundary
        cur_speed: float, speed in m/s at the boundary
        traj_codebook: (K, shift+1, 3) LOCAL (x, y, heading) trajectories.
            Waypoint 0 ≈ current state; waypoints 1..shift are future steps.

    Returns:
        (shift, 2) float32 array of (acceleration, steering) in [-1, 1]
    """
    traj_local = traj_codebook[int(token_idx)]  # (shift+1, 3)
    num_steps = traj_local.shape[0] - 1  # shift

    cos_h = np.cos(cur_heading)
    sin_h = np.sin(cur_heading)
    lx = traj_local[:, 0]
    ly = traj_local[:, 1]

    # Rotate local (x, y) → world coordinates
    wx = cur_pos[0] + lx * cos_h - ly * sin_h
    wy = cur_pos[1] + lx * sin_h + ly * cos_h
    world_xy = np.stack([wx, wy], axis=1)  # (shift+1, 2)
    world_heading = cur_heading + traj_local[:, 2]  # (shift+1,)

    actions = np.zeros((num_steps, 2), dtype=np.float32)

    for k in range(num_steps):
        pos_k = world_xy[k]
        head_k = world_heading[k]

        disp = float(np.linalg.norm(world_xy[k + 1] - world_xy[k]))
        speed_k = disp / DT
        if k == 0:
            speed_k = 0.5 * speed_k + 0.5 * float(cur_speed)

        tgt = world_xy[k + 1]
        dx = float(tgt[0] - pos_k[0])
        dy = float(tgt[1] - pos_k[1])
        dist = np.sqrt(dx * dx + dy * dy)

        if dist < 1e-4:
            continue

        desired_heading = np.arctan2(dy, dx)
        heading_error = (desired_heading - head_k + np.pi) % (2 * np.pi) - np.pi
        steering = float(np.clip(heading_error * K_STEER, -1.0, 1.0))

        desired_speed = dist / DT
        accel_raw = (desired_speed - speed_k) / DT
        acceleration = float(np.clip(accel_raw / MAX_ACCEL, -1.0, 1.0))

        actions[k, 0] = acceleration
        actions[k, 1] = steering

    return actions


def decode_tokens_batch(
    token_indices: np.ndarray,
    cur_pos: np.ndarray,
    cur_heading: np.ndarray,
    cur_speed: np.ndarray,
    traj_codebook: np.ndarray,
) -> np.ndarray:
    """Vectorized decode: N tokens → (N, shift, 2) action arrays.

    Args:
        token_indices: (N,) int array
        cur_pos: (N, 2) world positions at token boundaries
        cur_heading: (N,) world headings in radians
        cur_speed: (N,) speeds in m/s
        traj_codebook: (K, shift+1, 3) LOCAL trajectories

    Returns:
        (N, shift, 2) float32 array of (acceleration, steering) in [-1, 1]
    """
    N = len(token_indices)
    num_steps = traj_codebook.shape[1] - 1  # shift

    actions = np.zeros((N, num_steps, 2), dtype=np.float32)

    traj_local = traj_codebook[token_indices.astype(int)]  # (N, shift+1, 3)

    cos_h = np.cos(cur_heading)
    sin_h = np.sin(cur_heading)
    lx = traj_local[:, :, 0]
    ly = traj_local[:, :, 1]

    wx = cur_pos[:, 0:1] + lx * cos_h[:, None] - ly * sin_h[:, None]
    wy = cur_pos[:, 1:2] + lx * sin_h[:, None] + ly * cos_h[:, None]
    world_xy = np.stack([wx, wy], axis=2)  # (N, shift+1, 2)
    world_heading = cur_heading[:, None] + traj_local[:, :, 2]  # (N, shift+1)

    for k in range(num_steps):
        pos_k = world_xy[:, k]
        head_k = world_heading[:, k]

        disp = np.linalg.norm(world_xy[:, k + 1] - world_xy[:, k], axis=1)
        speed_k = disp / DT
        if k == 0:
            speed_k = 0.5 * speed_k + 0.5 * cur_speed

        tgt = world_xy[:, k + 1]
        dx = tgt[:, 0] - pos_k[:, 0]
        dy = tgt[:, 1] - pos_k[:, 1]
        dist = np.sqrt(dx * dx + dy * dy)

        valid = dist >= 1e-4

        desired_heading = np.arctan2(dy, dx)
        heading_error = (desired_heading - head_k + np.pi) % (2 * np.pi) - np.pi
        steering = np.clip(heading_error * K_STEER, -1.0, 1.0)

        desired_speed = np.where(valid, dist / DT, speed_k)
        accel_raw = (desired_speed - speed_k) / DT
        acceleration = np.clip(accel_raw / MAX_ACCEL, -1.0, 1.0)

        actions[:, k, 0] = np.where(valid, acceleration, 0.0)
        actions[:, k, 1] = np.where(valid, steering, 0.0)

    return actions
