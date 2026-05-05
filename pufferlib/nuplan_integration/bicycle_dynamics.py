# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Python port of PufferDrive's classic bicycle dynamics model.

Reimplements move_dynamics() from drive.h
for use outside the C simulation environment.
"""

import math
import numpy as np

MAX_SPEED = 45.0  # Cap below nuPlan's perfect_tracking_controller threshold (50 m/s)


def step(
    x: float,
    y: float,
    heading: float,
    vx: float,
    vy: float,
    acceleration: float,
    steering: float,
    vehicle_length: float,
    dt: float = 0.1,
) -> tuple[float, float, float, float, float]:
    """Execute one bicycle model dynamics step.

    Matches drive.h move_dynamics() for CLASSIC dynamics exactly.

    Args:
        x, y: Current position (world frame).
        heading: Current heading angle (radians).
        vx, vy: Current velocity components (world frame).
        acceleration: Longitudinal acceleration (m/s^2).
        steering: Steering angle (radians, positive = left).
        vehicle_length: Vehicle length (meters), used as wheelbase proxy.
        dt: Timestep in seconds.

    Returns:
        (new_x, new_y, new_heading, new_vx, new_vy)
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Signed speed: positive if moving in heading direction
    speed_magnitude = math.sqrt(vx * vx + vy * vy)
    v_dot_heading = vx * cos_h + vy * sin_h
    signed_speed = math.copysign(speed_magnitude, v_dot_heading)

    # Update speed with acceleration
    signed_speed = signed_speed + acceleration * dt
    signed_speed = np.clip(signed_speed, -MAX_SPEED, MAX_SPEED)

    # Slip angle
    beta = math.tanh(0.5 * math.tan(steering))

    # Yaw rate
    yaw_rate = (signed_speed * math.cos(beta) * math.tan(steering)) / vehicle_length

    # New velocity
    new_vx = signed_speed * math.cos(heading + beta)
    new_vy = signed_speed * math.sin(heading + beta)

    # Update position and heading
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_heading = heading + yaw_rate * dt

    return new_x, new_y, new_heading, new_vx, new_vy


def propagate_trajectory(
    x: float,
    y: float,
    heading: float,
    vx: float,
    vy: float,
    acceleration: float,
    steering: float,
    vehicle_length: float,
    num_steps: int = 80,
    dt: float = 0.1,
    constant_velocity_after: int = 1,
) -> np.ndarray:
    """Propagate trajectory: first step with given action, rest with constant velocity.

    Args:
        x, y, heading, vx, vy: Initial state.
        acceleration, steering: Action for the first step(s).
        vehicle_length: Vehicle length.
        num_steps: Total trajectory points (including initial state).
        dt: Timestep.
        constant_velocity_after: After this many steps, switch to constant velocity
            (zero acceleration, zero steering).

    Returns:
        np.ndarray shape (num_steps, 5): [x, y, heading, vx, vy] per step.
    """
    trajectory = np.zeros((num_steps, 5), dtype=np.float64)
    trajectory[0] = [x, y, heading, vx, vy]

    for i in range(1, num_steps):
        if i <= constant_velocity_after:
            accel_i, steer_i = acceleration, steering
        else:
            accel_i, steer_i = 0.0, 0.0

        x, y, heading, vx, vy = step(
            x, y, heading, vx, vy, accel_i, steer_i, vehicle_length, dt
        )
        trajectory[i] = [x, y, heading, vx, vy]

    return trajectory
