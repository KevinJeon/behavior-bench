# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Generates 80-point nuPlan trajectories from a single PPO action.

Step 1: Apply the PPO action via bicycle dynamics.
Steps 2-80: Constant-velocity extrapolation (zero acceleration/steering).
Returns an InterpolatedTrajectory of EgoState objects for nuPlan.
"""

import math

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

from .bicycle_dynamics import propagate_trajectory

# nuPlan default vehicle
_VEHICLE_PARAMS = get_pacifica_parameters()


def build_trajectory(
    ego_state: EgoState,
    acceleration: float,
    steering: float,
    num_steps: int = 80,
    dt: float = 0.1,
    brake_to_stop: bool = False,
) -> InterpolatedTrajectory:
    """Build an 80-point nuPlan trajectory from a single PPO action.

    Args:
        ego_state: Current ego state from nuPlan.
        acceleration: Longitudinal acceleration from PPO (m/s^2).
        steering: Steering angle from PPO (radians).
        num_steps: Number of trajectory points (8s at 10Hz = 80).
        dt: Timestep in seconds.
        brake_to_stop: If True, apply acceleration over ALL steps (for goal braking).

    Returns:
        InterpolatedTrajectory with num_steps EgoState waypoints.
    """
    # Extract current state (use center, matching PufferDrive convention)
    x = ego_state.center.x
    y = ego_state.center.y
    heading = ego_state.center.heading
    vehicle_length = ego_state.car_footprint.length

    # nuPlan velocity is in body frame (longitudinal, lateral) — convert to global
    vx_body = ego_state.dynamic_car_state.center_velocity_2d.x
    vy_body = ego_state.dynamic_car_state.center_velocity_2d.y
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx = vx_body * cos_h - vy_body * sin_h
    vy = vx_body * sin_h + vy_body * cos_h

    # Propagate: normally 1 step with action then constant velocity,
    # but for braking apply deceleration over all steps until stopped
    traj = propagate_trajectory(
        x, y, heading, vx, vy,
        acceleration, steering, vehicle_length,
        num_steps=num_steps, dt=dt,
        constant_velocity_after=num_steps if brake_to_stop else 1,
    )

    # Convert to nuPlan EgoState list
    # We need to convert center positions back to rear_axle for nuPlan
    rear_axle_to_center = _VEHICLE_PARAMS.rear_axle_to_center
    start_time_us = ego_state.time_us

    # nuPlan requires the trajectory to include the current time as the first point
    ego_states = [ego_state]

    for i in range(1, num_steps):
        t_x, t_y, t_heading, t_vx, t_vy = traj[i]
        cos_h = math.cos(t_heading)
        sin_h = math.sin(t_heading)

        # Convert center to rear axle position
        ra_x = t_x - rear_axle_to_center * cos_h
        ra_y = t_y - rear_axle_to_center * sin_h

        # Convert global velocity to body frame (longitudinal, lateral)
        # nuPlan's build_from_rear_axle expects body-frame velocity
        vx_body = t_vx * cos_h + t_vy * sin_h
        vy_body = -t_vx * sin_h + t_vy * cos_h

        time_us = start_time_us + int(i * dt * 1e6)

        state = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(ra_x, ra_y, t_heading),
            rear_axle_velocity_2d=StateVector2D(vx_body, vy_body),
            rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
            tire_steering_angle=0.0,
            time_point=TimePoint(time_us),
            vehicle_parameters=_VEHICLE_PARAMS,
        )
        ego_states.append(state)

    return InterpolatedTrajectory(ego_states)
