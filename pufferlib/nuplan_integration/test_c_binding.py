# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Test the C binding pipeline for observation computation.

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.test_c_binding
"""

import glob
import os
import tempfile

import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential


_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


def load_scenario():
    """Load one nuPlan scenario."""
    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "mini", "*.db")))
    if not db_files:
        db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "trainval", "*.db")))[:1]

    print(f"Found {len(db_files)} db files")

    builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=data_root,
        db_files=db_files,
        map_version="nuplan-maps-v1.0",
    )

    scenarios = list(builder.get_scenarios(
        ScenarioFilter(
            scenario_types=None,
            scenario_tokens=None,
            log_names=None,
            map_names=None,
            num_scenarios_per_type=1,
            limit_total_scenarios=1,
            timestamp_threshold_s=None,
            ego_displacement_minimum_m=None,
            expand_scenarios=False,
            remove_invalid_goals=False,
            shuffle=False,
        ),
        Sequential(),
    ))

    assert scenarios, "No scenarios found!"
    return scenarios[0]


def tracked_objects_to_array(tracked_objects):
    """Convert tracked objects to Nx7 float32 array."""
    agents = tracked_objects.get_tracked_objects_of_types(_AGENT_TYPES)
    if not agents:
        return np.zeros((0, 7), dtype=np.float32)
    rows = []
    for agent in agents:
        rows.append([
            agent.box.center.x, agent.box.center.y,
            agent.box.center.heading,
            agent.velocity.x, agent.velocity.y,
            agent.box.width, agent.box.length,
        ])
    return np.array(rows, dtype=np.float32)


def main():
    from pufferlib.ocean.drive import binding
    from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary

    # Step 1: Load scenario
    print("=== Step 1: Load nuPlan scenario ===")
    scenario = load_scenario()
    ego_state = scenario.initial_ego_state
    print(f"  Scenario: {scenario.scenario_type} / {scenario.token}")
    print(f"  Ego: ({ego_state.center.x:.1f}, {ego_state.center.y:.1f})")
    print(f"  Iterations: {scenario.get_number_of_iterations()}")

    # Step 2: Create .bin
    print("\n=== Step 2: Create .bin from nuPlan ===")
    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())
    bin_path = tempfile.mktemp(suffix=".bin")
    nuplan_to_binary(scenario, map_api, route_ids, bin_path)
    bin_size = os.path.getsize(bin_path)
    print(f"  Binary written: {bin_path} ({bin_size} bytes)")

    # Step 3: Init C env
    print("\n=== Step 3: Init C observation env ===")
    obs_buf = np.zeros(1120, dtype=np.float32)
    env_handle = binding.init_obs_env(bin_path, obs_buf)
    print(f"  C env handle: {env_handle}")
    assert env_handle != 0, "init_obs_env returned NULL!"

    # Step 4: Compute observations for initial state
    print("\n=== Step 4: Compute observations (initial state) ===")
    last_idx = scenario.get_number_of_iterations() - 1
    last_state = scenario.get_ego_state_at_iteration(last_idx)
    goal_x, goal_y = last_state.center.x, last_state.center.y

    detections = scenario.initial_tracked_objects.tracked_objects
    agents = tracked_objects_to_array(detections)
    print(f"  Tracked agents: {agents.shape[0]}")

    binding.compute_obs_external(
        env_handle,
        ego_state.center.x,
        ego_state.center.y,
        ego_state.center.heading,
        ego_state.dynamic_car_state.center_velocity_2d.x,
        ego_state.dynamic_car_state.center_velocity_2d.y,
        ego_state.car_footprint.width,
        ego_state.car_footprint.length,
        goal_x,
        goal_y,
        agents,
    )

    print(f"  Obs buffer shape: {obs_buf.shape}")
    print(f"  Non-zero: {np.count_nonzero(obs_buf)} / {len(obs_buf)}")
    print(f"  All finite: {np.all(np.isfinite(obs_buf))}")
    print(f"  Min: {obs_buf.min():.6f}, Max: {obs_buf.max():.6f}")

    # Parse observation structure
    ego_obs = obs_buf[:7]
    partner_obs = obs_buf[7:7 + 31 * 7].reshape(31, 7)
    road_obs = obs_buf[7 + 31 * 7:].reshape(128, 7)

    print(f"\n  Ego features: {ego_obs}")

    active_partners = np.sum(np.any(partner_obs != 0, axis=1))
    print(f"  Active partners: {active_partners} / 31")

    active_roads = np.sum(np.any(road_obs != 0, axis=1))
    print(f"  Active road segments: {active_roads} / 128")

    # Road type distribution
    road_types = road_obs[:, 6]  # type is last feature
    active_mask = np.any(road_obs != 0, axis=1)
    if active_mask.any():
        types_active = road_types[active_mask]
        for t_val, name in [(0.0, "LANE"), (1.0, "LINE"), (2.0, "EDGE")]:
            count = np.sum(np.abs(types_active - t_val) < 0.01)
            print(f"    {name}: {count}")

    # Step 5: Test multiple timesteps
    print("\n=== Step 5: Test multiple timesteps ===")
    n_test = min(5, scenario.get_number_of_iterations())
    for i in range(n_test):
        state = scenario.get_ego_state_at_iteration(i)
        binding.compute_obs_external(
            env_handle,
            state.center.x,
            state.center.y,
            state.center.heading,
            state.dynamic_car_state.center_velocity_2d.x,
            state.dynamic_car_state.center_velocity_2d.y,
            state.car_footprint.width,
            state.car_footprint.length,
            goal_x,
            goal_y,
            agents,  # using initial agents for simplicity
        )
        nonzero = np.count_nonzero(obs_buf)
        finite = np.all(np.isfinite(obs_buf))
        print(f"  t={i}: nonzero={nonzero}, finite={finite}, "
              f"ego_pos=({state.center.x:.1f}, {state.center.y:.1f})")

    # Step 6: Plot observations
    print("\n=== Step 6: Plot observations ===")
    from pufferlib.nuplan_integration.visualize_obs import plot_obs

    # Re-compute for initial state (t=0)
    binding.compute_obs_external(
        env_handle,
        ego_state.center.x,
        ego_state.center.y,
        ego_state.center.heading,
        ego_state.dynamic_car_state.center_velocity_2d.x,
        ego_state.dynamic_car_state.center_velocity_2d.y,
        ego_state.car_footprint.width,
        ego_state.car_footprint.length,
        goal_x,
        goal_y,
        agents,
    )

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scenario_videos")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "c_binding_obs_t0.png")
    plot_obs(obs_buf, title="C Binding Observation (t=0)", save_path=plot_path)

    # Cleanup
    os.unlink(bin_path)
    print(f"\n=== PASSED: All {n_test} timesteps produced valid observations ===")


if __name__ == "__main__":
    main()
