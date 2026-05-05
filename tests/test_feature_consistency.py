# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Test that observations from pre-built binaries match nuPlan integration observations.

Compares two observation paths for the same scenario at timestep 0:
  Path A: Load pre-built binary in Drive env → compute_observations() in C
  Path B: Load nuPlan scenario → nuplan_to_binary() → init_obs_env() + compute_obs_external()

For the same agent (SDC), observations must be identical.

Usage:
    NUPLAN_MAPS_ROOT=/path/to/nuplan/maps \
    NUPLAN_DATA_ROOT=/path/to/nuplan/dataset \
    python -m pytest tests/test_feature_consistency.py -v -s
"""

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

GPUDRIVE_JSON_DIR = Path(os.environ.get("GPUDRIVE_JSON_DIR", ""))
GPUDRIVE_BIN_DIR = Path(os.environ.get("DRIVE_BINARIES_DATA_ROOT", ""))
NUPLAN_DATA_ROOT = os.environ.get("NUPLAN_DATA_ROOT", "")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT", "")

# Observation layout constants
EGO_FEATURES = 7
PARTNER_FEATURES = 7
MAX_OBS_PARTNERS = 31
ROAD_FEATURES = 7
MAX_ROAD_SEGMENTS = 128
OBS_SIZE = EGO_FEATURES + PARTNER_FEATURES * MAX_OBS_PARTNERS + ROAD_FEATURES * MAX_ROAD_SEGMENTS


def _read_bin_header(bin_path):
    """Read num_objects and num_roads from binary header."""
    with open(bin_path, "rb") as f:
        sdc_idx = struct.unpack("i", f.read(4))[0]
        n_tracks = struct.unpack("i", f.read(4))[0]
        f.read(n_tracks * 4)
        num_objects = struct.unpack("i", f.read(4))[0]
        num_roads = struct.unpack("i", f.read(4))[0]
    return sdc_idx, num_objects, num_roads


def _pick_test_scenarios(n=5):
    """Pick n scenarios that have <=128 objects and <=400 total entities."""
    json_files = sorted(GPUDRIVE_JSON_DIR.glob("*.json"))
    bin_files = sorted(GPUDRIVE_BIN_DIR.glob("*.bin"))
    assert len(json_files) == len(bin_files), (
        f"Mismatch: {len(json_files)} JSONs vs {len(bin_files)} binaries"
    )

    selected = []
    for i, (jf, bf) in enumerate(zip(json_files, bin_files)):
        _, num_objects, num_roads = _read_bin_header(str(bf))
        if num_objects <= 128 and num_objects + num_roads <= 400:
            selected.append((i, jf, bf))
        if len(selected) >= n:
            break
    return selected


def _load_nuplan_scenario(json_path):
    """Load the nuPlan scenario matching a gpudrive JSON's metadata."""
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    with open(json_path) as f:
        metadata = json.load(f)["metadata"]

    db_path = os.path.join(NUPLAN_DATA_ROOT, "nuplan-v1.1", "trainval", f"{metadata['log_name']}.db")
    if not os.path.exists(db_path):
        pytest.skip(f"DB not found: {db_path}")

    builder = NuPlanScenarioBuilder(
        data_root=NUPLAN_DATA_ROOT, map_root=NUPLAN_MAPS_ROOT,
        sensor_root=NUPLAN_DATA_ROOT, db_files=[db_path], map_version="nuplan-maps-v1.0",
    )

    target_ts = metadata["initial_lidar_timestamp"]
    scenarios = list(builder.get_scenarios(
        ScenarioFilter(
            scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
            num_scenarios_per_type=None, limit_total_scenarios=None,
            timestamp_threshold_s=None, ego_displacement_minimum_m=None,
            expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
        ), Sequential(),
    ))

    for s in scenarios:
        try:
            if abs(s.initial_ego_state.time_point.time_us - target_ts) < 100_000:
                return s
        except Exception:
            continue

    pytest.skip(f"No matching scenario found for {json_path.name}")


def _obs_from_binary(bin_path, map_idx):
    """Path A: Load binary via init_obs_env, compute obs using JSON data at step 0.

    Uses the same compute_obs_external path as the nuPlan integration,
    but with data from the gpudrive JSON (matching the binary exactly).
    """
    import json as json_mod
    from pufferlib.ocean.drive import binding

    # Read ego/agent state from gpudrive JSON
    json_files = sorted(GPUDRIVE_JSON_DIR.glob("*.json"))
    with open(json_files[map_idx]) as f:
        jdata = json_mod.load(f)

    sdc_idx = jdata["metadata"]["sdc_track_index"]
    sdc = jdata["objects"][sdc_idx]

    # Init obs env from the pre-built binary
    obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)
    env_handle = binding.init_obs_env(str(bin_path), obs_buf)

    # SDC state at t=0
    ego_x = sdc["position"][0]["x"]
    ego_y = sdc["position"][0]["y"]
    ego_h = sdc["heading"][0]
    ego_vx = sdc["velocity"][0]["x"]
    ego_vy = sdc["velocity"][0]["y"]
    ego_w = sdc["width"] * 0.7  # 0.7x shrink matching should_control_agent()
    ego_l = sdc["length"] * 0.7
    goal_x = sdc.get("goalPosition", sdc["position"][-1]).get("x", 0)
    goal_y = sdc.get("goalPosition", sdc["position"][-1]).get("y", 0)

    # Build agents array from JSON (all valid non-SDC objects)
    agents_rows = []
    for i, obj in enumerate(jdata["objects"]):
        if i == sdc_idx:
            continue
        if not obj["valid"][0]:
            continue
        p = obj["position"][0]
        if p["x"] == -10000:
            continue
        v = obj["velocity"][0]
        agents_rows.append([p["x"], p["y"], obj["heading"][0],
                            v["x"], v["y"], obj["width"], obj["length"]])

    agents = np.array(agents_rows, dtype=np.float32) if agents_rows else np.zeros((0, 7), dtype=np.float32)

    binding.compute_obs_external(
        env_handle,
        ego_x, ego_y, ego_h,
        ego_vx, ego_vy,
        ego_w, ego_l,
        goal_x, goal_y, agents,
    )

    return obs_buf.copy()


def _obs_from_nuplan(scenario, bin_path):
    """Path B: nuPlan scenario → nuplan_to_binary → init_obs_env + compute_obs_external.

    This mirrors the actual evaluation pipeline: convert nuPlan scenario to binary,
    then compute observations using nuPlan's ego/agent state.
    """
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from pufferlib.ocean.drive import binding
    from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary, TRAJECTORY_LENGTH

    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())

    # Convert nuPlan scenario to binary (roads from nuPlan map API)
    tmp_bin = tempfile.mktemp(suffix=".bin")
    nuplan_to_binary(scenario, map_api, route_ids, tmp_bin)

    # Init obs env
    obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)
    env_handle = binding.init_obs_env(tmp_bin, obs_buf)

    # Get ego state and goal (capped at TRAJECTORY_LENGTH, subsampled 2x for 20Hz→10Hz)
    ego_state = scenario.initial_ego_state
    n_iter = scenario.get_number_of_iterations()
    goal_idx = min(2 * (TRAJECTORY_LENGTH - 1), n_iter - 1)
    goal_state = scenario.get_ego_state_at_iteration(goal_idx)
    goal_x, goal_y = goal_state.center.x, goal_state.center.y

    # Build agents array from nuPlan tracked objects
    agent_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    agents_list = scenario.initial_tracked_objects.tracked_objects.get_tracked_objects_of_types(agent_types)
    if agents_list:
        rows = [[a.box.center.x, a.box.center.y, a.box.center.heading,
                 a.velocity.x, a.velocity.y, a.box.width, a.box.length]
                for a in agents_list]
        agents = np.array(rows, dtype=np.float32)
    else:
        agents = np.zeros((0, 7), dtype=np.float32)

    # Convert ego velocity from body frame to global frame
    import math
    h = ego_state.center.heading
    vx_body = ego_state.dynamic_car_state.center_velocity_2d.x
    vy_body = ego_state.dynamic_car_state.center_velocity_2d.y
    cos_h = math.cos(h)
    sin_h = math.sin(h)
    vx_global = vx_body * cos_h - vy_body * sin_h
    vy_global = vx_body * sin_h + vy_body * cos_h

    # Apply 0.7x bounding box shrink to match Drive env's should_control_agent()
    ego_width = ego_state.car_footprint.width * 0.7
    ego_length = ego_state.car_footprint.length * 0.7

    binding.compute_obs_external(
        env_handle,
        ego_state.center.x, ego_state.center.y, h,
        vx_global, vy_global,
        ego_width, ego_length,
        goal_x, goal_y, agents,
    )

    os.unlink(tmp_bin)
    return obs_buf.copy()


def _compare_obs(obs_a, obs_b, label=""):
    """Compare two observation vectors and return per-section stats."""
    diff = np.abs(obs_a - obs_b)
    ego_diff = diff[:EGO_FEATURES]
    partner_diff = diff[EGO_FEATURES:EGO_FEATURES + MAX_OBS_PARTNERS * PARTNER_FEATURES]
    road_diff = diff[EGO_FEATURES + MAX_OBS_PARTNERS * PARTNER_FEATURES:]

    stats = {
        "total_max": diff.max(),
        "total_mean": diff.mean(),
        "ego_max": ego_diff.max(),
        "ego_mean": ego_diff.mean(),
        "partner_max": partner_diff.max(),
        "partner_mean": partner_diff.mean(),
        "road_max": road_diff.max(),
        "road_mean": road_diff.mean(),
        "match_rate": np.sum(diff < 0.01) / len(diff),
    }

    if label:
        print(f"\n  [{label}] Comparison:")
        print(f"    Ego:      max={stats['ego_max']:.6f}  mean={stats['ego_mean']:.6f}")
        print(f"    Partners: max={stats['partner_max']:.6f}  mean={stats['partner_mean']:.6f}")
        print(f"    Roads:    max={stats['road_max']:.6f}  mean={stats['road_mean']:.6f}")
        print(f"    Match rate (diff<0.01): {stats['match_rate']*100:.1f}%")

    return stats


# ---- Tests ----

scenarios = _pick_test_scenarios(5)


@pytest.mark.parametrize("map_idx,json_path,bin_path", scenarios,
                         ids=[f"map_{i}" for i, _, _ in scenarios])
def test_binary_vs_nuplan_obs(map_idx, json_path, bin_path):
    """Observations from pre-built binary must match nuPlan integration for same scenario."""
    print(f"\n  Loading nuPlan scenario for {json_path.name}...")
    scenario = _load_nuplan_scenario(json_path)

    print(f"  Computing obs from binary (map_idx={map_idx})...")
    obs_binary = _obs_from_binary(bin_path, map_idx)

    print(f"  Computing obs from nuPlan integration...")
    obs_nuplan = _obs_from_nuplan(scenario, bin_path)

    stats = _compare_obs(obs_binary, obs_nuplan, label=json_path.stem)

    # Ego features should be very close (same agent, same position)
    assert stats["ego_max"] < 0.1, (
        f"Ego features differ too much: max_diff={stats['ego_max']:.6f}"
    )

    # Partner features should be close (same agents, same ordering by distance)
    assert stats["partner_max"] < 0.5, (
        f"Partner features differ too much: max_diff={stats['partner_max']:.6f}"
    )

    # Overall match rate should be high
    assert stats["match_rate"] > 0.5, (
        f"Overall match rate too low: {stats['match_rate']*100:.1f}%"
    )


if __name__ == "__main__":
    print(f"JSON dir: {GPUDRIVE_JSON_DIR}")
    print(f"BIN dir:  {GPUDRIVE_BIN_DIR}")
    print(f"Picked {len(scenarios)} test scenarios")

    for map_idx, json_path, bin_path in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario {map_idx}: {json_path.name}")

        scenario = _load_nuplan_scenario(json_path)
        obs_binary = _obs_from_binary(bin_path, map_idx)
        obs_nuplan = _obs_from_nuplan(scenario, bin_path)
        _compare_obs(obs_binary, obs_nuplan, label=json_path.stem)

    print("\nAll done!")
