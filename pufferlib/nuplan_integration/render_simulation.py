# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Render PufferDrive policy rollout on a nuPlan scenario as MP4 video.

Creates a bird's-eye view animation showing observations, actions, and
planned trajectory at each timestep.

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.render_simulation
"""

import math
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import torch

from pufferlib.nuplan_integration.observation_builder import (
    ObservationBuilder, OBS_SIZE,
    EGO_FEATURES, PARTNER_FEATURES, ROAD_FEATURES,
    MAX_PARTNERS, MAX_ROAD_SEGMENTS,
)
from pufferlib.nuplan_integration.visualize_obs import parse_obs
from pufferlib.nuplan_integration.bicycle_dynamics import propagate_trajectory


# Must match planner.py
ACCEL_VALUES = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
STEER_VALUES = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
NUM_STEER = len(STEER_VALUES)


def _load_policy(weights_path, input_size=64, hidden_size=256, device="cpu"):
    """Load PPO policy with LSTM wrapper."""
    from pufferlib.models import LSTMWrapper
    from pufferlib.ocean.torch import Drive as DrivePolicy
    from pufferlib.nuplan_integration.planner import _MockDriveEnv

    mock_env = _MockDriveEnv()
    base_policy = DrivePolicy(mock_env, input_size=input_size, hidden_size=hidden_size)
    policy = LSTMWrapper(mock_env, base_policy, input_size=hidden_size, hidden_size=hidden_size)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    policy.load_state_dict(cleaned, strict=False)
    policy.eval()
    return policy, device


def _policy_step(policy, obs, lstm_h, lstm_c, device):
    """Run one policy step, return action indices and updated LSTM state."""
    obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    state = {"lstm_h": lstm_h, "lstm_c": lstm_c}

    with torch.no_grad():
        action_logits, value = policy.forward_eval(obs_tensor, state)

    lstm_h = state["lstm_h"]
    lstm_c = state["lstm_c"]

    if isinstance(action_logits, (list, tuple)):
        logits = action_logits[0] if len(action_logits) == 1 else action_logits
    else:
        logits = action_logits

    if isinstance(logits, (list, tuple)):
        accel_idx = int(torch.argmax(logits[0], dim=-1).item())
        steer_idx = int(torch.argmax(logits[1], dim=-1).item())
    else:
        flat_idx = int(torch.argmax(logits, dim=-1).item())
        accel_idx = flat_idx // NUM_STEER
        steer_idx = flat_idx % NUM_STEER

    return accel_idx, steer_idx, lstm_h, lstm_c


def _get_trajectory_points_ego_frame(ego_state, accel, steer, num_steps=80, dt=0.1):
    """Propagate trajectory and return points in ego-centric frame."""
    x = ego_state.center.x
    y = ego_state.center.y
    heading = ego_state.center.heading
    vx = ego_state.dynamic_car_state.center_velocity_2d.x
    vy = ego_state.dynamic_car_state.center_velocity_2d.y
    length = ego_state.car_footprint.length
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    traj = propagate_trajectory(x, y, heading, vx, vy, accel, steer, length,
                                num_steps=num_steps, dt=dt, constant_velocity_after=1)

    # Convert to ego frame
    points = []
    for t_x, t_y, t_h, t_vx, t_vy in traj:
        dx = t_x - x
        dy = t_y - y
        # Ego frame: rel_x forward, rel_y left (matches observation convention)
        rel_x = dx * cos_h + dy * sin_h
        rel_y = -dx * sin_h + dy * cos_h
        points.append((rel_y, rel_x))  # swap for plot coords (lateral, longitudinal)
    return points


def render_frame(ax, obs, step_idx, accel, steer, traj_points):
    """Render one frame of the BEV visualization."""
    ax.clear()
    parsed = parse_obs(obs)
    ego = parsed["ego"]

    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.2, color="white")
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")

    # Road segments
    type_colors = {0.0: "#4a7c59", 1.0: "#c4a35a", 2.0: "#8b4513"}
    for seg in parsed["road_segments"]:
        color = type_colors.get(seg["type"], "#666666")
        dx = seg["half_len"] * math.cos(seg["heading"])
        dy = seg["half_len"] * math.sin(seg["heading"])
        lw = 1.5 if seg["type"] == 2.0 else 1.0
        ax.plot([seg["y"] - dy, seg["y"] + dy], [seg["x"] - dx, seg["x"] + dx],
                color=color, linewidth=lw, alpha=0.7)

    # Partners
    for p in parsed["partners"]:
        cos_h = math.cos(p["heading"])
        sin_h = math.sin(p["heading"])
        hw, hl = p["width"] / 2, p["length"] / 2
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        rotated = [(cx * cos_h - cy * sin_h + p["rel_y"],
                     cx * sin_h + cy * cos_h + p["rel_x"]) for cx, cy in corners]
        polygon = plt.Polygon(rotated, closed=True, facecolor="#e74c3c",
                              edgecolor="white", alpha=0.7, linewidth=0.5)
        ax.add_patch(polygon)

    # Ego vehicle
    hw, hl = ego["width"] / 2, ego["length"] / 2
    ego_rect = plt.Rectangle((-hw, -hl), ego["width"], ego["length"],
                              facecolor="#3498db", edgecolor="white", linewidth=1.5, alpha=0.9)
    ax.add_patch(ego_rect)
    ax.plot(0, 0, "o", color="white", markersize=3)

    # Goal
    gx, gy = ego["goal_rel_y"], ego["goal_rel_x"]
    ax.plot(gx, gy, "*", color="#f1c40f", markersize=15, markeredgecolor="white",
            markeredgewidth=0.5, zorder=10)

    # Trajectory
    if traj_points:
        tx = [p[0] for p in traj_points]
        ty = [p[1] for p in traj_points]
        ax.plot(tx, ty, "-", color="#2ecc71", linewidth=2, alpha=0.8, zorder=8)
        # Mark first few steps more prominently
        ax.plot(tx[:10], ty[:10], "o", color="#2ecc71", markersize=3, alpha=0.9, zorder=9)

    # Auto-scale
    all_x = [0, gx] + [p["rel_y"] for p in parsed["partners"]]
    all_y = [0, gy] + [p["rel_x"] for p in parsed["partners"]]
    if parsed["road_segments"]:
        all_x += [s["y"] for s in parsed["road_segments"]]
        all_y += [s["x"] for s in parsed["road_segments"]]
    if traj_points:
        all_x += [p[0] for p in traj_points]
        all_y += [p[1] for p in traj_points]
    pad = 10
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # Info text
    accel_str = f"{accel:+.1f}" if accel is not None else "N/A"
    steer_str = f"{steer:.2f}" if steer is not None else "N/A"
    info = f"Step {step_idx}  |  accel={accel_str} m/s²  |  steer={steer_str}  |  speed={ego['speed']:.1f} m/s"
    ax.set_title(info, fontsize=11, color="white", fontweight="bold",
                 pad=8, backgroundcolor="#2d2d44")


def main():
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")
    weights_path = os.environ.get("PUFFERDRIVE_WEIGHTS_PATH",
        os.environ.get("PUFFERDRIVE_WEIGHTS_PATH", ""))

    # Find db files
    db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "mini", "*.db")))
    if not db_files:
        db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "trainval", "*.db")))[:1]
    print(f"Found {len(db_files)} db files")

    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root, map_root=map_root, sensor_root=data_root,
        db_files=db_files, map_version="nuplan-maps-v1.0",
    )

    scenarios = list(scenario_builder.get_scenarios(
        ScenarioFilter(
            scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
            num_scenarios_per_type=1, limit_total_scenarios=1,
            timestamp_threshold_s=None, ego_displacement_minimum_m=None,
            expand_scenarios=False, remove_invalid_goals=False, shuffle=False,
        ),
        Sequential(),
    ))

    if not scenarios:
        print("No scenarios found!")
        return

    scenario = scenarios[0]
    num_iterations = scenario.get_number_of_iterations()
    print(f"Scenario: {scenario.scenario_type} / {scenario.token} ({num_iterations} steps)")

    # Goal = last expert state
    last_state = scenario.get_ego_state_at_iteration(num_iterations - 1)
    goal = (last_state.center.x, last_state.center.y)

    # Build observation builder
    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())
    obs_builder = ObservationBuilder(map_api=map_api, route_roadblock_ids=route_ids, goal=goal)

    # Load policy
    print("Loading policy...")
    policy, device = _load_policy(weights_path)
    hidden_size = 256
    lstm_h = torch.zeros(1, hidden_size, device=device)
    lstm_c = torch.zeros(1, hidden_size, device=device)

    # Collect frames
    print("Running policy rollout...")
    frames = []  # list of (obs, accel, steer, traj_points)

    for i in range(num_iterations):
        ego_state = scenario.get_ego_state_at_iteration(i)
        detections = scenario.get_tracked_objects_at_iteration(i)

        obs = obs_builder.build(ego_state, detections.tracked_objects)

        accel_idx, steer_idx, lstm_h, lstm_c = _policy_step(
            policy, obs, lstm_h, lstm_c, device)

        accel = float(ACCEL_VALUES[accel_idx])
        steer = float(STEER_VALUES[steer_idx])

        traj_points = _get_trajectory_points_ego_frame(ego_state, accel, steer)

        frames.append((obs.copy(), accel, steer, traj_points, i))

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{num_iterations}")

    # Render video using cv2
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "nuplan_simulation.mp4")
    output_path = os.path.abspath(output_path)
    print(f"Rendering {len(frames)} frames to {output_path}...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    dpi = 100
    writer = None

    for frame_idx, (obs, accel, steer, traj_points, step_idx) in enumerate(frames):
        render_frame(ax, obs, step_idx, accel, steer, traj_points)
        fig.savefig("/tmp/_nuplan_frame.png", dpi=dpi, facecolor="#1a1a2e", bbox_inches="tight")

        img = cv2.imread("/tmp/_nuplan_frame.png")
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 5, (w, h))
        writer.write(img)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Rendered {frame_idx+1}/{len(frames)}")

    if writer is not None:
        writer.release()
    plt.close()
    os.remove("/tmp/_nuplan_frame.png")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
