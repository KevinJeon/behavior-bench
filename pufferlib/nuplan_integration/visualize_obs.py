# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Visualize PufferDrive observation features from a nuPlan scenario.

Creates a bird's-eye view plot showing ego, partners, and road segments
as the PPO policy sees them (ego-centric frame).

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.visualize_obs
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pufferlib.nuplan_integration.observation_builder import (
    EGO_FEATURES, PARTNER_FEATURES, ROAD_FEATURES,
    MAX_PARTNERS, MAX_ROAD_SEGMENTS, MAX_SPEED,
    MAX_VEH_WIDTH, MAX_VEH_LEN, MAX_ROAD_SEGMENT_LENGTH, MAX_ROAD_SCALE,
    OBS_SIZE,
)


def parse_obs(obs: np.ndarray) -> dict:
    """Parse a 1120-float observation vector into structured components."""
    assert obs.shape == (OBS_SIZE,), f"Expected ({OBS_SIZE},), got {obs.shape}"

    result = {
        "ego": {
            "goal_rel_x": obs[0] / 0.005,  # undo scaling
            "goal_rel_y": obs[1] / 0.005,
            "speed": obs[2] * MAX_SPEED,
            "width": obs[3] * MAX_VEH_WIDTH,
            "length": obs[4] * MAX_VEH_LEN,
            "collision": obs[5],
            "respawn": obs[6],
        },
        "partners": [],
        "road_segments": [],
    }

    # Partners
    idx = EGO_FEATURES
    for i in range(MAX_PARTNERS):
        rel_x = obs[idx] / 0.02
        rel_y = obs[idx + 1] / 0.02
        width = obs[idx + 2] * MAX_VEH_WIDTH
        length = obs[idx + 3] * MAX_VEH_LEN
        cos_h = obs[idx + 4]
        sin_h = obs[idx + 5]
        speed = obs[idx + 6] * MAX_SPEED

        # Skip zero-padded slots
        if abs(rel_x) > 0.01 or abs(rel_y) > 0.01 or abs(speed) > 0.01:
            result["partners"].append({
                "rel_x": rel_x, "rel_y": rel_y,
                "width": width, "length": length,
                "heading": math.atan2(sin_h, cos_h),
                "speed": speed,
            })
        idx += PARTNER_FEATURES

    # Road segments
    idx = EGO_FEATURES + MAX_PARTNERS * PARTNER_FEATURES
    for i in range(MAX_ROAD_SEGMENTS):
        x = obs[idx] / 0.02
        y = obs[idx + 1] / 0.02
        half_len = obs[idx + 2] * MAX_ROAD_SEGMENT_LENGTH
        width_val = obs[idx + 3] * MAX_ROAD_SCALE
        cos_a = obs[idx + 4]
        sin_a = obs[idx + 5]
        type_val = obs[idx + 6]

        if abs(x) > 0.01 or abs(y) > 0.01:
            result["road_segments"].append({
                "x": x, "y": y,
                "half_len": half_len,
                "heading": math.atan2(sin_a, cos_a),
                "type": type_val,  # 0=lane, 1=line, 2=edge
            })
        idx += ROAD_FEATURES

    return result


def plot_obs(obs: np.ndarray, title: str = "PufferDrive Observation (Ego Frame)",
             save_path: str = None):
    """Plot observation as bird's-eye view in ego-centric frame."""
    parsed = parse_obs(obs)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left: Bird's eye view ---
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_title("Bird's-Eye View (Ego-Centric Frame)")
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.2, color="white")

    # Road segments
    type_colors = {0.0: "#4a7c59", 1.0: "#c4a35a", 2.0: "#8b4513"}
    type_names = {0.0: "Lane", 1.0: "Centerline", 2.0: "Boundary"}
    for seg in parsed["road_segments"]:
        color = type_colors.get(seg["type"], "#666666")
        dx = seg["half_len"] * math.cos(seg["heading"])
        dy = seg["half_len"] * math.sin(seg["heading"])
        lw = 1.5 if seg["type"] == 2.0 else 1.0
        ax.plot(
            [seg["y"] - dy, seg["y"] + dy],
            [seg["x"] - dx, seg["x"] + dx],
            color=color, linewidth=lw, alpha=0.7,
        )

    # Partners
    for p in parsed["partners"]:
        cos_h = math.cos(p["heading"])
        sin_h = math.sin(p["heading"])
        # Draw as oriented rectangle
        hw, hl = p["width"] / 2, p["length"] / 2
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        rotated = []
        for cx, cy in corners:
            rx = cx * cos_h - cy * sin_h + p["rel_y"]
            ry = cx * sin_h + cy * cos_h + p["rel_x"]
            rotated.append((rx, ry))
        polygon = plt.Polygon(rotated, closed=True, facecolor="#e74c3c",
                              edgecolor="white", alpha=0.7, linewidth=0.5)
        ax.add_patch(polygon)
        # Speed arrow
        arrow_len = min(abs(p["speed"]) * 0.3, 5.0)
        ax.annotate("", xy=(p["rel_y"] + arrow_len * sin_h, p["rel_x"] + arrow_len * cos_h),
                     xytext=(p["rel_y"], p["rel_x"]),
                     arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5))

    # Ego vehicle (at origin, facing up)
    ego = parsed["ego"]
    hw, hl = ego["width"] / 2, ego["length"] / 2
    ego_rect = plt.Rectangle((-hw, -hl), ego["width"], ego["length"],
                              facecolor="#3498db", edgecolor="white",
                              linewidth=1.5, alpha=0.9)
    ax.add_patch(ego_rect)
    ax.plot(0, 0, "o", color="white", markersize=3)

    # Goal point
    gx, gy = ego["goal_rel_y"], ego["goal_rel_x"]  # swap for plot coords
    ax.plot(gx, gy, "*", color="#f1c40f", markersize=15, markeredgecolor="white",
            markeredgewidth=0.5, zorder=10)
    ax.annotate("GOAL", (gx, gy), textcoords="offset points", xytext=(8, 8),
                color="#f1c40f", fontsize=9, fontweight="bold")

    # Auto-scale with padding
    all_x = [0, gx] + [p["rel_y"] for p in parsed["partners"]]
    all_y = [0, gy] + [p["rel_x"] for p in parsed["partners"]]
    if parsed["road_segments"]:
        all_x += [s["y"] for s in parsed["road_segments"]]
        all_y += [s["x"] for s in parsed["road_segments"]]
    pad = 10
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # Legend
    legend_items = [
        mpatches.Patch(color="#3498db", label=f"Ego (speed={ego['speed']:.1f} m/s)"),
        mpatches.Patch(color="#e74c3c", label=f"Partners ({len(parsed['partners'])})"),
        mpatches.Patch(color="#f1c40f", label="Goal"),
    ]
    for tv, tc in type_colors.items():
        name = type_names.get(tv, f"Type {tv}")
        count = sum(1 for s in parsed["road_segments"] if s["type"] == tv)
        if count > 0:
            legend_items.append(mpatches.Patch(color=tc, label=f"{name} ({count} segs)"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor="#2d2d44", edgecolor="white", labelcolor="white")

    # --- Right: Raw feature heatmap ---
    ax2 = axes[1]
    ax2.set_title("Raw Observation Vector (1120 floats)")

    # Reshape for visualization: show structure
    ego_feat = obs[:EGO_FEATURES]
    partner_feat = obs[EGO_FEATURES:EGO_FEATURES + MAX_PARTNERS * PARTNER_FEATURES].reshape(MAX_PARTNERS, PARTNER_FEATURES)
    road_feat = obs[EGO_FEATURES + MAX_PARTNERS * PARTNER_FEATURES:].reshape(MAX_ROAD_SEGMENTS, ROAD_FEATURES)

    # Stack: ego (1x7), partners (31x7), roads (128x7)
    combined = np.vstack([ego_feat.reshape(1, -1), partner_feat, road_feat])
    im = ax2.imshow(combined, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                    interpolation="nearest")
    plt.colorbar(im, ax=ax2, shrink=0.8, label="Feature value")

    # Labels
    ax2.set_xlabel("Feature index (0-6)")
    ax2.set_ylabel("Entity index")
    ax2.axhline(0.5, color="yellow", linewidth=1.5, linestyle="--")
    ax2.axhline(MAX_PARTNERS + 0.5, color="yellow", linewidth=1.5, linestyle="--")
    ax2.text(7.2, 0, "Ego", fontsize=8, va="center", color="yellow")
    ax2.text(7.2, 16, "Partners", fontsize=8, va="center", color="yellow")
    ax2.text(7.2, MAX_PARTNERS + 64, "Roads", fontsize=8, va="center", color="yellow")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.savefig("nuplan_obs_visualization.png",
                     dpi=150, bbox_inches="tight")
        print("Saved to nuplan_obs_visualization.png")
    plt.close()


def main():
    """Load a nuPlan scenario and visualize the observation."""
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.abstract_map import SemanticMapLayer
    from nuplan.database.nuplan_db_orm.nuplandb import NuPlanDB
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils import discover_log_dbs
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    from pufferlib.nuplan_integration.observation_builder import ObservationBuilder

    import glob

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    # Find mini split db files
    db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "mini", "*.db")))
    if not db_files:
        db_files = sorted(glob.glob(os.path.join(data_root, "nuplan-v1.1", "trainval", "*.db")))[:1]
    print(f"Found {len(db_files)} db files")

    print("Building scenario builder...")
    scenario_builder = NuPlanScenarioBuilder(
        data_root=data_root,
        map_root=map_root,
        sensor_root=data_root,
        db_files=db_files,
        map_version="nuplan-maps-v1.0",
    )

    print("Discovering scenarios...")
    # Get first available scenario
    scenarios = scenario_builder.get_scenarios(
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
    )

    scenarios = list(scenarios)
    if not scenarios:
        print("No scenarios found!")
        return

    scenario = scenarios[0]
    print(f"Using scenario: {scenario.scenario_type} / {scenario.token}")

    # Get initial state
    ego_state = scenario.initial_ego_state
    detections_tracks = scenario.initial_tracked_objects
    detections = detections_tracks.tracked_objects
    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())

    # Goal = last point of expert trajectory
    last_idx = scenario.get_number_of_iterations() - 1
    last_state = scenario.get_ego_state_at_iteration(last_idx)
    goal = (last_state.center.x, last_state.center.y)

    print(f"Ego position: ({ego_state.center.x:.1f}, {ego_state.center.y:.1f})")
    print(f"Goal (expert traj end): ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"Route roadblocks: {len(route_ids)}")

    # Build observation
    obs_builder = ObservationBuilder(
        map_api=map_api,
        route_roadblock_ids=route_ids,
        goal=goal,
    )
    obs = obs_builder.build(ego_state, detections)

    print(f"Observation shape: {obs.shape}")
    print(f"Non-zero elements: {np.count_nonzero(obs)} / {len(obs)}")

    # Parse and print summary
    parsed = parse_obs(obs)
    ego = parsed["ego"]
    print(f"\nEgo: speed={ego['speed']:.1f} m/s, size={ego['width']:.1f}x{ego['length']:.1f}m")
    print(f"Goal: ({ego['goal_rel_x']:.1f}, {ego['goal_rel_y']:.1f}) m relative")
    print(f"Partners: {len(parsed['partners'])}")
    print(f"Road segments: {len(parsed['road_segments'])}")

    type_counts = {}
    for s in parsed["road_segments"]:
        t = s["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  Lane: {type_counts.get(0.0, 0)}, Centerline: {type_counts.get(1.0, 0)}, Boundary: {type_counts.get(2.0, 0)}")

    # Plot
    title = f"PufferDrive Obs - {scenario.scenario_type} (t=0)"
    plot_obs(obs, title=title)


if __name__ == "__main__":
    main()
