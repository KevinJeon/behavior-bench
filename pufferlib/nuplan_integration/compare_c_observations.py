# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Compare C-binding observations: our nuplan_to_binary vs ScenarioMax pipeline.

Creates side-by-side bird's-eye plots and computes numerical differences.

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.compare_c_observations
"""

import glob
import math
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# ScenarioMax
sys.path.insert(0, os.environ.get("SCENARIOMAX_ROOT", ""))
from scenariomax.raw_to_unified.datasets.nuplan.extractor import convert_nuplan_scenario
from scenariomax.unified_to_gpudrive.converter.roadgraph import convert_map_features

from pufferlib.ocean.drive.drive import save_map_binary
from pufferlib.ocean.drive import binding
from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary
from pufferlib.nuplan_integration.visualize_obs import parse_obs

_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


def load_scenario():
    """Load scenario via ScenarioMax loader (handles subsampling to 0.1s)."""
    from scenariomax.raw_to_unified.datasets.nuplan.load import get_nuplan_scenarios

    data_root = os.environ.get("NUPLAN_DATA_ROOT", "")
    map_root = os.environ.get("NUPLAN_MAPS_ROOT", "")

    # Find mini split
    data_dir = os.path.join(data_root, "nuplan-v1.1", "splits", "mini")
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(data_root, "nuplan-v1.1", "splits", "val14")

    scenarios = get_nuplan_scenarios(data_dir, map_root, num_files=1)
    assert scenarios, "No scenarios found!"
    return scenarios[0]


def scenariomax_to_bin(scenario, output_path):
    """Convert scenario via ScenarioMax pipeline to .bin.

    Uses ScenarioMax extractor + GPUDrive roadgraph converter, then builds
    the map_data dict for save_map_binary, skipping trimesh/collision.
    """
    from scenariomax.unified_to_gpudrive import utils as gpudrive_utils
    from scenariomax.unified_to_gpudrive.converter.state import _extract_obj

    unified = convert_nuplan_scenario(scenario, version="v1.1")
    roads, _ = convert_map_features(unified["static_map_elements"])
    if roads is None:
        print("  WARNING: convert_map_features returned None (3D structure)")
        roads = []
    print(f"  ScenarioMax roads extracted: {len(roads)}")

    # Extract objects from unified dynamic_agents (same as state.convert_track_features_to_objects but without trimesh)
    objects = []
    for idx, (obj_id, agent_data) in enumerate(unified["dynamic_agents"].items()):
        obj = _extract_obj(idx, obj_id, agent_data)
        objects.append(obj)

    # Find SDC index
    sdc_index = 0
    for i, obj in enumerate(objects):
        if obj.get("is_sdc", False):
            sdc_index = i
            break

    map_data = {
        "objects": objects,
        "roads": roads if roads else [],
        "metadata": {
            "sdc_track_index": sdc_index,
            "tracks_to_predict": [],
        },
    }
    save_map_binary(map_data, output_path, unique_map_id=0)
    return output_path


def compute_obs_from_bin(bin_path, ego_state, tracked_objects, goal_x, goal_y, label=""):
    """Load a .bin, init C env, compute observations."""
    obs_buf = np.zeros(1120, dtype=np.float32)
    env_handle = binding.init_obs_env(bin_path, obs_buf)

    bin_size = os.path.getsize(bin_path)
    print(f"  [{label}] Binary: {bin_size} bytes")

    # Build agents array
    agents_list = tracked_objects.get_tracked_objects_of_types(_AGENT_TYPES)
    if agents_list:
        rows = [[a.box.center.x, a.box.center.y, a.box.center.heading,
                 a.velocity.x, a.velocity.y, a.box.width, a.box.length]
                for a in agents_list]
        agents = np.array(rows, dtype=np.float32)
    else:
        agents = np.zeros((0, 7), dtype=np.float32)

    print(f"  [{label}] Ego: ({ego_state.center.x:.1f}, {ego_state.center.y:.1f}), agents: {len(agents_list) if agents_list else 0}")

    binding.compute_obs_external(
        env_handle,
        ego_state.center.x, ego_state.center.y, ego_state.center.heading,
        ego_state.dynamic_car_state.center_velocity_2d.x,
        ego_state.dynamic_car_state.center_velocity_2d.y,
        ego_state.car_footprint.width, ego_state.car_footprint.length,
        goal_x, goal_y, agents,
    )

    nonzero = np.count_nonzero(obs_buf)
    print(f"  [{label}] Obs nonzero: {nonzero}/1120")
    return obs_buf.copy()


def plot_bev(ax, parsed, title):
    """Plot bird's-eye view on given axes."""
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Lateral (m)")
    ax.set_ylabel("Longitudinal (m)")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.2, color="white")

    type_colors = {0.0: "#4a7c59", 1.0: "#c4a35a", 2.0: "#8b4513"}
    type_names = {0.0: "Lane", 1.0: "Line", 2.0: "Edge"}

    for seg in parsed["road_segments"]:
        color = type_colors.get(seg["type"], "#666666")
        dx = seg["half_len"] * math.cos(seg["heading"])
        dy = seg["half_len"] * math.sin(seg["heading"])
        lw = 1.5 if seg["type"] == 2.0 else 1.0
        ax.plot([seg["y"] - dy, seg["y"] + dy],
                [seg["x"] - dx, seg["x"] + dx],
                color=color, linewidth=lw, alpha=0.7)

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

    ego = parsed["ego"]
    hw, hl = ego["width"] / 2, ego["length"] / 2
    ego_rect = plt.Rectangle((-hw, -hl), ego["width"], ego["length"],
                              facecolor="#3498db", edgecolor="white",
                              linewidth=1.5, alpha=0.9)
    ax.add_patch(ego_rect)
    ax.plot(0, 0, "o", color="white", markersize=3)

    gx, gy = ego["goal_rel_y"], ego["goal_rel_x"]
    ax.plot(gx, gy, "*", color="#f1c40f", markersize=15, markeredgecolor="white",
            markeredgewidth=0.5, zorder=10)

    # Auto-scale
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
        mpatches.Patch(color="#3498db", label=f"Ego (v={ego['speed']:.1f} m/s)"),
        mpatches.Patch(color="#e74c3c", label=f"Partners ({len(parsed['partners'])})"),
    ]
    for tv, tc in type_colors.items():
        name = type_names.get(tv, f"Type {tv}")
        count = sum(1 for s in parsed["road_segments"] if s["type"] == tv)
        if count > 0:
            legend_items.append(mpatches.Patch(color=tc, label=f"{name} ({count})"))
    ax.legend(handles=legend_items, loc="upper left", fontsize=7,
              facecolor="#2d2d44", edgecolor="white", labelcolor="white")


def main():
    print("=== Loading scenario ===")
    scenario = load_scenario()
    ego_state = scenario.initial_ego_state
    detections = scenario.initial_tracked_objects.tracked_objects
    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())
    last_idx = scenario.get_number_of_iterations() - 1
    last_state = scenario.get_ego_state_at_iteration(last_idx)
    goal_x, goal_y = last_state.center.x, last_state.center.y
    print(f"  Scenario: {scenario.scenario_type} / {scenario.token}")
    print(f"  Ego: ({ego_state.center.x:.1f}, {ego_state.center.y:.1f})")

    # --- Our pipeline ---
    print("\n=== Our pipeline (nuplan_to_binary) ===")
    our_bin = tempfile.mktemp(suffix="_ours.bin")
    nuplan_to_binary(scenario, map_api, route_ids, our_bin)
    our_obs = compute_obs_from_bin(our_bin, ego_state, detections, goal_x, goal_y, label="Ours")
    our_parsed = parse_obs(our_obs)
    print(f"  Partners: {len(our_parsed['partners'])}")
    print(f"  Road segments: {len(our_parsed['road_segments'])}")

    # --- ScenarioMax pipeline ---
    print("\n=== ScenarioMax pipeline ===")
    smax_bin = tempfile.mktemp(suffix="_smax.bin")
    scenariomax_to_bin(scenario, smax_bin)
    # ScenarioMax centers all coords at ego initial position, so we pass ego-relative coords
    # to compute_obs_external. The .bin has (0,0)-centered data.
    # We need a mock ego_state at (0,0) — but we can't mock easily.
    # Instead, just load the smax .bin and compute obs without external state updates
    # by using the positions from the .bin itself (which set_start_position already did).
    # For a fair comparison, we skip external state update for ScenarioMax and just
    # call compute_observations directly after init.
    smax_obs_buf = np.zeros(1120, dtype=np.float32)
    smax_handle = binding.init_obs_env(smax_bin, smax_obs_buf)
    # For ScenarioMax, the ego is already at traj_x[0] (near 0,0 after centering + set_means)
    # Just call compute_observations via compute_obs_external with the position from the .bin
    # The ego in the .bin is centered — its traj_x[0]/traj_y[0] after set_means is near 0
    # Pass (0,0) as ego position since data is ego-centered, world_mean is also ~0
    binding.compute_obs_external(
        smax_handle,
        0.0, 0.0,  # ego at origin (ScenarioMax centering)
        ego_state.center.heading,
        ego_state.dynamic_car_state.center_velocity_2d.x,
        ego_state.dynamic_car_state.center_velocity_2d.y,
        ego_state.car_footprint.width, ego_state.car_footprint.length,
        goal_x - ego_state.center.x, goal_y - ego_state.center.y,  # goal also relative
        None,  # agents already in .bin
    )
    smax_obs = smax_obs_buf.copy()
    smax_bin_size = os.path.getsize(smax_bin)
    print(f"  [SMax] Binary: {smax_bin_size} bytes, obs nonzero: {np.count_nonzero(smax_obs)}/1120")
    smax_parsed = parse_obs(smax_obs)
    print(f"  Partners: {len(smax_parsed['partners'])}")
    print(f"  Road segments: {len(smax_parsed['road_segments'])}")

    # --- Numerical comparison ---
    print("\n=== Numerical comparison ===")
    diff = np.abs(our_obs - smax_obs)
    print(f"  Max absolute diff: {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Matching elements (diff < 0.01): {np.sum(diff < 0.01)} / {len(diff)}")

    # Per-section comparison
    ego_diff = diff[:7]
    partner_diff = diff[7:7 + 31 * 7]
    road_diff = diff[7 + 31 * 7:]
    print(f"  Ego features   - max diff: {ego_diff.max():.6f}, mean: {ego_diff.mean():.6f}")
    print(f"  Partner features - max diff: {partner_diff.max():.6f}, mean: {partner_diff.mean():.6f}")
    print(f"  Road features   - max diff: {road_diff.max():.6f}, mean: {road_diff.mean():.6f}")

    # Road type counts
    for label, parsed in [("Ours", our_parsed), ("ScenarioMax", smax_parsed)]:
        type_counts = {}
        for s in parsed["road_segments"]:
            t = s["type"]
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  {label} road types: LANE={type_counts.get(0.0, 0)}, "
              f"LINE={type_counts.get(1.0, 0)}, EDGE={type_counts.get(2.0, 0)}")

    # --- Plot ---
    print("\n=== Generating comparison plot ===")
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    plot_bev(axes[0], our_parsed, "Ours (nuplan_to_binary)")
    plot_bev(axes[1], smax_parsed, "ScenarioMax")

    # Diff heatmap
    ax3 = axes[2]
    ax3.set_title("Absolute Difference")
    combined_ours = np.vstack([
        our_obs[:7].reshape(1, 7),
        our_obs[7:7 + 31 * 7].reshape(31, 7),
        our_obs[7 + 31 * 7:].reshape(128, 7),
    ])
    combined_smax = np.vstack([
        smax_obs[:7].reshape(1, 7),
        smax_obs[7:7 + 31 * 7].reshape(31, 7),
        smax_obs[7 + 31 * 7:].reshape(128, 7),
    ])
    diff_map = np.abs(combined_ours - combined_smax)
    im = ax3.imshow(diff_map, aspect="auto", cmap="hot", vmin=0, vmax=0.5,
                    interpolation="nearest")
    plt.colorbar(im, ax=ax3, shrink=0.8, label="Abs diff")
    ax3.set_xlabel("Feature index (0-6)")
    ax3.set_ylabel("Entity index")
    ax3.axhline(0.5, color="cyan", linewidth=1, linestyle="--")
    ax3.axhline(31.5, color="cyan", linewidth=1, linestyle="--")
    ax3.text(7.2, 0, "Ego", fontsize=8, va="center", color="cyan")
    ax3.text(7.2, 16, "Partners", fontsize=8, va="center", color="cyan")
    ax3.text(7.2, 96, "Roads", fontsize=8, va="center", color="cyan")

    fig.suptitle(f"C-Binding Observation Comparison: Ours vs ScenarioMax\n"
                 f"Max diff={diff.max():.4f}, Match rate={np.sum(diff < 0.01)/len(diff)*100:.1f}%",
                 fontsize=13)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scenario_videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "c_obs_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Cleanup
    os.unlink(our_bin)
    os.unlink(smax_bin)
    print("\nDone!")


if __name__ == "__main__":
    main()
