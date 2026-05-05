# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Compare road features: ScenarioMax pipeline vs. our observation_builder.

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.compare_road_features
"""

import json
import math
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

# ── ScenarioMax imports (need ScenarioMax venv or sys.path) ──
sys.path.insert(0, os.environ.get("SCENARIOMAX_ROOT", ""))
from scenariomax.raw_to_unified.datasets.nuplan.extractor import convert_nuplan_scenario
from scenariomax.raw_to_unified.datasets.nuplan.load import get_nuplan_scenarios
from scenariomax.unified_to_gpudrive.converter.roadgraph import convert_map_features, TYPE_MAPPING

# ── PufferDrive imports ──
from pufferlib.nuplan_integration.observation_builder import (
    ROAD_TYPE_EDGE,
    ROAD_TYPE_LANE,
    ROAD_TYPE_LINE,
    ObservationBuilder,
    _simplify_nuplan_polyline,
    _polyline_to_segments,
)
from pufferlib.ocean.drive.drive import simplify_polyline


# Drive.h type mapping (same as save_map_binary)
def map_element_id_to_drive_type(map_element_id):
    """Maps GPUDrive map_element_id to drive.h road type (4=LANE, 5=LINE, 6=EDGE)."""
    if 0 <= map_element_id <= 3:
        return 4  # ROAD_LANE
    elif 5 <= map_element_id <= 13:
        return 5  # ROAD_LINE
    elif 14 <= map_element_id <= 16:
        return 6  # ROAD_EDGE
    return -1  # filtered


def road_type_to_obs_val(drive_type):
    """Convert drive.h type to observation value (entity->type - 4.0f)."""
    return float(drive_type - 4)


def extract_scenariomax_segments(gpudrive_json, ego_x, ego_y, radius=100.0):
    """Extract road segments from GPUDrive JSON, applying same processing as save_map_binary + drive.h."""
    segments = []
    max_dist_sq = radius * radius

    for road in gpudrive_json.get("roads", []):
        map_element_id = road.get("map_element_id", 0)
        road_type_word = road.get("type", "")

        # Apply same overrides as save_map_binary
        if road_type_word == "lane":
            map_element_id = 2
        elif road_type_word == "road_edge":
            map_element_id = 15

        drive_type = map_element_id_to_drive_type(map_element_id)
        if drive_type < 4:
            continue  # skip non-road elements

        obs_type = road_type_to_obs_val(drive_type)

        geometry = road.get("geometry", [])
        # Apply same simplification as save_map_binary
        if len(geometry) > 10 and map_element_id <= 16:
            geometry = simplify_polyline(geometry, 0.1, 250)

        # Convert to segments (same as _polyline_to_segments)
        for j in range(len(geometry) - 1):
            sx, sy = geometry[j]["x"], geometry[j]["y"]
            ex, ey = geometry[j + 1]["x"], geometry[j + 1]["y"]
            mid_x = (sx + ex) * 0.5
            mid_y = (sy + ey) * 0.5
            dx = mid_x - ego_x
            dy = mid_y - ego_y
            dist_sq = dx * dx + dy * dy
            if dist_sq > max_dist_sq:
                continue
            seg_dx = ex - mid_x
            seg_dy = ey - mid_y
            half_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
            segments.append((mid_x, mid_y, half_len, obs_type))

    return segments


def extract_ours_segments(scenario, ego_x, ego_y):
    """Extract road segments using our observation_builder pipeline."""
    map_api = scenario.map_api
    route_ids = list(scenario.get_route_roadblock_ids())
    last_state = scenario.get_ego_state_at_iteration(scenario.get_number_of_iterations() - 1)
    goal = (last_state.center.x, last_state.center.y)

    obs_builder = ObservationBuilder(map_api, route_ids, goal)
    raw_segments = obs_builder._extract_road_segments(ego_x, ego_y)

    segments = []
    for dist_sq, mid_x, mid_y, half_len, dx_n, dy_n, type_val in raw_segments:
        segments.append((mid_x, mid_y, half_len, type_val))

    return segments


def count_types(segments):
    """Count segments per road type."""
    counts = {0.0: 0, 1.0: 0, 2.0: 0}
    for _, _, _, t in segments:
        if t in counts:
            counts[t] += 1
    return counts


def plot_comparison(smax_segs, ours_segs, ego_x, ego_y, output_path):
    """Side-by-side scatter plot of road segments colored by type."""
    type_colors = {0.0: "#3498db", 1.0: "#e67e22", 2.0: "#e74c3c"}
    type_labels = {0.0: "LANE (0)", 1.0: "LINE (1)", 2.0: "EDGE (2)"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Road Segment Comparison: ScenarioMax vs. Ours", fontsize=14)

    for ax, segs, title in [
        (axes[0], smax_segs, "ScenarioMax Pipeline"),
        (axes[1], ours_segs, "Our ObservationBuilder"),
    ]:
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")

        for t_val in [0.0, 1.0, 2.0]:
            xs = [s[0] for s in segs if s[3] == t_val]
            ys = [s[1] for s in segs if s[3] == t_val]
            if xs:
                ax.scatter(xs, ys, c=type_colors[t_val], s=3, alpha=0.6,
                           label=f"{type_labels[t_val]} ({len(xs)})")

        # Mark ego position
        ax.plot(ego_x, ego_y, "k*", markersize=15, zorder=10, label="Ego")
        ax.legend(fontsize=9, loc="upper right")

        # Set limits around ego
        ax.set_xlim(ego_x - 110, ego_x + 110)
        ax.set_ylim(ego_y - 110, ego_y + 110)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


def main():
    maps_path = os.environ.get("NUPLAN_MAPS_ROOT", "")
    data_path = os.environ.get("NUPLAN_DATA_ROOT", "")

    # Find the mini split or any available split
    splits = ["nuplan-v1.1/splits/mini", "nuplan-v1.1/splits/val14"]
    data_dir = None
    for split in splits:
        candidate = os.path.join(data_path, split)
        if os.path.isdir(candidate):
            data_dir = candidate
            break

    if data_dir is None:
        # Try to find any directory with .db files
        for root, dirs, files in os.walk(data_path):
            if any(f.endswith(".db") for f in files):
                data_dir = root
                break

    if data_dir is None:
        print(f"ERROR: No .db files found under {data_path}")
        return

    print(f"Using data dir: {data_dir}")
    print(f"Using maps dir: {maps_path}")

    # Step 1: Load one scenario
    print("\n=== Step 1: Loading nuPlan scenario ===")
    scenarios = get_nuplan_scenarios(data_dir, maps_path, num_files=1)
    scenario = scenarios[0]
    print(f"  Scenario: {scenario.scenario_name}")
    print(f"  Map: {scenario.map_api.map_name}")
    print(f"  Iterations: {scenario.get_number_of_iterations()}")

    ego_state = scenario.initial_ego_state
    ego_x = ego_state.center.x
    ego_y = ego_state.center.y
    print(f"  Ego position: ({ego_x:.1f}, {ego_y:.1f})")

    # Step 2: ScenarioMax pipeline
    print("\n=== Step 2: ScenarioMax → GPUDrive roads ===")
    unified = convert_nuplan_scenario(scenario, version="v1.1")

    # Convert map features directly (skip collision/trimesh)
    roads, _ = convert_map_features(unified["static_map_elements"])
    if roads is None:
        print("  ERROR: Road conversion returned None (3D structure)")
        return

    gpudrive_json = {"roads": roads}
    print(f"  Roads extracted: {len(roads)}")

    # Save JSON for inspection
    json_path = "/tmp/scenariomax_test.json"
    with open(json_path, "w") as f:
        json.dump(gpudrive_json, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")

    # ScenarioMax centers coordinates at initial ego position, so ego is at (0,0)
    # We need to offset ScenarioMax segments back to absolute coords for comparison
    smax_segs_raw = extract_scenariomax_segments(gpudrive_json, 0.0, 0.0)
    # Shift to absolute coordinates
    smax_segs = [(mx + ego_x, my + ego_y, hl, t) for mx, my, hl, t in smax_segs_raw]
    smax_counts = count_types(smax_segs)
    print(f"  Segments within 100m: {len(smax_segs)}")
    print(f"  Type distribution: LANE={smax_counts[0.0]}, LINE={smax_counts[1.0]}, EDGE={smax_counts[2.0]}")

    # Step 3: Our pipeline
    print("\n=== Step 3: Our ObservationBuilder ===")
    ours_segs = extract_ours_segments(scenario, ego_x, ego_y)
    ours_counts = count_types(ours_segs)
    print(f"  Segments within 100m: {len(ours_segs)}")
    print(f"  Type distribution: LANE={ours_counts[0.0]}, LINE={ours_counts[1.0]}, EDGE={ours_counts[2.0]}")

    # Step 4: Compare
    print("\n=== Step 4: Comparison ===")
    print(f"  {'Metric':<30s} {'ScenarioMax':>12s} {'Ours':>12s} {'Match':>8s}")
    print(f"  {'-'*62}")
    total_smax = len(smax_segs)
    total_ours = len(ours_segs)
    print(f"  {'Total segments':<30s} {total_smax:>12d} {total_ours:>12d} {'~' if abs(total_smax - total_ours) < total_smax * 0.2 else '✗':>8s}")
    for t_val, t_name in [(0.0, "LANE (type 0)"), (1.0, "LINE (type 1)"), (2.0, "EDGE (type 2)")]:
        s = smax_counts[t_val]
        o = ours_counts[t_val]
        match = "✓" if abs(s - o) < max(s, 1) * 0.2 else "✗"
        print(f"  {t_name:<30s} {s:>12d} {o:>12d} {match:>8s}")

    # Step 5: Plot
    print("\n=== Step 5: Generating comparison plot ===")
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "scenario_videos", "road_features_comparison.png")
    out_path = os.path.abspath(out_path)
    plot_comparison(smax_segs, ours_segs, ego_x, ego_y, out_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
