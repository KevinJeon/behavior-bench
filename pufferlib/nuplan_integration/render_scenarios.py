# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Render all nuPlan simulation scenarios as MP4 videos.

Uses the ScenarioVisualizer for proper map rendering with:
- Lane polygons, intersections, crosswalks, stop lines
- Agent boxes with heading arrows (vehicles, pedestrians, bicycles)
- Ego vehicle with heading arrow
- Ego trajectory with time-gradient alpha
- Expert trajectory (green dashed)

Usage:
    python -m pufferlib.nuplan_integration.render_scenarios \
        /path/to/simulation_log_dir
"""

import glob
import lzma
import math
import os
import pickle
import sys
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import msgpack
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.utils.serialization.to_scene import tracked_object_types

simulation_tile_agent_style: Dict[str, Any] = {
    "ego": {"fill_color": "#DF7162", "fill_alpha": 1.0, "line_color": "#000000", "line_width": 1},
    "vehicles": {"fill_color": "#6A9BDD", "fill_alpha": 1.0, "line_color": "#000000", "line_width": 0.5},
    "pedestrians": {"fill_color": "#D5E8D4", "fill_alpha": 1.0, "line_color": "#000000", "line_width": 1},
    "bicycles": {"fill_color": "#FF4D4D", "fill_alpha": 0.5, "line_color": "#FF4D4D", "line_width": 1},
    "genericobjects": {"fill_color": "#AE4DFF", "fill_alpha": 0.5, "line_color": "#AE4DFF", "line_width": 1},
    "traffic_cone": {"fill_color": "#ed2a2a", "fill_alpha": 1.0, "line_color": "#ed2a2a", "line_width": 1},
    "barrier": {"fill_color": "#FFFFFF", "fill_alpha": 0.5, "line_color": "#FFFFFF", "line_width": 1},
    "czone_sign": {"fill_color": "#00f6ff", "fill_alpha": 0.5, "line_color": "#00f6ff", "line_width": 1},
}

simulation_map_layer_color = {
    SemanticMapLayer.LANE: {"fill_color": "#e8f0f8", "fill_color_alpha": 1, "line_color": "#e8f0f8"},
    SemanticMapLayer.WALKWAYS: {"fill_color": "#FFFFFF", "fill_color_alpha": 0.5, "line_color": "#ffffff"},
    SemanticMapLayer.CARPARK_AREA: {"fill_color": "#ffffff", "fill_color_alpha": 0.5, "line_color": "#ffffff"},
    SemanticMapLayer.PUDO: {"fill_color": "#AF75A7", "fill_color_alpha": 0.3, "line_color": "#AF75A7"},
    SemanticMapLayer.INTERSECTION: {"fill_color": "#e8f0f8", "fill_color_alpha": 1.0, "line_color": "#e8f0f8"},
    SemanticMapLayer.STOP_LINE: {"fill_color": "#dedede", "fill_color_alpha": 0.4, "line_color": "#c0c0c0"},
    SemanticMapLayer.CROSSWALK: {"fill_color": "#dedede", "fill_color_alpha": 0.4, "line_color": "#c0c0c0"},
    SemanticMapLayer.ROADBLOCK: {"fill_color": "#e8f0f8", "fill_color_alpha": 0.2, "line_color": "#d1dbe2"},
    SemanticMapLayer.BASELINE_PATHS: {"line_color": "#bbbbbb", "line_color_alpha": 1.0},
    SemanticMapLayer.LANE_CONNECTOR: {"line_color": "#bbbbbb", "line_color_alpha": 1.0},
}

EGO_COLOR = "#DF7162"
AGENT_COLOR = "#6A9BDD"
EXPERT_COLOR = "#2ecc71"


class ScenarioVisualizer:
    def __init__(self, map_api, scenario, boundary_x=(50, 50), boundary_y=(50, 50), radius=150):
        self._map_api = map_api
        self.scenario = scenario
        self._boundary_x = boundary_x
        self._boundary_y = boundary_y
        self._radius = radius
        self._layer_names = [
            SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.LANE,
            SemanticMapLayer.CROSSWALK, SemanticMapLayer.INTERSECTION,
            SemanticMapLayer.STOP_LINE, SemanticMapLayer.WALKWAYS,
            SemanticMapLayer.CARPARK_AREA,
        ]

    def cache_map(self, center):
        """Pre-load map data once for the whole scenario."""
        self._nearest_vector_map = self._map_api.get_proximal_map_objects(
            center, self._radius, self._layer_names)
        self._load_map_data()
        # Pre-extract polygon coords and line coords to avoid re-querying
        self._cached_polygons = []
        layers = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
            (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
            (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
            (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]),
            (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]),
            (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA]),
        ]
        if self.scenario.get_route_roadblock_ids():
            layers.append((SemanticMapLayer.ROADBLOCK, simulation_map_layer_color[SemanticMapLayer.ROADBLOCK]))
        for layer_name, color in layers:
            for obj in self._nearest_vector_map.get(layer_name, []):
                xs, ys = zip(*obj.polygon.exterior.coords)
                self._cached_polygons.append((xs, ys, color))
        self._cached_lines = []
        for layer_name, color in [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
            (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]),
        ]:
            for obj in self._nearest_vector_map.get(layer_name, []):
                path = obj.baseline_path.discrete_path
                lx = [p.x for p in path]
                ly = [p.y for p in path]
                self._cached_lines.append((lx, ly, color))

    def render_frame(self, sample, ax, ego_trail_xs=None, ego_trail_ys=None,
                     expert_xs=None, expert_ys=None, expert_step=None,
                     step=0, num_steps=0, scenario_type=""):
        center = sample.ego_state.center

        ax.set_xlim(center.x - self._boundary_x[0], center.x + self._boundary_x[1])
        ax.set_ylim(center.y - self._boundary_y[0], center.y + self._boundary_y[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        ax.set_facecolor("white")

        # Use cached map data (from cache_map) for speed
        for xs, ys, color in self._cached_polygons:
            ax.fill(xs, ys, color=color["fill_color"], alpha=color["fill_color_alpha"],
                    edgecolor=color["line_color"])
        for lx, ly, color in self._cached_lines:
            ax.plot(lx, ly, color=color["line_color"], linestyle='dashed',
                    alpha=color["line_color_alpha"], linewidth=0.8)

        # Expert trajectory (full, dashed)
        if expert_xs is not None and expert_ys is not None:
            ax.plot(expert_xs, expert_ys, "--", color=EXPERT_COLOR,
                    linewidth=2, alpha=0.5, zorder=4)
            if expert_step is not None and expert_step < len(expert_xs):
                ax.plot(expert_xs[expert_step], expert_ys[expert_step], "o",
                        color=EXPERT_COLOR, markersize=6, markeredgecolor="white",
                        markeredgewidth=1, zorder=12)

        # Ego trail with alpha gradient
        if ego_trail_xs is not None and ego_trail_ys is not None and len(ego_trail_xs) > 1:
            for i in range(len(ego_trail_xs) - 1):
                frac = i / max(len(ego_trail_xs) - 1, 1)
                alpha = 0.3 + 0.7 * frac
                ax.plot(ego_trail_xs[i:i+2], ego_trail_ys[i:i+2], color=EGO_COLOR,
                        linewidth=2.5, alpha=alpha, zorder=8, solid_capstyle='round')

        # Agents
        self._add_agents(sample, ax)

        # Ego vehicle (on top)
        self._add_ego_agent(sample, ax)

        # Title
        ax.set_title(
            f"{scenario_type}  |  t={step * 0.1:.1f}s  ({step}/{num_steps - 1})",
            fontsize=11, fontweight="bold", pad=6,
        )

    def _load_map_data(self):
        if SemanticMapLayer.STOP_LINE in self._nearest_vector_map:
            self._nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
                sp for sp in self._nearest_vector_map[SemanticMapLayer.STOP_LINE]
                if sp.stop_line_type != StopLineType.TURN_STOP
            ]

    def _render_map_polygon_layers(self, ax):
        layers = [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.LANE]),
            (SemanticMapLayer.INTERSECTION, simulation_map_layer_color[SemanticMapLayer.INTERSECTION]),
            (SemanticMapLayer.STOP_LINE, simulation_map_layer_color[SemanticMapLayer.STOP_LINE]),
            (SemanticMapLayer.CROSSWALK, simulation_map_layer_color[SemanticMapLayer.CROSSWALK]),
            (SemanticMapLayer.WALKWAYS, simulation_map_layer_color[SemanticMapLayer.WALKWAYS]),
            (SemanticMapLayer.CARPARK_AREA, simulation_map_layer_color[SemanticMapLayer.CARPARK_AREA]),
        ]
        if self.scenario.get_route_roadblock_ids():
            layers.append((SemanticMapLayer.ROADBLOCK, simulation_map_layer_color[SemanticMapLayer.ROADBLOCK]))
        for layer_name, color in layers:
            for obj in self._nearest_vector_map.get(layer_name, []):
                xs, ys = zip(*obj.polygon.exterior.coords)
                ax.fill(xs, ys, color=color["fill_color"], alpha=color["fill_color_alpha"],
                        edgecolor=color["line_color"])

    def _render_map_line_layers(self, ax):
        for layer_name, color in [
            (SemanticMapLayer.LANE, simulation_map_layer_color[SemanticMapLayer.BASELINE_PATHS]),
            (SemanticMapLayer.LANE_CONNECTOR, simulation_map_layer_color[SemanticMapLayer.LANE_CONNECTOR]),
        ]:
            for obj in self._nearest_vector_map.get(layer_name, []):
                path = obj.baseline_path.discrete_path
                ax.plot([p.x for p in path], [p.y for p in path],
                        color=color["line_color"], linestyle='dashed',
                        alpha=color["line_color_alpha"], linewidth=0.8)

    def _add_ego_agent(self, sample, ax):
        fp = sample.ego_state.car_footprint
        corners = fp.all_corners()
        xs = [c.x for c in corners] + [corners[0].x]
        ys = [c.y for c in corners] + [corners[0].y]
        ax.fill(xs, ys, color=simulation_tile_agent_style["ego"]["fill_color"],
                alpha=simulation_tile_agent_style["ego"]["fill_alpha"],
                edgecolor=simulation_tile_agent_style["ego"]["line_color"],
                linewidth=simulation_tile_agent_style["ego"]["line_width"], zorder=10)
        al = fp.length / 10.0
        ax.arrow(fp.center.x, fp.center.y,
                 al * np.cos(fp.center.heading), al * np.sin(fp.center.heading),
                 head_width=0.5, head_length=0.7, fc="black", ec="black", zorder=11)

    def _add_agents(self, sample, ax):
        if isinstance(sample.observation, DetectionsTracks):
            for name, otype in tracked_object_types.items():
                if name not in ('vehicles', 'pedestrians', 'bicycles'):
                    continue
                style = simulation_tile_agent_style[name]
                for obj in sample.observation.tracked_objects.get_tracked_objects_of_type(otype):
                    corners = obj.box.all_corners()
                    xs = [c.x for c in corners] + [corners[0].x]
                    ys = [c.y for c in corners] + [corners[0].y]
                    ax.fill(xs, ys, color=style["fill_color"], alpha=style["fill_alpha"],
                            edgecolor=style["line_color"], linewidth=style["line_width"], zorder=10)
                    al = obj.box.length / 10.0
                    ax.arrow(obj.box.center.x, obj.box.center.y,
                             al * np.cos(obj.box.center.heading),
                             al * np.sin(obj.box.center.heading),
                             head_width=0.5, head_length=0.7, fc="black", ec="black", zorder=11)


def load_sim_log(msgpack_path):
    with lzma.open(msgpack_path, "rb") as f:
        data = msgpack.unpackb(f.read())
        return pickle.loads(data)


def render_scenario_mp4(sim_log, output_path, fps=10):
    history = sim_log.simulation_history
    scenario = sim_log.scenario
    map_api = history.map_api
    num_steps = len(history.data)
    num_expert = scenario.get_number_of_iterations()
    scenario_type = getattr(scenario, "scenario_type", "unknown")

    # Extract ego trajectory
    ego_xs = [s.ego_state.center.x for s in history.data]
    ego_ys = [s.ego_state.center.y for s in history.data]

    # Extract expert trajectory
    expert_xs, expert_ys = [], []
    for i in range(num_expert):
        es = scenario.get_ego_state_at_iteration(i)
        expert_xs.append(es.center.x)
        expert_ys.append(es.center.y)

    sv = ScenarioVisualizer(map_api, scenario,
                            boundary_x=(50, 50), boundary_y=(50, 50), radius=150)

    # Cache map data once using midpoint of all trajectories
    from nuplan.common.actor_state.state_representation import Point2D
    mid_x = np.mean(ego_xs + expert_xs)
    mid_y = np.mean(ego_ys + expert_ys)
    sv.cache_map(Point2D(mid_x, mid_y))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    dpi = 100
    writer = None
    tmp_frame = f"/tmp/_scenario_frame_{os.getpid()}.png"

    for step in range(num_steps):
        ax.clear()
        sample = history.data[step]

        sv.render_frame(
            sample, ax,
            ego_trail_xs=ego_xs[:step + 1],
            ego_trail_ys=ego_ys[:step + 1],
            expert_xs=expert_xs,
            expert_ys=expert_ys,
            expert_step=step if step < num_expert else None,
            step=step,
            num_steps=num_steps,
            scenario_type=scenario_type,
        )

        fig.savefig(tmp_frame, dpi=dpi, facecolor="white")
        img = cv2.imread(tmp_frame)
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        else:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    if writer is not None:
        writer.release()
    plt.close()
    if os.path.exists(tmp_frame):
        os.remove(tmp_frame)


def _init_obs_env(scenario, map_api, route_roadblock_ids):
    """Initialize C binding obs env for computing observations."""
    import tempfile
    from pufferlib.ocean.drive import binding
    from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary

    bin_path = tempfile.mktemp(suffix=".bin")
    nuplan_to_binary(
        scenario=scenario,
        map_api=map_api,
        route_roadblock_ids=list(route_roadblock_ids),
        output_path=bin_path,
    )
    obs_size = (binding.EGO_FEATURES_CLASSIC
                + binding.PARTNER_FEATURES * binding.MAX_OBS_PARTNERS
                + binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS)
    obs_buf = np.zeros(obs_size, dtype=np.float32)
    c_env_handle = binding.init_obs_env(bin_path, obs_buf)

    # Goal = last expert state
    last_idx = scenario.get_number_of_iterations() - 1
    last_state = scenario.get_ego_state_at_iteration(last_idx)
    goal_x, goal_y = last_state.center.x, last_state.center.y

    return c_env_handle, obs_buf, goal_x, goal_y, bin_path


def _compute_obs_for_sample(sample, c_env_handle, obs_buf, goal_x, goal_y):
    """Compute observation vector for a single simulation sample."""
    from pufferlib.ocean.drive import binding
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

    ego = sample.ego_state
    tracked = sample.observation.tracked_objects

    agent_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    agents_list = tracked.get_tracked_objects_of_types(agent_types)
    if agents_list:
        rows = []
        for a in agents_list:
            # Agent velocities are already in global frame
            rows.append([a.box.center.x, a.box.center.y, a.box.center.heading,
                         a.velocity.x, a.velocity.y, a.box.width, a.box.length])
        agents = np.array(rows, dtype=np.float32)
    else:
        agents = np.zeros((0, 7), dtype=np.float32)

    # nuPlan velocity is in body frame — convert to global frame
    ego_h = ego.center.heading
    cos_h, sin_h = math.cos(ego_h), math.sin(ego_h)
    ego_vx_body = ego.dynamic_car_state.center_velocity_2d.x
    ego_vy_body = ego.dynamic_car_state.center_velocity_2d.y
    ego_vx = ego_vx_body * cos_h - ego_vy_body * sin_h
    ego_vy = ego_vx_body * sin_h + ego_vy_body * cos_h

    # Apply 0.7x bounding box shrink to match Drive env's should_control_agent()
    binding.compute_obs_external(
        c_env_handle,
        ego.center.x, ego.center.y, ego_h,
        ego_vx, ego_vy,
        ego.car_footprint.width * 0.7, ego.car_footprint.length * 0.7,
        goal_x, goal_y,
        agents,
    )
    return obs_buf.copy()


def _render_obs_frame(obs, ax):
    """Render observation bird's-eye view on given axes."""
    import math
    from pufferlib.nuplan_integration.visualize_obs import parse_obs

    parsed = parse_obs(obs)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.2, color="white")

    # Road segments
    # Ego frame: rel_x = forward, rel_y = left
    # Plot: x-axis = right (negate rel_y), y-axis = forward (rel_x)
    type_colors = {0.0: "#4a7c59", 1.0: "#c4a35a", 2.0: "#8b4513"}
    for seg in parsed["road_segments"]:
        color = type_colors.get(seg["type"], "#666666")
        dx = seg["half_len"] * math.cos(seg["heading"])
        dy = seg["half_len"] * math.sin(seg["heading"])
        lw = 1.5 if seg["type"] == 2.0 else 1.0
        ax.plot([-seg["y"] + dy, -seg["y"] - dy],
                [seg["x"] - dx, seg["x"] + dx],
                color=color, linewidth=lw, alpha=0.7)

    # Partners (mirror: negate rel_y and heading for horizontal flip)
    for p in parsed["partners"]:
        mh = -p["heading"]  # mirror heading
        cos_h, sin_h = math.cos(mh), math.sin(mh)
        hw, hl = p["width"] / 2, p["length"] / 2
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        px, py = -p["rel_y"], p["rel_x"]
        rotated = [(cx * cos_h - cy * sin_h + px,
                     cx * sin_h + cy * cos_h + py) for cx, cy in corners]
        polygon = plt.Polygon(rotated, closed=True, facecolor="#e74c3c",
                              edgecolor="white", alpha=0.7, linewidth=0.5)
        ax.add_patch(polygon)
        arrow_len = min(abs(p["speed"]) * 0.3, 5.0)
        ax.annotate("", xy=(px + arrow_len * sin_h, py + arrow_len * cos_h),
                     xytext=(px, py),
                     arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.5))

    # Ego
    ego = parsed["ego"]
    hw, hl = ego["width"] / 2, ego["length"] / 2
    ego_rect = plt.Rectangle((-hw, -hl), ego["width"], ego["length"],
                              facecolor="#3498db", edgecolor="white",
                              linewidth=1.5, alpha=0.9)
    ax.add_patch(ego_rect)
    ax.plot(0, 0, "o", color="white", markersize=3)

    # Goal
    gx, gy = -ego["goal_rel_y"], ego["goal_rel_x"]
    ax.plot(gx, gy, "*", color="#f1c40f", markersize=15, markeredgecolor="white",
            markeredgewidth=0.5, zorder=10)

    # Auto-scale
    all_x = [0, gx] + [-p["rel_y"] for p in parsed["partners"]]
    all_y = [0, gy] + [p["rel_x"] for p in parsed["partners"]]
    if parsed["road_segments"]:
        all_x += [-s["y"] for s in parsed["road_segments"]]
        all_y += [s["x"] for s in parsed["road_segments"]]
    pad = 10
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_title(f"Policy Input  |  spd={ego['speed']:.1f} m/s  partners={len(parsed['partners'])}",
                 fontsize=10, fontweight="bold", color="white", pad=6)


def render_scenario_with_obs_mp4(sim_log, output_path, fps=10):
    """Render side-by-side: nuPlan scene (left) + policy observation (right)."""
    history = sim_log.simulation_history
    scenario = sim_log.scenario
    map_api = history.map_api
    num_steps = len(history.data)
    num_expert = scenario.get_number_of_iterations()
    scenario_type = getattr(scenario, "scenario_type", "unknown")

    ego_xs = [s.ego_state.center.x for s in history.data]
    ego_ys = [s.ego_state.center.y for s in history.data]

    expert_xs, expert_ys = [], []
    for i in range(num_expert):
        es = scenario.get_ego_state_at_iteration(i)
        expert_xs.append(es.center.x)
        expert_ys.append(es.center.y)

    sv = ScenarioVisualizer(map_api, scenario,
                            boundary_x=(50, 50), boundary_y=(50, 50), radius=150)
    from nuplan.common.actor_state.state_representation import Point2D
    mid_x = np.mean(ego_xs + expert_xs)
    mid_y = np.mean(ego_ys + expert_ys)
    sv.cache_map(Point2D(mid_x, mid_y))

    # Init C obs env
    route_ids = list(scenario.get_route_roadblock_ids())
    c_env_handle, obs_buf, goal_x, goal_y, bin_path = _init_obs_env(
        scenario, map_api, route_ids)

    fig, (ax_scene, ax_obs) = plt.subplots(1, 2, figsize=(16, 8))
    dpi = 100
    writer = None
    tmp_frame = f"/tmp/_scenario_obs_frame_{os.getpid()}.png"

    for step in range(num_steps):
        ax_scene.clear()
        ax_obs.clear()
        sample = history.data[step]

        # Left: nuPlan scene
        sv.render_frame(
            sample, ax_scene,
            ego_trail_xs=ego_xs[:step + 1],
            ego_trail_ys=ego_ys[:step + 1],
            expert_xs=expert_xs,
            expert_ys=expert_ys,
            expert_step=step if step < num_expert else None,
            step=step,
            num_steps=num_steps,
            scenario_type=scenario_type,
        )

        # Right: policy observation
        obs = _compute_obs_for_sample(sample, c_env_handle, obs_buf, goal_x, goal_y)
        _render_obs_frame(obs, ax_obs)

        fig.savefig(tmp_frame, dpi=dpi, facecolor="white")
        img = cv2.imread(tmp_frame)
        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        else:
            img = cv2.resize(img, (w, h))
        writer.write(img)

    if writer is not None:
        writer.release()
    plt.close()
    if os.path.exists(tmp_frame):
        os.remove(tmp_frame)
    # Cleanup temp bin
    if os.path.exists(bin_path):
        os.remove(bin_path)


def main():
    if len(sys.argv) < 2:
        base = os.environ.get("NUPLAN_EXP_ROOT", "")
        runs = sorted(glob.glob(os.path.join(base, "*")))
        if not runs:
            print("No simulation runs found!")
            return
        sim_dir = runs[-1]
    else:
        sim_dir = sys.argv[1]

    log_files = sorted(glob.glob(os.path.join(sim_dir, "simulation_log", "**", "*.msgpack.xz"), recursive=True))
    print(f"Found {len(log_files)} scenario logs in {sim_dir}")

    output_dir = os.path.join(sim_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)

    for i, log_file in enumerate(log_files):
        parts = log_file.split("/simulation_log/")
        if len(parts) > 1:
            scenario_name = parts[1].split("/")[1]
        else:
            scenario_name = f"scenario_{i}"

        output_path = os.path.join(output_dir, f"{scenario_name}.mp4")
        print(f"[{i + 1}/{len(log_files)}] {scenario_name}...")

        try:
            sim_log = load_sim_log(log_file)
            render_scenario_mp4(sim_log, output_path)
            print(f"  -> {output_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
