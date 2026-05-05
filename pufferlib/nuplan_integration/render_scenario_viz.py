# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Render nuPlan simulation scenarios using map + agent visualization.
Adapted from ScenarioVisualizer for single-planner PufferDrive results.

Usage:
    python -m pufferlib.nuplan_integration.render_scenario_viz \
        --exp_dir /path/to/simulation/output \
        [--scenario behind_pedestrian_on_pickup_dropoff] \
        [--out_dir scenario_videos]
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from pathlib import Path
import glob
import os
import argparse
import cv2

from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.utils.serialization.to_scene import tracked_object_types
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType

# ── Style config ──────────────────────────────────────────────────────────────

simulation_tile_agent_style = {
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
PLANNER_LABEL = "PufferDrive PPO"


# ── ScenarioVisualizer ────────────────────────────────────────────────────────

class ScenarioVisualizer:
    def __init__(self, scenario, boundary_x=(100, 100), boundary_y=(100, 100), radius=150):
        self._map_api = scenario.map_api
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

    def visualize_single_sample(self, sample, ax):
        self._center = sample.ego_state.center
        self._nearest_vector_map = self._map_api.get_proximal_map_objects(
            self._center, self._radius, self._layer_names)
        ax.set_xlim(self._center.x - self._boundary_x[0], self._center.x + self._boundary_x[1])
        ax.set_ylim(self._center.y - self._boundary_y[0], self._center.y + self._boundary_y[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        self._load_map_data()
        self._render_map_polygon_layers(ax)
        self._render_map_line_layers(ax)
        self._add_ego_agent(sample, ax)
        self._add_agents(sample, ax)

    def render_map_only(self, center, ax):
        self._center = center
        self._nearest_vector_map = self._map_api.get_proximal_map_objects(
            self._center, self._radius, self._layer_names)
        ax.set_xlim(self._center.x - self._boundary_x[0], self._center.x + self._boundary_x[1])
        ax.set_ylim(self._center.y - self._boundary_y[0], self._center.y + self._boundary_y[1])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
        self._load_map_data()
        self._render_map_polygon_layers(ax)
        self._render_map_line_layers(ax)

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


# ── Helper functions ──────────────────────────────────────────────────────────

def extract_all_trajectories(sim_log, n_steps, step=2):
    history = sim_log.simulation_history.data
    ego_xs, ego_ys = [], []
    agent_trajs = {}
    total_frames = 0
    for t in range(0, n_steps, step):
        sample = history[t]
        ego_xs.append(sample.ego_state.center.x)
        ego_ys.append(sample.ego_state.center.y)
        total_frames += 1
        if isinstance(sample.observation, DetectionsTracks):
            for name, otype in tracked_object_types.items():
                if name not in ('vehicles', 'bicycles', 'pedestrians'):
                    continue
                for obj in sample.observation.tracked_objects.get_tracked_objects_of_type(otype):
                    tid = obj.track_token
                    if tid not in agent_trajs:
                        agent_trajs[tid] = {"xs": [], "ys": [], "type": name, "last_frame": 0}
                    agent_trajs[tid]["xs"].append(obj.box.center.x)
                    agent_trajs[tid]["ys"].append(obj.box.center.y)
                    agent_trajs[tid]["last_frame"] = total_frames
    agent_trajs = {tid: t for tid, t in agent_trajs.items()
                   if t["type"] != "pedestrians" or t["last_frame"] >= total_frames - 2}
    return ego_xs, ego_ys, agent_trajs


def draw_ego_trajectory(ax, ego_xs, ego_ys, linewidth=2.5):
    for i in range(len(ego_xs) - 1):
        frac = i / max(len(ego_xs) - 1, 1)
        alpha = 0.3 + 0.7 * frac
        ax.plot(ego_xs[i:i+2], ego_ys[i:i+2], color=EGO_COLOR,
                linewidth=linewidth, alpha=alpha, zorder=8, solid_capstyle='round')


def draw_agent_trajectories(ax, agent_trajs, linewidth=1.5, min_ped_points=5):
    for tid, traj in agent_trajs.items():
        n = len(traj["xs"])
        if n < 2:
            continue
        if traj["type"] == "pedestrians" and n < min_ped_points:
            continue
        color = "#6aaa5e" if traj["type"] == "pedestrians" else AGENT_COLOR
        lw = 1.0 if traj["type"] == "pedestrians" else linewidth
        ax.plot(traj["xs"], traj["ys"], color=color,
                linewidth=lw, alpha=0.6, zorder=5)


def auto_zoom(ax, ego_xs, ego_ys, margin=15):
    xmin, xmax = min(ego_xs), max(ego_xs)
    ymin, ymax = min(ego_ys), max(ego_ys)
    dx = max(xmax - xmin, 20)
    dy = max(ymax - ymin, 20)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(dx, dy) / 2 + margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


def fig_to_bgr(fig, dpi=150):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# ── Rendering functions ───────────────────────────────────────────────────────

def render_trajectory_overview(sim_log, out_path):
    """Single-planner trajectory overview: map + full trajectory + final state."""
    n_steps = len(sim_log.simulation_history.data)
    mid = n_steps // 2

    sv = ScenarioVisualizer(
        scenario=sim_log.scenario,
        boundary_x=(200, 200), boundary_y=(200, 200), radius=300,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    center_sample = sim_log.simulation_history.data[mid]
    sv.render_map_only(center_sample.ego_state.center, ax)

    ego_xs, ego_ys, agent_trajs = extract_all_trajectories(sim_log, n_steps, step=2)
    draw_agent_trajectories(ax, agent_trajs)
    draw_ego_trajectory(ax, ego_xs, ego_ys, linewidth=3)

    last_sample = sim_log.simulation_history.data[n_steps - 1]
    sv._add_ego_agent(last_sample, ax)
    sv._add_agents(last_sample, ax)

    auto_zoom(ax, ego_xs, ego_ys)
    ax.set_title(PLANNER_LABEL, fontsize=14, fontweight="bold")

    # Legend
    legend_handles = [
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                       facecolor=EGO_COLOR, edgecolor="black", linewidth=0.8),
        Line2D([0], [0], color=EGO_COLOR, linewidth=3, solid_capstyle='round'),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                       facecolor=AGENT_COLOR, edgecolor="black", linewidth=0.8),
        Line2D([0], [0], color=AGENT_COLOR, linewidth=1.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#D5E8D4",
               markeredgecolor="black", markersize=12, markeredgewidth=0.8, linewidth=0),
    ]
    legend_labels = ["Ego Vehicle", "Ego Trajectory", "Vehicles", "Vehicle Trajectories", "Pedestrians"]
    ax.legend(legend_handles, legend_labels, loc="lower right", fontsize=10,
              frameon=True, edgecolor="black", fancybox=False)

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def render_sequence_mp4(sim_log, out_path, n_cols=5, fps=5):
    """Render MP4 video: each frame shows map + agents at current time + ego trajectory so far."""
    history = sim_log.simulation_history.data
    n_steps = len(history)

    sv = ScenarioVisualizer(
        scenario=sim_log.scenario,
        boundary_x=(35, 35), boundary_y=(35, 35), radius=150,
    )

    # Pre-extract full ego trajectory for drawing partial paths
    ego_xs, ego_ys, _ = extract_all_trajectories(sim_log, n_steps, step=1)

    writer = None
    frame_step = 2  # render every 2nd timestep for smoother video
    for t in range(0, n_steps, frame_step):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        sample = history[t]
        sv.visualize_single_sample(sample, ax)

        # Draw ego trajectory up to current time
        idx = t + 1
        if idx > 1:
            draw_ego_trajectory(ax, ego_xs[:idx], ego_ys[:idx], linewidth=3)

        # Time label
        ax.set_title(f"{PLANNER_LABEL}  |  t = {t * 0.1:.1f}s", fontsize=14, fontweight="bold")

        img = fig_to_bgr(fig)
        plt.close(fig)

        if writer is None:
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        writer.write(img)

    if writer is not None:
        writer.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def find_sim_logs(exp_dir):
    """Find all simulation log paths in an experiment directory."""
    pattern = os.path.join(exp_dir, "simulation_log", "*", "*", "*", "*", "*.msgpack.xz")
    logs = {}
    for path in glob.glob(pattern):
        parts = Path(path).parts
        # .../simulation_log/planner/scenario_type/logname/token/token.msgpack.xz
        scenario_type = parts[-4]
        logs[scenario_type] = path
    return logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", required=True, help="Simulation experiment directory")
    parser.add_argument("--scenario", default=None, help="Single scenario type to render (default: all)")
    parser.add_argument("--out_dir", default="scenario_videos", help="Output directory")
    parser.add_argument("--mode", default="both", choices=["mp4", "pdf", "both"],
                        help="Output mode: mp4 video, pdf overview, or both")
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    sim_logs = find_sim_logs(args.exp_dir)
    if args.scenario:
        if args.scenario not in sim_logs:
            print(f"Scenario '{args.scenario}' not found. Available: {list(sim_logs.keys())}")
            return
        sim_logs = {args.scenario: sim_logs[args.scenario]}

    for scenario_type, log_path in sorted(sim_logs.items()):
        print(f"Loading {scenario_type}...")
        sim_log = SimulationLog.load_data(Path(log_path))
        n = len(sim_log.simulation_history.data)
        print(f"  {n} timesteps ({n * 0.1:.1f}s)")

        if args.mode in ("pdf", "both"):
            pdf_out = out_dir / f"{scenario_type}_overview.pdf"
            print(f"  Rendering overview -> {pdf_out.name}")
            render_trajectory_overview(sim_log, pdf_out)

        if args.mode in ("mp4", "both"):
            mp4_out = out_dir / f"{scenario_type}.mp4"
            print(f"  Rendering MP4 -> {mp4_out.name}")
            render_sequence_mp4(sim_log, mp4_out, fps=args.fps)

    print("Done!")


if __name__ == "__main__":
    main()
