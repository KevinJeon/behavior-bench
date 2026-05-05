# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Visualize PufferDrive PPO simulation results (Non-Reactive vs Reactive).

Adapted from DiffuSearch visualization. Generates:
1. N x 2 grid overview (selected scenarios)
2. Per-scenario trajectory overview plots
3. Per-scenario sequence plots (timestep snapshots)

Usage:
    export NUPLAN_MAPS_ROOT='/path/to/nuplan/maps'
    export NUPLAN_DATA_ROOT='/path/to/nuplan/dataset'
    python -m pufferlib.nuplan_integration.visualize_scenarios
"""

import glob
import os
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.simulation_log import SimulationLog
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
NONREACTIVE_LABEL = "Non-Reactive Agents"
REACTIVE_LABEL = "Reactive Agents"


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


# ── Experiment directories ────────────────────────────────────────────────────
NONREACTIVE_DIR = (
    os.environ.get("NUPLAN_SIM_NONREACTIVE", "")
)
REACTIVE_DIR = (
    os.environ.get("NUPLAN_SIM_REACTIVE", "")
)
PLANNER = "PufferDrivePPO"
OUT_DIR = Path("scenario_videos/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COLS = 5


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


def auto_zoom(ax, ego_xs, ego_ys, margin=15):
    xmin, xmax = min(ego_xs), max(ego_xs)
    ymin, ymax = min(ego_ys), max(ego_ys)
    dx = max(xmax - xmin, 20)
    dy = max(ymax - ymin, 20)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    half = max(dx, dy) / 2 + margin
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)


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


def load_scores(exp_dir):
    pattern = os.path.join(exp_dir, "aggregator_metric", "*.parquet")
    agg_file = max(glob.glob(pattern), key=os.path.getmtime)
    df = pd.read_parquet(agg_file)
    return df[df["num_scenarios"].isna()].copy().set_index("scenario")


def build_path(exp_dir, planner, stype, logname, token):
    return Path(exp_dir) / "simulation_log" / planner / stype / logname / token / f"{token}.msgpack.xz"


def make_legend_row(fig, gs_row):
    ax_leg = fig.add_subplot(gs_row)
    ax_leg.set_axis_off()
    legend_handles = [
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                       facecolor=EGO_COLOR, edgecolor="black", linewidth=0.8),
        Line2D([0], [0], color=EGO_COLOR, linewidth=3, solid_capstyle='round'),
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                       facecolor=AGENT_COLOR, edgecolor="black", linewidth=0.8),
        Line2D([0], [0], color=AGENT_COLOR, linewidth=1.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#D5E8D4",
               markeredgecolor="black", markersize=12, markeredgewidth=0.8, linewidth=0),
        Line2D([0], [0], color="#6aaa5e", linewidth=1.0),
    ]
    legend_labels = ["Ego Vehicle", "Ego Trajectory", "Surrounding Vehicles",
                     "Surrounding Trajectories", "Pedestrians", "Ped. Trajectories"]
    leg = ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=6,
                        fontsize=20, frameon=True, edgecolor="black", fancybox=False,
                        handlelength=1.5, handleheight=1.0, handletextpad=0.3,
                        bbox_to_anchor=(0.0, 0, 1.0, 1.0), mode="expand",
                        borderaxespad=0)
    leg.get_frame().set_linewidth(1.0)


def main():
    print("Loading scores...")
    nr_df = load_scores(NONREACTIVE_DIR)
    re_df = load_scores(REACTIVE_DIR)

    # Load all scenarios
    all_tokens = list(nr_df.index)
    sim_data = {}
    for token in all_tokens:
        stype = nr_df.loc[token, "scenario_type"]
        logname = nr_df.loc[token, "log_name"]
        print(f"Loading {token} ({stype})...")

        nr_path = build_path(NONREACTIVE_DIR, PLANNER, stype, logname, token)
        re_path = build_path(REACTIVE_DIR, PLANNER, stype, logname, token)

        if not nr_path.exists() or not re_path.exists():
            print(f"  Skipping (missing log file)")
            continue

        nr_log = SimulationLog.load_data(nr_path)
        re_log = SimulationLog.load_data(re_path)

        n_total = min(len(nr_log.simulation_history.data),
                      len(re_log.simulation_history.data))

        sim_data[token] = {
            "nonreactive": nr_log,
            "reactive": re_log,
            "n_steps": n_total,
            "scenario_type": stype,
            "score_nr": nr_df.loc[token, "score"],
            "score_re": re_df.loc[token, "score"],
        }

    print(f"\nLoaded {len(sim_data)} scenarios.")

    # Select 5 interesting scenarios for grid plot
    # Pick scenarios with highest combined score or most interesting types
    scored = sorted(sim_data.items(),
                    key=lambda x: max(x[1]["score_nr"], x[1]["score_re"]),
                    reverse=True)
    grid_tokens = [t for t, _ in scored[:5]]
    if len(grid_tokens) < 5:
        # Fill with remaining
        for t in sim_data:
            if t not in grid_tokens:
                grid_tokens.append(t)
            if len(grid_tokens) >= 5:
                break

    # ── Plot Type 1: N x 2 grid (selected scenarios) ─────────────────────────
    print("\n=== Generating N x 2 grid plot ===")
    n_grid = len(grid_tokens)
    fig = plt.figure(figsize=(5 * n_grid, 10.5))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, n_grid, height_ratios=[1, 1, 0.12], hspace=0.05, wspace=0.05)

    axes = np.empty((2, n_grid), dtype=object)
    for r in range(2):
        for c in range(n_grid):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for col, token in enumerate(grid_tokens):
        info = sim_data[token]
        n_total = info["n_steps"]
        mid = n_total // 2

        sv = ScenarioVisualizer(
            scenario=info["nonreactive"].scenario,
            boundary_x=(200, 200), boundary_y=(200, 200), radius=300,
        )

        for row, (label, sim_log) in enumerate([
            (NONREACTIVE_LABEL, info["nonreactive"]),
            (REACTIVE_LABEL, info["reactive"]),
        ]):
            ax = axes[row, col]
            ax.set_facecolor("white")
            center_sample = sim_log.simulation_history.data[mid]

            sv.render_map_only(center_sample.ego_state.center, ax)

            ego_xs, ego_ys, agent_trajs = extract_all_trajectories(sim_log, n_total, step=2)
            draw_agent_trajectories(ax, agent_trajs)
            draw_ego_trajectory(ax, ego_xs, ego_ys, linewidth=3)

            last_sample = sim_log.simulation_history.data[n_total - 1]
            sv._add_ego_agent(last_sample, ax)
            sv._add_agents(last_sample, ax)

            auto_zoom(ax, ego_xs, ego_ys)

        # Column header
        stype_short = info["scenario_type"].replace("_", " ").title()
        if len(stype_short) > 25:
            stype_short = stype_short[:22] + "..."
        nr_score = info["score_nr"]
        re_score = info["score_re"]
        axes[0, col].set_title(f"{stype_short}\nNR={nr_score:.2f} / R={re_score:.2f}",
                               color="black", fontsize=16)

    axes[0, 0].set_ylabel(NONREACTIVE_LABEL, color="black", fontsize=22)
    axes[1, 0].set_ylabel(REACTIVE_LABEL, color="black", fontsize=22)

    make_legend_row(fig, gs[2, :])

    out = OUT_DIR / "all_scenarios_grid.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"  Saved {out}")

    # ── Plot Type 2: Per-scenario trajectory overview ─────────────────────────
    print("\n=== Generating trajectory overview plots ===")
    for token, info in sim_data.items():
        n_steps = info["n_steps"]
        mid = n_steps // 2

        sv = ScenarioVisualizer(
            scenario=info["nonreactive"].scenario,
            boundary_x=(200, 200), boundary_y=(200, 200), radius=300,
        )

        fig, axes_traj = plt.subplots(1, 2, figsize=(14, 7))
        fig.patch.set_facecolor("white")

        for ax_idx, (label, sim_log) in enumerate([
            (NONREACTIVE_LABEL, info["nonreactive"]),
            (REACTIVE_LABEL, info["reactive"]),
        ]):
            ax = axes_traj[ax_idx]
            ax.set_facecolor("white")
            center_sample = sim_log.simulation_history.data[mid]

            sv.render_map_only(center_sample.ego_state.center, ax)

            ego_xs, ego_ys, agent_trajs = extract_all_trajectories(sim_log, n_steps, step=2)
            draw_agent_trajectories(ax, agent_trajs)
            draw_ego_trajectory(ax, ego_xs, ego_ys, linewidth=3)

            last_t = min(n_steps - 1, len(sim_log.simulation_history.data) - 1)
            last_sample = sim_log.simulation_history.data[last_t]
            sv._add_ego_agent(last_sample, ax)
            sv._add_agents(last_sample, ax)

            auto_zoom(ax, ego_xs, ego_ys)

            score_key = "score_nr" if ax_idx == 0 else "score_re"
            ax.set_title(f"{label} (score={info[score_key]:.2f})",
                         color="black", fontsize=13, fontweight="bold")

        fig.suptitle(info["scenario_type"].replace("_", " ").title(),
                     fontsize=15, fontweight="bold")
        plt.tight_layout()
        out = OUT_DIR / f"{info['scenario_type']}_trajectory.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"  Saved {out.name}")

    # ── Plot Type 3: Sequence plots (2 rows x N_COLS timesteps) ───────────────
    print("\n=== Generating sequence plots ===")
    for token, info in sim_data.items():
        n_total = info["n_steps"]
        timesteps = np.linspace(0, n_total - 1, N_COLS, dtype=int).tolist()

        fig = plt.figure(figsize=(4 * N_COLS, 8.5))
        fig.patch.set_facecolor("white")
        gs_seq = gridspec.GridSpec(3, N_COLS, height_ratios=[1, 1, 0.08],
                                  hspace=0.08, wspace=0.0)

        axes_seq = np.empty((2, N_COLS), dtype=object)
        for r in range(2):
            for c in range(N_COLS):
                axes_seq[r, c] = fig.add_subplot(gs_seq[r, c])

        sv = ScenarioVisualizer(
            scenario=info["nonreactive"].scenario,
            boundary_x=(35, 35), boundary_y=(35, 35), radius=150,
        )

        nr_ego_xs, nr_ego_ys, _ = extract_all_trajectories(info["nonreactive"], n_total, step=2)
        re_ego_xs, re_ego_ys, _ = extract_all_trajectories(info["reactive"], n_total, step=2)

        for col, t in enumerate(timesteps):
            ax_nr = axes_seq[0, col]
            ax_nr.set_facecolor("white")
            sv.visualize_single_sample(info["nonreactive"].simulation_history.data[t], ax_nr)
            idx = max(1, t // 2 + 1)
            draw_ego_trajectory(ax_nr, nr_ego_xs[:idx], nr_ego_ys[:idx])
            ax_nr.set_title(f"t={t*0.1:.1f}s", color="black", fontsize=20)

            ax_re = axes_seq[1, col]
            ax_re.set_facecolor("white")
            sv.visualize_single_sample(info["reactive"].simulation_history.data[t], ax_re)
            draw_ego_trajectory(ax_re, re_ego_xs[:idx], re_ego_ys[:idx])

        axes_seq[0, 0].set_ylabel(NONREACTIVE_LABEL, color="black", fontsize=18)
        axes_seq[1, 0].set_ylabel(REACTIVE_LABEL, color="black", fontsize=18)

        fig.suptitle(info["scenario_type"].replace("_", " ").title(),
                     fontsize=16, fontweight="bold", y=0.98)

        # Legend row
        ax_leg_seq = fig.add_subplot(gs_seq[2, :])
        ax_leg_seq.set_axis_off()
        seq_handles = [
            FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                           facecolor=EGO_COLOR, edgecolor="black", linewidth=0.8),
            Line2D([0], [0], color=EGO_COLOR, linewidth=3, solid_capstyle='round'),
            FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.1",
                           facecolor=AGENT_COLOR, edgecolor="black", linewidth=0.8),
            Line2D([0], [0], marker='s', color='w', markerfacecolor="#D5E8D4",
                   markeredgecolor="black", markersize=12, markeredgewidth=0.8, linewidth=0),
        ]
        seq_labels = ["Ego Vehicle", "Ego Past Trajectory", "Surrounding Vehicles", "Pedestrians"]
        leg_seq = ax_leg_seq.legend(seq_handles, seq_labels, loc="center", ncol=4,
                                   fontsize=18, frameon=True, edgecolor="black", fancybox=False,
                                   handlelength=1.5, handleheight=1.0, handletextpad=0.3,
                                   bbox_to_anchor=(0.0, 0, 1.0, 1.0), mode="expand",
                                   borderaxespad=0)
        leg_seq.get_frame().set_linewidth(1.0)

        out = OUT_DIR / f"{info['scenario_type']}_sequence.pdf"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
        print(f"  Saved {out.name}")

    print(f"\nDone! All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
