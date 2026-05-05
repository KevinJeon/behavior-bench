# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Plot test for IDM trajectories / lane connectivity in the Drive environment.

Modes:
  --mode lanes   : Plot only lane polylines, color connected lanes the same
  --mode idm     : Plot IDM agent trajectories on top of simulator state

Usage:
    python tests/test_idm_plot.py --map-ids 0 --mode lanes
    python tests/test_idm_plot.py --map-ids 0-55 --mode lanes
    python tests/test_idm_plot.py --map-ids 0 --mode idm --steps 40
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("DRIVE_BINARIES_DATA_ROOT", "")

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.viz import plot_simulator_state

MOVEMENT_IDM = 1


# ── Union-Find for connected components ──────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def components(self):
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return list(groups.values())


# ── Lane connectivity plot ───────────────────────────────────────────────────

def _find_enriched_json(map_id, split):
    """Find the enriched JSON file corresponding to a map_id in binariesv3."""
    conn_dir = Path(os.environ.get("GPUDRIVE_CONNECTIVITY_ROOT", "")) / split
    if conn_dir.exists():
        json_files = sorted(conn_dir.glob("*.json"))
        if map_id < len(json_files):
            return json_files[map_id]
    return None


def plot_lane_connectivity(map_id, split, output_dir, episode_length):
    """Plot lanes: trace 3-lane chains forward via exit_lanes, color each chain."""
    bin_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    bin_dir = os.path.join(bin_root, split)
    available_maps = len(glob.glob(os.path.join(bin_dir, "*.bin")))
    if available_maps == 0:
        available_maps = 100

    # Load env to get centered coordinates
    env = Drive(
        split=split, map_id=map_id, num_maps=available_maps,
        action_type="continuous", episode_length=episode_length, resample_frequency=-1,
    )
    env.reset()
    state = env.get_state()[0]
    entities = state["entities"]
    env.close()

    # Separate objects from roads by finding where road types start
    num_objects = 0
    for e in entities:
        if e["type"] in (1, 2, 3):
            num_objects += 1
        else:
            break
    road_entities = entities[num_objects:]

    # Load enriched JSON for exit_lanes
    json_path = _find_enriched_json(map_id, split)
    if json_path is None:
        raise FileNotFoundError(f"No enriched JSON for map {map_id} in split {split}")

    with open(json_path) as f:
        jdata = json.load(f)
    json_roads = jdata.get("roads", [])

    # Build lane lookup: id -> (road_index, exit_lanes)
    lane_id_to_idx = {}
    lane_exit_map = {}
    for j, road in enumerate(json_roads):
        if road.get("type") != "lane":
            continue
        lid = road["id"]
        lane_id_to_idx[lid] = j
        lane_exit_map[lid] = road.get("exit_lanes", [])

    # Find start lanes: lanes that have no entry_lanes (chain beginnings)
    all_lane_ids = set(lane_id_to_idx.keys())
    has_entry = set()
    for j, road in enumerate(json_roads):
        if road.get("type") != "lane":
            continue
        for eid in road.get("exit_lanes", []):
            if eid in all_lane_ids:
                has_entry.add(eid)
    start_lanes = [lid for lid in all_lane_ids if lid not in has_entry]
    # If no pure start lanes, use all lanes as potential starts
    if not start_lanes:
        start_lanes = list(all_lane_ids)

    # Pick ONE start lane and trace 3 lanes forward
    chain_depth = 3
    # Pick first start lane that has exit_lanes
    chosen_start = None
    for sid in start_lanes:
        if len(lane_exit_map.get(sid, [])) > 0:
            chosen_start = sid
            break
    if chosen_start is None and start_lanes:
        chosen_start = start_lanes[0]
    elif chosen_start is None:
        chosen_start = list(all_lane_ids)[0]

    # Trace forward from chosen start
    chain = [chosen_start]
    current = chosen_start
    for _ in range(chain_depth - 1):
        exits = lane_exit_map.get(current, [])
        next_lane = None
        for eid in exits:
            if eid in all_lane_ids:
                next_lane = eid
                break
        if next_lane is None:
            break
        chain.append(next_lane)
        current = next_lane

    # Color: 1st chain lane = red, 2nd = orange, 3rd = green, others = blue
    chain_colors = ["red", "orange", "green"]
    chain_set = set(chain)
    lane_id_to_color = {}
    for i, lid in enumerate(chain):
        lane_id_to_color[lid] = chain_colors[i] if i < len(chain_colors) else "green"
    for lid in all_lane_ids:
        if lid not in lane_id_to_color:
            lane_id_to_color[lid] = "blue"

    # Compute axis limits from road entities
    road_x, road_y = [], []
    for e in entities:
        if e["type"] in (4, 5, 6):
            tx, ty = e.get("traj_x", []), e.get("traj_y", [])
            if len(tx) > 0:
                road_x.extend(tx)
                road_y.extend(ty)
    margin = 15.0
    if road_x:
        axis_limits = (min(road_x) - margin, max(road_x) + margin,
                       min(road_y) - margin, max(road_y) + margin)
    else:
        axis_limits = None

    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_facecolor("white")

    # Background: road edges (type 6)
    for e in entities:
        if e["type"] == 6:
            tx, ty = e.get("traj_x", []), e.get("traj_y", [])
            if len(tx) > 1:
                ax.plot(tx, ty, color="black", linewidth=1.5, alpha=0.6, zorder=1)

    # Road lines (type 5) as dashed
    for e in entities:
        if e["type"] == 5:
            tx, ty = e.get("traj_x", []), e.get("traj_y", [])
            if len(tx) > 1:
                ax.plot(tx, ty, color="gray", linewidth=0.5, linestyle="--", alpha=0.4, zorder=1)

    # Draw lanes colored by chain
    num_lanes_plotted = 0
    for j, road in enumerate(json_roads):
        if road.get("type") != "lane":
            continue
        lid = road["id"]
        color = lane_id_to_color.get(lid, "lightgray")

        # Get centered coordinates from state dict
        if j < len(road_entities):
            re = road_entities[j]
            tx, ty = re.get("traj_x", []), re.get("traj_y", [])
        else:
            continue

        if len(tx) < 2:
            continue

        is_chain = lid in chain_set
        z = 10 if is_chain else 5
        lw = 3.5 if is_chain else 2.0
        ax.plot(tx, ty, color=color, linewidth=lw, alpha=0.85, zorder=z)
        # Arrow at midpoint showing direction
        mid = len(tx) // 2
        if mid + 1 < len(tx):
            dx = tx[mid + 1] - tx[mid]
            dy = ty[mid + 1] - ty[mid]
            ax.annotate("", xy=(tx[mid] + dx * 0.3, ty[mid] + dy * 0.3),
                        xytext=(tx[mid], ty[mid]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                        zorder=z + 1)
        num_lanes_plotted += 1

    if axis_limits:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    ax.set_aspect("equal")
    ax.set_title(f"Lane Chain (3-deep) — Map {map_id} ({split}), "
                 f"{num_lanes_plotted} lanes, {len(chain)} red", fontsize=14)

    out_path = os.path.join(output_dir, f"lanes_map{map_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path, num_lanes_plotted, len(chain)


# ── IDM trajectory plot (existing) ───────────────────────────────────────────

def _get_agent_positions(env):
    state = env.get_state()[0]
    entities = state["entities"]
    active = state["active_agent_indices"]
    positions = {}
    for action_idx, entity_idx in enumerate(active):
        e = entities[entity_idx]
        positions[action_idx] = (e["x"], e["y"])
    return positions


def collect_idm_trajectories(env, num_steps):
    num_agents = env.num_agents
    positions = _get_agent_positions(env)
    trajectories = {i: [positions[i]] for i in range(num_agents) if i in positions}
    removed = set()  # agents that have been removed (goal reached / collision)
    all_indices = list(range(num_agents))
    binding.vec_set_movement_mode(env.c_envs, all_indices, MOVEMENT_IDM)
    neutral_actions = np.zeros((num_agents, 2), dtype=np.float32)
    INVALID = -10000.0
    for _ in range(num_steps):
        env.step(neutral_actions)
        positions = _get_agent_positions(env)
        for i in trajectories:
            if i in removed:
                continue  # stop recording once removed
            if i in positions:
                x, y = positions[i]
                if x == INVALID or y == INVALID:
                    removed.add(i)
                    continue
                # Stop recording once agent stops moving
                prev = trajectories[i][-1]
                if abs(x - prev[0]) < 0.01 and abs(y - prev[1]) < 0.01:
                    removed.add(i)
                    continue
                trajectories[i].append((x, y))
    for i in trajectories:
        trajectories[i] = np.array(trajectories[i], dtype=np.float32)
    # Filter out agents whose initial position was already invalid
    trajectories = {i: t for i, t in trajectories.items()
                    if len(t) >= 2 and not np.any(t == INVALID)}
    return trajectories


def plot_idm_trajectories(map_id, split, steps, episode_length, output_dir):
    bin_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    bin_dir = os.path.join(bin_root, split)
    available_maps = len(glob.glob(os.path.join(bin_dir, "*.bin")))
    if available_maps == 0:
        available_maps = 100
    env = Drive(
        split=split, map_id=map_id, num_maps=available_maps,
        action_type="continuous", episode_length=episode_length, resample_frequency=-1,
        goal_behavior=3,  # GOAL_REMOVE: remove agents when they reach goal
    )
    env.reset()
    num_agents = env.num_agents
    initial_state = env.get_state()[0]
    snapshot = env.create_snapshot()
    trajectories = collect_idm_trajectories(env, steps)
    env.restore_snapshot(snapshot)
    env.free_snapshot(snapshot)

    # Compute axis limits: zoom to agent trajectories area (with road context)
    agent_x, agent_y = [], []
    for traj in trajectories.values():
        if len(traj) >= 2:
            agent_x.extend(traj[:, 0].tolist())
            agent_y.extend(traj[:, 1].tolist())
    # Also include initial agent positions from state
    for entity in initial_state.get("entities", []):
        if entity["type"] in (1, 2, 3) and entity.get("x", -10000) != -10000:
            agent_x.append(entity["x"])
            agent_y.append(entity["y"])

    if agent_x:
        margin = 40.0
        axis_limits = (min(agent_x) - margin, max(agent_x) + margin,
                       min(agent_y) - margin, max(agent_y) + margin)
    else:
        # Fallback to full road view
        road_x, road_y = [], []
        for entity in initial_state.get("entities", []):
            if entity["type"] in (4, 5, 6):
                tx, ty = entity.get("traj_x", []), entity.get("traj_y", [])
                if len(tx) > 0:
                    road_x.extend(tx)
                    road_y.extend(ty)
        margin = 15.0
        axis_limits = (min(road_x) - margin, max(road_x) + margin,
                       min(road_y) - margin, max(road_y) + margin) if road_x else None

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_facecolor("white")
    plot_simulator_state(initial_state, test_agent_idx=0, ax=ax, axis_limits=axis_limits)

    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_agents)]

    for agent_idx, traj in trajectories.items():
        if len(traj) < 2:
            continue
        dist = np.sqrt((traj[-1, 0] - traj[0, 0]) ** 2 + (traj[-1, 1] - traj[0, 1]) ** 2)
        if dist < 0.1:
            continue
        color = colors[agent_idx]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0, alpha=0.8, zorder=20,
                label=f"Agent {agent_idx}")
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=40, marker="o", zorder=21,
                   edgecolors="black", linewidths=0.5)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=80, marker="*", zorder=21,
                   edgecolors="black", linewidths=0.5)

    ax.set_title(f"IDM Trajectories — Map {map_id} ({split}), {steps} steps, {num_agents} agents",
                 fontsize=14)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    out_path = os.path.join(output_dir, f"idm_map{map_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    env.close()
    return out_path, num_agents


# ── IDM video (step-by-step animation) ────────────────────────────────────────

def make_idm_video(map_id, split, steps, episode_length, output_dir, fps=10):
    """Render each IDM step as a frame and save as mp4."""
    import tempfile

    bin_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    bin_dir = os.path.join(bin_root, split)
    available_maps = len(glob.glob(os.path.join(bin_dir, "*.bin")))
    if available_maps == 0:
        available_maps = 100

    env_kwargs = dict(
        split=split, map_id=map_id, num_maps=available_maps,
        action_type="continuous", episode_length=episode_length, resample_frequency=-1,
        goal_behavior=3,  # GOAL_REMOVE: remove agents when they reach goal
    )

    # Dry-run in a separate env to compute axis limits without corrupting
    # route_progress (not saved/restored by snapshots).
    env_dry = Drive(**env_kwargs)
    env_dry.reset()
    num_agents = env_dry.num_agents
    all_indices = list(range(num_agents))
    binding.vec_set_movement_mode(env_dry.c_envs, all_indices, MOVEMENT_IDM)
    neutral_actions = np.zeros((num_agents, 2), dtype=np.float32)
    INVALID = -10000.0
    all_agent_x, all_agent_y = [], []
    initial_state = env_dry.get_state()[0]
    for entity in initial_state.get("entities", []):
        if entity["type"] in (1, 2, 3) and entity.get("x", -10000) != -10000:
            all_agent_x.append(entity["x"])
            all_agent_y.append(entity["y"])
    for _ in range(steps):
        env_dry.step(neutral_actions)
        state = env_dry.get_state()[0]
        for action_idx, entity_idx in enumerate(state["active_agent_indices"]):
            e = state["entities"][entity_idx]
            if e["x"] != INVALID and e["y"] != INVALID:
                all_agent_x.append(e["x"])
                all_agent_y.append(e["y"])
    env_dry.close()

    if all_agent_x:
        margin = 40.0
        axis_limits = (min(all_agent_x) - margin, max(all_agent_x) + margin,
                       min(all_agent_y) - margin, max(all_agent_y) + margin)
    else:
        axis_limits = None

    # Fresh env for actual rendering (clean route_progress)
    env = Drive(**env_kwargs)
    env.reset()
    num_agents = env.num_agents
    all_indices = list(range(num_agents))
    binding.vec_set_movement_mode(env.c_envs, all_indices, MOVEMENT_IDM)

    cmap = plt.cm.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(num_agents)]

    # Collect trail history — stop tracking once agent is removed
    trails = {i: [] for i in range(num_agents)}
    removed_agents = set()

    # Save frames to temp dir
    frame_dir = tempfile.mkdtemp()
    frame_paths = []

    for step in range(steps):
        state = env.get_state()[0]
        entities = state["entities"]
        active = state["active_agent_indices"]

        # Collect positions (stop once removed)
        for action_idx, entity_idx in enumerate(active):
            if action_idx in removed_agents:
                continue
            e = entities[entity_idx]
            x, y = e["x"], e["y"]
            if x == INVALID or y == INVALID:
                removed_agents.add(action_idx)
                continue
            # Stop tracking once agent stops moving
            if len(trails[action_idx]) >= 1:
                prev = trails[action_idx][-1]
                if abs(x - prev[0]) < 0.01 and abs(y - prev[1]) < 0.01:
                    removed_agents.add(action_idx)
                    continue
            trails[action_idx].append((x, y))

        # Draw frame
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_facecolor("white")
        plot_simulator_state(state, test_agent_idx=0, ax=ax, axis_limits=axis_limits)

        # Draw IDM routes (thin dashed lines)
        for action_idx, entity_idx in enumerate(active):
            if action_idx in removed_agents:
                continue
            e = entities[entity_idx]
            rx, ry = e.get("route_x"), e.get("route_y")
            if rx is not None and ry is not None and len(rx) >= 2:
                ax.plot(rx, ry, color=colors[action_idx], linewidth=2.0,
                        linestyle="--", alpha=0.7, zorder=15)

        # Draw trails
        for action_idx, trail in trails.items():
            if len(trail) < 1:
                continue
            arr = np.array(trail)
            color = colors[action_idx]
            if len(arr) >= 2:
                ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=1.5,
                        alpha=0.6, zorder=20)

        ax.set_title(f"IDM Step {step}/{steps} — Map {map_id} ({split}), {num_agents} agents",
                     fontsize=12)
        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
        ax.set_aspect("equal")

        frame_path = os.path.join(frame_dir, f"frame_{step:04d}.png")
        fig.savefig(frame_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(frame_path)

        # Step simulation
        env.step(neutral_actions)

    env.close()

    # Encode frames to GIF using Pillow
    from PIL import Image as PILImage
    out_path = os.path.join(output_dir, f"idm_map{map_id}.gif")
    images = [PILImage.open(fp) for fp in frame_paths]
    duration = int(1000 / fps)  # ms per frame
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=duration, loop=0, optimize=True)

    # Clean up frames
    for fp in frame_paths:
        os.remove(fp)
    os.rmdir(frame_dir)

    return out_path, num_agents


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_map_ids(s):
    if "-" in s and "," not in s:
        start, end = s.split("-")
        return list(range(int(start.strip()), int(end.strip())))
    else:
        return [int(x.strip()) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Plot IDM / lane connectivity for Drive env")
    parser.add_argument("--map-ids", type=str, default="0")
    parser.add_argument("--mode", choices=["idm", "lanes", "video"], default="lanes")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--split", type=str, default="pufferhard")
    parser.add_argument("--episode-length", type=int, default=91)
    parser.add_argument("--output-dir", type=str, default="artifacts/idm_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    map_ids = parse_map_ids(args.map_ids)
    print(f"Plotting {len(map_ids)} maps ({args.split}), mode={args.mode}")

    for idx, map_id in enumerate(map_ids):
        try:
            if args.mode == "lanes":
                out_path, n_lanes, n_comp = plot_lane_connectivity(
                    map_id, args.split, args.output_dir, args.episode_length)
                print(f"  [{idx+1}/{len(map_ids)}] Map {map_id}: "
                      f"{n_lanes} lanes, {n_comp} components -> {out_path}")
            elif args.mode == "video":
                out_path, num_agents = make_idm_video(
                    map_id, args.split, args.steps, args.episode_length, args.output_dir)
                print(f"  [{idx+1}/{len(map_ids)}] Map {map_id}: "
                      f"{num_agents} agents -> {out_path}")
            else:
                out_path, num_agents = plot_idm_trajectories(
                    map_id, args.split, args.steps, args.episode_length, args.output_dir)
                print(f"  [{idx+1}/{len(map_ids)}] Map {map_id}: "
                      f"{num_agents} agents -> {out_path}")
        except Exception as e:
            print(f"  [{idx+1}/{len(map_ids)}] Map {map_id}: FAILED ({e})")

    print(f"Done. {len(map_ids)} plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
