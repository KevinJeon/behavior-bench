# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Visualize policy observations for 20 random scenarios as MP4 videos.

Usage:
    DRIVE_BINARIES_DATA_ROOT=/path/to/binaries \
    python scripts/visualize_obs_scenarios.py
"""
import os
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive.drive import Drive
from pufferlib.nuplan_integration.visualize_obs import parse_obs
import math


def render_obs_frame(obs, ax, step, num_steps):
    """Render observation bird's-eye view on given axes."""
    parsed = parse_obs(obs)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    ax.grid(True, alpha=0.2, color="white")

    # Road segments
    type_colors = {0.0: "#4a7c59", 1.0: "#c4a35a", 2.0: "#8b4513"}
    for seg in parsed["road_segments"]:
        color = type_colors.get(seg["type"], "#666666")
        dx = seg["half_len"] * math.cos(seg["heading"])
        dy = seg["half_len"] * math.sin(seg["heading"])
        lw = 1.5 if seg["type"] == 2.0 else 1.0
        ax.plot([seg["y"] - dy, seg["y"] + dy],
                [seg["x"] - dx, seg["x"] + dx],
                color=color, linewidth=lw, alpha=0.7)

    # Partners
    for p in parsed["partners"]:
        cos_h, sin_h = math.cos(p["heading"]), math.sin(p["heading"])
        hw, hl = p["width"] / 2, p["length"] / 2
        corners = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        rotated = [(cx * cos_h - cy * sin_h + p["rel_y"],
                     cx * sin_h + cy * cos_h + p["rel_x"]) for cx, cy in corners]
        polygon = plt.Polygon(rotated, closed=True, facecolor="#e74c3c",
                              edgecolor="white", alpha=0.7, linewidth=0.5)
        ax.add_patch(polygon)
        arrow_len = min(abs(p["speed"]) * 0.3, 5.0)
        ax.annotate("", xy=(p["rel_y"] + arrow_len * sin_h, p["rel_x"] + arrow_len * cos_h),
                     xytext=(p["rel_y"], p["rel_x"]),
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
    gx, gy = ego["goal_rel_y"], ego["goal_rel_x"]
    ax.plot(gx, gy, "*", color="#f1c40f", markersize=15, markeredgecolor="white",
            markeredgewidth=0.5, zorder=10)

    # Fixed view range for consistent frame size
    view_range = 60
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)

    ax.set_title(
        f"t={step * 0.1:.1f}s ({step}/{num_steps})  |  "
        f"spd={ego['speed']:.1f} m/s  |  partners={len(parsed['partners'])}",
        color="white", fontsize=10)
    ax.tick_params(colors="white")


def main():
    data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT",
                               "")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "experiments", "obs_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    num_scenarios = 20
    episode_length = 150
    fps = 10

    # Pick 20 random map indices
    # Count available binaries
    bin_dir = os.path.join(data_root, "training")
    all_bins = sorted([f for f in os.listdir(bin_dir) if f.endswith(".bin")])
    total_maps = len(all_bins)
    print(f"Found {total_maps} binary maps in {bin_dir}")

    # Filter maps that fit within MAX_AGENTS=128 by reading num_objects from binary header
    import struct
    print("Scanning binaries for object counts...", flush=True)
    valid_indices = []
    for i, bin_name in enumerate(all_bins):
        bin_path = os.path.join(bin_dir, bin_name)
        with open(bin_path, "rb") as f:
            sdc_idx = struct.unpack("i", f.read(4))[0]
            n_tracks = struct.unpack("i", f.read(4))[0]
            f.read(n_tracks * 4)  # skip track indices
            num_objects = struct.unpack("i", f.read(4))[0]
        if num_objects <= 128:
            valid_indices.append(i)
    print(f"Maps with <=128 objects: {len(valid_indices)}/{total_maps}", flush=True)

    random.seed(42)
    map_indices = sorted(random.sample(valid_indices, min(num_scenarios, len(valid_indices))))
    print(f"Selected map indices: {map_indices}")

    # Also filter by total entities (objects + roads) to avoid segfaults
    print("Filtering by total entity count...", flush=True)
    safe_indices = []
    for idx in valid_indices:
        bin_path = os.path.join(bin_dir, all_bins[idx])
        with open(bin_path, "rb") as f:
            sdc_idx = struct.unpack("i", f.read(4))[0]
            n_tracks = struct.unpack("i", f.read(4))[0]
            f.read(n_tracks * 4)
            num_objects = struct.unpack("i", f.read(4))[0]
            num_roads = struct.unpack("i", f.read(4))[0]
        if num_objects + num_roads <= 400:
            safe_indices.append(idx)
    print(f"Maps with <=128 objects and <=400 total entities: {len(safe_indices)}/{total_maps}", flush=True)

    random.seed(42)
    map_indices = sorted(random.sample(safe_indices, min(num_scenarios, len(safe_indices))))
    print(f"Selected map indices: {map_indices}", flush=True)

    # Process one scenario at a time to keep memory low
    rendered = 0
    for scenario_idx, map_id in enumerate(map_indices):
        print(f"\n[{rendered+1}/{num_scenarios}] Rendering map {map_id} ({all_bins[map_id]})...", flush=True)

        try:
            env = Drive(
                data_root=data_root,
                split="training",
                num_maps=1,
                num_agents=128,
                episode_length=episode_length,
                resample_frequency=episode_length + 10,
                dt=0.1,
                init_mode="create_all_valid",
                control_mode="control_vehicles",
                goal_behavior=0,
                collision_behavior=0,
                offroad_behavior=0,
                map_id=map_id,
            )

            obs, _ = env.reset(seed=42)
            num_agents = obs.shape[0]

            # Pick agent 0 (sdc / ego)
            agent_idx = 0

            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            writer = None
            tmp_frame = f"/tmp/_obs_viz_frame_{os.getpid()}.png"
            out_path = os.path.join(out_dir, f"scenario_{map_id:06d}.mp4")

            for step in range(episode_length + 1):
                ax.clear()

                agent_obs = obs[agent_idx]
                render_obs_frame(agent_obs, ax, step, episode_length)

                fig.savefig(tmp_frame, dpi=100, facecolor="#1a1a2e",
                            pad_inches=0.1)
                img = cv2.imread(tmp_frame)
                if writer is None:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                else:
                    img = cv2.resize(img, (w, h))
                writer.write(img)

                if step < episode_length:
                    actions = np.zeros((num_agents, *env.single_action_space.shape), dtype=np.int32)
                    obs, _, terminals, truncations, _ = env.step(actions)

            writer.release()
            plt.close(fig)
            if os.path.exists(tmp_frame):
                os.remove(tmp_frame)

            print(f"  -> {out_path}", flush=True)
            rendered += 1
        except Exception as e:
            print(f"  SKIPPED: {e}", flush=True)

    print(f"\nDone! {rendered} videos in {out_dir}")


if __name__ == "__main__":
    main()
