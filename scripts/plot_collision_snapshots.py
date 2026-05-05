#!/usr/bin/env python3
# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Diagnostic plot for collision snapshots from the evaluator.

Reads collision_snapshots.json and generates a grid plot showing:
- Ego and other vehicle bounding boxes at state_before (1 step before collision)
- Front bumper line (yellow)
- Velocity vectors
- Relative angle, distance, collision type, at-fault classification

Usage:
    python scripts/plot_collision_snapshots.py <path_to_collision_snapshots.json> [--max N]
"""

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _get_box_corners(x, y, heading, length, width):
    """4 corners: front-left, front-right, rear-right, rear-left."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    hl = length / 2
    hw = width / 2
    return [
        (x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h),
        (x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h),
        (x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h),
        (x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h),
    ]


def _draw_vehicle(ax, entity, color, label, draw_bumper=False):
    """Draw a vehicle with bounding box, heading arrow, and velocity vector."""
    x, y = entity["x"], entity["y"]
    h = entity["heading"]
    L = entity.get("length", 4.0)
    W = entity.get("width", 2.0)
    vx = entity.get("vx", 0)
    vy = entity.get("vy", 0)

    # Bounding box
    corners = _get_box_corners(x, y, h, L, W)
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor="black",
                        alpha=0.6, linewidth=1.2, zorder=3)
    ax.add_patch(poly)

    # Front bumper line
    if draw_bumper:
        fl, fr = corners[0], corners[1]
        ax.plot([fl[0], fr[0]], [fl[1], fr[1]], color="#ffd600",
                linewidth=3, zorder=5, alpha=0.9)

    # Heading arrow
    cos_h, sin_h = math.cos(h), math.sin(h)
    ax.annotate("", xy=(x + L/2 * cos_h * 0.8, y + L/2 * sin_h * 0.8),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                zorder=4)

    # Velocity vector
    speed = math.hypot(vx, vy)
    if speed > 0.1:
        scale = 0.3
        ax.annotate("", xy=(x + vx * scale, y + vy * scale),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5),
                    zorder=6)
        ax.text(x + vx * scale, y + vy * scale + 0.5,
                f"{speed:.1f} m/s", ha="center", fontsize=6, color=color, zorder=7)

    # Label
    ax.text(x, y - W * 0.8, label, ha="center", va="top", fontsize=7,
            fontweight="bold", color=color, zorder=7)


def _draw_projected(ax, entity, dt, color):
    """Draw projected position after dt seconds (dashed outline)."""
    x = entity["x"] + entity.get("vx", 0) * dt
    y = entity["y"] + entity.get("vy", 0) * dt
    h = entity["heading"]
    L = entity.get("length", 4.0)
    W = entity.get("width", 2.0)

    corners = _get_box_corners(x, y, h, L, W)
    poly = plt.Polygon(corners, closed=True, facecolor="none", edgecolor=color,
                        alpha=0.5, linewidth=1.0, linestyle="--", zorder=2)
    ax.add_patch(poly)

    # Front bumper of projected ego
    fl, fr = corners[0], corners[1]
    ax.plot([fl[0], fr[0]], [fl[1], fr[1]], color="#ffd600",
            linewidth=2, zorder=2, alpha=0.4, linestyle="--")

    return x, y


def plot_snapshots(snapshots, output_path, max_plots=20):
    """Generate grid of collision diagnostic plots."""
    n = min(len(snapshots), max_plots)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n):
        snap = snapshots[i]
        ax = axes[i]
        ego = snap["ego"]
        other = snap["other"]
        ctype = snap["collision_type"]
        fault = snap["at_fault"]
        map_id = snap["map_id"]
        step = snap["step"]

        fault_str = "AT FAULT" if fault else "NOT AT FAULT"
        fault_color = "#d32f2f" if fault else "#2e7d32"

        # Draw vehicles at state_before
        _draw_vehicle(ax, ego, "#42a5f5", "ego", draw_bumper=True)
        _draw_vehicle(ax, other, "#ef5350", "other")

        # Draw projected positions (dt=0.5s forward)
        dt = 0.5
        px_ego, py_ego = _draw_projected(ax, ego, dt, "#42a5f5")
        px_other, py_other = _draw_projected(ax, other, dt, "#ef5350")

        # Connection line between vehicles
        dist = math.hypot(other["x"] - ego["x"], other["y"] - ego["y"])
        mid_x = (ego["x"] + other["x"]) / 2
        mid_y = (ego["y"] + other["y"]) / 2
        ax.plot([ego["x"], other["x"]], [ego["y"], other["y"]],
                color="gray", linewidth=0.8, linestyle=":", alpha=0.5, zorder=1)
        ax.text(mid_x, mid_y + 0.3, f"{dist:.1f}m", ha="center",
                fontsize=6, color="gray", zorder=1)

        # Compute relative angle
        dx = other["x"] - ego["x"]
        dy = other["y"] - ego["y"]
        d = math.hypot(dx, dy)
        if d > 1e-6:
            ego_dir = (math.cos(ego["heading"]), math.sin(ego["heading"]))
            dot = (ego_dir[0] * dx + ego_dir[1] * dy) / d
            dot = max(-1, min(1, dot))
            angle_deg = math.degrees(math.acos(dot))
        else:
            angle_deg = 0

        # Projected distance
        proj_dist = math.hypot(px_other - px_ego, py_other - py_ego)

        # Info text
        ego_speed = math.hypot(ego.get("vx", 0), ego.get("vy", 0))
        other_speed = math.hypot(other.get("vx", 0), other.get("vy", 0))
        info = (
            f"Map {map_id} step {step}\n"
            f"dist={dist:.2f}m  proj={proj_dist:.2f}m\n"
            f"angle={angle_deg:.1f}deg\n"
            f"ego_v={ego_speed:.1f}  other_v={other_speed:.1f} m/s"
        )
        # Show active planner if available (hybrid planner)
        active_planner = snap.get("active_planner")
        epistemic_val = snap.get("epistemic")
        if active_planner:
            info += f"\nplanner={active_planner}"
            if epistemic_val is not None:
                info += f"  MI={epistemic_val:.3f}"

        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                zorder=10)

        # Title with planner badge
        title = f"{ctype}\n{fault_str}"
        if active_planner:
            planner_color = "#1565c0" if active_planner == "PDM" else "#e65100"
            title += f"  [{active_planner}]"
        ax.set_title(title, fontsize=9,
                     color=fault_color, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.set_facecolor("#fafafa")

        # Auto-scale
        all_x = [ego["x"], other["x"], px_ego, px_other]
        all_y = [ego["y"], other["y"], py_ego, py_other]
        pad = 6.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Collision Diagnostics — {n} collisions\n"
        "Solid = state_before | Dashed = projected +0.5s | Yellow = front bumper",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <collision_snapshots.json> [--max N]")
        sys.exit(1)

    json_path = sys.argv[1]
    max_plots = 20
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        max_plots = int(sys.argv[idx + 1])

    with open(json_path) as f:
        snapshots = json.load(f)

    print(f"Loaded {len(snapshots)} collision snapshots")

    # Separate at-fault and not-at-fault
    at_fault = [s for s in snapshots if s["at_fault"]]
    not_fault = [s for s in snapshots if not s["at_fault"]]
    print(f"  At-fault: {len(at_fault)}")
    print(f"  Not at-fault: {len(not_fault)}")

    # Count by collision type
    from collections import Counter
    types = Counter(s["collision_type"] for s in snapshots)
    for t, c in types.most_common():
        fault_count = sum(1 for s in snapshots if s["collision_type"] == t and s["at_fault"])
        print(f"  {t}: {c} total, {fault_count} at-fault")

    # Plot not-at-fault collisions (those are the suspicious ones)
    out_dir = os.path.dirname(json_path)
    if not_fault:
        out_path = os.path.join(out_dir, "collision_not_at_fault.png")
        plot_snapshots(not_fault, out_path, max_plots=max_plots)

    if at_fault:
        out_path = os.path.join(out_dir, "collision_at_fault.png")
        plot_snapshots(at_fault, out_path, max_plots=min(max_plots, 8))


if __name__ == "__main__":
    main()
