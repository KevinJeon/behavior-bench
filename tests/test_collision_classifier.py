# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for at-fault collision classification (PDM scorer logic).

Tests classify_collision(), is_at_fault(), ego_crossed_lane_divider(),
and find_colliding_entity() with synthetic entity dicts.
Generates a visual overview plot of all 5 CollisionType scenarios
(each with at-fault and not-at-fault variant), showing timesteps
leading up to the collision.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pytest

from pufferlib.evaluation.collision_classifier import (
    CollisionType,
    classify_collision,
    find_colliding_entity,
    is_at_fault,
    ego_crossed_lane_divider,
    ego_has_lateral_drift,
    ego_heading_misaligned,
    min_point_to_polyline_dist,
    _get_box_corners,
    _is_agent_behind,
    _get_agent_relative_angle,
    STOPPED_SPEED_THRESHOLD,
    ROAD_LANE_TYPE,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_entity(x, y, heading, vx, vy, length=4.0, width=2.0, removed=0):
    return dict(
        x=x, y=y, heading=heading,
        vx=vx, vy=vy,
        length=length, width=width,
        removed=removed,
    )


def make_lane(traj_x, traj_y):
    """Create a lane entity (type=4) from trajectory points."""
    return dict(
        type=ROAD_LANE_TYPE,
        traj_x=list(traj_x),
        traj_y=list(traj_y),
        x=traj_x[0], y=traj_y[0],
        vx=0, vy=0, heading=0,
        length=0, width=0, removed=0,
    )


# ---------------------------------------------------------------------------
# Plotting utilities — timestep-based collision visualization
# ---------------------------------------------------------------------------

SCENARIOS = []  # (ego, other, ctype, fault, label, lanes_or_None)


def _record(ego, other, ctype, fault, label, lanes=None):
    SCENARIOS.append((dict(ego), dict(other), ctype, fault, label, lanes))


def _advance_entity(e, t):
    """Return a copy of entity advanced by t seconds."""
    return dict(
        e,
        x=e["x"] + e.get("vx", 0) * t,
        y=e["y"] + e.get("vy", 0) * t,
    )


def _draw_box(ax, x, y, length, width, heading, color, alpha,
              linewidth=0.8, zorder=3, linestyle="-"):
    """Draw a single rotated rectangle."""
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    dx_l, dy_l = length / 2 * cos_h, length / 2 * sin_h
    dx_w, dy_w = width / 2 * (-sin_h), width / 2 * cos_h
    corners = np.array([
        [x - dx_l - dx_w, y - dy_l - dy_w],
        [x + dx_l - dx_w, y + dy_l - dy_w],
        [x + dx_l + dx_w, y + dy_l + dy_w],
        [x - dx_l + dx_w, y - dy_l + dy_w],
    ])
    poly = plt.Polygon(corners, closed=True, facecolor=color, edgecolor="black",
                        alpha=alpha, linewidth=linewidth, zorder=zorder,
                        linestyle=linestyle)
    ax.add_patch(poly)


def _draw_front_bumper(ax, entity, color="#ffd600", linewidth=2.5):
    """Draw the front bumper line used for classification."""
    corners = _get_box_corners(
        entity["x"], entity["y"], entity["heading"],
        entity["length"], entity["width"],
    )
    fl, fr = corners[0], corners[1]
    ax.plot([fl[0], fr[0]], [fl[1], fr[1]], color=color, linewidth=linewidth,
            linestyle="-", zorder=7, alpha=0.9)


def _draw_timesteps(ax, ego_orig, other_orig, num_steps=5, dt=0.2):
    """Draw vehicles at multiple timesteps leading to collision.

    Steps -num_steps..-1 are 'before' (fading in),
    step 0 is collision moment (solid).
    """
    ego_color = "#42a5f5"
    other_color = "#ef5350"

    # Collect all positions for auto-scaling
    all_x, all_y = [], []

    for step in range(-num_steps, 1):
        t = step * dt
        ego_t = _advance_entity(ego_orig, t)
        other_t = _advance_entity(other_orig, t)

        if step < 0:
            # Past timesteps — faded
            alpha = 0.15 + 0.12 * (num_steps + step)
            lw = 0.5
            ls = "--"
        else:
            # Collision moment — solid
            alpha = 0.85
            lw = 1.6
            ls = "-"

        _draw_box(ax, ego_t["x"], ego_t["y"],
                  ego_t["length"], ego_t["width"], ego_t["heading"],
                  ego_color, alpha, linewidth=lw, zorder=3 + step, linestyle=ls)
        _draw_box(ax, other_t["x"], other_t["y"],
                  other_t["length"], other_t["width"], other_t["heading"],
                  other_color, alpha, linewidth=lw, zorder=3 + step, linestyle=ls)

        for e in (ego_t, other_t):
            all_x.append(e["x"])
            all_y.append(e["y"])

        # Timestep label
        if step <= 0:
            label = f"t={step * dt:.1f}s" if step < 0 else "t=0 (collision)"
            mid_x = (ego_t["x"] + other_t["x"]) / 2
            mid_y = max(ego_t["y"], other_t["y"]) + 2.5
            ax.text(mid_x, mid_y, label, ha="center", va="bottom",
                    fontsize=5.5, color="#555", alpha=max(0.4, alpha),
                    zorder=2)

    # Draw front bumper on collision moment ego
    ego_col = _advance_entity(ego_orig, 0)
    _draw_front_bumper(ax, ego_col)

    # Vehicle labels
    ego_col = _advance_entity(ego_orig, 0)
    other_col = _advance_entity(other_orig, 0)
    ax.text(ego_col["x"], ego_col["y"] - ego_col["width"] * 0.8,
            "ego", ha="center", va="top", fontsize=7,
            fontweight="bold", color=ego_color, zorder=8)
    ax.text(other_col["x"], other_col["y"] - other_col["width"] * 0.8,
            "other", ha="center", va="top", fontsize=7,
            fontweight="bold", color=other_color, zorder=8)

    # Velocity arrows at collision moment
    for e, c in [(ego_col, ego_color), (other_col, other_color)]:
        speed = math.hypot(e.get("vx", 0), e.get("vy", 0))
        if speed > 0.1:
            scale = 0.35
            ax.annotate("", xy=(e["x"] + e["vx"] * scale, e["y"] + e["vy"] * scale),
                        xytext=(e["x"], e["y"]),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=2.0),
                        zorder=9)

    # Impact marker
    cx = (ego_col["x"] + other_col["x"]) / 2
    cy = (ego_col["y"] + other_col["y"]) / 2
    ax.plot(cx, cy, marker="*", markersize=16, color="#ff6f00",
            markeredgecolor="#e65100", markeredgewidth=0.8, zorder=10)

    return all_x, all_y


def _draw_lanes(ax, lanes, xlim):
    """Draw lane centerlines."""
    if not lanes:
        return
    for traj_x, traj_y in lanes:
        ax.plot(traj_x, traj_y, color="#bdbdbd", linewidth=1.5,
                linestyle="-", zorder=0, alpha=0.6)


def plot_all_scenarios(path):
    """Create a 2-column grid: at-fault (left) vs not-at-fault (right) per type."""
    n = len(SCENARIOS)
    if n == 0:
        return

    cols = 2  # at-fault | not-at-fault
    rows = 5  # 5 collision types
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4))

    for i, (ego_orig, other_orig, ctype, fault, label, lanes) in enumerate(SCENARIOS):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        fault_str = "AT FAULT" if fault else "NOT AT FAULT"
        fault_color = "#d32f2f" if fault else "#2e7d32"

        # Get collision type name
        ctype_name = ctype.name if isinstance(ctype, CollisionType) else str(ctype)

        # Draw timesteps leading to collision
        all_x, all_y = _draw_timesteps(ax, ego_orig, other_orig,
                                        num_steps=5, dt=0.15)

        # Draw lanes if present
        if lanes:
            lane_data = [(l["traj_x"], l["traj_y"]) for l in lanes
                         if l.get("type") == ROAD_LANE_TYPE]
            _draw_lanes(ax, lane_data, (min(all_x), max(all_x)))

        ax.set_title(f"{label}\n{fault_str}", fontsize=9,
                     color=fault_color, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, linestyle=":")
        ax.set_facecolor("#fafafa")

        # Auto-scale
        pad = 6.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    for j in range(n, rows * cols):
        axes[j // cols, j % cols].set_visible(False)

    fig.suptitle("PDM At-Fault Collision Classification\n"
                 "5 Timesteps before collision shown (fading in)",
                 fontsize=13, fontweight="bold")
    legend_text = (
        "Dashed = past positions  |  "
        "Solid = collision (t=0)  |  "
        "Yellow line = front bumper  |  "
        "\u2605 = impact point"
    )
    fig.text(0.5, 0.005, legend_text, ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved scenario overview: {path}")


# ===================================================================
# 1. The 5 CollisionType scenarios — at-fault + not-at-fault each
# ===================================================================

class TestFiveCollisionTypes:
    """One at-fault and one not-at-fault test per CollisionType."""

    # --- STOPPED_EGO_COLLISION ---

    def test_stopped_ego_not_at_fault(self):
        """Ego stationary, other drives into ego -> STOPPED_EGO, NOT at fault."""
        ego = make_entity(0, 0, 0, 0, 0)
        other = make_entity(-6, 0, 0, 8, 0)
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.STOPPED_EGO_COLLISION
        fault = is_at_fault(ctype, ego, other)
        assert fault is False
        _record(ego, other, ctype, fault,
                "STOPPED_EGO\nEgo stationary, other hits ego")

    def test_stopped_ego_still_not_at_fault_with_lanes(self):
        """Ego stationary even in multiple lanes -> STOPPED_EGO, NOT at fault."""
        ego = make_entity(0, 1.75, 0, 0, 0)  # between two lanes
        other = make_entity(-6, 1.75, 0, 8, 0)
        lanes = [make_lane([0, 50], [0, 0]), make_lane([0, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.STOPPED_EGO_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is False  # Stopped ego is NEVER at fault

    # --- STOPPED_TRACK_COLLISION ---

    def test_stopped_track_at_fault(self):
        """Ego moving hits parked vehicle -> STOPPED_TRACK, AT FAULT."""
        ego = make_entity(0, 0, 0, 8, 0)
        other = make_entity(6, 0, 0, 0, 0)
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.STOPPED_TRACK_COLLISION
        fault = is_at_fault(ctype, ego, other)
        assert fault is True
        _record(ego, other, ctype, fault,
                "STOPPED_TRACK\nEgo hits parked vehicle")

    def test_stopped_track_always_at_fault(self):
        """Ego moving hits stationary, even if other is behind -> STOPPED_TRACK, always AT FAULT."""
        # Other is behind ego but stopped — ego still at fault for STOPPED_TRACK
        ego = make_entity(0, 0, math.pi, -5, 0)  # backing up
        other = make_entity(3, 0, 0, 0, 0)  # parked ahead (behind ego's heading)
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.STOPPED_TRACK_COLLISION
        fault = is_at_fault(ctype, ego, other)
        assert fault is True

    # --- ACTIVE_FRONT_COLLISION ---

    def test_active_front_at_fault(self):
        """Both moving, ego front bumper hits other -> ACTIVE_FRONT, AT FAULT."""
        ego = make_entity(0, 0, 0, 10, 0)
        other = make_entity(3, 0.5, math.pi, -3, 0)  # close enough for bumper contact
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_FRONT_COLLISION
        fault = is_at_fault(ctype, ego, other)
        assert fault is True
        _record(ego, other, ctype, fault,
                "ACTIVE_FRONT\nEgo front bumper hits other")

    def test_active_front_always_at_fault(self):
        """Front collision is always at-fault, even if ego is in own lane."""
        ego = make_entity(10, 0, 0, 10, 0)
        other = make_entity(13, 0, math.pi, 5, 0)  # head-on, close contact
        lanes = [make_lane([0, 50], [0, 0]), make_lane([0, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_FRONT_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is True  # ACTIVE_FRONT is ALWAYS at fault

    # --- ACTIVE_REAR_COLLISION ---

    def test_active_rear_not_at_fault(self):
        """Other rear-ends ego, ego in own lane -> ACTIVE_REAR, NOT at fault."""
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(-5, 0, 0, 12, 0)
        lanes = [make_lane([-20, 50], [0, 0]), make_lane([-20, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_REAR_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is False
        _record(ego, other, ctype, fault,
                "ACTIVE_REAR\nOther rear-ends ego (in lane)",
                lanes=lanes)

    def test_active_rear_at_fault_lane_change(self):
        """Ego cuts off other by changing lanes -> ACTIVE_REAR, AT FAULT."""
        # Ego heading misaligned (0.25 rad), other directly behind
        ego = make_entity(0, 1.75, 0.25, 5, 2)
        other = make_entity(-6, 1.75, 0, 12, 0)  # directly behind ego
        lanes = [make_lane([-20, 50], [0, 0]), make_lane([-20, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_REAR_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is True
        _record(ego, other, ctype, fault,
                "ACTIVE_REAR\nEgo cuts off other (lane change)",
                lanes=lanes)

    # --- ACTIVE_LATERAL_COLLISION ---

    def test_active_lateral_not_at_fault(self):
        """Side collision, ego in own lane -> ACTIVE_LATERAL, NOT at fault."""
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(1, -5, math.pi / 2, 0, 8)
        lanes = [make_lane([-20, 50], [0, 0]), make_lane([-20, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_LATERAL_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is False
        _record(ego, other, ctype, fault,
                "ACTIVE_LATERAL\nSide collision (ego in lane)",
                lanes=lanes)

    def test_active_lateral_at_fault_lane_change(self):
        """Ego changes lane, side collision -> ACTIVE_LATERAL, AT FAULT."""
        # Ego at y=1.75 (straddling two lanes), other alongside in lane 1
        ego = make_entity(0, 1.75, 0, 5, -2)
        other = make_entity(-1, 0, 0, 5, 0)  # alongside, slightly behind
        lanes = [make_lane([-20, 50], [0, 0]), make_lane([-20, 50], [3.5, 3.5])]
        entities = [ego, other] + lanes
        ctype = classify_collision(ego, other)
        assert ctype == CollisionType.ACTIVE_LATERAL_COLLISION
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is True
        _record(ego, other, ctype, fault,
                "ACTIVE_LATERAL\nEgo changes lane into other",
                lanes=lanes)


# ===================================================================
# 2. classify_collision() detailed tests
# ===================================================================

class TestClassifyCollision:

    def test_stopped_ego(self):
        ego = make_entity(0, 0, 0, 0, 0)
        other = make_entity(-5, 0, math.pi, 5, 0)
        assert classify_collision(ego, other) == CollisionType.STOPPED_EGO_COLLISION

    def test_stopped_track(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(5, 0, 0, 0, 0)
        assert classify_collision(ego, other) == CollisionType.STOPPED_TRACK_COLLISION

    def test_active_rear(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(-5, 0, 0, 8, 0)
        assert classify_collision(ego, other) == CollisionType.ACTIVE_REAR_COLLISION

    def test_active_front(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(3, 0, math.pi, 3, 0)  # close enough for bumper contact
        assert classify_collision(ego, other) == CollisionType.ACTIVE_FRONT_COLLISION

    def test_active_lateral_right(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(1, -4, math.pi / 2, 0, 5)
        assert classify_collision(ego, other) == CollisionType.ACTIVE_LATERAL_COLLISION

    def test_active_lateral_left(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(1, 4, -math.pi / 2, 0, -5)
        assert classify_collision(ego, other) == CollisionType.ACTIVE_LATERAL_COLLISION

    def test_front_angled_heading(self):
        """Ego heading pi/4, other ahead in ego's local frame -> ACTIVE_FRONT."""
        h = math.pi / 4
        ego = make_entity(0, 0, h, 5 * math.cos(h), 5 * math.sin(h))
        # Place other close enough that front bumper intersects
        dist = 3.0
        other = make_entity(dist * math.cos(h), dist * math.sin(h),
                           h + math.pi, 3, 0)
        assert classify_collision(ego, other) == CollisionType.ACTIVE_FRONT_COLLISION

    def test_rear_angled_heading(self):
        """Ego heading pi/6, other behind in ego's local frame -> ACTIVE_REAR."""
        h = math.pi / 6
        cos_h, sin_h = math.cos(h), math.sin(h)
        ego = make_entity(0, 0, h, 5 * cos_h, 5 * sin_h)
        other = make_entity(-5 * cos_h, -5 * sin_h, h, 8 * cos_h, 8 * sin_h)
        assert classify_collision(ego, other) == CollisionType.ACTIVE_REAR_COLLISION

    def test_barely_stopped_ego(self):
        v = STOPPED_SPEED_THRESHOLD - 0.01
        ego = make_entity(0, 0, 0, v, 0)
        other = make_entity(5, 0, math.pi, 5, 0)
        assert classify_collision(ego, other) == CollisionType.STOPPED_EGO_COLLISION

    def test_barely_moving_ego(self):
        v_ego = STOPPED_SPEED_THRESHOLD + 0.01
        v_other = STOPPED_SPEED_THRESHOLD - 0.01
        ego = make_entity(0, 0, 0, v_ego, 0)
        other = make_entity(5, 0, 0, v_other, 0)
        assert classify_collision(ego, other) == CollisionType.STOPPED_TRACK_COLLISION


# ===================================================================
# 3. is_at_fault() tests
# ===================================================================

class TestIsAtFault:

    def test_active_front_always_fault(self):
        assert is_at_fault(CollisionType.ACTIVE_FRONT_COLLISION) is True

    def test_stopped_track_always_fault(self):
        assert is_at_fault(CollisionType.STOPPED_TRACK_COLLISION) is True

    def test_stopped_ego_never_fault(self):
        assert is_at_fault(CollisionType.STOPPED_EGO_COLLISION) is False

    def test_active_rear_aligned_not_fault(self):
        ego = make_entity(10, 0, 0, 5, 0)
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        entities = [ego, lane1, lane2]
        assert is_at_fault(CollisionType.ACTIVE_REAR_COLLISION, ego, entities=entities) is False

    def test_active_rear_misaligned_is_fault(self):
        ego = make_entity(10, 0.5, 0.3, 5, 2)
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        entities = [ego, lane1, lane2]
        assert is_at_fault(CollisionType.ACTIVE_REAR_COLLISION, ego, entities=entities) is True

    def test_lateral_in_multiple_lanes_is_fault(self):
        ego = make_entity(0, 1.5, 0, 5, -2, width=2.0)
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        entities = [ego, lane1, lane2]
        assert is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, ego, entities=entities) is True

    def test_lateral_in_single_lane_not_fault(self):
        ego = make_entity(10, 0, 0, 5, 0, width=2.0)
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [7, 7])
        entities = [ego, lane1, lane2]
        assert is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, ego, entities=entities) is False

    def test_lateral_without_entities_not_fault(self):
        ego = make_entity(0, 0, 0, 5, 0)
        assert is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, ego) is False

    def test_lateral_heading_misaligned_is_fault(self):
        ego = make_entity(10, 0.5, 0.3, 5, 2, width=2.0)
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        entities = [ego, lane1, lane2]
        assert is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, ego, entities=entities) is True

    def test_unknown_type_not_fault(self):
        assert is_at_fault("something_unknown") is False

    def test_backward_compat_string_active_front(self):
        assert is_at_fault("active_front") is True

    def test_backward_compat_string_stopped_ego(self):
        assert is_at_fault("stopped_ego") is False


# ===================================================================
# 4. Geometry helpers tests
# ===================================================================

class TestGeometryHelpers:

    def test_box_corners_axis_aligned(self):
        corners = _get_box_corners(0, 0, 0, 4, 2)
        assert abs(corners[0][0] - 2.0) < 1e-6  # front-left x
        assert abs(corners[0][1] - 1.0) < 1e-6  # front-left y
        assert abs(corners[1][0] - 2.0) < 1e-6  # front-right x
        assert abs(corners[1][1] - (-1.0)) < 1e-6

    def test_box_corners_rotated(self):
        corners = _get_box_corners(0, 0, math.pi / 2, 4, 2)
        assert abs(corners[0][0] - (-1.0)) < 1e-6
        assert abs(corners[0][1] - 2.0) < 1e-6

    def test_is_agent_behind_directly_behind(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(-5, 0, 0, 0, 0)
        assert _is_agent_behind(ego, other) is True

    def test_is_agent_behind_directly_ahead(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(5, 0, 0, 0, 0)
        assert _is_agent_behind(ego, other) is False

    def test_is_agent_behind_at_side(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(0, 5, 0, 0, 0)
        assert _is_agent_behind(ego, other) is False

    def test_relative_angle_ahead(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(5, 0, 0, 0, 0)
        angle = _get_agent_relative_angle(ego, other)
        assert abs(angle) < 0.01

    def test_relative_angle_behind(self):
        ego = make_entity(0, 0, 0, 5, 0)
        other = make_entity(-5, 0, 0, 0, 0)
        angle = _get_agent_relative_angle(ego, other)
        assert abs(angle - math.pi) < 0.01


# ===================================================================
# 5. ego_crossed_lane_divider() tests
# ===================================================================

class TestEgoCrossedLaneDivider:

    def test_centered_in_single_lane(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [5, 5])
        positions = [(5, 0), (7, 0), (9, 0), (11, 0)]
        assert ego_crossed_lane_divider(positions, 2.0, [lane1, lane2]) is False

    def test_straddling_two_lanes(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        positions = [(5, 0), (7, 0.5), (9, 1.2), (11, 1.75)]
        assert ego_crossed_lane_divider(positions, 2.0, [lane1, lane2]) is True

    def test_only_current_position_straddling(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        positions = [(5, 0), (7, 0), (9, 0), (11, 1.75)]
        assert ego_crossed_lane_divider(positions, 2.0, [lane1, lane2]) is True

    def test_past_position_straddling(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        positions = [(5, 0), (7, 1.75), (9, 3.0), (11, 3.5)]
        assert ego_crossed_lane_divider(positions, 2.0, [lane1, lane2]) is True

    def test_lanes_far_apart(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [10, 10])
        positions = [(5, 0), (7, 0.5), (9, 1.0), (11, 0.5)]
        assert ego_crossed_lane_divider(positions, 2.0, [lane1, lane2]) is False

    def test_no_lane_entities(self):
        vehicle = make_entity(20, 0, 0, 3, 0)
        vehicle["type"] = 1
        assert ego_crossed_lane_divider([(10, 0)], 2.0, [vehicle]) is False

    def test_single_position(self):
        lane1 = make_lane([0, 50], [0, 0])
        lane2 = make_lane([0, 50], [3.5, 3.5])
        assert ego_crossed_lane_divider([(10, 1.75)], 2.0, [lane1, lane2]) is True


# ===================================================================
# 6. ego_has_lateral_drift() tests
# ===================================================================

class TestEgoHasLateralDrift:

    def test_straight_no_drift(self):
        positions = [(0, 0), (5, 0), (10, 0), (15, 0)]
        assert ego_has_lateral_drift(positions, heading=0.0) is False

    def test_lane_change_drift(self):
        positions = [(0, 0), (5, 0.3), (10, 0.7), (15, 1.2)]
        assert ego_has_lateral_drift(positions, heading=0.0) is True

    def test_small_drift_below_threshold(self):
        positions = [(0, 0), (5, 0.05), (10, 0.1), (15, 0.05)]
        assert ego_has_lateral_drift(positions, heading=0.0) is False

    def test_angled_heading_straight(self):
        h = math.pi / 4
        cos_h, sin_h = math.cos(h), math.sin(h)
        positions = [(i * cos_h, i * sin_h) for i in range(4)]
        assert ego_has_lateral_drift(positions, heading=h) is False

    def test_angled_heading_with_drift(self):
        h = math.pi / 4
        cos_h, sin_h = math.cos(h), math.sin(h)
        lat_x, lat_y = -sin_h, cos_h
        positions = [
            (i * 5 * cos_h + i * 0.3 * lat_x,
             i * 5 * sin_h + i * 0.3 * lat_y)
            for i in range(4)
        ]
        assert ego_has_lateral_drift(positions, heading=h, threshold=0.5) is True

    def test_single_position(self):
        assert ego_has_lateral_drift([(0, 0)], heading=0.0) is False

    def test_empty_positions(self):
        assert ego_has_lateral_drift([], heading=0.0) is False


# ===================================================================
# 7. min_point_to_polyline_dist() tests
# ===================================================================

class TestMinPointToPolylineDist:

    def test_point_on_segment(self):
        dist = min_point_to_polyline_dist(5, 0, [0, 10], [0, 0])
        assert abs(dist) < 0.01

    def test_point_perpendicular(self):
        dist = min_point_to_polyline_dist(5, 3, [0, 10], [0, 0])
        assert abs(dist - 3.0) < 0.01

    def test_point_beyond_segment(self):
        dist = min_point_to_polyline_dist(15, 0, [0, 10], [0, 0])
        assert abs(dist - 5.0) < 0.01

    def test_multi_segment(self):
        dist = min_point_to_polyline_dist(5, 2, [0, 5, 10], [0, 0, 0])
        assert abs(dist - 2.0) < 0.01


# ===================================================================
# 8. find_colliding_entity() tests
# ===================================================================

class TestFindCollidingEntity:

    def _entities(self, *args):
        return [make_entity(x, y, 0, 0, 0) for x, y in args]

    def test_finds_closest_active(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = self._entities((3, 0), (10, 0), (5, 0))
        result = find_colliding_entity(ego, entities, [0, 1, 2], [])
        assert result is not None
        assert result["x"] == 3

    def test_finds_closest_static(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = self._entities((10, 0), (2, 0))
        result = find_colliding_entity(ego, entities, [], [0, 1])
        assert result is not None
        assert result["x"] == 2

    def test_skips_removed(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = [
            make_entity(2, 0, 0, 0, 0, removed=1),
            make_entity(5, 0, 0, 0, 0),
        ]
        result = find_colliding_entity(ego, entities, [0, 1], [])
        assert result is not None
        assert result["x"] == 5

    def test_skips_invalid_position(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = [
            make_entity(-10000, 0, 0, 0, 0),
            make_entity(5, 0, 0, 0, 0),
        ]
        result = find_colliding_entity(ego, entities, [0, 1], [])
        assert result is not None
        assert result["x"] == 5

    def test_no_match_beyond_max_dist(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = self._entities((20, 0), (30, 0))
        result = find_colliding_entity(ego, entities, [0, 1], [])
        assert result is None

    def test_skips_ego_itself(self):
        ego = make_entity(5, 5, 0, 0, 0)
        entities = [
            make_entity(5, 5, 0, 0, 0),
            make_entity(8, 5, 0, 0, 0),
        ]
        result = find_colliding_entity(ego, entities, [0, 1], [])
        assert result is not None
        assert result["x"] == 8

    def test_empty_indices(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = self._entities((2, 0), (5, 0))
        result = find_colliding_entity(ego, entities, [], [])
        assert result is None

    def test_index_out_of_range(self):
        ego = make_entity(0, 0, 0, 0, 0)
        entities = self._entities((2, 0))
        result = find_colliding_entity(ego, entities, [0, 99], [])
        assert result is not None
        assert result["x"] == 2


# ===================================================================
# 9. Integration: full evaluator pipeline
# ===================================================================

class TestFullPipeline:

    @staticmethod
    def _two_lanes():
        return [
            make_lane([0, 50], [0, 0]),
            make_lane([0, 50], [3.5, 3.5]),
        ]

    @pytest.mark.parametrize("scenario,expected_fault", [
        ("ego_hits_parked", True),
        ("ego_rear_ended", False),
        ("ego_stopped_hit", False),
        ("ego_front_moving", True),
        ("ego_lane_change", True),
        ("parallel_scrape", False),
    ])
    def test_pipeline(self, scenario, expected_fault):
        ego, entities, active, static = self._build_scenario(scenario)

        other = find_colliding_entity(ego, entities, active, static)
        assert other is not None, f"No colliding entity found for {scenario}"

        ctype = classify_collision(ego, other)
        fault = is_at_fault(ctype, ego, other, entities=entities)
        assert fault is expected_fault, (
            f"Scenario '{scenario}': expected fault={expected_fault}, "
            f"got {fault} (type={ctype.name})"
        )

    def _build_scenario(self, name):
        lanes = self._two_lanes()

        if name == "ego_hits_parked":
            ego = make_entity(0, 0, 0, 8, 0)
            other = make_entity(5, 0, 0, 0, 0)
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        if name == "ego_rear_ended":
            ego = make_entity(0, 0, 0, 5, 0)
            other = make_entity(-4, 0, 0, 10, 0)
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        if name == "ego_stopped_hit":
            ego = make_entity(0, 0, 0, 0, 0)
            other = make_entity(4, 0, math.pi, 8, 0)
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        if name == "ego_front_moving":
            ego = make_entity(0, 0, 0, 10, 0)
            other = make_entity(3, 0.5, math.pi, -3, 0)  # close contact
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        if name == "ego_lane_change":
            ego = make_entity(0, 1.75, 0, 5, -3)
            other = make_entity(1, 3.5, 0, 5, 0)
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        if name == "parallel_scrape":
            ego = make_entity(0, 0, 0, 5, 0)
            other = make_entity(0.5, 3.5, 0, 5, 0)
            entities = [ego, other] + lanes
            return ego, entities, [1], []

        raise ValueError(f"Unknown scenario: {name}")


# ===================================================================
# Generate overview plot
# ===================================================================

def test_generate_overview_plot():
    """Generate the 5x2 grid: at-fault + not-at-fault for each CollisionType."""
    SCENARIOS.clear()

    lanes = [make_lane([-20, 50], [0, 0]), make_lane([-20, 50], [3.5, 3.5])]

    # Row 1: STOPPED_EGO (not at fault) | (still not at fault even between lanes)
    ego = make_entity(0, 0, 0, 0, 0)
    other = make_entity(-6, 0, 0, 8, 0)
    ctype = classify_collision(ego, other)
    _record(ego, other, ctype, is_at_fault(ctype, ego, other),
            "STOPPED_EGO\nEgo stationary, other hits ego")

    ego2 = make_entity(0, 1.75, 0, 0, 0)
    other2 = make_entity(-6, 1.75, 0, 8, 0)
    ctype2 = classify_collision(ego2, other2)
    _record(ego2, other2, ctype2,
            is_at_fault(ctype2, ego2, other2, entities=[ego2, other2] + lanes),
            "STOPPED_EGO\nEgo stopped (between lanes)")

    # Row 2: STOPPED_TRACK (at fault) | (at fault even when backing up)
    ego = make_entity(0, 0, 0, 8, 0)
    other = make_entity(6, 0, 0, 0, 0)
    ctype = classify_collision(ego, other)
    _record(ego, other, ctype, is_at_fault(ctype, ego, other),
            "STOPPED_TRACK\nEgo hits parked vehicle")

    ego2 = make_entity(0, 0, math.pi, -5, 0)
    other2 = make_entity(3, 0, 0, 0, 0)
    ctype2 = classify_collision(ego2, other2)
    _record(ego2, other2, ctype2, is_at_fault(ctype2, ego2, other2),
            "STOPPED_TRACK\nEgo backing into parked car")

    # Row 3: ACTIVE_FRONT (at fault) | (at fault, head-on)
    ego = make_entity(0, 0, 0, 10, 0)
    other = make_entity(3, 0.5, math.pi, -3, 0)
    ctype = classify_collision(ego, other)
    _record(ego, other, ctype, is_at_fault(ctype, ego, other),
            "ACTIVE_FRONT\nEgo front bumper hits other")

    ego2 = make_entity(0, 0, 0, 10, 0)
    other2 = make_entity(3, 0, math.pi, 5, 0)
    ctype2 = classify_collision(ego2, other2)
    _record(ego2, other2, ctype2, is_at_fault(ctype2, ego2, other2),
            "ACTIVE_FRONT\nHead-on collision")

    # Row 4: ACTIVE_REAR (not at fault) | (at fault — ego lane change)
    ego = make_entity(0, 0, 0, 5, 0)
    other = make_entity(-5, 0, 0, 12, 0)
    ctype = classify_collision(ego, other)
    _record(ego, other, ctype,
            is_at_fault(ctype, ego, other, entities=[ego, other] + lanes),
            "ACTIVE_REAR\nOther rear-ends ego (in lane)",
            lanes=lanes)

    ego2 = make_entity(0, 1.75, 0.25, 5, 2)
    other2 = make_entity(-6, 1.75, 0, 12, 0)
    ctype2 = classify_collision(ego2, other2)
    _record(ego2, other2, ctype2,
            is_at_fault(ctype2, ego2, other2, entities=[ego2, other2] + lanes),
            "ACTIVE_REAR\nEgo cuts off other (lane change)",
            lanes=lanes)

    # Row 5: ACTIVE_LATERAL (not at fault) | (at fault — ego lane change)
    ego = make_entity(0, 0, 0, 5, 0)
    other = make_entity(1, -5, math.pi / 2, 0, 8)
    ctype = classify_collision(ego, other)
    _record(ego, other, ctype,
            is_at_fault(ctype, ego, other, entities=[ego, other] + lanes),
            "ACTIVE_LATERAL\nSide collision (ego in lane)",
            lanes=lanes)

    ego2 = make_entity(0, 1.75, 0, 5, -2)
    other2 = make_entity(-1, 0, 0, 5, 0)
    ctype2 = classify_collision(ego2, other2)
    _record(ego2, other2, ctype2,
            is_at_fault(ctype2, ego2, other2, entities=[ego2, other2] + lanes),
            "ACTIVE_LATERAL\nEgo changes lane into other",
            lanes=lanes)

    plot_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "collision_classifier_tests.png"
    )
    plot_all_scenarios(os.path.abspath(plot_path))
    assert os.path.exists(os.path.abspath(plot_path)), "Plot was not saved"
