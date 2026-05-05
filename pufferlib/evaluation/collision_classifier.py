# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""At-fault collision classification following nuPlan / PDM scorer logic.

Classifies collisions between ego and other agents into types:
- STOPPED_EGO_COLLISION: Ego is stationary -> not at fault
- STOPPED_TRACK_COLLISION: Other is stationary, ego moving -> at fault
- ACTIVE_FRONT_COLLISION: Ego front bumper hits other -> at fault
- ACTIVE_REAR_COLLISION: Other is behind ego -> not at fault
- ACTIVE_LATERAL_COLLISION: Side collision -> at fault only if ego is in multiple lanes

Reference:
  https://github.com/autonomousvision/tuplan_garage/blob/main/
    tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py
  https://github.com/autonomousvision/tuplan_garage/blob/main/
    tuplan_garage/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py
"""

import math
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

STOPPED_SPEED_THRESHOLD = 0.05  # m/s (matching nuPlan/PDM)
ROAD_LANE_TYPE = 4


class CollisionType(IntEnum):
    """Collision types matching nuPlan's CollisionType enum."""
    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


# String-to-enum mapping for backward compatibility
_STR_TO_COLLISION_TYPE = {
    "stopped_ego": CollisionType.STOPPED_EGO_COLLISION,
    "stopped_track": CollisionType.STOPPED_TRACK_COLLISION,
    "active_front": CollisionType.ACTIVE_FRONT_COLLISION,
    "active_rear": CollisionType.ACTIVE_REAR_COLLISION,
    "active_lateral": CollisionType.ACTIVE_LATERAL_COLLISION,
}


def _get_box_corners(x, y, heading, length, width):
    """Get 4 corners of a rotated bounding box.

    Returns corners in order: front-left, front-right, rear-right, rear-left.
    This matches the nuPlan/PDM convention where the front bumper edge
    is corners[0] to corners[1].
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    hl = length / 2
    hw = width / 2

    return [
        (x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h),   # front-left
        (x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h),   # front-right
        (x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h),   # rear-right
        (x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h),   # rear-left
    ]


def _get_agent_relative_angle(ego: Dict, other: Dict) -> float:
    """Angle between ego's heading direction and the vector to other.

    Matches nuPlan's ``get_agent_relative_angle`` from
    ``nuplan.planning.simulation.observation.idm.utils``.

    Returns angle in radians in [0, pi].
    """
    dx = other["x"] - ego["x"]
    dy = other["y"] - ego["y"]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0

    ego_heading = ego.get("heading", 0.0)
    ego_dir_x = math.cos(ego_heading)
    ego_dir_y = math.sin(ego_heading)

    dot = (ego_dir_x * dx + ego_dir_y * dy) / dist
    dot = max(-1.0, min(1.0, dot))
    return math.acos(dot)


def _is_agent_behind(ego: Dict, other: Dict, angle_tolerance_deg: float = 150.0) -> bool:
    """True if other agent is behind ego (angle > tolerance).

    Matches nuPlan's ``is_agent_behind`` with default 150 degree threshold.
    """
    return _get_agent_relative_angle(ego, other) > math.radians(angle_tolerance_deg)


def _closing_speed(ego: Dict, other: Dict) -> float:
    """Relative closing speed along the ego→other axis.

    Positive means ego is approaching other (closing the gap).
    """
    dx = other["x"] - ego["x"]
    dy = other["y"] - ego["y"]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return 0.0
    # Unit vector from ego to other
    ux, uy = dx / dist, dy / dist
    # Relative velocity of ego w.r.t. other, projected onto ego→other axis
    rel_vx = ego.get("vx", 0) - other.get("vx", 0)
    rel_vy = ego.get("vy", 0) - other.get("vy", 0)
    return rel_vx * ux + rel_vy * uy


def classify_collision(ego: Dict, other: Dict) -> CollisionType:
    """Classify collision type using approach-angle logic.

    Since we typically have ``state_before`` (one timestep before the actual
    collision) rather than the exact collision state, we use the approach
    angle instead of geometric (Shapely) overlap to determine front vs lateral.

    Priority order (matching ``get_collision_type`` from tuplan_garage):
    1. Ego stopped -> STOPPED_EGO_COLLISION
    2. Track stopped -> STOPPED_TRACK_COLLISION
    3. Other behind ego (angle > 150 deg) -> ACTIVE_REAR_COLLISION
    4. Other in front of ego (angle < 60 deg) + ego closing -> ACTIVE_FRONT_COLLISION
    5. Otherwise -> ACTIVE_LATERAL_COLLISION

    Args:
        ego: dict with keys x, y, vx, vy, heading, length, width
        other: dict with same keys

    Returns:
        CollisionType enum value
    """
    ego_speed = math.hypot(ego.get("vx", 0), ego.get("vy", 0))
    other_speed = math.hypot(other.get("vx", 0), other.get("vy", 0))

    # 1. Ego stopped
    if ego_speed <= STOPPED_SPEED_THRESHOLD:
        return CollisionType.STOPPED_EGO_COLLISION

    # 2. Track stopped
    if other_speed <= STOPPED_SPEED_THRESHOLD:
        return CollisionType.STOPPED_TRACK_COLLISION

    # 3. Other is behind ego
    if _is_agent_behind(ego, other):
        return CollisionType.ACTIVE_REAR_COLLISION

    # 4. Front collision: other is roughly in front of ego and ego is closing in
    angle = _get_agent_relative_angle(ego, other)
    if angle < math.radians(60) and _closing_speed(ego, other) > 0.5:
        return CollisionType.ACTIVE_FRONT_COLLISION

    # 5. Lateral collision
    return CollisionType.ACTIVE_LATERAL_COLLISION


def _point_to_segment_dist_sq(px, py, ax, ay, bx, by):
    """Squared distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return (px - proj_x) ** 2 + (py - proj_y) ** 2


def min_point_to_polyline_dist(px, py, traj_x, traj_y):
    """Minimum distance from a point to a polyline."""
    best_sq = float("inf")
    n = min(len(traj_x), len(traj_y))
    for i in range(n - 1):
        d_sq = _point_to_segment_dist_sq(
            px, py, traj_x[i], traj_y[i], traj_x[i + 1], traj_y[i + 1]
        )
        if d_sq < best_sq:
            best_sq = d_sq
    return math.sqrt(best_sq) if best_sq < float("inf") else float("inf")


def _position_in_multiple_lanes(
    x: float, y: float, width: float, lanes: list, lane_half_width: float,
) -> bool:
    """Check if a position overlaps multiple lane centerlines."""
    threshold = width / 2.0 + lane_half_width
    count = 0
    for traj_x, traj_y in lanes:
        dist = min_point_to_polyline_dist(x, y, traj_x, traj_y)
        if dist < threshold:
            count += 1
        if count >= 2:
            return True
    return False


def ego_crossed_lane_divider(
    ego_positions: List[tuple],
    ego_width: float,
    entities: List[Dict],
    lane_half_width: float = 1.75,
) -> bool:
    """Check if ego was in multiple lanes at any of the given positions.

    Args:
        ego_positions: list of (x, y) positions (most recent last)
        ego_width: ego vehicle width
        entities: all entities from env.get_state()
        lane_half_width: half-width of a lane in meters

    Returns:
        True if any position overlaps 2 or more lanes
    """
    # Extract lane polylines once
    lanes = []
    for e in entities:
        if e.get("type") != ROAD_LANE_TYPE:
            continue
        traj_x = e.get("traj_x")
        traj_y = e.get("traj_y")
        if not traj_x or not traj_y:
            continue
        lanes.append((traj_x, traj_y))

    if len(lanes) < 2:
        return False

    for x, y in ego_positions:
        if _position_in_multiple_lanes(x, y, ego_width, lanes, lane_half_width):
            return True
    return False


def _nearest_lane_heading(
    x: float, y: float, entities: List[Dict],
) -> Optional[float]:
    """Get the heading of the nearest lane segment at position (x, y).

    Returns the direction (in radians) of the closest lane segment,
    or None if no lanes are found.
    """
    best_dist_sq = float("inf")
    best_heading = None

    for e in entities:
        if e.get("type") != ROAD_LANE_TYPE:
            continue
        traj_x = e.get("traj_x")
        traj_y = e.get("traj_y")
        if not traj_x or not traj_y:
            continue
        n = min(len(traj_x), len(traj_y))
        for i in range(n - 1):
            ax, ay = traj_x[i], traj_y[i]
            bx, by = traj_x[i + 1], traj_y[i + 1]
            d_sq = _point_to_segment_dist_sq(x, y, ax, ay, bx, by)
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_heading = math.atan2(by - ay, bx - ax)

    return best_heading


def ego_heading_misaligned(
    ego: Dict,
    entities: List[Dict],
    angle_threshold: float = 0.08,  # ~4.6 degrees
) -> bool:
    """Check if ego's heading is misaligned with the nearest lane direction.

    If ego is driving at a significant angle to the lane, it's crossing
    the lane boundary and is at fault.

    Args:
        ego: ego entity dict with x, y, heading
        entities: all entities (for lane data)
        angle_threshold: minimum angle difference in radians (~0.1 = 5.7 deg)

    Returns:
        True if ego heading differs from lane direction by more than threshold
    """
    lane_heading = _nearest_lane_heading(ego["x"], ego["y"], entities)
    if lane_heading is None:
        return False
    ego_heading = ego.get("heading", 0.0)
    # Normalize angle difference to [-pi, pi]
    diff = ego_heading - lane_heading
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return abs(diff) > angle_threshold


def ego_has_lateral_drift(
    ego_position_history: List[tuple],
    heading: float,
    threshold: float = 0.3,
) -> bool:
    """Check if ego drifted laterally (perpendicular to a reference heading).

    Projects position history onto the lateral axis (perpendicular to heading).
    If the max lateral displacement exceeds threshold, ego was changing lanes.

    Args:
        ego_position_history: list of (x, y) positions (oldest first)
        heading: reference heading in radians (ideally the lane direction)
        threshold: lateral drift threshold in meters

    Returns:
        True if lateral drift exceeds threshold
    """
    if len(ego_position_history) < 2:
        return False

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Project all positions onto the lateral axis relative to first position
    x0, y0 = ego_position_history[0]
    lateral_values = []
    for x, y in ego_position_history:
        dx = x - x0
        dy = y - y0
        # Lateral component: perpendicular to heading
        lateral = -dx * sin_h + dy * cos_h
        lateral_values.append(lateral)

    # Check if there's significant lateral movement
    lat_range = max(lateral_values) - min(lateral_values)
    return lat_range > threshold


def is_at_fault(
    collision_type,
    ego: Dict = None,
    other: Dict = None,
    entities: List[Dict] = None,
    ego_position_history: List[tuple] = None,
) -> bool:
    """Determine if collision is ego's fault (PDM scorer logic).

    Matching ``_calculate_no_at_fault_collision`` from tuplan_garage:
    - ACTIVE_FRONT_COLLISION or STOPPED_TRACK_COLLISION -> always at fault
    - ACTIVE_LATERAL_COLLISION + ego in multiple lanes -> at fault
    - STOPPED_EGO_COLLISION -> not at fault
    - ACTIVE_REAR_COLLISION -> not at fault (unless ego changed lanes)
    - ACTIVE_LATERAL_COLLISION in own lane -> not at fault

    Args:
        collision_type: CollisionType enum or string
        ego: ego entity dict
        other: other entity dict
        entities: all entities (needed for lane check)
        ego_position_history: recent (x,y) positions

    Returns:
        True if collision is ego's fault
    """
    # Backward compat: convert string to enum
    if isinstance(collision_type, str):
        collision_type = _STR_TO_COLLISION_TYPE.get(collision_type)
        if collision_type is None:
            return False

    # Collision with VRU (pedestrian/cyclist): vehicle is always at fault
    if other is not None:
        other_type = other.get("type", 1)
        if other_type in (2, 3):  # PEDESTRIAN=2, CYCLIST=3
            return True

    # Always at fault
    if collision_type in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION):
        return True

    # Never at fault
    if collision_type == CollisionType.STOPPED_EGO_COLLISION:
        return False

    # Conditional: lateral or rear — at fault only if ego is in multiple lanes
    if collision_type in (CollisionType.ACTIVE_LATERAL_COLLISION, CollisionType.ACTIVE_REAR_COLLISION):
        if ego is None:
            return False
        positions = ego_position_history or [(ego["x"], ego["y"])]
        if entities is not None:
            if not ego_crossed_lane_divider(positions, ego.get("width", 2.0), entities):
                return False
            # Ego overlaps multiple lanes — additionally require actual lateral
            # drift to distinguish real lane changes from simply being near a
            # lane boundary (common in curves or narrow roads).
            ego_heading = ego.get("heading", 0.0)
            if not ego_has_lateral_drift(positions, ego_heading, threshold=0.3):
                return False
            return True

    return False


def find_colliding_entity(
    ego: Dict,
    entities: List[Dict],
    active_indices: List[int],
    static_indices: List[int],
    max_dist: float = 15.0,
    dt: float = 0.5,
) -> Optional[Dict]:
    """Find the entity that ego collided with.

    Since we use state_before (one step before collision), entities may not
    yet be touching. We project each candidate forward by dt using linear
    motion and pick the one with the smallest projected distance.

    Args:
        ego: ego entity dict with x, y, vx, vy, length, width
        entities: list of all entity dicts from env.get_state()
        active_indices: indices of active (controlled) agents
        static_indices: indices of static (expert replay) agents
        max_dist: maximum search radius in meters
        dt: projection time in seconds (one simulation step)

    Returns:
        Entity dict of the closest other agent, or None
    """
    ego_x = ego["x"] + ego.get("vx", 0) * dt
    ego_y = ego["y"] + ego.get("vy", 0) * dt
    best_dist = max_dist
    best_entity = None

    candidate_indices = list(active_indices or []) + list(static_indices or [])

    for idx in candidate_indices:
        if idx >= len(entities):
            continue
        e = entities[idx]
        if e.get("removed", 0) or e.get("x", -10000) < -9000:
            continue
        # Skip ego itself
        if abs(e["x"] - ego["x"]) < 0.01 and abs(e["y"] - ego["y"]) < 0.01:
            continue

        ex = e["x"] + e.get("vx", 0) * dt
        ey = e["y"] + e.get("vy", 0) * dt
        dist = math.hypot(ex - ego_x, ey - ego_y)
        if dist < best_dist:
            best_dist = dist
            best_entity = e

    return best_entity
