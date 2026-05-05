# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Converts nuPlan scenario data into PufferDrive's observation format.

Reimplements compute_observations() from drive.h (lines 1933-2112) in Python,
reading from nuPlan's EgoState, TrackedObjects, and Map API instead of C structs.

Observation layout (CLASSIC dynamics, 1120 floats):
  [0:7]     Ego features (goal_rel_x, goal_rel_y, speed, width, length, collision, respawn)
  [7:224]   Partner features (31 agents x 7 features)
  [224:1120] Road features (128 segments x 7 features)
"""

import logging
import math
from typing import List

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer

from pufferlib.ocean.drive.drive import simplify_polyline

log = logging.getLogger(__name__)

# Must match drive.h constants exactly
MAX_SPEED = 100.0
MAX_VEH_WIDTH = 15.0
MAX_VEH_LEN = 30.0
MAX_ROAD_SEGMENT_LENGTH = 100.0
MAX_ROAD_SCALE = 100.0

EGO_FEATURES = 7
PARTNER_FEATURES = 7
ROAD_FEATURES = 7
MAX_PARTNERS = 31  # MAX_AGENTS - 1
MAX_ROAD_SEGMENTS = 128
MAX_PARTNER_DIST_SQ = 2500.0  # 50m radius
MAP_QUERY_RADIUS = 250.0  # meters, must match ScenarioMax DEFAULT_MAP_EXTRACTION_RADIUS_METERS
ROAD_SEGMENT_DIST_SQ = 10000.0  # 100m radius for road segment filtering

# Spatial grid for road segment selection (matches drive.h)
GRID_CELL_SIZE = 5.0  # meters per cell
VISION_RANGE = 21  # 21x21 grid

OBS_SIZE = EGO_FEATURES + PARTNER_FEATURES * MAX_PARTNERS + ROAD_FEATURES * MAX_ROAD_SEGMENTS  # 1120

# PufferDrive road type values (entity->type - 4.0f)
ROAD_TYPE_LANE = 0.0       # ROAD_LANE = 4 -> 4-4=0 (centerlines)
ROAD_TYPE_LINE = 1.0       # ROAD_LINE = 5 -> 5-4=1 (lane dividers between lanes)
ROAD_TYPE_EDGE = 2.0       # ROAD_EDGE = 6 -> 6-4=2 (outer road boundaries)

# Agent types to include as partners
_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


def _rotate_to_ego_frame(
    dx: float, dy: float, cos_h: float, sin_h: float
) -> tuple[float, float]:
    """Rotate world-frame delta into ego-centric frame. Matches drive.h convention."""
    rel_x = dx * cos_h + dy * sin_h
    rel_y = -dx * sin_h + dy * cos_h
    return rel_x, rel_y


def _build_spiral_order(size: int = VISION_RANGE) -> list:
    """Build spiral order matching drive.h init_neighbor_offsets()."""
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    x, y = 0, 0
    direction = 0
    steps_to_take = 1
    steps_taken = 0
    segments_completed = 0
    max_offsets = size * size
    half = size // 2
    cells = [(half, half)]  # center
    while len(cells) < max_offsets:
        x += dx[direction]
        y += dy[direction]
        if abs(x) <= half and abs(y) <= half:
            cells.append((x + half, y + half))
        steps_taken += 1
        if steps_taken == steps_to_take:
            steps_taken = 0
            direction = (direction + 1) % 4
            segments_completed += 1
            if segments_completed % 2 == 0:
                steps_to_take += 1
    return cells


_SPIRAL_ORDER = _build_spiral_order()


def _simplify_nuplan_polyline(discrete_path, center_x=0.0, center_y=0.0):
    """Convert nuPlan discrete_path to dicts and simplify via Visvalingham-Whyatt.

    Uses the same simplify_polyline as PufferDrive's save_map_binary (threshold=0.1, max_len=250).
    Coordinates are centered before simplification (matching ScenarioMax) to avoid
    floating-point precision loss with large absolute coordinates.
    """
    if len(discrete_path) < 2:
        return [{"x": p.x, "y": p.y} for p in discrete_path]
    # Center coordinates for numerically stable simplification
    geometry = [{"x": p.x - center_x, "y": p.y - center_y, "z": 0.0} for p in discrete_path]
    # Only simplify if > 10 points (matching save_map_binary condition)
    if len(geometry) > 10:
        geometry = simplify_polyline(geometry, 0.1, 250)
    # Shift back to absolute coordinates
    for pt in geometry:
        pt["x"] += center_x
        pt["y"] += center_y
    return geometry


def _polyline_to_segments(
    polyline, ego_x: float, ego_y: float, type_val: float,
    max_dist_sq: float = ROAD_SEGMENT_DIST_SQ,
    bbox: list = None,
) -> list:
    """Convert a polyline (list of dicts with 'x', 'y') into road segments.

    If bbox is provided as [min_x, max_x, min_y, max_y], updates it with all
    polyline points (needed for global grid bounding box, matching drive.h).
    """
    segments = []
    for j in range(len(polyline) - 1):
        sx, sy = polyline[j]["x"], polyline[j]["y"]
        ex, ey = polyline[j + 1]["x"], polyline[j + 1]["y"]

        # Update bounding box with all points (matching init_grid_map)
        if bbox is not None:
            for px, py in ((sx, sy), (ex, ey)):
                if px < bbox[0]: bbox[0] = px
                if px > bbox[1]: bbox[1] = px
                if py < bbox[2]: bbox[2] = py
                if py > bbox[3]: bbox[3] = py

        mid_x = (sx + ex) * 0.5
        mid_y = (sy + ey) * 0.5

        rel_mx = mid_x - ego_x
        rel_my = mid_y - ego_y
        dist_sq = rel_mx * rel_mx + rel_my * rel_my

        if dist_sq > max_dist_sq:
            continue

        seg_dx = ex - mid_x
        seg_dy = ey - mid_y
        half_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

        if half_len > 0:
            dx_norm = seg_dx / half_len
            dy_norm = seg_dy / half_len
        else:
            dx_norm, dy_norm = 1.0, 0.0

        segments.append(
            (dist_sq, mid_x, mid_y, half_len, dx_norm, dy_norm, type_val)
        )
    return segments


class ObservationBuilder:
    """Builds PufferDrive observation vectors from nuPlan data."""

    def __init__(
        self,
        map_api: AbstractMap,
        route_roadblock_ids: List[str],
        goal: tuple[float, float],
    ):
        self._map_api = map_api
        self._route_roadblock_ids = route_roadblock_ids
        self._goal = goal
        self._logged_road_info = False

    def _get_goal(self) -> tuple[float, float]:
        """Return the goal position (last point of expert trajectory)."""
        return self._goal

    def _extract_road_segments(
        self, ego_x: float, ego_y: float
    ) -> tuple:
        """Extract road segments from nearby lanes and their boundaries.

        Uses ROADBLOCK/ROADBLOCK_CONNECTOR queries (like ScenarioMax) and
        subsamples dense polylines to get evenly-spaced segments.

        Returns:
            (segments, bbox) where bbox is [min_x, max_x, min_y, max_y] for
            global grid computation (matching drive.h init_grid_map).
        """
        ego_point = Point2D(ego_x, ego_y)
        segments = []
        bbox = [float('inf'), float('-inf'), float('inf'), float('-inf')]

        # Query ROADBLOCK/CONNECTOR for lanes + INTERSECTION for edge polygons
        road_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
        edge_layers = [SemanticMapLayer.INTERSECTION]

        try:
            proximal_objects = self._map_api.get_proximal_map_objects(
                ego_point, MAP_QUERY_RADIUS, road_layers + edge_layers
            )
        except Exception as e:
            log.warning("Failed to query map objects: %s", e)
            return segments

        # Get boundaries DB layer for boundary type classification
        try:
            boundaries_db = self._map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)
        except Exception:
            boundaries_db = None

        seen_lane_ids = set()
        seen_boundary_ids = set()
        num_lanes = 0
        block_polygons = []

        # Collect intersection polygons for edge boundary union
        for intersection in proximal_objects.get(SemanticMapLayer.INTERSECTION, []):
            try:
                block_polygons.append(intersection.polygon)
            except Exception:
                pass

        for layer in road_layers:
            is_roadblock = (layer == SemanticMapLayer.ROADBLOCK)
            for block in proximal_objects.get(layer, []):
                try:
                    edges = (
                        sorted(block.interior_edges, key=lambda lane: lane.index)
                        if is_roadblock
                        else block.interior_edges
                    )
                except Exception:
                    continue

                # Collect ROADBLOCK polygons for edge boundary union
                if is_roadblock:
                    try:
                        block_polygons.append(block.polygon)
                    except Exception:
                        pass

                for lane in edges:
                    # Deduplicate lanes that appear in multiple blocks
                    if lane.id in seen_lane_ids:
                        continue
                    seen_lane_ids.add(lane.id)
                    num_lanes += 1

                    # 1. Baseline path (centerline) -> ROAD_TYPE_LANE
                    try:
                        polyline = _simplify_nuplan_polyline(lane.baseline_path.discrete_path, ego_x, ego_y)
                        segments.extend(
                            _polyline_to_segments(polyline, ego_x, ego_y, ROAD_TYPE_LANE, bbox=bbox)
                        )
                    except Exception:
                        pass

                    # 2. Lane boundaries -> ROAD_TYPE_LINE (ROADBLOCK only, between adjacent lanes)
                    if is_roadblock and boundaries_db is not None:
                        try:
                            adjacent = lane.adjacent_edges
                            if adjacent[0] and adjacent[1]:
                                for boundary in [lane.left_boundary, lane.right_boundary]:
                                    if boundary.id in seen_boundary_ids:
                                        continue
                                    seen_boundary_ids.add(boundary.id)
                                    # Check boundary type via DB (filter UNKNOWN)
                                    try:
                                        btype_fid = int(
                                            boundaries_db.loc[[str(boundary.id)]]["boundary_type_fid"].iloc[0]
                                        )
                                    except Exception:
                                        continue  # skip unknown boundaries
                                    poly = _simplify_nuplan_polyline(boundary.discrete_path, ego_x, ego_y)
                                    segments.extend(
                                        _polyline_to_segments(poly, ego_x, ego_y, ROAD_TYPE_LINE, bbox=bbox)
                                    )
                        except Exception:
                            pass

        # 3. Road edge boundaries via polygon union -> ROAD_TYPE_EDGE
        #    Uses ROADBLOCK + INTERSECTION polygons (matches ScenarioMax)
        if block_polygons:
            try:
                merged = unary_union(block_polygons)
                boundary_geoms = gpd.GeoSeries(merged).boundary.explode(index_parts=True)
                for boundary_line in boundary_geoms[0]:
                    points = list(boundary_line.coords)
                    # Reverse point order and center coordinates (matching ScenarioMax)
                    polyline = [{"x": p[0] - ego_x, "y": p[1] - ego_y, "z": 0.0} for p in reversed(points)]
                    if len(polyline) > 10:
                        polyline = simplify_polyline(polyline, 0.1, 250)
                    # Shift back to absolute coordinates
                    for pt in polyline:
                        pt["x"] += ego_x
                        pt["y"] += ego_y
                    segments.extend(
                        _polyline_to_segments(polyline, ego_x, ego_y, ROAD_TYPE_EDGE, bbox=bbox)
                    )
            except Exception as e:
                log.warning("Failed to extract road edge boundaries: %s", e)

        if not self._logged_road_info:
            log.info(
                "Road feature extraction: %d lanes queried (radius=%.0fm), %d segments after simplification",
                num_lanes, MAP_QUERY_RADIUS, len(segments),
            )
            self._logged_road_info = True

        return segments, bbox

    def build(
        self,
        ego_state: EgoState,
        tracked_objects: TrackedObjects,
    ) -> np.ndarray:
        """Build PufferDrive observation vector from nuPlan data.

        Args:
            ego_state: Current ego vehicle state.
            tracked_objects: Detected surrounding agents.

        Returns:
            np.ndarray of shape (1120,) float32.
        """
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        # --- Ego state extraction ---
        # Use center position (PufferDrive uses vehicle center, not rear axle)
        ego_x = ego_state.center.x
        ego_y = ego_state.center.y
        heading = ego_state.center.heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        vx = ego_state.dynamic_car_state.center_velocity_2d.x
        vy = ego_state.dynamic_car_state.center_velocity_2d.y
        speed_mag = math.sqrt(vx * vx + vy * vy)
        v_dot_h = vx * cos_h + vy * sin_h
        signed_speed = math.copysign(speed_mag, v_dot_h)

        width = ego_state.car_footprint.width
        length = ego_state.car_footprint.length

        # --- Ego features ---
        goal_x, goal_y = self._get_goal()
        dx_goal = goal_x - ego_x
        dy_goal = goal_y - ego_y
        rel_goal_x, rel_goal_y = _rotate_to_ego_frame(dx_goal, dy_goal, cos_h, sin_h)

        obs[0] = rel_goal_x * 0.005
        obs[1] = rel_goal_y * 0.005
        obs[2] = signed_speed / MAX_SPEED
        obs[3] = width / MAX_VEH_WIDTH
        obs[4] = length / MAX_VEH_LEN
        obs[5] = 0.0  # collision_state (nuPlan handles separately)
        obs[6] = 0.0  # respawn_flag (not applicable)

        # --- Partner features ---
        obs_idx = EGO_FEATURES
        agents = tracked_objects.get_tracked_objects_of_types(_AGENT_TYPES)

        # Compute distances and sort
        agent_dists = []
        for agent in agents:
            ax = agent.box.center.x
            ay = agent.box.center.y
            dx = ax - ego_x
            dy = ay - ego_y
            dist_sq = dx * dx + dy * dy
            if dist_sq <= MAX_PARTNER_DIST_SQ:
                agent_dists.append((dist_sq, agent))

        agent_dists.sort(key=lambda x: x[0])

        cars_seen = 0
        for dist_sq, agent in agent_dists:
            if cars_seen >= MAX_PARTNERS:
                break

            dx = agent.box.center.x - ego_x
            dy = agent.box.center.y - ego_y
            rel_x, rel_y = _rotate_to_ego_frame(dx, dy, cos_h, sin_h)

            # Relative heading: cos(other - ego), sin(other - ego)
            other_h = agent.box.center.heading
            other_cos = math.cos(other_h)
            other_sin = math.sin(other_h)
            rel_heading_cos = other_cos * cos_h + other_sin * sin_h
            rel_heading_sin = other_sin * cos_h - other_cos * sin_h

            # Other agent's signed speed
            other_vx = agent.velocity.x
            other_vy = agent.velocity.y
            other_speed_mag = math.sqrt(other_vx * other_vx + other_vy * other_vy)
            other_v_dot_h = other_vx * other_cos + other_vy * other_sin
            other_signed_speed = math.copysign(other_speed_mag, other_v_dot_h)

            obs[obs_idx] = rel_x * 0.02
            obs[obs_idx + 1] = rel_y * 0.02
            obs[obs_idx + 2] = agent.box.width / MAX_VEH_WIDTH
            obs[obs_idx + 3] = agent.box.length / MAX_VEH_LEN
            obs[obs_idx + 4] = rel_heading_cos
            obs[obs_idx + 5] = rel_heading_sin
            obs[obs_idx + 6] = other_signed_speed / MAX_SPEED

            obs_idx += PARTNER_FEATURES
            cars_seen += 1

        # Zero-pad remaining partner slots (already zero from np.zeros)
        obs_idx = EGO_FEATURES + MAX_PARTNERS * PARTNER_FEATURES

        # --- Road features (global grid selection, matches drive.h) ---
        segments, bbox = self._extract_road_segments(ego_x, ego_y)

        # Build global grid matching drive.h init_grid_map():
        # Grid origin = (min_x, min_y), cell assignment via floor((x - origin) / GRID_CELL_SIZE)
        half = VISION_RANGE // 2
        selected = []
        if segments and bbox[0] < bbox[1]:
            grid_origin_x = bbox[0]  # top_left_x = min_x
            grid_origin_y = bbox[2]  # bottom_right_y = min_y
            # Assign segments to global grid cells
            grid = {}
            for seg in segments:
                _, mid_x, mid_y, half_len, dx_n, dy_n, type_val = seg
                gx = int((mid_x - grid_origin_x) / GRID_CELL_SIZE)
                gy = int((mid_y - grid_origin_y) / GRID_CELL_SIZE)
                key = (gx, gy)
                if key not in grid:
                    grid[key] = []
                grid[key].append(seg)

            # Find ego's cell in global grid
            ego_gx = int((ego_x - grid_origin_x) / GRID_CELL_SIZE)
            ego_gy = int((ego_y - grid_origin_y) / GRID_CELL_SIZE)

            # Collect segments using spiral offsets from ego's cell
            # (matching drive.h cache_neighbor_offsets: all segments from cell, then next)
            for cell_offset in _SPIRAL_ORDER:
                gx = ego_gx + (cell_offset[0] - half)
                gy = ego_gy + (cell_offset[1] - half)
                key = (gx, gy)
                if key not in grid:
                    continue
                for seg in grid[key]:
                    selected.append(seg)
                    if len(selected) >= MAX_ROAD_SEGMENTS:
                        break
                if len(selected) >= MAX_ROAD_SEGMENTS:
                    break

        for k, (_, mid_x, mid_y, half_len, dx_n, dy_n, type_val) in enumerate(
            selected
        ):
            rel_x = mid_x - ego_x
            rel_y = mid_y - ego_y
            x_obs, y_obs = _rotate_to_ego_frame(rel_x, rel_y, cos_h, sin_h)

            # Direction in ego frame
            cos_angle = dx_n * cos_h + dy_n * sin_h
            sin_angle = -dx_n * sin_h + dy_n * cos_h

            obs[obs_idx] = x_obs * 0.02
            obs[obs_idx + 1] = y_obs * 0.02
            obs[obs_idx + 2] = half_len / MAX_ROAD_SEGMENT_LENGTH
            obs[obs_idx + 3] = 0.1 / MAX_ROAD_SCALE  # width is always 0.1
            obs[obs_idx + 4] = cos_angle
            obs[obs_idx + 5] = sin_angle
            obs[obs_idx + 6] = type_val

            obs_idx += ROAD_FEATURES

        return obs
