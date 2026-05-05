# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Convert nuPlan scenario data to PufferDrive .bin format.

Creates a binary file readable by drive.h's load_map_binary(), containing:
- Road segments (from ObservationBuilder._extract_road_segments)
- Ego vehicle trajectory (from scenario expert trajectory)
- Tracked objects at initial timestep

This enables using the C compute_observations() for exact 1:1 observation matching.
"""

import math
import struct
from typing import List, Tuple

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from .observation_builder import (
    ROAD_TYPE_EDGE,
    ROAD_TYPE_LANE,
    ROAD_TYPE_LINE,
    ObservationBuilder,
    _simplify_nuplan_polyline,
)


# Must match drive.h entity types
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6

TRAJECTORY_LENGTH = 151

_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]

_TYPE_MAP = {
    TrackedObjectType.VEHICLE: VEHICLE,
    TrackedObjectType.PEDESTRIAN: PEDESTRIAN,
    TrackedObjectType.BICYCLE: CYCLIST,
}

_ROAD_OBS_TO_BIN = {
    ROAD_TYPE_LANE: ROAD_LANE,
    ROAD_TYPE_LINE: ROAD_LINE,
    ROAD_TYPE_EDGE: ROAD_EDGE,
}


def nuplan_to_binary(
    scenario: "AbstractScenario",
    map_api: "AbstractMap",
    route_roadblock_ids: List[str],
    output_path: str,
) -> str:
    """Convert a nuPlan scenario to PufferDrive .bin format.

    Args:
        scenario: nuPlan scenario
        map_api: nuPlan map API
        route_roadblock_ids: Route roadblock IDs
        output_path: Path to write the .bin file

    Returns:
        output_path
    """
    ego_state = scenario.initial_ego_state
    ego_x = ego_state.center.x
    ego_y = ego_state.center.y

    # Goal = last position of the expert ego trajectory
    expert_traj = list(scenario.get_expert_ego_trajectory())
    last_state = expert_traj[-1]
    goal = (last_state.center.x, last_state.center.y)

    # --- Extract road polylines (matching ScenarioMax exactly) ---
    obs_builder = ObservationBuilder(map_api, route_roadblock_ids, goal)
    road_polylines = _extract_road_polylines(obs_builder, ego_x, ego_y)

    # --- Extract ego trajectory ---
    ego_trajectory = _extract_ego_trajectory(scenario)

    # --- Extract initial tracked objects ---
    initial_objects = _extract_tracked_objects(scenario)

    # --- Write binary ---
    _write_binary(output_path, ego_trajectory, initial_objects, road_polylines)

    return output_path


def _extract_road_polylines(
    obs_builder: ObservationBuilder, ego_x: float, ego_y: float
) -> list:
    """Extract road polylines with their types for binary writing.

    Returns list of (drive_type, polyline_points) tuples.
    Each polyline_points is a list of (x, y, z) tuples.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    from pufferlib.ocean.drive.drive import simplify_polyline

    ego_point = Point2D(ego_x, ego_y)
    from .observation_builder import MAP_QUERY_RADIUS

    road_layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    edge_layers = [SemanticMapLayer.INTERSECTION]

    proximal_objects = obs_builder._map_api.get_proximal_map_objects(
        ego_point, MAP_QUERY_RADIUS, road_layers + edge_layers
    )

    try:
        boundaries_db = obs_builder._map_api._get_vector_map_layer(
            SemanticMapLayer.BOUNDARIES
        )
    except Exception:
        boundaries_db = None

    roads = []
    seen_lane_ids = set()
    seen_boundary_ids = set()
    block_polygons = []

    # Intersection polygons
    for intersection in proximal_objects.get(SemanticMapLayer.INTERSECTION, []):
        try:
            block_polygons.append(intersection.polygon)
        except Exception:
            pass

    for layer in road_layers:
        is_roadblock = layer == SemanticMapLayer.ROADBLOCK
        for block in proximal_objects.get(layer, []):
            try:
                edges = (
                    sorted(block.interior_edges, key=lambda lane: lane.index)
                    if is_roadblock
                    else block.interior_edges
                )
            except Exception:
                continue

            if is_roadblock:
                try:
                    block_polygons.append(block.polygon)
                except Exception:
                    pass

            for lane in edges:
                if lane.id in seen_lane_ids:
                    continue
                seen_lane_ids.add(lane.id)

                # Centerline -> ROAD_LANE
                try:
                    polyline = _simplify_nuplan_polyline(
                        lane.baseline_path.discrete_path, ego_x, ego_y
                    )
                    pts = [(p["x"], p["y"], 0.0) for p in polyline]
                    if len(pts) >= 2:
                        roads.append((ROAD_LANE, pts))
                except Exception:
                    pass

                # Boundaries -> ROAD_LINE
                if is_roadblock and boundaries_db is not None:
                    try:
                        adjacent = lane.adjacent_edges
                        if adjacent[0] and adjacent[1]:
                            for boundary in [lane.left_boundary, lane.right_boundary]:
                                if boundary.id in seen_boundary_ids:
                                    continue
                                seen_boundary_ids.add(boundary.id)
                                try:
                                    int(
                                        boundaries_db.loc[[str(boundary.id)]][
                                            "boundary_type_fid"
                                        ].iloc[0]
                                    )
                                except Exception:
                                    continue
                                poly = _simplify_nuplan_polyline(
                                    boundary.discrete_path, ego_x, ego_y
                                )
                                pts = [(p["x"], p["y"], 0.0) for p in poly]
                                if len(pts) >= 2:
                                    roads.append((ROAD_LINE, pts))
                    except Exception:
                        pass

    # Edge boundaries via polygon union -> ROAD_EDGE
    if block_polygons:
        try:
            merged = unary_union(block_polygons)
            boundary_geoms = gpd.GeoSeries(merged).boundary.explode(
                index_parts=True
            )
            for boundary_line in boundary_geoms[0]:
                points = list(boundary_line.coords)
                polyline = [
                    {"x": p[0] - ego_x, "y": p[1] - ego_y, "z": 0.0}
                    for p in reversed(points)
                ]
                if len(polyline) > 10:
                    polyline = simplify_polyline(polyline, 0.1, 250)
                for pt in polyline:
                    pt["x"] += ego_x
                    pt["y"] += ego_y
                pts = [(p["x"], p["y"], 0.0) for p in polyline]
                if len(pts) >= 2:
                    roads.append((ROAD_EDGE, pts))
        except Exception:
            pass

    # Split any road with too many points into segments of max MAX_ROAD_PTS.
    # Waymo roads have <=33 pts; the C code allocates based on array_size so
    # very large polylines cause heap corruption on resample.
    MAX_ROAD_PTS = 50
    split_roads = []
    for rtype, pts in roads:
        while len(pts) > MAX_ROAD_PTS:
            split_roads.append((rtype, pts[:MAX_ROAD_PTS]))
            pts = pts[MAX_ROAD_PTS - 1:]  # overlap by 1 for continuity
        if len(pts) >= 2:
            split_roads.append((rtype, pts))

    return split_roads


def _extract_ego_trajectory(scenario: "AbstractScenario") -> dict:
    """Extract ego trajectory in GPUDrive object format.

    nuPlan runs at 20Hz (dt=0.05s) but PufferDrive uses 10Hz (dt=0.1s),
    so we subsample every 2nd nuPlan iteration.
    """
    n_iter = scenario.get_number_of_iterations()
    positions = []
    velocities = []
    headings = []
    valids = []

    # Subsample: take every 2nd nuPlan iteration (20Hz → 10Hz)
    for i in range(0, min(n_iter, 2 * TRAJECTORY_LENGTH), 2):
        state = scenario.get_ego_state_at_iteration(i)
        positions.append(
            {"x": state.center.x, "y": state.center.y, "z": 0.0}
        )
        # nuPlan velocity is in body frame — convert to global frame
        h = state.center.heading
        vx_body = state.dynamic_car_state.center_velocity_2d.x
        vy_body = state.dynamic_car_state.center_velocity_2d.y
        cos_h = math.cos(h)
        sin_h = math.sin(h)
        velocities.append(
            {
                "x": vx_body * cos_h - vy_body * sin_h,
                "y": vx_body * sin_h + vy_body * cos_h,
                "z": 0.0,
            }
        )
        headings.append(h)
        valids.append(1)

    # Pad to TRAJECTORY_LENGTH
    while len(positions) < TRAJECTORY_LENGTH:
        positions.append(positions[-1] if positions else {"x": 0, "y": 0, "z": 0})
        velocities.append({"x": 0, "y": 0, "z": 0})
        headings.append(headings[-1] if headings else 0.0)
        valids.append(0)

    ego_state = scenario.initial_ego_state
    expert_traj = list(scenario.get_expert_ego_trajectory())
    goal_state = expert_traj[-1]

    return {
        "type": VEHICLE,
        "id": 0,
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "valids": valids,
        "width": ego_state.car_footprint.width,
        "length": ego_state.car_footprint.length,
        "height": 1.5,
        "goal_x": goal_state.center.x,
        "goal_y": goal_state.center.y,
        "mark_as_expert": 1,
    }


def _extract_tracked_objects(scenario: "AbstractScenario") -> list:
    """Extract all tracked objects at initial timestep as GPUDrive objects.

    Writes all agents (no limit) so that world_mean matches the training binary.
    The observation only shows the closest MAX_OBS_PARTNERS (31), but all agents
    contribute to the world_mean centering and grid computation.
    """
    objects = []
    tracked = scenario.initial_tracked_objects.tracked_objects
    agents = tracked.get_tracked_objects_of_types(_AGENT_TYPES)

    for agent in agents:
        obj_type = _TYPE_MAP.get(agent.tracked_object_type, VEHICLE)

        positions = [
            {"x": agent.box.center.x, "y": agent.box.center.y, "z": 0.0}
        ]
        # Agent velocities are already in global frame
        velocities = [{"x": agent.velocity.x, "y": agent.velocity.y, "z": 0.0}]
        headings = [agent.box.center.heading]
        valids = [1]

        # Pad
        while len(positions) < TRAJECTORY_LENGTH:
            positions.append(positions[-1])
            velocities.append({"x": 0, "y": 0, "z": 0})
            headings.append(headings[-1])
            valids.append(0)

        # Set goal far enough from start to pass MIN_DISTANCE_TO_GOAL (2.0m) filter
        # in set_active_agents(). Project 10m in heading direction.
        heading = agent.box.center.heading
        goal_x = agent.box.center.x + 10.0 * math.cos(heading)
        goal_y = agent.box.center.y + 10.0 * math.sin(heading)

        objects.append(
            {
                "type": obj_type,
                "id": len(objects) + 1,
                "positions": positions,
                "velocities": velocities,
                "headings": headings,
                "valids": valids,
                "width": agent.box.width,
                "length": agent.box.length,
                "height": agent.box.height if hasattr(agent.box, "height") else 1.5,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "mark_as_expert": 0,
            }
        )

    return objects


def _write_binary(
    output_path: str,
    ego: dict,
    objects: list,
    roads: list,
) -> None:
    """Write PufferDrive .bin format matching save_map_binary."""
    all_objects = [ego] + objects
    num_objects = len(all_objects)
    num_roads = len(roads)

    with open(output_path, "wb") as f:
        # Metadata
        f.write(struct.pack("i", 0))  # sdc_track_index = 0 (ego is first object)
        f.write(struct.pack("i", 0))  # num_tracks_to_predict = 0

        # Entity counts
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))

        # Write objects (ego + tracked agents)
        for obj in all_objects:
            f.write(struct.pack("i", 0))  # scenario_id / unique_map_id
            f.write(struct.pack("i", obj["type"]))
            f.write(struct.pack("i", 0))  # id
            f.write(struct.pack("i", TRAJECTORY_LENGTH))  # array_size

            # Position arrays (x, y, z)
            positions = obj["positions"]
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    f.write(struct.pack("f", float(positions[i].get(coord, 0.0))))

            # Velocity arrays (vx, vy, vz)
            velocities = obj["velocities"]
            for coord in ["x", "y", "z"]:
                for i in range(TRAJECTORY_LENGTH):
                    f.write(struct.pack("f", float(velocities[i].get(coord, 0.0))))

            # Heading array
            headings = obj["headings"]
            for i in range(TRAJECTORY_LENGTH):
                f.write(struct.pack("f", float(headings[i])))

            # Valid array
            valids = obj["valids"]
            for i in range(TRAJECTORY_LENGTH):
                f.write(struct.pack("i", int(valids[i])))

            # Scalar fields
            f.write(struct.pack("f", float(obj["width"])))
            f.write(struct.pack("f", float(obj["length"])))
            f.write(struct.pack("f", float(obj.get("height", 1.5))))
            f.write(struct.pack("f", float(obj.get("goal_x", 0.0))))
            f.write(struct.pack("f", float(obj.get("goal_y", 0.0))))
            f.write(struct.pack("f", 0.0))  # goal_z
            f.write(struct.pack("i", int(obj.get("mark_as_expert", 0))))

            # Exit lanes (none for objects)
            f.write(struct.pack("i", 0))

        # Write roads
        for road_type, points in roads:
            f.write(struct.pack("i", 0))  # scenario_id
            f.write(struct.pack("i", road_type))
            f.write(struct.pack("i", 0))  # id
            f.write(struct.pack("i", len(points)))  # array_size

            # Position arrays
            for coord_idx in range(3):  # x, y, z
                for pt in points:
                    f.write(struct.pack("f", float(pt[coord_idx])))

            # Scalar fields
            f.write(struct.pack("f", 0.0))  # width
            f.write(struct.pack("f", 0.0))  # length
            f.write(struct.pack("f", 0.0))  # height
            f.write(struct.pack("f", 0.0))  # goal_x
            f.write(struct.pack("f", 0.0))  # goal_y
            f.write(struct.pack("f", 0.0))  # goal_z
            f.write(struct.pack("i", 0))  # mark_as_expert

            # Exit lanes (none for roads)
            f.write(struct.pack("i", 0))
