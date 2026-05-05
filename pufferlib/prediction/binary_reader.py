# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Binary reader for Waymo .bin scenario files.

Inverse of save_map_binary() in pufferlib/ocean/drive/drive.py:530.
Reads binary files containing agent trajectories and road geometry.
"""

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def read_binary_scenario(filepath: str) -> Dict[str, Any]:
    """Read a .bin scenario file and return structured data.

    Args:
        filepath: Path to the .bin file.

    Returns:
        Dictionary with keys:
            'sdc_track_index': int
            'tracks_to_predict': List[int]
            'objects': List[dict] - agent data (vehicles, pedestrians, cyclists)
            'roads': List[dict] - road geometry (lanes, lines, edges, crosswalks)
    """
    with open(filepath, 'rb') as f:
        # Read sdc_track_index
        sdc_track_index = struct.unpack('i', f.read(4))[0]

        # Read tracks_to_predict
        num_tracks = struct.unpack('i', f.read(4))[0]
        tracks_to_predict = []
        for _ in range(num_tracks):
            track_idx = struct.unpack('i', f.read(4))[0]
            tracks_to_predict.append(track_idx)

        # Read counts
        num_objects = struct.unpack('i', f.read(4))[0]
        num_roads = struct.unpack('i', f.read(4))[0]

        # Read objects
        objects = []
        for _ in range(num_objects):
            obj = _read_object(f)
            objects.append(obj)

        # Read roads
        roads = []
        for _ in range(num_roads):
            road = _read_road(f)
            roads.append(road)

    return {
        'sdc_track_index': sdc_track_index,
        'tracks_to_predict': tracks_to_predict,
        'objects': objects,
        'roads': roads,
    }


def _read_object(f) -> Dict[str, Any]:
    """Read a single object (agent) from binary file."""
    scenario_id = struct.unpack('i', f.read(4))[0]
    obj_type = struct.unpack('i', f.read(4))[0]
    obj_id = struct.unpack('i', f.read(4))[0]
    array_size = struct.unpack('i', f.read(4))[0]

    # Read trajectory arrays
    traj_x = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_y = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_z = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_vx = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_vy = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_vz = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_heading = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    traj_valid = np.array(struct.unpack(f'{array_size}i', f.read(4 * array_size)), dtype=np.int32)

    # Read scalar fields
    width = struct.unpack('f', f.read(4))[0]
    length = struct.unpack('f', f.read(4))[0]
    height = struct.unpack('f', f.read(4))[0]
    goal_x = struct.unpack('f', f.read(4))[0]
    goal_y = struct.unpack('f', f.read(4))[0]
    goal_z = struct.unpack('f', f.read(4))[0]
    mark_as_expert = struct.unpack('i', f.read(4))[0]

    # Objects have 0 exit_lanes
    num_exit_lanes = struct.unpack('i', f.read(4))[0]
    exit_lanes = []
    for _ in range(num_exit_lanes):
        exit_lanes.append(struct.unpack('i', f.read(4))[0])

    return {
        'scenario_id': scenario_id,
        'type': obj_type,  # 1=vehicle, 2=pedestrian, 3=cyclist
        'id': obj_id,
        'traj_x': traj_x,
        'traj_y': traj_y,
        'traj_z': traj_z,
        'traj_vx': traj_vx,
        'traj_vy': traj_vy,
        'traj_vz': traj_vz,
        'traj_heading': traj_heading,
        'traj_valid': traj_valid,
        'width': width,
        'length': length,
        'height': height,
        'goal_position': np.array([goal_x, goal_y, goal_z], dtype=np.float32),
        'mark_as_expert': mark_as_expert,
    }


def _read_road(f) -> Dict[str, Any]:
    """Read a single road element from binary file."""
    scenario_id = struct.unpack('i', f.read(4))[0]
    road_type = struct.unpack('i', f.read(4))[0]
    road_id = struct.unpack('i', f.read(4))[0]
    array_size = struct.unpack('i', f.read(4))[0]

    # Read polyline coordinate arrays
    polyline_x = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    polyline_y = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)
    polyline_z = np.array(struct.unpack(f'{array_size}f', f.read(4 * array_size)), dtype=np.float32)

    # Read scalar fields
    width = struct.unpack('f', f.read(4))[0]
    length = struct.unpack('f', f.read(4))[0]
    height = struct.unpack('f', f.read(4))[0]
    goal_x = struct.unpack('f', f.read(4))[0]
    goal_y = struct.unpack('f', f.read(4))[0]
    goal_z = struct.unpack('f', f.read(4))[0]
    mark_as_expert = struct.unpack('i', f.read(4))[0]

    # Read exit_lanes connectivity
    num_exit_lanes = struct.unpack('i', f.read(4))[0]
    exit_lanes = []
    for _ in range(num_exit_lanes):
        exit_lanes.append(struct.unpack('i', f.read(4))[0])

    return {
        'scenario_id': scenario_id,
        'type': road_type,  # 4=lane, 5=line, 6=edge, 7-10=other
        'id': road_id,
        'polyline_x': polyline_x,
        'polyline_y': polyline_y,
        'polyline_z': polyline_z,
        'exit_lanes': exit_lanes,
    }


def list_binary_files(data_dir: str, split: str = 'training') -> List[str]:
    """List all .bin files in a data directory."""
    data_path = Path(data_dir) / split
    if not data_path.exists():
        data_path = Path(data_dir)
    return sorted([str(p) for p in data_path.glob('*.bin')])
