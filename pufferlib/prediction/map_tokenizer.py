# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Map tokenizer: converts raw road polylines into pt_token format.

Adapts SMART's map tokenization (SMART/smart/datasets/preprocess.py:403-468
and SMART/smart/model/smart.py:257-319) for our .bin road data.

Pipeline:
1. Take polyline_x, polyline_y from each road
2. Interpolate to 0.5m spacing
3. Split into 5m chunks (sampled to ~3 points for matching)
4. Match each chunk to the 1024-entry codebook via distance minimization
5. Produce pt_token structure compatible with SMARTMapDecoder
"""

import os
import math
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional


_TOKENS_DIR = Path(__file__).parent / 'smart' / 'tokens'
_MAP_TOKEN_PATH = _TOKENS_DIR / 'map_traj_token5.pkl'


def load_map_codebook(path: Optional[str] = None) -> Dict:
    """Load the map token codebook (1024 templates).

    Returns dict with:
        'traj_src': torch.Tensor (1024, 11, 2) - full templates
        'sample_pt': torch.Tensor (1024, 3, 2) - 3 sample points per template
        'traj_end_theta': torch.Tensor (1024,) - end heading per template
    """
    if path is None:
        path = str(_MAP_TOKEN_PATH)
    raw = pickle.load(open(path, 'rb'))
    traj_src = raw['traj_src']  # numpy (1024, 11, 2)

    # Compute sample points (3 evenly spaced from each template)
    indices = torch.linspace(0, traj_src.shape[1] - 1, steps=3).long()
    sample_pt = torch.from_numpy(traj_src[:, indices.numpy()]).float()

    # Compute end heading
    traj_end_theta = np.arctan2(
        traj_src[:, -1, 1] - traj_src[:, -2, 1],
        traj_src[:, -1, 0] - traj_src[:, -2, 0])

    return {
        'traj_src': torch.from_numpy(traj_src).float(),
        'sample_pt': sample_pt,
        'traj_end_theta': torch.from_numpy(traj_end_theta).float(),
    }


def interpolate_polyline(points_xy, distance=0.5, split_distance=5.0):
    """Interpolate a polyline to uniform spacing and split into chunks.

    Adapted from SMART/smart/datasets/preprocess.py:33-114.

    Args:
        points_xy: np.ndarray (N, 2) of polyline x, y coordinates.
        distance: Interpolation spacing in meters.
        split_distance: Length of each chunk in meters.

    Returns:
        torch.Tensor (num_chunks, num_sample_pts, 3) with [x, y, heading] per point,
        or None if polyline is too short.
    """
    if len(points_xy) < 2:
        return None

    # Compute cumulative distance and headings between consecutive points
    diffs = np.diff(points_xy, axis=0)
    segment_lengths = np.sqrt((diffs**2).sum(axis=1))
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])

    # Split on large gaps OR heading changes (matching original SMART)
    # Original: SMART/smart/datasets/preprocess.py:interplating_polyline
    segments = []
    current_segment = [points_xy[0]]
    current_dist = [0.0]

    for i in range(len(segment_lengths)):
        d = segment_lengths[i]
        should_split = False

        if d > 10:
            should_split = True
        elif d > 3 and i > 0:
            heading_diff = abs(headings[i] - headings[i - 1])
            heading_diff = min(heading_diff, 2 * np.pi - heading_diff)
            if heading_diff > np.pi / 4:
                should_split = True
            elif heading_diff > np.pi / 8:
                should_split = True
            elif heading_diff > 0.1:
                should_split = True

        if should_split:
            # Start new segment
            if len(current_segment) >= 2:
                segments.append((np.array(current_segment), np.array(current_dist)))
            current_segment = [points_xy[i + 1]]
            current_dist = [0.0]
        else:
            current_dist.append(current_dist[-1] + segment_lengths[i])
            current_segment.append(points_xy[i + 1])

    if len(current_segment) >= 2:
        segments.append((np.array(current_segment), np.array(current_dist)))

    multi_polylines_list = []
    polyline_size = int(split_distance / distance)  # = 10

    for pts, dist_along in segments:
        if len(dist_along) < 2 or dist_along[-1] < 0.01:
            continue

        new_dist = np.arange(0, dist_along[-1], distance)
        new_dist = np.concatenate([new_dist, [dist_along[-1]]])
        new_x = np.interp(new_dist, dist_along, pts[:, 0])
        new_y = np.interp(new_dist, dist_along, pts[:, 1])

        new_polylines = torch.from_numpy(np.stack([new_x, new_y], axis=1)).float()

        # Compute heading
        heading = torch.atan2(
            new_polylines[1:, 1] - new_polylines[:-1, 1],
            new_polylines[1:, 0] - new_polylines[:-1, 0])
        heading = torch.cat([heading, heading[-1:]])
        new_polylines = torch.cat([new_polylines, heading.unsqueeze(-1)], dim=-1)

        # Split into chunks using unfold
        if new_polylines.shape[0] >= polyline_size + 1:
            chunks = new_polylines.unfold(0, polyline_size + 1, polyline_size)
            chunks = chunks.transpose(1, 2)
            # Sample to 3 points (like SMART: every 5th point from 11)
            chunks = chunks[:, ::5, :]
            multi_polylines_list.append(chunks)

        # Handle remainder
        remainder_start = (
            ((new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1) * polyline_size
            if new_polylines.shape[0] >= polyline_size + 1 else 0)
        remainder = new_polylines[remainder_start:]
        if len(remainder) >= 3:
            indices = torch.linspace(0, len(remainder) - 1, steps=3).long()
            last_chunk = remainder[indices].unsqueeze(0)
            multi_polylines_list.append(last_chunk)

    if not multi_polylines_list:
        return None

    return torch.cat(multi_polylines_list, dim=0)


def tokenize_roads(roads: List[Dict], map_codebook: Dict) -> Dict:
    """Tokenize road polylines from binary reader into pt_token format.

    Args:
        roads: List of road dicts from binary_reader.read_binary_scenario().
        map_codebook: Dict from load_map_codebook().

    Returns:
        pt_token dict compatible with SMARTMapDecoder:
            'position': (M, 3) - x, y, z of first point per chunk
            'orientation': (M,) - heading angle
            'token_idx': (M,) - index into 1024 codebook
            'type': (M,) - road marking type
            'pl_type': (M,) - polygon type
            'side': (M,) - side indicator (0 for simplified)
            'num_nodes': int
        map_save dict for token matching:
            'traj_pos': (M, 3, 2) - sample points per chunk
            'traj_theta': (M,) - heading per chunk
        token2pl_edge_index: (2, M) - token to polygon mapping
        map_polygon dict:
            'light_type': (num_polygons,)
    """
    sample_pt = map_codebook['sample_pt']  # (1024, 3, 2)

    all_chunks = []
    all_types = []
    all_pl_types = []
    all_pl_indices = []
    polygon_idx = 0

    # Map road_type from binary to a simplified polygon type
    # Binary types: 4=lane, 5=line, 6=edge, 8=crosswalk
    # Excluded: 7=stop_sign, 9=speed_bump, 10+=other (matching original SMART)
    ALLOWED_ROAD_TYPES = {4, 5, 6, 8}
    road_type_to_pl_type = {4: 0, 5: 1, 6: 2, 8: 3}

    for road in roads:
        road_type = road.get('type', 4)
        if road_type not in ALLOWED_ROAD_TYPES:
            continue

        points = np.stack([road['polyline_x'], road['polyline_y']], axis=1)
        chunks = interpolate_polyline(points)
        if chunks is None:
            polygon_idx += 1
            continue

        num_chunks = chunks.shape[0]
        all_chunks.append(chunks)
        all_types.append(torch.full((num_chunks,), min(road_type, 16), dtype=torch.long))
        pl_type = road_type_to_pl_type.get(road_type, 0)
        all_pl_types.append(torch.full((num_chunks,), pl_type, dtype=torch.long))
        all_pl_indices.append(torch.full((num_chunks,), polygon_idx, dtype=torch.long))
        polygon_idx += 1

    if not all_chunks:
        # Return empty tensors
        return _empty_pt_token()

    all_chunks = torch.cat(all_chunks, dim=0)  # (M, 3, 3) - [x, y, heading]
    all_types = torch.cat(all_types, dim=0)
    all_pl_types = torch.cat(all_pl_types, dim=0)
    all_pl_indices = torch.cat(all_pl_indices, dim=0)

    M = all_chunks.shape[0]

    # Extract position and heading from first point of each chunk
    traj_pos = all_chunks[:, :, :2]  # (M, 3, 2)
    traj_theta = all_chunks[:, 0, 2]  # (M,) heading at first point

    # Token matching: transform to local coords and match to codebook
    cos = traj_theta.cos()
    sin = traj_theta.sin()
    rot_mat = traj_theta.new_zeros(M, 2, 2)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    # Transform to local coordinate frame
    traj_pos_local = torch.bmm(
        (traj_pos - traj_pos[:, 0:1, :]),
        rot_mat)

    # Distance to each template
    distance = torch.sum(
        (sample_pt[None] - traj_pos_local.unsqueeze(1))**2,
        dim=(-2, -1))
    token_idx = torch.argmin(distance, dim=1)

    # Build pt_token structure
    position = torch.cat([
        traj_pos[:, 0, :],
        torch.zeros(M, 1)
    ], dim=-1)  # (M, 3)

    token2pl = torch.stack([
        torch.arange(M),
        all_pl_indices
    ], dim=0)

    # Compute masks for map next-token prediction
    # A token predicts the next token within the same polyline
    pt_valid_mask = torch.ones(M, dtype=torch.bool)
    same_polyline = all_pl_indices[:-1] == all_pl_indices[1:]  # (M-1,)
    pt_pred_mask = torch.zeros(M, dtype=torch.bool)
    pt_target_mask = torch.zeros(M, dtype=torch.bool)
    pt_pred_mask[:-1] = same_polyline   # token i predicts token i+1
    pt_target_mask[1:] = same_polyline  # token i+1 is the target

    # Map polygon metadata
    num_polygons = polygon_idx

    return {
        'pt_token': {
            'position': position,
            'orientation': traj_theta,
            'token_idx': token_idx,
            'type': all_types,
            'pl_type': all_pl_types,
            'side': torch.zeros(M, dtype=torch.long),
            'num_nodes': M,
            'pt_valid_mask': pt_valid_mask,
            'pt_pred_mask': pt_pred_mask,
            'pt_target_mask': pt_target_mask,
        },
        'map_save': {
            'traj_pos': traj_pos,
            'traj_theta': traj_theta,
        },
        'token2pl_edge_index': torch.stack([
            torch.arange(M), all_pl_indices], dim=0),
        'map_polygon': {
            'type': torch.zeros(num_polygons, dtype=torch.long),
            'num_nodes': num_polygons,
        },
    }


def _empty_pt_token():
    """Return empty pt_token structure when no road data is available."""
    return {
        'pt_token': {
            'position': torch.zeros(1, 3),
            'orientation': torch.zeros(1),
            'token_idx': torch.zeros(1, dtype=torch.long),
            'type': torch.zeros(1, dtype=torch.long),
            'pl_type': torch.zeros(1, dtype=torch.long),
            'side': torch.zeros(1, dtype=torch.long),
            'num_nodes': 1,
            'pt_valid_mask': torch.ones(1, dtype=torch.bool),
            'pt_pred_mask': torch.zeros(1, dtype=torch.bool),
            'pt_target_mask': torch.zeros(1, dtype=torch.bool),
        },
        'map_save': {
            'traj_pos': torch.zeros(1, 3, 2),
            'traj_theta': torch.zeros(1),
        },
        'token2pl_edge_index': torch.zeros(2, 1, dtype=torch.long),
        'map_polygon': {
            'type': torch.zeros(1, dtype=torch.long),
            'num_nodes': 1,
        },
    }
