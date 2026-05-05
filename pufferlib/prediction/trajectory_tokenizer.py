# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Trajectory tokenizer: match agent trajectories to SMART motion token codebook.

Adapted from SMART/smart/datasets/preprocess.py and
treeplanner/.../token_processor.py.

Each motion token is a 4-corner polygon contour displacement (left_front,
right_front, right_back, left_back) in local coordinates. Token matching
finds the codebook entry closest to the actual bounding box at each shift
boundary.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_TOKENS_DIR = Path(__file__).parent / 'smart' / 'tokens'
_MOTION_TOKEN_PATH = _TOKENS_DIR / 'cluster_frame_5_2048.pkl'


def load_motion_codebook(path: Optional[str] = None) -> Dict:
    """Load the motion token codebook (2048 templates per agent type).

    Returns dict with:
        'token': {'veh': (2048, 4, 2), ...} - final polygon displacement
        'token_all': {'veh': (2048, 6, 4, 2), ...} - all sub-step displacements
        'traj': {'veh': (2048, 6, 3), ...} - trajectory (x, y, heading)
    """
    if path is None:
        path = str(_MOTION_TOKEN_PATH)
    return pickle.load(open(path, 'rb'))


def cal_polygon_contour(x, y, theta, width, length):
    """Compute 4-corner bounding box polygon for vehicles.

    Returns: np.ndarray (N, 4, 2) with corners:
        [left_front, right_front, right_back, left_back]
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Half dimensions
    hw = 0.5 * width
    hl = 0.5 * length

    # left_front
    lf_x = x + hl * cos_t - hw * sin_t
    lf_y = y + hl * sin_t + hw * cos_t
    # right_front
    rf_x = x + hl * cos_t + hw * sin_t
    rf_y = y + hl * sin_t - hw * cos_t
    # right_back
    rb_x = x - hl * cos_t + hw * sin_t
    rb_y = y - hl * sin_t - hw * cos_t
    # left_back
    lb_x = x - hl * cos_t - hw * sin_t
    lb_y = y - hl * sin_t + hw * cos_t

    contour = np.stack([
        np.stack([lf_x, lf_y], axis=-1),
        np.stack([rf_x, rf_y], axis=-1),
        np.stack([rb_x, rb_y], axis=-1),
        np.stack([lb_x, lb_y], axis=-1),
    ], axis=-2)  # (N, 4, 2)

    return contour


def match_tokens(positions: np.ndarray,
                 headings: np.ndarray,
                 valid_masks: np.ndarray,
                 codebook: np.ndarray,
                 shift: int = 5,
                 width: float = 2.0,
                 length: float = 4.8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match trajectory segments to motion token codebook.

    Adapted from treeplanner/.../token_processor.py:match_token().

    Args:
        positions: (N, T, 2) agent positions [x, y] over T timesteps.
        headings: (N, T) agent headings.
        valid_masks: (N, T) bool validity masks.
        codebook: (2048, 4, 2) vehicle token codebook (polygon displacements).
        shift: Token shift (default 5 = 0.5s at 10Hz).
        width: Vehicle width for polygon contour (default 2.0m).
        length: Vehicle length for polygon contour (default 4.8m).

    Returns:
        token_idx: (N, num_tokens) int64 codebook indices.
        token_contour: (N, num_tokens, 4, 2) float matched polygon corners in world coords.
    """
    N, T, _ = positions.shape
    token_size, contour_dim, feat_dim = codebook.shape  # (2048, 4, 2)

    # Expand codebook for all agents: (N, 2048*4, 2)
    codebook_flat = codebook.reshape(1, token_size * contour_dim, feat_dim)
    codebook_flat = np.repeat(codebook_flat, N, axis=0)

    # Initial state
    prev_pos = positions[:, 0, :].copy()  # (N, 2)
    prev_heading = headings[:, 0].copy()  # (N,)

    token_index_list = []
    token_contour_list = []

    for i in range(shift, T, shift):
        cur_pos = positions[:, i, :]  # (N, 2)
        cur_heading = headings[:, i]  # (N,)

        # Rotation matrix from prev heading (local -> world)
        cos_t = np.cos(prev_heading)
        sin_t = np.sin(prev_heading)
        # Standard rotation: [[cos, -sin], [sin, cos]]
        # einsum 'nij,nkj->nki' computes R @ p for each point
        rot = np.zeros((N, 2, 2))
        rot[:, 0, 0] = cos_t
        rot[:, 0, 1] = -sin_t
        rot[:, 1, 0] = sin_t
        rot[:, 1, 1] = cos_t

        # Transform codebook to world coords
        # codebook_flat: (N, 2048*4, 2), rot: (N, 2, 2)
        agent_token_world = np.einsum('nij,nkj->nki', rot, codebook_flat)
        agent_token_world = agent_token_world.reshape(N, token_size, contour_dim, feat_dim)
        agent_token_world += prev_pos[:, None, None, :]  # translate

        # Compute actual polygon contour at current position
        cur_contour = cal_polygon_contour(
            cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)  # (N, 4, 2)

        # Find closest codebook entry (average distance across 4 corners)
        # cur_contour: (N, 1, 4, 2) vs agent_token_world: (N, 2048, 4, 2)
        dist = np.sqrt(np.sum(
            (cur_contour[:, None, :, :] - agent_token_world) ** 2,
            axis=-1))  # (N, 2048, 4)
        avg_dist = np.mean(dist, axis=2)  # (N, 2048)
        token_idx = np.argmin(avg_dist, axis=1)  # (N,)

        # Get matched contour
        token_contour = agent_token_world[
            np.arange(N), token_idx]  # (N, 4, 2)

        # Update prev_pos/heading using MATCHED contour (not GT)
        # This is critical: matches the original SMART behavior
        valid_prev = valid_masks[:, max(0, i - shift)]
        matched_pos = token_contour.mean(axis=1)  # (N, 2) center of polygon
        diff_xy = token_contour[:, 0, :] - token_contour[:, 3, :]  # left_front - left_back
        matched_heading = np.arctan2(diff_xy[:, 1], diff_xy[:, 0])

        # Only update for valid agents; keep GT for others
        prev_pos[valid_prev] = matched_pos[valid_prev]
        prev_heading[valid_prev] = matched_heading[valid_prev]

        token_index_list.append(token_idx[:, None])
        token_contour_list.append(token_contour[:, None, :, :])

    token_indices = np.concatenate(token_index_list, axis=1)  # (N, 18)
    token_contours = np.concatenate(token_contour_list, axis=1)  # (N, 18, 4, 2)

    return torch.from_numpy(token_indices).long(), torch.from_numpy(token_contours).float()


def compute_token_data(positions: np.ndarray,
                       headings: np.ndarray,
                       valid_masks: np.ndarray,
                       shift: int = 5,
                       codebook: Optional[np.ndarray] = None,
                       agent_types: Optional[np.ndarray] = None) -> Dict:
    """Compute motion token indices for all agents in a scenario.

    Args:
        positions: (N, T, 2) pre-filled agent positions (invalid steps
            already interpolated from nearest valid).
        headings: (N, T) pre-filled agent headings.
        valid_masks: (N, T) bool validity masks.
        shift: Token shift (default 5).
        codebook: Vehicle codebook (2048, 4, 2). If None, loads default.
        agent_types: (N,) SMART type indices (0=veh, 1=ped, 2=cyc).
            If None, all agents use vehicle codebook.

    Returns:
        Dict with:
            'token_idx': (N, num_tokens) - codebook indices
            'token_pos': (N, num_tokens, 2) - matched token positions (center of polygon)
            'token_heading': (N, num_tokens) - matched token headings
    """
    N = positions.shape[0]

    if agent_types is not None and np.any(agent_types > 0):
        # Per-type codebook matching
        all_codebooks = load_motion_codebook()
        type_to_key = {0: 'veh', 1: 'ped', 2: 'cyc'}
        T_tokens = (positions.shape[1] - shift) // shift + 1  # number of tokens

        token_idx_all = torch.zeros(N, T_tokens, dtype=torch.long)
        token_contour_all = torch.zeros(N, T_tokens, 4, 2)

        for t_idx, t_key in type_to_key.items():
            mask = (agent_types == t_idx)
            if not np.any(mask):
                continue
            cb = all_codebooks['token'][t_key]
            idx, contour = match_tokens(
                positions[mask], headings[mask], valid_masks[mask],
                cb, shift=shift)
            token_idx_all[mask] = idx
            token_contour_all[mask] = contour

        token_contour = token_contour_all
        token_idx = token_idx_all
    else:
        if codebook is None:
            data = load_motion_codebook()
            codebook = data['token']['veh']

        token_idx, token_contour = match_tokens(
            positions, headings, valid_masks, codebook, shift=shift)

    # Derive matched positions/headings from contours (matching original SMART)
    # Position: mean of 4 polygon corners
    token_pos = token_contour.mean(dim=2)  # (N, num_tokens, 2)
    # Heading: atan2(left_front - left_back)
    diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]  # left_front - left_back
    token_heading = torch.atan2(diff_xy[:, :, 1], diff_xy[:, :, 0])  # (N, num_tokens)

    return {
        'token_idx': token_idx,
        'token_pos': token_pos,
        'token_heading': token_heading,
    }
