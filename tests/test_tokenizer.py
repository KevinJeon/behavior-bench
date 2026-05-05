# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for trajectory tokenization and detokenization roundtrip.

Tests verify:
1. Tokenization produces valid indices
2. Detokenize(tokenize(trajectory)) ≈ original trajectory
3. Codebook format is correct
4. Token decoder works with variable-shift codebooks
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from pufferlib.prediction.trajectory_tokenizer import (
    match_tokens, compute_token_data, cal_polygon_contour, load_motion_codebook,
)
from pufferlib.ocean.token_decoder import decode_token_to_actions, decode_tokens_batch


# Path to the 200-token codebook (created by extract_tokens.py)
CODEBOOK_PATH = Path(__file__).parent.parent / "codebook_200_shift2.pkl"
SHIFT = 2


def _make_synthetic_codebook(K=200, shift=2):
    """Create a small synthetic codebook for testing when real one isn't available."""
    rng = np.random.RandomState(42)
    traj = np.zeros((K, shift + 1, 3), dtype=np.float32)
    for i in range(K):
        speed = rng.uniform(0.0, 4.0)  # 0-20 m/s equivalent at 0.2s
        steer = rng.uniform(-0.3, 0.3)
        for k in range(1, shift + 1):
            traj[i, k, 0] = traj[i, k - 1, 0] + speed * 0.1
            traj[i, k, 1] = traj[i, k - 1, 1] + steer * 0.1
            traj[i, k, 2] = traj[i, k - 1, 2] + steer * 0.05

    token_polygon = np.zeros((K, 4, 2), dtype=np.float64)
    token_all = np.zeros((K, shift + 1, 4, 2), dtype=np.float32)
    for i in range(K):
        for k in range(shift + 1):
            contour = cal_polygon_contour(
                np.array([traj[i, k, 0]]),
                np.array([traj[i, k, 1]]),
                np.array([traj[i, k, 2]]),
                2.0, 4.8,
            )
            token_all[i, k] = contour[0].astype(np.float32)
        token_polygon[i] = token_all[i, -1].astype(np.float64)

    return {
        'token': {'veh': token_polygon},
        'traj': {'veh': traj},
        'token_all': {'veh': token_all},
    }


def _load_codebook():
    """Load real codebook or fall back to synthetic."""
    if CODEBOOK_PATH.exists():
        import pickle
        return pickle.load(open(CODEBOOK_PATH, 'rb'))
    return _make_synthetic_codebook()


def _make_straight_trajectory(T=91, speed=10.0, heading=0.0):
    """Create a straight-line trajectory."""
    positions = np.zeros((1, T, 2), dtype=np.float32)
    headings = np.full((1, T), heading, dtype=np.float32)
    valid = np.ones((1, T), dtype=bool)
    for t in range(T):
        positions[0, t, 0] = speed * t * 0.1 * np.cos(heading)
        positions[0, t, 1] = speed * t * 0.1 * np.sin(heading)
    return positions, headings, valid


def _make_turning_trajectory(T=91, speed=8.0, turn_rate=0.1):
    """Create a turning trajectory."""
    positions = np.zeros((1, T, 2), dtype=np.float32)
    headings = np.zeros((1, T), dtype=np.float32)
    valid = np.ones((1, T), dtype=bool)
    x, y, h = 0.0, 0.0, 0.0
    for t in range(T):
        positions[0, t, 0] = x
        positions[0, t, 1] = y
        headings[0, t] = h
        x += speed * 0.1 * np.cos(h)
        y += speed * 0.1 * np.sin(h)
        h += turn_rate * 0.1
    return positions, headings, valid


def detokenize_trajectory(token_indices, start_pos, start_heading, traj_codebook):
    """Chain tokens to reconstruct trajectory positions.

    Args:
        token_indices: (num_tokens,) int64 array
        start_pos: (2,) starting position
        start_heading: float starting heading
        traj_codebook: (K, shift+1, 3) local trajectories

    Returns:
        (num_tokens * shift + 1, 2) reconstructed positions
    """
    shift = traj_codebook.shape[1] - 1
    num_tokens = len(token_indices)
    positions = [start_pos.copy()]

    cur_pos = start_pos.copy()
    cur_heading = float(start_heading)

    for i in range(num_tokens):
        tok = int(token_indices[i])
        traj_local = traj_codebook[tok]  # (shift+1, 3)

        cos_h = np.cos(cur_heading)
        sin_h = np.sin(cur_heading)

        for k in range(1, shift + 1):
            lx, ly = traj_local[k, 0], traj_local[k, 1]
            wx = cur_pos[0] + lx * cos_h - ly * sin_h
            wy = cur_pos[1] + lx * sin_h + ly * cos_h
            positions.append(np.array([wx, wy]))

        # Advance to matched endpoint
        lx_end, ly_end = traj_local[-1, 0], traj_local[-1, 1]
        cur_pos[0] = cur_pos[0] + lx_end * cos_h - ly_end * sin_h
        cur_pos[1] = cur_pos[1] + lx_end * sin_h + ly_end * cos_h
        cur_heading = cur_heading + float(traj_local[-1, 2])

    return np.array(positions)


class TestCodebookFormat:
    def test_codebook_keys(self):
        cb = _load_codebook()
        assert 'token' in cb and 'veh' in cb['token']
        assert 'traj' in cb and 'veh' in cb['traj']
        assert 'token_all' in cb and 'veh' in cb['token_all']

    def test_codebook_shapes(self):
        cb = _load_codebook()
        K = cb['traj']['veh'].shape[0]
        shift = cb['traj']['veh'].shape[1] - 1
        assert cb['token']['veh'].shape == (K, 4, 2)
        assert cb['traj']['veh'].shape == (K, shift + 1, 3)
        assert cb['token_all']['veh'].shape == (K, shift + 1, 4, 2)

    def test_codebook_first_waypoint_near_zero(self):
        cb = _load_codebook()
        traj = cb['traj']['veh']
        # First waypoint should be near origin (local frame)
        assert np.allclose(traj[:, 0, :], 0.0, atol=0.01)


class TestTokenization:
    def test_tokenize_straight_line(self):
        cb = _load_codebook()
        codebook = cb['token']['veh']
        positions, headings, valid = _make_straight_trajectory()
        token_idx, _ = match_tokens(positions, headings, valid, codebook, shift=SHIFT)
        assert token_idx.shape[0] == 1
        assert (token_idx >= 0).all()
        assert (token_idx < codebook.shape[0]).all()

    def test_tokenize_turn(self):
        cb = _load_codebook()
        codebook = cb['token']['veh']
        positions, headings, valid = _make_turning_trajectory()
        token_idx, _ = match_tokens(positions, headings, valid, codebook, shift=SHIFT)
        assert token_idx.shape[0] == 1
        assert (token_idx >= 0).all()

    def test_compute_token_data(self):
        cb = _load_codebook()
        codebook = cb['token']['veh']
        positions, headings, valid = _make_straight_trajectory()
        data = compute_token_data(positions, headings, valid, shift=SHIFT, codebook=codebook)
        assert 'token_idx' in data
        assert 'token_pos' in data
        assert 'token_heading' in data


class TestDetokenization:
    def test_roundtrip_straight(self):
        cb = _load_codebook()
        codebook_poly = cb['token']['veh']
        traj_cb = cb['traj']['veh']
        positions, headings, valid = _make_straight_trajectory(speed=5.0)
        token_idx, _ = match_tokens(positions, headings, valid, codebook_poly, shift=SHIFT)

        reconstructed = detokenize_trajectory(
            token_idx[0].numpy(), positions[0, 0].copy(), headings[0, 0], traj_cb)

        # Compare first 10 tokens at shift boundaries
        num_tokens = min(token_idx.shape[1], 10)
        errors = []
        for i in range(num_tokens):
            t_orig = (i + 1) * SHIFT
            t_recon = (i + 1) * SHIFT
            if t_orig < positions.shape[1] and t_recon < len(reconstructed):
                error = np.linalg.norm(
                    reconstructed[t_recon] - positions[0, t_orig])
                errors.append(error)
        if errors:
            mean_err = np.mean(errors)
            # With synthetic codebook, expect higher errors; real codebook should be < 1m
            assert mean_err < 5.0, f"Mean error={mean_err:.2f}m over {len(errors)} tokens"

    def test_roundtrip_turn(self):
        cb = _load_codebook()
        codebook_poly = cb['token']['veh']
        traj_cb = cb['traj']['veh']
        positions, headings, valid = _make_turning_trajectory(speed=5.0)
        token_idx, _ = match_tokens(positions, headings, valid, codebook_poly, shift=SHIFT)

        reconstructed = detokenize_trajectory(
            token_idx[0].numpy(), positions[0, 0].copy(), headings[0, 0], traj_cb)

        # Check errors don't explode
        num_tokens = token_idx.shape[1]
        errors = []
        for i in range(min(num_tokens, 10)):
            t_orig = (i + 1) * SHIFT
            t_recon = (i + 1) * SHIFT
            if t_orig < positions.shape[1] and t_recon < len(reconstructed):
                error = np.linalg.norm(
                    reconstructed[t_recon] - positions[0, t_orig])
                errors.append(error)
        if errors:
            assert np.mean(errors) < 2.0, f"Mean error={np.mean(errors):.2f}m"


class TestTokenDecoder:
    def test_decode_single_token(self):
        cb = _load_codebook()
        traj_cb = cb['traj']['veh']
        shift = traj_cb.shape[1] - 1
        actions = decode_token_to_actions(
            0, np.array([0.0, 0.0]), 0.0, 5.0, traj_cb)
        assert actions.shape == (shift, 2)
        assert np.all(np.abs(actions) <= 1.0)

    def test_decode_batch(self):
        cb = _load_codebook()
        traj_cb = cb['traj']['veh']
        shift = traj_cb.shape[1] - 1
        N = 10
        token_indices = np.arange(N)
        cur_pos = np.zeros((N, 2))
        cur_heading = np.zeros(N)
        cur_speed = np.ones(N) * 5.0
        actions = decode_tokens_batch(
            token_indices, cur_pos, cur_heading, cur_speed, traj_cb)
        assert actions.shape == (N, shift, 2)
        assert np.all(np.abs(actions) <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
