# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Visualize raw input data for a few scenarios.

Creates diagnostic plots showing:
1. Raw global positions (UTM coords) with valid/invalid markers
2. Map features (roads by type, colored)
3. Agent trajectories (history + future) with bounding boxes
4. Motion vectors at valid/invalid boundaries
5. Map token positions vs raw road polylines

Usage:
    python -m pufferlib.prediction.visualize_data
"""

import os
import sys
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

from pufferlib.prediction.binary_reader import read_binary_scenario, list_binary_files
from pufferlib.prediction.trajectory_tokenizer import compute_token_data, load_motion_codebook
from pufferlib.prediction.map_tokenizer import tokenize_roads, load_map_codebook


ROAD_TYPE_COLORS = {
    4: ('#555555', 'Lane'),
    5: ('#AAAAAA', 'Line'),
    6: ('#FF8800', 'Edge'),
    7: ('#00AA00', 'Crosswalk'),
    8: ('#0088FF', 'SpeedBump'),
    9: ('#AA00AA', 'StopSign'),
    10: ('#FF0000', 'Other'),
}

AGENT_TYPE_NAMES = {1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'}
AGENT_TYPE_COLORS = {1: '#2196F3', 2: '#4CAF50', 3: '#FF9800'}

OUT_DIR = Path('analysis/data_viz')


def draw_box(ax, x, y, heading, width, length, color='blue', alpha=0.5, label=None):
    """Draw a rotated bounding box."""
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    hw, hl = width / 2, length / 2
    corners = [
        (-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)
    ]
    rotated = [(x + cx * cos_h - cy * sin_h,
                y + cx * sin_h + cy * cos_h) for cx, cy in corners]
    polygon = plt.Polygon(rotated, closed=True, facecolor=color,
                          edgecolor='black', alpha=alpha, linewidth=0.5,
                          label=label)
    ax.add_patch(polygon)


def plot_scenario_overview(scenario, scenario_idx, out_dir):
    """Plot 1: Full scene overview with raw global coordinates."""
    objects = scenario['objects']
    roads = scenario['roads']
    sdc_idx = scenario['sdc_track_index']

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # -- Left: Full scene with all roads and agent trajectories --
    ax = axes[0]
    ax.set_title(f'Scenario {scenario_idx}: Full Scene (Global UTM Coords)', fontsize=12)

    # Draw roads
    for road in roads:
        road_type = road['type']
        color, label = ROAD_TYPE_COLORS.get(road_type, ('#888888', f'Type{road_type}'))
        ax.plot(road['polyline_x'], road['polyline_y'],
                color=color, linewidth=0.8, alpha=0.6)

    # Draw agent trajectories
    t_split = 11  # historical / future boundary
    for i, obj in enumerate(objects):
        valid = obj['traj_valid'].astype(bool)
        color = AGENT_TYPE_COLORS.get(obj['type'], '#888888')
        is_sdc = (i == sdc_idx)

        # History (solid)
        hist_valid = valid[:t_split]
        if hist_valid.any():
            hx = obj['traj_x'][:t_split][hist_valid]
            hy = obj['traj_y'][:t_split][hist_valid]
            ax.plot(hx, hy, color=color, linewidth=1.5 if is_sdc else 0.8,
                    alpha=0.9 if is_sdc else 0.5)

        # Future (dashed)
        fut_valid = valid[t_split:]
        if fut_valid.any():
            fx = obj['traj_x'][t_split:][fut_valid]
            fy = obj['traj_y'][t_split:][fut_valid]
            ax.plot(fx, fy, color=color, linewidth=1.5 if is_sdc else 0.8,
                    linestyle='--', alpha=0.7 if is_sdc else 0.3)

        # Draw box at t=10 (last historical step)
        if valid[t_split - 1]:
            draw_box(ax, obj['traj_x'][t_split - 1], obj['traj_y'][t_split - 1],
                     obj['traj_heading'][t_split - 1],
                     obj['width'], obj['length'],
                     color='red' if is_sdc else color,
                     alpha=0.8 if is_sdc else 0.4)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m, UTM)')
    ax.set_ylabel('Y (m, UTM)')

    # Add legend
    legend_elements = [
        mpatches.Patch(color=c, label=l) for c, l in [
            ('#555555', 'Lane'), ('#AAAAAA', 'Line'), ('#FF8800', 'Edge'),
            ('#2196F3', 'Vehicle'), ('#4CAF50', 'Pedestrian'), ('#FF9800', 'Cyclist'),
            ('red', 'SDC'),
        ]]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # -- Right: Zoomed view centered on SDC --
    ax2 = axes[1]
    if sdc_idx >= 0 and sdc_idx < len(objects):
        sdc = objects[sdc_idx]
        if sdc['traj_valid'][t_split - 1]:
            cx = sdc['traj_x'][t_split - 1]
            cy = sdc['traj_y'][t_split - 1]
        else:
            valid_t = np.where(sdc['traj_valid'])[0]
            cx = sdc['traj_x'][valid_t[-1]] if len(valid_t) > 0 else 0
            cy = sdc['traj_y'][valid_t[-1]] if len(valid_t) > 0 else 0
    else:
        cx, cy = 0, 0

    ax2.set_title(f'Scenario {scenario_idx}: Zoomed on SDC (±80m)', fontsize=12)

    for road in roads:
        road_type = road['type']
        color, _ = ROAD_TYPE_COLORS.get(road_type, ('#888888', ''))
        ax2.plot(road['polyline_x'], road['polyline_y'],
                 color=color, linewidth=1.0, alpha=0.7)

    for i, obj in enumerate(objects):
        valid = obj['traj_valid'].astype(bool)
        color = AGENT_TYPE_COLORS.get(obj['type'], '#888888')
        is_sdc = (i == sdc_idx)

        hist_valid = valid[:t_split]
        if hist_valid.any():
            hx = obj['traj_x'][:t_split][hist_valid]
            hy = obj['traj_y'][:t_split][hist_valid]
            ax2.plot(hx, hy, color=color, linewidth=2.0 if is_sdc else 1.0,
                     alpha=0.9 if is_sdc else 0.6)

        fut_valid = valid[t_split:]
        if fut_valid.any():
            fx = obj['traj_x'][t_split:][fut_valid]
            fy = obj['traj_y'][t_split:][fut_valid]
            ax2.plot(fx, fy, color=color, linewidth=2.0 if is_sdc else 1.0,
                     linestyle='--', alpha=0.8 if is_sdc else 0.4)

        if valid[t_split - 1]:
            draw_box(ax2, obj['traj_x'][t_split - 1], obj['traj_y'][t_split - 1],
                     obj['traj_heading'][t_split - 1],
                     obj['width'], obj['length'],
                     color='red' if is_sdc else color,
                     alpha=0.8 if is_sdc else 0.5)

    ax2.set_xlim(cx - 80, cx + 80)
    ax2.set_ylim(cy - 80, cy + 80)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m, UTM)')
    ax2.set_ylabel('Y (m, UTM)')

    plt.tight_layout()
    plt.savefig(out_dir / f'scenario_{scenario_idx:04d}_overview.png', dpi=150)
    plt.close()


def plot_valid_invalid_analysis(scenario, scenario_idx, out_dir):
    """Plot 2: Show valid vs invalid timesteps and the motion vectors."""
    objects = scenario['objects']
    sdc_idx = scenario['sdc_track_index']
    T = len(objects[0]['traj_x']) if objects else 91

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # -- Top left: Valid mask heatmap --
    ax = axes[0, 0]
    N = min(len(objects), 32)
    valid_matrix = np.zeros((N, T))
    for i in range(N):
        valid_matrix[i] = objects[i]['traj_valid'][:T]
    ax.imshow(valid_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Agent index')
    ax.set_title(f'Scenario {scenario_idx}: Valid Mask (green=valid, red=invalid)')
    ax.axvline(x=10.5, color='white', linewidth=2, label='History/Future boundary')
    ax.legend(fontsize=8)

    # -- Top right: Position values for SDC (show raw UTM coords) --
    ax = axes[0, 1]
    if sdc_idx >= 0 and sdc_idx < len(objects):
        sdc = objects[sdc_idx]
        timesteps = np.arange(T)
        valid = sdc['traj_valid'].astype(bool)

        ax.plot(timesteps[valid], sdc['traj_x'][:T][valid], 'b-o', markersize=2, label='x (valid)')
        ax.plot(timesteps[~valid], sdc['traj_x'][:T][~valid], 'bx', markersize=4, label='x (invalid)')
        ax.plot(timesteps[valid], sdc['traj_y'][:T][valid], 'r-o', markersize=2, label='y (valid)')
        ax.plot(timesteps[~valid], sdc['traj_y'][:T][~valid], 'rx', markersize=4, label='y (invalid)')
        ax.axvline(x=10.5, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Position (m, UTM)')
        ax.set_title(f'SDC Raw Positions (Agent {sdc_idx})')
        ax.legend(fontsize=8)
    else:
        ax.set_title('No SDC found')

    # -- Bottom left: Motion vectors magnitude --
    ax = axes[1, 0]
    if sdc_idx >= 0 and sdc_idx < len(objects):
        sdc = objects[sdc_idx]
        dx = np.diff(sdc['traj_x'][:T])
        dy = np.diff(sdc['traj_y'][:T])
        motion_mag = np.sqrt(dx**2 + dy**2)

        # Color by valid transition
        valid = sdc['traj_valid'][:T].astype(bool)
        both_valid = valid[:-1] & valid[1:]

        ax.bar(np.arange(T-1)[both_valid], motion_mag[both_valid],
               color='green', alpha=0.7, label='Both valid')
        ax.bar(np.arange(T-1)[~both_valid], motion_mag[~both_valid],
               color='red', alpha=0.7, label='Invalid transition')
        ax.axvline(x=10, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Motion magnitude (m)')
        ax.set_title(f'SDC Motion Vectors (red = invalid transition)')
        ax.set_yscale('log')
        ax.legend(fontsize=8)

    # -- Bottom right: All agents position spread --
    ax = axes[1, 1]
    all_x_valid, all_y_valid = [], []
    all_x_invalid, all_y_invalid = [], []
    for obj in objects[:32]:
        valid = obj['traj_valid'][:T].astype(bool)
        all_x_valid.append(obj['traj_x'][:T][valid])
        all_y_valid.append(obj['traj_y'][:T][valid])
        all_x_invalid.append(obj['traj_x'][:T][~valid])
        all_y_invalid.append(obj['traj_y'][:T][~valid])

    all_x_valid = np.concatenate(all_x_valid) if all_x_valid else np.array([])
    all_y_valid = np.concatenate(all_y_valid) if all_y_valid else np.array([])
    all_x_invalid = np.concatenate(all_x_invalid) if all_x_invalid else np.array([])
    all_y_invalid = np.concatenate(all_y_invalid) if all_y_invalid else np.array([])

    ax.scatter(all_x_valid, all_y_valid, s=1, c='green', alpha=0.3, label='Valid')
    if len(all_x_invalid) > 0:
        ax.scatter(all_x_invalid, all_y_invalid, s=3, c='red', alpha=0.5, label='Invalid (0,0?)')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('All Agent Positions: Valid (green) vs Invalid (red)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f'scenario_{scenario_idx:04d}_valid_analysis.png', dpi=150)
    plt.close()


def plot_map_tokens(scenario, scenario_idx, map_codebook, out_dir):
    """Plot 3: Compare raw road polylines vs tokenized map tokens."""
    roads = scenario['roads']
    objects = scenario['objects']
    sdc_idx = scenario['sdc_track_index']

    map_data = tokenize_roads(roads, map_codebook)
    pt_token = map_data['pt_token']
    map_save = map_data['map_save']

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))

    # Find center
    t_split = 11
    if sdc_idx >= 0 and sdc_idx < len(objects):
        sdc = objects[sdc_idx]
        if sdc['traj_valid'][t_split - 1]:
            cx = sdc['traj_x'][t_split - 1]
            cy = sdc['traj_y'][t_split - 1]
        else:
            valid_t = np.where(sdc['traj_valid'])[0]
            cx = sdc['traj_x'][valid_t[-1]] if len(valid_t) > 0 else 0
            cy = sdc['traj_y'][valid_t[-1]] if len(valid_t) > 0 else 0
    else:
        cx, cy = 0, 0

    # -- Left: Raw road polylines --
    ax = axes[0]
    ax.set_title(f'Scenario {scenario_idx}: Raw Road Polylines', fontsize=12)
    for road in roads:
        road_type = road['type']
        color, label = ROAD_TYPE_COLORS.get(road_type, ('#888888', f'Type{road_type}'))
        ax.plot(road['polyline_x'], road['polyline_y'],
                color=color, linewidth=1.0, alpha=0.7)
    ax.set_xlim(cx - 80, cx + 80)
    ax.set_ylim(cy - 80, cy + 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    legend_elements = [
        mpatches.Patch(color=c, label=l) for _, (c, l) in ROAD_TYPE_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=8)

    # -- Right: Tokenized map (pt_token positions + arrows showing token direction) --
    ax = axes[1]
    ax.set_title(f'Scenario {scenario_idx}: Map Tokens ({pt_token["num_nodes"]} tokens)', fontsize=12)

    # Draw token positions with arrows
    positions = pt_token['position'].numpy()  # (M, 3)
    orientations = pt_token['orientation'].numpy()  # (M,)
    token_idx = pt_token['token_idx'].numpy()  # (M,)
    pl_types = pt_token['pl_type'].numpy()  # (M,)

    pl_type_colors = {0: '#555555', 1: '#AAAAAA', 2: '#FF8800', 3: '#00AA00'}
    pl_type_names = {0: 'Lane', 1: 'Line', 2: 'Edge', 3: 'Other'}

    for pl_type in np.unique(pl_types):
        mask = pl_types == pl_type
        color = pl_type_colors.get(pl_type, '#888888')
        ax.scatter(positions[mask, 0], positions[mask, 1],
                   s=8, c=color, alpha=0.6,
                   label=f'{pl_type_names.get(pl_type, "?")} ({mask.sum()})')
        # Draw direction arrows
        arrow_len = 2.0
        for j in np.where(mask)[0]:
            dx = arrow_len * math.cos(orientations[j])
            dy = arrow_len * math.sin(orientations[j])
            ax.arrow(positions[j, 0], positions[j, 1], dx, dy,
                     head_width=0.5, head_length=0.3,
                     fc=color, ec=color, alpha=0.3)

    # Also draw the traj_pos (3 sample points per token) to show actual road shape
    traj_pos = map_save['traj_pos'].numpy()  # (M, 3, 2)
    polygon_idx = map_data['token2pl_edge_index'][1].numpy()
    for pg_idx in np.unique(polygon_idx):
        mask = polygon_idx == pg_idx
        pts = traj_pos[mask]  # (K, 3, 2)
        # Draw connected segments
        for k in range(pts.shape[0]):
            ax.plot(pts[k, :, 0], pts[k, :, 1], 'k-', linewidth=0.3, alpha=0.3)

    ax.set_xlim(cx - 80, cx + 80)
    ax.set_ylim(cy - 80, cy + 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / f'scenario_{scenario_idx:04d}_map_tokens.png', dpi=150)
    plt.close()


def plot_model_input(scenario, scenario_idx, map_codebook, motion_codebook, out_dir):
    """Plot 4: Show exactly what the model receives as input.

    This replicates dataset._process_scenario() and shows:
    - Agent positions as model sees them (centered on SDC)
    - Token positions overlaid on GT trajectory
    - Motion token distribution (2048 SMART tokens)
    - Token validity heatmap
    """
    objects = scenario['objects']
    roads = scenario['roads']
    sdc_idx = scenario['sdc_track_index']
    t_split = 11
    T = 91
    shift = 5

    # Filter to vehicles only (type == 1)
    vehicle_indices = [i for i, obj in enumerate(objects) if obj['type'] == 1]
    if not vehicle_indices:
        vehicle_indices = [sdc_idx] if sdc_idx >= 0 else [0]
    objects = [objects[i] for i in vehicle_indices]
    if sdc_idx in vehicle_indices:
        sdc_idx = vehicle_indices.index(sdc_idx)
    else:
        sdc_idx = 0

    N = len(objects)

    # Select closest agents to SDC
    if N > 16 and sdc_idx >= 0:
        sdc_obj = objects[sdc_idx]
        sdc_x = sdc_obj['traj_x'][t_split - 1]
        sdc_y = sdc_obj['traj_y'][t_split - 1]
        distances = []
        for obj in objects:
            ox = obj['traj_x'][t_split - 1]
            oy = obj['traj_y'][t_split - 1]
            distances.append((ox - sdc_x)**2 + (oy - sdc_y)**2)
        sorted_indices = np.argsort(distances)[:16]
        objects = [objects[i] for i in sorted_indices]
        sdc_idx = int(np.where(sorted_indices == sdc_idx)[0][0]) if sdc_idx in sorted_indices else 0
        N = len(objects)

    # Build filled positions/headings (same as dataset._process_scenario)
    positions = np.zeros((N, T, 2))
    headings = np.zeros((N, T))
    valid_masks = np.zeros((N, T), dtype=bool)
    for i, obj in enumerate(objects):
        t_max = min(len(obj['traj_x']), T)
        positions[i, :t_max, 0] = obj['traj_x'][:t_max]
        positions[i, :t_max, 1] = obj['traj_y'][:t_max]
        headings[i, :t_max] = obj['traj_heading'][:t_max]
        valid_masks[i, :t_max] = obj['traj_valid'][:t_max].astype(bool)

    # Fill invalid with nearest valid
    for i in range(N):
        v = valid_masks[i]
        if not v.any():
            continue
        first_valid = np.where(v)[0][0]
        if first_valid > 0:
            positions[i, :first_valid] = positions[i, first_valid]
            headings[i, :first_valid] = headings[i, first_valid]
        for t in range(first_valid + 1, T):
            if not v[t]:
                positions[i, t] = positions[i, t - 1]
                headings[i, t] = headings[i, t - 1]

    # Scene center
    center_x = positions[sdc_idx, t_split - 1, 0]
    center_y = positions[sdc_idx, t_split - 1, 1]

    # Compute motion tokens
    token_data = compute_token_data(
        positions, headings, valid_masks, shift=shift,
        codebook=motion_codebook)
    token_idx = token_data['token_idx'].numpy()  # (N, 18)

    # GT positions at shift boundaries
    shift_indices = list(range(shift, T, shift))  # [5, 10, ..., 90]
    num_tokens = len(shift_indices)
    token_pos_x = positions[:, shift_indices, 0] - center_x  # (N, 18)
    token_pos_y = positions[:, shift_indices, 1] - center_y
    token_valid = np.zeros((N, num_tokens), dtype=bool)
    for k, t_idx in enumerate(shift_indices):
        t_prev = t_idx - shift
        token_valid[:, k] = valid_masks[:, t_idx] & valid_masks[:, t_prev]

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # -- (0,0): Centered scene with roads + trajectories + token positions --
    ax = axes[0, 0]
    ax.set_title(f'Centered Scene + Token Positions (shift={shift})\n'
                 f'N={N} vehicles, {num_tokens} tokens/agent', fontsize=11)

    for road in roads:
        road_type = road['type']
        color, _ = ROAD_TYPE_COLORS.get(road_type, ('#888888', ''))
        ax.plot(road['polyline_x'] - center_x, road['polyline_y'] - center_y,
                color=color, linewidth=0.8, alpha=0.4)

    for i in range(N):
        v = valid_masks[i]
        is_sdc = (i == sdc_idx)
        color = '#FF0000' if is_sdc else '#2196F3'

        x_c = positions[i, :, 0] - center_x
        y_c = positions[i, :, 1] - center_y

        # GT trajectory
        if v.any():
            # History (solid)
            hist_v = v[:t_split]
            if hist_v.any():
                ax.plot(x_c[:t_split][hist_v], y_c[:t_split][hist_v],
                        '-', color=color, linewidth=2.0 if is_sdc else 0.8, alpha=0.8)
            # Future (dashed)
            fut_v = v[t_split:]
            if fut_v.any():
                ax.plot(x_c[t_split:][fut_v], y_c[t_split:][fut_v],
                        '--', color=color, linewidth=2.0 if is_sdc else 0.8, alpha=0.5)

        # Token positions (circles at shift boundaries)
        tv = token_valid[i]
        if tv.any():
            ax.scatter(token_pos_x[i, tv], token_pos_y[i, tv],
                       s=40 if is_sdc else 15, c='magenta', marker='o',
                       edgecolors='black', linewidths=0.5, zorder=5,
                       alpha=0.9 if is_sdc else 0.6)

        # Box at t=10
        if v[t_split - 1]:
            draw_box(ax, x_c[t_split-1], y_c[t_split-1],
                     headings[i, t_split-1],
                     objects[i]['width'], objects[i]['length'],
                     color=color, alpha=0.7)

    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, centered)')
    ax.set_ylabel('Y (m, centered)')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')

    # -- (0,1): SDC token detail --
    ax = axes[0, 1]
    ax.set_title(f'SDC Token Detail\nGT trajectory + token positions at t=5,10,...,90', fontsize=11)

    sdc_x_c = positions[sdc_idx, :, 0] - center_x
    sdc_y_c = positions[sdc_idx, :, 1] - center_y
    sdc_v = valid_masks[sdc_idx]

    if sdc_v.any():
        ax.plot(sdc_x_c[sdc_v], sdc_y_c[sdc_v], 'b-', linewidth=1.5, alpha=0.6, label='GT trajectory')

    # Token positions with labels
    sdc_tv = token_valid[sdc_idx]
    for k in range(num_tokens):
        if sdc_tv[k]:
            t = shift_indices[k]
            color = '#FF6600' if k < 2 else '#CC00CC'  # orange=hist, magenta=future
            ax.scatter(token_pos_x[sdc_idx, k], token_pos_y[sdc_idx, k],
                       s=60, c=color, marker='o', edgecolors='black',
                       linewidths=0.8, zorder=5)
            ax.annotate(f't={t}\ntk={token_idx[sdc_idx, k]}',
                        (token_pos_x[sdc_idx, k], token_pos_y[sdc_idx, k]),
                        fontsize=6, ha='left', va='bottom')

            # Draw heading arrow at token position
            heading_k = headings[sdc_idx, t]
            arrow_len = 2.0
            ax.arrow(token_pos_x[sdc_idx, k], token_pos_y[sdc_idx, k],
                     arrow_len * math.cos(heading_k), arrow_len * math.sin(heading_k),
                     head_width=0.5, head_length=0.3, fc=color, ec=color, alpha=0.6)

    # Roads around SDC
    for road in roads:
        road_type = road['type']
        color, _ = ROAD_TYPE_COLORS.get(road_type, ('#888888', ''))
        ax.plot(road['polyline_x'] - center_x, road['polyline_y'] - center_y,
                color=color, linewidth=0.8, alpha=0.3)

    margin = 40
    sdc_valid_x = sdc_x_c[sdc_v]
    sdc_valid_y = sdc_y_c[sdc_v]
    if len(sdc_valid_x) > 0:
        mid_x = (sdc_valid_x.min() + sdc_valid_x.max()) / 2
        mid_y = (sdc_valid_y.min() + sdc_valid_y.max()) / 2
        ax.set_xlim(mid_x - margin, mid_x + margin)
        ax.set_ylim(mid_y - margin, mid_y + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m, centered)')
    ax.set_ylabel('Y (m, centered)')
    ax.legend(fontsize=8, loc='upper right')

    # -- (0,2): Token positions vs GT positions at shift boundaries --
    ax = axes[0, 2]
    ax.set_title('Token Pos vs GT Pos at Shift Boundaries\n(should overlap perfectly)', fontsize=11)

    gt_at_shifts_x = positions[:, shift_indices, 0] - center_x
    gt_at_shifts_y = positions[:, shift_indices, 1] - center_y

    all_gt_x = gt_at_shifts_x[token_valid].flatten()
    all_gt_y = gt_at_shifts_y[token_valid].flatten()
    all_tk_x = token_pos_x[token_valid].flatten()
    all_tk_y = token_pos_y[token_valid].flatten()

    if len(all_gt_x) > 0:
        errors = np.sqrt((all_gt_x - all_tk_x)**2 + (all_gt_y - all_tk_y)**2)
        ax.scatter(all_gt_x, all_gt_y, s=5, c='blue', alpha=0.4, label=f'GT pos (N={len(all_gt_x)})')
        ax.scatter(all_tk_x, all_tk_y, s=5, c='red', alpha=0.4, label='Token pos')
        ax.set_xlabel('X (m, centered)')
        ax.set_ylabel('Y (m, centered)')
        stats_txt = (f'Mean error: {errors.mean():.3f}m\n'
                     f'Max error: {errors.max():.3f}m\n'
                     f'Median error: {np.median(errors):.3f}m')
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_aspect('equal')
    ax.legend(fontsize=8)

    # -- (1,0): Motion token distribution --
    ax = axes[1, 0]
    all_valid_tokens = token_idx[token_valid].flatten()
    if len(all_valid_tokens) > 0:
        ax.hist(all_valid_tokens, bins=100, range=(0, 2048), color='steelblue', alpha=0.7)
        ax.set_xlabel('Motion Token Index (0..2047)')
        ax.set_ylabel('Count')
        ax.set_title(f'Motion Token Distribution (SMART 2048 vocab)\n'
                     f'{len(all_valid_tokens)} valid tokens from {N} agents\n'
                     f'Unique tokens: {len(np.unique(all_valid_tokens))}/2048')
    else:
        ax.set_title('No valid motion tokens found')

    # -- (1,1): Token validity heatmap --
    ax = axes[1, 1]
    ax.imshow(token_valid[:min(N, 32)].astype(float), aspect='auto',
              cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel(f'Token step (0..{num_tokens-1}, each = {shift} timesteps)')
    ax.set_ylabel('Agent index')
    ax.set_title(f'Token Validity (green=valid, red=invalid)\n'
                 f'Hist tokens: 0-1, Future tokens: 2-17')
    ax.axvline(x=1.5, color='white', linewidth=2, label='History/Future boundary')
    ax.legend(fontsize=8)

    # -- (1,2): Map token stats --
    ax = axes[1, 2]
    map_data = tokenize_roads(roads, map_codebook)
    pt_token = map_data['pt_token']
    map_token_idx = pt_token['token_idx'].numpy()

    ax.hist(map_token_idx, bins=50, color='coral', alpha=0.7)
    ax.set_xlabel('Map Token index (0..1023)')
    ax.set_ylabel('Count')
    ax.set_title(f'Map Token Distribution\n{len(map_token_idx)} tokens from {len(roads)} roads')

    stats_text = (
        f"Unique tokens: {len(np.unique(map_token_idx))}/1024\n"
        f"Map tokens: {pt_token['num_nodes']}\n"
        f"pt_pred_mask sum: {pt_token['pt_pred_mask'].sum().item()}\n"
        f"pt_target_mask sum: {pt_token['pt_target_mask'].sum().item()}\n"
        f"pt_valid_mask sum: {pt_token['pt_valid_mask'].sum().item()}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_dir / f'scenario_{scenario_idx:04d}_model_input.png', dpi=150)
    plt.close()


def plot_coordinate_statistics(scenarios, out_dir):
    """Plot 5: Global statistics across multiple scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    all_x, all_y = [], []
    all_invalid_x, all_invalid_y = [], []
    max_motion_valid, max_motion_invalid = [], []
    num_agents_list = []
    num_tokens_list = []

    for scenario in scenarios:
        objects = scenario['objects']
        num_agents_list.append(len(objects))

        for obj in objects:
            valid = obj['traj_valid'].astype(bool)
            all_x.extend(obj['traj_x'][valid].tolist())
            all_y.extend(obj['traj_y'][valid].tolist())
            all_invalid_x.extend(obj['traj_x'][~valid].tolist())
            all_invalid_y.extend(obj['traj_y'][~valid].tolist())

            dx = np.diff(obj['traj_x'])
            dy = np.diff(obj['traj_y'])
            motion = np.sqrt(dx**2 + dy**2)
            both_valid = valid[:-1] & valid[1:]
            if both_valid.any():
                max_motion_valid.append(motion[both_valid].max())
            invalid_trans = ~both_valid
            if invalid_trans.any():
                max_motion_invalid.append(motion[invalid_trans].max())

    # -- (0,0): Position distribution --
    ax = axes[0, 0]
    ax.hist2d(all_x, all_y, bins=100, cmap='viridis')
    ax.set_xlabel('X (m, UTM)')
    ax.set_ylabel('Y (m, UTM)')
    ax.set_title(f'Valid Position Distribution ({len(scenarios)} scenarios)')
    ax.set_aspect('equal')

    # -- (0,1): Invalid positions --
    ax = axes[0, 1]
    if all_invalid_x:
        ax.hist2d(all_invalid_x, all_invalid_y, bins=100, cmap='Reds')
        ax.set_title(f'Invalid Position Distribution\n'
                     f'({len(all_invalid_x)} points, expect cluster at (0,0))')
    else:
        ax.set_title('No invalid positions')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # -- (1,0): Motion magnitude distribution --
    ax = axes[1, 0]
    if max_motion_valid:
        ax.hist(max_motion_valid, bins=50, alpha=0.7, color='green', label='Valid transitions')
    if max_motion_invalid:
        ax.hist(max_motion_invalid, bins=50, alpha=0.7, color='red', label='Invalid transitions')
    ax.set_xlabel('Max motion per agent (m/step)')
    ax.set_ylabel('Count')
    ax.set_title('Max Motion Magnitude (Valid vs Invalid transitions)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # -- (1,1): Agents per scenario --
    ax = axes[1, 1]
    ax.hist(num_agents_list, bins=30, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of agents')
    ax.set_ylabel('Count')
    ax.set_title(f'Agents per Scenario (mean={np.mean(num_agents_list):.1f})')

    plt.tight_layout()
    plt.savefig(out_dir / 'global_statistics.png', dpi=150)
    plt.close()


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    files = list_binary_files(data_dir, 'training')
    map_codebook = load_map_codebook()
    motion_data = load_motion_codebook()
    motion_codebook = motion_data['token']['veh']  # (2048, 4, 2)

    # Pick 5 diverse scenarios (spread across the dataset)
    indices = [0, 100, 1000, 5000, 20000]
    scenarios = []

    for idx in indices:
        if idx >= len(files):
            continue
        print(f"Loading scenario {idx}: {files[idx]}")
        scenario = read_binary_scenario(files[idx])
        scenarios.append(scenario)

        print(f"  Objects: {len(scenario['objects'])}, Roads: {len(scenario['roads'])}")
        print(f"  SDC index: {scenario['sdc_track_index']}")
        if scenario['objects']:
            sdc = scenario['objects'][scenario['sdc_track_index']]
            valid = sdc['traj_valid'].astype(bool)
            print(f"  SDC valid timesteps: {valid.sum()}/{len(valid)}")
            if valid.any():
                vt = np.where(valid)[0]
                print(f"  SDC position range: x=[{sdc['traj_x'][valid].min():.1f}, {sdc['traj_x'][valid].max():.1f}]"
                      f"  y=[{sdc['traj_y'][valid].min():.1f}, {sdc['traj_y'][valid].max():.1f}]")

        print(f"  Generating plots...")
        plot_scenario_overview(scenario, idx, out_dir)
        plot_valid_invalid_analysis(scenario, idx, out_dir)
        plot_map_tokens(scenario, idx, map_codebook, out_dir)
        plot_model_input(scenario, idx, map_codebook, motion_codebook, out_dir)

    # Global statistics
    print(f"\nGenerating global statistics from {len(scenarios)} scenarios...")
    plot_coordinate_statistics(scenarios, out_dir)

    print(f"\nAll plots saved to {out_dir}/")
    print("Files:")
    for f in sorted(out_dir.glob('*.png')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
