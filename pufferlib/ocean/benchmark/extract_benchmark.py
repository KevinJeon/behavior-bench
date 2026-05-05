# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Extract interactivity scores from binary scenario files.

This script reads binary files from a specified split directory and calculates
an interactivity score for each scenario based on:
- Number of agents
- Changes in acceleration and steering (computed from velocity trajectories)
- Average distance to the nearest agent

Usage:
    python extract_benchmark.py --split /path/to/binaries/training
    python extract_benchmark.py --split /path/to/binaries/testing --output scores.csv
    python extract_benchmark.py --split /path/to/binaries/training --plot --plot-dir plots/
    python extract_benchmark.py --split /path/to/binaries/training --plot --plot-top 20
"""

import argparse
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Entity type constants (matching drive.h)
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
ROAD_LANE = 4


@dataclass
class Entity:
    """Represents an entity loaded from a binary file."""
    scenario_id: int
    entity_type: int
    entity_id: int
    array_size: int
    traj_x: np.ndarray
    traj_y: np.ndarray
    traj_z: np.ndarray
    traj_vx: Optional[np.ndarray]
    traj_vy: Optional[np.ndarray]
    traj_vz: Optional[np.ndarray]
    traj_heading: Optional[np.ndarray]
    traj_valid: Optional[np.ndarray]
    width: float
    length: float
    height: float
    goal_position_x: float
    goal_position_y: float
    goal_position_z: float
    mark_as_expert: int
    exit_lane_ids: List[int]

    @property
    def is_agent(self) -> bool:
        """Check if this entity is an agent (vehicle, pedestrian, or cyclist)."""
        return self.entity_type in (VEHICLE, PEDESTRIAN, CYCLIST)


@dataclass
class ScenarioData:
    """Data loaded from a binary scenario file."""
    sdc_track_index: int
    num_tracks_to_predict: int
    tracks_to_predict_indices: List[int]
    num_objects: int
    num_roads: int
    entities: List[Entity]
    filename: str


def load_binary_file(filepath: str) -> Optional[ScenarioData]:
    """
    Load a binary scenario file.

    Args:
        filepath: Path to the binary file

    Returns:
        ScenarioData object or None if loading fails
    """
    try:
        with open(filepath, 'rb') as f:
            # Read sdc_track_index
            sdc_track_index = struct.unpack('i', f.read(4))[0]

            # Read tracks_to_predict
            num_tracks_to_predict = struct.unpack('i', f.read(4))[0]
            tracks_to_predict_indices = []
            for _ in range(num_tracks_to_predict):
                idx = struct.unpack('i', f.read(4))[0]
                tracks_to_predict_indices.append(idx)

            # Read num_objects and num_roads
            num_objects = struct.unpack('i', f.read(4))[0]
            num_roads = struct.unpack('i', f.read(4))[0]
            num_entities = num_objects + num_roads

            entities = []
            for _ in range(num_entities):
                # Read base entity data
                scenario_id = struct.unpack('i', f.read(4))[0]
                entity_type = struct.unpack('i', f.read(4))[0]
                entity_id = struct.unpack('i', f.read(4))[0]
                array_size = struct.unpack('i', f.read(4))[0]

                # Read trajectory arrays
                traj_x = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                traj_y = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                traj_z = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()

                # Read velocity and heading for agents (VEHICLE, PEDESTRIAN, CYCLIST)
                traj_vx = traj_vy = traj_vz = traj_heading = traj_valid = None
                if entity_type in (VEHICLE, PEDESTRIAN, CYCLIST):
                    traj_vx = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                    traj_vy = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                    traj_vz = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                    traj_heading = np.frombuffer(f.read(array_size * 4), dtype=np.float32).copy()
                    traj_valid = np.frombuffer(f.read(array_size * 4), dtype=np.int32).copy()

                # Read scalar fields
                width = struct.unpack('f', f.read(4))[0]
                length = struct.unpack('f', f.read(4))[0]
                height = struct.unpack('f', f.read(4))[0]
                goal_position_x = struct.unpack('f', f.read(4))[0]
                goal_position_y = struct.unpack('f', f.read(4))[0]
                goal_position_z = struct.unpack('f', f.read(4))[0]
                mark_as_expert = struct.unpack('i', f.read(4))[0]

                # Read exit_lanes connectivity (added for lane chaining)
                exit_lane_count = struct.unpack('i', f.read(4))[0]
                exit_lane_ids = []
                for _ in range(exit_lane_count):
                    exit_lane_ids.append(struct.unpack('i', f.read(4))[0])

                entity = Entity(
                    scenario_id=scenario_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    array_size=array_size,
                    traj_x=traj_x,
                    traj_y=traj_y,
                    traj_z=traj_z,
                    traj_vx=traj_vx,
                    traj_vy=traj_vy,
                    traj_vz=traj_vz,
                    traj_heading=traj_heading,
                    traj_valid=traj_valid,
                    width=width,
                    length=length,
                    height=height,
                    goal_position_x=goal_position_x,
                    goal_position_y=goal_position_y,
                    goal_position_z=goal_position_z,
                    mark_as_expert=mark_as_expert,
                    exit_lane_ids=exit_lane_ids,
                )
                entities.append(entity)

            return ScenarioData(
                sdc_track_index=sdc_track_index,
                num_tracks_to_predict=num_tracks_to_predict,
                tracks_to_predict_indices=tracks_to_predict_indices,
                num_objects=num_objects,
                num_roads=num_roads,
                entities=entities,
                filename=os.path.basename(filepath),
            )

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_agent_goal_distance(entity: Entity) -> float:
    """
    Get the initial distance from an agent to its goal.

    Args:
        entity: Entity to check

    Returns:
        Goal distance in meters, or -1 if invalid
    """
    if not entity.is_agent or entity.array_size == 0:
        return -1.0

    # Check validity at timestep 0
    if entity.traj_valid is not None and not entity.traj_valid[0]:
        return -1.0

    x0, y0 = entity.traj_x[0], entity.traj_y[0]
    gx, gy = entity.goal_position_x, entity.goal_position_y

    # Skip invalid goals
    if abs(gx) > 1e6 or abs(gy) > 1e6:
        return -1.0

    return float(np.sqrt((gx - x0)**2 + (gy - y0)**2))


def compute_acceleration_changes(entity: Entity, dt: float = 0.1) -> Tuple[float, float]:
    """
    Compute total acceleration and steering changes for an agent.

    Args:
        entity: Entity with velocity trajectories
        dt: Time step in seconds

    Returns:
        Tuple of (total_accel_change, total_steering_change)
    """
    if not entity.is_agent or entity.traj_vx is None or entity.traj_valid is None:
        return 0.0, 0.0

    # Get valid indices
    valid_mask = entity.traj_valid.astype(bool)
    if valid_mask.sum() < 3:
        return 0.0, 0.0

    # Compute speed from velocity components
    speed = np.sqrt(entity.traj_vx**2 + entity.traj_vy**2)

    # Compute acceleration (change in speed)
    accel = np.diff(speed) / dt

    # Compute acceleration changes (jerk-like metric)
    accel_changes = np.abs(np.diff(accel))

    # Compute heading changes (steering-like metric)
    heading = entity.traj_heading
    # Handle angle wrapping
    heading_diff = np.diff(heading)
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    steering_changes = np.abs(np.diff(heading_diff))

    # Only count valid timesteps - use length of the computed arrays
    n_accel = len(accel_changes)
    n_steer = len(steering_changes)

    # For accel_changes (length n-2), we need valid at indices 0..n-3 (first n-2 elements)
    valid_accel = valid_mask[:n_accel] & valid_mask[1:n_accel+1] & valid_mask[2:n_accel+2]
    # For steering_changes (length n-2), similar logic
    valid_steer = valid_mask[:n_steer] & valid_mask[1:n_steer+1] & valid_mask[2:n_steer+2]

    total_accel_change = np.sum(accel_changes[valid_accel]) if n_accel > 0 else 0.0
    total_steering_change = np.sum(steering_changes[valid_steer]) if n_steer > 0 else 0.0

    return float(total_accel_change), float(total_steering_change)


def compute_avg_min_distance(scenario: ScenarioData) -> float:
    """
    Compute the average minimum distance between agents over all timesteps.

    Args:
        scenario: Scenario data with entities

    Returns:
        Average minimum distance to nearest agent
    """
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) < 2:
        return float('inf')

    # Find common timestep range
    min_len = min(a.array_size for a in agents)
    if min_len == 0:
        return float('inf')

    min_distances = []

    for t in range(min_len):
        # Get valid positions at this timestep
        positions = []
        for agent in agents:
            if agent.traj_valid is not None and agent.traj_valid[t]:
                positions.append((agent.traj_x[t], agent.traj_y[t]))

        if len(positions) < 2:
            continue

        # Compute pairwise distances
        positions = np.array(positions)
        for i in range(len(positions)):
            distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
            # Exclude self (distance = 0)
            distances[i] = float('inf')
            min_dist = np.min(distances)
            min_distances.append(min_dist)

    if len(min_distances) == 0:
        return float('inf')

    return float(np.mean(min_distances))


def compute_avg_agents_in_radius(scenario: ScenarioData, radius: float = 10.0, min_goal_dist: float = 0.0) -> float:
    """
    Compute the average number of agents within a given radius for each agent.

    This is averaged across all agents and all valid timesteps.

    Args:
        scenario: Scenario data with entities
        radius: Radius in meters to count nearby agents (default: 10m)
        min_goal_dist: Only consider agents with goal distance >= this value

    Returns:
        Average number of agents within radius (excluding self)
    """
    # Filter agents by minimum goal distance
    agents = []
    for e in scenario.entities:
        if not e.is_agent:
            continue
        goal_dist = get_agent_goal_distance(e)
        if goal_dist >= min_goal_dist:
            agents.append(e)

    if len(agents) < 2:
        return 0.0

    # Find common timestep range
    min_len = min(a.array_size for a in agents)
    if min_len == 0:
        return 0.0

    counts = []

    for t in range(min_len):
        # Get valid positions at this timestep
        positions = []
        for agent in agents:
            if agent.traj_valid is not None and agent.traj_valid[t]:
                positions.append((agent.traj_x[t], agent.traj_y[t]))

        if len(positions) < 2:
            continue

        # Compute pairwise distances and count agents within radius
        positions = np.array(positions)
        for i in range(len(positions)):
            distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
            # Count agents within radius (excluding self at distance 0)
            agents_in_radius = np.sum((distances > 0) & (distances <= radius))
            counts.append(agents_in_radius)

    if len(counts) == 0:
        return 0.0

    return float(np.mean(counts))


def compute_avg_goal_distance(scenario: ScenarioData) -> float:
    """
    Compute the average initial distance from agents to their goals.

    Args:
        scenario: Scenario data with entities

    Returns:
        Average goal distance in meters (from initial position to goal)
    """
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) == 0:
        return 0.0

    distances = []
    for agent in agents:
        if agent.array_size == 0:
            continue
        # Check if agent is valid at timestep 0
        if agent.traj_valid is not None and not agent.traj_valid[0]:
            continue

        # Get initial position
        x0, y0 = agent.traj_x[0], agent.traj_y[0]
        # Get goal position
        gx, gy = agent.goal_position_x, agent.goal_position_y

        # Skip invalid goals (very large values indicate no goal)
        if abs(gx) > 1e6 or abs(gy) > 1e6:
            continue

        dist = np.sqrt((gx - x0)**2 + (gy - y0)**2)
        distances.append(dist)

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))


def scenario_to_viz_format(scenario: ScenarioData, timestep: int = 0) -> dict:
    """
    Convert ScenarioData to the format expected by plot_simulator_state.

    Args:
        scenario: ScenarioData object
        timestep: Which timestep to visualize (default: 0, the initial state)

    Returns:
        Dictionary in viz format with entities and active_agent_indices
    """
    entities = []
    active_agent_indices = []

    for idx, entity in enumerate(scenario.entities):
        # Clamp timestep to valid range
        t = min(timestep, entity.array_size - 1) if entity.array_size > 0 else 0

        # Determine validity
        valid = 1
        if entity.is_agent and entity.traj_valid is not None:
            valid = entity.traj_valid[t]

        # Build entity dict for visualization
        entity_dict = {
            'type': entity.entity_type,
            'x': entity.traj_x[t] if entity.array_size > 0 else 0,
            'y': entity.traj_y[t] if entity.array_size > 0 else 0,
            'width': entity.width,
            'length': entity.length,
            'goal_position_x': entity.goal_position_x,
            'goal_position_y': entity.goal_position_y,
            'traj_x': entity.traj_x,
            'traj_y': entity.traj_y,
            'valid': valid,
            'collision_state': 0,
        }

        # Add heading for agents
        if entity.is_agent and entity.traj_heading is not None:
            entity_dict['heading'] = entity.traj_heading[t]
        else:
            entity_dict['heading'] = 0.0

        entities.append(entity_dict)

        # Track active agents (vehicles, pedestrians, cyclists)
        if entity.is_agent:
            active_agent_indices.append(idx)

    return {
        'entities': entities,
        'active_agent_indices': active_agent_indices,
        'static_car_indices': [],
    }


def compute_axis_limits(scenario: ScenarioData, timestep: int = 0, padding: float = 30.0) -> tuple:
    """
    Compute axis limits based on AGENT positions only (not roads).
    Always includes the ego agent (first agent) in the limits.

    Args:
        scenario: ScenarioData object
        timestep: Which timestep to use for positions
        padding: Padding around the agents in meters

    Returns:
        Tuple of (xmin, xmax, ymin, ymax)
    """
    x_coords = []
    y_coords = []

    # Get agents
    agents = [e for e in scenario.entities if e.is_agent]

    # Always include ego agent (first agent) if valid
    if len(agents) > 0:
        ego = agents[0]
        if ego.array_size > 0:
            t = min(timestep, ego.array_size - 1)
            x, y = ego.traj_x[t], ego.traj_y[t]
            if abs(x) < 1e6 and abs(y) < 1e6:
                x_coords.append(x)
                y_coords.append(y)
            # Always include ego goal
            gx, gy = ego.goal_position_x, ego.goal_position_y
            if abs(gx) < 1e6 and abs(gy) < 1e6:
                x_coords.append(gx)
                y_coords.append(gy)

    # Add other agent positions
    for entity in agents[1:]:  # Skip ego, already added
        if entity.array_size == 0:
            continue
        t = min(timestep, entity.array_size - 1)

        # Check validity
        if entity.traj_valid is not None and not entity.traj_valid[t]:
            continue

        # Skip invalid positions
        x, y = entity.traj_x[t], entity.traj_y[t]
        if abs(x) > 1e6 or abs(y) > 1e6:
            continue

        x_coords.append(x)
        y_coords.append(y)

        # Also include goals
        gx, gy = entity.goal_position_x, entity.goal_position_y
        if abs(gx) < 1e6 and abs(gy) < 1e6:
            x_coords.append(gx)
            y_coords.append(gy)

    if not x_coords or not y_coords:
        return (-100, 100, -100, 100)

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    # Add padding
    xmin -= padding
    xmax += padding
    ymin -= padding
    ymax += padding

    # Make it square-ish for better visualization
    x_range = xmax - xmin
    y_range = ymax - ymin
    if x_range > y_range:
        diff = (x_range - y_range) / 2
        ymin -= diff
        ymax += diff
    else:
        diff = (y_range - x_range) / 2
        xmin -= diff
        xmax += diff

    return (xmin, xmax, ymin, ymax)


def plot_scenario(scenario: ScenarioData, scores: dict, output_path: str, min_goal_dist: float = 5.0) -> bool:
    """
    Plot a scenario with its interactivity score.

    Always uses timestep 0 (initial state). Ego agent (idx=0) is highlighted in red.

    Args:
        scenario: ScenarioData object
        scores: Dictionary with interactivity scores
        output_path: Path to save the plot
        min_goal_dist: Minimum goal distance for filtering agents

    Returns:
        True if plot was created, False if skipped (ego not valid)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pufferlib.viz import plot_simulator_state

    # Always use timestep 0
    timestep = 0

    # Check if ego is valid at timestep 0
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) == 0:
        print(f"[SKIP] {scenario.filename}: No agents")
        return False

    ego = agents[0]
    if ego.array_size == 0:
        print(f"[SKIP] {scenario.filename}: Ego has no trajectory")
        return False

    if ego.traj_valid is not None and not ego.traj_valid[timestep]:
        print(f"[SKIP] {scenario.filename}: Ego not valid at t=0")
        return False

    # Convert to viz format
    viz_data = scenario_to_viz_format(scenario, timestep)

    # Compute axis limits based on entity positions
    axis_limits = compute_axis_limits(scenario, timestep)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot the scenario (ego agent idx=0 will be handled by plot_simulator_state with test_agent_idx=0)
    plot_simulator_state(viz_data, test_agent_idx=0, ax=ax, axis_limits=axis_limits)

    # Highlight ego agent with red filled rectangle
    x, y = ego.traj_x[timestep], ego.traj_y[timestep]
    heading = ego.traj_heading[timestep] if ego.traj_heading is not None else 0.0
    length, width = ego.length, ego.width

    # Create rotated rectangle for ego highlight (filled red)
    from matplotlib.transforms import Affine2D
    rect = patches.Rectangle(
        (-length/2, -width/2), length, width,
        linewidth=4, edgecolor='darkred', facecolor='red', alpha=0.7, zorder=100
    )
    transform = Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)

    # Also add a marker at the center for visibility at any zoom level
    ax.plot(x, y, 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=1, zorder=101)

    # Plot ego goal as green star
    gx, gy = ego.goal_position_x, ego.goal_position_y
    if abs(gx) < 1e6 and abs(gy) < 1e6:
        ax.plot(gx, gy, 'g*', markersize=20, markeredgecolor='darkgreen', markeredgewidth=2, zorder=102, label='Ego Goal')
        # Draw line from ego to goal
        ax.plot([x, gx], [y, gy], 'g--', linewidth=2, alpha=0.5, zorder=99)

    # Add title with interactivity score
    title = (
        f"{scenario.filename}\n"
        f"Interactivity: {scores['interactivity_score']:.3f} | "
        f"Visible@t0: {scores['agents_valid_t0']} | "
        f"In 10m: {scores['avg_agents_in_radius']:.1f} | "
        f"Ego Goal: {scores['ego_goal_distance']:.0f}m"
    )
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Save
    fig.tight_layout()
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return True


def plot_scenario_gif(scenario: ScenarioData, scores: dict, output_path: str,
                      min_goal_dist: float = 5.0, fps: int = 10, step: int = 1,
                      ego_idx: int = 0) -> bool:
    """
    Render an animated GIF of a scenario showing agent movement over time.

    Args:
        scenario: ScenarioData object
        scores: Dictionary with interactivity scores
        output_path: Path to save the GIF
        min_goal_dist: Minimum goal distance for filtering
        fps: Frames per second
        step: Timestep increment per frame (1 = every step, 2 = every other, etc.)
        ego_idx: Index of the ego agent in the agent list

    Returns:
        True if GIF was created, False if skipped
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    from pufferlib.viz import plot_simulator_state
    from PIL import Image
    import io

    agents = [e for e in scenario.entities if e.is_agent]
    if ego_idx >= len(agents):
        return False

    ego = agents[ego_idx]
    if ego.array_size < 2 or ego.traj_valid is None:
        return False

    # Find max timestep across all agents
    max_t = max(e.array_size for e in agents if e.array_size > 0)

    # Compute fixed axis limits across all timesteps (use timestep 0)
    axis_limits = compute_axis_limits(scenario, timestep=0)

    # Map ego_idx (agent-only index) to entity index for plot_simulator_state
    agent_count = 0
    ego_entity_idx = 0
    for i, e in enumerate(scenario.entities):
        if e.is_agent:
            if agent_count == ego_idx:
                ego_entity_idx = i
                break
            agent_count += 1

    frames = []
    timesteps = list(range(0, max_t, step))

    for t in timesteps:
        fig, ax = plt.subplots(figsize=(10, 10))

        viz_data = scenario_to_viz_format(scenario, timestep=t)
        plot_simulator_state(viz_data, test_agent_idx=ego_entity_idx, ax=ax, axis_limits=axis_limits)

        # Highlight ego with red rectangle
        et = min(t, ego.array_size - 1)
        if ego.traj_valid is not None and ego.traj_valid[et]:
            x, y = ego.traj_x[et], ego.traj_y[et]
            heading = ego.traj_heading[et] if ego.traj_heading is not None else 0.0
            rect = patches.Rectangle(
                (-ego.length / 2, -ego.width / 2), ego.length, ego.width,
                linewidth=3, edgecolor='darkred', facecolor='red', alpha=0.7, zorder=100
            )
            transform = Affine2D().rotate(heading).translate(x, y) + ax.transData
            rect.set_transform(transform)
            ax.add_patch(rect)

            # Draw ego trail (past positions)
            trail_t = list(range(max(0, t - 20), t + 1))
            trail_x = [ego.traj_x[min(tt, ego.array_size - 1)] for tt in trail_t
                        if ego.traj_valid[min(tt, ego.array_size - 1)]]
            trail_y = [ego.traj_y[min(tt, ego.array_size - 1)] for tt in trail_t
                        if ego.traj_valid[min(tt, ego.array_size - 1)]]
            if len(trail_x) > 1:
                ax.plot(trail_x, trail_y, 'r-', linewidth=2, alpha=0.5, zorder=99)

        # Ego goal
        gx, gy = ego.goal_position_x, ego.goal_position_y
        if abs(gx) < 1e6 and abs(gy) < 1e6:
            ax.plot(gx, gy, 'g*', markersize=18, markeredgecolor='darkgreen',
                    markeredgewidth=2, zorder=102)

        ax.set_title(
            f"{scenario.filename} (ego={ego_idx})  |  Score: {scores['interactivity_score']:.3f}  |  t={t}/{max_t-1}",
            fontsize=11, fontweight='bold'
        )

        fig.tight_layout()

        # Render to PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if not frames:
        return False

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    return True


def plot_scenario_paper(scenario: ScenarioData, scores: dict, output_path: str,
                        ego_idx: int = 0, show_ego_trajectory: bool = True,
                        title: str = None, figsize: tuple = (10, 8)) -> bool:
    """
    Render a publication-quality static plot of a scenario at t=0.

    Args:
        scenario: ScenarioData object
        scores: Dictionary with interactivity scores
        output_path: Path to save (PNG). PDF is saved alongside automatically.
        ego_idx: Index of the ego agent in the agent list
        show_ego_trajectory: Whether to show ego GT trajectory
        title: Custom title (default: auto-generated from scores)
        figsize: Figure size in inches

    Returns:
        True if plot was created, False if skipped
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
    })

    agents = [e for e in scenario.entities if e.is_agent]
    if ego_idx >= len(agents):
        return False

    ego = agents[ego_idx]
    if ego.array_size < 2 or ego.traj_valid is None or not ego.traj_valid[0]:
        return False

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_facecolor('#f8f8f8')

    # --- Draw road infrastructure ---
    ROAD_LANE = 4
    ROAD_LINE = 5
    ROAD_EDGE = 6
    STOP_SIGN = 7
    CROSSWALK = 8

    for entity in scenario.entities:
        if entity.entity_type == ROAD_LANE:
            ax.plot(entity.traj_x, entity.traj_y, color='#999999', linewidth=0.6,
                    zorder=1, alpha=0.7)
        elif entity.entity_type == ROAD_LINE:
            ax.plot(entity.traj_x, entity.traj_y, color='#bbbbbb', linewidth=0.5,
                    linestyle='--', zorder=1, alpha=0.6)
        elif entity.entity_type == ROAD_EDGE:
            ax.plot(entity.traj_x, entity.traj_y, color='#333333', linewidth=1.2,
                    zorder=2)
        elif entity.entity_type == STOP_SIGN:
            ax.scatter(entity.traj_x, entity.traj_y, color='#cc0000', s=40,
                       marker='H', zorder=3, alpha=0.8)
        elif entity.entity_type == CROSSWALK:
            from matplotlib.patches import Polygon
            points = np.column_stack([entity.traj_x, entity.traj_y])
            if len(points) >= 3:
                ax.add_patch(Polygon(points, facecolor='none', edgecolor='#888888',
                                     linewidth=1, alpha=0.4, hatch='//', zorder=2))

    # --- Draw ego trajectory only ---
    if show_ego_trajectory:
        valid_mask = ego.traj_valid[:ego.array_size] if ego.traj_valid is not None else np.ones(ego.array_size)
        tx = ego.traj_x[:ego.array_size]
        ty = ego.traj_y[:ego.array_size]
        valid_x = tx[valid_mask.astype(bool)]
        valid_y = ty[valid_mask.astype(bool)]
        if len(valid_x) >= 2:
            ax.plot(valid_x, valid_y, color='#cc3333', linewidth=1.5, alpha=0.5,
                    zorder=3, linestyle='-')

    # --- Draw vehicles at t=0 with direction arrows ---
    ego_x, ego_y = ego.traj_x[0], ego.traj_y[0]

    for a_idx, agent in enumerate(agents):
        if agent.array_size == 0:
            continue
        if agent.traj_valid is not None and not agent.traj_valid[0]:
            continue

        x, y = agent.traj_x[0], agent.traj_y[0]
        if abs(x) > 1e6 or abs(y) > 1e6:
            continue

        heading = agent.traj_heading[0] if agent.traj_heading is not None else 0.0
        length, width = agent.length, agent.width

        if a_idx == ego_idx:
            facecolor = '#cc3333'
            edgecolor = '#661111'
            alpha = 0.85
            linewidth = 2.0
            zorder = 100
            arrow_color = '#ffffff'
        else:
            facecolor = '#4477aa'
            edgecolor = '#223355'
            alpha = 0.6
            linewidth = 1.0
            zorder = 10
            arrow_color = '#ffffff'

        rect = patches.Rectangle(
            (-length / 2, -width / 2), length, width,
            linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,
            alpha=alpha, zorder=zorder,
        )
        transform = Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(transform)
        ax.add_patch(rect)

        # Chevron ">" direction indicator — smaller, shifted forward
        chev_len = length * 0.12
        chev_spread = width * 0.22
        fwd_shift = length * 0.15
        cx = x + np.cos(heading) * fwd_shift
        cy = y + np.sin(heading) * fwd_shift
        tip_x = cx + np.cos(heading) * chev_len
        tip_y = cy + np.sin(heading) * chev_len
        perp_x = -np.sin(heading) * chev_spread
        perp_y = np.cos(heading) * chev_spread
        back_x = cx - np.cos(heading) * chev_len * 0.5
        back_y = cy - np.sin(heading) * chev_len * 0.5
        ax.plot([back_x + perp_x, tip_x, back_x - perp_x],
                [back_y + perp_y, tip_y, back_y - perp_y],
                color='white', linewidth=1.4 if a_idx == ego_idx else 0.9,
                solid_capstyle='round', solid_joinstyle='round',
                alpha=0.95, zorder=zorder + 1)

    # --- Goal pin ---
    gx, gy = ego.goal_position_x, ego.goal_position_y
    if abs(gx) < 1e6 and abs(gy) < 1e6:
        # Pin: circle on a stick
        pin_radius = 1.8
        pin_stick_len = 4.5
        # Stick
        ax.plot([gx, gx], [gy, gy + pin_stick_len], color='#116622',
                linewidth=2.0, zorder=104, solid_capstyle='round')
        # Circle head
        pin_circle = patches.Circle((gx, gy + pin_stick_len), pin_radius,
                                     facecolor='#22aa44', edgecolor='#116622',
                                     linewidth=1.5, zorder=105)
        ax.add_patch(pin_circle)
        # Inner dot
        ax.plot(gx, gy + pin_stick_len, 'o', color='white', markersize=3,
                zorder=106)

    # --- Axis limits: tight around ego + goal only ---
    x_coords = [ego_x]
    y_coords = [ego_y]
    if abs(gx) < 1e6:
        x_coords.append(gx)
        y_coords.append(gy)

    pad = 15.0
    xmin, xmax = min(x_coords) - pad, max(x_coords) + pad
    ymin, ymax = min(y_coords) - pad, max(y_coords) + pad
    # Make square
    x_range = xmax - xmin
    y_range = ymax - ymin
    size = max(x_range, y_range, 50.0)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    ax.set_xlim(cx - size / 2, cx + size / 2)
    ax.set_ylim(cy - size / 2, cy + size / 2)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#cc3333',
               markeredgecolor='#661111', markersize=10, label='Ego Vehicle'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4477aa',
               markeredgecolor='#223355', markersize=10, label='Other Vehicles'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#22aa44',
               markeredgecolor='#116622', markersize=8, label='Goal'),
        Line2D([0], [0], color='#333333', linewidth=1.2, label='Road Edge'),
    ]
    if show_ego_trajectory:
        legend_elements.append(
            Line2D([0], [0], color='#cc3333', linewidth=1.5, alpha=0.5, label='Ego Trajectory')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              framealpha=0.9, edgecolor='#cccccc', fancybox=False)

    # --- Title ---
    if title is None:
        score_val = scores.get('interactivity_score', 0)
        n_agents = scores.get('agents_valid_t0', len(agents))
        title = f"Score: {score_val:.3f}  |  Agents: {n_agents}"

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    # Also save PDF
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def build_chained_lane_polyline(start_lane: Entity, all_lanes: dict, start_idx: int, max_length: float = 200.0):
    """Build a polyline by following exit_lanes until max_length is reached."""
    points_x = list(start_lane.traj_x[start_idx:])
    points_y = list(start_lane.traj_y[start_idx:])

    total_length = 0.0
    for i in range(len(points_x) - 1):
        total_length += np.sqrt((points_x[i+1] - points_x[i])**2 + (points_y[i+1] - points_y[i])**2)

    current_lane = start_lane
    visited = {current_lane.entity_id}
    while total_length < max_length and current_lane.exit_lane_ids:
        next_lane = None
        for eid in current_lane.exit_lane_ids:
            if eid in all_lanes and eid not in visited:
                next_lane = all_lanes[eid]
                break
        if next_lane is None:
            break

        visited.add(next_lane.entity_id)
        for i in range(1, next_lane.array_size):
            seg_len = np.sqrt((next_lane.traj_x[i] - points_x[-1])**2 + (next_lane.traj_y[i] - points_y[-1])**2)
            total_length += seg_len
            points_x.append(next_lane.traj_x[i])
            points_y.append(next_lane.traj_y[i])
            if total_length >= max_length:
                break
        current_lane = next_lane

    return np.array(points_x, dtype=np.float32), np.array(points_y, dtype=np.float32)


def compute_trajectory_overlaps(scenario: ScenarioData, radius: float = 3.0) -> int:
    """Count how many unique other agents come within radius of ego's GT trajectory."""
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) < 2:
        return 0
    ego = agents[0]
    if ego.array_size == 0 or ego.traj_valid is None:
        return 0

    overlaps = 0
    for agent in agents[1:]:
        if agent.array_size == 0 or agent.traj_valid is None:
            continue
        min_len = min(ego.array_size, agent.array_size)
        for t in range(min_len):
            if not ego.traj_valid[t] or not agent.traj_valid[t]:
                continue
            dx = ego.traj_x[t] - agent.traj_x[t]
            dy = ego.traj_y[t] - agent.traj_y[t]
            if np.sqrt(dx**2 + dy**2) <= radius:
                overlaps += 1
                break  # Count each agent only once
    return overlaps


def _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> bool:
    """Check if segment (ax1,ay1)-(ax2,ay2) intersects segment (bx1,by1)-(bx2,by2)."""
    dx_a = ax2 - ax1
    dy_a = ay2 - ay1
    dx_b = bx2 - bx1
    dy_b = by2 - by1

    d1 = dx_b * (ay1 - by1) - dy_b * (ax1 - bx1)
    d2 = dx_b * (ay2 - by1) - dy_b * (ax2 - bx1)
    d3 = dx_a * (by1 - ay1) - dy_a * (bx1 - ax1)
    d4 = dx_a * (by2 - ay1) - dy_a * (bx2 - ax1)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def compute_trajectory_crossings(scenario: ScenarioData, min_goal_dist: float = 5.0, ego_idx: int = 0) -> int:
    """
    Count how many unique other agents have GT trajectories that geometrically
    cross the ego's GT trajectory (line segment intersection).

    Args:
        scenario: Scenario data with entities
        min_goal_dist: Only consider agents with goal distance >= this value
        ego_idx: Index of the ego agent in the agent list

    Returns:
        Number of unique agents whose trajectory crosses ego's trajectory
    """
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) < 2:
        return 0
    ego = agents[ego_idx]
    if ego.array_size < 2 or ego.traj_valid is None:
        return 0

    # Build ego segments (only valid consecutive timesteps)
    ego_segs = []
    for t in range(ego.array_size - 1):
        if ego.traj_valid[t] and ego.traj_valid[t + 1]:
            ego_segs.append((ego.traj_x[t], ego.traj_y[t],
                             ego.traj_x[t + 1], ego.traj_y[t + 1]))
    if not ego_segs:
        return 0

    # Ego bounding box for pre-filter
    ego_all_x = np.concatenate([[s[0], s[2]] for s in ego_segs])
    ego_all_y = np.concatenate([[s[1], s[3]] for s in ego_segs])
    ego_xmin, ego_xmax = ego_all_x.min() - 20.0, ego_all_x.max() + 20.0
    ego_ymin, ego_ymax = ego_all_y.min() - 20.0, ego_all_y.max() + 20.0

    crossings = 0
    for i, agent in enumerate(agents):
        if i == ego_idx:
            continue
        if agent.array_size < 2 or agent.traj_valid is None:
            continue
        # Filter by goal distance
        goal_dist = get_agent_goal_distance(agent)
        if goal_dist >= 0 and goal_dist < min_goal_dist:
            continue

        # Bounding box pre-filter
        valid_mask = agent.traj_valid.astype(bool)
        if valid_mask.sum() < 2:
            continue
        ax = agent.traj_x[valid_mask]
        ay = agent.traj_y[valid_mask]
        if ax.max() < ego_xmin or ax.min() > ego_xmax or \
           ay.max() < ego_ymin or ay.min() > ego_ymax:
            continue

        # Check segment intersections
        found = False
        for t in range(agent.array_size - 1):
            if not agent.traj_valid[t] or not agent.traj_valid[t + 1]:
                continue
            bx1, by1 = agent.traj_x[t], agent.traj_y[t]
            bx2, by2 = agent.traj_x[t + 1], agent.traj_y[t + 1]
            for ax1, ay1, ax2, ay2 in ego_segs:
                if _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
                    found = True
                    break
            if found:
                break
        if found:
            crossings += 1

    return crossings


def compute_proximity_weighted_ego_dynamics(
    scenario: ScenarioData, min_goal_dist: float = 5.0, dt: float = 0.1, ego_idx: int = 0
) -> Tuple[float, float]:
    """
    Compute ego acceleration and steering changes weighted by proximity to nearest agent.

    Changes near other agents are weighted higher (1/dist), changes in isolation
    contribute little. This captures ego maneuvering *because of* other agents.

    Args:
        scenario: Scenario data with entities
        min_goal_dist: Only consider nearby agents with goal distance >= this value
        dt: Time step in seconds
        ego_idx: Index of the ego agent in the agent list

    Returns:
        Tuple of (weighted_accel_score, weighted_steering_score)
    """
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) < 2:
        return 0.0, 0.0
    ego = agents[ego_idx]
    if ego.array_size < 3 or ego.traj_vx is None or ego.traj_valid is None:
        return 0.0, 0.0

    # Filter other agents by goal distance
    others = []
    for i, a in enumerate(agents):
        if i == ego_idx:
            continue
        if a.traj_valid is None:
            continue
        gd = get_agent_goal_distance(a)
        if gd >= 0 and gd < min_goal_dist:
            continue
        others.append(a)
    if not others:
        return 0.0, 0.0

    # Compute ego speed and acceleration changes per timestep
    speed = np.sqrt(ego.traj_vx**2 + ego.traj_vy**2)
    accel = np.diff(speed) / dt
    accel_changes = np.abs(np.diff(accel))  # length = array_size - 2

    heading = ego.traj_heading
    heading_diff = np.diff(heading)
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    steering_changes = np.abs(np.diff(heading_diff))  # length = array_size - 2

    weighted_accel = 0.0
    weighted_steer = 0.0
    n = len(accel_changes)

    for t in range(n):
        # Need valid at t, t+1, t+2 for the change computation
        if not (ego.traj_valid[t] and ego.traj_valid[t + 1] and ego.traj_valid[t + 2]):
            continue

        # Find minimum distance to any valid other agent at timestep t+1 (center of window)
        tc = t + 1
        ex, ey = ego.traj_x[tc], ego.traj_y[tc]
        min_dist = float('inf')
        for other in others:
            if tc >= other.array_size or other.traj_valid is None or not other.traj_valid[tc]:
                continue
            dx = other.traj_x[tc] - ex
            dy = other.traj_y[tc] - ey
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d

        if min_dist == float('inf'):
            continue

        weight = 1.0 / max(min_dist, 1.0)
        weighted_accel += accel_changes[t] * weight
        weighted_steer += steering_changes[t] * weight

    return float(weighted_accel), float(weighted_steer)


def compute_ttc_critical_steps(scenario: ScenarioData, ttc_threshold: float = 3.0, ego_idx: int = 0) -> int:
    """Count timesteps where ego has TTC < threshold with any other agent."""
    agents = [e for e in scenario.entities if e.is_agent]
    if len(agents) < 2:
        return 0
    ego = agents[ego_idx]
    if ego.array_size == 0 or ego.traj_valid is None or ego.traj_vx is None:
        return 0

    critical_steps = 0
    for t in range(ego.array_size):
        if not ego.traj_valid[t]:
            continue
        for i, agent in enumerate(agents):
            if i == ego_idx:
                continue
            if t >= agent.array_size or agent.traj_valid is None or not agent.traj_valid[t]:
                continue
            if agent.traj_vx is None:
                continue
            dx = agent.traj_x[t] - ego.traj_x[t]
            dy = agent.traj_y[t] - ego.traj_y[t]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 50.0:
                continue
            dvx = ego.traj_vx[t] - agent.traj_vx[t]
            dvy = ego.traj_vy[t] - agent.traj_vy[t]
            if dist > 0:
                closing_speed = (dx * dvx + dy * dvy) / dist
            else:
                closing_speed = 0.0
            if closing_speed > 0.1:
                ttc = dist / closing_speed
                if ttc < ttc_threshold:
                    critical_steps += 1
                    break  # Count this timestep once
    return critical_steps


def compute_ego_heading_change(scenario: ScenarioData, ego_idx: int = 0) -> float:
    """Total absolute heading change of ego over its GT trajectory."""
    agents = [e for e in scenario.entities if e.is_agent]
    if not agents:
        return 0.0
    ego = agents[ego_idx]
    if ego.array_size < 2 or ego.traj_heading is None or ego.traj_valid is None:
        return 0.0

    total_change = 0.0
    prev_heading = None
    for t in range(ego.array_size):
        if not ego.traj_valid[t]:
            continue
        if prev_heading is not None:
            diff = abs(np.arctan2(np.sin(ego.traj_heading[t] - prev_heading),
                                   np.cos(ego.traj_heading[t] - prev_heading)))
            total_change += diff
        prev_heading = ego.traj_heading[t]
    return total_change


def compute_ego_goal_lane_multiplier(scenario: ScenarioData, lane_length: float = 300.0, tolerance: float = 5.0, ego_idx: int = 0) -> float:
    """
    Check if ego's goal lies on the ego's current lane.
    Returns 1.0 if goal is on a different lane (harder), 0.5 if on same lane (easier).
    """
    agents = [e for e in scenario.entities if e.is_agent]
    if not agents:
        return 1.0
    ego = agents[ego_idx]
    if ego.array_size == 0 or ego.traj_valid is None or not ego.traj_valid[0]:
        return 1.0

    ego_x, ego_y = ego.traj_x[0], ego.traj_y[0]
    ego_heading = ego.traj_heading[0] if ego.traj_heading is not None else 0.0
    goal_x, goal_y = ego.goal_position_x, ego.goal_position_y

    if abs(goal_x) > 1e6 or abs(goal_y) > 1e6:
        return 1.0  # No valid goal

    # Find all lanes
    lanes = [e for e in scenario.entities if e.entity_type == ROAD_LANE and e.array_size > 1]
    if not lanes:
        return 1.0

    # Find closest lane to ego position with heading alignment
    best_lane = None
    best_dist = float('inf')
    for lane in lanes:
        dx = lane.traj_x - ego_x
        dy = lane.traj_y - ego_y
        dists = np.sqrt(dx**2 + dy**2)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        # Check heading alignment at closest point
        if min_idx < lane.array_size - 1:
            lane_dx = lane.traj_x[min_idx + 1] - lane.traj_x[min_idx]
            lane_dy = lane.traj_y[min_idx + 1] - lane.traj_y[min_idx]
            lane_heading = np.arctan2(lane_dy, lane_dx)
            heading_diff = abs(np.arctan2(np.sin(ego_heading - lane_heading), np.cos(ego_heading - lane_heading)))
            if heading_diff > np.pi / 4:  # >45 degrees = wrong direction
                continue

        if min_dist < best_dist:
            best_dist = min_dist
            best_lane = lane

    if best_lane is None or best_dist > 10.0:
        return 1.0

    # Find start index on best lane
    dx = best_lane.traj_x - ego_x
    dy = best_lane.traj_y - ego_y
    dists_to_ego = np.sqrt(dx**2 + dy**2)
    start_idx = np.argmin(dists_to_ego)

    # Build chained polyline following exit_lanes up to ~lane_length
    lane_lookup = {e.entity_id: e for e in scenario.entities if e.entity_type == ROAD_LANE and e.array_size > 1}
    lane_points_x, lane_points_y = build_chained_lane_polyline(best_lane, lane_lookup, start_idx, max_length=lane_length)

    # Check if goal is within tolerance of this lane segment
    goal_dx = lane_points_x - goal_x
    goal_dy = lane_points_y - goal_y
    goal_dists = np.sqrt(goal_dx**2 + goal_dy**2)

    if np.min(goal_dists) <= tolerance:
        return 0.5  # Goal on same lane = easier
    return 1.0  # Goal on different lane = harder


def compute_interactivity_score(scenario: ScenarioData, min_goal_dist: float = 10.0) -> Optional[dict]:
    """
    Compute interactivity score for a scenario.

    Only considers agents with goal distance >= min_goal_dist.
    Returns None if ego agent (idx=0) has goal < min_goal_dist.

    Args:
        scenario: Scenario data
        min_goal_dist: Minimum goal distance to consider an agent (default: 5m)

    Returns:
        Dictionary with score components and total score, or None if scenario should be skipped
    """
    all_agents = [e for e in scenario.entities if e.is_agent]
    if len(all_agents) == 0:
        return None

    # Check ego agent (first agent, idx=0) goal distance
    ego_agent = all_agents[0]
    ego_goal_dist = get_agent_goal_distance(ego_agent)
    if ego_goal_dist < min_goal_dist:
        return None  # Skip this scenario

    # Filter agents by minimum goal distance AND valid at t=0
    filtered_agents = []
    agents_valid_t0 = 0
    for agent in all_agents:
        goal_dist = get_agent_goal_distance(agent)
        if goal_dist >= min_goal_dist:
            filtered_agents.append(agent)
            # Check if valid at t=0
            if agent.array_size > 0 and agent.traj_valid is not None and agent.traj_valid[0]:
                agents_valid_t0 += 1

    num_agents = len(filtered_agents)
    if num_agents == 0:
        return None

    # Count agents within 50m of ego at t=0
    ego_x, ego_y = ego_agent.traj_x[0], ego_agent.traj_y[0]
    agents_near_ego = 0
    for agent in filtered_agents:
        if agent.array_size == 0 or agent.traj_valid is None or not agent.traj_valid[0]:
            continue
        dx = agent.traj_x[0] - ego_x
        dy = agent.traj_y[0] - ego_y
        if np.sqrt(dx**2 + dy**2) <= 50.0:
            agents_near_ego += 1
    num_agents_nearby = agents_near_ego

    # Require minimum agents visible at t=0 for the scenario to be interesting
    min_visible_t0 = 5
    if agents_valid_t0 < min_visible_t0:
        return None  # Skip scenarios with too few visible agents at t=0

    # Compute acceleration and steering changes for filtered agents only
    total_accel_change = 0.0
    total_steering_change = 0.0
    for agent in filtered_agents:
        accel, steer = compute_acceleration_changes(agent)
        total_accel_change += accel
        total_steering_change += steer

    # Normalize by number of filtered agents
    avg_accel_change = total_accel_change / max(num_agents, 1)
    avg_steering_change = total_steering_change / max(num_agents, 1)

    # Compute average minimum distance (for reference, using all agents)
    avg_min_distance = compute_avg_min_distance(scenario)

    # Compute average number of filtered agents within 10m radius
    avg_agents_in_radius = compute_avg_agents_in_radius(scenario, radius=10.0, min_goal_dist=min_goal_dist)

    # Compute interactivity score
    # Higher score = more interactive (more agents, more behavior changes, more crowded, longer goal)

    # Density score: average agents in 10m radius (normalized, max ~3 agents = very dense)
    density_score = min(avg_agents_in_radius / 3.0, 1.0)

    # Goal distance score: longer distances = more interesting
    # Normalize: 10m (min threshold) = 0, 50m = 1.0
    goal_dist_score = min(max(ego_goal_dist - min_goal_dist, 0) / 40.0, 1.0)

    # New interaction metrics
    trajectory_overlaps = compute_trajectory_overlaps(scenario, radius=5.0)
    ttc_critical_steps = compute_ttc_critical_steps(scenario, ttc_threshold=3.0)
    ego_heading_change = compute_ego_heading_change(scenario)

    # Lane-change multiplier: 1.0 if goal on different lane, 0.5 if on same lane
    lane_multiplier = compute_ego_goal_lane_multiplier(scenario)

    # Combined interactivity score
    interactivity_score = (
        0.10 * min(num_agents_nearby / 15.0, 1.0) +      # Agents within 50m of ego
        0.10 * min(avg_accel_change / 30.0, 1.0) +        # Acceleration changes
        0.05 * min(avg_steering_change / 2.0, 1.0) +      # Steering changes
        0.10 * density_score +                              # Average agents within 10m radius
        0.15 * goal_dist_score +                            # Ego goal distance
        0.25 * min(trajectory_overlaps / 5.0, 1.0) +      # Agents crossing ego GT path
        0.20 * min(ttc_critical_steps / 20.0, 1.0) +      # Critical TTC timesteps
        0.05 * min(ego_heading_change / (np.pi / 2), 1.0)  # Ego heading change
    ) * lane_multiplier

    return {
        'filename': scenario.filename,
        'num_agents': num_agents,
        'num_agents_nearby': num_agents_nearby,
        'num_agents_total': len(all_agents),
        'agents_valid_t0': agents_valid_t0,
        'num_objects': scenario.num_objects,
        'num_roads': scenario.num_roads,
        'ego_goal_distance': ego_goal_dist,
        'avg_accel_change': avg_accel_change,
        'avg_steering_change': avg_steering_change,
        'avg_min_distance': avg_min_distance if avg_min_distance != float('inf') else -1,
        'avg_agents_in_radius': avg_agents_in_radius,
        'density_score': density_score,
        'goal_dist_score': goal_dist_score,
        'trajectory_overlaps': trajectory_overlaps,
        'ttc_critical_steps': ttc_critical_steps,
        'ego_heading_change': ego_heading_change,
        'lane_multiplier': lane_multiplier,
        'interactivity_score': interactivity_score,
    }


def compute_interactivity_score_v2(scenario: ScenarioData, min_goal_dist: float = 10.0, ego_idx: int = 0) -> Optional[dict]:
    """
    Compute interactivity score v2 for a scenario.

    Designed to correlate with PPO policy performance (higher score = harder scenario).
    Components:
    - Geometric trajectory crossings (ego vs other agents)
    - Proximity-weighted ego acceleration/steering changes
    - TTC critical timesteps for ego
    - Number of agents near ego
    - Ego goal distance
    - Ego heading change

    Args:
        scenario: Scenario data
        min_goal_dist: Minimum goal distance for ego and agent filtering
        ego_idx: Index of the ego agent in the agent list

    Returns:
        Dictionary with score components and total score, or None if scenario should be skipped
    """
    all_agents = [e for e in scenario.entities if e.is_agent]
    if ego_idx >= len(all_agents):
        return None

    # Check ego agent is a car (filter out motorcycles, pedestrians, cyclists)
    ego_agent = all_agents[ego_idx]
    if ego_agent.entity_type != VEHICLE or ego_agent.length < 3.0:
        return None

    # Check ego agent goal distance
    ego_goal_dist = get_agent_goal_distance(ego_agent)
    if ego_goal_dist < min_goal_dist:
        return None

    # Count agents valid at t=0
    agents_valid_t0 = 0
    for agent in all_agents:
        if agent.array_size > 0 and agent.traj_valid is not None and agent.traj_valid[0]:
            agents_valid_t0 += 1

    if agents_valid_t0 < 3:
        return None

    # Count agents within 40m of ego at t=0 (with goal filter)
    ego_x, ego_y = ego_agent.traj_x[0], ego_agent.traj_y[0]
    agents_near_ego = 0
    for i, agent in enumerate(all_agents):
        if i == ego_idx:
            continue
        if agent.array_size == 0 or agent.traj_valid is None or not agent.traj_valid[0]:
            continue
        gd = get_agent_goal_distance(agent)
        if 0 <= gd < min_goal_dist:
            continue
        dx = agent.traj_x[0] - ego_x
        dy = agent.traj_y[0] - ego_y
        if np.sqrt(dx * dx + dy * dy) <= 40.0:
            agents_near_ego += 1

    # Compute components
    trajectory_crossings = compute_trajectory_crossings(scenario, min_goal_dist=min_goal_dist, ego_idx=ego_idx)
    weighted_accel, weighted_steer = compute_proximity_weighted_ego_dynamics(
        scenario, min_goal_dist=min_goal_dist, ego_idx=ego_idx
    )
    ttc_critical_steps = compute_ttc_critical_steps(scenario, ttc_threshold=3.0, ego_idx=ego_idx)
    ego_heading_change = compute_ego_heading_change(scenario, ego_idx=ego_idx)
    lane_multiplier = compute_ego_goal_lane_multiplier(scenario, ego_idx=ego_idx)

    # Normalize components to [0, 1]
    # Normalization limits chosen from p75-p90 of training set distribution
    c_crossings = min(trajectory_crossings / 4.0, 1.0)
    c_weighted_accel = min(weighted_accel / 60.0, 1.0)
    c_weighted_steer = min(weighted_steer / 0.1, 1.0)
    c_ttc = min(ttc_critical_steps / 60.0, 1.0)
    c_agents = min(agents_near_ego / 10.0, 1.0)
    c_goal_dist = min(max(ego_goal_dist - min_goal_dist, 0) / 100.0, 1.0)
    c_heading = min(ego_heading_change / (np.pi / 2), 1.0)

    # Weighted combination
    interactivity_score = (
        0.30 * c_crossings +
        0.15 * c_weighted_accel +
        0.15 * c_weighted_steer +
        0.20 * c_ttc +
        0.10 * c_agents +
        0.10 * c_goal_dist
    ) * lane_multiplier

    return {
        'filename': scenario.filename,
        'ego_agent_idx': ego_idx,
        'num_agents_total': len(all_agents),
        'agents_valid_t0': agents_valid_t0,
        'agents_near_ego': agents_near_ego,
        'ego_goal_distance': ego_goal_dist,
        'trajectory_crossings': trajectory_crossings,
        'weighted_accel': weighted_accel,
        'weighted_steer': weighted_steer,
        'ttc_critical_steps': ttc_critical_steps,
        'ego_heading_change': ego_heading_change,
        'lane_multiplier': lane_multiplier,
        'c_crossings': c_crossings,
        'c_weighted_accel': c_weighted_accel,
        'c_weighted_steer': c_weighted_steer,
        'c_ttc': c_ttc,
        'c_agents': c_agents,
        'c_goal_dist': c_goal_dist,
        'c_heading': c_heading,
        'interactivity_score': interactivity_score,
    }


def compute_all_ego_scores(scenario: ScenarioData, min_goal_dist: float = 10.0) -> list:
    """Compute interactivity scores for every valid car agent as ego.

    Returns:
        List of score dicts, one per qualifying agent.
    """
    agents = [e for e in scenario.entities if e.is_agent]
    results = []
    for idx in range(len(agents)):
        scores = compute_interactivity_score_v2(scenario, min_goal_dist, ego_idx=idx)
        if scores is not None:
            results.append(scores)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract interactivity scores from binary scenario files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help='Path to split directory (e.g., /path/to/binaries/training)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: interactivity_scores_<split>.csv)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)',
    )
    parser.add_argument(
        '--sort-by',
        type=str,
        choices=['interactivity_score', 'num_agents', 'avg_min_distance', 'avg_accel_change', 'avg_agents_in_radius'],
        default='interactivity_score',
        help='Sort results by this column (descending)',
    )
    parser.add_argument(
        '--min-goal-dist',
        type=float,
        default=10.0,
        help='Minimum goal distance for ego agent and filtering (default: 10m)',
    )
    parser.add_argument(
        '--extract',
        type=int,
        default=None,
        help='Extract top N scenarios to --extract-dir',
    )
    parser.add_argument(
        '--extract-dir',
        type=str,
        default=None,
        help='Directory to copy extracted scenario files to',
    )
    parser.add_argument(
        '--random',
        type=int,
        default=None,
        help='Randomly extract N scenarios to --extract-dir (skips scoring)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible random extraction (default: 42)',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots for scenarios',
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='scenario_plots',
        help='Directory to save plots',
    )
    parser.add_argument(
        '--plot-top',
        type=int,
        default=None,
        help='Only plot top N scenarios (by interactivity score)',
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.random and args.extract:
        parser.error("Cannot use both --extract and --random. Choose one.")
    if (args.random or args.extract) and not args.extract_dir:
        parser.error("--extract-dir is required when using --extract or --random")

    # Check if split directory exists
    split_path = Path(args.split)
    if not split_path.exists():
        print(f"Error: Split directory not found: {args.split}")
        return

    # Get all binary files
    binary_files = sorted(split_path.glob('*'))
    binary_files = [f for f in binary_files if f.is_file()]

    if args.limit:
        binary_files = binary_files[:args.limit]

    print(f"Found {len(binary_files)} binary files in {args.split}")

    # Random extraction mode - skip scoring entirely
    if args.random:
        import random
        import shutil

        if args.random > len(binary_files):
            print(f"Warning: Requested {args.random} but only {len(binary_files)} files available. Extracting all.")
            args.random = len(binary_files)

        random.seed(args.seed)
        selected_files = random.sample(binary_files, args.random)
        selected_files.sort()  # Sort for deterministic ordering after sampling

        extract_dir = Path(args.extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRandomly extracting {len(selected_files)} scenarios (seed={args.seed}) to {extract_dir}/")

        manifest_rows = []
        for idx, src_path in enumerate(tqdm(selected_files, desc="Extracting random scenarios")):
            new_filename = f"map_{idx:06d}.bin"
            dst_path = extract_dir / new_filename
            shutil.copy2(src_path, dst_path)
            manifest_rows.append({
                'new_filename': new_filename,
                'original_filename': src_path.name,
                'seed': args.seed,
            })

        # Save manifest
        manifest_path = extract_dir / "manifest.csv"
        with open(manifest_path, 'w') as f:
            f.write('new_filename,original_filename,seed\n')
            for row in manifest_rows:
                f.write(f"{row['new_filename']},{row['original_filename']},{row['seed']}\n")

        print(f"Extracted {len(manifest_rows)} scenario files to {extract_dir}/")
        print(f"Manifest saved to {manifest_path}")
        return  # Exit early, no scoring needed

    print(f"Filtering: ego goal >= {args.min_goal_dist}m, agents with goal >= {args.min_goal_dist}m, min 5 visible at t=0")

    # Process files
    results = []
    scenarios = {}  # Store scenarios for plotting
    skipped = 0
    for filepath in tqdm(binary_files, desc="Processing scenarios"):
        scenario = load_binary_file(str(filepath))
        if scenario is not None:
            scores = compute_interactivity_score(scenario, min_goal_dist=args.min_goal_dist)
            if scores is not None:
                results.append(scores)
                if args.plot:
                    scenarios[scenario.filename] = scenario
            else:
                skipped += 1

    print(f"\nProcessed {len(results)} scenarios successfully (skipped {skipped} not meeting criteria)")

    # Sort results
    if args.sort_by == 'avg_min_distance':
        # For distance, lower is more interactive, but -1 means no data
        results.sort(key=lambda x: x[args.sort_by] if x[args.sort_by] > 0 else float('inf'))
    else:
        results.sort(key=lambda x: x[args.sort_by], reverse=True)

    # Extract top N scenarios if requested
    if args.extract and args.extract_dir:
        import shutil
        extract_dir = Path(args.extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        scenarios_to_extract = results[:args.extract]
        print(f"\nExtracting top {len(scenarios_to_extract)} scenarios to {extract_dir}/")
        print(f"Files will be renamed to sequential format: map_000000.bin, map_000001.bin, ...")

        extracted_scenarios = []  # Track successfully extracted scenarios
        for idx, scores in enumerate(tqdm(scenarios_to_extract, desc="Extracting scenarios")):
            src_path = split_path / scores['filename']
            if src_path.exists():
                # Rename to sequential format for C binding compatibility
                new_filename = f"map_{idx:06d}.bin"
                dst_path = extract_dir / new_filename
                shutil.copy2(src_path, dst_path)
                # Store mapping in scores for manifest
                scores['original_filename'] = scores['filename']
                scores['new_filename'] = new_filename
                extracted_scenarios.append(scores)
            else:
                print(f"[WARN] Source file not found: {src_path}")

        print(f"Extracted {len(extracted_scenarios)} scenario files to {extract_dir}/")

        # Also save a manifest with scores (includes original->new filename mapping)
        manifest_path = extract_dir / "manifest.csv"
        with open(manifest_path, 'w') as f:
            if extracted_scenarios:
                # Put new_filename and original_filename first for clarity
                headers = ['new_filename', 'original_filename'] + [h for h in extracted_scenarios[0].keys()
                           if h not in ('new_filename', 'original_filename', 'filename')]
                f.write(','.join(headers) + '\n')
                for row in extracted_scenarios:
                    values = [str(row.get(h, '')) for h in headers]
                    f.write(','.join(values) + '\n')
        print(f"Manifest saved to {manifest_path}")

    # Generate plots if requested
    if args.plot and scenarios:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Determine which scenarios to plot
        if args.plot_top:
            scenarios_to_plot = results[:args.plot_top]
        else:
            scenarios_to_plot = results

        print(f"\nGenerating {len(scenarios_to_plot)} plots...")
        for scores in tqdm(scenarios_to_plot, desc="Plotting scenarios"):
            filename = scores['filename']
            if filename in scenarios:
                output_file = plot_dir / f"{Path(filename).stem}_score{scores['interactivity_score']:.3f}.png"
                if not plot_scenario(scenarios[filename], scores, str(output_file), min_goal_dist=args.min_goal_dist):
                    continue  # Skip if ego not valid

        print(f"Plots saved to {plot_dir}/")

    # Output CSV
    split_name = split_path.name
    output_path = args.output or f'interactivity_scores_{split_name}.csv'

    with open(output_path, 'w') as f:
        # Write header
        if results:
            headers = list(results[0].keys())
            f.write(','.join(headers) + '\n')

            # Write data
            for row in results:
                values = [str(row.get(h, '')) for h in headers]
                f.write(','.join(values) + '\n')

    print(f"Results saved to {output_path}")

    # Print summary statistics
    if results:
        scores = [r['interactivity_score'] for r in results]
        num_agents_list = [r['num_agents'] for r in results]
        distances = [r['avg_min_distance'] for r in results if r['avg_min_distance'] > 0]

        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"Total scenarios:     {len(results)}")
        print(f"\nInteractivity Score:")
        print(f"  Mean:              {np.mean(scores):.4f}")
        print(f"  Std:               {np.std(scores):.4f}")
        print(f"  Min:               {np.min(scores):.4f}")
        print(f"  Max:               {np.max(scores):.4f}")
        print(f"\nNumber of Agents:")
        print(f"  Mean:              {np.mean(num_agents_list):.1f}")
        print(f"  Min:               {np.min(num_agents_list)}")
        print(f"  Max:               {np.max(num_agents_list)}")
        if distances:
            print(f"\nAvg Min Distance:")
            print(f"  Mean:              {np.mean(distances):.2f}m")
            print(f"  Min:               {np.min(distances):.2f}m")
            print(f"  Max:               {np.max(distances):.2f}m")

        agents_in_radius = [r['avg_agents_in_radius'] for r in results]
        print(f"\nAvg Agents in 10m Radius (filtered):")
        print(f"  Mean:              {np.mean(agents_in_radius):.2f}")
        print(f"  Min:               {np.min(agents_in_radius):.2f}")
        print(f"  Max:               {np.max(agents_in_radius):.2f}")

        ego_goal_dists = [r['ego_goal_distance'] for r in results]
        print(f"\nEgo Goal Distance:")
        print(f"  Mean:              {np.mean(ego_goal_dists):.1f}m")
        print(f"  Min:               {np.min(ego_goal_dists):.1f}m")
        print(f"  Max:               {np.max(ego_goal_dists):.1f}m")
        print("=" * 60)

        # Print top 10 most interactive scenarios
        print("\nTop 10 Most Interactive Scenarios:")
        print("-" * 110)
        print(f"{'Filename':<40} {'Score':>8} {'Vis@t0':>8} {'In 10m':>8} {'EgoGoal':>8} {'Accel':>10}")
        print("-" * 110)
        for r in results[:10]:
            print(f"{r['filename']:<40} {r['interactivity_score']:>8.4f} {r['agents_valid_t0']:>8} {r['avg_agents_in_radius']:>8.2f} {r['ego_goal_distance']:>7.0f}m {r['avg_accel_change']:>10.2f}")


if __name__ == '__main__':
    main()
