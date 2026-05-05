# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Configuration dataclasses for evaluation."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


@dataclass
class ActionConfig:
    """Configuration for action space."""

    accel_min: float = -4.0  # m/s²
    accel_max: float = 4.0  # m/s²
    steer_min: float = -1.0
    steer_max: float = 1.0
    # Discrete action values (7 accel × 13 steer = 91 options)
    accel_values: np.ndarray = None
    steer_values: np.ndarray = None

    def __post_init__(self):
        if self.accel_values is None:
            self.accel_values = np.array(
                [-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32
            )
        if self.steer_values is None:
            self.steer_values = np.linspace(
                self.steer_min, self.steer_max, 13, dtype=np.float32
            )

    @property
    def bounds(self) -> tuple:
        """Return (lower_bound, upper_bound) arrays."""
        lb = np.array([self.accel_min, self.steer_min], dtype=np.float32)
        ub = np.array([self.accel_max, self.steer_max], dtype=np.float32)
        return lb, ub

    @property
    def neutral_action(self) -> np.ndarray:
        """Return neutral action (no acceleration, no steering)."""
        return np.array([0.0, 0.0], dtype=np.float32)

    @property
    def neutral_action_index(self) -> int:
        """Return discrete index for neutral action."""
        # accel_idx=3 (0.0 m/s²), steer_idx=6 (0.0)
        return 3 * len(self.steer_values) + 6


@dataclass
class EvaluatorConfig:
    """Configuration for parallel map evaluator."""

    # Map selection
    num_maps: int = 10
    map_pool_size: int = 1000

    # Episode settings
    episode_length: int = 91
    action_type: str = "continuous"  # "continuous" or "discrete"

    # Goal/termination behavior
    # goal_behavior: 0=respawn, 1=generate_new, 2=stop, 3=remove, 5=sample_lane_ahead
    goal_behavior: int = 1
    # For goal_behavior=5: probability of sampling on a parallel lane
    goal_lane_change_prob: float = 0.0
    # Arc-length of the new sampled goal ahead of the agent (m).
    goal_target_distance: float = 20.0
    # termination_mode: 0=terminate at episode_length only, 1=terminate when all agents done
    termination_mode: int = 1
    collision_behavior: int = 2  # 0=ignore, 1=stop, 2=remove
    offroad_behavior: int = 2  # 0=ignore, 1=stop, 2=remove

    # Output
    output_dir: str = "artifacts/eval"

    # Scene visualization (road, agents, trajectories per step → GIF)
    render: bool = False  # Kept for backwards compatibility
    viz: bool = False

    # Planner-specific visualization (PDM proposals, etc.)
    save_iteration_gifs: bool = False  # Kept for backwards compatibility
    planner_viz: bool = False
    iteration_gif_steps: Optional[List[int]] = None  # Only save GIFs for these steps (None = all steps)

    # Dataset split
    split: str = "testing"

    # Uncertainty estimation
    compute_uncertainty: bool = False
    ensemble_weight_paths: List[str] = field(default_factory=list)

    # Action config
    action_config: ActionConfig = field(default_factory=ActionConfig)

    # Drive env dynamics + reward-conditioning (needed by DriveConditionedPaper).
    # Defaults preserve existing behavior (classic dynamics, no conditioning).
    dynamics_model: str = "classic"
    reward_conditioning: bool = False
    creward_deterministic: bool = False
    # When True, the env emits the 10-dim jerk ego obs layout even in classic
    # dynamics. Enables mixing ConditionedPaper with classic-trained planners
    # (PPO/SMART/WorldModel) in a single rollout; those planners get wrapped
    # in ClassicObsView at the registry level to see a 7-dim ego view.
    emit_jerk_ego_obs: bool = False
    # Per-profile creward coefficients. Keys: delta_goal, alpha_collision,
    # alpha_boundary, alpha_comfort, alpha_l_align, alpha_vel_align,
    # alpha_l_center, alpha_center_bias, alpha_reverse.
    creward_ego: Dict[str, float] = field(default_factory=dict)
    # List of traffic creward profiles. Traffic agents are assigned a profile
    # by entity index (cycled round-robin). Empty list -> no override.
    creward_traffic: List[Dict[str, float]] = field(default_factory=list)
    # Optional global env-reward overrides (apply to all agents). Typical
    # keys: reward_velocity, reward_timestep, reward_vehicle_collision, etc.
    env_reward_overrides: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Merge backwards-compatible flags: render → viz, save_iteration_gifs → planner_viz
        if self.render and not self.viz:
            self.viz = True
        if self.save_iteration_gifs and not self.planner_viz:
            self.planner_viz = True
