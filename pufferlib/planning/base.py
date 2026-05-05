# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Abstract base class for trajectory planners."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class PlanResult:
    """Internal result from planning (for use within planner implementations)."""

    best_action_sequence: np.ndarray  # Shape: (H, action_dim)
    first_action: np.ndarray  # Shape: (action_dim,)
    warm_start: Optional[np.ndarray]  # For MPC-style planners

    # Optional trajectory proposals (for visualization)
    trajectory_proposals: Optional[np.ndarray] = None  # Shape: (N, H+1, 2) or None
    trajectory_costs: Optional[np.ndarray] = None  # Shape: (N,) or None
    elite_indices: Optional[np.ndarray] = None  # Shape: (K,) or None


class BasePlanner(ABC):
    """
    Abstract base class for trajectory planners.

    Planners are self-contained and manage their own environment interactions.
    Each planner plans for a single agent.
    """

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
    ):
        """
        Initialize the planner.

        Args:
            horizon: Planning horizon (number of timesteps)
            action_dim: Dimension of action space
            action_lb: Lower bounds for actions, shape (action_dim,)
            action_ub: Upper bounds for actions, shape (action_dim,)
        """
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_lb = np.asarray(action_lb, dtype=np.float32)
        self.action_ub = np.asarray(action_ub, dtype=np.float32)

    @abstractmethod
    def plan(self, current_step: int = 0, obs: Optional[np.ndarray] = None, extract_trajectories: bool = False) -> np.ndarray:
        """
        Plan next action for the agent.

        Args:
            current_step: Current episode step (for episode boundary handling)
            obs: Current observation from environment (optional, used by learned planners)
            extract_trajectories: If True, extract trajectory positions for visualization.

        Returns:
            first_action: Action to execute now, shape (action_dim,)
        """
        pass

    @abstractmethod
    def plot(self, ax, state, axis_limits=None):
        """
        Plot trajectory proposals on given matplotlib axes.

        Args:
            ax: Matplotlib axes object
            state: Environment state dict (from env.get_state())
            axis_limits: Optional tuple (xmin, xmax, ymin, ymax)
        """
        pass

    @property
    @abstractmethod
    def population_size(self) -> int:
        """Number of candidate sequences evaluated per iteration."""
        pass

    @property
    def supports_trajectory_proposals(self) -> bool:
        """Whether this planner generates trajectory proposals for visualization."""
        return False

    def reset(self):
        """Reset planner state (e.g., clear warm start history)."""
        pass
