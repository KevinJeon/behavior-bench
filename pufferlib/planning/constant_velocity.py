# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Constant velocity planner - drives straight."""

import numpy as np
from .base import BasePlanner


class ConstantVelocityPlanner(BasePlanner):
    """
    Simple planner that always returns neutral action (drive straight).

    Useful for non-ego agents that should maintain constant velocity
    without active planning.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        horizon: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
    ):
        """
        Initialize constant velocity planner.

        Args:
            env: Single environment (not used, for interface compatibility)
            agent_idx: Index of agent being planned for
            horizon: Planning horizon (not used)
            action_lb: Action lower bounds
            action_ub: Action upper bounds
        """
        super().__init__(horizon, len(action_lb), action_lb, action_ub)
        self.env = env
        self.agent_idx = agent_idx

        # Neutral action: zero acceleration, zero steering
        self.neutral_action = np.array([0.0, 0.0], dtype=np.float32)

    @property
    def population_size(self) -> int:
        """Number of candidates (always 1 for constant velocity)."""
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        """No trajectory proposals for constant velocity."""
        return False

    def plan(self, current_step: int = 0, obs=None, extract_trajectories: bool = False) -> np.ndarray:
        """
        Plan next action (always neutral).

        Args:
            current_step: Current episode step (ignored)
            obs: Observation (ignored for this planner)
            extract_trajectories: Ignored for this planner

        Returns:
            Neutral action (action_dim,)
        """
        return self.neutral_action.copy()

    def plot(self, ax, state, axis_limits=None):
        """No visualization for constant velocity planner."""
        pass

    def reset(self):
        """No state to reset."""
        pass
