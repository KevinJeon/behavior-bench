# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""IDM (Intelligent Driver Model) planner.

Uses the C-level move_idm function which handles both longitudinal
(car-following) and lateral (lane-following) control directly in
the simulation step, bypassing the normal action buffer.
"""

import numpy as np
from .base import BasePlanner

MOVEMENT_IDM = 1


class IDMPlanner(BasePlanner):
    """
    Planner that uses the Intelligent Driver Model (IDM) for driving.

    Unlike other planners, IDM operates directly at the C level during
    env.step(). This planner sets the agents' movement_mode flag to IDM
    via the binding, then returns dummy actions (which are ignored by
    move_idm in c_step).

    The IDM handles:
    - Longitudinal control: car-following with adaptive cruise behavior
    - Lateral control: lane-following via lookahead steering
    """

    def __init__(
        self,
        env,
        agent_indices: list,
        horizon: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        target_velocity: float = 15.0,
        min_gap: float = 1.0,
        headway_time: float = 1.5,
        accel_max: float = 1.0,
        decel_max: float = 2.0,
    ):
        super().__init__(horizon, len(action_lb), action_lb, action_ub)
        self.env = env
        self.agent_indices = agent_indices
        self.target_velocity = target_velocity
        self.min_gap = min_gap
        self.headway_time = headway_time
        self.accel_max = accel_max
        self.decel_max = decel_max
        self.neutral_action = np.array([0.0, 0.0], dtype=np.float32)

        # Set movement mode to IDM via C binding
        self._set_idm_mode()
        self._set_target_velocity()
        self._set_idm_params()

    def _set_idm_mode(self):
        from pufferlib.ocean.drive import binding
        binding.vec_set_movement_mode(
            self.env.c_envs, self.agent_indices, MOVEMENT_IDM
        )

    def _set_target_velocity(self):
        from pufferlib.ocean.drive import binding
        binding.vec_set_idm_target_velocity(
            self.env.c_envs, self.agent_indices, self.target_velocity
        )

    def _set_idm_params(self):
        from pufferlib.ocean.drive import binding
        binding.vec_set_idm_params(
            self.env.c_envs, self.min_gap, self.headway_time,
            self.accel_max, self.decel_max
        )

    @property
    def population_size(self) -> int:
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        return False

    def plan(self, current_step: int = 0, obs=None, extract_trajectories: bool = False) -> np.ndarray:
        """Return dummy action(s). IDM movement happens in c_step."""
        if obs is not None and obs.ndim >= 2:
            # Batch mode: return (batch_size, 2) dummy actions
            return np.zeros((obs.shape[0], 2), dtype=np.float32)
        return self.neutral_action.copy()

    def plot(self, ax, state, axis_limits=None):
        pass

    def reset(self):
        """Re-set IDM mode (in case env was reset and flags cleared)."""
        self._set_idm_mode()
        self._set_target_velocity()
        self._set_idm_params()
