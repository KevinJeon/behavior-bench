# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Expert traffic controller — agents follow ground-truth trajectories.

Uses the C engine's move_expert() directly by setting MOVEMENT_EXPERT mode
on all non-ego active agents. No P-controller needed.
"""

import numpy as np

from pufferlib.planning.base import BasePlanner
from pufferlib.ocean.drive import binding


MOVEMENT_EXPERT = 2


class ExpertPlanner(BasePlanner):
    """Traffic controller that replays ground-truth trajectories via C engine."""

    def __init__(self, env, agent_idx, action_lb, action_ub):
        super().__init__(env, agent_idx, action_lb, action_ub)
        # Set all non-ego active agents to MOVEMENT_EXPERT
        other_indices = [i for i in range(env.num_agents) if i != agent_idx]
        if other_indices:
            binding.vec_set_movement_mode(env.c_envs, other_indices, MOVEMENT_EXPERT)

    def plan(self, current_step=0, obs=None, extract_trajectories=False):
        # move_expert() in C engine handles positioning — return dummy actions
        if obs is not None:
            n = np.atleast_2d(obs).shape[0]
            return np.zeros((n, 2), dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def plot(self, ax, state, axis_limits=None):
        pass

    def population_size(self):
        return 0
