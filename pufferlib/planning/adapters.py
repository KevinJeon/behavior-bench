# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Adapter wrappers for cross-dynamics planner interoperability.

When the env runs in classic dynamics with `emit_jerk_ego_obs=True` (so
DriveConditionedPaper can be mixed in), the ego obs has the 10-dim jerk
layout + 9 creward features. Classic-trained neural planners (PPO, SMART,
WorldModel) expect the legacy 7-dim ego layout with no creward appendix.

`ClassicObsView` slices the jerk-format obs back to the classic layout
before forwarding to the wrapped planner, and forwards all other
`BasePlanner` attributes/methods unchanged.

Layout mapping (per agent):
    jerk ego [0..5]  -> classic ego [0..5]   (goal_x, goal_y, speed, width,
                                              length, collision_state)
    jerk ego [9]     -> classic ego [6]      (respawn_flag)
    jerk ego [6..8]  -> dropped              (steering_angle, a_long, a_lat)
    partner + road   -> unchanged            (same sizes in both layouts)
    creward [9 tail] -> dropped
"""

from __future__ import annotations

import numpy as np

from .base import BasePlanner


_JERK_EGO_DIM = 10
_CLASSIC_EGO_DIM = 7
_CREWARD_DIM = 9


class ClassicObsView(BasePlanner):
    """Wrap a classic-trained planner so it sees classic-layout obs when the
    env is emitting the jerk-layout (mixed-env case)."""

    def __init__(self, inner: BasePlanner, env):
        super().__init__(
            horizon=getattr(inner, "horizon", 1),
            action_dim=getattr(inner, "action_dim", 2),
            action_lb=getattr(inner, "action_lb", np.array([-1.0, -1.0], dtype=np.float32)),
            action_ub=getattr(inner, "action_ub", np.array([1.0, 1.0], dtype=np.float32)),
        )
        self.inner = inner
        self.env = env
        # Cached slice layout — derived from env dims.
        self._partner_dim = env.max_partner_objects * env.partner_features
        self._road_dim = env.max_road_objects * env.road_features
        # Whether the inner planner expects creward-less obs (True for all
        # classic-trained neural planners).

    # -- main dispatch ------------------------------------------------------

    def plan(self, current_step: int = 0, obs=None, extract_trajectories: bool = False, **kwargs):
        if obs is None:
            return self.inner.plan(current_step=current_step, obs=obs,
                                   extract_trajectories=extract_trajectories, **kwargs)
        classic_obs = self._slice_to_classic(obs)
        return self.inner.plan(current_step=current_step, obs=classic_obs,
                               extract_trajectories=extract_trajectories, **kwargs)

    def _slice_to_classic(self, obs: np.ndarray) -> np.ndarray:
        """Convert jerk-format obs (10-ego + partner + road + 9-creward) to
        classic-format (7-ego + partner + road)."""
        single = obs.ndim == 1
        obs2 = np.atleast_2d(obs)
        # Only slice if the obs has the jerk-ego layout. Detect by size:
        expected_jerk = _JERK_EGO_DIM + self._partner_dim + self._road_dim
        if obs2.shape[1] < expected_jerk:
            # Already classic (no wrapping needed). Pass-through.
            return obs if single else obs2
        # ego [0..5] kept, [6..8] dropped, [9] mapped to classic [6].
        ego = np.concatenate([obs2[:, 0:6], obs2[:, 9:10]], axis=1)
        partner = obs2[:, _JERK_EGO_DIM:_JERK_EGO_DIM + self._partner_dim]
        road = obs2[:, _JERK_EGO_DIM + self._partner_dim:
                    _JERK_EGO_DIM + self._partner_dim + self._road_dim]
        classic = np.concatenate([ego, partner, road], axis=1).astype(obs.dtype, copy=False)
        return classic[0] if single else classic

    # -- pass-through -------------------------------------------------------

    def plot(self, ax, state, axis_limits=None):
        return self.inner.plot(ax, state, axis_limits=axis_limits)

    def reset(self):
        return self.inner.reset()

    @property
    def population_size(self):
        return getattr(self.inner, "population_size", 1)

    @property
    def supports_trajectory_proposals(self):
        return getattr(self.inner, "supports_trajectory_proposals", False)

    # Forward everything else (last_logits, LSTM save/restore, etc.)
    def __getattr__(self, name):
        # Called only when the attribute isn't found on self — delegate to inner.
        inner = self.__dict__.get("inner", None)
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)
