# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""PDM (Plan-based Decision Making) planner implementation.

Evaluates a fixed set of IDM proposals (velocity × lateral offset combinations)
in parallel batch environments, picks the best by reward, and applies the
winning IDM parameters to the ego agent for the next step.

The ego runs in IDM mode (like the proposals), so there is no inverse dynamics
mismatch — the real environment uses the same IDM controller as the evaluation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .base import BasePlanner
from pufferlib.ocean.drive.batch_env import DriveBatch
from pufferlib.ocean.drive import binding

log = logging.getLogger("planning")

MOVEMENT_DYNAMICS = 0
MOVEMENT_IDM = 1


@dataclass
class PDMConfig:
    """Configuration for PDM planner."""

    horizon: int = 40
    episode_length: int = 91
    target_speed_kmh: float = 80.0  # Speed limit in km/h (penalty if exceeded)
    velocity_fractions: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    lateral_offsets: Tuple[float, ...] = (-1.0, 0.0, 1.0)
    max_velocity_kmh: float = 100.0  # Maximum velocity for IDM proposals (legacy, use max_velocity)
    max_velocity: float = 0.0  # Maximum velocity in m/s (0 = use max_velocity_kmh / 3.6)
    idm_min_gap: float = 1.0        # IDM: minimum gap to lead vehicle [m]
    idm_headway_time: float = 1.5   # IDM: desired time headway [s]
    idm_accel_max: float = 1.0      # IDM: maximum acceleration [m/s²]
    idm_decel_max: float = 2.0      # IDM: maximum deceleration [m/s²]
    goal_reduction_epsilon: float = 1.0  # Min goal-distance reduction (m) for hybrid pdm_score viability check
    # Planner for other agents during proposal evaluation:
    #   "same"               = use the real other_planner (perfect prediction)
    #   "idm"                = IDM (C-level, no actions needed)
    #   "constant_velocity"  = constant velocity projection
    proposal_other_planner: str = "constant_velocity"


class PDMPlanner(BasePlanner):
    """
    Plan-based Decision Making planner.

    Evaluates a fixed set of IDM proposals (velocity × lateral offset combinations)
    in parallel batch environments, picks the best by reward, and applies the
    winning IDM parameters on the ego agent.

    The ego uses MOVEMENT_IDM in the real environment — the returned action is
    a neutral [0,0] (ignored by IDM). The real effect comes from setting
    idm_target_velocity and idm_lateral_offset on the ego entity.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: PDMConfig,
        other_planner=None,
    ):
        super().__init__(
            horizon=config.horizon,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.num_agents = env.num_agents

        # Build the proposal-internal other planner
        if config.proposal_other_planner == "same":
            self._proposal_other_planner = other_planner
        elif config.proposal_other_planner == "idm":
            self._proposal_other_planner = None  # IDM is handled at C level
            # Set other agents to IDM mode in batch envs (done after batch creation below)
            self._proposal_other_is_idm = True
        elif config.proposal_other_planner == "constant_velocity":
            self._proposal_other_planner = None
        else:
            raise ValueError(
                f"Unknown proposal_other_planner: {config.proposal_other_planner}. "
                f"Use 'same', 'idm', or 'constant_velocity'."
            )
        if not hasattr(self, '_proposal_other_is_idm'):
            self._proposal_other_is_idm = False

        # Build proposal grid: velocity × lateral offset
        max_vel_ms = config.max_velocity if config.max_velocity > 0 else config.max_velocity_kmh / 3.6
        self._velocities = []
        self._offsets = []
        for frac in config.velocity_fractions:
            for off in config.lateral_offsets:
                self._velocities.append(frac * max_vel_ms)
                self._offsets.append(off)
        self._num_proposals = len(self._velocities)

        # Precompute numpy arrays for binding calls
        self._vel_array = np.array(self._velocities, dtype=np.float32)
        self._off_array = np.array(self._offsets, dtype=np.float32)

        # Create batch environment for parallel proposal evaluation
        self.batch_env = DriveBatch.from_env(env, self._num_proposals, ego_agent_idx=agent_idx)

        # Internal state for visualization
        self._last_trajectory_proposals = None
        self._last_trajectory_costs = None
        self._last_best_idx = None
        self._current_step = 0

        # Viability stats (set by _evaluate_proposals; consumed by hybrid pdm_score).
        # Split by failure cause so the hybrid can distinguish "stuck" from "unsafe":
        #   collision_unsafe → all proposals collide/offroad → PPO would also be risky
        #   goal_stuck       → safe proposals exist but none reduces goal_dist → PPO may slip
        self.last_num_viable_proposals = self._num_proposals
        self.last_num_collision_unsafe = 0
        self.last_num_goal_stuck = 0
        self.last_best_score = 0.0

        # Neutral action (IDM ignores actions)
        self.neutral_action = np.array([0.0, 0.0], dtype=np.float32)

        # Set ego to IDM mode in the real environment
        binding.vec_set_movement_mode(
            self.env.c_envs, [self.agent_idx], MOVEMENT_IDM
        )

        # Set IDM params on real env and batch envs
        binding.vec_set_idm_params(
            self.env.c_envs, config.idm_min_gap, config.idm_headway_time,
            config.idm_accel_max, config.idm_decel_max
        )
        binding.vec_set_idm_params(
            self.batch_env.c_envs, config.idm_min_gap, config.idm_headway_time,
            config.idm_accel_max, config.idm_decel_max
        )

    @property
    def population_size(self) -> int:
        return self._num_proposals

    @property
    def supports_trajectory_proposals(self) -> bool:
        return True

    def plan(
        self,
        current_step: int = 0,
        obs: Optional[np.ndarray] = None,
        extract_trajectories: bool = False,
        apply_result: bool = True,
    ) -> np.ndarray:
        """
        Plan next action by evaluating IDM proposals.

        Args:
            apply_result: If True, apply best IDM params to the real env.
                If False, only store them internally (use apply_best_proposal()
                to apply later). Set to False for counterfactual evaluation.

        Returns:
            Neutral action [0, 0] — the real control comes from IDM with the
            best proposal's velocity/offset parameters applied to the ego.
        """
        self._current_step = current_step

        # 1. Create snapshot
        snapshot = self.env.create_snapshot()
        snapshot_handle = snapshot[0]

        # 2. Evaluate proposals
        self._evaluate_proposals(
            snapshot_handle, extract_trajectories=extract_trajectories,
            apply_result=apply_result,
        )

        # 3. Cleanup
        self.env.free_snapshot(snapshot)

        # Return neutral action (IDM handles movement)
        return self.neutral_action.copy()

    def apply_best_proposal(self):
        """Apply the last-computed best IDM params to the real env."""
        binding.vec_set_idm_proposals(
            self.env.c_envs, self.agent_idx,
            self.last_best_vel, self.last_best_off,
        )

    def _evaluate_proposals(
        self, snapshot_handle, extract_trajectories: bool = False,
        apply_result: bool = True,
    ):
        """
        Evaluate all IDM proposals and optionally apply the best one's parameters.

        Steps:
        1. Restore all batch envs to snapshot
        2. Set IDM mode with per-proposal velocity/offset
        3. Step H times for scoring
        4. Score proposals, pick best
        5. Apply best proposal's velocity/offset to ego in real env
        """
        N = self._num_proposals
        H = self.horizon

        # Effective horizon (episode boundary)
        effective_H = max(0, min(H, self.config.episode_length - self._current_step))

        # Restore all batch envs to snapshot
        self.batch_env.restore_snapshot_broadcast(snapshot_handle)

        # Set IDM mode with per-proposal params on ego
        binding.vec_set_idm_proposals(
            self.batch_env.c_envs, self.agent_idx, self._vel_array, self._off_array
        )

        # If proposal_other_planner is "idm", set other agents to IDM in batch envs
        other_indices = [i for i in range(self.num_agents) if i != self.agent_idx]
        if self._proposal_other_is_idm and other_indices:
            binding.vec_set_movement_mode(
                self.batch_env.c_envs, other_indices, MOVEMENT_IDM
            )
        # If proposal_other_planner is "constant_velocity", force DYNAMICS mode
        # (agents may be in IDM mode from main env's other_planner)
        elif self._proposal_other_planner is None and other_indices:
            binding.vec_set_movement_mode(
                self.batch_env.c_envs, other_indices, MOVEMENT_DYNAMICS
            )

        # Accumulators
        total_collision_rewards = np.zeros(N, dtype=np.float32)
        total_offroad_rewards = np.zeros(N, dtype=np.float32)
        total_jerk_rewards = np.zeros(N, dtype=np.float32)
        total_speed_rewards = np.zeros(N, dtype=np.float32)
        total_lane_distance = np.zeros(N, dtype=np.float32)
        final_goal_dists = np.zeros(N, dtype=np.float32)
        done_mask = np.zeros(N, dtype=np.bool_)

        # Initial goal distance per proposal env (before any rollout step) — used
        # for the hybrid pdm_score viability check.
        _, _, _, init_goal_dist, _, _, _ = self.batch_env.get_decomposed_rewards()
        initial_goal_dists = init_goal_dist[self.batch_env.ego_indices].copy()

        target_speed_normalized = (self.config.target_speed_kmh / 3.6) / 100.0

        # Action buffer: ego uses neutral (IDM ignores it), other agents need real actions
        all_actions = np.zeros((N, self.num_agents, 2), dtype=np.float32)

        # Trajectory tracking
        trajectories = None
        if extract_trajectories:
            trajectories = np.zeros((N, H + 1, 2), dtype=np.float32)
            x, y = self.batch_env.get_ego_positions()
            trajectories[:, 0, 0] = x
            trajectories[:, 0, 1] = y

        obs = self.batch_env.observations

        for t in range(effective_H):
            # Other agent actions
            if self.num_agents > 1 and not self._proposal_other_is_idm:
                if self._proposal_other_planner is not None:
                    first_other = other_indices[0] if other_indices else 1
                    other_obs = obs[first_other::self.num_agents]
                    other_action = self._proposal_other_planner.plan(obs=other_obs)
                else:
                    other_action = self.neutral_action

                for i in range(self.num_agents):
                    if i != self.agent_idx:
                        all_actions[:, i, :] = other_action

            # Ego gets neutral action (IDM ignores it)
            all_actions[:, self.agent_idx, :] = 0.0
            flat_actions = all_actions.reshape(-1, 2)

            # Step
            obs, rewards, terms, truncs = self.batch_env.step(flat_actions)

            # Track done state
            ego_terms = terms[self.batch_env.ego_indices]
            ego_truncs = truncs[self.batch_env.ego_indices]
            previously_done = done_mask.copy()
            done_mask |= ego_terms | ego_truncs

            # Decomposed rewards
            coll_rew, offr_rew, goal_rew, goal_dist, jerk_rew, lane_dist, _ = (
                self.batch_env.get_decomposed_rewards()
            )
            ego_coll = coll_rew[self.batch_env.ego_indices].copy()
            ego_offr = offr_rew[self.batch_env.ego_indices].copy()
            ego_dist = goal_dist[self.batch_env.ego_indices].copy()
            ego_jerk = jerk_rew[self.batch_env.ego_indices].copy()
            ego_lane_dist = lane_dist[self.batch_env.ego_indices].copy()

            ego_jerk[previously_done] = 0
            ego_lane_dist[previously_done] = 0

            total_collision_rewards += ego_coll
            total_offroad_rewards += ego_offr
            total_jerk_rewards += ego_jerk
            total_lane_distance += ego_lane_dist

            # Speed penalty
            ego_obs = obs[self.batch_env.ego_indices]
            ego_speed_normalized = ego_obs[:, 2]
            ego_speed_rew = np.where(
                ego_speed_normalized > target_speed_normalized, -1.0, 0.0
            )
            ego_speed_rew[previously_done] = 0
            total_speed_rewards += ego_speed_rew

            # Goal distance
            was_active = ~previously_done
            final_goal_dists = np.where(was_active, ego_dist, final_goal_dists)

            # Trajectory positions
            if extract_trajectories:
                x, y = self.batch_env.get_ego_positions()
                for i in range(N):
                    if done_mask[i]:
                        trajectories[i, t + 1, :] = trajectories[i, t, :]
                    else:
                        trajectories[i, t + 1, 0] = x[i]
                        trajectories[i, t + 1, 1] = y[i]

        # Fill remaining trajectory positions
        if extract_trajectories and effective_H < H:
            for t in range(effective_H, H):
                trajectories[:, t + 1, :] = trajectories[:, effective_H, :]

        # === Score proposals ===
        # goal_dist score [0, 2]
        gd_min, gd_max = final_goal_dists.min(), final_goal_dists.max()
        if gd_max > gd_min:
            goal_dist_score = 2.0 * (
                1.0 - (final_goal_dists - gd_min) / (gd_max - gd_min)
            )
        else:
            goal_dist_score = 2.0 * np.ones(N, dtype=np.float32)

        # lane_dist score [0, 1]
        ld_min, ld_max = total_lane_distance.min(), total_lane_distance.max()
        if ld_max > ld_min:
            lane_dist_score = 1.0 - (total_lane_distance - ld_min) / (
                ld_max - ld_min
            )
        else:
            lane_dist_score = np.ones(N, dtype=np.float32)

        # speed score: bool
        speed_score = np.where(total_speed_rewards < 0, 0.0, 1.0).astype(np.float32)

        # jerk score [0, 1]
        jk_min, jk_max = total_jerk_rewards.min(), total_jerk_rewards.max()
        if jk_max > jk_min:
            jerk_score = (total_jerk_rewards - jk_min) / (jk_max - jk_min)
        else:
            jerk_score = np.ones(N, dtype=np.float32)

        # collision/offroad flags
        has_collision = total_collision_rewards < 0
        has_offroad = total_offroad_rewards < 0

        reward = goal_dist_score + lane_dist_score + speed_score + jerk_score
        reward[has_collision | has_offroad] = 0.0

        costs = -reward

        # Pick best proposal
        best_idx = np.argmin(costs)
        self._last_best_idx = best_idx

        # Viability stats for hybrid switch_mode=pdm_score (does NOT affect best_idx).
        goal_not_reduced = final_goal_dists >= (initial_goal_dists - self.config.goal_reduction_epsilon)
        unsafe_mask = has_collision | has_offroad
        viable_mask = ~(unsafe_mask | goal_not_reduced)
        self.last_num_viable_proposals = int(viable_mask.sum())
        # Split-failure counters: lets hybrid distinguish "PDM has no safe option"
        # (collision_unsafe → don't hand the wheel to PPO, it'll crash too) from
        # "PDM has safe but stuck options" (goal_stuck → PPO may find a way through).
        self.last_num_collision_unsafe = int(unsafe_mask.sum())
        self.last_num_goal_stuck = int((~unsafe_mask & goal_not_reduced).sum())
        self.last_best_score = float(reward.max())

        log.debug(
            "PDM best: proposal %d (vel=%.1f m/s, off=%.1f m), reward=%.3f",
            best_idx,
            self._vel_array[best_idx],
            self._off_array[best_idx],
            -costs[best_idx],
        )

        # Store trajectories for visualization
        if extract_trajectories:
            sorted_idx = np.argsort(costs)
            self._last_trajectory_proposals = trajectories[sorted_idx]
            self._last_trajectory_costs = costs[sorted_idx]

        # Store best params for later application
        self.last_best_vel = np.array([self._vel_array[best_idx]], dtype=np.float32)
        self.last_best_off = np.array([self._off_array[best_idx]], dtype=np.float32)

        # Apply best proposal's IDM params to ego in real env (if requested)
        if apply_result:
            binding.vec_set_idm_proposals(
                self.env.c_envs, self.agent_idx,
                self.last_best_vel, self.last_best_off,
            )

    def plot(self, ax, state, axis_limits=None):
        """Plot proposal trajectories on given matplotlib axes."""
        if self._last_trajectory_proposals is None:
            return

        H = self.horizon
        effective_H = max(0, min(H, self.config.episode_length - self._current_step))
        if effective_H < 5:
            return

        all_trajs = self._last_trajectory_proposals
        draw_length = min(effective_H + 1, all_trajs.shape[1])

        # Draw non-best trajectories (lightblue) with endpoints
        end_idx = draw_length - 1
        for i in range(1, len(all_trajs)):
            traj = all_trajs[i]
            ax.plot(
                traj[:draw_length, 0],
                traj[:draw_length, 1],
                color="lightblue",
                linewidth=1.0,
                alpha=0.5,
                zorder=10,
            )
            ax.scatter(
                traj[end_idx, 0],
                traj[end_idx, 1],
                color="lightblue",
                s=30,
                marker="o",
                alpha=0.6,
                zorder=11,
            )

        # Draw best trajectory (red)
        best_traj = all_trajs[0]
        ax.plot(
            best_traj[:draw_length, 0],
            best_traj[:draw_length, 1],
            color="red",
            linewidth=2.5,
            alpha=0.9,
            zorder=15,
        )
        ax.scatter(
            best_traj[end_idx, 0],
            best_traj[end_idx, 1],
            color="red",
            s=80,
            marker="*",
            zorder=16,
        )

    def reset(self):
        """Reset planner state."""
        self._last_trajectory_proposals = None
        self._last_trajectory_costs = None
        self._last_best_idx = None
        self._current_step = 0
        # Re-apply IDM mode on ego
        binding.vec_set_movement_mode(
            self.env.c_envs, [self.agent_idx], MOVEMENT_IDM
        )


class MultiAgentPDMPlanner(BasePlanner):
    """PDM traffic controller: wraps one PDMPlanner per traffic agent.

    Public shape mirrors IDMPlanner so the evaluator's traffic path works
    unchanged. Each inner PDM owns its own DriveBatch (num_proposals copies),
    so expect N × num_proposals batch envs and proportional compute.
    """

    def __init__(self, env, agent_indices, action_lb, action_ub, config,
                 other_planner=None):
        super().__init__(
            horizon=config.horizon,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_indices = list(agent_indices)
        self._inner = [
            PDMPlanner(env=env, agent_idx=i, action_lb=action_lb,
                       action_ub=action_ub, config=config,
                       other_planner=other_planner)
            for i in self.agent_indices
        ]
        log.info("MultiAgentPDMPlanner: %d traffic agents × %d proposals each",
                 len(self._inner), self._inner[0]._num_proposals if self._inner else 0)

    def plan(self, current_step=0, obs=None, extract_trajectories=False):
        N = len(self._inner)
        out = np.zeros((N, 2), dtype=np.float32)
        obs2d = np.atleast_2d(obs) if obs is not None else None
        for k, p in enumerate(self._inner):
            agent_obs = obs2d[k] if (obs2d is not None and k < obs2d.shape[0]) else None
            out[k] = p.plan(current_step=current_step, obs=agent_obs)
        return out

    def reset(self):
        for p in self._inner:
            p.reset()

    @property
    def population_size(self):
        return 1

    @property
    def supports_trajectory_proposals(self):
        return False

    def plot(self, ax, state, axis_limits=None):
        pass
