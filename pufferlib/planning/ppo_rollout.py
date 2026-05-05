# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""PPO-rollout planner with two strategies:

- ``topk_argmax``: Sample top-K (accel, steer) joint actions from the current
  PPO logits at t=0; each branch then follows PPO-argmax for H-1 more steps
  with constant-velocity other traffic. Score the K final trajectories.

- ``beam_search``: Maintain B beams expanded by C children each step (B·C
  parallel envs total). At every step, score all candidates with the same
  outcome × behavior formula, prune to the top-B, and replicate winners back
  into B·C envs via per-env snapshot/restore. Returns the t=0 action of the
  beam with the highest final score.

Both strategies use the same paper-style score:

    Score = (1 − Coll) · (1 − Off) · S_goal · (w_cmf·S_cmf + w_align·S_align + w_ctr·S_ctr)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .base import BasePlanner
from .policy import PPOPlanner
from pufferlib.ocean.drive.batch_env import DriveBatch
from pufferlib.ocean.drive import binding

log = logging.getLogger("planning")

MOVEMENT_DYNAMICS = 0
MOVEMENT_IDM = 1


@dataclass
class PPORolloutConfig:
    """Configuration for PPORolloutPlanner."""

    # Strategy: "topk_argmax" (one-shot at t=0, argmax thereafter) or
    # "beam_search" (B beams × C children per step, prune by score).
    strategy: str = "beam_search"

    # topk_argmax: number of branches at t=0 (= number of batch envs).
    top_k: int = 8

    # beam_search:
    beam_width: int = 4       # B
    branch_factor: int = 4    # C → B·C envs total per step

    horizon: int = 10
    episode_length: int = 91

    # Behavior-score weights (sum should be 1 by convention; not enforced).
    w_cmf: float = 1.0 / 3.0
    w_align: float = 1.0 / 3.0
    w_ctr: float = 1.0 / 3.0
    # Lane-distance scale (m): values >= scale → S_ctr saturates at 0.
    lane_dist_scale: float = 2.0


class PPORolloutPlanner(BasePlanner):
    """K-branch / beam-search PPO rollout around the current policy."""

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        ppo_planner: PPOPlanner,
        config: PPORolloutConfig,
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
        self.ppo = ppo_planner
        self.num_agents = env.num_agents
        self._other_planner = other_planner

        if ppo_planner.config.policy_action_type != "discrete":
            raise ValueError(
                "PPORolloutPlanner currently only supports discrete PPO policies "
                f"(got policy_action_type={ppo_planner.config.policy_action_type})"
            )
        self._num_accel = len(ppo_planner.config.accel_values)
        self._num_steer = len(ppo_planner.config.steer_values)
        self._num_joint = self._num_accel * self._num_steer

        if config.strategy == "topk_argmax":
            self._n_envs = min(config.top_k, self._num_joint)
            self._B = self._n_envs
            self._C = 1
        elif config.strategy == "beam_search":
            self._B = max(1, min(config.beam_width, self._num_joint))
            self._C = max(1, min(config.branch_factor, self._num_joint))
            self._n_envs = self._B * self._C
        else:
            raise ValueError(
                f"Unknown strategy {config.strategy!r}; "
                "use 'topk_argmax' or 'beam_search'"
            )

        # Cloned batch env (one per branch; ego runs in DYNAMICS)
        self.batch_env = DriveBatch.from_env(
            env, self._n_envs, ego_agent_idx=agent_idx
        )

        self._other_indices = [
            i for i in range(self.num_agents) if i != agent_idx
        ]

        # Pre-allocated action buffer (N, num_agents, 2)
        self._all_actions = np.zeros(
            (self._n_envs, self.num_agents, 2), dtype=np.float32
        )
        self._neutral_action = np.zeros(2, dtype=np.float32)

        # Per-call diagnostics
        self.last_best_idx = 0
        self.last_best_score = 0.0
        self.last_scores = None
        self._last_trajectory_proposals = None
        self._last_trajectory_costs = None
        self._current_step = 0

    @property
    def population_size(self) -> int:
        return self._n_envs

    @property
    def supports_trajectory_proposals(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _topk_actions_from_logits(
        self, logits, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (actions (B, k, 2), flat_indices (B, k)) from PPO logits.

        Logits can be a single tensor (B, n_joint) or a (accel, steer) tuple.
        Action values are normalized to [-1, 1] via the PPO LUT (matches
        PPOPlanner.plan() output convention).
        """
        if isinstance(logits, (list, tuple)):
            if len(logits) == 1:
                joint_logp = torch.log_softmax(
                    torch.nan_to_num(logits[0], neginf=-1e9), dim=-1
                )
            else:
                accel_logp = torch.log_softmax(
                    torch.nan_to_num(logits[0], neginf=-1e9), dim=-1
                )  # (B, n_accel)
                steer_logp = torch.log_softmax(
                    torch.nan_to_num(logits[1], neginf=-1e9), dim=-1
                )  # (B, n_steer)
                # Joint log-prob under independence: outer-sum then flatten
                joint_logp = (
                    accel_logp[..., :, None] + steer_logp[..., None, :]
                ).reshape(accel_logp.shape[0], -1)  # (B, n_joint)
        else:
            joint_logp = torch.log_softmax(
                torch.nan_to_num(logits, neginf=-1e9), dim=-1
            )

        # joint_logp: (B, n_joint) — pick top-k per row.
        k = min(k, joint_logp.shape[-1])
        topk = torch.topk(joint_logp, k=k, dim=-1)
        flat_idx = topk.indices  # (B, k)
        accel_idx = flat_idx // self._num_steer
        steer_idx = flat_idx % self._num_steer

        actions = torch.stack(
            [self.ppo._accel_lut[accel_idx], self.ppo._steer_lut[steer_idx]],
            dim=-1,
        )  # (B, k, 2)
        return actions, flat_idx

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        init_gd: np.ndarray, final_gd: np.ndarray,
        sum_jerk: np.ndarray, sum_lane_dist: np.ndarray,
        sum_lane_align: np.ndarray, step_counts: np.ndarray,
        has_coll: np.ndarray, has_off: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Score every branch with outcome × behavior. Returns (scores, components)."""
        eps = 1e-3

        # S_goal: progress relative to start, clipped to [0,1]
        denom = np.maximum(init_gd, eps)
        s_goal = np.clip((init_gd - final_gd) / denom, 0.0, 1.0).astype(np.float32)
        s_goal = np.where(init_gd < eps, np.float32(1.0), s_goal)

        active_steps = np.maximum(step_counts, 1.0)
        mean_jerk = sum_jerk / active_steps
        mean_lane_dist = sum_lane_dist / active_steps
        mean_lane_align = sum_lane_align / active_steps  # |theta_f| ∈ [0, π]

        # S_cmf: less jerk → higher score, min/max-normalized across branches
        jk_min, jk_max = mean_jerk.min(), mean_jerk.max()
        if jk_max > jk_min + 1e-9:
            s_cmf = (1.0 - (mean_jerk - jk_min) / (jk_max - jk_min)).astype(np.float32)
        else:
            s_cmf = np.ones_like(mean_jerk, dtype=np.float32)

        s_align = np.clip(1.0 - mean_lane_align / float(np.pi), 0.0, 1.0).astype(np.float32)
        s_ctr = np.clip(
            1.0 - mean_lane_dist / max(self.config.lane_dist_scale, 1e-3),
            0.0, 1.0,
        ).astype(np.float32)

        outcome = (
            (~has_coll).astype(np.float32)
            * (~has_off).astype(np.float32)
            * s_goal
        )
        behavior = (
            self.config.w_cmf * s_cmf
            + self.config.w_align * s_align
            + self.config.w_ctr * s_ctr
        ).astype(np.float32)
        scores = outcome * behavior
        return scores, dict(
            s_goal=s_goal, s_cmf=s_cmf, s_align=s_align, s_ctr=s_ctr,
            outcome=outcome, behavior=behavior,
        )

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def plan(
        self,
        current_step: int = 0,
        obs: Optional[np.ndarray] = None,
        extract_trajectories: bool = False,
    ) -> np.ndarray:
        self._current_step = current_step
        if self.config.strategy == "topk_argmax":
            return self._plan_topk_argmax(current_step, extract_trajectories)
        return self._plan_beam_search(current_step, extract_trajectories)

    # ------------------------------------------------------------------
    # Strategy 1: top-K at t=0, argmax thereafter
    # ------------------------------------------------------------------

    def _plan_topk_argmax(
        self, current_step: int, extract_trajectories: bool
    ) -> np.ndarray:
        K = self._n_envs
        H = self.horizon
        effective_H = max(0, min(H, self.config.episode_length - current_step))

        # Top-K joint actions at t=0 from current PPO logits (B=1)
        actions_t0_t, _ = self._topk_actions_from_logits(self.ppo.last_logits, k=K)
        actions_t0 = actions_t0_t[0].cpu().numpy().astype(np.float32)  # (K, 2)

        snapshot = self.env.create_snapshot()
        try:
            self.batch_env.restore_snapshot_broadcast(snapshot[0])
            binding.vec_set_movement_mode(
                self.batch_env.c_envs, [self.agent_idx], MOVEMENT_DYNAMICS
            )
            if self._other_indices:
                binding.vec_set_movement_mode(
                    self.batch_env.c_envs, self._other_indices, MOVEMENT_DYNAMICS
                )

            self.ppo.save_lstm_state()
            try:
                self.ppo.restore_lstm_state_broadcast(K)

                _, _, _, init_goal_dist, _, _, _ = (
                    self.batch_env.get_decomposed_rewards()
                )
                init_gd = init_goal_dist[self.batch_env.ego_indices].copy()

                sum_jerk = np.zeros(K, dtype=np.float32)
                sum_lane_dist = np.zeros(K, dtype=np.float32)
                sum_lane_align = np.zeros(K, dtype=np.float32)
                step_counts = np.zeros(K, dtype=np.float32)
                has_coll = np.zeros(K, dtype=np.bool_)
                has_off = np.zeros(K, dtype=np.bool_)
                final_gd = init_gd.copy()
                done_mask = np.zeros(K, dtype=np.bool_)

                trajectories = None
                if extract_trajectories:
                    trajectories = np.zeros((K, H + 1, 2), dtype=np.float32)
                    x, y = self.batch_env.get_ego_positions()
                    trajectories[:, 0, 0] = x
                    trajectories[:, 0, 1] = y

                for t in range(effective_H):
                    if t == 0:
                        ego_actions = actions_t0
                    else:
                        ego_obs = self.batch_env.observations[
                            self.batch_env.ego_indices
                        ]
                        ego_actions = self.ppo.plan(
                            current_step=current_step + t, obs=ego_obs
                        )
                        ego_actions = np.atleast_2d(ego_actions).astype(
                            np.float32, copy=False
                        )

                    self._all_actions[:, self.agent_idx, :] = ego_actions
                    if self._other_indices:
                        for i in self._other_indices:
                            self._all_actions[:, i, :] = self._neutral_action

                    _, _, terms, truncs = self.batch_env.step(
                        self._all_actions.reshape(-1, 2)
                    )
                    ego_terms = terms[self.batch_env.ego_indices]
                    ego_truncs = truncs[self.batch_env.ego_indices]
                    prev_done = done_mask.copy()
                    done_mask = done_mask | ego_terms | ego_truncs
                    active = ~prev_done

                    coll, offr, _, gd, jerk, lane_dist, lane_align = (
                        self.batch_env.get_decomposed_rewards()
                    )
                    has_coll = has_coll | (coll[self.batch_env.ego_indices] < 0)
                    has_off = has_off | (offr[self.batch_env.ego_indices] < 0)
                    sum_jerk += np.where(
                        active, np.abs(jerk[self.batch_env.ego_indices]), 0.0
                    )
                    sum_lane_dist += np.where(
                        active, lane_dist[self.batch_env.ego_indices], 0.0
                    )
                    sum_lane_align += np.where(
                        active, lane_align[self.batch_env.ego_indices], 0.0
                    )
                    step_counts += active.astype(np.float32)
                    final_gd = np.where(
                        active, gd[self.batch_env.ego_indices], final_gd
                    )

                    if extract_trajectories:
                        x, y = self.batch_env.get_ego_positions()
                        for i in range(K):
                            if done_mask[i]:
                                trajectories[i, t + 1, :] = trajectories[i, t, :]
                            else:
                                trajectories[i, t + 1, 0] = x[i]
                                trajectories[i, t + 1, 1] = y[i]

                if extract_trajectories and effective_H < H:
                    for t in range(effective_H, H):
                        trajectories[:, t + 1, :] = trajectories[:, effective_H, :]

            finally:
                self.ppo.restore_lstm_state()
        finally:
            self.env.free_snapshot(snapshot)

        scores, comp = self._compute_score(
            init_gd, final_gd, sum_jerk, sum_lane_dist, sum_lane_align,
            step_counts, has_coll, has_off,
        )

        best_idx = int(np.argmax(scores))
        self.last_best_idx = best_idx
        self.last_best_score = float(scores[best_idx])
        self.last_scores = scores

        if extract_trajectories:
            sorted_idx = np.argsort(-scores)
            self._last_trajectory_proposals = trajectories[sorted_idx]
            self._last_trajectory_costs = -scores[sorted_idx]

        log.debug(
            "PPO-rollout (topk_argmax) best=%d score=%.3f", best_idx, scores[best_idx]
        )
        return actions_t0[best_idx].astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Strategy 2: beam search (B beams × C children per step)
    # ------------------------------------------------------------------

    def _plan_beam_search(
        self, current_step: int, extract_trajectories: bool
    ) -> np.ndarray:
        B, C, N = self._B, self._C, self._n_envs
        H = self.horizon
        effective_H = max(0, min(H, self.config.episode_length - current_step))

        # 1. Top-B joint actions at t=0 from current PPO logits (real env, B_logits=1)
        top_b_t, _ = self._topk_actions_from_logits(self.ppo.last_logits, k=B)
        top_b_actions = top_b_t[0].cpu().numpy().astype(np.float32)  # (B, 2)
        # Replicate each top-B action C times → t=0 actions for N envs
        actions_t = np.repeat(top_b_actions, C, axis=0)  # (N, 2)
        # t=0 action tag: which top-B index each env descends from
        t0_action_tag = np.repeat(np.arange(B), C).astype(np.int32)

        snapshot = self.env.create_snapshot()
        snapshot_handle = snapshot[0]
        try:
            self.batch_env.restore_snapshot_broadcast(snapshot_handle)
            binding.vec_set_movement_mode(
                self.batch_env.c_envs, [self.agent_idx], MOVEMENT_DYNAMICS
            )
            if self._other_indices:
                binding.vec_set_movement_mode(
                    self.batch_env.c_envs, self._other_indices, MOVEMENT_DYNAMICS
                )

            self.ppo.save_lstm_state()
            try:
                self.ppo.restore_lstm_state_broadcast(N)

                _, _, _, init_goal_dist, _, _, _ = (
                    self.batch_env.get_decomposed_rewards()
                )
                init_gd = init_goal_dist[self.batch_env.ego_indices].copy()

                sum_jerk = np.zeros(N, dtype=np.float32)
                sum_lane_dist = np.zeros(N, dtype=np.float32)
                sum_lane_align = np.zeros(N, dtype=np.float32)
                step_counts = np.zeros(N, dtype=np.float32)
                has_coll = np.zeros(N, dtype=np.bool_)
                has_off = np.zeros(N, dtype=np.bool_)
                final_gd = init_gd.copy()
                done_mask = np.zeros(N, dtype=np.bool_)

                trajectories = None
                if extract_trajectories:
                    trajectories = np.zeros((N, H + 1, 2), dtype=np.float32)
                    x, y = self.batch_env.get_ego_positions()
                    trajectories[:, 0, 0] = x
                    trajectories[:, 0, 1] = y

                for t in range(effective_H):
                    # Apply current actions to the N envs and step
                    self._all_actions[:, self.agent_idx, :] = actions_t
                    if self._other_indices:
                        for i in self._other_indices:
                            self._all_actions[:, i, :] = self._neutral_action

                    _, _, terms, truncs = self.batch_env.step(
                        self._all_actions.reshape(-1, 2)
                    )
                    ego_terms = terms[self.batch_env.ego_indices]
                    ego_truncs = truncs[self.batch_env.ego_indices]
                    prev_done = done_mask.copy()
                    done_mask = done_mask | ego_terms | ego_truncs
                    active = ~prev_done

                    coll, offr, _, gd, jerk, lane_dist, lane_align = (
                        self.batch_env.get_decomposed_rewards()
                    )
                    has_coll = has_coll | (coll[self.batch_env.ego_indices] < 0)
                    has_off = has_off | (offr[self.batch_env.ego_indices] < 0)
                    sum_jerk += np.where(
                        active, np.abs(jerk[self.batch_env.ego_indices]), 0.0
                    )
                    sum_lane_dist += np.where(
                        active, lane_dist[self.batch_env.ego_indices], 0.0
                    )
                    sum_lane_align += np.where(
                        active, lane_align[self.batch_env.ego_indices], 0.0
                    )
                    step_counts += active.astype(np.float32)
                    final_gd = np.where(
                        active, gd[self.batch_env.ego_indices], final_gd
                    )

                    if extract_trajectories:
                        x, y = self.batch_env.get_ego_positions()
                        for i in range(N):
                            if done_mask[i]:
                                trajectories[i, t + 1, :] = trajectories[i, t, :]
                            else:
                                trajectories[i, t + 1, 0] = x[i]
                                trajectories[i, t + 1, 1] = y[i]

                    # Score current N branches (cumulative since t=0)
                    scores, _ = self._compute_score(
                        init_gd, final_gd, sum_jerk, sum_lane_dist,
                        sum_lane_align, step_counts, has_coll, has_off,
                    )

                    # Last step: keep accumulators for final scoring; no expansion
                    if t == effective_H - 1:
                        break

                    # Pick top-B winners
                    winner_idx = np.argsort(-scores)[:B]

                    # Forward PPO on winner obs to obtain logits at the next state
                    winner_obs = self.batch_env.observations[
                        self.batch_env.ego_indices
                    ][winner_idx]
                    # Gather the LSTM state at the winners (current state is (N, hidden))
                    winner_lstm_h = self.ppo.lstm_h[winner_idx] if self.ppo.lstm_h is not None else None
                    winner_lstm_c = self.ppo.lstm_c[winner_idx] if self.ppo.lstm_c is not None else None
                    self.ppo.lstm_h = winner_lstm_h
                    self.ppo.lstm_c = winner_lstm_c

                    # ppo.plan returns argmax actions (we ignore them), but stores
                    # last_logits (B, ...) — exactly what we need for top-C per beam.
                    _ = self.ppo.plan(
                        current_step=current_step + t + 1, obs=winner_obs
                    )
                    next_actions_t, _ = self._topk_actions_from_logits(
                        self.ppo.last_logits, k=C
                    )  # (B, C, 2)
                    actions_t = next_actions_t.reshape(B * C, 2).cpu().numpy().astype(np.float32)

                    # Replicate winners' env state into N envs via per-env snapshots.
                    # Layout: env[b*C + c] gets winner_idx[b]'s state, for c in [0, C).
                    all_snaps = binding.vec_create_snapshot(self.batch_env.c_envs)
                    new_snaps = []
                    for b in range(B):
                        snap_b = all_snaps[int(winner_idx[b])]
                        for _c in range(C):
                            new_snaps.append(snap_b)
                    binding.vec_restore_snapshot(self.batch_env.c_envs, new_snaps)
                    binding.vec_free_snapshot(all_snaps)

                    # Replicate accumulators: gather on winners then repeat C times.
                    sum_jerk = np.repeat(sum_jerk[winner_idx], C)
                    sum_lane_dist = np.repeat(sum_lane_dist[winner_idx], C)
                    sum_lane_align = np.repeat(sum_lane_align[winner_idx], C)
                    step_counts = np.repeat(step_counts[winner_idx], C)
                    has_coll = np.repeat(has_coll[winner_idx], C)
                    has_off = np.repeat(has_off[winner_idx], C)
                    final_gd = np.repeat(final_gd[winner_idx], C)
                    init_gd = np.repeat(init_gd[winner_idx], C)
                    done_mask = np.repeat(done_mask[winner_idx], C)
                    t0_action_tag = np.repeat(t0_action_tag[winner_idx], C)

                    # Replicate LSTM state (winner_lstm_* shape (B, hidden) post-forward
                    # → (N, hidden) by repeat_interleave, matching env layout).
                    if self.ppo.lstm_h is not None:
                        self.ppo.lstm_h = self.ppo.lstm_h.repeat_interleave(C, dim=0)
                        self.ppo.lstm_c = self.ppo.lstm_c.repeat_interleave(C, dim=0)

                    # Also replicate trajectory rows (so plot stays consistent)
                    if extract_trajectories:
                        trajectories = np.repeat(
                            trajectories[winner_idx], C, axis=0
                        )

                if extract_trajectories and effective_H < H:
                    for tt in range(effective_H, H):
                        trajectories[:, tt + 1, :] = trajectories[:, effective_H, :]

            finally:
                self.ppo.restore_lstm_state()
        finally:
            self.env.free_snapshot(snapshot)

        # Final score across the surviving N branches
        scores, comp = self._compute_score(
            init_gd, final_gd, sum_jerk, sum_lane_dist, sum_lane_align,
            step_counts, has_coll, has_off,
        )
        best = int(np.argmax(scores))
        self.last_best_idx = best
        self.last_best_score = float(scores[best])
        self.last_scores = scores

        if extract_trajectories:
            sorted_idx = np.argsort(-scores)
            self._last_trajectory_proposals = trajectories[sorted_idx]
            self._last_trajectory_costs = -scores[sorted_idx]

        # Return the t=0 action of the beam with the best cumulative score.
        best_t0_tag = int(t0_action_tag[best])
        log.debug(
            "PPO-rollout (beam_search B=%d C=%d) best=%d t0_tag=%d score=%.3f",
            B, C, best, best_t0_tag, scores[best],
        )
        return top_b_actions[best_t0_tag].astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(self, ax, state, axis_limits=None):
        if self._last_trajectory_proposals is None:
            return

        H = self.horizon
        effective_H = max(0, min(H, self.config.episode_length - self._current_step))
        if effective_H < 5:
            return

        all_trajs = self._last_trajectory_proposals
        draw_length = min(effective_H + 1, all_trajs.shape[1])
        end_idx = draw_length - 1

        for i in range(1, len(all_trajs)):
            traj = all_trajs[i]
            ax.plot(
                traj[:draw_length, 0], traj[:draw_length, 1],
                color="lightgreen", linewidth=1.0, alpha=0.5, zorder=10,
            )
            ax.scatter(
                traj[end_idx, 0], traj[end_idx, 1],
                color="lightgreen", s=30, marker="o", alpha=0.6, zorder=11,
            )

        best_traj = all_trajs[0]
        ax.plot(
            best_traj[:draw_length, 0], best_traj[:draw_length, 1],
            color="darkgreen", linewidth=2.5, alpha=0.9, zorder=15,
        )
        ax.scatter(
            best_traj[end_idx, 0], best_traj[end_idx, 1],
            color="darkgreen", s=80, marker="*", zorder=16,
        )

    def reset(self):
        self._last_trajectory_proposals = None
        self._last_trajectory_costs = None
        self.last_best_idx = 0
        self.last_best_score = 0.0
        self.last_scores = None
        self._current_step = 0

    def close(self):
        """Release the cloned batch env. Must be called per scenario or the
        per-map allocation accumulates (each map leaks B*C Drive C-envs)."""
        if getattr(self, "batch_env", None) is not None:
            self.batch_env.close()
            self.batch_env = None
