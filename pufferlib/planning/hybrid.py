# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Hybrid PPO+PDM planner with uncertainty-based switching.

Combines a PPO policy with a PDM planner. Uses an ensemble of PPO policies
to estimate epistemic uncertainty (mutual information). When uncertainty
exceeds a threshold, switches from PPO to PDM for safer planning.

The PPO LSTM state is always updated (even when PDM is used) to maintain
temporal coherence.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .base import BasePlanner
from .policy import PPOPlanner, PPOConfig
from .pdm import PDMPlanner, PDMConfig
from .ppo_rollout import PPORolloutPlanner, PPORolloutConfig
from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.batch_env import DriveBatch
from pufferlib.evaluation.uncertainty import PolicyEnsemble, compute_epistemic, compute_aleatoric

log = logging.getLogger("planning")

MOVEMENT_DYNAMICS = 0
MOVEMENT_IDM = 1


@dataclass
class HybridConfig:
    ppo_config: PPOConfig = None
    pdm_config: PDMConfig = None
    ensemble_weights: List[str] = field(default_factory=list)
    epistemic_threshold: float = 0.8
    pdm_min_steps: int = 1  # minimum consecutive PDM steps after a switch (1 = no lock-in)
    force_ppo: bool = False  # always use PPO (ignore uncertainty switching)
    force_pdm: bool = False  # always use PDM (ignore uncertainty switching)
    switch_mode: str = "epistemic"  # "epistemic" | "value_variance" | "pdm_score"
    value_variance_threshold: float = 0.1
    lookahead_steps: int = 0  # 0=disabled, N>0 = propagate PPO N steps ahead to check future uncertainty
    # PPO mode (orthogonal to switch_mode): how to act once PPO is selected.
    #   "single_step" = take a single argmax action from the current PPO forward
    #   "rollout"     = top-K PPO action sampling + short batch-env rollout, scored
    #                   with a paper-style outcome × behavior score (see ppo_rollout.py)
    ppo_mode: str = "single_step"
    ppo_rollout_config: PPORolloutConfig = None


class HybridPPOPDMPlanner(BasePlanner):
    """Hybrid planner that switches between PPO and PDM based on epistemic uncertainty.

    When epistemic uncertainty (mutual information across ensemble) exceeds the
    threshold, ego is switched to IDM mode and PDM plans. Otherwise, ego stays
    in dynamics mode and PPO's action is used.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: HybridConfig,
        other_planner=None,
    ):
        super().__init__(
            horizon=config.pdm_config.horizon,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.threshold = config.epistemic_threshold
        self._using_pdm = False

        # Create PPO planner (always runs for LSTM state update)
        self.ppo = PPOPlanner(env, agent_idx, action_lb, action_ub, config.ppo_config)

        # Create PDM planner — its __init__ sets MOVEMENT_IDM on the real env
        self.pdm = PDMPlanner(
            env, agent_idx, action_lb, action_ub, config.pdm_config, other_planner
        )

        # Optional: PPO-rollout branch evaluator (used when ppo_mode == "rollout")
        self.ppo_mode = config.ppo_mode
        if self.ppo_mode == "rollout":
            ppo_rollout_cfg = config.ppo_rollout_config or PPORolloutConfig()
            self.ppo_rollout = PPORolloutPlanner(
                env, agent_idx, action_lb, action_ub,
                ppo_planner=self.ppo, config=ppo_rollout_cfg,
                other_planner=other_planner,
            )
        else:
            self.ppo_rollout = None

        # Reset to DYNAMICS mode (PPO is the default)
        binding.vec_set_movement_mode(env.c_envs, [agent_idx], MOVEMENT_DYNAMICS)

        # Create ensemble for epistemic uncertainty
        self.ensemble = PolicyEnsemble(
            config.ensemble_weights, env, config.ppo_config,
            ego_agent_idx=agent_idx,
        )

        # Lookahead env for forward uncertainty simulation
        self._lookahead_steps = config.lookahead_steps
        self._other_planner = other_planner
        if self._lookahead_steps > 0:
            self._lookahead_env = DriveBatch.from_env(env, 1)
        else:
            self._lookahead_env = None

        # Tracking
        self.last_epistemic = 0.0
        self._pdm_step_count = 0
        self._total_step_count = 0
        self._pdm_min_steps = config.pdm_min_steps
        self._pdm_remaining = 0  # remaining locked-in PDM steps

        # Per-step history (for diagnostics / visualization)
        self._step_planners: List[str] = []     # "PPO" or "PDM" per step
        self._step_epistemics: List[float] = []  # epistemic uncertainty per step
        self._step_aleatorics: List[float] = []  # aleatoric uncertainty per step
        self._step_value_vars: List[float] = []  # value prediction variance per step
        self._step_lookahead_triggered: List[bool] = []  # whether lookahead triggered PDM
        self._step_num_viable: List[int] = []   # PDM viable proposal count per step
        self._step_best_score: List[float] = []  # PDM best proposal reward per step

        log.info(
            "HybridPPOPDMPlanner: threshold=%.2f, ensemble_size=%d, pdm_min_steps=%d, lookahead_steps=%d",
            self.threshold, len(config.ensemble_weights), self._pdm_min_steps, self._lookahead_steps,
        )

    @property
    def switch_steps(self) -> List[int]:
        """Steps where the active planner changed (PPO↔PDM)."""
        switches = []
        for i in range(1, len(self._step_planners)):
            if self._step_planners[i] != self._step_planners[i - 1]:
                switches.append(i)
        return switches

    @property
    def population_size(self) -> int:
        return self.pdm.population_size

    @property
    def supports_trajectory_proposals(self) -> bool:
        return True

    def plan(
        self,
        current_step: int = 0,
        obs: Optional[np.ndarray] = None,
        extract_trajectories: bool = False,
    ) -> np.ndarray:
        self._total_step_count += 1

        # 1. PPO forward (always, to keep LSTM state updated)
        ppo_action = self.ppo.plan(current_step=current_step, obs=obs)

        # 2. Ensemble forward → epistemic uncertainty + value variance
        ensemble_logits, ensemble_values = self.ensemble.forward_all_with_values(obs)
        all_logits = [self.ppo.last_logits] + ensemble_logits
        epistemic = compute_epistemic(all_logits)
        aleatoric = compute_aleatoric(self.ppo.last_logits)
        all_values = [self.ppo.last_value] + ensemble_values
        value_var = float(np.var(all_values))
        self.last_epistemic = epistemic

        # 3. Always run PDM (on batch_env copy, no side effects on real env)
        pdm_action = self.pdm.plan(
            current_step=current_step,
            obs=obs,
            extract_trajectories=extract_trajectories,
            apply_result=False,
        )

        # 4. Switch based on uncertainty (with PDM lock-in and lookahead)
        lookahead_triggered = False
        if self.config.force_ppo:
            use_pdm = False
        elif self.config.force_pdm:
            use_pdm = True
        elif self._pdm_remaining > 0:
            # PDM locked in — keep using it
            use_pdm = True
            self._pdm_remaining -= 1
        elif self.config.switch_mode == "value_variance" and value_var > self.config.value_variance_threshold:
            use_pdm = True
            self._pdm_remaining = self._pdm_min_steps - 1
        elif self.config.switch_mode == "epistemic" and epistemic > self.threshold:
            use_pdm = True
            self._pdm_remaining = self._pdm_min_steps - 1
        elif self.config.switch_mode == "pdm_score":
            # Fall back to PPO only when NO PDM proposal is viable, i.e. every
            # proposal collides, leaves the road, or fails to reduce the goal
            # distance by at least goal_reduction_epsilon (default 1.0 m).
            use_pdm = self.pdm.last_num_viable_proposals > 0
        elif self._lookahead_steps > 0 and self._lookahead_exceeds_threshold(obs, current_step):
            # Current uncertainty below threshold, but future uncertainty exceeds it
            use_pdm = True
            lookahead_triggered = True
            self._pdm_remaining = self._pdm_min_steps - 1
            log.debug("Lookahead triggered PDM switch at step %d", current_step)
        else:
            use_pdm = False

        if use_pdm:
            self.pdm.apply_best_proposal()
            binding.vec_set_movement_mode(
                self.env.c_envs, [self.agent_idx], MOVEMENT_IDM
            )
            self._using_pdm = True
            self._pdm_step_count += 1
            self._step_planners.append("PDM")
            self._step_epistemics.append(epistemic)
            self._step_aleatorics.append(aleatoric)
            self._step_value_vars.append(value_var)
            self._step_lookahead_triggered.append(lookahead_triggered)
            self._step_num_viable.append(self.pdm.last_num_viable_proposals)
            self._step_best_score.append(self.pdm.last_best_score)
            self.last_counterfactual_action = ppo_action
            self.last_counterfactual_mode = MOVEMENT_DYNAMICS
            return pdm_action
        else:
            binding.vec_set_movement_mode(
                self.env.c_envs, [self.agent_idx], MOVEMENT_DYNAMICS
            )
            self._using_pdm = False
            self._step_planners.append("PPO")
            self._step_epistemics.append(epistemic)
            self._step_aleatorics.append(aleatoric)
            self._step_value_vars.append(value_var)
            self._step_lookahead_triggered.append(False)
            self._step_num_viable.append(self.pdm.last_num_viable_proposals)
            self._step_best_score.append(self.pdm.last_best_score)
            self.last_counterfactual_action = pdm_action
            self.last_counterfactual_mode = MOVEMENT_IDM
            if self.ppo_rollout is not None:
                # PPO-rollout reads ppo.last_logits (already set by ppo.plan above)
                # and returns the best of K branch-evaluated top-K actions.
                return self.ppo_rollout.plan(
                    current_step=current_step,
                    obs=obs,
                    extract_trajectories=extract_trajectories,
                )
            return ppo_action

    def _lookahead_exceeds_threshold(self, obs: np.ndarray, current_step: int) -> bool:
        """Propagate PPO forward N steps in a cloned env to detect future uncertainty spikes.

        Returns True if epistemic uncertainty exceeds the threshold at any lookahead step.
        """
        N = self._lookahead_steps
        if N <= 0 or self._lookahead_env is None:
            return False

        num_agents = self.env.num_agents

        # Save all LSTM states
        self.ppo.save_lstm_state()
        self.ensemble.save_all_lstm_states()
        if self._other_planner is not None and hasattr(self._other_planner, 'save_lstm_state'):
            self._other_planner.save_lstm_state()

        # Snapshot env and restore to lookahead batch_env
        snapshot = self.env.create_snapshot()
        snapshot_handle = snapshot[0]
        self._lookahead_env.restore_snapshot_broadcast(snapshot_handle)

        exceeded = False
        lookahead_obs = obs.copy()

        for step_i in range(N):
            # PPO forward on current lookahead obs
            ppo_action = self.ppo.plan(current_step=current_step + step_i, obs=lookahead_obs)

            # Build action array for all agents
            all_actions = np.zeros((num_agents, 2), dtype=np.float32)
            all_actions[self.agent_idx, :] = ppo_action

            # Other agents
            if num_agents > 1:
                other_indices = [i for i in range(num_agents) if i != self.agent_idx]
                if self._other_planner is not None:
                    batch_obs = self._lookahead_env.observations  # (num_agents, obs_dim)
                    other_obs = batch_obs[other_indices]
                    other_actions = self._other_planner.plan(obs=other_obs)
                    other_actions = np.atleast_2d(other_actions)
                    # Broadcast single action to all others if needed
                    if other_actions.shape[0] == 1 and len(other_indices) > 1:
                        other_actions = np.tile(other_actions, (len(other_indices), 1))
                    for idx, agent_i in enumerate(other_indices):
                        all_actions[agent_i, :] = other_actions[min(idx, len(other_actions) - 1)]

            # Step lookahead env
            new_obs, _, terms, truncs = self._lookahead_env.step(all_actions)

            # Check if ego terminated — stop rollout but don't trigger PDM
            # (only epistemic uncertainty should decide, not collision detection)
            if terms[self.agent_idx] or truncs[self.agent_idx]:
                break

            # Get new ego observation
            lookahead_obs = new_obs[self.agent_idx]

            # Ensemble forward on new obs to compute uncertainty
            ensemble_logits, _ = self.ensemble.forward_all_with_values(lookahead_obs)
            all_logits = [self.ppo.last_logits] + ensemble_logits
            epistemic = compute_epistemic(all_logits)

            if epistemic > self.threshold:
                exceeded = True
                break

        # Restore all LSTM states
        self.ppo.restore_lstm_state()
        self.ensemble.restore_all_lstm_states()
        if self._other_planner is not None and hasattr(self._other_planner, 'restore_lstm_state'):
            self._other_planner.restore_lstm_state()

        # Free snapshot
        self.env.free_snapshot(snapshot)

        return exceeded

    def plot(self, ax, state, axis_limits=None):
        if self._using_pdm:
            self.pdm.plot(ax, state, axis_limits)
        elif self.ppo_rollout is not None:
            self.ppo_rollout.plot(ax, state, axis_limits)

    def reset(self):
        self.ppo.reset()
        self.pdm.reset()
        self.ensemble.reset()
        if self.ppo_rollout is not None:
            self.ppo_rollout.reset()
        self._using_pdm = False

    def close(self):
        """Release all cloned batch envs owned by this planner.

        Must be called per scenario; without it each map leaks the PDM batch
        (15 envs), the PPO-rollout batch (B·C envs when ppo_mode=rollout) and
        the optional lookahead env (1 env when lookahead_steps>0). At ~30+
        leaked Drive C-envs per scenario, OOM hits after a few hundred maps.
        """
        if hasattr(self, "pdm") and self.pdm is not None and \
                getattr(self.pdm, "batch_env", None) is not None:
            self.pdm.batch_env.close()
            self.pdm.batch_env = None
        if self.ppo_rollout is not None:
            self.ppo_rollout.close()
        if self._lookahead_env is not None:
            self._lookahead_env.close()
            self._lookahead_env = None
        # Reset to dynamics mode
        binding.vec_set_movement_mode(
            self.env.c_envs, [self.agent_idx], MOVEMENT_DYNAMICS
        )

        if self._total_step_count > 0:
            pdm_pct = 100.0 * self._pdm_step_count / self._total_step_count
            lookahead_count = sum(self._step_lookahead_triggered)
            log.info(
                "Hybrid stats: PDM used %d/%d steps (%.1f%%), lookahead triggered %d",
                self._pdm_step_count, self._total_step_count, pdm_pct, lookahead_count,
            )
        self._pdm_step_count = 0
        self._total_step_count = 0
        self._pdm_remaining = 0
        self._step_planners = []
        self._step_epistemics = []
        self._step_aleatorics = []
        self._step_value_vars = []
        self._step_lookahead_triggered = []
        self._step_num_viable = []
        self._step_best_score = []
