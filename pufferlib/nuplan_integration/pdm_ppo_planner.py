# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Hybrid PDM+PPO planner for nuPlan evaluation.

Combines a PPO policy with a PDM planner. Uses an ensemble of PPO policies
to estimate epistemic uncertainty (mutual information). When uncertainty
exceeds a threshold, switches from PPO to PDM for safer planning.

The PPO LSTM state is always updated (even when PDM is used) to maintain
temporal coherence for when PPO resumes control.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Type

import numpy as np
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import (
    AbstractTrajectory,
)

from .planner import PufferDrivePlanner, _MockDriveEnv, ACCEL_VALUES, STEER_VALUES, NUM_STEER
from .pdm_planner import PDMNuPlanPlanner  # wraps tuplan_garage PDMClosedPlanner
from .trajectory_filler import build_trajectory

log = logging.getLogger(__name__)


@dataclass
class PDMPPOConfig:
    """Configuration for the hybrid PDM+PPO nuPlan planner."""

    # PPO policy
    weights_path: str = ""
    ensemble_weights: List[str] = field(default_factory=list)
    input_size: int = 64
    hidden_size: int = 256
    device: str = "cuda"
    stochastic: bool = False

    # PDM parameters
    pdm_velocity_fractions: str = "0.0,0.2,0.4,0.6,0.8,1.0"
    pdm_lateral_offsets: str = "-1.5,-0.5,0.0,0.5,1.5"
    pdm_idm_desired_velocity: float = 15.0

    # Switching parameters
    epistemic_threshold: float = 0.8
    pdm_min_steps: int = 1
    switch_mode: str = "epistemic"  # "epistemic" or "value_variance"
    value_variance_threshold: float = 0.1

    # Trajectory
    trajectory_steps: int = 80
    trajectory_dt: float = 0.1


class PDMPPONuPlanPlanner(AbstractPlanner):
    """Hybrid planner: PPO with PDM fallback based on ensemble uncertainty.

    At each step:
    1. Always run PPO forward pass (keeps LSTM state updated)
    2. Compute epistemic uncertainty from ensemble disagreement
    3. Compute PDM's best proposal trajectory
    4. If uncertainty > threshold: use PDM trajectory
       Else: use PPO trajectory

    The PPO's LSTM state is always maintained regardless of which planner's
    trajectory is used, ensuring smooth transitions when switching back to PPO.
    """

    requires_scenario: bool = True

    def __init__(
        self,
        scenario: AbstractScenario,
        weights_path: str = "",
        ensemble_weights: str = "",
        input_size: int = 64,
        hidden_size: int = 256,
        device: str = "cuda",
        stochastic: bool = False,
        epistemic_threshold: float = 0.8,
        pdm_min_steps: int = 1,
        switch_mode: str = "epistemic",
        value_variance_threshold: float = 0.1,
        pdm_velocity_fractions: str = "0.0,0.2,0.4,0.6,0.8,1.0",
        pdm_lateral_offsets: str = "-1.5,-0.5,0.0,0.5,1.5",
        pdm_idm_desired_velocity: float = 15.0,
        trajectory_steps: int = 80,
        trajectory_dt: float = 0.1,
    ):
        super().__init__()
        self._scenario = scenario
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        ew = [p.strip() for p in ensemble_weights.split(",") if p.strip()] if ensemble_weights else []

        self._config = PDMPPOConfig(
            weights_path=weights_path,
            ensemble_weights=ew,
            input_size=input_size,
            hidden_size=hidden_size,
            device=device,
            stochastic=stochastic,
            pdm_velocity_fractions=pdm_velocity_fractions,
            pdm_lateral_offsets=pdm_lateral_offsets,
            pdm_idm_desired_velocity=pdm_idm_desired_velocity,
            epistemic_threshold=epistemic_threshold,
            pdm_min_steps=pdm_min_steps,
            switch_mode=switch_mode,
            value_variance_threshold=value_variance_threshold,
            trajectory_steps=trajectory_steps,
            trajectory_dt=trajectory_dt,
        )

        # Sub-planners (initialized lazily in initialize())
        self._ppo_planner = None
        self._pdm_planner = None

        # Ensemble members
        self._ensemble_policies = []
        self._ensemble_lstm_h = []
        self._ensemble_lstm_c = []

        # Switching state
        self._pdm_remaining = 0
        self._ppo_remaining = 0
        self._using_pdm = False
        self._step_count = 0
        self._pdm_step_count = 0
        self._last_ego_speed = 10.0  # assume moving initially

        # Action smoothing for PPO (EMA)
        self._smooth_alpha = 0.5  # 0=full smoothing, 1=no smoothing
        self._last_ppo_accel = 0.0
        self._last_ppo_steer = 0.0

    def name(self) -> str:
        return "PDMPPOHybrid"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        # Initialize PPO planner
        self._ppo_planner = PufferDrivePlanner(
            scenario=self._scenario,
            weights_path=self._config.weights_path,
            input_size=self._config.input_size,
            hidden_size=self._config.hidden_size,
            device=self._config.device,
            stochastic=self._config.stochastic,
            trajectory_steps=self._config.trajectory_steps,
            trajectory_dt=self._config.trajectory_dt,
        )
        self._ppo_planner.initialize(initialization)

        # Initialize PDM planner (tuplan_garage PDMClosedPlanner wrapper)
        self._pdm_planner = PDMNuPlanPlanner(
            speed_limit_fraction=self._config.pdm_velocity_fractions,
            lateral_offsets=self._config.pdm_lateral_offsets,
            fallback_target_velocity=self._config.pdm_idm_desired_velocity,
        )
        self._pdm_planner.initialize(initialization)

        # Load ensemble models for uncertainty estimation
        self._load_ensemble()

        log.info(
            "PDMPPONuPlanPlanner initialized: ensemble_size=%d, threshold=%.2f, "
            "switch_mode=%s, pdm_min_steps=%d",
            len(self._ensemble_policies),
            self._config.epistemic_threshold,
            self._config.switch_mode,
            self._config.pdm_min_steps,
        )

    def _load_ensemble(self) -> None:
        """Load ensemble PPO policies for uncertainty estimation."""
        if not self._config.ensemble_weights:
            log.warning("No ensemble weights provided — uncertainty will always be 0")
            return

        from pufferlib.models import LSTMWrapper
        from pufferlib.ocean.torch import Drive as DrivePolicy

        mock_env = _MockDriveEnv()
        hidden_size = self._config.hidden_size

        for weight_path in self._config.ensemble_weights:
            base = DrivePolicy(mock_env, input_size=self._config.input_size, hidden_size=hidden_size)
            policy = LSTMWrapper(mock_env, base, input_size=hidden_size, hidden_size=hidden_size)
            policy = policy.to(self._device)

            checkpoint = torch.load(weight_path, map_location=self._device, weights_only=False)
            if "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                sd = checkpoint["state_dict"]
            else:
                sd = checkpoint
            cleaned = {k.removeprefix("module."): v for k, v in sd.items()}
            policy.load_state_dict(cleaned, strict=False)
            policy.eval()

            self._ensemble_policies.append(policy)
            self._ensemble_lstm_h.append(
                torch.zeros(1, hidden_size, device=self._device)
            )
            self._ensemble_lstm_c.append(
                torch.zeros(1, hidden_size, device=self._device)
            )

        log.info("Loaded %d ensemble members", len(self._ensemble_policies))

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        self._step_count += 1

        # 1. Always run PPO (keeps LSTM state updated, gets logits + value)
        ppo_trajectory = self._ppo_planner.compute_planner_trajectory(current_input)

        # 2. Compute uncertainty from ensemble
        epistemic, value_var = self._compute_uncertainty(current_input)

        # 3. Always compute PDM trajectory
        pdm_trajectory = self._pdm_planner.compute_planner_trajectory(current_input)

        # Track ego speed for "stuck" detection
        ego_state = current_input.history.current_state[0]
        self._last_ego_speed = abs(ego_state.dynamic_car_state.center_velocity_2d.x)

        # 4. Switch decision
        use_pdm = self._should_use_pdm(epistemic, value_var)

        # Signal to HybridEgoController which tracking mode to use
        from pufferlib.nuplan_integration import hybrid_controller
        hybrid_controller.USE_PDM = use_pdm

        if use_pdm:
            self._using_pdm = True
            self._pdm_step_count += 1
            # Reset smoothing state when switching to PDM
            self._last_ppo_accel = 0.0
            self._last_ppo_steer = 0.0
            result = pdm_trajectory
        else:
            self._using_pdm = False
            result = ppo_trajectory

        # Log PDM usage ratio periodically
        if self._step_count % 50 == 0 or self._step_count == 1:
            pdm_pct = 100.0 * self._pdm_step_count / self._step_count
            pdm_best = self._pdm_planner.last_best_score
            pdm_mean = self._pdm_planner.last_score_mean
            pdm_viable = self._pdm_planner.last_num_above_0
            log.info(
                "Step %d: PDM %d/%d (%.1f%%), pdm_best=%.3f, pdm_mean=%.3f, viable=%d, using=%s",
                self._step_count, self._pdm_step_count, self._step_count,
                pdm_pct, pdm_best, pdm_mean, pdm_viable, "PDM" if use_pdm else "PPO",
            )

        return result

    def _smooth_ppo_trajectory(self, current_input, ppo_trajectory):
        """Apply exponential moving average smoothing to PPO actions for comfort."""
        import math
        # Extract the raw accel/steer that PPO used (stored in planner)
        raw_accel = self._ppo_planner._last_steering  # this is actually steering
        # We need to get the actual action values - read from the PPO planner's last action
        # The PPO planner stores _last_steering but not _last_accel.
        # Instead, reconstruct from the trajectory: compare first two states
        states = ppo_trajectory.get_sampled_trajectory()
        if len(states) < 2:
            return ppo_trajectory

        ego_state = states[0]
        next_state = states[1]

        # Estimate acceleration from velocity change
        v0 = ego_state.dynamic_car_state.speed
        v1 = next_state.dynamic_car_state.speed
        dt = self._config.trajectory_dt
        raw_accel = (v1 - v0) / dt if dt > 0 else 0.0

        # Estimate steering from heading change
        h0 = ego_state.center.heading
        h1 = next_state.center.heading
        dh = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi
        raw_steer = dh  # approximate

        # EMA smoothing
        alpha = self._smooth_alpha
        smooth_accel = alpha * raw_accel + (1 - alpha) * self._last_ppo_accel
        smooth_steer = alpha * raw_steer + (1 - alpha) * self._last_ppo_steer
        self._last_ppo_accel = smooth_accel
        self._last_ppo_steer = smooth_steer

        # Rebuild trajectory with smoothed actions
        return build_trajectory(
            ego_state, smooth_accel, smooth_steer,
            num_steps=self._config.trajectory_steps,
            dt=self._config.trajectory_dt,
        )

    def _compute_uncertainty(
        self, current_input: PlannerInput
    ) -> tuple[float, float]:
        """Compute epistemic uncertainty and value variance from ensemble.

        Returns (epistemic_mi, value_variance).
        """
        if not self._ensemble_policies:
            return 0.0, 0.0

        from pufferlib.ocean.drive import binding

        # Get the observation that was computed during PPO forward pass
        obs_tensor = torch.from_numpy(self._ppo_planner._obs_buf).float().unsqueeze(0).to(self._device)

        # Get PPO's logits and value
        ppo_logits = self._ppo_planner._policy  # need last logits
        # Re-forward PPO to get logits (LSTM state already updated, but we need logits)
        # Actually, PPO already ran — we need to extract its last logits
        # The PPO planner doesn't expose logits directly, so we re-forward with current LSTM
        # This is a no-op for LSTM state since we save/restore

        saved_h = self._ppo_planner._lstm_h.clone()
        saved_c = self._ppo_planner._lstm_c.clone()

        # Rewind LSTM to before PPO ran (we can't, it already advanced)
        # Instead, just forward ensemble members with current obs
        all_logits = []
        all_values = []

        # Forward each ensemble member
        for i, policy in enumerate(self._ensemble_policies):
            state = {
                "lstm_h": self._ensemble_lstm_h[i],
                "lstm_c": self._ensemble_lstm_c[i],
            }
            with torch.no_grad():
                logits, value = policy.forward_eval(obs_tensor, state)
            self._ensemble_lstm_h[i] = state["lstm_h"]
            self._ensemble_lstm_c[i] = state["lstm_c"]
            all_logits.append(logits)
            all_values.append(float(value.mean().item()))

        if len(all_logits) < 2:
            return 0.0, 0.0

        # Compute epistemic uncertainty (mutual information across ensemble)
        epistemic = self._compute_epistemic_mi(all_logits)
        value_var = float(np.var(all_values))

        return epistemic, value_var

    def _compute_epistemic_mi(self, all_logits) -> float:
        """Compute epistemic uncertainty as mutual information across ensemble members."""
        # Collect softmax probabilities from each member
        all_probs = []
        for logits in all_logits:
            if isinstance(logits, (list, tuple)):
                # Multi-head: concatenate
                probs = [torch.softmax(l, dim=-1) for l in logits]
            else:
                probs = [torch.softmax(logits, dim=-1)]
            all_probs.append(probs)

        # Compute MI per head
        total_mi = 0.0
        num_heads = len(all_probs[0])

        for head_idx in range(num_heads):
            head_probs = [member[head_idx] for member in all_probs]
            stacked = torch.stack(head_probs, dim=0)  # (K, batch, num_actions)

            # Mean probability across ensemble
            mean_probs = stacked.mean(dim=0)  # (batch, num_actions)

            # Entropy of mean: H(p_bar)
            entropy_mean = -torch.sum(
                mean_probs * torch.log(mean_probs + 1e-10), dim=-1
            ).mean().item()

            # Mean entropy: mean(H(p_k))
            member_entropies = -torch.sum(
                stacked * torch.log(stacked + 1e-10), dim=-1
            )  # (K, batch)
            mean_entropy = member_entropies.mean().item()

            # MI = H(p_bar) - mean(H(p_k))
            total_mi += max(0.0, entropy_mean - mean_entropy)

        return total_mi

    def _should_use_pdm(self, epistemic: float, value_var: float) -> bool:
        """Decide whether to use PDM based on PDM's own confidence.

        Default is PDM. Switch to PPO only when PDM signals weakness.
        """
        cfg = self._config

        # If PPO is locked in, keep using PPO
        if self._ppo_remaining > 0:
            self._ppo_remaining -= 1
            return False  # use PPO

        pdm_best = self._pdm_planner.last_best_score
        pdm_mean = self._pdm_planner.last_score_mean

        if cfg.switch_mode == "pdm_score":
            # PDM best score below threshold → use PPO
            if pdm_best < cfg.epistemic_threshold:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "pdm_score_mean":
            # PDM mean score below threshold → early warning, use PPO
            if pdm_mean < cfg.epistemic_threshold:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "pdm_viable":
            # Switch to PPO when few proposals are viable (early warning)
            pdm_viable = self._pdm_planner.last_num_above_0
            if pdm_viable <= cfg.epistemic_threshold:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "pdm_score_stuck":
            # PDM best score < 0.1 OR ego is stuck (speed < 0.1 m/s)
            ego_speed = self._last_ego_speed
            if pdm_best < 0.1 or ego_speed < 0.1:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "pdm_score_brake":
            # PDM best score < 0.1 OR emergency brake triggered
            pdm_brake = getattr(self._pdm_planner, '_last_emergency_brake', False)
            if pdm_best < 0.1 or pdm_brake:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "pdm_combined":
            # Combined: best_score < 0.1 OR score_mean < threshold
            if pdm_best < 0.1 or pdm_mean < cfg.epistemic_threshold:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False
        elif cfg.switch_mode == "epistemic":
            if epistemic < cfg.epistemic_threshold:
                self._ppo_remaining = cfg.pdm_min_steps - 1
                return False

        return True  # default: use PDM
