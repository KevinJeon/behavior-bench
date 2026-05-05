# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""DriveConditionedPaper policy-based planner.

Wraps the paper-exact Gigaflow MLP (pufferlib.ocean.torch.DriveConditionedPaper)
for eval-time inference. The policy emits a Categorical over 12 discrete
jerk combinations (4 longitudinal × 3 lateral, see drive.h JERK_LONG/JERK_LAT).

Two output-encoding modes, chosen from the env's dynamics_model at init:

1. **jerk-env** (dynamics_model == "jerk") — native mode. Emit normalized
   continuous `(a_long_norm, a_lat_norm) ∈ [-1, 1]²` via a precomputed LUT.
   The C env's jerk branch applies the asymmetric mapping internally
   (negative side scaled by 15, positive by 4).

2. **classic-env** (dynamics_model == "classic") — interoperability mode.
   The env's classic branch expects `(accel, steer)` where accel ∈ [-1, 1]
   maps to [-4, 4] m/s² and steer ∈ [-1, 1] rad. We decode the policy's
   jerk-idx to `(Δa_long, Δa_lat)` per-step, integrate against the current
   agent state read from the ego obs (speed, length, a_long, a_lat —
   exposed via the 10-dim jerk obs layout that the env emits when
   `emit_jerk_ego_obs=1`), and produce a one-step target (accel, steer).
   Longitudinal is a perfect one-step integration; lateral inverts the
   centripetal bicycle relation `a_lat ≈ v²·tan(steer)/L`, which is
   accurate for v > ~1 m/s.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from .base import BasePlanner

log = logging.getLogger("planning")


@dataclass
class ConditionedPaperConfig:
    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: int = 128
    hidden_size: int = 1024
    stochastic: bool = False


# Must match drive.h constants.
_JERK_LONG = (-15.0, -4.0, 0.0, 4.0)          # m/s³
_JERK_LAT = (-4.0, 0.0, 4.0)                  # m/s³
_DT = 0.1                                      # env timestep
_MAX_SPEED = 100.0                             # drive.h:130
_MAX_VEH_LEN = 30.0                            # drive.h:131
_MAX_ACCEL_CLASSIC = 4.0                       # ACCELERATION_VALUES[6]
_MAX_STEER_CLASSIC = 1.0                       # STEERING_VALUES[12]  (radians)
# Jerk-env clip ranges (drive.h:2010, 2016).
_A_LONG_CLIP = (-5.0, 2.5)
_A_LAT_CLIP = (-4.0, 4.0)
# Jerk-env steering rate limit (drive.h:2035) and angle clip (drive.h:2036).
# Wheelbase = 0.6 * length (drive.h:804).
_STEER_RATE_LIMIT = 0.6 * _DT                  # rad per step
_STEER_ANGLE_CLIP = 0.55                       # rad
_WHEELBASE_FRACTION = 0.6


class ConditionedPaperPlanner(BasePlanner):
    """Policy planner for DriveConditionedPaper.

    Returns shape (2,) for single-obs input and (N, 2) for batch input.
    Actions are normalized continuous in [-1, 1], consumed directly by the
    jerk-dynamics Drive env.
    """

    def __init__(self, env, agent_idx, action_lb, action_ub, config):
        super().__init__(
            horizon=1,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.device = torch.device(config.device)

        from pufferlib.ocean.torch import DriveConditionedPaper
        from pufferlib.ocean.drive.drive import Drive

        # Build a throwaway env matching training exactly so the policy sees
        # action_space=MultiDiscrete([12]), ego_dim=10, creward_dim=9.
        policy_env = Drive(
            episode_length=env.episode_length,
            action_type="discrete",
            dynamics_model="jerk",
            reward_conditioning=True,
            max_controlled_agents=1,
            split=env.split,
        )
        self.policy = DriveConditionedPaper(
            policy_env,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.policy.eval()
        policy_env.close()

        # Per-idx jerk values (kept on CPU for cheap lookup in plan()).
        jerk_long_per_idx = np.array(
            [_JERK_LONG[i // 3] for i in range(12)], dtype=np.float32)
        jerk_lat_per_idx = np.array(
            [_JERK_LAT[i % 3] for i in range(12)], dtype=np.float32)
        self._jerk_long_lut_np = jerk_long_per_idx
        self._jerk_lat_lut_np = jerk_lat_per_idx

        # Mode: native jerk-env LUT (normalized jerk-actions) or classic-env
        # (state-aware conversion per step).
        self._classic_mode = (getattr(env, "dynamics_model", "classic") == "classic")

        if not self._classic_mode:
            # Pre-normalize the jerk-idx LUT for the jerk-env path (on device).
            long_norm = np.empty(12, dtype=np.float32)
            lat_norm = np.empty(12, dtype=np.float32)
            for idx in range(12):
                a_long = _JERK_LONG[idx // 3]
                a_lat = _JERK_LAT[idx % 3]
                long_norm[idx] = a_long / (-_JERK_LONG[0]) if a_long < 0 else a_long / _JERK_LONG[3]
                lat_norm[idx] = a_lat / _JERK_LAT[2]
            self._long_lut = torch.from_numpy(long_norm).to(self.device)
            self._lat_lut = torch.from_numpy(lat_norm).to(self.device)
        else:
            # Classic-env: keep jerk-idx → (jerk_long, jerk_lat) LUT on device
            # for vectorized gather; the actual classic action is derived
            # per-step from the ego obs state.
            self._long_lut = torch.from_numpy(jerk_long_per_idx).to(self.device)
            self._lat_lut = torch.from_numpy(jerk_lat_per_idx).to(self.device)

        self._obs_buffer = None

        if config.weights_path:
            self._load_weights(config.weights_path)

    def _load_weights(self, weights_path):
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k[len("module."):] if k.startswith("module.") else k
            new_state_dict[new_k] = v
        result = self.policy.load_state_dict(new_state_dict, strict=False)
        if result.missing_keys:
            log.warning("Missing keys when loading DriveConditionedPaper weights: %s",
                        result.missing_keys)
        if result.unexpected_keys:
            log.warning("Unexpected keys when loading DriveConditionedPaper weights: %s",
                        result.unexpected_keys)
        log.info("Loaded DriveConditionedPaper weights from %s", weights_path)

    @property
    def population_size(self):
        return 1

    @property
    def supports_trajectory_proposals(self):
        return False

    def plan(self, current_step=0, obs=None, extract_trajectories=False):
        if obs is None:
            raise ValueError("ConditionedPaperPlanner requires observation input")

        single = obs.ndim == 1
        obs = np.atleast_2d(obs)
        batch_size = obs.shape[0]

        if self._obs_buffer is None or self._obs_buffer.shape[0] != batch_size:
            self._obs_buffer = torch.empty(
                batch_size, obs.shape[1], dtype=torch.float32, device=self.device)
        self._obs_buffer.copy_(torch.as_tensor(obs, dtype=torch.float32))

        with torch.inference_mode():
            action, _value = self.policy(self._obs_buffer)

        # Discrete action: model returns tuple (logits,) where logits is (B, 12).
        if isinstance(action, (list, tuple)):
            logits = action[0]
        else:
            logits = action

        if self.config.stochastic:
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            idx = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
        else:
            idx = torch.argmax(logits, dim=-1)

        if not self._classic_mode:
            # Jerk-env path: env expects normalized (a_long_norm, a_lat_norm).
            result = torch.stack([self._long_lut[idx], self._lat_lut[idx]], dim=-1).cpu().numpy()
            return result[0] if single else result

        # Classic-env path: decode jerk-idx + current agent state → classic
        # (accel, steer) target executable in one step of classic dynamics.
        idx_np = idx.cpu().numpy()
        jerk_long = self._jerk_long_lut_np[idx_np]       # (B,) m/s³
        jerk_lat = self._jerk_lat_lut_np[idx_np]         # (B,) m/s³

        # Recover physical state from the (jerk-format) obs columns:
        #   obs[:,2] = signed_speed / MAX_SPEED      -> speed (m/s)
        #   obs[:,4] = length     / MAX_VEH_LEN      -> length (m)
        #   obs[:,6] = steering_angle / π            -> steering_angle (rad)
        #   obs[:,7] = a_long asymmetrically normalized (see drive.h:2239-2241)
        #   obs[:,8] = a_lat      / JERK_LAT[2]      -> a_lat (m/s²)
        speed = obs[:, 2] * _MAX_SPEED
        length = np.maximum(obs[:, 4] * _MAX_VEH_LEN, 1e-3)
        # CLASSIC env yaw_rate uses `length`, not the JERK env's wheelbase
        # (= 0.6 * length). Inverting `a_lat ≈ v²·tan(steer)/length` requires
        # length here so the classic env actually produces the desired a_lat.
        wheelbase = length
        current_steer = obs[:, 6] * np.pi
        a_long_cur = np.where(obs[:, 7] < 0, obs[:, 7] * -_JERK_LONG[0], obs[:, 7] * _JERK_LONG[3])
        a_lat_cur = obs[:, 8] * _JERK_LAT[2]

        # One-step integration of jerk dynamics, matched to drive.h:1994-2008.
        # Sign-change → hard-stop at 0 (matches C code lines 1998, 2013).
        target_a_long_raw = a_long_cur + jerk_long * _DT
        target_a_long = np.where(a_long_cur * target_a_long_raw < 0, 0.0,
                                 np.clip(target_a_long_raw, *_A_LONG_CLIP))
        target_a_lat_raw = a_lat_cur + jerk_lat * _DT
        target_a_lat = np.where(a_lat_cur * target_a_lat_raw < 0, 0.0,
                                np.clip(target_a_lat_raw, *_A_LAT_CLIP))

        # Classic-env normalization (C classic branch, drive.h:1921-1922):
        #   accel = action[0] * ACCELERATION_VALUES[6] = action[0] * 4
        accel_norm = np.clip(target_a_long / _MAX_ACCEL_CLASSIC, -1.0, 1.0)

        # Steering: replicate the jerk-env pipeline (drive.h:2032-2036) exactly
        # so the classic env sees the same rate-limited, clipped steering the
        # policy implicitly assumes. Raw atan(a_lat·L/v²) without limits
        # causes violent lateral accel spikes and triggers the comfort threshold.
        v_new = speed + 0.5 * (target_a_long + a_long_cur) * _DT  # drive.h:2022
        v_sq = np.maximum(v_new * v_new, 1e-5)
        signed_curvature_des = target_a_lat / v_sq
        target_steer_raw = np.arctan(signed_curvature_des * wheelbase)
        delta_steer = np.clip(target_steer_raw - current_steer,
                              -_STEER_RATE_LIMIT, _STEER_RATE_LIMIT)
        new_steer = np.clip(current_steer + delta_steer,
                            -_STEER_ANGLE_CLIP, _STEER_ANGLE_CLIP)

        steer_norm = np.clip(new_steer / _MAX_STEER_CLASSIC, -1.0, 1.0).astype(np.float32)
        accel_norm = accel_norm.astype(np.float32)

        result = np.stack([accel_norm, steer_norm], axis=-1)
        return result[0] if single else result

    def plot(self, ax, state, axis_limits=None):
        pass
