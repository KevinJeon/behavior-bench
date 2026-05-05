# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Policy-based planner using trained neural network."""

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

import torch

from .base import BasePlanner

log = logging.getLogger("planning")


@dataclass
class PPOConfig:
    """Configuration for policy planner."""

    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Network architecture (must match checkpoint)
    input_size: int = 64
    hidden_size: int = 256
    # Action type the policy was trained with
    policy_action_type: str = "discrete"
    # Stochastic sampling (for diverse rollouts)
    stochastic: bool = False
    temperature: float = 1.0
    # Discrete action space (7 accel × 13 steer)
    accel_values: np.ndarray = None
    steer_values: np.ndarray = None
    # Policy class in pufferlib.ocean.torch (e.g. "Drive" or "DriveConditioned").
    policy_class_name: str = "Drive"
    # Whether the policy was trained with reward_conditioning (appends 9-dim
    # creward block to obs). Must match the training config of the checkpoint.
    reward_conditioning: bool = False

    def __post_init__(self):
        if self.accel_values is None:
            self.accel_values = np.array(
                [-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32
            )
        if self.steer_values is None:
            self.steer_values = np.linspace(-1.0, 1.0, 13, dtype=np.float32)


class PPOPlanner(BasePlanner):
    """
    Planner that uses a trained neural network policy with LSTM.

    The policy outputs discrete actions which are converted to continuous.
    Uses LSTMWrapper to maintain temporal state across steps.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: PPOConfig,
    ):
        super().__init__(
            horizon=1,  # Policy is reactive, no horizon
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.device = torch.device(config.device)

        # Import and create policy with matching architecture
        from pufferlib.ocean import torch as pt
        from pufferlib.ocean.drive.drive import Drive
        from pufferlib.models import LSTMWrapper

        policy_cls = getattr(pt, config.policy_class_name)

        # Create a temporary env with the correct action type for the policy
        policy_env = Drive(
            episode_length=env.episode_length,
            action_type=config.policy_action_type,
            max_controlled_agents=1,
            split=env.split,
            reward_conditioning=config.reward_conditioning,
        )

        base_policy = policy_cls(
            policy_env,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
        )

        # Wrap with LSTM (matching training architecture)
        self.policy = LSTMWrapper(
            policy_env, base_policy,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.policy.eval()

        # Close the temporary env
        policy_env.close()

        # LSTM hidden state (initialized on first plan() call)
        self.lstm_h = None
        self.lstm_c = None

        # Pre-allocated observation buffer (avoids repeated CPU→GPU alloc+transfer)
        self._obs_buffer = None

        # Pre-computed GPU lookup tables for action decoding (avoids GPU→CPU sync per call)
        accel_norm = config.accel_values / np.max(np.abs(config.accel_values))
        steer_norm = config.steer_values / np.max(np.abs(config.steer_values))
        self._accel_lut = torch.from_numpy(accel_norm).float().to(self.device)
        self._steer_lut = torch.from_numpy(steer_norm).float().to(self.device)
        self._num_steer = len(config.steer_values)

        # Load weights
        if config.weights_path:
            self._load_weights(config.weights_path)

    def _load_weights(self, weights_path: str):
        """Load policy weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the checkpoint is the state dict directly
            state_dict = checkpoint

        # Remove prefixes
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            # Remove 'module.' prefix (from DataParallel)
            if new_key.startswith("module."):
                new_key = new_key[7:]
            new_state_dict[new_key] = v

        result = self.policy.load_state_dict(new_state_dict, strict=False)
        if result.missing_keys:
            log.warning("Missing keys when loading weights: %s", result.missing_keys)
        if result.unexpected_keys:
            log.warning("Unexpected keys when loading weights: %s", result.unexpected_keys)
        log.info("Loaded policy weights from %s", weights_path)

    @property
    def last_value(self) -> float:
        """Lazy GPU sync — only triggers when uncertainty analysis actually reads it."""
        if hasattr(self, '_last_value_tensor') and self._last_value_tensor is not None:
            return self._last_value_tensor.mean().item()
        return 0.0

    @last_value.setter
    def last_value(self, val):
        pass  # ignored — use _last_value_tensor instead

    @property
    def population_size(self) -> int:
        """Number of candidates (always 1 for policy planner)."""
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        """No trajectory proposals for policy planner."""
        return False

    def plan(
        self, current_step: int = 0, obs: np.ndarray = None, extract_trajectories: bool = False
    ) -> np.ndarray:
        """
        Plan next action using the policy network.

        Supports both single obs (obs_dim,) and batch obs (N, obs_dim).
        - Single: uses persistent LSTM state across steps, returns (2,)
        - Batch: uses zero LSTM state (no cross-call persistence), returns (N, 2)

        Args:
            current_step: Current episode step (ignored)
            obs: Observation array, shape (obs_dim,) or (N, obs_dim)
            extract_trajectories: Ignored for this planner

        Returns:
            Continuous action (2,) or (N, 2) matching input batch dimension
        """
        if obs is None:
            raise ValueError("PPOPlanner requires observation input")

        single = obs.ndim == 1
        obs = np.atleast_2d(obs)
        batch_size = obs.shape[0]

        # Reuse pre-allocated GPU buffer (avoids alloc + .float() + .to() per call)
        if self._obs_buffer is None or self._obs_buffer.shape[0] != batch_size:
            self._obs_buffer = torch.empty(
                batch_size, obs.shape[1], dtype=torch.float32, device=self.device)
        self._obs_buffer.copy_(torch.as_tensor(obs, dtype=torch.float32))

        # LSTM state: persistent for both single and batch modes
        if self.lstm_h is None or self.lstm_h.shape[0] != batch_size:
            self.lstm_h = torch.zeros(batch_size, self.config.hidden_size, device=self.device)
            self.lstm_c = torch.zeros(batch_size, self.config.hidden_size, device=self.device)
        state = {"lstm_h": self.lstm_h, "lstm_c": self.lstm_c}

        with torch.inference_mode():
            action_logits, value = self.policy.forward_eval(self._obs_buffer, state)

        # Store logits/value lazily (no GPU sync — only materialized when accessed)
        self.last_logits = action_logits
        self._last_value_tensor = value

        # Persist LSTM state
        self.lstm_h = state["lstm_h"]
        self.lstm_c = state["lstm_c"]

        # Decode actions entirely on GPU — single CPU transfer at the end
        if isinstance(action_logits, (list, tuple)):
            if len(action_logits) == 1:
                flat_idx = self._sample_action_gpu(action_logits[0])
                accel_idx = flat_idx // self._num_steer
                steer_idx = flat_idx % self._num_steer
            else:
                accel_idx = self._sample_action_gpu(action_logits[0])
                steer_idx = self._sample_action_gpu(action_logits[1])
        elif hasattr(action_logits, 'mean'):
            result = torch.stack([
                action_logits.mean[:, 0].clamp(self.action_lb[0], self.action_ub[0]),
                action_logits.mean[:, 1].clamp(self.action_lb[1], self.action_ub[1]),
            ], dim=-1).cpu().numpy().astype(np.float32)
            return result[0] if single else result
        else:
            flat_idx = self._sample_action_gpu(action_logits)
            accel_idx = flat_idx // self._num_steer
            steer_idx = flat_idx % self._num_steer

        # GPU LUT lookup + single CPU transfer
        result = torch.stack(
            [self._accel_lut[accel_idx], self._steer_lut[steer_idx]], dim=-1
        ).cpu().numpy()
        return result[0] if single else result

    def _sample_action_gpu(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample action index on GPU (no CPU transfer)."""
        if self.config.stochastic:
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            return torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def _sample_action(self, logits: torch.Tensor) -> np.ndarray:
        """Sample action index from logits (stochastic) or take argmax (deterministic)."""
        if self.config.stochastic:
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            idx = torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
            return idx.cpu().numpy()
        else:
            return torch.argmax(logits, dim=-1).cpu().numpy()

    def plot(self, ax, state, axis_limits=None):
        """No visualization for policy planner."""
        pass

    def save_lstm_state(self):
        """Save current LSTM state for later restoration."""
        if self.lstm_h is not None:
            self._saved_lstm_h = self.lstm_h.clone()
            self._saved_lstm_c = self.lstm_c.clone()
        else:
            self._saved_lstm_h = None
            self._saved_lstm_c = None

    def restore_lstm_state(self):
        """Restore previously saved LSTM state."""
        self.lstm_h = self._saved_lstm_h
        self.lstm_c = self._saved_lstm_c

    def restore_lstm_state_broadcast(self, batch_size: int):
        """Restore saved LSTM state, broadcast to given batch size.

        Saved state shape: (M, hidden_size) where M = num_other_agents.
        Repeats the full agent pattern N = batch_size // M times:
        [agent1, agent2, ..., agentM, agent1, agent2, ..., agentM, ...]

        This matches the interleaved obs layout from batch_env.
        """
        if self._saved_lstm_h is None:
            self.lstm_h = None
            self.lstm_c = None
            return
        M = self._saved_lstm_h.shape[0]
        N = batch_size // M
        self.lstm_h = self._saved_lstm_h.repeat(N, 1)
        self.lstm_c = self._saved_lstm_c.repeat(N, 1)

    def reset(self):
        """Reset LSTM state for new episode."""
        self.lstm_h = None
        self.lstm_c = None


@dataclass
class WorldModelConfig:
    """Configuration for world model planner."""

    weights_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: int = 64
    hidden_size: int = 256
    transition_loss_coef: float = 1.0
    policy_action_type: str = "discrete"
    stochastic: bool = False
    temperature: float = 1.0
    accel_values: np.ndarray = None
    steer_values: np.ndarray = None

    def __post_init__(self):
        if self.accel_values is None:
            self.accel_values = np.array(
                [-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32
            )
        if self.steer_values is None:
            self.steer_values = np.linspace(-1.0, 1.0, 13, dtype=np.float32)


class WorldModelPlanner(BasePlanner):
    """
    Planner using a trained DriveLatentWorldModel with LatentWorldModelWrapper.

    Like PPOPlanner but uses the latent world model architecture.
    Reactive (horizon=1): encodes observation, decodes action at each step.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: WorldModelConfig,
    ):
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

        from pufferlib.ocean.torch import DriveLatentWorldModel, LatentWorldModelWrapper
        from pufferlib.ocean.drive.drive import Drive

        # Create temporary env with correct action type
        policy_env = Drive(
            episode_length=env.episode_length,
            action_type=config.policy_action_type,
            max_controlled_agents=1,
            split=env.split,
        )

        base_policy = DriveLatentWorldModel(
            policy_env,
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            transition_loss_coef=config.transition_loss_coef,
        )

        self.policy = LatentWorldModelWrapper(
            policy_env, base_policy,
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
        ).to(self.device)
        self.policy.eval()

        policy_env.close()

        # Latent state (stored in lstm_h slot)
        self.lstm_h = None
        self.lstm_c = None

        self._obs_buffer = None

        accel_norm = config.accel_values / np.max(np.abs(config.accel_values))
        steer_norm = config.steer_values / np.max(np.abs(config.steer_values))
        self._accel_lut = torch.from_numpy(accel_norm).float().to(self.device)
        self._steer_lut = torch.from_numpy(steer_norm).float().to(self.device)
        self._num_steer = len(config.steer_values)

        if config.weights_path:
            self._load_weights(config.weights_path)

    def _load_weights(self, weights_path: str):
        """Load policy weights from checkpoint."""
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[7:]
            new_state_dict[new_key] = v

        result = self.policy.load_state_dict(new_state_dict, strict=False)
        if result.missing_keys:
            log.warning("Missing keys when loading weights: %s", result.missing_keys)
        if result.unexpected_keys:
            log.warning("Unexpected keys when loading weights: %s", result.unexpected_keys)
        log.info("Loaded world model weights from %s", weights_path)

    @property
    def last_value(self) -> float:
        if hasattr(self, '_last_value_tensor') and self._last_value_tensor is not None:
            return self._last_value_tensor.mean().item()
        return 0.0

    @last_value.setter
    def last_value(self, val):
        pass

    @property
    def population_size(self) -> int:
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        return False

    def plan(
        self, current_step: int = 0, obs: np.ndarray = None, extract_trajectories: bool = False
    ) -> np.ndarray:
        if obs is None:
            raise ValueError("WorldModelPlanner requires observation input")

        single = obs.ndim == 1
        obs = np.atleast_2d(obs)
        batch_size = obs.shape[0]

        if self._obs_buffer is None or self._obs_buffer.shape[0] != batch_size:
            self._obs_buffer = torch.empty(
                batch_size, obs.shape[1], dtype=torch.float32, device=self.device)
        self._obs_buffer.copy_(torch.as_tensor(obs, dtype=torch.float32))

        if self.lstm_h is None or self.lstm_h.shape[0] != batch_size:
            self.lstm_h = torch.zeros(batch_size, self.config.hidden_size, device=self.device)
            self.lstm_c = torch.zeros(batch_size, self.config.hidden_size, device=self.device)
        state = {"lstm_h": self.lstm_h, "lstm_c": self.lstm_c}

        with torch.inference_mode():
            action_logits, value = self.policy.forward_eval(self._obs_buffer, state)

        self.last_logits = action_logits
        self._last_value_tensor = value

        self.lstm_h = state["lstm_h"]
        self.lstm_c = state["lstm_c"]

        if isinstance(action_logits, (list, tuple)):
            if len(action_logits) == 1:
                flat_idx = self._sample_action_gpu(action_logits[0])
                accel_idx = flat_idx // self._num_steer
                steer_idx = flat_idx % self._num_steer
            else:
                accel_idx = self._sample_action_gpu(action_logits[0])
                steer_idx = self._sample_action_gpu(action_logits[1])
        elif hasattr(action_logits, 'mean'):
            result = torch.stack([
                action_logits.mean[:, 0].clamp(self.action_lb[0], self.action_ub[0]),
                action_logits.mean[:, 1].clamp(self.action_lb[1], self.action_ub[1]),
            ], dim=-1).cpu().numpy().astype(np.float32)
            return result[0] if single else result
        else:
            flat_idx = self._sample_action_gpu(action_logits)
            accel_idx = flat_idx // self._num_steer
            steer_idx = flat_idx % self._num_steer

        result = torch.stack(
            [self._accel_lut[accel_idx], self._steer_lut[steer_idx]], dim=-1
        ).cpu().numpy()
        return result[0] if single else result

    def _sample_action_gpu(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.stochastic:
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
            return torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)
        return torch.argmax(logits, dim=-1)

    def plot(self, ax, state, axis_limits=None):
        pass

    def save_lstm_state(self):
        if self.lstm_h is not None:
            self._saved_lstm_h = self.lstm_h.clone()
            self._saved_lstm_c = self.lstm_c.clone()
        else:
            self._saved_lstm_h = None
            self._saved_lstm_c = None

    def restore_lstm_state(self):
        self.lstm_h = self._saved_lstm_h
        self.lstm_c = self._saved_lstm_c

    def restore_lstm_state_broadcast(self, batch_size: int):
        if self._saved_lstm_h is None:
            self.lstm_h = None
            self.lstm_c = None
            return
        M = self._saved_lstm_h.shape[0]
        N = batch_size // M
        self.lstm_h = self._saved_lstm_h.repeat(N, 1)
        self.lstm_c = self._saved_lstm_c.repeat(N, 1)

    def reset(self):
        self.lstm_h = None
        self.lstm_c = None


