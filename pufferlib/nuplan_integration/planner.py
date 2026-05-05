# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""PufferDrive PPO planner for nuPlan evaluation.

Implements nuPlan's AbstractPlanner interface, converting nuPlan scenario data
into PufferDrive observations, running the PPO policy, and returning a trajectory.

Uses the C compute_observations() via binding for exact 1:1 observation matching
with the training environment.
"""

import json
import logging
import math
import os
import tempfile
from typing import List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
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

from .trajectory_filler import build_trajectory

log = logging.getLogger(__name__)

_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


class _MockDriveEnv:
    """Minimal mock of PufferDrive's Drive env for policy construction.

    Provides only the attributes that DrivePolicy and LSTMWrapper read during __init__,
    avoiding the need to load binary scenario data.
    """

    def __init__(self):
        import gymnasium

        from pufferlib.ocean.drive import binding
        obs_size = (binding.EGO_FEATURES_CLASSIC
                    + binding.PARTNER_FEATURES * binding.MAX_OBS_PARTNERS
                    + binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        # Discrete action: 7 accel * 13 steer = 91
        self.single_action_space = gymnasium.spaces.MultiDiscrete([91])

        # Attributes read by DrivePolicy.__init__
        self.max_partner_objects = binding.MAX_OBS_PARTNERS
        self.partner_features = binding.PARTNER_FEATURES
        self.max_road_objects = binding.MAX_ROAD_SEGMENT_OBSERVATIONS
        self.road_features = binding.ROAD_FEATURES
        self.dynamics_model = "classic"

# Discrete action space (must match training config)
ACCEL_VALUES = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
STEER_VALUES = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
NUM_STEER = len(STEER_VALUES)


class PufferDrivePlanner(AbstractPlanner):
    """nuPlan planner that uses a PufferDrive-trained PPO policy."""

    requires_scenario: bool = True

    def __init__(
        self,
        scenario: AbstractScenario,
        weights_path: str,
        input_size: int = 64,
        hidden_size: int = 256,
        device: str = "cuda",
        stochastic: bool = False,
        trajectory_steps: int = 80,
        trajectory_dt: float = 0.1,
        policy_name: str = "Drive",  # "Drive" or "DriveTransformer"
    ):
        super().__init__()
        self._scenario = scenario
        self._weights_path = weights_path
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._stochastic = stochastic
        self._trajectory_steps = trajectory_steps
        self._trajectory_dt = trajectory_dt
        self._policy_name = policy_name

        self._policy = None
        self._c_env_handle = None
        self._obs_buf = None
        self._bin_path = None
        self._goal_x = 0.0
        self._goal_y = 0.0
        self._lstm_h = None
        self._lstm_c = None
        self._goal_reached = False
        self._last_steering = 0.0

        # Per-step logging (activated by PPO_LOG_DIR env var)
        self._log_dir = os.environ.get("PPO_LOG_DIR", "")
        self._step_log = []  # accumulates per-step data
        self._last_value = 0.0
        self._last_entropy = 0.0

    def name(self) -> str:
        return "PufferDrivePPO"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Initialize planner with route and map data, load PPO policy.

        Creates a temporary .bin file from nuPlan data, loads it into a
        lightweight C Drive env for compute_observations().
        """
        from pufferlib.ocean.drive import binding

        from .nuplan_to_binary import nuplan_to_binary

        # Goal = last position of the expert ego trajectory
        expert_traj = list(self._scenario.get_expert_ego_trajectory())
        goal_state = expert_traj[-1]
        self._goal_x = goal_state.center.x
        self._goal_y = goal_state.center.y

        # Store map API for speed limit queries
        self._map_api = initialization.map_api

        # Create temp .bin from nuPlan scenario
        self._bin_path = tempfile.mktemp(suffix=".bin")
        nuplan_to_binary(
            scenario=self._scenario,
            map_api=initialization.map_api,
            route_roadblock_ids=list(initialization.route_roadblock_ids),
            output_path=self._bin_path,
        )

        # Allocate observation buffer (shared with C)
        obs_size = (binding.EGO_FEATURES_CLASSIC
                    + binding.PARTNER_FEATURES * binding.MAX_OBS_PARTNERS
                    + binding.ROAD_FEATURES * binding.MAX_ROAD_SEGMENT_OBSERVATIONS)
        self._obs_buf = np.zeros(obs_size, dtype=np.float32)

        # Init C env: loads .bin, builds grid, caches neighbors
        self._c_env_handle = binding.init_obs_env(self._bin_path, self._obs_buf)

        self._load_policy()
        self._reset_lstm()

        log.info(
            "PufferDrivePlanner initialized with C binding: %d route roadblocks, bin=%s, device=%s",
            len(initialization.route_roadblock_ids),
            self._bin_path,
            self._device,
        )

    def _load_policy(self) -> None:
        """Load PPO policy with LSTM wrapper (or LatentWorldModelWrapper)."""
        mock_env = _MockDriveEnv()

        if self._policy_name == "DriveLatentWorldModel":
            from pufferlib.ocean.torch import DriveLatentWorldModel, LatentWorldModelWrapper

            base_policy = DriveLatentWorldModel(
                mock_env,
                input_size=self._input_size,
                hidden_size=self._hidden_size,
            )
            self._policy = LatentWorldModelWrapper(
                mock_env,
                base_policy,
                input_size=self._hidden_size,
                hidden_size=self._hidden_size,
            ).to(self._device)
        else:
            from pufferlib.models import LSTMWrapper

            if self._policy_name == "DriveTransformer":
                from pufferlib.ocean.torch import DriveTransformer as PolicyClass
            elif self._policy_name == "DriveGameFormer":
                from pufferlib.ocean.torch import DriveGameFormer as PolicyClass
            else:
                from pufferlib.ocean.torch import Drive as PolicyClass

            base_policy = PolicyClass(
                mock_env,
                input_size=self._input_size,
                hidden_size=self._hidden_size,
            )

            self._policy = LSTMWrapper(
                mock_env,
                base_policy,
                input_size=self._hidden_size,
                hidden_size=self._hidden_size,
            ).to(self._device)

        # Load weights
        checkpoint = torch.load(
            self._weights_path, map_location=self._device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove prefixes from DataParallel etc.
        cleaned = {}
        for k, v in state_dict.items():
            key = k.removeprefix("module.")
            cleaned[key] = v

        result = self._policy.load_state_dict(cleaned, strict=False)
        if result.missing_keys:
            log.warning("Missing keys: %s", result.missing_keys)
        if result.unexpected_keys:
            log.warning("Unexpected keys: %s", result.unexpected_keys)

        self._policy.eval()
        log.info("Loaded PPO weights from %s", self._weights_path)

    def _reset_lstm(self) -> None:
        self._lstm_h = torch.zeros(1, self._hidden_size, device=self._device)
        self._lstm_c = torch.zeros(1, self._hidden_size, device=self._device)

    @staticmethod
    def _body_to_global_velocity(vx_body, vy_body, heading):
        """Convert velocity from body frame (longitudinal, lateral) to global frame.

        nuPlan's center_velocity_2d and rear_axle_velocity_2d are in the body frame
        where vx = forward speed, vy = lateral speed.  PufferDrive expects global frame.
        """
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        return (vx_body * cos_h - vy_body * sin_h,
                vx_body * sin_h + vy_body * cos_h)

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """Compute trajectory: build obs -> PPO forward -> bicycle propagation."""
        from pufferlib.ocean.drive import binding

        ego_state, observation = current_input.history.current_state
        tracked_objects = observation.tracked_objects

        # Check if goal is reached — brake with last steering to stay on lane
        dx = ego_state.center.x - self._goal_x
        dy = ego_state.center.y - self._goal_y
        dist_to_goal = math.sqrt(dx * dx + dy * dy)
        speed = abs(ego_state.dynamic_car_state.center_velocity_2d.x)

        if dist_to_goal < 2.0:
            self._goal_reached = True

        if self._goal_reached:
            if speed < 0.1:
                # Already stopped — hold position with zero action
                return build_trajectory(
                    ego_state, 0.0, 0.0,
                    num_steps=self._trajectory_steps,
                    dt=self._trajectory_dt,
                )
            # Brake to stop immediately (1 step)
            brake_accel = -speed / self._trajectory_dt
            return build_trajectory(
                ego_state,
                brake_accel,
                self._last_steering,
                num_steps=self._trajectory_steps,
                dt=self._trajectory_dt,
            )

        # Build agents array from tracked objects [N, 7]: x, y, heading, vx, vy, width, length
        agents = self._tracked_objects_to_array(tracked_objects)

        # Convert ego velocity from body frame to global frame
        ego_vx, ego_vy = self._body_to_global_velocity(
            ego_state.dynamic_car_state.center_velocity_2d.x,
            ego_state.dynamic_car_state.center_velocity_2d.y,
            ego_state.center.heading,
        )

        # Compute observations via C (updates self._obs_buf in-place)
        binding.compute_obs_external(
            self._c_env_handle,
            ego_state.center.x,
            ego_state.center.y,
            ego_state.center.heading,
            ego_vx,
            ego_vy,
            ego_state.car_footprint.width,
            ego_state.car_footprint.length,
            self._goal_x,
            self._goal_y,
            agents,
        )

        obs_tensor = torch.from_numpy(self._obs_buf).float().unsqueeze(0).to(self._device)

        # PPO forward pass with LSTM
        state = {"lstm_h": self._lstm_h, "lstm_c": self._lstm_c}
        with torch.no_grad():
            action_logits, value = self._policy.forward_eval(obs_tensor, state)

        # Persist LSTM state
        self._lstm_h = state["lstm_h"]
        self._lstm_c = state["lstm_c"]

        # Store value for logging/hybrid planner
        self._last_value = float(value.mean().item())

        # Decode action
        if isinstance(action_logits, (list, tuple)):
            logits = action_logits[0] if len(action_logits) == 1 else action_logits
        else:
            logits = action_logits

        # Compute entropy for logging
        def _entropy(l):
            p = torch.softmax(l, dim=-1)
            return float(-(p * torch.log(p + 1e-10)).sum(dim=-1).mean().item())
        if isinstance(logits, (list, tuple)):
            self._last_entropy = sum(_entropy(l) for l in logits)
        else:
            self._last_entropy = _entropy(logits)

        if isinstance(logits, (list, tuple)):
            # Multi-head: separate accel and steer logits
            accel_idx = self._sample(logits[0])
            steer_idx = self._sample(logits[1])
        else:
            # Single-head: flat index
            flat_idx = self._sample(logits)
            accel_idx = flat_idx // NUM_STEER
            steer_idx = flat_idx % NUM_STEER

        acceleration = float(ACCEL_VALUES[accel_idx])
        steering = float(STEER_VALUES[steer_idx])

        # Save steering for goal braking (preserves lane curvature)
        self._last_steering = steering

        # Post-processing: clamp acceleration when over speed limit
        MAX_SPEED = 80.0 / 3.6  # 80 km/h fallback when no lane speed limit found
        if acceleration > 0:
            speed_limit = self._get_speed_limit(ego_state)
            if speed_limit is None:
                speed_limit = MAX_SPEED
            if speed > speed_limit:
                acceleration = 0.0

        # Log per-step data if logging is enabled
        if self._log_dir:
            step_data = {
                "value": self._last_value,
                "entropy": self._last_entropy,
                "accel": acceleration,
                "steer": steering,
                "speed": speed,
                "dist_to_goal": dist_to_goal,
            }
            self._step_log.append(step_data)
            self._log_step_to_csv(step_data)

        # Build trajectory
        return build_trajectory(
            ego_state,
            acceleration,
            steering,
            num_steps=self._trajectory_steps,
            dt=self._trajectory_dt,
        )

    def _log_step_to_csv(self, step_data):
        """Append one step to a shared CSV file (thread/process safe)."""
        if not self._log_dir:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        csv_path = os.path.join(self._log_dir, "ppo_steps.csv")
        log_name = getattr(self._scenario, 'log_name', '')
        scenario_type = getattr(self._scenario, 'scenario_type', '')
        step_idx = len(self._step_log) - 1

        row = f"{log_name},{scenario_type},{step_idx},{step_data['value']:.6f},{step_data['entropy']:.6f},{step_data['accel']:.4f},{step_data['steer']:.4f},{step_data['speed']:.4f},{step_data['dist_to_goal']:.4f}\n"

        import fcntl
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if f.tell() == 0:
                f.write("log_name,scenario_type,step,value,entropy,accel,steer,speed,dist_to_goal\n")
            f.write(row)
            fcntl.flock(f, fcntl.LOCK_UN)

    def _compute_lane_following_steering(self, ego_state) -> float | None:
        """Compute steering angle to follow the nearest lane's centerline.

        Returns None if no lane is found (ego is offroad).
        """
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D

        try:
            point = Point2D(ego_state.center.x, ego_state.center.y)
            nearest = self._map_api.get_proximal_map_objects(
                point, 10.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            lanes = nearest.get(SemanticMapLayer.LANE, []) + nearest.get(SemanticMapLayer.LANE_CONNECTOR, [])
            if not lanes:
                return None

            ego_x, ego_y = ego_state.center.x, ego_state.center.y
            ego_h = ego_state.center.heading

            # Find the closest lane by minimum distance to its centerline
            best_lane = None
            best_dist = float('inf')
            best_closest_idx = 0
            for lane_candidate in lanes:
                path_c = lane_candidate.baseline_path.discrete_path
                for i, p in enumerate(path_c):
                    d = math.sqrt((p.x - ego_x)**2 + (p.y - ego_y)**2)
                    if d < best_dist:
                        best_dist = d
                        best_lane = lane_candidate
                        best_closest_idx = i

            if best_lane is None:
                return None

            path = best_lane.baseline_path.discrete_path
            closest_idx = best_closest_idx
            min_dist = best_dist

            # Lookahead point ~5m ahead along centerline
            speed = abs(ego_state.dynamic_car_state.center_velocity_2d.x)
            lookahead = max(5.0, speed * 0.5)
            target_idx = closest_idx
            accumulated = 0.0
            for i in range(closest_idx, len(path) - 1):
                dx = path[i+1].x - path[i].x
                dy = path[i+1].y - path[i].y
                accumulated += math.sqrt(dx*dx + dy*dy)
                target_idx = i + 1
                if accumulated >= lookahead:
                    break

            # Compute steering angle using pure pursuit
            target = path[min(target_idx, len(path) - 1)]
            dx = target.x - ego_x
            dy = target.y - ego_y
            # Transform to ego frame
            cos_h = math.cos(ego_h)
            sin_h = math.sin(ego_h)
            local_x = dx * cos_h + dy * sin_h   # forward
            local_y = -dx * sin_h + dy * cos_h  # left

            # Pure pursuit: steering = atan2(2 * L * local_y, dist^2)
            dist_sq = local_x**2 + local_y**2
            if dist_sq < 0.01:
                return 0.0
            wheelbase = ego_state.car_footprint.length
            steering = math.atan2(2.0 * wheelbase * local_y, dist_sq)
            return max(-0.5, min(0.5, steering))  # clamp to reasonable range

        except Exception:
            return None

    def _get_speed_limit(self, ego_state) -> float | None:
        """Query the speed limit at the ego's current position from the map."""
        from nuplan.common.maps.abstract_map import SemanticMapLayer
        from nuplan.common.actor_state.state_representation import Point2D

        try:
            point = Point2D(ego_state.center.x, ego_state.center.y)
            nearest = self._map_api.get_proximal_map_objects(
                point, 5.0, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
            )
            lanes = nearest.get(SemanticMapLayer.LANE, []) + nearest.get(SemanticMapLayer.LANE_CONNECTOR, [])
            if lanes:
                return lanes[0].speed_limit_mps
        except Exception:
            pass
        return None

    _NUPLAN_TYPE_MAP = {
        TrackedObjectType.VEHICLE: 1.0,
        TrackedObjectType.PEDESTRIAN: 2.0,
        TrackedObjectType.BICYCLE: 3.0,
    }

    def _tracked_objects_to_array(self, tracked_objects) -> np.ndarray:
        """Convert nuPlan tracked objects to Nx8 float32 array.

        Columns: [x, y, heading, vx, vy, width, length, type]
        Agent velocities are already in global frame (unlike ego velocity).
        Type: 1=vehicle, 2=pedestrian, 3=cyclist (matching C entity types).
        """
        agents = tracked_objects.get_tracked_objects_of_types(_AGENT_TYPES)
        if not agents:
            return np.zeros((0, 8), dtype=np.float32)

        rows = []
        for agent in agents:
            rows.append([
                agent.box.center.x,
                agent.box.center.y,
                agent.box.center.heading,
                agent.velocity.x,
                agent.velocity.y,
                agent.box.width,
                agent.box.length,
                self._NUPLAN_TYPE_MAP.get(agent.tracked_object_type, 1.0),
            ])
        return np.array(rows, dtype=np.float32)

    def _sample(self, logits: torch.Tensor) -> int:
        """Sample or argmax from logits."""
        if self._stochastic:
            probs = torch.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return int(idx.item())
        return int(torch.argmax(logits, dim=-1).item())
