# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""SMART trajectory prediction model as a nuPlan planner.

Uses the SMART autoregressive motion-token model to predict ego trajectories.
Unlike the PPO planner, SMART directly outputs position trajectories without
needing bicycle dynamics.
"""

import logging
import math
import os
import tempfile
from collections import defaultdict
from typing import List, Type

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
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
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)

log = logging.getLogger(__name__)

_VEHICLE_PARAMS = get_pacifica_parameters()
_AGENT_TYPES = [
    TrackedObjectType.VEHICLE,
    TrackedObjectType.PEDESTRIAN,
    TrackedObjectType.BICYCLE,
]


class SMARTNuPlanPlanner(AbstractPlanner):
    """SMART trajectory prediction as nuPlan planner."""

    requires_scenario: bool = True

    def __init__(
        self,
        scenario: AbstractScenario,
        weights_path: str,
        device: str = "cpu",
        temperature: float = 1.0,
        greedy: bool = True,
        shift: int = 5,
        num_historical_steps: int = 11,
        trajectory_steps: int = 80,
        trajectory_dt: float = 0.1,
    ):
        self._scenario = scenario
        self._weights_path = weights_path
        self._device = device
        self._temperature = temperature
        self._greedy = greedy
        self._shift = shift
        self._num_historical_steps = num_historical_steps
        self._trajectory_steps = trajectory_steps
        self._trajectory_dt = trajectory_dt

        # Lazy-loaded
        self._model = None
        self._motion_codebook_data = None
        self._map_codebook = None
        self._map_data = None
        self._history = None
        self._track_id_to_idx = {}
        self._step = 0
        self._initialized = False

    # -- AbstractPlanner interface --

    def name(self) -> str:
        return "SMARTNuPlanPlanner"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        from pufferlib.planning.smart import load_smart_model, HistoryBuffer
        from pufferlib.prediction.map_tokenizer import tokenize_roads
        from pufferlib.nuplan_integration.nuplan_to_binary import nuplan_to_binary
        from pufferlib.prediction.binary_reader import read_binary_scenario

        # Load model + codebooks
        self._model, self._motion_codebook_data, self._map_codebook = load_smart_model(
            self._weights_path, self._device
        )

        # Create temporary binary from scenario for map tokenization
        tmp_bin = tempfile.mktemp(suffix=".bin")
        route_ids = list(self._scenario.get_route_roadblock_ids())
        nuplan_to_binary(
            self._scenario,
            self._scenario.map_api,
            route_ids,
            tmp_bin,
        )

        # Read binary and tokenize roads
        scenario_data = read_binary_scenario(tmp_bin)
        self._map_data = tokenize_roads(scenario_data["roads"], self._map_codebook)
        os.unlink(tmp_bin)

        # Initialize history buffer
        self._history = HistoryBuffer(max_agents=256, max_steps=200)
        self._track_id_to_idx = {}
        self._step = 0
        self._initialized = True

        log.info(
            "SMARTNuPlanPlanner initialized: %d road tokens, device=%s",
            self._map_data["pt_token"]["position"].shape[0] if self._map_data else 0,
            self._device,
        )

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        from pufferlib.prediction.trajectory_tokenizer import compute_token_data

        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]

        # Extract tracked objects
        tracked = observation.tracked_objects
        agents = tracked.get_tracked_objects_of_types(_AGENT_TYPES)

        # Build agent arrays: ego first, then tracked objects
        ego_x, ego_y = ego_state.center.x, ego_state.center.y
        ego_h = ego_state.center.heading
        ego_w = ego_state.car_footprint.width
        ego_l = ego_state.car_footprint.length

        # Map tracked objects to consistent indices via track_token
        all_x = [ego_x]
        all_y = [ego_y]
        all_h = [ego_h]
        all_w = [ego_w]
        all_l = [ego_l]

        for agent in agents:
            tid = agent.track_token
            if tid not in self._track_id_to_idx:
                self._track_id_to_idx[tid] = len(self._track_id_to_idx) + 1  # 0 = ego
            all_x.append(agent.box.center.x)
            all_y.append(agent.box.center.y)
            all_h.append(agent.box.center.heading)
            all_w.append(agent.box.width)
            all_l.append(agent.box.length)

        num_agents = len(all_x)
        x = np.array(all_x, dtype=np.float32)
        y = np.array(all_y, dtype=np.float32)
        h = np.array(all_h, dtype=np.float32)
        w = np.array(all_w, dtype=np.float32)
        l = np.array(all_l, dtype=np.float32)

        # Append to history
        self._history.append(x, y, h, w, l, num_agents)
        self._step += 1

        # Not enough history yet — return constant-velocity trajectory
        if self._history.num_steps < self._num_historical_steps:
            return self._constant_velocity_trajectory(ego_state)

        # Build HeteroData and run SMART inference
        data = self._build_hetero_data()
        if data is None:
            return self._constant_velocity_trajectory(ego_state)

        from torch_geometric.data import Batch
        batch = Batch.from_data_list([data])
        batch = batch.to(self._device)

        with torch.no_grad():
            result = self._model.inference(
                batch,
                greedy=self._greedy,
                max_steps=1,  # Only 1 token (5 steps) — rest filled by constant velocity
            )

        # Extract ego trajectory (agent 0 in filtered data)
        pred_traj = result["pred_traj"][0].cpu().numpy()  # (T_pred, 2)
        pred_head = result["pred_head"][0].cpu().numpy()  # (T_pred,)

        # Convert from centered coords back to global
        center = self._center_pos
        pred_traj[:, 0] += center[0]
        pred_traj[:, 1] += center[1]

        # SMART with max_steps=1 only predicts 5 steps (1 token × shift=5)
        # Fill remaining trajectory with constant-velocity extrapolation
        num_predicted = min(len(pred_traj), self._shift)
        if num_predicted < self._trajectory_steps:
            full_traj = np.zeros((self._trajectory_steps, 2), dtype=np.float32)
            full_head = np.zeros(self._trajectory_steps, dtype=np.float32)
            full_traj[:num_predicted] = pred_traj[:num_predicted]
            full_head[:num_predicted] = pred_head[:num_predicted]

            # Constant velocity from last predicted point
            last_pos = full_traj[num_predicted - 1]
            last_head = full_head[num_predicted - 1]
            if num_predicted >= 2:
                dx = full_traj[num_predicted - 1, 0] - full_traj[num_predicted - 2, 0]
                dy = full_traj[num_predicted - 1, 1] - full_traj[num_predicted - 2, 1]
            else:
                dx = (last_pos[0] - ego_state.center.x) / self._trajectory_dt
                dy = (last_pos[1] - ego_state.center.y) / self._trajectory_dt
                dx *= self._trajectory_dt
                dy *= self._trajectory_dt

            for i in range(num_predicted, self._trajectory_steps):
                full_traj[i, 0] = full_traj[i - 1, 0] + dx
                full_traj[i, 1] = full_traj[i - 1, 1] + dy
                full_head[i] = last_head

            pred_traj = full_traj
            pred_head = full_head

        # Build nuPlan trajectory from predicted positions
        return self._positions_to_trajectory(ego_state, pred_traj, pred_head)

    # -- Internal methods --

    def _build_hetero_data(self):
        """Build HeteroData from history buffer + cached map data."""
        from pufferlib.prediction.trajectory_tokenizer import compute_token_data
        from torch_geometric.data import HeteroData

        hist = self._history
        N = hist.num_agents
        T_hist = self._num_historical_steps
        T_future = self._trajectory_steps
        T = T_hist + T_future
        shift = self._shift

        actual_steps = min(hist.num_steps, T_hist)

        # Build arrays
        position = np.zeros((N, T, 2), dtype=np.float32)
        heading = np.zeros((N, T), dtype=np.float32)
        velocity = np.zeros((N, T, 2), dtype=np.float32)
        valid_mask = np.zeros((N, T), dtype=bool)

        start = T_hist - actual_steps
        src_start = hist.num_steps - actual_steps
        position[:N, start:T_hist] = hist.position[:N, src_start:hist.num_steps]
        heading[:N, start:T_hist] = hist.heading[:N, src_start:hist.num_steps]
        velocity[:N, start:T_hist] = hist.velocity[:N, src_start:hist.num_steps]
        valid_mask[:N, start:T_hist] = hist.valid[:N, src_start:hist.num_steps]

        # Filter to currently valid agents
        currently_valid = valid_mask[:, T_hist - 1].copy()
        currently_valid[0] = True  # ego always valid
        valid_indices = np.where(currently_valid)[0]
        M = len(valid_indices)

        if M == 0:
            return None

        ego_idx = 0  # ego is always first

        position = position[valid_indices]
        heading = heading[valid_indices]
        velocity = velocity[valid_indices]
        valid_mask = valid_mask[valid_indices]

        # Shape
        shape_arr = np.zeros((M, T, 3), dtype=np.float32)
        shape_arr[:, :, 0] = hist.width[valid_indices, np.newaxis]
        shape_arr[:, :, 1] = hist.length[valid_indices, np.newaxis]

        # Forward/backward fill invalid positions
        t_indices = np.arange(T)
        for i in range(M):
            v = valid_mask[i]
            if not v.any():
                continue
            first_valid = int(np.argmax(v))
            if first_valid > 0:
                position[i, :first_valid] = position[i, first_valid]
                heading[i, :first_valid] = heading[i, first_valid]
                velocity[i, :first_valid] = velocity[i, first_valid]
            fill_idx = np.maximum.accumulate(np.where(v, t_indices, 0))
            position[i] = position[i, fill_idx]
            heading[i] = heading[i, fill_idx]
            velocity[i] = velocity[i, fill_idx]

        # Scene center
        center_pos = position[ego_idx, T_hist - 1].copy()
        self._center_pos = center_pos.copy()

        # Motion tokens for historical portion
        motion_cb = self._motion_codebook_data["token"]["veh"]
        num_total_tokens = (T - 1) // shift
        token_data = compute_token_data(
            positions=position[:, :T_hist],
            headings=heading[:, :T_hist],
            valid_masks=valid_mask[:, :T_hist],
            shift=shift,
            codebook=motion_cb,
        )
        num_hist_tokens = token_data["token_idx"].shape[1]
        if num_hist_tokens < num_total_tokens:
            pad = torch.zeros(M, num_total_tokens - num_hist_tokens, dtype=torch.long)
            token_data["token_idx"] = torch.cat([token_data["token_idx"], pad], dim=1)

        # Center positions
        position -= center_pos[np.newaxis, np.newaxis, :]

        # Token positions/headings
        shift_indices = list(range(shift, T, shift))
        num_tokens = len(shift_indices)
        token_pos = torch.from_numpy(position[:, shift_indices, :]).float()
        token_heading = torch.from_numpy(heading[:, shift_indices]).float()
        center_torch = torch.from_numpy(center_pos).float()
        num_ht = min(token_data["token_pos"].shape[1], num_tokens)
        token_pos[:, :num_ht] = token_data["token_pos"] - center_torch
        token_heading[:, :num_ht] = token_data["token_heading"]

        # Token validity
        vm = valid_mask[:, shift_indices]
        vm_prev = valid_mask[:, [max(0, s - shift) for s in shift_indices]]
        token_valid = torch.from_numpy(vm & vm_prev)

        # Category
        category = torch.full((M,), 3, dtype=torch.long)
        category[ego_idx] = 5

        # Build HeteroData
        data = HeteroData()
        data["agent"]["position"] = torch.from_numpy(position).float()
        data["agent"]["heading"] = torch.from_numpy(heading).float()
        data["agent"]["velocity"] = torch.from_numpy(velocity).float()
        data["agent"]["valid_mask"] = torch.from_numpy(valid_mask).bool()
        data["agent"]["type"] = torch.zeros(M, dtype=torch.long)
        data["agent"]["shape"] = torch.from_numpy(shape_arr).float()
        data["agent"]["category"] = category
        data["agent"]["num_nodes"] = M
        data["agent"]["av_index"] = ego_idx
        data["agent"]["token_pos"] = token_pos
        data["agent"]["token_heading"] = token_heading
        data["agent"]["token_idx"] = token_data["token_idx"]
        data["agent"]["agent_valid_mask"] = token_valid

        # Map tokens
        if self._map_data is not None:
            for key, val in self._map_data["pt_token"].items():
                if key == "position":
                    val = val.clone()
                    val[:, :2] -= center_torch
                data["pt_token"][key] = val

            if "map_save" in self._map_data:
                traj_pos = self._map_data["map_save"]["traj_pos"].clone()
                traj_pos -= center_torch.unsqueeze(0)
                data["pt_token"]["traj_pos"] = traj_pos

            if "token2pl_edge_index" in self._map_data:
                data["pt_token"]["polygon_idx"] = self._map_data["token2pl_edge_index"][1]

        return data

    def _positions_to_trajectory(
        self, ego_state: EgoState, positions: np.ndarray, headings: np.ndarray
    ) -> InterpolatedTrajectory:
        """Convert predicted (x, y) positions and headings to nuPlan trajectory."""
        rear_axle_to_center = _VEHICLE_PARAMS.rear_axle_to_center
        start_time_us = ego_state.time_us
        dt = self._trajectory_dt

        ego_states = [ego_state]

        for i in range(min(len(positions), self._trajectory_steps - 1)):
            t_x = float(positions[i, 0])
            t_y = float(positions[i, 1])
            t_heading = float(headings[i])
            cos_h = math.cos(t_heading)
            sin_h = math.sin(t_heading)

            # Compute velocity from position diff
            if i == 0:
                prev_x, prev_y = ego_state.center.x, ego_state.center.y
            else:
                prev_x, prev_y = float(positions[i - 1, 0]), float(positions[i - 1, 1])
            vx = (t_x - prev_x) / dt
            vy = (t_y - prev_y) / dt

            # Center to rear axle
            ra_x = t_x - rear_axle_to_center * cos_h
            ra_y = t_y - rear_axle_to_center * sin_h

            # Global to body velocity
            vx_body = vx * cos_h + vy * sin_h
            vy_body = -vx * sin_h + vy * cos_h

            time_us = start_time_us + int((i + 1) * dt * 1e6)

            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(ra_x, ra_y, t_heading),
                rear_axle_velocity_2d=StateVector2D(vx_body, vy_body),
                rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
                tire_steering_angle=0.0,
                time_point=TimePoint(time_us),
                vehicle_parameters=_VEHICLE_PARAMS,
            )
            ego_states.append(state)

        return InterpolatedTrajectory(ego_states)

    def _constant_velocity_trajectory(
        self, ego_state: EgoState
    ) -> InterpolatedTrajectory:
        """Fallback: constant velocity trajectory during history bootstrap."""
        from .trajectory_filler import build_trajectory
        return build_trajectory(ego_state, 0.0, 0.0, self._trajectory_steps, self._trajectory_dt)
