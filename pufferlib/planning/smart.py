# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""SMART prediction model as a planner.

Uses the SMART trajectory prediction model to predict future positions
for all agents, then converts predictions to (acceleration, steering)
actions via a proportional controller.

Can be used as ego planner or traffic agent controller.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, HeteroData

from pufferlib.planning.base import BasePlanner
from pufferlib.prediction.smart.prediction_model import PredictionModel
from pufferlib.prediction.trajectory_tokenizer import (
    compute_token_data,
    load_motion_codebook,
)
from pufferlib.prediction.map_tokenizer import tokenize_roads, load_map_codebook
from pufferlib.prediction.binary_reader import read_binary_scenario

log = logging.getLogger(__name__)


def load_smart_model(weights_path, device="cuda"):
    """Load SMART model and codebooks once. Returns (model, motion_codebook_data, map_codebook)."""
    device = device if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if weights_path:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_cfg = ckpt.get("config", {})
        # Config may be nested with 'model' key (from .ini) or flat
        if 'model' in model_cfg and isinstance(model_cfg['model'], dict):
            model_cfg = model_cfg['model']
        model = PredictionModel(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded SMART model from %s (epoch %d)", weights_path, ckpt.get("epoch", -1))
    else:
        model = PredictionModel({})
        log.info("Created SMART model with random weights (no checkpoint)")

    model.eval()
    model.to(device)

    motion_codebook_data = load_motion_codebook()
    map_codebook = load_map_codebook()
    return model, motion_codebook_data, map_codebook


@dataclass
class SMARTConfig:
    """Configuration for SMART planner."""

    weights_path: str = ""
    device: str = "cuda"
    temperature: float = 1.0
    greedy: bool = False
    repredict_interval: int = 5  # Re-run inference every N steps (=shift for autoregressive)
    num_historical_steps: int = 11
    shift: int = 5
    # Proportional controller gains
    k_accel: float = 1.0
    k_steer: float = 2.0
    max_accel: float = 4.0  # m/s^2 (matches action space bounds)
    dt: float = 0.1  # Timestep in seconds


class HistoryBuffer:
    """Accumulates agent states over time for SMART inference."""

    def __init__(self, max_agents: int, max_steps: int):
        self.max_agents = max_agents
        self.max_steps = max_steps
        self.position = np.zeros((max_agents, max_steps, 2), dtype=np.float32)
        self.heading = np.zeros((max_agents, max_steps), dtype=np.float32)
        self.velocity = np.zeros((max_agents, max_steps, 2), dtype=np.float32)
        self.valid = np.zeros((max_agents, max_steps), dtype=bool)
        self.width = np.zeros(max_agents, dtype=np.float32)
        self.length = np.zeros(max_agents, dtype=np.float32)
        self.agent_type = np.zeros(max_agents, dtype=np.int32)  # raw entity type (1=veh, 2=ped, 3=cyc)
        self.num_agents = 0
        self.num_steps = 0

    def clear(self):
        self.position[:] = 0
        self.heading[:] = 0
        self.velocity[:] = 0
        self.valid[:] = False
        self.agent_type[:] = 0
        self.num_agents = 0
        self.num_steps = 0

    def append(self, x, y, heading, width, length, num_agents, agent_type=None):
        """Append current agent states to history."""
        t = self.num_steps
        if t >= self.max_steps:
            return
        n = min(num_agents, self.max_agents)
        self.num_agents = max(self.num_agents, n)

        self.position[:n, t, 0] = x[:n]
        self.position[:n, t, 1] = y[:n]
        self.heading[:n, t] = heading[:n]
        # Only mark agents with valid positions and non-zero dimensions
        # (removed agents have x=-10000, inactive slots have width/length=0)
        self.valid[:n, t] = (
            (x[:n] > -9000) & (y[:n] > -9000)
            & (width[:n] > 0) & (length[:n] > 0)
        )
        self.width[:n] = width[:n]
        self.length[:n] = length[:n]
        if agent_type is not None:
            self.agent_type[:n] = agent_type[:n]

        # Compute velocity from position differences (only for valid agents)
        if t > 0:
            dt = 0.1
            valid_now = self.valid[:n, t] & self.valid[:n, t - 1]
            dx = self.position[:n, t, 0] - self.position[:n, t - 1, 0]
            dy = self.position[:n, t, 1] - self.position[:n, t - 1, 1]
            self.velocity[:n, t, 0] = np.where(valid_now, dx / dt, 0.0)
            self.velocity[:n, t, 1] = np.where(valid_now, dy / dt, 0.0)
        self.num_steps = t + 1


class SMARTPlanner(BasePlanner):
    """Planner using SMART prediction model.

    Predicts future trajectories for all agents using the SMART model,
    then converts predicted positions to actions via proportional control.
    """

    def __init__(
        self,
        env,
        agent_idx: int,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        config: SMARTConfig,
        model=None,
        motion_codebook_data=None,
        map_codebook=None,
        map_env_index: int = 0,
    ):
        super().__init__(
            horizon=config.num_historical_steps,
            action_dim=len(action_lb),
            action_lb=action_lb,
            action_ub=action_ub,
        )
        self.env = env
        self.agent_idx = agent_idx
        self.config = config
        self.neutral_action = np.array([0.0, 0.0], dtype=np.float32)
        self.map_env_index = map_env_index

        if model is not None:
            self.model = model
            self.device = next(model.parameters()).device
        else:
            device = config.device if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            ckpt = torch.load(config.weights_path, map_location="cpu", weights_only=False)
            model_cfg = ckpt.get("config", {})
            if 'model' in model_cfg and isinstance(model_cfg['model'], dict):
                model_cfg = model_cfg['model']
            self.model = PredictionModel(model_cfg)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.model.to(self.device)
            log.info("Loaded SMART model from %s (epoch %d)",
                     config.weights_path, ckpt.get("epoch", -1))

        if motion_codebook_data is not None:
            self.motion_codebook_data = motion_codebook_data
            self.map_codebook = map_codebook
        else:
            self.motion_codebook_data = load_motion_codebook()
            self.map_codebook = load_map_codebook()
        self.motion_codebook_veh = self.motion_codebook_data["token"]["veh"]

        # State
        self.history = HistoryBuffer(
            max_agents=128,
            max_steps=200,  # enough for full sim (91 steps) with margin
        )
        self.map_data = None  # Cached tokenized roads
        self._predicted_positions = None  # (N, T_future, 2)
        self._predicted_headings = None  # (N, T_future)
        self._prediction_step = -1
        self._center_pos = None  # Scene center used for centering/un-centering
        self._valid_agent_map = None  # Mapping: filtered idx -> full idx
        self._map_loaded = False

    def reset(self):
        """Reset state for new episode."""
        self.history.clear()
        self._predicted_positions = None
        self._predicted_headings = None
        self._prediction_step = -1
        self._center_pos = None
        self._valid_agent_map = None
        self._map_loaded = False
        self.map_data = None

    def _load_map(self):
        """Load and tokenize map data from .bin file."""
        if self._map_loaded:
            return

        env = self.env
        idx = self.map_env_index
        map_id = env.map_ids[idx] if hasattr(env, "map_ids") and len(env.map_ids) > idx else 0
        bin_path = os.path.join(env.data_root, env.split, f"map_{map_id:06d}.bin")

        if not os.path.isfile(bin_path):
            log.warning("Map .bin file not found: %s", bin_path)
            self._map_loaded = True
            return

        scenario = read_binary_scenario(bin_path)
        self.map_data = tokenize_roads(scenario["roads"], self.map_codebook)
        self._scenario_objects = scenario["objects"]  # cache for history bootstrap
        self._map_loaded = True
        log.debug("Loaded map tokens from %s", bin_path)

    def _bootstrap_speeds_from_state(self, cur_speeds, num_agents):
        """Fill cur_speeds for all agents from environment state (vx, vy).

        This is called once at bootstrap to get accurate initial speeds for
        all agents, not just the ego. Without this, backward extrapolation
        creates static histories for non-ego agents.
        """
        state = self.env.get_state()
        if isinstance(state, list):
            state = state[self.map_env_index] if state else {}
        entities = state.get("entities", [])
        active = state.get("active_agent_indices", [])
        for i, entity_idx in enumerate(active):
            if i >= num_agents:
                break
            if entity_idx < len(entities):
                e = entities[entity_idx]
                vx = e.get("vx", 0.0)
                vy = e.get("vy", 0.0)
                cur_speeds[i] = np.sqrt(vx**2 + vy**2)

    def _get_active_agent_indices(self):
        """Get the active_agent_indices mapping from the environment.

        Returns list of entity indices, where position i in the returned list
        corresponds to position i in _get_agent_states() output.
        """
        state = self.env.get_state()
        if isinstance(state, list):
            state = state[self.map_env_index] if state else {}
        return state.get("active_agent_indices", [])

    def _load_ground_truth_history(self):
        """Pre-fill history buffer with ground-truth historical data from .bin file.

        This breaks the closed-loop cold-start problem: without real historical
        velocities, SMART predicts stationary agents → controller follows →
        agents stay still. Ground-truth history gives SMART realistic motion
        context from step 0.

        Reorders .bin data to match active_agent_indices order (same order as
        _get_agent_states() returns).
        """
        if not hasattr(self, '_scenario_objects') or self._scenario_objects is None:
            return

        objects = self._scenario_objects
        T_hist = self.config.num_historical_steps  # 11

        # Get active agent mapping: position-in-active -> entity index (= .bin index)
        active_indices = self._get_active_agent_indices()
        if not active_indices:
            log.warning("No active_agent_indices, falling back to .bin order")
            active_indices = list(range(len(objects)))

        num_agents = len(active_indices)
        num_objects = len(objects)

        for t in range(T_hist):
            x = np.full(num_agents, -10000.0, dtype=np.float32)
            y = np.full(num_agents, -10000.0, dtype=np.float32)
            h = np.zeros(num_agents, dtype=np.float32)
            w = np.zeros(num_agents, dtype=np.float32)
            l = np.zeros(num_agents, dtype=np.float32)

            for i, entity_idx in enumerate(active_indices):
                if entity_idx >= num_objects:
                    continue
                obj = objects[entity_idx]
                if t < len(obj['traj_x']) and obj['traj_valid'][t]:
                    x[i] = obj['traj_x'][t]
                    y[i] = obj['traj_y'][t]
                    h[i] = obj['traj_heading'][t]
                w[i] = obj['width']
                l[i] = obj['length']

            self.history.append(x, y, h, w, l, num_agents)

        log.info("Bootstrapped GT history: %d steps, %d agents (reordered by active_agent_indices)",
                 T_hist, num_agents)

    def _load_ground_truth_predictions(self):
        """Load GT future trajectory (bin steps 11-90) as predictions for warm-up period."""
        if not hasattr(self, '_scenario_objects') or self._scenario_objects is None:
            return

        objects = self._scenario_objects
        T_hist = self.config.num_historical_steps  # 11
        T_future = 80  # steps 11..90
        num_agents = min(len(objects), self.history.max_agents)

        positions = np.full((num_agents, T_future, 2), -10000.0, dtype=np.float32)
        headings = np.zeros((num_agents, T_future), dtype=np.float32)
        for j in range(num_agents):
            obj = objects[j]
            last_valid_pos = None
            last_valid_head = 0.0
            # Find last valid historical position as fallback
            for t in range(T_hist - 1, -1, -1):
                if obj['traj_valid'][t]:
                    last_valid_pos = np.array([obj['traj_x'][t], obj['traj_y'][t]])
                    last_valid_head = obj['traj_heading'][t]
                    break
            for t in range(T_future):
                bin_t = T_hist + t  # 11..90
                if bin_t < len(obj['traj_x']) and obj['traj_valid'][bin_t]:
                    positions[j, t, 0] = obj['traj_x'][bin_t]
                    positions[j, t, 1] = obj['traj_y'][bin_t]
                    headings[j, t] = obj['traj_heading'][bin_t]
                    last_valid_pos = positions[j, t].copy()
                    last_valid_head = headings[j, t]
                elif last_valid_pos is not None:
                    # Forward-fill: hold last valid position (agent stops)
                    positions[j, t] = last_valid_pos
                    headings[j, t] = last_valid_head

        self._predicted_positions = positions
        self._predicted_headings = headings
        log.debug("Loaded GT future predictions for %d agents", num_agents)

    def _get_agent_states(self):
        """Get current agent states from environment (for this map's agents only)."""
        env = self.env
        total_agents = env.num_agents
        x = np.zeros(total_agents, dtype=np.float32)
        y = np.zeros(total_agents, dtype=np.float32)
        z = np.zeros(total_agents, dtype=np.float32)
        heading = np.zeros(total_agents, dtype=np.float32)
        ids = np.zeros(total_agents, dtype=np.int32)
        length = np.zeros(total_agents, dtype=np.float32)
        width = np.zeros(total_agents, dtype=np.float32)
        agent_type = np.zeros(total_agents, dtype=np.int32)

        from pufferlib.ocean.drive import binding
        binding.vec_get_global_agent_state(
            env.c_envs, x, y, z, heading, ids, length, width, agent_type)

        # Slice to this map's agents
        idx = self.map_env_index
        cur = env.agent_offsets[idx]
        nxt = env.agent_offsets[idx + 1]
        num_agents = nxt - cur
        if not hasattr(self, '_agent_state_call_count'):
            self._agent_state_call_count = 0
        self._agent_state_call_count += 1
        if self._agent_state_call_count <= 3:
            log.info("_get_agent_states[%d]: total_agents=%d, offsets=[%d:%d], num_agents=%d",
                     self._agent_state_call_count, total_agents, cur, nxt, num_agents)
            log.info("  raw x[:5] = %s", x[:5])
            log.info("  sliced x[:5] = %s", x[cur:nxt][:5])
        return (x[cur:nxt], y[cur:nxt], heading[cur:nxt], width[cur:nxt],
                length[cur:nxt], num_agents, agent_type[cur:nxt])

    def _build_hetero_data(self) -> Optional[HeteroData]:
        """Build HeteroData from history buffer + cached map data.

        Only includes agents that were valid in at least one historical step
        (filters out inactive/removed agents). Stores a mapping from filtered
        indices back to full indices in self._valid_agent_map.
        """
        hist = self.history
        N = hist.num_agents
        T_hist = self.config.num_historical_steps
        T_future = 80
        T = T_hist + T_future
        shift = self.config.shift

        if N == 0 or hist.num_steps < T_hist:
            return None

        # Pad history to T_hist if we have fewer steps
        actual_steps = min(hist.num_steps, T_hist)

        # Build full-length arrays (T=91) with history filled, future zeroed
        position_full = np.zeros((N, T, 2), dtype=np.float32)
        heading_full = np.zeros((N, T), dtype=np.float32)
        velocity_full = np.zeros((N, T, 2), dtype=np.float32)
        valid_mask_full = np.zeros((N, T), dtype=bool)

        # Fill historical portion (right-aligned to T_hist)
        start = T_hist - actual_steps
        src_start = hist.num_steps - actual_steps
        position_full[:N, start:T_hist] = hist.position[:N, src_start:hist.num_steps]
        heading_full[:N, start:T_hist] = hist.heading[:N, src_start:hist.num_steps]
        velocity_full[:N, start:T_hist] = hist.velocity[:N, src_start:hist.num_steps]
        valid_mask_full[:N, start:T_hist] = hist.valid[:N, src_start:hist.num_steps]

        # Filter to only agents that are currently valid (last historical step)
        # This excludes removed/despawned agents whose ghost positions would
        # cause SMART to predict phantom trajectories
        currently_valid = valid_mask_full[:, T_hist - 1].copy()  # (N,)
        # Always include ego
        ego_idx_full = self.agent_idx if self.agent_idx < N else 0
        currently_valid[ego_idx_full] = True
        valid_indices = np.where(currently_valid)[0]  # indices into full array
        M = len(valid_indices)  # number of active agents

        if M == 0:
            return None

        # Store mapping: filtered index -> full index
        self._valid_agent_map = valid_indices
        # Ego index in filtered array
        ego_idx = int(np.where(valid_indices == ego_idx_full)[0][0])

        log.debug("_build_hetero_data: %d/%d agents active, ego %d->%d",
                  M, N, ego_idx_full, ego_idx)

        # Slice to active agents only
        position = position_full[valid_indices]
        heading = heading_full[valid_indices]
        velocity = velocity_full[valid_indices]
        valid_mask = valid_mask_full[valid_indices]

        # Fill shape (vectorized, constant over time)
        shape_arr = np.zeros((M, T, 3), dtype=np.float32)
        shape_arr[:, :, 0] = hist.width[valid_indices, np.newaxis]
        shape_arr[:, :, 1] = hist.length[valid_indices, np.newaxis]

        # Fill invalid positions with nearest valid (vectorized forward-fill)
        t_indices = np.arange(T)
        for i in range(M):
            v = valid_mask[i]
            if not v.any():
                continue
            first_valid = int(np.argmax(v))
            # Back-fill before first valid
            if first_valid > 0:
                position[i, :first_valid] = position[i, first_valid]
                heading[i, :first_valid] = heading[i, first_valid]
                velocity[i, :first_valid] = velocity[i, first_valid]
            # Forward-fill: build index array where each invalid step maps to
            # its nearest preceding valid step
            fill_idx = np.maximum.accumulate(np.where(v, t_indices, 0))
            position[i] = position[i, fill_idx]
            heading[i] = heading[i, fill_idx]
            velocity[i] = velocity[i, fill_idx]

        # Heading cleaning (matching dataset.py / original SMART):
        # If heading jumps by more than 1.0 rad between consecutive valid
        # frames, copy previous heading forward.
        for i in range(M):
            v = valid_mask[i]
            if not v.any():
                continue
            valid_indices_i = np.where(v)[0]
            prev_t = valid_indices_i[0]
            for t in valid_indices_i[1:]:
                hdiff = abs(heading[i, t] - heading[i, prev_t])
                hdiff = min(hdiff, 2 * np.pi - hdiff)
                if hdiff > 1.0:
                    heading[i, t] = heading[i, prev_t]
                prev_t = t

        # Scene center: ego at last historical step
        center_pos = position[ego_idx, T_hist - 1].copy()
        self._center_pos = center_pos.copy()

        # Compute motion tokens for HISTORICAL portion only (2 tokens, not 18).
        # The model generates future tokens autoregressively during inference.
        # Map raw entity types (1=veh, 2=ped, 3=cyc) to SMART indices (0, 1, 2)
        smart_types = np.clip(hist.agent_type[valid_indices] - 1, 0, 2)
        num_total_tokens = (T - 1) // shift  # 18
        token_data = compute_token_data(
            positions=position[:, :T_hist],
            headings=heading[:, :T_hist],
            valid_masks=valid_mask[:, :T_hist],
            shift=shift,
            codebook=self.motion_codebook_veh,
            agent_types=smart_types,
        )
        num_hist_tokens = token_data['token_idx'].shape[1]  # 2
        if num_hist_tokens < num_total_tokens:
            pad = torch.zeros(M, num_total_tokens - num_hist_tokens, dtype=torch.long)
            token_data['token_idx'] = torch.cat([token_data['token_idx'], pad], dim=1)

        # Center positions
        position -= center_pos[np.newaxis, np.newaxis, :]

        # Token positions/headings at shift boundaries
        shift_indices = list(range(shift, T, shift))
        num_tokens = len(shift_indices)
        # Use centered GT as fallback for future tokens (model zeros them anyway)
        token_pos = torch.from_numpy(position[:, shift_indices, :]).float()
        token_heading = torch.from_numpy(heading[:, shift_indices]).float()
        # Override historical tokens with matched codebook values (matching dataset.py:211-212)
        center_pos_torch = torch.from_numpy(center_pos).float()
        num_ht = min(token_data['token_pos'].shape[1], num_tokens)
        # Fall back to raw position when codebook quantization error is too large.
        # This prevents the model from starting predictions from a position
        # several meters away from the agent's actual location (happens when
        # heading diverges from direction of motion during initial steps).
        max_quant_err = 2.0  # meters
        for i in range(M):
            for ht_idx in range(num_ht):
                raw_global = position_full[valid_indices[i], shift_indices[ht_idx]]
                matched_global = token_data['token_pos'][i, ht_idx].numpy()
                err = np.sqrt(np.sum((raw_global - matched_global)**2))
                if err > max_quant_err:
                    token_data['token_pos'][i, ht_idx] = torch.tensor(
                        raw_global, dtype=torch.float32)
                    token_data['token_heading'][i, ht_idx] = float(
                        heading_full[valid_indices[i], shift_indices[ht_idx]])
                    log.debug("codebook fallback agent %d token %d: err=%.1fm", i, ht_idx, err)
        token_pos[:, :num_ht] = token_data['token_pos'] - center_pos_torch
        token_heading[:, :num_ht] = token_data['token_heading']

        # Token validity (vectorized)
        vm = valid_mask[:, shift_indices]
        vm_prev = valid_mask[:, [max(0, s - shift) for s in shift_indices]]
        token_valid = torch.from_numpy(vm & vm_prev)

        # Category: 3 = predict, 5 = SDC (ego agent)
        category = torch.full((M,), 3, dtype=torch.long)
        category[ego_idx] = 5

        # Build HeteroData
        data = HeteroData()
        data["agent"]["position"] = torch.from_numpy(position).float()
        data["agent"]["heading"] = torch.from_numpy(heading).float()
        data["agent"]["velocity"] = torch.from_numpy(velocity).float()
        data["agent"]["valid_mask"] = torch.from_numpy(valid_mask).bool()
        data["agent"]["type"] = torch.from_numpy(smart_types).long()
        data["agent"]["shape"] = torch.from_numpy(shape_arr).float()
        data["agent"]["category"] = category
        data["agent"]["num_nodes"] = M
        data["agent"]["av_index"] = ego_idx
        data["agent"]["token_pos"] = token_pos
        data["agent"]["token_heading"] = token_heading
        data["agent"]["token_idx"] = token_data["token_idx"]
        data["agent"]["agent_valid_mask"] = token_valid

        # Map tokens
        if self.map_data is not None:
            center_torch = torch.from_numpy(center_pos).float()
            for key, val in self.map_data["pt_token"].items():
                if key == "position":
                    val = val.clone()
                    val[:, :2] -= center_torch
                data["pt_token"][key] = val

            if "map_save" in self.map_data:
                traj_pos = self.map_data["map_save"]["traj_pos"].clone()
                traj_pos -= center_torch.unsqueeze(0)
                data["pt_token"]["traj_pos"] = traj_pos

            if "token2pl_edge_index" in self.map_data:
                data["pt_token"]["polygon_idx"] = self.map_data["token2pl_edge_index"][1]

        return data

    def prepare_step(self, current_step: int) -> bool:
        """Prepare for inference: load map, get states, append to history.

        Returns True if this planner needs a new prediction this step.
        """
        self._load_map()
        if self.history.num_steps == 0:
            self._load_ground_truth_history()
        x, y, heading, width, length, num_agents, agent_type = self._get_agent_states()
        self.history.append(x, y, heading, width, length, num_agents, agent_type)

        should_predict = (
            self._predicted_positions is None
            or (current_step - self._prediction_step) >= self.config.repredict_interval
        )
        return should_predict and self.history.num_steps >= self.config.num_historical_steps

    def inject_predictions(self, pred_traj: np.ndarray, pred_head: np.ndarray,
                           current_step: int):
        """Inject externally-computed predictions (from batch inference).

        Args:
            pred_traj: (M, 80, 2) predicted positions in centered coords (filtered agents)
            pred_head: (M, 80) predicted headings (filtered agents)
            current_step: current simulation step
        """
        filtered_positions = pred_traj.copy()
        filtered_headings = pred_head.copy()
        filtered_positions += self._center_pos[np.newaxis, np.newaxis, :]

        # Map filtered predictions back to full agent indices.
        # Initialize with -10000 so the P-controller skips agents without predictions.
        N_full = self.history.num_agents
        num_steps = filtered_positions.shape[1]
        self._predicted_positions = np.full((N_full, num_steps, 2), -10000.0, dtype=np.float32)
        self._predicted_headings = np.zeros((N_full, num_steps), dtype=np.float32)
        if hasattr(self, '_valid_agent_map') and self._valid_agent_map is not None:
            for fi, full_i in enumerate(self._valid_agent_map):
                if full_i < N_full:
                    self._predicted_positions[full_i] = filtered_positions[fi]
                    self._predicted_headings[full_i] = filtered_headings[fi]
        else:
            n = min(filtered_positions.shape[0], N_full)
            self._predicted_positions[:n] = filtered_positions[:n]
            self._predicted_headings[:n] = filtered_headings[:n]
        self._prediction_step = current_step

    def _run_inference(self):
        """Run SMART inference for one token (repredict_interval steps).

        Only generates enough tokens to cover repredict_interval steps,
        enabling true autoregressive execution: predict → execute → feed back
        real positions → predict next token.
        """
        data = self._build_hetero_data()
        if data is None:
            return

        data = data.to(self.device)
        with torch.no_grad():
            result = self.model.inference(
                data,
                greedy=self.config.greedy,
                temperature=self.config.temperature,
                max_steps=self.config.repredict_interval,
            )

        # pred_traj is always (M, 80, 2) but only first num_generated steps are valid
        import math
        num_generated = min(
            math.ceil(self.config.repredict_interval / self.config.shift) * self.config.shift,
            result["pred_traj"].shape[1],
        )
        filtered_positions = result["pred_traj"][:, :num_generated].cpu().numpy()
        filtered_headings = result["pred_head"][:, :num_generated].cpu().numpy()

        # Un-center predictions
        filtered_positions += self._center_pos[np.newaxis, np.newaxis, :]

        # Map filtered predictions back to full agent indices.
        # Initialize with -10000 so the P-controller skips agents without predictions.
        N_full = self.history.num_agents
        self._predicted_positions = np.full((N_full, num_generated, 2), -10000.0, dtype=np.float32)
        self._predicted_headings = np.zeros((N_full, num_generated), dtype=np.float32)
        if hasattr(self, '_valid_agent_map') and self._valid_agent_map is not None:
            for fi, full_i in enumerate(self._valid_agent_map):
                if full_i < N_full:
                    self._predicted_positions[full_i] = filtered_positions[fi]
                    self._predicted_headings[full_i] = filtered_headings[fi]
        else:
            n = min(filtered_positions.shape[0], N_full)
            self._predicted_positions[:n] = filtered_positions[:n]
            self._predicted_headings[:n] = filtered_headings[:n]

        ego_pred = self._predicted_positions[self.agent_idx]
        ego_cur = self.history.position[self.agent_idx, self.history.num_steps - 1]
        log.info("inference: %d active, ego_idx=%d, ego_cur=(%.1f,%.1f), ego_pred[0]=(%.1f,%.1f), ego_pred[-1]=(%.1f,%.1f)",
                 len(self._valid_agent_map) if hasattr(self, '_valid_agent_map') else N_full,
                 self.agent_idx, ego_cur[0], ego_cur[1],
                 ego_pred[0, 0], ego_pred[0, 1], ego_pred[-1, 0], ego_pred[-1, 1])

    def _positions_to_actions(
        self,
        agent_indices: np.ndarray,
        current_step: int,
        cur_positions: np.ndarray,
        cur_headings: np.ndarray,
        cur_speeds: np.ndarray,
    ) -> np.ndarray:
        """Convert predicted positions to (acceleration, steering) actions.

        Uses a proportional controller to track predicted positions.

        Args:
            agent_indices: (n,) which agents to control
            current_step: current simulation step
            cur_positions: (num_agents, 2) fresh positions from _get_agent_states
            cur_headings: (num_agents,) fresh headings from _get_agent_states
            cur_speeds: (num_agents,) speeds from obs[:, 2] * MAX_SPEED
        """
        n = len(agent_indices)
        actions = np.zeros((n, 2), dtype=np.float32)

        if self._predicted_positions is None:
            return actions

        # Which sub-step of the prediction to use
        steps_since_prediction = current_step - self._prediction_step
        pred_idx = max(0, min(steps_since_prediction, self._predicted_positions.shape[1] - 1))

        cfg = self.config
        for i, agent_i in enumerate(agent_indices):
            if agent_i >= self._predicted_positions.shape[0]:
                continue

            cur_x = cur_positions[agent_i, 0]
            cur_y = cur_positions[agent_i, 1]
            cur_h = cur_headings[agent_i]
            cur_spd = cur_speeds[agent_i]

            # Target position from prediction (skip invalid targets)
            tgt_x = self._predicted_positions[agent_i, pred_idx, 0]
            tgt_y = self._predicted_positions[agent_i, pred_idx, 1]
            if tgt_x < -9000 or tgt_y < -9000:
                continue

            dx = tgt_x - cur_x
            dy = tgt_y - cur_y
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 1e-4:
                continue

            # Heading
            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - cur_h
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            steering = np.clip(heading_error * 2.0, -1.0, 1.0)

            # Acceleration (speed_error / dt / max_accel)
            desired_speed = dist / cfg.dt
            speed_error = desired_speed - cur_spd
            required_accel = speed_error / cfg.dt  # m/s²
            acceleration = np.clip(required_accel / cfg.max_accel, -1.0, 1.0)

            actions[i, 0] = acceleration
            actions[i, 1] = steering

        return actions

    def plan(
        self,
        current_step: int = 0,
        obs: Optional[np.ndarray] = None,
        extract_trajectories: bool = False,
    ) -> np.ndarray:
        """Plan action(s) using SMART predictions.

        Supports both single (ego) and batch (traffic) modes:
        - Single obs (obs_dim,) -> returns (2,)
        - Batch obs (N, obs_dim) -> returns (N, 2)
        """
        # Lazy load map (also caches scenario objects for history bootstrap)
        self._load_map()

        # Get agent states from simulator
        x, y, heading, width, length, num_agents, agent_type = self._get_agent_states()
        cur_x, cur_y = x[:num_agents], y[:num_agents]
        cur_h = heading[:num_agents]
        cur_w, cur_l = width[:num_agents], length[:num_agents]

        # Get initial speeds from obs (needed for backward-extrapolation bootstrap)
        cur_speeds = np.zeros(num_agents, dtype=np.float32)
        if obs is not None:
            obs_2d = np.atleast_2d(obs)
            if obs.ndim >= 2:
                # Traffic mode: obs excludes ego (self.agent_idx).
                # Map obs rows back to internal agent indices, skipping ego.
                obs_row = 0
                for ai in range(num_agents):
                    if ai == self.agent_idx:
                        continue
                    if obs_row < obs_2d.shape[0]:
                        cur_speeds[ai] = obs_2d[obs_row, 2] * 100.0
                    obs_row += 1
            else:
                # Ego mode: obs[2] is ego's own speed
                cur_speeds[self.agent_idx] = obs_2d[0, 2] * 100.0
        # Override with history-based speeds when available (all agents, more accurate)
        if self.history.num_steps >= 2:
            t = self.history.num_steps - 1
            dx = self.history.position[:num_agents, t, 0] - self.history.position[:num_agents, t - 1, 0]
            dy = self.history.position[:num_agents, t, 1] - self.history.position[:num_agents, t - 1, 1]
            cur_speeds[:num_agents] = np.sqrt(dx**2 + dy**2) / self.config.dt

        # Bootstrap history on first call via backward-extrapolation.
        # Get all agent speeds from environment state (not just obs) for accuracy.
        if self.history.num_steps == 0:
            self._bootstrap_speeds_from_state(cur_speeds, num_agents)
            T_hist = self.config.num_historical_steps
            # Extrapolate T_hist-1 steps back (not T_hist) to avoid duplicating
            # the current position. The real append below fills the last slot,
            # giving T_hist total entries with proper velocity at the transition.
            for t_idx in range(T_hist - 1):
                dt_back = (T_hist - 1 - t_idx) * self.config.dt
                hist_x = cur_x - cur_speeds[:num_agents] * np.cos(cur_h) * dt_back
                hist_y = cur_y - cur_speeds[:num_agents] * np.sin(cur_h) * dt_back
                self.history.append(hist_x, hist_y, cur_h, cur_w, cur_l, num_agents, agent_type)

        # Append current agent states to history (accumulate real trajectory)
        self.history.append(cur_x, cur_y, cur_h, cur_w, cur_l, num_agents, agent_type)

        # Debug: log agent positions vs predictions for first few agents
        if current_step > 0 and self._predicted_positions is not None:
            steps_since = current_step - self._prediction_step
            pidx = max(0, min(steps_since, self._predicted_positions.shape[1] - 1))
            for dbg_i in range(min(3, num_agents)):
                px, py = cur_x[dbg_i], cur_y[dbg_i]
                if dbg_i < self._predicted_positions.shape[0]:
                    tx, ty = self._predicted_positions[dbg_i, pidx, 0], self._predicted_positions[dbg_i, pidx, 1]
                    err = np.sqrt((px - tx)**2 + (py - ty)**2)
                    log.info("step %d agent %d: pos=(%.1f,%.1f) pred_target=(%.1f,%.1f) err=%.2fm pidx=%d",
                             current_step, dbg_i, px, py, tx, ty, err, pidx)

        # Re-predict if needed (autoregressive: every repredict_interval steps)
        should_predict = (
            self._predicted_positions is None
            or (current_step - self._prediction_step) >= self.config.repredict_interval
        )

        if should_predict and self.history.num_steps >= self.config.num_historical_steps:
            self._run_inference()
            self._prediction_step = current_step

        # Build fresh state arrays for P-controller
        cur_positions = np.stack([cur_x, cur_y], axis=-1)  # (M, 2)
        cur_headings = cur_h

        # Determine which agents we control
        if obs is not None and obs.ndim >= 2:
            # Batch mode (traffic controller): all agents except ego
            agent_indices = np.array(
                [i for i in range(num_agents) if i != self.agent_idx],
                dtype=np.int32)
            return self._positions_to_actions(agent_indices, current_step,
                                              cur_positions, cur_headings, cur_speeds)
        else:
            # Single mode (ego): agent 0
            agent_indices = np.array([self.agent_idx], dtype=np.int32)
            actions = self._positions_to_actions(agent_indices, current_step,
                                                 cur_positions, cur_headings, cur_speeds)
            return actions[0]

    def plot(self, ax, state, axis_limits=None):
        """Plot predicted trajectories."""
        if self._predicted_positions is None:
            return
        N = self._predicted_positions.shape[0]
        t = self.history.num_steps - 1
        if t < 0:
            return

        # Compute global→local offset using ego agent's current position.
        # Use the position from when predictions were made (_prediction_step)
        # to stay consistent with the prediction coordinate frame.
        entities = state.get("entities", [])
        active = state.get("active_agent_indices", [])
        if not active or self.agent_idx >= len(active) or active[self.agent_idx] >= len(entities):
            return
        ego_entity = entities[active[self.agent_idx]]
        local_x = ego_entity.get("x", 0)
        local_y = ego_entity.get("y", 0)
        global_x = self.history.position[self.agent_idx, t, 0]
        global_y = self.history.position[self.agent_idx, t, 1]
        offset = np.array([global_x - local_x, global_y - local_y], dtype=np.float32)

        # Plot predicted trajectories
        H_full = self._predicted_positions.shape[1]
        H_other = min(self.config.repredict_interval, H_full)

        for i in range(N):
            if not self.history.valid[i, t]:
                continue

            if i == self.agent_idx:
                # Ego trajectory: plot full prediction in blue
                traj = self._predicted_positions[i, :H_full] - offset
                ax.plot(traj[:, 0], traj[:, 1], "b-o", linewidth=2.5, alpha=0.9, markersize=4, zorder=95)
            else:
                traj = self._predicted_positions[i, :H_other] - offset
                # Skip if trajectory start is outside visible area
                if axis_limits is not None:
                    xmin, xmax, ymin, ymax = axis_limits
                    if traj[0, 0] < xmin or traj[0, 0] > xmax or traj[0, 1] < ymin or traj[0, 1] > ymax:
                        continue
                ax.plot(traj[:, 0], traj[:, 1], "g-", linewidth=1.5, alpha=0.7, zorder=90)

    @property
    def population_size(self) -> int:
        return 1

    @property
    def supports_trajectory_proposals(self) -> bool:
        return False


class BatchSMARTController:
    """Coordinates batched SMART inference across multiple maps.

    Each map has its own SMARTPlanner (with HistoryBuffer and map data).
    This controller collects their HeteroData, batches them via PyG
    Batch.from_data_list(), runs a single model.inference(), and
    distributes results back.
    """

    def __init__(self, planners, model, device, greedy=False, temperature=1.0,
                 inference_chunk_size=50):
        self.planners = planners
        self.model = model
        self.device = device
        self.greedy = greedy
        self.temperature = temperature
        self.inference_chunk_size = inference_chunk_size

    def reset(self):
        for p in self.planners:
            p.reset()

    def step(self, current_step, env, obs):
        """Run one batched inference step for all maps.

        Returns:
            actions: np.ndarray (total_agents, 2)
        """
        total_agents = env.num_agents
        actions = np.zeros((total_agents, 2), dtype=np.float32)

        # Phase 1: Prepare all planners (get states, append history)
        ready_indices = []
        data_list = []

        for i, planner in enumerate(self.planners):
            needs_prediction = planner.prepare_step(current_step)
            if needs_prediction:
                data = planner._build_hetero_data()
                if data is not None:
                    ready_indices.append(i)
                    data_list.append(data)

        # Phase 2: Chunked batched inference
        if data_list:
            all_pred_traj = []
            all_pred_head = []
            chunk_size = self.inference_chunk_size

            for chunk_start in range(0, len(data_list), chunk_size):
                chunk = data_list[chunk_start:chunk_start + chunk_size]
                if len(chunk) == 1:
                    batch_data = chunk[0].to(self.device)
                else:
                    batch_data = Batch.from_data_list(chunk).to(self.device)

                with torch.no_grad():
                    repredict = self.planners[0].config.repredict_interval
                    result = self.model.inference(
                        batch_data,
                        greedy=self.greedy,
                        temperature=self.temperature,
                        max_steps=repredict,
                    )

                all_pred_traj.append(result["pred_traj"].cpu().numpy())
                all_pred_head.append(result["pred_head"].cpu().numpy())
                del batch_data, result

            pred_traj = np.concatenate(all_pred_traj) if len(all_pred_traj) > 1 else all_pred_traj[0]
            pred_head = np.concatenate(all_pred_head) if len(all_pred_head) > 1 else all_pred_head[0]

            # Phase 3: Split results back to individual planners
            if len(data_list) == 1:
                self.planners[ready_indices[0]].inject_predictions(
                    pred_traj, pred_head, current_step)
            else:
                agent_counts = []
                for d in data_list:
                    n = d["agent"]["num_nodes"]
                    agent_counts.append(n if isinstance(n, int) else n.item())

                offset = 0
                for idx, count in zip(ready_indices, agent_counts):
                    self.planners[idx].inject_predictions(
                        pred_traj[offset:offset + count],
                        pred_head[offset:offset + count],
                        current_step,
                    )
                    offset += count

        # Phase 4: Convert predictions to actions (per-planner)
        from pufferlib.ocean.drive import binding
        all_x = np.zeros(total_agents, dtype=np.float32)
        all_y = np.zeros(total_agents, dtype=np.float32)
        all_z = np.zeros(total_agents, dtype=np.float32)
        all_h = np.zeros(total_agents, dtype=np.float32)
        all_ids = np.zeros(total_agents, dtype=np.int32)
        all_len = np.zeros(total_agents, dtype=np.float32)
        all_wid = np.zeros(total_agents, dtype=np.float32)
        all_type = np.zeros(total_agents, dtype=np.int32)
        binding.vec_get_global_agent_state(
            env.c_envs, all_x, all_y, all_z, all_h, all_ids, all_len, all_wid, all_type)

        for i, planner in enumerate(self.planners):
            cur = env.agent_offsets[i]
            nxt = env.agent_offsets[i + 1]
            n_agents = nxt - cur
            agent_indices = np.arange(n_agents, dtype=np.int32)
            cur_positions = np.stack([all_x[cur:nxt], all_y[cur:nxt]], axis=-1)
            cur_headings = all_h[cur:nxt]
            # obs not available here; use history velocity as fallback
            cur_speeds = np.zeros(n_agents, dtype=np.float32)
            if obs is not None and obs.ndim >= 2:
                map_obs = obs[cur:nxt]
                cur_speeds = map_obs[:, 2] * 100.0
            map_actions = planner._positions_to_actions(
                agent_indices, current_step, cur_positions, cur_headings, cur_speeds)
            actions[cur:cur + n_agents] = map_actions[:n_agents]

        return actions
