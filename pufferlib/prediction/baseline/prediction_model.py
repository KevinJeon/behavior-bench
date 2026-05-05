# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Lightweight baseline joint prediction model.

GRU agent encoder + MLP map encoder + MLP fusion + joint multi-modal WTA loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from pufferlib.prediction.simpl.utils import group_tokens_to_lanes


class BaselinePredictionModel(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        def _int(key, default):
            v = config.get(key, default)
            return int(v) if v is not None else default

        def _float(key, default):
            v = config.get(key, default)
            return float(v) if v is not None else default

        self._num_historical_steps = _int('num_historical_steps', 11)
        self._num_future_steps = _int('num_future_steps', 80)
        hidden_dim = _int('hidden_dim', 128)
        num_modes = _int('num_modes', 6)
        self.num_modes = num_modes
        self.reg_loss_weight = _float('reg_loss_weight', 1.0)
        self.cls_loss_weight = _float('cls_loss_weight', 0.1)

        T = self._num_future_steps
        in_channels = 6  # dx, dy, cos_h, sin_h, speed, valid
        gru_hidden = 64

        # Agent encoder: GRU over displacement history
        self.agent_gru = nn.GRU(
            input_size=in_channels, hidden_size=gru_hidden,
            num_layers=1, batch_first=False)
        self.agent_proj = nn.Linear(gru_hidden, hidden_dim)

        # Map encoder: MLP + MaxPool (like Drive in torch.py)
        self.map_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Scene-level mode logits: pool agent feats → K logits
        self.mode_head = nn.Linear(hidden_dim, num_modes)

        # Per-agent trajectory head
        self.traj_head = nn.Linear(hidden_dim, num_modes * T * 2)

    @property
    def num_historical_steps(self):
        return self._num_historical_steps

    def _prepare_agent_data(self, data):
        """Extract displacement features, centers, headings, GT from data.

        Same logic as SIMPL's _prepare_agent_data.
        """
        position = data['agent']['position']    # (N, 91, 2)
        heading = data['agent']['heading']       # (N, 91)
        velocity = data['agent']['velocity']     # (N, 91, 2)
        valid_mask = data['agent']['valid_mask'] # (N, 91)

        T_h = self._num_historical_steps
        T_f = self._num_future_steps

        centers = position[:, T_h - 1, :]
        headings_last = heading[:, T_h - 1]

        hist_pos = position[:, :T_h, :]
        hist_head = heading[:, :T_h]
        hist_valid = valid_mask[:, :T_h]

        dx = hist_pos[:, 1:, 0] - hist_pos[:, :-1, 0]
        dy = hist_pos[:, 1:, 1] - hist_pos[:, :-1, 1]
        cos_h = hist_head[:, 1:].cos()
        sin_h = hist_head[:, 1:].sin()
        speed = velocity[:, 1:T_h].norm(dim=-1)
        valid_flag = (hist_valid[:, 1:] & hist_valid[:, :-1]).float()

        disp_features = torch.stack([dx, dy, cos_h, sin_h, speed, valid_flag], dim=1)
        invalid = valid_flag.unsqueeze(1).expand_as(disp_features) == 0
        disp_features = disp_features.masked_fill(invalid, 0.0)

        gt_future = position[:, T_h:T_h + T_f, :] - centers.unsqueeze(1)
        gt_valid = valid_mask[:, T_h:T_h + T_f]

        cos_h_last = headings_last.cos()
        sin_h_last = headings_last.sin()
        gt_x = gt_future[..., 0] * cos_h_last.unsqueeze(1) + gt_future[..., 1] * sin_h_last.unsqueeze(1)
        gt_y = -gt_future[..., 0] * sin_h_last.unsqueeze(1) + gt_future[..., 1] * cos_h_last.unsqueeze(1)
        gt_local = torch.stack([gt_x, gt_y], dim=-1)

        if isinstance(data, Batch) and hasattr(data, '_slice_dict') and 'agent' in data._slice_dict:
            slices = data._slice_dict['agent']['position']
            agent_batch = torch.cat([
                torch.full((slices[i + 1] - slices[i],), i,
                           dtype=torch.long, device=position.device)
                for i in range(len(slices) - 1)
            ])
        else:
            agent_batch = torch.zeros(position.shape[0], dtype=torch.long,
                                      device=position.device)

        return disp_features, centers, headings_last, agent_batch, gt_local, gt_valid

    def forward(self, data) -> dict:
        disp_features, centers, headings, agent_batch, gt_future, gt_valid = \
            self._prepare_agent_data(data)

        N = centers.shape[0]
        K = self.num_modes
        T = self._num_future_steps

        # Agent encoding: GRU
        # disp_features: (N, 6, 10) → transpose to (10, N, 6) for GRU
        gru_input = disp_features.permute(2, 0, 1)  # (10, N, 6)
        _, h_n = self.agent_gru(gru_input)  # h_n: (1, N, 64)
        agent_feats = self.agent_proj(h_n.squeeze(0))  # (N, 128)

        # Map encoding: MLP + MaxPool per lane, then MaxPool per scene
        lane_feats_raw, lane_mask, _, _, lane_batch = \
            group_tokens_to_lanes(data, max_nodes=10)

        # Per-lane: MLP on each node then MaxPool over valid nodes
        N_l, max_nodes, _ = lane_feats_raw.shape
        lane_encoded = self.map_encoder(lane_feats_raw)  # (N_l, max_nodes, 128)
        lane_encoded = lane_encoded.masked_fill(~lane_mask.unsqueeze(-1), float('-inf'))
        lane_feats, _ = lane_encoded.max(dim=1)  # (N_l, 128)
        # Fix lanes with no valid nodes (all -inf)
        all_invalid = ~lane_mask.any(dim=1)
        if all_invalid.any():
            lane_feats[all_invalid] = 0.0

        # MaxPool per scene → map_feat (B, 128)
        batch_ids = agent_batch.unique()
        B = batch_ids.shape[0]
        map_feat = torch.zeros(B, agent_feats.shape[1], device=agent_feats.device)
        for b in range(B):
            l_mask = lane_batch == b
            if l_mask.any():
                map_feat[b], _ = lane_feats[l_mask].max(dim=0)

        # Broadcast map_feat to each agent
        agent_map = map_feat[agent_batch]  # (N, 128)

        # Fusion
        fused = self.fusion(torch.cat([agent_feats, agent_map], dim=-1))  # (N, 128)

        # Scene-level mode logits
        scene_feats = torch.zeros(B, fused.shape[1], device=fused.device)
        for b in range(B):
            a_mask = agent_batch == b
            scene_feats[b], _ = fused[a_mask].max(dim=0)
        scene_logits = self.mode_head(scene_feats)  # (B, K)

        # Per-agent logits: broadcast scene logits to agents
        agent_logits = scene_logits[agent_batch]  # (N, K)

        # Per-agent trajectory prediction
        traj_flat = self.traj_head(fused)  # (N, K*T*2)
        traj = traj_flat.view(N, K, T, 2)

        probs = F.softmax(agent_logits, dim=-1)

        return {
            'traj': traj,
            'probs': probs,
            'logits': agent_logits,
            'scene_logits': scene_logits,
            'gt_future': gt_future,
            'gt_valid': gt_valid,
            'centers': centers,
            'headings': headings,
            'agent_batch': agent_batch,
            'category': data['agent'].get('category', None),
        }

    def compute_loss(self, pred: dict) -> dict:
        """Joint WTA loss: mode selection is per-scene, not per-agent."""
        traj = pred['traj']            # (N, K, T, 2)
        scene_logits = pred['scene_logits']  # (B, K)
        agent_batch = pred['agent_batch']    # (N,)
        gt = pred['gt_future']         # (N, T, 2)
        valid = pred['gt_valid']       # (N, T)
        category = pred.get('category', None)

        N, K, T, _ = traj.shape
        device = traj.device

        # Filter agents with future data
        if category is not None:
            pred_mask = category >= 3
        else:
            pred_mask = valid.any(dim=1)

        if pred_mask.sum() == 0:
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'agent_cls_loss': torch.tensor(0.0, device=device),
                'map_cls_loss': torch.tensor(0.0, device=device),
                'agent_accuracy': torch.tensor(0.0, device=device),
            }

        traj_m = traj[pred_mask]            # (M, K, T, 2)
        gt_m = gt[pred_mask]                # (M, T, 2)
        valid_m = valid[pred_mask]          # (M, T)
        batch_m = agent_batch[pred_mask]    # (M,)

        # Joint WTA: sum endpoint distances per mode across agents in each scene
        gt_endpoint = gt_m[:, -1, :]                    # (M, 2)
        traj_endpoint = traj_m[:, :, -1, :]             # (M, K, 2)
        endpoint_dist = (traj_endpoint - gt_endpoint.unsqueeze(1)).norm(dim=-1)  # (M, K)

        # Sum per scene
        batch_ids = scene_logits.shape[0]
        scene_dist = torch.zeros(batch_ids, K, device=device)
        for b in range(batch_ids):
            b_mask = batch_m == b
            if b_mask.any():
                scene_dist[b] = endpoint_dist[b_mask].sum(dim=0)

        # Best joint mode per scene
        best_scene_mode = scene_dist.argmin(dim=-1)  # (B,)

        # Per-agent best mode (from scene selection)
        best_mode = best_scene_mode[batch_m]  # (M,)

        # Regression loss
        M = traj_m.shape[0]
        best_traj = traj_m[torch.arange(M, device=device), best_mode]  # (M, T, 2)
        valid_expanded = valid_m.unsqueeze(-1).expand_as(best_traj)
        reg_loss = F.smooth_l1_loss(
            best_traj[valid_expanded], gt_m[valid_expanded], reduction='mean')

        # Classification loss: cross-entropy on scene logits
        cls_loss = F.cross_entropy(scene_logits, best_scene_mode)

        # Mode accuracy
        pred_mode = scene_logits.argmax(dim=-1)
        mode_acc = (pred_mode == best_scene_mode).float().mean()

        total_loss = self.reg_loss_weight * reg_loss + self.cls_loss_weight * cls_loss

        return {
            'loss': total_loss,
            'agent_cls_loss': cls_loss.detach(),
            'map_cls_loss': torch.tensor(0.0, device=device),
            'agent_accuracy': mode_acc.detach(),
        }

    def inference(self, data, greedy: bool = False,
                  temperature: float = 1.0) -> dict:
        self.eval()
        pred = self.forward(data)

        traj = pred['traj']
        probs = pred['probs']
        centers = pred['centers']
        headings = pred['headings']
        gt_local = pred['gt_future']
        gt_valid = pred['gt_valid']

        if greedy:
            selected = probs.argmax(dim=-1)
        else:
            scaled_logits = pred['logits'] / max(temperature, 1e-8)
            selected = torch.multinomial(F.softmax(scaled_logits, dim=-1), 1).squeeze(-1)

        N = traj.shape[0]
        selected_traj = traj[torch.arange(N, device=traj.device), selected]

        cos_h = headings.cos()
        sin_h = headings.sin()

        scene_x = selected_traj[..., 0] * cos_h.unsqueeze(1) - selected_traj[..., 1] * sin_h.unsqueeze(1)
        scene_y = selected_traj[..., 0] * sin_h.unsqueeze(1) + selected_traj[..., 1] * cos_h.unsqueeze(1)
        pred_scene = torch.stack([scene_x, scene_y], dim=-1) + centers.unsqueeze(1)

        gt_scene_x = gt_local[..., 0] * cos_h.unsqueeze(1) - gt_local[..., 1] * sin_h.unsqueeze(1)
        gt_scene_y = gt_local[..., 0] * sin_h.unsqueeze(1) + gt_local[..., 1] * cos_h.unsqueeze(1)
        gt_scene = torch.stack([gt_scene_x, gt_scene_y], dim=-1) + centers.unsqueeze(1)

        return {
            'pred_traj': pred_scene,
            'gt': gt_scene,
            'valid_mask': gt_valid,
        }
