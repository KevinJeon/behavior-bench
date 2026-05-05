# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Agent decoder adapted from SMART/smart/modules/agent_decoder.py.

Key design:
- 2048 motion tokens from pre-computed trajectory codebook (k-means on Waymo)
- shift=5: each token covers a 5-timestep segment (0.5s at 10Hz)
- MLPEmbedding(input_dim=8) for token embedding (flattened 4-corner polygon)
- Codebook-based autoregressive inference (no bicycle dynamics)
- Vehicles only (single token embedding head)
- Token sequence: T=91 steps -> 18 token positions at [5, 10, ..., 90]
"""

import math
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph

from pufferlib.prediction.smart.mlp_layer import MLPLayer
from pufferlib.prediction.smart.attention_layer import AttentionLayer
from pufferlib.prediction.smart.fourier_embedding import FourierEmbedding, MLPEmbedding
from pufferlib.prediction.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle

import numpy as np


class SMARTAgentDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_data: Dict,
                 num_actions: int = 2048,
                 shift: int = 5,
                 hist_drop_prob: float = 0.1,
                 num_t_layers: Optional[int] = None,
                 num_pt2a_layers: Optional[int] = None,
                 num_a2a_layers: Optional[int] = None) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.shift = shift
        self.num_historical_tokens = (num_historical_steps - 1) // shift  # 2
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_t_layers = num_t_layers or num_layers
        self.num_pt2a_layers = num_pt2a_layers or num_layers
        self.num_a2a_layers = num_a2a_layers or num_layers
        self.num_layers = max(self.num_t_layers, self.num_pt2a_layers, self.num_a2a_layers)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_actions = num_actions

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8  # 4 corners x 2 coords

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        # Motion token embedding: MLP on flattened polygon contour (4x2=8D)
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a, hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t, hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands)

        self.fusion_emb = MLPEmbedding(
            input_dim=hidden_dim * 2, hidden_dim=hidden_dim)

        self.t_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout,
                bipartite=False, has_pos_emb=True)
            for _ in range(self.num_t_layers)])
        self.pt2a_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout,
                bipartite=True, has_pos_emb=True)
            for _ in range(self.num_pt2a_layers)])
        self.a2a_attn_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout,
                bipartite=False, has_pos_emb=True)
            for _ in range(self.num_a2a_layers)])

        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            output_dim=num_actions)

        # Store codebook data (numpy arrays, converted to tensors on demand)
        self.trajectory_token = token_data['token']
        self.trajectory_token_all = token_data['token_all']

        self.beam_size = 5
        self.hist_drop_prob = hist_drop_prob
        self.apply(weight_init)

    def agent_token_embedding(self, data, agent_category, agent_token_index,
                              pos_a, head_vector_a, valid_mask=None,
                              inference=False):
        """Compute agent feature embeddings at token level.

        Uses MLPEmbedding on codebook polygon contours (4x2=8D).
        When inference=True, also returns codebook trajectory data and
        intermediate embeddings needed for autoregressive updates.
        """
        num_agent, num_tokens, traj_dim = pos_a.shape
        motion_vector_a = torch.cat([
            pos_a.new_zeros(num_agent, 1, self.input_dim),
            pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        if valid_mask is not None:
            both_valid = torch.cat([
                valid_mask[:, :1],
                valid_mask[:, 1:] & valid_mask[:, :-1],
            ], dim=1)
            motion_vector_a = motion_vector_a * both_valid.unsqueeze(-1)

        # Compute token embeddings from codebook polygon contours per type
        device = pos_a.device
        type_keys = ['veh', 'ped', 'cyc']
        type_emb_modules = [self.token_emb_veh, self.token_emb_ped, self.token_emb_cyc]
        token_embs_per_type = []  # list of (2048, hidden_dim) per type
        token_raw_per_type = []   # list of (2048, 4, 2) per type
        for key, emb_module in zip(type_keys, type_emb_modules):
            raw = torch.from_numpy(self.trajectory_token[key]).clone().to(device).float()
            token_raw_per_type.append(raw)
            token_embs_per_type.append(emb_module(raw.view(raw.shape[0], -1)))
        # Stack: (3, 2048, hidden_dim)
        all_token_embs = torch.stack(token_embs_per_type, dim=0)
        self._all_token_embs = all_token_embs

        # Prepare sub-step trajectory data for inference
        if inference:
            traj_all_per_type = []
            for i, key in enumerate(type_keys):
                raw_all = torch.from_numpy(
                    self.trajectory_token_all[key]).clone().to(device).float()
                traj_all_per_type.append(torch.cat([
                    raw_all[:, :self.shift],
                    token_raw_per_type[i][:, None, ...]
                ], dim=1))  # (2048, shift+1, 4, 2)
            # Stack: (3, 2048, shift+1, 4, 2)
            all_token_traj_all = torch.stack(traj_all_per_type, dim=0)

        # Look up embeddings per agent type
        agent_types = data['agent']['type'].long()  # (num_agent,)
        # Select the right type's embedding table for each agent
        per_agent_emb_table = all_token_embs[agent_types]  # (num_agent, 2048, hidden_dim)
        # Index by token: per_agent_emb_table[i, agent_token_index[i, j]] for each (i, j)
        agent_token_emb = per_agent_emb_table[
            torch.arange(num_agent, device=device).unsqueeze(1).expand(-1, num_tokens),
            agent_token_index
        ]  # (num_agent, num_tokens, hidden_dim)

        if inference:
            # Select per-agent trajectory tables
            agent_token_traj_all = all_token_traj_all[agent_types]  # (num_agent, 2048, shift+1, 4, 2)

        categorical_embs = [
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(
                repeats=num_tokens, dim=0),
            self.shape_emb(data['agent']['shape'][
                :, self.num_historical_steps - 1, :
            ]).repeat_interleave(repeats=num_tokens, dim=0),
        ]
        feature_a = torch.stack([
            torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_a,
                nbr_vector=motion_vector_a[:, :, :2]),
        ], dim=-1)

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=categorical_embs)
        x_a = x_a.view(-1, num_tokens, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return (feat_a, agent_token_traj_all,
                    agent_token_emb, categorical_embs)
        return feat_a

    def build_temporal_edge(self, pos_a, head_a, head_vector_a,
                            num_agent, mask, inference_mask=None):
        """Build temporal self-attention edges."""
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        if self.hist_drop_prob > 0 and self.training:
            keep_mask = torch.bernoulli(
                torch.ones_like(mask, dtype=torch.float) * (1 - self.hist_drop_prob)
            ).bool()
            hist_mask = mask & keep_mask
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t.contiguous())[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span // self.shift]

        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack([
            torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_t[edge_index_t[1]],
                nbr_vector=rel_pos_t[:, :2]),
            rel_head_t,
            (edge_index_t[0] - edge_index_t[1]).float()], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a,
                               batch_s, mask_s):
        """Build agent-to-agent interaction edges."""
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s,
            loop=False, max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]

        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack([
            torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_s[edge_index_a2a[1]],
                nbr_vector=rel_pos_a2a[:, :2]),
            rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(self, data, num_tokens, pos_a, head_a,
                             head_vector_a, mask, batch_s, batch_pl):
        """Build map-to-agent cross-attention edges."""
        mask_pl2a = mask.clone().transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)

        pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = pos_pl.repeat(num_tokens, 1)
        orient_pl = orient_pl.repeat(num_tokens)

        edge_index_pl2a = radius(
            x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
            batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack([
            torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=head_vector_s[edge_index_pl2a[1]],
                nbr_vector=rel_pos_pl2a[:, :2]),
            rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def _build_batch_vectors(self, data, num_tokens, device):
        """Build batch vectors for spatial edge construction."""
        if isinstance(data, Batch):
            agent_num_nodes = data['agent']['num_nodes']
            if isinstance(agent_num_nodes, int):
                agent_batch = torch.zeros(
                    agent_num_nodes, dtype=torch.long, device=device)
            else:
                agent_batch = torch.arange(
                    len(agent_num_nodes), device=device
                ).repeat_interleave(agent_num_nodes)
            batch_s = torch.cat([
                agent_batch + data.num_graphs * t
                for t in range(num_tokens)], dim=0)
            batch_pl = torch.cat([
                data['pt_token']['batch'] + data.num_graphs * t
                for t in range(num_tokens)], dim=0)
        else:
            batch_s = torch.arange(
                num_tokens, device=device
            ).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(
                num_tokens, device=device
            ).repeat_interleave(data['pt_token']['num_nodes'])
        return batch_s, batch_pl

    def _run_encoder(self, data, map_enc, feat_a, pos_a, head_a, head_vector_a,
                     num_agent, num_tokens, mask, pl2a_mask, batch_s, batch_pl,
                     inference_mask=None):
        """Run temporal + spatial + map attention layers."""
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)

        mask_s = (inference_mask if inference_mask is not None else mask
                  ).transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a, head_a, head_vector_a, batch_s, mask_s)
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data, num_tokens, pos_a, head_a, head_vector_a,
            inference_mask if inference_mask is not None else pl2a_mask,
            batch_s, batch_pl)

        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i % self.num_t_layers](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(
                -1, num_tokens, self.hidden_dim
            ).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i % self.num_pt2a_layers](
                (map_enc['x_pt'].repeat_interleave(
                    repeats=num_tokens, dim=0
                ).reshape(-1, num_tokens, self.hidden_dim
                ).transpose(0, 1).reshape(-1, self.hidden_dim), feat_a),
                r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i % self.num_a2a_layers](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_tokens, -1, self.hidden_dim).transpose(0, 1)

        return feat_a

    def forward(self, data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training forward pass (teacher-forced) on token-level data.

        Data shapes (token-level, shift=5):
            'token_pos': (N, 18, 2)
            'token_heading': (N, 18)
            'token_idx': (N, 18)
            'agent_valid_mask': (N, 18)
        """
        pos_a = data['agent']['token_pos']
        head_a = data['agent']['token_heading']
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_tokens, traj_dim = pos_a.shape
        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx']

        mask = data['agent']['agent_valid_mask'].clone()

        feat_a = self.agent_token_embedding(
            data, agent_category, agent_token_index, pos_a, head_vector_a,
            valid_mask=mask)

        batch_s, batch_pl = self._build_batch_vectors(
            data, num_tokens, pos_a.device)

        # Map-to-agent mask: only agents to predict (cat 3) + SDC (cat 5)
        pl2a_mask = mask.clone()
        pl2a_mask[(agent_category != 3) & (agent_category != 5)] = False

        feat_a = self._run_encoder(
            data, map_enc, feat_a, pos_a, head_a, head_vector_a,
            num_agent, num_tokens, mask, pl2a_mask, batch_s, batch_pl)

        # Predict next motion token
        next_token_prob = self.token_predict_head(feat_a)

        # GT: next token is the token at k+1
        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)

        # Loss mask: cat 3 + SDC cat 5, valid at current and neighbors
        eval_mask = mask.clone()
        eval_mask[(agent_category != 3) & (agent_category != 5)] = False
        next_token_eval_mask = (
            eval_mask
            * eval_mask.roll(shifts=-1, dims=1)
            * eval_mask.roll(shifts=1, dims=1))
        next_token_eval_mask[:, -1] = False

        return {
            'x_a': feat_a,
            'next_token_prob': next_token_prob,
            'next_token_idx_gt': next_token_index_gt,
            'next_token_eval_mask': next_token_eval_mask,
        }

    def inference(self, data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor],
                  greedy: bool = False,
                  temperature: float = 1.0,
                  ego_condition: bool = False,
                  max_steps: int = 0) -> Dict[str, torch.Tensor]:
        """Autoregressive inference with codebook-based trajectory generation.

        For each future token, predicts a codebook index, looks up the
        sub-step polygon trajectory from token_all, rotates/translates to
        world coords, and extracts position (mean of 4 corners) and heading
        (atan2 of left_front - left_back).
        """
        T_full = data['agent']['position'].shape[1]  # 91
        num_future = T_full - self.num_historical_steps  # 80

        pos_a = data['agent']['token_pos'].clone()
        head_a = data['agent']['token_heading'].clone()
        num_agent, num_tokens, traj_dim = pos_a.shape

        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx'].clone()
        agent_types = data['agent']['type'].long()
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]

        # Zero out future token positions/headings
        pos_a[:, self.num_historical_tokens:] = 0
        head_a[:, self.num_historical_tokens:] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        # Token-level validity
        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        agent_valid_mask[:, self.num_historical_tokens:] = True
        agent_valid_mask[~eval_mask] = False

        # Get initial embeddings with codebook data for inference
        (feat_a, agent_token_traj_all,
         agent_token_emb, categorical_embs) = self.agent_token_embedding(
            data, agent_category, agent_token_index,
            pos_a, head_vector_a, inference=True)
        # agent_token_traj_all: (N, 2048, shift+1, 4, 2)

        # Output tensors
        pred_traj = torch.zeros(num_agent, num_future, 2, device=pos_a.device)
        pred_head = torch.zeros(num_agent, num_future, device=pos_a.device)
        next_token_idx_list = []

        mask = agent_valid_mask.clone()
        feat_a_t_dict = {}

        batch_s, batch_pl = self._build_batch_vectors(
            data, num_tokens, pos_a.device)

        num_recurrent = num_future // self.shift  # 16
        if max_steps > 0:
            import math
            num_recurrent = min(num_recurrent, math.ceil(max_steps / self.shift))

        # Ego-conditioning: override ego's position at the query timestep
        # with GT for the next step, so a2a edges see ego one step ahead.
        # No data shifting or extra edges needed.
        if ego_condition:
            ego_cond_mask = (agent_category == 5)
            gt_token_pos = data['agent']['token_pos']       # (N, 18, 2)
            gt_token_heading = data['agent']['token_heading']  # (N, 18)
        else:
            ego_cond_mask = None

        for t in range(num_recurrent):
            # Build inference mask (KV-cache style)
            if t == 0:
                inference_mask = mask.clone()
                inference_mask[:, self.num_historical_tokens + t:] = False
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[:, self.num_historical_tokens + t - 1] = True

            # Ego-conditioning (teacher forcing): set ego's position at query
            # timestep to GT, so a2a edges see ego's true position
            if ego_cond_mask is not None and ego_cond_mask.any():
                query_tok = self.num_historical_tokens - 1 + t
                gt_idx = query_tok  # teacher forcing: GT at same position
                if gt_idx < num_tokens:
                    pos_a[ego_cond_mask, query_tok] = gt_token_pos[
                        ego_cond_mask, gt_idx]
                    head_a[ego_cond_mask, query_tok] = gt_token_heading[
                        ego_cond_mask, gt_idx]
                    head_vector_a = torch.stack(
                        [head_a.cos(), head_a.sin()], dim=-1)

            # Build edges
            edge_index_t, r_t = self.build_temporal_edge(
                pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)
            mask_s = inference_mask.transpose(0, 1).reshape(-1)
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a, head_a, head_vector_a, batch_s, mask_s)
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                data, num_tokens, pos_a, head_a, head_vector_a,
                inference_mask, batch_s, batch_pl)

            # Run attention layers with KV caching
            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                feat_a = feat_a.reshape(-1, self.hidden_dim)
                feat_a = self.t_attn_layers[i % self.num_t_layers](feat_a, r_t, edge_index_t)
                feat_a = feat_a.reshape(
                    -1, num_tokens, self.hidden_dim
                ).transpose(0, 1).reshape(-1, self.hidden_dim)
                feat_a = self.pt2a_attn_layers[i % self.num_pt2a_layers](
                    (map_enc['x_pt'].repeat_interleave(
                        repeats=num_tokens, dim=0
                    ).reshape(-1, num_tokens, self.hidden_dim
                    ).transpose(0, 1).reshape(-1, self.hidden_dim), feat_a),
                    r_pl2a, edge_index_pl2a)
                feat_a = self.a2a_attn_layers[i % self.num_a2a_layers](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(
                    num_tokens, -1, self.hidden_dim).transpose(0, 1)

                if i + 1 not in feat_a_t_dict:
                    feat_a_t_dict[i + 1] = feat_a
                else:
                    feat_a_t_dict[i + 1][
                        :, self.num_historical_tokens - 1 + t
                    ] = feat_a[:, self.num_historical_tokens - 1 + t]

            # Predict next token from features at current position
            logits = self.token_predict_head(
                feat_a[:, self.num_historical_tokens - 1 + t])
            probs = torch.softmax(logits / temperature, dim=-1)

            if greedy:
                next_token_idx = probs.argmax(dim=-1, keepdim=True)
            else:
                # Top-k sampling (beam_size=5, then multinomial)
                topk_prob, topk_idx = torch.topk(
                    probs, k=self.beam_size, dim=-1)
                sample_idx = torch.multinomial(topk_prob, 1)
                next_token_idx = topk_idx.gather(-1, sample_idx)

            next_token_idx = next_token_idx.squeeze(-1)  # (N,)

            # Ego-conditioning: override predicted token with GT for ego
            if ego_cond_mask is not None and ego_cond_mask.any():
                next_token_idx = next_token_idx.clone()
                next_token_idx[ego_cond_mask] = agent_token_index[
                    ego_cond_mask, self.num_historical_tokens + t]

            # Look up sub-step trajectory from codebook (per-agent type)
            # agent_token_traj_all: (N, 2048, shift+1, 4, 2)
            next_token_traj = agent_token_traj_all[
                torch.arange(num_agent, device=next_token_idx.device), next_token_idx
            ]  # (N, 6, 4, 2)

            # Rotate by current heading and translate to world coords
            theta = head_a[:, self.num_historical_tokens - 1 + t]
            cos_t = theta.cos()
            sin_t = theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
            rot_mat[:, 0, 0] = cos_t
            rot_mat[:, 0, 1] = sin_t
            rot_mat[:, 1, 0] = -sin_t
            rot_mat[:, 1, 1] = cos_t

            # Rotate all corners: (N, 6, 4, 2) @ (N, 2, 2)
            rot_expanded = rot_mat.unsqueeze(1).unsqueeze(1).expand(
                -1, self.shift + 1, 1, -1, -1).reshape(-1, 2, 2)
            traj_rotated = torch.bmm(
                next_token_traj.reshape(-1, 4, 2),
                rot_expanded
            ).view(num_agent, self.shift + 1, 4, 2)

            # Translate to world coords
            origin = pos_a[:, self.num_historical_tokens - 1 + t, :]
            traj_world = traj_rotated + origin[:, None, None, :]

            # Extract positions (mean of 4 corners) and headings per sub-step
            # Skip index 0 (starting position), use indices 1..shift
            step_positions = traj_world[:, 1:].mean(dim=2)  # (N, 5, 2)
            diff_xy = (traj_world[:, 1:, 0, :]
                       - traj_world[:, 1:, 3, :])  # left_front - left_back
            step_headings = torch.atan2(
                diff_xy[:, :, 1], diff_xy[:, :, 0])  # (N, 5)

            # Record predictions for this token's sub-steps
            pred_traj[:, t * self.shift:(t + 1) * self.shift] = step_positions
            pred_head[:, t * self.shift:(t + 1) * self.shift] = step_headings

            # Update token-level state using final contour
            final_contour = traj_world[:, -1]  # (N, 4, 2)
            pos_a[:, self.num_historical_tokens + t] = final_contour.mean(dim=1)
            diff_final = final_contour[:, 0, :] - final_contour[:, 3, :]
            head_a[:, self.num_historical_tokens + t] = torch.atan2(
                diff_final[:, 1], diff_final[:, 0])

            next_token_idx_list.append(next_token_idx[:, None])

            # Update token embedding at predicted position (type-aware)
            per_agent_emb_table = self._all_token_embs[agent_types]  # (N, 2048, hidden_dim)
            agent_token_emb[
                :, self.num_historical_tokens + t
            ] = per_agent_emb_table[torch.arange(num_agent, device=next_token_idx.device), next_token_idx]

            # Recompute feat_a from updated positions/headings
            head_vector_a = torch.stack(
                [head_a.cos(), head_a.sin()], dim=-1)
            motion_vector_a = torch.cat([
                pos_a.new_zeros(num_agent, 1, self.input_dim),
                pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
            motion_vector_a[
                :, self.num_historical_tokens + 1 + t:
            ] = 0

            x_a = torch.stack([
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a,
                    nbr_vector=motion_vector_a[:, :, :2]),
            ], dim=-1)
            x_a = self.x_a_emb(
                continuous_inputs=x_a.view(-1, x_a.size(-1)),
                categorical_embs=categorical_embs)
            x_a = x_a.view(-1, num_tokens, self.hidden_dim)

            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        # GT validity for metrics
        gt_valid = data['agent']['valid_mask'][:, self.num_historical_steps:]
        gt_valid = gt_valid & eval_mask.unsqueeze(1)

        return {
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'gt': data['agent']['position'][
                :, self.num_historical_steps:, :self.input_dim
            ].contiguous(),
            'valid_mask': gt_valid,
            'next_token_idx': torch.cat(next_token_idx_list, dim=-1),
        }

    def get_token_logits(self, data: HeteroData,
                         map_enc: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Get raw logits for the NEXT motion token (first future step only).

        This is a lightweight version of inference() that stops after
        the first token prediction and returns logits instead of sampling.

        Args:
            data: HeteroData with agent/map features
            map_enc: Pre-computed map encoding

        Returns:
            logits: (N, num_actions) raw logits over 2048 motion tokens
            eval_mask: (N,) bool mask indicating valid agents
        """
        pos_a = data['agent']['token_pos'].clone()
        head_a = data['agent']['token_heading'].clone()
        num_agent, num_tokens, traj_dim = pos_a.shape

        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx'].clone()
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]

        # Zero out future token positions/headings
        pos_a[:, self.num_historical_tokens:] = 0
        head_a[:, self.num_historical_tokens:] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        agent_valid_mask[:, self.num_historical_tokens:] = True
        agent_valid_mask[~eval_mask] = False

        (feat_a, agent_token_traj_all,
         agent_token_emb, categorical_embs) = self.agent_token_embedding(
            data, agent_category, agent_token_index,
            pos_a, head_vector_a, inference=True)

        mask = agent_valid_mask.clone()
        mask[:, self.num_historical_tokens:] = False
        mask[:, self.num_historical_tokens] = True  # only first future step

        batch_s, batch_pl = self._build_batch_vectors(
            data, num_tokens, pos_a.device)

        # Build edges for t=0
        inference_mask = mask.clone()
        inference_mask[:, self.num_historical_tokens:] = False

        edge_index_t, r_t = self.build_temporal_edge(
            pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)
        mask_s = inference_mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a, head_a, head_vector_a, batch_s, mask_s)
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data, num_tokens, pos_a, head_a, head_vector_a,
            inference_mask, batch_s, batch_pl)

        # Run attention layers
        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i % self.num_t_layers](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(
                -1, num_tokens, self.hidden_dim
            ).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i % self.num_pt2a_layers](
                (map_enc['x_pt'].repeat_interleave(
                    repeats=num_tokens, dim=0
                ).reshape(-1, num_tokens, self.hidden_dim
                ).transpose(0, 1).reshape(-1, self.hidden_dim), feat_a),
                r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i % self.num_a2a_layers](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(
                num_tokens, -1, self.hidden_dim).transpose(0, 1)

        # Get logits at the query position (last historical token)
        logits = self.token_predict_head(
            feat_a[:, self.num_historical_tokens - 1])

        return logits, eval_mask
