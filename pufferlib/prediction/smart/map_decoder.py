# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Map decoder adapted from SMART/smart/modules/map_decoder.py.

Uses 1024 pre-computed road trajectory templates for map representation.
Encodes map tokens via self-attention and produces map embeddings
used as context for agent cross-attention.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import subgraph

from pufferlib.prediction.smart.attention_layer import AttentionLayer
from pufferlib.prediction.smart.mlp_layer import MLPLayer
from pufferlib.prediction.smart.fourier_embedding import FourierEmbedding, MLPEmbedding
from pufferlib.prediction.smart.utils import angle_between_2d_vectors, wrap_angle, weight_init


class SMARTMapDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers

        input_dim_r_pt2pt = 3 if input_dim == 2 else 4

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt, hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands)
        self.pt2pt_layers = nn.ModuleList([
            AttentionLayer(
                hidden_dim=hidden_dim, num_heads=num_heads,
                head_dim=head_dim, dropout=dropout,
                bipartite=False, has_pos_emb=True)
            for _ in range(num_layers)])

        self.token_size = 1024
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim,
            output_dim=self.token_size)
        input_dim_token = 22  # 11 points * 2 coords
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)

    def _sample_pt_pred(self, data: HeteroData):
        """BERT-style random masking for road NTP (matching original SMART).

        Randomly masks ~1/3 of non-first tokens within each polyline.
        Unmasked predecessors of masked tokens become predictors.
        """
        polygon_idx = data['pt_token']['polygon_idx']
        M = polygon_idx.shape[0]
        batch_idx = data['pt_token']['batch'] if isinstance(data, Batch) else torch.zeros(M, dtype=torch.long, device=polygon_idx.device)

        # Create unique polyline IDs across batch
        unique_poly_id = batch_idx * 100000 + polygon_idx

        # Identify non-first tokens within polylines (candidates for masking)
        same_polyline = unique_poly_id[:-1] == unique_poly_id[1:]
        is_not_first = torch.zeros(M, dtype=torch.bool, device=polygon_idx.device)
        is_not_first[1:] = same_polyline

        # Randomly mask ~1/3 of candidate tokens
        candidate_indices = is_not_first.nonzero(as_tuple=True)[0]
        if len(candidate_indices) == 0:
            return (torch.ones(M, dtype=torch.bool, device=polygon_idx.device),
                    torch.zeros(M, dtype=torch.bool, device=polygon_idx.device),
                    torch.zeros(M, dtype=torch.bool, device=polygon_idx.device))

        num_to_mask = max(1, len(candidate_indices) // 3)
        perm = torch.randperm(len(candidate_indices), device=polygon_idx.device)[:num_to_mask]
        masked_indices = candidate_indices[perm]

        # pt_valid_mask: all tokens except masked ones
        pt_valid_mask = torch.ones(M, dtype=torch.bool, device=polygon_idx.device)
        pt_valid_mask[masked_indices] = False

        # pt_pred_mask: unmasked predecessor of a masked token (same polyline)
        # pt_target_mask: the masked token itself
        pred_indices = masked_indices - 1
        valid_preds = pt_valid_mask[pred_indices]  # predecessor must be unmasked

        pt_pred_mask = torch.zeros(M, dtype=torch.bool, device=polygon_idx.device)
        pt_target_mask = torch.zeros(M, dtype=torch.bool, device=polygon_idx.device)
        pt_pred_mask[pred_indices[valid_preds]] = True
        pt_target_mask[masked_indices[valid_preds]] = True

        return pt_valid_mask, pt_pred_mask, pt_target_mask

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        # BERT-style random masking during training, static masks during eval
        if self.training:
            pt_valid_mask, pt_pred_mask, pt_target_mask = self._sample_pt_pred(data)
        else:
            pt_valid_mask = data['pt_token']['pt_valid_mask']
            pt_pred_mask = data['pt_token']['pt_pred_mask']
            pt_target_mask = data['pt_token']['pt_target_mask']

        pos_pt = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pt = data['pt_token']['orientation'].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)

        # Embed map tokens
        token_sample_pt = self.map_token['traj_src'].to(pos_pt.device).float()
        pt_token_emb_src = self.token_emb(
            token_sample_pt.view(token_sample_pt.shape[0], -1))
        pt_token_emb = pt_token_emb_src[data['pt_token']['token_idx']]

        x_pt = pt_token_emb

        # Zero out embeddings of masked tokens (BERT-style: hide content)
        x_pt[~pt_valid_mask] = 0.0

        # Add categorical embeddings
        x_pt_categorical_embs = [
            self.type_pt_emb(data['pt_token']['type'].long()),
            self.polygon_type_emb(data['pt_token']['pl_type'].long()),
        ]
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)

        # Build spatial edges (filter out edges from/to masked tokens)
        edge_index_pt2pt = radius_graph(
            x=pos_pt[:, :2], r=self.pl2pl_radius,
            batch=data['pt_token']['batch'] if isinstance(data, Batch) else None,
            loop=False, max_num_neighbors=100)

        # NOTE: Do NOT filter edges based on BERT masking (matching original SMART
        # where mask_pt=False). All map tokens can attend to all neighbors.
        # Masked tokens have zeroed embeddings (line 132) but still receive
        # context through attention from unmasked neighbors.

        # Compute relative features
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]])
        r_pt2pt = torch.stack([
            torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
            angle_between_2d_vectors(
                ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                nbr_vector=rel_pos_pt2pt[:, :2]),
            rel_orient_pt2pt], dim=-1)
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)

        # Self-attention layers
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        # Map next-token prediction (for training loss)
        next_token_prob = self.token_predict_head(x_pt[pt_pred_mask])
        next_token_index_gt = data['pt_token']['token_idx'][pt_target_mask]

        return {
            'x_pt': x_pt,
            'map_next_token_prob': next_token_prob,
            'map_next_token_idx_gt': next_token_index_gt,
            'map_next_token_eval_mask': pt_pred_mask[pt_pred_mask],
        }
