# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Prediction model: SMART-based world model with 91 action tokens.

Main model that combines map + agent encoders and provides
training (teacher-forced) and inference (autoregressive) modes.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from pufferlib.prediction.smart.smart_decoder import SMARTDecoder
from pufferlib.prediction.map_tokenizer import load_map_codebook
from pufferlib.prediction.trajectory_tokenizer import load_motion_codebook


class PredictionModel(nn.Module):

    def __init__(self, config: dict):
        super(PredictionModel, self).__init__()
        self.config = config

        # Parse config values (may be strings from .ini files)
        def _int(key, default):
            v = config.get(key, default)
            return int(v) if v is not None else default

        def _float(key, default):
            v = config.get(key, default)
            return float(v) if v is not None else default

        # Load map token codebook
        map_token = load_map_codebook(config.get('map_token_path', None))

        # Load motion token codebook (2048 trajectory templates)
        motion_token_data = load_motion_codebook(
            config.get('motion_token_path', None))

        self.encoder = SMARTDecoder(
            input_dim=_int('input_dim', 2),
            hidden_dim=_int('hidden_dim', 128),
            num_historical_steps=_int('num_historical_steps', 11),
            pl2pl_radius=_float('pl2pl_radius', 10.0),
            time_span=_int('time_span', None),
            pl2a_radius=_float('pl2a_radius', 30.0),
            a2a_radius=_float('a2a_radius', 60.0),
            num_freq_bands=_int('num_freq_bands', 64),
            num_map_layers=_int('num_map_layers', 3),
            num_agent_layers=_int('num_agent_layers', 6),
            num_heads=_int('num_heads', 8),
            head_dim=_int('head_dim', 16),
            dropout=_float('dropout', 0.1),
            map_token=map_token,
            token_data=motion_token_data,
            num_actions=_int('num_actions', 2048),
            shift=_int('shift', 5),
            hist_drop_prob=_float('hist_drop_prob', 0.1),
            num_t_layers=_int('num_t_layers', None),
            num_pt2a_layers=_int('num_pt2a_layers', None),
            num_a2a_layers=_int('num_a2a_layers', None),
        )

        self.cls_loss = nn.CrossEntropyLoss(
            label_smoothing=_float('label_smoothing', 0.1))
        self.map_cls_loss = nn.CrossEntropyLoss(
            label_smoothing=_float('label_smoothing', 0.1))

    @property
    def num_historical_steps(self):
        return self.encoder.agent_encoder.num_historical_steps

    def forward(self, data: HeteroData) -> dict:
        """Training forward pass (teacher-forced).

        Returns dict with:
            'next_token_prob': (N, T, 91) action logits
            'next_token_idx_gt': (N, T) GT action indices
            'next_token_eval_mask': (N, T) valid prediction mask
            'map_next_token_prob': map token predictions
            'map_next_token_idx_gt': map GT token indices
        """
        return self.encoder(data)

    def inference(self, data: HeteroData, greedy: bool = False,
                  temperature: float = 1.0,
                  ego_condition: bool = False,
                  max_steps: int = 0) -> dict:
        """Autoregressive inference.

        Args:
            max_steps: If > 0, only predict this many future sim steps
                       (e.g. 10 → 2 token steps instead of 16).

        Returns dict with:
            'pred_traj': (N, T_future, 2) predicted positions
            'pred_head': (N, T_future) predicted headings
            'next_token_idx': (N, num_future_tokens) predicted token indices
            'gt': (N, T_future, 2) ground-truth positions
            'valid_mask': (N, T_future) validity mask
        """
        return self.encoder.inference(data, greedy=greedy, temperature=temperature,
                                      ego_condition=ego_condition, max_steps=max_steps)

    def get_token_logits(self, data: HeteroData):
        """Get raw logits for the next motion token (first future step).

        Returns:
            logits: (N, num_actions) raw logits
            eval_mask: (N,) valid agent mask
        """
        return self.encoder.get_token_logits(data)

    def compute_loss(self, pred: dict) -> dict:
        """Compute training losses.

        Returns dict with:
            'loss': total loss
            'agent_cls_loss': agent action prediction loss
            'map_cls_loss': map token prediction loss
            'agent_accuracy': top-1 action prediction accuracy
        """
        # Agent action prediction loss
        mask = pred['next_token_eval_mask']
        agent_logits = pred['next_token_prob'][mask]
        agent_gt = pred['next_token_idx_gt'][mask]

        if agent_logits.numel() > 0:
            agent_cls_loss = self.cls_loss(agent_logits, agent_gt)
            agent_acc = (agent_logits.argmax(dim=-1) == agent_gt).float().mean()
        else:
            agent_cls_loss = torch.tensor(0.0, device=mask.device)
            agent_acc = torch.tensor(0.0, device=mask.device)

        # Map token prediction loss
        map_logits = pred.get('map_next_token_prob', None)
        map_gt = pred.get('map_next_token_idx_gt', None)

        if map_logits is not None and map_logits.numel() > 0:
            map_cls_loss = self.map_cls_loss(map_logits, map_gt)
        else:
            map_cls_loss = torch.tensor(0.0, device=mask.device)

        # Only agent CE loss, matching original SMART (map encoder learns
        # through gradient flow via agent decoder cross-attention)
        loss = agent_cls_loss

        return {
            'loss': loss,
            'agent_cls_loss': agent_cls_loss,
            'map_cls_loss': map_cls_loss,
            'agent_accuracy': agent_acc,
        }
