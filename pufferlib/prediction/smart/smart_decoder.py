# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""SMART decoder combining map and agent encoders.

Adapted from SMART/smart/modules/smart_decoder.py.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from pufferlib.prediction.smart.agent_decoder import SMARTAgentDecoder
from pufferlib.prediction.smart.map_decoder import SMARTMapDecoder


class SMARTDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token: Dict,
                 token_data: Dict,
                 num_actions: int = 2048,
                 shift: int = 5,
                 hist_drop_prob: float = 0.1,
                 num_t_layers: Optional[int] = None,
                 num_pt2a_layers: Optional[int] = None,
                 num_a2a_layers: Optional[int] = None) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token,
        )
        self.agent_encoder = SMARTAgentDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_data=token_data,
            num_actions=num_actions,
            shift=shift,
            hist_drop_prob=hist_drop_prob,
            num_t_layers=num_t_layers,
            num_pt2a_layers=num_pt2a_layers,
            num_a2a_layers=num_a2a_layers,
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}

    def inference(self, data: HeteroData,
                  greedy: bool = False,
                  temperature: float = 1.0,
                  ego_condition: bool = False,
                  max_steps: int = 0) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder.inference(
            data, map_enc, greedy=greedy, temperature=temperature,
            ego_condition=ego_condition, max_steps=max_steps)
        return {**map_enc, **agent_enc}

    def get_token_logits(self, data: HeteroData):
        """Get raw logits for the next motion token (first future step).

        Returns:
            logits: (N, num_actions) raw logits
            eval_mask: (N,) valid agent mask
        """
        map_enc = self.map_encoder(data)
        return self.agent_encoder.get_token_logits(data, map_enc)
