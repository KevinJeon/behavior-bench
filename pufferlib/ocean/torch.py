# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from PufferDrive V2.0
# (https://github.com/Emerge-Lab/PufferDrive/)
# Copyright (c) 2026 PufferDrive, licensed under the MIT license.

from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
import numpy as np
from typing import Tuple, Optional, Dict, List

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401
from torch.distributions import Dirichlet


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, action_chunk_size=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories

        # Determine ego dimension from environment's dynamics model
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)
    
    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value


class DriveConditioned(nn.Module):
    """Drive policy with Gigaflow-paper reward conditioning (Creward).

    Identical to Drive but reads an additional CREWARD_FEATURES-long channel
    after the road block and encodes it through a separate MLP before
    concatenating with the ego/partner/road features. Used in combination
    with `env.reward_conditioning=True` and the `drive_gigaflow_conditioning.ini`
    config.
    """
    def __init__(self, env, input_size=128, hidden_size=128, action_chunk_size=1, **kwargs):
        super().__init__()
        from pufferlib.ocean.drive import binding
        self.hidden_size = hidden_size
        self.action_chunk_size = action_chunk_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.creward_dim = binding.CREWARD_FEATURES

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.creward_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.creward_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # 4 branches now: ego, road, partner, creward
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(4 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        cre_dim = self.creward_dim

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]
        creward_obs = observations[:, ego_dim + partner_dim + road_dim : ego_dim + partner_dim + road_dim + cre_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)
        creward_features = self.creward_encoder(creward_obs)

        concat_features = torch.cat([ego_features, road_features, partner_features, creward_features], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value


class DriveTransformer(nn.Module):
    """Drive policy with ego-centric cross-attention over partner and road entities.

    Replaces the max-pool aggregation in Drive with a single cross-attention layer
    where the ego embedding queries against all partner and road entity embeddings.
    Drop-in replacement for Drive -- same encode_observations/decode_actions interface.
    """

    def __init__(self, env, input_size=128, hidden_size=128, num_heads=4, num_layers=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        # Per-entity encoders (same as Drive)
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # Type embeddings: 0=ego, 1=partner, 2=road
        self.type_embed = nn.Embedding(3, input_size)

        # Cross-attention layers
        self.num_layers = num_layers
        self.cross_attns = nn.ModuleList()
        self.attn_norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attns.append(
                nn.MultiheadAttention(input_size, num_heads, batch_first=True)
            )
            self.attn_norms.append(nn.LayerNorm(input_size))
            self.ffns.append(nn.Sequential(
                nn.Linear(input_size, input_size * 2),
                nn.GELU(),
                nn.Linear(input_size * 2, input_size),
            ))
            self.ffn_norms.append(nn.LayerNorm(input_size))

        # Output projection
        self.output_proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(input_size, hidden_size)),
            nn.GELU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        device = observations.device
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        # 1. Parse flat observation into structured components
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim:ego_dim + partner_dim + road_dim]

        # 2. Reshape into entity sets
        partner_objects = partner_obs.view(B, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(B, self.max_road_objects, self.road_features)

        # 3. Build padding masks BEFORE one-hot (True = ignore zero-padded slots)
        partner_mask = partner_objects.abs().sum(-1) == 0          # (B, P)
        road_mask = road_objects.abs().sum(-1) == 0                # (B, R)
        kv_mask = torch.cat([partner_mask, road_mask], dim=1)      # (B, P+R)

        # 4. One-hot encode road type (same as Drive)
        road_continuous = road_objects[:, :, :self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # 5. Encode each entity type
        ego_emb = self.ego_encoder(ego_obs).unsqueeze(1)          # (B, 1, D)
        partner_emb = self.partner_encoder(partner_objects)        # (B, P, D)
        road_emb = self.road_encoder(road_objects)                 # (B, R, D)

        # 6. Add type embeddings
        ego_emb = ego_emb + self.type_embed(torch.zeros(1, 1, dtype=torch.long, device=device))
        partner_emb = partner_emb + self.type_embed(torch.ones(1, 1, dtype=torch.long, device=device))
        road_emb = road_emb + self.type_embed(torch.full((1, 1), 2, dtype=torch.long, device=device))

        # 7. Build K/V sequence (include ego so attention never has all-masked keys)
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)  # ego is never masked
        kv = torch.cat([ego_emb, partner_emb, road_emb], dim=1)   # (B, 1+P+R, D)
        kv_mask = torch.cat([ego_mask, kv_mask], dim=1)            # (B, 1+P+R)

        # 8. Cross-attention layers
        q = ego_emb
        for i in range(self.num_layers):
            attn_out, _ = self.cross_attns[i](
                query=q, key=kv, value=kv,
                key_padding_mask=kv_mask,
            )
            q = self.attn_norms[i](q + attn_out)
            q = self.ffn_norms[i](q + self.ffns[i](q))

        # 9. Output projection
        out = q.squeeze(1)                                         # (B, D)
        return self.output_proj(out)                               # (B, hidden_size)

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value


class DriveNoGoal(Drive):
    """Drive policy that ignores the 2 goal features (obs[0:2]).

    Accepts the same full observation vector but only uses ego[2:7]
    (speed, width, length, collision, respawn) = 5 features.
    """

    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__(env, input_size=input_size, hidden_size=hidden_size, **kwargs)
        # Override ego_dim and ego_encoder for 5 features (no goal)
        self.ego_dim_no_goal = self.ego_dim - 2  # 5 for classic, 8 for jerk
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim_no_goal, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim  # full ego dim (7) for slicing the obs vector
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        # Skip first 2 features (goal_x, goal_y)
        ego_obs = observations[:, 2:ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding


class DriveGameFormer(nn.Module):
    """GameFormer-style encoder-decoder policy for Drive.

    Encoder: full self-attention over all entity tokens (agents + roads).
    Decoder: self-attention among agent tokens + cross-attention to scene encoding.
    No level-k iterative refinement -- single-pass decoder for PPO.
    """

    def __init__(self, env, input_size=128, hidden_size=128,
                 num_heads=4, num_enc_layers=2, ffn_expansion=2, dropout=0.0,
                 **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6
        self.num_agents = 1 + self.max_partner_objects  # ego + partners

        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7

        # --- Per-entity encoders ---
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # Type embeddings: 0=ego, 1=partner, 2=road
        self.type_embed = nn.Embedding(3, input_size)

        # --- Encoder: full self-attention over all tokens ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=input_size * ffn_expansion,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_enc_layers, enable_nested_tensor=False,
        )

        # --- Decoder: self-attention among agents + cross-attention to scene ---
        # Self-attention among agent tokens
        self.dec_self_norm = nn.LayerNorm(input_size)
        self.dec_self_attn = nn.MultiheadAttention(
            input_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.dec_self_ffn_norm = nn.LayerNorm(input_size)
        self.dec_self_ffn = nn.Sequential(
            nn.Linear(input_size, input_size * ffn_expansion),
            nn.GELU(),
            nn.Linear(input_size * ffn_expansion, input_size),
        )

        # Cross-attention: agents query full scene encoding
        self.dec_cross_norm = nn.LayerNorm(input_size)
        self.dec_cross_attn = nn.MultiheadAttention(
            input_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.dec_cross_ffn_norm = nn.LayerNorm(input_size)
        self.dec_cross_ffn = nn.Sequential(
            nn.Linear(input_size, input_size * ffn_expansion),
            nn.GELU(),
            nn.Linear(input_size * ffn_expansion, input_size),
        )

        # --- Output ---
        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, hidden_size)),
            nn.GELU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        device = observations.device
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        # 1. Parse flat observation
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim:ego_dim + partner_dim + road_dim]

        # 2. Reshape
        partner_objects = partner_obs.view(B, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(B, self.max_road_objects, self.road_features)

        # 3. Padding masks (True = ignore)
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        partner_mask = partner_objects.abs().sum(-1) == 0                 # (B, 31)
        road_mask = road_objects.abs().sum(-1) == 0                      # (B, 128)
        agent_mask = torch.cat([ego_mask, partner_mask], dim=1)          # (B, 32)

        # 4. One-hot road type
        road_continuous = road_objects[:, :, :self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # 5. Per-entity encoding
        ego_emb = self.ego_encoder(ego_obs).unsqueeze(1)         # (B, 1, D)
        partner_emb = self.partner_encoder(partner_objects)       # (B, 31, D)
        road_emb = self.road_encoder(road_objects)                # (B, 128, D)

        # 6. Type embeddings
        ego_emb = ego_emb + self.type_embed(torch.zeros(1, 1, dtype=torch.long, device=device))
        partner_emb = partner_emb + self.type_embed(torch.ones(1, 1, dtype=torch.long, device=device))
        road_emb = road_emb + self.type_embed(torch.full((1, 1), 2, dtype=torch.long, device=device))

        # 7. Concatenate all tokens: 1 ego + 31 partners + 128 roads = 160
        tokens = torch.cat([ego_emb, partner_emb, road_emb], dim=1)         # (B, 160, D)
        scene_mask = torch.cat([ego_mask, partner_mask, road_mask], dim=1)   # (B, 160)

        # === ENCODER: full self-attention ===
        enc_out = self.encoder(tokens, src_key_padding_mask=scene_mask)  # (B, 48, D)

        # === DECODER ===
        # Extract agent tokens from encoder output
        agents = enc_out[:, :self.num_agents]  # (B, 32, D)

        # Self-attention among agents (Pre-LN)
        agents_normed = self.dec_self_norm(agents)
        self_attn_out, _ = self.dec_self_attn(
            agents_normed, agents_normed, agents_normed,
            key_padding_mask=agent_mask,
        )
        agents = agents + self_attn_out
        agents = agents + self.dec_self_ffn(self.dec_self_ffn_norm(agents))

        # Cross-attention: agents query scene encoding (Pre-LN)
        agents_normed = self.dec_cross_norm(agents)
        cross_attn_out, _ = self.dec_cross_attn(
            agents_normed, enc_out, enc_out,
            key_padding_mask=scene_mask,
        )
        agents = agents + cross_attn_out
        agents = agents + self.dec_cross_ffn(self.dec_cross_ffn_norm(agents))

        # Extract ego token
        ego_out = agents[:, 0]  # (B, D)
        return self.output_proj(ego_out)  # (B, hidden_size)

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value


class DriveConditionedPaper(nn.Module):
    """Paper-exact Gigaflow MLP architecture with reward conditioning.

    Matches arXiv 2502.03349 App. C/D:
      - Paper App. C: "we do not share any parameters between the policy and
        the critic". Therefore actor and critic have their OWN per-entity
        encoders, their own backbones and their own heads — no shared weights.
      - Per-entity encoders: small 2-layer MLPs (ego, partner, road, creward),
        duplicated for actor and critic paths.
      - Permutation-invariant per-set encoding via channel-wise max-pool.
      - Separate 3-layer [hidden x hidden x hidden] backbones (paper App. D).
      - Feedforward. Use with `rnn_name = None` (no LSTM wrapper).

    Target parameter count: ~6M at input_size=128, hidden_size=1024.
    """

    def __init__(self, env, input_size=128, hidden_size=1024, **kwargs):
        super().__init__()
        from pufferlib.ocean.drive import binding
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.creward_dim = binding.CREWARD_FEATURES
        self.input_size = input_size

        # Build an independent MLP "tower" for one of {actor, critic}: per-entity
        # encoders + 3-layer backbone. Returns a nn.ModuleDict with the pieces
        # so encode_* can be run against either tower.
        def build_tower():
            tower = nn.ModuleDict()

            def entity_encoder(in_dim):
                return nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(in_dim, input_size)),
                    nn.LayerNorm(input_size),
                    nn.GELU(),
                    pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
                )
            tower["ego"] = entity_encoder(self.ego_dim)
            tower["partner"] = entity_encoder(self.partner_features)
            tower["road"] = entity_encoder(self.road_features_after_onehot)
            tower["creward"] = entity_encoder(self.creward_dim)

            backbone_in = 4 * input_size
            tower["backbone"] = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(backbone_in, hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.GELU(),
            )
            return tower

        self.actor_tower = build_tower()
        self.critic_tower = build_tower()

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def _parse_obs(self, observations):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        cre_dim = self.creward_dim

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]
        creward_obs = observations[:, ego_dim + partner_dim + road_dim
                                     : ego_dim + partner_dim + road_dim + cre_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects_oh = torch.cat([road_continuous, road_onehot], dim=2)
        return ego_obs, partner_objects, road_objects_oh, creward_obs

    def _tower_forward(self, tower, ego_obs, partner_objects, road_objects_oh, creward_obs):
        ego_feat = tower["ego"](ego_obs)
        partner_feat, _ = tower["partner"](partner_objects).max(dim=1)
        road_feat, _ = tower["road"](road_objects_oh).max(dim=1)
        creward_feat = tower["creward"](creward_obs)
        concat = torch.cat([ego_feat, partner_feat, road_feat, creward_feat], dim=1)
        return tower["backbone"](concat)

    def forward(self, observations, state=None, trunc_or_term_before=None, episode_ended=False):
        # Accept LSTMWrapper-compatible signature so pufferl can call this bare
        # policy when rnn_name=None. trunc_or_term_before/episode_ended are
        # unused by a feedforward policy.
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        # Run both towers independently. Pack outputs into a single tensor of
        # shape (B, 2*hidden_size) so that downstream code (RNN wrapper, decode)
        # can operate on a single "hidden" tensor; decode_actions splits it back.
        ego_obs, partner_objects, road_objects_oh, creward_obs = self._parse_obs(observations)
        actor_h = self._tower_forward(self.actor_tower, ego_obs, partner_objects, road_objects_oh, creward_obs)
        critic_h = self._tower_forward(self.critic_tower, ego_obs, partner_objects, road_objects_oh, creward_obs)
        return torch.cat([actor_h, critic_h], dim=1)

    def decode_actions(self, flat_hidden):
        actor_h, critic_h = torch.split(flat_hidden, self.hidden_size, dim=1)
        if self.is_continuous:
            parameters = self.actor(actor_h)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(actor_h)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(critic_h)
        return action, value


class DrivePaper(nn.Module):
    """Paper-exact Gigaflow MLP architecture without reward conditioning.

    Same independent-tower design as DriveConditionedPaper (arXiv 2502.03349
    App. C/D) but drops the creward branch. Ego dim auto-adapts to the env's
    dynamics model (7 for classic, 10 for jerk).
    """

    def __init__(self, env, input_size=128, hidden_size=1024, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.input_size = input_size

        def build_tower():
            tower = nn.ModuleDict()

            def entity_encoder(in_dim):
                return nn.Sequential(
                    pufferlib.pytorch.layer_init(nn.Linear(in_dim, input_size)),
                    nn.LayerNorm(input_size),
                    nn.GELU(),
                    pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
                )
            tower["ego"] = entity_encoder(self.ego_dim)
            tower["partner"] = entity_encoder(self.partner_features)
            tower["road"] = entity_encoder(self.road_features_after_onehot)

            backbone_in = 3 * input_size
            tower["backbone"] = nn.Sequential(
                pufferlib.pytorch.layer_init(nn.Linear(backbone_in, hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.GELU(),
                pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.GELU(),
            )
            return tower

        self.actor_tower = build_tower()
        self.critic_tower = build_tower()

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def _parse_obs(self, observations):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects_oh = torch.cat([road_continuous, road_onehot], dim=2)
        return ego_obs, partner_objects, road_objects_oh

    def _tower_forward(self, tower, ego_obs, partner_objects, road_objects_oh):
        ego_feat = tower["ego"](ego_obs)
        partner_feat, _ = tower["partner"](partner_objects).max(dim=1)
        road_feat, _ = tower["road"](road_objects_oh).max(dim=1)
        concat = torch.cat([ego_feat, partner_feat, road_feat], dim=1)
        return tower["backbone"](concat)

    def forward(self, observations, state=None, trunc_or_term_before=None, episode_ended=False):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        ego_obs, partner_objects, road_objects_oh = self._parse_obs(observations)
        actor_h = self._tower_forward(self.actor_tower, ego_obs, partner_objects, road_objects_oh)
        critic_h = self._tower_forward(self.critic_tower, ego_obs, partner_objects, road_objects_oh)
        return torch.cat([actor_h, critic_h], dim=1)

    def decode_actions(self, flat_hidden):
        actor_h, critic_h = torch.split(flat_hidden, self.hidden_size, dim=1)
        if self.is_continuous:
            parameters = self.actor(actor_h)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(actor_h)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(critic_h)
        return action, value


class DriveGameFormerConditioned(nn.Module):
    """DriveGameFormer + reward conditioning (Gigaflow paper S. 14).

    Adds one extra token holding the 9-D Creward vector with its own type id,
    so the Transformer encoder and decoder can attend to the reward conditioning
    signal like any other entity. Architecture otherwise identical to
    DriveGameFormer. Use with env.reward_conditioning=True and obs-dim 1163.
    """

    def __init__(self, env, input_size=128, hidden_size=128,
                 num_heads=4, num_enc_layers=2, ffn_expansion=2, dropout=0.0,
                 **kwargs):
        super().__init__()
        from pufferlib.ocean.drive import binding
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6
        self.num_agents = 1 + self.max_partner_objects  # ego + partners
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.creward_dim = binding.CREWARD_FEATURES

        # --- Per-entity encoders ---
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.creward_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.creward_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        # Type embeddings: 0=ego, 1=partner, 2=road, 3=creward
        self.type_embed = nn.Embedding(4, input_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=input_size * ffn_expansion,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_enc_layers, enable_nested_tensor=False,
        )

        # --- Decoder: self-attention among agents + cross-attention to scene ---
        self.dec_self_norm = nn.LayerNorm(input_size)
        self.dec_self_attn = nn.MultiheadAttention(
            input_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.dec_self_ffn_norm = nn.LayerNorm(input_size)
        self.dec_self_ffn = nn.Sequential(
            nn.Linear(input_size, input_size * ffn_expansion),
            nn.GELU(),
            nn.Linear(input_size * ffn_expansion, input_size),
        )

        self.dec_cross_norm = nn.LayerNorm(input_size)
        self.dec_cross_attn = nn.MultiheadAttention(
            input_size, num_heads, dropout=dropout, batch_first=True,
        )
        self.dec_cross_ffn_norm = nn.LayerNorm(input_size)
        self.dec_cross_ffn = nn.Sequential(
            nn.Linear(input_size, input_size * ffn_expansion),
            nn.GELU(),
            nn.Linear(input_size * ffn_expansion, input_size),
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, hidden_size)),
            nn.GELU(),
        )

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def encode_observations(self, observations, state=None):
        B = observations.shape[0]
        device = observations.device
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        cre_dim = self.creward_dim

        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim:ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim:ego_dim + partner_dim + road_dim]
        creward_obs = observations[:, ego_dim + partner_dim + road_dim
                                     : ego_dim + partner_dim + road_dim + cre_dim]

        partner_objects = partner_obs.view(B, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(B, self.max_road_objects, self.road_features)

        # Padding masks (True = ignore)
        ego_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        partner_mask = partner_objects.abs().sum(-1) == 0
        road_mask = road_objects.abs().sum(-1) == 0
        creward_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        agent_mask = torch.cat([ego_mask, partner_mask], dim=1)

        road_continuous = road_objects[:, :, :self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        # Per-entity encoding
        ego_emb = self.ego_encoder(ego_obs).unsqueeze(1)
        partner_emb = self.partner_encoder(partner_objects)
        road_emb = self.road_encoder(road_objects)
        creward_emb = self.creward_encoder(creward_obs).unsqueeze(1)

        # Type embeddings
        ego_emb = ego_emb + self.type_embed(torch.zeros(1, 1, dtype=torch.long, device=device))
        partner_emb = partner_emb + self.type_embed(torch.ones(1, 1, dtype=torch.long, device=device))
        road_emb = road_emb + self.type_embed(torch.full((1, 1), 2, dtype=torch.long, device=device))
        creward_emb = creward_emb + self.type_embed(torch.full((1, 1), 3, dtype=torch.long, device=device))

        # Concatenate: 1 ego + 31 partners + 128 roads + 1 creward = 161 tokens
        tokens = torch.cat([ego_emb, partner_emb, road_emb, creward_emb], dim=1)
        scene_mask = torch.cat([ego_mask, partner_mask, road_mask, creward_mask], dim=1)

        enc_out = self.encoder(tokens, src_key_padding_mask=scene_mask)
        agents = enc_out[:, :self.num_agents]

        # Self-attention among agents
        agents_normed = self.dec_self_norm(agents)
        self_attn_out, _ = self.dec_self_attn(
            agents_normed, agents_normed, agents_normed,
            key_padding_mask=agent_mask,
        )
        agents = agents + self_attn_out
        agents = agents + self.dec_self_ffn(self.dec_self_ffn_norm(agents))

        # Cross-attention: agents query full scene encoding (incl. creward token)
        agents_normed = self.dec_cross_norm(agents)
        cross_attn_out, _ = self.dec_cross_attn(
            agents_normed, enc_out, enc_out,
            key_padding_mask=scene_mask,
        )
        agents = agents + cross_attn_out
        agents = agents + self.dec_cross_ffn(self.dec_cross_ffn_norm(agents))

        ego_out = agents[:, 0]
        return self.output_proj(ego_out)

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value


class DriveLatentWorldModel(nn.Module):
    """Drive policy with a latent world model for N-step imagination.

    Encoder maps observations to latent z_t, from which action and value are predicted.
    A transition model predicts z_t from (z_{t-1}, a_{t-1}), enabling multi-step
    rollouts without stepping through the environment.
    """

    def __init__(self, env, input_size=128, hidden_size=256, transition_loss_coef=1.0, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.transition_loss_coef = transition_loss_coef
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6

        # Encoder (identical to Drive)
        self.ego_dim = 10 if env.dynamics_model == "jerk" else 7
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )
        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )

        # Action space
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        # Transition model: (z_{t-1}, action_embedding) -> z_t_hat
        action_input_dim = sum(self.atn_dim)  # one-hot encoded actions
        self.transition_model = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size + action_input_dim, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
        )

        # Actor and value heads
        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)
        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)

        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)
        embedding = F.relu(self.shared_embedding(concat_features))
        return embedding

    def transition(self, z_prev, action_prev):
        """Predict next latent from previous latent and action."""
        action_onehots = []
        offset = 0
        for dim_size in self.atn_dim:
            a = action_prev[:, offset] if action_prev.dim() > 1 else action_prev
            action_onehots.append(F.one_hot(a.long(), num_classes=dim_size).float())
            offset += 1
        action_embed = torch.cat(action_onehots, dim=-1)
        transition_input = torch.cat([z_prev, action_embed], dim=-1)
        return self.transition_model(transition_input)

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        value = self.value_fn(flat_hidden)
        return action, value

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_eval(self, x, state=None):
        return self.forward(x, state)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def imagine(self, z_0, n_steps):
        """Generate N-step action proposals from initial latent z_0 (for evaluation)."""
        z = z_0
        actions_list = []
        values_list = []
        for _ in range(n_steps):
            logits, value = self.decode_actions(z)
            action = pufferlib.pytorch.sample_logits(logits)[0]
            actions_list.append(action)
            values_list.append(value)
            z = self.transition(z, action)
        return actions_list, values_list


class LatentWorldModelWrapper(nn.Module):
    """Wrapper for DriveLatentWorldModel following the LSTMWrapper protocol.

    During eval: encodes obs, stores latent in state for continuity.
    During training: iterates over timesteps, computes transition consistency loss.
    """

    def __init__(self, env, policy, input_size=256, hidden_size=256,
                 imagination_horizon=4, **kwargs):
        super().__init__()
        self.policy = policy
        self.obs_shape = env.single_observation_space.shape
        self.hidden_size = hidden_size
        self.is_continuous = self.policy.is_continuous
        self.atn_dim = self.policy.atn_dim
        self.imagination_horizon = imagination_horizon
        self._transition_loss = None

    def forward_eval(self, observations, state):
        z = self.policy.encode_observations(observations)
        state["lstm_h"] = z.detach()
        state["lstm_c"] = torch.zeros_like(z)
        logits, value = self.policy.decode_actions(z)
        return logits, value

    def forward(self, observations, state, trunc_or_term_before, episode_ended=False):
        device = observations.device
        x = observations
        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        values = torch.zeros(B, TT, device=device)
        logits = torch.zeros(B, TT, sum(self.atn_dim), device=device)
        transition_losses = []

        z_prev = state["lstm_h"]  # latent from previous BPTT window (or None)
        K = self.imagination_horizon
        steps_since_encode = 0  # counts steps since last encoder re-anchor

        for t in range(TT):
            mask = trunc_or_term_before[:, t] == 1.0
            if z_prev is not None and mask.any():
                z_prev = torch.where(mask.unsqueeze(-1), torch.zeros_like(z_prev), z_prev)

            # Re-anchor from encoder every K steps, at first step, or after episode reset
            use_encoder = (z_prev is None) or (steps_since_encode >= K) or mask.any()

            if use_encoder:
                z_t_encoded = self.policy.encode_observations(x[:, t, :])
                if t > 0 and z_prev is not None and not mask.all():
                    # Transition step for non-reset agents
                    a_prev = state["action"][:, t - 1, :]
                    z_t_trans = self.policy.transition(z_prev, a_prev)
                    # Consistency loss: transition should match encoder
                    valid = ~mask
                    if valid.any():
                        loss = F.mse_loss(z_t_trans[valid], z_t_encoded.detach()[valid])
                        transition_losses.append(loss)
                # Use encoder output and reset imagination counter
                z_t = z_t_encoded
                steps_since_encode = 0
            else:
                # Imagination: use transition model only
                a_prev = state["action"][:, t - 1, :]
                z_t = self.policy.transition(z_prev, a_prev)
                # Consistency loss against encoder target
                with torch.no_grad():
                    z_t_encoded = self.policy.encode_observations(x[:, t, :])
                valid = ~mask
                if valid.any():
                    loss = F.mse_loss(z_t[valid], z_t_encoded[valid])
                    transition_losses.append(loss)

            logits_t, value_t = self.policy.decode_actions(z_t)
            logits[:, t, :] = logits_t[0] if isinstance(logits_t, tuple) else logits_t
            values[:, t] = value_t.flatten()

            z_prev = z_t  # no detach within imagination window
            steps_since_encode += 1

        state["lstm_h"] = z_prev.detach()
        state["lstm_c"] = torch.zeros_like(z_prev)

        if episode_ended:
            state["lstm_h"] = state["lstm_h"].detach()
            state["lstm_c"] = state["lstm_c"].detach()

        if transition_losses:
            self._transition_loss = self.policy.transition_loss_coef * torch.stack(transition_losses).mean()
        else:
            self._transition_loss = torch.tensor(0.0, device=device)

        logits = logits.reshape(B * TT, sum(self.atn_dim))
        logits_tuple = (logits,)
        return logits_tuple, values

    def compute_auxiliary_loss(self):
        loss = self._transition_loss if self._transition_loss is not None else torch.tensor(0.0)
        logs = {"transition_loss": loss.item()}
        self._transition_loss = None
        return loss, logs
