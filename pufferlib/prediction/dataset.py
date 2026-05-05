# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Datasets for prediction model training.

WaymoBinaryDataset: Loads .bin files for supervised pretraining.
OnPolicyDataset: Loads .pt files from transition collection for fine-tuning.
"""

import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch_geometric.data import HeteroData, Batch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pufferlib.prediction.binary_reader import read_binary_scenario, list_binary_files
from pufferlib.prediction.trajectory_tokenizer import compute_token_data, load_motion_codebook
from pufferlib.prediction.map_tokenizer import tokenize_roads, load_map_codebook


class WaymoBinaryDataset(Dataset):
    """Dataset that loads Waymo .bin files and produces HeteroData for training.

    Processes each scenario into the format expected by the SMART-based model:
    - Agent trajectories with positions, headings, velocities
    - GT motion tokens via polygon contour matching (original SMART codebook)
    - Map tokens via road polyline tokenization
    - Vehicles (type==1), pedestrians (type==2), and cyclists (type==3) are included.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'training',
                 num_historical_steps: int = 11,
                 num_future_steps: int = 80,
                 max_agents: int = -1,
                 max_files: int = -1,
                 shift: int = 5,
                 cache_dir: Optional[str] = None,
                 num_actions: int = -1):
        self.data_dir = data_dir
        self.split = split
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.max_agents = max_agents
        self.shift = shift
        # Handle string "None" from .ini config
        self.cache_dir = None if cache_dir in (None, 'None', '') else cache_dir
        self.cache_hits = 0
        self.cache_misses = 0

        self.file_list = list_binary_files(data_dir, split)
        if max_files > 0:
            self.file_list = self.file_list[:max_files]

        # Log cache status
        if self.cache_dir:
            cache_split_dir = os.path.join(self.cache_dir, self.split)
            if os.path.isdir(cache_split_dir):
                cached = len([f for f in os.listdir(cache_split_dir) if f.endswith('.pt')])
            else:
                cached = 0
            print(f"  [{split}] {len(self.file_list)} files, "
                  f"cache: {cached}/{len(self.file_list)} "
                  f"({cache_split_dir})")
        else:
            print(f"  [{split}] {len(self.file_list)} files, cache: disabled")

        self.map_codebook = load_map_codebook()
        motion_data = load_motion_codebook()
        codebook_veh = motion_data['token']['veh']  # (2048, 4, 2)
        if num_actions > 0 and num_actions < codebook_veh.shape[0]:
            self.motion_codebook_veh = codebook_veh[:num_actions]
        else:
            self.motion_codebook_veh = codebook_veh

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> HeteroData:
        filepath = self.file_list[idx]

        # Check cache (split subdirectory prevents train/val collision)
        if self.cache_dir:
            cache_path = os.path.join(
                self.cache_dir, self.split,
                Path(filepath).stem + '.pt')
            if os.path.exists(cache_path):
                self.cache_hits += 1
                return torch.load(cache_path, weights_only=False)
            self.cache_misses += 1

        scenario = read_binary_scenario(filepath)
        data = self._process_scenario(scenario)

        # Save to cache
        if self.cache_dir:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(data, cache_path)

        return data

    def _process_scenario(self, scenario: dict) -> HeteroData:
        """Convert binary scenario to HeteroData."""
        objects = scenario['objects']
        roads = scenario['roads']
        sdc_index = scenario['sdc_track_index']

        T = self.num_historical_steps + self.num_future_steps  # 91

        # --- Filter to vehicles, pedestrians, and cyclists (types 1, 2, 3) ---
        agent_indices = [i for i, obj in enumerate(objects) if obj['type'] in (1, 2, 3)]
        if len(agent_indices) == 0:
            agent_indices = [sdc_index] if sdc_index >= 0 else [0]
        objects = [objects[i] for i in agent_indices]
        # Update sdc_index
        if sdc_index in agent_indices:
            sdc_index = agent_indices.index(sdc_index)
        else:
            sdc_index = 0

        N = len(objects)

        # Select closest agents to SDC if max_agents limit is set
        if self.max_agents > 0 and N > self.max_agents and sdc_index >= 0:
            sdc_obj = objects[sdc_index]
            sdc_x = sdc_obj['traj_x'][self.num_historical_steps - 1]
            sdc_y = sdc_obj['traj_y'][self.num_historical_steps - 1]
            distances = []
            for obj in objects:
                ox = obj['traj_x'][self.num_historical_steps - 1]
                oy = obj['traj_y'][self.num_historical_steps - 1]
                distances.append((ox - sdc_x)**2 + (oy - sdc_y)**2)
            sorted_indices = np.argsort(distances)[:self.max_agents]
            objects = [objects[i] for i in sorted_indices]
            sdc_index = int(np.where(sorted_indices == sdc_index)[0][0]) if sdc_index in sorted_indices else 0
            N = len(objects)

        # Build agent tensors (per-step, full T=91 resolution)
        position = torch.zeros(N, T, 2)
        heading = torch.zeros(N, T)
        velocity = torch.zeros(N, T, 2)
        valid_mask = torch.zeros(N, T, dtype=torch.bool)
        agent_type = torch.zeros(N, dtype=torch.long)
        shape = torch.zeros(N, T, 3)  # width, length, height

        for i, obj in enumerate(objects):
            t_max = min(len(obj['traj_x']), T)
            position[i, :t_max, 0] = torch.from_numpy(obj['traj_x'][:t_max])
            position[i, :t_max, 1] = torch.from_numpy(obj['traj_y'][:t_max])
            heading[i, :t_max] = torch.from_numpy(obj['traj_heading'][:t_max])
            velocity[i, :t_max, 0] = torch.from_numpy(obj['traj_vx'][:t_max])
            velocity[i, :t_max, 1] = torch.from_numpy(obj['traj_vy'][:t_max])
            valid_mask[i, :t_max] = torch.from_numpy(
                obj['traj_valid'][:t_max]).bool()
            # Map raw Waymo types to SMART indices: 1->0 (veh), 2->1 (ped), 3->2 (cyc)
            agent_type[i] = obj['type'] - 1
            shape[i, :, 0] = obj['width']
            shape[i, :, 1] = obj['length']
            shape[i, :, 2] = obj['height']

        # --- Determine scene center (SDC at last historical step) ---
        t_center = self.num_historical_steps - 1
        sdc_i = max(0, min(sdc_index, N - 1))
        if valid_mask[sdc_i, t_center]:
            center_pos = position[sdc_i, t_center].clone()
        else:
            valid_t = valid_mask[sdc_i].nonzero(as_tuple=False)
            if len(valid_t) > 0:
                center_pos = position[sdc_i, valid_t[0].item()].clone()
            else:
                center_pos = torch.zeros(2)

        # --- Fill invalid positions with nearest valid per agent ---
        for i in range(N):
            v = valid_mask[i]
            if not v.any():
                position[i] = center_pos
                continue
            first_valid = v.nonzero(as_tuple=False)[0].item()
            if first_valid > 0:
                position[i, :first_valid] = position[i, first_valid]
                heading[i, :first_valid] = heading[i, first_valid]
                velocity[i, :first_valid] = velocity[i, first_valid]
            for t in range(first_valid + 1, T):
                if not v[t]:
                    position[i, t] = position[i, t - 1]
                    heading[i, t] = heading[i, t - 1]
                    velocity[i, t] = velocity[i, t - 1]

        # --- Heading cleaning (matching original SMART) ---
        # If heading jumps by more than 1.0 rad between consecutive valid
        # frames, copy previous heading forward.
        for i in range(N):
            v = valid_mask[i]
            if not v.any():
                continue
            prev_t = v.nonzero(as_tuple=False)[0].item()
            for t in range(prev_t + 1, T):
                if v[t]:
                    hdiff = abs(float(heading[i, t] - heading[i, prev_t]))
                    hdiff = min(hdiff, 2 * math.pi - hdiff)
                    if hdiff > 1.0:
                        heading[i, t] = heading[i, prev_t]
                    prev_t = t

        # --- Compute GT motion tokens via codebook matching ---
        # Pass filled (un-centered) positions/headings so agents with
        # invalid t=0 data still get correct matching.
        token_data = compute_token_data(
            positions=position.numpy(),
            headings=heading.numpy(),
            valid_masks=valid_mask.numpy(),
            shift=self.shift, codebook=self.motion_codebook_veh,
            agent_types=agent_type.numpy())
        # token_data keys: token_idx (N, 18)

        # --- Center all agent positions on scene center ---
        position = position - center_pos.unsqueeze(0).unsqueeze(0)

        # --- Token positions and headings from codebook matching ---
        # Use matched contour positions/headings (not GT) to match original SMART.
        # The model must learn to predict next tokens from the same distribution
        # it sees during autoregressive inference.
        shift_indices = list(range(self.shift, T, self.shift))  # [5, 10, ..., 90]
        num_tokens = len(shift_indices)
        # Center matched positions the same way as GT positions
        token_pos = token_data['token_pos'] - center_pos.unsqueeze(0)  # (N, 18, 2)
        token_heading_gt = token_data['token_heading']                  # (N, 18)
        token_valid = torch.zeros(N, num_tokens, dtype=torch.bool)
        for k, t_idx in enumerate(shift_indices):
            t_prev = t_idx - self.shift
            token_valid[:, k] = valid_mask[:, t_idx] & valid_mask[:, t_prev]

        # Category: 3 for agents to predict, 0 for background
        # Following original SMART: max 32 prediction targets, closest to SDC
        category = torch.zeros(N, dtype=torch.long)
        has_future = valid_mask[:, self.num_historical_steps:].any(dim=1)
        vehicle2pred = has_future.clone()
        if sdc_index >= 0 and sdc_index < N:
            vehicle2pred[sdc_index] = False  # SDC gets category=5 separately
        max_pred_agents = 32
        if vehicle2pred.sum() > max_pred_agents:
            # Keep closest 32 to SDC (matching original SMART clip logic)
            sdc_pos = position[max(0, sdc_index), self.num_historical_steps - 1, :2]
            dist = torch.norm(position[:, self.num_historical_steps - 1, :2] - sdc_pos, dim=-1)
            dist[~vehicle2pred] = float('inf')
            _, sorted_idx = dist.sort()
            vehicle2pred.fill_(False)
            vehicle2pred[sorted_idx[:max_pred_agents]] = True
        category[vehicle2pred] = 3
        if sdc_index >= 0 and sdc_index < N:
            category[sdc_index] = 5

        # Tokenize map
        map_data = tokenize_roads(roads, self.map_codebook)

        # --- Center map token positions on scene center ---
        map_data['pt_token']['position'][:, :2] -= center_pos
        map_data['map_save']['traj_pos'] -= center_pos.unsqueeze(0)

        # Build HeteroData
        data = HeteroData()

        # Agent data (per-step, full resolution for GT evaluation)
        data['agent'] = {}
        data['agent']['position'] = position      # (N, 91, 2)
        data['agent']['heading'] = heading         # (N, 91)
        data['agent']['velocity'] = velocity       # (N, 91, 2)
        data['agent']['valid_mask'] = valid_mask   # (N, 91)
        data['agent']['type'] = agent_type         # (N,)
        data['agent']['shape'] = shape             # (N, 91, 3)
        data['agent']['category'] = category       # (N,)
        data['agent']['num_nodes'] = N
        data['agent']['av_index'] = max(0, sdc_index)

        # Token-level data for agent decoder
        # Use GT positions/headings at shift boundaries (already centered)
        data['agent']['token_pos'] = token_pos                        # (N, 18, 2)
        data['agent']['token_heading'] = token_heading_gt             # (N, 18)
        data['agent']['token_idx'] = token_data['token_idx']          # (N, 18)
        data['agent']['agent_valid_mask'] = token_valid               # (N, 18)

        # Map token data
        for key, val in map_data['pt_token'].items():
            data['pt_token'][key] = val

        # Map visualization data
        data['pt_token']['traj_pos'] = map_data['map_save']['traj_pos']
        data['pt_token']['polygon_idx'] = map_data['token2pl_edge_index'][1]

        return data


class OnPolicyDataset(Dataset):
    """Dataset for fine-tuning on policy-collected transitions."""

    def __init__(self, transition_dir: str, max_files: int = -1):
        self.transition_dir = transition_dir
        self.file_list = sorted(Path(transition_dir).glob('*.pt'))
        if max_files > 0:
            self.file_list = self.file_list[:max_files]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> HeteroData:
        return torch.load(str(self.file_list[idx]), weights_only=False)


def create_dataloaders(config: dict, mode: str = 'pretrain',
                       distributed: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        config: Configuration dict.
        mode: 'pretrain' or 'finetune'.
        distributed: If True, use DistributedSampler for multi-GPU training.
    """
    from torch.utils.data.distributed import DistributedSampler

    data_cfg = config.get('data', config)
    model_cfg = config.get('model', config)
    batch_size = int(data_cfg.get('batch_size', 32))
    num_workers = int(data_cfg.get('num_workers', 8))
    shift = int(data_cfg.get('shift', 5))
    num_actions = int(model_cfg.get('num_actions', -1))

    if mode == 'pretrain':
        train_dataset = WaymoBinaryDataset(
            data_dir=data_cfg.get('data_dir', 'experiments/prediction'),
            split=data_cfg.get('train_split', 'training'),
            num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
            num_future_steps=int(data_cfg.get('num_future_steps', 80)),
            max_agents=int(data_cfg.get('max_agents', -1)),
            max_files=int(data_cfg.get('max_train_files', -1)),
            shift=shift,
            cache_dir=data_cfg.get('cache_dir', None),
            num_actions=num_actions,
        )
        val_dataset = WaymoBinaryDataset(
            data_dir=data_cfg.get('data_dir', 'experiments/prediction'),
            split=data_cfg.get('val_split', 'validation'),
            num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
            num_future_steps=int(data_cfg.get('num_future_steps', 80)),
            max_agents=int(data_cfg.get('max_agents', -1)),
            max_files=int(data_cfg.get('max_val_files', 500)),
            shift=shift,
            cache_dir=data_cfg.get('cache_dir', None),
            num_actions=num_actions,
        )
    elif mode == 'finetune':
        finetune_cfg = config.get('finetune', config)
        transition_dir = finetune_cfg.get('transition_dir', 'experiments/transitions')
        mix_ratio = float(finetune_cfg.get('mix_ratio', 0.5))

        onpolicy_dataset = OnPolicyDataset(transition_dir=transition_dir)

        if mix_ratio < 1.0 and mix_ratio > 0.0:
            waymo_dataset = WaymoBinaryDataset(
                data_dir=data_cfg.get('data_dir', 'experiments/prediction'),
                split=data_cfg.get('train_split', 'training'),
                num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
                num_future_steps=int(data_cfg.get('num_future_steps', 80)),
                max_agents=int(data_cfg.get('max_agents', -1)),
                max_files=int(len(onpolicy_dataset) * (1 - mix_ratio) / mix_ratio),
                shift=shift,
                cache_dir=data_cfg.get('cache_dir', None),
                num_actions=num_actions,
            )
            train_dataset = ConcatDataset([onpolicy_dataset, waymo_dataset])
        else:
            train_dataset = onpolicy_dataset

        val_dataset = WaymoBinaryDataset(
            data_dir=data_cfg.get('data_dir', 'experiments/prediction'),
            split=data_cfg.get('val_split', 'validation'),
            num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
            num_future_steps=int(data_cfg.get('num_future_steps', 80)),
            max_agents=int(data_cfg.get('max_agents', -1)),
            max_files=int(data_cfg.get('max_val_files', 500)),
            shift=shift,
            cache_dir=data_cfg.get('cache_dir', None),
            num_actions=num_actions,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Samplers for distributed training
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_hetero,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_hetero,
    )
    return train_loader, val_loader


def _collate_hetero(batch: List[HeteroData]) -> Batch:
    """Custom collate function for HeteroData batching."""
    return Batch.from_data_list(batch)
