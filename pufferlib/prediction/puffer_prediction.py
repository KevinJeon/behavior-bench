# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Training script for SMART-based prediction model.

Supports three commands:
- warmup-cache: Pre-build dataset cache with multiprocessing
- pretrain: Supervised pretraining on Waymo .bin files
- finetune: Fine-tuning on policy-collected transitions

Usage:
    # 1. Build cache (recommended before training, uses 80 workers by default)
    python -m pufferlib.prediction.puffer_prediction warmup-cache --config config/prediction/smart.ini

    # 2. Train (single GPU)
    python -m pufferlib.prediction.puffer_prediction pretrain --config config/prediction/smart.ini

    # 2. Train (multi-GPU)
    torchrun --nproc_per_node=8 -m pufferlib.prediction.puffer_prediction pretrain --config config/prediction/smart.ini
"""

import argparse
import configparser
import math
import os
import time

import matplotlib
matplotlib.use('Agg')  # non-interactive backend to avoid tkinter threading issues

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from pufferlib.prediction.dataset import create_dataloaders


def _create_model(model_cfg):
    """Create prediction model based on model_type in config."""
    model_type = model_cfg.get('model_type', 'smart')
    if model_type == 'simpl':
        from pufferlib.prediction.simpl.prediction_model import SIMPLPredictionModel
        return SIMPLPredictionModel(model_cfg)
    elif model_type == 'baseline':
        from pufferlib.prediction.baseline.prediction_model import BaselinePredictionModel
        return BaselinePredictionModel(model_cfg)
    else:
        from pufferlib.prediction.smart.prediction_model import PredictionModel
        return PredictionModel(model_cfg)


# Road visualization colors and styles (matching viz.py)
ROAD_COLORS = {
    4: (0.83, 0.83, 0.83),   # Lane - light gray
    5: (0.31, 0.31, 0.31),   # Line - gray
    6: (0.0, 0.0, 0.0),      # Edge - black
    7: (1.0, 0.0, 0.0),      # Stop sign - red
}
ROAD_LW = {4: 1, 5: 1, 6: 2}
ROAD_LS = {4: '-', 5: '--', 6: '-'}

AGENT_COLORS = {
    0: np.array([65, 105, 225]) / 255,    # Vehicle - Royal Blue
    1: np.array([0, 200, 0]) / 255,       # Pedestrian - Green
    2: np.array([255, 0, 255]) / 255,     # Cyclist - Magenta
}
PRED_COLORS = ['#e6194b', '#f58231', '#911eb4', '#42d4f4', '#f032e6']


def _draw_box(ax, x, y, length, width, heading, color, alpha=0.7):
    """Draw an oriented bounding box with heading arrow."""
    c, s = np.cos(heading), np.sin(heading)
    u = np.array([c, s])
    ut = np.array([s, -c])
    pt = np.array([x, y])

    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    ax.plot([tl[0], tr[0], br[0], bl[0], tl[0]],
            [tl[1], tr[1], br[1], bl[1], tl[1]],
            color=color, alpha=alpha, linewidth=1, zorder=4)

    # Heading arrow
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u
    ax.plot([cl[0], cf[0], cr[0]], [cl[1], cf[1], cr[1]],
            color=color, alpha=alpha, linewidth=1, zorder=4)


def setup_distributed():
    """Initialize distributed training if launched via torchrun."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


class PufferPrediction:
    """Training manager for the prediction model."""

    def __init__(self, config, model, train_loader, val_loader, device='cuda',
                 wandb_run=None, local_rank=0):
        self.config = config
        self.device = device
        self.wandb_run = wandb_run
        self.local_rank = local_rank

        # Move model to device
        self.model = model.to(device)

        # Wrap in DDP if distributed
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[local_rank],
                             find_unused_parameters=True)
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

        self.train_loader = train_loader
        self.val_loader = val_loader

        train_cfg = config.get('train', config)
        self.lr = float(train_cfg.get('learning_rate', 1e-3))
        self.max_grad_norm = float(train_cfg.get('max_grad_norm', 0.5))
        self.warmup_epochs = int(train_cfg.get('warmup_epochs', 5))
        self.checkpoint_interval = int(train_cfg.get('checkpoint_interval', 5))
        self.checkpoint_dir = train_cfg.get('checkpoint_dir',
                                            train_cfg.get('data_dir', 'experiments/prediction'))

        self.weight_decay = float(train_cfg.get('weight_decay', 0.1))
        if self.weight_decay > 0:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        # Use fp32 (matching original SMART precision: 32)
        self.use_amp = str(train_cfg.get('use_amp', 'false')).lower() in ('true', '1', 'yes')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.epoch = 0
        self.global_step = 0

        max_epochs = int(train_cfg.get('max_epochs', 100))
        warmup_steps = self.warmup_epochs
        total_steps = max_epochs
        decay_epoch = int(train_cfg.get('decay_epoch', 0))

        if decay_epoch > 0:
            # SIMPL polyline schedule: warmup 1e-4->1e-3, hold, decay to 1e-4
            init_lr = 1e-4
            peak_lr = self.lr
            final_lr = 1e-4
            init_ratio = init_lr / peak_lr
            final_ratio = final_lr / peak_lr

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup from init_lr to peak_lr
                    return init_ratio + (1.0 - init_ratio) * current_step / max(1, warmup_steps)
                elif current_step < decay_epoch:
                    # Hold at peak_lr
                    return 1.0
                else:
                    # Linear decay from peak_lr to final_lr
                    progress = (current_step - decay_epoch) / max(1, total_steps - decay_epoch)
                    return 1.0 - (1.0 - final_ratio) * min(progress, 1.0)
        else:
            # SMART cosine schedule: linear warmup + cosine decay
            lr_min_ratio = float(train_cfg.get('lr_min_ratio', 0.01))

            def lr_lambda(current_step):
                if current_step + 1 < warmup_steps:
                    return float(current_step + 1) / float(max(1, warmup_steps))
                return max(lr_min_ratio, 0.5 * (1.0 + math.cos(
                    math.pi * (current_step - warmup_steps)
                    / float(max(1, total_steps - warmup_steps)))))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda)

    def train_epoch(self) -> dict:
        """Run one training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        total_agent_loss = 0
        total_map_loss = 0
        total_acc = 0
        num_batches = 0
        total_batches = len(self.train_loader)

        # Set epoch for distributed sampler
        if hasattr(self.train_loader, 'sampler') and \
                isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.epoch)

        nan_batches = 0
        import time as _time
        _t_epoch = _time.perf_counter()
        _t_last_log = _t_epoch

        for batch in self.train_loader:
            batch = batch.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.model(batch)
                losses = self.raw_model.compute_loss(pred)
                loss = losses['loss']

            # Skip NaN/Inf batches to prevent poisoning the model
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_agent_loss += losses['agent_cls_loss'].item()
            total_map_loss += losses['map_cls_loss'].item()
            total_acc += losses['agent_accuracy'].item()
            num_batches += 1
            self.global_step += 1

            if self.wandb_run and num_batches % 100 == 0:
                self.wandb_run.log({
                    'train/loss_step': loss.item(),
                    'train/agent_loss_step': losses['agent_cls_loss'].item(),
                    'train/accuracy_step': losses['agent_accuracy'].item(),
                    'global_step': self.global_step,
                })

            # Progress bar
            _t_now = _time.perf_counter()
            if is_main_process() and _t_now - _t_last_log >= 30:
                _elapsed = _t_now - _t_epoch
                _pct = 100 * num_batches / total_batches
                _steps_per_sec = num_batches / _elapsed
                _eta = (_elapsed / num_batches) * (total_batches - num_batches)
                _avg_loss = total_loss / num_batches
                _avg_acc = total_acc / num_batches
                # Cache stats from dataset
                _ds = self.train_loader.dataset
                _cache_info = ""
                if hasattr(_ds, 'cache_hits'):
                    _total_access = _ds.cache_hits + _ds.cache_misses
                    if _total_access > 0:
                        _hit_rate = 100 * _ds.cache_hits / _total_access
                        _cache_info = f" | cache: {_hit_rate:.0f}%"
                print(f"\r  Epoch {self.epoch} [{num_batches}/{total_batches} "
                      f"({_pct:.0f}%)] {_steps_per_sec:.1f} it/s | "
                      f"loss={_avg_loss:.4f} acc={_avg_acc:.4f} | "
                      f"ETA: {_eta/60:.0f}min{_cache_info}",
                      end="", flush=True)
                _t_last_log = _t_now

        if is_main_process():
            _elapsed = _time.perf_counter() - _t_epoch
            _ds = self.train_loader.dataset
            _cache_info = ""
            if hasattr(_ds, 'cache_hits'):
                _total_access = _ds.cache_hits + _ds.cache_misses
                if _total_access > 0:
                    _hit_rate = 100 * _ds.cache_hits / _total_access
                    _cache_info = (f" | cache: {_ds.cache_hits}/{_total_access}"
                                   f" ({_hit_rate:.0f}% hit)")
            print(f"\r  Epoch {self.epoch} done: {num_batches} batches "
                  f"in {_elapsed/60:.1f}min "
                  f"({num_batches/_elapsed:.1f} it/s){_cache_info}")

        if nan_batches > 0 and is_main_process():
            print(f"  Warning: {nan_batches} NaN batches skipped")

        metrics = {
            'train_loss': total_loss / max(num_batches, 1),
            'train_agent_loss': total_agent_loss / max(num_batches, 1),
            'train_map_loss': total_map_loss / max(num_batches, 1),
            'train_accuracy': total_acc / max(num_batches, 1),
        }
        self.epoch += 1
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation (teacher-forced loss + accuracy)."""
        self.raw_model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        total_batches = len(self.val_loader)

        import time as _time
        _t_start = _time.perf_counter()
        _t_last_log = _t_start

        for batch in self.val_loader:
            batch = batch.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred = self.raw_model(batch)
                losses = self.raw_model.compute_loss(pred)

            total_loss += losses['loss'].item()
            total_acc += losses['agent_accuracy'].item()
            num_batches += 1

            _t_now = _time.perf_counter()
            if is_main_process() and _t_now - _t_last_log >= 30:
                _pct = 100 * num_batches / total_batches
                _steps_per_sec = num_batches / (_t_now - _t_start)
                _eta = ((_t_now - _t_start) / num_batches) * (total_batches - num_batches)
                print(f"\r  Val [{num_batches}/{total_batches} ({_pct:.0f}%)] "
                      f"{_steps_per_sec:.1f} it/s | ETA: {_eta/60:.0f}min",
                      end="", flush=True)
                _t_last_log = _t_now

        if is_main_process() and total_batches > 0:
            _elapsed = _time.perf_counter() - _t_start
            print(f"\r  Val done: {num_batches} batches in "
                  f"{_elapsed/60:.1f}min ({num_batches/_elapsed:.1f} it/s)")

        return {
            'val_loss': total_loss / max(num_batches, 1),
            'val_accuracy': total_acc / max(num_batches, 1),
        }

    @torch.no_grad()
    def compute_trajectory_metrics(self, loader, max_batches=None, ks=(1, 5)) -> dict:
        """Compute minADE/minFDE for k=1,5 via autoregressive rollout.

        Args:
            loader: DataLoader to evaluate on.
            max_batches: Limit number of batches (None = full dataset).
            ks: Tuple of k values for min-of-k metrics.

        Returns:
            Dict with minADE_k and minFDE_k for each k.
        """
        self.model.eval()
        max_k = max(ks)
        ade_sums = {k: 0.0 for k in ks}
        fde_sums = {k: 0.0 for k in ks}
        total_agents = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(self.device)

            # Run max_k rollouts (first greedy, rest sampled)
            all_preds = []
            for sample_i in range(max_k):
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    result = self.raw_model.inference(
                        batch, greedy=(sample_i == 0))
                all_preds.append(result['pred_traj'])

            gt = result['gt']              # (N, T, 2)
            valid = result['valid_mask']   # (N, T)
            preds_stack = torch.stack(all_preds, dim=0)  # (max_k, N, T, 2)

            # Vectorized metric computation (no per-agent loop)
            has_valid = valid.any(dim=1)  # (N,)
            if not has_valid.any():
                continue

            num_valid_per_agent = valid.float().sum(dim=1).clamp(min=1)  # (N,)
            last_valid_idx = (valid.float() * torch.arange(
                valid.shape[1], device=valid.device).unsqueeze(0)
            ).max(dim=1).indices  # (N,)

            # Displacement errors: (max_k, N, T)
            disps = torch.norm(
                preds_stack - gt.unsqueeze(0), dim=-1)
            disps = disps * valid.unsqueeze(0).float()

            # ADE per sample per agent: (max_k, N)
            ade_per_sample = disps.sum(dim=-1) / num_valid_per_agent.unsqueeze(0)
            # FDE per sample per agent: (max_k, N)
            fde_per_sample = disps[:, torch.arange(gt.shape[0]), last_valid_idx]

            for k in ks:
                min_ade = ade_per_sample[:k].min(dim=0).values  # (N,)
                min_fde = fde_per_sample[:k].min(dim=0).values  # (N,)
                ade_sums[k] += min_ade[has_valid].sum().item()
                fde_sums[k] += min_fde[has_valid].sum().item()

            total_agents += has_valid.sum().item()

        n = max(total_agents, 1)
        metrics = {}
        for k in ks:
            metrics[f'minADE_{k}'] = ade_sums[k] / n
            metrics[f'minFDE_{k}'] = fde_sums[k] / n
        return metrics

    @torch.no_grad()
    def plot_trajectories(self, loader, num_scenes=5, num_samples=5):
        """Create full scene plots: map features, agent boxes, GT + predicted trajectories."""
        import matplotlib.pyplot as plt

        self.model.eval()
        num_hist = self.raw_model.num_historical_steps

        batch = next(iter(loader))
        batch = batch.to(self.device)

        # Run rollouts (first greedy, rest top-k sampled)
        all_preds = []
        for si in range(num_samples):
            with torch.amp.autocast('cuda'):
                result = self.raw_model.inference(
                    batch, greedy=(si == 0))
            all_preds.append(result['pred_traj'].cpu())

        gt = result['gt'].cpu()
        valid = result['valid_mask'].cpu()

        # Batch vectors
        agent_nn = batch['agent']['num_nodes']
        if isinstance(agent_nn, int):
            agent_b = torch.zeros(agent_nn, dtype=torch.long)
        else:
            agent_nn_cpu = agent_nn.cpu()
            agent_b = torch.arange(len(agent_nn_cpu)).repeat_interleave(agent_nn_cpu)
        # Use PyG's batch vector for pt_token (always available after Batch.from_data_list)
        pt_b = batch['pt_token']['batch'].cpu()

        # Agent data
        a_pos = batch['agent']['position'].cpu()
        a_head = batch['agent']['heading'].cpu()
        a_shape = batch['agent']['shape'].cpu()
        a_type = batch['agent']['type'].cpu()
        a_valid = batch['agent']['valid_mask'].cpu()

        # Map data
        has_map = 'polygon_idx' in batch['pt_token']
        if has_map:
            pt_pos = batch['pt_token']['position'].cpu()  # (M, 3) token centers
            pt_type = batch['pt_token']['type'].cpu()
            pt_poly = batch['pt_token']['polygon_idx'].cpu()

        n_sc = min(num_scenes, agent_b.max().item() + 1)
        ncols = min(n_sc, 5)
        nrows = math.ceil(n_sc / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
        if n_sc == 1:
            axes = [axes]
        else:
            axes = axes.flatten().tolist()

        for sc in range(n_sc):
            ax = axes[sc]

            # --- Map features ---
            if has_map:
                pm = pt_b == sc
                sc_pos_map = pt_pos[pm]  # (K, 3) - x, y, z
                sc_ptype = pt_type[pm]
                sc_poly = pt_poly[pm]

                for poly_id in sc_poly.unique():
                    sel = sc_poly == poly_id
                    pts = sc_pos_map[sel, :2].numpy()  # (K, 2) token center positions
                    rtype = sc_ptype[sel][0].item()
                    color = ROAD_COLORS.get(rtype, (0.5, 0.5, 0.5))
                    lw = ROAD_LW.get(rtype, 1)
                    ls = ROAD_LS.get(rtype, '-')
                    ax.plot(pts[:, 0], pts[:, 1], color=color,
                            linewidth=lw, linestyle=ls, zorder=1)

            # --- Agents in this scenario ---
            am = agent_b == sc
            sc_pos = a_pos[am]
            sc_head = a_head[am]
            sc_shape = a_shape[am]
            sc_atype = a_type[am]
            sc_avalid = a_valid[am]
            sc_gt = gt[am]
            sc_valid = valid[am]
            sc_preds = [p[am] for p in all_preds]
            n_agents = sc_pos.shape[0]

            t_last = num_hist - 1

            for i in range(n_agents):
                atype = sc_atype[i].item()
                color = AGENT_COLORS.get(atype, np.array([0.5, 0.5, 0.5]))

                # Draw bounding box at last historical step
                if sc_avalid[i, t_last]:
                    x = sc_pos[i, t_last, 0].item()
                    y = sc_pos[i, t_last, 1].item()
                    h = sc_head[i, t_last].item()
                    w = sc_shape[i, 0, 0].item()
                    l = sc_shape[i, 0, 1].item()
                    if l > 0 and w > 0:
                        _draw_box(ax, x, y, l, w, h, color, alpha=0.7)

                # History trajectory
                hv = sc_avalid[i, :num_hist]
                if hv.sum() > 1:
                    hp = sc_pos[i, :num_hist][hv].numpy()
                    ax.plot(hp[:, 0], hp[:, 1], color=color, linewidth=0.8,
                            linestyle=':', alpha=0.5, zorder=2)

                # GT future trajectory (blue)
                fv = sc_valid[i]
                if fv.sum() > 1:
                    gp = sc_gt[i][fv].numpy()
                    ax.plot(gp[:, 0], gp[:, 1], color='blue', linewidth=1.5,
                            linestyle='-', alpha=0.6, zorder=3,
                            label='GT' if i == 0 else None)

                # Predicted trajectories (vehicles only, all samples)
                if atype == 0 and sc_avalid[i, t_last] and fv.sum() > 0:
                    for si in range(num_samples):
                        pp = sc_preds[si][i].numpy()
                        lbl = None
                        if i == 0:
                            lbl = 'greedy' if si == 0 else (
                                f'sample' if si == 1 else None)
                        ax.plot(pp[:, 0], pp[:, 1],
                                color=PRED_COLORS[si % len(PRED_COLORS)],
                                linewidth=1.0, alpha=0.7, zorder=3,
                                label=lbl)

            # View bounds centered on agents
            valid_at_t = sc_avalid[:, t_last]
            if valid_at_t.any():
                vp = sc_pos[valid_at_t, t_last, :2]
                cx = vp[:, 0].mean().item()
                cy = vp[:, 1].mean().item()
                spread = max(
                    (vp[:, 0].max() - vp[:, 0].min()).item(),
                    (vp[:, 1].max() - vp[:, 1].min()).item())
                pad = max(spread / 2 + 40, 60)
                ax.set_xlim(cx - pad, cx + pad)
                ax.set_ylim(cy - pad, cy + pad)

            ax.set_aspect('equal')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.legend(fontsize=6, loc='upper right')
            ax.set_title(f'Scene {sc + 1}', fontsize=9)

        for idx in range(n_sc, len(axes)):
            axes[idx].set_visible(False)
        fig.suptitle(f'Epoch {self.epoch}', fontsize=11)
        fig.tight_layout()
        return fig

    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        if path is None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.checkpoint_dir, f'epoch_{self.epoch:03d}.pt')
        ckpt = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
        }
        if self.wandb_run is not None:
            ckpt['wandb_run_id'] = self.wandb_run.id
        torch.save(ckpt, path)
        return path

    def load_checkpoint(self, path):
        """Load training state from a checkpoint for resuming training.

        Backward compatible with older checkpoints that lack
        scheduler/scaler/wandb_run_id keys.

        Returns:
            The full checkpoint dict (caller can read wandb_run_id etc).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            # Fast-forward scheduler for old checkpoints
            for _ in range(ckpt.get('epoch', 0)):
                self.scheduler.step()

        if 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])

        self.epoch = ckpt.get('epoch', 0)
        self.global_step = ckpt.get('global_step', 0)
        return ckpt

    def print_dashboard(self, train_metrics, val_metrics,
                        train_traj=None, val_traj=None):
        """Print training progress."""
        lr = self.scheduler.get_last_lr()[0]
        line = (f"Epoch {self.epoch:3d} | "
                f"LR: {lr:.2e} | "
                f"TrLoss: {train_metrics['train_loss']:.4f} | "
                f"TrAcc: {train_metrics['train_accuracy']:.4f} | "
                f"VlLoss: {val_metrics['val_loss']:.4f} | "
                f"VlAcc: {val_metrics['val_accuracy']:.4f}")
        if train_traj:
            line += (f" | TrADE1: {train_traj['minADE_1']:.2f}"
                     f" | TrFDE1: {train_traj['minFDE_1']:.2f}"
                     f" | TrADE5: {train_traj['minADE_5']:.2f}"
                     f" | TrFDE5: {train_traj['minFDE_5']:.2f}")
        if val_traj:
            line += (f" | VlADE1: {val_traj['minADE_1']:.2f}"
                     f" | VlFDE1: {val_traj['minFDE_1']:.2f}"
                     f" | VlADE5: {val_traj['minADE_5']:.2f}"
                     f" | VlFDE5: {val_traj['minFDE_5']:.2f}")
        print(line)


def _setup_wandb_metrics(wandb_run):
    """Configure wandb metric x-axes for proper charting."""
    import wandb
    wandb.define_metric("global_step")
    wandb.define_metric("epoch")
    # Step-level metrics use global_step as x-axis
    wandb.define_metric("train/loss_step", step_metric="global_step")
    wandb.define_metric("train/agent_loss_step", step_metric="global_step")
    wandb.define_metric("train/accuracy_step", step_metric="global_step")
    # Epoch-level metrics use epoch as x-axis
    wandb.define_metric("train/train_*", step_metric="epoch")
    wandb.define_metric("train/minADE_*", step_metric="epoch")
    wandb.define_metric("train/minFDE_*", step_metric="epoch")
    wandb.define_metric("train/lr", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch_time_s", step_metric="epoch")


def load_config(config_path: str) -> dict:
    """Load configuration from .ini file."""
    parser = configparser.ConfigParser()
    parser.read(config_path)
    config = {}
    for section in parser.sections():
        config[section] = dict(parser[section])
    return config


def warmup_cache(args):
    """Pre-build dataset cache using multiprocessing for fast training startup."""
    from multiprocessing import Pool
    from pathlib import Path
    import time as _time

    config = load_config(args.config)
    data_cfg = config.get('data', config)
    cache_dir = data_cfg.get('cache_dir', None)
    if not cache_dir or cache_dir in ('None', ''):
        print("ERROR: cache_dir not set in config. Set [data] cache_dir = /path/to/cache")
        return

    data_dir = data_cfg.get('data_dir', '')
    num_workers = args.workers
    splits = [data_cfg.get('train_split', 'training'),
              data_cfg.get('val_split', 'validation')]

    # Build a dataset instance to reuse config
    from pufferlib.prediction.dataset import WaymoBinaryDataset
    from pufferlib.prediction.binary_reader import list_binary_files

    ds_kwargs = dict(
        data_dir=data_dir,
        num_historical_steps=int(data_cfg.get('num_historical_steps', 11)),
        num_future_steps=int(data_cfg.get('num_future_steps', 80)),
        max_agents=int(data_cfg.get('max_agents', -1)),
        shift=int(data_cfg.get('shift', 5)),
        cache_dir=cache_dir,
        num_actions=int(config.get('model', {}).get('num_actions', -1)),
    )

    def _process_one(item):
        idx, split = item
        try:
            ds = WaymoBinaryDataset(split=split, max_files=-1, **ds_kwargs)
            ds[idx]
            return 'ok'
        except Exception as e:
            return f'err:{e}'

    for split in splits:
        files = list_binary_files(data_dir, split)
        cache_split_dir = os.path.join(cache_dir, split)

        # Count existing cache files
        cached = 0
        todo = []
        for i, f in enumerate(files):
            cp = os.path.join(cache_split_dir, Path(f).stem + '.pt')
            if os.path.exists(cp):
                cached += 1
            else:
                todo.append((i, split))

        print(f"[{split}] {len(files)} total, {cached} cached, {len(todo)} to process")
        if not todo:
            continue

        t0 = _time.time()
        with Pool(num_workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(_process_one, todo, chunksize=64)):
                results.append(r)
                if (i + 1) % 5000 == 0:
                    elapsed = _time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(todo) - i - 1) / rate / 60
                    errs = sum(1 for x in results if str(x).startswith('err'))
                    print(f"  {i+1}/{len(todo)} ({rate:.0f}/s, ETA {eta:.0f}min, {errs} errors)")

        elapsed = _time.time() - t0
        errs = sum(1 for x in results if str(x).startswith('err'))
        print(f"  Done: {len(results)} in {elapsed/60:.1f}min ({errs} errors)")


def pretrain(args):
    """Phase 1: Supervised pretraining on Waymo .bin files."""
    rank, local_rank, world_size = setup_distributed()
    config = load_config(args.config)
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    device = f'cuda:{local_rank}'
    max_epochs = int(train_cfg.get('max_epochs', 100))
    traj_eval_interval = int(train_cfg.get('traj_eval_interval', 5))

    # Pre-load checkpoint metadata for wandb resume
    resume_ckpt_meta = None
    if args.resume is not None:
        resume_ckpt_meta = torch.load(args.resume, map_location='cpu',
                                      weights_only=False)

    # WandB init (rank 0 only)
    wandb_run = None
    if is_main_process():
        try:
            import wandb
            resume_run_id = args.wandb_run_id
            if resume_run_id is None and resume_ckpt_meta is not None:
                resume_run_id = resume_ckpt_meta.get('wandb_run_id')

            if resume_run_id:
                wandb_run = wandb.init(
                    id=resume_run_id,
                    project='puffer-prediction',
                    config={**model_cfg, **train_cfg, 'world_size': world_size},
                    resume="allow",
                )
            else:
                wandb_run = wandb.init(
                    project='puffer-prediction',
                    config={**model_cfg, **train_cfg, 'world_size': world_size},
                    name=f'pretrain_{time.strftime("%Y%m%d_%H%M%S")}',
                )
            _setup_wandb_metrics(wandb_run)
        except ImportError:
            print("wandb not installed, logging to stdout only")
        except Exception as e:
            print(f"wandb init failed: {e}, logging to stdout only")

    if is_main_process():
        print(f"Creating dataloaders (pretrain mode, world_size={world_size})...")

    train_loader, val_loader = create_dataloaders(
        config, mode='pretrain', distributed=(world_size > 1))

    # Fixed eval subset (2000 scenarios, deterministic) for trajectory metrics
    eval_size = int(train_cfg.get('eval_size', 2000))
    eval_size = min(eval_size, len(val_loader.dataset))
    eval_gen = torch.Generator().manual_seed(42)
    eval_indices = torch.randperm(
        len(val_loader.dataset), generator=eval_gen)[:eval_size].tolist()
    eval_subset = torch.utils.data.Subset(val_loader.dataset, eval_indices)
    eval_loader = torch.utils.data.DataLoader(
        eval_subset,
        batch_size=int(config.get('data', {}).get('batch_size', 4)),
        shuffle=False,
        num_workers=int(config.get('data', {}).get('num_workers', 8)),
        pin_memory=True,
        collate_fn=val_loader.collate_fn,
    )

    if is_main_process():
        print(f"Train: {len(train_loader.dataset)} scenarios, "
              f"Val: {len(val_loader.dataset)} scenarios, "
              f"Eval subset: {len(eval_subset)} scenarios")

    model = _create_model(model_cfg)
    if is_main_process():
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    trainer = PufferPrediction(
        config, model, train_loader, val_loader, device=device,
        wandb_run=wandb_run, local_rank=local_rank)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        if is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        start_epoch = trainer.epoch
        if is_main_process():
            print(f"  Resumed at epoch {start_epoch}, "
                  f"global_step {trainer.global_step}")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_metrics = trainer.train_epoch()
        trainer.scheduler.step()
        val_metrics = trainer.validate()

        # Compute trajectory metrics on fixed eval subset (rank 0 only)
        train_traj = None
        val_traj = None
        traj_fig = None
        train_traj_fig = None
        is_fast_inference = model_cfg.get('model_type', 'smart') == 'simpl'
        train_max_batches = None if is_fast_inference else 3
        val_max_batches = None if is_fast_inference else 50
        if is_main_process() and (epoch + 1) % traj_eval_interval == 0:
            train_traj = trainer.compute_trajectory_metrics(
                train_loader, max_batches=train_max_batches, ks=(1, 5))
            val_traj = trainer.compute_trajectory_metrics(
                eval_loader, max_batches=val_max_batches, ks=(1, 5))
            traj_fig = trainer.plot_trajectories(eval_loader, num_scenes=10)
            train_traj_fig = trainer.plot_trajectories(
                train_loader, num_scenes=10)

        elapsed = time.time() - t0

        if is_main_process():
            trainer.print_dashboard(train_metrics, val_metrics,
                                    train_traj, val_traj)
            print(f"  ({elapsed:.1f}s)")

            # Log to wandb
            if wandb_run:
                import wandb as _wb
                log_dict = {
                    'epoch': epoch + 1,
                    'train/lr': trainer.scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch_time_s': elapsed,
                }
                if train_traj:
                    log_dict.update({
                        f'train/{k}': v for k, v in train_traj.items()})
                if val_traj:
                    log_dict.update({
                        f'val/{k}': v for k, v in val_traj.items()})
                if traj_fig is not None:
                    log_dict['val/trajectories'] = _wb.Image(traj_fig)
                    import matplotlib.pyplot as plt
                    plt.close(traj_fig)
                if train_traj_fig is not None:
                    log_dict['train/trajectories'] = _wb.Image(train_traj_fig)
                    import matplotlib.pyplot as plt
                    plt.close(train_traj_fig)
                wandb_run.log(log_dict)

            if (epoch + 1) % int(train_cfg.get('checkpoint_interval', 5)) == 0:
                path = trainer.save_checkpoint()
                print(f"  Saved checkpoint: {path}")

        # Synchronize all processes at epoch boundary
        if dist.is_initialized():
            dist.barrier()

    if wandb_run:
        wandb_run.finish()
    cleanup_distributed()


def finetune(args):
    """Phase 2: Fine-tuning on policy-collected transitions."""
    rank, local_rank, world_size = setup_distributed()
    config = load_config(args.config)
    model_cfg = config.get('model', {})
    finetune_cfg = config.get('finetune', {})

    device = f'cuda:{local_rank}'
    max_epochs = int(finetune_cfg.get('max_epochs', 50))
    pretrain_ckpt = finetune_cfg.get('pretrain_ckpt', '')
    traj_eval_interval = int(config.get('train', {}).get('traj_eval_interval', 5))

    # Pre-load checkpoint metadata for wandb resume
    resume_ckpt_meta = None
    if args.resume is not None:
        resume_ckpt_meta = torch.load(args.resume, map_location='cpu',
                                      weights_only=False)

    # WandB init (rank 0 only)
    wandb_run = None
    if is_main_process():
        try:
            import wandb
            resume_run_id = args.wandb_run_id
            if resume_run_id is None and resume_ckpt_meta is not None:
                resume_run_id = resume_ckpt_meta.get('wandb_run_id')

            if resume_run_id:
                wandb_run = wandb.init(
                    id=resume_run_id,
                    project='puffer-prediction',
                    config={**model_cfg, **finetune_cfg, 'world_size': world_size},
                    resume="allow",
                )
            else:
                wandb_run = wandb.init(
                    project='puffer-prediction',
                    config={**model_cfg, **finetune_cfg, 'world_size': world_size},
                    name=f'finetune_{time.strftime("%Y%m%d_%H%M%S")}',
                )
            _setup_wandb_metrics(wandb_run)
        except ImportError:
            print("wandb not installed, logging to stdout only")
        except Exception as e:
            print(f"wandb init failed: {e}, logging to stdout only")

    if is_main_process():
        print(f"Creating dataloaders (finetune mode, world_size={world_size})...")

    train_loader, val_loader = create_dataloaders(
        config, mode='finetune', distributed=(world_size > 1))

    if is_main_process():
        print(f"Train: {len(train_loader.dataset)} scenarios, "
              f"Val: {len(val_loader.dataset)} scenarios")

    model = _create_model(model_cfg)

    # Skip pretrain_ckpt when resuming (resume includes model weights)
    if args.resume is None and pretrain_ckpt and os.path.exists(pretrain_ckpt):
        if is_main_process():
            print(f"Loading pretrained checkpoint: {pretrain_ckpt}")
        ckpt = torch.load(pretrain_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])

    # Override learning rate for fine-tuning
    config['train']['learning_rate'] = finetune_cfg.get('learning_rate', '1e-4')

    trainer = PufferPrediction(
        config, model, train_loader, val_loader, device=device,
        wandb_run=wandb_run, local_rank=local_rank)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        if is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        start_epoch = trainer.epoch
        if is_main_process():
            print(f"  Resumed at epoch {start_epoch}, "
                  f"global_step {trainer.global_step}")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_metrics = trainer.train_epoch()
        trainer.scheduler.step()
        val_metrics = trainer.validate()

        train_traj = None
        val_traj = None
        traj_fig = None
        train_traj_fig = None
        if is_main_process() and (epoch + 1) % traj_eval_interval == 0:
            train_traj = trainer.compute_trajectory_metrics(
                train_loader, max_batches=3, ks=(1, 5))
            val_traj = trainer.compute_trajectory_metrics(
                val_loader, max_batches=3, ks=(1, 5))
            traj_fig = trainer.plot_trajectories(val_loader, num_scenes=5)
            train_traj_fig = trainer.plot_trajectories(
                train_loader, num_scenes=5)

        elapsed = time.time() - t0

        if is_main_process():
            trainer.print_dashboard(train_metrics, val_metrics,
                                    train_traj, val_traj)
            print(f"  ({elapsed:.1f}s)")

            if wandb_run:
                import wandb as _wb
                log_dict = {
                    'epoch': epoch + 1,
                    'train/lr': trainer.scheduler.get_last_lr()[0],
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch_time_s': elapsed,
                }
                if train_traj:
                    log_dict.update({
                        f'train/{k}': v for k, v in train_traj.items()})
                if val_traj:
                    log_dict.update({
                        f'val/{k}': v for k, v in val_traj.items()})
                if traj_fig is not None:
                    log_dict['val/trajectories'] = _wb.Image(traj_fig)
                    import matplotlib.pyplot as plt
                    plt.close(traj_fig)
                if train_traj_fig is not None:
                    log_dict['train/trajectories'] = _wb.Image(train_traj_fig)
                    import matplotlib.pyplot as plt
                    plt.close(train_traj_fig)
                wandb_run.log(log_dict)

            if (epoch + 1) % int(finetune_cfg.get('checkpoint_interval', 5)) == 0:
                path = trainer.save_checkpoint()
                print(f"  Saved checkpoint: {path}")

        if dist.is_initialized():
            dist.barrier()

    if wandb_run:
        wandb_run.finish()
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description='SMART Prediction Model Training')
    subparsers = parser.add_subparsers(dest='command')

    pretrain_parser = subparsers.add_parser('pretrain', help='Supervised pretraining')
    pretrain_parser.add_argument('--config', type=str,
                                default='config/prediction/smart.ini')
    pretrain_parser.add_argument('--resume', type=str, default=None,
                                help='Path to checkpoint to resume training from')
    pretrain_parser.add_argument('--wandb-run-id', type=str, default=None,
                                help='WandB run ID to resume (overrides checkpoint value)')

    finetune_parser = subparsers.add_parser('finetune', help='RL fine-tuning')
    finetune_parser.add_argument('--config', type=str,
                                 default='config/prediction/smart.ini')
    finetune_parser.add_argument('--resume', type=str, default=None,
                                 help='Path to checkpoint to resume training from')
    finetune_parser.add_argument('--wandb-run-id', type=str, default=None,
                                 help='WandB run ID to resume (overrides checkpoint value)')

    cache_parser = subparsers.add_parser('warmup-cache',
                                          help='Pre-build dataset cache with multiprocessing')
    cache_parser.add_argument('--config', type=str,
                              default='config/prediction/smart.ini')
    cache_parser.add_argument('--workers', type=int, default=80,
                              help='Number of parallel workers (default: 80)')

    args = parser.parse_args()

    if args.command == 'pretrain':
        pretrain(args)
    elif args.command == 'finetune':
        finetune(args)
    elif args.command == 'warmup-cache':
        warmup_cache(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
