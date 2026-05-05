# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Uncertainty estimation for policy planners.

Computes aleatoric (entropy) and epistemic (mutual information) uncertainty
from policy logits. Supports single-model and ensemble evaluation.
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger("evaluation")


def _logits_to_probs(logits_tuple):
    """Convert multi-discrete logits tuple to list of probability tensors."""
    if isinstance(logits_tuple, torch.Tensor):
        return [F.softmax(logits_tuple, dim=-1)]
    return [F.softmax(l, dim=-1) for l in logits_tuple]


def _entropy_from_probs(probs_list):
    """Compute entropy from list of probability tensors (one per action head).

    Returns scalar entropy summed across action heads.
    """
    total = 0.0
    for p in probs_list:
        # Clamp to avoid log(0)
        p = p.clamp(min=1e-8)
        total += -(p * p.log()).sum(dim=-1)
    return total


def compute_aleatoric(logits) -> float:
    """Compute aleatoric uncertainty (entropy) from policy logits.

    Args:
        logits: tuple of tensors (multi-discrete) or single tensor,
                shape (batch, num_actions) per head. Batch dim = 1 for ego.

    Returns:
        Entropy as a float (summed across action heads, averaged over batch).
    """
    if isinstance(logits, torch.distributions.Normal):
        return logits.entropy().sum(dim=-1).mean().item()

    probs = _logits_to_probs(logits)
    ent = _entropy_from_probs(probs)
    return ent.mean().item()


def compute_epistemic(logits_list: List) -> float:
    """Compute epistemic uncertainty (mutual information) across ensemble.

    MI = H(mean_probs) - mean(H(probs))
    Higher MI means the models disagree → epistemic uncertainty.

    Args:
        logits_list: list of logits from N ensemble members,
                     each is a tuple of tensors or single tensor.

    Returns:
        Mutual information as a float.
    """
    if len(logits_list) < 2:
        return 0.0

    # Convert all to probabilities
    all_probs = [_logits_to_probs(l) for l in logits_list]
    num_heads = len(all_probs[0])

    mi_total = 0.0
    for head_idx in range(num_heads):
        # Stack probs for this head across ensemble: (N, batch, num_actions)
        stacked = torch.stack([p[head_idx] for p in all_probs], dim=0)
        # Mean probs across ensemble
        mean_probs = stacked.mean(dim=0).clamp(min=1e-8)
        # H(mean_probs)
        h_mean = -(mean_probs * mean_probs.log()).sum(dim=-1)
        # mean(H(probs))
        per_model_h = -(stacked.clamp(min=1e-8) * stacked.clamp(min=1e-8).log()).sum(dim=-1)
        mean_h = per_model_h.mean(dim=0)
        # MI = H(mean) - mean(H)
        mi_total += (h_mean - mean_h).mean()

    return mi_total.item()


class PolicyEnsemble:
    """Ensemble of PPO policies with different weights for epistemic uncertainty.

    Creates N copies of the policy network, each loaded with different
    checkpoint weights. Maintains separate LSTM states per member.
    """

    def __init__(self, weight_paths: List[str], env, config, ego_agent_idx: int = 0):
        """Create ensemble from multiple checkpoint paths.

        Args:
            weight_paths: list of checkpoint file paths
            env: Drive environment (for policy construction)
            config: PPOConfig
            ego_agent_idx: which agent is ego
        """
        from pufferlib.planning.policy import PPOPlanner, PPOConfig

        self.members = []
        self.lstm_states = []

        for path in weight_paths:
            member_config = PPOConfig(
                weights_path=path,
                device=config.device,
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                policy_action_type=config.policy_action_type,
            )
            planner = PPOPlanner(
                env=env,
                agent_idx=ego_agent_idx,
                action_lb=np.zeros(2),  # Not used for forward only
                action_ub=np.ones(2),
                config=member_config,
            )
            self.members.append(planner)

        log.info("Created policy ensemble with %d members", len(self.members))

    def forward_all(self, obs: np.ndarray) -> List:
        """Forward observation through all ensemble members.

        Args:
            obs: observation array, shape (obs_dim,) or (N, obs_dim)

        Returns:
            List of logits from each ensemble member.
        """
        logits_list = []
        for member in self.members:
            single = obs.ndim == 1
            obs_2d = np.atleast_2d(obs)
            batch_size = obs_2d.shape[0]

            obs_tensor = torch.from_numpy(obs_2d).float().to(member.device)

            if member.lstm_h is None or member.lstm_h.shape[0] != batch_size:
                member.lstm_h = torch.zeros(
                    batch_size, member.config.hidden_size, device=member.device
                )
                member.lstm_c = torch.zeros(
                    batch_size, member.config.hidden_size, device=member.device
                )
            state = {"lstm_h": member.lstm_h, "lstm_c": member.lstm_c}

            with torch.no_grad():
                logits, _ = member.policy.forward_eval(obs_tensor, state)

            member.lstm_h = state["lstm_h"]
            member.lstm_c = state["lstm_c"]
            logits_list.append(logits)

        return logits_list

    def forward_all_with_values(self, obs: np.ndarray) -> Tuple[List, List[float]]:
        """Forward observation through all ensemble members, returning logits and values.

        Args:
            obs: observation array, shape (obs_dim,) or (N, obs_dim)

        Returns:
            Tuple of (logits_list, values) where values is a list of scalar
            value predictions from each ensemble member.
        """
        logits_list = []
        values = []
        for member in self.members:
            single = obs.ndim == 1
            obs_2d = np.atleast_2d(obs)
            batch_size = obs_2d.shape[0]

            obs_tensor = torch.from_numpy(obs_2d).float().to(member.device)

            if member.lstm_h is None or member.lstm_h.shape[0] != batch_size:
                member.lstm_h = torch.zeros(
                    batch_size, member.config.hidden_size, device=member.device
                )
                member.lstm_c = torch.zeros(
                    batch_size, member.config.hidden_size, device=member.device
                )
            state = {"lstm_h": member.lstm_h, "lstm_c": member.lstm_c}

            with torch.no_grad():
                logits, value = member.policy.forward_eval(obs_tensor, state)

            member.lstm_h = state["lstm_h"]
            member.lstm_c = state["lstm_c"]
            logits_list.append(logits)
            values.append(value.mean().item())

        return logits_list, values

    def save_all_lstm_states(self):
        """Save LSTM states for all ensemble members."""
        for member in self.members:
            member.save_lstm_state()

    def restore_all_lstm_states(self):
        """Restore LSTM states for all ensemble members."""
        for member in self.members:
            member.restore_lstm_state()

    def reset(self):
        """Reset LSTM states for all ensemble members."""
        for member in self.members:
            member.lstm_h = None
            member.lstm_c = None


def plot_uncertainty_correlation(
    rewards: List[float],
    aleatoric: List[float],
    epistemic: Optional[List[float]],
    output_dir: str,
    value_variance: Optional[List[float]] = None,
):
    """Generate uncertainty vs reward correlation plots.

    Creates scatter plots and binned plots showing the relationship
    between uncertainty and reward across maps.

    Args:
        rewards: per-map total rewards
        aleatoric: per-map mean aleatoric uncertainty
        epistemic: per-map mean epistemic uncertainty (None if no ensemble)
        output_dir: directory to save plots
        value_variance: per-map mean value prediction variance (None if no ensemble)
    """
    os.makedirs(output_dir, exist_ok=True)
    rewards = np.array(rewards)
    aleatoric = np.array(aleatoric)

    has_epistemic = epistemic is not None and len(epistemic) > 0 and any(v > 0 for v in epistemic)
    has_value_var = value_variance is not None and len(value_variance) > 0
    num_plots = 1 + int(has_epistemic) + int(has_value_var)

    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Aleatoric scatter
    ax = axes[plot_idx]
    ax.scatter(aleatoric, rewards, alpha=0.4, s=15)
    if len(rewards) > 2:
        corr = np.corrcoef(aleatoric, rewards)[0, 1]
        # Trend line
        z = np.polyfit(aleatoric, rewards, 1)
        p = np.poly1d(z)
        x_range = np.linspace(aleatoric.min(), aleatoric.max(), 100)
        ax.plot(x_range, p(x_range), "r--", alpha=0.7)
        ax.set_title(f"Aleatoric vs Reward ($\\rho$={corr:.3f})")
    else:
        ax.set_title("Aleatoric vs Reward")
    ax.set_xlabel("Aleatoric Uncertainty (entropy)")
    ax.set_ylabel("Step Reward")
    plot_idx += 1

    # Epistemic scatter
    if has_epistemic:
        epistemic = np.array(epistemic)
        ax = axes[plot_idx]
        ax.scatter(epistemic, rewards, alpha=0.4, s=15)
        if len(rewards) > 2:
            corr = np.corrcoef(epistemic, rewards)[0, 1]
            z = np.polyfit(epistemic, rewards, 1)
            p = np.poly1d(z)
            x_range = np.linspace(epistemic.min(), epistemic.max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.7)
            ax.set_title(f"Epistemic vs Reward ($\\rho$={corr:.3f})")
        else:
            ax.set_title("Epistemic vs Reward")
        ax.set_xlabel("Epistemic Uncertainty (MI)")
        ax.set_ylabel("Step Reward")
        plot_idx += 1

    # Value variance scatter
    if has_value_var:
        value_variance = np.array(value_variance)
        ax = axes[plot_idx]
        ax.scatter(value_variance, rewards, alpha=0.4, s=15)
        if len(rewards) > 2:
            corr = np.corrcoef(value_variance, rewards)[0, 1]
            z = np.polyfit(value_variance, rewards, 1)
            p = np.poly1d(z)
            x_range = np.linspace(value_variance.min(), value_variance.max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.7)
            ax.set_title(f"Value Variance vs Reward ($\\rho$={corr:.3f})")
        else:
            ax.set_title("Value Variance vs Reward")
        ax.set_xlabel("Value Prediction Variance")
        ax.set_ylabel("Step Reward")

    fig.tight_layout()
    path = os.path.join(output_dir, "uncertainty_correlation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved uncertainty correlation plot: %s", path)

    # Binned plot
    _plot_binned(
        rewards, aleatoric,
        epistemic if has_epistemic else None,
        output_dir,
        value_variance=value_variance if has_value_var else None,
    )

    return path


def _plot_binned(rewards, aleatoric, epistemic, output_dir, n_bins=5, value_variance=None):
    """Create binned uncertainty vs reward plot."""
    has_epistemic = epistemic is not None
    has_value_var = value_variance is not None
    num_plots = 1 + int(has_epistemic) + int(has_value_var)

    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    panels = [(aleatoric, "Aleatoric")]
    if has_epistemic:
        panels.append((epistemic, "Epistemic"))
    if has_value_var:
        panels.append((value_variance, "Value Variance"))

    for idx, (unc, label) in enumerate(panels):
        ax = axes[idx]
        bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            ax.text(0.5, 0.5, "Not enough variation", transform=ax.transAxes, ha="center")
            continue

        bin_means = []
        bin_stds = []
        bin_centers = []
        for i in range(len(bin_edges) - 1):
            mask = (unc >= bin_edges[i]) & (unc < bin_edges[i + 1])
            if i == len(bin_edges) - 2:
                mask = (unc >= bin_edges[i]) & (unc <= bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(rewards[mask].mean())
                bin_stds.append(rewards[mask].std())
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        ax.bar(range(len(bin_means)), bin_means, yerr=bin_stds, alpha=0.7, capsize=4)
        ax.set_xticks(range(len(bin_centers)))
        ax.set_xticklabels([f"{c:.2f}" for c in bin_centers], rotation=45)
        ax.set_xlabel(f"{label} Uncertainty")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Reward by {label} Uncertainty Bin")

    fig.tight_layout()
    path = os.path.join(output_dir, "uncertainty_binned.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
