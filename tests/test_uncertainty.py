# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for uncertainty estimation module."""

import math
import pytest
import torch

from pufferlib.evaluation.uncertainty import (
    compute_aleatoric,
    compute_epistemic,
    _logits_to_probs,
    _entropy_from_probs,
)


class TestComputeAleatoric:
    """Tests for aleatoric uncertainty (entropy)."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform logits should give maximum entropy."""
        # 7 actions, uniform logits → entropy = log(7) ≈ 1.946
        logits = (torch.zeros(1, 7),)  # multi-discrete tuple with 1 head
        ent = compute_aleatoric(logits)
        assert abs(ent - math.log(7)) < 0.01

    def test_peaked_distribution_low_entropy(self):
        """Very peaked distribution should have near-zero entropy."""
        logits = torch.full((1, 7), -100.0)
        logits[0, 3] = 100.0
        ent = compute_aleatoric((logits,))
        assert ent < 0.01

    def test_multi_discrete_sums_heads(self):
        """Multi-discrete: entropy should be sum across heads."""
        head1 = torch.zeros(1, 7)  # entropy = log(7)
        head2 = torch.zeros(1, 13)  # entropy = log(13)
        logits = (head1, head2)
        ent = compute_aleatoric(logits)
        expected = math.log(7) + math.log(13)
        assert abs(ent - expected) < 0.01

    def test_single_tensor_discrete(self):
        """Single tensor (not tuple) should work."""
        logits = torch.zeros(1, 91)  # flat discrete
        ent = compute_aleatoric(logits)
        assert abs(ent - math.log(91)) < 0.01

    def test_batch_averages(self):
        """Batch of different distributions: should average entropy."""
        logits = torch.zeros(2, 7)
        # First sample: uniform (entropy=log7)
        # Second sample: peaked (entropy≈0)
        logits[1, :] = -100.0
        logits[1, 0] = 100.0
        ent = compute_aleatoric((logits,))
        # Average of log(7) and ~0 ≈ log(7)/2
        assert ent < math.log(7)
        assert ent > 0.0

    def test_continuous_normal(self):
        """Normal distribution entropy."""
        loc = torch.zeros(1, 2)
        scale = torch.ones(1, 2)
        dist = torch.distributions.Normal(loc, scale)
        ent = compute_aleatoric(dist)
        # Entropy of N(0,1) = 0.5*log(2*pi*e) ≈ 1.4189 per dim, 2 dims
        expected = 2 * 0.5 * math.log(2 * math.pi * math.e)
        assert abs(ent - expected) < 0.01


class TestComputeEpistemic:
    """Tests for epistemic uncertainty (mutual information)."""

    def test_identical_models_zero_mi(self):
        """If all models agree, MI should be zero."""
        logits = (torch.tensor([[0.0, 1.0, 2.0]]),)
        # All 3 "models" have identical logits
        mi = compute_epistemic([logits, logits, logits])
        assert abs(mi) < 1e-5

    def test_disagreeing_models_positive_mi(self):
        """If models disagree, MI should be positive."""
        # Model 1: peaked on action 0
        l1 = (torch.tensor([[10.0, -10.0, -10.0]]),)
        # Model 2: peaked on action 1
        l2 = (torch.tensor([[-10.0, 10.0, -10.0]]),)
        # Model 3: peaked on action 2
        l3 = (torch.tensor([[-10.0, -10.0, 10.0]]),)
        mi = compute_epistemic([l1, l2, l3])
        # Mean probs ≈ [1/3, 1/3, 1/3] → H(mean) ≈ log(3)
        # Each model H ≈ 0 → mean(H) ≈ 0
        # MI ≈ log(3) ≈ 1.099
        assert mi > 0.5

    def test_single_model_returns_zero(self):
        """Single model should return 0 (can't compute MI)."""
        logits = (torch.tensor([[1.0, 2.0, 3.0]]),)
        mi = compute_epistemic([logits])
        assert mi == 0.0

    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        mi = compute_epistemic([])
        assert mi == 0.0

    def test_multi_discrete_ensemble(self):
        """Multi-discrete logits (tuple) should work."""
        # 2 heads: accel (7) and steer (13)
        l1 = (torch.zeros(1, 7), torch.zeros(1, 13))
        l2 = (torch.zeros(1, 7), torch.zeros(1, 13))
        mi = compute_epistemic([l1, l2])
        # Identical → MI = 0
        assert abs(mi) < 1e-5

    def test_mi_bounded_by_entropy(self):
        """MI should not exceed H(mean_probs)."""
        l1 = (torch.randn(1, 5),)
        l2 = (torch.randn(1, 5),)
        mi = compute_epistemic([l1, l2])
        # Compute H(mean_probs) as upper bound
        p1 = torch.softmax(l1[0], dim=-1)
        p2 = torch.softmax(l2[0], dim=-1)
        mean_p = (p1 + p2) / 2
        h_mean = -(mean_p * mean_p.clamp(min=1e-8).log()).sum().item()
        assert mi <= h_mean + 1e-5


class TestHelpers:
    """Tests for helper functions."""

    def test_logits_to_probs_single_tensor(self):
        """Single tensor should return list with one prob tensor."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        probs = _logits_to_probs(logits)
        assert len(probs) == 1
        assert abs(probs[0].sum().item() - 1.0) < 1e-5

    def test_logits_to_probs_tuple(self):
        """Tuple of tensors should return list of prob tensors."""
        logits = (torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0]]))
        probs = _logits_to_probs(logits)
        assert len(probs) == 2
        assert abs(probs[0].sum().item() - 1.0) < 1e-5
        assert abs(probs[1].sum().item() - 1.0) < 1e-5

    def test_entropy_from_probs_uniform(self):
        """Uniform distribution entropy."""
        probs = [torch.tensor([[0.25, 0.25, 0.25, 0.25]])]
        ent = _entropy_from_probs(probs)
        assert abs(ent.item() - math.log(4)) < 0.01
