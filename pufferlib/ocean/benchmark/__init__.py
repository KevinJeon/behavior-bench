# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from PufferDrive V2.0
# (https://github.com/Emerge-Lab/PufferDrive/)
# Copyright (c) 2026 PufferDrive, licensed under the MIT license.

"""Benchmark utilities for the Drive environment."""

from .extract_benchmark import (
    load_binary_file,
    compute_interactivity_score,
    compute_avg_agents_in_radius,
    compute_avg_goal_distance,
    get_agent_goal_distance,
    ScenarioData,
    Entity,
)

__all__ = [
    'load_binary_file',
    'compute_interactivity_score',
    'compute_avg_agents_in_radius',
    'compute_avg_goal_distance',
    'get_agent_goal_distance',
    'ScenarioData',
    'Entity',
]
