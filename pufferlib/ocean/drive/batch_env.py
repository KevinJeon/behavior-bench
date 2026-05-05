# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Batch environment wrapper for parallel planner rollouts."""

from typing import Tuple
import numpy as np

from pufferlib.ocean.drive import binding
from pufferlib.ocean.drive.drive import Drive


class DriveBatch:
    """
    Batch environment for parallel planner rollouts.

    This class manages N cloned environments that share the same map
    but can be independently stepped for parallel rollout evaluation.
    """

    def __init__(
        self,
        c_envs,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        truncations: np.ndarray,
        collision_rewards: np.ndarray,
        offroad_rewards: np.ndarray,
        goal_rewards: np.ndarray,
        goal_distances: np.ndarray,
        jerk_rewards: np.ndarray,
        lane_distances: np.ndarray,
        lane_alignments: np.ndarray,
        num_envs: int,
        agents_per_env: int,
        ego_agent_idx: int = 0,
    ):
        self.c_envs = c_envs
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.truncations = truncations
        # Decomposed rewards for CEM
        self.collision_rewards = collision_rewards
        self.offroad_rewards = offroad_rewards
        self.goal_rewards = goal_rewards
        self.goal_distances = goal_distances
        self.jerk_rewards = jerk_rewards
        self.lane_distances = lane_distances
        self.lane_alignments = lane_alignments
        self.num_envs = num_envs
        self.agents_per_env = agents_per_env
        self.ego_agent_idx = ego_agent_idx

        # Precompute indices for ego agent in each environment
        self.ego_indices = np.arange(
            ego_agent_idx, num_envs * agents_per_env, agents_per_env, dtype=np.int32
        )

        # Preallocate arrays for fast position extraction
        self._pos_x = np.zeros(num_envs, dtype=np.float32)
        self._pos_y = np.zeros(num_envs, dtype=np.float32)

        # Preallocate arrays for all-agent position extraction
        total = num_envs * agents_per_env
        self._all_x = np.zeros(total, dtype=np.float32)
        self._all_y = np.zeros(total, dtype=np.float32)
        self._all_z = np.zeros(total, dtype=np.float32)
        self._all_heading = np.zeros(total, dtype=np.float32)
        self._all_ids = np.zeros(total, dtype=np.int32)
        self._all_length = np.zeros(total, dtype=np.float32)
        self._all_width = np.zeros(total, dtype=np.float32)
        self._all_type = np.zeros(total, dtype=np.int32)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute one step with given actions."""
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        return self.observations, self.rewards, self.terminals, self.truncations

    def get_decomposed_rewards(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get decomposed rewards from the last step.

        Returns:
            Tuple of (collision_rewards, offroad_rewards, goal_rewards, goal_distances, jerk_rewards, lane_distances, lane_alignments)
            Each array has shape (total_agents,)
        """
        return self.collision_rewards, self.offroad_rewards, self.goal_rewards, self.goal_distances, self.jerk_rewards, self.lane_distances, self.lane_alignments

    def restore_snapshot_broadcast(self, snapshot_handle):
        """Restore all environments to the same snapshot state."""
        binding.vec_restore_snapshot_broadcast(self.c_envs, snapshot_handle)

    def get_state(self):
        """Get state from all environments (expensive - use sparingly)."""
        return binding.vec_get(self.c_envs)

    def get_ego_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ego agent positions for all environments.

        This is a fast alternative to get_state() when only positions are needed.

        Returns:
            Tuple of (x_positions, y_positions), each shape (num_envs,)
        """
        binding.vec_get_ego_positions(self.c_envs, self._pos_x, self._pos_y, self.ego_agent_idx)
        return self._pos_x.copy(), self._pos_y.copy()

    def get_all_agent_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions for all active agents across all batch environments.

        Uses vec_get_global_agent_state C binding for fast extraction.

        Returns:
            Tuple of (x_positions, y_positions), each shape (num_envs * agents_per_env,)
        """
        binding.vec_get_global_agent_state(
            self.c_envs,
            self._all_x, self._all_y, self._all_z,
            self._all_heading, self._all_ids,
            self._all_length, self._all_width,
            self._all_type,
        )
        return self._all_x.copy(), self._all_y.copy()

    def get_global_agent_state(self) -> dict:
        """
        Get full state for all agents across all batch environments.

        Returns:
            Dict with keys: x, y, z, heading, ids, length, width, type
            Each value shape (num_envs * agents_per_env,)
        """
        binding.vec_get_global_agent_state(
            self.c_envs,
            self._all_x, self._all_y, self._all_z,
            self._all_heading, self._all_ids,
            self._all_length, self._all_width,
            self._all_type,
        )
        return {
            "x": self._all_x.copy(),
            "y": self._all_y.copy(),
            "z": self._all_z.copy(),
            "heading": self._all_heading.copy(),
            "ids": self._all_ids.copy(),
            "length": self._all_length.copy(),
            "width": self._all_width.copy(),
            "type": self._all_type.copy(),
        }

    def close(self):
        """Clean up resources."""
        binding.vec_close(self.c_envs)

    @classmethod
    def from_env(cls, env: Drive, num_envs: int, ego_agent_idx: int = 0) -> "DriveBatch":
        """
        Create a batch environment cloned from a single environment.

        Args:
            env: Source Drive environment (must have exactly 1 sub-environment)
            num_envs: Number of cloned environments to create
            ego_agent_idx: Which agent is ego (for correct ego_indices)

        Returns:
            DriveBatch instance
        """
        agents_per_env = env.num_agents
        total_agents = num_envs * agents_per_env

        observations = np.zeros((total_agents, env.num_obs), dtype=np.float32)
        rewards = np.zeros(total_agents, dtype=np.float32)
        terminals = np.zeros(total_agents, dtype=np.bool_)
        truncations = np.zeros(total_agents, dtype=np.bool_)

        # Decomposed rewards for CEM
        collision_rewards = np.zeros(total_agents, dtype=np.float32)
        offroad_rewards = np.zeros(total_agents, dtype=np.float32)
        goal_rewards = np.zeros(total_agents, dtype=np.float32)
        goal_distances = np.zeros(total_agents, dtype=np.float32)
        jerk_rewards = np.zeros(total_agents, dtype=np.float32)
        lane_distances = np.zeros(total_agents, dtype=np.float32)
        lane_alignments = np.zeros(total_agents, dtype=np.float32)

        # Action shape depends on action type
        if env._action_type_flag == 0:  # Discrete
            actions = np.zeros((total_agents, 1), dtype=np.int32)
        else:  # Continuous
            actions = np.zeros((total_agents, 2), dtype=np.float32)

        c_envs = binding.vec_clone_from_env(
            env.c_envs,
            observations,
            actions,
            rewards,
            terminals,
            truncations,
            collision_rewards,
            offroad_rewards,
            goal_rewards,
            goal_distances,
            jerk_rewards,
            lane_distances,
            lane_alignments,
            num_envs,
            agents_per_env,
        )

        return cls(
            c_envs,
            observations,
            actions,
            rewards,
            terminals,
            truncations,
            collision_rewards,
            offroad_rewards,
            goal_rewards,
            goal_distances,
            jerk_rewards,
            lane_distances,
            lane_alignments,
            num_envs,
            agents_per_env,
            ego_agent_idx=ego_agent_idx,
        )
