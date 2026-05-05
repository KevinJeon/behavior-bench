# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Demo: clone a single Drive env into a batch in C, step with batched actions,
and plot the resulting states.
"""

import os
import sys
import numpy as np
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.viz import plot_simulator_state, VizConfig


class DriveBatch:
    def __init__(self, c_envs, observations, actions, rewards, terminals, truncations):
        self.c_envs = c_envs
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.truncations = truncations

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        return (self.observations, self.rewards, self.terminals, self.truncations, [])

    def get_state(self):
        return binding.vec_get(self.c_envs)

    def close(self):
        binding.vec_close(self.c_envs)


def clone_batch_from_env(env, num_envs):
    agents_per_env = env.num_agents
    obs_shape = (num_envs * agents_per_env, env.num_obs)
    observations = np.zeros(obs_shape, dtype=np.float32)
    rewards = np.zeros(num_envs * agents_per_env, dtype=np.float32)
    terminals = np.zeros(num_envs * agents_per_env, dtype=np.bool_)
    truncations = np.zeros(num_envs * agents_per_env, dtype=np.bool_)

    if env._action_type_flag == 0:
        actions = np.zeros((num_envs * agents_per_env, 1), dtype=np.int32)
    else:
        actions = np.zeros((num_envs * agents_per_env, 2), dtype=np.float32)

    c_envs = binding.vec_clone_from_env(
        env.c_envs,
        observations,
        actions,
        rewards,
        terminals,
        truncations,
        num_envs,
        agents_per_env,
    )
    return DriveBatch(c_envs, observations, actions, rewards, terminals, truncations)


def main():
    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        print("Error: DRIVE_BINARIES_DATA_ROOT not set")
        sys.exit(1)

    env = Drive(
        num_agents=1,
        num_maps=10,
        resample_frequency=-1,
        episode_length=91,
        action_type="discrete",
        max_controlled_agents=1,
    )
    env.reset()

    num_warmup_steps = np.random.randint(2, 6)
    for _ in range(num_warmup_steps):
        accel_idx = np.random.randint(0, 7, env.num_agents)
        steer_idx = np.random.randint(0, 13, env.num_agents)
        actions = (accel_idx * 13 + steer_idx).astype(np.int32).reshape(-1, 1)
        env.step(actions)

    original_state = env.get_state()[0]
    original_entities = original_state.get("entities") or []
    original_active_idx = original_state.get("active_agent_indices") or []
    if original_active_idx:
        original_ego = original_entities[original_active_idx[0]]
        print(
            f"original before: ({original_ego.get('x', 0.0):.3f}, {original_ego.get('y', 0.0):.3f})"
        )

    num_envs = 100
    clone_start = time.perf_counter()
    batch = clone_batch_from_env(env, num_envs)
    clone_elapsed = time.perf_counter() - clone_start
    print(f"clone 100 envs (init once): {clone_elapsed:.6f}s")

    snapshot = env.create_snapshot()
    if not snapshot or len(snapshot) != 1:
        raise ValueError("Expected single-env snapshot from base env")
    restore_start = time.perf_counter()
    binding.vec_restore_snapshot_broadcast(batch.c_envs, snapshot[0])
    restore_elapsed = time.perf_counter() - restore_start
    print(f"restore snapshot broadcast (100 envs): {restore_elapsed:.6f}s")
    binding.vec_free_snapshot(snapshot)

    states_before = batch.get_state()
    for i, state in enumerate(states_before):
        entities = state.get("entities") or []
        active_idx = state.get("active_agent_indices") or []
        if active_idx:
            ego = entities[active_idx[0]]
            print(f"clone {i} before: ({ego.get('x', 0.0):.3f}, {ego.get('y', 0.0):.3f})")

    base_actions = np.zeros((num_envs, env.num_agents, 1), dtype=np.int32)
    for i in range(num_envs):
        accel_idx = i % 7
        steer_idx = (i * 3) % 13
        base_actions[i, 0, 0] = accel_idx * 13 + steer_idx

    num_steps = 40
    step_start = time.perf_counter()
    for _ in range(num_steps):
        batch.step(base_actions.reshape(-1, 1))
    step_elapsed = time.perf_counter() - step_start
    print(
        f"step 100 envs ({num_steps} steps): {step_elapsed:.6f}s "
        f"avg {step_elapsed / num_steps:.6f}s/step"
    )

    states = batch.get_state()
    output_dir = os.path.join("artifacts", "drive_batch_clone_demo")
    os.makedirs(output_dir, exist_ok=True)

    viz_config = VizConfig()
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_simulator_state(original_state, viz_config=viz_config.__dict__, test_agent_idx=0, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "original_before.png"), dpi=150)
    plt.close(fig)

    for i, state in enumerate(states):
        entities = state.get("entities") or []
        active_idx = state.get("active_agent_indices") or []
        if active_idx:
            ego = entities[active_idx[0]]
            print(f"clone {i} after: ({ego.get('x', 0.0):.3f}, {ego.get('y', 0.0):.3f})")

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_simulator_state(state, viz_config=viz_config.__dict__, test_agent_idx=0, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"clone_env_{i}.png"), dpi=150)
        plt.close(fig)

    original_state_after = env.get_state()[0]
    original_after_entities = original_state_after.get("entities") or []
    original_after_active_idx = original_state_after.get("active_agent_indices") or []
    if original_after_active_idx:
        original_after_ego = original_after_entities[original_after_active_idx[0]]
        print(
            f"original after: ({original_after_ego.get('x', 0.0):.3f}, {original_after_ego.get('y', 0.0):.3f})"
        )
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_simulator_state(original_state_after, viz_config=viz_config.__dict__, test_agent_idx=0, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "original_after.png"), dpi=150)
    plt.close(fig)

    batch.close()
    env.close()
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
