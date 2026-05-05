# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pufferlib.ocean.drive.batch_env import DriveBatch
from pufferlib.ocean.drive.drive import Drive
from pufferlib.viz import plot_simulator_state


def _make_drive_env():
    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        pytest.skip("DRIVE_BINARIES_DATA_ROOT is not set")
    try:
        env = Drive(
            num_agents=32,
            num_maps=1,
            resample_frequency=0,
            episode_length=91,
            use_all_maps=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(str(exc))
    env.reset(seed=0)
    return env


def _fill_actions(actions, value):
    actions[:] = value
    return actions


def _slice_env(obs, env_idx, agents_per_env):
    start = env_idx * agents_per_env
    end = start + agents_per_env
    return obs[start:end]


def test_drive_batch_clone_same_actions_match():
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=3)
    try:
        per_env_actions = np.zeros(
            (batch_env.agents_per_env, batch_env.actions.shape[1]),
            dtype=batch_env.actions.dtype,
        )
        action_value = 1 if np.issubdtype(batch_env.actions.dtype, np.integer) else 0.25
        _fill_actions(per_env_actions, action_value)
        actions = np.tile(per_env_actions, (batch_env.num_envs, 1))

        for _ in range(3):
            obs, rewards, terminals, truncations = batch_env.step(actions)

            ref_obs = _slice_env(obs, 0, batch_env.agents_per_env)
            ref_rewards = _slice_env(rewards, 0, batch_env.agents_per_env)
            ref_terminals = _slice_env(terminals, 0, batch_env.agents_per_env)
            ref_truncations = _slice_env(truncations, 0, batch_env.agents_per_env)

            for env_idx in range(1, batch_env.num_envs):
                np.testing.assert_array_equal(
                    _slice_env(obs, env_idx, batch_env.agents_per_env),
                    ref_obs,
                )
                np.testing.assert_array_equal(
                    _slice_env(rewards, env_idx, batch_env.agents_per_env),
                    ref_rewards,
                )
                np.testing.assert_array_equal(
                    _slice_env(terminals, env_idx, batch_env.agents_per_env),
                    ref_terminals,
                )
                np.testing.assert_array_equal(
                    _slice_env(truncations, env_idx, batch_env.agents_per_env),
                    ref_truncations,
                )
    finally:
        batch_env.close()
        env.close()


def test_drive_batch_clone_isolation_from_original():
    env = _make_drive_env()
    batch_env_a = DriveBatch.from_env(env, num_envs=1)
    batch_env_b = DriveBatch.from_env(env, num_envs=1)
    try:
        initial_env_obs = env.observations.copy()
        initial_clone_b_obs = batch_env_b.observations.copy()

        actions = np.zeros_like(batch_env_a.actions)
        action_value = 2 if np.issubdtype(batch_env_a.actions.dtype, np.integer) else 0.5
        _fill_actions(actions, action_value)

        for _ in range(3):
            batch_env_a.step(actions)

        np.testing.assert_array_equal(env.observations, initial_env_obs)
        np.testing.assert_array_equal(batch_env_b.observations, initial_clone_b_obs)
    finally:
        batch_env_a.close()
        batch_env_b.close()
        env.close()


def test_drive_batch_clone_plot(tmp_path):
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=2)
    try:
        actions = np.zeros_like(batch_env.actions)
        action_value = 1 if np.issubdtype(batch_env.actions.dtype, np.integer) else 0.25
        _fill_actions(actions, action_value)

        states = [batch_env.get_state()]
        for _ in range(2):
            batch_env.step(actions)
            states.append(batch_env.get_state())

        num_steps = len(states)
        fig, axes = plt.subplots(
            batch_env.num_envs,
            num_steps,
            figsize=(4 * num_steps, 4 * batch_env.num_envs),
        )
        axes = np.atleast_2d(axes)

        for env_idx in range(batch_env.num_envs):
            for step_idx, state_set in enumerate(states):
                ax = axes[env_idx, step_idx]
                plot_simulator_state(state_set[env_idx], ax=ax)
                ax.set_title(f"env {env_idx} step {step_idx}")
                ax.axis("off")

        output_path = tmp_path / "drive_batch_clone_plot.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        assert output_path.exists() and output_path.stat().st_size > 0
    finally:
        batch_env.close()
        env.close()


def test_drive_batch_clone_step_changes_observation():
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=1)
    try:
        initial_obs = batch_env.observations.copy()
        actions = np.zeros_like(batch_env.actions)
        action_value = 1 if np.issubdtype(batch_env.actions.dtype, np.integer) else 0.25
        _fill_actions(actions, action_value)

        obs_changed = False
        for _ in range(3):
            obs, _, _, _ = batch_env.step(actions)
            if not np.array_equal(obs, initial_obs):
                obs_changed = True
                break

        assert obs_changed, "Observation did not change after stepping the clone"
    finally:
        batch_env.close()
        env.close()


def test_drive_batch_restore_snapshot_broadcast():
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=3)
    snapshot = None
    try:
        snapshot = env.create_snapshot()
        snapshot_handle = snapshot[0]
        initial_obs = batch_env.observations.copy()

        actions = np.zeros_like(batch_env.actions)
        action_value = 1 if np.issubdtype(batch_env.actions.dtype, np.integer) else 0.25
        _fill_actions(actions, action_value)
        batch_env.step(actions)

        assert not np.array_equal(
            batch_env.observations,
            initial_obs,
        ), "Observation did not change before restore"

        batch_env.restore_snapshot_broadcast(snapshot_handle)
        np.testing.assert_array_equal(batch_env.observations, initial_obs)
    finally:
        if snapshot is not None:
            env.free_snapshot(snapshot)
        batch_env.close()
        env.close()


def test_snapshot_restores_terminals_correctly():
    """Test that terminals are properly restored from snapshot."""
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=5)
    snapshot = None
    try:
        # Create snapshot at initial state
        snapshot = env.create_snapshot()
        snapshot_handle = snapshot[0]

        # Do some steps to cause some agents to crash/stop
        actions = np.ones_like(batch_env.actions) * 0.5
        for _ in range(5):
            obs, rewards, terminals, truncations = batch_env.step(actions)

        # Check that some agents are terminated
        assert terminals.sum() > 0, "Expected some agents to be terminated after steps"

        # Restore snapshot
        batch_env.restore_snapshot_broadcast(snapshot_handle)

        # Step once more with same actions
        obs2, rewards2, terminals2, truncations2 = batch_env.step(actions)

        # After restore, do the same steps again and compare
        batch_env.restore_snapshot_broadcast(snapshot_handle)
        obs_first, rewards_first, terminals_first, truncations_first = batch_env.step(actions)

        # The terminals from the first step after restore should match
        np.testing.assert_array_equal(
            terminals_first,
            terminals2,
            err_msg="Terminals after restore don't match - stopped/removed flags not properly restored!",
        )

    finally:
        if snapshot is not None:
            env.free_snapshot(snapshot)
        batch_env.close()
        env.close()


def test_snapshot_restore_multiple_iterations():
    """Test that multiple snapshot restores work correctly (simulates CEM iterations)."""
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=10)
    snapshot = None
    try:
        # Create snapshot at initial state
        snapshot = env.create_snapshot()
        snapshot_handle = snapshot[0]

        # Simulate multiple CEM iterations
        terminals_history = []
        for iteration in range(5):
            # Restore snapshot (like CEM does before each iteration)
            batch_env.restore_snapshot_broadcast(snapshot_handle)

            # Do rollout with fixed actions
            actions = np.ones_like(batch_env.actions) * 0.3
            obs, rewards, terminals, truncations = batch_env.step(actions)

            terminated_count = terminals.sum()
            terminals_history.append(terminated_count)

        # With the SAME actions and SAME snapshot, all iterations should have
        # the SAME number of terminated agents
        for i in range(1, len(terminals_history)):
            assert (
                terminals_history[i] == terminals_history[0]
            ), f"Iteration {i} has {terminals_history[i]} terminated agents, but iteration 0 had {terminals_history[0]}. This suggests accumulation! History: {terminals_history}"

    finally:
        if snapshot is not None:
            env.free_snapshot(snapshot)
        batch_env.close()
        env.close()


def test_snapshot_restore_multiple_steps_per_iteration():
    """Test snapshot restore with multiple steps per iteration (like CEM rollouts)."""
    env = _make_drive_env()
    batch_env = DriveBatch.from_env(env, num_envs=50)  # Same as CEM population size
    snapshot = None
    try:
        # Create snapshot at initial state
        snapshot = env.create_snapshot()
        snapshot_handle = snapshot[0]

        # Simulate CEM iterations with multi-step rollouts
        first_step_terminals_history = []
        for iteration in range(10):
            # Restore snapshot (like CEM does before each iteration)
            batch_env.restore_snapshot_broadcast(snapshot_handle)

            # Do multi-step rollout (like CEM horizon)
            for step_idx in range(10):
                actions = np.ones_like(batch_env.actions) * 0.3
                obs, rewards, terminals, truncations = batch_env.step(actions)

                # Track terminals after the FIRST step of each iteration
                if step_idx == 0:
                    first_step_terminals_history.append(terminals.sum())

        # After restore, the FIRST step should always have the SAME number of terminals
        print(f"First-step terminals across iterations: {first_step_terminals_history}")
        for i in range(1, len(first_step_terminals_history)):
            assert (
                first_step_terminals_history[i] == first_step_terminals_history[0]
            ), f"Iteration {i} first-step has {first_step_terminals_history[i]} terminated, but iteration 0 had {first_step_terminals_history[0]}. Accumulation detected! History: {first_step_terminals_history}"

    finally:
        if snapshot is not None:
            env.free_snapshot(snapshot)
        batch_env.close()
        env.close()
