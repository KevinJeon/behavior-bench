# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Test that removed agents stay removed when using goal_behavior=3 (REMOVE)."""

import os
import sys
import numpy as np
import pytest

# Skip if data root not set
if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
    pytest.skip("DRIVE_BINARIES_DATA_ROOT not set", allow_module_level=True)

from pufferlib.ocean.drive.drive import Drive


def get_active_agent_positions(state):
    """Extract positions of all active (non-removed) agents from state."""
    entities = state.get("entities", [])
    active_indices = state.get("active_agent_indices", [])

    positions = {}
    for idx in (active_indices or []):
        if idx < len(entities):
            e = entities[idx]
            removed = e.get("removed", 0)
            stopped = e.get("stopped", 0)
            x = e.get("x", 0)
            y = e.get("y", 0)
            valid = e.get("valid", 1)
            positions[idx] = {
                "x": x,
                "y": y,
                "removed": removed,
                "stopped": stopped,
                "valid": valid,
            }
    return positions


def count_visible_agents(state):
    """Count agents that should be visible (not removed, not at -10000)."""
    positions = get_active_agent_positions(state)
    visible = 0
    for idx, data in positions.items():
        if data["removed"] == 0 and data["x"] > -9000:
            visible += 1
    return visible


class TestAgentRemovalPersistence:
    """Test that agents stay removed after reaching goal with goal_behavior=3."""

    def test_removed_agents_stay_removed_no_planner(self):
        """Test basic removal persistence without any planner."""
        env = Drive(
            use_all_maps=False,
            resample_frequency=910,
            episode_length=91,
            action_type="continuous",
            max_controlled_agents=128,
            goal_behavior=3,  # REMOVE
            collision_behavior=2,  # REMOVE
            offroad_behavior=2,  # REMOVE
            termination_mode=1,
            map_id=0,
            split="testing",
            goal_speed=100,
            goal_radius=2,
        )

        obs, _ = env.reset()
        num_agents = env.num_agents

        # Track removed agents across steps
        ever_removed = set()

        for step in range(min(30, env.episode_length)):
            # Get state before step
            state = env.get_state()
            if isinstance(state, list):
                state = state[0] if state else {}

            positions = get_active_agent_positions(state)

            # Check which agents are removed
            currently_removed = set()
            for idx, data in positions.items():
                if data["removed"] == 1 or data["x"] < -9000:
                    currently_removed.add(idx)

            # Verify removed agents stay removed
            for idx in ever_removed:
                if idx in positions:
                    agent_data = positions[idx]
                    assert agent_data["removed"] == 1 or agent_data["x"] < -9000, \
                        f"Step {step}: Agent {idx} was removed but is now back! " \
                        f"removed={agent_data['removed']}, x={agent_data['x']}"

            # Update ever_removed
            ever_removed.update(currently_removed)

            # Take random actions
            actions = np.random.uniform(-1, 1, (num_agents, 2)).astype(np.float32)
            obs, rewards, dones, truncs, info = env.step(actions)

            # Stop if episode is done
            if truncs[0]:
                break

        env.close()
        print(f"Test passed: {len(ever_removed)} agents were removed and stayed removed")

    def test_get_state_returns_consistent_data(self):
        """Test that multiple get_state() calls return the same data."""
        env = Drive(
            use_all_maps=False,
            resample_frequency=910,
            episode_length=91,
            action_type="continuous",
            max_controlled_agents=128,
            goal_behavior=3,
            map_id=0,
            split="testing",
        )

        obs, _ = env.reset()
        num_agents = env.num_agents

        for step in range(5):
            # Get state multiple times
            state1 = env.get_state()
            state2 = env.get_state()

            if isinstance(state1, list):
                state1 = state1[0] if state1 else {}
            if isinstance(state2, list):
                state2 = state2[0] if state2 else {}

            pos1 = get_active_agent_positions(state1)
            pos2 = get_active_agent_positions(state2)

            # Check consistency
            assert len(pos1) == len(pos2), f"Step {step}: Different number of agents!"

            for idx in pos1:
                assert idx in pos2, f"Step {step}: Agent {idx} missing in second call"
                assert pos1[idx]["x"] == pos2[idx]["x"], \
                    f"Step {step}: Agent {idx} x differs: {pos1[idx]['x']} vs {pos2[idx]['x']}"
                assert pos1[idx]["removed"] == pos2[idx]["removed"], \
                    f"Step {step}: Agent {idx} removed differs"

            # Step
            actions = np.random.uniform(-1, 1, (num_agents, 2)).astype(np.float32)
            obs, rewards, dones, truncs, info = env.step(actions)

            if truncs[0]:
                break

        env.close()
        print("Test passed: get_state() returns consistent data")


if __name__ == "__main__":
    test = TestAgentRemovalPersistence()

    print("=" * 60)
    print("Test 1: Removed agents stay removed (no planner)")
    print("=" * 60)
    test.test_removed_agents_stay_removed_no_planner()

    print("\n" + "=" * 60)
    print("Test 2: get_state() consistency")
    print("=" * 60)
    test.test_get_state_returns_consistent_data()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
