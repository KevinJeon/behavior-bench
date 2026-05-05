# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Demo: send continuous actions to Drive and plot before/after states.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.ocean.drive.drive import Drive
from pufferlib.viz import plot_simulator_state, VizConfig


def main():
    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        print("Error: DRIVE_BINARIES_DATA_ROOT not set")
        sys.exit(1)

    num_agents = 1
    print("Creating Drive environment...")
    env = Drive(
        num_agents=num_agents,
        num_maps=1,
        resample_frequency=-1,
        episode_length=91,
        action_type="continuous",
        max_controlled_agents=1,
    )
    print("Resetting environment...")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")

    output_dir = os.path.join("artifacts", "drive_continuous_action_demo")
    os.makedirs(output_dir, exist_ok=True)

    # Plot initial state
    print("Getting initial state...")
    states = env.get_state()
    if not states:
        print("Warning: get_state returned empty list")
        env.close()
        return

    state_before = states[0]
    viz_config = VizConfig()

    fig, ax = plt.subplots(figsize=(10, 10))
    plot_simulator_state(state_before, viz_config=viz_config.__dict__, test_agent_idx=0, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "step_00_before.png"), dpi=150)
    plt.close(fig)
    print("Saved initial state plot")

    # Continuous actions are float32 in range [-1, 1]
    # Format: [acceleration, steering]
    action_sequence = np.array(
        [
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [0.5, 0.0],   # Accelerate forward
            [1.0, 0.2],   # Full throttle, slight right
            [0.5, -0.3],  # Medium throttle, left turn
            [0.0, 0.0],   # Coast
            [-0.5, 0.1],  # Brake, slight right
        ],
        dtype=np.float32,
    )

    # Step through environment and plot each step
    for idx, action in enumerate(action_sequence):
        actions = np.ascontiguousarray(action.reshape(num_agents, 2), dtype=np.float32)
        obs, rewards, dones, truncs, infos = env.step(actions)
        print(f"Step {idx + 1}: action={action.tolist()} reward={float(rewards[0]):.4f}")

        # Plot state after each step
        states = env.get_state()
        if states:
            state = states[0]
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_simulator_state(state, viz_config=viz_config.__dict__, test_agent_idx=0, ax=ax)
            ax.set_title(f"Step {idx + 1}: accel={action[0]:.2f}, steer={action[1]:.2f}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"step_{idx + 1:02d}.png"), dpi=150)
            plt.close(fig)

    env.close()
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
