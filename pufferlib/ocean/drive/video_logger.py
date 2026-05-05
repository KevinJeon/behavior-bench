# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Matplotlib-based video logger for traffic mix visualization.

Renders top-down episodes with roads (from viz.py) and agents colored by
movement mode (PPO=blue, IDM=green, Expert=gold). Logs GIF videos to wandb.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pufferlib.viz import plot_entity, _draw_agent_box

# Movement mode constants (matching drive.h)
MOVEMENT_DYNAMICS = 0  # PPO
MOVEMENT_IDM = 1
MOVEMENT_EXPERT = 2

MODE_COLORS = {
    MOVEMENT_DYNAMICS: ("#4285F4", "#2a5db0"),  # blue (face, edge)
    MOVEMENT_IDM: ("#34A853", "#1e6e30"),        # green
    MOVEMENT_EXPERT: ("#FBBC04", "#b88a02"),      # gold
}
COLLISION_COLORS = ("#EA4335", "#a22e24")
STATIC_COLORS = ("#9AA0A6", "#6e7278")

MODE_LABELS = {
    MOVEMENT_DYNAMICS: "PPO",
    MOVEMENT_IDM: "IDM",
    MOVEMENT_EXPERT: "Expert",
}

INVALID_POSITION = -10000.0


def _render_frame_on_ax(ax, state, axis_limits=None):
    """Render one frame onto ax. Returns axis_limits for reuse."""
    if isinstance(state, list):
        if not state:
            return axis_limits
        state = state[0]

    entities = state.get("entities")
    if not entities:
        return axis_limits

    active_set = set(state.get("active_agent_indices") or [])
    static_set = set(state.get("static_car_indices") or [])

    # 1) Draw road elements (types 4-9) using viz.py
    for idx, entity in enumerate(entities):
        if entity.get("type", 0) in (4, 5, 6, 7, 8, 9):
            plot_entity(ax, entity, idx, [], [])

    # 2) Draw vehicles/pedestrians/cyclists with traffic-mix colors
    agent_xs, agent_ys = [], []
    mode_counts = {}

    for idx, entity in enumerate(entities):
        etype = entity.get("type", 0)
        if etype not in (1, 2, 3):
            continue
        x, y = entity.get("x", INVALID_POSITION), entity.get("y", INVALID_POSITION)
        if x <= INVALID_POSITION or entity.get("removed", 0) or entity.get("stopped", 0) or entity.get("valid", 1) == 0:
            continue

        heading = entity.get("heading", 0)
        length = max(entity.get("length", 4.5), 0.5)
        width = max(entity.get("width", 2.0), 0.3)
        is_active = idx in active_set
        collision = entity.get("collision_state", 0) > 0
        mode = entity.get("movement_mode", MOVEMENT_DYNAMICS)

        if collision:
            face, edge = COLLISION_COLORS
        elif is_active or mode != MOVEMENT_DYNAMICS:
            face, edge = MODE_COLORS.get(mode, STATIC_COLORS)
            label = MODE_LABELS.get(mode, "?")
            mode_counts[label] = mode_counts.get(label, 0) + 1
        else:
            face, edge = STATIC_COLORS

        alpha = 0.75 if is_active else 0.35
        zorder = 10 if is_active else 5
        _draw_agent_box(ax, x, y, length, width, heading, face, edge, alpha, 1.0, zorder)

        agent_xs.append(x)
        agent_ys.append(y)

    # 3) Legend
    from matplotlib.patches import Patch
    legend_items = []
    for mode_val, label in MODE_LABELS.items():
        c = mode_counts.get(label, 0)
        if c > 0:
            legend_items.append(Patch(color=MODE_COLORS[mode_val][0], label=f"{label} ({c})"))
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", fontsize=7,
                  facecolor="#f0f0f0", edgecolor="#999999")

    # 4) Axis limits: compute from agents on first frame, reuse thereafter
    if axis_limits is None and agent_xs:
        pad = 25.0
        axis_limits = (min(agent_xs) - pad, max(agent_xs) + pad,
                       min(agent_ys) - pad, max(agent_ys) + pad)

    if axis_limits:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])

    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return axis_limits


def run_and_log_episodes(policy, driver_env, device, num_episodes=5):
    """Run episodes directly on driver_env, render frames, log to wandb.

    Uses the actual training env (driver_env) so the visualization reflects
    the exact same agent distribution (PPO/IDM/Expert) as in training.

    Args:
        policy: Trained policy (torch module).
        driver_env: Drive env instance (the actual training env).
        device: torch device for policy inference.
        num_episodes: Number of scenarios to record.

    Returns:
        dict of wandb-loggable videos.
    """
    import torch

    if not getattr(driver_env, "mix_traffic", False):
        return {}

    logs = {}
    try:
        import wandb
    except ImportError:
        return {}

    import pufferlib.pytorch
    import tempfile
    from PIL import Image

    use_rnn = hasattr(policy, "lstm")
    num_agents = driver_env.num_agents
    episode_length = driver_env.episode_length or 91

    for ep in range(num_episodes):
        frames = []
        axis_limits = None

        driver_env.resample_maps()
        obs, _ = driver_env.reset(seed=ep)

        if use_rnn:
            lstm_h = torch.zeros(num_agents, policy.lstm.hidden_size, device=device)
            lstm_c = torch.zeros(num_agents, policy.lstm.hidden_size, device=device)

        def _capture():
            nonlocal axis_limits
            state = driver_env.get_state()
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=80)
            axis_limits = _render_frame_on_ax(ax, state, axis_limits)
            fig.patch.set_facecolor("white")
            plt.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
            plt.close(fig)

        _capture()  # initial frame

        for step_i in range(episode_length - 1):
            obs_tensor = torch.as_tensor(obs).to(device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            with torch.no_grad():
                rnn_state = {}
                if use_rnn:
                    rnn_state["lstm_h"] = lstm_h
                    rnn_state["lstm_c"] = lstm_c

                if hasattr(policy, "forward_eval"):
                    logits, _ = policy.forward_eval(obs_tensor, rnn_state)
                else:
                    logits, _ = policy(obs_tensor)

                if use_rnn:
                    lstm_h = rnn_state.get("lstm_h", lstm_h)
                    lstm_c = rnn_state.get("lstm_c", lstm_c)

                actions, _, _ = pufferlib.pytorch.sample_logits(logits)
                actions = actions.cpu().numpy()

            if actions.ndim == 0:
                actions = actions.reshape(1)
            if actions.shape[0] == 1 and obs.shape[0] > 1:
                actions = np.broadcast_to(actions, (obs.shape[0],) + actions.shape[1:])

            obs, rewards, terminals, truncations, info = driver_env.step(actions)
            _capture()

            if terminals.all() or truncations.all():
                break

        if frames:
            pil_frames = [Image.fromarray(f[:, :, :3]) for f in frames]
            tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            pil_frames[0].save(
                tmp.name, save_all=True, append_images=pil_frames[1:],
                duration=100, loop=0,
            )
            logs[f"traffic_mix/episode_{ep}"] = wandb.Video(tmp.name, format="gif")
            print(f"[VideoLogger] Episode {ep}: {len(frames)} frames")

    return logs
