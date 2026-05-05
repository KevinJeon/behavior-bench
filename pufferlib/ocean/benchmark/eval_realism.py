# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
WOSAC Realism Evaluation for Drive Planners.

Evaluates how well a single planner matches distributional properties of
human driving behavior using WOSAC (Waymo Open Sim Agents Challenge) metrics.

ALL agents in the scene are controlled by the same planner — no ego/traffic
distinction.

Usage:
    python pufferlib/ocean/benchmark/eval_realism.py --planner.type idm --map-ids 0-19
    python pufferlib/ocean/benchmark/eval_realism.py --planner.type smart --map-ids 0-19
    python pufferlib/ocean/benchmark/eval_realism.py --planner.type ppo --map-ids 0-19 \
        --planner.ppo.weights-path path/to/weights.pt
"""

import argparse
import ast
import configparser
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

log = logging.getLogger("realism")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir,
                     planners=None):
    """Render per-map visualization for the current step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pufferlib.viz import plot_simulator_state

    states = env.get_state()
    if isinstance(states, dict):
        states = [states]

    n = min(viz_maps, len(states))
    for map_i in range(n):
        map_viz_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
        os.makedirs(map_viz_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 12))
        plot_simulator_state(states[map_i], ax=ax)

        # Overlay SMART predicted trajectories (transform global -> state coords)
        if planners is not None and map_i < len(planners):
            planner = planners[map_i]
            if planner._predicted_positions is not None:
                t = planner.history.num_steps - 1
                state = states[map_i]
                entities = state.get("entities", [])
                active = state.get("active_agent_indices", [])
                if t >= 0 and active:
                    N = planner._predicted_positions.shape[0]
                    for j in range(min(N, len(active))):
                        entity_idx = active[j]
                        if entity_idx >= len(entities):
                            continue
                        entity = entities[entity_idx]
                        local_x = entity.get("x", 0)
                        local_y = entity.get("y", 0)
                        if local_x < -9000:  # removed agent
                            continue
                        global_x = planner.history.position[j, t, 0]
                        global_y = planner.history.position[j, t, 1]
                        offset = np.array([global_x - local_x, global_y - local_y],
                                          dtype=np.float32)
                        traj = planner._predicted_positions[j] - offset  # (80, 2)
                        ax.plot(traj[:, 0], traj[:, 1], '-', color='red',
                                alpha=0.3, linewidth=1, zorder=50)

        ax.set_title(f"Map {map_i} | Rollout {rollout_idx + 1} | Step {step_idx + 1}/{sim_steps}")
        fig.tight_layout()
        fig.savefig(os.path.join(map_viz_dir, f"step_{step_idx:03d}.png"), dpi=100)
        plt.close(fig)


def _create_gif(image_dir, output_path, fps=10):
    """Create GIF from step PNGs in a directory."""
    import glob as globlib
    from PIL import Image

    pattern = os.path.join(image_dir, "step_*.png")
    image_files = sorted(globlib.glob(pattern))
    if not image_files:
        return

    frames = [Image.open(f) for f in image_files]
    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


# ---------------------------------------------------------------------------
# Config loading (mirrors registry.py pattern but uses realism.ini)
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "config"
)
_DEFAULT_CONFIG = os.path.join(_CONFIG_DIR, "realism.ini")


def _puffer_type(value):
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _build_nested_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    result = defaultdict(dict)
    for key, value in flat.items():
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return dict(result)


def load_realism_config(argv=None) -> Dict[str, Any]:
    """Load realism config from INI + CLI overrides."""
    p = configparser.ConfigParser()
    p.read(_DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(
        description="WOSAC Realism Evaluation for Drive Planners",
    )

    for section in p.sections():
        for key in p[section]:
            fmt = f"--{section}.{key}"
            default = _puffer_type(p[section][key])
            parser.add_argument(fmt.replace("_", "-"), default=default, type=_puffer_type)

    parser.add_argument("--map-ids", type=str, default=None,
                        help="Map IDs: 'all', range '0-100', or comma-separated '0,5,10'")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom realism.ini")

    args = vars(parser.parse_args(argv))

    custom_config = args.pop("config", None)
    if custom_config and os.path.isfile(custom_config):
        p2 = configparser.ConfigParser()
        p2.read([_DEFAULT_CONFIG, custom_config])
        for section in p2.sections():
            for key in p2[section]:
                flat_key = f"{section}.{key}"
                if flat_key not in args or args[flat_key] == _puffer_type(p[section].get(key, "")):
                    args[flat_key] = _puffer_type(p2[section][key])

    map_ids_str = args.pop("map_ids", None)
    output_dir = args.pop("output_dir", None)

    config = _build_nested_dict(args)
    config["map_ids"] = map_ids_str
    config["output_dir"] = output_dir
    return config


# ---------------------------------------------------------------------------
# Map ID parsing (same as eval.py)
# ---------------------------------------------------------------------------

def _parse_map_ids(map_ids_str, split):
    data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    split_dir = os.path.join(data_root, split)

    if map_ids_str is None or map_ids_str.lower() == "all":
        if os.path.isdir(split_dir):
            bin_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".bin"))
            return list(range(len(bin_files)))
        log.error("Split directory not found: %s", split_dir)
        sys.exit(1)

    if "-" in map_ids_str and "," not in map_ids_str:
        start, end = map_ids_str.split("-")
        return list(range(int(start.strip()), int(end.strip()) + 1))

    return [int(x.strip()) for x in map_ids_str.split(",")]


# ---------------------------------------------------------------------------
# Planner creation (all agents, single planner type)
# ---------------------------------------------------------------------------

def _create_all_agent_planner(planner_type, type_cfg, env, num_agents, action_config, episode_length,
                               smart_model=None, smart_codebooks=None, map_env_index=0):
    """Create a planner that controls ALL agents in the scene.

    Returns a callable: step(current_step, obs) -> actions (num_agents, 2)
    """
    from pufferlib.evaluation.config import ActionConfig
    ac_lb, ac_ub = action_config.bounds

    if planner_type == "idm":
        from pufferlib.planning.idm import IDMPlanner
        agent_indices = list(range(num_agents))
        horizon = int(type_cfg.get("horizon", 40))
        tv = float(type_cfg.get("target_velocity", 15.0))
        planner = IDMPlanner(
            env=env, agent_indices=agent_indices, horizon=horizon,
            action_lb=ac_lb, action_ub=ac_ub, target_velocity=tv,
        )
        neutral = np.zeros((num_agents, 2), dtype=np.float32)

        def step_fn(current_step, obs):
            return neutral

        step_fn.reset = planner.reset
        return step_fn

    elif planner_type == "smart":
        from pufferlib.planning.smart import SMARTPlanner, SMARTConfig
        smart_cfg = SMARTConfig(
            weights_path=str(type_cfg.get("weights_path", "")),
            device=str(type_cfg.get("device", "cuda")),
            temperature=float(type_cfg.get("temperature", 1.0)),
            greedy=bool(type_cfg.get("greedy", False)),
            repredict_interval=int(type_cfg.get("repredict_interval", 1)),
        )
        planner = SMARTPlanner(
            env=env, agent_idx=0, action_lb=ac_lb, action_ub=ac_ub, config=smart_cfg,
            model=smart_model,
            motion_codebook_data=smart_codebooks[0] if smart_codebooks else None,
            map_codebook=smart_codebooks[1] if smart_codebooks else None,
            map_env_index=map_env_index,
        )

        def step_fn(current_step, obs):
            # Single-agent call triggers history recording + inference
            planner.plan(current_step=current_step, obs=obs[0])
            # Extract actions for ALL agents from cached predictions
            all_indices = np.arange(num_agents, dtype=np.int32)
            return planner._positions_to_actions(all_indices, current_step)

        step_fn.reset = planner.reset
        return step_fn

    elif planner_type == "ppo":
        from pufferlib.planning.policy import PPOPlanner, PPOConfig
        ppo_cfg = PPOConfig(
            weights_path=str(type_cfg.get("weights_path", "")),
            device=str(type_cfg.get("device", "cuda")),
            stochastic=bool(type_cfg.get("stochastic", False)),
            temperature=float(type_cfg.get("temperature", 1.0)),
        )
        planner = PPOPlanner(
            env=env, agent_idx=0, action_lb=ac_lb, action_ub=ac_ub, config=ppo_cfg,
        )

        def step_fn(current_step, obs):
            # Batch inference for all agents
            return planner.plan(current_step=current_step, obs=obs)

        step_fn.reset = planner.reset
        return step_fn

    elif planner_type == "constant_velocity":
        neutral = np.zeros((num_agents, 2), dtype=np.float32)

        def step_fn(current_step, obs):
            return neutral

        step_fn.reset = lambda: None
        return step_fn

    elif planner_type == "pdm":
        from pufferlib.planning.pdm import PDMPlanner, PDMConfig
        vf = type_cfg.get("velocity_fractions", (0.2, 0.4, 0.6, 0.8, 1.0))
        lo = type_cfg.get("lateral_offsets", (-1.0, 0.0, 1.0))
        pdm_cfg = PDMConfig(
            horizon=int(type_cfg.get("horizon", 40)),
            episode_length=episode_length,
            velocity_fractions=tuple(vf) if not isinstance(vf, tuple) else vf,
            lateral_offsets=tuple(lo) if not isinstance(lo, tuple) else lo,
            proposal_other_planner=str(type_cfg.get("proposal_other", "constant_velocity")),
        )
        log.warning("PDM is a single-agent planner — only agent 0 will be actively controlled. "
                     "Other agents get neutral actions.")
        planner = PDMPlanner(
            env=env, agent_idx=0, action_lb=ac_lb, action_ub=ac_ub,
            config=pdm_cfg, other_planner=None,
        )

        def step_fn(current_step, obs):
            actions = np.zeros((num_agents, 2), dtype=np.float32)
            actions[0] = planner.plan(current_step=current_step, obs=obs[0])
            return actions

        step_fn.reset = planner.reset
        return step_fn

    else:
        raise ValueError(f"Unknown planner type: {planner_type}")


def _generate_random_trajectories(gt, num_rollouts, sim_steps):
    """Generate WOSAC 2023 kinematic random baseline trajectories.

    Samples (dx, dy, d_heading) from N(mu=1.0, sigma=0.1) at each timestep
    and propagates in the agent's local coordinate frame. No simulator needed.
    """
    num_agents = gt["x"].shape[0]

    # Initial positions from GT (step 0)
    x0 = gt["x"][:, 0, 0]       # (num_agents,)
    y0 = gt["y"][:, 0, 0]
    h0 = gt["heading"][:, 0, 0]

    sim = {
        "x": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "y": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "z": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "heading": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
        "id": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.int32),
    }
    # All steps are valid (no simulator removal)
    sim_valid = np.ones((num_agents, num_rollouts, sim_steps), dtype=bool)

    for r in range(num_rollouts):
        # Sample random deltas: shape (num_agents, sim_steps, 3) for (dx, dy, d_heading)
        samples = np.random.normal(loc=1.0, scale=0.1, size=(num_agents, sim_steps, 3)).astype(np.float32)

        x = x0.copy()
        y = y0.copy()
        h = h0.copy()

        for t in range(sim_steps):
            sim["x"][:, r, t] = x
            sim["y"][:, r, t] = y
            sim["heading"][:, r, t] = h
            sim["id"][:, r, t] = gt["id"][:, 0]

            dx = samples[:, t, 0]
            dy = samples[:, t, 1]
            dh = samples[:, t, 2]

            # Propagate in agent's local frame
            cos_h = np.cos(h)
            sin_h = np.sin(h)
            x += dx * cos_h - dy * sin_h
            y += dx * sin_h + dy * cos_h
            h += dh

    return sim, sim_valid


# ---------------------------------------------------------------------------
# Batch evaluation for a single map (legacy path)
# ---------------------------------------------------------------------------

def _evaluate_single_map(env, map_id, map_idx, planner_type, type_cfg, action_config,
                          num_rollouts, sim_steps, episode_length,
                          smart_model=None, smart_codebooks=None):
    """Run rollouts for a single map. Returns (gt, sim, sim_valid, agent_state, road_edges)."""
    obs, _ = env.reset()
    num_agents = env.num_agents

    gt = env.get_ground_truth_trajectories()
    agent_state = env.get_global_agent_state()
    road_edges = env.get_road_edge_polylines()

    if planner_type == "random":
        map_sim, map_sim_valid = _generate_random_trajectories(gt, num_rollouts, sim_steps)
    else:
        map_sim = {
            "x": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
            "y": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
            "z": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.float32),
            "id": np.zeros((num_agents, num_rollouts, sim_steps), dtype=np.int32),
        }
        map_sim_valid = np.ones((num_agents, num_rollouts, sim_steps), dtype=bool)

        # Create planner ONCE outside rollout loop (shared model, no reload)
        step_fn = _create_all_agent_planner(
            planner_type, type_cfg, env, num_agents, action_config, episode_length,
            smart_model=smart_model, smart_codebooks=smart_codebooks,
        )

        for rollout_idx in range(num_rollouts):
            obs, _ = env.reset()
            step_fn.reset()

            for step_idx in range(sim_steps):
                state = env.get_global_agent_state()
                map_sim["x"][:, rollout_idx, step_idx] = state["x"][:num_agents]
                map_sim["y"][:, rollout_idx, step_idx] = state["y"][:num_agents]
                map_sim["z"][:, rollout_idx, step_idx] = state["z"][:num_agents]
                map_sim["heading"][:, rollout_idx, step_idx] = state["heading"][:num_agents]
                map_sim["id"][:, rollout_idx, step_idx] = state["id"][:num_agents]

                obs = np.atleast_2d(obs)
                actions = step_fn(step_idx, obs)
                actions = np.atleast_2d(actions)

                if actions.shape[0] < num_agents:
                    full_actions = np.zeros((num_agents, 2), dtype=np.float32)
                    full_actions[:actions.shape[0]] = actions
                    actions = full_actions

                obs, rewards, dones, truncs, infos = env.step(actions[:num_agents])

            # Detect removed agents (x=-10000 means removed/invalid)
            for agent_i in range(num_agents):
                x0 = map_sim["x"][agent_i, rollout_idx, 0]
                # Agent already invalid at start
                if x0 < -9000:
                    map_sim_valid[agent_i, rollout_idx, :] = False
                    continue
                for t in range(1, sim_steps):
                    xt = map_sim["x"][agent_i, rollout_idx, t]
                    if xt < -9000 or abs(xt - x0) > 5000:
                        map_sim_valid[agent_i, rollout_idx, t:] = False
                        break

        del step_fn

    return gt, map_sim, map_sim_valid, agent_state, road_edges


# ---------------------------------------------------------------------------
# Batch map evaluation
# ---------------------------------------------------------------------------

def _evaluate_batch(env, planner_type, type_cfg, action_config,
                    num_rollouts, sim_steps, episode_length,
                    smart_model=None, smart_codebooks=None,
                    viz=False, viz_maps=0, output_dir=None):
    """Run rollouts for ALL maps in a batched Drive env.

    The Drive env contains num_envs maps simultaneously. Each map's agents
    are at env.agent_offsets[i]:env.agent_offsets[i+1].

    Returns (gt, sim, sim_valid, agent_state, road_edges) for all maps combined.
    """
    obs, _ = env.reset()
    total_agents = env.num_agents

    gt = env.get_ground_truth_trajectories()
    agent_state = env.get_global_agent_state()
    road_edges = env.get_road_edge_polylines()

    if planner_type == "random":
        map_sim, map_sim_valid = _generate_random_trajectories(gt, num_rollouts, sim_steps)
    else:
        map_sim = {
            "x": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
            "y": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
            "z": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
            "heading": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
            "id": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.int32),
        }
        map_sim_valid = np.ones((total_agents, num_rollouts, sim_steps), dtype=bool)

        if planner_type == "smart":
            # Batched SMART inference: one forward pass for all maps
            from pufferlib.planning.smart import SMARTPlanner, SMARTConfig, BatchSMARTController
            from pufferlib.evaluation.config import ActionConfig as _AC
            ac_lb, ac_ub = action_config.bounds

            smart_cfg = SMARTConfig(
                weights_path=str(type_cfg.get("weights_path", "")),
                device=str(type_cfg.get("device", "cuda")),
                temperature=float(type_cfg.get("temperature", 1.0)),
                greedy=bool(type_cfg.get("greedy", False)),
                repredict_interval=int(type_cfg.get("repredict_interval", 1)),
            )

            planners = []
            for i in range(env.num_envs):
                planner = SMARTPlanner(
                    env=env, agent_idx=0, action_lb=ac_lb, action_ub=ac_ub,
                    config=smart_cfg, model=smart_model,
                    motion_codebook_data=smart_codebooks[0] if smart_codebooks else None,
                    map_codebook=smart_codebooks[1] if smart_codebooks else None,
                    map_env_index=i,
                )
                planners.append(planner)

            batch_ctrl = BatchSMARTController(
                planners=planners,
                model=smart_model,
                device=next(smart_model.parameters()).device,
                greedy=smart_cfg.greedy,
                temperature=smart_cfg.temperature,
            )

            import time as _time
            for rollout_idx in range(num_rollouts):
                obs, _ = env.reset()
                batch_ctrl.reset()
                _t_rollout = _time.perf_counter()

                for step_idx in range(sim_steps):
                    state = env.get_global_agent_state()
                    map_sim["x"][:, rollout_idx, step_idx] = state["x"][:total_agents]
                    map_sim["y"][:, rollout_idx, step_idx] = state["y"][:total_agents]
                    map_sim["z"][:, rollout_idx, step_idx] = state["z"][:total_agents]
                    map_sim["heading"][:, rollout_idx, step_idx] = state["heading"][:total_agents]
                    map_sim["id"][:, rollout_idx, step_idx] = state["id"][:total_agents]

                    actions = batch_ctrl.step(step_idx, env, obs)
                    obs, rewards, dones, truncs, infos = env.step(actions[:total_agents])

                    if viz and viz_maps > 0:
                        _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir,
                                         planners=planners)

                    elapsed = _time.perf_counter() - _t_rollout
                    avg_step = elapsed / (step_idx + 1)
                    eta = avg_step * (sim_steps - step_idx - 1)
                    log.info("  Rollout %d/%d  Step %d/%d  (%.1fs elapsed, ETA %.0fs)",
                             rollout_idx + 1, num_rollouts, step_idx + 1, sim_steps,
                             elapsed, eta)

                # Detect removed agents (x=-10000 means removed/invalid)
                for agent_i in range(total_agents):
                    x0 = map_sim["x"][agent_i, rollout_idx, 0]
                    # Agent already invalid at start
                    if x0 < -9000:
                        map_sim_valid[agent_i, rollout_idx, :] = False
                        continue
                    for t in range(1, sim_steps):
                        xt = map_sim["x"][agent_i, rollout_idx, t]
                        if xt < -9000 or abs(xt - x0) > 5000:
                            map_sim_valid[agent_i, rollout_idx, t:] = False
                            break

                # Create GIFs for this rollout
                if viz and viz_maps > 0:
                    for map_i in range(min(viz_maps, env.num_envs)):
                        img_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
                        gif_path = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}.gif")
                        _create_gif(img_dir, gif_path)
                    log.info("  Saved viz GIFs for %d maps", min(viz_maps, env.num_envs))

            del batch_ctrl, planners

        elif planner_type == "ppo":
            # Batched PPO: discrete policy → sample discrete action → convert to continuous
            # (C++ backend ignores discrete actions, so we convert manually)
            import pufferlib.pytorch
            from pufferlib.ocean.drive.drive import Drive
            from pufferlib.ocean.torch import Drive as DrivePolicy
            from pufferlib.models import LSTMWrapper

            ppo_device = torch.device(str(type_cfg.get("device", "cuda")))
            weights_path = str(type_cfg.get("weights_path", ""))
            hidden_size = 256

            # Discrete action mapping (7 accel × 13 steer = 91 actions)
            _accel_vals = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
            _steer_vals = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
            _num_steer = len(_steer_vals)

            # Build policy with discrete action space (matching training architecture)
            policy_env = Drive(
                action_type="discrete", episode_length=episode_length,
                max_controlled_agents=1,
            )
            base_policy = DrivePolicy(policy_env, input_size=64, hidden_size=hidden_size)
            policy = LSTMWrapper(
                policy_env, base_policy, input_size=hidden_size, hidden_size=hidden_size,
            ).to(ppo_device)
            policy.eval()
            policy_env.close()

            # Load weights
            ckpt = torch.load(weights_path, map_location=ppo_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            new_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
            result = policy.load_state_dict(new_sd, strict=False)
            if result.missing_keys:
                log.warning("PPO missing keys: %s", result.missing_keys)
            log.info("Loaded PPO weights from %s", weights_path)

            import time as _time
            for rollout_idx in range(num_rollouts):
                obs, _ = env.reset()
                lstm_state = dict(
                    lstm_h=torch.zeros(total_agents, hidden_size, device=ppo_device),
                    lstm_c=torch.zeros(total_agents, hidden_size, device=ppo_device),
                )
                _t_rollout = _time.perf_counter()

                for step_idx in range(sim_steps):
                    agent_st = env.get_global_agent_state()
                    map_sim["x"][:, rollout_idx, step_idx] = agent_st["x"][:total_agents]
                    map_sim["y"][:, rollout_idx, step_idx] = agent_st["y"][:total_agents]
                    map_sim["z"][:, rollout_idx, step_idx] = agent_st["z"][:total_agents]
                    map_sim["heading"][:, rollout_idx, step_idx] = agent_st["heading"][:total_agents]
                    map_sim["id"][:, rollout_idx, step_idx] = agent_st["id"][:total_agents]

                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(obs[:total_agents]).float().to(ppo_device)
                        logits, value = policy.forward_eval(ob_tensor, lstm_state)
                        # Temperature scaling for rollout diversity
                        ppo_temperature = float(type_cfg.get("temperature", 1.0))
                        if ppo_temperature != 1.0:
                            if isinstance(logits, torch.Tensor):
                                logits = logits / ppo_temperature
                            elif isinstance(logits, (list, tuple)):
                                logits = [l / ppo_temperature for l in logits]
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)

                    # Convert discrete action index → continuous (accel, steer)
                    flat_idx = action.cpu().numpy().flatten()
                    accel_idx = flat_idx // _num_steer
                    steer_idx = flat_idx % _num_steer
                    accel = _accel_vals[accel_idx] / np.max(np.abs(_accel_vals))
                    steer = _steer_vals[steer_idx]
                    cont_actions = np.stack([accel, steer], axis=-1).astype(np.float32)

                    obs, rewards, dones, truncs, infos = env.step(cont_actions[:total_agents])

                    if viz and viz_maps > 0:
                        _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir)

                    elapsed = _time.perf_counter() - _t_rollout
                    avg_step = elapsed / (step_idx + 1)
                    eta = avg_step * (sim_steps - step_idx - 1)
                    log.info("  Rollout %d/%d  Step %d/%d  (%.1fs elapsed, ETA %.0fs)",
                             rollout_idx + 1, num_rollouts, step_idx + 1, sim_steps,
                             elapsed, eta)

                # Detect removed agents (x=-10000 means removed/invalid)
                for agent_i in range(total_agents):
                    x0 = map_sim["x"][agent_i, rollout_idx, 0]
                    # Agent already invalid at start
                    if x0 < -9000:
                        map_sim_valid[agent_i, rollout_idx, :] = False
                        continue
                    for t in range(1, sim_steps):
                        xt = map_sim["x"][agent_i, rollout_idx, t]
                        if xt < -9000 or abs(xt - x0) > 5000:
                            map_sim_valid[agent_i, rollout_idx, t:] = False
                            break

                # Create GIFs for this rollout
                if viz and viz_maps > 0:
                    for map_i in range(min(viz_maps, env.num_envs)):
                        img_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
                        gif_path = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}.gif")
                        _create_gif(img_dir, gif_path)
                    log.info("  Saved viz GIFs for %d maps", min(viz_maps, env.num_envs))

            del policy

        elif planner_type in ("conditioned", "conditioned_mix",
                              "conditioned_aggr", "conditioned_normal",
                              "conditioned_caut"):
            # Batched DriveConditioned (PPO-arch + 9-dim reward-conditioning branch).
            # Same action decoding as ppo; env must emit the 9-float creward
            # block, which requires reward_conditioning=1 at env-init time.
            import pufferlib.pytorch
            from pufferlib.ocean.drive.drive import Drive
            from pufferlib.ocean.torch import DriveConditioned as DriveConditionedPolicy
            from pufferlib.models import LSTMWrapper

            ppo_device = torch.device(str(type_cfg.get("device", "cuda")))
            weights_path = str(type_cfg.get("weights_path", ""))
            hidden_size = 256

            _accel_vals = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
            _steer_vals = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
            _num_steer = len(_steer_vals)

            policy_env = Drive(
                action_type="discrete", episode_length=episode_length,
                max_controlled_agents=1,
                reward_conditioning=True,
            )
            base_policy = DriveConditionedPolicy(policy_env, input_size=64, hidden_size=hidden_size)
            policy = LSTMWrapper(
                policy_env, base_policy, input_size=hidden_size, hidden_size=hidden_size,
            ).to(ppo_device)
            policy.eval()
            policy_env.close()

            ckpt = torch.load(weights_path, map_location=ppo_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            new_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
            result = policy.load_state_dict(new_sd, strict=False)
            if result.missing_keys:
                log.warning("Conditioned missing keys: %s", result.missing_keys)
            log.info("Loaded Conditioned PPO weights from %s", weights_path)

            import time as _time
            for rollout_idx in range(num_rollouts):
                obs, _ = env.reset()
                lstm_state = dict(
                    lstm_h=torch.zeros(total_agents, hidden_size, device=ppo_device),
                    lstm_c=torch.zeros(total_agents, hidden_size, device=ppo_device),
                )
                _t_rollout = _time.perf_counter()

                for step_idx in range(sim_steps):
                    agent_st = env.get_global_agent_state()
                    map_sim["x"][:, rollout_idx, step_idx] = agent_st["x"][:total_agents]
                    map_sim["y"][:, rollout_idx, step_idx] = agent_st["y"][:total_agents]
                    map_sim["z"][:, rollout_idx, step_idx] = agent_st["z"][:total_agents]
                    map_sim["heading"][:, rollout_idx, step_idx] = agent_st["heading"][:total_agents]
                    map_sim["id"][:, rollout_idx, step_idx] = agent_st["id"][:total_agents]

                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(obs[:total_agents]).float().to(ppo_device)
                        logits, value = policy.forward_eval(ob_tensor, lstm_state)
                        ppo_temperature = float(type_cfg.get("temperature", 1.0))
                        if ppo_temperature != 1.0:
                            if isinstance(logits, torch.Tensor):
                                logits = logits / ppo_temperature
                            elif isinstance(logits, (list, tuple)):
                                logits = [l / ppo_temperature for l in logits]
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)

                    flat_idx = action.cpu().numpy().flatten()
                    accel_idx = flat_idx // _num_steer
                    steer_idx = flat_idx % _num_steer
                    accel = _accel_vals[accel_idx] / np.max(np.abs(_accel_vals))
                    steer = _steer_vals[steer_idx]
                    cont_actions = np.stack([accel, steer], axis=-1).astype(np.float32)

                    obs, rewards, dones, truncs, infos = env.step(cont_actions[:total_agents])

                    if viz and viz_maps > 0:
                        _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir)

                    elapsed = _time.perf_counter() - _t_rollout
                    avg_step = elapsed / (step_idx + 1)
                    eta = avg_step * (sim_steps - step_idx - 1)
                    log.info("  Rollout %d/%d  Step %d/%d  (%.1fs elapsed, ETA %.0fs)",
                             rollout_idx + 1, num_rollouts, step_idx + 1, sim_steps,
                             elapsed, eta)

                for agent_i in range(total_agents):
                    x0 = map_sim["x"][agent_i, rollout_idx, 0]
                    if x0 < -9000:
                        map_sim_valid[agent_i, rollout_idx, :] = False
                        continue
                    for t in range(1, sim_steps):
                        xt = map_sim["x"][agent_i, rollout_idx, t]
                        if xt < -9000 or abs(xt - x0) > 5000:
                            map_sim_valid[agent_i, rollout_idx, t:] = False
                            break

                if viz and viz_maps > 0:
                    for map_i in range(min(viz_maps, env.num_envs)):
                        img_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
                        gif_path = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}.gif")
                        _create_gif(img_dir, gif_path)
                    log.info("  Saved viz GIFs for %d maps", min(viz_maps, env.num_envs))

            del policy

        elif planner_type == "ppo_nogoal":
            # Batched PPO with DriveNoGoal policy (ignores goal features in observation)
            import pufferlib.pytorch
            from pufferlib.ocean.drive.drive import Drive
            from pufferlib.ocean.torch import DriveNoGoal as DriveNoGoalPolicy
            from pufferlib.models import LSTMWrapper

            ppo_device = torch.device(str(type_cfg.get("device", "cuda")))
            weights_path = str(type_cfg.get("weights_path", ""))
            hidden_size = 256

            # Discrete action mapping (7 accel × 13 steer = 91 actions)
            _accel_vals = np.array([-4.0, -2.67, -1.33, 0.0, 1.33, 2.67, 4.0], dtype=np.float32)
            _steer_vals = np.linspace(-1.0, 1.0, 13, dtype=np.float32)
            _num_steer = len(_steer_vals)

            # Build policy with discrete action space (DriveNoGoal architecture)
            policy_env = Drive(
                action_type="discrete", episode_length=episode_length,
                max_controlled_agents=1,
            )
            base_policy = DriveNoGoalPolicy(policy_env, input_size=64, hidden_size=hidden_size)
            policy = LSTMWrapper(
                policy_env, base_policy, input_size=hidden_size, hidden_size=hidden_size,
            ).to(ppo_device)
            policy.eval()
            policy_env.close()

            # Load weights
            ckpt = torch.load(weights_path, map_location=ppo_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            new_sd = {k.removeprefix("module."): v for k, v in state_dict.items()}
            result = policy.load_state_dict(new_sd, strict=False)
            if result.missing_keys:
                log.warning("PPO NoGoal missing keys: %s", result.missing_keys)
            log.info("Loaded PPO NoGoal weights from %s", weights_path)

            import time as _time
            for rollout_idx in range(num_rollouts):
                obs, _ = env.reset()
                lstm_state = dict(
                    lstm_h=torch.zeros(total_agents, hidden_size, device=ppo_device),
                    lstm_c=torch.zeros(total_agents, hidden_size, device=ppo_device),
                )
                _t_rollout = _time.perf_counter()

                for step_idx in range(sim_steps):
                    agent_st = env.get_global_agent_state()
                    map_sim["x"][:, rollout_idx, step_idx] = agent_st["x"][:total_agents]
                    map_sim["y"][:, rollout_idx, step_idx] = agent_st["y"][:total_agents]
                    map_sim["z"][:, rollout_idx, step_idx] = agent_st["z"][:total_agents]
                    map_sim["heading"][:, rollout_idx, step_idx] = agent_st["heading"][:total_agents]
                    map_sim["id"][:, rollout_idx, step_idx] = agent_st["id"][:total_agents]

                    with torch.no_grad():
                        ob_tensor = torch.as_tensor(obs[:total_agents]).float().to(ppo_device)
                        logits, value = policy.forward_eval(ob_tensor, lstm_state)
                        ppo_temperature = float(type_cfg.get("temperature", 1.0))
                        if ppo_temperature != 1.0:
                            if isinstance(logits, torch.Tensor):
                                logits = logits / ppo_temperature
                            elif isinstance(logits, (list, tuple)):
                                logits = [l / ppo_temperature for l in logits]
                        action, _, _ = pufferlib.pytorch.sample_logits(logits)

                    # Convert discrete action index → continuous (accel, steer)
                    flat_idx = action.cpu().numpy().flatten()
                    accel_idx = flat_idx // _num_steer
                    steer_idx = flat_idx % _num_steer
                    accel = _accel_vals[accel_idx] / np.max(np.abs(_accel_vals))
                    steer = _steer_vals[steer_idx]
                    cont_actions = np.stack([accel, steer], axis=-1).astype(np.float32)

                    obs, rewards, dones, truncs, infos = env.step(cont_actions[:total_agents])

                    if viz and viz_maps > 0:
                        _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir)

                    elapsed = _time.perf_counter() - _t_rollout
                    avg_step = elapsed / (step_idx + 1)
                    eta = avg_step * (sim_steps - step_idx - 1)
                    log.info("  Rollout %d/%d  Step %d/%d  (%.1fs elapsed, ETA %.0fs)",
                             rollout_idx + 1, num_rollouts, step_idx + 1, sim_steps,
                             elapsed, eta)

                # Detect removed agents (x=-10000 means removed/invalid)
                for agent_i in range(total_agents):
                    x0 = map_sim["x"][agent_i, rollout_idx, 0]
                    # Agent already invalid at start
                    if x0 < -9000:
                        map_sim_valid[agent_i, rollout_idx, :] = False
                        continue
                    for t in range(1, sim_steps):
                        xt = map_sim["x"][agent_i, rollout_idx, t]
                        if xt < -9000 or abs(xt - x0) > 5000:
                            map_sim_valid[agent_i, rollout_idx, t:] = False
                            break

                # Create GIFs for this rollout
                if viz and viz_maps > 0:
                    for map_i in range(min(viz_maps, env.num_envs)):
                        img_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
                        gif_path = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}.gif")
                        _create_gif(img_dir, gif_path)
                    log.info("  Saved viz GIFs for %d maps", min(viz_maps, env.num_envs))

            del policy

        else:
            # Sequential path for other planners (idm, pdm, etc.)
            step_fns = []
            for i in range(env.num_envs):
                cur = env.agent_offsets[i]
                nxt = env.agent_offsets[i + 1]
                n_agents = nxt - cur
                fn = _create_all_agent_planner(
                    planner_type, type_cfg, env, n_agents, action_config, episode_length,
                    smart_model=smart_model, smart_codebooks=smart_codebooks,
                    map_env_index=i,
                )
                step_fns.append(fn)

            for rollout_idx in range(num_rollouts):
                obs, _ = env.reset()
                for fn in step_fns:
                    fn.reset()

                for step_idx in range(sim_steps):
                    state = env.get_global_agent_state()
                    map_sim["x"][:, rollout_idx, step_idx] = state["x"][:total_agents]
                    map_sim["y"][:, rollout_idx, step_idx] = state["y"][:total_agents]
                    map_sim["z"][:, rollout_idx, step_idx] = state["z"][:total_agents]
                    map_sim["heading"][:, rollout_idx, step_idx] = state["heading"][:total_agents]
                    map_sim["id"][:, rollout_idx, step_idx] = state["id"][:total_agents]

                    # Compute actions per map, then step all at once
                    actions = np.zeros((total_agents, 2), dtype=np.float32)
                    for i, fn in enumerate(step_fns):
                        cur = env.agent_offsets[i]
                        nxt = env.agent_offsets[i + 1]
                        n_agents = nxt - cur
                        map_obs = np.atleast_2d(obs[cur:nxt] if hasattr(obs, '__getitem__') else obs)
                        map_actions = fn(step_idx, map_obs)
                        map_actions = np.atleast_2d(map_actions)
                        actions[cur:cur + min(map_actions.shape[0], n_agents)] = map_actions[:n_agents]

                    obs, rewards, dones, truncs, infos = env.step(actions[:total_agents])

                    if viz and viz_maps > 0:
                        _render_viz_step(env, step_idx, rollout_idx, sim_steps, viz_maps, output_dir)

                # Detect removed agents (x=-10000 means removed/invalid)
                for agent_i in range(total_agents):
                    x0 = map_sim["x"][agent_i, rollout_idx, 0]
                    # Agent already invalid at start
                    if x0 < -9000:
                        map_sim_valid[agent_i, rollout_idx, :] = False
                        continue
                    for t in range(1, sim_steps):
                        xt = map_sim["x"][agent_i, rollout_idx, t]
                        if xt < -9000 or abs(xt - x0) > 5000:
                            map_sim_valid[agent_i, rollout_idx, t:] = False
                            break

                # Create GIFs for this rollout
                if viz and viz_maps > 0:
                    for map_i in range(min(viz_maps, env.num_envs)):
                        img_dir = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}")
                        gif_path = os.path.join(output_dir, f"map_{map_i:03d}", f"rollout_{rollout_idx:03d}.gif")
                        _create_gif(img_dir, gif_path)
                    log.info("  Saved viz GIFs for %d maps", min(viz_maps, env.num_envs))

            for fn in step_fns:
                del fn

    return gt, map_sim, map_sim_valid, agent_state, road_edges


# ---------------------------------------------------------------------------
# Ground-truth evaluation (sim = GT, sanity check)
# ---------------------------------------------------------------------------

def _evaluate_ground_truth(env, num_rollouts, sim_steps):
    """Use ground-truth trajectories as simulated — meta-score should be ~1.0."""
    obs, _ = env.reset()
    gt = env.get_ground_truth_trajectories()
    agent_state = env.get_global_agent_state()
    road_edges = env.get_road_edge_polylines()

    # Repeat GT across rollouts: (N, 1, T) → (N, num_rollouts, T)
    sim = {}
    for key in ("x", "y", "z", "heading"):
        sim[key] = np.repeat(gt[key], num_rollouts, axis=1)
    # id: (N, 1) → broadcast to (N, num_rollouts, sim_steps)
    id_expanded = np.broadcast_to(gt["id"][:, :, np.newaxis],
                                  (gt["id"].shape[0], num_rollouts, sim_steps)).copy()
    sim["id"] = id_expanded

    total_agents = gt["x"].shape[0]
    sim_valid = np.ones((total_agents, num_rollouts, sim_steps), dtype=bool)

    return gt, sim, sim_valid, agent_state, road_edges


# ---------------------------------------------------------------------------
# Open-loop SMART evaluation (single-shot prediction, no simulator stepping)
# ---------------------------------------------------------------------------

def _evaluate_open_loop(env, planner_type, type_cfg, action_config,
                        num_rollouts, sim_steps, episode_length,
                        smart_model, smart_codebooks):
    """Generate trajectories via single-shot SMART prediction — no env.step().

    Uses dataset.py (WaymoBinaryDataset) for data preparation to match the
    training pipeline exactly (SDC centering, vehicles only, full GT).
    Predictions are mapped back to env agents by matching initial positions.
    """
    if planner_type != "smart":
        raise ValueError(f"open_loop eval_mode only supports planner.type=smart, got '{planner_type}'")

    obs, _ = env.reset()
    total_agents = env.num_agents
    gt = env.get_ground_truth_trajectories()
    agent_state = env.get_global_agent_state()
    road_edges = env.get_road_edge_polylines()

    sim = {
        "x": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
        "y": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
        "z": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
        "heading": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.float32),
        "id": np.zeros((total_agents, num_rollouts, sim_steps), dtype=np.int32),
    }
    sim_valid = np.ones((total_agents, num_rollouts, sim_steps), dtype=bool)

    num_hist = 11
    pred_steps = min(80, sim_steps - num_hist)
    model_device = next(smart_model.parameters()).device
    greedy = bool(type_cfg.get("greedy", False))
    temperature = float(type_cfg.get("temperature", 1.0))

    # Load scenarios via dataset.py (same pipeline as training)
    from pufferlib.prediction.dataset import WaymoBinaryDataset, read_binary_scenario
    from scipy.spatial import cKDTree

    dataset = WaymoBinaryDataset(
        data_dir=env.data_root, split=env.split,
        num_historical_steps=11, num_future_steps=80, shift=5,
        max_agents=-1, max_files=env.num_envs,
    )
    log.info("Loaded %d scenarios via dataset.py pipeline", len(dataset))

    # Pre-build mapping: for each env map, match env agents → dataset agents
    # by initial position at t=num_hist-1 (=10)
    agent_mappings = []   # (ds_indices, matched_mask, center_pos) per map
    data_list = []

    for map_i in range(env.num_envs):
        cur = env.agent_offsets[map_i]
        nxt = env.agent_offsets[map_i + 1]
        n_env = nxt - cur

        # Env positions at t=10
        env_pos = np.stack([
            gt['x'][cur:nxt, 0, num_hist - 1],
            gt['y'][cur:nxt, 0, num_hist - 1],
        ], axis=-1)  # (n_env, 2)

        # Dataset HeteroData for this scenario
        ds_data = dataset[map_i]
        data_list.append(ds_data)

        # Recover world-coordinate positions: read center_pos from binary
        map_id = env.map_ids[map_i] if hasattr(env, "map_ids") else map_i
        bin_path = os.path.join(env.data_root, env.split, f"map_{map_id:06d}.bin")
        scenario = read_binary_scenario(bin_path)
        objects = scenario['objects']
        sdc_track = scenario['sdc_track_index']
        veh_orig_indices = [i for i, o in enumerate(objects) if o['type'] == 1]
        if sdc_track in veh_orig_indices:
            sdc_veh_idx = veh_orig_indices.index(sdc_track)
        else:
            sdc_veh_idx = 0
        sdc_obj = objects[veh_orig_indices[sdc_veh_idx]]
        center_pos = np.array([
            sdc_obj['traj_x'][num_hist - 1],
            sdc_obj['traj_y'][num_hist - 1],
        ], dtype=np.float32)

        # Un-center dataset positions to get world coords
        ds_pos_centered = ds_data['agent']['position'][:, num_hist - 1, :].numpy()
        ds_pos_world = ds_pos_centered + center_pos

        # Match env agents to dataset agents by position
        if n_env > 0 and len(ds_pos_world) > 0:
            tree = cKDTree(ds_pos_world.astype(np.float64))
            dists, indices = tree.query(env_pos.astype(np.float64))
            matched = dists < 0.1
        else:
            matched = np.zeros(n_env, dtype=bool)
            indices = np.zeros(n_env, dtype=int)

        agent_mappings.append((indices, matched, center_pos))

    n_matched = sum(m.sum() for _, m, _ in agent_mappings)
    log.info("Matched %d/%d env agents to dataset vehicles", n_matched, total_agents)

    import time as _time
    from torch_geometric.data import Batch

    for rollout_idx in range(num_rollouts):
        _t_rollout = _time.perf_counter()

        # Fill observed steps from GT
        for key in ("x", "y", "z", "heading"):
            sim[key][:, rollout_idx, :num_hist] = gt[key][:, 0, :num_hist]
        sim["id"][:, rollout_idx, :] = gt["id"][:, 0, np.newaxis]

        # Chunked batch inference using dataset.py data
        CHUNK_SIZE = 50
        all_pred_traj = [None] * len(data_list)
        all_pred_head = [None] * len(data_list)

        for chunk_start in range(0, len(data_list), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(data_list))
            chunk = [data_list[i] for i in range(chunk_start, chunk_end)]

            if len(chunk) == 1:
                batch_data = chunk[0].to(model_device)
            else:
                batch_data = Batch.from_data_list(chunk).to(model_device)

            with torch.no_grad():
                result = smart_model.inference(
                    batch_data,
                    greedy=(greedy or rollout_idx == 0),
                    temperature=temperature,
                )
            pred_traj = result["pred_traj"].cpu().numpy()
            pred_head = result["pred_head"].cpu().numpy()

            del batch_data, result
            torch.cuda.empty_cache()

            # Split predictions by scenario
            if len(chunk) == 1:
                all_pred_traj[chunk_start] = pred_traj
                all_pred_head[chunk_start] = pred_head
            else:
                agent_counts = [data_list[i]["agent"]["num_nodes"]
                                for i in range(chunk_start, chunk_end)]
                offset = 0
                for j, count in enumerate(agent_counts):
                    all_pred_traj[chunk_start + j] = pred_traj[offset:offset + count]
                    all_pred_head[chunk_start + j] = pred_head[offset:offset + count]
                    offset += count

        # Map predictions back to env agents
        for map_i in range(env.num_envs):
            cur = env.agent_offsets[map_i]
            nxt = env.agent_offsets[map_i + 1]
            n_env = nxt - cur
            indices, matched, center_pos = agent_mappings[map_i]

            ds_pred = all_pred_traj[map_i]  # (n_ds, 80, 2) centered
            ds_head = all_pred_head[map_i]  # (n_ds, 80)

            # Un-center predictions to world coords
            ds_pred_world = ds_pred + center_pos[np.newaxis, np.newaxis, :]

            for j in range(n_env):
                if matched[j]:
                    ds_j = indices[j]
                    ps = min(pred_steps, ds_pred_world.shape[1])
                    sim["x"][cur + j, rollout_idx, num_hist:num_hist + ps] = \
                        ds_pred_world[ds_j, :ps, 0]
                    sim["y"][cur + j, rollout_idx, num_hist:num_hist + ps] = \
                        ds_pred_world[ds_j, :ps, 1]
                    sim["heading"][cur + j, rollout_idx, num_hist:num_hist + ps] = \
                        ds_head[ds_j, :ps]
                    # Fill remaining steps
                    filled = num_hist + ps
                    if filled < sim_steps:
                        sim["x"][cur + j, rollout_idx, filled:] = \
                            ds_pred_world[ds_j, ps - 1, 0]
                        sim["y"][cur + j, rollout_idx, filled:] = \
                            ds_pred_world[ds_j, ps - 1, 1]
                        sim["heading"][cur + j, rollout_idx, filled:] = \
                            ds_head[ds_j, ps - 1]
                else:
                    # Unmatched agent: hold last known position
                    sim["x"][cur + j, rollout_idx, num_hist:] = \
                        gt["x"][cur + j, 0, num_hist - 1]
                    sim["y"][cur + j, rollout_idx, num_hist:] = \
                        gt["y"][cur + j, 0, num_hist - 1]
                    sim["heading"][cur + j, rollout_idx, num_hist:] = \
                        gt["heading"][cur + j, 0, num_hist - 1]

        elapsed = _time.perf_counter() - _t_rollout
        log.info("  Rollout %d/%d  (%.1fs)", rollout_idx + 1, num_rollouts, elapsed)
        torch.cuda.empty_cache()

    return gt, sim, sim_valid, agent_state, road_edges


def _compute_and_collect(evaluator, gt, sim, sim_valid, agent_state, road_edges, num_rollouts):
    """Apply sim_valid masking to GT, compute WOSAC metrics, return DataFrame."""
    sim_valid_all = np.all(sim_valid, axis=1, keepdims=True)
    n_excluded = int(np.sum(gt["valid"].astype(bool) & ~sim_valid_all))
    gt["valid"] = gt["valid"] * sim_valid_all.astype(gt["valid"].dtype)
    if n_excluded > 0:
        log.info("Excluded %d agent-steps where sim agent was removed but GT was valid", n_excluded)

    sim["id"][:] = gt["id"][..., np.newaxis]

    results = evaluator.compute_metrics(
        gt, sim,
        {"length": agent_state["length"], "width": agent_state["width"]},
        road_edges,
        aggregate_results=False,
    )
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _visualize_scenarios(gt, sim, agent_state, road_edges, results_df,
                         output_dir, num_maps=5, num_rollouts_to_show=5):
    """Plot GT vs. simulated trajectories with road edges per scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Patch

    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    scenario_ids = gt["scenario_id"][:, 0]  # (num_agents,)
    unique_scenarios = np.unique(scenario_ids)
    unique_scenarios = unique_scenarios[unique_scenarios >= 0]

    if len(unique_scenarios) > num_maps:
        unique_scenarios = unique_scenarios[:num_maps]

    num_rollouts = sim["x"].shape[1]
    n_show = min(num_rollouts_to_show, num_rollouts)

    for scenario_id in unique_scenarios:
        agent_mask = scenario_ids == scenario_id
        n_agents = int(agent_mask.sum())

        # Get ADE from results if available
        ade_str = ""
        if results_df is not None and scenario_id in results_df.index:
            ade_val = results_df.loc[scenario_id, "ade"]
            ade_str = f" | ADE={ade_val:.1f}"

        fig, ax = plt.subplots(figsize=(14, 14))

        # 1) Road edges
        re_lengths = road_edges["lengths"]
        re_sids = road_edges["scenario_id"]
        re_x = road_edges["x"]
        re_y = road_edges["y"]
        pt_idx = 0
        for i in range(len(re_lengths)):
            length = re_lengths[i]
            if re_sids[i] == scenario_id:
                ax.plot(re_x[pt_idx:pt_idx + length], re_y[pt_idx:pt_idx + length],
                        "k-", linewidth=1, alpha=0.5)
            pt_idx += length

        # 2) GT trajectories (green) and 3) Sim trajectories (blue)
        agent_indices = np.where(agent_mask)[0]
        gt_valid = gt["valid"][agent_mask, 0, :]  # (n_agents, sim_steps)

        for local_i, global_i in enumerate(agent_indices):
            valid = gt_valid[local_i].astype(bool)
            if not valid.any():
                continue

            # GT trajectory
            gx = gt["x"][global_i, 0, valid]
            gy = gt["y"][global_i, 0, valid]
            label_gt = "GT" if local_i == 0 else None
            ax.plot(gx, gy, "-", color="green", linewidth=2, alpha=0.7,
                    label=label_gt, zorder=10)
            ax.plot(gx[0], gy[0], "*", color="darkgreen", markersize=10, zorder=15)

            # Sim trajectories (multiple rollouts, only valid timesteps)
            for r in range(n_show):
                sx = sim["x"][global_i, r, valid]
                sy = sim["y"][global_i, r, valid]
                label_sim = "Prediction" if local_i == 0 and r == 0 else None
                ax.plot(sx, sy, "-", color="blue", linewidth=1, alpha=0.2,
                        label=label_sim, zorder=5)

            # Bounding box at start (GT)
            t0 = np.where(valid)[0][0]
            try:
                from pufferlib.ocean.benchmark.geometry_utils import get_2d_box_corners
                box = torch.as_tensor(
                    [[gt["x"][global_i, 0, t0], gt["y"][global_i, 0, t0],
                      agent_state["length"][global_i], agent_state["width"][global_i],
                      gt["heading"][global_i, 0, t0]]],
                    dtype=torch.float32,
                )
                corners = get_2d_box_corners(box)[0].cpu().numpy()
                poly = MplPolygon(corners, facecolor="green", edgecolor="darkgreen",
                                  linewidth=1, alpha=0.3, zorder=12)
                ax.add_patch(poly)
            except Exception:
                pass

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Scenario {scenario_id} | {n_agents} agents{ade_str}")

        legend_elements = [
            Patch(facecolor="green", alpha=0.5, label="GT"),
            Patch(facecolor="blue", alpha=0.3, label=f"Predictions ({n_show} rollouts)"),
            Patch(facecolor="none", edgecolor="black", label="Road Edges"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, f"scenario_{scenario_id}.png"), dpi=150)
        plt.close(fig)

    log.info("Saved %d scenario visualizations to %s", len(unique_scenarios), viz_dir)


def _visualize_scenarios_gif(gt, sim, agent_state, road_edges, results_df,
                              output_dir, num_maps=5, rollout_idx=0, fps=10):
    """Create per-scenario animated GIFs showing agents moving along GT and sim trajectories.

    Each frame shows: road edges, GT positions (green), sim positions (blue),
    GT trajectory history (green trail), sim trajectory history (blue trail),
    and bounding boxes at current positions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon, Patch, FancyArrowPatch
    from PIL import Image
    import tempfile

    gif_dir = os.path.join(output_dir, "gif")
    os.makedirs(gif_dir, exist_ok=True)

    scenario_ids = gt["scenario_id"][:, 0]
    unique_scenarios = np.unique(scenario_ids)
    unique_scenarios = unique_scenarios[unique_scenarios >= 0]

    if len(unique_scenarios) > num_maps:
        unique_scenarios = unique_scenarios[:num_maps]

    sim_steps = gt["x"].shape[2] if gt["x"].ndim == 3 else gt["x"].shape[1]
    r = min(rollout_idx, sim["x"].shape[1] - 1)

    for scenario_id in unique_scenarios:
        agent_mask = scenario_ids == scenario_id
        agent_indices = np.where(agent_mask)[0]
        n_agents = len(agent_indices)

        # Get ADE
        ade_str = ""
        if results_df is not None and scenario_id in results_df.index:
            ade_val = results_df.loc[scenario_id, "ade"]
            ade_str = f" | ADE={ade_val:.1f}"

        # Precompute road edges for this scenario
        re_segments = []
        pt_idx = 0
        for i in range(len(road_edges["lengths"])):
            length = road_edges["lengths"][i]
            if road_edges["scenario_id"][i] == scenario_id:
                re_segments.append((
                    road_edges["x"][pt_idx:pt_idx + length],
                    road_edges["y"][pt_idx:pt_idx + length],
                ))
            pt_idx += length

        # Compute axis limits from GT + road edges
        all_x, all_y = [], []
        gt_valid = gt["valid"][agent_mask, 0, :].astype(bool)
        for local_i, global_i in enumerate(agent_indices):
            v = gt_valid[local_i]
            if v.any():
                all_x.append(gt["x"][global_i, 0, v])
                all_y.append(gt["y"][global_i, 0, v])
        for rx, ry in re_segments:
            all_x.append(rx)
            all_y.append(ry)
        if not all_x:
            continue
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        margin = 20
        xlim = (all_x.min() - margin, all_x.max() + margin)
        ylim = (all_y.min() - margin, all_y.max() + margin)

        # Render each frame
        frames = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for t in range(sim_steps):
                fig, ax = plt.subplots(figsize=(10, 10))

                # Road edges
                for rx, ry in re_segments:
                    ax.plot(rx, ry, "k-", linewidth=0.8, alpha=0.4)

                for local_i, global_i in enumerate(agent_indices):
                    v = gt_valid[local_i]
                    if not v.any():
                        continue

                    # GT trail (history up to t)
                    t_end = min(t + 1, sim_steps)
                    v_hist = v[:t_end]
                    if v_hist.any():
                        gx = gt["x"][global_i, 0, :t_end][v_hist]
                        gy = gt["y"][global_i, 0, :t_end][v_hist]
                        ax.plot(gx, gy, "-", color="green", linewidth=1.5, alpha=0.5)

                    # Sim trail (history up to t)
                    sx_hist = sim["x"][global_i, r, :t_end]
                    sy_hist = sim["y"][global_i, r, :t_end]
                    valid_sim = sx_hist > -9000
                    if valid_sim.any():
                        ax.plot(sx_hist[valid_sim], sy_hist[valid_sim],
                                "-", color="blue", linewidth=1.5, alpha=0.5)

                    # Current positions with bounding boxes
                    if t < sim_steps and v[t]:
                        gx_t = float(gt["x"][global_i, 0, t])
                        gy_t = float(gt["y"][global_i, 0, t])
                        gh_t = float(gt["heading"][global_i, 0, t])
                        w = float(agent_state["width"][global_i])
                        l = float(agent_state["length"][global_i])

                        # GT bounding box
                        corners_gt = _bbox_corners(gx_t, gy_t, gh_t, l, w)
                        poly_gt = MplPolygon(corners_gt, facecolor="green",
                                             edgecolor="darkgreen", linewidth=1,
                                             alpha=0.4, zorder=12)
                        ax.add_patch(poly_gt)

                        # Sim bounding box
                        sx_t = float(sim["x"][global_i, r, t])
                        sy_t = float(sim["y"][global_i, r, t])
                        sh_t = float(sim["heading"][global_i, r, t])
                        if sx_t > -9000:
                            corners_sim = _bbox_corners(sx_t, sy_t, sh_t, l, w)
                            poly_sim = MplPolygon(corners_sim, facecolor="blue",
                                                  edgecolor="darkblue", linewidth=1,
                                                  alpha=0.4, zorder=11)
                            ax.add_patch(poly_sim)

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect("equal")
                ax.set_title(f"Scenario {scenario_id} | Step {t}/{sim_steps-1} | "
                             f"{n_agents} agents{ade_str}")

                legend_elements = [
                    Patch(facecolor="green", alpha=0.4, label="GT"),
                    Patch(facecolor="blue", alpha=0.4, label="Sim"),
                    Patch(facecolor="none", edgecolor="black", label="Road Edges"),
                ]
                ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

                frame_path = os.path.join(tmpdir, f"frame_{t:03d}.png")
                fig.savefig(frame_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                frames.append(Image.open(frame_path).copy())

            # Save GIF
            if frames:
                gif_path = os.path.join(gif_dir, f"scenario_{scenario_id}.gif")
                duration = int(1000 / fps)
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:],
                    duration=duration, loop=0,
                )

        log.info("  Saved GIF for scenario %d (%d agents, %d frames)",
                 scenario_id, n_agents, sim_steps)

    log.info("Saved %d scenario GIFs to %s", len(unique_scenarios), gif_dir)


def _bbox_corners(x, y, heading, length, width):
    """Compute 4 corners of a bounding box."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = np.array([
        [x + hl * cos_h - hw * sin_h, y + hl * sin_h + hw * cos_h],
        [x + hl * cos_h + hw * sin_h, y + hl * sin_h - hw * cos_h],
        [x - hl * cos_h + hw * sin_h, y - hl * sin_h - hw * cos_h],
        [x - hl * cos_h - hw * sin_h, y - hl * sin_h + hw * cos_h],
    ])
    return corners


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Deterministic
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        print("ERROR: DRIVE_BINARIES_DATA_ROOT not set")
        sys.exit(1)

    config = load_realism_config()
    realism_cfg = config.get("realism", {})
    planner_cfg = config.get("planner", {})
    planner_type = planner_cfg.get("type", "idm")
    type_cfg = planner_cfg.get(planner_type, {})

    split = str(realism_cfg.get("split", "training"))
    num_rollouts = int(realism_cfg.get("num_rollouts", 32))
    init_steps = int(realism_cfg.get("init_steps", 10))
    episode_length = int(realism_cfg.get("episode_length", 91))
    device = str(realism_cfg.get("device", "cuda"))
    sim_steps = episode_length - init_steps  # 81
    viz = str(realism_cfg.get("viz", "False")).lower() in ("true", "1", "yes")
    viz_maps = int(realism_cfg.get("viz_maps", 5))
    eval_mode = str(realism_cfg.get("eval_mode", "closed_loop"))

    # For open_loop prediction: use init_steps=0 so GT covers full 91 steps (0-90).
    # The model expects steps 0-10 as history, not steps 10-20.
    if eval_mode == "open_loop":
        init_steps = 0
        sim_steps = episode_length - init_steps  # 91

    map_ids = _parse_map_ids(config.get("map_ids"), split)
    num_target_maps = len(map_ids)

    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_wosac_{planner_type}"
    output_dir = config.get("output_dir")
    if output_dir is None:
        exp_root = os.environ.get("PUFFER_EXP_ROOT", "experiments")
        output_dir = os.path.join(exp_root, run_id)
    else:
        output_dir = os.path.join(output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "realism.log"), mode="w"),
        ],
    )

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    log.info("=" * 60)
    log.info("WOSAC REALISM EVALUATION")
    log.info("=" * 60)
    log.info("Output:      %s", output_dir)
    log.info("Eval mode:   %s", eval_mode)
    log.info("Planner:     %s", planner_type)
    for k, v in type_cfg.items():
        if v != "" and v is not None:
            log.info("  %s: %s", k, v)
    log.info("Split:       %s", split)
    log.info("Maps:        %d  (%s)", num_target_maps, config.get("map_ids", "all"))
    log.info("Rollouts:    %d", num_rollouts)
    log.info("Init steps:  %d  |  Sim steps: %d", init_steps, sim_steps)
    if viz:
        log.info("Viz:         ON  (first %d maps)", viz_maps)
    log.info("=" * 60)

    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.evaluation.config import ActionConfig
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    action_config = ActionConfig()

    # Build WOSACEvaluator config
    wosac_config = {
        "eval": {
            "wosac_init_steps": init_steps,
            "wosac_num_rollouts": num_rollouts,
        },
        "train": {"device": device},
    }
    evaluator = WOSACEvaluator(wosac_config)

    # Pre-load SMART model once if needed
    smart_model = None
    smart_codebooks = None
    if planner_type == "smart":
        from pufferlib.planning.smart import load_smart_model
        smart_model, motion_cb, map_cb = load_smart_model(
            str(type_cfg.get("weights_path", "")),
            str(type_cfg.get("device", "cuda")),
        )
        smart_codebooks = (motion_cb, map_cb)

    total_start = time.perf_counter()

    # --- Load all maps deterministically with use_all_maps=True ---
    # Read env settings from [env] section of realism.ini
    realism_ini = os.path.join(_CONFIG_DIR, "realism.ini")
    env_cfg = config.get("env", {})
    action_type = str(env_cfg.get("action_type", "continuous"))
    control_mode = "control_wosac"
    env_extra_kwargs = {}
    if planner_type in ("conditioned", "conditioned_mix",
                        "conditioned_aggr", "conditioned_normal",
                        "conditioned_caut"):
        # DriveConditioned needs the 9-float creward block appended to the obs.
        env_extra_kwargs["reward_conditioning"] = True
        # Prefer creward_profiles list (round-robin across agents in C); fall
        # back to the legacy explicit creward.* fields for older [planner.conditioned].
        profiles = type_cfg.get("creward_profiles", None)
        if profiles:
            env_extra_kwargs["creward_deterministic"] = True
            env_extra_kwargs["creward_traffic"] = list(profiles)
            log.info("Loaded %d creward profile(s) for %s", len(profiles), planner_type)
        else:
            creward_ego = {k: float(v) for k, v in (type_cfg.get("creward", {}) or {}).items()
                           if v != "" and v is not None}
            if creward_ego:
                env_extra_kwargs["creward_deterministic"] = True
                env_extra_kwargs["creward_traffic"] = [tuple([
                    creward_ego.get("delta_goal", 0.0),
                    creward_ego.get("alpha_collision", 0.0),
                    creward_ego.get("alpha_boundary", 0.0),
                    creward_ego.get("alpha_comfort", 0.0),
                    creward_ego.get("alpha_l_align", 0.0),
                    creward_ego.get("alpha_vel_align", 0.0),
                    creward_ego.get("alpha_l_center", 0.0),
                    creward_ego.get("alpha_center_bias", 0.0),
                    creward_ego.get("alpha_reverse", 0.0),
                ])]
                log.info("Pinned creward: %s", creward_ego)

    env = Drive(
        num_maps=num_target_maps,
        num_agents=num_target_maps * 50,  # enough to cover all maps
        split=split,
        control_mode=control_mode,
        init_mode="create_all_valid",
        init_steps=init_steps,
        episode_length=episode_length,
        action_type=action_type,
        goal_behavior=int(env_cfg.get("goal_behavior", 3)),
        collision_behavior=int(env_cfg.get("collision_behavior", 0)),
        offroad_behavior=int(env_cfg.get("offroad_behavior", 0)),
        max_controlled_agents=-1,
        use_all_maps=True,
        ini_file=realism_ini,
        **env_extra_kwargs,
    )

    log.info("Loaded %d maps, %d agents (use_all_maps=True)", env.num_envs, env.num_agents)

    # --- Dispatch by eval_mode ---
    if eval_mode == "closed_loop":
        gt, sim, sim_valid, agent_state, road_edges = _evaluate_batch(
            env, planner_type, type_cfg, action_config,
            num_rollouts, sim_steps, episode_length,
            smart_model=smart_model, smart_codebooks=smart_codebooks,
            viz=viz, viz_maps=viz_maps, output_dir=output_dir,
        )
    elif eval_mode == "open_loop":
        gt, sim, sim_valid, agent_state, road_edges = _evaluate_open_loop(
            env, planner_type, type_cfg, action_config,
            num_rollouts, sim_steps, episode_length,
            smart_model, smart_codebooks,
        )
    elif eval_mode == "ground_truth":
        gt, sim, sim_valid, agent_state, road_edges = _evaluate_ground_truth(
            env, num_rollouts, sim_steps,
        )
    else:
        log.error("Unknown eval_mode: %s (expected: closed_loop, open_loop, ground_truth)", eval_mode)
        sys.exit(1)

    results = _compute_and_collect(
        evaluator, gt, sim, sim_valid, agent_state, road_edges, num_rollouts,
    )

    env.close()
    torch.cuda.empty_cache()
    gc.collect()

    if results is None or len(results) == 0:
        log.error("No maps processed!")
        sys.exit(1)

    log.info("-" * 60)
    log.info("Total scenarios: %d", len(results))

    # Print results
    log.info("=" * 60)
    log.info("WOSAC REALISM RESULTS — %s", planner_type.upper())
    log.info("=" * 60)

    meta_score = results["realism_meta_score"].mean()
    log.info("Realism meta-score:  %.4f", meta_score)
    # Exclude agents with no valid steps (ADE=0 from division-by-zero guard)
    valid_ade = results[results["ade"] > 0]
    log.info("ADE:                 %.4f  (%d/%d agents with valid steps)",
             valid_ade["ade"].mean() if len(valid_ade) > 0 else 0.0,
             len(valid_ade), len(results))
    log.info("minADE:              %.4f", valid_ade["min_ade"].mean() if len(valid_ade) > 0 else 0.0)
    log.info("-" * 60)

    # Per-metric breakdown
    metric_names = [
        "linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration",
        "distance_to_nearest_object", "time_to_collision", "collision_indication",
        "distance_to_road_edge", "offroad_indication",
    ]
    for name in metric_names:
        col = f"likelihood_{name}"
        if col in results.columns:
            log.info("  %-35s %.4f", name, results[col].mean())

    # WOSAC bucketed metrics (from Waymo aggregate_metrics_to_buckets)
    _BUCKETS = {
        "Kinematic": ["linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration"],
        "Interactive": ["distance_to_nearest_object", "collision_indication", "time_to_collision"],
        "Map-based": ["distance_to_road_edge", "offroad_indication"],
    }
    _WEIGHTS = {
        "linear_speed": 0.05, "linear_acceleration": 0.05,
        "angular_speed": 0.05, "angular_acceleration": 0.05,
        "distance_to_nearest_object": 0.10, "collision_indication": 0.25, "time_to_collision": 0.10,
        "distance_to_road_edge": 0.10, "offroad_indication": 0.25,
    }
    log.info("-" * 60)
    for bucket_name, bucket_metrics in _BUCKETS.items():
        w_sum = sum(_WEIGHTS[m] for m in bucket_metrics)
        score_waymo = sum(_WEIGHTS[m] * results[f"likelihood_{m}"].mean() for m in bucket_metrics) / w_sum
        score_puffer = np.mean([results[f"likelihood_{m}"].mean() for m in bucket_metrics])
        log.info("  %-25s waymo=%.4f  puffer=%.4f", bucket_name, score_waymo, score_puffer)

    log.info("=" * 60)

    total_time = time.perf_counter() - total_start
    log.info("Total time: %.1fs", total_time)

    # Save results
    results_path = os.path.join(output_dir, "wosac_results.csv")
    results.to_csv(results_path)
    log.info("Per-scenario results: %s", results_path)

    # Save aggregate
    agg = {
        "planner_type": planner_type,
        "realism_meta_score": float(meta_score),
        "ade": float(valid_ade["ade"].mean()) if len(valid_ade) > 0 else 0.0,
        "min_ade": float(valid_ade["min_ade"].mean()) if len(valid_ade) > 0 else 0.0,
        "num_maps": len(results),
        "total_agents": int(results["num_agents"].sum()) if "num_agents" in results.columns else len(results),
        "num_rollouts": num_rollouts,
    }
    for name in metric_names:
        col = f"likelihood_{name}"
        if col in results.columns:
            agg[col] = float(results[col].mean())
    for bucket_name, bucket_metrics in _BUCKETS.items():
        key = bucket_name.lower().replace('-', '_')
        w_sum = sum(_WEIGHTS[m] for m in bucket_metrics)
        agg[f"bucket_{key}_waymo"] = float(
            sum(_WEIGHTS[m] * results[f"likelihood_{m}"].mean() for m in bucket_metrics) / w_sum
        )
        agg[f"bucket_{key}_puffer"] = float(
            np.mean([results[f"likelihood_{m}"].mean() for m in bucket_metrics])
        )

    agg_path = os.path.join(output_dir, "wosac_summary.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    log.info("Summary: %s", agg_path)

    # Visualize scenarios
    if viz:
        _visualize_scenarios(gt, sim, agent_state, road_edges, results,
                             output_dir, num_maps=viz_maps)
        _visualize_scenarios_gif(gt, sim, agent_state, road_edges, results,
                                  output_dir, num_maps=viz_maps)

    return agg


if __name__ == "__main__":
    main()
