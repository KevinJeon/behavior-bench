# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Planner registry and config-driven planner creation.

Provides:
- load_eval_config(): Load evaluation.ini + CLI overrides (dot-notation)
- create_ego_planner(): Create ego planner from config dict
- create_traffic_controller(): Create traffic controller from config dict
"""

import argparse
import ast
import configparser
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional

import numpy as np

from pufferlib.planning.base import BasePlanner
from pufferlib.planning.adapters import ClassicObsView
from pufferlib.evaluation.config import ActionConfig

# DriveConditioned policy variants that share the same PPOPlanner backend.
# Each variant points to a different INI section ([planner|traffic].<name>) with
# its own weights_path / creward_profiles, but they all instantiate identically.
_CONDITIONED_VARIANTS = {
    "conditioned", "conditioned_mix",
    "conditioned_aggr", "conditioned_normal", "conditioned_caut",
}

# Classic-trained neural planners that need the jerk-obs-view adapter when
# the env emits the 10-dim jerk ego layout (emit_jerk_ego_obs=True).
_CLASSIC_NEURAL_PLANNERS = {"ppo", "smart", "world_model", "hybrid"} | _CONDITIONED_VARIANTS


def _maybe_wrap_for_jerk_obs(planner: BasePlanner, planner_type: str, env) -> BasePlanner:
    """Wrap a classic-trained neural planner in ClassicObsView when the env
    is emitting jerk-layout obs (mixed-env case with ConditionedPaper)."""
    if planner_type not in _CLASSIC_NEURAL_PLANNERS:
        return planner
    if not getattr(env, "emit_jerk_ego_obs", False):
        return planner
    return ClassicObsView(planner, env)


# ---------------------------------------------------------------------------
# Config loading (follows pufferl.py pattern)
# ---------------------------------------------------------------------------

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "config")
_DEFAULT_CONFIG = os.path.join(_CONFIG_DIR, "evaluation.ini")


def _puffer_type(value):
    """Convert string to Python literal (int, float, bool, tuple, etc.)."""
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _build_nested_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild nested dict from dot-notation keys.

    'planner.cem.horizon' -> config['planner']['cem']['horizon']
    """
    result = defaultdict(dict)
    for key, value in flat.items():
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return dict(result)


def load_eval_config(argv=None) -> Dict[str, Any]:
    """Load evaluation config from INI file with CLI overrides.

    Args:
        argv: Optional list of CLI args (for testing). None = sys.argv.

    Returns:
        Nested dict: config["eval"], config["planner"], config["traffic"], etc.
    """
    p = configparser.ConfigParser()
    p.read(_DEFAULT_CONFIG)

    parser = argparse.ArgumentParser(
        description="Evaluation Framework for Drive Planners",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Register all INI keys as --section.key CLI args
    for section in p.sections():
        for key in p[section]:
            fmt = f"--{section}.{key}"
            default = _puffer_type(p[section][key])
            parser.add_argument(
                fmt.replace("_", "-"),
                default=default,
                type=_puffer_type,
            )

    # Extra CLI-only args (not in INI)
    parser.add_argument("--map-ids", type=str, default=None,
                        help="Map IDs: 'all', range '0-100', or comma-separated '0,5,10'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom evaluation.ini to layer on top of defaults")

    args = vars(parser.parse_args(argv))

    # If custom config provided, reload with it layered on defaults
    custom_config = args.pop("config", None)
    if custom_config and os.path.isfile(custom_config):
        p2 = configparser.ConfigParser()
        p2.read([_DEFAULT_CONFIG, custom_config])
        # Re-parse to pick up new defaults (CLI still wins)
        for section in p2.sections():
            for key in p2[section]:
                flat_key = f"{section}.{key}"
                if flat_key not in args or args[flat_key] == _puffer_type(p[section].get(key, "")):
                    args[flat_key] = _puffer_type(p2[section][key])

    # Pull out non-dotted keys before nesting
    map_ids_str = args.pop("map_ids", None)
    output_dir = args.pop("output_dir", None)

    config = _build_nested_dict(args)

    # Attach non-dotted keys at top level
    config["map_ids"] = map_ids_str
    config["output_dir"] = output_dir

    return config


# ---------------------------------------------------------------------------
# Planner registry
# ---------------------------------------------------------------------------

def _get_planner_class(planner_type: str):
    """Lazy import to avoid circular imports and heavy torch loading."""
    if planner_type == "pdm":
        from pufferlib.planning.pdm import PDMPlanner, PDMConfig
        return PDMPlanner, PDMConfig
    elif planner_type == "ppo":
        from pufferlib.planning.policy import PPOPlanner, PPOConfig
        return PPOPlanner, PPOConfig
    elif planner_type == "idm":
        from pufferlib.planning.idm import IDMPlanner
        return IDMPlanner, None
    elif planner_type == "constant_velocity":
        from pufferlib.planning.constant_velocity import ConstantVelocityPlanner
        return ConstantVelocityPlanner, None
    elif planner_type == "hybrid":
        from pufferlib.planning.hybrid import HybridPPOPDMPlanner, HybridConfig
        return HybridPPOPDMPlanner, HybridConfig
    elif planner_type == "smart":
        from pufferlib.planning.smart import SMARTPlanner, SMARTConfig
        return SMARTPlanner, SMARTConfig
    elif planner_type == "expert":
        from pufferlib.planning.expert import ExpertPlanner
        return ExpertPlanner, None
    elif planner_type == "world_model":
        from pufferlib.planning.policy import WorldModelPlanner, WorldModelConfig
        return WorldModelPlanner, WorldModelConfig
    elif planner_type == "conditioned_paper":
        from pufferlib.planning.conditioned_paper import (
            ConditionedPaperPlanner, ConditionedPaperConfig,
        )
        return ConditionedPaperPlanner, ConditionedPaperConfig
    elif planner_type == "conditioned_jerk":
        from pufferlib.planning.conditioned_jerk import (
            ConditionedJerkPlanner, ConditionedJerkConfig,
        )
        return ConditionedJerkPlanner, ConditionedJerkConfig
    elif planner_type in _CONDITIONED_VARIANTS:
        from pufferlib.planning.policy import PPOPlanner, PPOConfig
        return PPOPlanner, PPOConfig
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")


def _build_pdm_config(cfg: dict, episode_length: int = 91):
    from pufferlib.planning.pdm import PDMConfig
    vf = cfg.get("velocity_fractions", (0.2, 0.4, 0.6, 0.8, 1.0))
    lo = cfg.get("lateral_offsets", (-1.0, 0.0, 1.0))
    return PDMConfig(
        horizon=int(cfg.get("horizon", 40)),
        episode_length=episode_length,
        velocity_fractions=tuple(vf) if not isinstance(vf, tuple) else vf,
        lateral_offsets=tuple(lo) if not isinstance(lo, tuple) else lo,
        proposal_other_planner=str(cfg.get("proposal_other", "constant_velocity")),
        max_velocity=float(cfg.get("max_velocity", 0.0)),
        idm_min_gap=float(cfg.get("min_gap", 1.0)),
        idm_headway_time=float(cfg.get("headway_time", 1.5)),
        idm_accel_max=float(cfg.get("accel_max", 1.0)),
        idm_decel_max=float(cfg.get("decel_max", 2.0)),
    )


def _build_ppo_config(cfg: dict):
    from pufferlib.planning.policy import PPOConfig
    return PPOConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
    )


def _build_smart_config(cfg: dict):
    from pufferlib.planning.smart import SMARTConfig
    return SMARTConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
        temperature=float(cfg.get("temperature", 1.0)),
        greedy=bool(cfg.get("greedy", False)),
        repredict_interval=int(cfg.get("repredict_interval", 5)),
    )


def _build_conditioned_config(cfg: dict):
    from pufferlib.planning.policy import PPOConfig
    return PPOConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
        policy_class_name="DriveConditioned",
        reward_conditioning=True,
    )


def _build_conditioned_paper_config(cfg: dict):
    from pufferlib.planning.conditioned_paper import ConditionedPaperConfig
    return ConditionedPaperConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
        input_size=int(cfg.get("input_size", 128)),
        hidden_size=int(cfg.get("hidden_size", 1024)),
        stochastic=str(cfg.get("stochastic", "false")).lower() in ("true", "1", "yes"),
    )


def _build_conditioned_jerk_config(cfg: dict):
    from pufferlib.planning.conditioned_jerk import ConditionedJerkConfig
    return ConditionedJerkConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
        input_size=int(cfg.get("input_size", 64)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        stochastic=str(cfg.get("stochastic", "false")).lower() in ("true", "1", "yes"),
    )


def _build_world_model_config(cfg: dict):
    from pufferlib.planning.policy import WorldModelConfig
    return WorldModelConfig(
        weights_path=str(cfg.get("weights_path", "")),
        device=str(cfg.get("device", "cuda")),
    )


def _build_hybrid_config(cfg: dict, episode_length: int = 91):
    from pufferlib.planning.hybrid import HybridConfig
    from pufferlib.planning.policy import PPOConfig
    from pufferlib.planning.pdm import PDMConfig
    from pufferlib.planning.ppo_rollout import PPORolloutConfig

    ppo_config = PPOConfig(
        weights_path=str(cfg.get("ppo_weights_path", "")),
        device=str(cfg.get("device", "cuda")),
    )
    pdm_config = PDMConfig(horizon=40, episode_length=episode_length)
    ensemble_str = cfg.get("ensemble_weights", "")
    ensemble_weights = [p.strip() for p in str(ensemble_str).split(",") if p.strip()]

    ppo_mode = str(cfg.get("ppo_mode", "single_step"))
    ppo_rollout_config = None
    if ppo_mode == "rollout":
        ppo_rollout_config = PPORolloutConfig(
            strategy=str(cfg.get("ppo_rollout_strategy", "beam_search")),
            top_k=int(cfg.get("ppo_rollout_top_k", 8)),
            beam_width=int(cfg.get("ppo_rollout_beam_width", 4)),
            branch_factor=int(cfg.get("ppo_rollout_branch_factor", 4)),
            horizon=int(cfg.get("ppo_rollout_horizon", 10)),
            episode_length=episode_length,
            w_cmf=float(cfg.get("ppo_rollout_w_cmf", 1.0 / 3.0)),
            w_align=float(cfg.get("ppo_rollout_w_align", 1.0 / 3.0)),
            w_ctr=float(cfg.get("ppo_rollout_w_ctr", 1.0 / 3.0)),
            lane_dist_scale=float(cfg.get("ppo_rollout_lane_dist_scale", 2.0)),
        )

    return HybridConfig(
        ppo_config=ppo_config,
        pdm_config=pdm_config,
        ensemble_weights=ensemble_weights,
        epistemic_threshold=float(cfg.get("epistemic_threshold", 0.8)),
        pdm_min_steps=int(cfg.get("pdm_min_steps", 1)),
        force_ppo=str(cfg.get("force_ppo", "false")).lower() in ("true", "1", "yes"),
        force_pdm=str(cfg.get("force_pdm", "false")).lower() in ("true", "1", "yes"),
        switch_mode=str(cfg.get("switch_mode", "epistemic")),
        value_variance_threshold=float(cfg.get("value_variance_threshold", 0.1)),
        lookahead_steps=int(cfg.get("lookahead_steps", 0)),
        ppo_mode=ppo_mode,
        ppo_rollout_config=ppo_rollout_config,
    )


def create_ego_planner(
    config: Dict[str, Any],
    env,
    action_config: ActionConfig,
    traffic_controller: Optional[BasePlanner] = None,
    ego_agent_idx: int = 0,
) -> BasePlanner:
    """Create ego planner from config dict.

    Args:
        config: Full nested config (from load_eval_config)
        env: Drive environment
        action_config: Action space configuration
        traffic_controller: Traffic planner (passed to PDM for scoring)

    Returns:
        Initialized planner instance
    """
    planner_type = config["planner"]["type"]
    type_cfg = config["planner"].get(planner_type, {})
    episode_length = int(config["eval"].get("episode_length", 91))
    ac_lb, ac_ub = action_config.bounds

    cls, _ = _get_planner_class(planner_type)

    if planner_type == "pdm":
        pdm_cfg = _build_pdm_config(type_cfg, episode_length)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=pdm_cfg, other_planner=traffic_controller)
    elif planner_type == "ppo":
        ppo_cfg = _build_ppo_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=ppo_cfg)
    elif planner_type == "idm":
        horizon = int(type_cfg.get("horizon", 40))
        tv = float(type_cfg.get("target_velocity", 15.0))
        planner = cls(env=env, agent_indices=[ego_agent_idx], horizon=horizon,
                      action_lb=ac_lb, action_ub=ac_ub, target_velocity=tv,
                      min_gap=float(type_cfg.get("min_gap", 1.0)),
                      headway_time=float(type_cfg.get("headway_time", 1.5)),
                      accel_max=float(type_cfg.get("accel_max", 1.0)),
                      decel_max=float(type_cfg.get("decel_max", 2.0)))
    elif planner_type == "constant_velocity":
        planner = cls(env=env, agent_idx=ego_agent_idx, horizon=1,
                      action_lb=ac_lb, action_ub=ac_ub)
    elif planner_type == "world_model":
        wm_cfg = _build_world_model_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=wm_cfg)
    elif planner_type == "hybrid":
        hybrid_cfg = _build_hybrid_config(type_cfg, episode_length)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=hybrid_cfg, other_planner=traffic_controller)
    elif planner_type == "smart":
        smart_cfg = _build_smart_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=smart_cfg)
    elif planner_type == "conditioned_paper":
        cpp_cfg = _build_conditioned_paper_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cpp_cfg)
    elif planner_type == "conditioned_jerk":
        cj_cfg = _build_conditioned_jerk_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cj_cfg)
    elif planner_type in _CONDITIONED_VARIANTS:
        cond_cfg = _build_conditioned_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cond_cfg)
    else:
        raise ValueError(f"Unknown ego planner type: {planner_type}")

    return _maybe_wrap_for_jerk_obs(planner, planner_type, env)


def create_traffic_controller(
    config: Dict[str, Any],
    env,
    action_config: ActionConfig,
    ego_agent_idx: int = 0,
) -> BasePlanner:
    """Create traffic agent controller from config dict.

    Args:
        config: Full nested config (from load_eval_config)
        env: Drive environment
        action_config: Action space configuration

    Returns:
        Initialized planner instance for traffic agents
    """
    traffic_type = config["traffic"]["type"]
    type_cfg = config["traffic"].get(traffic_type, {})
    episode_length = int(config["eval"].get("episode_length", 91))
    ac_lb, ac_ub = action_config.bounds

    cls, _ = _get_planner_class(traffic_type)

    if traffic_type == "pdm":
        from pufferlib.planning.pdm import MultiAgentPDMPlanner
        pdm_cfg = _build_pdm_config(type_cfg, episode_length)
        other_indices = [i for i in range(env.num_agents) if i != ego_agent_idx]
        planner = MultiAgentPDMPlanner(
            env=env, agent_indices=other_indices,
            action_lb=ac_lb, action_ub=ac_ub, config=pdm_cfg,
            other_planner=None,
        )
    elif traffic_type == "ppo":
        ppo_cfg = _build_ppo_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=ppo_cfg)
    elif traffic_type == "idm":
        other_indices = [i for i in range(env.num_agents) if i != ego_agent_idx]
        horizon = int(type_cfg.get("horizon", 40))
        tv = float(type_cfg.get("target_velocity", 15.0))
        planner = cls(env=env, agent_indices=other_indices, horizon=horizon,
                      action_lb=ac_lb, action_ub=ac_ub, target_velocity=tv,
                      min_gap=float(type_cfg.get("min_gap", 1.0)),
                      headway_time=float(type_cfg.get("headway_time", 1.5)),
                      accel_max=float(type_cfg.get("accel_max", 1.0)),
                      decel_max=float(type_cfg.get("decel_max", 2.0)))
    elif traffic_type == "constant_velocity":
        planner = cls(env=env, agent_idx=ego_agent_idx, horizon=1,
                      action_lb=ac_lb, action_ub=ac_ub)
    elif traffic_type == "smart":
        smart_cfg = _build_smart_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=smart_cfg)
    elif traffic_type == "world_model":
        wm_cfg = _build_world_model_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=wm_cfg)
    elif traffic_type == "expert":
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub)
    elif traffic_type == "conditioned_paper":
        cpp_cfg = _build_conditioned_paper_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cpp_cfg)
    elif traffic_type == "conditioned_jerk":
        cj_cfg = _build_conditioned_jerk_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cj_cfg)
    elif traffic_type in _CONDITIONED_VARIANTS:
        cond_cfg = _build_conditioned_config(type_cfg)
        planner = cls(env=env, agent_idx=ego_agent_idx, action_lb=ac_lb, action_ub=ac_ub,
                      config=cond_cfg)
    else:
        raise ValueError(f"Unknown traffic controller type: {traffic_type}")

    return _maybe_wrap_for_jerk_obs(planner, traffic_type, env)
