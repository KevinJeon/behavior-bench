# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Evaluation Framework for Drive Planners.

Config-driven evaluation with dot-notation CLI overrides.
Defaults are in pufferlib/config/evaluation.ini.

Usage:
    # PDM ego vs IDM traffic (defaults)
    python pufferlib/ocean/benchmark/eval.py --map-ids 0-10

    # Override planner type and params
    python pufferlib/ocean/benchmark/eval.py --planner.type ppo --traffic.type smart \\
        --traffic.smart.weights-path path/to/weights.pt

    # SMART as traffic controller
    python pufferlib/ocean/benchmark/eval.py --traffic.type smart \\
        --traffic.smart.weights-path experiments/smart_epoch_030.pt

    # Enable visualization
    python pufferlib/ocean/benchmark/eval.py --eval.viz True --eval.planner-viz True --map-ids 5
"""

import logging
import os
import sys
import uuid
from datetime import datetime
from functools import partial

from pufferlib.planning.registry import (
    load_eval_config,
    create_ego_planner,
    create_traffic_controller,
)
from pufferlib.evaluation import Evaluator, EvaluatorConfig

log = logging.getLogger("eval")


def _parse_map_ids(map_ids_str, split):
    """Parse map IDs from string or discover from split directory."""
    data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
    split_dir = os.path.join(data_root, split)

    if map_ids_str is None:
        if os.path.isdir(split_dir):
            bin_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".bin"))
            log.info("Using all %d .bin files from %s", len(bin_files), split_dir)
            return list(range(len(bin_files)))
        else:
            log.error("Split directory not found: %s", split_dir)
            sys.exit(1)

    if map_ids_str.lower() == "all":
        if os.path.isdir(split_dir):
            bin_files = sorted(f for f in os.listdir(split_dir) if f.endswith(".bin"))
            return list(range(len(bin_files)))
        log.error("Split directory not found: %s", split_dir)
        sys.exit(1)

    if "-" in map_ids_str and "," not in map_ids_str:
        start, end = map_ids_str.split("-")
        return list(range(int(start.strip()), int(end.strip()) + 1))

    return [int(x.strip()) for x in map_ids_str.split(",")]


def main():
    # Deterministic evaluation
    import random
    import numpy as np
    import torch

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Verify environment
    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        print("ERROR: DRIVE_BINARIES_DATA_ROOT not set")
        sys.exit(1)

    # Load config (INI defaults + CLI overrides)
    config = load_eval_config()

    eval_cfg = config.get("eval", {})
    planner_cfg = config.get("planner", {})
    traffic_cfg = config.get("traffic", {})

    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output_dir = config.get("output_dir")
    if output_dir is None:
        exp_root = os.environ.get("PUFFER_EXP_ROOT", "experiments")
        output_dir = os.path.join(exp_root, run_id)
    else:
        output_dir = os.path.join(output_dir, run_id)

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "eval.log"), mode="w"),
        ],
    )

    # Parse map IDs
    split = str(eval_cfg.get("split", "pufferhard"))
    map_ids = _parse_map_ids(config.get("map_ids"), split)

    # Build EvaluatorConfig
    # Parse ensemble weight paths: check planner-level config first, then eval-level fallback
    ensemble_str = ""
    planner_type = planner_cfg.get("type", "pdm")
    planner_type_cfg = planner_cfg.get(planner_type, {})
    if planner_type_cfg.get("ensemble_weights", ""):
        ensemble_str = str(planner_type_cfg["ensemble_weights"])
    elif eval_cfg.get("ensemble_weight_paths", ""):
        ensemble_str = str(eval_cfg["ensemble_weight_paths"])
    ensemble_paths = [p.strip() for p in ensemble_str.split(",") if p.strip()]

    # Auto-enable uncertainty when ensemble weights are provided
    compute_uncertainty = str(eval_cfg.get("compute_uncertainty", "False")).lower() in ("true", "1", "yes")
    if ensemble_paths:
        compute_uncertainty = True

    # Reward-conditioning values for DriveConditionedPaper / DriveConditioned
    # come from the planner / traffic sub-configs directly:
    #   planner.<type>.creward.*  -> ego creward profile
    #   planner.<type>.reward.*   -> global env reward overrides
    #   traffic.<type>.creward_profiles -> list of traffic profiles
    # where <type> ∈ {conditioned_paper, conditioned}.
    creward_ego = {}
    creward_traffic = []
    env_reward_overrides = {}
    creward_deterministic = False
    from pufferlib.planning.registry import _CONDITIONED_VARIANTS
    _CREWARD_PLANNER_TYPES = ("conditioned_paper", "conditioned_jerk") + tuple(_CONDITIONED_VARIANTS)
    if planner_cfg.get("type") in _CREWARD_PLANNER_TYPES:
        p_cfg = planner_cfg.get(planner_cfg["type"], {}) or {}
        creward_ego = {k: float(v) for k, v in (p_cfg.get("creward", {}) or {}).items()
                       if v != "" and v is not None}
        for k, v in (p_cfg.get("reward", {}) or {}).items():
            if v != "" and v is not None:
                env_reward_overrides[f"reward_{k}"] = float(v)
        if creward_ego:
            creward_deterministic = True
    if traffic_cfg.get("type") in _CREWARD_PLANNER_TYPES:
        t_cfg = traffic_cfg.get(traffic_cfg["type"], {}) or {}
        profiles = t_cfg.get("creward_profiles", None)
        if profiles:
            creward_traffic = list(profiles)  # drive.py normalizes tuples/dicts
            creward_deterministic = True

    # Auto-detect: if ConditionedPaper is in the mix (ego or traffic), the env
    # must emit the 10-dim jerk ego obs layout so the paper policy sees its
    # native obs format. Classic-trained neural planners get auto-wrapped in
    # ClassicObsView (in registry.py) to strip the extra dims back out.
    # DriveConditioned uses the 7-dim classic ego layout, so it does not
    # require emit_jerk_ego_obs — just reward_conditioning.
    conditioned_paper_in_mix = (
        planner_cfg.get("type") in ("conditioned_paper", "conditioned_jerk")
        or traffic_cfg.get("type") in ("conditioned_paper", "conditioned_jerk")
    )
    any_conditioned_in_mix = (
        planner_cfg.get("type") in _CREWARD_PLANNER_TYPES
        or traffic_cfg.get("type") in _CREWARD_PLANNER_TYPES
    )
    eval_dynamics = str(eval_cfg.get("dynamics_model", "classic"))
    eval_reward_cond = str(eval_cfg.get("reward_conditioning", "0")).lower() in ("true", "1", "yes")
    emit_jerk_ego_obs = conditioned_paper_in_mix and eval_dynamics != "jerk"
    if any_conditioned_in_mix and not eval_reward_cond:
        eval_reward_cond = True
        log.info("Auto-enabling reward_conditioning=1 (conditioned policy in the mix)")
    if emit_jerk_ego_obs:
        log.info("Mixed eval detected (ConditionedPaper + classic-trained planners in %s env): "
                 "emit_jerk_ego_obs=1; classic-trained planners wrapped in ClassicObsView",
                 eval_dynamics)

    evaluator_config = EvaluatorConfig(
        episode_length=int(eval_cfg.get("episode_length", 91)),
        action_type=str(eval_cfg.get("action_type", "continuous")),
        output_dir=output_dir,
        split=split,
        viz=bool(eval_cfg.get("viz", False)),
        planner_viz=bool(eval_cfg.get("planner_viz", False)),
        goal_behavior=int(eval_cfg.get("goal_behavior", 3)),
        goal_lane_change_prob=float(eval_cfg.get("goal_lane_change_prob", 0.0)),
        goal_target_distance=float(eval_cfg.get("goal_target_distance", 20.0)),
        termination_mode=int(eval_cfg.get("termination_mode", 1)),
        collision_behavior=int(eval_cfg.get("collision_behavior", 2)),
        offroad_behavior=int(eval_cfg.get("offroad_behavior", 2)),
        compute_uncertainty=compute_uncertainty,
        ensemble_weight_paths=ensemble_paths,
        dynamics_model=eval_dynamics,
        reward_conditioning=eval_reward_cond,
        emit_jerk_ego_obs=emit_jerk_ego_obs,
        creward_deterministic=creward_deterministic,
        creward_ego=creward_ego,
        creward_traffic=creward_traffic,
        env_reward_overrides=env_reward_overrides,
    )

    # Save config
    import json
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Log configuration
    log.info("=" * 60)
    log.info("EVALUATION CONFIG")
    log.info("=" * 60)
    log.info("Output:        %s", output_dir)
    log.info("Config saved:  %s", config_path)
    log.info("Maps:          %d  |  Split: %s", len(map_ids), split)
    log.info("Episodes:      length=%d", int(eval_cfg.get("episode_length", 91)))
    log.info("Action type:   %s", eval_cfg.get("action_type", "continuous"))
    log.info("-" * 60)
    log.info("Ego:           %s", planner_cfg.get("type", "pdm"))
    planner_type = planner_cfg.get("type", "pdm")
    type_params = planner_cfg.get(planner_type, {})
    for k, v in type_params.items():
        if v != "" and v is not None:
            log.info("  %s: %s", k, v)
    log.info("Traffic:       %s", traffic_cfg.get("type", "idm"))
    traffic_type = traffic_cfg.get("type", "idm")
    traffic_params = traffic_cfg.get(traffic_type, {})
    for k, v in traffic_params.items():
        if v != "" and v is not None:
            log.info("  %s: %s", k, v)
    log.info("-" * 60)
    goal_names = {0: "respawn", 1: "generate_new", 2: "stop", 3: "remove",
                  4: "continue", 5: "sample_lane_ahead"}
    term_names = {0: "episode_length", 1: "all_agents_done"}
    log.info("Goal:          %d (%s)", int(eval_cfg.get("goal_behavior", 3)),
             goal_names.get(int(eval_cfg.get("goal_behavior", 3)), "?"))
    log.info("Terminate:     %d (%s)", int(eval_cfg.get("termination_mode", 1)),
             term_names.get(int(eval_cfg.get("termination_mode", 1)), "?"))
    log.info("Collision:     %d  |  Offroad: %d",
             int(eval_cfg.get("collision_behavior", 2)),
             int(eval_cfg.get("offroad_behavior", 2)))
    log.info("Viz:           %s  |  Planner viz: %s",
             eval_cfg.get("viz", False), eval_cfg.get("planner_viz", False))
    log.info("=" * 60)

    # Create planner factories
    ego_factory = partial(create_ego_planner, config)
    traffic_factory = partial(create_traffic_controller, config)

    # Run evaluation
    evaluator = Evaluator(evaluator_config, ego_factory, traffic_factory)
    summary = evaluator.run(map_ids=map_ids)

    log.info("Evaluation complete!")
    return summary


if __name__ == "__main__":
    main()
