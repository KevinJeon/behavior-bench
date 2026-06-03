# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""
Save action trajectories for PBT replay / zeroshot evaluation.

Config-driven CLI (like eval.py), using pufferl.load_config for Drive env/train
settings. Implements the logic from pufferl.zero_shot (hc/pbt branch) via
OtherReplayEvaluator.save_replay / collect_rollouts / replay_rollouts.

Usage:
    # use_all_maps: save all maps in one env, then auto replay-test verify
    DRIVE_BINARIES_DATA_ROOT=/data/puffer/after_ws/bin/ \\
    python pufferlib/ocean/benchmark/save_trajectories.py \\
        --config config/ocean/drive_gigaflow_classic_simple.ini \\
        --load-multiple-model-path /path/ego.pt /path/other.pt \\
        --output-dir /data/puffer/after_ws/test/replay_allmaps \\
        --use-all-maps --num-rollouts 10 \\
        --env.num-maps 100 --env.control-mode control_pbt

    # Sequential maps (one map per env load)
    python pufferlib/ocean/benchmark/save_trajectories.py ... --sequential-maps --env.num-maps 100

    # Population collection (optional advanced mode)
    python pufferlib/ocean/benchmark/save_trajectories.py \\
        --config config/ocean/drive_gigaflow_classic_simple.ini \\
        --trajectories.mode save-population \\
        --population-path /data/puffer/popul_lane_nominal/ \\
        --output-dir /data/puffer/popul_lane_nominal/replay
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("save_trajectories")

_DEFAULT_INI = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "..",
    "..",
    "config",
    "save_trajectories.ini",
)


def _parse_extra_argv(argv):
    """Parse script-only flags before pufferl.load_config (avoids unknown-arg errors)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--trajectories.mode",
        "--trajectories-mode",
        "--mode",
        dest="trajectories.mode",
        default=None,
        choices=("save-replay", "save-population", "replay-test"),
    )
    parser.add_argument(
        "--trajectories.output-dir",
        "--trajectories-output-dir",
        "--output-dir",
        dest="output_dir",
        default=None,
    )
    parser.add_argument(
        "--trajectories.verify-replay",
        "--trajectories-verify-replay",
        dest="trajectories.verify_replay",
        default=None,
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
    )
    parser.add_argument(
        "--trajectories.replay-npz",
        "--trajectories-replay-npz",
        "--replay-npz",
        dest="replay_npz",
        default=None,
    )
    parser.add_argument(
        "--load-multiple-model-path",
        dest="load_multiple_model_path",
        nargs=2,
        default=None,
        metavar=("EGO_CKPT", "OTHER_CKPT"),
    )
    parser.add_argument(
        "--trajectories.num-rollouts",
        "--trajectories-num-rollouts",
        "--num-rollouts",
        dest="num_rollouts",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--trajectories.sequential-maps",
        "--trajectories-sequential-maps",
        "--sequential-maps",
        dest="sequential_maps",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--trajectories.use-all-maps",
        "--trajectories-use-all-maps",
        "--use-all-maps",
        dest="use_all_maps",
        action="store_true",
        default=None,
    )
    # PBT / population flags (not in drive_gigaflow_classic_simple.ini)
    parser.add_argument("--pbt.population-path", "--population-path", dest="population_path", default=None)
    parser.add_argument("--pbt.ego-ratio", "--ego-ratio", dest="ego_ratio", type=float, default=None)
    parser.add_argument("--pbt.pbt-mode", dest="pbt_mode", default=None)
    parser.add_argument("--pbt.num-collect-rollout", dest="num_collect_rollout", type=int, default=None)
    parser.add_argument("--pbt.collect-start-idx", dest="collect_start_idx", type=int, default=None)
    parser.add_argument("--pbt.collect-end-idx", dest="collect_end_idx", type=int, default=None)
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def _setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "save_trajectories.log"), mode="w"),
        ],
    )


def _replay_output_dir(trajectories_cfg, args=None) -> str:
    explicit = (trajectories_cfg.get("output_dir") or "").strip()
    if explicit:
        return explicit
    if args:
        population = (args.get("pbt", {}).get("population_path") or "").strip()
        if population:
            return os.path.join(population, "replay")
    exp_root = os.environ.get("PUFFER_EXP_ROOT", "experiments")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return os.path.join(exp_root, "trajectories", run_id)


def _configure_vecenv(args):
    backend = args.get("eval", {}).get("backend", "PufferEnv")
    args["vec"] = dict(
        backend=backend,
        num_envs=1,
        num_workers=1,
        batch_size=1,
        zero_copy=True,
    )
    args["env"]["episode_length"] = int(args["env"].get("episode_length", 91))


def _stack_ragged_1d(arrays, dtype=np.int32, pad_value=-1):
    """Stack 1D arrays of different lengths; pad tail with pad_value."""
    flat = [np.asarray(a, dtype=dtype).ravel() for a in arrays]
    lengths = np.array([len(a) for a in flat], dtype=np.int32)
    max_len = int(lengths.max()) if len(flat) else 0
    stacked = np.full((len(flat), max_len), pad_value, dtype=dtype)
    for i, a in enumerate(flat):
        stacked[i, : len(a)] = a
    return stacked, lengths


def _stack_rollout_actions(rollout_actions):
    """Stack action buffers; pad agent axis when maps have different agent counts."""
    shapes = {a.shape for a in rollout_actions}
    if len(shapes) == 1:
        return np.stack(rollout_actions, axis=0)

    max_agents = max(a.shape[0] for a in rollout_actions)
    horizon = rollout_actions[0].shape[1]
    action_dim = rollout_actions[0].shape[2]
    num_agents = np.zeros(len(rollout_actions), dtype=np.int32)
    stacked = np.zeros(
        (len(rollout_actions), max_agents, horizon, action_dim),
        dtype=rollout_actions[0].dtype,
    )
    for i, actions in enumerate(rollout_actions):
        n = actions.shape[0]
        num_agents[i] = n
        stacked[i, :n] = actions
    return stacked, num_agents


def _load_policies_from_population(args, vecenv, env_name):
    from pufferlib.pufferl import load_policy

    population_path = args["pbt"]["population_path"]
    checkpoints = sorted(
        os.path.join(population_path, f)
        for f in os.listdir(population_path)
        if f.endswith(".pt")
    )
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints in {population_path}")

    policies = []
    for ckpt in checkpoints:
        run_args = copy.deepcopy(args)
        run_args["load_model_path"] = ckpt
        policy = load_policy(run_args, vecenv, env_name)
        policies.append(policy.eval())
        log.info("Loaded population member: %s", ckpt)
    return policies


def _log_collect_summary(rollout_idx, total, results, num_steps=None):
    metrics = results if isinstance(results, dict) else {}
    log.info(
        "collect %d/%d steps=%s score=%.4f completion=%.4f collision=%.4f offroad=%.4f return=%.4f",
        rollout_idx,
        total,
        num_steps if num_steps is not None else metrics.get("num_steps", "?"),
        float(metrics.get("score", float("nan"))),
        float(metrics.get("completion_rate", float("nan"))),
        float(metrics.get("collision_rate", float("nan"))),
        float(metrics.get("offroad_rate", float("nan"))),
        float(metrics.get("episode_return", float("nan"))),
    )


def _save_use_all_maps_npz(
    output_dir,
    rollout_actions,
    agent_offsets,
    map_ids,
    num_maps,
    ego_indices_list=None,
    num_steps_list=None,
):
    """Persist use_all_maps tensors and metadata."""
    agent_offsets = np.asarray(agent_offsets, dtype=np.int32)
    map_ids = np.asarray(map_ids, dtype=np.int32).ravel()
    if len(rollout_actions) == 1:
        actions = rollout_actions[0].astype(np.int16)
    else:
        actions = np.stack(rollout_actions, axis=0).astype(np.int16)

    out_npz = os.path.join(output_dir, "other_actions.npz")
    payload = dict(
        actions=actions,
        agent_offsets=agent_offsets,
        map_ids=map_ids,
        num_maps=int(num_maps),
        num_rollouts=int(len(rollout_actions)),
        use_all_maps=np.int8(1),
    )
    if ego_indices_list is not None:
        payload["ego_indices"] = np.stack(ego_indices_list, axis=0).astype(np.int32)
    if num_steps_list is not None:
        payload["num_steps"] = np.asarray(num_steps_list, dtype=np.int32)
    np.savez_compressed(out_npz, **payload)
    return out_npz, actions


def _save_replay_npy_bundle(output_dir, actions, agent_offsets, map_ids):
    """Save replay tensors as separate .npy files for fast load."""
    actions_path = os.path.join(output_dir, "other_actions.npy")
    offsets_path = os.path.join(output_dir, "agent_offsets.npy")
    map_ids_path = os.path.join(output_dir, "map_ids.npy")
    np.save(actions_path, np.asarray(actions))
    np.save(offsets_path, np.asarray(agent_offsets, dtype=np.int32))
    np.save(map_ids_path, np.asarray(map_ids, dtype=np.int32))
    return actions_path, offsets_path, map_ids_path


def _num_save_replay_rollouts(trajectories_cfg, pbt, env_cfg) -> int:
    if trajectories_cfg.get("sequential_maps"):
        return int(env_cfg.get("num_maps", 1))
    n = trajectories_cfg.get("num_rollouts")
    if n is None:
        start = int(pbt.get("collect_start_idx", 0))
        end = pbt.get("collect_end_idx")
        if end is not None:
            n = int(end) - start
        else:
            n = int(pbt.get("num_collect_rollout", 1))
    n = int(n)
    if n < 1:
        raise ValueError(f"num_rollouts must be >= 1, got {n}")
    return n


def run_save_replay(args, vecenv, env_name, trajectories_cfg):
    from pufferlib.ocean.benchmark.evaluator import OtherReplayEvaluator
    from pufferlib.pufferl import load_env, load_policy

    paths = args.get("load_multiple_model_path")
    if not paths or len(paths) != 2:
        raise ValueError(
            "save-replay requires --load-multiple-model-path EGO.pt OTHER.pt"
        )

    pbt = args.setdefault("pbt", {})
    sequential_maps = bool(trajectories_cfg.get("sequential_maps", False))
    use_all_maps = bool(args["env"].get("use_all_maps", False))
    if sequential_maps and use_all_maps:
        raise ValueError("--sequential-maps and --use-all-maps cannot be used together")
    num_rollouts = _num_save_replay_rollouts(trajectories_cfg, pbt, args["env"])

    args["env"]["control_mode"] = args["env"].get("control_mode") or "control_pbt"
    args["env"]["pbt_mode"] = "none"
    args["env"]["ego_ratio"] = float(pbt.get("ego_ratio", 0.25))

    output_dir = _replay_output_dir(trajectories_cfg, args)
    os.makedirs(output_dir, exist_ok=True)
    evaluator = OtherReplayEvaluator(args, output_dir=output_dir)
    log.info("Ego policy:   %s", paths[0])
    log.info("Other policy: %s", paths[1])

    if use_all_maps:
        num_maps = int(args["env"]["num_maps"])
        args["env"]["use_all_maps"] = True
        args["env"]["map_id"] = -1
        args["env"]["resample_frequency"] = 910
        log.info(
            "use_all_maps: %d stochastic rollouts over %d maps -> %s",
            num_rollouts,
            num_maps,
            output_dir,
        )

        if vecenv is None:
            vecenv = load_env(env_name, args)
        args["load_model_path"] = paths[0]
        policy1 = load_policy(args, vecenv, env_name)
        args2 = copy.deepcopy(args)
        args2["load_model_path"] = paths[1]
        policy2 = load_policy(args2, vecenv, env_name)
        vecenv.close()
        vecenv = None

        rollout_actions = []
        collect_metrics = []
        ego_indices_list = []
        num_steps_list = []
        saved_offsets = None
        saved_map_ids = None
        for i in range(num_rollouts):
            log.info("use_all_maps collect %d/%d", i + 1, num_rollouts)
            vecenv = load_env(env_name, args)
            actions, offsets, map_ids, ego_idx, results, num_steps = (
                evaluator.collect_replay_rollout(args, vecenv, policy1, policy2)
            )
            vecenv.close()
            rollout_actions.append(actions)
            collect_metrics.append(results)
            ego_indices_list.append(ego_idx)
            num_steps_list.append(num_steps)
            _log_collect_summary(i + 1, num_rollouts, results, num_steps=num_steps)
            if saved_offsets is None:
                saved_offsets = offsets
                saved_map_ids = map_ids
            elif not np.array_equal(saved_offsets, offsets):
                log.warning("agent_offsets changed between rollouts (map layout differs)")

        out_npz, actions_saved = _save_use_all_maps_npz(
            output_dir,
            rollout_actions,
            saved_offsets,
            saved_map_ids,
            num_maps,
            ego_indices_list=ego_indices_list,
            num_steps_list=num_steps_list,
        )
        _save_replay_npy_bundle(output_dir, actions_saved, saved_offsets, saved_map_ids)
        log.info("Wrote %s actions=%s", out_npz, actions_saved.shape)

        result = {
            "npz_path": out_npz,
            "num_rollouts": num_rollouts,
            "use_all_maps": True,
            "collect_metrics": collect_metrics,
        }
        if trajectories_cfg.get("verify_replay", True):
            log.info("Running replay-test verification (use_all_maps)")
            verify_result = run_replay_test(
                args,
                env_name,
                trajectories_cfg,
                npz_path=out_npz,
                policy1=policy1,
                collect_metrics=collect_metrics,
            )
            result["verify"] = verify_result
        return result

    if vecenv is None:
        vecenv = load_env(env_name, args)
    args["load_model_path"] = paths[0]
    policy1 = load_policy(args, vecenv, env_name)

    args2 = copy.deepcopy(args)
    args2["load_model_path"] = paths[1]
    policy2 = load_policy(args2, vecenv, env_name)
    vecenv.close()
    vecenv = None

    if sequential_maps:
        log.info(
            "Sequential maps: map_id 0..%d (%d rollouts) -> %s",
            num_rollouts - 1,
            num_rollouts,
            output_dir,
        )
    else:
        log.info("Collecting %d rollout(s) -> %s", num_rollouts, output_dir)

    rollout_actions = []
    rollout_offsets = []
    rollout_map_ids = []

    for i in range(num_rollouts):
        run_args = copy.deepcopy(args)
        if sequential_maps:
            run_args["env"]["map_id"] = i
            run_args["env"]["use_all_maps"] = False
            run_args["env"]["resample_frequency"] = 910
            log.info("save-replay map %d/%d (map_id=%d)", i + 1, num_rollouts, i)
        else:
            log.info("save-replay rollout %d/%d", i + 1, num_rollouts)

        vecenv = load_env(env_name, run_args)
        actions, offsets, map_ids, ego_idx, results, num_steps = evaluator.collect_replay_rollout(
            run_args, vecenv, policy1, policy2
        )
        vecenv.close()
        _log_collect_summary(i + 1, num_rollouts, results, num_steps=num_steps)
        rollout_actions.append(actions)
        rollout_offsets.append(offsets)
        rollout_map_ids.append(map_ids)

    if num_rollouts == 1:
        out_path = os.path.join(output_dir, "other_actions.npy")
        np.save(out_path, rollout_actions[0])
        np.save(os.path.join(output_dir, "agent_offsets.npy"), np.asarray(rollout_offsets[0], dtype=np.int32))
        np.save(os.path.join(output_dir, "map_ids.npy"), np.asarray(rollout_map_ids[0], dtype=np.int32))
        log.info("Wrote %s shape=%s", out_path, rollout_actions[0].shape)
        return {"other_actions_path": out_path, "num_rollouts": 1}

    if sequential_maps:
        # For sequential maps, store one contiguous action tensor:
        # (sum_agents_over_maps, horizon, action_dim)
        num_agents_per_map = np.asarray([a.shape[0] for a in rollout_actions], dtype=np.int32)
        actions_concat = np.concatenate(rollout_actions, axis=0).astype(np.int16)
        map_agent_offsets = np.zeros(num_rollouts + 1, dtype=np.int32)
        map_agent_offsets[1:] = np.cumsum(num_agents_per_map, dtype=np.int32)
        out_npz = os.path.join(output_dir, "other_actions.npz")
        np.savez_compressed(
            out_npz,
            actions=actions_concat,
            num_maps=num_rollouts,
            map_indices=np.arange(num_rollouts, dtype=np.int32),
            num_agents_per_map=num_agents_per_map,
            map_agent_offsets=map_agent_offsets,
            sequential_maps=np.int8(1),
        )
        _save_replay_npy_bundle(
            output_dir,
            actions_concat,
            map_agent_offsets,
            np.arange(num_rollouts, dtype=np.int32),
        )
        log.info("Wrote %s actions=%s", out_npz, actions_concat.shape)
        result = {"npz_path": out_npz, "num_rollouts": num_rollouts}
        if trajectories_cfg.get("verify_replay", False):
            result["verify"] = run_replay_test(
                args, env_name, trajectories_cfg, npz_path=out_npz
            )
        return result

    stacked = _stack_rollout_actions(rollout_actions)
    if isinstance(stacked, tuple):
        actions_stack, num_agents_per_rollout = stacked
        log.info(
            "Padded variable agent counts across rollouts: %s",
            num_agents_per_rollout.tolist(),
        )
    else:
        actions_stack = stacked
        num_agents_per_rollout = np.full(num_rollouts, actions_stack.shape[1], dtype=np.int32)

    offsets_stack, offsets_lengths = _stack_ragged_1d(rollout_offsets)
    map_ids_stack, map_ids_lengths = _stack_ragged_1d(rollout_map_ids)
    out_npz = os.path.join(output_dir, "other_actions.npz")
    npz_kwargs = dict(
        actions=actions_stack.astype(np.int16),
        agent_offsets=offsets_stack,
        agent_offsets_lengths=offsets_lengths,
        map_ids=map_ids_stack,
        map_ids_lengths=map_ids_lengths,
        num_agents_per_rollout=num_agents_per_rollout,
        num_rollouts=num_rollouts,
    )
    if sequential_maps:
        npz_kwargs["map_indices"] = np.arange(num_rollouts, dtype=np.int32)
        npz_kwargs["sequential_maps"] = np.int8(1)
    np.savez_compressed(out_npz, **npz_kwargs)
    _save_replay_npy_bundle(output_dir, actions_stack.astype(np.int16), offsets_stack, map_ids_stack)
    log.info("Wrote %s actions=%s", out_npz, actions_stack.shape)

    verify = trajectories_cfg.get("verify_replay", False)
    if verify:
        run_replay_test(args, env_name, trajectories_cfg, npz_path=out_npz, shard=actions_stack)

    return {"npz_path": out_npz, "num_rollouts": num_rollouts}


def run_save_population(args, vecenv, env_name, trajectories_cfg):
    from pufferlib.ocean.benchmark.evaluator import OtherReplayEvaluator

    pbt = args.setdefault("pbt", {})
    start = int(pbt.get("collect_start_idx", 0))
    end = int(pbt.get("collect_end_idx", pbt.get("num_collect_rollout", 50)))
    num_collect = end - start
    if num_collect <= 0:
        raise ValueError(f"Invalid collect shard: [{start}, {end})")

    args["env"]["control_mode"] = args["env"].get("control_mode", "control_vehicles")
    args["env"]["pbt_mode"] = "none"

    policies = _load_policies_from_population(args, vecenv, env_name)
    replay_dir = _replay_output_dir(trajectories_cfg, args)
    evaluator = OtherReplayEvaluator(args, output_dir=replay_dir)

    vecenv.close()
    vecenv = None

    rollout_actions = []
    rollout_offsets = []
    rollout_map_ids = []

    from pufferlib.pufferl import load_env

    for local_i, rollout_i in enumerate(range(start, end)):
        log.info("Collect rollout %d (%d/%d in shard)", rollout_i, local_i + 1, num_collect)
        if vecenv is None:
            vecenv = load_env(env_name, args)
        other_action_buf, agent_offsets, map_ids = evaluator.collect_rollouts(
            args, vecenv, policies
        )
        rollout_actions.append(other_action_buf.astype(np.int16))
        rollout_offsets.append(np.asarray(agent_offsets))
        rollout_map_ids.append(np.asarray(map_ids))

        vecenv.close()
        vecenv = None

    stacked = _stack_rollout_actions(rollout_actions)
    if isinstance(stacked, tuple):
        total_actions, num_agents_per_rollout = stacked
    else:
        total_actions = stacked
        num_agents_per_rollout = np.full(num_collect, total_actions.shape[1], dtype=np.int32)

    total_offsets, offsets_lengths = _stack_ragged_1d(rollout_offsets)
    total_map_ids, map_ids_lengths = _stack_ragged_1d(rollout_map_ids)

    out_npz = os.path.join(replay_dir, "other_actions.npz")
    os.makedirs(replay_dir, exist_ok=True)
    np.savez_compressed(
        out_npz,
        actions=total_actions,
        agent_offsets=total_offsets,
        agent_offsets_lengths=offsets_lengths,
        map_ids=total_map_ids,
        map_ids_lengths=map_ids_lengths,
        num_agents_per_rollout=num_agents_per_rollout,
        collect_start_idx=start,
        collect_end_idx=end,
    )
    _save_replay_npy_bundle(replay_dir, total_actions, total_offsets, total_map_ids)
    log.info("Wrote %s  shape=%s", out_npz, total_actions.shape)

    verify = trajectories_cfg.get("verify_replay", True)
    if verify:
        run_replay_test(args, env_name, trajectories_cfg, npz_path=out_npz, shard=total_actions)

    return {"npz_path": out_npz, "num_rollouts": num_collect}


def _replay_maps_from_npz(data, actions):
    """Return list of (map_id, action_slice) for replay-test."""
    if "map_agent_offsets" in data:
        map_agent_offsets = np.asarray(data["map_agent_offsets"], dtype=np.int32)
        map_indices = (
            np.asarray(data["map_indices"], dtype=np.int32)
            if "map_indices" in data
            else np.arange(len(map_agent_offsets) - 1, dtype=np.int32)
        )
        out = []
        for i, map_id in enumerate(map_indices):
            start = int(map_agent_offsets[i])
            end = int(map_agent_offsets[i + 1])
            out.append((int(map_id), actions[start:end]))
        return out

    num_agents_per_map = None
    if "num_agents_per_map" in data:
        num_agents_per_map = np.asarray(data["num_agents_per_map"], dtype=np.int32)
    elif "num_agents_per_rollout" in data:
        num_agents_per_map = np.asarray(data["num_agents_per_rollout"], dtype=np.int32)

    map_indices = (
        np.asarray(data["map_indices"], dtype=np.int32)
        if "map_indices" in data
        else np.arange(actions.shape[0], dtype=np.int32)
    )

    # Legacy stacked sequential format: (num_maps, max_agents, horizon, action_dim)
    if actions.ndim == 4 and num_agents_per_map is not None:
        out = []
        for i, map_id in enumerate(map_indices):
            n = int(num_agents_per_map[i])
            out.append((int(map_id), actions[i, :n]))
        return out

    return None


def _compare_collect_replay_metrics(collect_metrics, replay_metrics, rollout_idx):
    keys = ("score", "completion_rate", "collision_rate", "offroad_rate", "episode_return", "num_steps")
    deltas = []
    for key in keys:
        c = float(collect_metrics.get(key, float("nan")))
        r = float(replay_metrics.get(key, float("nan")))
        deltas.append(abs(r - c))
    log.info(
        "verify rollout %d max_delta=%.6f score=%.4f",
        rollout_idx,
        max(deltas) if deltas else float("nan"),
        float(replay_metrics.get("score", float("nan"))),
    )


def _log_done_summary(result):
    parts = []
    if result.get("npz_path"):
        parts.append(f"npz={result['npz_path']}")
    elif result.get("other_actions_path"):
        parts.append(f"npy={result['other_actions_path']}")
    if "num_rollouts" in result:
        parts.append(f"rollouts={result['num_rollouts']}")
    verify = result.get("verify") or {}
    verified = verify.get("verified_rollouts")
    if verified is not None:
        parts.append(f"verified={verified}")
    log.info("Done: %s", " ".join(parts))


def run_replay_test(
    args,
    env_name,
    trajectories_cfg,
    npz_path=None,
    shard=None,
    policy1=None,
    collect_metrics=None,
):
    from pufferlib.ocean.benchmark.evaluator import OtherReplayEvaluator
    from pufferlib.pufferl import load_env, load_policy

    replay_dir = _replay_output_dir(trajectories_cfg, args)
    npz_path = npz_path or trajectories_cfg.get("replay_npz") or os.path.join(
        replay_dir, "other_actions.npz"
    )
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Replay npz not found: {npz_path}")

    data = np.load(npz_path)
    actions = shard if shard is not None else data["actions"]
    num_agents_per_rollout = data["num_agents_per_rollout"] if "num_agents_per_rollout" in data else None
    sequential_maps = bool(data["sequential_maps"].item()) if "sequential_maps" in data else False
    use_all_maps = bool(data["use_all_maps"].item()) if "use_all_maps" in data else bool(
        args["env"].get("use_all_maps", False)
    )

    args["env"]["pbt_mode"] = "none"
    args["env"]["control_mode"] = args["env"].get("control_mode", "control_pbt")
    args["env"]["population_path"] = replay_dir
    if sequential_maps and "num_maps" in data:
        args["env"]["num_maps"] = int(data["num_maps"])
    elif sequential_maps and actions.ndim >= 3:
        args["env"]["num_maps"] = int(
            data["map_indices"].shape[0]
            if "map_indices" in data
            else (data["map_agent_offsets"].shape[0] - 1 if "map_agent_offsets" in data else actions.shape[0])
        )

    evaluator = OtherReplayEvaluator(args, output_dir=replay_dir)

    if policy1 is None and args.get("load_multiple_model_path"):
        probe_args = copy.deepcopy(args)
        probe_args["load_model_path"] = args["load_multiple_model_path"][0]
        probe_env = load_env(env_name, probe_args)
        policy1 = load_policy(probe_args, probe_env, env_name)
        probe_env.close()

    if use_all_maps and actions.ndim in (3, 4):
        run_args = copy.deepcopy(args)
        run_args["env"]["use_all_maps"] = True
        run_args["env"]["map_id"] = -1
        if "num_maps" in data:
            run_args["env"]["num_maps"] = int(data["num_maps"])
        rollout_tensors = actions if actions.ndim == 4 else actions[None, ...]
        horizon = int(rollout_tensors.shape[-2])
        run_args["env"]["resample_frequency"] = horizon
        num_steps_arr = data["num_steps"] if "num_steps" in data else None
        ego_indices_arr = data["ego_indices"] if "ego_indices" in data else None

        verified = 0
        replay_metrics_all = []
        for r in range(rollout_tensors.shape[0]):
            rollout_actions = rollout_tensors[r]
            steps = int(num_steps_arr[r]) if num_steps_arr is not None else horizon
            ego_idx = ego_indices_arr[r] if ego_indices_arr is not None else None
            vecenv = load_env(env_name, run_args)
            replay_metrics = evaluator.replay_rollouts(
                run_args,
                vecenv,
                rollout_actions,
                ego_indices=ego_idx,
                num_steps=steps,
                exact_actions=True,
            )
            vecenv.close()
            replay_metrics_all.append(replay_metrics)
            if collect_metrics is not None and r < len(collect_metrics):
                _compare_collect_replay_metrics(collect_metrics[r], replay_metrics, r + 1)
            verified += 1
        return {
            "npz_path": npz_path,
            "verified_rollouts": verified,
            "replay_metrics": replay_metrics_all,
        }

    per_map = _replay_maps_from_npz(data, actions)
    if per_map is not None:
        log.info("Replay-test %s with %d maps (per-map env)", npz_path, len(per_map))
        for i, (map_id, rollout_actions) in enumerate(per_map):
            run_args = copy.deepcopy(args)
            run_args["env"]["map_id"] = int(map_id)
            run_args["env"]["use_all_maps"] = False
            run_args["env"]["resample_frequency"] = int(rollout_actions.shape[1])
            if sequential_maps:
                run_args["env"]["num_maps"] = max(int(map_id) + 1, int(args["env"].get("num_maps", 100)))
            vecenv = load_env(env_name, run_args)
            log.info(
                "replay_rollouts map %d/%d (map_id=%d, agents=%d)",
                i + 1,
                len(per_map),
                int(map_id),
                rollout_actions.shape[0],
            )
            replay_metrics = evaluator.replay_rollouts(
                run_args, vecenv, rollout_actions, policy1=policy1
            )
            vecenv.close()
        return {"npz_path": npz_path, "verified_rollouts": len(per_map)}

    log.info("Replay-test %s with %d rollouts", npz_path, len(actions))
    vecenv = load_env(env_name, args)
    for i in range(len(actions)):
        log.info("replay_rollouts %d/%d", i + 1, len(actions))
        rollout_actions = actions[i]
        if num_agents_per_rollout is not None:
            n = int(num_agents_per_rollout[i])
            rollout_actions = rollout_actions[:n]
        replay_metrics = evaluator.replay_rollouts(args, vecenv, rollout_actions, policy1=policy1)
        if i < len(actions) - 1:
            vecenv.close()
            vecenv = load_env(env_name, args)
    vecenv.close()
    return {"npz_path": npz_path, "verified_rollouts": len(actions)}


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if "DRIVE_BINARIES_DATA_ROOT" not in os.environ:
        print("ERROR: DRIVE_BINARIES_DATA_ROOT not set", file=sys.stderr)
        sys.exit(1)

    extra, remaining = _parse_extra_argv(argv)
    sys.argv = [sys.argv[0]] + remaining

    from pufferlib import pufferl

    _DRIVE_CLASSIC_INI = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "..",
            "..",
            "config",
            "ocean",
            "drive_gigaflow_classic_simple.ini",
        )
    )
    if not any(a.startswith("--config") for a in remaining):
        remaining = ["--config", _DRIVE_CLASSIC_INI] + remaining
        sys.argv = [sys.argv[0]] + remaining
    else:
        idx = remaining.index("--config") if "--config" in remaining else -1
        if idx >= 0 and idx + 1 < len(remaining):
            user_ini = remaining[idx + 1]
            if not os.path.isfile(user_ini):
                raise FileNotFoundError(
                    f"Config not found: {user_ini}\n"
                    f"Use an existing drive ini, e.g.:\n  --config {_DRIVE_CLASSIC_INI}"
                )

    env_name = "puffer_drive"
    args = pufferl.load_config(env_name)

    package = args.get("package")
    if package in (None, "None", ""):
        raise pufferlib.APIUsageError(
            "Config did not set [base] package=ocean (only default.ini loaded?). "
            "Pass a valid drive ini, e.g. --config config/ocean/drive_gigaflow_classic_simple.ini"
        )
    if args["env"].get("num_maps") is None:
        raise pufferlib.APIUsageError(
            "Config missing [env] settings (num_maps, control_mode, ...). "
            "Check --config path points to a drive *.ini file."
        )

    extra_d = vars(extra)
    trajectories_cfg = args.setdefault("trajectories", {})
    if extra_d.get("trajectories.mode"):
        trajectories_cfg["mode"] = extra_d["trajectories.mode"]
    elif extra.load_multiple_model_path:
        trajectories_cfg["mode"] = "save-replay"
    if extra.output_dir:
        trajectories_cfg["output_dir"] = extra.output_dir
    if extra_d.get("trajectories.verify_replay") is not None:
        trajectories_cfg["verify_replay"] = extra_d["trajectories.verify_replay"]
    if extra.replay_npz:
        trajectories_cfg["replay_npz"] = extra.replay_npz
    if extra.load_multiple_model_path:
        args["load_multiple_model_path"] = list(extra.load_multiple_model_path)
    if extra.num_rollouts is not None:
        trajectories_cfg["num_rollouts"] = extra.num_rollouts
    if extra.sequential_maps:
        trajectories_cfg["sequential_maps"] = True
    if extra.use_all_maps:
        args["env"]["use_all_maps"] = True
        args["env"]["resample_frequency"] = 910
        if trajectories_cfg.get("verify_replay") is None:
            trajectories_cfg["verify_replay"] = True

    pbt = args.setdefault("pbt", {})
    if extra.population_path:
        pbt["population_path"] = extra.population_path
    if extra.ego_ratio is not None:
        pbt["ego_ratio"] = extra.ego_ratio
    if extra.pbt_mode:
        pbt["pbt_mode"] = extra.pbt_mode
    if extra.num_collect_rollout is not None:
        pbt["num_collect_rollout"] = extra.num_collect_rollout
    if extra.collect_start_idx is not None:
        pbt["collect_start_idx"] = extra.collect_start_idx
    if extra.collect_end_idx is not None:
        pbt["collect_end_idx"] = extra.collect_end_idx
    pbt.setdefault("ego_ratio", 0.25)

    mode = trajectories_cfg.get("mode", "save-replay")
    output_dir = _replay_output_dir(trajectories_cfg, args)
    _setup_logging(output_dir)

    _configure_vecenv(args)
    args["env"]["split"] = args.get("eval", {}).get("split", args["env"].get("split", "training"))

    from pufferlib.pufferl import load_env

    log.info("Mode: %s | output: %s", mode, output_dir)
    log.info(
        "Env split=%s num_maps=%s control_mode=%s use_all_maps=%s",
        args["env"].get("split"),
        args["env"].get("num_maps"),
        args["env"].get("control_mode"),
        args["env"].get("use_all_maps", False),
    )

    if mode == "save-replay":
        vecenv = load_env(env_name, args)
        result = run_save_replay(args, vecenv, env_name, trajectories_cfg)
    elif mode == "save-population":
        result = run_save_population(args, None, env_name, trajectories_cfg)
    elif mode == "replay-test":
        result = run_replay_test(args, env_name, trajectories_cfg)
    else:
        raise ValueError(f"Unknown trajectories.mode: {mode}")

    _log_done_summary(result)
    return result


if __name__ == "__main__":
    main()
