# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Map evaluator for trajectory planners."""

import logging
import os
import time
import glob as globlib
from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import rich
import rich.box
from rich.table import Table
from rich.console import Console

from pufferlib.planning.base import BasePlanner
from pufferlib.ocean.drive.drive import Drive
from pufferlib.ocean.drive import binding
from pufferlib.viz import plot_simulator_state, VizConfig, compute_axis_limits_for_ego_and_goal

from .config import EvaluatorConfig, ActionConfig
from .collision_classifier import CollisionType, classify_collision, is_at_fault
from .metrics import MapMetrics, MetricsWriter
from .uncertainty import compute_aleatoric, compute_epistemic, PolicyEnsemble, plot_uncertainty_correlation

log = logging.getLogger("evaluator")


# Type aliases for planner factory functions
EgoPlannerFactory = Callable[[Drive, ActionConfig, Optional[BasePlanner], int], BasePlanner]
OtherPlannerFactory = Callable[[Drive, ActionConfig, int], BasePlanner]


class ActionConverter:
    """Handles conversion between continuous and discrete action spaces."""

    def __init__(self, config: ActionConfig):
        self.config = config
        self._neutral_idx = self.to_index(config.neutral_action)

    def to_index(self, action: np.ndarray) -> int:
        """Convert single continuous action to discrete index."""
        accel_idx = int(np.argmin(np.abs(self.config.accel_values - action[0])))
        steer_idx = int(np.argmin(np.abs(self.config.steer_values - action[1])))
        return accel_idx * len(self.config.steer_values) + steer_idx

    def to_index_batch(self, actions: np.ndarray) -> np.ndarray:
        """Convert batch of continuous actions to discrete indices."""
        accel_idx = np.argmin(
            np.abs(self.config.accel_values[None, :] - actions[:, 0:1]), axis=1
        )
        steer_idx = np.argmin(
            np.abs(self.config.steer_values[None, :] - actions[:, 1:2]), axis=1
        )
        return (accel_idx * len(self.config.steer_values) + steer_idx).astype(np.int32)

    @property
    def neutral_index(self) -> int:
        """Return discrete index for neutral action."""
        return self._neutral_idx


@dataclass
class MapContext:
    """Holds all state for evaluation on a single map."""

    map_idx: int
    map_id: int
    env: Drive
    planners: List[BasePlanner]
    converter: ActionConverter
    output_dir: str
    config: EvaluatorConfig
    metrics: MapMetrics
    step_idx: int = 0
    done: bool = False
    axis_limits: Optional[tuple] = None


class Evaluator:
    """
    Evaluates planners across multiple maps sequentially.

    Args:
        config: Evaluator configuration
        ego_planner_factory: Factory function to create ego planner (agent 0)
        other_planner_factory: Factory function to create planner for other agents
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        ego_planner_factory: EgoPlannerFactory,
        other_planner_factory: OtherPlannerFactory,
    ):
        self.config = config
        self.ego_planner_factory = ego_planner_factory
        self.other_planner_factory = other_planner_factory
        self.converter = ActionConverter(config.action_config)
        self.all_metrics: List[MapMetrics] = []
        self._total_maps = 0  # set in run()
        self._start_time = 0.0
        self._console = Console()
        self._hybrid_map_data: List[Dict[str, Any]] = []  # per-map hybrid diagnostics
        # Step-level uncertainty data (aggregated across all maps)
        self._step_aleatoric_all: List[float] = []
        self._step_epistemic_all: List[float] = []
        self._step_value_variance_all: List[float] = []
        self._step_rewards_all: List[float] = []

    def _load_manifest(self) -> Dict[int, int]:
        """Load ego_agent_idx mapping from manifest.csv in the split directory.

        The manifest maps sequential map IDs (derived from new_filename like
        map_000042.bin -> 42) to ego_agent_idx values.
        """
        import csv as _csv
        import re as _re
        data_root = os.environ.get("DRIVE_BINARIES_DATA_ROOT", "")
        manifest_path = os.path.join(data_root, self.config.split, "manifest.csv")
        if not os.path.isfile(manifest_path):
            return {}
        mapping = {}
        with open(manifest_path, "r") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if "ego_agent_idx" not in row:
                    break  # manifest doesn't have ego_agent_idx column
                # Derive sequential map_id from new_filename (e.g., map_000042.bin -> 42)
                fname = row.get("new_filename", "")
                m = _re.search(r"map_(\d+)", fname)
                if m:
                    mid = int(m.group(1))
                else:
                    continue
                ego_idx = int(row["ego_agent_idx"])
                mapping[mid] = ego_idx
        if mapping:
            log.info("Loaded manifest with %d entries from %s", len(mapping), manifest_path)
        return mapping

    def run(self, map_ids: List[int], ego_agent_indices: Optional[Dict[int, int]] = None) -> Dict[str, Any]:
        """
        Run evaluation on specified maps sequentially.

        Args:
            map_ids: List of specific map IDs to evaluate.
            ego_agent_indices: Optional dict mapping map_id -> ego_agent_idx.
                If None, attempts to load from manifest.csv in the split directory.
                If no manifest found, defaults to 0 for all maps.

        Returns:
            Summary statistics dictionary
        """
        if not map_ids:
            raise ValueError("map_ids must be provided and non-empty")

        # Load ego agent indices from manifest if not provided
        if ego_agent_indices is None:
            ego_agent_indices = self._load_manifest()

        self._total_maps = len(map_ids)
        self._start_time = time.perf_counter()
        total_start = self._start_time
        all_collision_snapshots = []

        for map_idx, map_id in enumerate(map_ids):
            ego_idx = ego_agent_indices.get(map_id, 0)
            metrics, snapshots = self._evaluate_map(map_idx, map_id, ego_agent_idx=ego_idx)
            all_collision_snapshots.extend(snapshots)
            if metrics is not None:
                self.all_metrics.append(metrics)
            else:
                log.warning("  Skipped map %d (ID=%d)", map_idx, map_id)
            self._print_dashboard()

        total_time = time.perf_counter() - total_start

        writer = MetricsWriter(self.config.output_dir)
        per_map_path = writer.write_per_map(self.all_metrics)
        summary_path = writer.write_summary(self.all_metrics)
        summary = writer.compute_summary_dict(self.all_metrics)
        summary["total_time_s"] = total_time

        log.info("Results: %s", per_map_path)

        # Save collision snapshots for diagnostics
        if all_collision_snapshots:
            import json
            snap_path = os.path.join(self.config.output_dir, "collision_snapshots.json")
            with open(snap_path, "w") as f:
                json.dump(all_collision_snapshots, f, indent=2)
            log.info("Collision snapshots: %s (%d collisions)", snap_path, len(all_collision_snapshots))

        # Generate uncertainty correlation plots (per-step data)
        if self.config.compute_uncertainty and self._step_rewards_all:
            rw = np.array(self._step_rewards_all)
            al = np.array(self._step_aleatoric_all) if self._step_aleatoric_all else None
            ep = np.array(self._step_epistemic_all) if self._step_epistemic_all else None
            vv = np.array(self._step_value_variance_all) if self._step_value_variance_all else None
            if al is not None and len(al) == len(rw):
                plot_uncertainty_correlation(
                    self._step_rewards_all, list(al),
                    list(ep) if ep is not None and len(ep) == len(rw) else None,
                    self.config.output_dir,
                    value_variance=list(vv) if vv is not None and len(vv) == len(rw) else None,
                )
                # Log correlations
                if len(rw) > 2:
                    corr_al = np.corrcoef(al, rw)[0, 1]
                    log.info("Aleatoric-Reward correlation: rho=%.3f", corr_al)
                if ep is not None and len(ep) == len(rw) and len(rw) > 2:
                    corr_ep = np.corrcoef(ep, rw)[0, 1]
                    log.info("Epistemic-Reward correlation:  rho=%.3f", corr_ep)
                if vv is not None and len(vv) == len(rw) and len(rw) > 2:
                    corr_vv = np.corrcoef(vv, rw)[0, 1]
                    log.info("Value-Variance-Reward correlation: rho=%.3f", corr_vv)

        # Aggregate hybrid planner diagnostics
        if self._hybrid_map_data:
            self._save_hybrid_summary()

        return summary

    def _evaluate_map(self, map_idx: int, map_id: int, ego_agent_idx: int = 0) -> Optional[MapMetrics]:
        """Evaluate a single map and return metrics.

        Args:
            map_idx: Sequential index in this evaluation run.
            map_id: Map ID in the split.
            ego_agent_idx: Which active agent to treat as ego (position in active_agent_indices).
        """
        map_start_time = time.perf_counter()
        try:
            # Create environment
            env = self._create_env(map_id, human_agent_idx=ego_agent_idx)
            if env is None:
                return None

            # Reset and get agent count
            obs, _ = env.reset()
            num_agents = env.num_agents
            actual_map_id = env.map_ids[0] if env.map_ids else -1

            # Map entity index from manifest → active agent position
            if ego_agent_idx != 0:
                state = env.get_state()
                if isinstance(state, list):
                    state = state[0] if state else {}
                active_indices = state.get("active_agent_indices", [])
                if active_indices:
                    try:
                        ego_agent_idx = list(active_indices).index(ego_agent_idx)
                    except ValueError:
                        log.warning("Ego entity %d not in active agents for map %d, skipping", ego_agent_idx, map_id)
                        env.close()
                        return None, []

            # Create planners
            other_planner = None
            if num_agents > 1:
                other_planner = self.other_planner_factory(env, self.config.action_config, ego_agent_idx)

            ego_planner = self.ego_planner_factory(
                env, self.config.action_config, other_planner, ego_agent_idx
            )

            planners = [ego_planner]
            if other_planner is not None:
                planners.append(other_planner)

            # Setup ensemble for epistemic uncertainty (skip if ego planner has its own)
            ensemble = None
            if self.config.compute_uncertainty and self.config.ensemble_weight_paths:
                if not hasattr(ego_planner, 'ensemble'):
                    ensemble = PolicyEnsemble(
                        self.config.ensemble_weight_paths, env, ego_planner.config,
                        ego_agent_idx=ego_agent_idx,
                    )

            # Setup output directory
            map_output_dir = os.path.join(self.config.output_dir, f"map_{map_id:03d}")
            if self.config.viz or self.config.planner_viz:
                os.makedirs(map_output_dir, exist_ok=True)

            # Compute fixed axis limits
            axis_limits = None
            if self.config.viz:
                initial_state = env.get_state()
                if isinstance(initial_state, list) and initial_state:
                    initial_state = initial_state[0]
                if initial_state:
                    axis_limits = compute_axis_limits_for_ego_and_goal(
                        initial_state, test_agent_idx=ego_agent_idx, padding=30.0
                    )

            # Create metrics
            metrics = MapMetrics(map_id=actual_map_id)
            at_fault_steps = 0
            had_collision = False
            ego_position_history = []  # last N (x,y) for lane-change detection
            POSITION_HISTORY_LEN = 4
            collision_snapshots = []  # diagnostic collision data
            collision_info = None  # (collision_type, at_fault) for rendering
            step_aleatoric = []
            step_epistemic = []
            step_value_variance = []
            step_rewards = []  # per-step rewards for hybrid diagnostics
            counterfactual_rewards = []  # per-step counterfactual rewards
            collision_step = None  # step at which first collision occurred
            # Per-component reward sums (ego-only). Tracked in float64 to avoid
            # drift on long episodes; converted to dict at end of map.
            component_names = getattr(env, "REWARD_COMPONENT_NAMES", ())
            component_totals = np.zeros(len(component_names), dtype=np.float64)
            # Parallel α-less behavior sums (violation counts, seconds of reversing, etc.).
            component_totals_raw = np.zeros(len(component_names), dtype=np.float64)
            ego_speed_sum = 0.0
            ego_speed_steps = 0

            # Run episode
            step_idx = 0
            done = False
            while not done:
                # Get state before planning
                state_before = env.get_state()
                if isinstance(state_before, list):
                    state_before = state_before[0] if state_before else {}

                # Plan
                final_actions = np.zeros((num_agents, 2), dtype=np.float32)
                t0 = time.perf_counter()
                final_actions[ego_agent_idx, :] = planners[0].plan(current_step=step_idx, obs=obs[ego_agent_idx], extract_trajectories=self.config.viz or self.config.planner_viz)
                planning_time_ms = (time.perf_counter() - t0) * 1000
                metrics.planning_time_ms += planning_time_ms

                # Compute uncertainty from ego planner logits
                if self.config.compute_uncertainty and hasattr(planners[0], 'last_logits'):
                    step_aleatoric.append(compute_aleatoric(planners[0].last_logits))
                    if ensemble is not None:
                        ensemble_logits, ensemble_values = ensemble.forward_all_with_values(obs[ego_agent_idx])
                        all_logits = [planners[0].last_logits] + ensemble_logits
                        step_epistemic.append(compute_epistemic(all_logits))
                        # Value head variance across ensemble
                        ego_value = getattr(planners[0], 'last_value', None)
                        if ego_value is not None:
                            all_values = [ego_value] + ensemble_values
                            step_value_variance.append(float(np.var(all_values)))

                # Other agents — pass all non-ego observations as batch
                if len(planners) > 1 and num_agents > 1:
                    other_mask = np.ones(num_agents, dtype=bool)
                    other_mask[ego_agent_idx] = False
                    other_obs = obs[other_mask]  # (num_agents-1, obs_dim)
                    other_actions = planners[1].plan(current_step=step_idx, obs=other_obs)  # (num_agents-1, 2) or (2,)
                    other_actions = np.atleast_2d(other_actions)
                    final_actions[other_mask, :] = other_actions

                # Counterfactual step (hybrid planner: step copy with other planner)
                ego_planner = planners[0]
                cf_reward = None
                if hasattr(ego_planner, 'last_counterfactual_action'):
                    MOVEMENT_IDM = 1
                    MOVEMENT_DYNAMICS = 0
                    snapshot = env.create_snapshot()

                    # If counterfactual is PDM, apply its IDM params
                    if ego_planner.last_counterfactual_mode == MOVEMENT_IDM:
                        ego_planner.pdm.apply_best_proposal()
                    binding.vec_set_movement_mode(
                        env.c_envs, [ego_agent_idx], ego_planner.last_counterfactual_mode
                    )

                    cf_actions = final_actions.copy()
                    cf_actions[ego_agent_idx, :] = ego_planner.last_counterfactual_action
                    _, cf_rewards_arr, _, _, _ = env.step(cf_actions)
                    cf_reward = float(cf_rewards_arr[ego_agent_idx])

                    env.restore_snapshot(snapshot)
                    env.free_snapshot(snapshot)

                    # Restore chosen mode
                    chosen_mode = MOVEMENT_IDM if ego_planner._using_pdm else MOVEMENT_DYNAMICS
                    binding.vec_set_movement_mode(env.c_envs, [ego_agent_idx], chosen_mode)

                counterfactual_rewards.append(cf_reward)

                # Step
                obs, rewards, dones, truncs, infos = env.step(final_actions)

                real_reward = float(rewards[ego_agent_idx])
                step_rewards.append(real_reward)

                # Update metrics
                metrics.total_reward += real_reward
                metrics.num_steps += 1

                # Accumulate per-component breakdown for the ego agent only.
                rc_buf = getattr(env, "reward_components", None)
                if rc_buf is not None and len(component_names) > 0:
                    component_totals += rc_buf[ego_agent_idx]
                raw_buf = getattr(env, "reward_components_raw", None)
                step_components_raw = None
                step_components_weighted = None
                if raw_buf is not None and len(component_names) > 0:
                    component_totals_raw += raw_buf[ego_agent_idx]
                    step_components_raw = raw_buf[ego_agent_idx].copy()
                if rc_buf is not None and len(component_names) > 0:
                    step_components_weighted = rc_buf[ego_agent_idx].copy()

                # Track goal distance (skip if agent is done/removed — obs may be invalid)
                if not dones[ego_agent_idx]:
                    obs = np.atleast_2d(obs)
                    rel_goal_x = obs[ego_agent_idx, 0]
                    rel_goal_y = obs[ego_agent_idx, 1]
                    metrics.final_goal_distance = float(np.hypot(rel_goal_x, rel_goal_y))

                # Track ego position for lane-change detection
                if state_before:
                    sb_entities = state_before.get("entities", [])
                    sb_active = state_before.get("active_agent_indices", [])
                    if sb_active and ego_agent_idx < len(sb_active) and sb_active[ego_agent_idx] < len(sb_entities):
                        eb = sb_entities[sb_active[ego_agent_idx]]
                        if eb.get("x", -10000) > -9000:
                            ego_position_history.append((eb["x"], eb["y"]))
                            if len(ego_position_history) > POSITION_HISTORY_LEN:
                                ego_position_history.pop(0)
                            vx = eb.get("vx", 0.0)
                            vy = eb.get("vy", 0.0)
                            ego_speed_sum += float(np.hypot(vx, vy))
                            ego_speed_steps += 1

                # Render - use state_before so trajectories align with vehicle position
                state_after = env.get_state()
                if isinstance(state_after, list):
                    state_after = state_after[0] if state_after else {}

                # Classify collision if one occurred.
                # The C++ engine saves actual collision positions (collision_x/y)
                # and the colliding entity index (collided_with_index) before
                # clearing positions to -10000. We use these real positions
                # combined with velocity/heading from state_before.
                if state_after and state_before and not had_collision:
                    after_entities = state_after.get("entities", [])
                    after_active = state_after.get("active_agent_indices", [])
                    if after_active and ego_agent_idx < len(after_active) and after_active[ego_agent_idx] < len(after_entities):
                        ego_after = after_entities[after_active[ego_agent_idx]]
                        if ego_after.get("collision_state", 0) == 1:
                            had_collision = True
                            collision_step = step_idx
                            before_entities = state_before.get("entities", [])
                            before_active = state_before.get("active_agent_indices", [])
                            other_idx = ego_after.get("collided_with_index", -1)
                            if (other_idx >= 0
                                    and other_idx < len(before_entities)
                                    and before_active
                                    and ego_agent_idx < len(before_active)
                                    and before_active[ego_agent_idx] < len(before_entities)):
                                ego_before = before_entities[before_active[ego_agent_idx]]
                                other_before = before_entities[other_idx]
                                # Build collision dicts: real positions from C++,
                                # velocity/heading/dimensions from state_before
                                ego_col = dict(ego_before,
                                    x=ego_after["collision_x"],
                                    y=ego_after["collision_y"],
                                )
                                other_col = dict(other_before,
                                    x=ego_after["collision_other_x"],
                                    y=ego_after["collision_other_y"],
                                )
                                ctype = classify_collision(ego_col, other_col)
                                fault = is_at_fault(
                                    ctype, ego_col, other_col,
                                    entities=before_entities,
                                    ego_position_history=ego_position_history,
                                )
                                collision_info = {
                                    "type": ctype,
                                    "at_fault": fault,
                                    "ego_x": ego_after["collision_x"],
                                    "ego_y": ego_after["collision_y"],
                                    "other_x": ego_after["collision_other_x"],
                                    "other_y": ego_after["collision_other_y"],
                                    "other_entity_idx": other_idx,
                                }
                                if fault:
                                    at_fault_steps += 1
                                log.debug(
                                    "  COLLISION step=%d type=%s fault=%s other_idx=%d",
                                    step_idx, ctype, fault, other_idx,
                                )
                                # Save collision snapshot for diagnostics
                                _keys = ["x", "y", "vx", "vy", "heading", "length", "width"]
                                snap = {
                                    "map_id": actual_map_id,
                                    "step": step_idx,
                                    "ego": {k: ego_col.get(k, 0) for k in _keys},
                                    "other": {k: other_col.get(k, 0) for k in _keys},
                                    "collision_type": ctype.name if hasattr(ctype, 'name') else str(ctype),
                                    "at_fault": fault,
                                    "ego_position_history": list(ego_position_history[-4:]),
                                }
                                ego_planner = planners[0]
                                if hasattr(ego_planner, '_using_pdm'):
                                    snap["active_planner"] = "PDM" if ego_planner._using_pdm else "PPO"
                                    snap["epistemic"] = getattr(ego_planner, 'last_epistemic', None)
                                collision_snapshots.append(snap)
                            else:
                                log.debug(
                                    "  COLLISION step=%d: no collided_with_index",
                                    step_idx,
                                )

                if self.config.viz:
                    traffic_planner = planners[1] if len(planners) > 1 else None
                    self._render_step(
                        map_output_dir, step_idx, state_before, final_actions[ego_agent_idx],
                        planners[0], axis_limits, map_idx, env_reward=real_reward,
                        collision_info=collision_info,
                        traffic_planner=traffic_planner,
                        test_agent_idx=ego_agent_idx,
                        component_names=component_names,
                        step_components_raw=step_components_raw,
                        step_components_weighted=step_components_weighted,
                    )

                # Save planner-specific visualization (PDM proposals, etc.)
                if self.config.planner_viz:
                    ego_planner = planners[0]
                    should_save = (
                        hasattr(ego_planner, 'has_iteration_history') and
                        ego_planner.has_iteration_history and
                        (self.config.iteration_gif_steps is None or step_idx in self.config.iteration_gif_steps)
                    )
                    if should_save:
                        video_path = os.path.join(map_output_dir, f"planner_iter_step_{step_idx:03d}.mp4")
                        ego_planner.save_iteration_video(video_path, state_before, axis_limits)

                step_idx += 1

                # Check done condition based on goal_behavior:
                # - goal_behavior 0 (RESPAWN) or 1 (GENERATE_NEW): continue on ego done, only end on truncation/step limit
                # - goal_behavior 2 (STOP) or 3 (REMOVE): end when ego is done (reached goal/stopped/removed)
                should_end_on_done = self.config.goal_behavior >= 2  # STOP or REMOVE
                # Under non-terminating goal_behavior (RESPAWN / GENERATE_NEW) the env
                # keeps the ego alive after reaching its goal, but the benchmark
                # episode should still end at first ego goal-reach. Detect via the
                # monotonic counter on the ego agent log.
                ego_reached_goal = False
                if hasattr(env, 'get_agent_log'):
                    try:
                        _ego_log = env.get_agent_log(ego_agent_idx)
                        if isinstance(_ego_log, dict) and _ego_log.get("goals_reached_this_episode", 0) > 0:
                            ego_reached_goal = True
                    except Exception:
                        pass
                if (truncs[ego_agent_idx]
                        or step_idx >= self.config.episode_length
                        or (should_end_on_done and dones[ego_agent_idx])
                        or ego_reached_goal):
                    done = True
                    # Render final state without trajectory overlay
                    if self.config.viz:
                        if collision_info is not None:
                            # Build collision state from state_before, move both
                            # vehicles to their actual collision positions
                            import copy
                            final_state = copy.deepcopy(state_before)
                            active = final_state.get("active_agent_indices", [])
                            ents = final_state.get("entities", [])
                            if active and ego_agent_idx < len(active) and active[ego_agent_idx] < len(ents):
                                ego = ents[active[ego_agent_idx]]
                                ego["x"] = collision_info["ego_x"]
                                ego["y"] = collision_info["ego_y"]
                            oidx = collision_info.get("other_entity_idx", -1)
                            if 0 <= oidx < len(ents):
                                other = ents[oidx]
                                other["x"] = collision_info["other_x"]
                                other["y"] = collision_info["other_y"]
                        else:
                            final_state = state_after
                        self._render_step(
                            map_output_dir, step_idx, final_state, final_actions[ego_agent_idx],
                            None, axis_limits, map_idx, env_reward=real_reward,
                            collision_info=collision_info,
                            test_agent_idx=ego_agent_idx,
                            component_names=component_names,
                            step_components_raw=step_components_raw,
                            step_components_weighted=step_components_weighted,
                        )
                    # Extract metrics from env info
                    ego_info = None
                    if infos is not None and len(infos) > 0:
                        ego_info = infos[ego_agent_idx] if isinstance(infos, (list, np.ndarray)) and len(infos) > ego_agent_idx else infos
                    elif hasattr(env, 'get_agent_log'):
                        ego_info = env.get_agent_log(ego_agent_idx)
                    elif hasattr(env, 'get_episode_log'):
                        ego_info = env.get_episode_log()

                    if isinstance(ego_info, dict) and ego_info:
                        completion_rate = ego_info.get("completion_rate", -1)
                        metrics.goal_reached = completion_rate > 0 if completion_rate != -1 else False
                        metrics.collision_rate = ego_info.get("collision_rate", -1)
                        metrics.offroad_rate = ego_info.get("offroad_rate", -1)
                        # Store additional metrics
                        metrics.episode_return = ego_info.get("episode_return", -1)
                        metrics.score = ego_info.get("score", -1)
                        metrics.lane_alignment_rate = ego_info.get("lane_alignment_rate", -1)
                        # Derived [0, 1] benchmark scores (higher = better).
                        steps_alive = max(ego_info.get("lane_distance_count", 0.0), 1.0)
                        viol = ego_info.get("comfort_violations", 0.0)
                        aligned = ego_info.get("lane_aligned_steps", 0.0)
                        mean_dist = ego_info.get("lane_distance_avg", 0.0)  # already pre-averaged in binding
                        metrics.score_comfort  = max(0.0, 1.0 - min(viol / (3.0 * steps_alive), 1.0))
                        metrics.score_l_align  = max(0.0, min(aligned / steps_alive, 1.0))
                        metrics.score_l_center = max(0.0, 1.0 - min(mean_dist / 2.0, 1.0))

                    # Set at-fault collision rate (1.0 if any at-fault collision, 0.0 otherwise)
                    metrics.at_fault_collision_rate = 1.0 if at_fault_steps > 0 else 0.0
                    metrics.mean_speed_mps = ego_speed_sum / max(ego_speed_steps, 1)

                    # Set uncertainty metrics
                    if step_aleatoric:
                        metrics.mean_aleatoric = float(np.mean(step_aleatoric))
                    if step_epistemic:
                        metrics.mean_epistemic = float(np.mean(step_epistemic))
                    if step_value_variance:
                        metrics.mean_value_variance = float(np.mean(step_value_variance))

                    # Collect step-level data for per-step plots
                    if self.config.compute_uncertainty:
                        self._step_aleatoric_all.extend(step_aleatoric)
                        self._step_epistemic_all.extend(step_epistemic)
                        self._step_value_variance_all.extend(step_value_variance)
                        self._step_rewards_all.extend(step_rewards)


            # Hybrid planner diagnostics: save per-step CSV + timeline plot
            if hasattr(ego_planner, '_step_planners') and ego_planner._step_planners:
                os.makedirs(map_output_dir, exist_ok=True)
                self._save_hybrid_diagnostics(
                    map_output_dir, ego_planner, step_rewards, collision_step,
                    counterfactual_rewards,
                )

            # Create video
            if self.config.viz:
                video_path = os.path.join(map_output_dir, "episode.mp4")
                self._create_video(map_output_dir, video_path)

            # Cleanup. Prefer the planner's own close() (Hybrid handles its
            # PDM, PPO-rollout and lookahead batch envs internally); fall back
            # to the historical hard-coded paths for planners that predate it.
            if hasattr(ego_planner, 'close') and callable(ego_planner.close):
                ego_planner.close()
            else:
                if hasattr(ego_planner, 'batch_env'):
                    ego_planner.batch_env.close()
                if hasattr(ego_planner, 'pdm') and hasattr(ego_planner.pdm, 'batch_env'):
                    ego_planner.pdm.batch_env.close()
            env.close()

            # Free GPU memory explicitly
            del ego_planner, planners
            if other_planner is not None:
                del other_planner
            if ensemble is not None:
                del ensemble
            import torch as _torch, gc as _gc
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            _gc.collect()

            # Set total evaluation time
            metrics.total_time_s = time.perf_counter() - map_start_time

            # Attach per-component reward breakdown (ego-only episode sums).
            if len(component_names) > 0:
                metrics.reward_components = {
                    name: float(component_totals[i])
                    for i, name in enumerate(component_names)
                }
                metrics.reward_components_raw = {
                    name: float(component_totals_raw[i])
                    for i, name in enumerate(component_names)
                }

            return metrics, collision_snapshots

        except Exception as e:
            log.exception("Error evaluating map %d (ID %d)", map_idx, map_id)
            return None, []

    def _count_visible_agents(self, state: Dict) -> int:
        """Count agents that are visible (not removed, not at -10000)."""
        entities = state.get("entities", [])
        active_indices = state.get("active_agent_indices", [])
        visible = 0
        for idx in (active_indices or []):
            if idx < len(entities):
                e = entities[idx]
                removed = e.get("removed", 0)
                x = e.get("x", 0)
                if removed == 0 and x > -9000:
                    visible += 1
        return visible

    def _create_env(self, map_id: int, human_agent_idx: int = 0) -> Optional[Drive]:
        """Create environment for a map.

        Args:
            map_id: Map ID to load.
            human_agent_idx: Which agent in the scenario to treat as ego (default 0).
        """
        if map_id < 0:
            raise ValueError(f"map_id must be non-negative, got {map_id}")

        eval_ini = os.path.join(os.path.dirname(__file__), "..", "config", "evaluation.ini")
        env_kwargs = dict(
            use_all_maps=False,
            resample_frequency=910,
            episode_length=self.config.episode_length,
            action_type=self.config.action_type,
            dynamics_model=self.config.dynamics_model,
            reward_conditioning=self.config.reward_conditioning,
            creward_deterministic=self.config.creward_deterministic,
            # Pass the entity index (from manifest) directly so the C env can
            # dispatch creward_ego to the correct agent regardless of how
            # human_agent_idx is interpreted elsewhere.
            ego_entity_idx=human_agent_idx,
            creward_ego=self.config.creward_ego,
            creward_traffic=self.config.creward_traffic,
            max_controlled_agents=-1,
            goal_behavior=self.config.goal_behavior,
            goal_lane_change_prob=self.config.goal_lane_change_prob,
            goal_target_distance=self.config.goal_target_distance,
            collision_behavior=self.config.collision_behavior,
            offroad_behavior=self.config.offroad_behavior,
            termination_mode=self.config.termination_mode,
            map_id=map_id,
            split=self.config.split,
            goal_speed=100,
            goal_radius=2.0,
            human_agent_idx=human_agent_idx,
            control_mode="control_evaluation",
            num_agents=128,
            ini_file=eval_ini,
        )
        # Optional global env-reward overrides (e.g. velocity, timestep) come
        # from config.env_reward_overrides and are forwarded only when set.
        for k, v in (self.config.env_reward_overrides or {}).items():
            env_kwargs[k] = v
        # emit_jerk_ego_obs forwarded only when enabled (drive.py only accepts
        # it once the Python-side plumbing is completed; stripping on False
        # keeps all standard eval paths working).
        if getattr(self.config, "emit_jerk_ego_obs", False):
            env_kwargs["emit_jerk_ego_obs"] = True
        env = Drive(**env_kwargs)

        if env.num_envs == 0 or env.num_agents == 0:
            env.close()
            return None

        return env

    def _render_step(self, output_dir, step_idx, state, action0, planner, axis_limits, map_idx, env_reward=None, collision_info=None, traffic_planner=None, test_agent_idx=0, component_names=None, step_components_raw=None, step_components_weighted=None):
        """Render a single step."""
        viz_config = VizConfig()
        fig, ax = plt.subplots(figsize=(12, 12))

        plot_simulator_state(
            state,
            viz_config=viz_config.__dict__,
            test_agent_idx=test_agent_idx,
            ax=ax,
            axis_limits=axis_limits,
        )

        if planner is not None:
            planner.plot(ax, state, axis_limits=axis_limits)

        if traffic_planner is not None and hasattr(traffic_planner, 'plot'):
            traffic_planner.plot(ax, state, axis_limits=axis_limits)

        # Compute ego speed from state
        ego_speed = 0.0
        active_indices = state.get("active_agent_indices", [])
        entities = state.get("entities", [])
        if test_agent_idx < len(active_indices) and active_indices[test_agent_idx] < len(entities):
            ego = entities[active_indices[test_agent_idx]]
            vx = ego.get("vx", 0.0)
            vy = ego.get("vy", 0.0)
            ego_speed = (vx**2 + vy**2) ** 0.5

        title = f"Map {map_idx} | Step {step_idx + 1} | v={ego_speed:.1f} m/s | a=[{action0[0]:+.2f}, {action0[1]:+.2f}]"
        # Show active planner for hybrid planner
        if hasattr(planner, '_using_pdm'):
            active = "PDM" if planner._using_pdm else "PPO"
            mi = getattr(planner, 'last_epistemic', 0.0)
            title += f" | {active} (MI={mi:.3f})"
        # Show collision info (type + at-fault)
        if collision_info is not None:
            ctype = collision_info["type"]
            fault = collision_info["at_fault"]
            ctype_name = ctype.name if hasattr(ctype, 'name') else str(ctype)
            fault_str = "AT FAULT" if fault else "NO FAULT"
            title += f" | COLLISION: {ctype_name} ({fault_str})"
        ax.set_title(title, fontsize=14)

        # Add reward breakdown text box for all elite proposals
        reward_lines = []
        scores = getattr(planner, '_last_reward_scores', None) if planner is not None else None
        if scores is not None:
            elite_size = getattr(planner.config, 'elite_size', len(scores))
            n_elites = min(elite_size, len(scores))
            costs = planner._last_trajectory_costs
            # Columns: [0]goal_d [1]lane_d [2]speed [3]jerk [4]collision [5]offroad
            reward_lines.append("  #  rew   gd   ld  spd  jrk col ofr")
            for i in range(n_elites):
                s = scores[i]
                r = float(-costs[i]) if costs is not None else 0.0
                reward_lines.append(
                    f" {i:>2d} {r:.2f} {s[0]:.2f} {s[1]:.2f}  {s[2]:.0f}  {s[3]:.2f}  {s[4]:.0f}   {s[5]:.0f}"
                )
        if env_reward is not None:
            reward_lines.append(f"\nEnv reward: {env_reward:+.4f}")
        if component_names and step_components_raw is not None:
            reward_lines.append("Per-component (raw | weighted):")
            for i, name in enumerate(component_names):
                raw_v = float(step_components_raw[i])
                w_v = float(step_components_weighted[i]) if step_components_weighted is not None else 0.0
                reward_lines.append(f"  {name:<12s} {raw_v:+8.3f} | {w_v:+8.4f}")
        if reward_lines:
            text = "\n".join(reward_lines)
            ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=6,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        fig.tight_layout()
        output_path = os.path.join(output_dir, f"step_{step_idx:03d}.png")
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def _create_video(self, image_dir: str, output_path: str, fps: int = 10):
        """Create GIF from images."""
        pattern = os.path.join(image_dir, "step_*.png")
        image_files = sorted(globlib.glob(pattern))

        if not image_files:
            return

        gif_path = output_path.replace(".mp4", ".gif")

        try:
            frames = [Image.open(f) for f in image_files]
            duration = int(1000 / fps)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
            )
        except Exception:
            pass

    def _save_hybrid_summary(self):
        """Aggregate hybrid planner stats across all maps and save summary."""
        import json

        data = self._hybrid_map_data
        n_maps = len(data)
        total_switches = sum(d["num_switches"] for d in data)
        total_pdm = sum(d["pdm_steps"] for d in data)
        total_steps = sum(d["total_steps"] for d in data)
        collisions_total = sum(1 for d in data if d["had_collision"])
        collisions_near_switch = sum(1 for d in data if d["collision_near_switch"])

        ppo_rewards = [d["mean_reward_ppo"] for d in data if d["mean_reward_ppo"] != 0.0]
        pdm_rewards = [d["mean_reward_pdm"] for d in data if d["mean_reward_pdm"] != 0.0]

        summary = {
            "num_maps": n_maps,
            "total_switches": total_switches,
            "mean_switches_per_map": total_switches / n_maps if n_maps else 0,
            "pdm_step_fraction": total_pdm / total_steps if total_steps else 0,
            "mean_reward_ppo_steps": float(np.mean(ppo_rewards)) if ppo_rewards else None,
            "mean_reward_pdm_steps": float(np.mean(pdm_rewards)) if pdm_rewards else None,
            "collisions_total": collisions_total,
            "collisions_near_switch": collisions_near_switch,
            "collision_near_switch_rate": collisions_near_switch / collisions_total if collisions_total else 0,
            # Exclude raw step data from JSON (too large)
            "per_map": [
                {k: v for k, v in d.items() if not k.startswith("_")}
                for d in data
            ],
        }

        path = os.path.join(self.config.output_dir, "hybrid_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        # Aggregate plot across all maps
        self._plot_hybrid_aggregate(data)
        self._plot_hybrid_uncertainty_correlation(data)

        log.info("-" * 60)
        log.info("HYBRID PLANNER DIAGNOSTICS")
        log.info("-" * 60)
        log.info("PDM usage:     %d/%d steps (%.1f%%)", total_pdm, total_steps,
                 100.0 * total_pdm / total_steps if total_steps else 0)
        log.info("Switches:      %d total (%.1f/map)", total_switches,
                 total_switches / n_maps if n_maps else 0)
        if ppo_rewards:
            log.info("Mean rew PPO:  %+.4f", float(np.mean(ppo_rewards)))
        if pdm_rewards:
            log.info("Mean rew PDM:  %+.4f", float(np.mean(pdm_rewards)))
        if collisions_total > 0:
            log.info("Collisions:    %d total, %d near switch (%.0f%%)",
                     collisions_total, collisions_near_switch,
                     100.0 * collisions_near_switch / collisions_total)
        log.info("Summary: %s", path)
        log.info("-" * 60)

    def _plot_hybrid_aggregate(self, data):
        """Create aggregate plots across all maps for hybrid planner analysis.

        2×2 figure:
        1. Re-switch frequency: how many additional switches within N steps?
        2. Switch count vs map outcome (reward), colored by collision
        3. Switch gap distribution + collision rate by gap size
        4. Counterfactual advantage by planner (violin plots)
        """
        output_dir = self.config.output_dir
        N_VALUES = [5, 10, 15, 20]

        # Collect all switch step indices per map (for re-switch analysis)
        all_switch_lists = []
        for d in data:
            if d["switch_steps"]:
                all_switch_lists.append((d["switch_steps"], d["total_steps"]))

        # --- Build 2×2 figure ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        total_switches = sum(d["num_switches"] for d in data)

        # ================================================================
        # Panel 1: Re-switch frequency within N steps (KEPT AS-IS)
        # ================================================================
        ax1 = axes[0, 0]

        if total_switches > 0:
            mean_reswitches = []
            for N in N_VALUES:
                reswitch_counts = []
                for switches, _ in all_switch_lists:
                    for i, s in enumerate(switches):
                        count = 0
                        for j in range(i + 1, len(switches)):
                            if switches[j] - s <= N:
                                count += 1
                            else:
                                break
                        reswitch_counts.append(count)
                mean_reswitches.append(
                    float(np.mean(reswitch_counts)) if reswitch_counts else 0.0
                )

            x_pos = np.arange(len(N_VALUES))
            bars = ax1.bar(
                x_pos, mean_reswitches,
                color=["#66bb6a", "#42a5f5", "#ab47bc", "#ef5350"],
                alpha=0.8, edgecolor="black", linewidth=0.8,
            )
            for bar, val in zip(bars, mean_reswitches):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
                )
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([str(n) for n in N_VALUES])
            ax1.set_xlabel("Window size N (steps)")
            ax1.set_ylabel("Mean additional switches within N steps")
            ax1.set_title("Re-switch frequency after each switch")
            ax1.text(
                0.97, 0.97,
                f"Total switches: {total_switches}\n"
                f"Maps with switches: {len(all_switch_lists)}/{len(data)}",
                transform=ax1.transAxes, fontsize=9, verticalalignment="top",
                horizontalalignment="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
        else:
            ax1.text(0.5, 0.5, "No switches occurred",
                     transform=ax1.transAxes, ha="center", va="center", fontsize=12)
            ax1.set_title("Re-switch frequency after each switch")

        # ================================================================
        # Panel 2: Switch count vs map outcome
        # ================================================================
        ax2 = axes[0, 1]

        switch_counts = []
        total_rewards = []
        had_collisions = []
        for d in data:
            switch_counts.append(d["num_switches"])
            total_rewards.append(sum(d["_step_rewards"]))
            had_collisions.append(d["had_collision"])

        switch_counts = np.array(switch_counts)
        total_rewards = np.array(total_rewards)
        had_collisions = np.array(had_collisions)

        # Scatter: no collision (green circles) then collision (red x)
        no_coll = ~had_collisions
        if no_coll.any():
            ax2.scatter(
                switch_counts[no_coll], total_rewards[no_coll],
                color="#4caf50", marker="o", alpha=0.4, s=18,
                label=f"No collision ({no_coll.sum()})", zorder=3,
            )
        if had_collisions.any():
            ax2.scatter(
                switch_counts[had_collisions], total_rewards[had_collisions],
                color="#ef5350", marker="x", alpha=0.6, s=25,
                label=f"Collision ({had_collisions.sum()})", zorder=4,
            )

        # Regression line
        if len(switch_counts) > 2 and switch_counts.std() > 0:
            coeffs = np.polyfit(switch_counts, total_rewards, 1)
            x_fit = np.linspace(switch_counts.min(), switch_counts.max(), 50)
            ax2.plot(x_fit, np.polyval(coeffs, x_fit), color="black",
                     linewidth=1.5, linestyle="--", alpha=0.7, zorder=5)
            corr = np.corrcoef(switch_counts, total_rewards)[0, 1]

            # Collision rate by switch count bucket
            buckets = [(0, 0), (1, 3), (4, 8), (9, 15), (16, 100)]
            bucket_labels = []
            bucket_coll_rates = []
            for lo, hi in buckets:
                mask = (switch_counts >= lo) & (switch_counts <= hi)
                if mask.sum() > 0:
                    bucket_labels.append(f"{lo}-{hi}" if lo != hi else str(lo))
                    bucket_coll_rates.append(had_collisions[mask].mean())

            coll_text = "Collision rate by #switches:\n"
            for lbl, rate in zip(bucket_labels, bucket_coll_rates):
                coll_text += f"  {lbl:>5s}: {rate:.0%}\n"

            ax2.text(
                0.03, 0.97,
                f"r = {corr:.3f}\n{coll_text}",
                transform=ax2.transAxes, fontsize=8, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )

        ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax2.set_xlabel("Number of switches")
        ax2.set_ylabel("Total map reward")
        ax2.set_title("Switches vs map outcome")
        ax2.legend(fontsize=8, loc="upper right")

        # ================================================================
        # Panel 3: Switch gap distribution + collision rate by min gap
        # ================================================================
        ax3 = axes[1, 0]

        # Compute gaps between consecutive switches
        all_gaps = []
        per_map_min_gap = []  # (min_gap, had_collision) per map with ≥2 switches
        for d in data:
            sw = d["switch_steps"]
            if len(sw) >= 2:
                gaps = [sw[i+1] - sw[i] for i in range(len(sw) - 1)]
                all_gaps.extend(gaps)
                per_map_min_gap.append((min(gaps), d["had_collision"]))

        if all_gaps:
            # Histogram of gap sizes — build bins up to max gap
            max_gap = max(all_gaps)
            bin_edges = [1, 2, 3, 4, 5]
            bin_labels = ["1", "2", "3", "4"]
            if max_gap >= 5:
                bin_edges.append(10)
                bin_labels.append("5-9")
            if max_gap >= 10:
                bin_edges.append(20)
                bin_labels.append("10-19")
            if max_gap >= 20:
                bin_edges.append(max_gap + 1)
                bin_labels.append("20+")
            else:
                bin_edges[-1] = max_gap + 1  # extend last bin
            counts, _ = np.histogram(all_gaps, bins=bin_edges)

            x_pos = np.arange(len(bin_labels))
            ax3.bar(x_pos, counts, color="#42a5f5", alpha=0.7, edgecolor="black",
                    linewidth=0.8, label="Gap count")
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(bin_labels)
            ax3.set_xlabel("Gap between consecutive switches (steps)")
            ax3.set_ylabel("Count", color="#42a5f5")
            ax3.tick_params(axis="y", labelcolor="#42a5f5")

            # Overlay: collision rate for maps whose min gap falls in each bucket
            ax3b = ax3.twinx()
            if per_map_min_gap:
                min_gaps = np.array([g for g, _ in per_map_min_gap])
                min_colls = np.array([c for _, c in per_map_min_gap])
                coll_rates = []
                for i in range(len(bin_edges) - 1):
                    mask = (min_gaps >= bin_edges[i]) & (min_gaps < bin_edges[i+1])
                    if mask.sum() > 0:
                        coll_rates.append(min_colls[mask].mean())
                    else:
                        coll_rates.append(np.nan)
                coll_rates = np.array(coll_rates, dtype=float)
                valid = ~np.isnan(coll_rates)
                ax3b.plot(x_pos[valid], coll_rates[valid], color="#ef5350",
                          linewidth=2, marker="s", markersize=6, zorder=5,
                          label="Collision rate (min gap)")
                ax3b.set_ylabel("Collision rate", color="#ef5350")
                ax3b.tick_params(axis="y", labelcolor="#ef5350")
                ax3b.set_ylim(0, 1)
                ax3b.legend(fontsize=8, loc="upper right")

            # Annotate
            short_gap_pct = sum(1 for g in all_gaps if g <= 3) / len(all_gaps) * 100
            ax3.text(
                0.97, 0.55,
                f"Gaps ≤ 3 steps: {short_gap_pct:.0f}%\n"
                f"Total gaps: {len(all_gaps)}",
                transform=ax3.transAxes, fontsize=9, verticalalignment="top",
                horizontalalignment="right", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
            ax3.set_title("Switch gap distribution + collision rate")
        else:
            ax3.text(0.5, 0.5, "Not enough switches for gap analysis",
                     transform=ax3.transAxes, ha="center", va="center", fontsize=12)
            ax3.set_title("Switch gap distribution")

        # ================================================================
        # Panel 4: Counterfactual advantage by planner (violin)
        # ================================================================
        ax4 = axes[1, 1]

        ppo_deltas = []  # CF_reward - chosen_reward when PPO was chosen
        pdm_deltas = []  # CF_reward - chosen_reward when PDM was chosen
        for d in data:
            cf_rewards = d.get("_step_cf_rewards", [])
            rewards = d["_step_rewards"]
            planners_list = d["_step_planners"]
            n = min(len(rewards), len(planners_list), len(cf_rewards))
            for i in range(n):
                cf_r = cf_rewards[i]
                if cf_r is None:
                    continue
                delta = cf_r - rewards[i]
                if planners_list[i] == "PPO":
                    ppo_deltas.append(delta)
                else:
                    pdm_deltas.append(delta)

        if ppo_deltas or pdm_deltas:
            violin_data = []
            violin_positions = []
            violin_labels = []

            if ppo_deltas:
                violin_data.append(ppo_deltas)
                violin_positions.append(0)
                violin_labels.append(f"PPO chosen\n(n={len(ppo_deltas)})")
            if pdm_deltas:
                violin_data.append(pdm_deltas)
                violin_positions.append(1)
                violin_labels.append(f"PDM chosen\n(n={len(pdm_deltas)})")

            parts = ax4.violinplot(
                violin_data, positions=violin_positions,
                showmeans=True, showmedians=True, showextrema=False,
            )
            colors = ["#4caf50", "#2196f3"]
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(colors[i] if i < len(colors) else "#999")
                body.set_alpha(0.6)
            parts["cmeans"].set_color("black")
            parts["cmedians"].set_color("orange")

            ax4.axhline(y=0, color="gray", linestyle="-", linewidth=1)
            ax4.set_xticks(violin_positions)
            ax4.set_xticklabels(violin_labels, fontsize=9)
            ax4.set_ylabel("Counterfactual advantage\n(CF reward - chosen reward)")
            ax4.set_title("Would the other planner have been better?")

            # Annotations
            texts = []
            if ppo_deltas:
                ppo_arr = np.array(ppo_deltas)
                ppo_better_pct = (ppo_arr < 0).mean() * 100  # negative = PPO was better
                texts.append(
                    f"PPO chosen:\n"
                    f"  mean adv: {ppo_arr.mean():+.4f}\n"
                    f"  PDM better: {(ppo_arr > 0).sum()}/{len(ppo_arr)} ({(ppo_arr > 0).mean()*100:.0f}%)"
                )
            if pdm_deltas:
                pdm_arr = np.array(pdm_deltas)
                texts.append(
                    f"PDM chosen:\n"
                    f"  mean adv: {pdm_arr.mean():+.4f}\n"
                    f"  PPO better: {(pdm_arr > 0).sum()}/{len(pdm_arr)} ({(pdm_arr > 0).mean()*100:.0f}%)"
                )
            ax4.text(
                0.03, 0.97, "\n".join(texts),
                transform=ax4.transAxes, fontsize=8, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
        else:
            ax4.text(0.5, 0.5, "No counterfactual data",
                     transform=ax4.transAxes, ha="center", va="center", fontsize=12)
            ax4.set_title("Counterfactual advantage")

        fig.suptitle(
            f"Hybrid Planner Aggregate  |  {len(data)} maps  |  "
            f"{total_switches} switches  |  "
            f"{sum(1 for d in data if d['had_collision'])} collisions",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "hybrid_aggregate.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("Aggregate plot: %s", plot_path)

    def _plot_hybrid_uncertainty_correlation(self, data):
        """Create uncertainty/value-variance vs reward correlation plots.

        1×3 figure:
        1. Aleatoric uncertainty vs per-step reward
        2. Epistemic uncertainty vs per-step reward
        3. Value prediction variance vs per-step reward
        """
        output_dir = self.config.output_dir

        # Collect all per-step data across maps
        all_aleatorics = []
        all_epistemics = []
        all_value_vars = []
        all_rewards = []
        all_planners = []

        for d in data:
            rewards = d.get("_step_rewards", [])
            aleatorics = d.get("_step_aleatorics", [])
            epistemics = d.get("_step_epistemics", [])
            value_vars = d.get("_step_value_vars", [])
            planners = d.get("_step_planners", [])
            n = min(len(rewards), len(epistemics))
            if not aleatorics or not value_vars:
                continue
            n = min(n, len(aleatorics), len(value_vars))
            for i in range(n):
                all_rewards.append(rewards[i])
                all_aleatorics.append(aleatorics[i])
                all_epistemics.append(epistemics[i])
                all_value_vars.append(value_vars[i])
                all_planners.append(planners[i] if i < len(planners) else "PPO")

        if not all_rewards:
            return

        rewards = np.array(all_rewards)
        aleatorics = np.array(all_aleatorics)
        epistemics = np.array(all_epistemics)
        value_vars = np.array(all_value_vars)
        is_ppo = np.array([p == "PPO" for p in all_planners])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, unc, label in [
            (axes[0], aleatorics, "Aleatoric Uncertainty (entropy)"),
            (axes[1], epistemics, "Epistemic Uncertainty (MI)"),
            (axes[2], value_vars, "Value Prediction Variance"),
        ]:
            # Subsample for readability
            n_pts = len(rewards)
            if n_pts > 5000:
                idx = np.random.default_rng(42).choice(n_pts, 5000, replace=False)
            else:
                idx = np.arange(n_pts)

            # Scatter colored by planner
            ppo_mask = is_ppo[idx]
            pdm_mask = ~ppo_mask
            ax.scatter(unc[idx[ppo_mask]], rewards[idx[ppo_mask]],
                       alpha=0.25, s=10, c="#2ecc71", label="PPO", rasterized=True)
            ax.scatter(unc[idx[pdm_mask]], rewards[idx[pdm_mask]],
                       alpha=0.25, s=10, c="#3498db", label="PDM", rasterized=True)

            # Trend line + correlation (on full data)
            if len(rewards) > 2 and np.std(unc) > 1e-10:
                corr = np.corrcoef(unc, rewards)[0, 1]
                z = np.polyfit(unc, rewards, 1)
                p = np.poly1d(z)
                x_range = np.linspace(unc.min(), unc.max(), 100)
                ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2)
                ax.set_title(f"{label}\nr = {corr:.3f}  (n={len(rewards)})", fontsize=10)
            else:
                ax.set_title(label, fontsize=10)

            ax.set_xlabel(label.split("(")[0].strip(), fontsize=9)
            ax.set_ylabel("Per-step Reward", fontsize=9)
            ax.legend(fontsize=8, loc="best", markerscale=2)

        fig.suptitle("Hybrid Planner: Uncertainty vs Reward Correlation", fontsize=13, fontweight="bold")
        fig.tight_layout()
        plot_path = os.path.join(output_dir, "hybrid_uncertainty_correlation.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info("Uncertainty correlation plot: %s", plot_path)

    def _save_hybrid_diagnostics(self, output_dir, planner, step_rewards,
                                 collision_step, counterfactual_rewards=None):
        """Save per-step CSV and timeline plot for hybrid planner."""
        import csv as _csv

        planners = planner._step_planners
        epistemics = planner._step_epistemics
        threshold = planner.threshold
        self._hybrid_threshold = threshold  # Store for aggregate plot
        n = len(planners)
        has_cf = counterfactual_rewards and any(r is not None for r in counterfactual_rewards)

        # Per-step CSV
        csv_path = os.path.join(output_dir, "hybrid_steps.csv")
        cumulative = 0.0
        header = ["step", "planner", "epistemic", "reward", "cumulative_reward"]
        if has_cf:
            header.append("counterfactual_reward")
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(header)
            for i in range(n):
                r = step_rewards[i] if i < len(step_rewards) else 0.0
                cumulative += r
                row = [i, planners[i], f"{epistemics[i]:.4f}", f"{r:.4f}", f"{cumulative:.4f}"]
                if has_cf:
                    cf_r = counterfactual_rewards[i] if i < len(counterfactual_rewards) and counterfactual_rewards[i] is not None else ""
                    row.append(f"{cf_r:.4f}" if isinstance(cf_r, float) else cf_r)
                w.writerow(row)

        # Collect aggregate data for this map
        switch_steps = []
        for i in range(1, n):
            if planners[i] != planners[i - 1]:
                switch_steps.append(i)

        ppo_rewards = [step_rewards[i] for i in range(min(n, len(step_rewards))) if planners[i] == "PPO"]
        pdm_rewards = [step_rewards[i] for i in range(min(n, len(step_rewards))) if planners[i] == "PDM"]

        collision_near_switch = False
        if collision_step is not None:
            for s in switch_steps:
                if abs(collision_step - s) <= 5:
                    collision_near_switch = True
                    break

        # Build CF rewards list (None → 0.0 for storage)
        cf_list = []
        if has_cf:
            cf_list = [
                counterfactual_rewards[i] if i < len(counterfactual_rewards) and counterfactual_rewards[i] is not None else 0.0
                for i in range(n)
            ]

        self._hybrid_map_data.append({
            "map_dir": output_dir,
            "total_steps": n,
            "pdm_steps": sum(1 for p in planners if p == "PDM"),
            "num_switches": len(switch_steps),
            "switch_steps": switch_steps,
            "mean_reward_ppo": float(np.mean(ppo_rewards)) if ppo_rewards else 0.0,
            "mean_reward_pdm": float(np.mean(pdm_rewards)) if pdm_rewards else 0.0,
            "had_collision": collision_step is not None,
            "collision_step": collision_step,
            "collision_near_switch": collision_near_switch,
            # Raw per-step data for aggregate analysis
            "_step_planners": list(planners),
            "_step_rewards": list(step_rewards[:n]),
            "_step_epistemics": list(epistemics[:n]),
            "_step_aleatorics": list(planner._step_aleatorics[:n]),
            "_step_value_vars": list(planner._step_value_vars[:n]),
            "_step_cf_rewards": cf_list,
        })

        # Timeline plot
        self._plot_hybrid_timeline(
            output_dir, planners, epistemics, step_rewards, threshold, collision_step,
        )

    def _plot_hybrid_timeline(self, output_dir, step_planners, step_epistemics,
                               step_rewards, threshold, collision_step):
        """Generate a two-panel timeline plot for hybrid planner diagnostics."""
        n = len(step_planners)
        if n == 0:
            return

        steps = np.arange(n)
        epistemics = np.array(step_epistemics[:n])
        rewards = np.array(step_rewards[:n]) if len(step_rewards) >= n else np.zeros(n)
        cum_rewards = np.cumsum(rewards)

        # Detect switch points
        switch_steps = []
        for i in range(1, n):
            if step_planners[i] != step_planners[i - 1]:
                switch_steps.append(i)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

        # Helper: shade background by active planner
        def shade_planner_regions(ax):
            i = 0
            while i < n:
                j = i + 1
                while j < n and step_planners[j] == step_planners[i]:
                    j += 1
                color = "#d4edda" if step_planners[i] == "PPO" else "#cce5ff"
                ax.axvspan(i - 0.5, j - 0.5, alpha=0.3, color=color, linewidth=0)
                i = j

        # Top panel: Epistemic uncertainty
        shade_planner_regions(ax1)
        ax1.plot(steps, epistemics, color="black", linewidth=1.2, label="Epistemic (MI)")
        ax1.axhline(y=threshold, color="orange", linestyle="--", linewidth=1.0,
                     label=f"Threshold ({threshold:.2f})")
        for s in switch_steps:
            ax1.axvline(x=s, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        if collision_step is not None and collision_step < n:
            ax1.axvline(x=collision_step, color="red", linewidth=1.5, label="Collision")
        ax1.set_ylabel("Epistemic Uncertainty")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_xlim(-0.5, n - 0.5)

        # Bottom panel: Cumulative reward
        shade_planner_regions(ax2)
        ax2.plot(steps, cum_rewards, color="black", linewidth=1.2, label="Cumulative Reward")
        for s in switch_steps:
            ax2.axvline(x=s, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        if collision_step is not None and collision_step < n:
            ax2.axvline(x=collision_step, color="red", linewidth=1.5)
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_xlabel("Step")
        ax2.set_xlim(-0.5, n - 0.5)

        # Legend for planner colors (shared)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d4edda", alpha=0.5, label="PPO"),
            Patch(facecolor="#cce5ff", alpha=0.5, label="PDM"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right", fontsize=8)

        fig.suptitle(
            f"Hybrid Planner Timeline  |  {sum(1 for p in step_planners if p == 'PDM')}/{n} PDM steps  |  {len(switch_steps)} switches",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "hybrid_timeline.png"), dpi=120)
        plt.close(fig)

    def _compute_dashboard_stats(self):
        """Compute aggregated stats for the dashboard. Returns None if no metrics yet."""
        metrics = self.all_metrics
        n = len(metrics)
        if n == 0:
            return None
        elapsed = time.perf_counter() - self._start_time

        collision_rates = [m.collision_rate for m in metrics if m.collision_rate >= 0]
        offroad_rates = [m.offroad_rate for m in metrics if m.offroad_rate >= 0]
        at_fault_rates = [m.at_fault_collision_rate for m in metrics if m.at_fault_collision_rate >= 0]
        goal_reached = sum(1 for m in metrics if m.goal_reached)
        rewards = [m.total_reward for m in metrics]
        goal_dists = [m.final_goal_distance for m in metrics if m.final_goal_distance < float("inf")]
        plan_times = [m.avg_planning_time_ms for m in metrics if m.num_steps > 0]
        map_times = [m.total_time_s for m in metrics]
        lane_rates = [m.lane_alignment_rate for m in metrics if m.lane_alignment_rate > 0]

        def _stat(vals):
            if not vals:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            a = np.array(vals)
            return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}

        return {
            "n": n,
            "total_maps": self._total_maps,
            "elapsed": elapsed,
            "goal_reached": goal_reached,
            "avg_coll": np.mean(collision_rates) * 100 if collision_rates else 0,
            "avg_offr": np.mean(offroad_rates) * 100 if offroad_rates else 0,
            "avg_fault": np.mean(at_fault_rates) * 100 if at_fault_rates else 0,
            "avg_lane": np.mean(lane_rates) * 100 if lane_rates else 0,
            "reward": _stat(rewards),
            "goal_dist": _stat(goal_dists),
            "plan_time": _stat(plan_times),
            "avg_map_time": np.mean(map_times) if map_times else 0,
            "maps_per_min": n / elapsed * 60 if elapsed > 0 else 0,
            "last": metrics[-1],
        }

    def _log_dashboard(self, stats):
        """Write plain-text dashboard to eval.log."""
        if stats is None:
            return
        n = stats["n"]
        r = stats["reward"]
        gd = stats["goal_dist"]
        pt = stats["plan_time"]
        last = stats["last"]
        elapsed = stats["elapsed"]

        eta_str = ""
        if n < stats["total_maps"]:
            eta = elapsed / n * (stats["total_maps"] - n)
            eta_str = f"  ETA: {int(eta)//60}m {int(eta)%60}s"
        else:
            eta_str = "  Done"

        em, es = int(elapsed) // 60, int(elapsed) % 60
        goal_icon = "+" if last.goal_reached else "-"
        coll_icon = "!" if last.collision_rate > 0 else " "

        log.info("--- %d/%d maps | %dm%ds%s ---", n, stats["total_maps"], em, es, eta_str)
        log.info("  Goal: %d/%d | Coll: %.1f%% | Fault: %.1f%% | Offroad: %.1f%% | Lane: %.1f%%",
                 stats["goal_reached"], n, stats["avg_coll"], stats["avg_fault"],
                 stats["avg_offr"], stats["avg_lane"])
        log.info("  Reward: %+.2f +/- %.2f | GoalDist: %.1fm | Plan: %.0fms | Map: %.1fs | %.1f maps/min",
                 r["mean"], r["std"], gd["mean"], pt["mean"],
                 stats["avg_map_time"], stats["maps_per_min"])
        log.info("  [%s%s] Map %d (ID=%d)  r=%+.2f  gd=%.1fm  t=%.1fs",
                 goal_icon, coll_icon, n - 1, last.map_id,
                 last.total_reward, last.final_goal_distance, last.total_time_s)
        if last.reward_components:
            # Always show all components so the user can see which ones are
            # effectively zero vs small-but-nonzero. Split across two lines to
            # keep the dashboard readable (6 + 5).
            items = list(last.reward_components.items())
            def _fmt(v):
                return f"{v:+.1e}" if (v != 0.0 and abs(v) < 0.01) else f"{v:+.4f}"
            parts = [f"{k}={_fmt(v)}" for k, v in items]
            log.info("          rewards: %s", "  ".join(parts[:6]))
            if len(parts) > 6:
                log.info("                   %s", "  ".join(parts[6:]))
        if last.reward_components_raw:
            # Behavior metrics (α-less, episode-summed). Same layout.
            items_raw = list(last.reward_components_raw.items())
            def _fmt_raw(v):
                return f"{v:+.1e}" if (v != 0.0 and abs(v) < 0.01) else f"{v:+.4f}"
            parts_raw = [f"{k}={_fmt_raw(v)}" for k, v in items_raw]
            log.info("          raw:     %s", "  ".join(parts_raw[:6]))
            if len(parts_raw) > 6:
                log.info("                   %s", "  ".join(parts_raw[6:]))

    def _print_dashboard(self, c1="[cyan]", c2="[white]", b1="[bright_cyan]", b2="[bright_white]"):
        """Print a rich dashboard to terminal and log plain-text version to eval.log."""
        stats = self._compute_dashboard_stats()

        # Log plain-text to eval.log
        self._log_dashboard(stats)

        metrics = self.all_metrics
        n = len(metrics)
        elapsed = time.perf_counter() - self._start_time

        dashboard = Table(box=rich.box.ROUNDED, expand=True, show_header=False, border_style="bright_cyan")

        # Header row
        header = Table(box=None, expand=True, show_header=False)
        header.add_column(justify="left", width=40)
        header.add_column(justify="right", width=40)
        remaining = ""
        if n > 0 and self._total_maps > n:
            eta = elapsed / n * (self._total_maps - n)
            m, s = int(eta) // 60, int(eta) % 60
            remaining = f"{c1}ETA: {b2}{m}{c2}m {b2}{s}{c2}s"
        elif n >= self._total_maps:
            remaining = f"{b2}Done"
        h, rm = int(elapsed) // 3600, int(elapsed) % 3600
        m, s = rm // 60, rm % 60
        uptime = f"{b2}{h}{c2}h {b2}{m}{c2}m {b2}{s}{c2}s" if h else f"{b2}{m}{c2}m {b2}{s}{c2}s"
        header.add_row(
            f"{b1}PufferDrive Eval {b2}{n}{c2}/{b2}{self._total_maps} {c2}maps",
            f"{c1}Elapsed: {uptime}  {remaining}",
        )
        dashboard.add_row(header)

        if stats is None:
            dashboard.add_row(f"{c2}  No maps evaluated yet")
            with self._console.capture() as capture:
                self._console.print(dashboard)
            print("\033[0;0H\033[J" + capture.get())
            return

        # Left: Summary stats
        left = Table(box=None, expand=True)
        left.add_column(f"{c1}Metric", justify="left", width=20)
        left.add_column(f"{c1}Value", justify="right", width=20)

        left.add_row(f"{c2}Goal Reached", f"{b2}{stats['goal_reached']}{c2}/{b2}{n}")
        left.add_row(f"{c2}Collision Rate", f"{b2}{stats['avg_coll']:.1f}{c2}%")
        left.add_row(f"{c2}At-Fault Coll.", f"{b2}{stats['avg_fault']:.1f}{c2}%")
        left.add_row(f"{c2}Offroad Rate", f"{b2}{stats['avg_offr']:.1f}{c2}%")
        left.add_row(f"{c2}Lane Alignment", f"{b2}{stats['avg_lane']:.1f}{c2}%")

        # Right: Performance stats
        r = stats["reward"]
        gd = stats["goal_dist"]
        pt = stats["plan_time"]
        right = Table(box=None, expand=True)
        right.add_column(f"{c1}Metric", justify="left", width=20)
        right.add_column(f"{c1}Value", justify="right", width=20)

        right.add_row(f"{c2}Reward", f"{b2}{r['mean']:+.2f} {c2}+/- {b2}{r['std']:.2f}")
        right.add_row(f"{c2}Goal Distance", f"{b2}{gd['mean']:.1f}{c2}m +/- {b2}{gd['std']:.1f}{c2}m")
        right.add_row(f"{c2}Plan Time", f"{b2}{pt['mean']:.0f}{c2}ms +/- {b2}{pt['std']:.0f}{c2}ms")
        right.add_row(f"{c2}Map Time", f"{b2}{stats['avg_map_time']:.1f}{c2}s")
        right.add_row(f"{c2}Throughput", f"{b2}{stats['maps_per_min']:.1f} {c2}maps/min")

        monitor = Table(box=None, expand=True, pad_edge=False)
        monitor.add_row(left, right)
        dashboard.add_row(monitor)

        # Last map row
        last = stats["last"]
        last_icon = "[green]+[/green]" if last.goal_reached else "[red]-[/red]"
        coll_icon = "[red]![/red]" if last.collision_rate > 0 else " "
        last_row = (
            f"  {last_icon}{coll_icon} Map {n-1} (ID={last.map_id})  "
            f"r={b2}{last.total_reward:+.2f}{c2}  "
            f"gd={b2}{last.final_goal_distance:.1f}{c2}m  "
            f"t={b2}{last.total_time_s:.1f}{c2}s"
        )
        dashboard.add_row(last_row)

        with self._console.capture() as capture:
            self._console.print(dashboard)
        print("\033[0;0H\033[J" + capture.get())

    def _log_summary(self, summary: Dict, total_time: float):
        """Log summary statistics."""
        r = summary["reward"]
        gd = summary["goal_distance"]
        pt = summary["planning_time_ms"]
        mt = summary["map_time_s"]
        cr = summary["collision_rate"]
        afcr = summary["at_fault_collision_rate"]
        orr = summary["offroad_rate"]

        log.info("=" * 60)
        log.info("RESULTS  (%d maps, %.1fs)", summary["num_maps"], total_time)
        log.info("-" * 60)
        log.info("Reward:      %+.2f +/- %.2f  (min=%+.2f, max=%+.2f)", r["mean"], r["std"], r["min"], r["max"])
        log.info("Goal dist:   %.1f +/- %.1fm  (min=%.1fm)", gd["mean"], gd["std"], gd["min"])
        log.info("Goal:        %d/%d reached", summary["goal_reached_count"], summary["num_maps"])
        log.info("Collision:   %.1f%%", cr * 100 if cr >= 0 else -1)
        log.info("At-fault:    %.1f%%", afcr * 100 if afcr >= 0 else -1)
        log.info("Offroad:     %.1f%%", orr * 100 if orr >= 0 else -1)
        log.info("Plan time:   %.1f +/- %.1f ms/step", pt["mean"], pt["std"])
        log.info("Map time:    %.1f +/- %.1fs", mt["mean"], mt["std"])
        log.info("=" * 60)
