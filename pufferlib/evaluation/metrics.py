# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Metrics collection and CSV output."""

import csv
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class MapMetrics:
    """Metrics for a single map evaluation."""

    map_id: int = -1
    total_reward: float = 0.0
    num_steps: int = 0
    goal_reached: bool = False
    collision_rate: float = -1.0  # % of steps with collision
    at_fault_collision_rate: float = -1.0  # % of steps with at-fault collision
    offroad_rate: float = -1.0  # % of steps offroad
    final_goal_distance: float = float("inf")
    planning_time_ms: float = 0.0
    total_time_s: float = 0.0
    # Uncertainty metrics
    mean_aleatoric: float = -1.0
    mean_epistemic: float = -1.0
    mean_value_variance: float = -1.0
    # Additional metrics from env info
    episode_return: float = 0.0
    score: float = 0.0
    lane_alignment_rate: float = 0.0
    mean_speed_mps: float = 0.0
    # Per-scenario benchmark scores in [0, 1] (higher = better). Derived in
    # evaluator.py from raw env-log accumulators; see the plan for formulas.
    score_comfort: float = -1.0
    score_l_align: float = -1.0
    score_l_center: float = -1.0
    # Per-episode sum of each reward component for the ego agent. Keys are
    # entries from Drive.REWARD_COMPONENT_NAMES (collision, offroad, goal,
    # jerk_legacy, velocity, comfort, l_align, l_center, timestep, reverse,
    # speed_limit). Empty when per-component breakdown isn't available.
    reward_components: Dict[str, float] = field(default_factory=dict)
    # Parallel α-less behavior-metric dict. Same keys, but values are the
    # pre-coefficient "behavior term" summed over the episode (e.g. violation
    # counts, seconds of reversing, count of goal events). Lets cross-policy
    # comparison on behavior independent of reward conditioning.
    reward_components_raw: Dict[str, float] = field(default_factory=dict)

    @property
    def avg_planning_time_ms(self) -> float:
        """Average planning time per step."""
        return self.planning_time_ms / max(self.num_steps, 1)


class MetricsWriter:
    """Writes evaluation metrics to CSV files."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_per_map(
        self, metrics: List[MapMetrics], filename: str = "per_map.csv"
    ) -> str:
        """
        Write per-map metrics to CSV.

        Returns:
            Path to the written CSV file
        """
        path = os.path.join(self.output_dir, filename)
        has_uncertainty = any(m.mean_aleatoric >= 0 for m in metrics)
        # Discover the set of reward-component names present across all maps,
        # preserving insertion order from the first map that has them.
        component_names: List[str] = []
        for m in metrics:
            for name in m.reward_components:
                if name not in component_names:
                    component_names.append(name)
        fieldnames = [
            "map_id",
            "total_reward",
            "num_steps",
            "goal_reached",
            "collision_rate",
            "at_fault_collision_rate",
            "offroad_rate",
            "final_goal_distance",
            "avg_planning_time_ms",
            "total_time_s",
            "mean_speed_mps",
            "score_comfort",
            "score_l_align",
            "score_l_center",
        ]
        if has_uncertainty:
            fieldnames.extend(["mean_aleatoric", "mean_epistemic"])
        fieldnames.extend(f"r_{n}" for n in component_names)
        fieldnames.extend(f"raw_{n}" for n in component_names)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics:
                row = {
                    "map_id": m.map_id,
                    "total_reward": f"{m.total_reward:.2f}",
                    "num_steps": m.num_steps,
                    "goal_reached": m.goal_reached,
                    "collision_rate": f"{m.collision_rate*100:.1f}%",
                    "at_fault_collision_rate": f"{m.at_fault_collision_rate*100:.1f}%" if m.at_fault_collision_rate >= 0 else "N/A",
                    "offroad_rate": f"{m.offroad_rate*100:.1f}%",
                    "final_goal_distance": f"{m.final_goal_distance:.2f}",
                    "avg_planning_time_ms": f"{m.avg_planning_time_ms:.1f}",
                    "total_time_s": f"{m.total_time_s:.2f}",
                    "mean_speed_mps": f"{m.mean_speed_mps:.3f}",
                    "score_comfort":  f"{m.score_comfort:.4f}"  if m.score_comfort  >= 0 else "N/A",
                    "score_l_align":  f"{m.score_l_align:.4f}"  if m.score_l_align  >= 0 else "N/A",
                    "score_l_center": f"{m.score_l_center:.4f}" if m.score_l_center >= 0 else "N/A",
                }
                if has_uncertainty:
                    row["mean_aleatoric"] = f"{m.mean_aleatoric:.4f}" if m.mean_aleatoric >= 0 else "N/A"
                    row["mean_epistemic"] = f"{m.mean_epistemic:.4f}" if m.mean_epistemic >= 0 else "N/A"
                for n in component_names:
                    row[f"r_{n}"] = f"{m.reward_components.get(n, 0.0):+.4f}"
                for n in component_names:
                    row[f"raw_{n}"] = f"{m.reward_components_raw.get(n, 0.0):+.4f}"
                writer.writerow(row)

        return path

    def write_summary(
        self, metrics: List[MapMetrics], filename: str = "summary.csv"
    ) -> str:
        """
        Write summary statistics to CSV.

        Returns:
            Path to the written CSV file
        """
        if not metrics:
            return ""

        path = os.path.join(self.output_dir, filename)

        # Compute statistics
        rewards = [m.total_reward for m in metrics]
        steps = [m.num_steps for m in metrics]
        goal_dists = [m.final_goal_distance for m in metrics]
        planning_times = [m.avg_planning_time_ms for m in metrics]
        map_times = [m.total_time_s for m in metrics]
        collision_rates = [m.collision_rate for m in metrics if m.collision_rate >= 0]
        at_fault_collision_rates = [m.at_fault_collision_rate for m in metrics if m.at_fault_collision_rate >= 0]
        offroad_rates = [m.offroad_rate for m in metrics if m.offroad_rate >= 0]

        goal_reached_count = sum(1 for m in metrics if m.goal_reached)
        n = len(metrics)

        stats = [
            ("total_reward", np.mean(rewards), np.std(rewards), np.min(rewards), np.max(rewards)),
            ("num_steps", np.mean(steps), np.std(steps), np.min(steps), np.max(steps)),
            ("final_goal_distance", np.mean(goal_dists), np.std(goal_dists), np.min(goal_dists), np.max(goal_dists)),
            ("avg_planning_time_ms", np.mean(planning_times), np.std(planning_times), np.min(planning_times), np.max(planning_times)),
            ("map_time_s", np.mean(map_times), np.std(map_times), np.min(map_times), np.max(map_times)),
            ("goal_reached_rate", goal_reached_count / n, "", "", ""),
            ("collision_rate", np.mean(collision_rates) if collision_rates else -1, "", "", ""),
            ("at_fault_collision_rate", np.mean(at_fault_collision_rates) if at_fault_collision_rates else -1, "", "", ""),
            ("offroad_rate", np.mean(offroad_rates) if offroad_rates else -1, "", "", ""),
        ]
        for key in ("score_comfort", "score_l_align", "score_l_center"):
            vals = [getattr(m, key) for m in metrics if getattr(m, key) >= 0]
            if vals:
                stats.append((key, np.mean(vals), np.std(vals), np.min(vals), np.max(vals)))
        # Per-component reward stats (ego, sum-over-episode per map).
        component_names: List[str] = []
        for m in metrics:
            for name in m.reward_components:
                if name not in component_names:
                    component_names.append(name)
        for name in component_names:
            vals = [m.reward_components.get(name, 0.0) for m in metrics]
            stats.append((
                f"r_{name}",
                np.mean(vals), np.std(vals), np.min(vals), np.max(vals),
            ))
        # Raw behavior metrics (α-less). Union of keys in case some maps
        # don't have them populated (e.g. pre-feature envs).
        raw_names: List[str] = []
        for m in metrics:
            for name in m.reward_components_raw:
                if name not in raw_names:
                    raw_names.append(name)
        for name in raw_names:
            vals = [m.reward_components_raw.get(name, 0.0) for m in metrics]
            stats.append((
                f"raw_{name}",
                np.mean(vals), np.std(vals), np.min(vals), np.max(vals),
            ))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "min", "max"])
            for name, mean, std, min_val, max_val in stats:
                if isinstance(std, str):
                    writer.writerow([name, f"{mean:.4f}", "", "", ""])
                else:
                    writer.writerow([
                        name,
                        f"{mean:.4f}",
                        f"{std:.4f}",
                        f"{min_val:.4f}",
                        f"{max_val:.4f}",
                    ])

        return path

    def compute_summary_dict(self, metrics: List[MapMetrics]) -> Dict[str, Any]:
        """Compute summary statistics as a dictionary."""
        if not metrics:
            return {"error": "No maps completed"}

        rewards = [m.total_reward for m in metrics]
        steps = [m.num_steps for m in metrics]
        goal_dists = [m.final_goal_distance for m in metrics]
        planning_times = [m.avg_planning_time_ms for m in metrics]
        map_times = [m.total_time_s for m in metrics]
        collision_rates = [m.collision_rate for m in metrics if m.collision_rate >= 0]
        at_fault_collision_rates = [m.at_fault_collision_rate for m in metrics if m.at_fault_collision_rate >= 0]
        offroad_rates = [m.offroad_rate for m in metrics if m.offroad_rate >= 0]
        episode_returns = [m.episode_return for m in metrics if m.episode_return >= 0]
        scores = [m.score for m in metrics if m.score >= 0]
        lane_alignments = [m.lane_alignment_rate for m in metrics if m.lane_alignment_rate >= 0]
        aleatoric_vals = [m.mean_aleatoric for m in metrics if m.mean_aleatoric >= 0]
        epistemic_vals = [m.mean_epistemic for m in metrics if m.mean_epistemic >= 0]

        goal_reached_count = sum(1 for m in metrics if m.goal_reached)

        return {
            "num_maps": len(metrics),
            "reward": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
            },
            "steps": {
                "mean": float(np.mean(steps)),
                "total": int(np.sum(steps)),
            },
            "goal_distance": {
                "mean": float(np.mean(goal_dists)),
                "std": float(np.std(goal_dists)),
                "min": float(np.min(goal_dists)),
            },
            "goal_reached_count": goal_reached_count,
            "goal_reached_rate": goal_reached_count / len(metrics),
            "collision_rate": float(np.mean(collision_rates)) if collision_rates else -1,
            "at_fault_collision_rate": float(np.mean(at_fault_collision_rates)) if at_fault_collision_rates else -1,
            "offroad_rate": float(np.mean(offroad_rates)) if offroad_rates else -1,
            "episode_return": {
                "mean": float(np.mean(episode_returns)) if episode_returns else -1,
                "std": float(np.std(episode_returns)) if episode_returns else -1,
            },
            "score": {
                "mean": float(np.mean(scores)) if scores else -1,
                "std": float(np.std(scores)) if scores else -1,
            },
            "lane_alignment_rate": {
                "mean": float(np.mean(lane_alignments)) if lane_alignments else -1,
            },
            "planning_time_ms": {
                "mean": float(np.mean(planning_times)),
                "std": float(np.std(planning_times)),
            },
            "map_time_s": {
                "mean": float(np.mean(map_times)),
                "std": float(np.std(map_times)),
                "min": float(np.min(map_times)),
                "max": float(np.max(map_times)),
            },
            "aleatoric": {
                "mean": float(np.mean(aleatoric_vals)) if aleatoric_vals else -1,
                "std": float(np.std(aleatoric_vals)) if aleatoric_vals else -1,
            },
            "epistemic": {
                "mean": float(np.mean(epistemic_vals)) if epistemic_vals else -1,
                "std": float(np.std(epistemic_vals)) if epistemic_vals else -1,
            },
        }
