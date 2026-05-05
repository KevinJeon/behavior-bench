#!/usr/bin/env python3
# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Integration tests for all planner/traffic combinations.

Runs each planner configuration on 5 maps with minimal parameters
to verify nothing crashes. Skips weight-dependent planners if weights
are not found.

Usage:
    pytest tests/test_eval_planners.py -v
    python -m pytest tests/test_eval_planners.py -v -x
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.evaluation import Evaluator, EvaluatorConfig
from pufferlib.planning.registry import create_ego_planner, create_traffic_controller
from functools import partial

# ---------------------------------------------------------------------------
# Paths to optional model weights
# ---------------------------------------------------------------------------
SMART_WEIGHTS = os.path.join("experiments", "smart_epoch_030.pt")
PPO_WEIGHTS = os.environ.get("PPO_WEIGHTS_PATH", "")

HAS_SMART_WEIGHTS = os.path.isfile(SMART_WEIGHTS)
HAS_PPO_WEIGHTS = PPO_WEIGHTS != "" and os.path.isfile(PPO_WEIGHTS)
HAS_DATA = "DRIVE_BINARIES_DATA_ROOT" in os.environ

MAP_IDS = [0, 1, 2, 3, 4]
EPISODE_LENGTH = 20  # Short episodes for fast tests


def _build_config(ego_type, traffic_type, ego_kwargs=None, traffic_kwargs=None):
    """Build a minimal config dict for testing."""
    ego_kwargs = ego_kwargs or {}
    traffic_kwargs = traffic_kwargs or {}

    config = {
        "eval": {
            "split": "pufferhard",
            "episode_length": EPISODE_LENGTH,
            "action_type": "continuous",
            "viz": False,
            "planner_viz": False,
            "goal_behavior": 3,
            "termination_mode": 1,
            "collision_behavior": 2,
            "offroad_behavior": 2,
        },
        "planner": {"type": ego_type, ego_type: dict(ego_kwargs)},
        "traffic": {"type": traffic_type, traffic_type: dict(traffic_kwargs)},
    }
    return config


def _run_evaluation(config, map_ids=None):
    """Run a short evaluation and return summary."""
    if map_ids is None:
        map_ids = MAP_IDS

    with tempfile.TemporaryDirectory() as tmpdir:
        eval_cfg = config["eval"]
        evaluator_config = EvaluatorConfig(
            episode_length=int(eval_cfg.get("episode_length", EPISODE_LENGTH)),
            action_type=str(eval_cfg.get("action_type", "continuous")),
            output_dir=tmpdir,
            split=str(eval_cfg.get("split", "pufferhard")),
            viz=False,
            planner_viz=False,
            goal_behavior=int(eval_cfg.get("goal_behavior", 3)),
            termination_mode=int(eval_cfg.get("termination_mode", 1)),
            collision_behavior=int(eval_cfg.get("collision_behavior", 2)),
            offroad_behavior=int(eval_cfg.get("offroad_behavior", 2)),
        )

        ego_factory = partial(create_ego_planner, config)
        traffic_factory = partial(create_traffic_controller, config)

        evaluator = Evaluator(evaluator_config, ego_factory, traffic_factory)
        summary = evaluator.run(map_ids=map_ids)
        return summary


@unittest.skipUnless(HAS_DATA, "DRIVE_BINARIES_DATA_ROOT not set")
class TestPlannerCombinations(unittest.TestCase):
    """Test all planner/traffic combinations on 5 maps."""

    def _assert_valid_summary(self, summary, num_maps=5):
        self.assertEqual(summary["num_maps"], num_maps)
        self.assertGreater(summary["reward"]["mean"], -1000)

    # --- PDM ego ---

    def test_pdm_vs_idm(self):
        """PDM ego planner with IDM traffic."""
        config = _build_config("pdm", "idm", ego_kwargs={"horizon": 5})
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    # --- IDM ego ---

    def test_idm_vs_idm(self):
        """IDM ego planner with IDM traffic."""
        config = _build_config("idm", "idm")
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    # --- Constant velocity ego ---

    def test_cv_vs_idm(self):
        """Constant velocity ego with IDM traffic."""
        config = _build_config("constant_velocity", "idm")
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    # --- PPO ego (requires weights) ---

    @unittest.skipUnless(HAS_PPO_WEIGHTS, "PPO weights not found")
    def test_ppo_vs_idm(self):
        """PPO ego planner with IDM traffic."""
        config = _build_config(
            "ppo", "idm",
            ego_kwargs={"weights_path": PPO_WEIGHTS},
        )
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    # --- SMART planner (requires weights) ---

    @unittest.skipUnless(HAS_SMART_WEIGHTS, "SMART weights not found")
    def test_smart_ego_vs_idm(self):
        """SMART ego planner with IDM traffic."""
        config = _build_config(
            "smart", "idm",
            ego_kwargs={"weights_path": SMART_WEIGHTS},
        )
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    @unittest.skipUnless(HAS_SMART_WEIGHTS, "SMART weights not found")
    def test_pdm_vs_smart_traffic(self):
        """PDM ego with SMART traffic controller."""
        config = _build_config(
            "pdm", "smart",
            ego_kwargs={"horizon": 5},
            traffic_kwargs={"weights_path": SMART_WEIGHTS},
        )
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)

    # --- PPO as traffic (requires weights) ---

    @unittest.skipUnless(HAS_PPO_WEIGHTS, "PPO weights not found")
    def test_pdm_vs_ppo_traffic(self):
        """PDM ego with PPO traffic controller."""
        config = _build_config(
            "pdm", "ppo",
            ego_kwargs={"horizon": 5},
            traffic_kwargs={"weights_path": PPO_WEIGHTS},
        )
        summary = _run_evaluation(config)
        self._assert_valid_summary(summary)


class TestEvalConfig(unittest.TestCase):
    """Test config loading and CLI parsing (no data needed)."""

    def test_load_eval_config_defaults(self):
        """Test that eval config loads with defaults."""
        from pufferlib.planning.registry import load_eval_config
        config = load_eval_config(argv=[])
        self.assertIn("eval", config)
        self.assertIn("planner", config)
        self.assertIn("traffic", config)
        self.assertEqual(config["planner"]["type"], "pdm")
        self.assertEqual(config["traffic"]["type"], "idm")

    def test_load_eval_config_overrides(self):
        """Test that CLI overrides work."""
        from pufferlib.planning.registry import load_eval_config
        config = load_eval_config(argv=[
            "--planner.type", "pdm",
            "--traffic.type", "constant_velocity",
            "--planner.pdm.horizon", "20",
        ])
        self.assertEqual(config["planner"]["type"], "pdm")
        self.assertEqual(config["traffic"]["type"], "constant_velocity")
        self.assertEqual(config["planner"]["pdm"]["horizon"], 20)

    def test_evaluator_config_viz_compat(self):
        """Test that render/save_iteration_gifs backwards compat works."""
        # Old-style: render=True should set viz=True
        cfg = EvaluatorConfig(render=True)
        self.assertTrue(cfg.viz)

        # New-style: viz=True directly
        cfg2 = EvaluatorConfig(viz=True)
        self.assertTrue(cfg2.viz)

        # Planner viz backwards compat
        cfg3 = EvaluatorConfig(save_iteration_gifs=True)
        self.assertTrue(cfg3.planner_viz)


if __name__ == "__main__":
    unittest.main(verbosity=2)
