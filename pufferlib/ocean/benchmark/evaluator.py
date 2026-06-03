# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from PufferDrive V2.0
# (https://github.com/Emerge-Lab/PufferDrive/)
# Copyright (c) 2026 PufferDrive, licensed under the MIT license.

"""WOSAC evaluation class for PufferDrive."""

import torch
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
import configparser
import os

import pufferlib
from pufferlib.ocean.benchmark import metrics
from pufferlib.ocean.benchmark import estimators


_METRIC_FIELD_NAMES = [
    "linear_speed",
    "linear_acceleration",
    "angular_speed",
    "angular_acceleration",
    "distance_to_nearest_object",
    "time_to_collision",
    "collision_indication",
    "distance_to_road_edge",
    "offroad_indication",
]


class WOSACEvaluator:
    """Evaluates policys on the Waymo Open Sim Agent Challenge (WOSAC) in PufferDrive. Info and links in the readme."""

    def __init__(self, config: Dict):
        self.config = config
        self.num_steps = 91  # Hardcoded for WOSAC (9.1s at 10Hz)
        self.init_steps = config.get("eval", {}).get("wosac_init_steps", 0)
        self.sim_steps = self.num_steps - self.init_steps
        self.num_rollouts = config.get("eval", {}).get("wosac_num_rollouts", 32)
        self.device = config.get("train", {}).get("device", "cuda")

        wosac_metrics_path = os.path.join(os.path.dirname(__file__), "wosac.ini")
        self.metrics_config = configparser.ConfigParser()
        self.metrics_config.read(wosac_metrics_path)

    def _compute_metametric(self, metrics: pd.Series) -> float:
        metametric = 0.0
        for field_name in _METRIC_FIELD_NAMES:
            likelihood_field_name = "likelihood_" + field_name
            weight = self.metrics_config.getfloat(field_name, "metametric_weight")
            metric_score = metrics[likelihood_field_name]
            metametric += weight * metric_score

        return metametric

    def _get_histogram_params(self, metric_name: str):
        return (
            self.metrics_config.getfloat(metric_name, "histogram.min_val"),
            self.metrics_config.getfloat(metric_name, "histogram.max_val"),
            self.metrics_config.getint(metric_name, "histogram.num_bins"),
            self.metrics_config.getfloat(metric_name, "histogram.additive_smoothing_pseudocount"),
            self.metrics_config.getboolean(metric_name, "independent_timesteps"),
        )

    def collect_ground_truth_trajectories(self, puffer_env):
        """Collect ground truth data for evaluation.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading', 'id'
                        each of shape (num_agents, 1, num_steps) for trajectory data
        """
        return puffer_env.get_ground_truth_trajectories()

    def collect_simulated_trajectories(self, args, puffer_env, policy):
        """Roll out policy in env and collect trajectories.
        Returns:
            trajectories: dict with keys 'x', 'y', 'z', 'heading' each of shape
                (num_agents, num_rollouts, num_steps)
        """

        driver = puffer_env.driver_env
        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        trajectories = {
            "x": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "y": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "z": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "heading": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.float32),
            "id": np.zeros((num_agents, self.num_rollouts, self.sim_steps), dtype=np.int32),
        }

        for rollout_idx in range(self.num_rollouts):
            print(f"\rCollecting rollout {rollout_idx + 1}/{self.num_rollouts}...", end="", flush=True)
            obs, info = puffer_env.reset()
            state = {}
            if args["train"]["use_rnn"]:
                state = dict(
                    lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                    lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
                )

            for time_idx in range(self.sim_steps):
                # Get global state
                agent_state = driver.get_global_agent_state()
                trajectories["x"][:, rollout_idx, time_idx] = agent_state["x"]
                trajectories["y"][:, rollout_idx, time_idx] = agent_state["y"]
                trajectories["z"][:, rollout_idx, time_idx] = agent_state["z"]
                trajectories["heading"][:, rollout_idx, time_idx] = agent_state["heading"]
                trajectories["id"][:, rollout_idx, time_idx] = agent_state["id"]

                # Step policy
                with torch.no_grad():
                    ob_tensor = torch.as_tensor(obs).to(device)
                    logits, value = policy.forward_eval(ob_tensor, state)
                    action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                    action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

                if isinstance(logits, torch.distributions.Normal):
                    action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

                obs, _, _, _, _ = puffer_env.step(action_np)

        return trajectories

    def compute_metrics(
        self,
        ground_truth_trajectories: Dict,
        simulated_trajectories: Dict,
        agent_state: Dict,
        road_edge_polylines: Dict,
        aggregate_results: bool = False,
    ) -> Dict:
        """Compute realism metrics comparing simulated and ground truth trajectories.

        Args:
            ground_truth_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id', 'scenario_id', 'valid']
            simulated_trajectories: Dict with keys ['x', 'y', 'z', 'heading', 'id']
            agent_state: Dict with length and width of agents.
            road_edge_polylines: Dict with keys ['x', 'y', 'lengths', 'scenario_id']

        Note: z-position currently not used.

        Returns:
            Dictionary with scores per scenario_id
        """
        # Ensure the id order matches exactly for simulated and ground truth
        assert np.array_equal(simulated_trajectories["id"][:, 0:1, 0], ground_truth_trajectories["id"]), (
            "Agent IDs don't match between simulated and ground truth trajectories"
        )

        eval_mask = ground_truth_trajectories["id"][:, 0] >= 0

        # Extract trajectories
        sim_x = simulated_trajectories["x"]
        sim_y = simulated_trajectories["y"]
        sim_heading = simulated_trajectories["heading"]
        ref_x = ground_truth_trajectories["x"]
        ref_y = ground_truth_trajectories["y"]
        ref_heading = ground_truth_trajectories["heading"]
        ref_valid = ground_truth_trajectories["valid"]
        agent_length = agent_state["length"]
        agent_width = agent_state["width"]
        scenario_ids = ground_truth_trajectories["scenario_id"]

        # is_vehicle flag for TTC filtering (only vehicles, not pedestrians/cyclists)
        is_vehicle = ground_truth_trajectories.get("is_vehicle")

        # We evaluate the metrics only for the Tracks to Predict.
        eval_sim_x = sim_x[eval_mask]
        eval_sim_y = sim_y[eval_mask]
        eval_sim_heading = sim_heading[eval_mask]
        eval_ref_x = ref_x[eval_mask]
        eval_ref_y = ref_y[eval_mask]
        eval_ref_heading = ref_heading[eval_mask]
        eval_ref_valid = ref_valid[eval_mask]
        eval_agent_length = agent_length[eval_mask]
        eval_agent_width = agent_width[eval_mask]
        eval_scenario_ids = scenario_ids[eval_mask]

        if is_vehicle is not None:
            eval_is_vehicle = is_vehicle[eval_mask]
        else:
            eval_is_vehicle = None

        # Compute features
        # Kinematics-related features
        sim_linear_speed, sim_linear_accel, sim_angular_speed, sim_angular_accel = metrics.compute_kinematic_features(
            eval_sim_x, eval_sim_y, eval_sim_heading
        )

        ref_linear_speed, ref_linear_accel, ref_angular_speed, ref_angular_accel = metrics.compute_kinematic_features(
            eval_ref_x, eval_ref_y, eval_ref_heading
        )

        # Get the log speed (linear and angular) validity. Since this is computed by
        # a delta between steps i-1 and i+1, we verify that both of these are
        # valid (logical and).
        speed_validity, acceleration_validity = metrics.compute_kinematic_validity(ref_valid[eval_mask])

        # Interaction-related features
        sim_signed_distances, sim_collision_per_step, sim_time_to_collision = metrics.compute_interaction_features(
            sim_x, sim_y, sim_heading, scenario_ids, agent_length, agent_width, eval_mask, device=self.device
        )

        ref_signed_distances, ref_collision_per_step, ref_time_to_collision = metrics.compute_interaction_features(
            ref_x,
            ref_y,
            ref_heading,
            scenario_ids,
            agent_length,
            agent_width,
            eval_mask,
            device=self.device,
            valid=ref_valid,
        )

        # Map-based features
        sim_distance_to_road_edge, sim_offroad_per_step = metrics.compute_map_features(
            eval_sim_x,
            eval_sim_y,
            eval_sim_heading,
            eval_scenario_ids,
            eval_agent_length,
            eval_agent_width,
            road_edge_polylines,
            device=self.device,
        )

        ref_distance_to_road_edge, ref_offroad_per_step = metrics.compute_map_features(
            eval_ref_x,
            eval_ref_y,
            eval_ref_heading,
            eval_scenario_ids,
            eval_agent_length,
            eval_agent_width,
            road_edge_polylines,
            device=self.device,
            valid=eval_ref_valid,
        )

        # Compute realism metrics
        # Average Displacement Error (ADE) and minADE
        # Note: This metric is not included in the scoring meta-metric, as per WOSAC rules.
        ade, min_ade = metrics.compute_displacement_error(
            eval_sim_x, eval_sim_y, eval_ref_x, eval_ref_y, eval_ref_valid
        )

        # Log-likelihood metrics
        # Kinematic features log-likelihoods
        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "linear_speed"
        )
        linear_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_speed,
            sim_values=sim_linear_speed,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "linear_acceleration"
        )
        linear_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_linear_accel,
            sim_values=sim_linear_accel,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "angular_speed"
        )
        angular_speed_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_angular_speed,
            sim_values=sim_angular_speed,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "angular_acceleration"
        )
        angular_accel_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_angular_accel,
            sim_values=sim_angular_accel,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "distance_to_nearest_object"
        )
        distance_to_nearest_object_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_signed_distances,
            sim_values=sim_signed_distances,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "time_to_collision"
        )
        time_to_collision_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_time_to_collision,
            sim_values=sim_time_to_collision,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        # Map-based features log-likelihoods
        min_val, max_val, num_bins, additive_smoothing, independent_timesteps = self._get_histogram_params(
            "distance_to_road_edge"
        )
        distance_to_road_edge_log_likelihood = estimators.log_likelihood_estimate_timeseries(
            log_values=ref_distance_to_road_edge,
            sim_values=sim_distance_to_road_edge,
            treat_timesteps_independently=independent_timesteps,
            min_val=min_val,
            max_val=max_val,
            num_bins=num_bins,
            additive_smoothing=additive_smoothing,
            sanity_check=False,
        )

        speed_log_ll = metrics._reduce_average_with_validity(
            linear_speed_log_likelihood,
            speed_validity[:, 0, :],
            axis=1,
        )

        accel_log_ll = metrics._reduce_average_with_validity(
            linear_accel_log_likelihood,
            acceleration_validity[:, 0, :],
            axis=1,
        )

        angular_speed_log_ll = metrics._reduce_average_with_validity(
            angular_speed_log_likelihood,
            speed_validity[:, 0, :],
            axis=1,
        )

        angular_accel_log_ll = metrics._reduce_average_with_validity(
            angular_accel_log_likelihood,
            acceleration_validity[:, 0, :],
            axis=1,
        )

        distance_to_nearest_object_log_ll = metrics._reduce_average_with_validity(
            distance_to_nearest_object_log_likelihood,
            eval_ref_valid[:, 0, :],
            axis=1,
        )

        # TTC is computed only for vehicles (not pedestrians/cyclists)
        if eval_is_vehicle is not None:
            ttc_valid = eval_ref_valid & eval_is_vehicle[..., None].astype(bool)
        else:
            ttc_valid = eval_ref_valid
        time_to_collision_log_ll = metrics._reduce_average_with_validity(
            time_to_collision_log_likelihood,
            ttc_valid[:, 0, :],
            axis=1,
        )

        distance_to_road_edge_log_ll = metrics._reduce_average_with_validity(
            distance_to_road_edge_log_likelihood,
            eval_ref_valid[:, 0, :],
            axis=1,
        )

        # Collision likelihood is computed by aggregating in time. For invalid objects
        # in the logged scenario, we need to filter possible collisions in simulation.
        # `sim_collision_indication` shape: (n_samples, n_objects).

        sim_collision_indication = np.any(np.where(eval_ref_valid, sim_collision_per_step, False), axis=2)
        ref_collision_indication = np.any(np.where(eval_ref_valid, ref_collision_per_step, False), axis=2)

        sim_num_collisions = np.mean(sim_collision_indication, axis=1)
        ref_num_collisions = np.mean(ref_collision_indication, axis=1)

        collision_log_ll = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_collision_indication[:, 0],
            sim_values=sim_collision_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )

        # Offroad likelihood (same pattern as collision)
        sim_offroad_indication = np.any(np.where(eval_ref_valid, sim_offroad_per_step, False), axis=2)
        ref_offroad_indication = np.any(np.where(eval_ref_valid, ref_offroad_per_step, False), axis=2)

        sim_num_offroad = np.mean(sim_offroad_indication, axis=1)
        ref_num_offroad = np.mean(ref_offroad_indication, axis=1)

        offroad_log_ll = estimators.log_likelihood_estimate_scenario_level(
            log_values=ref_offroad_indication[:, 0],
            sim_values=sim_offroad_indication,
            min_val=0.0,
            max_val=1.0,
            num_bins=2,
            use_bernoulli=True,
        )

        # Get agent IDs
        eval_agent_ids = ground_truth_trajectories["id"][eval_mask]

        # Store log-likelihoods in DataFrame (exp applied after scene-level averaging)
        df = pd.DataFrame(
            {
                "agent_id": eval_agent_ids.flatten(),
                "scenario_id": eval_scenario_ids.flatten(),
                "num_collisions_sim": sim_num_collisions.flatten(),
                "num_collisions_ref": ref_num_collisions.flatten(),
                "num_offroad_sim": sim_num_offroad.flatten(),
                "num_offroad_ref": ref_num_offroad.flatten(),
                "ade": ade,
                "min_ade": min_ade,
                "likelihood_linear_speed": speed_log_ll,
                "likelihood_linear_acceleration": accel_log_ll,
                "likelihood_angular_speed": angular_speed_log_ll,
                "likelihood_angular_acceleration": angular_accel_log_ll,
                "likelihood_distance_to_nearest_object": distance_to_nearest_object_log_ll,
                "likelihood_time_to_collision": time_to_collision_log_ll,
                "likelihood_collision_indication": collision_log_ll,
                "likelihood_distance_to_road_edge": distance_to_road_edge_log_ll,
                "likelihood_offroad_indication": offroad_log_ll,
            }
        )

        scene_level_results = df.groupby("scenario_id")[
            [
                "ade",
                "min_ade",
                "num_collisions_sim",
                "num_collisions_ref",
                "num_offroad_sim",
                "num_offroad_ref",
                "likelihood_linear_speed",
                "likelihood_linear_acceleration",
                "likelihood_angular_speed",
                "likelihood_angular_acceleration",
                "likelihood_distance_to_nearest_object",
                "likelihood_time_to_collision",
                "likelihood_collision_indication",
                "likelihood_distance_to_road_edge",
                "likelihood_offroad_indication",
            ]
        ].mean()

        # Convert log-likelihoods to probabilities AFTER scene-level averaging
        likelihood_cols = [c for c in scene_level_results.columns if c.startswith("likelihood_")]
        scene_level_results[likelihood_cols] = np.exp(scene_level_results[likelihood_cols])

        scene_level_results["realism_meta_score"] = scene_level_results.apply(self._compute_metametric, axis=1)
        scene_level_results["num_agents"] = df.groupby("scenario_id").size()
        scene_level_results = scene_level_results[
            ["num_agents"] + [col for col in scene_level_results.columns if col != "num_agents"]
        ]

        if aggregate_results:
            aggregate_metrics = scene_level_results.mean().to_dict()
            aggregate_metrics["total_num_agents"] = scene_level_results["num_agents"].sum()
            # Convert numpy types to Python native types
            return {k: v.item() if hasattr(v, "item") else v for k, v in aggregate_metrics.items()}
        else:
            print("\n Scene-level results:\n")
            print(scene_level_results)

            print(f"\n Overall realism meta score: {scene_level_results['realism_meta_score'].mean():.4f}")
            print(f"\n Overall minADE: {scene_level_results['min_ade'].mean():.4f}")
            print(f"\n Overall ADE: {scene_level_results['ade'].mean():.4f}")

            # print(f"\n Full agent-level results:\n")
            # print(df)
            return scene_level_results

    def _quick_sanity_check(self, gt_trajectories, simulated_trajectories, agent_idx=None, max_agents_to_plot=10):
        if agent_idx is None:
            agent_indices = range(np.clip(simulated_trajectories["x"].shape[0], 1, max_agents_to_plot))

        else:
            agent_indices = [agent_idx]

        for agent_idx in agent_indices:
            valid_mask = gt_trajectories["valid"][agent_idx, 0, :] == 1
            invalid_mask = ~valid_mask

            last_valid_idx = np.where(valid_mask)[0][-1] if valid_mask.any() else 0
            goal_x = gt_trajectories["x"][agent_idx, 0, last_valid_idx]
            goal_y = gt_trajectories["y"][agent_idx, 0, last_valid_idx]
            goal_radius = 2.0  # Note: Hardcoded here; ideally pass from config

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            axs[0].set_title(f"Simulated rollouts (x, y) for agent id: {simulated_trajectories['id'][agent_idx, 0][0]}")

            for i in range(self.num_rollouts):
                # Sample random color for each rollout
                color = plt.cm.tab20(i % 20)
                axs[0].scatter(
                    simulated_trajectories["x"][agent_idx, i, :],
                    simulated_trajectories["y"][agent_idx, i, :],
                    alpha=0.1,
                    color=color,
                )

            axs[1].set_title(
                f"Simulated rollouts (x, y) and GT; agent id: {simulated_trajectories['id'][agent_idx, 0][0]}"
            )

            axs[1].scatter(
                simulated_trajectories["x"][agent_idx, :, valid_mask],
                simulated_trajectories["y"][agent_idx, :, valid_mask],
                color="b",
                alpha=0.1,
                zorder=4,
            )

            axs[1].scatter(
                gt_trajectories["x"][agent_idx, 0, valid_mask],
                gt_trajectories["y"][agent_idx, 0, valid_mask],
                color="g",
                label="Ground truth",
                alpha=0.5,
            )

            axs[1].scatter(
                gt_trajectories["x"][agent_idx, 0, 0],
                gt_trajectories["y"][agent_idx, 0, 0],
                color="darkgreen",
                marker="*",
                s=200,
                label="Log start",
                zorder=5,
                alpha=0.5,
            )
            axs[1].scatter(
                simulated_trajectories["x"][agent_idx, :, 0],
                simulated_trajectories["y"][agent_idx, :, 0],
                color="darkblue",
                marker="*",
                s=200,
                label="Agent start",
                zorder=5,
                alpha=0.5,
            )

            circle = plt.Circle(
                (goal_x, goal_y),
                goal_radius,
                color="g",
                fill=False,
                linewidth=2,
                linestyle="--",
                label=f"Goal radius ({goal_radius}m)",
                zorder=0,
            )
            axs[1].add_patch(circle)

            axs[1].set_xlabel("x")
            axs[1].set_ylabel("y")
            axs[1].legend()
            axs[1].set_aspect("equal", adjustable="datalim")

            axs[2].set_title(f"Heading timeseries for agent ID: {simulated_trajectories['id'][agent_idx, 0][0]}")
            time_steps = list(range(self.sim_steps))
            for r in range(self.num_rollouts):
                axs[2].plot(
                    time_steps,
                    simulated_trajectories["heading"][agent_idx, r, :],
                    color="b",
                    alpha=0.1,
                    label="Simulated" if r == 0 else "",
                )
            axs[2].plot(time_steps, gt_trajectories["heading"][agent_idx, 0, :], color="g", label="Ground truth")

            if invalid_mask.any():
                invalid_timesteps = np.where(invalid_mask)[0]
                axs[2].scatter(
                    invalid_timesteps,
                    gt_trajectories["heading"][agent_idx, 0, invalid_mask],
                    color="r",
                    marker="^",
                    s=100,
                    label="Invalid",
                    zorder=6,
                    edgecolors="darkred",
                    linewidths=1,
                )

            axs[2].set_xlabel("Time step")
            axs[2].legend()

            plt.tight_layout()

            plt.savefig(f"trajectory_comparison_agent_{agent_idx}.png")


class HumanReplayEvaluator:
    """Evaluates policies against human replays in PufferDrive."""

    def __init__(self, config: Dict):
        self.config = config
        self.sim_steps = 91 - self.config["env"]["init_steps"]

    def rollout(self, args, puffer_env, policy):
        """Roll out policy in env with human replays. Store statistics.

        In human replay mode, only the SDC (self-driving car) is controlled by the policy
        while all other agents replay their human trajectories. This tests how compatible
        the policy is with (static) human partners.

        Args:
            args: Config dict with train settings (device, use_rnn, etc.)
            puffer_env: PufferLib environment wrapper
            policy: Trained policy to evaluate

        Returns:
            dict: Aggregated metrics including:
                - avg_collisions_per_agent: Average collisions per agent
                - avg_offroad_per_agent: Average offroad events per agent
        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]

        obs, info = puffer_env.reset()
        state = {}
        if args["train"]["use_rnn"]:
            state = dict(
                lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
                lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
            )

        for time_idx in range(self.sim_steps):
            # Step policy
            with torch.no_grad():
                ob_tensor = torch.as_tensor(obs).to(device)
                logits, value = policy.forward_eval(ob_tensor, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                action_np = action.cpu().numpy().reshape(puffer_env.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action_np = np.clip(action_np, puffer_env.action_space.low, puffer_env.action_space.high)

            obs, rewards, dones, truncs, info_list = puffer_env.step(action_np)

            if len(info_list) > 0:  # Happens at the end of episode
                results = info_list[0]
                return results

class OtherReplayEvaluator:
    """Evaluates policies against other policies replays in PufferDrive."""

    def __init__(self, config: Dict, mode: str = None, exp: str = None, output_dir: str = None):
        self.config = config
        self.exp = exp
        self.mode = mode
        self.output_dir = output_dir
        self.sim_steps = int(config.get("env", {}).get("episode_length", 91))

    @staticmethod
    def _model_tag(path: str) -> str:
        """Short id from checkpoint path (matches legacy [-11:-3] on model_*.pt paths)."""
        stem = os.path.basename(path).replace(".pt", "")
        if stem.startswith("model_"):
            return stem[len("model_") :]
        return stem[-8:] if len(stem) >= 8 else stem

    def _replay_buffer_dir(self) -> str:
        if self.output_dir:
            return self.output_dir
        if self.mode:
            return f"/data/puffer/experiments/{self.mode}/other_action_buffer"
        population = self.config.get("pbt", {}).get("population_path", "")
        return os.path.join(population, "replay")

    @staticmethod
    def _ego_indices_from_reset(puffer_env, infos, num_agents_per_env=None):
        """Resolve global ego agent indices after reset (PBT or legacy offset heuristic)."""
        import numpy as np

        info = infos[0] if isinstance(infos, (list, tuple)) and infos else infos
        if isinstance(info, dict) and "ego_indices" in info:
            return np.asarray(info["ego_indices"], dtype=np.int64)

        ego_indices = []
        if isinstance(info, dict) and "agent_offsets" in info:
            offsets = info["agent_offsets"]
            for i in range(len(offsets) - 1):
                if offsets[i + 1] > offsets[i]:
                    ego_indices.append(int(offsets[i]))
            return np.asarray(ego_indices, dtype=np.int64)

        n = num_agents_per_env or getattr(puffer_env.driver_env, "num_agents", 0)
        offsets = getattr(puffer_env.driver_env, "agent_offsets", None)
        if offsets is not None and len(offsets) > 1:
            for i in range(len(offsets) - 1):
                if offsets[i + 1] > offsets[i]:
                    ego_indices.append(int(offsets[i]))
            return np.asarray(ego_indices, dtype=np.int64)

        if n > 0:
            return np.asarray([0], dtype=np.int64)
        return np.asarray([], dtype=np.int64)

    def save_result(self, path, res):
        import json
        import os
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = [data]
        data.append(res)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def play_reactive(self, args, puffer_env, policy1, policy2):
        """Roll out policy in env with human replays. Store statistics.

        Args:


        Returns:

        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]
        obs, infos = puffer_env.reset()
        ego_indices = self._ego_indices_from_reset(
            puffer_env, infos, num_agents_per_env=args["env"].get("num_agents"),
        ).tolist()
        other_mask = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        other_mask[ego_indices] = False
        state_ego = dict(
        lstm_h=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        lstm_c=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        )
        state_other = dict(
        lstm_h=torch.zeros(obs.shape[0]- len(ego_indices), policy2.hidden_size, device=device),
        lstm_c=torch.zeros(obs.shape[0] - len(ego_indices), policy2.hidden_size, device=device),
        )
        ego_speed = 0
        for time_idx in range(self.sim_steps):
            # Step policy
            with torch.no_grad():
                total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                # ego action
                ob_ego = ob_tensor[ego_indices]
                ego_speed += ob_ego[:, 2].mean()
                logits_ego, value_ego = policy1.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()

                # other action
                ob_other = ob_tensor[other_mask]
                logits_other, value_other = policy2.forward_eval(ob_other, state_other)
                action_other, logprob_other, _ = pufferlib.pytorch.sample_logits(logits_other)
                action_other = action_other.cpu().numpy()
    
            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            if isinstance(logits_other, torch.distributions.Normal):  
                action_other = np.clip(action_other, puffer_env.action_space.low, puffer_env.action_space.high)
            total_actions[ego_indices] = action_ego
            total_actions[other_mask.cpu().numpy()] = action_other
            obs, rewards, dones, truncs, info_list = puffer_env.step(total_actions)

            if len(info_list) > 0:  # Happens at the end of episode
                results = info_list[0]
                ego_speed /= (time_idx + 1)
                results["ego_speed"] = ego_speed.item()
                if args['load_multiple_model_path'][1][-11:-3] == args['load_multiple_model_path'][0][-11:-3]:
                    other_name = "selfplay"
                else:
                    other_name = args['load_multiple_model_path'][1][-11:-3]
                res_dict = {f"{args['load_multiple_model_path'][0][-11:-3]}_vs_{other_name}": results}
                print(res_dict)
                self.save_result(f"/data/puffer/results/{self.exp}/{self.mode}/zeroshot_reactive.json", res_dict)
                return results

    def collect_replay_rollout(self, args, puffer_env, policy1, policy2):
        """One ego-vs-other rollout; returns per-agent actions and env layout metadata.

        Actions are stored with shape (num_agents, horizon, 1) so rollouts stack even when
        PBT ego sampling changes how many agents are controlled vs replayed.
        """
        import numpy as np
        import torch
        import pufferlib

        device = args["train"]["device"]
        obs, infos = puffer_env.reset()
        num_agents = obs.shape[0]
        agent_offsets = np.asarray(puffer_env.agent_offsets.copy())
        map_ids = np.asarray(puffer_env.map_ids.copy())
        print(len(agent_offsets), len(map_ids), obs.shape)
        ego_indices = self._ego_indices_from_reset(
            puffer_env, infos, num_agents_per_env=args["env"].get("num_agents"),
        ).tolist()
        other_mask = torch.ones(num_agents, dtype=torch.bool, device=device)
        other_mask[ego_indices] = False
        other_mask_np = other_mask.cpu().numpy()
        state_ego = dict(
            lstm_h=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
            lstm_c=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        )
        state_other = dict(
            lstm_h=torch.zeros(num_agents - len(ego_indices), policy2.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents - len(ego_indices), policy2.hidden_size, device=device),
        )
        resample_freq = int(args["env"].get("resample_frequency") or 0)
        if resample_freq > 0:
            horizon = resample_freq
        else:
            horizon = int(args["env"].get("episode_length", self.sim_steps))
        all_action_buf = np.zeros((num_agents, horizon, 1), dtype=np.int16)
        ego_indices_np = np.asarray(ego_indices, dtype=np.int64)
        ego_speed = 0.0
        results = None
        num_steps = 0
        for time_idx in range(horizon):
            with torch.no_grad():
                total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                ob_ego = ob_tensor[ego_indices]
                ego_speed += ob_ego[:, 2].mean().item()
                logits_ego, value_ego = policy1.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()

                ob_other = ob_tensor[other_mask]
                logits_other, value_other = policy2.forward_eval(ob_other, state_other)
                action_other, logprob_other, _ = pufferlib.pytorch.sample_logits(logits_other)
                action_other = action_other.cpu().numpy()

            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            if isinstance(logits_other, torch.distributions.Normal):
                action_other = np.clip(action_other, puffer_env.action_space.low, puffer_env.action_space.high)
            total_actions[ego_indices] = action_ego
            total_actions[other_mask_np] = action_other
            all_action_buf[:, time_idx] = total_actions.astype(np.int16)
            obs, rewards, dones, truncs, info_list = puffer_env.step(total_actions)
            num_steps = time_idx + 1

            if len(info_list) > 0:
                results = info_list[0]
            # Per-agent terminals fire throughout the rollout; only stop early
            # when not using a fixed resample horizon.
            if resample_freq == 0 and (np.any(truncs) or np.any(dones)):
                break

        log = self._fetch_episode_log(puffer_env)
        if log:
            results = log
        if results is None:
            raise RuntimeError(
                f"collect_replay_rollout ended without episode info after {num_steps} steps"
            )
        results = dict(results)
        results["ego_speed"] = ego_speed / max(num_steps, 1)
        results["num_steps"] = num_steps
        return all_action_buf, agent_offsets, map_ids, ego_indices_np, results, num_steps

    def save_replay(self, args, puffer_env, policy1, policy2):
        """Roll out once and write ``other_actions.npy`` (single-rollout API)."""
        import numpy as np

        all_action_buf, agent_offsets, map_ids, ego_indices, results, num_steps = (
            self.collect_replay_rollout(args, puffer_env, policy1, policy2)
        )
        other_action_buf = all_action_buf
        buffer_dir = self._replay_buffer_dir()
        os.makedirs(buffer_dir, exist_ok=True)
        ego_tag = self._model_tag(args["load_multiple_model_path"][0])
        out_path = os.path.join(buffer_dir, "other_actions.npy")
        np.save(out_path, other_action_buf)
        results["other_actions_path"] = out_path
        res_dict = {f"{ego_tag}_vs_selfplay": results}
        print(res_dict)
        results_path = os.path.join(os.path.dirname(buffer_dir), "zeroshot_replay.json")
        self.save_result(results_path, res_dict)
        return results
            
    def play_replay(self, args, puffer_env, policy1, policy2):
        """Roll out policy in env with human replays. Store statistics.

        Args:


        Returns:

        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]
        obs, infos = puffer_env.reset()
        ego_indices = self._ego_indices_from_reset(
            puffer_env, infos, num_agents_per_env=args["env"].get("num_agents"),
        ).tolist()
        other_mask = torch.ones(obs.shape[0], dtype=torch.bool, device=device)
        other_mask[ego_indices] = False
        state_ego = dict(
        lstm_h=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        lstm_c=torch.zeros(len(ego_indices), policy1.hidden_size, device=device),
        )
        other_path = os.path.join(self._replay_buffer_dir(), "other_actions.npy")
        other_action_npy = np.load(other_path)
        other_mask_np = other_mask.cpu().numpy()
        ego_speed = 0
        for time_idx in range(self.sim_steps):
            # Step policy
            with torch.no_grad():
                total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                # ego action
                ob_ego = ob_tensor[ego_indices]
                ego_speed += ob_ego[:, 2].mean()
                logits_ego, value_ego = policy1.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()
            
            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(action_ego, puffer_env.action_space.low, puffer_env.action_space.high)

            if other_action_npy.shape[0] == num_agents:
                total_actions[other_mask_np] = other_action_npy[other_mask_np, time_idx]
            else:
                # Legacy: (num_other, horizon, 1) from older collectors
                total_actions[other_mask_np] = other_action_npy[:, time_idx]
            total_actions[ego_indices] = action_ego
            obs, rewards, dones, truncs, info_list = puffer_env.step(total_actions)

            if len(info_list) > 0:  # Happens at the end of episode
                results = info_list[0]
                ego_speed /= (time_idx + 1)
                results["ego_speed"] = ego_speed.item()
                res_dict = {f"{args['load_multiple_model_path'][0][-11:-3]}_vs_{args['load_multiple_model_path'][1][-11:-3]}": results}
                print(res_dict)
                self.save_result(f"/data/puffer/results/{self.mode}/zeroshot.json", res_dict)
                return results

    def collect_rollouts(self, args, puffer_env, policies):
        import numpy as np
        import torch
        import pufferlib
        from tqdm import tqdm

        num_agents = puffer_env.observation_space.shape[0]
        device = args["train"]["device"]
        obs, info_list = puffer_env.reset()
        map_ids = puffer_env.map_ids.copy()
        agent_offsets = puffer_env.agent_offsets.copy()
        print(len(agent_offsets), len(map_ids), obs.shape)
        pool = np.arange(obs.shape[0], dtype=np.int64)
        pool = np.random.permutation(pool)
        base, rem = divmod(obs.shape[0], len(policies))
        counts = [base + (i < rem) for i in range(len(policies))]
        other_indices = []
        p = 0
        states = []
        for i, count in enumerate(counts):
            indices = pool[p:p+count]
            p += count
            other_indices.append(indices)
            states.append(dict(
                lstm_h=torch.zeros(count, policies[i].hidden_size, device=device),
                lstm_c=torch.zeros(count, policies[i].hidden_size, device=device),
            ))
        horizon = int(args["env"]["resample_frequency"])
        other_action_buf = np.zeros((obs.shape[0], horizon, 1))
        replay_dir = self._replay_buffer_dir()
        os.makedirs(replay_dir, exist_ok=True)
        total_results = []
        ar = np.arange(num_agents, dtype=np.int64)
        other_masks = [np.isin(ar, other_indices[policy_idx]) for policy_idx in range(len(policies))]
        total_actions = np.zeros((obs.shape[0], 1), dtype=np.int64)
        with torch.inference_mode():
            for time_idx in tqdm(
                range(args["env"]["resample_frequency"]),
                desc="collect_rollouts",
                leave=False,
            ):
                total_actions.fill(0)
                ob_tensor = torch.as_tensor(obs, device=device)

                for policy_idx, policy in enumerate(policies):
                    other_mask = other_masks[policy_idx]
                    ob_other = ob_tensor[other_mask]
                    logits_other, value_other = policy.forward_eval(ob_other, states[policy_idx])
                    action_other, logprob_other, _ = pufferlib.pytorch.sample_logits(logits_other)
                    action_other = action_other.cpu().numpy()
                    if isinstance(logits_other, torch.distributions.Normal):
                        action_other = np.clip(
                            action_other, puffer_env.action_space.low, puffer_env.action_space.high
                        )
                    other_action_buf[other_mask, time_idx] = action_other
                    total_actions[other_mask] = action_other

                obs, rewards, dones, truncs, info_list = puffer_env.step(total_actions)
                if len(info_list) > 0:  # Happens at the end of episode
                    results = info_list[0]
                    total_results.append(results)
        df = pd.DataFrame(total_results)
        mean_per_key = df.mean(numeric_only=True).to_dict()
        print(mean_per_key)
        return other_action_buf, agent_offsets, map_ids

    @staticmethod
    def _episode_metrics(results):
        import pandas as pd

        if isinstance(results, list):
            if not results:
                return {}
            df = pd.DataFrame(results)
            return df.mean(numeric_only=True).to_dict()
        return dict(results)

    @staticmethod
    def _fetch_episode_log(puffer_env):
        from pufferlib.ocean.drive import binding

        driver = getattr(puffer_env, "driver_env", puffer_env)
        if hasattr(driver, "get_episode_log"):
            log = driver.get_episode_log()
            if log:
                return log
        try:
            return binding.vec_log(driver.c_envs, driver.num_agents)
        except Exception:
            return None

    @staticmethod
    def log_map_agent_layout(agent_offsets, map_ids, actions, label="layout"):
        import numpy as np

        offsets = np.asarray(agent_offsets, dtype=np.int32)
        map_ids = np.asarray(map_ids, dtype=np.int32).ravel()
        actions = np.asarray(actions)
        lines = [f"{label}: total_agents={actions.shape[0]}"]
        for i in range(len(map_ids)):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            seg = actions[start:end]
            other_rows = int(np.any(seg != 0, axis=(1, 2)).sum())
            lines.append(
                f"  map_slot[{i:3d}] map_id={int(map_ids[i]):3d} "
                f"agent_idx=[{start:4d}:{end:4d}) count={end - start} saved_other_rows={other_rows}"
            )
        print("\n".join(lines))

    @staticmethod
    def _apply_fixed_ego_indices(puffer_env, ego_indices):
        import numpy as np

        driver = puffer_env.driver_env
        ego_indices = np.asarray(ego_indices, dtype=np.int64)
        driver.ego_indices = ego_indices
        other_mask = np.ones(driver.num_agents, dtype=bool)
        other_mask[ego_indices] = False
        driver.other_indices = np.flatnonzero(other_mask).astype(np.int64)

    def replay_rollouts(
        self,
        args,
        puffer_env,
        loaded_actions,
        policy1=None,
        ego_indices=None,
        num_steps=None,
        exact_actions=True,
    ):
        """Replay saved trajectories.

        exact_actions=True replays the full saved action tensor (ego+other).
        exact_actions=False uses policy1 for ego and saved actions for others.
        """
        import numpy as np
        import torch
        import pufferlib

        num_agents = puffer_env.observation_space.shape[0]
        if loaded_actions.shape[0] != num_agents:
            raise ValueError(
                f"Replay action agents {loaded_actions.shape[0]} != env agents {num_agents}"
            )
        obs, infos = puffer_env.reset()
        if ego_indices is not None:
            self._apply_fixed_ego_indices(puffer_env, ego_indices)

        horizon = int(num_steps or loaded_actions.shape[1])
        horizon = min(horizon, int(loaded_actions.shape[1]))
        results = None

        if exact_actions or policy1 is None:
            for time_idx in range(horizon):
                obs, rewards, dones, truncs, info_list = puffer_env.step(
                    loaded_actions[:, time_idx].astype(np.int64)
                )
                if len(info_list) > 0:
                    results = info_list[0]
            log = self._fetch_episode_log(puffer_env)
            if log:
                results = log
            if results is None:
                raise RuntimeError(
                    f"replay_rollouts ended without episode info after {horizon} steps"
                )
            results = dict(results)
            results["num_steps"] = horizon
            return results

        device = args["train"]["device"]
        ego_idx = self._ego_indices_from_reset(
            puffer_env, infos, num_agents_per_env=args["env"].get("num_agents"),
        ).tolist()
        other_mask = torch.ones(num_agents, dtype=torch.bool, device=device)
        other_mask[ego_idx] = False
        other_mask_np = other_mask.cpu().numpy()
        state_ego = dict(
            lstm_h=torch.zeros(len(ego_idx), policy1.hidden_size, device=device),
            lstm_c=torch.zeros(len(ego_idx), policy1.hidden_size, device=device),
        )
        ego_speed = 0.0
        for time_idx in range(horizon):
            with torch.no_grad():
                total_actions = np.zeros((num_agents, 1), dtype=np.int64)
                ob_tensor = torch.as_tensor(obs).to(device)
                ob_ego = ob_tensor[ego_idx]
                ego_speed += ob_ego[:, 2].mean().item()
                logits_ego, value_ego = policy1.forward_eval(ob_ego, state_ego)
                action_ego, logprob_ego, _ = pufferlib.pytorch.sample_logits(logits_ego)
                action_ego = action_ego.cpu().numpy()
            if isinstance(logits_ego, torch.distributions.Normal):
                action_ego = np.clip(
                    action_ego, puffer_env.action_space.low, puffer_env.action_space.high
                )
            total_actions[other_mask_np] = loaded_actions[other_mask_np, time_idx]
            total_actions[ego_idx] = action_ego
            obs, rewards, dones, truncs, info_list = puffer_env.step(total_actions)
            if len(info_list) > 0:
                results = info_list[0]
        log = self._fetch_episode_log(puffer_env)
        if log:
            results = log
        if results is None:
            raise RuntimeError(
                f"replay_rollouts ended without episode info after {horizon} steps"
            )
        results = dict(results)
        results["ego_speed"] = ego_speed / max(horizon, 1)
        results["num_steps"] = horizon
        return results