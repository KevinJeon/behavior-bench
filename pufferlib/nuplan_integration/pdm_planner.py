# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""PDM (Predictive Driver Model) planner for nuPlan evaluation.

Wraps the PDMClosedPlanner from tuplan_garage (vendored in pufferlib/nuplan_integration/pdm/)
with a Hydra-compatible constructor that accepts flat parameters.

Reference: Dauner et al., "Parting with Misconceptions about Learning-based Vehicle Motion Planning"
           https://github.com/autonomousvision/tuplan_garage
"""

import json
import logging
import os
from typing import List, Optional, Type

import numpy as np

from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from pufferlib.nuplan_integration.pdm.pdm_closed_planner import PDMClosedPlanner
from pufferlib.nuplan_integration.pdm.proposal.batch_idm_policy import BatchIDMPolicy

log = logging.getLogger(__name__)


class PDMNuPlanPlanner(PDMClosedPlanner):
    """PDM planner for nuPlan, wrapping tuplan_garage's PDMClosedPlanner.

    Hydra instantiates this class with flat parameters. The constructor
    builds the TrajectorySampling, BatchIDMPolicy, and lateral_offsets
    objects expected by the parent class.

    Uses the exact same parameters as tuplan_garage for matching performance.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        # Trajectory output: 80 poses at 0.1s = 8s
        trajectory_num_poses: int = 80,
        trajectory_interval: float = 0.1,
        # Proposal horizon: 40 poses at 0.1s = 4s
        proposal_num_poses: int = 40,
        proposal_interval: float = 0.1,
        # IDM policy parameters (tuplan_garage defaults)
        speed_limit_fraction: str = "0.2,0.4,0.6,0.8,1.0",
        fallback_target_velocity: float = 15.0,
        min_gap_to_lead_agent: float = 1.0,
        headway_time: float = 1.5,
        accel_max: float = 1.5,
        decel_max: float = 3.0,
        # Lateral offsets from centerline [m] (tuplan_garage default: [-1.0, 1.0])
        lateral_offsets: str = "-1.0,1.0",
        # Map radius [m]
        map_radius: float = 50.0,
    ):
        trajectory_sampling = TrajectorySampling(
            num_poses=trajectory_num_poses,
            interval_length=trajectory_interval,
        )
        proposal_sampling = TrajectorySampling(
            num_poses=proposal_num_poses,
            interval_length=proposal_interval,
        )

        # Handle Hydra passing numeric types instead of strings (e.g., 1.0 instead of "1.0")
        if isinstance(speed_limit_fraction, (int, float)):
            slf = [float(speed_limit_fraction)]
        else:
            slf = [float(x) for x in str(speed_limit_fraction).split(",") if x.strip()]

        if lateral_offsets is None or (isinstance(lateral_offsets, str) and not lateral_offsets.strip()):
            lat = []
        elif isinstance(lateral_offsets, (int, float)):
            lat = [float(lateral_offsets)]
        else:
            lat = [float(x) for x in str(lateral_offsets).split(",") if x.strip()]

        idm_policies = BatchIDMPolicy(
            speed_limit_fraction=slf,
            fallback_target_velocity=fallback_target_velocity,
            min_gap_to_lead_agent=min_gap_to_lead_agent,
            headway_time=headway_time,
            accel_max=accel_max,
            decel_max=decel_max,
        )

        super().__init__(
            trajectory_sampling=trajectory_sampling,
            proposal_sampling=proposal_sampling,
            idm_policies=idm_policies,
            lateral_offsets=lat,
            map_radius=map_radius,
        )

        # Expose proposal stats for hybrid switching
        self.last_best_score = 1.0
        self.last_score_mean = 1.0  # mean across all proposals (early warning)
        self.last_num_above_0 = 15  # how many proposals are viable

        # Per-step logging (activated by PDM_LOG_DIR env var)
        self._log_dir = os.environ.get("PDM_LOG_DIR", "")
        self._step_log = []
        self._scenario_ref = None  # set during initialize()

    def name(self) -> str:
        return "PDMNuPlan"

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks

    def _get_closed_loop_trajectory(self, current_input):
        """Override to capture the best proposal score before returning trajectory."""
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

        ego_state, observation = current_input.history.current_state

        self._observation.update(
            ego_state, observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        self._update_proposal_manager(ego_state)

        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        )

        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )

        proposal_scores = self._scorer.score_proposals(
            simulated_proposals_array,
            ego_state, self._observation,
            self._centerline, self._route_lane_dict,
            self._drivable_area_map, self._map_api,
        )

        # Store proposal stats for hybrid switching
        self.last_best_score = float(np.max(proposal_scores))
        self.last_score_mean = float(np.mean(proposal_scores))
        self.last_num_above_0 = int((proposal_scores > 0).sum())

        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )
        emergency_brake = trajectory is not None
        self._last_emergency_brake = emergency_brake

        if trajectory is None:
            trajectory = self._generator.generate_trajectory(np.argmax(proposal_scores))

        # Log per-step data if logging is enabled
        if self._log_dir:
            step_data = {
                "best_score": self.last_best_score,
                "score_var": float(np.var(proposal_scores)),
                "score_mean": float(np.mean(proposal_scores)),
                "num_above_0": int((proposal_scores > 0).sum()),
                "num_proposals": len(proposal_scores),
                "emergency_brake": emergency_brake,
            }
            self._step_log.append(step_data)
            self._log_step_to_csv(step_data)

            # Export simulated proposal trajectories + scenario context
            if os.environ.get("PDM_EXPORT_PROPOSALS"):
                self._export_proposals(
                    simulated_proposals_array, proposal_scores,
                    ego_state, observation,
                )

        return trajectory

    def initialize(self, initialization):
        """Override to capture scenario info for logging."""
        super().initialize(initialization)
        self._init_info = initialization

    def _export_proposals(self, proposals_array, proposal_scores,
                          ego_state, observation):
        """Export proposal trajectories + scenario context as .npz."""
        os.makedirs(self._log_dir, exist_ok=True)
        step_idx = len(self._step_log) - 1
        out_path = os.path.join(self._log_dir, f"proposals_step_{step_idx:04d}.npz")

        # Ego state
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_heading = ego_state.rear_axle.heading
        ego_vel = ego_state.dynamic_car_state.speed

        # Other agents: positions, headings, dimensions
        agents_data = []
        for tracked_obj in observation.tracked_objects:
            agents_data.append([
                tracked_obj.center.x,
                tracked_obj.center.y,
                tracked_obj.center.heading,
                tracked_obj.box.length,
                tracked_obj.box.width,
            ])
        agents_array = np.array(agents_data, dtype=np.float32) if agents_data else np.empty((0, 5))

        # Route/centerline (if available)
        centerline_xy = np.empty((0, 2))
        if self._centerline is not None:
            try:
                cl_coords = np.array(self._centerline.linestring.coords)
                centerline_xy = cl_coords[:, :2].astype(np.float32)
            except Exception:
                pass

        # All nearby lane centerlines and boundaries from the map
        lanes_list = []
        lane_boundaries_list = []
        try:
            from nuplan.common.actor_state.state_representation import Point2D
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer
            ego_point = Point2D(ego_x, ego_y)
            nearby = self._map_api.get_proximal_map_objects(
                ego_point, self._map_radius,
                [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
            )
            for layer in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
                for roadblock in nearby[layer]:
                    for lane in roadblock.interior_edges:
                        # Lane centerline (baseline path)
                        try:
                            bl = lane.baseline_path.discrete_path
                            coords = np.array([(p.x, p.y) for p in bl], dtype=np.float64)
                            lanes_list.append(coords)
                        except Exception:
                            pass
                        # Lane polygon (road surface)
                        try:
                            poly = lane.polygon
                            ext = np.array(poly.exterior.coords, dtype=np.float64)[:, :2]
                            lane_boundaries_list.append(ext)
                        except Exception:
                            pass
        except Exception as e:
            log.debug("Lane export failed: %s", e)

        # Pack lanes: concatenate with NaN separators for easy plotting
        if lanes_list:
            sep = np.array([[np.nan, np.nan]], dtype=np.float64)
            lanes_packed = np.concatenate(
                [np.concatenate([l, sep]) for l in lanes_list], axis=0
            )
        else:
            lanes_packed = np.empty((0, 2), dtype=np.float64)

        if lane_boundaries_list:
            sep = np.array([[np.nan, np.nan]], dtype=np.float64)
            boundaries_packed = np.concatenate(
                [np.concatenate([b, sep]) for b in lane_boundaries_list], axis=0
            )
        else:
            boundaries_packed = np.empty((0, 2), dtype=np.float64)

        np.savez_compressed(
            out_path,
            trajectories=proposals_array[:, :, :2],  # [N, H+1, 2]
            scores=proposal_scores,
            best_idx=int(np.argmax(proposal_scores)),
            ego=np.array([ego_x, ego_y, ego_heading, ego_vel], dtype=np.float32),
            agents=agents_array,        # [M, 5] (x, y, heading, length, width)
            centerline=centerline_xy,   # [K, 2]
            lanes=lanes_packed,          # [L, 2] with NaN separators
            lane_boundaries=boundaries_packed,  # [B, 2] with NaN separators
        )

    def _log_step_to_csv(self, step_data):
        """Append one step to a shared CSV file (process safe via flock)."""
        if not self._log_dir:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        csv_path = os.path.join(self._log_dir, "pdm_steps.csv")

        # Use route hash as scenario identifier
        route_ids = self._init_info.route_roadblock_ids if self._init_info else []
        route_hash = str(abs(hash(tuple(route_ids))))[:12]
        step_idx = len(self._step_log) - 1

        row = f"{route_hash},{step_idx},{step_data['best_score']:.6f},{step_data['score_var']:.10f},{step_data['score_mean']:.6f},{step_data['num_above_0']},{step_data['num_proposals']},{step_data['emergency_brake']}\n"

        import fcntl
        with open(csv_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            if f.tell() == 0:
                f.write("route_hash,step,best_score,score_var,score_mean,num_above_0,num_proposals,emergency_brake\n")
            f.write(row)
            fcntl.flock(f, fcntl.LOCK_UN)
