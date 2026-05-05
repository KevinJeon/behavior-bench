# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

"""Hybrid ego controller that switches between perfect tracking and two-stage.

When the planner signals PPO mode, uses perfect tracking (PPO outputs a full
trajectory that should be followed exactly). When PDM mode, uses the two-stage
controller (LQR tracker + kinematic bicycle model) which PDM is designed for.

Communication: The planner sets a module-level flag `USE_PDM` that this controller
reads at each step. This avoids needing a direct reference between planner and
controller (which nuPlan's architecture doesn't support).
"""

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel
from nuplan.planning.simulation.controller.tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

# Module-level flag set by PDMPPONuPlanPlanner at each step
USE_PDM = False


class HybridEgoController(AbstractEgoController):
    """Switches between perfect tracking (PPO) and two-stage (PDM) dynamically.

    Reads the module-level `USE_PDM` flag to decide which mode to use.
    - USE_PDM=True:  two-stage controller (LQR tracker + kinematic bicycle)
    - USE_PDM=False: perfect tracking (follow trajectory exactly)
    """

    def __init__(
        self,
        scenario: AbstractScenario,
        tracker: AbstractTracker,
        motion_model: AbstractMotionModel,
    ):
        self._scenario = scenario
        self._tracker = tracker
        self._motion_model = motion_model
        self._current_state = None

    def reset(self) -> None:
        self._current_state = None

    def get_state(self) -> EgoState:
        if self._current_state is None:
            self._current_state = self._scenario.initial_ego_state
        return self._current_state

    def update_state(
        self,
        current_iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        ego_state: EgoState,
        trajectory: AbstractTrajectory,
    ) -> None:
        if USE_PDM:
            # Two-stage: LQR tracker + kinematic bicycle (what PDM expects)
            sampling_time = next_iteration.time_point - current_iteration.time_point
            dynamic_state = self._tracker.track_trajectory(
                current_iteration, next_iteration, ego_state, trajectory
            )
            self._current_state = self._motion_model.propagate_state(
                state=ego_state, ideal_dynamic_state=dynamic_state, sampling_time=sampling_time
            )
        else:
            # Perfect tracking: follow PPO trajectory exactly
            self._current_state = trajectory.get_state_at_time(next_iteration.time_point)
            assert self._current_state is not None
            if self._current_state.dynamic_car_state.speed >= 50:
                raise RuntimeError(
                    f"Velocity too high: {self._current_state.dynamic_car_state.speed}"
                )
