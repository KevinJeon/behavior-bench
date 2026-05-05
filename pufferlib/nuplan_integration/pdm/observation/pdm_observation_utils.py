# Copyright (c) 2026 Copyright holder of the paper "Scaling RL for Autonomous Driving Is Not Enough: A Behavior Benchmark for True Generalization" submitted to NeurIPS2026 for review.
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from TuPlan Garage
# (https://github.com/autonomousvision/tuplan_garage)
# Copyright (c) 2023 Daniel Dauner, Marcel Hallgarten, Andreas Geiger, and Kashyap Chitta.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from shapely.geometry import Polygon

from pufferlib.nuplan_integration.pdm.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)

DRIVABLE_MAP_LAYERS = [
    SemanticMapLayer.ROADBLOCK,
    SemanticMapLayer.ROADBLOCK_CONNECTOR,
    SemanticMapLayer.CARPARK_AREA,
]


def get_drivable_area_map(
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float = 50,
) -> PDMOccupancyMap:

    # query all drivable map elements around ego position
    position: Point2D = ego_state.center.point
    drivable_area = map_api.get_proximal_map_objects(
        position, map_radius, DRIVABLE_MAP_LAYERS
    )

    # collect lane polygons in list, save on-route indices
    drivable_polygons: List[Polygon] = []
    drivable_polygon_ids: List[str] = []

    for type in [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]:
        for roadblock in drivable_area[type]:
            for lane in roadblock.interior_edges:
                drivable_polygons.append(lane.polygon)
                drivable_polygon_ids.append(lane.id)

    for carpark in drivable_area[SemanticMapLayer.CARPARK_AREA]:
        drivable_polygons.append(carpark.polygon)
        drivable_polygon_ids.append(carpark.id)

    # create occupancy map with lane polygons
    drivable_area_map = PDMOccupancyMap(drivable_polygon_ids, drivable_polygons)

    return drivable_area_map
