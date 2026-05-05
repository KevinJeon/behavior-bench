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

from typing import Dict, List

import numpy as np
import numpy.typing as npt
import shapely.vectorized
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import Geometry
from shapely.strtree import STRtree


class PDMOccupancyMap:
    """Occupancy map class of PDM, based on shapely's str-tree."""

    def __init__(
        self,
        tokens: List[str],
        geometries: npt.NDArray[np.object_],
        node_capacity: int = 10,
    ):
        """
        Constructor of PDMOccupancyMap
        :param tokens: list of tracked tokens
        :param geometries: list/array of polygons
        :param node_capacity: max number of child nodes in str-tree, defaults to 10
        """
        assert len(tokens) == len(
            geometries
        ), f"PDMOccupancyMap: Tokens/Geometries ({len(tokens)}/{len(geometries)}) have unequal length!"

        self._tokens: List[str] = tokens
        self._token_to_idx: Dict[str, int] = {
            token: idx for idx, token in enumerate(tokens)
        }

        self._geometries = geometries
        self._node_capacity = node_capacity
        self._str_tree = STRtree(self._geometries, node_capacity)

    def __getitem__(self, token) -> Geometry:
        """
        Retrieves geometry of token.
        :param token: geometry identifier
        :return: Geometry of token
        """
        return self._geometries[self._token_to_idx[token]]

    def __len__(self) -> int:
        """
        Number of geometries in the occupancy map
        :return: int
        """
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        """
        Getter for track tokens in occupancy map
        :return: list of strings
        """
        return self._tokens

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """
        Getter for track tokens in occupancy map
        :return: dictionary of tokens and indices
        """
        return self._token_to_idx

    def intersects(self, geometry: Geometry) -> List[str]:
        """
        Searches for intersecting geometries in the occupancy map
        :param geometry: geometries to query
        :return: list of tokens for intersecting geometries
        """
        indices = self.query(geometry, predicate="intersects")
        return [self._tokens[idx] for idx in indices]

    def query(self, geometry: Geometry, predicate=None):
        """
        Function to directly calls shapely's query function on str-tree
        :param geometry: geometries to query
        :param predicate: see shapely, defaults to None
        :return: query output
        """
        return self._str_tree.query(geometry, predicate=predicate)

    def points_in_polygons(
        self, points: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.bool_]:
        """
        Determines wether input-points are in polygons of the occupancy map
        :param points: input-points
        :return: boolean array of shape (polygons, input-points)
        """
        output = np.zeros((len(self._geometries), len(points)), dtype=bool)
        for i, polygon in enumerate(self._geometries):
            output[i] = shapely.vectorized.contains(polygon, points[:, 0], points[:, 1])

        return output
