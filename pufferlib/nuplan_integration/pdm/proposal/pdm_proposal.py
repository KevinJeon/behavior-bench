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

from dataclasses import dataclass
from typing import List

from shapely.geometry import LineString

from pufferlib.nuplan_integration.pdm.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from pufferlib.nuplan_integration.pdm.utils.pdm_path import PDMPath


@dataclass
class PDMProposal:
    """Dataclass for storing proposal information."""

    proposal_idx: int
    lateral_idx: int
    longitudinal_idx: int
    path: PDMPath

    @property
    def linestring(self) -> LineString:
        """Getter for linestring of proposal's path."""
        return self.path.linestring

    @property
    def length(self):
        """Getter for length [m] of proposal's path."""
        return self.path.length


class PDMProposalManager:
    """Class to store and manage lateral and longitudinal combination of proposals."""

    def __init__(
        self,
        lateral_proposals: List[PDMPath],
        longitudinal_policies: BatchIDMPolicy,
    ):
        """
        Constructor for PDMProposalManager
        :param lateral_proposals: list of path's to follow
        :param longitudinal_policies: IDM policy class (batch-wise)
        """

        self._num_lateral_proposals: int = len(lateral_proposals)
        self._num_longitudinal_proposals: int = longitudinal_policies.num_policies
        self._longitudinal_policies: BatchIDMPolicy = longitudinal_policies

        self._proposals: List[PDMProposal] = []
        proposal_idx = 0

        for lateral_idx in range(self._num_lateral_proposals):
            for longitudinal_idx in range(self._num_longitudinal_proposals):
                self._proposals.append(
                    PDMProposal(
                        proposal_idx=proposal_idx,
                        lateral_idx=lateral_idx,
                        longitudinal_idx=longitudinal_idx,
                        path=lateral_proposals[lateral_idx],
                    )
                )
                proposal_idx += 1

    def __len__(self) -> int:
        """Returns number of proposals (paths x policies)."""
        return len(self._proposals)

    def __getitem__(self, proposal_idx) -> PDMProposal:
        """
        Returns the requested proposal.
        :param proposal_idx: index for each proposal
        :return: PDMProposal dataclass
        """
        return self._proposals[proposal_idx]

    def update(self, speed_limit_mps: float) -> None:
        """
        Updates target velocities of IDM policies with current speed-limit.
        :param speed_limit_mps: current speed-limit [m/s]
        """
        self._longitudinal_policies.update(speed_limit_mps)

    @property
    def num_lateral_proposals(self) -> int:
        return self._num_lateral_proposals

    @property
    def num_longitudinal_proposals(self) -> int:
        return self._longitudinal_policies._num_longitudinal_proposals

    @property
    def max_target_velocity(self) -> float:
        return self._longitudinal_policies.max_target_velocity

    @property
    def longitudinal_policies(self) -> BatchIDMPolicy:
        return self._longitudinal_policies
