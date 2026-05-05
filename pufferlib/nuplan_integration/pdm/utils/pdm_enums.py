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

from enum import IntEnum


class StateIndex:
    """Index mapping for array representation of ego states."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _VELOCITY_X = 3
    _VELOCITY_Y = 4
    _ACCELERATION_X = 5
    _ACCELERATION_Y = 6
    _STEERING_ANGLE = 7
    _STEERING_RATE = 8
    _ANGULAR_VELOCITY = 9
    _ANGULAR_ACCELERATION = 10

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_")
            and not attribute.startswith("__")
            and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def VELOCITY_X(cls):
        return cls._VELOCITY_X

    @classmethod
    @property
    def VELOCITY_Y(cls):
        return cls._VELOCITY_Y

    @classmethod
    @property
    def ACCELERATION_X(cls):
        return cls._ACCELERATION_X

    @classmethod
    @property
    def ACCELERATION_Y(cls):
        return cls._ACCELERATION_Y

    @classmethod
    @property
    def STEERING_ANGLE(cls):
        return cls._STEERING_ANGLE

    @classmethod
    @property
    def STEERING_RATE(cls):
        return cls._STEERING_RATE

    @classmethod
    @property
    def ANGULAR_VELOCITY(cls):
        return cls._ANGULAR_VELOCITY

    @classmethod
    @property
    def ANGULAR_ACCELERATION(cls):
        return cls._ANGULAR_ACCELERATION

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)

    @classmethod
    @property
    def VELOCITY_2D(cls):
        # assumes velocity X, Y have subsequent indices
        return slice(cls._VELOCITY_X, cls._VELOCITY_Y + 1)

    @classmethod
    @property
    def ACCELERATION_2D(cls):
        # assumes acceleration X, Y have subsequent indices
        return slice(cls._ACCELERATION_X, cls._ACCELERATION_Y + 1)


class SE2Index(IntEnum):
    """Index mapping for state se2 (x,y,θ) arrays."""

    X = 0
    Y = 1
    HEADING = 2


class DynamicStateIndex(IntEnum):
    """Index mapping for dynamic car state (output of controller)."""

    ACCELERATION_X = 0
    STEERING_RATE = 1


class StateIDMIndex(IntEnum):
    """Index mapping for IDM states."""

    PROGRESS = 0
    VELOCITY = 1


class LeadingAgentIndex(IntEnum):
    """Index mapping for leading agent state (for IDM policies)."""

    PROGRESS = 0
    VELOCITY = 1
    LENGTH_REAR = 2


class BBCoordsIndex(IntEnum):
    """Index mapping for corners and center of bounding boxes."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


class EgoAreaIndex(IntEnum):
    """Index mapping for area of ego agent (used in PDMScorer)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2


class MultiMetricIndex(IntEnum):
    """Index mapping multiplicative metrics (used in PDMScorer)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    DRIVING_DIRECTION = 2


class WeightedMetricIndex(IntEnum):
    """Index mapping weighted metrics (used in PDMScorer)."""

    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2
