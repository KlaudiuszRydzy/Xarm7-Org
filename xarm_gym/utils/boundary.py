from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

# from gello.robots.xarm_robot import RobotState


@dataclass(frozen=True)
class PartialRobotState:
    """Represents a partial state of the robot
    inspired by gello RobotState
    """

    cartesian: Optional[Union[List[float], np.ndarray]] = None
    gripper: Optional[Union[float, np.ndarray]] = None
    joints: Optional[Union[List[float], np.ndarray]] = None
    aa: Optional[Union[List[float], np.ndarray]] = None

    def __add__(self, other: "PartialRobotState") -> "PartialRobotState":
        """assumes the second robot state is a delta"""
        raise NotImplementedError("double check for correctness adding aa")
        return PartialRobotState(
            cartesian=self.cartesian + other.cartesian,
            gripper=self.gripper + other.gripper,
            joints=self.joints + other.joints,
            aa=self.aa + other.aa,
        )


class Boundary(ABC):
    @abstractmethod
    def contains(self, state: PartialRobotState) -> bool:
        pass


@dataclass
class CartesianBoundary(Boundary):
    """Represents an axis-aligned bounding box in 3D space."""

    min: PartialRobotState
    max: PartialRobotState

    def post_init(self):
        assert len(self.min.cartesian) == 3
        assert len(self.max.cartesian) == 3

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given point is inside the bounding box.

        Returns:
            bool: True if the point is inside the box, False otherwise.
        """

        return np.logical_and(
            (self.min.cartesian <= state.cartesian),
            (state.cartesian <= self.max.cartesian),
        ).all()


@dataclass
class JointBoundary(Boundary):
    """Represents a joint limit in the robot's configuration space."""

    min: PartialRobotState
    max: PartialRobotState

    def post_init(self):
        assert len(self.min.joins) == 7
        assert len(self.max.joins) == 7

    def contains(self, state: PartialRobotState) -> bool:
        """Checks if the given robot state is within the joint limits.

        Returns:
            bool: True if the state is within the limits, False otherwise.
        """
        if not isinstance(state, PartialRobotState):
            raise ValueError("State must be a RobotState object.")

        return np.logical_and(
            (self.min.joints <= state.joints), (state.joints <= self.max.joints)
        ).all()


@dataclass
class CompositeBoundary(Boundary):
    bounds: List[Boundary]

    def add(self, b: Union[CartesianBoundary, JointBoundary]) -> None:
        self.bounds.append(b)

    def clear(self) -> None:
        self.bounds = []


class ANDBoundary(CompositeBoundary):
    """logical AND operation between multiple boundaries."""

    def contains(self, state: PartialRobotState) -> bool:
        return all([b.contains(state) for b in self.bounds])


class ORBoundary(CompositeBoundary):
    """logical OR operation between multiple boundaries."""

    def contains(self, state: PartialRobotState) -> bool:
        return any([b.contains(state) for b in self.bounds])


@dataclass
class NOTBoundary(Boundary):
    """logical NOT operation on a single boundary."""

    bound: Boundary

    def contains(self, state: PartialRobotState) -> bool:
        return not self.bound.contains(state)
