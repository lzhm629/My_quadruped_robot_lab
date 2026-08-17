"""Deployment constants for the OpenRobot wheel-fixed trot policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "source"
    / "My_quadruped_robot_lab"
    / "My_quadruped_robot_lab"
    / "assets"
    / "data"
    / "openrobot_wheelfixed"
    / "mjcf"
    / "openrobot_wheelfixed.xml"
)
DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "logs"
    / "rsl_rl"
    / "openrobot_wheelfixed_trot"
    / "2026-08-11_17-34-38"
    / "exported"
    / "policy.pt"
)

JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


@dataclass(frozen=True)
class RobotConfig:
    default_joint_pos: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.7, -0.3, 0.0, -0.7, 0.3, 0.0, 0.7, -0.3, 0.0, -0.7, 0.3], dtype=np.float64
        )
    )
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.10, 0.20] * 4, dtype=np.float64)
    )
    lower_joint_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.2, 0.0, -0.6, -0.2, -1.5, -1.6, -0.2, 0.0, -0.6, -0.2, -1.5, -1.6],
            dtype=np.float64,
        )
    )
    upper_joint_limits: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.2, 1.5, 1.6, 0.2, 0.0, 0.6, 0.2, 1.5, 1.6, 0.2, 0.0, 0.6],
            dtype=np.float64,
        )
    )
    stiffness: np.ndarray = field(
        default_factory=lambda: np.array([1500.0, 1500.0, 1500.0] * 4, dtype=np.float64)
    )
    damping: np.ndarray = field(
        default_factory=lambda: np.array([40.0, 40.0, 60.0] * 4, dtype=np.float64)
    )
    torque_limits: np.ndarray = field(
        default_factory=lambda: np.array([603.0, 603.0, 900.0] * 4, dtype=np.float64)
    )


@dataclass(frozen=True)
class PolicyConfig:
    physics_dt: float = 0.005
    decimation: int = 4
    cycle_time: float = 0.8
    history_length: int = 10
    action_clip: float = 10.0

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

