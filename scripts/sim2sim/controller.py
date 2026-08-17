"""Action processing and delayed explicit PD control."""

from __future__ import annotations

from collections import deque

import numpy as np

from config import RobotConfig


class DelayedPDController:
    def __init__(self, cfg: RobotConfig, hip_thigh_delay: int, calf_delay: int):
        if min(hip_thigh_delay, calf_delay) < 0:
            raise ValueError("Actuator delays must be non-negative")
        self.cfg = cfg
        self.delays = np.array([hip_thigh_delay, hip_thigh_delay, calf_delay] * 4, dtype=np.int64)
        self._history: deque[np.ndarray] = deque(maxlen=int(self.delays.max()) + 1)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        for _ in range(int(self.delays.max()) + 1):
            self._history.append(self.cfg.default_joint_pos.copy())

    def action_to_target(self, action: np.ndarray, action_clip: float) -> np.ndarray:
        clipped_action = np.clip(np.asarray(action, dtype=np.float64), -action_clip, action_clip)
        target = self.cfg.default_joint_pos + clipped_action * self.cfg.action_scale
        return np.clip(target, self.cfg.lower_joint_limits, self.cfg.upper_joint_limits)

    def compute(self, target: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        self._history.append(np.asarray(target, dtype=np.float64).copy())
        delayed_target = np.empty_like(target, dtype=np.float64)
        for joint_index, delay in enumerate(self.delays):
            delayed_target[joint_index] = self._history[-int(delay) - 1][joint_index]
        torque = self.cfg.stiffness * (delayed_target - joint_pos) - self.cfg.damping * joint_vel
        return np.clip(torque, -self.cfg.torque_limits, self.cfg.torque_limits)

