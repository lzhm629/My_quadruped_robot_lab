"""IsaacLab-compatible policy observation construction."""

from __future__ import annotations

from collections import OrderedDict, deque

import numpy as np


TERM_WIDTHS = OrderedDict(
    gait_clock=2,
    velocity_commands=3,
    base_ang_vel=3,
    projected_gravity=3,
    joint_pos=12,
    joint_vel=12,
    actions=12,
)


class ObservationHistory:
    """Maintain one oldest-to-newest history buffer per IsaacLab observation term."""

    def __init__(self, history_length: int):
        self.history_length = history_length
        self._buffers: dict[str, deque[np.ndarray]] = {}

    def reset(self) -> None:
        self._buffers.clear()

    def update(self, terms: dict[str, np.ndarray]) -> np.ndarray:
        unknown = set(terms) ^ set(TERM_WIDTHS)
        if unknown:
            raise ValueError(f"Observation term mismatch: {sorted(unknown)}")

        flattened = []
        for name, width in TERM_WIDTHS.items():
            value = np.asarray(terms[name], dtype=np.float32).reshape(-1)
            if value.shape != (width,):
                raise ValueError(f"Observation term {name!r} has shape {value.shape}, expected {(width,)}")
            if name not in self._buffers:
                self._buffers[name] = deque(
                    (value.copy() for _ in range(self.history_length)), maxlen=self.history_length
                )
            else:
                self._buffers[name].append(value.copy())
            flattened.append(np.concatenate(tuple(self._buffers[name])))

        observation = np.concatenate(flattened).astype(np.float32, copy=False)
        expected_width = sum(TERM_WIDTHS.values()) * self.history_length
        if observation.shape != (expected_width,):
            raise RuntimeError(f"Built observation shape {observation.shape}, expected {(expected_width,)}")
        return observation


def build_observation_terms(
    *,
    policy_step: int,
    policy_dt: float,
    cycle_time: float,
    command: np.ndarray,
    base_angular_velocity: np.ndarray,
    projected_gravity: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    default_joint_pos: np.ndarray,
    previous_action: np.ndarray,
) -> dict[str, np.ndarray]:
    phase = np.remainder(policy_step * policy_dt, cycle_time) / cycle_time
    angle = 2.0 * np.pi * phase
    return {
        "gait_clock": np.array([np.sin(angle), np.cos(angle)]),
        "velocity_commands": command * np.array([2.0, 2.0, 0.25]),
        "base_ang_vel": base_angular_velocity * 0.25,
        "projected_gravity": projected_gravity,
        "joint_pos": joint_pos - default_joint_pos,
        "joint_vel": joint_vel * 0.05,
        "actions": previous_action,
    }

