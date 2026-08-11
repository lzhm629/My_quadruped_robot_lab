"""Rewards that preserve the intent of the legacy Go2 trot task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    return torch.remainder(env.episode_length_buf * env.step_dt, cycle_time) / cycle_time


def _moving(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.1) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command, dim=-1) > threshold


def trot_gait_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    cycle_time: float = 0.5,
    threshold: float = 5.0,
) -> torch.Tensor:
    """Reward alternating diagonal contacts: FL/RR then FR/RL."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    contact = force > threshold
    first_diagonal_stance = _phase(env, cycle_time) < 0.5
    expected = torch.stack(
        (first_diagonal_stance, ~first_diagonal_stance, ~first_diagonal_stance, first_diagonal_stance), dim=-1
    )
    return torch.all(contact == expected, dim=-1).float() * _moving(env, command_name)


def feet_clearance_trot(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    cycle_time: float = 0.5,
    target_height: float = 0.06,
) -> torch.Tensor:
    """Track a sinusoidal swing-foot clearance for the two diagonal pairs."""
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - 0.02
    phase = _phase(env, cycle_time)
    target = torch.abs(torch.sin(2.0 * math.pi * phase)) * target_height
    first_diagonal_stance = phase < 0.5
    expected_stance = torch.stack(
        (first_diagonal_stance, ~first_diagonal_stance, ~first_diagonal_stance, first_diagonal_stance), dim=-1
    )
    swing = (~expected_stance).float()
    error = torch.sum(torch.abs(heights - target.unsqueeze(-1)) * swing, dim=-1)
    return torch.exp(-10.0 * error) * _moving(env, command_name)


def all_feet_contact_without_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 0.1,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward keeping all feet grounded for a near-zero velocity command."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    all_contact = torch.all(force > contact_threshold, dim=-1)
    command = env.command_manager.get_command(command_name)
    stationary = torch.linalg.vector_norm(command, dim=-1) < command_threshold
    return (all_contact & stationary).float()


def hip_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize ab/adduction joints moving away from the neutral pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=-1)

