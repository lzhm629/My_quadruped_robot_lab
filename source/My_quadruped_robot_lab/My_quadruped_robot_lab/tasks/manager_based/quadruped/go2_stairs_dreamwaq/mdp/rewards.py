"""Additional reward terms used by the legacy DreamWaQ task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_relative_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    relative_height = asset.data.root_pos_w[:, 2] - heights.mean(dim=1)
    return torch.square(relative_height - target_height)


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    older = getattr(env, "_dreamwaq_older_action", previous)
    cost = torch.sum(torch.square(action - 2.0 * previous + older), dim=-1)
    env._dreamwaq_older_action = previous.clone()
    return cost


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def power_distribution(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.var(power, dim=-1)


def hip_position_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=-1)


def stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.any(torch.norm(forces[..., :2], dim=-1) > 5.0 * torch.abs(forces[..., 2]), dim=-1).float()


def zero_command_hip_symmetry(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    hip = asset.data.joint_pos[:, asset_cfg.joint_ids]
    symmetry = torch.abs(hip[:, 0] + hip[:, 1]) + torch.abs(hip[:, 2] + hip[:, 3])
    stopped = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=-1) < 0.1
    return symmetry * stopped
