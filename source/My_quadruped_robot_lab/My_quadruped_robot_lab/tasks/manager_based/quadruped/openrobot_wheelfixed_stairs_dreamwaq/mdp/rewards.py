"""OpenRobot-specific rewards for foot-supported DreamWaQ stair locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def normalized_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize torque relative to each actuator's configured effort limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return torch.sum(torch.square(torque / limits), dim=-1)


def normalized_joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize normalized mechanical power without depending on robot scale."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    effort_limits = asset.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    velocity_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return torch.sum(torch.abs((torque / effort_limits) * (velocity / velocity_limits)), dim=-1)


def foot_landing_impact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize only the portion of foot contact force above a landing threshold."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak_force = torch.max(torch.norm(forces, dim=-1), dim=1).values
    return torch.sum(torch.square(torch.clamp(peak_force - threshold, min=0.0) / threshold), dim=-1)


def foot_slip(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize horizontal foot speed only while the corresponding foot is in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    contacts = torch.norm(forces, dim=-1) > 1.0
    velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.sum(torch.square(velocity_xy), dim=-1) * contacts, dim=-1)
