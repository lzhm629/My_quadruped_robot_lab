"""DreamWaQ observation and privileged-state terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def terrain_height_profile(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the 187 legacy base-relative terrain samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    heights = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.clamp(asset.data.root_pos_w[:, 2:3] - 0.5 - heights, -1.0, 1.0) * 5.0


def material_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return one mean static-friction value per environment."""
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().to(asset.device)
    return materials[..., 0].mean(dim=1, keepdim=True)


def material_restitution(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return one mean restitution value per environment."""
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().to(asset.device)
    return materials[..., 2].mean(dim=1, keepdim=True)


def stiffness_multipliers(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    default = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return asset.data.joint_stiffness[:, asset_cfg.joint_ids] / default


def damping_multipliers(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    default = asset.data.default_joint_damping[:, asset_cfg.joint_ids].clamp_min(1.0e-6)
    return asset.data.joint_damping[:, asset_cfg.joint_ids] / default
