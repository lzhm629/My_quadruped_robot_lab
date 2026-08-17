"""Observation terms for the OpenRobot stair-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def base_euler_xyz(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Return base roll, pitch, and yaw in the legacy stairs observation order."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack((roll, pitch, yaw), dim=-1)


def terrain_heights(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner")
) -> torch.Tensor:
    """Return absolute world-frame terrain heights for the privileged critic input."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_heights = sensor.data.ray_hits_w[..., 2]
    return torch.nan_to_num(ray_heights, nan=0.0, posinf=0.0, neginf=0.0)
