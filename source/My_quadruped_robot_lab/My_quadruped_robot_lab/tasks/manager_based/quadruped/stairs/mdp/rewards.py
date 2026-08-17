"""Reward terms specific to stair climbing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_relative_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height relative to the locally scanned stair surface."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    ray_heights = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    terrain_height = torch.mean(ray_heights, dim=1)
    return torch.square(asset.data.root_pos_w[:, 2] - terrain_height - target_height)


def idle_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    velocity_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize failure to advance while a forward command is active."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    commanded = command[:, 0] > command_threshold
    stalled = asset.data.root_lin_vel_b[:, 0] < velocity_threshold
    return (commanded & stalled).float()
