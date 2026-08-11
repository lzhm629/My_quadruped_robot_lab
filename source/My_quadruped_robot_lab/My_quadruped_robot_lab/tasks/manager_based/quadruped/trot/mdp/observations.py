"""Trot-specific observation terms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_clock(env: ManagerBasedRLEnv, cycle_time: float = 0.5) -> torch.Tensor:
    """Return the sine/cosine clock used by the legacy trot policy."""
    phase = torch.remainder(env.episode_length_buf * env.step_dt, cycle_time) / cycle_time
    return torch.stack((torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)), dim=-1)


def foot_contact_state(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0
) -> torch.Tensor:
    """Return binary foot contacts in the order requested by ``sensor_cfg``."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return (force > threshold).to(dtype=torch.float32)


def action_rate_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return one privileged scalar describing the action-rate magnitude."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.linalg.vector_norm(delta, dim=-1, keepdim=True)


def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return base height as a column tensor."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]

