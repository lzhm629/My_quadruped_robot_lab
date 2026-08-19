"""Terrain-only curriculum for OpenRobot DreamWaQ stairs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Promote sustained 2.5 m traversals and demote early or short failures."""
    terrain: TerrainImporter = env.scene.terrain
    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())

    asset: Articulation = env.scene[asset_cfg.name]
    displacement = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    distance = torch.norm(displacement, dim=1)
    episode_fraction = env.episode_length_buf[env_ids].float() / float(env.max_episode_length)

    move_up = (distance >= 2.5) & (episode_fraction >= 0.8)
    move_down = ((distance < 1.25) | (episode_fraction < 0.5)) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
