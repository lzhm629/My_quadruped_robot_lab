"""Discrete one-way stair terrains for the wheel-fixed OpenRobot."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass


def discrete_stairs_terrain(
    difficulty: float, cfg: "DiscreteStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build a straight staircase that rises along the positive x-axis."""
    level = min(int(difficulty * len(cfg.step_depths)), len(cfg.step_depths) - 1)
    step_depth = cfg.step_depths[level]
    terrain_length, terrain_width = cfg.size
    stair_start = cfg.start_platform_length
    stair_end = stair_start + cfg.num_steps * step_depth
    if stair_end >= terrain_length:
        raise ValueError("The staircase does not fit inside the configured terrain length.")

    ground = trimesh.creation.box(
        (terrain_length, terrain_width, cfg.foundation_thickness),
        trimesh.transformations.translation_matrix(
            (terrain_length / 2.0, terrain_width / 2.0, -cfg.foundation_thickness / 2.0)
        ),
    )
    ground.visual.vertex_colors = (0, 0, 0, 255)
    meshes = [ground]
    for step_index in range(cfg.num_steps):
        step_start = stair_start + step_index * step_depth
        step_height = (step_index + 1) * cfg.step_height
        step_length = terrain_length - step_start
        step = trimesh.creation.box(
            (step_length, terrain_width, step_height),
            trimesh.transformations.translation_matrix(
                (step_start + step_length / 2.0, terrain_width / 2.0, step_height / 2.0)
            ),
        )
        step.visual.vertex_colors = (255, 255, 255, 255)
        meshes.append(step)

    origin = np.array([cfg.spawn_x, terrain_width / 2.0, 0.0])
    return meshes, origin


@configclass
class DiscreteStairsTerrainCfg(SubTerrainBaseCfg):
    """Three-level staircase with fixed height and discrete tread depths."""

    function = discrete_stairs_terrain
    step_height: float = 0.15
    step_depths: tuple[float, ...] = (0.30, 0.28, 0.25)
    num_steps: int = 5
    start_platform_length: float = 2.0
    spawn_x: float = 1.0
    foundation_thickness: float = 0.1


OPENROBOT_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=1,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=1.0,
    border_height=0.1,
    num_rows=3,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={"stairs_up": DiscreteStairsTerrainCfg(proportion=1.0)},
)

OPENROBOT_STAIRS_PLAY_TERRAINS_CFG = OPENROBOT_STAIRS_TERRAINS_CFG.replace(
    curriculum=True,
    num_rows=1,
    difficulty_range=(0.999, 1.0),
)
