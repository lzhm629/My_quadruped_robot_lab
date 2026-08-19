"""Curriculum and exact evaluation terrains for OpenRobot DreamWaQ stairs."""

from dataclasses import MISSING

import numpy as np

from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from My_quadruped_robot_lab.tasks.manager_based.quadruped.go2_stairs_dreamwaq.terrain_cfg import (
    RoughPyramidSlopeCfg,
)


@height_field_to_mesh
def variable_pyramid_stairs(difficulty: float, cfg: "VariablePyramidStairsCfg") -> np.ndarray:
    """Generate pyramid stairs whose tread depth and riser height both scale with difficulty."""
    step_depth = cfg.step_depth_range[0] + difficulty * (
        cfg.step_depth_range[1] - cfg.step_depth_range[0]
    )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    if cfg.depth_jitter > 0.0:
        step_depth += np.random.uniform(-cfg.depth_jitter, cfg.depth_jitter) * difficulty
    if cfg.height_jitter > 0.0:
        step_height += np.random.uniform(-cfg.height_jitter, cfg.height_jitter) * difficulty
    if cfg.inverted:
        step_height *= -1.0

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_pixels = max(1, int(round(step_depth / cfg.horizontal_scale)))
    height_units = int(round(step_height / cfg.vertical_scale))
    platform_pixels = int(round(cfg.platform_width / cfg.horizontal_scale))

    height_field = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    current_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_pixels and (stop_y - start_y) > platform_pixels:
        start_x += step_pixels
        stop_x -= step_pixels
        start_y += step_pixels
        stop_y -= step_pixels
        current_height += height_units
        height_field[start_x:stop_x, start_y:stop_y] = current_height
    return height_field


@configclass
class VariablePyramidStairsCfg(HfTerrainBaseCfg):
    """Configuration for variable-depth stairs centered on a flat platform."""

    function = variable_pyramid_stairs
    step_depth_range: tuple[float, float] = MISSING
    step_height_range: tuple[float, float] = MISSING
    platform_width: float = 2.0
    inverted: bool = False
    depth_jitter: float = 0.0
    height_jitter: float = 0.0


OPENROBOT_DREAMWAQ_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=1,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    border_height=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "smooth_slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.30), platform_width=2.0
        ),
        "smooth_slope_down": HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.30), platform_width=2.0, inverted=True
        ),
        "rough_slope": RoughPyramidSlopeCfg(
            proportion=0.10,
            max_slope=0.30,
            platform_width=2.0,
            noise_amplitude=0.04,
        ),
        # From the center platform, a normal pyramid is traversed downwards.
        "stairs_down": VariablePyramidStairsCfg(
            proportion=0.35,
            step_depth_range=(0.38, 0.25),
            step_height_range=(0.06, 0.15),
            platform_width=2.0,
            depth_jitter=0.02,
            height_jitter=0.01,
        ),
        # An inverted pyramid rises from its center platform towards the border.
        "stairs_up": VariablePyramidStairsCfg(
            proportion=0.35,
            step_depth_range=(0.38, 0.25),
            step_height_range=(0.06, 0.15),
            platform_width=2.0,
            inverted=True,
            depth_jitter=0.02,
            height_jitter=0.01,
        ),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(-0.06, 0.06),
            noise_step=0.005,
            downsampled_scale=0.2,
        ),
    },
)


def _exact_stairs(*, inverted: bool) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=1,
        curriculum=False,
        size=(8.0, 8.0),
        border_width=25.0,
        border_height=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(1.0, 1.0),
        use_cache=False,
        sub_terrains={
            "exact_stairs": VariablePyramidStairsCfg(
                proportion=1.0,
                step_depth_range=(0.25, 0.25),
                step_height_range=(0.15, 0.15),
                platform_width=2.0,
                inverted=inverted,
            )
        },
    )


OPENROBOT_DREAMWAQ_PLAY_UP_TERRAINS_CFG = _exact_stairs(inverted=True)
OPENROBOT_DREAMWAQ_PLAY_DOWN_TERRAINS_CFG = _exact_stairs(inverted=False)
