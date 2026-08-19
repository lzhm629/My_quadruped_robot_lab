"""Mixed curriculum terrain matching the legacy Go2 DreamWaQ task."""

from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate

from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg,
    HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainGeneratorCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def rough_pyramid_slope(difficulty: float, cfg: "RoughPyramidSlopeCfg") -> np.ndarray:
    """Combine the legacy positive pyramid slope with downsampled uniform roughness."""
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    center_x, center_y = width // 2, length // 2
    x = (center_x - np.abs(center_x - np.arange(width))) / center_x
    y = (center_y - np.abs(center_y - np.arange(length))) / center_y
    slope = difficulty * cfg.max_slope
    height_max = slope * cfg.size[0] * 0.5 / cfg.vertical_scale
    height_field = height_max * x[:, None] * y[None, :]
    platform = int(cfg.platform_width / cfg.horizontal_scale / 2)
    platform_height = height_field[center_x - platform, center_y - platform]
    height_field = np.clip(height_field, 0.0, platform_height)

    coarse_x = max(2, int(cfg.size[0] / cfg.downsampled_scale))
    coarse_y = max(2, int(cfg.size[1] / cfg.downsampled_scale))
    noise_limit = int(cfg.noise_amplitude / cfg.vertical_scale)
    noise_step = max(1, int(cfg.noise_step / cfg.vertical_scale))
    values = np.arange(-noise_limit, noise_limit + noise_step, noise_step)
    coarse_noise = np.random.choice(values, size=(coarse_x, coarse_y))
    spline = interpolate.RectBivariateSpline(
        np.linspace(0.0, cfg.size[0], coarse_x), np.linspace(0.0, cfg.size[1], coarse_y), coarse_noise
    )
    noise = spline(np.linspace(0.0, cfg.size[0], width), np.linspace(0.0, cfg.size[1], length))
    return np.rint(height_field + noise).astype(np.int16)


@configclass
class RoughPyramidSlopeCfg(HfTerrainBaseCfg):
    function = rough_pyramid_slope
    max_slope: float = MISSING
    platform_width: float = 3.0
    noise_amplitude: float = 0.05
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


GO2_DREAMWAQ_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=1,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    border_height=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "smooth_slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.4), platform_width=3.0
        ),
        "smooth_slope_down": HfPyramidSlopedTerrainCfg(
            proportion=0.075, slope_range=(0.0, 0.4), platform_width=3.0, inverted=True
        ),
        "rough_slope": RoughPyramidSlopeCfg(
            proportion=0.15, max_slope=0.4, platform_width=3.0
        ),
        "stairs_up": HfPyramidStairsTerrainCfg(
            proportion=0.30, step_height_range=(0.05, 0.23), step_width=0.31, platform_width=3.0
        ),
        "stairs_down": HfPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.05, 0.23),
            step_width=0.31,
            platform_width=3.0,
            inverted=True,
        ),
        "rough": HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(-0.08, 0.08), noise_step=0.005, downsampled_scale=0.2
        ),
    },
)


GO2_DREAMWAQ_PLAY_TERRAINS_CFG = GO2_DREAMWAQ_TERRAINS_CFG.replace(
    curriculum=False,
    num_rows=5,
    num_cols=5,
    difficulty_range=(0.6, 0.9),
)
