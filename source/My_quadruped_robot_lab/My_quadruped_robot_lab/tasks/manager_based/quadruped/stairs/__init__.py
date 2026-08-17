"""Gymnasium registration for the OpenRobot stairs tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="openrobot_wheelfixed_stairs",
    entry_point=f"{__name__}.env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openrobot_env_cfg:OpenRobotStairsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenRobotStairsPPORunnerCfg",
    },
)

gym.register(
    id="openrobot_wheelfixed_stairs_play",
    entry_point=f"{__name__}.env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openrobot_env_cfg:OpenRobotStairsEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenRobotStairsPPORunnerCfg",
    },
)
