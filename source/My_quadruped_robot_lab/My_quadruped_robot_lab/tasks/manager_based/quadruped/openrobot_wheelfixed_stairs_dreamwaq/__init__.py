"""Gymnasium registrations for OpenRobot wheel-fixed stairs DreamWaQ."""

import gymnasium as gym

from . import agents


_RUNNER_CFG = (
    f"{agents.__name__}.rsl_rl_dreamwaq_cfg:"
    "OpenRobotWheelFixedStairsDreamWaQRunnerCfg"
)

gym.register(
    id="openrobot_wheelfixed_stairs_dreamwaq",
    entry_point=f"{__name__}.env:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:OpenRobotWheelFixedStairsDreamWaQEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": _RUNNER_CFG,
    },
)

gym.register(
    id="openrobot_wheelfixed_stairs_dreamwaq_play_up",
    entry_point=f"{__name__}.env:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:OpenRobotWheelFixedStairsDreamWaQEnvCfg_PLAY_UP"
        ),
        "rsl_rl_cfg_entry_point": _RUNNER_CFG,
    },
)

gym.register(
    id="openrobot_wheelfixed_stairs_dreamwaq_play_down",
    entry_point=f"{__name__}.env:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:OpenRobotWheelFixedStairsDreamWaQEnvCfg_PLAY_DOWN"
        ),
        "rsl_rl_cfg_entry_point": _RUNNER_CFG,
    },
)
