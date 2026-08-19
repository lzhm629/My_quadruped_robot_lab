"""Gymnasium registrations for Go2 stairs DreamWaQ."""

import gymnasium as gym

from . import agents


gym.register(
    id="go2_stairs_dreamwaq",
    entry_point=f"{__name__}.env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2StairsDreamWaQEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:Go2StairsDreamWaQRunnerCfg",
    },
)

gym.register(
    id="go2_stairs_dreamwaq_play",
    entry_point=f"{__name__}.env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:Go2StairsDreamWaQEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_dreamwaq_cfg:Go2StairsDreamWaQRunnerCfg",
    },
)
