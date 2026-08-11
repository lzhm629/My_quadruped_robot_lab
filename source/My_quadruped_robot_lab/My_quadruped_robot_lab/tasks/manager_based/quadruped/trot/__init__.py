"""Gymnasium registration for the Go2 trot task."""

import gymnasium as gym

from . import agents


gym.register(
    id="OpenRobot-Velocity-Trot-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trot_env_cfg:Go2TrotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TrotPPORunnerCfg",
    },
)

gym.register(
    id="OpenRobot-Velocity-Trot-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.trot_env_cfg:Go2TrotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TrotPPORunnerCfg",
    },
)

gym.register(
    id="OpenRobot-Velocity-Trot-OpenRobot-WheelFixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openrobot_env_cfg:OpenRobotTrotEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.openrobot_rsl_rl_ppo_cfg:OpenRobotTrotPPORunnerCfg"
        ),
    },
)

gym.register(
    id="OpenRobot-Velocity-Trot-OpenRobot-WheelFixed-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openrobot_env_cfg:OpenRobotTrotEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.openrobot_rsl_rl_ppo_cfg:OpenRobotTrotPPORunnerCfg"
        ),
    },
)
