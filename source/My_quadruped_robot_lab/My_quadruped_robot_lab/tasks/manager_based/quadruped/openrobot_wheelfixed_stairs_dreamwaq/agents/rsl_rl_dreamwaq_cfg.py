"""DreamWaQ runner configuration for the wheel-fixed OpenRobot stairs task."""

from isaaclab.utils import configclass

from My_quadruped_robot_lab.tasks.manager_based.quadruped.go2_stairs_dreamwaq.agents.rsl_rl_dreamwaq_cfg import (
    Go2StairsDreamWaQRunnerCfg,
)


@configclass
class OpenRobotWheelFixedStairsDreamWaQRunnerCfg(Go2StairsDreamWaQRunnerCfg):
    experiment_name = "openrobot_wheelfixed_stairs_dreamwaq"
