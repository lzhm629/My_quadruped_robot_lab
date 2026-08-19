"""MDP terms for OpenRobot stairs DreamWaQ."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403
from My_quadruped_robot_lab.tasks.manager_based.quadruped.go2_stairs_dreamwaq.mdp.observations import *  # noqa: F401, F403
from My_quadruped_robot_lab.tasks.manager_based.quadruped.go2_stairs_dreamwaq.mdp.rewards import *  # noqa: F401, F403

from .curriculums import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
