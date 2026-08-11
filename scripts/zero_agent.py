# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_steps", type=int, default=None, help="Stop after this many environment steps.")
parser.add_argument(
    "--disable_base_contact_termination",
    action="store_true",
    help="Keep fallen robots in the scene instead of resetting them on base contact.",
)
parser.add_argument(
    "--report_robot_state",
    action="store_true",
    help="Print root pose and joint state ranges after the rollout.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import My_quadruped_robot_lab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    if args_cli.disable_base_contact_termination and hasattr(env_cfg.terminations, "base_contact"):
        env_cfg.terminations.base_contact = None
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    step = 0
    terminated_count = 0
    truncated_count = 0
    while simulation_app.is_running() and (args_cli.num_steps is None or step < args_cli.num_steps):
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            _, _, terminated, truncated, _ = env.step(actions)
            terminated_count += int(terminated.sum().item())
            truncated_count += int(truncated.sum().item())
            step += 1

    print(
        f"[INFO]: Completed {step} steps; terminated={terminated_count}, truncated={truncated_count}"
    )
    if args_cli.report_robot_state:
        robot = env.unwrapped.scene["robot"]
        print(f"[INFO]: joint_names={robot.joint_names}")
        print(f"[INFO]: root_pos_w={robot.data.root_pos_w.detach().cpu().tolist()}")
        print(f"[INFO]: root_quat_w={robot.data.root_quat_w.detach().cpu().tolist()}")
        print(f"[INFO]: joint_pos={robot.data.joint_pos.detach().cpu().tolist()}")
        print(f"[INFO]: joint_vel={robot.data.joint_vel.detach().cpu().tolist()}")
        print(f"[INFO]: computed_torque={robot.data.computed_torque.detach().cpu().tolist()}")
        print(f"[INFO]: applied_torque={robot.data.applied_torque.detach().cpu().tolist()}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
