"""Replay and export a registered DreamWaQ locomotion task."""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Play a DreamWaQ locomotion task.")
parser.add_argument("--task", type=str, default="go2_stairs_dreamwaq_play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--real-time", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
PROJECT_DIR = Path(__file__).resolve().parents[2]
if not any(arg.startswith("hydra.run.dir=") for arg in hydra_args):
    hydra_args.append(f"hydra.run.dir={PROJECT_DIR}/outputs/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import My_quadruped_robot_lab  # noqa: E402,F401
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from My_quadruped_robot_lab.dreamwaq import DreamWaQRunner  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=str(PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name / "videos"), episode_trigger=lambda _: True, video_length=args_cli.video_length)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        root = PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name
        checkpoint = get_checkpoint_path(str(root), "-1", "-1")
    runner = DreamWaQRunner(env, agent_cfg.to_dict(), None, agent_cfg.device)
    print(f"[INFO] Loading DreamWaQ checkpoint: {checkpoint}")
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.inference_policy()
    export_path = Path(checkpoint).parent / "exported" / "policy_dreamwaq.pt"
    runner.export(str(export_path))
    print(f"[INFO] Exported policy to {export_path}")
    obs, extras = env.get_observations()
    dt = env.unwrapped.step_dt
    steps = 0
    while simulation_app.is_running():
        start = time.time()
        groups = extras["observations"]
        with torch.inference_mode():
            actions = policy(obs.to(runner.device), groups["history"].to(runner.device))
            obs, _, dones, extras = env.step(actions)
        steps += 1
        if args_cli.video and steps >= args_cli.video_length:
            break
        if args_cli.real_time:
            remaining = dt - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
