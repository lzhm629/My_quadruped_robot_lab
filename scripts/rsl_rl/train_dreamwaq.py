"""Train a registered DreamWaQ locomotion task."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Train a DreamWaQ locomotion task.")
parser.add_argument("--task", type=str, default="go2_stairs_dreamwaq")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

PROJECT_DIR = Path(__file__).resolve().parents[2]
if not any(arg.startswith("hydra.run.dir=") for arg in hydra_args):
    hydra_args.append(f"hydra.run.dir={PROJECT_DIR}/outputs/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import My_quadruped_robot_lab  # noqa: E402,F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from My_quadruped_robot_lab.dreamwaq import DreamWaQRunner  # noqa: E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    log_root = PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name
    log_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    cfg_dict = agent_cfg.to_dict()
    runner = DreamWaQRunner(env, cfg_dict, str(log_dir), agent_cfg.device)
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading DreamWaQ checkpoint: {resume_path}")
        runner.load(resume_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    dump_pickle(str(log_dir / "params" / "env.pkl"), env_cfg)
    dump_pickle(str(log_dir / "params" / "agent.pkl"), agent_cfg)
    runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
