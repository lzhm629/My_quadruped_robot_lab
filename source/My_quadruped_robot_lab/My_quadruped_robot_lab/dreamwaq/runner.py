"""Training and inference runner for Isaac Lab DreamWaQ tasks."""

from __future__ import annotations

import os
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from .algorithm import PPODreamWaQ
from .modules import ActorCriticDreamWaQ, DreamWaQInference


class DreamWaQRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None, device: str):
        self.env, self.cfg, self.log_dir, self.device = env, train_cfg, log_dir, device
        obs, extras = env.get_observations()
        observations = extras["observations"]
        required = {"critic", "history", "velocity", "reconstruction"}
        missing = required.difference(observations)
        if missing:
            raise KeyError(f"DreamWaQ observation groups missing: {sorted(missing)}")
        dims = {name: tensor.shape[-1] for name, tensor in observations.items()}
        expected = {"policy": 45, "history": 225, "critic": 783, "velocity": 3, "reconstruction": 45}
        if {name: dims[name] for name in expected} != expected:
            raise ValueError(f"DreamWaQ dimensions differ from contract: got {dims}, expected {expected}")

        policy_cfg = train_cfg["policy"].copy()
        policy_cfg.pop("class_name", None)
        self.policy = ActorCriticDreamWaQ(
            actor_obs_dim=dims["policy"], critic_obs_dim=dims["critic"], action_dim=env.num_actions,
            history_dim=train_cfg["history_dim"], latent_dim=train_cfg["latent_dim"],
            explicit_dim=train_cfg["explicit_dim"], actor_hidden_dims=policy_cfg["actor_hidden_dims"],
            critic_hidden_dims=policy_cfg["critic_hidden_dims"], activation_name=policy_cfg["activation"],
            init_noise_std=policy_cfg["init_noise_std"],
        ).to(device)
        algorithm_cfg = train_cfg["algorithm"].copy()
        algorithm_cfg.pop("class_name", None)
        self.alg = PPODreamWaQ(
            self.policy, algorithm_cfg, train_cfg["vae_learning_rate"], train_cfg["vae_kl_weight"]
        )
        self.num_steps = train_cfg["num_steps_per_env"]
        self.save_interval = train_cfg["save_interval"]
        self.alg.init_storage(self.num_steps, env.num_envs, 45, 783, 225, env.num_actions, device)
        self.current_iteration = 0
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10) if log_dir else None

    @staticmethod
    def _unpack(obs, extras):
        groups = extras["observations"]
        return obs, groups["critic"], groups["history"], groups["velocity"], groups["reconstruction"]

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=self.env.max_episode_length)
        state = self._unpack(*self.env.get_observations())
        rewards_window, lengths_window = deque(maxlen=100), deque(maxlen=100)
        running_rewards = torch.zeros(self.env.num_envs, device=self.device)
        running_lengths = torch.zeros(self.env.num_envs, device=self.device)
        for iteration in range(self.current_iteration, self.current_iteration + num_learning_iterations):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps):
                    obs, critic, history, velocity, reconstruction = (x.to(self.device) for x in state)
                    actions = self.alg.act(obs, critic, history, velocity, reconstruction)
                    next_obs, rewards, dones, extras = self.env.step(actions)
                    self.alg.process_step(rewards.to(self.device), dones.to(self.device), extras)
                    state = self._unpack(next_obs, extras)
                    running_rewards += rewards
                    running_lengths += 1
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if len(done_ids):
                        rewards_window.extend(running_rewards[done_ids].cpu().tolist())
                        lengths_window.extend(running_lengths[done_ids].cpu().tolist())
                        running_rewards[done_ids] = 0.0
                        running_lengths[done_ids] = 0.0
                self.alg.compute_returns(state[1].to(self.device))
            losses = self.alg.update()
            elapsed = time.time() - start
            if self.writer:
                for name, value in losses.items():
                    self.writer.add_scalar(f"Loss/{name}", value, iteration)
                self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
                self.writer.add_scalar("Policy/action_noise", self.policy.std.mean().item(), iteration)
                self.writer.add_scalar("Perf/iteration_seconds", elapsed, iteration)
                if rewards_window:
                    self.writer.add_scalar("Train/mean_reward", sum(rewards_window) / len(rewards_window), iteration)
                    self.writer.add_scalar("Train/mean_episode_length", sum(lengths_window) / len(lengths_window), iteration)
            if iteration % 10 == 0:
                mean_reward = sum(rewards_window) / len(rewards_window) if rewards_window else 0.0
                print(
                    f"iteration {iteration} | reward {mean_reward:.3f} | "
                    f"ppo {losses['surrogate']:.4f} | value {losses['value']:.4f} | "
                    f"vae {losses['vae']:.4f} | vel {losses['velocity']:.4f} | {elapsed:.2f}s"
                )
            if self.log_dir and iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"), iteration)
        self.current_iteration += num_learning_iterations
        if self.log_dir:
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"), self.current_iteration)

    def save(self, path: str, iteration: int | None = None):
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "vae_optimizer_state_dict": self.alg.vae_optimizer.state_dict(),
            "iter": self.current_iteration if iteration is None else iteration,
            "config": self.cfg,
        }, path)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.vae_optimizer.load_state_dict(checkpoint["vae_optimizer_state_dict"])
        self.current_iteration = checkpoint.get("iter", 0)

    def inference_policy(self):
        self.policy.eval()
        return self.policy.act_inference

    def export(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        module = DreamWaQInference(self.policy).to("cpu").eval()
        traced = torch.jit.trace(module, (torch.zeros(1, 45), torch.zeros(1, 225)))
        traced.save(path)
        self.policy.to(self.device)
