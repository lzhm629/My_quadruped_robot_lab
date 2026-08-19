"""Rollout storage carrying DreamWaQ history and auxiliary targets."""

from __future__ import annotations

import torch


class DreamWaQStorage:
    def __init__(self, num_steps, num_envs, obs_dim, critic_dim, history_dim, action_dim, device):
        shape = (num_steps, num_envs)
        self.obs = torch.zeros(*shape, obs_dim, device=device)
        self.critic = torch.zeros(*shape, critic_dim, device=device)
        self.history = torch.zeros(*shape, history_dim, device=device)
        self.velocity = torch.zeros(*shape, 3, device=device)
        self.reconstruction = torch.zeros(*shape, obs_dim, device=device)
        self.actions = torch.zeros(*shape, action_dim, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.dones = torch.zeros(*shape, 1, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.log_prob = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, action_dim, device=device)
        self.sigma = torch.zeros(*shape, action_dim, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)
        self.advantages = torch.zeros(*shape, 1, device=device)
        self.num_steps, self.num_envs, self.step = num_steps, num_envs, 0

    def add(self, transition: dict[str, torch.Tensor]):
        for name, tensor in transition.items():
            target = getattr(self, name)[self.step]
            target.copy_(tensor.view_as(target))
        self.step += 1

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0.0
        for step in reversed(range(self.num_steps)):
            next_values = last_values if step == self.num_steps - 1 else self.values[step + 1]
            alive = 1.0 - self.dones[step]
            delta = self.rewards[step] + alive * gamma * next_values - self.values[step]
            advantage = delta + alive * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages.copy_(self.returns - self.values)
        self.advantages.sub_(self.advantages.mean()).div_(self.advantages.std() + 1.0e-8)

    def batches(self, num_mini_batches, num_epochs):
        flattened = {name: getattr(self, name).flatten(0, 1) for name in (
            "obs", "critic", "history", "velocity", "reconstruction", "actions", "values",
            "log_prob", "mu", "sigma", "returns", "advantages", "dones"
        )}
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // num_mini_batches
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.obs.device)
            for start in range(0, mini_batch_size * num_mini_batches, mini_batch_size):
                ids = indices[start:start + mini_batch_size]
                yield {name: tensor[ids] for name, tensor in flattened.items()}

    def clear(self):
        self.step = 0
