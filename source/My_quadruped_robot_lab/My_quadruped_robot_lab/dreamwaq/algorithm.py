"""PPO with the DreamWaQ context-estimation objective."""

from __future__ import annotations

import torch
import torch.nn as nn

from .modules import ActorCriticDreamWaQ
from .storage import DreamWaQStorage


class PPODreamWaQ:
    def __init__(self, policy: ActorCriticDreamWaQ, cfg: dict, vae_learning_rate: float, vae_kl_weight: float):
        self.policy = policy
        self.cfg = cfg
        self.learning_rate = cfg["learning_rate"]
        self.vae_kl_weight = vae_kl_weight
        self.rl_parameters = list(policy.actor.parameters()) + list(policy.critic.parameters()) + [policy.std]
        self.optimizer = torch.optim.Adam(self.rl_parameters, lr=self.learning_rate)
        self.vae_optimizer = torch.optim.Adam(policy.vae.parameters(), lr=vae_learning_rate)
        self.storage: DreamWaQStorage | None = None
        self.transition: dict[str, torch.Tensor] = {}

    def init_storage(self, num_steps, num_envs, obs_dim, critic_dim, history_dim, action_dim, device):
        self.storage = DreamWaQStorage(num_steps, num_envs, obs_dim, critic_dim, history_dim, action_dim, device)

    def act(self, obs, critic, history, velocity, reconstruction):
        actions = self.policy.act(obs, history)
        self.transition = {
            "obs": obs, "critic": critic, "history": history, "velocity": velocity,
            "reconstruction": reconstruction, "actions": actions.detach(),
            "values": self.policy.evaluate(critic).detach(),
            "log_prob": self.policy.get_actions_log_prob(actions).detach().unsqueeze(-1),
            "mu": self.policy.action_mean.detach(), "sigma": self.policy.action_std.detach(),
        }
        return actions.detach()

    def process_step(self, rewards, dones, extras):
        rewards = rewards.clone()
        if "time_outs" in extras:
            rewards += self.cfg["gamma"] * self.transition["values"].squeeze(-1) * extras["time_outs"].float()
        self.transition["rewards"] = rewards.unsqueeze(-1)
        self.transition["dones"] = dones.float().unsqueeze(-1)
        self.storage.add(self.transition)
        self.transition = {}

    def compute_returns(self, critic):
        self.storage.compute_returns(
            self.policy.evaluate(critic).detach(), self.cfg["gamma"], self.cfg["lam"]
        )

    def update(self):
        totals = {name: 0.0 for name in ("value", "surrogate", "entropy", "vae", "velocity", "reconstruction", "kl")}
        updates = 0
        for batch in self.storage.batches(self.cfg["num_mini_batches"], self.cfg["num_learning_epochs"]):
            self.policy.act(batch["obs"], batch["history"])
            log_prob = self.policy.get_actions_log_prob(batch["actions"]).unsqueeze(-1)
            value = self.policy.evaluate(batch["critic"])
            mu, sigma = self.policy.action_mean, self.policy.action_std

            if self.cfg["schedule"] == "adaptive" and self.cfg["desired_kl"] is not None:
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(sigma / batch["sigma"] + 1.0e-5)
                        + (batch["sigma"].square() + (batch["mu"] - mu).square()) / (2.0 * sigma.square())
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > 2.0 * self.cfg["desired_kl"]:
                        self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < 0.5 * self.cfg["desired_kl"]:
                        self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    self.optimizer.param_groups[0]["lr"] = self.learning_rate

            ratio = torch.exp(log_prob - batch["log_prob"])
            surrogate = -batch["advantages"] * ratio
            clipped = -batch["advantages"] * ratio.clamp(1.0 - self.cfg["clip_param"], 1.0 + self.cfg["clip_param"])
            surrogate_loss = torch.maximum(surrogate, clipped).mean()

            if self.cfg["use_clipped_value_loss"]:
                value_clipped = batch["values"] + (value - batch["values"]).clamp(
                    -self.cfg["clip_param"], self.cfg["clip_param"]
                )
                value_loss = torch.maximum(
                    (value - batch["returns"]).square(), (value_clipped - batch["returns"]).square()
                ).mean()
            else:
                value_loss = (value - batch["returns"]).square().mean()
            entropy = self.policy.entropy.mean()
            rl_loss = surrogate_loss + self.cfg["value_loss_coef"] * value_loss - self.cfg["entropy_coef"] * entropy
            self.optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(self.rl_parameters, self.cfg["max_grad_norm"])
            self.optimizer.step()

            _, decoded, sampled_velocity, latent_mean, latent_logvar, _ = self.policy.vae(batch["history"])
            live = 1.0 - batch["dones"]
            velocity_loss = ((sampled_velocity - batch["velocity"]).square() * live).mean()
            reconstruction_loss = ((decoded - batch["reconstruction"]).square() * live).mean()
            kl_loss = -0.5 * torch.mean(
                torch.sum(1.0 + latent_logvar - latent_mean.square() - latent_logvar.exp(), dim=-1) * live.squeeze(-1)
            )
            vae_loss = velocity_loss + reconstruction_loss + self.vae_kl_weight * kl_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.vae.parameters(), self.cfg["max_grad_norm"])
            self.vae_optimizer.step()

            for name, value_item in (
                ("value", value_loss), ("surrogate", surrogate_loss), ("entropy", entropy),
                ("vae", vae_loss), ("velocity", velocity_loss),
                ("reconstruction", reconstruction_loss), ("kl", kl_loss),
            ):
                totals[name] += value_item.item()
            updates += 1
        self.storage.clear()
        return {name: value / updates for name, value in totals.items()}
