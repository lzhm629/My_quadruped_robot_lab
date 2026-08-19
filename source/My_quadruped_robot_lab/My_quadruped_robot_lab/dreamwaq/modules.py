"""DreamWaQ actor-critic and context encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def activation(name: str) -> nn.Module:
    table = {"elu": nn.ELU, "relu": nn.ReLU, "selu": nn.SELU, "lrelu": nn.LeakyReLU, "tanh": nn.Tanh}
    if name not in table:
        raise ValueError(f"Unsupported activation: {name}")
    return table[name]()


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_name: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), activation(activation_name)))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class ContextVAE(nn.Module):
    def __init__(self, history_dim: int, latent_dim: int, explicit_dim: int, reconstruction_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(history_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.latent_mean = nn.Linear(64, latent_dim)
        self.latent_logvar = nn.Sequential(nn.Linear(64, latent_dim), nn.Hardtanh(-5.0, 5.0))
        self.velocity_mean = nn.Linear(64, explicit_dim)
        self.velocity_logvar = nn.Sequential(nn.Linear(64, explicit_dim), nn.Hardtanh(-5.0, 5.0))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + explicit_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, reconstruction_dim)
        )

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def forward(self, history: torch.Tensor, deterministic: bool = False):
        encoded = self.encoder(history)
        latent_mean = self.latent_mean(encoded)
        latent_logvar = self.latent_logvar(encoded)
        velocity_mean = self.velocity_mean(encoded)
        velocity_logvar = self.velocity_logvar(encoded)
        if deterministic:
            latent, velocity = latent_mean, velocity_mean
        else:
            latent = self.sample(latent_mean, latent_logvar)
            velocity = self.sample(velocity_mean, velocity_logvar)
        code = torch.cat((velocity, latent), dim=-1)
        reconstruction = self.decoder(code)
        return code, reconstruction, velocity, latent_mean, latent_logvar, velocity_mean


class ActorCriticDreamWaQ(nn.Module):
    is_recurrent = False

    def __init__(
        self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int, history_dim: int,
        latent_dim: int, explicit_dim: int, actor_hidden_dims: list[int], critic_hidden_dims: list[int],
        activation_name: str = "elu", init_noise_std: float = 1.0,
    ):
        super().__init__()
        self.vae = ContextVAE(history_dim, latent_dim, explicit_dim, actor_obs_dim)
        self.actor = mlp(actor_obs_dim + latent_dim + explicit_dim, actor_hidden_dims, action_dim, activation_name)
        self.critic = mlp(critic_obs_dim, critic_hidden_dims, 1, activation_name)
        self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update_distribution(self, obs: torch.Tensor, history: torch.Tensor) -> None:
        code, _, _, _, _, _ = self.vae(history)
        mean = self.actor(torch.cat((code, obs), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        self.update_distribution(obs, history)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        code, _, _, _, _, _ = self.vae(history, deterministic=True)
        return self.actor(torch.cat((code, obs), dim=-1))

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass


class DreamWaQInference(nn.Module):
    """TorchScript-friendly policy with separate current and history inputs."""

    def __init__(self, policy: ActorCriticDreamWaQ):
        super().__init__()
        self.vae = policy.vae
        self.actor = policy.actor

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        code, _, _, _, _, _ = self.vae(history, deterministic=True)
        return self.actor(torch.cat((code, obs), dim=-1))
