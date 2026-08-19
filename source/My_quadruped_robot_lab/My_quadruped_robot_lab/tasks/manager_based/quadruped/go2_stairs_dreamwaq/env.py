"""Environment compatibility behavior for legacy DreamWaQ rewards."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


class PositiveRewardManagerBasedRLEnv(ManagerBasedRLEnv):
    """Clamp the summed reward after all manager terms are evaluated."""

    def step(self, action: torch.Tensor):
        observations, reward, terminated, truncated, extras = super().step(action)
        reward.clamp_(min=0.0)
        self.reward_buf = reward
        return observations, reward, terminated, truncated, extras
