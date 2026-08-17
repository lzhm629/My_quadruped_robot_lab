"""Environment compatibility behavior for the migrated stairs task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class PositiveRewardManagerBasedRLEnv(ManagerBasedRLEnv):
    """Clamp the summed reward to match the legacy stairs environment."""

    def step(self, action: torch.Tensor):
        observations, reward, terminated, truncated, extras = super().step(action)
        reward.clamp_(min=0.0)
        self.reward_buf = reward
        return observations, reward, terminated, truncated, extras
