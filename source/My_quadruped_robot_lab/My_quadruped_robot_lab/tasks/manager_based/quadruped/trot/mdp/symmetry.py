"""Left/right symmetry augmentation for quadruped trot observations and actions."""

from __future__ import annotations

import torch


def _manager_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _joint_mirror(env, flip_all_joint_signs: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    manager_env = _manager_env(env)
    names = list(manager_env.action_manager.get_term("joint_pos")._joint_names)
    name_to_index = {name: index for index, name in enumerate(names)}
    permutation = []
    signs = []
    for name in names:
        if name.startswith("FL_"):
            mirrored_name = "FR_" + name[3:]
        elif name.startswith("FR_"):
            mirrored_name = "FL_" + name[3:]
        elif name.startswith("RL_"):
            mirrored_name = "RR_" + name[3:]
        elif name.startswith("RR_"):
            mirrored_name = "RL_" + name[3:]
        else:
            raise ValueError(f"Unsupported quadruped joint name for symmetry: {name}")
        permutation.append(name_to_index[mirrored_name])
        signs.append(-1.0 if flip_all_joint_signs or name.endswith("_hip_joint") else 1.0)
    device = manager_env.device
    return torch.tensor(permutation, device=device), torch.tensor(signs, device=device)


def _transform_block(
    source: torch.Tensor,
    target: torch.Tensor,
    start: int,
    history: int,
    width: int,
    permutation: torch.Tensor | None = None,
    signs: torch.Tensor | tuple[float, ...] | None = None,
) -> int:
    end = start + history * width
    block = source[:, start:end].reshape(source.shape[0], history, width)
    if permutation is not None:
        block = block[..., permutation]
    if signs is not None:
        block = block * torch.as_tensor(signs, device=source.device, dtype=source.dtype)
    target[:, start:end] = block.reshape(source.shape[0], -1)
    return end


def _mirror_observation(
    obs: torch.Tensor, env, obs_type: str, flip_all_joint_signs: bool = False
) -> torch.Tensor:
    history = 10 if obs_type == "policy" else 3
    joint_permutation, joint_signs = _joint_mirror(env, flip_all_joint_signs)
    mirrored = obs.clone()
    index = 0
    index = _transform_block(obs, mirrored, index, history, 2, signs=(-1.0, -1.0))
    index = _transform_block(obs, mirrored, index, history, 3, signs=(1.0, -1.0, -1.0))
    index = _transform_block(obs, mirrored, index, history, 3, signs=(-1.0, 1.0, -1.0))
    index = _transform_block(obs, mirrored, index, history, 3, signs=(1.0, -1.0, 1.0))
    for _ in range(3):
        index = _transform_block(
            obs, mirrored, index, history, 12, permutation=joint_permutation, signs=joint_signs
        )
    if obs_type == "critic":
        index = _transform_block(obs, mirrored, index, history, 3, signs=(1.0, -1.0, 1.0))
        index = _transform_block(
            obs, mirrored, index, history, 12, permutation=joint_permutation, signs=joint_signs
        )
        index = _transform_block(obs, mirrored, index, history, 4, permutation=torch.tensor([1, 0, 3, 2], device=obs.device))
        index = _transform_block(obs, mirrored, index, history, 1)
        index = _transform_block(obs, mirrored, index, history, 1)
    if index != obs.shape[-1]:
        raise ValueError(f"Unexpected {obs_type} observation width: consumed {index}, received {obs.shape[-1]}")
    return mirrored


def _quadruped_trot_symmetry(obs, actions, env, obs_type: str, flip_all_joint_signs: bool):
    augmented_obs = None
    augmented_actions = None
    if obs is not None:
        augmented_obs = torch.cat(
            (obs, _mirror_observation(obs, env, obs_type, flip_all_joint_signs)), dim=0
        )
    if actions is not None:
        permutation, signs = _joint_mirror(env, flip_all_joint_signs)
        mirrored_actions = actions[:, permutation] * signs
        augmented_actions = torch.cat((actions, mirrored_actions), dim=0)
    return augmented_obs, augmented_actions


def quadruped_trot_symmetry(obs, actions, env, obs_type="policy"):
    """Mirror Go2 samples: only hip joint coordinates change sign."""
    return _quadruped_trot_symmetry(obs, actions, env, obs_type, flip_all_joint_signs=False)


def openrobot_trot_symmetry(obs, actions, env, obs_type="policy"):
    """Mirror OpenRobot samples whose left/right URDF joint axes are opposite."""
    return _quadruped_trot_symmetry(obs, actions, env, obs_type, flip_all_joint_signs=True)
