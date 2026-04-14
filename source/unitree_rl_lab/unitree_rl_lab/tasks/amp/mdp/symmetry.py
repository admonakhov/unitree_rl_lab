# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to specify the symmetry in the observation and action space for ANYmal."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "critic",
):


    # observations
    if obs is not None:
        num_envs = obs.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        obs_aug = torch.zeros(num_envs * 2, obs.shape[1], device=obs.device)
        # -- original
        obs_aug[:num_envs] = obs[:]
        # -- front-back
        obs_aug[num_envs : 2 * num_envs] = _transform_obs_front_back(env.unwrapped, obs, obs_type)

    else:
        obs_aug = None

    # actions
    if actions is not None:
        num_envs = actions.shape[0]
        # since we have 4 different symmetries, we need to augment the batch size by 4
        actions_aug = torch.zeros(num_envs * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:num_envs] = actions[:]
        actions_aug[num_envs : 2 * num_envs] = _transform_actions_front_back(actions)

    else:
        actions_aug = None

    return obs_aug, actions_aug



def _transform_obs_front_back(env: ManagerBasedRLEnv, obs: torch.Tensor, obs_type: str = "policy") -> torch.Tensor:
    """Applies a front-back symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes negating
    certain components of the linear and angular velocities, projected gravity, velocity commands,
    and flipping the joint positions, joint velocities, and last actions for the ANYmal robot.
    Additionally, if height-scan data is present, it is flipped along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.
        obs_type: The type of observation to augment. Defaults to "policy".

    Returns:
        The transformed observation tensor with front-back symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    # lin vel
    obs[:, :3] = obs[:, :3] * torch.tensor([-1, 1, 1], device=device)
    # ang vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([1, -1, -1], device=device)
    # projected gravity
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([-1, 1, 1], device=device)
    # velocity command
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([-1, 1, -1], device=device)
    # joint pos
    obs[:, 12:34] = _switch_anymal_joints_front_back(obs[:, 12:34])
    # joint vel
    obs[:, 34:56] = _switch_anymal_joints_front_back(obs[:, 34:56])
    # last actions
    obs[:, 56:78] = _switch_anymal_joints_front_back(obs[:, 56:78])

    # height-scan
    if obs_type == "critic":
        # handle asymmetric actor-critic formulation
        group_name = "critic" if "critic" in env.observation_manager.active_terms else "policy"
    else:
        group_name = "policy"

    # note: this is hard-coded for grid-pattern of ordering "xy" and size (1.6, 1.0)
    if "height_scan" in env.observation_manager.active_terms[group_name]:
        obs[:, 48:235] = obs[:, 48:235].view(-1, 11, 17).flip(dims=[2]).view(-1, 11 * 17)

    return obs



def _transform_actions_front_back(actions: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the front-back axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with front-back symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_anymal_joints_front_back(actions[:])
    return actions


def _switch_anymal_joints_front_back(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a front-back symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)
    joint_data_switched[..., :] *= -1.0

    return joint_data_switched
