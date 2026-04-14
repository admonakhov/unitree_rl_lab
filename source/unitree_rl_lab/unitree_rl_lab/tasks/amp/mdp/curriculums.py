# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
import numpy as np
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def curriculum_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_force: float,
    threshold_height: float,
    decay_rate: float = 20,  # 力衰减速率
    min_force: float = 0.0,  # 最小力值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    current_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    mean_height = torch.mean(current_heights).squeeze(-1)
    current_force = getattr(env, '_curriculum_force', max_force)

    if env.common_step_counter % env.max_episode_length == 0:
        if mean_height > threshold_height:
            force_reduction = decay_rate
            current_force = max(min_force, current_force - force_reduction)
            setattr(env, '_curriculum_force',current_force)
    return current_force

def curriculum_scale(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_scale: float,
    threshold_height: float,
    decay_scale: float = 0.05,  # 力衰减速率
    min_scale: float = 0.5,  # 最小力值
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Trunk"),
) -> None:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    current_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    mean_height = torch.mean(current_heights).squeeze(-1)

    current_scale = getattr(env, '_curriculum_scale', max_scale)
    if env.common_step_counter % env.max_episode_length == 0:

        if mean_height > threshold_height:
            force_reduction = decay_scale
            current_scale = max(min_scale, current_scale - force_reduction)
            setattr(env, '_curriculum_scale',current_scale)
    env.cfg.actions.joint_pos.scale = current_scale
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    return current_scale

def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)