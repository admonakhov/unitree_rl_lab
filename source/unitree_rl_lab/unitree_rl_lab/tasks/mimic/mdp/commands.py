from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Support single motion file or list of motion files
        motion_files = self.cfg.motion_file if isinstance(self.cfg.motion_file, (list, tuple)) else [self.cfg.motion_file]
        self.motions = [MotionLoader(f, self.body_indexes, device=self.device) for f in motion_files]
        # Per-env selected motion index
        if len(self.motions) == 1:
            self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            # motion assignment strategy: 'round_robin' or 'random'
            if getattr(self.cfg, "motion_assignment", "round_robin") == "random":
                self.motion_ids = torch.randint(len(self.motions), (self.num_envs,), device=self.device)
            else:
                # deterministic round-robin distribution across envs so each env uses one mocap
                self.motion_ids = (torch.arange(self.num_envs, device=self.device) % len(self.motions)).long()

        # Per-env time step within chosen motion
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # helper: motion lengths
        self.motion_lengths = torch.tensor([m.time_step_total for m in self.motions], device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Use maximum motion length to determine binning for adaptive sampling
        max_motion_len = int(self.motion_lengths.max().item())
        self.bin_count = int(max_motion_len // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        # Smoothed velocity command (for optional EMA smoothing)
        self.velocity_command_smoothed = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        # Gather joint positions per-env from assigned motions
        return self._gather_motion_attr("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._gather_motion_attr("joint_vel")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._gather_motion_attr("body_pos_w") + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._gather_motion_attr("body_quat_w")

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._gather_motion_attr("body_lin_vel_w")

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._gather_motion_attr("body_ang_vel_w")

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        # select anchor body position per env
        bp = self._gather_motion_attr("body_pos_w")
        return bp[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        bq = self._gather_motion_attr("body_quat_w")
        return bq[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        bl = self._gather_motion_attr("body_lin_vel_w")
        return bl[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        ba = self._gather_motion_attr("body_ang_vel_w")
        return ba[:, self.motion_anchor_body_index]

    def _gather_motion_attr(self, attr_name: str) -> torch.Tensor:
        """Assemble per-env tensor for given motion attribute (e.g. 'joint_pos', 'body_pos_w').

        Each MotionLoader stores data as [T, ...]. This returns a tensor shaped
        (num_envs, ...) where each env pulls from its assigned motion at
        the per-env time index in `self.time_steps`.
        """
        out = None
        device = self.device
        for i, m in enumerate(self.motions):
            idxs = (self.motion_ids == i).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            vals = getattr(m, attr_name)[self.time_steps[idxs]]
            if out is None:
                out = torch.zeros((self.num_envs,) + vals.shape[1:], device=device, dtype=vals.dtype)
            out[idxs] = vals
        if out is None:
            # no envs assigned? return empty tensor shaped for num_envs
            # attempt to construct from first motion's attr
            sample = getattr(self.motions[0], attr_name)
            out = torch.zeros((self.num_envs,) + sample.shape[1:], device=device, dtype=sample.dtype)
        return out

    def _gather_motion_initial(self, attr_name: str) -> torch.Tensor:
        """Return per-env initial (t=0) attribute from assigned motions."""
        out = None
        device = self.device
        for i, m in enumerate(self.motions):
            idxs = (self.motion_ids == i).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            vals = getattr(m, attr_name)[0]
            if out is None:
                out = torch.zeros((self.num_envs,) + vals.shape, device=device, dtype=vals.dtype)
            out[idxs] = vals.unsqueeze(0).expand(len(idxs), *vals.shape)
        if out is None:
            sample = getattr(self.motions[0], attr_name)[0]
            out = torch.zeros((self.num_envs,) + sample.shape, device=device, dtype=sample.dtype)
        return out

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # per-env motion lengths
            lengths_per_env = self.motion_lengths[self.motion_ids]
            current_bin_index = torch.clamp((self.time_steps * self.bin_count) // lengths_per_env.clamp(min=1), 0, self.bin_count - 1)
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # compute per-env motion lengths for selected envs and map sampled bin to time_steps
        lengths = self.motion_lengths[self.motion_ids[env_ids]]
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)) / self.bin_count * (lengths - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        # detect envs that reached the end of their assigned motion
        lengths_per_env = self.motion_lengths[self.motion_ids]
        env_ids = torch.where(self.time_steps >= lengths_per_env)[0]
        self._resample_command(env_ids)
        # if len(env_ids) > 0:
        #     # reset time to start of the same motion (disable resampling/mixing)
        #     self.time_steps[env_ids] = 0

        # Get current mocap data (assembled per-env across motions)
        body_pos_w = self.body_pos_w
        body_quat_w = self.body_quat_w
        body_lin_vel_w = self.body_lin_vel_w
        body_ang_vel_w = self.body_ang_vel_w

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        original_delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        if self.cfg.velocity_command_name is not None:
            velocity_commands = self._env.command_manager.get_command(self.cfg.velocity_command_name)
            # With probability zero_command_prob, set commands to zero for balance training
            is_zero_command = torch.rand(1, device=self.device) < self.cfg.zero_command_prob
            if is_zero_command:
                velocity_commands.zero_()
                # reset smoothed velocity when forcing zero commands
                self.velocity_command_smoothed.zero_()
            elif self.cfg.set_velocity_command:
                local_lin_vel = quat_apply(quat_inv(self.anchor_quat_w), self.anchor_lin_vel_w)
                target_vel = torch.stack(
                    (local_lin_vel[:, 0], local_lin_vel[:, 1], self.anchor_ang_vel_w[:, 2]), dim=1
                )

                # root_lin_vel = body_lin_vel_w[:, 0]
                # root_ang_vel = body_ang_vel_w[:, 0]
                # local_lin_vel = quat_apply(quat_inv(self.anchor_quat_w), root_lin_vel)
                # local_ang_vel = quat_apply(quat_inv(self.anchor_quat_w), root_ang_vel)
                # target_vel = torch.stack((local_lin_vel[:, 0], local_lin_vel[:, 1], local_ang_vel[:, 2]), dim=1)


                self.velocity_command_smoothed = (
                    self.cfg.velocity_smoothing_alpha * self.velocity_command_smoothed
                    + (1.0 - self.cfg.velocity_smoothing_alpha) * target_vel
                )

                velocity_commands[:, 0] = torch.trunc(self.velocity_command_smoothed[:, 0] * 25) / 25
                velocity_commands[:, 1] = torch.trunc(self.velocity_command_smoothed[:, 1] * 10) / 10
                velocity_commands[:, 2] = torch.trunc(self.velocity_command_smoothed[:, 2] * 5) / 5
                
            # Use original delta orientation (frames oriented relative to robot's current pose)
            delta_ori_w = original_delta_ori_w
        else:
            is_zero_command = torch.tensor(False, device=self.device)
            delta_ori_w = original_delta_ori_w

        # If zero command, use initial pose from mocap
        if is_zero_command:
            initial_body_pos = self._gather_motion_initial("body_pos_w")
            initial_body_quat = self._gather_motion_initial("body_quat_w")
            initial_joint_pos = self._gather_motion_initial("joint_pos")
            initial_joint_vel = self._gather_motion_initial("joint_vel")
            initial_anchor_pos_w = initial_body_pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins
            initial_anchor_quat_w = initial_body_quat[:, self.motion_anchor_body_index]
            initial_body_pos_w = initial_body_pos + self._env.scene.env_origins[:, None, :]
            initial_body_quat_w = initial_body_quat

            # Override with initial values
            anchor_pos_w_repeat = initial_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
            anchor_quat_w_repeat = initial_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
            body_pos_w = initial_body_pos_w
            body_quat_w = initial_body_quat_w
            body_lin_vel_w = torch.zeros_like(body_lin_vel_w)
            body_ang_vel_w = torch.zeros_like(body_ang_vel_w)
            joint_pos = initial_joint_pos
            joint_vel = torch.zeros_like(initial_joint_vel)

        self.body_quat_relative_w = quat_mul(delta_ori_w, body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | list[str] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    velocity_command_name: str | None = None  # Name of velocity command to align motion direction with
    set_velocity_command: bool = False  # If True, set the velocity command to match the current mocap velocity
    zero_command_prob: float = 0.0  # Probability of setting velocity commands to zero (for balance training)
    velocity_smoothing_alpha: float = 0.975  # EMA alpha for smoothing velocity commands (0..1)
    motion_assignment: str = "round_robin"  # 'round_robin' or 'random' per-env motion file assignment

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
