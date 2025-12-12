# Copyright (c) 2024, RoboVerse community
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Motion Capture (MoCap) tracking environment for Arcus 23DOF humanoid robot.

This configuration enables the robot to track and imitate motion capture data using:
- Motion commands from reference trajectories (BVH/CSV converted to NPZ)
- Body position and orientation tracking for key body parts
- Joint position tracking with observation noise
- Reward shaping for precise motion imitation

Usage:
    Before training, convert your motion capture data:
    python scripts/mimic/csv_to_npz.py -f path/to/motion.csv --input_fps 60
"""

from __future__ import annotations

import os
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
# Import MDP utilities for motion tracking
import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.arcus import ARCUS_A1_23DOF_CFG as ROBOT_CFG

##
# Velocity and Pose Ranges
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),      # Forward/backward velocity (m/s)
    "y": (-0.5, 0.5),      # Left/right velocity (m/s)
    "z": (-0.2, 0.2),      # Up/down velocity (m/s)
    "roll": (-0.52, 0.52),   # Roll angular velocity (rad/s)
    "pitch": (-0.52, 0.52),  # Pitch angular velocity (rad/s)
    "yaw": (-0.78, 0.78),    # Yaw angular velocity (rad/s)
}

##
# Scene definition
##
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)
@configclass
class Arcus23DofMocapSceneCfg(InteractiveSceneCfg):
    """Configuration for the mocap scene with Arcus 23DOF humanoid robot."""

    # Ground terrain with appropriate friction
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Arcus 23DOF robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # Contact sensor for tracking contacts
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command specifications for motion tracking."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        # Path to your motion file (convert with csv_to_npz.py before training)
        # Example: python scripts/mimic/csv_to_npz.py -f motion.csv --input_fps 60
        motion_file="/home/ant/UniTree/gym/retargeting/mocap/out.npz",
        anchor_body_name="torso_link",  # Main tracking reference body
        resampling_time_range=(1.0e9, 1.0e9),  # No resampling (use full motion)
        debug_vis=True,  # Visualize motion tracking in simulation

        # Pose randomization ranges
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },

        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),

        # Bodies to track (23DOF humanoid key body parts)
        body_names=[
            # Torso and pelvis
            "pelvis",
            "torso_link",
            # Left leg
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            # Right leg
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            # Left arm
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_rubber_hand",
            # Right arm
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_rubber_hand",
        ],
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Joint position control for all 23 DOF
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],  # All joints
        scale=1.0,  # Action scale (adjust based on your robot)
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for motion tracking."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network."""

        # Motion command (reference trajectory)
        motion_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"}
        )

        # Motion anchor orientation in body frame
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        # Base angular velocity
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2)
        )

        # Relative joint positions
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )

        # Relative joint velocities
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5)
        )

        # Previous action
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged observations for the critic network (asymmetric actor-critic)."""

        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"}
        )

        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"}
        )

        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"}
        )

        # Body positions in body frame
        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"}
        )

        # Body orientations in body frame
        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"}
        )

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

@configclass
class EventCfg:
    """Configuration for randomization events."""

    # Startup events
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # Interval events - push robot for robustness
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

@configclass
class RewardsCfg:
    """Reward terms for motion tracking."""

    # Regularization rewards
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7
    )

    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-5
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-1
    )

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # Motion tracking rewards
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    # Penalize unwanted contacts (exclude feet and hands)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    # Regex: all bodies except feet and hands
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_roll_rubber_hand$)(?!right_wrist_roll_rubber_hand$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if anchor position deviates too much
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )

    # Terminate if anchor orientation deviates too much
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8
        },
    )

    # Terminate if end-effector body positions deviate too much
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_roll_rubber_hand",
                "right_wrist_roll_rubber_hand",
            ],
        },
    )

##
# Environment configuration
##

@configclass
class Arcus23DofMocapEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Arcus 23DOF motion capture tracking environment."""

    # Scene settings
    scene: Arcus23DofMocapSceneCfg = Arcus23DofMocapSceneCfg(
        num_envs=4096,
        env_spacing=2.5
    )

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4  # Control frequency = 50Hz (sim at 200Hz)
        self.episode_length_s = 30.0  # Episode duration

        # Simulation settings
        self.sim.dt = 0.005  # 200 Hz physics simulation
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class Arcus23DofMocapPlayEnvCfg(Arcus23DofMocapEnvCfg):
    """Configuration for playing/visualizing trained policies."""

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4  # Control frequency = 50Hz (sim at 200Hz)
        self.episode_length_s = 30.0  # Episode duration

        # Simulation settings
        self.sim.dt = 0.005  # 200 Hz physics simulation
        # Override for single environment visualization
        self.scene.num_envs = 1
        self.episode_length_s = 1e9  # Infinite episode for visualization
