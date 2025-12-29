"""
Configuration for Unitree-g1-29dof locomotion environment with AMP (Adversarial Motion Priors).
Combines velocity commands for locomotion with motion tracking from mocap data to ensure natural motion.
"""
import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.arcus import LEG_JOINT_NAMES
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.mimic import mdp as mimic_mdp
from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommandCfg

# -----------------------
# Terrain generator
# -----------------------
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),  # Only flat terrain for training
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.0, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
    },
)

# -----------------------
# Scene
# -----------------------
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
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

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # height ray scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # сенсоры контактов
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # освещение
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# -----------------------
# Events (randomization, reset, pushes)
# -----------------------
@configclass
class EventCfg:
    """Configuration for events."""

    # startup randomize physics materials on robot bodies
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add a bit of base mass variation (startup)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset: external force/torque
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # reset base pose и скорость
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # reset joints
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # периодические толчки
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# -----------------------
# Commands
# -----------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Velocity command for locomotion
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        # allow more frequent resampling to improve responsiveness when turning
        resampling_time_range=(5.0, 30.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.2,
        # enable heading command so policy receives/uses target heading
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.6, 3.7),
            lin_vel_y=(-4.2, 3.9),
            ang_vel_z=(-9.0, 9.5),
            heading=(-3.14159, 3.14159),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.6, 3.7),
            lin_vel_y=(-4.2, 3.9),
            ang_vel_z=(-9.0, 9.5),
            heading=(-3.14159, 3.14159),
        ),
    )

    # Motion command from mocap data for AMP-style regularization
    motion = MotionCommandCfg(
        asset_name="robot",
        # Path to motion file (convert CSV to NPZ before training)
        # CSV file location: /home/ant/UniTree/gym/retargeting/mocap/walking_60fps.csv
        # To convert, run: python scripts/mimic/csv_to_npz_arcus.py -f /home/ant/UniTree/gym/retargeting/mocap/walking_60fps.csv --input_fps 60 --output_name /home/ant/UniTree/gym/unitree_rl_lab/poses/a1_23dof/walking_60fps.npz
        motion_file="mocap/walk12_g1_29 40.npz",
        anchor_body_name="torso_link",
        resampling_time_range=(10.0, 30.0),  # Enable resampling to allow motion changes with velocity commands
        debug_vis=True,
        # Pose randomization ranges - increased yaw for different directions
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.5, 0.5),  # Increased from (-0.1, 0.1) for more directional variety
        },
        velocity_range={
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "z": (-0.1, 0.1),
            "roll": (-0.26, 0.26),
            "pitch": (-0.26, 0.26),
            "yaw": (-1.0, 1.0),  # Increased from (-0.39, 0.39) for rotational variety
        },
        joint_position_range=(-0.02, 0.02),  # Further reduced for better arm tracking
        velocity_command_name="base_velocity",  # Link motion direction to velocity command for directional consistency
        set_velocity_command=True,  # Set the velocity command to match the current mocap velocity

        # Bodies to track
        body_names=[
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )


# -----------------------
# Actions
# -----------------------
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


# -----------------------
# Observations
# -----------------------
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Velocity command observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        # Note: motion_command and motion_anchor_ori_b are removed from policy observations
        # to avoid requiring motion data during deployment. They remain in critic observations
        # for AMP discriminator training during learning.

        def __post_init__(self):
            self.history_length = 10
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        # Note: Motion observations removed from critic to avoid deploy issues.
        # Motion command is still used in rewards for AMP training, but not needed
        # in observations during deployment. AMP discriminator training uses motion
        # data directly from the motion command, not through observations.

        def __post_init__(self):
            self.history_length = 10

    # privileged observations
    critic: CriticCfg = CriticCfg()


# -----------------------
# Rewards
# -----------------------
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Velocity tracking task rewards (adjusted since motion tracking is now stronger)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,  # Slightly reduced to balance with stronger motion tracking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # stronger angular velocity tracking to improve turning behavior
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- AMP-style motion tracking rewards (regularize motion to be natural)
    # Adjusted weights since motion now resamples more frequently
    # Anchor body tracking (torso) - critical for AMP
    motion_global_anchor_pos = RewTerm(
        func=mimic_mdp.motion_global_anchor_position_error_exp,
        weight=1.0,  # Anchor position is important for motion quality
        params={"command_name": "motion", "std": 0.25},  # Tighter tracking
    )
    motion_global_anchor_ori = RewTerm(
        func=mimic_mdp.motion_global_anchor_orientation_error_exp,
        weight=1.0,  # Anchor orientation is important
        params={"command_name": "motion", "std": 0.35},  # Tighter tracking
    )
    # Body relative tracking - for natural body coordination (includes arms)
    # Increased weights back for arms since arm movement is critical
    motion_body_pos = RewTerm(
        func=mimic_mdp.motion_relative_body_position_error_exp,
        weight=0.5,  # Increased back to prioritize arm movement
        params={"command_name": "motion", "std": 0.25},  # Tighter std for better arm tracking
    )
    motion_body_ori = RewTerm(
        func=mimic_mdp.motion_relative_body_orientation_error_exp,
        weight=0.5,  # Increased back to prioritize arm movement
        params={"command_name": "motion", "std": 0.25},  # Tighter std for better arm tracking
    )
    # Velocity tracking - for smooth motion
    motion_body_lin_vel = RewTerm(
        func=mimic_mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.8,  # Slightly increased
        params={"command_name": "motion", "std": 0.5},  # Tighter std
    )
    motion_body_ang_vel = RewTerm(
        func=mimic_mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.8,  # Slightly increased
        params={"command_name": "motion", "std": 1.5},  # Tighter std
    )

    # -- Specific arm motion tracking rewards (critical for natural arm swing)
    # Separate rewards for arm bodies to ensure strong arm movement replication
    arm_motion_pos = RewTerm(
        func=mimic_mdp.motion_relative_body_position_error_exp,
        weight=2.0,  # High weight for arm position tracking
        params={
            "command_name": "motion",
            "std": 0.1,  # Very tight tracking for arms
            "body_names": [
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_roll_rubber_hand",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_roll_rubber_hand",
            ],
        },
    )
    arm_motion_ori = RewTerm(
        func=mimic_mdp.motion_relative_body_orientation_error_exp,
        weight=2.0,  # High weight for arm orientation tracking
        params={
            "command_name": "motion",
            "std": 0.2,  # Tight tracking for arms
            "body_names": [
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_roll_rubber_hand",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_roll_rubber_hand",
            ],
        },
    )

    # -- Regularization rewards
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base penalties
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # joint penalties
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # reduce action-rate penalty so larger corrective actions are not overly discouraged
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # Arm movement rewards - encourage natural arm swing during walking
    # Reward arm joint velocity to encourage arm movement

    # joint deviation
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_.*_joint",
    #                 ".*_elbow_joint",
    #                 ".*_wrist_roll_joint",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"]),
    #     },
    # )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # joint_deviation_feet = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint"])},
    # )

    # orientation + height: penalize large body tilt and incorrect base height
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # base_height = RewTerm(func=mdp.base_height_l2, weight=-0.5, params={"target_height": 0.30})

    # feet-related rewards
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=1,
        params={
            "period": 1.2,
            "offset": [0.0, 0.5],
            "threshold": 0.5,
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.50,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )


# -----------------------
# Terminations
# -----------------------
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


# -----------------------
# Curriculum
# -----------------------
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Disabled terrain curriculum since using only flat terrain
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)


# -----------------------
# Полная конфигурация среды
# -----------------------
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment with AMP."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # ensure enough patches for many envs
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # enable curriculum in terrain generator если есть terrain_levels
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


# -----------------------
# Play config (меньше envs для отладки)
# -----------------------
@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-3.6, 3.7), lin_vel_y=(-4.2, 3.9), ang_vel_z=(-9.0, 9.5), heading=(-3.14159, 3.14159),
        )
        self.commands.base_velocity.ranges = limit_ranges

