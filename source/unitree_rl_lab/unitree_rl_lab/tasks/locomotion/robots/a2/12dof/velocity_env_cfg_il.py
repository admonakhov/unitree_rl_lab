# unitree_23dof_amp_velocity_env_cfg.py
"""
Configuration for Unitree-Arcus-A1 locomotion environment with AMP (Adversarial Motion Priors).
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

from unitree_rl_lab.assets.robots.arcus import ARCUS_A2_RET_CFG as ROBOT_CFG
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

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

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
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 1.0),
            "restitution_range": (0.0, 0.0), # Упругость столкновений
            "num_buckets": 64,
        },
    )

    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    # add_joint_default_pos = EventTerm(
    #     func=mimic_mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    #         "pos_distribution_params": (-0.1, 0.1),
    #         "operation": "add",
    #     },
    # )

    base_com = EventTerm(
        func=mimic_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.075, 0.075), "y": (-0.1, 0.1), "z": (-0.1, 0.05)},
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mimic_mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )



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
            lin_vel_x=(-0., 0),
            lin_vel_y=(-0, 0),
            ang_vel_z=(-0, 0),
            heading=(-0, 0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1, 1),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-3.14, 3.14),
        ),
    )

    # Motion command from mocap data for AMP-style regularization
    motion = MotionCommandCfg(
        asset_name="robot",
        # Path to motion file (convert CSV to NPZ before training)
        # CSV file location: /home/ant/UniTree/gym/retargeting/mocap/walking_60fps.csv
        # To convert, run: python scripts/mimic/csv_to_npz_arcus.py -f /home/ant/UniTree/gym/retargeting/mocap/walking_60fps.csv --input_fps 60 --output_name /home/ant/UniTree/gym/unitree_rl_lab/poses/a1_23dof/walking_60fps.npz
        motion_file=[
                    "mocap/arcus2/fast_forward.npz",
                    "mocap/arcus2/arc1_fast.npz",
                    "mocap/arcus2/arc2_fast.npz",
                    "mocap/arcus2/arc1.npz",
                    "mocap/arcus2/arc2.npz",
                    "mocap/arcus2/walk1.npz",
                    # "mocap/arcus2/walk2.npz",
                    "mocap/arcus2/stand.npz",
                    "mocap/arcus2/the_slowest.npz",
                    ],

        velocity_smoothing_alpha = 0.97,
        velocity_factor = 1.1,
        motion_assignment = 'round_robin', # round_robin or random
        anchor_body_name = "torso_link",

        adaptive_alpha = 0.000,
        adaptive_uniform_ratio = 1,
        # threshold_velocity_cmd = 0.16,
        resampling_time_range=(50.0, 1000.0),  # Enable resampling to allow motion changes with velocity commands
        debug_vis=True,
        
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.0),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.0, 0.0),  # Increased from (-0.1, 0.1) for more directional variety
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
            "torso_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            # "left_shoulder_roll_link",
            # "left_elbow_link",
            # "left_wrist_roll_link",
            # "right_shoulder_roll_link",
            # "right_elbow_link",
            # "right_wrist_roll_link",
        ],
    )


# -----------------------
# Actions
# -----------------------
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], 
        scale=0.15,
        use_default_offset=True
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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.5, n_max=0.5))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        command = ObsTerm(func=mimic_mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mimic_mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mimic_mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

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
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)



    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3)
    # -- tracking
    motion_global_anchor_pos = RewTerm(
        func=mimic_mdp.motion_global_anchor_position_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mimic_mdp.motion_global_anchor_orientation_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mimic_mdp.motion_relative_body_position_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mimic_mdp.motion_relative_body_orientation_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mimic_mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mimic_mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.25,
        params={"command_name": "motion", "std": 3.14},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

# -----------------------
# Terminations
# -----------------------
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.4})

# -----------------------
# Curriculum
# -----------------------
# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     # Disabled terrain curriculum since using only flat terrain
#     # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
#     # lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
#     pass


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
    curriculum = None
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0

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