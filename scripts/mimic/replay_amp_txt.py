"""This script demonstrates how to replay K1 robot motions from npz files.

.. code-block:: bash

    # Usage - Direct file path
    python replay_npz.py --motion <path_to_motion.npz>
    
    # Usage - From wandb registry
    python replay_npz.py --registry_name <wandb_registry_name>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import pickle
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion", type=str, default=None, help="Path to the motion txt file.")
parser.add_argument("--motion_pkl", type=str, default=None, help="Path to the motion pkl file to convert and replay.")
parser.add_argument("--fps", type=float, default=30.0, help="Target frames per second for replay.")
parser.add_argument("--save_path", type=str, default=None, help="Path to save the converted txt file")
parser.add_argument("--robot", choices=["booster_t1","booster_k1"], default="booster_t1", help="Which robot do you want to retarget")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from rsl_rl.utils import AMPLoaderDisplay
from scipy.spatial.transform import Rotation
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)

##
# Pre-defined configs
##
from unitree_rl_lab.assets.robots.arcus import ARCUS_A2_RET_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.amp.mdp.commands import MotionLoader

# Some numpy builds used in Isaac Sim don't expose the `numpy._core` package
# which can break unpickling of numpy arrays created elsewhere.
try:
    import numpy.core as _np_core

    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)
except Exception:
    pass


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion_file = None
    if args_cli.motion_pkl:
        if not os.path.isfile(args_cli.motion_pkl):
            raise FileNotFoundError(f"Motion pkl not found: {args_cli.motion_pkl}")
        if args_cli.save_path:
            motion_file = args_cli.save_path
        else:
            pkl_base, _ = os.path.splitext(args_cli.motion_pkl)
            motion_file = f"{pkl_base}.txt"
        convert_pkl_to_custom(args_cli.motion_pkl, motion_file, args_cli.fps)
    elif args_cli.motion:
        motion_file = args_cli.motion
    else:
        raise ValueError("Provide --motion or --motion_pkl.")

    if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    # Load amp txt 

    frame_cnt = 0
    # AMPLoaderDisplay and AMPLoader should be fixed num_joints !!!
    amp_loader_display = AMPLoaderDisplay(
        motion_files=[motion_file], device=scene.device, time_between_frames=sim_dt
    )

    print(amp_loader_display)
    motion_len = amp_loader_display.trajectory_num_frames[0]
    print(f"Loaded motion with {motion_len} frames from {motion_file}")
    print(robot.joint_names)
    # find elbow and ankle
    elbow_body_ids, _ = robot.find_bodies(name_keys=["left_elbow_link", "right_elbow_link"], preserve_order=True)
    feet_body_ids, _ = robot.find_bodies(name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
    left_arm_local_vec = torch.tensor([0.0, 0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))
    right_arm_local_vec = torch.tensor([0.0, -0.2, 0.0], device=scene.device).repeat((scene.num_envs, 1))

    dof_pos = torch.zeros((scene.num_envs, robot.num_joints), device=scene.device)
    dof_vel = torch.zeros((scene.num_envs, robot.num_joints), device=scene.device)
    root_state = torch.zeros((scene.num_envs, 13), device=scene.device)
    # amp_expert_frames
    all_frames = []
    # Simulation loop
    while simulation_app.is_running():
        while True:
            time = (frame_cnt % (motion_len)) * (1.0/args_cli.fps)
            visual_motion_frame = amp_loader_display.get_full_frame_at_time(0, time)
            dof_pos[:] = reorder(visual_motion_frame[6:6 + robot.num_joints])

            # print(len(visual_motion_frame))

            dof_vel[:] = reorder(visual_motion_frame[6 + 6 + robot.num_joints:6 + 6 + robot.num_joints + robot.num_joints])
            robot.write_joint_position_to_sim(dof_pos)
            robot.write_joint_velocity_to_sim(dof_vel)

            # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
            root_pos = visual_motion_frame[:3].clone()
            # ????
            root_pos[2] += 0.05
            euler = visual_motion_frame[3:6].cpu().numpy()
            quat_xyzw = Rotation.from_euler("XYZ", euler, degrees=False).as_quat()  # [x, y, z, w]
            quat_wxyz = torch.tensor(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=scene.device
            )
            lin_vel = visual_motion_frame[6 + robot.num_joints: 6 + robot.num_joints + 3].clone()
            ang_vel = torch.zeros_like(lin_vel)
            root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (scene.num_envs, 1))
            root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (scene.num_envs, 1))
            root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (scene.num_envs, 1))
            root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (scene.num_envs, 1))

            # find hand and feet pos
            # hand
            left_hand_pos = (robot.data.body_state_w[:, elbow_body_ids[0], :3] - robot.data.root_state_w[:, 0:3] + quat_apply(robot.data.body_state_w[:, elbow_body_ids[0], 3:7], left_arm_local_vec))
            right_hand_pos = (robot.data.body_state_w[:, elbow_body_ids[1], :3] - robot.data.root_state_w[:, 0:3] + quat_apply(robot.data.body_state_w[:, elbow_body_ids[1], 3:7], right_arm_local_vec))
            left_hand_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_hand_pos)
            right_hand_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_hand_pos)

            # foot
            left_foot_pos = (robot.data.body_state_w[:, feet_body_ids[0], :3] - robot.data.root_state_w[:, 0:3])
            right_foot_pos = (robot.data.body_state_w[:, feet_body_ids[1], :3] - robot.data.root_state_w[:, 0:3])
            left_foot_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), left_foot_pos)
            right_foot_pos = quat_apply(quat_conjugate(robot.data.root_state_w[:, 3:7]), right_foot_pos)

            # concate
            frame = torch.cat([dof_pos,dof_vel,left_hand_pos,right_hand_pos,left_foot_pos,right_foot_pos],dim=-1)
            if args_cli.save_path:
                frame = frame.cpu().numpy().reshape(-1)
                all_frames.append(frame)
            robot.write_root_state_to_sim(root_state)
            scene.write_data_to_sim()
            sim.render()  # We don't want physic (sim.step())
            scene.update(sim_dt)

            pos_lookat = root_state[0, :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
            frame_cnt += 1
            if frame_cnt >= (motion_len - 1):  
                break
        break
    if args_cli.save_path:
        all_frames_np = np.stack(all_frames, axis=0)
        np.savetxt(args_cli.save_path, all_frames_np, fmt='%f', delimiter=', ')

        with open(args_cli.save_path, 'r') as f:
            frames_data = f.readlines()

        frames_data_len = len(frames_data)
        with open(args_cli.save_path, 'w') as f:
            f.write('{\n')
            f.write('"LoopMode": "Wrap",\n')
            f.write(f'"FrameDuration": {1.0 / args_cli.fps:.3f},\n')
            f.write('"EnableCycleOffsetPosition": true,\n')
            f.write('"EnableCycleOffsetRotation": true,\n')
            f.write('"MotionWeight": 0.5,\n\n')
            f.write('"Frames":\n[\n')

            for i, line in enumerate(frames_data):
                line_start_str = '  ['
                if i == frames_data_len - 1:
                    f.write(line_start_str + line.rstrip() + ']\n')
                else:
                    f.write(line_start_str + line.rstrip() + '],\n')

            f.write(']\n}')

        print(f"✅ Successfully converted to {args_cli.save_path}")


def reorder(motion):
    idx = [0, 6, 12, 1, 7, 16, 13, 14, 15, 17, 21, 18, 19, 20, 22, 2, 8, 3, 9, 4, 10, 5, 11]
    return motion[idx]


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)
    print(motion_data.keys())
    idx = [0, 6, 12, 1, 7, 16, 13, 14, 15, 17, 21, 18, 19, 20, 22, 2, 8, 3, 9, 4, 10, 5, 11]
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]
    # dof_pos = dof_pos[:, idx]
    print(dof_pos.shape)

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

    q1_conj = quat_conjugate(root_rot_t[:-1])         
    dq = quat_mul(q1_conj, root_rot_t[1:])            
    axis_angle = axis_angle_from_quat(dq)             
    root_ang_vel = axis_angle / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    print(root_pos.shape, euler_angles.shape, dof_pos[:-1].shape, root_lin_vel.shape, root_ang_vel.shape, dof_vel.shape)
    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1, :23],  
         root_lin_vel, root_ang_vel, dof_vel[:, :23]),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")



def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
