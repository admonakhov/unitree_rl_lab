from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# This is the Cart root pose expressed in the G1 pelvis frame for the nominal
# velocity-task pose.  It aligns the left hand with arm_left_link before the
# fixed joints are authored.  The right-hand joint's local frame is computed
# from the stage, so it preserves the nominal closed-chain configuration too.
_CART_ROOT_OFFSET_B = (0.3, 0.0, -0.0935066)


def _num_cart_envs(env: ManagerBasedEnv, cart_env_ratio: float) -> int:
    if not 0.0 <= cart_env_ratio <= 1.0:
        raise ValueError(f"cart_env_ratio must be in [0, 1], got {cart_env_ratio}.")
    return round(env.scene.num_envs * cart_env_ratio)

def create_hand_cart_constraints(env: ManagerBasedEnv, env_ids: torch.Tensor | None, cart_env_ratio: float):
    """Create virtual spring constraints between G1 wrists and cart handles."""
    del env_ids

    from pxr import Gf, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    num_cart_envs = _num_cart_envs(env, cart_env_ratio)
    hand_cart_pairs = (("left_wrist_yaw_link", "arm_left_link"), ("right_wrist_yaw_link", "arm_right_link"))

    for env_id in range(num_cart_envs):
        env_path = f"/World/envs/env_{env_id}"
        for wrist_name, handle_name in hand_cart_pairs:
            joint_path = f"{env_path}/CartAttachments/{wrist_name}_to_{handle_name}"
            if stage.GetPrimAtPath(joint_path).IsValid():
                continue

            wrist_path = f"{env_path}/Robot/{wrist_name}"
            handle_path = f"{env_path}/Cart/{handle_name}"
            wrist_prim = stage.GetPrimAtPath(wrist_path)
            handle_prim = stage.GetPrimAtPath(handle_path)
            if not wrist_prim.IsValid() or not handle_prim.IsValid():
                raise RuntimeError(f"Cannot create hand-cart spring:\n{wrist_path}\n{handle_path}")

            joint = UsdPhysics.Joint.Define(stage, joint_path)
            joint.CreateExcludeFromArticulationAttr().Set(True)
            joint.CreateBody0Rel().SetTargets([wrist_path])
            joint.CreateBody1Rel().SetTargets([handle_path])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, Gf.Vec3f(0, 0, 0)))

            for axis in ("transX", "transY", "transZ"):
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr("force")
                drive.CreateMaxForceAttr(30.0)
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateStiffnessAttr(300.0)
                drive.CreateDampingAttr(50.0)

            for axis in ("rotX", "rotY", "rotZ"):
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr("force")
                drive.CreateMaxForceAttr(200.0)
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateStiffnessAttr(50.0)
                drive.CreateDampingAttr(10.0)

def filter_non_hand_robot_cart_collisions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    cart_env_ratio: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cart_cfg: SceneEntityCfg = SceneEntityCfg("cart"),
):
    """Disable direct robot-cart contacts; the D6 grasp drives provide the hand coupling."""
    del env_ids, robot_cfg, cart_cfg

    from pxr import Usd, UsdPhysics

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for env_id in range(_num_cart_envs(env, cart_env_ratio)):
        env_path = f"/World/envs/env_{env_id}"
        robot_path = f"{env_path}/Robot"
        cart_path = f"{env_path}/Cart"
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
            filtered_pairs.AddTarget(cart_path)


def reset_cart_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    cart_env_ratio: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cart_cfg: SceneEntityCfg = SceneEntityCfg("cart"),
):
    """Reset attached carts with their robots and park carts in no-cart environments."""
    robot: Articulation = env.scene[robot_cfg.name]
    cart: Articulation = env.scene[cart_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    assert env_ids is not None

    root_state = cart.data.default_root_state[env_ids].clone()
    cart_mask = env_ids < _num_cart_envs(env, cart_env_ratio)

    if torch.any(cart_mask):
        cart_ids = env_ids[cart_mask]
        offset_b = torch.tensor(_CART_ROOT_OFFSET_B, device=env.device).expand(cart_ids.numel(), -1)
        root_state[cart_mask, :3] = robot.data.root_pos_w[cart_ids] + quat_apply(robot.data.root_quat_w[cart_ids], offset_b)
        root_state[cart_mask, 3:7] = robot.data.root_quat_w[cart_ids]
        root_state[cart_mask, 7:13] = robot.data.root_vel_w[cart_ids]

    if torch.any(~cart_mask):
        no_cart_ids = env_ids[~cart_mask]
        root_state[~cart_mask, :3] = env.scene.env_origins[no_cart_ids]
        root_state[~cart_mask, 2] -= 10.0
        root_state[~cart_mask, 3:7] = 0.0
        root_state[~cart_mask, 3] = 1.0
        root_state[~cart_mask, 7:13] = 0.0

    cart.write_root_state_to_sim(root_state, env_ids)
