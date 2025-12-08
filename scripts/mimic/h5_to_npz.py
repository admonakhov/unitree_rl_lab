from isaaclab.app import AppLauncher

import torch

app_launcher = AppLauncher()

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp


import h5py
import numpy as np
import torch
from pathlib import Path


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
        rotations: shape (B, 4).
        dt: time step.
    Returns:
        shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
    return omega

def read_data(path):
    out = {}

    with h5py.File(path, 'r') as f:
        keypoints_output = f
        data = keypoints_output['persons']['1']
        for key in data.keys():
            out[key] = np.array(data[key])

    return out

def compare_data(df:dict, fps=15):
    out = {'fps':fps}
    struct = {'joint_pos':'joints', 
             'body_pos_w':'link_pos',
             'body_quat_w':'link_quat',}
    for key in struct.keys():
        out[key] = torch.tensor(df[struct[key]])
    
    out['joint_vel'] = torch.gradient(out['joint_pos'], spacing=1/fps, dim=0)[0]
    out['body_lin_vel_w'] = torch.gradient(out['body_pos_w'], spacing=1/fps, dim=0)[0]
    out['body_ang_vel_w'] = _so3_derivative(out['body_quat_w'], dt=1/fps)
    return out


if __name__ == '__main__':
    path = Path('poses/a1_23dof/w2.h5')
    data = read_data(path)
    data = compare_data(data)
    np.savez(path.parent/(path.stem+'.npz'), **data)
    