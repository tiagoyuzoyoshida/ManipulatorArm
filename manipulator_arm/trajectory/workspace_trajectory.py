import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from manipulator_arm.kinematics import inverse_kinematics_position


def linear_wtraj(transforms, gripper_list, points_hz: int, linear_vel: float):
    wtraj_list = []
    gripper_out = []

    for i in range(len(transforms) - 1):
        T1, T2 = transforms[i], transforms[i + 1]
        d_pos = float(np.linalg.norm(T2.t - T1.t))
        num_points = max(2, int((d_pos / max(linear_vel, 1e-6)) * points_hz))

        seg = T1.interp(T2, num_points)
        for k in range(len(seg)):
            wtraj_list.append(seg[k])
            if gripper_list:
                gripper_out.append(gripper_list[i])

    return wtraj_list, gripper_out


def append_cartesian_segment(waypoints: list, p_start: np.ndarray, p_end: np.ndarray, theta4: float, s_profile: np.ndarray, skip_first: bool):
    poses = rtb.ctraj(SE3(p_start), SE3(p_end), s_profile)
    start_idx = 1 if skip_first else 0
    for pose in poses[start_idx:]:
        waypoints.append(inverse_kinematics_position(pose.t, theta4=theta4))
