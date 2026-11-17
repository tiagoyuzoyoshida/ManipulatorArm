import numpy as np
from manipulator_arm.simulation.robot_description import DH_TABLE


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def forward_kinematics_chain(q) -> list:
    chain = []
    H = np.eye(4)

    for i, (jtype, theta_off, d_off, a, alpha) in enumerate(DH_TABLE):
        if jtype == "R":
            theta = float(q[i]) + theta_off
            d = d_off
        else:
            theta = theta_off
            d = float(q[i]) + d_off

        H = H @ dh_transform(theta, d, a, alpha)
        chain.append(H.copy())

    return chain


def forward_kinematics(q) -> np.ndarray:
    return forward_kinematics_chain(q)[-1]


def pose_to_xyz_rpy(T: np.ndarray):
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
    R = T[:3, :3]
    cos_p = np.hypot(R[0, 0], R[1, 0])

    if cos_p > 1e-9:
        yaw = np.atan2(R[1, 0], R[0, 0])
        pitch = np.atan2(-R[2, 0], cos_p)
        roll = np.atan2(R[2, 1], R[2, 2])
    else:
        yaw = 0.0
        pitch = np.copysign(np.pi / 2.0, -R[2, 0])
        roll = np.atan2(np.sign(pitch) * R[0, 1], R[1, 1])

    return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)
