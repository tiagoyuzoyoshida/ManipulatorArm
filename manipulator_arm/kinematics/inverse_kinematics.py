import numpy as np
from manipulator_arm.simulation.robot_description import JOINT_LIMITS, l1, l2, l4, l5
from .forward_kinematics import forward_kinematics_chain


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def within_limits(q) -> bool:
    for val, (q_min, q_max) in zip(q, JOINT_LIMITS):
        if val < q_min - 1e-6 or val > q_max + 1e-6:
            return False
    return True


def inverse_kinematics_position(pos, theta4: float = 0.0) -> np.ndarray:
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])

    r = np.hypot(px, py)
    zp = pz - l1
    l_eff = np.hypot(r, zp)
    d3 = l_eff - (l2 + l4 + l5)

    # Front and back reach solutions
    candidates = [
        np.array([np.atan2(py, px), wrap_pi(np.atan2(zp, r) - np.pi / 3.0), d3, theta4]),
        np.array([np.atan2(-py, -px), wrap_pi(np.atan2(zp, -r) - np.pi / 3.0), d3, theta4]),
    ]

    for q_sol in candidates:
        if within_limits(q_sol):
            return q_sol

    raise ValueError("Target position is outside joint limits.")


def inverse_kinematics(T: np.ndarray) -> np.ndarray:
    q_pos = inverse_kinematics_position(T[:3, 3], theta4=0.0)

    H_0_3 = forward_kinematics_chain(q_pos)[2]
    F = np.linalg.inv(H_0_3) @ T
    theta4 = np.atan2(F[1, 0], F[1, 1])

    q_sol = np.array([q_pos[0], q_pos[1], q_pos[2], theta4])
    if not within_limits(q_sol):
        raise ValueError("Target pose is outside joint limits.")
    return q_sol


def geometric_jacobian(q) -> np.ndarray:
    H_0_1, H_0_2, H_0_3, H_0_4 = forward_kinematics_chain(q)

    z0 = np.array([0.0, 0.0, 1.0])
    z1 = H_0_1[:3, 2]
    z2 = H_0_2[:3, 2]
    z3 = H_0_3[:3, 2]

    p1 = H_0_1[:3, 3]
    p3 = H_0_3[:3, 3]
    p4 = H_0_4[:3, 3]

    Jv = np.column_stack([
        np.cross(z0, p4),
        np.cross(z1, p4 - p1),
        z2,
        np.cross(z3, p4 - p3),
    ])
    Jw = np.column_stack([z0, z1, np.zeros(3), z3])

    return np.vstack([Jv, Jw])
