import numpy as np


def compute_waypoint_velocities(q_list, max_vel: float):
    q_list = np.asarray(q_list, dtype=float)
    n_pts, dof = q_list.shape
    dq = np.zeros((n_pts, dof))

    for i in range(1, n_pts - 1):
        dq[i] = np.clip(0.5 * (q_list[i + 1] - q_list[i - 1]), -max_vel, max_vel)

    return dq


def polynomial_traj(qi, qf, dqi, dqf, num_points: int):
    qi = np.asarray(qi, dtype=float)
    qf = np.asarray(qf, dtype=float)
    dqi = np.asarray(dqi, dtype=float)
    dqf = np.asarray(dqf, dtype=float)
    dof = qi.shape[0]

    # 5th-order polynomial boundary matrix at s = 1
    A = np.array([
        [1.0,  1.0,  1.0],
        [3.0,  4.0,  5.0],
        [6.0, 12.0, 20.0],
    ])
    B = np.vstack([(qf - qi) - dqi, dqf - dqi, np.zeros(dof)])
    a3, a4, a5 = np.linalg.solve(A, B)

    q_list, dq_list, ddq_list = [], [], []
    for i in range(num_points):
        s = i / max(1, num_points - 1)
        q_list.append(qi + dqi * s + a3 * s**3 + a4 * s**4 + a5 * s**5)
        dq_list.append(dqi + 3.0 * a3 * s**2 + 4.0 * a4 * s**3 + 5.0 * a5 * s**4)
        ddq_list.append(6.0 * a3 * s + 12.0 * a4 * s**2 + 20.0 * a5 * s**3)

    return np.array(q_list), np.array(dq_list), np.array(ddq_list)


def smooth_jtraj(q_list: np.ndarray, gripper_list, num_points: int, max_jvel: float):
    smooth_q = []
    gripper_out = []
    dq_list = compute_waypoint_velocities(q_list, max_jvel)

    for i in range(q_list.shape[0] - 1):
        seg, _, _ = polynomial_traj(q_list[i], q_list[i + 1], dq_list[i], dq_list[i + 1], num_points)
        smooth_q.extend(seg[1:] if i > 0 else seg)

        if gripper_list is not None:
            count = num_points - 1 if i > 0 else num_points
            gripper_out.extend([gripper_list[i]] * count)

    return np.array(smooth_q), gripper_out
