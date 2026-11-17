from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from manipulator_arm.simulation import URDF_PATH, load_robot

_GRAVITY = [0.0, 0.0, -9.81]

_JOINT_SPECS = (
    ("Joint 1 (theta1)", True, "deg", "deg/s", "deg/s^2", "N*m"),
    ("Joint 2 (theta2)", True, "deg", "deg/s", "deg/s^2", "N*m"),
    ("Joint 3 (d3)", False, "m", "m/s", "m/s^2", "N"),
    ("Joint 4 (theta4)", True, "deg", "deg/s", "deg/s^2", "N*m"),
)
_TITLES = ("Position", "Velocity", "Acceleration", "Torque / Force")
_COLORS = ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728")


def compute_trajectory_dynamics(traj: dict, urdf_path: Path = URDF_PATH) -> np.ndarray:
    robot = load_robot(urdf_path)
    tau = robot.rne(traj["q"], traj["dq"], traj["ddq"], gravity=_GRAVITY)
    return np.asarray(tau, dtype=float)


def plot_dynamics(traj: dict, tau: np.ndarray, out_path: Path = Path("outputs/dynamics.png"), show: bool = False) -> Path:
    t_s = traj["t_s"]
    q, dq, ddq = traj["q"], traj["dq"], traj["ddq"]

    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, constrained_layout=True)

    for row, (label, is_rev, pos_u, vel_u, acc_u, tau_u) in enumerate(_JOINT_SPECS):
        scale = 180.0 / np.pi if is_rev else 1.0
        series = (
            (q[:, row] * scale, pos_u),
            (dq[:, row] * scale, vel_u),
            (ddq[:, row] * scale, acc_u),
            (tau[:, row], tau_u),
        )
        for col, (values, unit) in enumerate(series):
            ax = axes[row, col]
            ax.plot(t_s, values, color=_COLORS[col], linewidth=1.8)
            ax.grid(True, linestyle="--", alpha=0.45)
            ax.set_ylabel(f"{label}\n[{unit}]" if col == 0 else f"[{unit}]", fontsize=9)
            if row == 0:
                ax.set_title(_TITLES[col], fontsize=11, fontweight="bold")
            if row == 3:
                ax.set_xlabel("Time [s]", fontsize=9)

    fig.suptitle(
        "4-DOF R-R-P-R Manipulator Trajectory Dynamics (RNE, g = -9.81 m/s^2)",
        fontsize=13,
        fontweight="bold",
    )

    resolved_out = Path(out_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(resolved_out, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return resolved_out
