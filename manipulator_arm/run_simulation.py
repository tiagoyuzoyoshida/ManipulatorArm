import argparse
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from manipulator_arm.dynamics import compute_trajectory_dynamics, plot_dynamics
from manipulator_arm.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    inverse_kinematics_position,
    pose_to_xyz_rpy,
)
from manipulator_arm.simulation import animate_trajectory, display_pose
from manipulator_arm.trajectory import generate_valve_trajectory, save_trajectory_gif


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4-DOF R-R-P-R manipulator simulation.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    fk = subparsers.add_parser("fk", help="Run Forward Kinematics.")
    fk.add_argument("--q", nargs=4, type=float, metavar=("THETA1", "THETA2", "D3", "THETA4"))
    fk.add_argument("--backend", choices=("swift", "pyplot", "none"), default="pyplot")
    fk.add_argument("--hold", action="store_true")

    ik = subparsers.add_parser("ik", help="Run Inverse Kinematics.")
    ik.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ik.add_argument("--rpy", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"))
    ik.add_argument("--theta4", type=float, default=0.0)
    ik.add_argument("--backend", choices=("swift", "pyplot", "none"), default="pyplot")
    ik.add_argument("--hold", action="store_true")

    traj = subparsers.add_parser("trajectory", help="Run Cartesian valve trajectory.")
    traj.add_argument("--backend", choices=("swift", "pyplot", "none"), default="pyplot")
    traj.add_argument("--gif", type=Path, default=Path("outputs/trajectory.gif"))
    traj.add_argument("--hold", action="store_true")

    dyn = subparsers.add_parser("dynamics", help="Run RNE inverse dynamics analysis.")
    dyn.add_argument("--out-dir", type=Path, default=Path("outputs"))
    dyn.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)

    return parser


def run_fk(q_input, backend: str, hold: bool):
    if q_input is None:
        raw = input("Enter q [theta1_deg theta2_deg d3_m theta4_deg]: ")
        q_input = [float(x) for x in raw.replace(",", " ").split()]

    q_rad = [
        math.radians(q_input[0]),
        math.radians(q_input[1]),
        float(q_input[2]),
        math.radians(q_input[3]),
    ]

    T = forward_kinematics(q_rad)
    display_pose(q_rad, backend=backend, hold=hold)

    x, y, z, r, p, yw = pose_to_xyz_rpy(T)
    print(
        f"End-effector pose: x={x:.4f} m, y={y:.4f} m, z={z:.4f} m | "
        f"roll={math.degrees(r):.2f} deg, pitch={math.degrees(p):.2f} deg, yaw={math.degrees(yw):.2f} deg"
    )


def run_ik(xyz_input, rpy_deg, theta4_deg: float, backend: str, hold: bool):
    if xyz_input is None:
        raw = input("Enter target position [x_m y_m z_m]: ")
        xyz_input = [float(x) for x in raw.replace(",", " ").split()]

    if rpy_deg is not None:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", np.radians(rpy_deg)).as_matrix()
        T[:3, 3] = xyz_input
        q_sol = inverse_kinematics(T)
    else:
        q_sol = inverse_kinematics_position(xyz_input, theta4=math.radians(theta4_deg))

    display_pose(q_sol, backend=backend, hold=hold)
    print(
        f"Joint solution: theta1={math.degrees(q_sol[0]):.2f} deg, "
        f"theta2={math.degrees(q_sol[1]):.2f} deg, "
        f"d3={q_sol[2]:.4f} m, "
        f"theta4={math.degrees(q_sol[3]):.2f} deg"
    )


def run_trajectory(backend: str, gif_path: Path, hold: bool):
    traj = generate_valve_trajectory()
    dt_s = float(traj["t_s"][1] - traj["t_s"][0])
    animate_trajectory(traj["q"], dt_s=dt_s, backend=backend, hold=hold, gripper_closed=traj["gripper_closed"])
    saved = save_trajectory_gif(traj["q"], out_path=gif_path, gripper_closed=traj["gripper_closed"])
    print(f"Trajectory completed ({len(traj['t_s'])} points). Saved GIF to: {saved}")


def run_dynamics(out_dir: Path, show: bool):
    traj = generate_valve_trajectory()
    tau = compute_trajectory_dynamics(traj)
    saved = plot_dynamics(traj, tau, out_path=out_dir / "dynamics.png", show=show)
    print(f"Saved dynamics plot to: {saved}")


def main():
    args = build_parser().parse_args()
    if args.command == "fk":
        run_fk(args.q, args.backend, args.hold)
    elif args.command == "ik":
        run_ik(args.xyz, args.rpy, args.theta4, args.backend, args.hold)
    elif args.command == "trajectory":
        run_trajectory(args.backend, args.gif, args.hold)
    elif args.command == "dynamics":
        run_dynamics(args.out_dir, args.show)
    else:
        run_trajectory("pyplot", Path("outputs/trajectory.gif"), False)


if __name__ == "__main__":
    main()
