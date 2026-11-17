from pathlib import Path
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3

from .robot_description import (
    ASSETS_DIR,
    ROBOT,
    URDF_PATH,
    base_floor_offset,
    load_robot,
    qz,
)

_OFFSCREEN_POSE = SE3(0.0, 0.0, -10.0).A
_VALVE_TOOL_OFFSET = 0.065


class SimulationCfg:
    dt: float = 0.05
    robot: rtb.Robot = ROBOT
    gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
    initial_q: np.ndarray = None


class Simulation:
    def __init__(self, cfg: SimulationCfg = None):
        cfg = cfg or SimulationCfg()
        self._robot = cfg.robot or load_robot()
        self._dt = cfg.dt
        self._gravity = cfg.gravity
        self._q0 = qz.copy() if cfg.initial_q is None else np.asarray(cfg.initial_q, dtype=float)
        self._robot_state = {
            "q": self._q0.copy(),
            "qd": np.zeros(4),
            "qdd": np.zeros(4),
            "tau": np.zeros(4),
        }

    def step(self, q_des: np.ndarray, qd_des: np.ndarray = None, qdd_des: np.ndarray = None):
        q = np.asarray(q_des, dtype=float)
        qd = np.zeros(4) if qd_des is None else np.asarray(qd_des, dtype=float)
        qdd = np.zeros(4) if qdd_des is None else np.asarray(qdd_des, dtype=float)

        tau = np.asarray(self._robot.rne(q, qd, qdd, gravity=list(self._gravity)), dtype=float).reshape(4)

        self._robot_state["q"] = q
        self._robot_state["qd"] = qd
        self._robot_state["qdd"] = qdd
        self._robot_state["tau"] = tau
        return self._robot_state


def display_pose(q, backend: str = "pyplot", hold: bool = False, urdf_path: Path = URDF_PATH):
    robot = load_robot(urdf_path)
    robot.base = SE3(0.0, 0.0, base_floor_offset)
    robot.q = np.asarray(q, dtype=float)

    if backend == "none":
        return robot

    env = rtb.backends.load_backend(backend)
    env.launch()
    env.add(robot)
    env.step(0.05)
    if hold:
        env.hold()
    else:
        env.close()
    return robot


def _setup_swift_extras(env, robot, q_arr, gripper_mask):
    robot.link_dict["ee_link"].geometry.clear()
    meshes_dir = ASSETS_DIR / "meshes"

    open_mesh = sg.Mesh(str((meshes_dir / "ee_gripper_open.stl").resolve()), color=(0.90, 0.62, 0.12, 1.0))
    closed_mesh = sg.Mesh(str((meshes_dir / "ee_gripper_closed.stl").resolve()), color=(0.90, 0.62, 0.12, 1.0))
    valve_mesh = sg.Mesh(str((meshes_dir / "valve_wheel.stl").resolve()), color=(0.88, 0.32, 0.12, 1.0))

    grasp_indices = np.flatnonzero(gripper_mask)
    first_grasp = int(grasp_indices[0]) if grasp_indices.size > 0 else 0
    t_valve = np.asarray(robot.fkine(q_arr[first_grasp]).A) @ SE3(0.0, 0.0, _VALVE_TOOL_OFFSET).A

    theta4_start = float(q_arr[first_grasp, 3])
    valve_angles = np.zeros(len(q_arr))
    angle = 0.0
    for i in range(len(q_arr)):
        if gripper_mask[i]:
            angle = float(q_arr[i, 3]) - theta4_start
        valve_angles[i] = angle

    env.add(open_mesh)
    env.add(closed_mesh)
    env.add(valve_mesh)
    return open_mesh, closed_mesh, valve_mesh, t_valve, valve_angles


def animate_trajectory(
    q_traj,
    dt_s: float = 0.05,
    backend: str = "pyplot",
    hold: bool = False,
    urdf_path: Path = URDF_PATH,
    gripper_closed=None,
):
    q_arr = np.asarray(q_traj, dtype=float)
    robot = load_robot(urdf_path)
    robot.base = SE3(0.0, 0.0, base_floor_offset)
    robot.q = q_arr[-1]

    if backend == "none":
        return robot

    gripper_mask = None if gripper_closed is None else np.asarray(gripper_closed, dtype=bool)
    env = rtb.backends.load_backend(backend)
    env.launch()

    swift_extras = None
    if backend == "swift":
        env.set_camera_pose([1.6, -1.4, 1.2], [0.25, 0.20, 0.65])
        if gripper_mask is not None:
            swift_extras = _setup_swift_extras(env, robot, q_arr, gripper_mask)

    env.add(robot)
    for i, q_row in enumerate(q_arr):
        robot.q = q_row
        if swift_extras is not None and gripper_mask is not None:
            open_mesh, closed_mesh, valve_mesh, t_valve, valve_angles = swift_extras
            ee_tf = np.asarray(robot.fkine(q_row).A)
            if gripper_mask[i]:
                closed_mesh.T = ee_tf
                open_mesh.T = _OFFSCREEN_POSE
            else:
                open_mesh.T = ee_tf
                closed_mesh.T = _OFFSCREEN_POSE
            valve_mesh.T = t_valve @ SE3.Rz(float(valve_angles[i])).A
        env.step(dt_s)

    if hold:
        env.hold()
    else:
        env.close()
    return robot
