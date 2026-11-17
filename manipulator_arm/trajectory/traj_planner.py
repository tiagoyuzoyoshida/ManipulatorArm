import math
from pathlib import Path
import numpy as np
import roboticstoolbox as rtb
import trimesh
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from spatialmath import SE3

from manipulator_arm.kinematics import forward_kinematics, inverse_kinematics_position
from manipulator_arm.simulation import (
    ASSETS_DIR,
    MESH_FILENAMES,
    URDF_PATH,
    base_floor_offset,
    load_robot,
)
from .workspace_trajectory import append_cartesian_segment

# Waypoints for the 5-phase valve-opening trajectory
Q_HOME = np.array([0.0, np.deg2rad(-10.0), 0.08, 0.0])
Q_APPROACH = np.array([np.deg2rad(40.0), np.deg2rad(-38.0), 0.08, 0.0])
Q_ENGAGE = np.array([np.deg2rad(40.0), np.deg2rad(-38.0), 0.24, 0.0])
Q_TURN = np.array([np.deg2rad(40.0), np.deg2rad(-38.0), 0.24, np.pi / 2.0])
Q_RETRACT = np.array([np.deg2rad(40.0), np.deg2rad(-38.0), 0.08, np.pi / 2.0])

VALVE_TOOL_OFFSET = 0.115

_BG_HEX = "#f2f4f8"
_GRID_HEX = "#d5dbe3"
_TEXT_HEX = "#1e242b"

_LINK_COLORS = (
    (0.26, 0.30, 0.35, 1.0),
    (0.82, 0.85, 0.89, 1.0),
    (0.16, 0.45, 0.74, 1.0),
    (0.88, 0.90, 0.93, 1.0),
    (0.26, 0.30, 0.35, 1.0),
    (0.90, 0.62, 0.12, 1.0),
)
_VALVE_COLOR = (0.88, 0.30, 0.12, 1.0)
_STAND_COLOR = (0.46, 0.51, 0.56, 1.0)


def generate_valve_trajectory(dt_s: float = 0.05, segment_steps: int = 25, dwell_steps: int = 5):
    p_home = forward_kinematics(Q_HOME)[:3, 3]
    p_approach = forward_kinematics(Q_APPROACH)[:3, 3]
    p_engage = forward_kinematics(Q_ENGAGE)[:3, 3]
    p_retract = forward_kinematics(Q_RETRACT)[:3, 3]

    s_profile = np.asarray(rtb.quintic(0.0, 1.0, segment_steps).q, dtype=float)
    waypoints = []
    gripper_states = []

    # 1. Home -> Approach
    append_cartesian_segment(waypoints, p_home, p_approach, 0.0, s_profile, skip_first=False)
    gripper_states.extend([False] * len(waypoints))

    # 2. Approach -> Engage
    before = len(waypoints)
    append_cartesian_segment(waypoints, p_approach, p_engage, 0.0, s_profile, skip_first=True)
    gripper_states.extend([False] * (len(waypoints) - before))

    # 3. Close gripper
    for _ in range(dwell_steps):
        waypoints.append(Q_ENGAGE.copy())
        gripper_states.append(True)

    # 4. Turn valve 90 deg
    t4_start, t4_end = float(Q_ENGAGE[3]), float(Q_TURN[3])
    for s_val in s_profile[1:]:
        t4 = t4_start + float(s_val) * (t4_end - t4_start)
        waypoints.append(inverse_kinematics_position(p_engage, theta4=t4))
        gripper_states.append(True)

    # 5. Open gripper and retract
    for _ in range(dwell_steps):
        waypoints.append(Q_TURN.copy())
        gripper_states.append(False)

    before = len(waypoints)
    append_cartesian_segment(waypoints, p_engage, p_retract, t4_end, s_profile, skip_first=True)
    gripper_states.extend([False] * (len(waypoints) - before))

    q_arr = np.vstack(waypoints)
    xyz_arr = np.vstack([forward_kinematics(q)[:3, 3] for q in q_arr])
    t_s = np.arange(len(q_arr)) * dt_s
    dq = np.gradient(q_arr, dt_s, axis=0)
    ddq = np.gradient(dq, dt_s, axis=0)

    return {
        "t_s": t_s,
        "q": q_arr,
        "dq": dq,
        "ddq": ddq,
        "xyz_m": xyz_arr,
        "gripper_closed": np.asarray(gripper_states, dtype=bool),
    }


def _load_mesh(mesh_path: Path):
    mesh = trimesh.load_mesh(mesh_path)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def _build_stand_mesh(t_valve: np.ndarray):
    p_bonnet = (t_valve @ np.array([0.0, 0.0, 0.115, 1.0]))[:3]
    p_floor = np.array([p_bonnet[0], p_bonnet[1], 0.0])
    p_mid = np.array([p_bonnet[0], p_bonnet[1], 0.50 * p_bonnet[2]])

    parts = [
        trimesh.creation.cylinder(radius=0.030, segment=(p_floor, p_bonnet), sections=24),
        trimesh.creation.cylinder(radius=0.095, segment=(p_floor, p_floor + [0.0, 0.0, 0.018]), sections=28),
        trimesh.creation.cylinder(radius=0.052, segment=(p_mid - [0.0, 0.0, 0.014], p_mid + [0.0, 0.0, 0.014]), sections=24),
    ]
    body = trimesh.creation.cylinder(radius=0.054, height=0.110, sections=24)
    body.apply_transform(t_valve @ SE3(0.0, 0.0, 0.115).A)
    parts.append(body)

    stand = trimesh.util.concatenate(parts)
    return np.asarray(stand.vertices, dtype=float), np.asarray(stand.faces, dtype=np.int64)


def _shade_mesh(verts, faces, tf, rgba, light_key, light_fill):
    v_tf = verts @ tf[:3, :3].T + tf[:3, 3]
    tris = v_tf[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    unorm = normals / np.where(norms > 1e-12, norms, 1.0)

    shade = 0.42 + 0.46 * np.clip(unorm @ light_key, 0.0, 1.0) + 0.16 * np.clip(unorm @ light_fill, 0.0, 1.0)
    colors = np.empty((len(tris), 4))
    colors[:, :3] = np.clip(np.asarray(rgba[:3]) * shade[:, None], 0.0, 1.0)
    colors[:, 3] = rgba[3]
    return tris, colors


def _draw_grid(ax):
    vals = (-0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.00)
    for c in vals:
        lw = 0.9 if abs(c) < 1e-6 else 0.55
        ax.plot([c, c], [vals[0], vals[-1]], [0.0, 0.0], color=_GRID_HEX, linewidth=lw, zorder=1)
        ax.plot([vals[0], vals[-1]], [c, c], [0.0, 0.0], color=_GRID_HEX, linewidth=lw, zorder=1)


def _phase_title(idx: int, total: int, is_closed: bool, valve_deg: float) -> str:
    if is_closed:
        if abs(valve_deg) < 2.0:
            return "Phase 3: Grasp Valve Handwheel"
        return f"Phase 4: Turn Valve ({valve_deg:+.0f} deg)"
    if valve_deg > 10.0:
        return "Phase 5: Release & Retract Boom"
    if idx < total * 0.25:
        return "Phase 1: Align with Manifold"
    return "Phase 2: Telescopic Extension (d3)"


def save_trajectory_gif(
    q_traj,
    out_path: Path = Path("outputs/trajectory.gif"),
    step: int = 3,
    fps: int = 15,
    urdf_path: Path = URDF_PATH,
    gripper_closed=None,
) -> Path:
    q_arr = np.asarray(q_traj, dtype=float)
    gripper_mask = np.zeros(len(q_arr), dtype=bool) if gripper_closed is None else np.asarray(gripper_closed, dtype=bool)

    robot = load_robot(urdf_path)
    robot.base = SE3(0.0, 0.0, base_floor_offset)

    meshes_dir = ASSETS_DIR / "meshes"
    link_meshes = [_load_mesh(meshes_dir / name) for name in MESH_FILENAMES]
    gripper_open = _load_mesh(meshes_dir / "ee_gripper_open.stl")
    gripper_closed_m = _load_mesh(meshes_dir / "ee_gripper_closed.stl")
    valve_mesh = _load_mesh(meshes_dir / "valve_wheel.stl")

    grasp_idx = np.flatnonzero(gripper_mask)
    first_grasp = int(grasp_idx[0]) if grasp_idx.size > 0 else 0
    t_valve_engage = (
        SE3(0.0, 0.0, base_floor_offset).A
        @ forward_kinematics(q_arr[first_grasp])
        @ SE3(0.0, 0.0, VALVE_TOOL_OFFSET).A
    )

    t4_start = float(q_arr[first_grasp, 3])
    valve_angles = np.zeros(len(q_arr))
    ang = 0.0
    for i in range(len(q_arr)):
        if gripper_mask[i]:
            ang = float(q_arr[i, 3]) - t4_start
        valve_angles[i] = ang

    stand_mesh = _build_stand_mesh(t_valve_engage)
    light_key = np.array([0.55, -0.60, 0.58]) / np.linalg.norm([0.55, -0.60, 0.58])
    light_fill = np.array([-0.60, 0.40, 0.69]) / np.linalg.norm([-0.60, 0.40, 0.69])
    stand_tris, stand_fc = _shade_mesh(stand_mesh[0], stand_mesh[1], np.eye(4), _STAND_COLOR, light_key, light_fill)

    ee_pts = np.vstack([forward_kinematics(q)[:3, 3] for q in q_arr]) + np.array([0.0, 0.0, base_floor_offset])
    indices = list(range(0, len(q_arr), step))
    if indices[-1] != len(q_arr) - 1:
        indices.append(len(q_arr) - 1)

    fig = Figure(figsize=(7.6, 6.2), dpi=105, facecolor=_BG_HEX)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d", facecolor=_BG_HEX)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.90)

    frames = []
    for idx in indices:
        fk_all = robot.fkine_all(q_arr[idx])
        is_closed = bool(gripper_mask[idx])
        valve_deg = math.degrees(float(valve_angles[idx]))

        ax.cla()
        ax.set_facecolor(_BG_HEX)
        ax.set_axis_off()
        _draw_grid(ax)

        all_tris, all_fc = [stand_tris], [stand_fc]
        for j, (verts, faces) in enumerate(link_meshes):
            if j == len(link_meshes) - 1:
                verts, faces = gripper_closed_m if is_closed else gripper_open
            tf = np.asarray(fk_all[j + 1].A)
            t_j, fc_j = _shade_mesh(verts, faces, tf, _LINK_COLORS[j], light_key, light_fill)
            all_tris.append(t_j)
            all_fc.append(fc_j)

        t_valve = t_valve_engage @ SE3.Rz(float(valve_angles[idx])).A
        v_tris, v_fc = _shade_mesh(valve_mesh[0], valve_mesh[1], t_valve, _VALVE_COLOR, light_key, light_fill)
        all_tris.append(v_tris)
        all_fc.append(v_fc)

        poly = Poly3DCollection(
            np.concatenate(all_tris, axis=0),
            facecolors=np.concatenate(all_fc, axis=0),
            linewidths=0.14,
            edgecolors=(0.12, 0.15, 0.18, 0.28),
            zsort="average",
        )
        ax.add_collection3d(poly)

        ax.plot(ee_pts[: idx + 1, 0], ee_pts[: idx + 1, 1], ee_pts[: idx + 1, 2], color="#e03131", linewidth=2.4, zorder=10)
        ax.scatter([ee_pts[idx, 0]], [ee_pts[idx, 1]], [ee_pts[idx, 2]], color="#e03131", s=28, zorder=11)

        phase_str = _phase_title(idx, len(q_arr), is_closed, valve_deg)
        gripper_str = "CLOSED" if is_closed else "OPEN"
        d3_mm = float(q_arr[idx, 2]) * 1000.0

        ax.set_xlim(-0.36, 0.94)
        ax.set_ylim(-0.36, 0.94)
        ax.set_zlim(0.0, 1.22)
        ax.set_box_aspect((1.30, 1.30, 1.22))
        ax.view_init(elev=22, azim=-34)
        ax.set_title(
            f"{phase_str}\nGripper: {gripper_str}  |  Boom d3: {d3_mm:.0f} mm  |  Valve: {valve_deg:+.0f}°",
            color=_TEXT_HEX,
            fontsize=10.5,
            fontweight="bold",
            pad=6,
        )

        canvas.draw()
        frames.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB"))

    resolved_out = Path(out_path)
    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        resolved_out,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(1000.0 / fps))),
        loop=0,
    )
    return resolved_out
