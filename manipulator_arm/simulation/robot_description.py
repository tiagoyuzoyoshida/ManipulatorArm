from pathlib import Path
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.models.URDF import URDFRobot

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
URDF_PATH = ASSETS_DIR / "urdf" / "manipulator.urdf"

MESH_FILENAMES = (
    "base_mobile.stl",
    "link_1.stl",
    "link_2.stl",
    "link_3.stl",
    "link_4.stl",
    "ee_gripper.stl",
)

# Dimensions in meters
l1 = 0.40
l2 = 0.35
l4 = 0.25
l5 = 0.15
phi_offset = np.pi / 6.0
base_floor_offset = 0.245

# Mass in kg
m0 = 25.0
m1 = 8.0
m2 = 6.0
m3 = 4.0
m4 = 2.0
m_grip = 0.5

# Centers of mass in meters
r0 = [0.0, 0.0, -0.07]
r1 = [0.0, 0.0, -l1 / 2.0]
r2 = [0.0, 0.0, l2 / 2.0]
r3 = [0.0, 0.0, -l4 / 2.0]
r4 = [0.0, 0.0, -l5 / 2.0]
r_grip = [0.0, 0.0, 0.03]

# Diagonal inertia matrices in kg*m^2
I0 = np.diag([0.60, 0.90, 1.10])
I1 = np.diag([0.12, 0.12, 0.04])
I2 = np.diag([0.08, 0.08, 0.02])
I3 = np.diag([0.03, 0.03, 0.01])
I4 = np.diag([0.01, 0.01, 0.005])
I_grip = np.diag([0.001, 0.001, 0.001])

# Joint limits (rad for revolute, m for prismatic)
JOINT_LIMITS = (
    (-np.pi, np.pi),
    (-3.0 * np.pi / 4.0, 3.0 * np.pi / 4.0),
    (0.0, 0.25),
    (-np.pi, np.pi),
)

# Standard home configuration
qz = np.array([0.0, np.deg2rad(-10.0), 0.08, 0.0])

# DH table: (joint_type, theta_offset, d_offset, a, alpha)
DH_TABLE = (
    ("R", 0.0, l1, 0.0, np.pi / 2.0),
    ("R", -phi_offset, 0.0, 0.0, -np.pi / 2.0),
    ("P", 0.0, l2 + l4, 0.0, 0.0),
    ("R", 0.0, l5, 0.0, 0.0),
)


def load_robot(urdf_path: Path = URDF_PATH) -> rtb.Robot:
    links, name, _ = URDFRobot.URDF_read(Path(urdf_path))
    return rtb.Robot(links, name=name)


ROBOT = load_robot()
