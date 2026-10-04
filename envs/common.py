"""Six-wheel geometry, command mixing and local SO101 CAD assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mujoco_models/tree_work.xml"
ASSET_MANIFEST_PATH = ROOT / "mujoco_models/so101_assets.json"
ARM_JOINTS = ("shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "wrist_yaw")
ARM_HOME = np.array([0., -1.70, 0., .50, 0., 0.])
BASE_JOINTS = ("base_height", "base_azimuth", "base_roll", "base_pitch")
STAGES = ("navigate", "extend", "realign", "operate", "retract", "done")
SCHEMA_VERSION = "tree_work_v5_single_ring_six_axis_combined"
TREE_RADIUS = 0.15
WHEEL_RADIUS = 0.032
WHEEL_ANGLES = np.arange(6, dtype=float) * np.pi / 3.0
WHEEL_SIGNS = np.where(np.arange(6) % 2 == 0, 1.0, -1.0)
ROLLER_ANGLE = np.pi / 4.0
CONTACT_STIFFNESS = 60000.0


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wheel_mixer(tree_radius=TREE_RADIUS, wheel_radius=WHEEL_RADIUS):
    """Wheel rad/s from [height m/s, azimuth rad/s]."""
    return np.column_stack((np.full(6, np.cos(ROLLER_ANGLE) / wheel_radius),
                            WHEEL_SIGNS * np.sin(ROLLER_ANGLE) * tree_radius / wheel_radius))


def wheel_commands(base_speeds, limit=30.0, tree_radius=TREE_RADIUS, wheel_radius=WHEEL_RADIUS):
    """One common scale preserves the rank-two manifold when saturating."""
    speeds = np.asarray(base_speeds, dtype=float)
    if speeds.shape != (2,) or not np.isfinite(speeds).all():
        raise ValueError("base_speeds must contain two finite values")
    command = wheel_mixer(tree_radius, wheel_radius) @ speeds
    command *= min(1.0, float(limit) / max(float(np.max(np.abs(command))), 1e-12))
    return command


def project_wheel_commands(command, tree_radius=TREE_RADIUS, wheel_radius=WHEEL_RADIUS):
    matrix = wheel_mixer(tree_radius, wheel_radius)
    return matrix @ (np.linalg.pinv(matrix) @ np.asarray(command, dtype=float))


def wheel_force_map(tree_radius=TREE_RADIUS):
    """[vertical force, tree-axis torque] from six rolling-direction forces."""
    return np.vstack((np.full(6, np.cos(ROLLER_ANGLE)),
                      WHEEL_SIGNS * np.sin(ROLLER_ANGLE) * tree_radius))


def wheel_force_allocation(tree_radius=TREE_RADIUS):
    return np.linalg.pinv(wheel_force_map(tree_radius))


def rolling_directions(azimuth=0.0, orientation=None):
    """Traction directions of the six inclined-axis omni wheels.

    The optional native root rotation carries these directions through passive
    tilt. A positive shaft rate moves its tree-side tread in the *opposite*
    direction, so positive wheel commands propel the chassis along this vector.
    """
    angles = WHEEL_ANGLES + (0.0 if orientation is not None else float(azimuth))
    tangent = np.column_stack((-np.sin(angles), np.cos(angles), np.zeros(6)))
    directions = WHEEL_SIGNS[:, None] * np.sin(ROLLER_ANGLE) * tangent + np.array([0.0, 0.0, np.cos(ROLLER_ANGLE)])
    return directions if orientation is None else directions @ np.asarray(orientation).reshape(3, 3).T


def wheel_axes():
    """Local shaft axes, fixed by the rolling-contact right-hand rule."""
    radial = np.column_stack((np.cos(WHEEL_ANGLES), np.sin(WHEEL_ANGLES), np.zeros(6)))
    return np.cross(radial, rolling_directions())


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_directory():
    manifest = json.loads(ASSET_MANIFEST_PATH.read_text())
    return (ROOT / manifest["asset_directory"]).resolve(strict=True)


def asset_contract():
    manifest = json.loads(ASSET_MANIFEST_PATH.read_text())
    asset_dir = asset_directory()
    for name, expected in manifest["files"].items():
        p = (asset_dir / name).resolve(strict=True)
        if not p.is_relative_to(asset_dir) or file_sha(p) != expected:
            raise ValueError(f"SO101 asset missing or changed: {name}")
    return {"source_commit": manifest["source_commit"], "license": manifest["license"],
            "files": manifest["files"], "manifest_sha256": file_sha(ASSET_MANIFEST_PATH)}


def model_contract():
    return {"schema_version": SCHEMA_VERSION, "model_sha256": file_sha(MODEL_PATH),
            "shared_geometry_sha256": file_sha(Path(__file__)),
            "arm_joint_order": list(ARM_JOINTS), "base_joint_order": list(BASE_JOINTS),
            "tool_sites": ["cut_site", "nozzle_site"], "assets": asset_contract()}


def load_model():
    asset_contract()
    tree = ET.parse(MODEL_PATH)
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        compiler = ET.SubElement(tree.getroot(), "compiler")
    compiler.set("meshdir", str(asset_directory()))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
