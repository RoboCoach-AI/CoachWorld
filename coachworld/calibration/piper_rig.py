"""Canonical dual-Piper rig geometry used by real dual-arm datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .urdf_kinematics import UrdfJoint, gripper_to_prismatic, joint_motion, make_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPER_URDF = (
    PROJECT_ROOT
    / "third_party/X-VLA/evaluation/SoftFold-Agilex/"
    "Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf"
)

DUAL_PIPER_MOUNT = {
    "name": "dual_piper_aloha_mount",
    "slot0_left": {"xyz": [0.23875, 0.30000, 0.77500], "rpy": [0.0, 0.0, 0.0]},
    "slot1_right": {"xyz": [0.23875, -0.30000, 0.77500], "rpy": [0.0, 0.0, 0.0]},
}


def fk_chain_positions(
    chain: list[UrdfJoint], q_values: np.ndarray, prefix: str = ""
) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    out = np.zeros((q_values.shape[0], 3), dtype=np.float64)
    for t, row in enumerate(q_values):
        q: dict[str, float] = {}
        for i in range(6):
            q[f"{prefix}joint{i + 1}"] = float(row[i])
            q[f"{prefix}link{i + 1}"] = float(row[i])
        if q_values.shape[1] >= 7:
            gripper = gripper_to_prismatic(float(row[6]))
            for name in ("joint7", "joint8", "link7", "link8"):
                q[f"{prefix}{name}"] = gripper / 2.0
        mat = np.eye(4, dtype=np.float64)
        for joint in chain:
            mat = mat @ make_transform(joint.origin_xyz, joint.origin_rpy)
            mat = mat @ joint_motion(joint, q.get(joint.name, 0.0))
        out[t] = mat[:3, 3]
    return out


def transform_points(points: np.ndarray, xyz: list[float], rpy: list[float]) -> np.ndarray:
    transform = make_transform(np.asarray(xyz, dtype=np.float64), np.asarray(rpy, dtype=np.float64))
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def slot_mount(slot: int) -> np.ndarray:
    key = "slot0_left" if int(slot) == 0 else "slot1_right"
    config = DUAL_PIPER_MOUNT[key]
    return make_transform(
        np.asarray(config["xyz"], dtype=np.float64),
        np.asarray(config["rpy"], dtype=np.float64),
    )


def camera_from_rig_from_anchor(camera_from_slot: np.ndarray, anchor_slot: int) -> np.ndarray:
    return np.asarray(camera_from_slot, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(
        slot_mount(anchor_slot)
    )


def build_shared_eef(qpos: np.ndarray, chain: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    left_local = fk_chain_positions(chain, qpos[:, 0:7])
    right_local = fk_chain_positions(chain, qpos[:, 7:14])
    left_cfg = DUAL_PIPER_MOUNT["slot0_left"]
    right_cfg = DUAL_PIPER_MOUNT["slot1_right"]
    left = transform_points(left_local, left_cfg["xyz"], left_cfg["rpy"])
    right = transform_points(right_local, right_cfg["xyz"], right_cfg["rpy"])
    return left.astype(np.float32), right.astype(np.float32)
