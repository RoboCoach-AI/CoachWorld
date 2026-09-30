"""Small URDF and rigid-alignment utilities shared by camera pipelines.

This module contains geometry only. Dataset discovery, reporting, and CLI
defaults belong outside the reusable package.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(x) for x in np.asarray(rpy, dtype=np.float64)]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(float(angle))
    s = math.sin(float(angle))
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rpy_matrix(rpy)
    out[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return out


def joint_motion(joint: UrdfJoint, q: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if joint.joint_type in {"revolute", "continuous"}:
        out[:3, :3] = axis_angle_matrix(joint.axis, float(q))
    elif joint.joint_type == "prismatic":
        out[:3, 3] = np.asarray(joint.axis, dtype=np.float64) * float(q)
    return out


def _parse_float_triplet(text: str | None, default: str = "0 0 0") -> np.ndarray:
    values = [float(x) for x in str(text or default).split()]
    if len(values) != 3:
        raise ValueError(f"expected 3 floats, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def parse_urdf(path: Path) -> dict[str, UrdfJoint]:
    root = ET.parse(path).getroot()
    joints: dict[str, UrdfJoint] = {}
    for elem in root.findall("joint"):
        parent_elem = elem.find("parent")
        child_elem = elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        origin = elem.find("origin")
        axis = elem.find("axis")
        name = str(elem.attrib["name"])
        joints[name] = UrdfJoint(
            name=name,
            joint_type=str(elem.attrib.get("type", "fixed")),
            parent=str(parent_elem.attrib["link"]),
            child=str(child_elem.attrib["link"]),
            origin_xyz=_parse_float_triplet(
                origin.attrib.get("xyz") if origin is not None else None
            ),
            origin_rpy=_parse_float_triplet(
                origin.attrib.get("rpy") if origin is not None else None
            ),
            axis=_parse_float_triplet(
                axis.attrib.get("xyz") if axis is not None else None, "0 0 1"
            ),
        )
    return joints


def chain_to_link(
    joints: dict[str, UrdfJoint], base_link: str, target_link: str
) -> list[UrdfJoint]:
    by_parent: dict[str, list[UrdfJoint]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)
    queue: list[tuple[str, list[UrdfJoint]]] = [(base_link, [])]
    seen: set[str] = set()
    while queue:
        link, chain = queue.pop(0)
        if link == target_link:
            return chain
        if link in seen:
            continue
        seen.add(link)
        for joint in by_parent.get(link, []):
            queue.append((joint.child, chain + [joint]))
    raise KeyError(f"no URDF chain from {base_link!r} to {target_link!r}")


def gripper_to_prismatic(raw: float) -> float:
    value = float(raw)
    if not np.isfinite(value):
        return 0.0
    unit = value / 5.35 if abs(value) > 1.5 else value
    return float(np.clip(unit, 0.0, 1.0) * 0.035)


def fk_positions(chain: list[UrdfJoint], q_values: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.ndim != 2 or q_values.shape[1] < 6:
        raise ValueError(f"q_values must be (T,>=6), got {q_values.shape}")
    out = np.zeros((q_values.shape[0], 3), dtype=np.float64)
    for t, row in enumerate(q_values):
        q = {f"joint{i + 1}": float(row[i]) for i in range(6)}
        if q_values.shape[1] >= 7:
            gripper = gripper_to_prismatic(float(row[6]))
            q["joint7"] = gripper
            q["joint8"] = -gripper
        mat = np.eye(4, dtype=np.float64)
        for joint in chain:
            mat = mat @ make_transform(joint.origin_xyz, joint.origin_rpy)
            mat = mat @ joint_motion(joint, q.get(joint.name, 0.0))
        out[t] = mat[:3, 3]
    return out


def fk_positions_named(
    chain: list[UrdfJoint], q_values: np.ndarray, joint_names: list[str]
) -> np.ndarray:
    return fk_transforms_named(chain, q_values, joint_names)[:, :3, 3]


def fk_transforms_named(
    chain: list[UrdfJoint], q_values: np.ndarray, joint_names: list[str]
) -> np.ndarray:
    """Return ``T_base_target`` for every row of named joint positions."""

    q_values = np.asarray(q_values, dtype=np.float64)
    if q_values.ndim != 2 or q_values.shape[1] < len(joint_names):
        raise ValueError(f"q_values must be (T,>={len(joint_names)}), got {q_values.shape}")
    out = np.zeros((q_values.shape[0], 4, 4), dtype=np.float64)
    for t, row in enumerate(q_values):
        q = {name: float(row[i]) for i, name in enumerate(joint_names)}
        mat = np.eye(4, dtype=np.float64)
        for joint in chain:
            mat = mat @ make_transform(joint.origin_xyz, joint.origin_rpy)
            mat = mat @ joint_motion(joint, q.get(joint.name, 0.0))
        out[t] = mat
    return out


def pose_matrices_to_xyz_rpy(poses: np.ndarray) -> np.ndarray:
    """Convert homogeneous poses to XYZ plus ZYX-composed roll/pitch/yaw."""

    matrices = np.asarray(poses, dtype=np.float64)
    if matrices.shape[-2:] != (4, 4):
        raise ValueError(f"poses must end with shape (4,4), got {matrices.shape}")
    rotation = matrices[..., :3, :3]
    pitch = np.arcsin(np.clip(-rotation[..., 2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    regular = np.abs(cos_pitch) > 1.0e-8

    roll = np.where(
        regular,
        np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2]),
        np.arctan2(-rotation[..., 1, 2], rotation[..., 1, 1]),
    )
    yaw = np.where(
        regular,
        np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0]),
        0.0,
    )
    return np.concatenate(
        [matrices[..., :3, 3], np.stack([roll, pitch, yaw], axis=-1)],
        axis=-1,
    )


def select_even_indices(length: int, max_frames: int) -> np.ndarray:
    length = int(length)
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    count = min(length, int(max_frames))
    if count == length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def safe_name(value: Any, max_len: int = 120) -> str:
    result = "".join(char if char.isalnum() or char in "._-" else "-" for char in str(value)).strip(
        "-"
    )
    return (result or "sample")[:max_len]


def rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if int(valid.sum()) < 3:
        mat = np.eye(4, dtype=np.float64)
        pred = np.full_like(dst, np.nan)
        stats = {"count": int(valid.sum()), "rmse": float("nan"), "median": float("nan")}
        return mat, stats, pred
    x = src[valid]
    y = dst[valid]
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    u, _, vt = np.linalg.svd((x - x_mean).T @ (y - y_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = y_mean - rotation @ x_mean
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotation
    mat[:3, 3] = translation
    pred = src @ rotation.T + translation[None, :]
    error = np.linalg.norm(pred[valid] - y, axis=1)
    stats = {
        "count": int(valid.sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median": float(np.median(error)),
        "p90": float(np.percentile(error, 90)),
        "max": float(np.max(error)),
    }
    return mat, stats, pred


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points[valid], axis=0), axis=1).sum())


def point_stats(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    if int(valid.sum()) == 0:
        return {"count": 0}
    values = points[valid]
    return {
        "count": int(len(values)),
        "min": [float(x) for x in values.min(axis=0)],
        "max": [float(x) for x in values.max(axis=0)],
        "mean": [float(x) for x in values.mean(axis=0)],
        "std": [float(x) for x in values.std(axis=0)],
        "path_length": path_length(values),
    }


def arm_distance_stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    count = min(len(left), len(right))
    if count <= 0:
        return {"count": 0}
    left = np.asarray(left[:count], dtype=np.float64)
    right = np.asarray(right[:count], dtype=np.float64)
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    if not np.any(valid):
        return {"count": 0}
    distance = np.linalg.norm(left[valid] - right[valid], axis=1)
    return {
        "count": int(len(distance)),
        "median": float(np.median(distance)),
        "min": float(np.min(distance)),
        "p10": float(np.percentile(distance, 10)),
        "p90": float(np.percentile(distance, 90)),
        "max": float(np.max(distance)),
        "overlap_lt_5cm": float(np.mean(distance < 0.05)),
        "overlap_lt_10cm": float(np.mean(distance < 0.10)),
    }


def transform_summary(mat: np.ndarray) -> dict[str, list]:
    mat = np.asarray(mat, dtype=np.float64)
    return {
        "translation": [float(x) for x in mat[:3, 3]],
        "rotation_matrix": [[float(x) for x in row] for row in mat[:3, :3]],
    }
