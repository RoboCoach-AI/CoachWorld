"""Shared RoboMind AgileX state and RGB adapters.

RoboMind 1.x and 2.0 store the same Cobot Magic joint state under different
HDF5 schemas.  This module normalizes only the observable source fields.  It
does not claim camera calibration or dataset admission.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import h5py
import numpy as np

from coachworld.calibration.urdf_kinematics import (
    UrdfJoint,
    chain_to_link,
    fk_transforms_named,
    parse_urdf,
)
from coachworld.data.robomind_raw import (
    ROBOMIND_EMBODIMENTS,
    decode_rgb_frame,
    infer_task,
    load_rgb_frames,
)


RoboMindVersion = Literal["1", "2"]

ALOHA_SHARED_BASE_LINK = "body_Link"
ALOHA_LEFT_EEF_LINK = "fl_link6"
ALOHA_RIGHT_EEF_LINK = "fr_link6"
ALOHA_LEFT_JOINT_NAMES = tuple(f"fl_joint{i}" for i in range(1, 7))
ALOHA_RIGHT_JOINT_NAMES = tuple(f"fr_joint{i}" for i in range(1, 7))

# Official aloha_new.urdf limits for fl/fr joint1..6.
ALOHA_NEW_ARM_JOINT_LOWER = (-2.618, 0.0, -2.697, -1.832, -1.22, -3.14)
ALOHA_NEW_ARM_JOINT_UPPER = (2.618, 3.14, 0.0, 1.832, 1.22, 3.14)


@dataclass(frozen=True)
class RoboMindAgilexState:
    """Episode state before FK or camera transforms are applied."""

    version: RoboMindVersion
    qpos: np.ndarray
    gripper: np.ndarray
    diagnostic_eef_xyz: np.ndarray
    camera_names: tuple[str, ...]
    source_fields: dict[str, str]

    @property
    def length(self) -> int:
        return int(self.qpos.shape[0])


@dataclass(frozen=True)
class RoboMindAlohaKinematics:
    """Official dual-arm FK rooted in the shared Aloha ``body_Link`` frame."""

    urdf_path: Path
    left_chain: tuple[UrdfJoint, ...]
    right_chain: tuple[UrdfJoint, ...]

    @classmethod
    def from_urdf(cls, path: str | Path) -> "RoboMindAlohaKinematics":
        urdf_path = Path(path).expanduser().resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(urdf_path)
        joints = parse_urdf(urdf_path)
        return cls(
            urdf_path=urdf_path,
            left_chain=tuple(chain_to_link(joints, ALOHA_SHARED_BASE_LINK, ALOHA_LEFT_EEF_LINK)),
            right_chain=tuple(chain_to_link(joints, ALOHA_SHARED_BASE_LINK, ALOHA_RIGHT_EEF_LINK)),
        )

    def eef_pose_matrices(self, qpos: np.ndarray) -> np.ndarray:
        """Return ``T_body_Link_eef`` with shape ``(T, 2, 4, 4)``."""

        values = np.asarray(qpos, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (2, 6):
            raise ValueError(f"RoboMind AgileX qpos must be (T,2,6), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("RoboMind AgileX qpos contains non-finite values")
        left = fk_transforms_named(
            list(self.left_chain), values[:, 0], list(ALOHA_LEFT_JOINT_NAMES)
        )
        right = fk_transforms_named(
            list(self.right_chain), values[:, 1], list(ALOHA_RIGHT_JOINT_NAMES)
        )
        return np.ascontiguousarray(np.stack([left, right], axis=1))


def gripper_distance_for_aloha_render(values: np.ndarray) -> np.ndarray:
    """Map RoboMind gripper fields to the Aloha URDF's 0..0.1 m opening."""

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return np.full(raw.shape, 0.05, dtype=np.float64)
    q01, q99 = np.percentile(finite, [1.0, 99.0])
    if q99 <= 0.15 and q01 >= -0.03:
        out = raw
    elif q99 <= 6.5 and q01 >= -0.1:
        out = raw / 5.35 * 0.1
    else:
        span = max(float(q99 - q01), 1.0e-6)
        out = (raw - float(q01)) / span * 0.1
    return np.nan_to_num(np.clip(out, 0.0, 0.1), nan=0.05)


def aloha_render_qpos(state: RoboMindAgilexState) -> np.ndarray:
    """Return CalibAll ``robomind_aloha`` qpos as left7 + right7."""

    left_gripper = gripper_distance_for_aloha_render(state.gripper[:, 0, 0])
    right_gripper = gripper_distance_for_aloha_render(state.gripper[:, 1, 0])
    return np.ascontiguousarray(
        np.concatenate(
            [
                state.qpos[:, 0, :6],
                left_gripper[:, None],
                state.qpos[:, 1, :6],
                right_gripper[:, None],
            ],
            axis=1,
        )
    )


def infer_version(f: h5py.File) -> RoboMindVersion:
    if "camera_observations/color_images" in f:
        return "2"
    if "observations/rgb_images" in f:
        return "1"
    raise KeyError("HDF5 is neither a supported RoboMind 1.x nor 2.0 episode")


def _matrix(f: h5py.File, key: str, *, min_width: int) -> np.ndarray:
    if key not in f:
        raise KeyError(f"missing {key}")
    values = np.asarray(f[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < min_width:
        raise ValueError(f"{key} must have shape (T,>={min_width}), got {values.shape}")
    return values


def _column(f: h5py.File, key: str) -> np.ndarray:
    if key not in f:
        raise KeyError(f"missing {key}")
    values = np.asarray(f[key], dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 1:
        raise ValueError(f"{key} must have shape (T,) or (T,>=1), got {values.shape}")
    return values[:, :1]


def _diagnostic_eef(f: h5py.File, key: str, length: int) -> np.ndarray:
    if key not in f:
        return np.full((length, 3), np.nan, dtype=np.float64)
    values = np.asarray(f[key], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 3:
        return np.full((length, 3), np.nan, dtype=np.float64)
    out = np.full((length, 3), np.nan, dtype=np.float64)
    count = min(length, len(values))
    out[:count] = values[:count, :3]
    return out


def load_state(f: h5py.File) -> RoboMindAgilexState:
    """Load dual-arm joint state without treating logged EEF as shared-frame."""

    version = infer_version(f)
    if version == "1":
        left_joint_key = "puppet/joint_position_left"
        right_joint_key = "puppet/joint_position_right"
        left_eef_key = "puppet/end_effector_left"
        right_eef_key = "puppet/end_effector_right"
        left_joint = _matrix(f, left_joint_key, min_width=7)
        right_joint = _matrix(f, right_joint_key, min_width=7)
        length = min(len(left_joint), len(right_joint))
        qpos = np.stack([left_joint[:length, :6], right_joint[:length, :6]], axis=1)
        gripper = np.stack([left_joint[:length, 6], right_joint[:length, 6]], axis=1)[..., None]
        cameras = tuple(sorted(f["observations/rgb_images"].keys()))
        source_fields = {
            "slot0_qpos": left_joint_key,
            "slot1_qpos": right_joint_key,
            "slot0_gripper": f"{left_joint_key}[:,6]",
            "slot1_gripper": f"{right_joint_key}[:,6]",
            "slot0_diagnostic_eef": left_eef_key,
            "slot1_diagnostic_eef": right_eef_key,
        }
    else:
        left_joint_key = "puppet/arm_left_position_align/data"
        right_joint_key = "puppet/arm_right_position_align/data"
        left_gripper_key = "puppet/end_effector_left_position_align/data"
        right_gripper_key = "puppet/end_effector_right_position_align/data"
        left_eef_key = "puppet/end_effector_left_pose_align/data"
        right_eef_key = "puppet/end_effector_right_pose_align/data"
        left_joint = _matrix(f, left_joint_key, min_width=6)
        right_joint = _matrix(f, right_joint_key, min_width=6)
        left_gripper = _column(f, left_gripper_key)
        right_gripper = _column(f, right_gripper_key)
        length = min(len(left_joint), len(right_joint), len(left_gripper), len(right_gripper))
        qpos = np.stack([left_joint[:length, :6], right_joint[:length, :6]], axis=1)
        gripper = np.stack([left_gripper[:length, 0], right_gripper[:length, 0]], axis=1)[..., None]
        cameras = tuple(sorted(f["camera_observations/color_images"].keys()))
        source_fields = {
            "slot0_qpos": left_joint_key,
            "slot1_qpos": right_joint_key,
            "slot0_gripper": left_gripper_key,
            "slot1_gripper": right_gripper_key,
            "slot0_diagnostic_eef": left_eef_key,
            "slot1_diagnostic_eef": right_eef_key,
        }

    diagnostic_eef = np.stack(
        [
            _diagnostic_eef(f, left_eef_key, length),
            _diagnostic_eef(f, right_eef_key, length),
        ],
        axis=1,
    )
    return RoboMindAgilexState(
        version=version,
        qpos=qpos,
        gripper=gripper,
        diagnostic_eef_xyz=diagnostic_eef,
        camera_names=cameras,
        source_fields=source_fields,
    )


def load_rgb(
    f: h5py.File,
    camera: str,
    indices: np.ndarray,
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Decode selected RGB frames from either RoboMind AgileX schema."""

    version = infer_version(f)
    indices = np.asarray(indices, dtype=np.int64)
    if version == "1":
        spec = ROBOMIND_EMBODIMENTS["h5_agilex_3rgb"]
        return load_rgb_frames(f, spec, camera, indices, target_size=target_size)

    key = f"camera_observations/color_images/{camera}"
    if key not in f:
        raise KeyError(f"missing {key}")
    h, w = target_size
    frames = []
    for index in indices:
        frame = decode_rgb_frame(f[key][int(index)], bgr_to_rgb=True)
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    return np.stack(frames, axis=0)


def task_name(path: Path, version: RoboMindVersion) -> str:
    if version == "1":
        return infer_task(path, "h5_agilex_3rgb")
    parts = list(path.parts)
    if "agilex" in parts:
        index = parts.index("agilex")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.parent.name


def list_v2_trajectories(root: Path) -> list[Path]:
    """List 2.0 episodes without recursively walking every data directory."""

    base = root / "data" / "agilex"
    if not base.exists():
        return sorted(root.rglob("trajectory.hdf5"))
    paths: list[Path] = []
    with os.scandir(base) as task_entries:
        for task_entry in task_entries:
            if not task_entry.is_dir():
                continue
            success_dir = Path(task_entry.path) / "success_episodes"
            if not success_dir.exists():
                continue
            with os.scandir(success_dir) as episode_entries:
                for episode_entry in episode_entries:
                    if not episode_entry.is_dir():
                        continue
                    candidate = Path(episode_entry.path) / "data" / "trajectory.hdf5"
                    if candidate.exists():
                        paths.append(candidate)
    return sorted(paths)


def signal_status(
    values: np.ndarray,
    *,
    min_width: int,
    path_eps: float = 1.0e-6,
    zero_eps: float = 1.0e-8,
) -> tuple[str, float]:
    """Classify missing placeholders separately from a valid static signal."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < min_width:
        return f"invalid_shape_{list(array.shape)}", float("nan")
    array = array[:, :min_width]
    if not np.isfinite(array).all():
        return "nonfinite", float("nan")
    if float(np.max(np.abs(array))) <= float(zero_eps):
        return "all_zero", 0.0
    path = float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())
    if path <= float(path_eps):
        return "constant_nonzero", path
    return "moving", path


def arm_joint_sanity_metrics(
    values: np.ndarray,
    *,
    limit_tolerance: float = 0.35,
    max_joint_step: float = 0.5,
) -> dict[str, Any]:
    """Separate temporal corruption from compatibility with aloha_new.urdf."""

    qpos = np.asarray(values, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] < 6:
        return {"status": f"invalid_shape_{list(qpos.shape)}", "sane": False}
    qpos = qpos[:, :6]
    if not np.isfinite(qpos).all():
        return {"status": "nonfinite", "sane": False}

    lower = np.asarray(ALOHA_NEW_ARM_JOINT_LOWER) - float(limit_tolerance)
    upper = np.asarray(ALOHA_NEW_ARM_JOINT_UPPER) + float(limit_tolerance)
    below = np.maximum(lower[None, :] - qpos, 0.0)
    above = np.maximum(qpos - upper[None, :], 0.0)
    violation = np.maximum(below, above)
    steps = np.linalg.norm(np.diff(qpos, axis=0), axis=1)
    max_step = float(np.max(steps)) if steps.size else 0.0
    max_violation = float(np.max(violation))
    within_joint_limits = max_violation <= 0.0
    temporally_sane = max_step <= float(max_joint_step)
    if not temporally_sane:
        status = "temporal_spike"
    elif not within_joint_limits:
        status = "temporally_sane_outside_aloha_new_limits"
    else:
        status = "temporally_sane_within_aloha_new_limits"
    return {
        "status": status,
        # Joint limits are diagnostic because released RoboMind trajectories
        # can use valid encoder ranges beyond the limits in this URDF revision.
        "sane": bool(temporally_sane),
        "within_joint_limits": bool(within_joint_limits),
        "temporally_sane": bool(temporally_sane),
        "joint_min": qpos.min(axis=0).astype(float).tolist(),
        "joint_max": qpos.max(axis=0).astype(float).tolist(),
        "max_abs": float(np.max(np.abs(qpos))),
        "limit_violation_values": int(np.count_nonzero(violation > 0.0)),
        "limit_violation_fraction": float(np.mean(violation > 0.0)),
        "max_limit_violation": max_violation,
        "max_step_norm": max_step,
        "p99_step_norm": float(np.percentile(steps, 99)) if steps.size else 0.0,
    }
