"""Convert dual-Piper qpos into the canonical arm-slot state contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from coachworld.calibration.piper_rig import slot_mount
from coachworld.calibration.urdf_kinematics import (
    UrdfJoint,
    gripper_to_prismatic,
    joint_motion,
    make_transform,
)
from .embodiment_adapter import ARM_SLOT_CONDITION_WIDTH, ArmSlotMetadata


def transform_intrinsics(K: np.ndarray, image_transform: dict[str, Any]) -> np.ndarray:
    result = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    scale_x, scale_y = [float(x) for x in image_transform["scale_xy"]]
    pad_x, pad_y = [float(x) for x in image_transform["pad_xy"]]
    result[0, 0] *= scale_x
    result[0, 2] = result[0, 2] * scale_x + pad_x
    result[1, 1] *= scale_y
    result[1, 2] = result[1, 2] * scale_y + pad_y
    return result


def vggt_sidecar_image_hw(summary: dict[str, Any]) -> tuple[int, int] | None:
    sample_dir = summary.get("vggt_sample_dir")
    if not sample_dir:
        return None
    predictions = Path(str(sample_dir)).expanduser().resolve() / "predictions.npz"
    if not predictions.exists():
        return None
    with np.load(predictions) as arrays:
        images = np.asarray(arrays["images"])
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"{predictions}: expected VGGT images (T,3,H,W), got {images.shape}")
    return int(images.shape[2]), int(images.shape[3])


def qpos_slot_health(
    qpos: np.ndarray, *, low_motion_threshold: float = 0.05
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for slot, column_slice in ((0, slice(0, 7)), (1, slice(7, 14))):
        values = np.asarray(qpos[:, column_slice], dtype=np.float64)
        differences = np.diff(values[:, :6], axis=0)
        path = float(np.sum(np.linalg.norm(differences, axis=1))) if differences.size else 0.0
        all_zero = bool(np.all(np.abs(values) < 1.0e-8))
        result[f"slot{slot}"] = {
            "all_zero": all_zero,
            "path_length": path,
            "low_motion": bool(path < float(low_motion_threshold)),
        }
    result["dual_qpos_present"] = bool(
        not result["slot0"]["all_zero"] and not result["slot1"]["all_zero"]
    )
    result["dual_qpos_motion_above_threshold"] = bool(
        not result["slot0"]["low_motion"] and not result["slot1"]["low_motion"]
    )
    return result


def _gripper_to_unit(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    unit = np.where(np.abs(values) > 1.5, values / 5.35, values)
    return np.clip(unit, 0.0, 1.0).astype(np.float32)


def _fk_matrices(chain: list[UrdfJoint], q_values: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q_values, dtype=np.float64)
    output = np.zeros((q_values.shape[0], 4, 4), dtype=np.float64)
    for t, row in enumerate(q_values):
        q: dict[str, float] = {}
        for i in range(6):
            q[f"joint{i + 1}"] = float(row[i])
            q[f"link{i + 1}"] = float(row[i])
        if q_values.shape[1] >= 7:
            gripper = gripper_to_prismatic(float(row[6]))
            for name in ("joint7", "joint8", "link7", "link8"):
                q[name] = gripper / 2.0
        transform = np.eye(4, dtype=np.float64)
        for joint in chain:
            transform = transform @ make_transform(joint.origin_xyz, joint.origin_rpy)
            transform = transform @ joint_motion(joint, q.get(joint.name, 0.0))
        output[t] = transform
    return output


def _rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float32)
    return np.stack(
        [
            rotation[..., 0, 0],
            rotation[..., 1, 0],
            rotation[..., 2, 0],
            rotation[..., 0, 1],
            rotation[..., 1, 1],
            rotation[..., 2, 1],
        ],
        axis=-1,
    )


def build_condition_from_qpos(
    qpos: np.ndarray,
    chain: list[UrdfJoint],
    *,
    slot_health: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, ArmSlotMetadata]:
    qpos = np.asarray(qpos, dtype=np.float64)[:, :14]
    local = [_fk_matrices(chain, qpos[:, 0:7]), _fk_matrices(chain, qpos[:, 7:14])]
    matrices = np.stack([slot_mount(slot)[None] @ local[slot] for slot in (0, 1)], axis=1)
    length = int(qpos.shape[0])
    mask = np.ones((length, 2), dtype=np.float32)
    if slot_health is not None:
        for slot in (0, 1):
            if bool((slot_health.get(f"slot{slot}") or {}).get("all_zero", False)):
                mask[:, slot] = 0.0
    gripper = np.stack(
        [_gripper_to_unit(qpos[:, 6]), _gripper_to_unit(qpos[:, 13])], axis=1
    ).astype(np.float32)
    condition = np.zeros((length, 2, ARM_SLOT_CONDITION_WIDTH), dtype=np.float32)
    condition[:, :, :3] = matrices[:, :, :3, 3].astype(np.float32)
    condition[:, :, 3:9] = _rotation_to_6d(matrices[:, :, :3, :3])
    condition[:, :, 9] = gripper
    condition[:, :, 10] = mask
    signals = {
        "observations.qpos": np.ascontiguousarray(qpos.astype(np.float32)),
        "observation.arm_slot.eef_pose_matrix": np.ascontiguousarray(
            matrices[:, :, :3, :4].reshape(length, 24).astype(np.float32)
        ),
        "observation.arm_slot.gripper_position": np.ascontiguousarray(gripper),
        "observation.arm_slot.mask": mask,
    }
    metadata = ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": bool(mask[:, 0].max() > 0.5)},
            {"slot": 1, "side": "right", "exists": bool(mask[:, 1].max() > 0.5)},
        ],
        source_fields=list(signals),
        transform=(
            "vifailback_qpos_piper_fk_gripper_base_dual_mount_to_"
            "arm_slot_xyz_rot6d_gripper"
        ),
    )
    return signals, np.ascontiguousarray(condition), metadata
