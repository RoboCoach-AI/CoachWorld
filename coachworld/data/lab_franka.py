"""Reviewed state conventions for the fixed-camera lab Franka datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from coachworld.calibration.urdf_kinematics import fk_transforms_named
from coachworld.data.embodiment_adapter import (
    ArmSlotMetadata,
    pack_arm_slot_eef_pose_matrices,
)


FRANKA_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
LAB_FRANKA_GRIPPER_RAW_OPEN = 3.0
LAB_FRANKA_GRIPPER_RAW_CLOSED = 230.0

# Recovered from 2,227 recorded Cartesian states in the 2026-07-21 capture.
# This is a reviewed controller/custom TCP convention, not a Panda URDF link.
LAB_FRANKA_LINK8_FROM_CUSTOM_TCP = np.eye(4, dtype=np.float64)
LAB_FRANKA_LINK8_FROM_CUSTOM_TCP[:3, :3] = Rotation.from_euler(
    "xyz",
    [-180.0, 0.0, 45.0],
    degrees=True,
).as_matrix()
LAB_FRANKA_LINK8_FROM_CUSTOM_TCP[:3, 3] = [0.049985, 0.049996, 0.024014]

# CalibAll's Robotiq 2F-85 fingertip-center TCP at zero mount yaw. A
# configured mount yaw is left-multiplied in the panda_link8 frame.
LAB_FRANKA_LINK8_FROM_ROBOTIQ_TCP_YAW0 = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, -0.0120509604],
        [0.0, 0.0, 1.0, 0.156334978],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _poses_from_link8_offset(
    qpos: np.ndarray,
    *,
    link8_chain: list[Any],
    link8_from_target: np.ndarray,
) -> np.ndarray:
    values = np.asarray(qpos, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 7:
        raise ValueError(f"expected Panda qpos with shape (T,>=7), got {values.shape}")
    base_from_link8 = fk_transforms_named(
        link8_chain,
        values[:, :7],
        FRANKA_JOINT_NAMES,
    )
    return np.einsum(
        "nij,jk->nik",
        base_from_link8,
        np.asarray(link8_from_target, dtype=np.float64),
    )


def custom_tcp_poses_from_qpos(
    qpos: np.ndarray,
    *,
    link8_chain: list[Any],
) -> np.ndarray:
    """Return ``T_base_custom_tcp`` for seven-joint Panda positions."""

    return _poses_from_link8_offset(
        qpos,
        link8_chain=link8_chain,
        link8_from_target=LAB_FRANKA_LINK8_FROM_CUSTOM_TCP,
    )


def link8_poses_from_qpos(
    qpos: np.ndarray,
    *,
    link8_chain: list[Any],
) -> np.ndarray:
    """Return ``T_base_link8`` for seven-joint Panda positions."""

    return _poses_from_link8_offset(
        qpos,
        link8_chain=link8_chain,
        link8_from_target=np.eye(4, dtype=np.float64),
    )


def robotiq_tcp_poses_from_qpos(
    qpos: np.ndarray,
    *,
    link8_chain: list[Any],
    mount_yaw_deg: float,
) -> np.ndarray:
    """Return the physical Robotiq fingertip-center TCP used by the URDF."""

    mount_yaw = np.eye(4, dtype=np.float64)
    mount_yaw[:3, :3] = Rotation.from_euler(
        "z",
        float(mount_yaw_deg),
        degrees=True,
    ).as_matrix()
    return _poses_from_link8_offset(
        qpos,
        link8_chain=link8_chain,
        link8_from_target=(
            mount_yaw @ LAB_FRANKA_LINK8_FROM_ROBOTIQ_TCP_YAW0
        ),
    )


def robotiq_tcp_arm_slot_condition(
    qpos: np.ndarray,
    gripper_raw: np.ndarray,
    *,
    link8_chain: list[Any],
    mount_yaw_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ArmSlotMetadata]:
    """Build the lab Franka condition from physical Robotiq TCP FK."""

    joints = np.asarray(qpos, dtype=np.float64)
    raw_gripper = np.asarray(gripper_raw, dtype=np.float64).reshape(-1)
    if joints.ndim != 2 or joints.shape[1] < 7:
        raise ValueError(f"expected qpos with shape (T,>=7), got {joints.shape}")
    if len(joints) != len(raw_gripper):
        raise ValueError(
            "qpos/gripper length mismatch: "
            f"{len(joints)} != {len(raw_gripper)}"
        )

    tcp_poses = robotiq_tcp_poses_from_qpos(
        joints,
        link8_chain=link8_chain,
        mount_yaw_deg=mount_yaw_deg,
    )
    gripper_open = gripper_open_unit(raw_gripper).astype(np.float32)
    pose_slots = np.broadcast_to(
        np.eye(4, dtype=np.float64),
        (len(joints), 2, 4, 4),
    ).copy()
    pose_slots[:, 0] = tcp_poses
    gripper_slots = np.zeros((len(joints), 2, 1), dtype=np.float32)
    gripper_slots[:, 0, 0] = gripper_open
    slot_mask = np.zeros((len(joints), 2), dtype=np.float32)
    slot_mask[:, 0] = 1.0
    condition = pack_arm_slot_eef_pose_matrices(
        pose_slots=pose_slots,
        gripper_slots=gripper_slots,
        slot_mask=slot_mask,
    )
    metadata = ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "inactive", "exists": False},
        ],
        source_fields=[
            "observation.arm_slot.cartesian_position",
            "observation.arm_slot.gripper_position",
            "observation.arm_slot.mask",
        ],
        transform=(
            "lab_franka_qpos_fk_robotiq_fingertip_tcp_"
            "to_arm_slot_xyz_rot6d_gripper"
        ),
    )
    return condition, tcp_poses, gripper_open, metadata


def gripper_open_unit(
    raw: np.ndarray,
    *,
    raw_open: float = LAB_FRANKA_GRIPPER_RAW_OPEN,
    raw_closed: float = LAB_FRANKA_GRIPPER_RAW_CLOSED,
) -> np.ndarray:
    """Map the lab controller's raw gripper value to ``0=closed, 1=open``."""

    if not np.isfinite(raw_open) or not np.isfinite(raw_closed):
        raise ValueError("gripper endpoints must be finite")
    if raw_closed <= raw_open:
        raise ValueError("raw_closed must exceed raw_open")
    values = np.asarray(raw, dtype=np.float64)
    return 1.0 - np.clip(
        (values - float(raw_open)) / (float(raw_closed) - float(raw_open)),
        0.0,
        1.0,
    )


def gripper_urdf_q(
    raw: np.ndarray,
    *,
    urdf_closed_q: float = 0.8,
) -> np.ndarray:
    """Map raw gripper state to CalibAll Robotiq ``0=open`` closure joints."""

    return (1.0 - gripper_open_unit(raw)) * float(urdf_closed_q)
