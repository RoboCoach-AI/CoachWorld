"""Embodiment adapters for CoachWorld action conditioning.

This module owns the model-side action condition contract.  Dataset builders
may parse different raw files, but they must emit the same tensor:

    arm_slot_eef_pose: (T, 2, 11)

The first 10 values are ``xyz + rot6d + gripper``.  The last value is the slot
existence mask.  Inactive slot values are forced to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from coachworld.data.canonical_robot_frame import matrix_to_rot6d


ARM_SLOT_EEF_POSE_VIEW = "arm_slot_eef_pose"
ARM_SLOT_EEF_POSE_DIM = 10
MAX_ARM_SLOTS = 2
ARM_SLOT_CONDITION_WIDTH = ARM_SLOT_EEF_POSE_DIM + 1
ARM_SLOT_VALUE_LAYOUT = "xyz_rot6d_gripper"
ARM_SLOT_MASK_STORAGE = "last_channel_exists_metadata"
ARM_SLOT_SOURCE_SEMANTICS = "gt_executed_eef_pose"
ROBOMIND1_AGILEX_GRIPPER_OPEN_SCALE = 5.35


@dataclass(frozen=True)
class ArmSlotMetadata:
    arm_slots: list[dict[str, Any]]
    source_fields: list[str]
    transform: str

    def condition_spec_extra(self) -> dict[str, Any]:
        active = [int(slot["slot"]) for slot in self.arm_slots if bool(slot.get("exists"))]
        inactive = [int(slot["slot"]) for slot in self.arm_slots if not bool(slot.get("exists"))]
        return {
            "source_fields": list(self.source_fields),
            "transform": self.transform,
            "feature_dim": ARM_SLOT_EEF_POSE_DIM,
            "arm_slots": {
                "max_slots": MAX_ARM_SLOTS,
                "active_slots": active,
                "inactive_slots": inactive,
            },
            "condition_semantics": ARM_SLOT_SOURCE_SEMANTICS,
        }


def arm_slot_condition_schema() -> dict[str, Any]:
    return {
        "condition_view": ARM_SLOT_EEF_POSE_VIEW,
        "feature_dim": ARM_SLOT_EEF_POSE_DIM,
        "value_layout": ARM_SLOT_VALUE_LAYOUT,
        "mask_storage": ARM_SLOT_MASK_STORAGE,
        "max_arm_slots": MAX_ARM_SLOTS,
    }


def rpy_to_rot6d(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw radians to first two rotation-matrix columns."""
    rpy = np.asarray(rpy, dtype=np.float32)
    if rpy.shape[-1] != 3:
        raise ValueError(f"rpy must end with dim=3, got {rpy.shape}")
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll), matching DROID/RoboMind xyz+rpy.
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r20 = -sp
    r21 = cp * sr
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def quat_wxyz_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz] quaternions to first two matrix columns."""
    q = np.asarray(quat, dtype=np.float32)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must end with dim=4, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1.0e-8):
        raise ValueError("invalid quaternion norm in robot pose condition")
    q = q / norm
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vectors to first two rotation-matrix columns."""
    vec = np.asarray(axis_angle, dtype=np.float32)
    if vec.shape[-1] != 3:
        raise ValueError(f"axis_angle must end with dim=3, got {vec.shape}")
    angle = np.linalg.norm(vec, axis=-1, keepdims=True)
    axis = np.divide(vec, np.maximum(angle, 1.0e-8), out=np.zeros_like(vec), where=angle > 1.0e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    theta = angle[..., 0]
    c = np.cos(theta)
    s = np.sin(theta)
    one_c = 1.0 - c
    r00 = c + x * x * one_c
    r01 = x * y * one_c - z * s
    r10 = y * x * one_c + z * s
    r11 = c + y * y * one_c
    r20 = z * x * one_c - y * s
    r21 = z * y * one_c + x * s
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def pack_arm_slot_eef_pose(
    *,
    cartesian_rpy_slots: np.ndarray,
    gripper_slots: np.ndarray,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Pack per-slot absolute EEF pose into the CoachWorld condition tensor."""
    cart = np.asarray(cartesian_rpy_slots, dtype=np.float32)
    grip = np.asarray(gripper_slots, dtype=np.float32)
    mask = np.asarray(slot_mask, dtype=np.float32)
    if cart.ndim != 3 or cart.shape[1:] != (MAX_ARM_SLOTS, 6):
        raise ValueError(f"cartesian_rpy_slots must be (T,2,6), got {cart.shape}")
    if grip.ndim != 3 or grip.shape[1:] != (MAX_ARM_SLOTS, 1):
        raise ValueError(f"gripper_slots must be (T,2,1), got {grip.shape}")
    if mask.ndim != 2 or mask.shape[1] != MAX_ARM_SLOTS:
        raise ValueError(f"slot_mask must be (T,2), got {mask.shape}")
    if cart.shape[0] != grip.shape[0] or cart.shape[0] != mask.shape[0]:
        raise ValueError(
            "condition lengths mismatch: "
            f"cart={cart.shape[0]} grip={grip.shape[0]} mask={mask.shape[0]}"
        )
    if not np.isfinite(cart).all() or not np.isfinite(grip).all() or not np.isfinite(mask).all():
        raise ValueError("non-finite value in arm-slot condition inputs")

    t = int(cart.shape[0])
    cond = np.zeros((t, MAX_ARM_SLOTS, ARM_SLOT_CONDITION_WIDTH), dtype=np.float32)
    for slot in range(MAX_ARM_SLOTS):
        xyz = cart[:, slot, :3]
        rot6d = rpy_to_rot6d(cart[:, slot, 3:6])
        values = np.concatenate([xyz, rot6d, grip[:, slot, :1]], axis=-1)
        cond[:, slot, :ARM_SLOT_EEF_POSE_DIM] = values
        cond[:, slot, ARM_SLOT_EEF_POSE_DIM] = (mask[:, slot] > 0.5).astype(np.float32)

    cond[..., :ARM_SLOT_EEF_POSE_DIM] *= cond[
        ..., ARM_SLOT_EEF_POSE_DIM : ARM_SLOT_EEF_POSE_DIM + 1
    ]
    if not np.isfinite(cond).all():
        raise ValueError("non-finite value in packed arm-slot condition")
    return np.ascontiguousarray(cond)


def pack_arm_slot_eef_pose_matrices(
    *,
    pose_slots: np.ndarray,
    gripper_slots: np.ndarray,
    slot_mask: np.ndarray,
) -> np.ndarray:
    """Pack homogeneous EEF poses without an intermediate Euler conversion."""

    poses = np.asarray(pose_slots, dtype=np.float64)
    grip = np.asarray(gripper_slots, dtype=np.float32)
    mask = np.asarray(slot_mask, dtype=np.float32)
    if poses.ndim != 4 or poses.shape[1:] != (MAX_ARM_SLOTS, 4, 4):
        raise ValueError(f"pose_slots must be (T,2,4,4), got {poses.shape}")
    if grip.ndim != 3 or grip.shape[1:] != (MAX_ARM_SLOTS, 1):
        raise ValueError(f"gripper_slots must be (T,2,1), got {grip.shape}")
    if mask.ndim != 2 or mask.shape[1] != MAX_ARM_SLOTS:
        raise ValueError(f"slot_mask must be (T,2), got {mask.shape}")
    if poses.shape[0] != grip.shape[0] or poses.shape[0] != mask.shape[0]:
        raise ValueError(
            "condition lengths mismatch: "
            f"poses={poses.shape[0]} grip={grip.shape[0]} mask={mask.shape[0]}"
        )
    if not np.isfinite(poses).all() or not np.isfinite(grip).all() or not np.isfinite(mask).all():
        raise ValueError("non-finite value in arm-slot pose inputs")

    cond = np.zeros(
        (poses.shape[0], MAX_ARM_SLOTS, ARM_SLOT_CONDITION_WIDTH),
        dtype=np.float32,
    )
    cond[..., :3] = poses[..., :3, 3].astype(np.float32)
    cond[..., 3:9] = matrix_to_rot6d(poses[..., :3, :3]).astype(np.float32)
    cond[..., 9:10] = grip
    cond[..., 10] = (mask > 0.5).astype(np.float32)
    cond[..., :ARM_SLOT_EEF_POSE_DIM] *= cond[..., 10:11]
    return np.ascontiguousarray(cond)


def robomind_agilex_gripper_to_unit(raw: np.ndarray, *, open_scale: float) -> np.ndarray:
    grip = np.asarray(raw, dtype=np.float32)
    if grip.ndim != 1:
        raise ValueError(f"AgileX gripper must be 1D, got {grip.shape}")
    scale = float(open_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"AgileX gripper open_scale must be positive, got {scale}")
    return np.clip(grip / scale, 0.0, 1.0).astype(np.float32)


def droid_franka_eef_pose_condition(
    *,
    cartesian_position: np.ndarray,
    gripper_position: np.ndarray,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build DROID Franka single-arm EEF condition."""
    cart = np.asarray(cartesian_position, dtype=np.float32)
    grip = np.asarray(gripper_position, dtype=np.float32)
    if cart.ndim != 2 or cart.shape[1] < 6:
        raise ValueError(f"DROID cartesian_position must be (T,>=6), got {cart.shape}")
    if grip.ndim != 2 or grip.shape[1] != 1:
        raise ValueError(f"DROID gripper_position must be (T,1), got {grip.shape}")
    if cart.shape[0] != grip.shape[0]:
        raise ValueError(f"DROID condition length mismatch: cart={cart.shape} grip={grip.shape}")

    t = int(cart.shape[0])
    cart_slots = np.zeros((t, MAX_ARM_SLOTS, 6), dtype=np.float32)
    grip_slots = np.zeros((t, MAX_ARM_SLOTS, 1), dtype=np.float32)
    mask = np.zeros((t, MAX_ARM_SLOTS), dtype=np.float32)
    cart_slots[:, 0] = cart[:, :6]
    grip_slots[:, 0, 0] = grip[:, 0]
    mask[:, 0] = 1.0
    meta = droid_franka_eef_pose_metadata()
    return pack_arm_slot_eef_pose(
        cartesian_rpy_slots=cart_slots,
        gripper_slots=grip_slots,
        slot_mask=mask,
    ), meta


def droid_franka_eef_pose_metadata() -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "inactive", "exists": False},
        ],
        source_fields=[
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
        ],
        transform="droid_xyz_rpy_gripper_to_arm_slot_xyz_rot6d_gripper",
    )


def robomind_franka_eef_pose_condition(
    *,
    end_effector: np.ndarray,
    joint_position: np.ndarray,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build RoboMind Franka single-arm EEF condition."""
    cart = np.asarray(end_effector, dtype=np.float32)
    joints = np.asarray(joint_position, dtype=np.float32)
    if cart.ndim != 2 or cart.shape[1] != 6:
        raise ValueError(f"RoboMind Franka end_effector must be (T,6), got {cart.shape}")
    if joints.ndim != 2 or joints.shape[1] < 1:
        raise ValueError(f"RoboMind Franka joint_position must be (T,>=1), got {joints.shape}")
    if cart.shape[0] != joints.shape[0]:
        raise ValueError(
            f"RoboMind Franka length mismatch: cart={cart.shape} joints={joints.shape}"
        )

    t = int(cart.shape[0])
    cart_slots = np.zeros((t, MAX_ARM_SLOTS, 6), dtype=np.float32)
    grip_slots = np.zeros((t, MAX_ARM_SLOTS, 1), dtype=np.float32)
    mask = np.zeros((t, MAX_ARM_SLOTS), dtype=np.float32)
    cart_slots[:, 0] = cart
    grip_slots[:, 0, 0] = joints[:, -1]
    mask[:, 0] = 1.0
    meta = robomind_franka_eef_pose_metadata()
    return pack_arm_slot_eef_pose(
        cartesian_rpy_slots=cart_slots,
        gripper_slots=grip_slots,
        slot_mask=mask,
    ), meta


def robomind_franka_eef_pose_metadata() -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "inactive", "exists": False},
        ],
        source_fields=["puppet/end_effector", "puppet/joint_position"],
        transform="robomind_franka_xyz_rpy_gripper_to_arm_slot_xyz_rot6d_gripper",
    )


def libero_franka_eef_pose_condition(
    *,
    ee_pos: np.ndarray,
    ee_ori_axis_angle: np.ndarray,
    gripper_states: np.ndarray,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build LIBERO Panda/Franka single-arm condition from robosuite state.

    LIBERO stores orientation as axis-angle, not roll/pitch/yaw.
    """
    pos = np.asarray(ee_pos, dtype=np.float32)
    ori = np.asarray(ee_ori_axis_angle, dtype=np.float32)
    grip = np.asarray(gripper_states, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"LIBERO ee_pos must be (T,3), got {pos.shape}")
    if ori.ndim != 2 or ori.shape[1] != 3:
        raise ValueError(f"LIBERO ee_ori must be axis-angle (T,3), got {ori.shape}")
    if grip.ndim != 2 or grip.shape[1] < 1:
        raise ValueError(f"LIBERO gripper_states must be (T,>=1), got {grip.shape}")
    if pos.shape[0] != ori.shape[0] or pos.shape[0] != grip.shape[0]:
        raise ValueError(
            f"LIBERO length mismatch: pos={pos.shape} ori={ori.shape} grip={grip.shape}"
        )
    if not np.isfinite(pos).all() or not np.isfinite(ori).all() or not np.isfinite(grip).all():
        raise ValueError("non-finite LIBERO EEF condition value")

    t = int(pos.shape[0])
    cond = np.zeros((t, MAX_ARM_SLOTS, ARM_SLOT_CONDITION_WIDTH), dtype=np.float32)
    gripper_unit = np.clip(grip[:, :1] / 0.04, 0.0, 1.0).astype(np.float32)
    cond[:, 0, :ARM_SLOT_EEF_POSE_DIM] = np.concatenate(
        [pos, axis_angle_to_rot6d(ori), gripper_unit],
        axis=-1,
    )
    cond[:, 0, ARM_SLOT_EEF_POSE_DIM] = 1.0
    meta = libero_franka_eef_pose_metadata()
    return np.ascontiguousarray(cond), meta


def libero_franka_eef_pose_metadata() -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "inactive", "exists": False},
        ],
        source_fields=[
            "obs/ee_pos",
            "obs/ee_ori",
            "obs/gripper_states",
        ],
        transform="libero_xyz_axis_angle_gripper_qpos_to_arm_slot_xyz_rot6d_gripper",
    )


def robomind_agilex_shared_fk_condition(
    *,
    eef_pose_slots: np.ndarray,
    gripper_raw_slots: np.ndarray,
    gripper_open_scale: float,
    source_fields: list[str] | None = None,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build dual-arm condition from official shared ``body_Link`` FK poses."""

    poses = np.asarray(eef_pose_slots, dtype=np.float64)
    raw_gripper = np.asarray(gripper_raw_slots, dtype=np.float32)
    if poses.ndim != 4 or poses.shape[1:] != (MAX_ARM_SLOTS, 4, 4):
        raise ValueError(f"RoboMind AgileX EEF poses must be (T,2,4,4), got {poses.shape}")
    if raw_gripper.ndim != 3 or raw_gripper.shape[1:] != (MAX_ARM_SLOTS, 1):
        raise ValueError(f"RoboMind AgileX gripper state must be (T,2,1), got {raw_gripper.shape}")
    if poses.shape[0] != raw_gripper.shape[0]:
        raise ValueError(
            f"RoboMind AgileX length mismatch: poses={poses.shape} gripper={raw_gripper.shape}"
        )

    t = int(poses.shape[0])
    grip_slots = np.zeros((t, MAX_ARM_SLOTS, 1), dtype=np.float32)
    for slot in range(MAX_ARM_SLOTS):
        grip_slots[:, slot, 0] = robomind_agilex_gripper_to_unit(
            raw_gripper[:, slot, 0], open_scale=gripper_open_scale
        )
    mask = np.ones((t, MAX_ARM_SLOTS), dtype=np.float32)
    meta = robomind_agilex_shared_fk_metadata(source_fields=source_fields)
    return pack_arm_slot_eef_pose_matrices(
        pose_slots=poses,
        gripper_slots=grip_slots,
        slot_mask=mask,
    ), meta


def robomind_agilex_shared_fk_metadata(
    *, source_fields: list[str] | None = None
) -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "right", "exists": True},
        ],
        source_fields=source_fields
        or [
            "puppet/joint_position_left",
            "puppet/joint_position_right",
        ],
        transform=(
            "robomind_agilex_qpos_official_aloha_urdf_body_Link_fk_to_arm_slot_xyz_rot6d_gripper"
        ),
    )


def robocoin_agilex_eef_pose_condition(
    *,
    eef_sim_pose_state: np.ndarray,
    gripper_open_scale_action: np.ndarray,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build RoboCOIN AgileX dual-arm EEF condition.

    RoboCOIN stores both arms in one 12D pose stream:
    ``[left_xyzrpy, right_xyzrpy]``.  Gripper open scale is already stored as
    two unit-scale channels in the observed action stream.
    """
    pose = np.asarray(eef_sim_pose_state, dtype=np.float32)
    grip = np.asarray(gripper_open_scale_action, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 12:
        raise ValueError(f"RoboCOIN eef_sim_pose_state must be (T,>=12), got {pose.shape}")
    if grip.ndim != 2 or grip.shape[1] < 2:
        raise ValueError(f"RoboCOIN gripper_open_scale_action must be (T,>=2), got {grip.shape}")
    if pose.shape[0] != grip.shape[0]:
        raise ValueError(f"RoboCOIN length mismatch: pose={pose.shape} grip={grip.shape}")

    t = int(pose.shape[0])
    cart_slots = np.zeros((t, MAX_ARM_SLOTS, 6), dtype=np.float32)
    grip_slots = np.zeros((t, MAX_ARM_SLOTS, 1), dtype=np.float32)
    mask = np.ones((t, MAX_ARM_SLOTS), dtype=np.float32)
    cart_slots[:, 0] = pose[:, :6]
    cart_slots[:, 1] = pose[:, 6:12]
    grip_slots[:, 0, 0] = np.clip(grip[:, 0], 0.0, 1.0)
    grip_slots[:, 1, 0] = np.clip(grip[:, 1], 0.0, 1.0)
    meta = robocoin_agilex_eef_pose_metadata()
    return pack_arm_slot_eef_pose(
        cartesian_rpy_slots=cart_slots,
        gripper_slots=grip_slots,
        slot_mask=mask,
    ), meta


def robocoin_agilex_eef_pose_metadata() -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": True},
            {"slot": 1, "side": "right", "exists": True},
        ],
        source_fields=[
            "eef_sim_pose_state",
            "gripper_open_scale_action",
        ],
        transform="robocoin_eef_sim_xyz_rpy_gripper_to_arm_slot_xyz_rot6d_gripper",
    )


def robotwin2_agilex_eef_pose_condition(
    *,
    left_endpose: np.ndarray,
    right_endpose: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, ArmSlotMetadata]:
    """Build RoboTwin2 clean dual-arm EEF condition from HDF5 endpose state.

    RoboTwin2 stores endpose as ``xyz + quaternion`` with transforms3d/Sapien
    quaternion order ``[qw, qx, qy, qz]``.
    """
    left = np.asarray(left_endpose, dtype=np.float32)
    right = np.asarray(right_endpose, dtype=np.float32)
    left_grip = np.asarray(left_gripper, dtype=np.float32).reshape(-1, 1)
    right_grip = np.asarray(right_gripper, dtype=np.float32).reshape(-1, 1)
    mask = np.asarray(slot_mask, dtype=np.float32)
    if left.ndim != 2 or left.shape[1] < 7 or right.ndim != 2 or right.shape[1] < 7:
        raise ValueError(
            f"RoboTwin2 endpose arrays must be (T,>=7), got left={left.shape} right={right.shape}"
        )
    if mask.ndim != 2 or mask.shape[1] != MAX_ARM_SLOTS:
        raise ValueError(f"slot_mask must be (T,2), got {mask.shape}")
    t = int(left.shape[0])
    if (
        right.shape[0] != t
        or left_grip.shape[0] != t
        or right_grip.shape[0] != t
        or mask.shape[0] != t
    ):
        raise ValueError(
            "RoboTwin2 length mismatch: "
            f"left={left.shape} right={right.shape} "
            f"left_gripper={left_grip.shape} right_gripper={right_grip.shape} mask={mask.shape}"
        )
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite RoboTwin2 endpose value")

    cond = np.zeros((t, MAX_ARM_SLOTS, ARM_SLOT_CONDITION_WIDTH), dtype=np.float32)
    slot_inputs = (
        (0, left[:, :3], left[:, 3:7], left_grip[:, :1]),
        (1, right[:, :3], right[:, 3:7], right_grip[:, :1]),
    )
    for slot, xyz, quat, grip in slot_inputs:
        values = np.concatenate(
            [xyz, quat_wxyz_to_rot6d(quat), np.clip(grip, 0.0, 1.0)],
            axis=-1,
        )
        cond[:, slot, :ARM_SLOT_EEF_POSE_DIM] = values
        cond[:, slot, ARM_SLOT_EEF_POSE_DIM] = (mask[:, slot] > 0.5).astype(np.float32)
    cond[..., :ARM_SLOT_EEF_POSE_DIM] *= cond[
        ..., ARM_SLOT_EEF_POSE_DIM : ARM_SLOT_EEF_POSE_DIM + 1
    ]
    if not np.isfinite(cond).all():
        raise ValueError("non-finite value in packed RoboTwin2 arm-slot condition")
    meta = robotwin2_agilex_eef_pose_metadata(
        slot0_exists=bool(np.any(mask[:, 0] > 0.5)),
        slot1_exists=bool(np.any(mask[:, 1] > 0.5)),
    )
    return np.ascontiguousarray(cond), meta


def robotwin2_agilex_eef_pose_metadata(
    *,
    slot0_exists: bool = True,
    slot1_exists: bool = True,
) -> ArmSlotMetadata:
    return ArmSlotMetadata(
        arm_slots=[
            {"slot": 0, "side": "left", "exists": bool(slot0_exists)},
            {"slot": 1, "side": "right", "exists": bool(slot1_exists)},
        ],
        source_fields=[
            "endpose.left_endpose",
            "endpose.right_endpose",
            "endpose.left_gripper",
            "endpose.right_gripper",
            "observation.arm_slot.mask",
        ],
        transform="robotwin2_xyz_quat_wxyz_gripper_mask_to_arm_slot_xyz_rot6d_gripper",
    )
