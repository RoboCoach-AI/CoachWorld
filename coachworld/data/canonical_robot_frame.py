"""Canonical robot-frame transforms for heterogeneous CoachWorld datasets.

The dual-arm canonical frame follows the RoboTwin Aloha world convention:

* +X points to the robot's physical right.
* +Y points forward from the robot toward the workspace.
* +Z points upward.
* The origin is the RoboTwin robot/world origin.

Real dual-Piper datasets use a shared rig frame with +X forward and +Y left.
RoboTwin places the nearly identical arm rig at ``[0, -0.65, 0]`` after a
+90 degree rotation around Z, which gives the fixed transform below.  This is
an embodiment/configuration transform; it is never fitted from trajectories.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import numpy as np


DUAL_ARM_CANONICAL_FRAME_NAME = "robotwin_world_x_right_y_forward_z_up"
SINGLE_ARM_CANONICAL_FRAME_NAME = "franka_base_x_forward_y_left_z_up"
# Backward-compatible alias for callers written before the single-arm contract.
CANONICAL_FRAME_NAME = DUAL_ARM_CANONICAL_FRAME_NAME

_IDENTITY = np.eye(4, dtype=np.float64)

T_CANONICAL_FROM_PIPER_RIG = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -0.65],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

PIPER_DATASETS = frozenset({"robocoin", "vifailback", "giagai"})
ROBOTWIN_DATASETS = frozenset({"robotwin", "robotwin2", "robotwin2_clean_50"})


def canonical_from_native(dataset: str) -> np.ndarray:
    """Return ``T_canonical_native`` for a supported dual-arm dataset."""

    name = str(dataset).strip().lower()
    if name in PIPER_DATASETS:
        return T_CANONICAL_FROM_PIPER_RIG.copy()
    if name in ROBOTWIN_DATASETS:
        return _IDENTITY.copy()
    raise KeyError(f"no canonical robot-frame transform registered for dataset={dataset!r}")


def rotation_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a robotics/MuJoCo ``[w,x,y,z]`` quaternion to a matrix."""

    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_transform_from_pos_quat_wxyz(
    position: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Build ``T_parent_child`` from a child pose expressed in its parent."""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_quat_wxyz(quat_wxyz)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    assert_rigid_transform(transform)
    return transform


def libero_base_from_world(model_xml: str) -> np.ndarray:
    """Recover ``T_robot_base_world`` from a LIBERO demo model XML."""

    root = ET.fromstring(str(model_xml))
    body = root.find(".//body[@name='robot0_base']")
    if body is None:
        raise ValueError("LIBERO model XML has no robot0_base body")
    position = np.fromstring(body.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float64)
    quat = np.fromstring(body.attrib.get("quat", "1 0 0 0"), sep=" ", dtype=np.float64)
    if position.shape != (3,) or quat.shape != (4,):
        raise ValueError(f"invalid robot0_base pose: pos={position}, quat={quat}")
    world_from_base = rigid_transform_from_pos_quat_wxyz(position, quat)
    base_from_world = np.linalg.inv(world_from_base)
    assert_rigid_transform(base_from_world)
    return base_from_world


def transform_xyz(xyz: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to an array whose final dimension is XYZ."""

    points = np.asarray(xyz, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"expected (...,3) XYZ points, got {points.shape}")
    T = np.asarray(T_target_source, dtype=np.float64).reshape(4, 4)
    return points @ T[:3, :3].T + T[:3, 3]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert first-two-column 6D rotations to proper rotation matrices."""

    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape[-1] != 6:
        raise ValueError(f"rot6d must end with dimension 6, got {values.shape}")
    first = values[..., :3]
    second = values[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= 1.0e-12):
        raise ValueError("rot6d first column has zero norm")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm <= 1.0e-12):
        raise ValueError("rot6d columns are linearly dependent")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=-1)


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Extract the first two columns using the CoachWorld rot6d layout."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end with shape (3,3), got {matrix.shape}")
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def transform_rot6d(rot6d: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    """Express EEF orientations in a target coordinate frame."""

    source_from_eef = rot6d_to_matrix(rot6d)
    target_from_source = np.asarray(T_target_source, dtype=np.float64).reshape(4, 4)[:3, :3]
    target_from_eef = np.einsum("ij,...jk->...ik", target_from_source, source_from_eef)
    return matrix_to_rot6d(target_from_eef)


def transform_arm_slot_condition(
    condition: np.ndarray,
    T_target_source: np.ndarray,
) -> np.ndarray:
    """Transform ``xyz+rot6d`` while preserving gripper and slot masks."""

    values = np.asarray(condition)
    if values.ndim != 3 or values.shape[-1] != 11:
        raise ValueError(f"arm-slot condition must be (T,slots,11), got {values.shape}")
    out = values.astype(np.float64, copy=True)
    active = out[..., 10] >= 0.5
    if bool(active.any()):
        out[..., :3][active] = transform_xyz(out[..., :3][active], T_target_source)
        out[..., 3:9][active] = transform_rot6d(out[..., 3:9][active], T_target_source)
    out[..., :10][~active] = 0.0
    dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype(np.float32)
    return np.ascontiguousarray(out.astype(dtype))


def transform_pose_matrices(poses: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    """Left-multiply homogeneous poses, transforming position and orientation."""

    matrices = np.asarray(poses, dtype=np.float64)
    if matrices.shape[-2:] != (4, 4):
        raise ValueError(f"expected (...,4,4) pose matrices, got {matrices.shape}")
    T = np.asarray(T_target_source, dtype=np.float64).reshape(4, 4)
    return np.einsum("ij,...jk->...ik", T, matrices)


def camera_from_canonical(
    T_camera_native: np.ndarray,
    T_canonical_native: np.ndarray,
) -> np.ndarray:
    """Rewrite a native-frame camera extrinsic for canonical-frame points."""

    camera_native = np.asarray(T_camera_native, dtype=np.float64).reshape(4, 4)
    canonical_native = np.asarray(T_canonical_native, dtype=np.float64).reshape(4, 4)
    return camera_native @ np.linalg.inv(canonical_native)


def assert_rigid_transform(T: np.ndarray, *, atol: float = 1.0e-8) -> None:
    """Raise when a candidate frame transform is not a finite proper SE(3)."""

    matrix = np.asarray(T, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("canonical transform must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol):
        raise ValueError("canonical transform rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol):
        raise ValueError("canonical transform rotation is not proper")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=atol):
        raise ValueError("canonical transform has an invalid homogeneous row")
