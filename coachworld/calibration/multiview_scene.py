"""Metric alignment helpers for calibrated multi-view scene reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SimilarityAlignment:
    """Similarity transform mapping reconstruction coordinates to robot base."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    camera_center_errors_m: np.ndarray
    camera_rotation_errors_deg: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        xyz = np.asarray(points, dtype=np.float64)
        return (
            self.scale * (xyz @ self.rotation.T) + self.translation[None, :]
        ).astype(np.float32)

    def transform_c2w(self, camera_to_world: np.ndarray) -> np.ndarray:
        c2w = np.asarray(camera_to_world, dtype=np.float64)
        single = c2w.ndim == 2
        if single:
            c2w = c2w[None, ...]
        out = np.tile(np.eye(4, dtype=np.float64), (len(c2w), 1, 1))
        out[:, :3, :3] = np.einsum(
            "ij,njk->nik", self.rotation, c2w[:, :3, :3]
        )
        out[:, :3, 3] = (
            self.scale
            * np.einsum("ij,nj->ni", self.rotation, c2w[:, :3, 3])
            + self.translation[None, :]
        )
        result = out.astype(np.float32)
        return result[0] if single else result

    def to_json(self) -> dict[str, object]:
        center = np.asarray(self.camera_center_errors_m, dtype=np.float64)
        rotation = np.asarray(self.camera_rotation_errors_deg, dtype=np.float64)
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.astype(float).tolist(),
            "translation": self.translation.astype(float).tolist(),
            "camera_center_errors_m": center.astype(float).tolist(),
            "camera_center_rms_m": float(np.sqrt(np.mean(center * center))),
            "camera_center_max_m": float(np.max(center)),
            "camera_rotation_errors_deg": rotation.astype(float).tolist(),
            "camera_rotation_rms_deg": float(
                np.sqrt(np.mean(rotation * rotation))
            ),
            "camera_rotation_max_deg": float(np.max(rotation)),
        }


def camera_to_world(camera_from_world: np.ndarray) -> np.ndarray:
    """Invert one or more homogeneous world-to-camera matrices."""

    mats = np.asarray(camera_from_world, dtype=np.float64)
    single = mats.ndim == 2
    if single:
        mats = mats[None, ...]
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {mats.shape}")
    result = np.linalg.inv(mats)
    return result[0] if single else result


def _mean_global_rotation(
    source_camera_to_world: np.ndarray,
    target_camera_to_world: np.ndarray,
) -> np.ndarray:
    candidates = np.einsum(
        "nij,nkj->nik",
        target_camera_to_world[:, :3, :3],
        source_camera_to_world[:, :3, :3],
    )
    return Rotation.from_matrix(candidates).mean().as_matrix()


def _scale_translation_for_rotation(
    source_centers: np.ndarray,
    target_centers: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray]:
    rotated = source_centers @ rotation.T
    source_mean = rotated.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = rotated - source_mean
    target_centered = target_centers - target_mean
    denom = float(np.sum(source_centered * source_centered))
    if denom <= 1e-12:
        raise ValueError("camera centers do not constrain similarity scale")
    scale = float(np.sum(source_centered * target_centered) / denom)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"invalid initial similarity scale: {scale}")
    translation = target_mean - scale * source_mean
    return scale, translation


def estimate_camera_similarity(
    source_camera_from_world: np.ndarray,
    target_camera_from_robot: np.ndarray,
    *,
    center_sigma_m: float = 0.02,
    rotation_sigma_deg: float = 2.0,
) -> SimilarityAlignment:
    """Align a reconstruction gauge to a metric robot-base frame.

    Both inputs are world/base-to-camera matrices for synchronized views.  The
    optimized transform combines camera-center and camera-orientation residuals,
    which avoids the mirror and axis ambiguities of center-only alignment.
    """

    source_w2c = np.asarray(source_camera_from_world, dtype=np.float64)
    target_w2c = np.asarray(target_camera_from_robot, dtype=np.float64)
    if source_w2c.shape != target_w2c.shape:
        raise ValueError(
            f"source/target camera shapes differ: {source_w2c.shape} vs "
            f"{target_w2c.shape}"
        )
    if source_w2c.ndim != 3 or source_w2c.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {source_w2c.shape}")
    if len(source_w2c) < 2:
        raise ValueError("at least two calibrated cameras are required")
    if not np.isfinite(source_w2c).all() or not np.isfinite(target_w2c).all():
        raise ValueError("camera matrices contain non-finite values")

    source_c2w = camera_to_world(source_w2c)
    target_c2w = camera_to_world(target_w2c)
    source_centers = source_c2w[:, :3, 3]
    target_centers = target_c2w[:, :3, 3]
    initial_rotation = _mean_global_rotation(source_c2w, target_c2w)
    initial_scale, initial_translation = _scale_translation_for_rotation(
        source_centers,
        target_centers,
        initial_rotation,
    )
    x0 = np.concatenate(
        [
            Rotation.from_matrix(initial_rotation).as_rotvec(),
            [np.log(initial_scale)],
            initial_translation,
        ]
    )
    center_sigma = max(float(center_sigma_m), 1e-6)
    rotation_sigma = max(float(np.deg2rad(rotation_sigma_deg)), 1e-6)

    def residual(params: np.ndarray) -> np.ndarray:
        global_rotation = Rotation.from_rotvec(params[:3]).as_matrix()
        scale = float(np.exp(params[3]))
        translation = params[4:7]
        aligned_centers = (
            scale * (source_centers @ global_rotation.T)
            + translation[None, :]
        )
        center_residual = (
            aligned_centers - target_centers
        ).reshape(-1) / center_sigma
        predicted_rotations = np.einsum(
            "ij,njk->nik", global_rotation, source_c2w[:, :3, :3]
        )
        relative = np.einsum(
            "nji,njk->nik",
            target_c2w[:, :3, :3],
            predicted_rotations,
        )
        rotation_residual = (
            Rotation.from_matrix(relative).as_rotvec().reshape(-1)
            / rotation_sigma
        )
        return np.concatenate([center_residual, rotation_residual])

    result = least_squares(
        residual,
        x0,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=300,
    )
    global_rotation = Rotation.from_rotvec(result.x[:3]).as_matrix()
    scale = float(np.exp(result.x[3]))
    translation = result.x[4:7]
    aligned_centers = (
        scale * (source_centers @ global_rotation.T) + translation[None, :]
    )
    center_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
    predicted_rotations = np.einsum(
        "ij,njk->nik", global_rotation, source_c2w[:, :3, :3]
    )
    relative = np.einsum(
        "nji,njk->nik",
        target_c2w[:, :3, :3],
        predicted_rotations,
    )
    rotation_errors = np.rad2deg(
        Rotation.from_matrix(relative).magnitude()
    )
    return SimilarityAlignment(
        scale=scale,
        rotation=global_rotation.astype(np.float64),
        translation=translation.astype(np.float64),
        camera_center_errors_m=center_errors.astype(np.float64),
        camera_rotation_errors_deg=rotation_errors.astype(np.float64),
    )


def valid_camera_from_robot_mask(
    camera_from_robot: np.ndarray,
    *,
    center_norm_range_m: tuple[float, float] = (0.2, 1.5),
    center_z_range_m: tuple[float, float] = (-0.1, 1.6),
    orthogonality_tolerance: float = 0.04,
    determinant_tolerance: float = 0.04,
) -> np.ndarray:
    """Reject malformed or physically implausible camera transforms."""

    mats = np.asarray(camera_from_robot, dtype=np.float64)
    if mats.ndim != 3 or mats.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {mats.shape}")
    finite = np.isfinite(mats).all(axis=(1, 2))
    rotations = mats[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)[None, ...]
    gram = np.einsum("nji,njk->nik", rotations, rotations)
    orthogonality = np.linalg.norm(gram - identity, axis=(1, 2))
    determinant = np.linalg.det(rotations)
    safe = mats.copy()
    safe[~finite] = np.eye(4, dtype=np.float64)
    centers = np.linalg.inv(safe)[:, :3, 3]
    norms = np.linalg.norm(centers, axis=1)
    return (
        finite
        & (orthogonality <= float(orthogonality_tolerance))
        & (np.abs(determinant - 1.0) <= float(determinant_tolerance))
        & (norms >= float(center_norm_range_m[0]))
        & (norms <= float(center_norm_range_m[1]))
        & (centers[:, 2] >= float(center_z_range_m[0]))
        & (centers[:, 2] <= float(center_z_range_m[1]))
    )


def diverse_camera_indices(
    camera_from_robot: np.ndarray,
    count: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Farthest-point sample real camera poses by center and view direction."""

    mats = np.asarray(camera_from_robot, dtype=np.float64)
    count = min(max(int(count), 0), len(mats))
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    c2w = camera_to_world(mats)
    centers = c2w[:, :3, 3]
    forwards = c2w[:, :3, 2]
    center_scale = np.maximum(
        np.quantile(np.abs(centers - np.median(centers, axis=0)), 0.8, axis=0),
        np.array([0.15, 0.15, 0.1], dtype=np.float64),
    )
    features = np.concatenate(
        [centers / center_scale[None, :], forwards * 0.8], axis=1
    )
    rng = np.random.default_rng(int(seed))
    centroid = features.mean(axis=0)
    distance = np.sum((features - centroid[None, :]) ** 2, axis=1)
    distance += rng.uniform(0.0, 1e-9, size=len(distance))
    selected = np.empty((count,), dtype=np.int64)
    selected[0] = int(np.argmax(distance))
    nearest = np.sum(
        (features - features[selected[0]][None, :]) ** 2, axis=1
    )
    for i in range(1, count):
        selected[i] = int(np.argmax(nearest))
        candidate_distance = np.sum(
            (features - features[selected[i]][None, :]) ** 2, axis=1
        )
        nearest = np.minimum(nearest, candidate_distance)
    return selected
