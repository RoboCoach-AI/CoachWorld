"""Robust joint camera intrinsics/extrinsics fitting from robot-frame EEF points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.optimize import least_squares

CameraModel = Literal["fixed", "focal_scale", "full"]


@dataclass(frozen=True)
class EEFCameraBundleSolution:
    """One camera fit and its unweighted pixel reprojection diagnostics."""

    model: CameraModel
    K: np.ndarray
    camera_from_robot: np.ndarray
    reprojection_errors_px: np.ndarray
    depths_m: np.ndarray
    optimization_success: bool
    optimization_message: str
    optimization_cost: float
    optimization_evaluations: int


def project_eef_points(
    object_points: np.ndarray,
    camera_from_robot: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project robot-frame points with ``camera_from_robot`` and pinhole ``K``."""

    points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(camera_from_robot, dtype=np.float64).reshape(4, 4)
    intrinsic = np.asarray(K, dtype=np.float64).reshape(3, 3)
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    depths = camera_points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera_points).all(axis=1) & (depths > 1.0e-8)
    uv[valid, 0] = (
        intrinsic[0, 0] * camera_points[valid, 0] / depths[valid] + intrinsic[0, 2]
    )
    uv[valid, 1] = (
        intrinsic[1, 1] * camera_points[valid, 1] / depths[valid] + intrinsic[1, 2]
    )
    return uv, depths


def _validate_inputs(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(object_points, dtype=np.float64)
    uv = np.asarray(image_points, dtype=np.float64)
    K = np.asarray(K_prior, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"object_points must be (N,3), got {xyz.shape}")
    if uv.shape != (len(xyz), 2):
        raise ValueError(f"image_points must be ({len(xyz)},2), got {uv.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K_prior must be (3,3), got {K.shape}")
    if len(xyz) < 6:
        raise ValueError("at least six 3D-2D correspondences are required")
    if not np.isfinite(xyz).all() or not np.isfinite(uv).all() or not np.isfinite(K).all():
        raise ValueError("camera bundle inputs contain non-finite values")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError("K_prior focal lengths must be positive")
    return xyz, uv, K


def _initial_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    *,
    reprojection_error_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        K.astype(np.float64),
        distortion,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(reprojection_error_px),
        iterationsCount=5000,
        confidence=0.999,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float64),
            image_points.astype(np.float64),
            K.astype(np.float64),
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
        )
        inliers = None
    if not ok:
        raise RuntimeError("OpenCV failed to initialize camera pose")
    if inliers is not None and len(inliers) >= 6:
        indices = inliers.reshape(-1)
        refined, rvec, tvec = cv2.solvePnP(
            object_points[indices].astype(np.float64),
            image_points[indices].astype(np.float64),
            K.astype(np.float64),
            distortion,
            rvec=np.asarray(rvec, dtype=np.float64),
            tvec=np.asarray(tvec, dtype=np.float64),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not refined:
            raise RuntimeError("OpenCV failed to refine the initialized camera pose")
    return np.asarray(rvec, dtype=np.float64).reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3)


def _decode_parameters(
    values: np.ndarray,
    *,
    model: CameraModel,
    K_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, _ = cv2.Rodrigues(np.asarray(values[:3], dtype=np.float64))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = values[3:6]
    K = np.asarray(K_prior, dtype=np.float64).copy()
    if model == "focal_scale":
        scale = float(np.exp(values[6]))
        K[0, 0] *= scale
        K[1, 1] *= scale
    elif model == "full":
        K[0, 0] *= float(np.exp(values[6]))
        K[1, 1] *= float(np.exp(values[7]))
        K[0, 2] += float(values[8])
        K[1, 2] += float(values[9])
    return transform, K


def solve_eef_camera_bundle(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K_prior: np.ndarray,
    *,
    model: CameraModel = "fixed",
    robust_scale_px: float = 8.0,
    focal_prior_rel: float = 0.10,
    principal_prior_px: float = 32.0,
    max_focal_delta_rel: float = 0.35,
    max_principal_delta_px: float = 96.0,
    minimum_depth_m: float = 0.05,
    max_nfev: int = 3000,
    initial_reprojection_error_px: float = 12.0,
) -> EEFCameraBundleSolution:
    """Fit a robust pinhole camera while keeping intrinsics near a supplied prior.

    ``fixed`` optimizes only camera pose. ``focal_scale`` adds one shared focal
    multiplier. ``full`` adds independent focal multipliers and principal-point
    offsets. Distortion is deliberately excluded because sparse EEF tracks do
    not constrain it reliably.
    """

    if model not in {"fixed", "focal_scale", "full"}:
        raise ValueError(f"unsupported camera model: {model}")
    xyz, observed_uv, prior = _validate_inputs(object_points, image_points, K_prior)
    if robust_scale_px <= 0.0:
        raise ValueError("robust_scale_px must be positive")
    if focal_prior_rel <= 0.0 or principal_prior_px <= 0.0:
        raise ValueError("intrinsic prior scales must be positive")
    rvec0, tvec0 = _initial_pose(
        xyz,
        observed_uv,
        prior,
        reprojection_error_px=float(initial_reprojection_error_px),
    )
    parameter_count = {"fixed": 6, "focal_scale": 7, "full": 10}[model]
    x0 = np.zeros(parameter_count, dtype=np.float64)
    x0[:3] = rvec0
    x0[3:6] = tvec0

    lower = np.full(parameter_count, -np.inf, dtype=np.float64)
    upper = np.full(parameter_count, np.inf, dtype=np.float64)
    focal_bound = float(np.log1p(max_focal_delta_rel))
    if model == "focal_scale":
        lower[6], upper[6] = -focal_bound, focal_bound
    elif model == "full":
        lower[6:8], upper[6:8] = -focal_bound, focal_bound
        lower[8:10], upper[8:10] = -float(max_principal_delta_px), float(
            max_principal_delta_px
        )

    prior_weight_px = float(robust_scale_px)
    minimum_depth = float(minimum_depth_m)

    def residual_vector(values: np.ndarray) -> np.ndarray:
        transform, K = _decode_parameters(values, model=model, K_prior=prior)
        camera_points = xyz @ transform[:3, :3].T + transform[:3, 3]
        depths = camera_points[:, 2]
        safe_depths = np.maximum(depths, minimum_depth)
        projected = np.column_stack(
            [
                K[0, 0] * camera_points[:, 0] / safe_depths + K[0, 2],
                K[1, 1] * camera_points[:, 1] / safe_depths + K[1, 2],
            ]
        )
        observation = (projected - observed_uv).reshape(-1)
        depth_penalty = np.maximum(0.0, minimum_depth - depths) * 1000.0
        priors: list[float] = []
        if model == "focal_scale":
            priors.append(values[6] / focal_prior_rel * prior_weight_px)
        elif model == "full":
            priors.extend((values[6:8] / focal_prior_rel * prior_weight_px).tolist())
            priors.extend(
                (values[8:10] / principal_prior_px * prior_weight_px).tolist()
            )
        return np.concatenate([observation, depth_penalty, np.asarray(priors)])

    optimized = least_squares(
        residual_vector,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=float(robust_scale_px),
        x_scale="jac",
        max_nfev=int(max_nfev),
    )
    transform, K = _decode_parameters(optimized.x, model=model, K_prior=prior)
    projected, depths = project_eef_points(xyz, transform, K)
    errors = np.linalg.norm(projected - observed_uv, axis=1)
    return EEFCameraBundleSolution(
        model=model,
        K=K,
        camera_from_robot=transform,
        reprojection_errors_px=errors,
        depths_m=depths,
        optimization_success=bool(optimized.success),
        optimization_message=str(optimized.message),
        optimization_cost=float(optimized.cost),
        optimization_evaluations=int(optimized.nfev),
    )
