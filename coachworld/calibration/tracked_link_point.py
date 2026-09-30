"""Shared camera initialization from RGB-D tracks attached to robot links.

Each observed point is assumed to be rigidly attached to a known robot link,
but its local surface coordinate is unknown.  The solver estimates one shared
``T_camera_body`` and one local 3-D point per track instead of incorrectly
treating a tracked wrist pixel as the link origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LinkPointTrack:
    """One RGB-D point track paired with body-from-link FK poses."""

    track_id: str
    body_from_link: np.ndarray
    camera_points: np.ndarray
    fit_mask: np.ndarray | None = None

    def validated(self) -> "LinkPointTrack":
        poses = np.asarray(self.body_from_link, dtype=np.float64)
        points = np.asarray(self.camera_points, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"{self.track_id}: body_from_link must be (N,4,4), got {poses.shape}")
        if points.shape != (len(poses), 3):
            raise ValueError(
                f"{self.track_id}: camera_points must be ({len(poses)},3), got {points.shape}"
            )
        finite = np.isfinite(poses).all(axis=(1, 2)) & np.isfinite(points).all(axis=1)
        if self.fit_mask is not None:
            mask = np.asarray(self.fit_mask, dtype=bool).reshape(-1)
            if mask.shape != (len(poses),):
                raise ValueError(f"{self.track_id}: fit_mask must be ({len(poses)},), got {mask.shape}")
            finite &= mask
        if int(finite.sum()) < 3:
            raise ValueError(f"{self.track_id}: fewer than three finite fit observations")
        return LinkPointTrack(
            track_id=str(self.track_id),
            body_from_link=np.ascontiguousarray(poses),
            camera_points=np.ascontiguousarray(points),
            fit_mask=np.ascontiguousarray(finite),
        )


@dataclass(frozen=True)
class TrackedLinkPointSolution:
    camera_from_body: np.ndarray
    local_points: dict[str, np.ndarray]
    selected_rotation_prior: int
    fit_residuals_m: np.ndarray
    optimization_success: bool
    optimization_message: str


def _linear_solution_for_rotation(
    tracks: Sequence[LinkPointTrack],
    rotation_camera_from_body: np.ndarray,
    *,
    local_point_regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve camera translation and per-track link points for fixed rotation."""

    rotation = np.asarray(rotation_camera_from_body, dtype=np.float64).reshape(3, 3)
    track_count = len(tracks)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for track_idx, track in enumerate(tracks):
        fit = np.asarray(track.fit_mask, dtype=bool)
        for body_from_link, observed in zip(track.body_from_link[fit], track.camera_points[fit]):
            row = np.zeros((3, 3 + 3 * track_count), dtype=np.float64)
            row[:, :3] = np.eye(3, dtype=np.float64)
            row[:, 3 + 3 * track_idx : 6 + 3 * track_idx] = rotation @ body_from_link[:3, :3]
            rows.append(row)
            targets.append(observed - rotation @ body_from_link[:3, 3])
    if local_point_regularization > 0.0:
        weight = float(np.sqrt(local_point_regularization))
        for track_idx in range(track_count):
            row = np.zeros((3, 3 + 3 * track_count), dtype=np.float64)
            row[:, 3 + 3 * track_idx : 6 + 3 * track_idx] = weight * np.eye(3)
            rows.append(row)
            targets.append(np.zeros(3, dtype=np.float64))
    matrix = np.concatenate(rows, axis=0)
    target = np.concatenate(targets, axis=0)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    translation = solution[:3]
    local_points = solution[3:].reshape(track_count, 3)
    residuals = _observation_residuals(tracks, rotation, translation, local_points)
    return translation, local_points, residuals


def _observation_residuals(
    tracks: Sequence[LinkPointTrack],
    rotation: np.ndarray,
    translation: np.ndarray,
    local_points: np.ndarray,
    *,
    fit_only: bool = True,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for track_idx, track in enumerate(tracks):
        mask = np.asarray(track.fit_mask, dtype=bool) if fit_only else np.ones(len(track.camera_points), dtype=bool)
        poses = track.body_from_link[mask]
        observed = track.camera_points[mask]
        finite = np.isfinite(observed).all(axis=1) & np.isfinite(poses).all(axis=(1, 2))
        poses = poses[finite]
        observed = observed[finite]
        body_points = (
            np.einsum("nij,j->ni", poses[:, :3, :3], local_points[track_idx])
            + poses[:, :3, 3]
        )
        predicted = body_points @ rotation.T + translation
        residuals.append(predicted - observed)
    if not residuals:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(residuals, axis=0)


def solve_tracked_link_points(
    tracks: Sequence[LinkPointTrack],
    rotation_priors: np.ndarray,
    *,
    robust_scale_m: float = 0.02,
    rotation_prior_sigma_deg: float = 20.0,
    local_point_sigma_m: float = 0.15,
    local_point_bound_m: float = 0.12,
    local_point_regularization: float = 1.0e-3,
    max_nfev: int = 500,
) -> TrackedLinkPointSolution:
    """Estimate one camera pose and per-track rigid surface coordinates.

    Rotation priors are ranked using a fixed-rotation linear solve.  Only the
    best candidate enters nonlinear least-squares; this is inexpensive even
    when hundreds of previously reviewed camera rotations are supplied.
    """

    validated = [track.validated() for track in tracks]
    if not validated:
        raise ValueError("at least one link-point track is required")
    priors = np.asarray(rotation_priors, dtype=np.float64)
    if priors.ndim != 3 or priors.shape[1:] != (3, 3) or len(priors) == 0:
        raise ValueError(f"rotation_priors must be (P,3,3), got {priors.shape}")
    if not np.isfinite(priors).all():
        raise ValueError("rotation_priors contain non-finite values")

    candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for prior_idx, rotation in enumerate(priors):
        translation, local_points, residuals = _linear_solution_for_rotation(
            validated,
            rotation,
            local_point_regularization=local_point_regularization,
        )
        norms = np.linalg.norm(residuals, axis=1)
        robust_score = float(np.median(norms) + 0.25 * np.quantile(norms, 0.9))
        local_penalty = float(np.mean(np.square(np.linalg.norm(local_points, axis=1) / local_point_sigma_m)))
        candidates.append((robust_score + 0.02 * local_penalty, prior_idx, translation, local_points))
    _, selected_prior, translation0, local_points0 = min(candidates, key=lambda row: row[0])
    rotation0 = priors[selected_prior]
    rotation0_obj = Rotation.from_matrix(rotation0)
    local_bound = float(local_point_bound_m)
    if not np.isfinite(local_bound) or local_bound <= 0.0:
        raise ValueError("local_point_bound_m must be finite and positive")
    local_points0 = np.clip(local_points0, -0.95 * local_bound, 0.95 * local_bound)
    x0 = np.concatenate([rotation0_obj.as_rotvec(), translation0, local_points0.reshape(-1)])
    robust_scale = float(robust_scale_m)
    rotation_sigma = float(np.deg2rad(rotation_prior_sigma_deg))

    def residual_vector(values: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(values[:3]).as_matrix()
        translation = values[3:6]
        local_points = values[6:].reshape(len(validated), 3)
        observation = _observation_residuals(validated, rotation, translation, local_points).reshape(-1)
        rotation_delta = (rotation0_obj.inv() * Rotation.from_matrix(rotation)).as_rotvec()
        rotation_prior = rotation_delta * (robust_scale / max(rotation_sigma, 1.0e-8))
        local_prior = local_points.reshape(-1) * (robust_scale / max(local_point_sigma_m, 1.0e-8))
        return np.concatenate([observation, rotation_prior, local_prior])

    optimized = least_squares(
        residual_vector,
        x0,
        bounds=(
            np.concatenate(
                [
                    np.full(6, -np.inf, dtype=np.float64),
                    np.full(3 * len(validated), -local_bound, dtype=np.float64),
                ]
            ),
            np.concatenate(
                [
                    np.full(6, np.inf, dtype=np.float64),
                    np.full(3 * len(validated), local_bound, dtype=np.float64),
                ]
            ),
        ),
        loss="soft_l1",
        f_scale=robust_scale,
        max_nfev=int(max_nfev),
        x_scale="jac",
    )
    values = optimized.x
    rotation = Rotation.from_rotvec(values[:3]).as_matrix()
    translation = values[3:6]
    local_points = values[6:].reshape(len(validated), 3)
    camera_from_body = np.eye(4, dtype=np.float64)
    camera_from_body[:3, :3] = rotation
    camera_from_body[:3, 3] = translation
    residuals = _observation_residuals(validated, rotation, translation, local_points)
    return TrackedLinkPointSolution(
        camera_from_body=camera_from_body,
        local_points={track.track_id: local_points[idx].copy() for idx, track in enumerate(validated)},
        selected_rotation_prior=int(selected_prior),
        fit_residuals_m=np.linalg.norm(residuals, axis=1),
        optimization_success=bool(optimized.success),
        optimization_message=str(optimized.message),
    )


def evaluate_tracked_link_points(
    tracks: Sequence[LinkPointTrack],
    solution: TrackedLinkPointSolution,
    *,
    fit_only: bool = False,
) -> np.ndarray:
    """Return per-observation metric residuals for fit or held-out auditing."""

    validated = [track.validated() for track in tracks]
    local_points = np.stack([solution.local_points[track.track_id] for track in validated], axis=0)
    residuals = _observation_residuals(
        validated,
        solution.camera_from_body[:3, :3],
        solution.camera_from_body[:3, 3],
        local_points,
        fit_only=fit_only,
    )
    return np.linalg.norm(residuals, axis=1)
