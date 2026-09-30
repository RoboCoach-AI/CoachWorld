"""Project canonical arm-slot EEF positions into calibrated image views."""

from __future__ import annotations

import numpy as np


def select_condition_at_latent_frames(
    condition: np.ndarray,
    latent_ids: list[int],
    *,
    vae_temporal_stride: int,
) -> np.ndarray:
    """Select raw condition rows aligned with latent-frame anchors."""

    values = np.asarray(condition, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 4:
        raise ValueError(
            "arm-slot condition must have shape (T,S,D+1) with XYZ and a mask, "
            f"got {values.shape}"
        )
    if values.shape[0] < 1:
        raise ValueError("arm-slot condition cannot be empty")
    stride = int(vae_temporal_stride)
    if stride <= 0:
        raise ValueError(f"vae_temporal_stride must be positive, got {stride}")
    frame_ids = np.asarray(
        [min(max(0, int(latent_id) * stride), values.shape[0] - 1) for latent_id in latent_ids],
        dtype=np.int64,
    )
    return np.ascontiguousarray(values[frame_ids])


def project_arm_slot_condition(
    condition: np.ndarray,
    camera_from_canonical: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_hw: tuple[int, int] | list[int] | np.ndarray,
    min_positive_depth: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Project raw canonical XYZ into every calibrated camera view.

    Args:
        condition: Raw, unnormalized ``(F,S,D+1)`` arm-slot condition. XYZ is
            read from the first three channels and the final channel is the
            binary ``slot_exists`` mask.
        camera_from_canonical: OpenCV extrinsics with shape ``(F,V,4,4)``.
        intrinsics: Target-image-space intrinsics with shape ``(F,V,3,3)``.
        image_hw: ``(height,width)`` or an array broadcastable to ``(F,V,2)``.

    Returns:
        ``eef_uv`` in target-image pixels, metric camera ``eef_depth``, an
        in-front-and-in-image ``eef_valid`` mask, ``eef_image_hw``, and the
        raw canonical gripper value aligned to the same latent frames.
        Missing slots carry NaN coordinates and depth. Existing but offscreen
        slots retain finite coordinates with ``eef_valid=false``.
    """

    values = np.asarray(condition, dtype=np.float32)
    viewmats = np.asarray(camera_from_canonical, dtype=np.float32)
    Ks = np.asarray(intrinsics, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 4:
        raise ValueError(
            "condition must have shape (F,S,D+1) with XYZ and a mask, "
            f"got {values.shape}"
        )
    if viewmats.ndim != 4 or viewmats.shape[-2:] != (4, 4):
        raise ValueError(f"camera_from_canonical must be (F,V,4,4), got {viewmats.shape}")
    if Ks.shape != viewmats.shape[:2] + (3, 3):
        raise ValueError(
            f"intrinsics must have shape {viewmats.shape[:2] + (3, 3)}, got {Ks.shape}"
        )
    if values.shape[0] != viewmats.shape[0]:
        raise ValueError(
            f"condition/camera frame mismatch: {values.shape[0]} != {viewmats.shape[0]}"
        )
    if not np.isfinite(viewmats).all() or not np.isfinite(Ks).all():
        raise ValueError("camera matrices must be finite")

    F, S = values.shape[:2]
    V = viewmats.shape[1]
    hw = np.asarray(image_hw, dtype=np.int64)
    if hw.shape == (2,):
        hw = np.broadcast_to(hw, (F, V, 2)).copy()
    elif hw.shape == (V, 2):
        hw = np.broadcast_to(hw[None], (F, V, 2)).copy()
    elif hw.shape != (F, V, 2):
        raise ValueError(
            f"image_hw must be (2,), (V,2), or (F,V,2); got {hw.shape}"
        )
    if np.any(hw <= 1):
        raise ValueError(f"image_hw values must exceed one pixel, got {hw}")

    xyz = values[..., :3]
    slot_exists = values[..., -1] >= 0.5
    xyz_finite = np.isfinite(xyz).all(axis=-1)
    points_camera = np.einsum(
        "fvij,fsj->fvsi", viewmats[..., :3, :3], xyz
    ) + viewmats[..., :3, 3][:, :, None, :]
    depth = points_camera[..., 2]
    pixels_h = np.einsum("fvij,fvsj->fvsi", Ks, points_camera)
    denominator = pixels_h[..., 2]
    projectable = (
        slot_exists[:, None, :]
        & xyz_finite[:, None, :]
        & np.isfinite(points_camera).all(axis=-1)
        & np.isfinite(pixels_h).all(axis=-1)
        & (np.abs(denominator) > float(min_positive_depth))
    )

    uv = np.full((F, V, S, 2), np.nan, dtype=np.float32)
    np.divide(
        pixels_h[..., :2],
        denominator[..., None],
        out=uv,
        where=projectable[..., None],
    )
    depth_out = np.where(projectable, depth, np.nan).astype(np.float32)
    height = hw[..., 0, None]
    width = hw[..., 1, None]
    valid = (
        projectable
        & (depth > float(min_positive_depth))
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] <= width - 1)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] <= height - 1)
    )
    # arm_slot_eef_pose is xyz + rot6d + gripper + slot_exists. Keep this
    # fail-closed instead of silently treating another condition schema as
    # gripper-aware projection input.
    if values.shape[-1] < 11:
        raise ValueError(
            "gripper-aware EEF projection requires xyz+rot6d+gripper+mask "
            f"(11 channels), got {values.shape[-1]}"
        )
    gripper = np.where(slot_exists, values[..., 9], np.nan).astype(np.float32)
    return {
        "eef_uv": np.ascontiguousarray(uv),
        "eef_depth": np.ascontiguousarray(depth_out),
        "eef_valid": np.ascontiguousarray(valid),
        "eef_image_hw": np.ascontiguousarray(hw),
        "eef_gripper": np.ascontiguousarray(gripper),
    }
