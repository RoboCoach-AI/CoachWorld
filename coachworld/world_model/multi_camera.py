"""Multi-camera latent encoding utilities.

Following Ctrl-World convention (evow/rollout/world_model_wrapper.py):
  - 3 camera views stacked in the **height** dimension of the latent
  - Latent shape: (C, T, H_total, W) where H_total = num_cameras * H_per_view
  - For Ctrl-World defaults: (4, T, 72, 40) where 72 = 3 * 24

This module handles:
  - Stacking multiple camera views into a single latent tensor
  - Splitting a stacked latent back into per-camera views
  - Resizing per-view latents to canonical sizes
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


# Ctrl-World defaults
DEFAULT_NUM_CAMERAS = 3
DEFAULT_PER_VIEW_H = 24
DEFAULT_PER_VIEW_W = 40


def stack_camera_latents(
    camera_latents: list[torch.Tensor],
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> torch.Tensor:
    """Stack per-camera latents by concatenating along the height dimension.

    Args:
        camera_latents: List of N tensors, each (C, T, h_i, w_i).
            Different cameras may have different spatial sizes.
        target_h: Target height per view (resize if needed). None = no resize.
        target_w: Target width per view (resize if needed). None = no resize.

    Returns:
        Stacked latent: (C, T, N*target_h, target_w).
    """
    resized = []
    for lat in camera_latents:
        if target_h is not None and target_w is not None:
            C, T, h, w = lat.shape
            if h != target_h or w != target_w:
                # Resize spatial dims: (C, T, h, w) -> (C*T, 1, h, w) -> interpolate -> reshape
                lat_flat = lat.reshape(C * T, 1, h, w).float()
                lat_flat = F.interpolate(lat_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
                lat = lat_flat.reshape(C, T, target_h, target_w).to(lat.dtype)
        resized.append(lat)
    return torch.cat(resized, dim=2)  # concat along height


def split_camera_latents(
    stacked: torch.Tensor,
    num_cameras: int = DEFAULT_NUM_CAMERAS,
) -> list[torch.Tensor]:
    """Split a height-stacked latent back into per-camera views.

    Args:
        stacked: (C, T, H_total, W) where H_total = num_cameras * h_per_view.
        num_cameras: Number of camera views.

    Returns:
        List of num_cameras tensors, each (C, T, h_per_view, W).
    """
    C, T, H_total, W = stacked.shape
    assert H_total % num_cameras == 0, (
        f"Height {H_total} not divisible by {num_cameras} cameras"
    )
    h_per_view = H_total // num_cameras
    return [stacked[:, :, i * h_per_view:(i + 1) * h_per_view, :] for i in range(num_cameras)]


def stack_camera_frames(
    camera_frames: list[np.ndarray],
    layout: str = "vertical",
) -> np.ndarray:
    """Stack RGB frames from multiple cameras for visualization.

    Args:
        camera_frames: List of N arrays, each (T, H, W, 3) or (H, W, 3).
        layout: "vertical" (height stack) or "horizontal" (width stack).

    Returns:
        Stacked array with same ndim as input.
    """
    axis = -3 if layout == "vertical" else -2  # H or W axis
    return np.concatenate(camera_frames, axis=axis)
