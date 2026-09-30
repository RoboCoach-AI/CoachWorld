"""Robot action adapter for WAN 2.2 DiT action conditioning.

Provides:
  1. ``make_robot_action_config`` — minimal config dict fed to WanModelAction.
  2. ``prepare_action_tensor`` — temporal alignment from raw action sequences
     to the DiT's per-frame token count.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# WAN 2.2 TI2V-5B fixed constants
VAE_TIME_COMPRESSION = 4  # 4 video frames → 1 latent frame
DIT_HIDDEN_DIM = 3072     # WanModelAction dim for the 5B variant


def make_robot_action_config(
    action_dim: int = 10,
    mid_dims: tuple[int, int] = (128, 256),
    latent_C: int = 48,
    latent_T: int = 11,
    latent_H: int = 12,
    latent_W: int = 20,
    latent_height_per_view: Optional[int] = None,
    vae_spatial_stride: int = 16,
    vae_temporal_stride: int = 4,
    action_schema: str = "fixed",
    max_arm_slots: int = 2,
    action_max_time_steps: int = 512,
    action_kv_enabled: bool = True,
    action_v_init_scale: float = 0.1,
    action_num_domains: int = 1,
    action_domain_prompt_tokens: int = 0,
    domain_aware_action_projection_enabled: bool = False,
    domain_aware_group_projection_enabled: bool = False,
    num_cameras: int = 1,
    multi_view_position_mode: str = "global",
    camera_id_embedding_enabled: bool = False,
    eef_projection_kv_enabled: bool = False,
) -> dict:
    """Minimal config dict for action conditioning.

    Args:
        action_dim: Per-arm condition dimensionality. The canonical CoachWorld
            contract is 10D EEF pose: xyz, rot6d, gripper.
        mid_dims: ActionEncoder MLP hidden widths.
        latent_C/T/H/W: Video-latent shape used by callers that need latent
            geometry for action-frame alignment or diagnostics.
        latent_height_per_view: Per-camera latent height before height stacking.
            If omitted, derived from latent_H // num_cameras.
        vae_spatial_stride: RGB-pixel stride per Wan latent pixel.
        vae_temporal_stride: Ratio between video-frame-rate action tokens
            and latent frames (4 for Wan2.2 VAE).
        action_max_time_steps: Maximum raw-frame condition length accepted by
            the action K/V encoder.
        num_cameras: Number of camera views stacked along latent height.
        multi_view_position_mode: RoPE layout for stacked views:
            ``global`` keeps historical global height positions;
            ``view_local`` resets spatial height positions inside each view.
        camera_id_embedding_enabled: Add a learned camera/view embedding to
            every visual token before the DiT blocks.
        eef_projection_kv_enabled: Add image-space EEF projection tokens to
            the same dedicated action K/V cross-attention path.
    """
    return {
        "action_dim": action_dim,
        "mid_dims": list(mid_dims),
        "latent_C": latent_C,
        "latent_T": latent_T,
        "latent_H": latent_H,
        "latent_W": latent_W,
        "latent_height_per_view": (
            int(latent_height_per_view)
            if latent_height_per_view is not None
            else max(1, int(latent_H) // max(1, int(num_cameras)))
        ),
        "latent_width": int(latent_W),
        "vae_spatial_stride": int(vae_spatial_stride),
        "vae_temporal_stride": vae_temporal_stride,
        "action_schema": action_schema,
        "max_arm_slots": max_arm_slots,
        "action_max_time_steps": int(action_max_time_steps),
        "action_kv_enabled": action_kv_enabled,
        "action_v_init_scale": float(action_v_init_scale),
        "action_num_domains": int(action_num_domains),
        "action_domain_prompt_tokens": int(action_domain_prompt_tokens),
        "domain_aware_action_projection_enabled": bool(domain_aware_action_projection_enabled),
        "domain_aware_group_projection_enabled": bool(domain_aware_group_projection_enabled),
        "num_cameras": num_cameras,
        "multi_view_position_mode": multi_view_position_mode,
        "camera_id_embedding_enabled": camera_id_embedding_enabled,
        "eef_projection_kv_enabled": bool(eef_projection_kv_enabled),
    }


def prepare_action_tensor(
    action_sequence: np.ndarray,
    num_video_frames: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Convert a robot action sequence to the per-video-frame tensor the DiT expects.

    WanModelAction.action_encoder consumes either fixed actions shaped
    ``(B, T_act, action_dim)`` or arm-slot conditions shaped
    ``(B, T_act, slots, action_dim + 1)``. For arm slots the final channel is
    an existence mask; values are the canonical 10D EEF pose condition. The DiT
    receives one condition token per raw video frame before VAE temporal
    compression, so ``T_act == num_video_frames``.

    Args:
        action_sequence: ``(T_actions, ...)`` normalized actions.
        num_video_frames: Number of raw (pre-compression) video frames the DiT
            input window represents — e.g. 17 for 5 latent frames.
        device: Target device.
        dtype: Target dtype.

    Returns:
        ``(1, num_video_frames, ...)``, or None if no actions.
    """
    if action_sequence is None or len(action_sequence) == 0:
        return None

    action_sequence = np.asarray(action_sequence)
    if action_sequence.ndim < 2:
        raise ValueError(
            f"action_sequence must have at least 2 dims (T,...), got {action_sequence.shape}"
        )
    T_actions = int(action_sequence.shape[0])

    if T_actions < num_video_frames:
        pad = np.repeat(action_sequence[-1:], num_video_frames - T_actions, axis=0)
        action_sequence = np.concatenate([action_sequence, pad], axis=0)
    elif T_actions > num_video_frames:
        # Actions are already at video-frame rate. Keep temporal alignment with
        # latent frames by taking the contiguous prefix for this window; do not
        # linspace-resample, which shifts latent-frame actions off 0/4/8/...
        # frame boundaries.
        action_sequence = action_sequence[:num_video_frames]

    tensor = torch.from_numpy(action_sequence).to(device=device, dtype=dtype)
    return tensor.unsqueeze(0)  # (1, T, action_dim)
