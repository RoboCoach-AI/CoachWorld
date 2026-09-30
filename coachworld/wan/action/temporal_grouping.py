"""Wan causal VAE action grouping.

Wan2.2 TI2V has a causal temporal VAE: latent 0 anchors RGB frame 0, while
latent i>0 summarizes raw frames [(i-1)*stride+1, i*stride].  Action modulation
must use exactly the same grouping.  This module is the only place where that
raw-frame to latent-frame mapping is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WanActionGroups:
    chunks: torch.Tensor
    """Flattened chunks shaped (B, T_latent, stride * D_flat)."""

    frame_indices: torch.Tensor
    """Raw action indices per latent chunk, shaped (T_latent, stride)."""


def wan_causal_group_indices(
    num_latent_frames: int,
    *,
    vae_temporal_stride: int = 4,
    device: torch.device | None = None,
) -> torch.Tensor:
    stride = int(vae_temporal_stride)
    t_latent = int(num_latent_frames)
    if stride <= 0:
        raise ValueError(f"vae_temporal_stride must be positive, got {stride}")
    if t_latent <= 0:
        raise ValueError(f"num_latent_frames must be positive, got {t_latent}")
    rows = []
    for i in range(t_latent):
        if i == 0:
            row = torch.zeros(stride, device=device, dtype=torch.long)
        else:
            start = (i - 1) * stride + 1
            row = torch.arange(start, start + stride, device=device, dtype=torch.long)
        rows.append(row)
    return torch.stack(rows, dim=0)


def group_wan_causal_actions(
    action_seq: torch.Tensor,
    *,
    num_latent_frames: int,
    vae_temporal_stride: int = 4,
) -> WanActionGroups:
    """Group raw-frame actions into Wan latent-frame chunks.

    Args:
        action_seq: Raw/video-frame action tensor shaped (B, T_raw, D_flat).
            The caller is responsible for flattening arm slots and applying slot
            masks before calling this function.
        num_latent_frames: Number of Wan latent frames in the DiT window.
        vae_temporal_stride: Wan temporal compression stride, normally 4.

    Returns:
        WanActionGroups with chunks shaped (B, T_latent, stride * D_flat).
    """
    if action_seq.dim() != 3:
        raise ValueError(
            "group_wan_causal_actions expects flattened action_seq "
            f"(B,T_raw,D), got {tuple(action_seq.shape)}"
        )
    if action_seq.shape[1] <= 0:
        raise ValueError("action_seq must contain at least one raw-frame action")
    frame_indices = wan_causal_group_indices(
        num_latent_frames,
        vae_temporal_stride=vae_temporal_stride,
        device=action_seq.device,
    )
    needed = int(frame_indices.max().item()) + 1
    if action_seq.shape[1] < needed:
        raise ValueError(
            f"raw action length {action_seq.shape[1]} is too short for "
            f"{num_latent_frames} Wan latent frames: need at least {needed}"
        )
    selected = action_seq.index_select(1, frame_indices.flatten())
    B, _, D = selected.shape
    chunks = selected.reshape(B, int(num_latent_frames), int(vae_temporal_stride), D)
    return WanActionGroups(
        chunks=chunks.reshape(B, int(num_latent_frames), int(vae_temporal_stride) * D),
        frame_indices=frame_indices,
    )
