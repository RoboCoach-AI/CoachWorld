"""Camera-aligned EEF control rasters for WAN latent conditioning."""

from __future__ import annotations

import torch


EEF_SPATIAL_CHANNELS_PER_SLOT = 3


def rasterize_eef_spatial_condition(
    *,
    uv: torch.Tensor,
    depth: torch.Tensor,
    valid: torch.Tensor,
    image_hw: torch.Tensor,
    gripper: torch.Tensor,
    latent_height_per_view: int,
    latent_width: int,
    sigma_px: float,
    depth_scale: float,
) -> torch.Tensor:
    """Rasterize slot EEF controls to the height-stacked latent layout.

    Args:
        uv: ``(B,F,V,S,2)`` target-image pixel coordinates.
        depth: ``(B,F,V,S)`` metric camera depth.
        valid: ``(B,F,V,S)`` in-image visibility.
        image_hw: ``(B,F,V,2)`` or ``(B,F,V,S,2)`` target image size.
        gripper: ``(B,F,S)`` or ``(B,F,V,S)`` canonical gripper state.

    Returns:
        ``(B,3*S,F,V*H_latent,W_latent)``. For each slot the three channels
        are Gaussian occupancy, normalized camera depth, and normalized
        gripper state. Views follow the same height-stacked order as RGB
        latents and PRoPE.
    """

    if uv.dim() == 4 and uv.shape[-1] == 2:
        uv = uv.unsqueeze(3)
        depth = depth.unsqueeze(3)
        valid = valid.unsqueeze(3)
    if uv.dim() != 5 or uv.shape[-1] != 2:
        raise ValueError(f"uv must be (B,F,V,S,2), got {tuple(uv.shape)}")
    if depth.shape != uv.shape[:-1] or valid.shape != uv.shape[:-1]:
        raise ValueError(
            "depth/valid must match uv leading dimensions; "
            f"uv={tuple(uv.shape)}, depth={tuple(depth.shape)}, valid={tuple(valid.shape)}"
        )

    batch, frames, views, slots, _ = uv.shape
    if image_hw.dim() == 4 and image_hw.shape == (batch, frames, views, 2):
        image_hw = image_hw.unsqueeze(3).expand(batch, frames, views, slots, 2)
    if image_hw.shape != uv.shape:
        raise ValueError(
            f"image_hw must be {(batch, frames, views, 2)} or {tuple(uv.shape)}, "
            f"got {tuple(image_hw.shape)}"
        )
    if gripper.shape == (batch, frames, slots):
        gripper = gripper.unsqueeze(2).expand(batch, frames, views, slots)
    if gripper.shape != uv.shape[:-1]:
        raise ValueError(
            f"gripper must be {(batch, frames, slots)} or {tuple(uv.shape[:-1])}, "
            f"got {tuple(gripper.shape)}"
        )

    h_lat = int(latent_height_per_view)
    w_lat = int(latent_width)
    if h_lat < 1 or w_lat < 1:
        raise ValueError(f"invalid latent size {h_lat}x{w_lat}")
    if float(sigma_px) <= 0.0 or float(depth_scale) <= 0.0:
        raise ValueError(
            f"sigma_px and depth_scale must be positive, got {sigma_px}, {depth_scale}"
        )

    dtype = uv.dtype if uv.is_floating_point() else torch.float32
    device = uv.device
    uv = uv.to(dtype=dtype)
    depth = depth.to(device=device, dtype=dtype)
    image_hw = image_hw.to(device=device, dtype=dtype).clamp_min(2.0)
    gripper = gripper.to(device=device, dtype=dtype)
    present = (
        torch.isfinite(uv).all(dim=-1)
        & torch.isfinite(depth)
        & torch.isfinite(gripper)
        & valid.to(device=device).bool()
    )

    height = image_hw[..., 0]
    width = image_hw[..., 1]
    u = torch.nan_to_num(uv[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
    v = torch.nan_to_num(uv[..., 1], nan=0.0, posinf=0.0, neginf=0.0)
    u_lat = u / (width - 1.0) * max(w_lat - 1, 1)
    v_lat = v / (height - 1.0) * max(h_lat - 1, 1)

    grid_y = torch.arange(h_lat, device=device, dtype=dtype).view(1, 1, 1, 1, h_lat, 1)
    grid_x = torch.arange(w_lat, device=device, dtype=dtype).view(1, 1, 1, 1, 1, w_lat)
    sigma_x = (float(sigma_px) / width * float(w_lat)).clamp_min(0.5)
    sigma_y = (float(sigma_px) / height * float(h_lat)).clamp_min(0.5)
    dist2 = (
        ((grid_x - u_lat[..., None, None]) / sigma_x[..., None, None]).square()
        + ((grid_y - v_lat[..., None, None]) / sigma_y[..., None, None]).square()
    )
    occupancy = torch.exp(-0.5 * dist2) * present[..., None, None].to(dtype=dtype)

    depth_value = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_value = (depth_value / float(depth_scale)).clamp(0.0, 1.0) * 2.0 - 1.0
    grip_value = torch.nan_to_num(gripper, nan=0.0, posinf=0.0, neginf=0.0)
    grip_value = grip_value.clamp(0.0, 1.0) * 2.0 - 1.0
    features = torch.stack(
        [
            occupancy,
            occupancy * depth_value[..., None, None],
            occupancy * grip_value[..., None, None],
        ],
        dim=4,
    )
    # (B,F,V,S,3,H,W) -> (B,S*3,F,V*H,W)
    return (
        features.permute(0, 3, 4, 1, 2, 5, 6)
        .contiguous()
        .reshape(batch, slots * EEF_SPATIAL_CHANNELS_PER_SLOT, frames, views * h_lat, w_lat)
    )
