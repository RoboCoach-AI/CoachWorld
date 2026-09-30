"""Differentiable metric-depth rendering for robot calibration."""

from __future__ import annotations

import nvdiffrast.torch as dr
import torch

from caliball.rendering.nvdiffrast_renderer import NVDiffrastRenderer
from caliball.rendering.nvdiffrast_utils import K_to_projection, transform_pos


def render_mask_metric_depth(
    renderer: NVDiffrastRenderer,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    intrinsic: torch.Tensor,
    camera_from_object: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render an anti-aliased mask and OpenCV camera-frame Z in metres.

    ``NVDiffrastRenderer.render_all`` exposes clip-space ``z / w``.  That value
    is useful for visibility but cannot be compared with an RGB-D observation.
    Here camera-frame vertex Z is rasterized as an attribute, preserving
    gradients with respect to ``camera_from_object``.
    """

    vertices, faces, intrinsic, camera_from_object, device = renderer._prepare_render_inputs(
        vertices,
        faces,
        intrinsic,
        camera_from_object,
    )
    projection = K_to_projection(intrinsic, renderer.H, renderer.W, device=device)
    clip_pose = renderer.opencv2blender @ camera_from_object
    clip_vertices = transform_pos(projection @ clip_pose, vertices, device=device).float()
    camera_vertices = transform_pos(camera_from_object, vertices, device=device)[0, :, :3]

    raster, _ = dr.rasterize(
        renderer.glctx,
        clip_vertices,
        faces,
        resolution=renderer.resolution,
    )
    attributes = torch.cat(
        [
            torch.ones((len(vertices), 1), dtype=torch.float32, device=device),
            camera_vertices[:, 2:3],
        ],
        dim=1,
    ).contiguous()
    interpolated, _ = dr.interpolate(attributes[None], raster, faces)
    mask = dr.antialias(
        interpolated[..., :1].contiguous(),
        raster.contiguous(),
        clip_vertices.contiguous(),
        faces.contiguous(),
    )[0, :, :, 0]
    metric_depth = interpolated[0, :, :, 1]
    hard_mask = raster[0, :, :, 3] > 0

    mask = torch.flip(mask, dims=[0])
    metric_depth = torch.flip(metric_depth, dims=[0])
    hard_mask = torch.flip(hard_mask, dims=[0])
    metric_depth = torch.where(hard_mask, metric_depth, torch.zeros_like(metric_depth))
    return mask, metric_depth


def render_solver_mask_metric_depth(
    solver: torch.nn.Module,
    camera_from_robot: torch.Tensor,
    link_poses: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render all links for a frame batch with nearest-surface metric depth."""

    frame_masks: list[torch.Tensor] = []
    frame_depths: list[torch.Tensor] = []
    for frame_poses in link_poses:
        link_masks: list[torch.Tensor] = []
        link_depths: list[torch.Tensor] = []
        for link_index in range(int(solver.nlinks)):
            camera_from_link = camera_from_robot @ frame_poses[link_index]
            mask, depth = render_mask_metric_depth(
                solver.renderer,
                getattr(solver, f"vertices_{link_index}"),
                getattr(solver, f"faces_{link_index}"),
                intrinsic,
                camera_from_link,
            )
            link_masks.append(mask)
            link_depths.append(depth)
        masks = torch.stack(link_masks, dim=0)
        depths = torch.stack(link_depths, dim=0)
        valid_depths = torch.where(masks > 0.01, depths, torch.full_like(depths, torch.inf))
        nearest_depth = valid_depths.amin(dim=0)
        nearest_depth = torch.where(
            torch.isfinite(nearest_depth),
            nearest_depth,
            torch.zeros_like(nearest_depth),
        )
        frame_masks.append(masks.sum(dim=0).clamp(max=1.0))
        frame_depths.append(nearest_depth)
    return torch.stack(frame_masks, dim=0), torch.stack(frame_depths, dim=0)
