"""Build fail-closed CoachWorld inputs from canonical state and camera geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from coachworld.data.eef_projection import project_arm_slot_condition
from coachworld.world_model.base import WorldModelInput


def camera_conditioning_required(model_config: Any) -> bool:
    """Return whether the configured model consumes calibrated camera geometry."""

    return bool(
        getattr(model_config, "prope_enabled", False)
        or getattr(model_config, "eef_projection_kv_enabled", False)
        or getattr(model_config, "eef_spatial_conditioning_enabled", False)
    )


def eef_projection_required(model_config: Any) -> bool:
    """Return whether projected EEF tensors are part of the checkpoint contract."""

    return bool(
        getattr(model_config, "eef_projection_kv_enabled", False)
        or getattr(model_config, "eef_spatial_conditioning_enabled", False)
    )


def select_runtime_condition_at_latent_frames(
    raw_condition: np.ndarray,
    latent_ids: list[int],
    *,
    vae_temporal_stride: int,
    generated_by_frame: dict[int, np.ndarray] | None = None,
    planned_by_frame: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    """Select raw metric states with planned/generated runtime precedence.

    Future anchors must use the same planned trajectory as the normalized
    action window. Already committed anchors use generated state. Recorded GT
    is only the final fallback.
    """

    values = np.asarray(raw_condition, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError(f"raw_condition must be non-empty (T,S,D), got {values.shape}")
    stride = int(vae_temporal_stride)
    if stride <= 0:
        raise ValueError(f"vae_temporal_stride must be positive, got {stride}")
    generated = generated_by_frame or {}
    planned = planned_by_frame or {}
    rows = []
    for latent_id in latent_ids:
        frame = max(0, int(latent_id) * stride)
        if frame in planned:
            row = planned[frame]
        elif frame in generated:
            row = generated[frame]
        else:
            row = values[min(frame, values.shape[0] - 1)]
        row_array = np.asarray(row, dtype=np.float32)
        if row_array.shape != values.shape[1:]:
            raise ValueError(
                f"runtime condition frame {frame} has shape {row_array.shape}, "
                f"expected {values.shape[1:]}"
            )
        rows.append(row_array)
    return np.ascontiguousarray(np.stack(rows, axis=0))


def build_world_model_input(
    *,
    model_config: Any,
    current_obs: torch.Tensor,
    action_sequence: np.ndarray,
    instruction: str = "",
    domain_id: int | None = None,
    embodiment_id: int | None = None,
    camera_setup_id: int | None = None,
    viewmats: np.ndarray | torch.Tensor | None = None,
    Ks: np.ndarray | torch.Tensor | None = None,
    raw_condition_at_latents: np.ndarray | None = None,
    image_hw: tuple[int, int] | list[int] | np.ndarray | None = None,
    history_obs: list[torch.Tensor] | None = None,
    history_actions: list[np.ndarray] | None = None,
) -> WorldModelInput:
    """Assemble the exact runtime condition consumed by a CoachWorld checkpoint.

    ``action_sequence`` is the normalized model condition. EEF projection must
    use the corresponding raw canonical state because metric XYZ is destroyed
    by per-root normalization.
    """

    needs_camera = camera_conditioning_required(model_config)
    if needs_camera and (viewmats is None or Ks is None):
        raise ValueError(
            "camera-aware checkpoint requires calibrated viewmats and Ks"
        )

    viewmats_np = _as_numpy(viewmats, "viewmats") if viewmats is not None else None
    Ks_np = _as_numpy(Ks, "Ks") if Ks is not None else None
    viewmats_tensor = _as_float_tensor(viewmats) if viewmats is not None else None
    Ks_tensor = _as_float_tensor(Ks) if Ks is not None else None

    projected_tensors: dict[str, torch.Tensor] = {}
    if eef_projection_required(model_config):
        if raw_condition_at_latents is None:
            raise ValueError(
                "EEF-conditioned checkpoint requires raw canonical state at "
                "the same latent anchors as K/T"
            )
        if image_hw is None:
            raise ValueError("EEF-conditioned checkpoint requires target image_hw")
        projected = project_arm_slot_condition(
            np.asarray(raw_condition_at_latents, dtype=np.float32),
            viewmats_np,
            Ks_np,
            image_hw=image_hw,
        )
        projected_tensors = {
            "eef_uv": torch.from_numpy(projected["eef_uv"]).float(),
            "eef_depth": torch.from_numpy(projected["eef_depth"]).float(),
            "eef_valid": torch.from_numpy(projected["eef_valid"]).bool(),
            "eef_image_hw": torch.from_numpy(projected["eef_image_hw"]).long(),
            "eef_gripper": torch.from_numpy(projected["eef_gripper"]).float(),
        }

    return WorldModelInput(
        current_obs=current_obs,
        action_sequence=np.ascontiguousarray(action_sequence),
        history_obs=list(history_obs or []),
        history_actions=list(history_actions or []),
        instruction=str(instruction),
        domain_id=domain_id,
        embodiment_id=embodiment_id,
        camera_setup_id=camera_setup_id,
        viewmats=viewmats_tensor,
        Ks=Ks_tensor,
        **projected_tensors,
    )


def _as_numpy(value: np.ndarray | torch.Tensor, label: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(array)


def _as_float_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).float()
