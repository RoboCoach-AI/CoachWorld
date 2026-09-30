"""Shared inference helpers for CoachWorld rollout and audit scripts."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

from coachworld.evaluator.video_io import save_h264_mp4
from coachworld.world_model.condition_config import (
    CONDITION_CONFIG_FIELDS as _CONDITION_CONFIG_FIELDS,
    validate_condition_config_match as _validate_condition_config_match,
)
from coachworld.world_model.wan_world_model import WanWorldModel

logger = logging.getLogger(__name__)

VAE_TEMPORAL_STRIDE = 4
# Historical import location retained without duplicating the contract.
CONDITION_CONFIG_FIELDS = _CONDITION_CONFIG_FIELDS


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint_dir(path: Path) -> Path:
    if (path / "dit_model.safetensors").exists() or (path / "dit_model.pt").exists():
        return path
    if path.exists():
        candidates = sorted(
            [
                p
                for p in path.glob("checkpoint-*")
                if (p / "dit_model.safetensors").exists() or (p / "dit_model.pt").exists()
            ],
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        if candidates:
            return candidates[-1]
    return path


def load_dit_weights(wm: WanWorldModel, checkpoint: Path) -> None:
    checkpoint = resolve_checkpoint_dir(checkpoint)
    st_path = checkpoint / "dit_model.safetensors"
    pt_path = checkpoint / "dit_model.pt"
    if st_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(st_path))
    elif pt_path.exists():
        state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
    else:
        existing = []
        if checkpoint.exists():
            existing = [str(p.relative_to(checkpoint)) for p in checkpoint.rglob("*") if p.is_file()][:40]
        raise FileNotFoundError(
            f"No dit_model.safetensors or dit_model.pt found in {checkpoint}. "
            f"First files under checkpoint: {existing}"
        )

    _ = wm.dit
    missing, unexpected = wm.dit.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded DiT checkpoint: %s (missing=%d unexpected=%d)",
        checkpoint,
        len(missing),
        len(unexpected),
    )
    allowed_missing = []
    if bool(getattr(wm.config, "prope_enabled", False)):
        allowed_missing.extend([k for k in missing if "prope_o" in k])
    illegal_missing = [k for k in missing if k not in set(allowed_missing)]
    illegal_unexpected = list(unexpected)
    if illegal_missing or illegal_unexpected:
        raise RuntimeError(
            "Checkpoint architecture mismatch; refusing to run inference with "
            f"missing={illegal_missing[:10]} unexpected={illegal_unexpected[:10]}"
        )


def validate_condition_config_match(cfg) -> None:
    _validate_condition_config_match(cfg.world_model, cfg.wm_training)


def build_world_model(cfg, *, device: str, num_steps: int | None = None) -> WanWorldModel:
    validate_condition_config_match(cfg)
    wc = copy.deepcopy(cfg.world_model)
    if num_steps is not None:
        wc.num_inference_steps = int(num_steps)
    return WanWorldModel(
        wc,
        device=device,
        num_cameras=int(getattr(cfg.wm_training, "num_cameras", 3)),
    )


def save_video(frames: np.ndarray, path: Path, fps: float) -> Path:
    return save_h264_mp4(frames, path, fps)


@torch.no_grad()
def decode_latents_in_chunks(
    world_model: WanWorldModel,
    latents: torch.Tensor,
    max_latents_per_chunk: int,
) -> np.ndarray:
    """Decode a latent sequence without dropping temporal boundary frames.

    Wan's VAE maps ``T`` latents to ``1 + 4 * (T - 1)`` RGB frames. Adjacent
    decode chunks therefore overlap by one latent anchor; the duplicate RGB
    anchor from the later chunk is removed after decoding.
    """

    total = int(latents.shape[1])
    chunk_size = int(max_latents_per_chunk)
    if total <= 0:
        raise ValueError("latents must contain at least one temporal item")
    if chunk_size < 2:
        raise ValueError("max_latents_per_chunk must be at least 2")

    chunks: list[np.ndarray] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        frames = world_model.decode(
            latents[:, start:end].to(world_model.device, dtype=world_model.dtype)
        )
        if frames is None:
            raise RuntimeError("WAN VAE decode returned None")
        if chunks:
            frames = frames[1:]
        chunks.append(frames)
        if end == total:
            break
        start = end - 1

    decoded = np.concatenate(chunks, axis=0)
    expected = 1 + VAE_TEMPORAL_STRIDE * (total - 1)
    if len(decoded) != expected:
        raise RuntimeError(
            f"chunked VAE decode produced {len(decoded)} frames for {total} "
            f"latents; expected {expected}"
        )
    return decoded
