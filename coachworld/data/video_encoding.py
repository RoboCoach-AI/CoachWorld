"""Strict RGB video decoding and WAN VAE encoding helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

VALID_RESIZE_MODES = {"direct_resize", "letterbox"}


def compute_image_transform(
    raw_hw: tuple[int, int],
    target_hw: tuple[int, int],
    *,
    resize_mode: str = "direct_resize",
) -> dict[str, object]:
    """Return the image-space transform used before VAE encoding.

    The returned ``scale_xy`` and ``pad_xy`` are the values needed to map raw
    image-space camera intrinsics into the target image space:

    ``fx' = fx * scale_x``, ``cx' = cx * scale_x + pad_x``.
    """

    raw_h, raw_w = int(raw_hw[0]), int(raw_hw[1])
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if raw_h <= 0 or raw_w <= 0:
        raise ValueError(f"invalid raw_hw={raw_hw!r}")
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"invalid target_hw={target_hw!r}")
    if resize_mode not in VALID_RESIZE_MODES:
        raise ValueError(f"invalid resize_mode={resize_mode!r}; expected {sorted(VALID_RESIZE_MODES)}")

    if resize_mode == "direct_resize":
        return {
            "raw_image_hw": [raw_h, raw_w],
            "target_image_hw": [target_h, target_w],
            "resize_mode": "direct_resize",
            "resized_hw": [target_h, target_w],
            "scale_xy": [target_w / raw_w, target_h / raw_h],
            "pad_xy": [0.0, 0.0],
            "pad_ltrb": [0, 0, 0, 0],
        }

    scale = min(target_w / raw_w, target_h / raw_h)
    resized_w = max(1, int(round(raw_w * scale)))
    resized_h = max(1, int(round(raw_h * scale)))
    if resized_w > target_w:
        resized_w = target_w
    if resized_h > target_h:
        resized_h = target_h
    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left
    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    return {
        "raw_image_hw": [raw_h, raw_w],
        "target_image_hw": [target_h, target_w],
        "resize_mode": "letterbox",
        "resized_hw": [resized_h, resized_w],
        "scale_xy": [resized_w / raw_w, resized_h / raw_h],
        "pad_xy": [float(pad_left), float(pad_top)],
        "pad_ltrb": [int(pad_left), int(pad_top), int(pad_right), int(pad_bottom)],
    }


def resize_video_tensor(
    x: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    resize_mode: str = "direct_resize",
    pad_value: float = 0.0,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected BCHW video tensor, got {tuple(x.shape)}")
    transform = compute_image_transform(
        (int(x.shape[-2]), int(x.shape[-1])),
        target_hw,
        resize_mode=resize_mode,
    )
    resized_h, resized_w = [int(v) for v in transform["resized_hw"]]
    x = torch.nn.functional.interpolate(
        x,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    if resize_mode == "letterbox":
        pad_left, pad_top, pad_right, pad_bottom = [int(v) for v in transform["pad_ltrb"]]
        x = torch.nn.functional.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=float(pad_value),
        )
    return x


def decode_video_frame_count_strict(video_path: str | Path, rgb_skip: int = 1) -> int:
    import av

    video_path = Path(video_path)
    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"failed to open video: {video_path}: {exc}") from exc

    count = 0
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, _frame in enumerate(container.decode(stream)):
            if i % int(rgb_skip) == 0:
                count += 1
    except Exception as exc:
        raise RuntimeError(f"failed to decode video: {video_path}: {exc}") from exc
    finally:
        container.close()
    if count <= 0:
        raise RuntimeError(f"decoded zero frames: {video_path}")
    return int(count)


def load_video_frames_strict(video_path: str | Path, rgb_skip: int = 1) -> np.ndarray:
    import av

    video_path = Path(video_path)
    try:
        container = av.open(str(video_path))
    except Exception as exc:
        raise RuntimeError(f"failed to open video: {video_path}: {exc}") from exc

    frames = []
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if i % int(rgb_skip) == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as exc:
        raise RuntimeError(f"failed to decode video: {video_path}: {exc}") from exc
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"decoded zero frames: {video_path}")
    return np.stack(frames)


def read_selected_video_frames(
    path: str | Path,
    wanted_indices: list[int],
    target_hw: tuple[int, int],
) -> dict[int, np.ndarray]:
    wanted = set(int(x) for x in wanted_indices)
    if not wanted:
        raise ValueError("wanted_indices is empty")
    target_h, target_w = target_hw
    frame_bytes = int(target_h) * int(target_w) * 3
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={int(target_w)}:{int(target_h)}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    out: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if not buf:
            break
        if len(buf) != frame_bytes:
            raise RuntimeError(f"partial frame while decoding {path}: got {len(buf)} bytes")
        if idx in wanted:
            out[idx] = (
                np.frombuffer(buf, dtype=np.uint8)
                .reshape(int(target_h), int(target_w), 3)
                .copy()
            )
        idx += 1
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}:\n{stderr}")
    missing = sorted(wanted - set(out))
    if missing:
        raise IndexError(f"video missing requested frames for {path}: first missing {missing[:10]}")
    return out


@torch.no_grad()
def encode_trajectory(
    vae,
    video_frames: np.ndarray,
    device: torch.device | str,
    target_size: tuple[int, int] | None = None,
    resize_mode: str = "direct_resize",
) -> torch.Tensor:
    if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
        raise ValueError(f"expected RGB frames with shape (T,H,W,3), got {video_frames.shape}")
    x = torch.from_numpy(np.asarray(video_frames, dtype=np.uint8)).permute(0, 3, 1, 2).float()
    if target_size is not None:
        x = resize_video_tensor(
            x,
            (int(target_size[0]), int(target_size[1])),
            resize_mode=resize_mode,
            pad_value=0.0,
        )
    x = x / 127.5 - 1.0
    x = x.permute(1, 0, 2, 3).to(device=device, dtype=torch.bfloat16)
    latents = vae.encode([x])
    return latents[0].half().cpu()
