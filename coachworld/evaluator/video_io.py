"""Video I/O helpers for CoachWorld evaluation artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


class VideoEncodingError(RuntimeError):
    """Raised when the environment cannot encode the requested eval video."""


def _validate_rgb_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB frames with shape (T,H,W,3), got {frames.shape}")
    if frames.shape[0] <= 0:
        raise ValueError("expected at least one frame")
    if frames.shape[1] % 2 or frames.shape[2] % 2:
        raise ValueError(
            f"H.264 yuv420p requires even height/width, got H={frames.shape[1]} W={frames.shape[2]}"
        )
    return frames


def _save_h264_mp4_with_imageio_ffmpeg(frames: np.ndarray, path: Path, fps: float) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = int(frames.shape[1]), int(frames.shape[2])
    fps_arg = f"{float(fps):.6g}"
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        fps_arg,
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    stdout, stderr = proc.communicate(frames.tobytes(order="C"))
    if proc.returncode != 0:
        raise VideoEncodingError(
            "imageio-ffmpeg/libx264 failed:\n"
            + stderr.decode("utf-8", errors="replace")
            + stdout.decode("utf-8", errors="replace")
        )


def save_h264_mp4(frames: np.ndarray, path: Path, fps: float) -> Path:
    """Save RGB frames as browser-compatible H.264/yuv420p MP4.

    This intentionally does not fall back to OpenCV ``mp4v`` because that codec
    produced MP4 files that did not play in Chrome/VSCode WebView on our cluster.
    The project eval environment must provide ``imageio-ffmpeg``.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _validate_rgb_frames(frames)
    _save_h264_mp4_with_imageio_ffmpeg(frames, path, fps)
    return path
