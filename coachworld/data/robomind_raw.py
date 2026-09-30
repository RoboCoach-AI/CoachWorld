"""RoboMind raw HDF5 helpers used by audits and visual reports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


@dataclass(frozen=True)
class RoboMindEmbodimentSpec:
    name: str
    cameras: tuple[str, ...]
    bgr_to_rgb: bool
    family: str


ROBOMIND_EMBODIMENTS: dict[str, RoboMindEmbodimentSpec] = {
    "h5_franka_3rgb": RoboMindEmbodimentSpec(
        name="h5_franka_3rgb",
        cameras=("camera_top", "camera_left", "camera_right"),
        bgr_to_rgb=True,
        family="franka_single_arm",
    ),
    "h5_agilex_3rgb": RoboMindEmbodimentSpec(
        name="h5_agilex_3rgb",
        cameras=("camera_front", "camera_left_wrist", "camera_right_wrist"),
        bgr_to_rgb=False,
        family="agilex_dual_arm",
    ),
}


def infer_embodiment(path: Path, allowed: set[str] | None = None) -> str | None:
    allowed = allowed or set(ROBOMIND_EMBODIMENTS)
    for part in path.parts:
        if part in allowed:
            return part
    return None


def infer_task(path: Path, embodiment: str) -> str:
    parts = list(path.parts)
    try:
        idx = parts.index(embodiment)
    except ValueError:
        return path.parent.parent.name
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return path.parent.parent.name


def discover_trajectory_files(
    root: Path,
    embodiments: set[str],
    *,
    limit_per_embodiment: int,
) -> dict[str, list[Path]]:
    found = {emb: [] for emb in sorted(embodiments)}
    for dirpath, _, filenames in os.walk(root):
        if "trajectory.hdf5" not in filenames:
            continue
        path = Path(dirpath) / "trajectory.hdf5"
        emb = infer_embodiment(path, embodiments)
        if emb is None or emb not in found:
            continue
        if len(found[emb]) < limit_per_embodiment:
            found[emb].append(path)
        if all(len(paths) >= limit_per_embodiment for paths in found.values()):
            break
    return found


def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_scalar(value.item())
        if value.size == 1:
            return decode_scalar(value.reshape(-1)[0])
    return str(value)


def read_language(f: h5py.File, fallback: str) -> str:
    if "language_raw" not in f:
        return fallback
    try:
        return decode_scalar(f["language_raw"][0]).strip() or fallback
    except Exception:
        return fallback


def dataset_info(ds: h5py.Dataset) -> dict[str, Any]:
    return {"shape": [int(x) for x in ds.shape], "dtype": str(ds.dtype)}


def group_dataset_info(f: h5py.File, group_name: str) -> dict[str, dict[str, Any]]:
    group = f.get(group_name)
    if group is None:
        return {}
    return {key: dataset_info(group[key]) for key in sorted(group.keys())}


def entry_to_uint8(entry: Any) -> np.ndarray:
    if isinstance(entry, bytes):
        return np.frombuffer(entry, dtype=np.uint8)
    if isinstance(entry, np.void):
        return np.frombuffer(bytes(entry), dtype=np.uint8)
    arr = np.asarray(entry)
    if arr.dtype == np.uint8:
        return arr.reshape(-1)
    if arr.dtype.kind in {"S", "V"}:
        return np.frombuffer(arr.tobytes(), dtype=np.uint8)
    return arr.astype(np.uint8).reshape(-1)


def raw_rgb_shape_from_size(size: int) -> tuple[int, int] | None:
    for h, w in (
        (480, 640),
        (640, 480),
        (720, 1280),
        (1280, 720),
        (1080, 1920),
        (1920, 1080),
        (240, 320),
        (320, 240),
    ):
        if int(size) == h * w * 3:
            return h, w
    return None


def decode_rgb_frame(entry: Any, *, bgr_to_rgb: bool) -> np.ndarray:
    buf = entry_to_uint8(entry)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        shape = raw_rgb_shape_from_size(int(buf.size))
        if shape is None:
            raise ValueError(f"failed to decode image buffer of size {buf.size}")
        h, w = shape
        img = buf.reshape(h, w, 3)
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_rgb_frames(
    f: h5py.File,
    spec: RoboMindEmbodimentSpec,
    camera: str,
    indices: np.ndarray,
    *,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    group_name = "observations/rgb_images"
    if group_name not in f:
        raise KeyError(f"missing {group_name}")
    group = f[group_name]
    if camera not in group:
        raise KeyError(f"missing camera {camera}; available={sorted(group.keys())}")
    ds = group[camera]
    frames = []
    for i in indices:
        img = decode_rgb_frame(ds[int(i)], bgr_to_rgb=spec.bgr_to_rgb)
        if target_size is not None:
            h, w = target_size
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
        frames.append(img)
    return np.stack(frames, axis=0)


def numeric_dataset(f: h5py.File, key: str) -> np.ndarray | None:
    if key not in f:
        return None
    arr = np.asarray(f[key][:], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return None
    return arr
