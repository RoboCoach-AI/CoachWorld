"""Strict helpers for LeRobot-format DROID raw data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from coachworld.data.camera_schema import DROID_VIDEO_KEYS, droid_view_name


def load_lerobot_info(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / "meta" / "info.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_path(root: str | Path, info: dict[str, Any], episode_id: int) -> Path:
    root = Path(root)
    chunk = int(episode_id) // int(info["chunks_size"])
    rel = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=int(episode_id),
    )
    return root / rel


def resolve_video_path(
    root: str | Path,
    info: dict[str, Any],
    episode_id: int,
    video_key: str,
) -> Path:
    root = Path(root)
    chunk = int(episode_id) // int(info["chunks_size"])
    rel = info["video_path"].format(
        episode_chunk=chunk,
        episode_index=int(episode_id),
        video_key=str(video_key),
    )
    return root / rel


def raw_video_path(root: str | Path, episode_id: int, video_key: str) -> Path:
    path = (
        Path(root)
        / "videos"
        / f"chunk-{int(episode_id) // 1000:03d}"
        / str(video_key)
        / f"episode_{int(episode_id):06d}.mp4"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_episode_dataframe(root: str | Path, info: dict[str, Any], episode_id: int) -> pd.DataFrame:
    path = resolve_data_path(root, info, episode_id)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"empty parquet: {path}")
    episodes = sorted({int(x) for x in df["episode_index"].tolist()})
    if episodes != [int(episode_id)]:
        raise ValueError(f"{path}: episode_index mismatch {episodes} != [{episode_id}]")
    return df


def episode_files_complete(
    root: str | Path,
    info: dict[str, Any],
    episode_id: int,
    video_keys: list[str] | None = None,
) -> bool:
    keys = video_keys if video_keys is not None else DROID_VIDEO_KEYS
    if not resolve_data_path(root, info, episode_id).exists():
        return False
    return all(resolve_video_path(root, info, episode_id, key).exists() for key in keys)


def episode_video_frame_count(
    root: str | Path,
    info: dict[str, Any],
    episode_id: int,
    rgb_skip: int,
) -> int:
    df = pd.read_parquet(resolve_data_path(root, info, episode_id), columns=["frame_index"])
    return int(len(df.iloc[:: int(rgb_skip)]))


def build_droid_views(
    root: str | Path,
    info: dict[str, Any],
    episode_id: int,
    video_keys: list[str],
) -> list[dict[str, Any]]:
    root = Path(root)
    views = []
    for view_id, key in enumerate(video_keys):
        path = resolve_video_path(root, info, episode_id, key)
        if not path.exists():
            raise FileNotFoundError(path)
        name = droid_view_name(key)
        views.append(
            {
                "view_id": int(view_id),
                "name": name,
                "source_key": key,
                "source_path": path.resolve().relative_to(root.resolve()).as_posix(),
                "intrinsics": None,
                "extrinsics_source": f"camera_extrinsics.{name}",
            }
        )
    return views
