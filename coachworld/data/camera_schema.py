"""Camera schema constants shared by data builders and evaluators."""

from __future__ import annotations


DROID_VIDEO_KEYS = [
    "observation.images.exterior_1_left",
    "observation.images.exterior_2_left",
    "observation.images.wrist_left",
]


def parse_camera_keys(spec: str, *, label: str = "camera keys") -> list[str]:
    keys = [item.strip() for item in str(spec).split(",") if item.strip()]
    if not keys:
        raise ValueError(f"{label} must contain at least one key")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicates: {keys}")
    return keys


def droid_view_name(video_key: str) -> str:
    prefix = "observation.images."
    if not str(video_key).startswith(prefix):
        raise ValueError(f"unexpected DROID video key: {video_key}")
    return str(video_key)[len(prefix) :]


def view_names_from_keys(video_keys: list[str]) -> list[str]:
    return [droid_view_name(key) for key in video_keys]
