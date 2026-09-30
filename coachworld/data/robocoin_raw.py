"""RoboCOIN raw LeRobot-v2.1 helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class RoboCoinDataset:
    root: Path
    name: str
    info: dict[str, Any]

    @property
    def robot_type(self) -> str:
        return str(self.info.get("robot_type", "unknown"))

    @property
    def fps(self) -> float:
        return float(self.info.get("fps", 0.0))

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes", 0))

    @property
    def video_keys(self) -> list[str]:
        features = self.info.get("features", {})
        return [
            key
            for key, spec in features.items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_dataset_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "meta/info.json").exists()
    )


def load_dataset(path: Path) -> RoboCoinDataset:
    return RoboCoinDataset(
        root=path,
        name=path.name,
        info=read_json(path / "meta/info.json"),
    )


def dataset_inventory(root: Path) -> dict[str, Any]:
    datasets = [load_dataset(path) for path in iter_dataset_dirs(root)]
    robot_counts: dict[str, int] = {}
    fps_counts: dict[str, int] = {}
    video_key_counts: dict[str, int] = {}
    total_episodes = 0
    total_frames = 0
    rows = []
    for ds in datasets:
        robot_counts[ds.robot_type] = robot_counts.get(ds.robot_type, 0) + 1
        fps_key = f"{ds.fps:g}"
        fps_counts[fps_key] = fps_counts.get(fps_key, 0) + 1
        vk_key = ",".join(ds.video_keys)
        video_key_counts[vk_key] = video_key_counts.get(vk_key, 0) + 1
        total_episodes += ds.total_episodes
        total_frames += int(ds.info.get("total_frames", 0))
        rows.append(
            {
                "name": ds.name,
                "robot_type": ds.robot_type,
                "fps": ds.fps,
                "episodes": ds.total_episodes,
                "frames": int(ds.info.get("total_frames", 0)),
                "video_keys": ds.video_keys,
            }
        )
    return {
        "root": str(root),
        "datasets": len(datasets),
        "episodes": total_episodes,
        "frames": total_frames,
        "robot_counts": robot_counts,
        "fps_counts": fps_counts,
        "video_key_counts": video_key_counts,
        "rows": rows,
    }


def episode_rows(dataset: RoboCoinDataset) -> list[dict[str, Any]]:
    return read_jsonl(dataset.root / "meta/episodes.jsonl")


def task_rows(dataset: RoboCoinDataset) -> list[dict[str, Any]]:
    return read_jsonl(dataset.root / "meta/tasks.jsonl")


def episode_data_path(dataset: RoboCoinDataset, episode_index: int) -> Path:
    chunk = episode_index // int(dataset.info.get("chunks_size", 1000))
    return dataset.root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def episode_video_path(dataset: RoboCoinDataset, video_key: str, episode_index: int) -> Path:
    chunk = episode_index // int(dataset.info.get("chunks_size", 1000))
    return (
        dataset.root
        / "videos"
        / f"chunk-{chunk:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )


def parquet_from_video_path(source_path: Path) -> Path:
    # .../<dataset>/videos/chunk-000/<video_key>/episode_000024.mp4
    source_path = Path(source_path).expanduser().resolve()
    dataset_dir = source_path.parents[3]
    chunk = source_path.parents[1].name
    return dataset_dir / "data" / chunk / f"{source_path.stem}.parquet"


def read_parquet_columns(path: Path, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    table = pq.read_table(path, columns=columns)
    out: dict[str, np.ndarray] = {}
    for key in table.column_names:
        values = table[key].to_pylist()
        out[key] = np.asarray(values)
    return out


def qpos_from_observation_state(
    state: np.ndarray,
    names: list[str],
    *,
    source: str = "observation.state",
) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{source}: expected observation.state (T,D), got {arr.shape}")

    def indices_for(prefix: str) -> list[int]:
        idxs: list[int] = []
        for joint_idx in range(1, 7):
            name = f"{prefix}_arm_joint_{joint_idx}_rad"
            if name not in names:
                raise KeyError(f"{source}: {name} missing from observation.state names")
            idxs.append(int(names.index(name)))
        gripper_name = f"{prefix}_gripper_open"
        if gripper_name not in names:
            raise KeyError(f"{source}: {gripper_name} missing from observation.state names")
        idxs.append(int(names.index(gripper_name)))
        return idxs

    if names:
        left = arr[:, indices_for("left")]
        right = arr[:, indices_for("right")]
        return np.ascontiguousarray(np.concatenate([left, right], axis=1))
    if arr.shape[1] >= 14:
        return np.ascontiguousarray(arr[:, :14])
    raise ValueError(f"{source}: cannot infer qpos without names from shape {arr.shape}")


def qpos_from_parquet(path: Path) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    dataset_dir = path.parents[2]
    info = read_json(dataset_dir / "meta" / "info.json")
    names = ((info.get("features") or {}).get("observation.state") or {}).get("names") or []
    columns = read_parquet_columns(path, ["observation.state"])
    return qpos_from_observation_state(
        np.asarray(columns["observation.state"], dtype=np.float64),
        [str(x) for x in names],
        source=str(path),
    )


def export_qpos_cache_from_summary(summary: dict[str, Any], out: Path) -> Path:
    parquet_value = summary.get("parquet_path") or summary.get("qpos_source")
    if parquet_value and str(parquet_value).endswith(".parquet"):
        parquet = Path(str(parquet_value)).expanduser().resolve()
    else:
        source = Path(str(summary["source_path"])).expanduser().resolve()
        parquet = parquet_from_video_path(source)
    qpos = qpos_from_parquet(parquet)
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=qpos)
    return parquet


def load_annotation_map(dataset: RoboCoinDataset, filename: str, key_field: str, text_field: str) -> dict[int, str]:
    path = dataset.root / "annotations" / filename
    if not path.exists():
        return {}
    mapping: dict[int, str] = {}
    for row in read_jsonl(path):
        if key_field in row and text_field in row:
            mapping[int(row[key_field])] = str(row[text_field])
    return mapping


def load_scene_annotations(dataset: RoboCoinDataset) -> dict[int, str]:
    path = dataset.root / "annotations/scene_annotations.jsonl"
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    for row in read_jsonl(path):
        if "episode_idx" in row and "scene_annotation" in row:
            out[int(row["episode_idx"])] = str(row["scene_annotation"])
    return out


def load_subtask_annotations(dataset: RoboCoinDataset) -> dict[int, str]:
    return load_annotation_map(
        dataset,
        "subtask_annotations.jsonl",
        "subtask_index",
        "subtask",
    )


def decode_indices(values: np.ndarray, mapping: dict[int, str], max_items: int = 16) -> list[str]:
    if values.size == 0:
        return []
    flat = np.asarray(values).reshape(-1)
    out: list[str] = []
    seen: set[int] = set()
    for value in flat:
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        out.append(mapping.get(idx, str(idx)))
        if len(out) >= max_items:
            break
    return out


def eef_pose_from_columns(columns: dict[str, np.ndarray], key: str = "eef_sim_pose_state") -> np.ndarray:
    if key not in columns:
        raise KeyError(f"missing {key}")
    arr = np.asarray(columns[key], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 12:
        raise ValueError(f"{key} must be (T,12+), got {arr.shape}")
    return arr[:, :12]


def slot_points_from_eef_pose(eef_pose: np.ndarray) -> list[np.ndarray]:
    return [eef_pose[:, 0:3], eef_pose[:, 6:9]]
