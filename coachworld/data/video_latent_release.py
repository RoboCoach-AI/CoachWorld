"""Build a portable release tree from a zero-copy video-latent collection."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


RELEASE_KIND = "coachworld_camera_aware_video_latent_release"


@dataclass(frozen=True)
class ReleaseFile:
    relative_path: Path
    source_path: Path
    source_was_symlink: bool
    size_bytes: int


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def iter_release_files(
    collection_root: str | Path,
    *,
    allowed_symlink_root: str | Path,
) -> Iterator[ReleaseFile]:
    """Yield files while resolving and validating every source symlink."""

    root = Path(collection_root).expanduser().resolve()
    allowed = Path(allowed_symlink_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"collection root does not exist: {root}")
    if not allowed.is_dir():
        raise FileNotFoundError(f"allowed symlink root does not exist: {allowed}")

    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for dirname in dirnames:
            directory = current_path / dirname
            if directory.is_symlink():
                raise ValueError(f"directory symlinks are not supported: {directory}")
        for filename in filenames:
            path = current_path / filename
            relative = path.relative_to(root)
            # Collection-level logs and local summaries are production artifacts,
            # not portable dataset payload. A sanitized release manifest is
            # generated separately below.
            if len(relative.parts) == 1:
                continue
            if len(relative.parts) == 2 and relative.name == "README.md":
                continue
            was_symlink = path.is_symlink()
            source = path.resolve(strict=True) if was_symlink else path
            if not source.is_file():
                raise ValueError(f"release source is not a regular file: {path}")
            if was_symlink and not _is_relative_to(source, allowed):
                raise ValueError(
                    f"symlink target escapes allowed asset root: {path} -> {source}"
                )
            yield ReleaseFile(
                relative_path=relative,
                source_path=source,
                source_was_symlink=was_symlink,
                size_bytes=source.stat().st_size,
            )


def _materialize_file(source: Path, destination: Path, *, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"invalid existing release path: {destination}")
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing release file has wrong size: {destination}")
        if mode == "hardlink" and not os.path.samefile(source, destination):
            raise ValueError(f"existing release file is not the expected hardlink: {destination}")
        return "reused"

    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError(
                f"cannot hardlink {source} to {destination}; use --mode copy only if "
                "duplicating the complete dataset is intentional"
            ) from exc
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported materialization mode: {mode!r}")
    return "created"


def _portable_root_record(record: dict[str, Any]) -> dict[str, Any]:
    output_name = Path(str(record["output_root"])).name
    family = str(record["family"])
    canonical_frame = (
        "franka_base_x_forward_y_left_z_up"
        if family.startswith("single_")
        else "robotwin_world_x_right_y_forward_z_up"
    )
    return {
        "dataset_key": str(record["dataset_key"]),
        "family": family,
        "path": output_name,
        "canonical_frame": canonical_frame,
        # Window counts belong to a trainer recipe, not to the video-latent
        # asset. Keep only stable split facts in the portable release.
        "splits": {
            split: {
                key: values[key]
                for key in ("episodes", "video_frames", "hours")
                if key in values
            }
            for split, values in record["splits"].items()
        },
        "filtered_source_entries": int(record.get("filtered_source_entries", 0)),
        "invalid_camera_entries": int(record.get("invalid_camera_entries", 0)),
        "admission_manifest_records": record.get("admission_manifest_records"),
        "projection_invariance_max_px": float(
            record.get("projection_invariance_max_px", 0.0)
        ),
    }


def _portable_totals(totals: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        split: {
            key: values[key]
            for key in ("episodes", "video_frames", "hours")
            if key in values
        }
        for split, values in totals.items()
    }


def _release_readme(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    roots = manifest["roots"]
    configs: list[str] = []
    for index, root in enumerate(roots):
        config = [
            f"- config_name: {root['dataset_key']}",
        ]
        if index == 0:
            config.append("  default: true")
        config.extend(
            [
                "  data_files:",
                "  - split: train",
                f"    path: {root['path']}/train.index.jsonl",
                "  - split: validation",
                f"    path: {root['path']}/val.index.jsonl",
            ]
        )
        configs.extend(config)
    front_matter = "\n".join(
        [
            "---",
            "license: other",
            "task:",
            "- video-generation",
            "task_categories:",
            "- robotics",
            "language:",
            "- zh",
            "- en",
            "tags:",
            "- robotics",
            "- world-model",
            "- embodied-ai",
            "- camera-aware",
            "- multi-embodiment",
            "- video-latent",
            "configs:",
            *configs,
            "---",
        ]
    )
    rows = "\n".join(
        "| {dataset_key} | {family} | {train:.3f} | {val:.3f} |".format(
            dataset_key=root["dataset_key"],
            family=root["family"],
            train=float(root["splits"]["train"]["hours"]),
            val=float(root["splits"]["val"]["hours"]),
        )
        for root in roots
    )
    return f"""{front_matter}

# CoachWorld Camera-Aware Video Latent v1

这是 CoachWorld 的 512x768、5 Hz、camera-aware `video_latent` 训练资产。它包含
{len(roots)} 个 production root、{int(totals['train']['episodes']):,} 个训练 episode
（{float(totals['train']['hours']):.3f} 小时）和
{int(totals['val']['episodes']):,} 个验证 episode
（{float(totals['val']['hours']):.3f} 小时）。

每个 root 都包含 Wan2.2 VAE latent、raw canonical arm-slot condition、目标图像坐标下的
相机内参 `K`、`T_camera_canonical`、domain/embodiment metadata 以及独立 q01/q99 统计。
`train.index.jsonl` 和 `val.index.jsonl` 是消费入口；所有 shard 都是普通文件，不依赖生产
机器上的绝对软链接。

| dataset_key | family | train hours | val hours |
| --- | --- | ---: | ---: |
{rows}

## 几何约定

- 单臂 canonical frame：`franka_base_x_forward_y_left_z_up`，对齐 DROID/Franka base
- 双臂 canonical frame：`robotwin_world_x_right_y_forward_z_up`，对齐 RoboTwin world
- 单臂只激活 slot0；双臂 slot0 为物理左臂、slot1 为物理右臂；缺失臂用 `slot_exists=0`
- camera convention：OpenCV，`p_camera = T_camera_canonical @ p_canonical`
- intrinsics space：letterbox 后的 512x768 target image
- condition：`xyz + rot6d + gripper + slot_exists`，存储值保持 canonical 米制物理量
- normalization：每个 production root 独立 q01/q99，仅在 reader 中作用于 numeric state

训练时可用同一份内嵌 K/T 把 raw canonical EEF 在线投影为 `uv/depth/valid`。可视化
heatmap/skeleton 不属于训练数据输入。

## 数据预览与读取

ModelScope 的 {len(roots)} 个 subset 分别指向各 production root 的 `train.index.jsonl` 和
`val.index.jsonl`。网页预览展示的是 episode 级索引、相机几何和 shard 定位元数据，
不是直接解码二进制 latent。实际训练读取 `.latent.f16.bin`、`.condition.f32.bin` 和
`.signal.f32.bin` 时，应使用 RoboCoach 的 video-latent reader，并以 index 中记录的
shard、offset、shape 和 dtype 为准。

## 发布状态

这是可移植数据包。公开发布前仍需逐项确认上游数据集的再分发许可和 attribution；
不要把仓库页面的默认 license 理解为所有上游数据衍生资产的统一授权。
"""


def _root_readme(record: dict[str, Any]) -> str:
    return f"""# CoachWorld production video_latent root

数据键：`{record['dataset_key']}`<br>
数据族：`{record['family']}`<br>
canonical frame：`{record['canonical_frame']}`

本 root 已嵌入目标图像坐标下的 `K` 与 `T_camera_canonical`，condition 与 camera 处于同一
canonical frame。存储 state 保持米制物理量，模型读取时使用本 root 的独立 q01/q99 做
numeric normalization。`train.index.jsonl` / `val.index.jsonl` 是入口；发布目录中的 shard
均为普通文件，不依赖生产机器的绝对软链接。
"""


def refresh_video_latent_release_metadata(
    release_root: str | Path,
) -> dict[str, Any]:
    """Regenerate release cards without traversing or rewriting the payload."""

    root = Path(release_root).expanduser().resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"release manifest does not exist: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("kind") != RELEASE_KIND:
        raise ValueError(f"unexpected release kind: {manifest.get('kind')!r}")
    roots = manifest.get("roots", [])
    if not roots:
        raise ValueError("release manifest contains no roots")

    _write_text(root / "README.md", _release_readme(manifest))
    for record in roots:
        production_root = root / str(record["path"])
        if not production_root.is_dir():
            raise FileNotFoundError(f"release root does not exist: {production_root}")
        _write_text(production_root / "README.md", _root_readme(record))
    return {
        "release_root": str(root),
        "metadata_files": len(roots) + 1,
        "configs": [str(record["dataset_key"]) for record in roots],
    }


def build_video_latent_release(
    collection_root: str | Path,
    output_root: str | Path,
    *,
    allowed_symlink_root: str | Path,
    mode: str = "hardlink",
) -> dict[str, Any]:
    """Materialize a portable, resumable release tree.

    Hardlink mode is intentionally the default: the production collection and
    release tree live on the same CFS filesystem, so a portable local tree does
    not need a second physical copy of the 874 GB payload.
    """

    source_root = Path(collection_root).expanduser().resolve()
    destination_root = Path(output_root).expanduser().resolve()
    if source_root == destination_root or _is_relative_to(destination_root, source_root):
        raise ValueError("release output must be outside the source collection")

    summary_path = source_root / "collection_summary.json"
    summary = _read_json(summary_path)
    if not bool(summary.get("complete")) or summary.get("missing_roots"):
        raise ValueError("source collection is not complete")
    roots = [_portable_root_record(record) for record in summary.get("roots", [])]
    if not roots:
        raise ValueError("source collection contains no production roots")
    expected_dirs = {root["path"] for root in roots}
    actual_dirs = {path.name for path in source_root.iterdir() if path.is_dir()}
    missing_dirs = sorted(expected_dirs - actual_dirs)
    if missing_dirs:
        raise ValueError(f"collection summary references missing roots: {missing_dirs}")

    destination_root.mkdir(parents=True, exist_ok=True)
    created = reused = source_symlinks = total_bytes = 0
    per_root: dict[str, dict[str, int]] = {}
    for item in iter_release_files(
        source_root,
        allowed_symlink_root=allowed_symlink_root,
    ):
        status = _materialize_file(
            item.source_path,
            destination_root / item.relative_path,
            mode=mode,
        )
        created += status == "created"
        reused += status == "reused"
        source_symlinks += int(item.source_was_symlink)
        total_bytes += item.size_bytes
        root_name = item.relative_path.parts[0]
        stats = per_root.setdefault(root_name, {"files": 0, "bytes": 0})
        stats["files"] += 1
        stats["bytes"] += item.size_bytes

    release_manifest = {
        "schema_version": 2,
        "kind": RELEASE_KIND,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "materialization": mode,
        "portable": True,
        "contains_symlinks": False,
        "target_fps": float(summary["target_fps"]),
        "target_image_hw": [int(x) for x in summary["target_image_hw"]],
        "roots": roots,
        "totals": _portable_totals(summary["totals"]),
        "payload": {
            "files": int(created + reused),
            "bytes": int(total_bytes),
            "source_symlinks_materialized": int(source_symlinks),
            "per_root": per_root,
        },
    }
    _write_json(destination_root / "release_manifest.json", release_manifest)
    refresh_video_latent_release_metadata(destination_root)

    remaining_links = [path for path in destination_root.rglob("*") if path.is_symlink()]
    if remaining_links:
        raise ValueError(f"release still contains symlinks: {remaining_links[:3]}")
    return release_manifest
