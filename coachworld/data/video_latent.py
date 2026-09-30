"""Strict metadata helpers for CoachWorld ``video_latent`` roots.

``video_latent`` stores per-view Wan latents plus source-aligned signals.  It
does not bind the data asset to one model's action tensor.  Model-specific
condition tensors live under ``model_condition_views`` and must declare their
source fields and transform.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from coachworld.data.action_domain import validate_domain_payload


SCHEMA_VERSION = 1
KIND = "coachworld_video_latent"
MANIFEST_NAME = "manifest.json"
VALID_SPLITS = {"train", "val"}

LATENT_LAYOUT = "T,V,C,H,W"
SIGNAL_TIMEBASE = "post_rgb_skip_video_grid"
ARM_SLOT_CONDITION_LAYOUT = "T,arm_slot,fields_plus_mask"

REQUIRED_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "source_dataset",
    "vae",
    "processing",
    "camera_schema",
    "signal_schema",
}

REQUIRED_INDEX_KEYS = {
    "dataset",
    "split",
    "episode_uid",
    "episode_id",
    "task",
    "text",
    "time",
    "views",
    "latent",
    "signals",
    "model_condition_views",
    "embodiment",
    "domain",
    "quality",
    "source",
}

REQUIRED_LATENT_KEYS = {"shard", "offset_bytes", "shape", "dtype", "layout"}
REQUIRED_SIGNALS_KEYS = {"shard", "offset_bytes", "dtype", "timebase", "fields"}
REQUIRED_FIELD_KEYS = {"shape", "offset_values"}
REQUIRED_CONDITION_KEYS = {
    "shard",
    "offset_bytes",
    "shape",
    "dtype",
    "layout",
    "source_fields",
    "transform",
}


def manifest_path(root: Path) -> Path:
    return Path(root) / MANIFEST_NAME


def split_index_path(root: Path, split: str) -> Path:
    assert_split(split)
    return Path(root) / f"{split}.index.jsonl"


def assert_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split={split!r}; expected one of {sorted(VALID_SPLITS)}")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    atomic_write_json(manifest_path(root), manifest)


def read_manifest(root: Path) -> dict[str, Any]:
    path = manifest_path(root)
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    missing = REQUIRED_MANIFEST_KEYS - set(manifest)
    if missing:
        raise ValueError(f"video_latent manifest missing keys: {sorted(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"video_latent schema_version mismatch: {manifest.get('schema_version')!r} != {SCHEMA_VERSION}"
        )
    if manifest.get("kind") != KIND:
        raise ValueError(f"video_latent kind mismatch: {manifest.get('kind')!r} != {KIND!r}")

    processing = manifest["processing"]
    if processing.get("latent_layout") != LATENT_LAYOUT:
        raise ValueError(f"processing.latent_layout must be {LATENT_LAYOUT!r}")
    if processing.get("latent_decode_contract") != "per_view_decode_then_stack_rgb":
        raise ValueError(
            "processing.latent_decode_contract must be 'per_view_decode_then_stack_rgb'"
        )
    video_keys = processing.get("video_keys")
    if not isinstance(video_keys, list) or not video_keys:
        raise ValueError("processing.video_keys must be a non-empty list")

    camera_schema = manifest["camera_schema"]
    if not bool(camera_schema.get("no_camera_duplication")):
        raise ValueError("camera_schema.no_camera_duplication must be true")
    num_views = int(camera_schema.get("num_views", len(video_keys)))
    if num_views != len(video_keys):
        raise ValueError(
            f"camera_schema.num_views={num_views} does not match "
            f"len(processing.video_keys)={len(video_keys)}"
        )
    if num_views <= 0:
        raise ValueError(f"camera_schema.num_views must be positive, got {num_views}")
    view_names = camera_schema.get("view_names")
    if view_names is not None:
        if not isinstance(view_names, list) or len(view_names) != num_views:
            raise ValueError(
                "camera_schema.view_names must be a list with length "
                f"num_views={num_views}"
            )
    view_local_required = bool(camera_schema.get("view_local_pe_required"))
    if num_views > 1 and not view_local_required:
        raise ValueError("multi-view video_latent must set view_local_pe_required=true")
    if num_views == 1 and view_local_required:
        raise ValueError("single-view video_latent must set view_local_pe_required=false")

    signal_schema = manifest["signal_schema"]
    if signal_schema.get("timebase") != SIGNAL_TIMEBASE:
        raise ValueError(f"signal_schema.timebase must be {SIGNAL_TIMEBASE!r}")
    if not bool(signal_schema.get("strict_alignment")):
        raise ValueError("signal_schema.strict_alignment must be true")


def _require_keys(obj: dict[str, Any], required: set[str], label: str) -> None:
    missing = required - set(obj)
    if missing:
        raise ValueError(f"{label} missing keys: {sorted(missing)}")


def _validate_shape(shape: Any, rank: int, label: str) -> list[int]:
    if not isinstance(shape, list) or len(shape) != rank:
        raise ValueError(f"{label}.shape must be rank-{rank} list, got {shape!r}")
    out = [int(x) for x in shape]
    if any(x <= 0 for x in out):
        raise ValueError(f"{label}.shape values must be positive, got {shape!r}")
    return out


def validate_index_entry(entry: dict[str, Any]) -> None:
    _require_keys(entry, REQUIRED_INDEX_KEYS, "index entry")
    assert_split(str(entry["split"]))
    domain = validate_domain_payload(entry["domain"])

    views = entry["views"]
    if not isinstance(views, list) or not views:
        raise ValueError("views must be a non-empty list")
    view_ids = [int(v.get("view_id")) for v in views]
    if view_ids != list(range(len(views))):
        raise ValueError(f"views must use contiguous view_id 0..V-1, got {view_ids}")

    time = entry["time"]
    for key in ("raw_fps", "fps", "rgb_skip", "num_raw_frames", "num_video_frames", "num_latent_frames"):
        if key not in time:
            raise ValueError(f"time missing key: {key}")
    num_video_frames = int(time["num_video_frames"])
    num_latent_frames = int(time["num_latent_frames"])

    latent = entry["latent"]
    _require_keys(latent, REQUIRED_LATENT_KEYS, "latent")
    if latent["layout"] != LATENT_LAYOUT:
        raise ValueError(f"latent.layout must be {LATENT_LAYOUT!r}, got {latent['layout']!r}")
    latent_shape = _validate_shape(latent["shape"], 5, "latent")
    if latent_shape[0] != num_latent_frames:
        raise ValueError(
            f"latent T={latent_shape[0]} does not match time.num_latent_frames={num_latent_frames}"
        )
    if latent_shape[1] != len(views):
        raise ValueError(f"latent V={latent_shape[1]} does not match len(views)={len(views)}")
    if latent.get("dtype") != "float16":
        raise ValueError(f"latent.dtype must be 'float16', got {latent.get('dtype')!r}")

    signals = entry["signals"]
    _require_keys(signals, REQUIRED_SIGNALS_KEYS, "signals")
    if signals["timebase"] != SIGNAL_TIMEBASE:
        raise ValueError(f"signals.timebase must be {SIGNAL_TIMEBASE!r}")
    if signals.get("dtype") != "float32":
        raise ValueError(f"signals.dtype must be 'float32', got {signals.get('dtype')!r}")
    fields = signals["fields"]
    if not isinstance(fields, dict) or not fields:
        raise ValueError("signals.fields must be a non-empty object")
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            raise ValueError(f"signals.fields[{name!r}] must be object")
        _require_keys(spec, REQUIRED_FIELD_KEYS, f"signals.fields[{name!r}]")
        shape = _validate_shape(spec["shape"], 2, f"signals.fields[{name!r}]")
        if shape[0] != num_video_frames:
            raise ValueError(
                f"signals field {name!r} T={shape[0]} does not match num_video_frames={num_video_frames}"
            )
        if int(spec["offset_values"]) < 0:
            raise ValueError(f"signals field {name!r} offset_values must be non-negative")

    conditions = entry["model_condition_views"]
    if not isinstance(conditions, dict):
        raise ValueError("model_condition_views must be an object")
    for name, spec in conditions.items():
        if not isinstance(spec, dict):
            raise ValueError(f"model_condition_views[{name!r}] must be object")
        _require_keys(spec, REQUIRED_CONDITION_KEYS, f"model_condition_views[{name!r}]")
        shape = _validate_shape(spec["shape"], 3, f"model_condition_views[{name!r}]")
        if shape[0] != num_video_frames:
            raise ValueError(
                f"condition {name!r} T={shape[0]} does not match num_video_frames={num_video_frames}"
            )
        if spec["layout"] != ARM_SLOT_CONDITION_LAYOUT:
            raise ValueError(
                f"condition {name!r} layout must be {ARM_SLOT_CONDITION_LAYOUT!r}, got {spec['layout']!r}"
            )
        if spec.get("dtype") != "float32":
            raise ValueError(f"condition {name!r} dtype must be 'float32'")
        source_fields = spec["source_fields"]
        if not isinstance(source_fields, list) or not source_fields:
            raise ValueError(f"condition {name!r} source_fields must be non-empty list")
        missing_sources = [field for field in source_fields if field not in fields]
        if missing_sources:
            raise ValueError(
                f"condition {name!r} source fields missing from signals: {missing_sources}"
            )

    quality = entry["quality"]
    for key in ("success", "suboptimal", "failure", "source_label"):
        if key not in quality:
            raise ValueError(f"quality missing key: {key}")

    embodiment = entry["embodiment"]
    arm_slots = embodiment.get("arm_slots")
    if not isinstance(arm_slots, list) or not arm_slots:
        raise ValueError("embodiment.arm_slots must be a non-empty list")
    domain_layout = list(domain["slot_layout"])
    if len(arm_slots) != len(domain_layout):
        raise ValueError(
            f"embodiment.arm_slots length={len(arm_slots)} does not match "
            f"domain.slot_layout length={len(domain_layout)}"
        )
    actual_layout = [
        str(slot.get("side")) if bool(slot.get("exists")) else "inactive"
        for slot in arm_slots
    ]
    if actual_layout != domain_layout:
        raise ValueError(
            f"embodiment arm slot layout {actual_layout!r} does not match "
            f"domain.slot_layout {domain_layout!r}"
        )


def iter_index(root: Path, split: str) -> Iterable[dict[str, Any]]:
    path = split_index_path(root, split)
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            try:
                validate_index_entry(entry)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            yield entry
