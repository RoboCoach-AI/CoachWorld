"""Shared binary shard writer for the CoachWorld video-latent contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .video_latent import ARM_SLOT_CONDITION_LAYOUT, LATENT_LAYOUT, SIGNAL_TIMEBASE


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_signals(
    signals: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    fields: dict[str, dict[str, Any]] = {}
    parts: list[np.ndarray] = []
    offset_values = 0
    for name in sorted(signals):
        array = np.ascontiguousarray(signals[name].astype(np.float32, copy=False))
        fields[name] = {
            "shape": [int(array.shape[0]), int(array.shape[1])],
            "offset_values": int(offset_values),
        }
        parts.append(array.reshape(-1))
        offset_values += int(array.size)
    flat = np.concatenate(parts).astype(np.float32, copy=False)
    return np.ascontiguousarray(flat), fields


class ShardWriter:
    def __init__(self, root: Path, split: str, shard_id: int, *, rank: int, world_size: int):
        self.split = split
        self.shard_id = int(shard_id)
        self.shard_dir = root / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        if int(world_size) > 1:
            self.prefix = f"{split}_rank{int(rank):02d}_shard_{shard_id:06d}"
        else:
            self.prefix = f"{split}_shard_{shard_id:06d}"
        self.latent_name = f"{self.prefix}.latent.f16.bin"
        self.signal_name = f"{self.prefix}.signal.f32.bin"
        self.condition_name = f"{self.prefix}.condition.f32.bin"
        self.latent_file = (self.shard_dir / self.latent_name).open("ab")
        self.signal_file = (self.shard_dir / self.signal_name).open("ab")
        self.condition_file = (self.shard_dir / self.condition_name).open("ab")
        self.items: list[dict[str, Any]] = []

    def close(self) -> None:
        for stream in (self.latent_file, self.signal_file, self.condition_file):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        metadata = {
            "split": self.split,
            "shard_id": self.shard_id,
            "latent": self.latent_name,
            "signal": self.signal_name,
            "condition": self.condition_name,
            "episodes": self.items,
        }
        path = self.shard_dir / f"{self.prefix}.meta.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def write(
        self,
        *,
        episode_uid: str,
        latent: np.ndarray,
        signals: np.ndarray,
        condition: np.ndarray,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        latent = np.ascontiguousarray(latent.astype(np.float16, copy=False))
        signals = np.ascontiguousarray(signals.astype(np.float32, copy=False))
        condition = np.ascontiguousarray(condition.astype(np.float32, copy=False))
        latent_offset = self.latent_file.tell()
        signal_offset = self.signal_file.tell()
        condition_offset = self.condition_file.tell()
        self.latent_file.write(latent.tobytes(order="C"))
        self.signal_file.write(signals.tobytes(order="C"))
        self.condition_file.write(condition.tobytes(order="C"))
        self.items.append(
            {
                "episode_uid": episode_uid,
                "latent_offset_bytes": int(latent_offset),
                "signal_offset_bytes": int(signal_offset),
                "condition_offset_bytes": int(condition_offset),
                "latent_shape": [int(x) for x in latent.shape],
                "signal_values": int(signals.size),
                "condition_shape": [int(x) for x in condition.shape],
            }
        )
        return (
            {
                "shard": self.latent_name,
                "offset_bytes": int(latent_offset),
                "shape": [int(x) for x in latent.shape],
                "dtype": "float16",
                "layout": LATENT_LAYOUT,
            },
            {
                "shard": self.signal_name,
                "offset_bytes": int(signal_offset),
                "dtype": "float32",
                "timebase": SIGNAL_TIMEBASE,
            },
            {
                "shard": self.condition_name,
                "offset_bytes": int(condition_offset),
                "shape": [int(x) for x in condition.shape],
                "dtype": "float32",
                "layout": ARM_SLOT_CONDITION_LAYOUT,
            },
        )
