"""Strict policy-condition override storage for WM simulator rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from coachworld.data.action_normalization import ActionNormalizer
from coachworld.data.video_latent import ARM_SLOT_CONDITION_LAYOUT, SIGNAL_TIMEBASE


SCHEMA_VERSION = 1
KIND = "coachworld_condition_override"
INDEX_NAME = "index.jsonl"
MANIFEST_NAME = "manifest.json"


def condition_override_key(sample_index: int, episode_id: int, frame_now: int) -> str:
    return f"{int(sample_index):06d}_ep-{int(episode_id):06d}_frame-{int(frame_now):06d}"


def condition_override_filename(sample_index: int, episode_id: int, frame_now: int) -> str:
    return f"{condition_override_key(sample_index, episode_id, frame_now)}.npy"


class ConditionOverrideStore:
    """Read externally supplied WM condition tensors for rollout.

    Stored arrays are raw, unnormalized condition tensors with shape
    ``(T, arm_slots, action_dim + 1)``. The last channel is a binary slot mask.
    The reader applies the same normalizer as ``video_latent`` before returning
    tensors to the world model.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        condition_view: str,
        action_dim: int,
        max_arm_slots: int,
        normalizer: ActionNormalizer,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.condition_view = str(condition_view)
        self.action_dim = int(action_dim)
        self.max_arm_slots = int(max_arm_slots)
        self.normalizer = normalizer
        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}")
        if self.max_arm_slots <= 0:
            raise ValueError(f"max_arm_slots must be positive, got {self.max_arm_slots}")

        self.manifest = self._read_manifest()
        self._validate_manifest(self.manifest)
        self.entries_by_key = self._read_index()
        if not self.entries_by_key:
            raise ValueError(f"condition override index is empty: {self.root / INDEX_NAME}")

    def _read_manifest(self) -> dict[str, Any]:
        path = self.root / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        return payload

    def _validate_manifest(self, payload: dict[str, Any]) -> None:
        if payload.get("kind") != KIND:
            raise ValueError(f"{self.root} kind={payload.get('kind')!r} != {KIND!r}")
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"{self.root} schema_version={payload.get('schema_version')!r} "
                f"!= {SCHEMA_VERSION}"
            )
        if payload.get("condition_view") != self.condition_view:
            raise ValueError(
                f"{self.root} condition_view={payload.get('condition_view')!r} "
                f"!= {self.condition_view!r}"
            )
        if int(payload.get("action_dim", -1)) != self.action_dim:
            raise ValueError(
                f"{self.root} action_dim={payload.get('action_dim')!r} != {self.action_dim}"
            )
        if int(payload.get("max_arm_slots", -1)) != self.max_arm_slots:
            raise ValueError(
                f"{self.root} max_arm_slots={payload.get('max_arm_slots')!r} "
                f"!= {self.max_arm_slots}"
            )
        if payload.get("layout") != ARM_SLOT_CONDITION_LAYOUT:
            raise ValueError(
                f"{self.root} layout={payload.get('layout')!r} "
                f"!= {ARM_SLOT_CONDITION_LAYOUT!r}"
            )
        if payload.get("timebase") != SIGNAL_TIMEBASE:
            raise ValueError(
                f"{self.root} timebase={payload.get('timebase')!r} != {SIGNAL_TIMEBASE!r}"
            )
        if bool(payload.get("values_normalized")):
            raise ValueError(
                f"{self.root} stores normalized override values. Store raw condition "
                "values and let ConditionOverrideStore apply the video_latent normalizer."
            )

    def _read_index(self) -> dict[str, dict[str, Any]]:
        path = self.root / INDEX_NAME
        if not path.exists():
            raise FileNotFoundError(path)
        entries: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                for key in ("sample_index", "episode_id", "frame_now", "path", "shape", "dtype"):
                    if key not in entry:
                        raise ValueError(f"{path}:{line_no} missing key {key!r}")
                if entry["dtype"] != "float32":
                    raise ValueError(f"{path}:{line_no} dtype must be float32, got {entry['dtype']!r}")
                shape = [int(x) for x in entry["shape"]]
                expected_rank = 3
                if len(shape) != expected_rank:
                    raise ValueError(f"{path}:{line_no} shape must be rank {expected_rank}, got {shape}")
                if shape[1] != self.max_arm_slots or shape[2] != self.action_dim + 1:
                    raise ValueError(
                        f"{path}:{line_no} shape={shape}, expected "
                        f"(T,{self.max_arm_slots},{self.action_dim + 1})"
                    )
                key = condition_override_key(
                    int(entry["sample_index"]),
                    int(entry["episode_id"]),
                    int(entry["frame_now"]),
                )
                if key in entries:
                    raise ValueError(f"{path}:{line_no} duplicate override key {key}")
                entries[key] = entry
        return entries

    def _entry(self, *, sample_index: int, episode_id: int, frame_now: int) -> dict[str, Any]:
        key = condition_override_key(sample_index, episode_id, frame_now)
        try:
            return self.entries_by_key[key]
        except KeyError as exc:
            raise KeyError(f"missing condition override for {key} in {self.root}") from exc

    def _load_array(self, entry: dict[str, Any]) -> np.ndarray:
        rel = Path(str(entry["path"]))
        if rel.is_absolute():
            raise ValueError(f"condition override path must be relative: {rel}")
        path = (self.root / rel).resolve()
        if self.root not in path.parents:
            raise ValueError(f"condition override path escapes root: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.float32:
            raise ValueError(f"{path} dtype={arr.dtype}, expected float32")
        expected_shape = tuple(int(x) for x in entry["shape"])
        if arr.shape != expected_shape:
            raise ValueError(f"{path} shape={arr.shape}, expected {expected_shape}")
        return np.asarray(arr, dtype=np.float32)

    def load_condition_window(
        self,
        *,
        sample_index: int,
        episode_id: int,
        frame_now: int,
        state_start: int,
        needed_frames: int,
    ) -> np.ndarray:
        needed = int(needed_frames)
        if needed <= 0:
            raise ValueError(f"needed_frames must be positive, got {needed}")
        entry = self._entry(
            sample_index=int(sample_index),
            episode_id=int(episode_id),
            frame_now=int(frame_now),
        )
        arr = self._load_array(entry)
        raw_start = int(state_start)
        left_pad = max(0, -raw_start)
        start = max(0, raw_start)
        end = min(start + max(0, needed - left_pad), int(arr.shape[0]))
        if start >= end:
            raise ValueError(
                f"empty override condition window for {entry['path']}: "
                f"state_start={raw_start}, needed={needed}, T={arr.shape[0]}"
            )
        window = np.asarray(arr[start:end], dtype=np.float32)
        if left_pad:
            prefix = np.repeat(np.asarray(arr[0:1], dtype=np.float32), left_pad, axis=0)
            window = np.concatenate([prefix, window], axis=0)
        if window.shape[0] < needed:
            pad = np.repeat(window[-1:], needed - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        elif window.shape[0] > needed:
            window = window[:needed]

        values = window[..., : self.action_dim]
        mask = window[..., self.action_dim : self.action_dim + 1]
        if not np.isfinite(values).all():
            raise ValueError(f"{entry['path']} condition values contain non-finite values")
        if not np.isfinite(mask).all():
            raise ValueError(f"{entry['path']} condition mask contains non-finite values")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError(f"{entry['path']} condition mask must be binary")

        values = self.normalizer.normalize(values).astype(np.float32)
        values = np.clip(values, -5.0, 5.0)
        values = values * mask
        return np.ascontiguousarray(np.concatenate([values, mask], axis=-1))

    def load_raw_condition_array(
        self,
        *,
        sample_index: int,
        episode_id: int,
        frame_now: int,
    ) -> np.ndarray:
        """Return the complete raw override for geometry-aware projection."""

        entry = self._entry(
            sample_index=int(sample_index),
            episode_id=int(episode_id),
            frame_now=int(frame_now),
        )
        array = self._load_array(entry).copy()
        values = array[..., : self.action_dim]
        mask = array[..., self.action_dim : self.action_dim + 1]
        if not np.isfinite(values).all():
            raise ValueError(f"{entry['path']} condition values contain non-finite values")
        if not np.isfinite(mask).all():
            raise ValueError(f"{entry['path']} condition mask contains non-finite values")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError(f"{entry['path']} condition mask must be binary")
        return np.ascontiguousarray(array)
