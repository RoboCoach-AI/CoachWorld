"""Index EEF image-projection sidecars generated from camera geometry smokes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EefProjectionRecord:
    sidecar_path: Path
    row_index: int
    uv: tuple[float, float] | None
    depth: float | None
    valid: bool | None
    image_hw: tuple[int, int]
    sample_id: str
    episode_uid: str


def camera_key_aliases(*values: str | None) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        aliases.add(text)
        leaf = text.split(".")[-1]
        aliases.add(leaf)
        if "exterior_1" in text or text in {"ext1", "exterior_1_left"}:
            aliases.update({"ext1", "exterior_1_left", "observation.images.exterior_1_left"})
        if "exterior_2" in text or text in {"ext2", "exterior_2_left"}:
            aliases.update({"ext2", "exterior_2_left", "observation.images.exterior_2_left"})
        if "wrist" in text:
            aliases.update({"wrist", "wrist_left", "observation.images.wrist_left"})
    return aliases


def parse_heatmap_size(value: Any) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.lower().replace(",", "x").strip()
        if not text:
            return None
        parts = [p for p in text.split("x") if p]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"heatmap size must be HxW or a 2-tuple, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"heatmap size must be positive, got {height}x{width}")
    return height, width


def _resize_heatmap_nn(heatmap: np.ndarray, out_hw: tuple[int, int] | None) -> np.ndarray:
    arr = np.asarray(heatmap, dtype=np.float32)
    if out_hw is None or arr.shape == tuple(out_hw):
        return arr.astype(np.float32, copy=False)
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if arr.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got {arr.shape}")
    src_h, src_w = arr.shape
    y = np.linspace(0, src_h - 1, out_h).round().astype(np.int64)
    x = np.linspace(0, src_w - 1, out_w).round().astype(np.int64)
    return arr[y[:, None], x[None, :]].astype(np.float32, copy=False)


class EefProjectionSidecarIndex:
    """Lookup EEF image projections by episode, source camera, and raw frame.

    The projection report stores rows in raw source-frame coordinates. Video
    latent training windows should therefore map latent frame ids through the
    ``frame_index`` signal before calling this index.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root if self.root.is_file() else self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"EEF projection manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.records: dict[tuple[int, str, int], EefProjectionRecord] = {}
        self._npz_cache: dict[Path, Any] = {}
        self.num_slots = 1
        self.output_num_slots: int | None = None
        self._build()

    @property
    def num_records(self) -> int:
        unique = {(rec.sidecar_path, rec.row_index) for rec in self.records.values()}
        return len(unique)

    @property
    def episode_ids(self) -> set[int]:
        return {key[0] for key in self.records}

    def _resolve_path(self, value: str) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.manifest_path.parent / path).resolve()

    def _build(self) -> None:
        for sample in self.manifest.get("samples", []):
            self._build_legacy_sample(sample)
        for sample in self.manifest.get("projection_samples", []):
            self._build_projection_sample(sample)

    def _build_legacy_sample(self, sample: dict[str, Any]) -> None:
        sidecar_raw = str(sample.get("sidecar", ""))
        if not sidecar_raw:
            return
        sidecar_path = self._resolve_path(sidecar_raw)
        episode_id = int(sample["episode_id"])
        sample_id = str(sample.get("sample_id", ""))
        episode_uid = str(sample.get("episode_uid", ""))
        for row_index, frame in enumerate(sample.get("frames", [])):
            raw_frame = int(frame["raw_frame_index"])
            uv_raw = frame.get("uv", [float("nan"), float("nan")])
            hw_raw = frame.get("image_hw", [0, 0])
            record = EefProjectionRecord(
                sidecar_path=sidecar_path,
                row_index=int(row_index),
                uv=(float(uv_raw[0]), float(uv_raw[1])),
                depth=float(frame.get("depth", 0.0)),
                valid=bool(frame.get("in_bounds", frame.get("valid", False))),
                image_hw=(int(hw_raw[0]), int(hw_raw[1])),
                sample_id=sample_id,
                episode_uid=episode_uid,
            )
            for alias in camera_key_aliases(
                frame.get("camera_key"),
                frame.get("camera_label"),
                sample.get("camera"),
            ):
                self.records[(episode_id, alias, raw_frame)] = record

    def _build_projection_sample(self, sample: dict[str, Any]) -> None:
        sidecar_raw = str(sample.get("sidecar", ""))
        if not sidecar_raw:
            return
        sidecar_path = self._resolve_path(sidecar_raw)
        if "episode_id" not in sample:
            raise KeyError(f"projection sample lacks episode_id: {sample.get('sample_id', '')}")
        episode_id = int(sample["episode_id"])
        sample_id = str(sample.get("sample_id", ""))
        episode_uid = str(sample.get("episode_uid", ""))
        npz = self._load_npz(sidecar_path)
        if "raw_frame_indices" in npz:
            raw_indices = np.asarray(npz["raw_frame_indices"], dtype=np.int64).reshape(-1)
        else:
            raw_indices = np.arange(np.asarray(npz["uv"]).shape[0], dtype=np.int64)
        uv = np.asarray(npz["uv"])
        if uv.ndim == 2 and uv.shape[-1] == 2:
            slots = 1
        elif uv.ndim == 3 and uv.shape[-1] == 2:
            slots = int(uv.shape[1])
        else:
            raise ValueError(f"{sidecar_path}: unsupported uv shape {uv.shape}")
        self.num_slots = max(self.num_slots, slots)
        image_hw = self._default_image_hw_from_npz(npz)
        aliases = camera_key_aliases(
            sample.get("camera_key"),
            sample.get("camera"),
            sample.get("camera_label"),
            sample.get("group"),
        )
        for row_index, raw_frame in enumerate(raw_indices):
            record = EefProjectionRecord(
                sidecar_path=sidecar_path,
                row_index=int(row_index),
                uv=None,
                depth=None,
                valid=None,
                image_hw=image_hw,
                sample_id=sample_id,
                episode_uid=episode_uid,
            )
            for alias in aliases:
                self.records[(episode_id, alias, int(raw_frame))] = record

    def _default_image_hw_from_npz(self, npz: Any) -> tuple[int, int]:
        if "image_hw" not in npz:
            return (0, 0)
        arr = np.asarray(npz["image_hw"])
        if arr.ndim == 2 and arr.shape[-1] == 2:
            return int(arr[0, 0]), int(arr[0, 1])
        if arr.ndim == 3 and arr.shape[-1] == 2:
            return int(arr[0, 0, 0]), int(arr[0, 0, 1])
        return (0, 0)

    def set_output_num_slots(self, num_slots: int) -> None:
        value = int(num_slots)
        if value < self.num_slots:
            raise ValueError(
                f"output num_slots={value} is smaller than sidecar num_slots={self.num_slots}"
            )
        self.output_num_slots = value

    def _get_record(
        self,
        *,
        episode_id: int,
        camera_key: str,
        raw_frame_index: int,
    ) -> EefProjectionRecord | None:
        for alias in camera_key_aliases(camera_key):
            record = self.records.get((int(episode_id), alias, int(raw_frame_index)))
            if record is not None:
                return record
        return None

    def _load_npz(self, path: Path) -> Any:
        resolved = Path(path).expanduser().resolve()
        cached = self._npz_cache.get(resolved)
        if cached is None:
            cached = np.load(resolved)
            self._npz_cache[resolved] = cached
        return cached

    def _row_array(
        self,
        npz: Any,
        key: str,
        row_index: int,
        *,
        fallback: Any,
        slots: int,
        tail: tuple[int, ...] = (),
        dtype: Any = np.float32,
    ) -> np.ndarray:
        if key in npz:
            row = np.asarray(npz[key][int(row_index)], dtype=dtype)
        else:
            row = np.asarray(fallback, dtype=dtype)
        if tail:
            if row.shape == tail:
                row = np.broadcast_to(row, (slots,) + tail)
            elif row.ndim == len(tail) + 1 and row.shape[-len(tail):] == tail:
                pass
            else:
                raise ValueError(f"{key} row has unsupported shape {row.shape}; expected {tail} or (S,{tail})")
        else:
            if row.ndim == 0:
                row = np.broadcast_to(row.reshape(1), (slots,))
            elif row.ndim == 1:
                pass
            else:
                raise ValueError(f"{key} row has unsupported shape {row.shape}; expected scalar or (S,)")
        if row.shape[0] > slots:
            return row[:slots].astype(dtype, copy=False)
        if row.shape[0] < slots:
            pad_shape = (slots - row.shape[0],) + row.shape[1:]
            pad = np.zeros(pad_shape, dtype=dtype)
            row = np.concatenate([row.astype(dtype, copy=False), pad], axis=0)
        return row.astype(dtype, copy=False)

    def _record_slot_count(self, record: EefProjectionRecord) -> int:
        npz = self._load_npz(record.sidecar_path)
        if "uv" in npz:
            uv = np.asarray(npz["uv"])
            if uv.ndim == 3 and uv.shape[-1] == 2:
                return int(uv.shape[1])
        return 1

    def _record_values(
        self,
        record: EefProjectionRecord,
        *,
        slots: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        npz = self._load_npz(record.sidecar_path)
        uv_fallback = record.uv if record.uv is not None else (float("nan"), float("nan"))
        depth_fallback = record.depth if record.depth is not None else float("nan")
        valid_fallback = record.valid if record.valid is not None else False
        uv = self._row_array(npz, "uv", record.row_index, fallback=uv_fallback, slots=slots, tail=(2,), dtype=np.float32)
        depth = self._row_array(npz, "depth", record.row_index, fallback=depth_fallback, slots=slots, dtype=np.float32)
        valid = self._row_array(npz, "valid", record.row_index, fallback=valid_fallback, slots=slots, dtype=np.bool_)
        image_hw = self._row_array(
            npz,
            "image_hw",
            record.row_index,
            fallback=record.image_hw,
            slots=slots,
            tail=(2,),
            dtype=np.int32,
        )
        return uv, depth, valid, image_hw

    def _record_heatmap(
        self,
        record: EefProjectionRecord,
        *,
        slots: int,
        heatmap_size: tuple[int, int],
    ) -> np.ndarray | None:
        npz = self._load_npz(record.sidecar_path)
        if "heatmap" not in npz:
            return None
        row = np.asarray(npz["heatmap"][record.row_index], dtype=np.float32)
        if row.ndim == 2:
            row = row[None, ...]
        if row.ndim != 3:
            raise ValueError(f"heatmap row must be (H,W) or (S,H,W), got {row.shape}")
        out = np.zeros((slots, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
        take = min(slots, row.shape[0])
        for slot in range(take):
            out[slot] = _resize_heatmap_nn(row[slot], heatmap_size)
        return out

    def lookup_sequence(
        self,
        *,
        episode_id: int,
        camera_keys: Iterable[str],
        raw_frame_indices: Iterable[int],
        load_heatmap: bool = False,
        heatmap_size: tuple[int, int] | None = None,
    ) -> dict[str, np.ndarray]:
        frame_ids = [int(x) for x in raw_frame_indices]
        keys = [str(x) for x in camera_keys]
        records: list[list[EefProjectionRecord | None]] = []
        slots = int(self.output_num_slots or self.num_slots)
        for raw_frame in frame_ids:
            row = []
            for camera_key in keys:
                record = self._get_record(
                    episode_id=int(episode_id),
                    camera_key=camera_key,
                    raw_frame_index=raw_frame,
                )
                row.append(record)
                if record is not None:
                    slots = max(slots, self._record_slot_count(record))
            records.append(row)
        uv = np.full((len(frame_ids), len(keys), slots, 2), np.nan, dtype=np.float32)
        depth = np.full((len(frame_ids), len(keys), slots), np.nan, dtype=np.float32)
        valid = np.zeros((len(frame_ids), len(keys), slots), dtype=np.bool_)
        image_hw = np.zeros((len(frame_ids), len(keys), slots, 2), dtype=np.int32)
        heatmap = None
        if load_heatmap:
            if heatmap_size is None:
                heatmap_size = self.default_heatmap_size()
            if heatmap_size is None:
                raise ValueError("cannot infer heatmap size from empty EEF sidecar index")
            heatmap = np.zeros((len(frame_ids), len(keys), slots, heatmap_size[0], heatmap_size[1]), dtype=np.float32)

        for t, row in enumerate(records):
            for v, record in enumerate(row):
                if record is None:
                    continue
                uv_row, depth_row, valid_row, hw_row = self._record_values(record, slots=slots)
                uv[t, v] = uv_row
                depth[t, v] = depth_row
                valid[t, v] = valid_row
                image_hw[t, v] = hw_row
                if heatmap is not None:
                    row_heatmap = self._record_heatmap(record, slots=slots, heatmap_size=heatmap_size)
                    if row_heatmap is not None:
                        heatmap[t, v] = row_heatmap

        out = {
            "eef_uv": np.ascontiguousarray(uv),
            "eef_depth": np.ascontiguousarray(depth),
            "eef_valid": np.ascontiguousarray(valid),
            "eef_image_hw": np.ascontiguousarray(image_hw),
        }
        if heatmap is not None:
            out["eef_heatmap"] = np.ascontiguousarray(heatmap)
        return out

    def default_heatmap_size(self) -> tuple[int, int] | None:
        for record in self.records.values():
            try:
                npz = self._load_npz(record.sidecar_path)
                heatmap = np.asarray(npz["heatmap"])
            except (FileNotFoundError, KeyError):
                continue
            if heatmap.ndim >= 3 and heatmap.shape[0] > 0:
                return int(heatmap.shape[-2]), int(heatmap.shape[-1])
        return None
