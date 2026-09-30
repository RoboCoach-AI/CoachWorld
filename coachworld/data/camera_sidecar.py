"""External camera sidecar lookup for video_latent roots.

The sidecar stores camera intrinsics and camera<-robot/world matrices in a
compact npz file.  It is intentionally separate from the original
``video_latent`` roots so calibration audits can be regenerated without
rewriting large latent shards.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from coachworld.data.eef_projection_sidecar import camera_key_aliases


@dataclass(frozen=True)
class CameraSidecarRecord:
    row_index: int
    split: str
    episode_id: int
    camera_key: str
    valid: bool
    validity: str
    source: str
    scene: str


class CameraSidecarIndex:
    """Lookup static camera matrices by split, episode, and source camera."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root if self.root.is_file() else self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"camera sidecar manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("kind") != "coachworld_camera_sidecar":
            raise ValueError(
                f"{manifest_path}: kind={self.manifest.get('kind')!r} "
                "!= 'coachworld_camera_sidecar'"
            )
        npz_raw = str(self.manifest.get("npz", "cameras.npz"))
        npz_path = Path(npz_raw).expanduser()
        if not npz_path.is_absolute():
            npz_path = manifest_path.parent / npz_path
        self.npz_path = npz_path.resolve()
        if not self.npz_path.exists():
            raise FileNotFoundError(self.npz_path)
        self.arrays = np.load(self.npz_path, allow_pickle=False)
        # NPZ members are compressed. Converting them inside every lookup
        # repeatedly decompresses the full table and turns an O(1) row lookup
        # into O(number_of_records) work. Keep the two hot arrays resident.
        self._camera_from_robot = np.asarray(
            self.arrays["camera_from_robot"], dtype=np.float32
        )
        self._intrinsics = np.asarray(self.arrays["K"], dtype=np.float32)
        self.records: dict[tuple[str, int, str], CameraSidecarRecord] = {}
        self._build()

    @property
    def num_records(self) -> int:
        return int(np.asarray(self.arrays["episode_id"]).shape[0])

    @property
    def num_valid_records(self) -> int:
        return int(np.asarray(self.arrays["valid"], dtype=np.bool_).sum())

    def _str_array(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            return np.asarray([""] * self.num_records)
        return np.asarray(self.arrays[key]).astype(str)

    def _build(self) -> None:
        episode_ids = np.asarray(self.arrays["episode_id"], dtype=np.int64)
        valid = np.asarray(self.arrays["valid"], dtype=np.bool_)
        split = self._str_array("split")
        camera_key = self._str_array("camera_key")
        source = self._str_array("source")
        validity = self._str_array("validity")
        scene = self._str_array("scene")
        if not (
            episode_ids.shape[0]
            == valid.shape[0]
            == split.shape[0]
            == camera_key.shape[0]
            == source.shape[0]
            == validity.shape[0]
            == scene.shape[0]
        ):
            raise ValueError(f"{self.npz_path}: camera sidecar arrays have inconsistent lengths")
        for i in range(int(episode_ids.shape[0])):
            record = CameraSidecarRecord(
                row_index=i,
                split=str(split[i]),
                episode_id=int(episode_ids[i]),
                camera_key=str(camera_key[i]),
                valid=bool(valid[i]),
                validity=str(validity[i]),
                source=str(source[i]),
                scene=str(scene[i]),
            )
            for alias in camera_key_aliases(record.camera_key):
                self.records[(record.split, record.episode_id, alias)] = record

    def _get_record(self, *, split: str, episode_id: int, camera_key: str) -> CameraSidecarRecord | None:
        for alias in camera_key_aliases(camera_key):
            record = self.records.get((str(split), int(episode_id), alias))
            if record is not None:
                return record
        return None

    def _untrusted_reason(self, record: CameraSidecarRecord) -> str:
        row = int(record.row_index)
        dataset = str(self.arrays["dataset"][row]) if "dataset" in self.arrays else ""
        if dataset != "droid":
            return ""
        if bool(self.manifest.get("droid_provenance_reviewed", False)):
            return ""
        if os.environ.get("COACHWORLD_TRUST_REVIEWED_DROID_SIDECAR", "").strip() == "1":
            return ""
        source = str(self.arrays["source"][row]) if "source" in self.arrays else ""
        source_intrinsics = (
            str(self.arrays["source_intrinsics"][row])
            if "source_intrinsics" in self.arrays
            else ""
        )
        if "vggt_omega_smoke_intrinsic" in source:
            return f"untrusted DROID K source={source!r}"
        if "caliball_droid_full_smoke" in source_intrinsics:
            return f"untrusted DROID K provenance={source_intrinsics!r}"
        if source_intrinsics.startswith("/") and not Path(source_intrinsics).exists():
            return f"DROID K provenance path does not exist: {source_intrinsics}"
        return ""

    def lookup_sequence(
        self,
        *,
        split: str,
        episode_id: int,
        camera_keys: Iterable[str],
        num_frames: int,
        require_valid: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(viewmats, Ks)`` with shapes ``(F,V,4,4)`` and ``(F,V,3,3)``."""
        keys = [str(x) for x in camera_keys]
        frames = int(num_frames)
        if frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        viewmats = np.zeros((frames, len(keys), 4, 4), dtype=np.float32)
        ks = np.zeros((frames, len(keys), 3, 3), dtype=np.float32)
        for view_id, key in enumerate(keys):
            record = self._get_record(split=str(split), episode_id=int(episode_id), camera_key=key)
            if record is None:
                raise KeyError(
                    f"camera sidecar has no record for split={split!r} "
                    f"episode_id={episode_id} camera_key={key!r}"
                )
            if require_valid and not record.valid:
                raise ValueError(
                    f"camera sidecar record is invalid for split={split!r} "
                    f"episode_id={episode_id} camera_key={key!r}: {record.validity}"
                )
            untrusted = self._untrusted_reason(record)
            if require_valid and untrusted:
                raise ValueError(
                    f"camera sidecar record is not trusted for split={split!r} "
                    f"episode_id={episode_id} camera_key={key!r}: {untrusted}"
                )
            T = self._camera_from_robot[record.row_index]
            K = self._intrinsics[record.row_index]
            if not np.isfinite(T).all() or not np.isfinite(K).all():
                raise ValueError(
                    f"camera sidecar record contains non-finite matrices for "
                    f"split={split!r} episode_id={episode_id} camera_key={key!r}"
                )
            viewmats[:, view_id] = T
            ks[:, view_id] = K
        return np.ascontiguousarray(viewmats), np.ascontiguousarray(ks)
