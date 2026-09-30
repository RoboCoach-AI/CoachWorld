"""Shared strict reader for ``video_latent`` roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from coachworld.data.camera_sidecar import CameraSidecarIndex
from coachworld.data.action_normalization import ActionNormalizer
from coachworld.data.video_latent import iter_index, read_manifest
from coachworld.world_model.multi_camera import stack_camera_latents


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return R = Rz(yaw) @ Ry(pitch) @ Rx(roll) for (..., 3) roll/pitch/yaw."""
    arr = np.asarray(rpy, dtype=np.float32)
    if arr.shape[-1] != 3:
        raise ValueError(f"rpy must end with 3 values, got {arr.shape}")
    roll, pitch, yaw = arr[..., 0], arr[..., 1], arr[..., 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    out = np.zeros(arr.shape[:-1] + (3, 3), dtype=np.float32)
    out[..., 0, 0] = cy * cp
    out[..., 0, 1] = cy * sp * sr - sy * cr
    out[..., 0, 2] = cy * sp * cr + sy * sr
    out[..., 1, 0] = sy * cp
    out[..., 1, 1] = sy * sp * sr + cy * cr
    out[..., 1, 2] = sy * sp * cr - cy * sr
    out[..., 2, 0] = -sp
    out[..., 2, 1] = cp * sr
    out[..., 2, 2] = cp * cr
    return out


def _invert_se3_np(transform: np.ndarray) -> np.ndarray:
    arr = np.asarray(transform, dtype=np.float32)
    if arr.shape[-2:] != (4, 4):
        raise ValueError(f"SE(3) transform must end with (4,4), got {arr.shape}")
    out = np.zeros_like(arr)
    r_inv = np.swapaxes(arr[..., :3, :3], -1, -2)
    out[..., :3, :3] = r_inv
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", r_inv, arr[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _as_4x4_matrix(value: Any, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = arr
        return out
    raise ValueError(f"{label} must be 4x4 or 3x4, got {arr.shape}")


def _pose6_to_viewmats(pose6: np.ndarray, *, convention: str) -> np.ndarray:
    """Convert xyz+rpy camera pose rows to PRoPE viewmats.

    PRoPE expects camera<-world matrices. Most dataset camera pose columns are
    stored as world<-camera poses, hence the default inversion.
    """
    arr = np.asarray(pose6, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"camera extrinsics must be (T,>=6) xyz+rpy, got {arr.shape}")
    mats = np.zeros((arr.shape[0], 4, 4), dtype=np.float32)
    mats[:, :3, :3] = _rpy_to_matrix(arr[:, 3:6])
    mats[:, :3, 3] = arr[:, :3]
    mats[:, 3, 3] = 1.0
    convention = str(convention)
    if convention == "world_from_camera":
        return _invert_se3_np(mats)
    if convention == "camera_from_world":
        return mats
    raise ValueError(
        "video_latent_camera_extrinsics_convention must be "
        f"'world_from_camera' or 'camera_from_world', got {convention!r}"
    )


def _intrinsics_from_view(view: dict[str, Any], *, fallback: str) -> np.ndarray:
    intrinsics = view.get("intrinsics")
    if intrinsics is None:
        if str(fallback) == "identity":
            return np.eye(3, dtype=np.float32)
        if str(fallback) == "error":
            raise ValueError(
                f"view {view.get('name', view.get('view_id'))!r} has intrinsics=null"
            )
        raise ValueError(
            "video_latent_camera_intrinsics_fallback must be 'identity' or 'error', "
            f"got {fallback!r}"
        )
    if isinstance(intrinsics, dict):
        for key in ("target", "K_target", "K", "raw"):
            if key in intrinsics:
                arr = np.asarray(intrinsics[key], dtype=np.float32)
                if arr.shape != (3, 3):
                    raise ValueError(
                        f"view {view.get('name', view.get('view_id'))!r} "
                        f"intrinsics[{key!r}] must be 3x3, got {arr.shape}"
                    )
                return arr
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics.get("cx", 0.0))
        cy = float(intrinsics.get("cy", 0.0))
        return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError(
            f"view {view.get('name', view.get('view_id'))!r} intrinsics must be 3x3, got {arr.shape}"
        )
    return arr


def _camera_from_world_from_view(view: dict[str, Any]) -> np.ndarray | None:
    """Return static camera<-world/rig matrix stored directly in view metadata.

    Older roots store camera pose as a signal field referenced by
    ``extrinsics_source``.  Newer reviewed roots store a fixed 4x4 matrix in
    ``views[].extrinsics``.  PRoPE consumes camera<-world/rig matrices, so the
    accepted keys below are all interpreted in that direction unless explicitly
    named world-from-camera.
    """
    extrinsics = view.get("extrinsics")
    if not isinstance(extrinsics, dict):
        return None
    camera_keys = (
        "T_camera_canonical",
        "T_camera_world",
        "T_camera_rig",
        "T_camera_rig_cv",
        "camera_from_world",
        "camera_from_robot",
        "extrinsic_cv",
        "extrinsic_cv_3x4",
    )
    for key in camera_keys:
        if key in extrinsics:
            return _as_4x4_matrix(
                extrinsics[key],
                label=f"view {view.get('name', view.get('view_id'))!r} extrinsics[{key!r}]",
            )
    world_keys = ("T_world_camera", "world_from_camera", "cam2world", "cam2world_gl")
    for key in world_keys:
        if key in extrinsics:
            return _invert_se3_np(
                _as_4x4_matrix(
                    extrinsics[key],
                    label=f"view {view.get('name', view.get('view_id'))!r} extrinsics[{key!r}]",
                )
            )
    return None


class VideoLatentRootReader:
    """Read one split of a ``video_latent`` root.

    This class is intentionally shared by training and rollout/eval. A mismatch
    in camera order, view count, latent geometry, action shape, or condition
    stats must fail before model code sees the batch.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_cameras: int,
        condition_view: str,
        action_dim: int,
        latent_height_per_view: int,
        latent_width: int,
        expected_video_keys: Optional[list[str]] = None,
        require_production_contract: bool = False,
        required_target_fps: float = 0.0,
        required_target_image_hw: Optional[list[int] | tuple[int, int]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.num_cameras = int(num_cameras)
        self.condition_view = str(condition_view)
        self.action_dim = int(action_dim)
        self.latent_height_per_view = int(latent_height_per_view)
        self.latent_width = int(latent_width)
        self.expected_video_keys = [str(x) for x in (expected_video_keys or [])]
        self.require_production_contract = bool(require_production_contract)
        self.required_target_fps = float(required_target_fps)
        self.required_target_image_hw = tuple(
            int(x) for x in (required_target_image_hw or [])
        )
        if self.required_target_image_hw and len(self.required_target_image_hw) != 2:
            raise ValueError(
                "required_target_image_hw must be empty or [height, width], got "
                f"{self.required_target_image_hw}"
            )
        if self.num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {self.num_cameras}")
        if self.latent_height_per_view <= 0 or self.latent_width <= 0:
            raise ValueError(
                "latent geometry must be positive, got "
                f"latent_height_per_view={self.latent_height_per_view}, "
                f"latent_width={self.latent_width}"
            )

        self.manifest = read_manifest(self.root)
        self._validate_manifest_geometry()
        if self.require_production_contract:
            self._validate_production_manifest()
        self.entries = list(iter_index(self.root, self.split))
        if not self.entries:
            raise ValueError(f"{self.root}/{self.split}.index.jsonl contains no entries")
        self.entries_by_episode: dict[int, dict[str, Any]] = {}
        for entry in self.entries:
            self.validate_entry_geometry(entry)
            if self.require_production_contract:
                self.validate_entry_production_contract(entry)
            episode_id = int(entry["episode_id"])
            if episode_id in self.entries_by_episode:
                raise ValueError(
                    f"{self.root} split={self.split} has duplicate episode_id={episode_id}"
                )
            self.entries_by_episode[episode_id] = entry
        self.normalizer = self._load_condition_normalizer()
        self._memmaps: dict[tuple[str, str], np.memmap] = {}

    def _validate_manifest_geometry(self) -> None:
        video_keys = self.manifest["processing"]["video_keys"]
        if self.expected_video_keys and video_keys != self.expected_video_keys:
            raise ValueError(
                f"{self.root} processing.video_keys={video_keys!r} "
                f"!= config video_latent_video_keys={self.expected_video_keys!r}"
            )
        if len(video_keys) != self.num_cameras:
            raise ValueError(
                f"{self.root} processing.video_keys has {len(video_keys)} view(s), "
                f"but config num_cameras={self.num_cameras}"
            )
        num_views = int(self.manifest["camera_schema"].get("num_views", len(video_keys)))
        if num_views != self.num_cameras:
            raise ValueError(
                f"{self.root} camera_schema.num_views={num_views} "
                f"!= config num_cameras={self.num_cameras}"
            )

    def _validate_production_manifest(self) -> None:
        contract = self.manifest.get("production_contract")
        if not isinstance(contract, dict) or contract.get("kind") != "coachworld_camera_aware_canonical_root":
            raise ValueError(
                f"{self.root} is not a final camera-aware canonical production root"
            )
        canonical_frame = str(contract.get("canonical_frame", "")).strip()
        if not canonical_frame:
            raise ValueError(f"{self.root} production_contract.canonical_frame is empty")
        if contract.get("camera_extrinsics_key") != "T_camera_canonical":
            raise ValueError(
                f"{self.root} production contract must require T_camera_canonical, got "
                f"{contract.get('camera_extrinsics_key')!r}"
            )
        if contract.get("condition_normalization") != "dataset_root_quantile_after_canonicalization":
            raise ValueError(
                f"{self.root} has unsupported production condition normalization "
                f"{contract.get('condition_normalization')!r}"
            )
        processing = self.manifest.get("processing", {})
        manifest_fps = float(processing.get("target_fps", 0.0))
        contract_fps = float(contract.get("target_fps", 0.0))
        if manifest_fps <= 0.0 or abs(manifest_fps - contract_fps) > 1e-6:
            raise ValueError(
                f"{self.root} target_fps disagreement: processing={manifest_fps}, "
                f"production_contract={contract_fps}"
            )
        if self.required_target_fps > 0.0 and abs(manifest_fps - self.required_target_fps) > 1e-6:
            raise ValueError(
                f"{self.root} target_fps={manifest_fps} != required {self.required_target_fps}"
            )
        manifest_hw = tuple(int(x) for x in processing.get("target_size", []))
        if self.required_target_image_hw and manifest_hw != self.required_target_image_hw:
            raise ValueError(
                f"{self.root} processing.target_size={manifest_hw} != required "
                f"{self.required_target_image_hw}"
            )
        condition_schema = self.manifest.get("condition_schema", {})
        if condition_schema.get("condition_view") != self.condition_view:
            raise ValueError(
                f"{self.root} production condition_view="
                f"{condition_schema.get('condition_view')!r} != {self.condition_view!r}"
            )

    @staticmethod
    def _validate_se3(matrix: np.ndarray, *, label: str) -> None:
        value = np.asarray(matrix, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError(f"{label} must be a finite 4x4 matrix")
        if not np.allclose(value[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            raise ValueError(f"{label} has invalid homogeneous bottom row")
        rotation = value[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
            raise ValueError(f"{label} rotation is not orthonormal")
        determinant = float(np.linalg.det(rotation))
        if abs(determinant - 1.0) > 2e-3:
            raise ValueError(f"{label} rotation determinant={determinant}, expected +1")

    def validate_entry_production_contract(self, entry: dict[str, Any]) -> None:
        uid = str(entry.get("episode_uid", "<unknown>"))
        contract = self.manifest["production_contract"]
        canonical_frame = str(contract["canonical_frame"])
        condition = entry.get("model_condition_views", {}).get(self.condition_view)
        if not isinstance(condition, dict):
            raise ValueError(f"{self.root} {uid} missing production condition view")
        if condition.get("canonical_frame") != canonical_frame:
            raise ValueError(
                f"{self.root} {uid} condition canonical_frame="
                f"{condition.get('canonical_frame')!r} != {canonical_frame!r}"
            )
        domain = entry.get("domain", {})
        if domain.get("canonical_action_frame") != canonical_frame or domain.get("canonicalized") is not True:
            raise ValueError(f"{self.root} {uid} domain is not canonicalized to {canonical_frame!r}")
        views = entry.get("views")
        if not isinstance(views, list) or len(views) != self.num_cameras:
            raise ValueError(f"{self.root} {uid} has invalid production views")
        for view in views:
            name = view.get("name", view.get("view_id"))
            intrinsics = view.get("intrinsics")
            if not isinstance(intrinsics, dict) or "target" not in intrinsics:
                raise ValueError(f"{self.root} {uid} view {name!r} lacks target-space K")
            K = np.asarray(intrinsics["target"], dtype=np.float64)
            if K.shape != (3, 3) or not np.isfinite(K).all():
                raise ValueError(f"{self.root} {uid} view {name!r} has invalid target-space K")
            if K[0, 0] <= 0.0 or K[1, 1] <= 0.0 or abs(float(K[2, 2]) - 1.0) > 1e-5:
                raise ValueError(f"{self.root} {uid} view {name!r} has nonphysical target-space K")
            image_hw = tuple(int(x) for x in view.get("image_transform", {}).get("target_image_hw", []))
            if self.required_target_image_hw and image_hw != self.required_target_image_hw:
                raise ValueError(
                    f"{self.root} {uid} view {name!r} target_image_hw={image_hw} != "
                    f"required {self.required_target_image_hw}"
                )
            extrinsics = view.get("extrinsics")
            if not isinstance(extrinsics, dict) or "T_camera_canonical" not in extrinsics:
                raise ValueError(f"{self.root} {uid} view {name!r} lacks T_camera_canonical")
            if extrinsics.get("canonical_frame") != canonical_frame:
                raise ValueError(
                    f"{self.root} {uid} view {name!r} extrinsics canonical frame mismatch"
                )
            self._validate_se3(
                np.asarray(extrinsics["T_camera_canonical"]),
                label=f"{self.root} {uid} view {name!r} T_camera_canonical",
            )

    def validate_entry_geometry(self, entry: dict[str, Any]) -> None:
        shape = [int(x) for x in entry["latent"]["shape"]]
        if len(shape) != 5:
            raise ValueError(f"{self.root} {entry['episode_uid']} latent shape must be rank 5, got {shape}")
        _, views, channels, height, width = shape
        if views != self.num_cameras:
            raise ValueError(
                f"{self.root} {entry['episode_uid']} latent V={views} "
                f"!= config num_cameras={self.num_cameras}"
            )
        if channels != 48:
            raise ValueError(f"{self.root} {entry['episode_uid']} latent C={channels} != 48")
        if height != self.latent_height_per_view or width != self.latent_width:
            raise ValueError(
                f"{self.root} {entry['episode_uid']} latent per-view HxW={height}x{width} "
                f"!= config {self.latent_height_per_view}x{self.latent_width}"
            )

    def entry_for_episode(self, episode_id: int) -> dict[str, Any]:
        try:
            return self.entries_by_episode[int(episode_id)]
        except KeyError as exc:
            raise KeyError(
                f"episode_id={episode_id} not found in "
                f"{self.root}/{self.split}.index.jsonl"
            ) from exc

    def entry_for_index(self, entry_index: int) -> dict[str, Any]:
        index = int(entry_index)
        if not 0 <= index < len(self.entries):
            raise IndexError(
                f"entry_index={index} outside [0, {len(self.entries)}) for "
                f"{self.root}/{self.split}.index.jsonl"
            )
        return self.entries[index]

    def _load_condition_normalizer(self) -> ActionNormalizer:
        path = self.root / "stats" / f"{self.condition_view}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing video_latent condition stats: {path}. "
                "Run scripts/data/build_video_latent_condition_stats.py before training/eval."
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("condition_view") != self.condition_view:
            raise ValueError(
                f"{path} condition_view={payload.get('condition_view')!r} "
                f"!= {self.condition_view!r}"
            )
        if int(payload.get("feature_dim")) != self.action_dim:
            raise ValueError(
                f"{path} feature_dim={payload.get('feature_dim')} "
                f"!= action_dim={self.action_dim}"
            )
        if self.require_production_contract:
            expected_scope = "dataset_root_quantile_after_canonicalization"
            if payload.get("normalization_scope") != expected_scope:
                raise ValueError(
                    f"{path} normalization_scope={payload.get('normalization_scope')!r} "
                    f"!= {expected_scope!r}"
                )
        return ActionNormalizer(
            q01=np.asarray(payload["q01"], dtype=np.float64),
            q99=np.asarray(payload["q99"], dtype=np.float64),
        )

    def _dtype_for_spec(self, spec: dict[str, Any]) -> np.dtype:
        dtype_name = str(spec["dtype"])
        if dtype_name == "float16":
            return np.dtype(np.float16)
        elif dtype_name == "float32":
            return np.dtype(np.float32)
        raise ValueError(f"Unsupported video_latent dtype={dtype_name!r}")

    def _memmap(self, spec: dict[str, Any]) -> np.ndarray:
        dtype = self._dtype_for_spec(spec)
        dtype_name = str(spec["dtype"])
        shard_name = str(spec["shard"])
        key = (shard_name, dtype_name)
        if key not in self._memmaps:
            shard_path = self.root / "shards" / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            size_bytes = shard_path.stat().st_size
            if size_bytes % dtype.itemsize != 0:
                raise ValueError(
                    f"{shard_path} byte size {size_bytes} is not divisible by dtype {dtype_name}"
                )
            self._memmaps[key] = np.memmap(
                shard_path,
                mode="r",
                dtype=dtype,
                shape=(size_bytes // dtype.itemsize,),
            )
        offset_bytes = int(spec["offset_bytes"])
        if offset_bytes % dtype.itemsize != 0:
            raise ValueError(
                f"{self.root} {shard_name} offset_bytes={offset_bytes} "
                f"is not divisible by dtype {dtype_name}"
            )
        shape = tuple(int(x) for x in spec["shape"])
        count = int(np.prod(shape))
        start = offset_bytes // dtype.itemsize
        end = start + count
        base = self._memmaps[key]
        if end > int(base.shape[0]):
            raise ValueError(
                f"{self.root} {shard_name} spec exceeds shard bounds: "
                f"start={start} count={count} shard_values={base.shape[0]}"
            )
        return base[start:end].reshape(shape)

    def load_signal_field(self, entry: dict[str, Any], field_name: str) -> np.ndarray:
        signals = entry.get("signals")
        if not isinstance(signals, dict):
            raise KeyError(f"{entry['episode_uid']} has no signals block")
        fields = signals.get("fields", {})
        field = fields.get(str(field_name))
        if field is None:
            raise KeyError(f"{entry['episode_uid']} missing signal field {field_name!r}")
        dtype = self._dtype_for_spec(signals)
        shard_name = str(signals["shard"])
        key = (shard_name, str(signals["dtype"]))
        if key not in self._memmaps:
            shard_path = self.root / "shards" / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            size_bytes = shard_path.stat().st_size
            self._memmaps[key] = np.memmap(
                shard_path,
                mode="r",
                dtype=dtype,
                shape=(size_bytes // dtype.itemsize,),
            )
        base = int(signals["offset_bytes"]) // dtype.itemsize
        start = base + int(field["offset_values"])
        shape = tuple(int(x) for x in field["shape"])
        count = int(np.prod(shape))
        end = start + count
        mmap = self._memmaps[key]
        if end > int(mmap.shape[0]):
            raise ValueError(
                f"{self.root} {shard_name} signal {field_name!r} exceeds shard bounds"
            )
        return np.asarray(mmap[start:end].reshape(shape), dtype=np.float32)

    def load_stacked_latents(
        self,
        entry: dict[str, Any],
        latent_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Return latents as ``(C,T,V*H,W)`` height-stacked by view."""
        spec = entry["latent"]
        if spec["layout"] != "T,V,C,H,W":
            raise ValueError(f"{entry['episode_uid']} latent layout must be T,V,C,H,W")
        arr = self._memmap(spec)
        if int(arr.shape[1]) != self.num_cameras:
            raise ValueError(
                f"{entry['episode_uid']} view count {arr.shape[1]} "
                f"!= configured num_cameras={self.num_cameras}"
            )
        if latent_ids is None:
            selected = torch.from_numpy(np.asarray(arr, dtype=np.float16).copy())
        else:
            ids = [int(i) for i in latent_ids]
            bad = [i for i in ids if i < 0 or i >= int(arr.shape[0])]
            if bad:
                raise IndexError(
                    f"{entry['episode_uid']} latent ids out of range: "
                    f"{bad[:10]} for T={arr.shape[0]}"
                )
            selected = torch.from_numpy(np.asarray(arr[ids], dtype=np.float16))
        cam_latents = [
            selected[:, view_id].permute(1, 0, 2, 3).contiguous()
            for view_id in range(self.num_cameras)
        ]
        return stack_camera_latents(cam_latents)

    def load_condition_window(
        self,
        entry: dict[str, Any],
        *,
        state_start: int,
        needed_frames: int,
    ) -> np.ndarray:
        conditions = entry["model_condition_views"]
        if self.condition_view not in conditions:
            raise KeyError(
                f"{entry['episode_uid']} missing condition view "
                f"{self.condition_view!r}"
            )
        spec = conditions[self.condition_view]
        arr = self._memmap(spec)
        if arr.ndim != 3 or int(arr.shape[-1]) != self.action_dim + 1:
            raise ValueError(
                f"{entry['episode_uid']} condition shape {arr.shape}, "
                f"expected (T,S,{self.action_dim + 1})"
            )
        needed = int(needed_frames)
        if needed <= 0:
            raise ValueError(f"needed_frames must be positive, got {needed}")
        raw_start = int(state_start)
        left_pad = max(0, -raw_start)
        start = max(0, raw_start)
        end = min(start + max(0, needed - left_pad), int(arr.shape[0]))
        if start >= end:
            raise ValueError(
                f"{entry['episode_uid']} empty condition window: "
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
        if not np.isfinite(mask).all():
            raise ValueError(f"{entry['episode_uid']} condition mask contains non-finite values")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError(f"{entry['episode_uid']} condition mask must be binary")
        values = self.normalizer.normalize(values).astype(np.float32)
        values = np.clip(values, -5.0, 5.0)
        values = values * mask
        return np.ascontiguousarray(np.concatenate([values, mask], axis=-1))

    def load_raw_condition_array(self, entry: dict[str, Any]) -> np.ndarray:
        """Return raw, unnormalized condition values for the configured view.

        Shape is ``(T, S, action_dim + 1)`` where the last channel is the
        binary slot-exists mask.  This is intended for rollout tooling that
        edits future state conditions before applying the same normalizer used
        by training.
        """
        conditions = entry["model_condition_views"]
        if self.condition_view not in conditions:
            raise KeyError(
                f"{entry['episode_uid']} missing condition view "
                f"{self.condition_view!r}"
            )
        spec = conditions[self.condition_view]
        arr = np.asarray(self._memmap(spec), dtype=np.float32).copy()
        if arr.ndim != 3 or int(arr.shape[-1]) != self.action_dim + 1:
            raise ValueError(
                f"{entry['episode_uid']} condition shape {arr.shape}, "
                f"expected (T,S,{self.action_dim + 1})"
            )
        mask = arr[..., self.action_dim : self.action_dim + 1]
        if not np.isfinite(arr).all():
            raise ValueError(f"{entry['episode_uid']} raw condition contains non-finite values")
        if np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError(f"{entry['episode_uid']} condition mask must be binary")
        return arr

    def load_camera_window(
        self,
        entry: dict[str, Any],
        *,
        latent_ids: list[int],
        camera_sidecar: CameraSidecarIndex | None = None,
        intrinsics_fallback: str = "identity",
        extrinsics_convention: str = "world_from_camera",
        vae_temporal_stride: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return camera matrices aligned with selected latent frames.

        Shapes are ``viewmats=(F,V,4,4)`` and ``Ks=(F,V,3,3)``. Current causal
        training is single-view, but keeping V in the reader avoids baking in
        that assumption at the data boundary.
        """
        ids = [int(x) for x in latent_ids]
        if not ids:
            raise ValueError("latent_ids must be non-empty")
        views = list(entry["views"])
        if len(views) != self.num_cameras:
            raise ValueError(
                f"{entry['episode_uid']} has {len(views)} view entries, "
                f"expected {self.num_cameras}"
            )
        if camera_sidecar is not None:
            camera_keys = [
                str(view.get("source_key") or view.get("name") or view.get("view_id"))
                for view in views
            ]
            return camera_sidecar.lookup_sequence(
                split=str(entry["split"]),
                episode_id=int(entry["episode_id"]),
                camera_keys=camera_keys,
                num_frames=len(ids),
            )
        num_video_frames = int(entry["time"]["num_video_frames"])
        frame_ids = [
            min(max(0, latent_id * int(vae_temporal_stride)), num_video_frames - 1)
            for latent_id in ids
        ]
        viewmats_by_view: list[np.ndarray] = []
        ks_by_view: list[np.ndarray] = []
        for view in views:
            source = view.get("extrinsics_source")
            static_camera_from_world = _camera_from_world_from_view(view)
            if static_camera_from_world is not None:
                selected_vm = np.broadcast_to(
                    static_camera_from_world,
                    (len(ids), 4, 4),
                ).astype(np.float32, copy=True)
            elif source:
                pose = self.load_signal_field(entry, str(source))
                mats = _pose6_to_viewmats(pose, convention=extrinsics_convention)
                selected_vm = mats[np.asarray(frame_ids, dtype=np.int64)]
            else:
                raise KeyError(
                    f"{entry['episode_uid']} view {view.get('name', view.get('view_id'))!r} "
                    "has neither extrinsics_source nor static view.extrinsics"
                )
            K = _intrinsics_from_view(view, fallback=intrinsics_fallback)
            selected_k = np.broadcast_to(K, (len(ids), 3, 3)).astype(np.float32, copy=True)
            viewmats_by_view.append(selected_vm.astype(np.float32, copy=False))
            ks_by_view.append(selected_k)
        viewmats = np.stack(viewmats_by_view, axis=1)
        Ks = np.stack(ks_by_view, axis=1)
        if not np.isfinite(viewmats).all() or not np.isfinite(Ks).all():
            raise ValueError(f"{entry['episode_uid']} camera matrices contain non-finite values")
        return np.ascontiguousarray(viewmats), np.ascontiguousarray(Ks)
