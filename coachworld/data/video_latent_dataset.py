"""Training dataset for CoachWorld ``video_latent`` shard roots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from coachworld.data.camera_sidecar import CameraSidecarIndex
from coachworld.data.eef_projection_sidecar import (
    EefProjectionSidecarIndex,
    parse_heatmap_size,
)
from coachworld.data.eef_projection import (
    project_arm_slot_condition,
    select_condition_at_latent_frames,
)
from coachworld.data.video_latent_reader import VideoLatentRootReader
from coachworld.world_model.action_contract import (
    action_start_frame_for_first_future_latent,
    latent_window_indices_for_first_future_frame,
    num_video_frames_for_latent_window,
)

logger = logging.getLogger(__name__)

VAE_TEMPORAL_STRIDE = 4
DEFAULT_CONDITION_VIEW = "arm_slot_eef_pose"


class VideoLatentTrainingDataset(Dataset):
    """Dataset backed by ``video_latent`` episode shards.

    It returns the current CoachWorld training batch contract: ``latent`` as
    ``(T,C,H,W)`` height-stacked Wan latents, ``action`` as normalized action
    conditioning, and ``text`` as task instruction.
    """

    def __init__(
        self,
        dataset_roots: list[str | Path],
        mode: str = "train",
        num_history: int = 4,
        num_frames: int = 3,
        num_cameras: int = 3,
        dataset_probs: Optional[list[float]] = None,
        condition_view: str = DEFAULT_CONDITION_VIEW,
        sample_stride_video_frames: int = 8,
        action_dim: int = 10,
        latent_height_per_view: int = 12,
        latent_width: int = 20,
        expected_video_keys: Optional[list[str]] = None,
        require_production_contract: bool = False,
        required_target_fps: float = 0.0,
        required_target_image_hw: Optional[list[int] | tuple[int, int]] = None,
        history_selector: str = "recent",
        history_offsets: Optional[list[int]] = None,
        history_dilations: Optional[list[int]] = None,
        history_collapse_prob: float = 0.0,
        history_eval_dilation: int = 2,
        generated_history_enabled: bool = False,
        generated_history_unroll_chunks: Optional[list[int]] = None,
        camera_conditioning: bool = False,
        camera_sidecar_roots: Optional[list[str | Path] | str | Path] = None,
        camera_intrinsics_fallback: str = "identity",
        camera_extrinsics_convention: str = "world_from_camera",
        eef_projection_roots: Optional[list[str | Path] | str | Path] = None,
        eef_projection_mode: str = "sidecar",
        eef_projection_load_heatmap: bool = False,
        eef_projection_heatmap_size: Optional[str | tuple[int, int]] = None,
    ) -> None:
        self.mode = str(mode)
        self.num_history = int(num_history)
        self.num_frames = int(num_frames)
        self.T_total = self.num_history + self.num_frames
        self.num_cameras = int(num_cameras)
        self.condition_view = str(condition_view)
        self.sample_stride_video_frames = int(sample_stride_video_frames)
        self.action_dim = int(action_dim)
        self.latent_height_per_view = int(latent_height_per_view)
        self.latent_width = int(latent_width)
        self.expected_video_keys = [str(x) for x in (expected_video_keys or [])]
        self.history_selector = str(history_selector)
        self.history_offsets = [int(x) for x in (history_offsets or [])]
        self.history_dilations = [int(x) for x in (history_dilations or [1, 2])]
        self.history_collapse_prob = float(history_collapse_prob)
        self.history_eval_dilation = int(history_eval_dilation)
        self.generated_history_enabled = bool(generated_history_enabled) and self.mode == "train"
        self.generated_history_unroll_chunks = sorted(
            {
                int(value)
                for value in (generated_history_unroll_chunks or [1])
            }
        )
        if any(value <= 0 for value in self.generated_history_unroll_chunks):
            raise ValueError(
                "generated_history_unroll_chunks must contain only positive integers, "
                f"got {self.generated_history_unroll_chunks}"
            )
        self.generated_history_max_unroll_chunks = max(
            self.generated_history_unroll_chunks
        )
        self.camera_conditioning = bool(camera_conditioning)
        self.camera_intrinsics_fallback = str(camera_intrinsics_fallback)
        self.camera_extrinsics_convention = str(camera_extrinsics_convention)
        self.eef_projection_mode = str(eef_projection_mode).strip().lower()
        self.eef_projection_load_heatmap = bool(eef_projection_load_heatmap)
        self.eef_projection_heatmap_size = parse_heatmap_size(eef_projection_heatmap_size)
        if self.sample_stride_video_frames <= 0:
            raise ValueError("sample_stride_video_frames must be positive")
        if self.history_selector not in {
            "recent",
            "sparse",
            "first_recent",
            "first_offset_recent",
        }:
            raise ValueError(
                f"unsupported history_selector={self.history_selector!r}; "
                "expected 'recent', 'sparse', 'first_recent', or 'first_offset_recent'"
            )
        if self.history_selector == "first_offset_recent" and len(self.history_offsets) != self.num_history - 1:
            raise ValueError(
                "history_selector='first_offset_recent' requires "
                f"num_history-1 offsets; got num_history={self.num_history}, "
                f"history_offsets={self.history_offsets}"
            )
        if self.history_selector == "sparse":
            if not self.history_dilations or any(x <= 0 for x in self.history_dilations):
                raise ValueError(
                    f"sparse history needs positive history_dilations, got {self.history_dilations}"
                )
            if not 0.0 <= self.history_collapse_prob <= 1.0:
                raise ValueError(
                    "history_collapse_prob must be in [0,1], got "
                    f"{self.history_collapse_prob}"
                )
            if self.history_eval_dilation <= 0:
                raise ValueError(
                    f"history_eval_dilation must be positive, got {self.history_eval_dilation}"
                )
        if self.num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {self.num_cameras}")
        if self.latent_height_per_view <= 0 or self.latent_width <= 0:
            raise ValueError(
                "latent geometry must be positive, got "
                f"latent_height_per_view={self.latent_height_per_view}, "
                f"latent_width={self.latent_width}"
            )
        if self.eef_projection_mode not in {"sidecar", "online"}:
            raise ValueError(
                "eef_projection_mode must be 'sidecar' or 'online', got "
                f"{self.eef_projection_mode!r}"
            )
        if self.eef_projection_mode == "online" and not self.camera_conditioning:
            raise ValueError("online EEF projection requires camera_conditioning=true")

        self.dataset_roots = [Path(p).expanduser().resolve() for p in dataset_roots]
        if not self.dataset_roots:
            raise ValueError("VideoLatentTrainingDataset needs at least one dataset root")
        self.camera_sidecar_indices = self._build_camera_sidecar_indices(camera_sidecar_roots)
        self.eef_projection_indices = self._build_eef_projection_indices(eef_projection_roots)
        self.dataset_probs = dataset_probs or [1.0 / len(self.dataset_roots)] * len(self.dataset_roots)
        if len(self.dataset_probs) != len(self.dataset_roots):
            raise ValueError(
                f"dataset_probs length {len(self.dataset_probs)} != dataset_roots length {len(self.dataset_roots)}"
            )
        prob_sum = sum(float(p) for p in self.dataset_probs)
        if prob_sum <= 0:
            raise ValueError("dataset_probs sum must be positive")
        self.dataset_probs = [float(p) / prob_sum for p in self.dataset_probs]

        self.manifests: list[dict[str, Any]] = []
        self.dataset_keys: list[str] = []
        self.entries_all: list[list[dict[str, Any]]] = []
        self.readers: list[VideoLatentRootReader] = []
        self.samples_all: list[list[tuple[int, int]]] = []

        total_samples = 0
        for ds_idx, root in enumerate(self.dataset_roots):
            reader = VideoLatentRootReader(
                root,
                split=self.mode,
                num_cameras=self.num_cameras,
                condition_view=self.condition_view,
                action_dim=self.action_dim,
                latent_height_per_view=self.latent_height_per_view,
                latent_width=self.latent_width,
                expected_video_keys=self.expected_video_keys,
                require_production_contract=require_production_contract,
                required_target_fps=required_target_fps,
                required_target_image_hw=required_target_image_hw,
            )
            self.readers.append(reader)
            self.manifests.append(reader.manifest)
            production_contract = reader.manifest.get("production_contract", {})
            self.dataset_keys.append(
                str(production_contract.get("dataset_key") or root.name)
            )
            entries = reader.entries
            self.entries_all.append(entries)

            samples: list[tuple[int, int]] = []
            for entry_idx, entry in enumerate(entries):
                num_video_frames = int(entry["time"]["num_video_frames"])
                num_latent_frames = int(entry["time"]["num_latent_frames"])
                # Keep every window that supports at least one generated chunk
                # followed by one supervised target chunk. Longer chains are
                # built per sample, so short episodes are not removed merely
                # because the configured maximum unroll is large.
                required_future = self.num_frames * (
                    2 if self.generated_history_enabled else 1
                )
                if num_latent_frames < self.num_history + required_future:
                    continue
                max_first_future_frame = max(
                    0,
                    (num_latent_frames - required_future) * VAE_TEMPORAL_STRIDE,
                )
                min_first_future_frame = (
                    VAE_TEMPORAL_STRIDE if self.history_selector == "sparse" else 0
                )
                for first_future_frame in range(
                    min_first_future_frame,
                    max_first_future_frame + 1,
                    self.sample_stride_video_frames,
                ):
                    if first_future_frame < num_video_frames:
                        samples.append((entry_idx, int(first_future_frame)))
            if not samples:
                raise ValueError(f"{root} split={self.mode} produced zero training windows")
            self.samples_all.append(samples)
            total_samples += len(samples)

        # Integer indexing remains a natural, lossless view of every root.
        # Training mixture weights are implemented by
        # ResumableDistributedMixtureSampler; they must not permanently prune
        # low-probability roots here.
        self._flat_index: list[tuple[int, int]] = []
        for ds_idx, samples in enumerate(self.samples_all):
            self._flat_index.extend(
                (ds_idx, sample_idx) for sample_idx in range(len(samples))
            )
        if not self._flat_index:
            raise ValueError(
                "dataset_probs produced zero indexed samples; at least one dataset must have positive probability"
            )

        logger.info(
            "VideoLatentTrainingDataset: %d roots, %d episode entries, %d windows, %d natural indexed windows, mode=%s",
            len(self.dataset_roots),
            sum(len(x) for x in self.entries_all),
            total_samples,
            len(self._flat_index),
            self.mode,
        )

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(f"source index tuple must have length 2, got {index!r}")
            return self.get_source_sample(int(index[0]), int(index[1]))
        ds_idx, sample_idx = self._flat_index[index % len(self._flat_index)]
        return self.get_source_sample(ds_idx, sample_idx)

    def _history_plan(
        self,
        *,
        dilation: int | None = None,
        collapsed: bool | None = None,
    ) -> tuple[int, bool]:
        if self.history_selector != "sparse":
            return 1, False
        if dilation is None:
            if self.mode == "train":
                dilation = int(np.random.choice(self.history_dilations))
            else:
                dilation = int(self.history_eval_dilation)
        if collapsed is None:
            collapsed = bool(
                self.mode == "train"
                and np.random.random() < self.history_collapse_prob
            )
        return int(dilation), bool(collapsed)

    def get_source_sample(
        self,
        dataset_index: int,
        sample_index: int,
        *,
        history_dilation: int | None = None,
        history_collapsed: bool | None = None,
    ) -> dict[str, Any]:
        """Load a sample by its root-local index.

        The distributed mixture sampler and validation code address these
        root-local windows directly, so sampling weights never remove source
        windows from the dataset.
        """
        ds_idx = int(dataset_index)
        sample_idx = int(sample_index)
        if ds_idx < 0 or ds_idx >= len(self.samples_all):
            raise IndexError(f"dataset_index={ds_idx} outside [0,{len(self.samples_all)})")
        if sample_idx < 0 or sample_idx >= len(self.samples_all[ds_idx]):
            raise IndexError(
                f"sample_index={sample_idx} outside [0,{len(self.samples_all[ds_idx])}) "
                f"for dataset_index={ds_idx}"
            )
        entry_idx, first_future_frame = self.samples_all[ds_idx][sample_idx]
        sample = self.get_episode_window(
            ds_idx,
            entry_idx,
            first_future_frame=int(first_future_frame),
            history_dilation=history_dilation,
            history_collapsed=history_collapsed,
        )
        if self.generated_history_enabled:
            entry = self.entries_all[ds_idx][entry_idx]
            num_latent_frames = int(entry["time"]["num_latent_frames"])
            chain = []
            for chunk_index in range(1, self.generated_history_max_unroll_chunks + 1):
                next_first_future_frame = (
                    int(first_future_frame)
                    + chunk_index * self.num_frames * VAE_TEMPORAL_STRIDE
                )
                next_first_future_latent = (
                    next_first_future_frame // VAE_TEMPORAL_STRIDE
                )
                if next_first_future_latent + self.num_frames > num_latent_frames:
                    break
                chain.append(
                    self.get_episode_window(
                        ds_idx,
                        entry_idx,
                        first_future_frame=next_first_future_frame,
                        history_dilation=int(sample["history_dilation"]),
                        history_collapsed=False,
                    )
                )
            if not chain:
                raise RuntimeError(
                    "generated-history sample has no following supervised window: "
                    f"dataset={self.dataset_keys[ds_idx]} entry={entry_idx} "
                    f"first_future_frame={first_future_frame}"
                )
            sample["generated_history_chain"] = chain
            sample["generated_history_max_unroll_chunks"] = (
                self.generated_history_max_unroll_chunks
            )
        return sample

    def get_episode_window(
        self,
        dataset_index: int,
        episode_entry_index: int,
        *,
        first_future_frame: int,
        history_dilation: int | None = None,
        history_collapsed: bool | None = None,
        allow_future_padding: bool = False,
    ) -> dict[str, Any]:
        """Load an arbitrary episode window for open- or closed-loop evaluation."""
        ds_idx = int(dataset_index)
        entry_idx = int(episode_entry_index)
        if ds_idx < 0 or ds_idx >= len(self.entries_all):
            raise IndexError(f"dataset_index={ds_idx} outside [0,{len(self.entries_all)})")
        if entry_idx < 0 or entry_idx >= len(self.entries_all[ds_idx]):
            raise IndexError(
                f"episode_entry_index={entry_idx} outside "
                f"[0,{len(self.entries_all[ds_idx])}) for dataset_index={ds_idx}"
            )
        entry = self.entries_all[ds_idx][entry_idx]
        dilation, collapsed = self._history_plan(
            dilation=history_dilation,
            collapsed=history_collapsed,
        )
        latent_ids = latent_window_indices_for_first_future_frame(
            first_future_frame=first_future_frame,
            video_length=int(entry["time"]["num_video_frames"]),
            history_frames=self.num_history,
            future_frames=self.num_frames,
            vae_temporal_stride=VAE_TEMPORAL_STRIDE,
            history_selector=self.history_selector,
            history_offsets=self.history_offsets,
            history_dilation=dilation,
            history_collapse=collapsed,
            allow_future_padding=bool(allow_future_padding),
        )
        first_future_latent = int(latent_ids[self.num_history])
        state_start = action_start_frame_for_first_future_latent(
            first_future_latent=first_future_latent,
            history_frames=self.num_history,
            vae_temporal_stride=VAE_TEMPORAL_STRIDE,
        )
        state_end = state_start + num_video_frames_for_latent_window(
            len(latent_ids),
            VAE_TEMPORAL_STRIDE,
        )
        reader = self.readers[ds_idx]
        latent = reader.load_stacked_latents(entry, latent_ids).permute(1, 0, 2, 3).contiguous()
        action = reader.load_condition_window(
            entry,
            state_start=state_start,
            needed_frames=state_end - state_start,
        )
        text = str(entry.get("text", {}).get("primary", "") or "")
        sample = {
            "latent": latent.float(),
            "action": torch.from_numpy(action).float(),
            "text": text,
            "dataset_index": int(ds_idx),
            "dataset_key": self.dataset_keys[ds_idx],
            "domain_id": int(entry["domain"]["domain_id"]),
            "embodiment_id": int(entry["domain"]["embodiment_id"]),
            "camera_setup_id": int(entry["domain"]["camera_setup_id"]),
            "history_latent_ids": torch.tensor(
                latent_ids[: self.num_history], dtype=torch.long
            ),
            "future_latent_ids": torch.tensor(
                latent_ids[self.num_history :], dtype=torch.long
            ),
            "history_dilation": int(dilation),
            "history_collapsed": bool(collapsed),
            "first_future_frame": int(first_future_frame),
            "episode_entry_index": int(entry_idx),
        }
        if self.camera_conditioning:
            camera_sidecar = self.camera_sidecar_indices[ds_idx]
            viewmats, Ks = reader.load_camera_window(
                entry,
                latent_ids=latent_ids,
                camera_sidecar=camera_sidecar,
                intrinsics_fallback=self.camera_intrinsics_fallback,
                extrinsics_convention=self.camera_extrinsics_convention,
                vae_temporal_stride=VAE_TEMPORAL_STRIDE,
            )
            sample["viewmats"] = torch.from_numpy(viewmats).float()
            sample["Ks"] = torch.from_numpy(Ks).float()
        eef_projection_index = self.eef_projection_indices[ds_idx]
        if self.eef_projection_mode == "online":
            if not self.camera_conditioning:
                raise RuntimeError("online EEF projection requires loaded camera matrices")
            raw_condition = reader.load_raw_condition_array(entry)
            condition_at_latents = select_condition_at_latent_frames(
                raw_condition,
                latent_ids,
                vae_temporal_stride=VAE_TEMPORAL_STRIDE,
            )
            target_image_hw = tuple(
                int(x) for x in reader.manifest["processing"]["target_size"]
            )
            eef_projection = project_arm_slot_condition(
                condition_at_latents,
                viewmats,
                Ks,
                image_hw=target_image_hw,
            )
            for key, value in eef_projection.items():
                tensor = torch.from_numpy(value)
                if key == "eef_valid":
                    sample[key] = tensor.bool()
                elif key == "eef_image_hw":
                    sample[key] = tensor.long()
                else:
                    sample[key] = tensor.float()
        elif eef_projection_index is not None:
            eef_projection = self._load_eef_projection_window(
                reader=reader,
                entry=entry,
                latent_ids=latent_ids,
                index=eef_projection_index,
            )
            for key, value in eef_projection.items():
                tensor = torch.from_numpy(value)
                if key == "eef_valid":
                    sample[key] = tensor.bool()
                elif key == "eef_image_hw":
                    sample[key] = tensor.long()
                else:
                    sample[key] = tensor.float()
        return sample

    def describe_source_sample(self, dataset_index: int, sample_index: int) -> dict[str, Any]:
        """Return JSON-safe provenance for a root-local validation window."""
        ds_idx = int(dataset_index)
        sample_idx = int(sample_index)
        entry_idx, first_future_frame = self.samples_all[ds_idx][sample_idx]
        entry = self.entries_all[ds_idx][entry_idx]
        dilation, collapsed = self._history_plan()
        latent_ids = latent_window_indices_for_first_future_frame(
            first_future_frame=int(first_future_frame),
            video_length=int(entry["time"]["num_video_frames"]),
            history_frames=self.num_history,
            future_frames=self.num_frames,
            vae_temporal_stride=VAE_TEMPORAL_STRIDE,
            history_selector=self.history_selector,
            history_offsets=self.history_offsets,
            history_dilation=dilation,
            history_collapse=collapsed,
        )
        return {
            "dataset_index": ds_idx,
            "dataset_root": str(self.dataset_roots[ds_idx]),
            "dataset_key": self.dataset_roots[ds_idx].name,
            "source_sample_index": sample_idx,
            "episode_entry_index": int(entry_idx),
            "episode_id": int(entry["episode_id"]),
            "episode_uid": str(entry.get("episode_uid", entry["episode_id"])),
            "num_video_frames": int(entry["time"]["num_video_frames"]),
            "num_latent_frames": int(entry["time"]["num_latent_frames"]),
            "first_future_frame": int(first_future_frame),
            "text": str(entry.get("text", {}).get("primary", "") or ""),
            "domain_id": int(entry["domain"]["domain_id"]),
            "embodiment_id": int(entry["domain"]["embodiment_id"]),
            "camera_setup_id": int(entry["domain"]["camera_setup_id"]),
            "history_selector": self.history_selector,
            "history_dilation": int(dilation),
            "history_collapsed": bool(collapsed),
            "history_latent_ids": [int(x) for x in latent_ids[: self.num_history]],
            "future_latent_ids": [int(x) for x in latent_ids[self.num_history :]],
        }

    def root_balanced_validation_refs(
        self,
        num_samples: int,
        *,
        seed: int,
        future_latents_required: int | None = None,
        window_fraction: float | None = None,
    ) -> list[tuple[int, int]]:
        """Select fixed validation windows across every dataset root.

        Episodes are selected deterministically from a seeded permutation and
        each selected episode contributes its middle available window. This
        avoids static episode boundaries while keeping the set independent of
        probability-expanded training/validation indexes.
        """
        requested = int(num_samples)
        if requested <= 0:
            return []
        num_roots = len(self.dataset_roots)
        if requested < num_roots:
            raise ValueError(
                f"root-balanced validation needs at least {num_roots} samples, got {requested}"
            )

        base, remainder = divmod(requested, num_roots)
        refs: list[tuple[int, int]] = []
        for ds_idx, samples in enumerate(self.samples_all):
            target = base + (1 if ds_idx < remainder else 0)
            by_entry: dict[int, list[int]] = {}
            for sample_idx, (entry_idx, first_future_frame) in enumerate(samples):
                if future_latents_required is not None:
                    first_future_latent = int(first_future_frame) // VAE_TEMPORAL_STRIDE
                    num_latents = int(
                        self.entries_all[ds_idx][int(entry_idx)]["time"]["num_latent_frames"]
                    )
                    if first_future_latent + int(future_latents_required) > num_latents:
                        continue
                by_entry.setdefault(int(entry_idx), []).append(int(sample_idx))
            entry_ids = sorted(by_entry)
            if not entry_ids:
                raise ValueError(f"dataset root {self.dataset_roots[ds_idx]} has no validation windows")

            rng = np.random.default_rng(int(seed) + ds_idx * 1009)
            order = rng.permutation(len(entry_ids)).tolist()
            selected_entries = [entry_ids[order[i % len(order)]] for i in range(target)]
            for repeat_idx, entry_idx in enumerate(selected_entries):
                candidates = by_entry[entry_idx]
                # Prefer a central window. If a tiny validation root must reuse
                # an episode, spread repeats around the midpoint deterministically.
                if len(candidates) == 1:
                    chosen = candidates[0]
                elif window_fraction is not None:
                    frac = min(max(float(window_fraction), 0.0), 1.0)
                    chosen = candidates[
                        min(
                            round(frac * (len(candidates) - 1)),
                            len(candidates) - 1,
                        )
                    ]
                else:
                    offsets = np.linspace(0.35, 0.65, max(1, target))
                    frac = float(offsets[min(repeat_idx, len(offsets) - 1)])
                    chosen = candidates[min(round(frac * (len(candidates) - 1)), len(candidates) - 1)]
                refs.append((ds_idx, int(chosen)))
        return refs

    def _build_camera_sidecar_indices(
        self,
        roots: Optional[list[str | Path] | str | Path],
    ) -> list[CameraSidecarIndex | None]:
        if roots is None or roots == "":
            return [None] * len(self.dataset_roots)
        if isinstance(roots, (str, Path)):
            root_values = [roots]
        else:
            root_values = list(roots)
        if not any(str(x) for x in root_values):
            return [None] * len(self.dataset_roots)
        if len(root_values) == 1:
            indices = [CameraSidecarIndex(root_values[0]) for _ in self.dataset_roots]
        elif len(root_values) == len(self.dataset_roots):
            indices = [CameraSidecarIndex(root) if str(root) else None for root in root_values]
        else:
            raise ValueError(
                "camera_sidecar_roots must be empty, length 1, or match dataset_roots; "
                f"got {len(root_values)} roots for {len(self.dataset_roots)} dataset roots"
            )
        for root, index in zip(self.dataset_roots, indices):
            if index is None:
                continue
            logger.info(
                "VideoLatentTrainingDataset: camera sidecar for %s has %d valid / %d records",
                root,
                index.num_valid_records,
                index.num_records,
            )
        return indices

    def _build_eef_projection_indices(
        self,
        roots: Optional[list[str | Path] | str | Path],
    ) -> list[EefProjectionSidecarIndex | None]:
        if self.eef_projection_mode == "online":
            if roots not in (None, "", []):
                raise ValueError(
                    "online EEF projection is computed from embedded K/T and cannot "
                    "also use eef_projection_roots"
                )
            return [None] * len(self.dataset_roots)
        if roots is None or roots == "":
            return [None] * len(self.dataset_roots)
        if isinstance(roots, (str, Path)):
            root_values = [roots]
        else:
            root_values = list(roots)
        if not any(str(x) for x in root_values):
            return [None] * len(self.dataset_roots)
        if len(root_values) == 1:
            indices = [EefProjectionSidecarIndex(root_values[0]) for _ in self.dataset_roots]
        elif len(root_values) == len(self.dataset_roots):
            indices = [EefProjectionSidecarIndex(root) if str(root) else None for root in root_values]
        else:
            raise ValueError(
                "eef_projection_roots must be empty, length 1, or match dataset_roots; "
                f"got {len(root_values)} roots for {len(self.dataset_roots)} dataset roots"
            )
        for root, index in zip(self.dataset_roots, indices):
            if index is None:
                continue
            logger.info(
                "VideoLatentTrainingDataset: EEF projection sidecar for %s has %d records across %d episodes",
                root,
                index.num_records,
                len(index.episode_ids),
            )
        global_slots = max((index.num_slots for index in indices if index is not None), default=1)
        for index in indices:
            if index is not None:
                index.set_output_num_slots(global_slots)
        return indices

    def _raw_frame_indices_for_latents(
        self,
        reader: VideoLatentRootReader,
        entry: dict[str, Any],
        latent_ids: list[int],
    ) -> list[int]:
        num_video_frames = int(entry["time"]["num_video_frames"])
        video_frame_ids = [
            min(max(0, int(latent_id) * VAE_TEMPORAL_STRIDE), num_video_frames - 1)
            for latent_id in latent_ids
        ]
        try:
            frame_index = reader.load_signal_field(entry, "frame_index").reshape(-1)
        except KeyError:
            return video_frame_ids
        return [int(round(float(frame_index[idx]))) for idx in video_frame_ids]

    def _load_eef_projection_window(
        self,
        *,
        reader: VideoLatentRootReader,
        entry: dict[str, Any],
        latent_ids: list[int],
        index: EefProjectionSidecarIndex,
    ) -> dict[str, np.ndarray]:
        views = list(entry["views"])
        camera_keys = [
            str(view.get("source_key") or view.get("name") or view.get("view_id"))
            for view in views
        ]
        raw_frame_indices = self._raw_frame_indices_for_latents(reader, entry, latent_ids)
        return index.lookup_sequence(
            episode_id=int(entry["episode_id"]),
            camera_keys=camera_keys,
            raw_frame_indices=raw_frame_indices,
            load_heatmap=self.eef_projection_load_heatmap,
            heatmap_size=self.eef_projection_heatmap_size,
        )
