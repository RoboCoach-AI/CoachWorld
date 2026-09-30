"""Batch collation for canonical video-latent samples."""

from __future__ import annotations

from typing import Any

import torch


def _has_consistent_key(batch: list[dict[str, Any]], key: str) -> bool:
    present = [key in item for item in batch]
    if any(present) and not all(present):
        raise RuntimeError(f"{key} must be present for every item in a batch")
    return bool(present[0])


def _pad_actions(actions: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(action.shape[0] for action in actions)
    trailing_shape = tuple(actions[0].shape[1:])
    for index, action in enumerate(actions):
        if tuple(action.shape[1:]) != trailing_shape:
            raise RuntimeError(
                "Action trailing shape mismatch inside batch: "
                f"item0={trailing_shape}, item{index}={tuple(action.shape[1:])}"
            )
    padded = actions[0].new_zeros((len(actions), max_len, *trailing_shape))
    for index, action in enumerate(actions):
        padded[index, : action.shape[0]] = action
    return padded


def collate_video_latent_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate video-latent windows and optional generated-history chains."""
    if not batch:
        raise ValueError("cannot collate an empty video-latent batch")

    out: dict[str, Any] = {
        "latent": torch.stack([item["latent"] for item in batch]),
        "action": _pad_actions([item["action"] for item in batch]),
        "text": [item["text"] for item in batch],
        "dataset_key": [str(item["dataset_key"]) for item in batch],
        "dataset_index": torch.tensor(
            [int(item["dataset_index"]) for item in batch],
            dtype=torch.long,
        ),
        "domain_id": torch.tensor(
            [int(item["domain_id"]) for item in batch],
            dtype=torch.long,
        ),
        "embodiment_id": torch.tensor(
            [int(item["embodiment_id"]) for item in batch],
            dtype=torch.long,
        ),
        "camera_setup_id": torch.tensor(
            [int(item["camera_setup_id"]) for item in batch],
            dtype=torch.long,
        ),
        "history_latent_ids": torch.stack(
            [item["history_latent_ids"] for item in batch]
        ),
        "future_latent_ids": torch.stack(
            [item["future_latent_ids"] for item in batch]
        ),
    }

    if _has_consistent_key(batch, "viewmats"):
        if not _has_consistent_key(batch, "Ks"):
            raise RuntimeError("viewmats requires Ks in every batch item")
        out["viewmats"] = torch.stack([item["viewmats"] for item in batch])
        out["Ks"] = torch.stack([item["Ks"] for item in batch])

    if _has_consistent_key(batch, "eef_uv"):
        for key in ("eef_depth", "eef_valid", "eef_image_hw"):
            if not _has_consistent_key(batch, key):
                raise RuntimeError(f"eef_uv requires {key} in every batch item")
            out[key] = torch.stack([item[key] for item in batch])
        out["eef_uv"] = torch.stack([item["eef_uv"] for item in batch])
        for key in ("eef_gripper", "eef_heatmap"):
            if _has_consistent_key(batch, key):
                out[key] = torch.stack([item[key] for item in batch])

    if _has_consistent_key(batch, "generated_history_chain"):
        available = [len(item["generated_history_chain"]) for item in batch]
        configured_depths = [
            int(item["generated_history_max_unroll_chunks"]) for item in batch
        ]
        if len(set(configured_depths)) != 1:
            raise RuntimeError(
                "generated_history_max_unroll_chunks differs inside one batch: "
                f"{configured_depths}"
            )
        chain_depth = configured_depths[0]
        if chain_depth <= 0 or min(available) <= 0:
            raise RuntimeError(
                "generated_history_chain must have positive configured/available "
                f"depths: configured={chain_depth}, available={available}"
            )

        out["generated_history_chain"] = []
        for step in range(chain_depth):
            target = collate_video_latent_batch(
                [
                    item["generated_history_chain"][
                        min(step, len(item["generated_history_chain"]) - 1)
                    ]
                    for item in batch
                ]
            )
            target["_generated_history_chain_active_mask"] = torch.tensor(
                [step < len(item["generated_history_chain"]) for item in batch],
                dtype=torch.bool,
            )
            out["generated_history_chain"].append(target)
        out["generated_history_available_chunks"] = torch.tensor(
            available,
            dtype=torch.long,
        )

    return out
