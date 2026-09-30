"""Mask candidate post-processing for robot-camera calibration.

The functions here are intentionally dataset-agnostic.  They turn raw SAM-like
mask predictions into explicit candidates, optionally split multi-object masks
into connected components, and apply slot-specific image-side gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskCandidate:
    mask: np.ndarray
    area: int
    cx: float
    cy: float
    area_frac: float
    bbox: tuple[int, int, int, int]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        payload = {
            "area": int(self.area),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "area_frac": float(self.area_frac),
            "bbox": [int(x) for x in self.bbox],
        }
        payload.update(self.meta)
        return payload


def mask_to_numpy(mask: Any, shape: tuple[int, int]) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    out = (np.asarray(mask).squeeze() > 0).astype(np.uint8)
    h, w = shape
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def candidates_from_raw_masks(
    raw_masks: Any,
    shape: tuple[int, int],
    *,
    min_area: int = 120,
    max_area_frac: float = 0.70,
) -> list[MaskCandidate]:
    h, w = shape
    out: list[MaskCandidate] = []
    for raw_idx, raw in enumerate(raw_masks):
        mask = mask_to_numpy(raw, shape)
        ys, xs = np.nonzero(mask)
        area = int(xs.size)
        if area < int(min_area) or area > int(max_area_frac * h * w):
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        out.append(
            MaskCandidate(
                mask=mask.astype(np.float32),
                area=area,
                cx=float(xs.mean()),
                cy=float(ys.mean()),
                area_frac=area / float(h * w),
                bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                meta={"raw_mask_index": int(raw_idx)},
            )
        )
    out.sort(key=lambda x: x.area, reverse=True)
    return out


def split_connected_components(
    candidates: list[MaskCandidate],
    shape: tuple[int, int],
    *,
    min_area: int = 120,
) -> tuple[list[MaskCandidate], dict[str, Any]]:
    h, w = shape
    split: list[MaskCandidate] = []
    component_counts: list[int] = []
    rejected_small = 0
    for parent_idx, candidate in enumerate(candidates):
        binary = (np.asarray(candidate.mask) > 0).astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        kept_for_parent = 0
        for comp_idx in range(1, n):
            area = int(stats[comp_idx, cv2.CC_STAT_AREA])
            if area < int(min_area):
                rejected_small += 1
                continue
            x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
            y = int(stats[comp_idx, cv2.CC_STAT_TOP])
            bw = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
            cx = float(centroids[comp_idx][0])
            cy = float(centroids[comp_idx][1])
            meta = {
                "parent_candidate_index": int(parent_idx),
                "component_index": int(comp_idx - 1),
                "parent_area_frac": float(candidate.area_frac),
                "parent_cx": float(candidate.cx),
                "parent_cy": float(candidate.cy),
                **candidate.meta,
            }
            split.append(
                MaskCandidate(
                    mask=(labels == comp_idx).astype(np.float32),
                    area=area,
                    cx=cx,
                    cy=cy,
                    area_frac=area / float(h * w),
                    bbox=(x, y, bw, bh),
                    meta=meta,
                )
            )
            kept_for_parent += 1
        component_counts.append(int(kept_for_parent))
    split.sort(key=lambda x: x.area, reverse=True)
    return split, {
        "enabled": True,
        "before_masks": int(len(candidates)),
        "after_components": int(len(split)),
        "component_counts": component_counts,
        "rejected_small_components": int(rejected_small),
        "min_area": int(min_area),
    }


def filter_by_image_side(
    candidates: list[MaskCandidate],
    *,
    slot: int,
    image_width: int,
    margin: float = 0.12,
) -> tuple[list[MaskCandidate], dict[str, Any]]:
    width = max(1.0, float(image_width))
    margin = float(np.clip(margin, 0.0, 0.49))
    if int(slot) == 0:
        keep = [i for i, cand in enumerate(candidates) if cand.cx / width <= 0.5 + margin]
        rule = f"slot0 cx_norm <= {0.5 + margin:.3f}"
    else:
        keep = [i for i, cand in enumerate(candidates) if cand.cx / width >= 0.5 - margin]
        rule = f"slot1 cx_norm >= {0.5 - margin:.3f}"
    keep_set = set(keep)
    return [candidates[i] for i in keep], {
        "enabled": True,
        "rule": rule,
        "slot": int(slot),
        "margin": margin,
        "before": int(len(candidates)),
        "after": int(len(keep)),
        "rejected": [
            {
                "candidate_index": int(i),
                "cx_norm": float(candidates[i].cx / width),
                "area_frac": float(candidates[i].area_frac),
                **candidates[i].meta,
            }
            for i in range(len(candidates))
            if i not in keep_set
        ],
    }


def candidate_arrays_and_meta(candidates: list[MaskCandidate]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    return [cand.mask.astype(np.float32) for cand in candidates], [cand.to_meta() for cand in candidates]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / float(union))


def nms_mask_candidates(candidates: list[MaskCandidate], iou_threshold: float = 0.88) -> list[MaskCandidate]:
    ordered = sorted(candidates, key=lambda x: (x.area, x.meta.get("sam_score", 0.0)), reverse=True)
    kept: list[MaskCandidate] = []
    for cand in ordered:
        if all(mask_iou(cand.mask, prev.mask) < float(iou_threshold) for prev in kept):
            kept.append(cand)
    return kept


def side_stats(mask: np.ndarray, image_width: int, *, center_drop_frac: float = 0.0) -> dict[str, float]:
    binary = np.asarray(mask) > 0
    ys, xs = np.nonzero(binary)
    area = int(xs.size)
    if area <= 0:
        return {
            "left_frac": 0.0,
            "right_frac": 0.0,
            "center_frac": 0.0,
            "cx_norm": 0.0,
        }
    width = max(1.0, float(image_width))
    split = 0.5 * width
    band = float(np.clip(center_drop_frac, 0.0, 0.20)) * width
    left = xs < split - band
    right = xs > split + band
    center = ~(left | right)
    return {
        "left_frac": float(left.sum() / area),
        "right_frac": float(right.sum() / area),
        "center_frac": float(center.sum() / area),
        "cx_norm": float(xs.mean() / width),
    }


def candidate_from_mask(mask: np.ndarray, shape: tuple[int, int], meta: dict[str, Any] | None = None) -> MaskCandidate | None:
    h, w = shape
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.nonzero(binary)
    area = int(xs.size)
    if area <= 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return MaskCandidate(
        mask=binary.astype(np.float32),
        area=area,
        cx=float(xs.mean()),
        cy=float(ys.mean()),
        area_frac=area / float(max(1, h * w)),
        bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
        meta=dict(meta or {}),
    )


def split_candidate_by_image_side(
    candidate: MaskCandidate,
    shape: tuple[int, int],
    *,
    min_area: int = 120,
    center_drop_frac: float = 0.02,
    min_side_frac: float = 0.30,
) -> tuple[MaskCandidate | None, MaskCandidate | None, dict[str, Any]]:
    h, w = shape
    stats = side_stats(candidate.mask, w, center_drop_frac=0.0)
    can_split = (
        stats["left_frac"] >= float(min_side_frac)
        and stats["right_frac"] >= float(min_side_frac)
        and candidate.cx / max(1.0, float(w)) > 0.20
        and candidate.cx / max(1.0, float(w)) < 0.80
    )
    info: dict[str, Any] = {
        "enabled": bool(can_split),
        "source_area": int(candidate.area),
        "source_area_frac": float(candidate.area_frac),
        "source_left_frac": float(stats["left_frac"]),
        "source_right_frac": float(stats["right_frac"]),
        "min_side_frac": float(min_side_frac),
        "center_drop_frac": float(center_drop_frac),
    }
    if not can_split:
        return None, None, info

    binary = np.asarray(candidate.mask) > 0
    _ys, xs_grid = np.mgrid[0:h, 0:w]
    split = 0.5 * float(w)
    band = float(np.clip(center_drop_frac, 0.0, 0.20)) * float(w)
    left_mask = binary & (xs_grid < split - band)
    right_mask = binary & (xs_grid > split + band)
    left = candidate_from_mask(
        left_mask,
        shape,
        {
            **candidate.meta,
            "slot_split_source": "image_side",
            "slot_split_from_area": int(candidate.area),
            "slot_split_side": "left",
        },
    )
    right = candidate_from_mask(
        right_mask,
        shape,
        {
            **candidate.meta,
            "slot_split_source": "image_side",
            "slot_split_from_area": int(candidate.area),
            "slot_split_side": "right",
        },
    )
    if left is not None and left.area < int(min_area):
        left = None
    if right is not None and right.area < int(min_area):
        right = None
    info["left_area"] = 0 if left is None else int(left.area)
    info["right_area"] = 0 if right is None else int(right.area)
    info["kept"] = bool(left is not None and right is not None)
    if left is None or right is None:
        return None, None, info
    return left, right, info


def build_dual_arm_slot_masks(
    raw_masks: Any,
    shape: tuple[int, int],
    *,
    min_area: int = 120,
    max_area_frac: float = 0.70,
    nms_iou: float = 0.88,
    min_side_purity: float = 0.68,
    side_margin: float = 0.12,
    center_drop_frac: float = 0.02,
    merged_min_side_frac: float = 0.30,
) -> dict[str, Any]:
    """Build strict slot0/slot1 masks from one frame of raw SAM masks.

    Ambiguous components are kept only for visualization/debug metadata and are
    intentionally excluded from slot masks.
    """
    h, w = shape
    raw_candidates = candidates_from_raw_masks(
        raw_masks,
        shape,
        min_area=int(min_area),
        max_area_frac=float(max_area_frac),
    )
    nms_candidates = nms_mask_candidates(raw_candidates, float(nms_iou))
    components_raw, split_info = split_connected_components(nms_candidates, shape, min_area=int(min_area))
    components = nms_mask_candidates(components_raw, float(nms_iou))

    slot_components: dict[int, list[MaskCandidate]] = {0: [], 1: []}
    ambiguous: list[dict[str, Any]] = []
    split_attempts: list[dict[str, Any]] = []
    margin = float(np.clip(side_margin, 0.0, 0.49))
    for comp_idx, comp in enumerate(components):
        stats = side_stats(comp.mask, w, center_drop_frac=float(center_drop_frac))
        cx_norm = float(comp.cx / max(1.0, float(w)))
        meta = {
            "component_candidate_index": int(comp_idx),
            **comp.to_meta(),
            **stats,
        }
        if stats["left_frac"] >= float(min_side_purity) and cx_norm <= 0.5 + margin:
            slot_components[0].append(comp)
            continue
        if stats["right_frac"] >= float(min_side_purity) and cx_norm >= 0.5 - margin:
            slot_components[1].append(comp)
            continue

        left, right, split_meta = split_candidate_by_image_side(
            comp,
            shape,
            min_area=int(min_area),
            center_drop_frac=float(center_drop_frac),
            min_side_frac=float(merged_min_side_frac),
        )
        split_meta["component_candidate_index"] = int(comp_idx)
        split_attempts.append(split_meta)
        if left is not None and right is not None:
            slot_components[0].append(left)
            slot_components[1].append(right)
        else:
            ambiguous.append({**meta, "reason": "side_purity_or_split_failed"})

    slot_masks: dict[int, np.ndarray | None] = {}
    for slot, comps in slot_components.items():
        if not comps:
            slot_masks[slot] = None
            continue
        union = np.zeros((h, w), dtype=np.uint8)
        for comp in comps:
            union |= (np.asarray(comp.mask) > 0).astype(np.uint8)
        slot_masks[slot] = union.astype(np.float32)

    return {
        "raw_candidates": raw_candidates,
        "nms_candidates": nms_candidates,
        "components": components,
        "slot_components": slot_components,
        "slot_masks": slot_masks,
        "ambiguous": ambiguous,
        "split_info": split_info,
        "split_attempts": split_attempts,
        "summary": {
            "raw_count": int(len(raw_candidates)),
            "nms_count": int(len(nms_candidates)),
            "component_count": int(len(components)),
            "slot0_component_count": int(len(slot_components[0])),
            "slot1_component_count": int(len(slot_components[1])),
            "ambiguous_count": int(len(ambiguous)),
            "slot0_area_frac": 0.0
            if slot_masks[0] is None
            else float(np.asarray(slot_masks[0] > 0).sum() / float(max(1, h * w))),
            "slot1_area_frac": 0.0
            if slot_masks[1] is None
            else float(np.asarray(slot_masks[1] > 0).sum() / float(max(1, h * w))),
        },
    }


def candidates_from_arrays_and_meta(masks: list[np.ndarray], metas: list[dict[str, Any]]) -> list[MaskCandidate]:
    out: list[MaskCandidate] = []
    for mask, meta in zip(masks, metas, strict=False):
        arr = np.asarray(mask, dtype=np.float32)
        bbox = meta.get("bbox", (0, 0, 0, 0))
        out.append(
            MaskCandidate(
                mask=arr,
                area=int(meta.get("area", np.asarray(arr > 0).sum())),
                cx=float(meta.get("cx", 0.0)),
                cy=float(meta.get("cy", 0.0)),
                area_frac=float(meta.get("area_frac", 0.0)),
                bbox=tuple(int(x) for x in bbox),
                meta={k: v for k, v in meta.items() if k not in {"area", "cx", "cy", "area_frac", "bbox"}},
            )
        )
    return out
