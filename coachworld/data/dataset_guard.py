"""Hard guards for DROID Wan latent datasets.

These checks catch the known legacy DROID latent root where RGB frames were
downsampled but action/state arrays were not. That root can still produce
correct-looking tensor shapes, so shape checks alone are not sufficient.
"""

from __future__ import annotations

import json
import logging
import math
import os
from itertools import islice
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

LEGACY_DROID_LATENT_BASENAME = "droid_latents"
SAFE_DROID_LATENT_BASENAME = "droid_latents_192x320"

COMMON_SEQUENCE_FIELDS = (
    "observation.state.cartesian_position",
    "observation.state.gripper_position",
    "action.cartesian_velocity",
    "action.gripper_position",
)

ACTION_SOURCE_FIELDS = {
    "commanded_action": ("action",),
    "commanded_original_action": ("action.original",),
    "commanded_joint_position_gripper": (
        "action.joint_position",
        "action.gripper_position",
    ),
    "commanded_joint_velocity_gripper": (
        "action.joint_velocity",
        "action.gripper_position",
    ),
    "commanded_cartesian_velocity_gripper": (
        "action.cartesian_velocity",
        "action.gripper_position",
    ),
    "abs_state_plus_cartesian_velocity_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_velocity",
        "action.gripper_position",
    ),
    "abs_state_quat_cartesian_velocity_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_velocity",
        "action.gripper_position",
    ),
    "causal_state_plus_cartesian_velocity_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_velocity",
        "action.gripper_position",
    ),
    "causal_state_quat_cartesian_velocity_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_velocity",
        "action.gripper_position",
    ),
    "abs_plus_delta_plus_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_position",
        "action.gripper_position",
    ),
    "abs_plus_chunk_delta_plus_gripper": (
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_position",
        "action.gripper_position",
    ),
}


def _normalized_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve(strict=False)


def _looks_like_droid_wan_root(path: Path) -> bool:
    parts = set(path.parts)
    return "droid_wan_raw" in parts or path.name.startswith("droid_latents")


def is_legacy_droid_latent_root(path: str | Path) -> bool:
    """Return True for the known legacy DROID latent root, not for suffixed roots."""
    root = _normalized_path(path)
    return root.name == LEGACY_DROID_LATENT_BASENAME and _looks_like_droid_wan_root(root)


def assert_not_legacy_droid_latent_root(path: str | Path, *, context: str = "dataset") -> None:
    """Refuse the old DROID latent root unless explicitly bypassed for archaeology."""
    if not is_legacy_droid_latent_root(path):
        return
    if os.environ.get("COACHWORLD_ALLOW_LEGACY_DROID_LATENTS") == "1":
        logger.warning(
            "COACHWORLD_ALLOW_LEGACY_DROID_LATENTS=1 set; allowing legacy %s root: %s",
            context,
            path,
        )
        return
    raise RuntimeError(
        f"Refusing to use legacy DROID latent {context} root: {path}\n"
        "This root is known to have rgb_skip/action-state timeline mismatch. "
        f"Use the regenerated {SAFE_DROID_LATENT_BASENAME} root instead, or set "
        "COACHWORLD_ALLOW_LEGACY_DROID_LATENTS=1 only for a clearly named historical ablation."
    )


def _sequence_len(value: object) -> int | None:
    if isinstance(value, (str, bytes, dict)):
        return None
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _finite_int(value: object) -> int | None:
    try:
        x = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return x


def validate_droid_annotation_contract(
    annotation: dict,
    *,
    annotation_path: str | Path | None = None,
    action_source: str | None = None,
    required_rgb_skip: int | None = 2,
) -> None:
    """Validate that annotation sequence fields live on the video-frame timeline."""
    where = str(annotation_path) if annotation_path is not None else "<annotation>"
    errors: list[str] = []

    video_length = _finite_int(annotation.get("video_length"))
    if video_length is None or video_length <= 0:
        errors.append(f"video_length is missing or invalid: {annotation.get('video_length')!r}")

    if required_rgb_skip is not None:
        rgb_skip = annotation.get("rgb_skip")
        if rgb_skip != required_rgb_skip:
            errors.append(f"rgb_skip={rgb_skip!r}, expected {required_rgb_skip}")

    fields = set(COMMON_SEQUENCE_FIELDS)
    if action_source:
        fields.update(ACTION_SOURCE_FIELDS.get(action_source, ()))

    if video_length is not None and video_length > 0:
        for field in sorted(fields):
            if field not in annotation:
                errors.append(f"missing field {field!r}")
                continue
            seq_len = _sequence_len(annotation[field])
            if seq_len is None:
                errors.append(f"field {field!r} is not a sequence")
            elif seq_len != video_length:
                errors.append(f"{field} length={seq_len}, video_length={video_length}")

    if errors:
        joined = "\n  - ".join(errors)
        raise RuntimeError(
            f"DROID annotation contract failed for {where}:\n  - {joined}\n"
            "The current 15D mainline requires action/state arrays to be downsampled "
            "onto the same timeline as decoded RGB frames."
        )


def _sample_annotation_paths(
    latent_root: Path,
    *,
    annotation_name: str,
    split: str,
    max_annotations: int,
) -> list[Path]:
    ann_dir = latent_root / annotation_name / split
    if not ann_dir.is_dir():
        return []
    return list(islice(ann_dir.glob("*.json"), max(0, max_annotations)))


def validate_droid_latent_dataset_root(
    latent_root: str | Path,
    *,
    meta_info_root: str | Path | None = None,
    annotation_name: str = "annotation",
    split: str | None = None,
    action_source: str | None = None,
    required_rgb_skip: int | None = 2,
    max_annotation_checks: int = 4,
    context: str = "dataset",
) -> None:
    """Validate path safety and a small annotation sample for a DROID latent root."""
    root = _normalized_path(latent_root)
    assert_not_legacy_droid_latent_root(root, context=context)
    if meta_info_root is not None:
        assert_not_legacy_droid_latent_root(Path(meta_info_root).parent, context="meta_info")

    if not _looks_like_droid_wan_root(root):
        return

    if max_annotation_checks <= 0:
        return

    splits: Iterable[str]
    if split:
        splits = (split,)
    else:
        splits = ("train", "val")

    checked = 0
    for split_name in splits:
        for ann_path in _sample_annotation_paths(
            root,
            annotation_name=annotation_name,
            split=split_name,
            max_annotations=max_annotation_checks,
        ):
            with open(ann_path) as f:
                annotation = json.load(f)
            validate_droid_annotation_contract(
                annotation,
                annotation_path=ann_path,
                action_source=action_source,
                required_rgb_skip=required_rgb_skip,
            )
            checked += 1

    if checked:
        logger.info(
            "Validated DROID annotation contract for %d sample(s) under %s",
            checked,
            root,
        )
