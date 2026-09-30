#!/usr/bin/env python3
"""Train the CoachWorld video-latent world model.

Example::

    torchrun --nproc_per_node=5 scripts/training/train_world_model.py \\
        --config configs/training/coachworld.yaml

Usage (warm-start model weights without optimizer state)::

    torchrun --nproc_per_node=5 scripts/training/train_world_model.py \\
        --config configs/training/coachworld.yaml \\
        --init_from /path/to/checkpoint

Config overrides via dotlist::

    torchrun --nproc_per_node=5 scripts/training/train_world_model.py \\
        --config configs/training/coachworld.yaml \\
        world_model.learning_rate=1e-5 \\
        wm_training.loss_scale=5.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from coachworld.config import load_config
from coachworld.data.video_latent import read_manifest, split_index_path
from coachworld.data.video_latent_dataset import VideoLatentTrainingDataset
from coachworld.world_model.causal_trainer import CausalWanTrainer
from coachworld.world_model.trainer import WMTrainer

logger = logging.getLogger(__name__)


def _video_latent_split_available(
    dataset_roots: list[str],
    *,
    split: str,
    condition_view: str,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for i, root_str in enumerate(dataset_roots):
        root = Path(root_str)
        checks = [
            (root / "manifest.json", "manifest"),
            (split_index_path(root, split), f"{split}.index"),
            (root / "stats" / f"{condition_view}.json", "condition_stats"),
        ]
        for path, label in checks:
            if not path.exists():
                missing.append(f"dataset[{i}] {label}: {path}")
    return not missing, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WAN 2.2 world model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/coachworld.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint (model + optimizer + step)",
    )
    parser.add_argument(
        "--init_from", type=str, default=None,
        help="Warm-start model weights from safetensors (no optimizer, step=0)",
    )
    # Remaining args treated as dotlist overrides
    args, unknown = parser.parse_known_args()
    args.overrides = [a for a in unknown if "=" in a]
    rejected = [a for a in unknown if "=" not in a]
    if rejected:
        parser.error(f"unrecognized arguments: {' '.join(rejected)}")
    return args


def main() -> None:
    args = parse_args()

    # Distributed setup
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        timeout_min = int(os.environ.get("COACHWORLD_DIST_TIMEOUT_MIN", "180"))
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_min))
        torch.cuda.set_device(local_rank)

    # Logging
    log_level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format=f"%(asctime)s [rank {rank}] %(levelname)s %(name)s: %(message)s",
    )

    # Load config
    cfg = load_config(args.config, args.overrides)
    logger.info("Config loaded from %s", args.config)

    # Seed
    seed = cfg.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Build datasets
    tc = cfg.wm_training
    wc = cfg.world_model
    causal_backend = str(getattr(wc, "backbone", "")) == "wan2.2_causal"
    dataset_format = str(getattr(tc, "dataset_format", "video_latent"))
    if dataset_format != "video_latent":
        raise ValueError(
            "scripts/training/train_world_model.py is video_latent-only. "
            f"Got wm_training.dataset_format={dataset_format!r}. "
            "This entry point requires video_latent datasets."
        )

    # Validate critical config before expensive operations
    if int(wc.max_train_steps) <= 0:
        raise ValueError("Set COACHWORLD_MAX_TRAIN_STEPS to a positive value.")
    if not tc.dataset_roots:
        raise ValueError(
            "wm_training.dataset_roots is empty. Set paths to video_latent roots."
        )
    for i, root in enumerate(tc.dataset_roots):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"dataset_roots[{i}] does not exist: {root}")
        read_manifest(Path(root))
    if not causal_backend and not wc.checkpoint:
        raise ValueError("world_model.checkpoint is empty. Set path to WAN 2.2 DiT weights.")

    train_ok, train_missing = _video_latent_split_available(
        tc.dataset_roots,
        split="train",
        condition_view=str(getattr(tc, "video_latent_condition_view")),
    )
    if not train_ok:
        raise FileNotFoundError(
            "Training split is incomplete:\n  - " + "\n  - ".join(train_missing)
        )

    val_ok, val_missing = _video_latent_split_available(
        tc.dataset_roots,
        split="val",
        condition_view=str(getattr(tc, "video_latent_condition_view")),
    )
    require_val = bool(getattr(tc, "require_val_split", False))
    if not val_ok and require_val:
        raise FileNotFoundError(
            "Validation split is required but incomplete:\n  - "
            + "\n  - ".join(val_missing)
        )
    if not val_ok:
        logger.warning(
            "Validation split incomplete; training will run without validation "
            "or diagnose(). Missing:\n  - %s",
            "\n  - ".join(val_missing),
        )

    train_dataset_probs = list(getattr(tc, "dataset_probs", []))
    val_dataset_probs = list(getattr(tc, "val_dataset_probs", [])) or train_dataset_probs

    train_dataset = VideoLatentTrainingDataset(
        dataset_roots=tc.dataset_roots,
        mode="train",
        num_history=wc.history_length,
        num_frames=wc.pred_frames,
        num_cameras=tc.num_cameras,
        dataset_probs=train_dataset_probs,
        condition_view=str(getattr(tc, "video_latent_condition_view")),
        sample_stride_video_frames=int(getattr(tc, "video_latent_sample_stride")),
        action_dim=int(wc.action_dim),
        latent_height_per_view=int(getattr(tc, "latent_height_per_view")),
        latent_width=int(getattr(tc, "latent_width")),
        expected_video_keys=list(getattr(tc, "video_latent_video_keys", [])),
        require_production_contract=bool(
            getattr(tc, "video_latent_require_production_contract", False)
        ),
        required_target_fps=float(
            getattr(tc, "video_latent_required_target_fps", 0.0)
        ),
        required_target_image_hw=list(
            getattr(tc, "video_latent_required_target_image_hw", [])
        ),
        history_selector=str(getattr(tc, "history_selector", "recent")),
        history_offsets=list(getattr(tc, "history_offsets", [])),
        history_dilations=list(getattr(tc, "history_dilations", [1, 2])),
        history_collapse_prob=float(getattr(tc, "history_collapse_prob", 0.0)),
        history_eval_dilation=int(getattr(tc, "history_eval_dilation", 2)),
        generated_history_enabled=bool(
            getattr(tc, "generated_history_enabled", False)
        ),
        generated_history_unroll_chunks=list(
            getattr(tc, "generated_history_unroll_chunks", [1])
        ),
        camera_conditioning=bool(getattr(tc, "video_latent_camera_conditioning", False)),
        camera_sidecar_roots=list(getattr(tc, "video_latent_camera_sidecar_roots", [])),
        camera_intrinsics_fallback=str(
            getattr(tc, "video_latent_camera_intrinsics_fallback", "identity")
        ),
        camera_extrinsics_convention=str(
            getattr(tc, "video_latent_camera_extrinsics_convention", "world_from_camera")
        ),
        eef_projection_roots=list(getattr(tc, "video_latent_eef_projection_roots", [])),
        eef_projection_mode=str(
            getattr(tc, "video_latent_eef_projection_mode", "sidecar")
        ),
        eef_projection_load_heatmap=bool(
            getattr(tc, "video_latent_eef_projection_load_heatmap", False)
        ),
        eef_projection_heatmap_size=str(
            getattr(tc, "video_latent_eef_projection_heatmap_size", "")
        ),
    )

    validation_enabled = (
        val_ok
        and (
            int(getattr(tc, "val_interval", 0)) > 0
            or int(getattr(tc, "diagnose_interval", 0)) > 0
        )
    )
    if val_ok and not validation_enabled:
        logger.info(
            "Validation split is present but disabled because both "
            "wm_training.val_interval and diagnose_interval are <= 0."
        )

    val_dataset = None
    if validation_enabled:
        val_dataset = VideoLatentTrainingDataset(
            dataset_roots=tc.dataset_roots,
            mode="val",
            num_history=wc.history_length,
            num_frames=wc.pred_frames,
            num_cameras=tc.num_cameras,
            dataset_probs=val_dataset_probs,
            condition_view=str(getattr(tc, "video_latent_condition_view")),
            sample_stride_video_frames=int(getattr(tc, "video_latent_sample_stride")),
            action_dim=int(wc.action_dim),
            latent_height_per_view=int(getattr(tc, "latent_height_per_view")),
            latent_width=int(getattr(tc, "latent_width")),
            expected_video_keys=list(getattr(tc, "video_latent_video_keys", [])),
            require_production_contract=bool(
                getattr(tc, "video_latent_require_production_contract", False)
            ),
            required_target_fps=float(
                getattr(tc, "video_latent_required_target_fps", 0.0)
            ),
            required_target_image_hw=list(
                getattr(tc, "video_latent_required_target_image_hw", [])
            ),
            history_selector=str(getattr(tc, "history_selector", "recent")),
            history_offsets=list(getattr(tc, "history_offsets", [])),
            history_dilations=list(getattr(tc, "history_dilations", [1, 2])),
            history_collapse_prob=0.0,
            history_eval_dilation=int(getattr(tc, "history_eval_dilation", 2)),
            generated_history_enabled=False,
            camera_conditioning=bool(getattr(tc, "video_latent_camera_conditioning", False)),
            camera_sidecar_roots=list(getattr(tc, "video_latent_camera_sidecar_roots", [])),
            camera_intrinsics_fallback=str(
                getattr(tc, "video_latent_camera_intrinsics_fallback", "identity")
            ),
            camera_extrinsics_convention=str(
                getattr(tc, "video_latent_camera_extrinsics_convention", "world_from_camera")
            ),
            eef_projection_roots=list(getattr(tc, "video_latent_eef_projection_roots", [])),
            eef_projection_mode=str(
                getattr(tc, "video_latent_eef_projection_mode", "sidecar")
            ),
            eef_projection_load_heatmap=bool(
                getattr(tc, "video_latent_eef_projection_load_heatmap", False)
            ),
            eef_projection_heatmap_size=str(
                getattr(tc, "video_latent_eef_projection_heatmap_size", "")
            ),
        )

    logger.info(
        "Datasets: train=%d, val=%s",
        len(train_dataset),
        "disabled" if val_dataset is None else str(len(val_dataset)),
    )

    # Encode text on demand with a bounded per-rank cache.
    if causal_backend:
        trainer = CausalWanTrainer(training_config=tc, model_config=wc)
    else:
        trainer = WMTrainer(training_config=tc, model_config=wc)
    trainer.setup(
        resume_from=args.resume if args.resume else None,
        init_from=args.init_from if args.init_from else None,
    )

    # Train
    trainer.train(
        train_dataset,
        val_dataset if val_dataset is not None and len(val_dataset) > 0 else None,
    )

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
