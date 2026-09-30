"""WAN 2.2 DiT trainer with FSDP and rectified flow loss.

Trains the action-conditioned WAN 2.2 world model on pre-extracted VAE
latents. Implements the same rectified flow objective as yy-wan-training
but within the CoachWorld framework.

Key design choices:
  - FSDP for WAN 5B parameter sharding across GPUs
  - Rectified flow: noisy = (1-sigma)*clean + sigma*noise, target = noise - clean
  - Action conditioning via dedicated action K/V plus optional action AdaLN;
    text K/V remains on the pretrained WAN path
  - Text embedding pre-cached at init (T5 not on training GPU)
  - Step-based training loop with resumable distributed mixture sampling

Reference:
  - third_party/yy-wan-training/cosmos_predict2/models/wan_warped_model.py (loss computation)
  - third_party/yy-wan-training/cosmos_predict2/models/wan_warped_model_causal_action.py (FSDP setup)
"""

from __future__ import annotations

import json
import gc
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from coachworld.config import WMTrainingConfig, WorldModelConfig
from coachworld.data.distributed_mixture_sampler import (
    ResumableDistributedMixtureSampler,
)
from coachworld.data.video_latent_collate import collate_video_latent_batch
from coachworld.evaluator.training_validation import (
    resolve_closed_loop_validation_protocol,
)
from coachworld.world_model.action_contract import (
    apply_kv_future_only_action_mask,
    full_window_enabled,
    num_video_frames_for_latent_window,
    validate_action_timing_contract,
)
from coachworld.world_model.condition_config import (
    CONDITION_CONFIG_FIELDS,
    validate_condition_config_match,
)
from coachworld.world_model.action_adapter import make_robot_action_config, prepare_action_tensor
from coachworld.world_model.eef_spatial_condition import (
    EEF_SPATIAL_CHANNELS_PER_SLOT,
    rasterize_eef_spatial_condition,
)
from coachworld.world_model.generated_history import (
    generated_history_step_is_active,
    select_generated_history_unroll_chunks,
)
from coachworld.world_model.text_encoder import T5TextEncoder, make_null_text_embedding
from coachworld.wan.action.temporal_grouping import wan_causal_group_indices
from coachworld.wan import FlowMatchScheduler, WanModelAction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training loop helpers (from lora_finetuner.py)
# ---------------------------------------------------------------------------


def _infinite_loader(dataloader: DataLoader) -> Iterator:
    """Yield batches forever by cycling through the dataloader.

    Calls sampler.set_epoch() at each epoch boundary so the distributed
    mixture schedule advances while preserving a resume offset in its first
    epoch.
    """
    sampler = dataloader.sampler
    epoch = int(getattr(sampler, "start_epoch", getattr(sampler, "epoch", 0)))
    while True:
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1


# ---------------------------------------------------------------------------
# WMTrainer
# ---------------------------------------------------------------------------


class WMTrainer:
    """WAN 2.2 DiT trainer with FSDP and rectified flow loss."""

    # WAN 2.2 constants
    VAE_TEMPORAL_STRIDE = 4
    PATCH_SIZE = (1, 2, 2)
    # Kept at the historical location for import-level compatibility.
    CONDITION_CONFIG_FIELDS = CONDITION_CONFIG_FIELDS

    def __init__(
        self,
        training_config: WMTrainingConfig,
        model_config: WorldModelConfig,
    ) -> None:
        self.tcfg = training_config
        self.mcfg = model_config
        self._validate_condition_config_match()
        validate_action_timing_contract(
            action_schema=getattr(self.tcfg, "action_schema", "fixed"),
            action_rate=getattr(self.tcfg, "action_condition_timestep_rate", "raw"),
        )
        if bool(getattr(self.mcfg, "prope_enabled", False)) and not bool(
            getattr(self.tcfg, "video_latent_camera_conditioning", False)
        ):
            raise ValueError(
                "world_model.prope_enabled=true requires "
                "wm_training.video_latent_camera_conditioning=true so real "
                "viewmats/Ks are present in each batch."
            )
        history_clean_prob = float(getattr(self.tcfg, "history_clean_prob", 0.0))
        if not 0.0 <= history_clean_prob <= 1.0:
            raise ValueError(
                f"wm_training.history_clean_prob must be in [0,1], got {history_clean_prob}"
            )
        if bool(getattr(self.tcfg, "gt_condition_replacement", False)) and history_clean_prob < 1.0:
            logger.warning(
                "gt_condition_replacement=true overrides history_clean_prob and history_noise_max; "
                "all history frames will remain exact"
            )
        generated_history_probability = float(
            getattr(self.tcfg, "generated_history_probability", 0.0)
        )
        if not 0.0 <= generated_history_probability <= 1.0:
            raise ValueError(
                "wm_training.generated_history_probability must be in [0,1], "
                f"got {generated_history_probability}"
            )
        generated_history_blend = float(
            getattr(self.tcfg, "generated_history_blend", 0.5)
        )
        if not 0.0 <= generated_history_blend <= 1.0:
            raise ValueError(
                "wm_training.generated_history_blend must be in [0,1], "
                f"got {generated_history_blend}"
            )
        if bool(getattr(self.tcfg, "generated_history_enabled", False)):
            generated_history_unroll_chunks = [
                int(value)
                for value in getattr(
                    self.tcfg,
                    "generated_history_unroll_chunks",
                    [1],
                )
            ]
            if (
                not generated_history_unroll_chunks
                or any(value <= 0 for value in generated_history_unroll_chunks)
            ):
                raise ValueError(
                    "generated_history_unroll_chunks must contain positive integers, "
                    f"got {generated_history_unroll_chunks}"
                )
            if int(getattr(self.tcfg, "generated_history_denoise_steps", 0)) <= 0:
                raise ValueError(
                    "generated_history_enabled=true requires "
                    "generated_history_denoise_steps > 0"
                )
            if not full_window_enabled(self.tcfg) or not full_window_enabled(self.mcfg):
                raise ValueError(
                    "generated-history training requires full-window H/F mode"
                )
        if (
            bool(getattr(self.mcfg, "eef_spatial_conditioning_enabled", False))
            and float(getattr(self.tcfg, "counterfactual_action_weight", 0.0)) > 0.0
        ):
            raise ValueError(
                "counterfactual_action_weight is incompatible with spatial EEF controls: "
                "a counterfactual 3D action needs its own projected raster"
            )

        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
        self.dtype = torch.bfloat16 if model_config.mixed_precision == "bf16" else torch.float32
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0
        self.fsdp_active = self.world_size > 1 and bool(self.mcfg.fsdp_enabled)

        # Will be initialized in setup()
        self._dit = None
        self._scheduler = None
        self._optimizer = None
        self._lr_scheduler = None
        self._generated_history_scheduler = None
        self._text_emb_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._global_step = 0
        self._accum_count = 0
        self._sampling_state: dict[str, object] | None = None

        # wandb (optional, main process only)
        self._wandb_run = None

    def _validate_condition_config_match(self) -> None:
        """Reject split-brain condition config before training starts.

        The same architecture switches are mirrored under ``world_model`` and
        ``wm_training`` for historical reasons. Training and inference must see
        the same values; otherwise an ablation can silently train one model and
        evaluate another. Overrides should therefore set both namespaces.
        """
        validate_condition_config_match(self.mcfg, self.tcfg)

    def _condition_dropout_rate(self, name: str) -> float:
        """Return the current training-time condition dropout probability."""
        fixed = float(getattr(self.tcfg, name))
        start = float(getattr(self.tcfg, f"{name}_start", -1.0))
        end = float(getattr(self.tcfg, f"{name}_end", -1.0))
        decay_steps = int(getattr(self.tcfg, f"{name}_decay_steps", 0))
        if start < 0.0 or end < 0.0 or decay_steps <= 0:
            return max(0.0, min(1.0, fixed))
        frac = min(1.0, max(0.0, float(self._global_step) / float(decay_steps)))
        rate = start + (end - start) * frac
        return max(0.0, min(1.0, rate))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        unique_texts: Optional[list[str]] = None,
        resume_from: Optional[str] = None,
        init_from: Optional[str] = None,
    ) -> None:
        """Initialize model, FSDP, optimizer, scheduler, text encoder.

        Call this after distributed init and before train().

        Args:
            unique_texts: (ignored, kept for backward compat) text encoding is
                now on-the-fly with a bounded per-rank LRU cache.
            resume_from: Resume a previous run (model + optimizer + step).
                Must be a DCP-compatible checkpoint directory (contains dcp/).
                Loaded AFTER FSDP wrapping via torch.distributed.checkpoint.
            init_from: Initialize model weights from a safetensors checkpoint
                (no optimizer, step=0). Loaded BEFORE FSDP wrapping. Use this
                for warm-starting a new run from a previous safetensors.
        """
        if self.is_main:
            visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
            logger.info(
                "Runtime topology: world_size=%d rank=%d local_device=%s "
                "visible_cuda=%d fsdp_active=%s fsdp_config=%s",
                self.world_size,
                self.rank,
                self.device,
                visible_cuda,
                self.fsdp_active,
                self.mcfg.fsdp_enabled,
            )
            if visible_cuda > 1 and self.world_size == 1:
                logger.warning(
                    "Multiple CUDA devices are visible but WORLD_SIZE=1; running "
                    "single-process training. Use torchrun to enable multi-GPU FSDP."
                )
        self._init_dit()
        # Pre-FSDP model init (safetensors warm-start, no optimizer)
        if init_from is not None:
            self.load_pretrained_weights(init_from)
        if self.fsdp_active:
            self._init_fsdp()
        elif self.world_size > 1 and self.is_main:
            logger.warning(
                "WORLD_SIZE=%d but world_model.fsdp_enabled=false; each rank "
                "will keep a full DiT copy. This is intended only for debugging.",
                self.world_size,
            )
        self._init_optimizer()
        self._init_scheduler()
        # Post-FSDP full resume (model + optimizer + step via DCP)
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        self._init_text_encoder()
        if self.is_main:
            self._init_wandb()

    def _init_dit(self) -> None:
        """Load WanModelAction with dedicated action conditioning.

        Action is injected through a root-level ActionEncoder whose output
        tokens feed a separate action K/V branch in every WAN block. The text
        branch keeps the pretrained K/V projections. Optional action AdaLN is
        computed from causal action chunks and added to WAN's time embedding.
        """
        num_cameras = int(self.tcfg.num_cameras)
        latent_h_per_cam = int(self.tcfg.latent_height_per_view)
        latent_w = int(self.tcfg.latent_width)
        if num_cameras <= 0:
            raise ValueError(f"wm_training.num_cameras must be positive, got {num_cameras}")
        if latent_h_per_cam <= 0 or latent_w <= 0:
            raise ValueError(
                "wm_training latent geometry must be positive, got "
                f"latent_height_per_view={latent_h_per_cam}, latent_width={latent_w}"
            )
        t_future = max(1, self.mcfg.pred_frames - 1)
        latent_h = num_cameras * latent_h_per_cam
        action_config = make_robot_action_config(
            action_dim=self.mcfg.action_dim,
            latent_C=48,
            latent_T=t_future,
            latent_H=latent_h,
            latent_W=latent_w,
            latent_height_per_view=latent_h_per_cam,
            vae_spatial_stride=int(getattr(self.mcfg, "prope_spatial_stride", 16)),
            vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
            action_schema=str(
                getattr(self.tcfg, "action_schema", getattr(self.mcfg, "action_schema", "fixed"))
            ),
            max_arm_slots=int(
                getattr(self.tcfg, "max_arm_slots", getattr(self.mcfg, "max_arm_slots", 2))
            ),
            action_max_time_steps=int(
                getattr(
                    self.tcfg,
                    "action_max_time_steps",
                    getattr(self.mcfg, "action_max_time_steps", 512),
                )
            ),
            action_kv_enabled=bool(
                getattr(self.tcfg, "action_kv_enabled", getattr(self.mcfg, "action_kv_enabled", True))
            ),
            action_v_init_scale=float(
                getattr(
                    self.tcfg,
                    "action_v_init_scale",
                    getattr(self.mcfg, "action_v_init_scale", 0.1),
                )
            ),
            action_num_domains=int(
                getattr(
                    self.tcfg,
                    "action_num_domains",
                    getattr(self.mcfg, "action_num_domains", 1),
                )
            ),
            action_domain_prompt_tokens=int(
                getattr(
                    self.tcfg,
                    "action_domain_prompt_tokens",
                    getattr(self.mcfg, "action_domain_prompt_tokens", 0),
                )
            ),
            domain_aware_action_projection_enabled=bool(
                getattr(
                    self.tcfg,
                    "domain_aware_action_projection_enabled",
                    getattr(self.mcfg, "domain_aware_action_projection_enabled", False),
                )
            ),
            domain_aware_group_projection_enabled=bool(
                getattr(
                    self.tcfg,
                    "domain_aware_group_projection_enabled",
                    getattr(self.mcfg, "domain_aware_group_projection_enabled", False),
                )
            ),
            num_cameras=num_cameras,
            multi_view_position_mode=str(
                getattr(
                    self.tcfg,
                    "multi_view_position_mode",
                    getattr(self.mcfg, "multi_view_position_mode", "global"),
                )
            ),
            camera_id_embedding_enabled=bool(
                getattr(
                    self.tcfg,
                    "camera_id_embedding_enabled",
                    getattr(self.mcfg, "camera_id_embedding_enabled", False),
                )
            ),
        )
        action_config["dense_action_film_enabled"] = bool(
            getattr(self.tcfg, "dense_action_film_enabled", False)
        )
        action_config["action_adaln_modulation_enabled"] = bool(
            getattr(self.tcfg, "action_adaln_modulation_enabled", False)
        )
        action_config["action_adaln_scale"] = float(
            getattr(
                self.tcfg,
                "action_adaln_scale",
                getattr(self.mcfg, "action_adaln_scale", 1.0),
            )
        )
        action_config["action_control_scale"] = float(
            getattr(
                self.tcfg,
                "action_control_scale",
                getattr(self.mcfg, "action_control_scale", 0.05),
            )
        )
        action_config["action_dense_rank"] = int(
            getattr(
                self.tcfg,
                "action_dense_rank",
                getattr(self.mcfg, "action_dense_rank", 128),
            )
        )
        action_config["action_dense_layernorm_enabled"] = bool(
            getattr(
                self.tcfg,
                "action_dense_layernorm_enabled",
                getattr(self.mcfg, "action_dense_layernorm_enabled", False),
            )
        )
        action_config["action_dense_relative_chunks"] = bool(
            getattr(
                self.tcfg,
                "action_dense_relative_chunks",
                getattr(self.mcfg, "action_dense_relative_chunks", False),
            )
        )
        action_config["eef_projection_kv_enabled"] = bool(
            getattr(
                self.tcfg,
                "eef_projection_kv_enabled",
                getattr(self.mcfg, "eef_projection_kv_enabled", False),
            )
        )
        logger.info(
            "Loading WAN DiT from %s (action_dim=%d, latent shape=(%d,%d,%d,%d), "
            "num_cameras=%d, action_schema=%s, max_arm_slots=%d, action_max_time_steps=%d, action_kv=%s, "
            "action_v_init_scale=%.3g, "
            "domains=%d, domain_prompt_tokens=%d, domain_aware_projection=%s, "
            "domain_aware_group_projection=%s, "
            "dense_action_film=%s, action_adaln=%s, "
            "action_adaln_scale=%.3g, action_control_scale=%.3g, action_dense_rank=%d, "
            "action_dense_layernorm=%s, action_dense_relative_chunks=%s, "
            "multi_view_position_mode=%s, camera_id_embedding=%s)",
            self.mcfg.checkpoint, self.mcfg.action_dim,
            48, t_future, latent_h, latent_w, num_cameras,
            action_config["action_schema"],
            action_config["max_arm_slots"],
            action_config["action_max_time_steps"],
            action_config["action_kv_enabled"],
            action_config["action_v_init_scale"],
            action_config["action_num_domains"],
            action_config["action_domain_prompt_tokens"],
            action_config["domain_aware_action_projection_enabled"],
            action_config["domain_aware_group_projection_enabled"],
            action_config["dense_action_film_enabled"],
            action_config["action_adaln_modulation_enabled"],
            action_config["action_adaln_scale"],
            action_config["action_control_scale"],
            action_config["action_dense_rank"],
            action_config["action_dense_layernorm_enabled"],
            action_config["action_dense_relative_chunks"],
            action_config["multi_view_position_mode"],
            action_config["camera_id_embedding_enabled"],
        )
        self._dit = WanModelAction.from_pretrained(
            self.mcfg.checkpoint,
            action_config=action_config,
            device_map=None, low_cpu_mem_usage=False,
        )
        # Warm-start the new action K/V in every cross-attn block from the
        # just-loaded pretrained text K/V (V1.1 LDA-style dedicated action
        # cross-attn). __init__ seeds from Xavier; here we refresh from the
        # real WAN pretrained weights so action attention starts with the
        # same attention pattern as text and then specializes.
        self._dit.post_load_warm_start_action_kv()
        if bool(getattr(self.mcfg, "prope_enabled", False)):
            self._dit.enable_prope(zero_init=bool(getattr(self.mcfg, "prope_zero_init", True)))
            logger.info(
                "Enabled non-causal per-patch PRoPE (zero_init=%s, spatial_stride=%d)",
                bool(getattr(self.mcfg, "prope_zero_init", True)),
                int(getattr(self.mcfg, "prope_spatial_stride", 16)),
            )
        extra_input_channels = 0
        if bool(
            getattr(self.tcfg, "condition_mask_enabled", False)
            or getattr(self.mcfg, "condition_mask_enabled", False)
        ):
            extra_input_channels += 1
        if bool(getattr(self.mcfg, "eef_spatial_conditioning_enabled", False)):
            extra_input_channels += (
                int(self.mcfg.max_arm_slots) * EEF_SPATIAL_CHANNELS_PER_SLOT
            )
        if extra_input_channels:
            self._dit.expand_input_channels(
                extra_channels=extra_input_channels,
                zero_init=True,
            )
            logger.info(
                "Expanded DiT patch embedding with %d conditioning channels "
                "(mask=%s, eef_spatial=%s, in_dim=%d)",
                extra_input_channels,
                bool(getattr(self.mcfg, "condition_mask_enabled", False)),
                bool(getattr(self.mcfg, "eef_spatial_conditioning_enabled", False)),
                self._dit.in_dim,
            )
        # Keep MASTER params in fp32 — FSDP2 MixedPrecisionPolicy downcasts to
        # bf16 during forward/backward and reduces grads in fp32. Storing
        # master params in bf16 was the root cause of the 500-step smoke-test's
        # grad_l2_action=inf (bf16 max ≈ 65504; cross-attn backward exceeded it
        # on the ActionEncoder path → poisoned total_norm → clip scaled grads
        # to 0 → no learning). Keep device move, drop dtype cast.
        self._dit.to(device=self.device)
        self._dit.train()
        self._dit.requires_grad_(True)

        if self.mcfg.gradient_checkpoint:
            self._dit.gradient_checkpoint = True

        # Defensive assert: the new ActionEncoder must be present at root.
        assert hasattr(self._dit, "action_encoder"), (
            "WanModelAction has no .action_encoder — action_config ignored or "
            "module import failed. Check make_robot_action_config() output."
        )

        num_params = sum(p.numel() for p in self._dit.parameters())
        trainable = sum(p.numel() for p in self._dit.parameters() if p.requires_grad)
        action_params = sum(p.numel() for p in self._dit.action_encoder.parameters())
        if getattr(self._dit, "eef_projection_encoder", None) is not None:
            action_params += sum(p.numel() for p in self._dit.eef_projection_encoder.parameters())
        logger.info(
            "DiT loaded: %.1fM params, %.1fM trainable, %.1fM in action-side encoders",
            num_params / 1e6, trainable / 1e6, action_params / 1e6,
        )

    def _init_fsdp(self) -> None:
        """Wrap DiT with FSDP2 + MixedPrecisionPolicy for stable bf16 training.

        MixedPrecisionPolicy:
          - param_dtype=bf16: params are downcast to bf16 for forward/backward
            compute. Combined with torch.autocast in compute_loss, all matmuls
            run in bf16 tensor cores.
          - reduce_dtype=fp32: grads are reduced across ranks in fp32. Prevents
            the bf16 overflow observed in V1 smoke-test (grad_l2_action=inf
            was poisoning clip_grad_norm_ and stalling learning).

        Master params stay in fp32 (we did NOT cast self._dit to bf16 before).
        AdamW's optimizer state is fp32. Checkpoint save gathers fp32 params.
        """
        from torch.distributed.device_mesh import init_device_mesh
        try:
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        shard_size = min(self.mcfg.fsdp_shard_size, self.world_size)
        replica_size = self.world_size // shard_size

        logger.info(
            "Initializing FSDP2: world_size=%d, shard_size=%d, replica_size=%d, "
            "param_dtype=%s, reduce_dtype=fp32",
            self.world_size, shard_size, replica_size, self.dtype,
        )

        dp_mesh = init_device_mesh(
            "cuda",
            (replica_size, shard_size),
            mesh_dim_names=("replicate", "shard"),
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.dtype,          # bf16 during compute
            reduce_dtype=torch.float32,      # fp32 grad all-reduce → no overflow
        )

        # Shard each block individually
        for block in self._dit.blocks:
            fully_shard(
                block, mesh=dp_mesh, mp_policy=mp_policy,
                reshard_after_forward=True,
            )

        # Shard the whole model (root parameters: patch_embedding,
        # text_embedding, time_embedding, time_projection, action_encoder,
        # head, img_emb — all fp32 master, bf16 compute, fp32 grad reduce).
        fully_shard(
            self._dit, mesh=dp_mesh, mp_policy=mp_policy,
            reshard_after_forward=True,
        )

        logger.info("FSDP2 initialized")

    @staticmethod
    def _is_action_side_param(name: str) -> bool:
        return (
            "action_encoder" in name
            or "eef_projection_encoder" in name
            or "action_adaln_encoder" in name
            or "action_dense_encoder" in name
            or "action_film" in name
            or "action_injector_" in name
            or "action_scale_shift_layer" in name
            or "action_decoder" in name
            or ".k_action." in name
            or ".v_action." in name
            or "norm_k_action" in name
        )

    @staticmethod
    def _is_warmstart_optional_param(name: str) -> bool:
        return (
            WMTrainer._is_action_side_param(name)
            or name == "camera_id_embedding"
            or "prope_o" in name
        )

    @staticmethod
    def _expand_warmstart_tensor(
        name: str,
        checkpoint_value: torch.Tensor,
        current_value: torch.Tensor,
    ) -> torch.Tensor | None:
        """Preserve existing domain rows when the domain registry grows.

        A model trained with N domains stores the domain-specific LoRA deltas
        on the leading axis.  Adding domains must not discard the already
        learned rows 0..N-1.  Other shape changes remain incompatible.
        """
        old_shape = tuple(checkpoint_value.shape)
        new_shape = tuple(current_value.shape)
        if name == "patch_embedding.weight":
            # Preserve every learned latent/condition channel and append only
            # the newly introduced conditioning channels. The current tensor
            # was constructed by expand_input_channels(zero_init=True), so its
            # suffix is the correct zero initialization for new EEF rasters.
            if (
                checkpoint_value.ndim == current_value.ndim == 5
                and old_shape[0] == new_shape[0]
                and old_shape[2:] == new_shape[2:]
                and old_shape[1] < new_shape[1]
            ):
                expanded = current_value.detach().cpu().clone()
                expanded[:, : old_shape[1]].copy_(
                    checkpoint_value.to(dtype=expanded.dtype, device="cpu")
                )
                return expanded
            return None
        if not WMTrainer._is_action_side_param(name):
            return None
        if checkpoint_value.ndim < 1 or checkpoint_value.ndim != current_value.ndim:
            return None
        if old_shape[1:] != new_shape[1:] or old_shape[0] >= new_shape[0]:
            return None
        if "delta_weight" not in name and "delta_bias" not in name:
            return None
        expanded = current_value.detach().cpu().clone()
        expanded[: old_shape[0]].copy_(checkpoint_value.to(dtype=expanded.dtype, device="cpu"))
        return expanded

    def _collect_dense_action_stats(self) -> dict[str, float]:
        """Collect last-forward dense action control activation magnitudes."""
        if self._dit is None or not hasattr(self._dit, "blocks"):
            return {}
        scale_vals = []
        shift_vals = []
        for block in self._dit.blocks:
            scale = getattr(block, "_last_action_scale_abs_mean", None)
            shift = getattr(block, "_last_action_shift_abs_mean", None)
            if scale is not None:
                scale_vals.append(float(scale.item()))
            if shift is not None:
                shift_vals.append(float(shift.item()))
        out = {}
        if scale_vals:
            out["dense_film_scale_abs_mean"] = float(np.mean(scale_vals))
        if shift_vals:
            out["dense_film_shift_abs_mean"] = float(np.mean(shift_vals))
        return out

    def _init_optimizer(self) -> None:
        """Set up AdamW optimizer with 3 disjoint param groups covering every
        trainable parameter:

          * ``action``   — all action-conditioning parameters: root-level
                           ``action_encoder.*`` and ``action_decoder.*``,
                           per-block ``k_action`` / ``v_action`` /
                           ``norm_k_action``. LR = base × ``action_lr_mult``,
                           WD = 0. These are warm-start or randomly
                           initialized and need aggressive learning; WD
                           would bias them toward zero which for a
                           newly-introduced pathway is the "ignore action"
                           absorbing state.
          * ``dit_wd``   — pretrained DiT 2-D+ weights (Linear.weight, Conv,
                           embeddings). LR = base, WD = 0.01.
          * ``dit_nowd`` — pretrained DiT 1-D weights (biases, LayerNorm
                           and RMSNorm weights). LR = base, WD = 0. Standard
                           transformer recipe: weight decay on norms/biases
                           hurts.
        """
        lr_mult = float(getattr(self.mcfg, "action_lr_mult", 1.0))

        def is_no_wd(name: str, p) -> bool:
            # Scalars, biases, LayerNorm / RMSNorm weights — anything 1-D.
            # Standard transformer recipe: weight decay only on 2-D+ weights.
            return p.ndim <= 1

        action_params, dit_wd_params, dit_nowd_params = [], [], []
        n_action = n_dit_wd = n_dit_nowd = 0
        for name, p in self._dit.named_parameters():
            if not p.requires_grad:
                continue
            if self._is_action_side_param(name):
                action_params.append(p)
                n_action += p.numel()
            elif is_no_wd(name, p):
                dit_nowd_params.append(p)
                n_dit_nowd += p.numel()
            else:
                dit_wd_params.append(p)
                n_dit_wd += p.numel()

        param_groups = [
            {
                "params": dit_wd_params,
                "lr": self.mcfg.learning_rate,
                "weight_decay": 0.01,
                "name": "dit_wd",
            },
            {
                "params": dit_nowd_params,
                "lr": self.mcfg.learning_rate,
                "weight_decay": 0.0,
                "name": "dit_nowd",
            },
            {
                "params": action_params,
                "lr": self.mcfg.learning_rate * lr_mult,
                "weight_decay": 0.0,
                "name": "action",
            },
        ]

        if self.is_main:
            logger.info(
                "AdamW: dit_wd lr=%.1e (%d params, wd=0.01) | "
                "dit_nowd lr=%.1e (%d params, wd=0) | "
                "action lr=%.1e (%d params, wd=0, mult=%.1fx)",
                self.mcfg.learning_rate, n_dit_wd,
                self.mcfg.learning_rate, n_dit_nowd,
                self.mcfg.learning_rate * lr_mult, n_action, lr_mult,
            )

        self._optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.mcfg.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,  # default; per-group overrides above
            eps=1e-8,
        )

        warmup_steps = max(0, int(self.tcfg.warmup_steps))
        schedule = str(getattr(self.tcfg, "lr_schedule", "constant")).strip().lower()
        if schedule not in {"constant", "cosine"}:
            raise ValueError(f"lr_schedule must be constant or cosine, got {schedule!r}")
        min_ratio = float(getattr(self.tcfg, "lr_min_ratio", 0.1))
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError(f"lr_min_ratio must be in [0,1], got {min_ratio}")
        total_updates = max(
            1,
            math.ceil(int(self.mcfg.max_train_steps) / int(self.mcfg.grad_accum_steps)),
        )

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if schedule == "constant":
                return 1.0
            decay_updates = max(1, total_updates - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_updates))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine

        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optimizer, lr_lambda
        )
        if self.is_main:
            logger.info(
                "LR schedule: %s, warmup=%d optimizer updates, total=%d, min_ratio=%.3f",
                schedule,
                warmup_steps,
                total_updates,
                min_ratio,
            )

    def _init_scheduler(self) -> None:
        """Set up FlowMatchScheduler for training."""
        # FlowMatchScheduler imported at module level from coachworld.wan

        self._scheduler = FlowMatchScheduler(
            shift=self.mcfg.flow_matching_shift,
            num_train_timesteps=1000,
            extra_one_step=True,
        )
        self._scheduler.set_timesteps(1000, device=self.device)
        logger.info("FlowMatchScheduler initialized (shift=%.1f)", self.mcfg.flow_matching_shift)
        if bool(getattr(self.tcfg, "generated_history_enabled", False)):
            steps = int(getattr(self.tcfg, "generated_history_denoise_steps", 2))
            self._generated_history_scheduler = FlowMatchScheduler(
                shift=self.mcfg.flow_matching_shift,
                num_train_timesteps=1000,
                extra_one_step=True,
            )
            self._generated_history_scheduler.set_timesteps(
                steps,
                device=self.device,
            )
            logger.info(
                "Generated-history rollout scheduler initialized (%d denoise steps)",
                steps,
            )

    def _resolve_text_encoder_device(self) -> str:
        """Resolve training-time T5 placement.

        ``auto`` is intentionally asymmetric:
        - single-process debug: CPU T5 keeps A800 VRAM for DiT/optimizer;
        - torchrun/FSDP: local CUDA avoids N huge CPU T5 copies and keeps text
          encoding behavior close to existing multi-GPU runs.
        """
        choice = os.environ.get(
            "COACHWORLD_TEXT_ENCODER_DEVICE",
            str(getattr(self.tcfg, "text_encoder_device", "auto")),
        ).strip()
        lowered = choice.lower()
        if lowered in {"", "auto"}:
            return str(self.device) if self.world_size > 1 else "cpu"
        if lowered in {"cuda", "local_cuda", "cuda:local"}:
            return str(self.device)
        return choice

    def _init_text_encoder(self) -> None:
        """Load T5 encoder resident on this rank's GPU for on-the-fly encoding.

        Pre-encoding the full unique-text vocabulary onto CPU can exceed
        available memory on large datasets.

        Keeping T5 resident costs ~11 GB VRAM per rank (UMT5-XXL in bf16) but
        makes memory deterministic and lets the trainer encode arbitrary
        instructions on demand. A small per-rank LRU cache avoids re-encoding
        prompts that recur within a few hundred steps.
        """
        if not bool(getattr(self.tcfg, "text_conditioning_enabled", True)):
            self._text_encoder = None
            self._text_emb_cache = OrderedDict()
            self._text_cache_limit = 0
            if self.is_main:
                logger.info("Text conditioning disabled; using null text embeddings.")
            return
        text_device = self._resolve_text_encoder_device()
        self._text_encoder = T5TextEncoder(
            wan_model_dir=self.mcfg.checkpoint,
            device=text_device,
            dtype=self.dtype,
        )
        self._text_encoder.load()
        # Each cached WAN T5 embedding is roughly 4 MiB. Keep this explicitly
        # configurable because batch=2 at 512x768 leaves less than 1 GiB of
        # driver-visible headroom on a 96 GiB card.
        self._text_emb_cache = OrderedDict()
        self._text_cache_limit = int(
            getattr(self.tcfg, "text_embedding_cache_limit", 32)
        )
        if self._text_cache_limit < 0:
            raise ValueError("text_embedding_cache_limit must be non-negative")
        if self.is_main:
            logger.info(
                "T5 encoder resident on %s; on-the-fly encoding with LRU "
                "cache (limit=%d). Training DiT device is %s.",
                text_device, self._text_cache_limit, self.device,
            )

    def _get_text_embedding(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of instructions via the resident T5 encoder.

        Returns (B, L, D) tensor on training device. Caches recent prompts in
        a bounded dict — FIFO eviction when full. Empty-string instructions
        map to a zero embedding (shares the CFG null branch).
        """
        if not bool(getattr(self.tcfg, "text_conditioning_enabled", True)):
            return make_null_text_embedding(
                batch_size=len(texts),
                device=str(self.device),
                dtype=self.dtype,
            )
        if self._text_encoder is None:
            raise RuntimeError("text encoder is not initialized while text conditioning is enabled")
        embeddings = []
        for text in texts:
            if not text:
                embeddings.append(
                    make_null_text_embedding(
                        device=str(self._text_encoder.device), dtype=self.dtype,
                    )
                )
                continue
            cached = self._text_emb_cache.get(text)
            if cached is not None:
                self._text_emb_cache.move_to_end(text)
                embeddings.append(cached)
                continue
            emb = self._text_encoder.encode_single(text)  # (1, L, D)
            if self._text_cache_limit > 0:
                while len(self._text_emb_cache) >= self._text_cache_limit:
                    self._text_emb_cache.popitem(last=False)
                self._text_emb_cache[text] = emb
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0).to(device=self.device, dtype=self.dtype)

    def _init_wandb(self) -> None:
        """Initialize wandb on main process."""
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.tcfg.wandb_project,
                name=self.tcfg.wandb_run_name or None,
                config={
                    "model": self.mcfg.__dict__ if hasattr(self.mcfg, '__dict__') else str(self.mcfg),
                    "training": self.tcfg.__dict__ if hasattr(self.tcfg, '__dict__') else str(self.tcfg),
                },
            )
        except Exception as e:
            logger.warning("Failed to init wandb: %s", e)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def _eef_projection_kwargs(self, batch: dict, start: int, end: int) -> dict:
        if not bool(
            getattr(
                self.tcfg,
                "eef_projection_kv_enabled",
                getattr(self.mcfg, "eef_projection_kv_enabled", False),
            )
        ):
            return {}
        required = ("eef_uv", "eef_depth", "eef_valid", "eef_image_hw")
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(
                "world_model.eef_projection_kv_enabled=true requires EEF projection "
                f"fields in the batch; missing {missing}. Configure "
                "wm_training.video_latent_eef_projection_mode='online' or provide "
                "wm_training.video_latent_eef_projection_roots."
            )
        return {
            "eef_uv": batch["eef_uv"][:, start:end].to(device=self.device, dtype=self.dtype),
            "eef_depth": batch["eef_depth"][:, start:end].to(device=self.device, dtype=self.dtype),
            "eef_valid": batch["eef_valid"][:, start:end].to(device=self.device),
            "eef_image_hw": batch["eef_image_hw"][:, start:end].to(device=self.device),
        }

    def _eef_spatial_condition(
        self,
        batch: dict,
        start: int,
        end: int,
        *,
        latent_height: int,
        latent_width: int,
        action_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if not bool(getattr(self.mcfg, "eef_spatial_conditioning_enabled", False)):
            return None
        required = ("eef_uv", "eef_depth", "eef_valid", "eef_image_hw", "eef_gripper")
        missing = [key for key in required if key not in batch]
        if missing:
            raise RuntimeError(
                "eef_spatial_conditioning_enabled=true requires online gripper-aware "
                f"EEF projection fields; missing {missing}"
            )
        spatial = rasterize_eef_spatial_condition(
            uv=batch["eef_uv"][:, start:end].to(device=self.device, dtype=self.dtype),
            depth=batch["eef_depth"][:, start:end].to(device=self.device, dtype=self.dtype),
            valid=batch["eef_valid"][:, start:end].to(device=self.device),
            image_hw=batch["eef_image_hw"][:, start:end].to(device=self.device),
            gripper=batch["eef_gripper"][:, start:end].to(device=self.device, dtype=self.dtype),
            latent_height_per_view=int(self.mcfg.latent_height_per_view),
            latent_width=int(self.mcfg.latent_width),
            sigma_px=float(self.mcfg.eef_spatial_sigma_px),
            depth_scale=float(self.mcfg.eef_spatial_depth_scale),
        )
        expected = (
            len(batch["latent"]),
            int(self.mcfg.max_arm_slots) * EEF_SPATIAL_CHANNELS_PER_SLOT,
            end - start,
            latent_height,
            latent_width,
        )
        if tuple(spatial.shape) != expected:
            raise RuntimeError(
                f"EEF spatial condition shape {tuple(spatial.shape)} != latent contract {expected}"
            )
        if action_drop_mask is not None:
            keep = (~action_drop_mask.to(device=self.device).bool()).to(dtype=spatial.dtype)
            spatial = spatial * keep.view(-1, 1, 1, 1, 1)
        return spatial

    @staticmethod
    def _concat_condition_channels(
        condition_mask: torch.Tensor | None,
        eef_spatial: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if condition_mask is None:
            return eef_spatial
        if eef_spatial is None:
            return condition_mask
        return torch.cat([condition_mask, eef_spatial], dim=1)

    @torch.no_grad()
    def _generate_history_source_future(self, batch: dict) -> torch.Tensor:
        """Generate one detached F-step chunk from the source H/F window."""
        if self._generated_history_scheduler is None:
            raise RuntimeError(
                "generated-history scheduler is not initialized"
            )

        latent_full = batch["latent"].to(device=self.device, dtype=self.dtype)
        latent_full = latent_full.permute(0, 2, 1, 3, 4).contiguous()
        B, C, T, H, W = latent_full.shape
        history_frames = int(self.mcfg.history_length)
        future_frames = int(self.mcfg.pred_frames)
        if T != history_frames + future_frames:
            raise RuntimeError(
                "generated-history source must be one full H/F window: "
                f"got T={T}, expected {history_frames + future_frames}"
            )

        actions = batch["action"]
        action_domain_ids = self._batch_action_domain_ids(
            batch,
            action_present=isinstance(actions, torch.Tensor),
        )
        action_dense_cond = None
        if isinstance(actions, torch.Tensor):
            dense_len = num_video_frames_for_latent_window(
                T,
                self.VAE_TEMPORAL_STRIDE,
            )
            actions_dense = actions[:, :dense_len, ...].contiguous()
            action_dense_cond = self._prepare_batch_actions(actions_dense, T)
            if (
                str(
                    getattr(
                        self.tcfg,
                        "action_condition_timestep_rate",
                        "raw",
                    )
                )
                == "latent"
            ):
                actions_sparse = actions_dense[
                    :, :: self.VAE_TEMPORAL_STRIDE, ...
                ][:, :T, ...].contiguous()
                action_cond = self._align_latent_rate_actions(
                    actions_sparse.to(device=self.device, dtype=self.dtype),
                    T,
                )
            else:
                action_cond = self._prepare_batch_actions(actions_dense, T)
        else:
            action_cond = self._prepare_batch_actions(actions, T)
            action_dense_cond = action_cond

        dense_zero_first_chunk = False
        dense_zero_prefix_chunks = 0
        if (
            action_cond is not None
            and bool(
                getattr(self.tcfg, "future_only_action_conditioning", False)
                or getattr(self.mcfg, "future_only_action_conditioning", False)
            )
        ):
            action_cond, action_mask = apply_kv_future_only_action_mask(
                action_cond,
                history_frames=history_frames,
                action_rate=str(
                    getattr(
                        self.tcfg,
                        "action_condition_timestep_rate",
                        "raw",
                    )
                ),
                vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
            )
            dense_zero_first_chunk = action_mask.dense_zero_first_chunk
            dense_zero_prefix_chunks = action_mask.dense_zero_prefix_chunks

        viewmats = None
        Ks = None
        if bool(getattr(self.mcfg, "prope_enabled", False)):
            viewmats = batch["viewmats"].to(
                device=self.device,
                dtype=self.dtype,
            )
            Ks = batch["Ks"].to(device=self.device, dtype=self.dtype)
        eef_projection_kwargs = self._eef_projection_kwargs(batch, 0, T)

        condition_mask = None
        if bool(
            getattr(self.tcfg, "condition_mask_enabled", False)
            or getattr(self.mcfg, "condition_mask_enabled", False)
        ):
            condition_mask = torch.zeros(
                B,
                1,
                T,
                H,
                W,
                device=self.device,
                dtype=self.dtype,
            )
            condition_mask[:, :, :history_frames] = 1.0
        eef_spatial = self._eef_spatial_condition(
            batch,
            0,
            T,
            latent_height=H,
            latent_width=W,
        )
        cond_concat = self._concat_condition_channels(
            condition_mask,
            eef_spatial,
        )
        text_emb = self._get_text_embedding(batch["text"])

        current = latent_full[:, :, :history_frames].float()
        noise = torch.randn(
            B,
            C,
            T,
            H,
            W,
            device=self.device,
            dtype=torch.float32,
        )
        rollout_latent = noise
        rollout_latent[:, :, :history_frames] = current
        seq_len = T * (H // self.PATCH_SIZE[1]) * (
            W // self.PATCH_SIZE[2]
        )

        for timestep in self._generated_history_scheduler.timesteps:
            timestep_batch = timestep.expand(B).to(
                device=self.device,
                dtype=self.dtype,
            )
            with torch.autocast("cuda", dtype=self.dtype):
                pred_list = self._dit(
                    [
                        rollout_latent[i].to(dtype=self.dtype)
                        for i in range(B)
                    ],
                    timestep_batch,
                    text_emb,
                    seq_len,
                    cond_concat=cond_concat,
                    action_seq=action_cond,
                    action_dense_seq=action_dense_cond,
                    action_dense_zero_first_chunk=dense_zero_first_chunk,
                    action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                    action_domain_ids=action_domain_ids,
                    viewmats=viewmats,
                    Ks=Ks,
                    **eef_projection_kwargs,
                )
                velocity = torch.stack(pred_list).float()
            rollout_latent = self._generated_history_scheduler.step(
                velocity,
                timestep,
                rollout_latent,
            )
            rollout_latent[:, :, :history_frames] = current

        return rollout_latent[:, :, history_frames:].to(
            dtype=self.dtype
        ).detach()

    def _advance_generated_history_batch(
        self,
        source_batch: dict,
        target_batch: dict,
    ) -> tuple[dict, dict[str, float]]:
        """Advance one chunk and inject detached predictions into target H."""
        source_future = self._generate_history_source_future(source_batch)
        next_batch = dict(target_batch)
        next_latent = next_batch["latent"].to(
            device=self.device,
            dtype=self.dtype,
        ).clone()
        source_ids = source_batch["future_latent_ids"].tolist()
        target_ids = next_batch["history_latent_ids"].tolist()

        allowed_keys = {
            str(value)
            for value in getattr(
                self.tcfg,
                "generated_history_dataset_keys",
                [],
            )
        }
        dataset_keys = [str(value) for value in source_batch["dataset_key"]]
        target_active = next_batch.get(
            "_generated_history_chain_active_mask"
        )
        if target_active is None:
            target_active = torch.ones(
                len(dataset_keys),
                dtype=torch.bool,
            )
        target_active = target_active.to(
            device=self.device,
            dtype=torch.bool,
        )
        eligible = torch.tensor(
            [
                not allowed_keys or dataset_key in allowed_keys
                for dataset_key in dataset_keys
            ],
            device=self.device,
            dtype=torch.bool,
        )
        eligible &= target_active
        blend = float(getattr(self.tcfg, "generated_history_blend", 0.5))
        replaced = 0
        delta_sum = 0.0
        replaced_samples = torch.zeros_like(eligible)
        for batch_index in range(next_latent.shape[0]):
            if not bool(eligible[batch_index].item()):
                continue
            source_lookup = {
                int(latent_id): position
                for position, latent_id in enumerate(source_ids[batch_index])
            }
            for history_position, latent_id in enumerate(
                target_ids[batch_index]
            ):
                source_position = source_lookup.get(int(latent_id))
                if source_position is None:
                    continue
                ground_truth = next_latent[
                    batch_index,
                    history_position,
                ]
                prediction = source_future[
                    batch_index,
                    :,
                    source_position,
                ]
                replacement = torch.lerp(
                    ground_truth,
                    prediction,
                    blend,
                )
                delta_sum += float(
                    (replacement - ground_truth).abs().mean().item()
                )
                next_latent[
                    batch_index,
                    history_position,
                ] = replacement
                replaced += 1
                replaced_samples[batch_index] = True

        if bool(eligible.any()) and replaced == 0:
            raise RuntimeError(
                "generated source F ids do not overlap next sparse-history ids"
            )
        next_batch["latent"] = next_latent
        next_batch["_generated_history_sample_mask"] = replaced_samples
        return next_batch, {
            "generated_history_active": 1.0,
            "generated_history_eligible_fraction": float(
                eligible.float().mean().item()
            ),
            "generated_history_sample_fraction": float(
                replaced_samples.float().mean().item()
            ),
            "generated_history_replaced_frames": float(replaced),
            "generated_history_delta": (
                delta_sum / float(replaced) if replaced else 0.0
            ),
            "generated_history_blend": blend,
            "generated_history_denoise_steps": float(
                getattr(self.tcfg, "generated_history_denoise_steps", 2)
            ),
        }

    def _build_generated_history_batch(
        self,
        batch: dict,
    ) -> tuple[dict, dict[str, float]]:
        """Unroll detached chunks and train from the accumulated model history."""
        chain = batch.get("generated_history_chain")
        if not isinstance(chain, list) or not chain:
            raise RuntimeError(
                "generated-history training is active but the dataset did not "
                "provide a non-empty generated_history_chain"
            )
        synchronized_chain_depth = len(chain)
        unroll_chunks = select_generated_history_unroll_chunks(
            global_step=self._global_step,
            configured_horizons=getattr(
                self.tcfg,
                "generated_history_unroll_chunks",
                [1],
            ),
            available_chunks=synchronized_chain_depth,
            seed=int(getattr(self.tcfg, "generated_history_seed", 0)),
        )
        current_batch = batch
        total_replaced = 0.0
        weighted_delta = 0.0
        eligible_fraction_sum = 0.0
        sample_fraction_sum = 0.0
        final_logs: dict[str, float] = {}
        for step in range(unroll_chunks):
            current_batch, step_logs = self._advance_generated_history_batch(
                current_batch,
                chain[step],
            )
            replaced = float(step_logs["generated_history_replaced_frames"])
            total_replaced += replaced
            weighted_delta += float(step_logs["generated_history_delta"]) * replaced
            eligible_fraction_sum += float(
                step_logs["generated_history_eligible_fraction"]
            )
            sample_fraction_sum += float(
                step_logs["generated_history_sample_fraction"]
            )
            final_logs = step_logs

        final_logs["generated_history_eligible_fraction"] = (
            eligible_fraction_sum / float(unroll_chunks)
        )
        final_logs["generated_history_sample_fraction"] = (
            sample_fraction_sum / float(unroll_chunks)
        )
        final_logs["generated_history_replaced_frames"] = total_replaced
        final_logs["generated_history_delta"] = (
            weighted_delta / total_replaced if total_replaced else 0.0
        )
        final_logs["generated_history_unroll_chunks"] = float(unroll_chunks)
        available = batch.get("generated_history_available_chunks")
        final_logs["generated_history_available_chunks"] = (
            float(available.float().mean().item())
            if isinstance(available, torch.Tensor)
            else float(synchronized_chain_depth)
        )
        return current_batch, final_logs

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Compute rectified flow loss with training/inference-aligned framing.

        Matches the inference-time setup in wan_world_model.py:step():
          * Full-window mode uses history_length + pred_frames latent frames
          * History/condition frames are near-clean state anchors
          * Future frames receive full random sigma
          * Velocity loss computed ONLY on future frames

        Also applies scheduled condition dropout. Text and action dropout are
        independent, but action dropout should normally remain zero for
        simulator training.
        """
        generated_history_logs = {
            "generated_history_active": 0.0,
            "generated_history_eligible_fraction": 0.0,
            "generated_history_sample_fraction": 0.0,
            "generated_history_replaced_frames": 0.0,
            "generated_history_delta": 0.0,
            "generated_history_unroll_chunks": 0.0,
            "generated_history_available_chunks": 0.0,
        }
        if (
            self._dit.training
            and generated_history_step_is_active(
                global_step=self._global_step,
                enabled=bool(
                    getattr(self.tcfg, "generated_history_enabled", False)
                ),
                start_step=int(
                    getattr(self.tcfg, "generated_history_start_step", 0)
                ),
                probability=float(
                    getattr(self.tcfg, "generated_history_probability", 0.0)
                ),
                seed=int(getattr(self.tcfg, "generated_history_seed", 0)),
            )
        ):
            batch, generated_history_logs = (
                self._build_generated_history_batch(batch)
            )

        # Full dataset window: (B, T_full, C, H, W) → (B, C, T_full, H, W)
        # T_full = num_history + num_frames (e.g. 11)
        latent_full = batch["latent"].to(device=self.device, dtype=self.dtype)
        latent_full = latent_full.permute(0, 2, 1, 3, 4)
        B, C, T_full, H, W = latent_full.shape

        full_window = full_window_enabled(self.tcfg) or full_window_enabled(self.mcfg)
        if full_window:
            # Train on the full history+future window. History frames are noisy
            # conditions, and loss is computed only on future frames.
            start = 0
            end = T_full
            latent = latent_full
            history_frames = min(max(1, int(self.mcfg.history_length)), T_full - 1)
        else:
            # Default CoachWorld short-window mode: one current anchor + future.
            T_model = self.mcfg.pred_frames
            start = max(0, self.mcfg.history_length - 1)
            end = start + T_model
            if end > T_full:
                # Defensive clamp (dataset window might be shorter than expected)
                start = max(0, T_full - T_model)
                end = T_full
            latent = latent_full[:, :, start:end, :, :]  # (B, C, T_model, H, W)
            history_frames = 1
        T = latent.shape[2]

        viewmats = None
        Ks = None
        if bool(getattr(self.mcfg, "prope_enabled", False)):
            if "viewmats" not in batch or "Ks" not in batch:
                raise RuntimeError(
                    "PRoPE is enabled but batch has no viewmats/Ks. "
                    "Set wm_training.video_latent_camera_conditioning=true."
                )
            viewmats = batch["viewmats"][:, start:end].to(
                device=self.device, dtype=self.dtype
            )
            Ks = batch["Ks"][:, start:end].to(device=self.device, dtype=self.dtype)
        eef_projection_kwargs = self._eef_projection_kwargs(batch, start, end)

        # Sample FUTURE sigma from scheduler (full range). Cast to self.dtype so
        # downstream arithmetic stays in bf16 — avoids fp32 gradients flowing
        # into bf16 model params (which crashes backward).
        num_sigmas = len(self._scheduler.sigmas)
        timestep_id = torch.randint(0, num_sigmas, (B,), device=self.device)
        timestep = self._scheduler.timesteps[timestep_id].to(
            dtype=self.dtype, device=self.device
        )  # (B,)
        sigma_future = self._scheduler.sigmas[timestep_id].to(
            dtype=self.dtype, device=self.device
        )  # (B,)
        sigma_future_bc = sigma_future.view(B, 1, 1, 1, 1)

        # Sample CURRENT/history sigma: small random noise in [0, history_noise_max]
        # so at inference (sigma=0 on current frame) the input stays in the
        # training distribution.
        hn_max = float(self.tcfg.history_noise_max)
        sigma_history = torch.rand(B, device=self.device, dtype=self.dtype) * hn_max
        generated_history_sample_mask = batch.get(
            "_generated_history_sample_mask"
        )
        if generated_history_sample_mask is not None:
            generated_history_sample_mask = generated_history_sample_mask.to(
                device=self.device,
                dtype=torch.bool,
            )
            sigma_history = sigma_history.masked_fill(
                generated_history_sample_mask,
                0.0,
            )
        sigma_history_bc = sigma_history.view(B, 1, 1, 1, 1)

        # Partition the sliced latent into history/condition vs future target.
        current_clean = latent[:, :, :history_frames, :, :]
        future_clean = latent[:, :, history_frames:, :, :]
        # --- Action conditioning ---
        actions = batch["action"]  # (B, N_raw, action_dim) or list
        action_domain_ids = self._batch_action_domain_ids(
            batch,
            action_present=isinstance(actions, torch.Tensor),
        )
        action_dense_cond = None
        if isinstance(actions, torch.Tensor):
            action_rate = str(
                getattr(self.tcfg, "action_condition_timestep_rate", "raw")
            )
            a_start = start * self.VAE_TEMPORAL_STRIDE
            dense_len = num_video_frames_for_latent_window(
                T, self.VAE_TEMPORAL_STRIDE
            )
            a_dense_end = min(a_start + dense_len, actions.shape[1])
            actions_dense_sliced = actions[:, a_start:a_dense_end, ...].contiguous()
            action_dense_cond = self._prepare_batch_actions(actions_dense_sliced, T)
            if action_rate == "latent":
                actions_sliced = actions[
                    :, a_start:a_dense_end:self.VAE_TEMPORAL_STRIDE, ...
                ][:, :T, ...].contiguous()
            else:
                actions_sliced = actions_dense_sliced
        else:
            actions_sliced = actions
        if (
            isinstance(actions_sliced, torch.Tensor)
            and str(getattr(self.tcfg, "action_condition_timestep_rate", "raw")) == "latent"
        ):
            action_cond = self._align_latent_rate_actions(
                actions_sliced.to(device=self.device, dtype=self.dtype), T
            )
        else:
            action_cond = self._prepare_batch_actions(actions_sliced, T)
            if action_dense_cond is None:
                action_dense_cond = action_cond
        action_cond_full_window = action_cond
        dense_zero_first_chunk = False
        dense_zero_prefix_chunks = 0
        if (
            action_cond is not None
            and bool(
                getattr(self.tcfg, "future_only_action_conditioning", False)
                or getattr(self.mcfg, "future_only_action_conditioning", False)
            )
        ):
            # Future-only action conditioning: drop sparse K/V tokens that align
            # with condition/history frames. Zeroing is not strict because the
            # action K/V projections have biases.
            action_cond, action_mask = apply_kv_future_only_action_mask(
                action_cond,
                history_frames=history_frames,
                action_rate=str(getattr(self.tcfg, "action_condition_timestep_rate", "raw")),
                vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
            )
            dense_zero_first_chunk = action_mask.dense_zero_first_chunk
            dense_zero_prefix_chunks = action_mask.dense_zero_prefix_chunks

        # --- History (current frame) noise: always standard Gaussian. ---
        noise_current = torch.randn_like(current_clean)
        if bool(
            getattr(self.tcfg, "gt_condition_replacement", False)
            or getattr(self.mcfg, "gt_condition_replacement", False)
        ):
            # Cosmos-style frame replacement: condition frames are kept exact.
            # We still train the velocity only on future frames, so the current
            # frame is pure conditioning context.
            noisy_current = current_clean
        else:
            corrupted_current = (
                (1.0 - sigma_history_bc) * current_clean
                + sigma_history_bc * noise_current
            )
            clean_prob = float(getattr(self.tcfg, "history_clean_prob", 0.0))
            if clean_prob > 0.0:
                use_clean = torch.rand(B, device=self.device) < clean_prob
                noisy_current = torch.where(
                    use_clean.view(B, 1, 1, 1, 1),
                    current_clean,
                    corrupted_current,
                )
            else:
                noisy_current = corrupted_current

        # Standard rectified-flow endpoint. Action/state is a condition only;
        # it never enters the source/target.
        noise_base = torch.randn_like(future_clean)
        x1_future = noise_base

        noisy_future = (
            (1.0 - sigma_future_bc) * future_clean + sigma_future_bc * x1_future
        )
        noisy_latent = torch.cat([noisy_current, noisy_future], dim=2)
        condition_mask = None
        if bool(
            getattr(self.tcfg, "condition_mask_enabled", False)
            or getattr(self.mcfg, "condition_mask_enabled", False)
        ):
            condition_mask = torch.zeros(
                B, 1, T, H, W, device=self.device, dtype=self.dtype
            )
            condition_mask[:, :, :history_frames, :, :] = 1.0

        # Target velocity (future frames only — current frame has no loss signal)
        target_future = x1_future - future_clean

        # ------------------------------------------------------------------
        # Text embedding
        # ------------------------------------------------------------------
        text_emb = self._get_text_embedding(batch["text"])  # (B, L, D)

        # ------------------------------------------------------------------
        # Scheduled training-time condition dropout.
        # ------------------------------------------------------------------
        # Text: zero the embedding for dropped samples (cross-attn still attends
        # to those slots but their K/V is ~0 after WAN's text_embedding MLP of
        # all-zero input → effectively learned null embedding).
        drop_text_mask = None
        cfg_text_rate = self._condition_dropout_rate("cfg_dropout_text")
        if cfg_text_rate > 0.0:
            drop_text_mask = (torch.rand(B, device=self.device) < cfg_text_rate)
            if drop_text_mask.any():
                # "Null text" is NOT strict zeros — passing zeros through the
                # pretrained self.text_embedding MLP yields a constant bias
                # vector repeated over all 512 positions, which degenerates
                # the cross-attn K/V (every text slot identical) and can
                # NaN flash_attn's bf16 softmax when head_dim is 128.
                # Small N(0, 0.01) noise preserves "~no text signal" semantics
                # while breaking position symmetry. Works uniformly in train,
                # val, and 3-branch CFG inference.
                null_text = torch.randn_like(text_emb) * 0.01
                expand_shape = (B,) + (1,) * (text_emb.dim() - 1)
                text_emb = torch.where(
                    drop_text_mask.view(*expand_shape), null_text, text_emb
                )

        # Action: pass a per-sample drop mask — the DiT zeros action tokens
        # for dropped samples inside _forward (zero V contributes zero to
        # cross-attn output, satisfying CFG "no-action" semantics without
        # the varlen flash_attn path which caused bf16 NaN).
        drop_act_mask = None
        cfg_action_rate = self._condition_dropout_rate("cfg_dropout_action")
        if cfg_action_rate > 0.0 and action_cond is not None:
            drop_act_mask = (
                torch.rand(B, device=self.device) < cfg_action_rate
            )

        eef_spatial = self._eef_spatial_condition(
            batch,
            start,
            end,
            latent_height=H,
            latent_width=W,
            action_drop_mask=drop_act_mask,
        )
        cond_concat = self._concat_condition_channels(condition_mask, eef_spatial)

        # Positional encoding seq_len
        pt, ph, pw = self.PATCH_SIZE
        seq_len = (T // pt) * (H // ph) * (W // pw)

        # ------------------------------------------------------------------
        # DiT forward — future timestep is the primary signal; the model
        # interprets slice index 0 as "already-nearly-denoised" context.
        # ``return_aux=True`` also returns an auxiliary action prediction
        # from the last-block visual tokens (LDA-1B style). That signal
        # forces the DiT's action cross-attn path to actually encode useful
        # information — without it, text + history over-determine the next
        # frame on most samples and the gate collapses toward zero.
        # ------------------------------------------------------------------
        with torch.autocast("cuda", dtype=self.dtype):
            aux_weight = float(getattr(self.tcfg, "aux_action_weight", 0.0))
            want_aux = aux_weight > 0.0 and action_cond is not None
            if want_aux:
                pred_list, aux_action_pred = self._dit(
                    [noisy_latent[i] for i in range(B)],
                    timestep,
                    text_emb,
                    seq_len,
                    cond_concat=cond_concat,
                    action_seq=action_cond,
                    action_dense_seq=action_dense_cond,
                    action_dense_zero_first_chunk=dense_zero_first_chunk,
                    action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                    action_drop_mask=drop_act_mask,
                    action_domain_ids=action_domain_ids,
                    viewmats=viewmats,
                    Ks=Ks,
                    **eef_projection_kwargs,
                    return_aux=True,
                )
            else:
                pred_list = self._dit(
                    [noisy_latent[i] for i in range(B)],
                    timestep,
                    text_emb,
                    seq_len,
                    cond_concat=cond_concat,
                    action_seq=action_cond,
                    action_dense_seq=action_dense_cond,
                    action_dense_zero_first_chunk=dense_zero_first_chunk,
                    action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                    action_drop_mask=drop_act_mask,
                    action_domain_ids=action_domain_ids,
                    viewmats=viewmats,
                    Ks=Ks,
                    **eef_projection_kwargs,
                )
                aux_action_pred = None
            # WanModelAction returns fp32 tensors after unpatchify. Keep a fp32
            # copy for counterfactual diagnostics/loss; small action-velocity
            # residuals can disappear if both true and alt predictions are
            # quantized to bf16 before the comparison.
            pred_fp32 = torch.stack(pred_list).float()  # (B, C, T, H, W)
            pred = pred_fp32.to(dtype=self.dtype)
        dense_action_stats = self._collect_dense_action_stats()

        # Main loss: future frames only. The current/anchor frame is input-only.
        pred_future = pred[:, :, history_frames:, :, :]
        main_loss = F.mse_loss(pred_future, target_future) * self.tcfg.loss_scale
        loss = main_loss

        # Auxiliary action reconstruction loss. Target = per-latent-frame
        # action. We subsample the video-frame-rate action_cond at the VAE
        # temporal stride (=4) so the target shape matches aux_action_pred's
        # (B, Fp, action_dim), where Fp=T (patch_t=1, no temporal patching).
        #
        # Mask: only supervise samples where action was NOT CFG-dropped.
        # On dropped samples, the DiT never received action input, so
        # asking it to reconstruct action would force the model to
        # hallucinate — poisoning training. On kept samples, the aux
        # gradient flows: (action → encoder → K/V → cross-attn → visual
        # → decoder → action), ensuring the whole chain is load-bearing.
        aux_loss_val = 0.0
        if aux_action_pred is not None:
            aux_loss = self._compute_aux_action_loss(
                aux_action_pred,
                action_cond_full_window,
                T_latent=T,
                history_frames=history_frames,
                drop_act_mask=drop_act_mask,
            )
            aux_loss_val = aux_loss.item()
            loss = loss + aux_weight * aux_loss

        cf_loss_val = 0.0
        cf_gap_val = 0.0
        cf_true_mse_val = 0.0
        cf_alt_mse_val = 0.0
        cf_pred_diff_val = 0.0
        cf_pred_diff_abs_mean_val = 0.0
        cf_pred_diff_abs_max_val = 0.0
        cf_alt_push_loss_val = 0.0
        cf_action_delta_val = 0.0
        cf_dense_action_delta_val = 0.0
        cf_active = False
        cf_mode = str(
            getattr(self.tcfg, "counterfactual_action_mode", "roll")
        )
        cf_weight = float(
            getattr(self.tcfg, "counterfactual_action_weight", 0.0)
        )
        cf_interval = max(
            1, int(getattr(self.tcfg, "counterfactual_action_interval", 1))
        )
        cf_active = (
            cf_weight > 0.0
            and action_cond is not None
            and ((self._global_step + 1) % cf_interval == 0)
        )
        if cf_active:
            action_alt_dense = None
            action_alt_domain_ids = action_domain_ids
            if cf_mode == "roll":
                if B <= 1:
                    action_alt = None
                else:
                    action_alt = action_cond.roll(shifts=1, dims=0)
                    action_alt_dense = action_dense_cond.roll(shifts=1, dims=0)
                    if action_domain_ids is not None:
                        action_alt_domain_ids = action_domain_ids.roll(shifts=1, dims=0)
            elif cf_mode == "time_reverse":
                action_alt = action_cond.flip(dims=(1,))
                action_alt_dense = action_dense_cond.flip(dims=(1,))
            elif cf_mode == "sign_flip":
                action_alt = -action_cond
                action_alt_dense = -action_dense_cond
            elif cf_mode == "zero":
                action_alt = torch.zeros_like(action_cond)
                action_alt_dense = torch.zeros_like(action_dense_cond)
            else:
                raise ValueError(
                    f"Unknown counterfactual_action_mode={cf_mode!r}; "
                    "expected roll, time_reverse, sign_flip, or zero"
                )
            if action_alt is not None:
                cf_action_delta_val = F.mse_loss(
                    action_cond.float(), action_alt.float()
                ).item()
                if hasattr(self._dit, "_chunk_actions_for_latent_frames"):
                    with torch.no_grad():
                        true_chunks = self._dit._chunk_actions_for_latent_frames(
                            action_dense_cond, T
                        )
                        alt_chunks = self._dit._chunk_actions_for_latent_frames(
                            action_alt_dense, T
                        )
                        cf_dense_action_delta_val = F.mse_loss(
                            true_chunks.float(), alt_chunks.float()
                        ).item()
        if cf_active and action_alt is not None:
            with torch.autocast("cuda", dtype=self.dtype):
                pred_shuf_list = self._dit(
                    [noisy_latent[i] for i in range(B)],
                    timestep,
                    text_emb,
                    seq_len,
                    cond_concat=cond_concat,
                    action_seq=action_alt,
                    action_dense_seq=action_alt_dense,
                    action_dense_zero_first_chunk=dense_zero_first_chunk,
                    action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                    action_drop_mask=None,
                    action_domain_ids=action_alt_domain_ids,
                    viewmats=viewmats,
                    Ks=Ks,
                    **eef_projection_kwargs,
                )
                pred_alt_fp32 = torch.stack(pred_shuf_list).float()
            # Counterfactual comparisons need fp32 precision. In bf16 the
            # true/alt MSE often quantizes to exactly the same scalar even
            # when the predictions differ, making the margin loss look stuck.
            pred_future_cf = pred_fp32[:, :, history_frames:, :, :]
            pred_alt_future_cf = pred_alt_fp32[:, :, history_frames:, :, :]
            target_future_cf = target_future.float()
            pred_true_ref_cf = pred_future_cf.detach()
            per_sample_true_mse_cf = F.mse_loss(
                pred_future_cf, target_future_cf, reduction="none"
            ).mean(dim=(1, 2, 3, 4))
            per_sample_alt_mse = F.mse_loss(
                pred_alt_future_cf, target_future_cf, reduction="none"
            ).mean(dim=(1, 2, 3, 4))
            per_sample_pred_diff = F.mse_loss(
                pred_true_ref_cf, pred_alt_future_cf, reduction="none"
            ).mean(dim=(1, 2, 3, 4))
            pred_abs_diff = (pred_true_ref_cf - pred_alt_future_cf).abs()
            margin = float(
                getattr(self.tcfg, "counterfactual_action_margin", 0.02)
            )
            if drop_act_mask is not None:
                keep = (~drop_act_mask).to(dtype=per_sample_true_mse_cf.dtype)
            else:
                keep = torch.ones_like(per_sample_true_mse_cf)
            denom = keep.sum().clamp(min=1.0)
            per_sample_gap = per_sample_alt_mse - per_sample_true_mse_cf
            cf_loss = (
                F.relu(margin - per_sample_gap)
                * keep
            ).sum() / denom
            true_mse = (per_sample_true_mse_cf * keep).sum() / denom
            alt_mse = (per_sample_alt_mse * keep).sum() / denom
            pred_diff = (per_sample_pred_diff * keep).sum() / denom
            pred_abs_diff_mean = pred_abs_diff.mean()
            pred_abs_diff_max = pred_abs_diff.max()
            cf_gap = alt_mse - true_mse
            cf_loss_val = cf_loss.item()
            cf_gap_val = cf_gap.item()
            cf_true_mse_val = true_mse.item()
            cf_alt_mse_val = alt_mse.item()
            cf_pred_diff_val = pred_diff.item()
            cf_pred_diff_abs_mean_val = pred_abs_diff_mean.item()
            cf_pred_diff_abs_max_val = pred_abs_diff_max.item()
            cf_alt_push_loss_val = cf_loss.item()
            loss = loss + cf_weight * cf_loss

        log_dict = {
            "loss": loss.item(),
            "loss_main": main_loss.item(),
            "loss_aux_action": aux_loss_val,
            "loss_counterfactual_action": cf_loss_val,
            "counterfactual_action_gap": cf_gap_val,
            "counterfactual_true_mse": cf_true_mse_val,
            "counterfactual_alt_mse": cf_alt_mse_val,
            "counterfactual_pred_diff": cf_pred_diff_val,
            "counterfactual_pred_diff_abs_mean": cf_pred_diff_abs_mean_val,
            "counterfactual_pred_diff_abs_max": cf_pred_diff_abs_max_val,
            "counterfactual_alt_push_loss": cf_alt_push_loss_val,
            "counterfactual_action_delta": cf_action_delta_val,
            "counterfactual_dense_action_delta": cf_dense_action_delta_val,
            "counterfactual_action_mode": cf_mode if cf_weight > 0.0 else "off",
            "counterfactual_action_active": int(cf_active),
            "counterfactual_action_interval": cf_interval,
            "sigma_future": sigma_future.mean().item(),
            "sigma_history": sigma_history.mean().item(),
            "latent_mean": latent.mean().item(),
            "latent_std": latent.std().item(),
            "cfg_drop_text_prob": cfg_text_rate,
            "cfg_drop_action_prob": cfg_action_rate,
            "cfg_drop_text_rate": (
                drop_text_mask.float().mean().item() if drop_text_mask is not None else 0.0
            ),
            "cfg_drop_action_rate": (
                drop_act_mask.float().mean().item() if drop_act_mask is not None else 0.0
            ),
        }
        log_dict.update(generated_history_logs)
        log_dict.update(dense_action_stats)
        return loss, log_dict

    def _prepare_batch_actions(
        self, actions, T_latent: int
    ) -> Optional[torch.Tensor]:
        """Prepare batched action conditioning tensor.

        Args:
            actions: (B, N_raw, action_dim) tensor or list of arrays.
            T_latent: Number of latent temporal frames.

        Returns:
            (B, N_video_frames, action_dim) tensor fed to WanModelAction via
            the action_seq kwarg, or None if no actions.
        """
        # Number of raw video frames corresponding to T_latent latent frames
        num_video_frames = (T_latent - 1) * self.VAE_TEMPORAL_STRIDE + 1

        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        elif isinstance(actions, list):
            actions_np = [a.numpy() if isinstance(a, torch.Tensor) else a for a in actions]
        else:
            return None

        batch_tensors = []
        for i in range(len(actions_np)):
            a = actions_np[i] if isinstance(actions_np, list) else actions_np[i]
            t = prepare_action_tensor(a, num_video_frames, self.device, dtype=torch.float32)
            if t is not None:
                batch_tensors.append(t)

        if not batch_tensors:
            return None

        return self._format_action_condition(torch.cat(batch_tensors, dim=0))

    def _batch_action_domain_ids(self, batch: dict, action_present: bool) -> Optional[torch.Tensor]:
        if not action_present:
            return None
        domain_features_enabled = bool(
            getattr(
                self.tcfg,
                "domain_aware_action_projection_enabled",
                getattr(self.mcfg, "domain_aware_action_projection_enabled", False),
            )
            or bool(
                getattr(
                    self.tcfg,
                    "domain_aware_group_projection_enabled",
                    getattr(self.mcfg, "domain_aware_group_projection_enabled", False),
                )
            )
            or int(
                getattr(
                    self.tcfg,
                    "action_domain_prompt_tokens",
                    getattr(self.mcfg, "action_domain_prompt_tokens", 0),
                )
            )
            > 0
        )
        if "domain_id" not in batch:
            if domain_features_enabled:
                raise ValueError(
                    "batch is missing domain_id but domain-aware action features are enabled"
                )
            return None
        return batch["domain_id"].to(self.device, dtype=torch.long)

    def _format_action_condition(self, action: torch.Tensor) -> torch.Tensor:
        schema = str(getattr(self.tcfg, "action_schema", getattr(self.mcfg, "action_schema", "fixed")))
        if schema == "fixed":
            if action.dim() != 3:
                raise ValueError(f"fixed action condition must be 3D, got {tuple(action.shape)}")
            return action
        if schema == "arm_slot":
            slots = int(getattr(self.tcfg, "max_arm_slots", getattr(self.mcfg, "max_arm_slots", 2)))
            action_dim = int(getattr(self.mcfg, "action_dim", action.shape[-1] - 1))
            if action.dim() == 4:
                if action.shape[2] != slots or action.shape[3] != action_dim + 1:
                    raise ValueError(
                        "arm-slot action condition shape mismatch: "
                        f"got {tuple(action.shape)}, expected (B,T,{slots},{action_dim + 1})"
                    )
                return action
            raise ValueError(
                "arm-slot action condition must be slot-aware (B,T,S,D+1); "
                f"got {tuple(action.shape)}"
            )
        raise ValueError(f"Unsupported action_schema={schema!r}")

    def _align_latent_rate_actions(
        self, actions: torch.Tensor, T_latent: int
    ) -> torch.Tensor:
        """Pad/truncate latent-rate actions to match the DiT latent window."""
        if actions.shape[1] == T_latent:
            return self._format_action_condition(actions)
        if actions.shape[1] == 0:
            empty_shape = list(actions.shape)
            empty_shape[1] = T_latent
            empty = torch.zeros(*empty_shape, device=actions.device, dtype=actions.dtype)
            return self._format_action_condition(empty)
        if actions.shape[1] < T_latent:
            pad_shape = list(actions.shape)
            pad_shape[1] = T_latent - actions.shape[1]
            pad = actions[:, -1:].expand(*pad_shape)
            return self._format_action_condition(torch.cat([actions, pad], dim=1))
        return self._format_action_condition(actions[:, :T_latent])

    def _latent_frame_action_targets(
        self,
        action_condition: torch.Tensor,
        T_latent: int,
    ) -> torch.Tensor:
        """Sample one action condition target per Wan latent frame.

        For raw-frame conditions this uses the last raw frame of each Wan
        causal latent group: [0, 4, 8, ...] for stride=4.  That matches the
        previous fixed-action aux target while making the causal grouping
        explicit and valid for slot-aware tensors too.
        """
        if str(getattr(self.tcfg, "action_condition_timestep_rate", "raw")) == "latent":
            target = action_condition[:, :T_latent, ...]
        else:
            indices = wan_causal_group_indices(
                T_latent,
                vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
                device=action_condition.device,
            )[:, -1]
            needed = int(indices.max().item()) + 1
            if action_condition.shape[1] < needed:
                if action_condition.shape[1] == 0:
                    shape = list(action_condition.shape)
                    shape[1] = needed
                    action_condition = torch.zeros(
                        *shape,
                        device=action_condition.device,
                        dtype=action_condition.dtype,
                    )
                else:
                    pad_shape = list(action_condition.shape)
                    pad_shape[1] = needed - action_condition.shape[1]
                    pad = action_condition[:, -1:, ...].expand(*pad_shape)
                    action_condition = torch.cat([action_condition, pad], dim=1)
            target = action_condition.index_select(1, indices)
        if target.shape[1] < T_latent:
            pad_shape = list(target.shape)
            pad_shape[1] = T_latent - target.shape[1]
            pad = target[:, -1:, ...].expand(*pad_shape)
            target = torch.cat([target, pad], dim=1)
        elif target.shape[1] > T_latent:
            target = target[:, :T_latent, ...]
        return target

    def _reduce_aux_per_sample(
        self,
        per_sample_loss: torch.Tensor,
        drop_act_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if drop_act_mask is None:
            return per_sample_loss.mean()
        keep = (~drop_act_mask).to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)
        denom = keep.sum().clamp(min=1.0)
        return (per_sample_loss * keep).sum() / denom

    def _compute_aux_action_loss(
        self,
        aux_action_pred,
        action_condition: torch.Tensor,
        *,
        T_latent: int,
        history_frames: int,
        drop_act_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        schema = str(getattr(self.tcfg, "action_schema", getattr(self.mcfg, "action_schema", "fixed")))
        target = self._latent_frame_action_targets(action_condition, T_latent)
        if schema == "fixed":
            if not torch.is_tensor(aux_action_pred):
                raise TypeError("fixed auxiliary action decoder must return a tensor")
            if target.dim() != 3:
                raise ValueError(f"fixed aux target must be (B,T,D), got {tuple(target.shape)}")
            target = target.to(device=aux_action_pred.device, dtype=aux_action_pred.dtype)
            per_frame_mse = F.mse_loss(aux_action_pred, target, reduction="none").mean(dim=2)
            if bool(
                getattr(self.tcfg, "future_only_action_conditioning", False)
                or getattr(self.mcfg, "future_only_action_conditioning", False)
            ):
                frame_keep = torch.ones_like(per_frame_mse)
                frame_keep[:, :history_frames] = 0.0
                per_sample = (per_frame_mse * frame_keep).sum(dim=1) / frame_keep.sum(
                    dim=1
                ).clamp(min=1.0)
            else:
                per_sample = per_frame_mse.mean(dim=1)
            return self._reduce_aux_per_sample(per_sample, drop_act_mask)

        if schema == "arm_slot":
            if not isinstance(aux_action_pred, dict):
                raise TypeError("arm-slot auxiliary action decoder must return a dict")
            values_pred = aux_action_pred["values"]
            mask_logits = aux_action_pred["mask_logits"]
            if target.dim() != 4:
                raise ValueError(f"arm-slot aux target must be (B,T,S,D+1), got {tuple(target.shape)}")
            action_dim = int(getattr(self.mcfg, "action_dim", target.shape[-1] - 1))
            target = target.to(device=values_pred.device, dtype=values_pred.dtype)
            values_target = target[..., :action_dim]
            mask_target = target[..., action_dim : action_dim + 1].clamp(0.0, 1.0)
            if values_pred.shape != values_target.shape:
                raise ValueError(
                    f"arm-slot aux values shape mismatch: pred={tuple(values_pred.shape)} "
                    f"target={tuple(values_target.shape)}"
                )
            if mask_logits.shape != mask_target.shape:
                raise ValueError(
                    f"arm-slot aux mask shape mismatch: pred={tuple(mask_logits.shape)} "
                    f"target={tuple(mask_target.shape)}"
                )
            slot_weight = mask_target.squeeze(-1)
            if bool(
                getattr(self.tcfg, "future_only_action_conditioning", False)
                or getattr(self.mcfg, "future_only_action_conditioning", False)
            ):
                frame_keep = torch.ones(
                    slot_weight.shape[:2],
                    device=slot_weight.device,
                    dtype=slot_weight.dtype,
                )
                frame_keep[:, :history_frames] = 0.0
                slot_weight = slot_weight * frame_keep[:, :, None]
            pose_mse = F.mse_loss(values_pred, values_target, reduction="none").mean(dim=-1)
            pose_denom = slot_weight.sum(dim=(1, 2)).clamp(min=1.0)
            pose_per_sample = (pose_mse * slot_weight).sum(dim=(1, 2)) / pose_denom

            mask_bce = F.binary_cross_entropy_with_logits(
                mask_logits.float(),
                mask_target.float(),
                reduction="none",
            ).squeeze(-1)
            if bool(
                getattr(self.tcfg, "future_only_action_conditioning", False)
                or getattr(self.mcfg, "future_only_action_conditioning", False)
            ):
                mask_bce = mask_bce * frame_keep[:, :, None]
                mask_denom = frame_keep.sum(dim=1).clamp(min=1.0) * mask_bce.shape[2]
                mask_per_sample = mask_bce.sum(dim=(1, 2)) / mask_denom
            else:
                mask_per_sample = mask_bce.mean(dim=(1, 2))
            mask_weight = float(getattr(self.tcfg, "aux_action_mask_weight", 0.05))
            per_sample = pose_per_sample + mask_weight * mask_per_sample.to(
                dtype=pose_per_sample.dtype
            )
            return self._reduce_aux_per_sample(per_sample, drop_act_mask)

        raise ValueError(f"Unsupported action_schema={schema!r}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        """Single training step with gradient accumulation.

        Returns log_dict with loss and grad_norm.
        """
        action_warmup_steps = int(getattr(self.tcfg, "action_warmup_steps", 0))
        action_warmup_active = self._global_step < action_warmup_steps
        loss, log_dict = self.compute_loss(batch)
        scaled_loss = loss / self.mcfg.grad_accum_steps
        scaled_loss.backward()
        if action_warmup_active:
            for name, p in self._dit.named_parameters():
                if p.grad is not None and not self._is_action_side_param(name):
                    p.grad = None
            log_dict["action_warmup_active"] = 1
        else:
            log_dict["action_warmup_active"] = 0
        log_dict["action_warmup_steps"] = action_warmup_steps

        self._accum_count += 1

        # Step optimizer every grad_accum_steps
        if self._accum_count % self.mcfg.grad_accum_steps == 0:
            # Per-block gradient norm monitoring (runs every diagnose_interval
            # optimizer steps). Cheap — just walks the parameter list post-
            # backward, before zero_grad.
            if (
                self.tcfg.diagnose_interval > 0
                and ((self._global_step + 1) % self.tcfg.diagnose_interval == 0)
            ):
                log_dict.update(self._compute_block_grad_norms())

            inf_count = 0
            large_count = 0
            clamp_val = float(self.tcfg.gradient_clip) * 10.0
            for p in self._dit.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                finite = torch.isfinite(g)
                inf_count += int((~finite).sum().item())
                if self.tcfg.gradient_clip > 0:
                    large_count += int(
                        (finite & (g.abs() > clamp_val)).sum().item()
                    )

            grad_health = torch.tensor(
                [inf_count, large_count],
                device=self.device,
                dtype=torch.int64,
            )
            if dist.is_initialized():
                dist.all_reduce(grad_health, op=dist.ReduceOp.SUM)
            inf_count, large_count = map(int, grad_health.tolist())
            log_dict["grad_inf_count"] = inf_count
            log_dict["grad_large_count"] = large_count

            skip_optimizer_step = inf_count > 0
            log_dict["optimizer_step_skipped_nonfinite"] = int(
                skip_optimizer_step
            )

            if not skip_optimizer_step and self.tcfg.gradient_clip > 0:
                # Clamp extreme finite grads before computing the total norm.
                # Non-finite grads skip the entire distributed optimizer step;
                # replacing them element-wise would produce a finite but
                # meaningless update direction.
                for p in self._dit.parameters():
                    if p.grad is not None:
                        p.grad.clamp_(min=-clamp_val, max=clamp_val)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._dit.parameters(), max_norm=self.tcfg.gradient_clip
                )
                log_dict["grad_norm"] = (
                    grad_norm.item() if torch.is_tensor(grad_norm)
                    else float(grad_norm)
                )

            if not skip_optimizer_step:
                self._optimizer.step()
                log_dict.update(self._sanitize_action_side_parameters())
            self._optimizer.zero_grad()

            if not skip_optimizer_step and self._lr_scheduler is not None:
                self._lr_scheduler.step()
                for g in self._optimizer.param_groups:
                    tag = g.get("name", "group")
                    log_dict[f"lr_{tag}"] = g["lr"]

        if torch.cuda.is_available():
            gib = float(1024**3)
            log_dict["gpu_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / gib
            log_dict["gpu_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / gib

        return log_dict

    @torch.no_grad()
    def _sanitize_action_side_parameters(self) -> dict:
        """Keep action-conditioning params finite and input-sensitive.

        We observed a checkpoint where prepared actions changed but
        action_tokens/dense_encoded/AdaLN were identical for all variants:
        root action encoder weights had exploded and dense/AdaLN Linear
        weights were exactly zero. This guard catches that failure mode right
        after optimizer.step(), before a checkpoint can persist it.
        """
        if self._dit is None:
            return {}

        repaired = 0
        clamped = 0
        max_abs = float(getattr(self.tcfg, "action_param_max_abs", 100.0))

        def repair_linear(layer: nn.Linear) -> None:
            nonlocal repaired
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            repaired += 1

        for name, module in self._dit.named_modules():
            if not (
                "action_encoder" in name
                or "eef_projection_encoder" in name
                or "action_dense_encoder" in name
                or "action_adaln_encoder" in name
            ):
                continue
            if isinstance(module, nn.Linear):
                w = module.weight
                if (
                    not torch.isfinite(w).all()
                    or w.detach().float().abs().max().item() > max_abs
                    or w.detach().float().abs().max().item() == 0.0
                ):
                    repair_linear(module)
                    continue
                bad = ~torch.isfinite(w)
                if bool(bad.any()):
                    w.nan_to_num_(nan=0.0, posinf=max_abs, neginf=-max_abs)
                    clamped += int(bad.sum().item())
                too_large = w.abs() > max_abs
                if bool(too_large.any()):
                    w.clamp_(min=-max_abs, max=max_abs)
                    clamped += int(too_large.sum().item())
                if module.bias is not None:
                    b = module.bias
                    bad_b = ~torch.isfinite(b)
                    if bool(bad_b.any()):
                        b.nan_to_num_(nan=0.0, posinf=max_abs, neginf=-max_abs)
                        clamped += int(bad_b.sum().item())
                    too_large_b = b.abs() > max_abs
                    if bool(too_large_b.any()):
                        b.clamp_(min=-max_abs, max=max_abs)
                        clamped += int(too_large_b.sum().item())
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                if module.weight is not None and not torch.isfinite(module.weight).all():
                    nn.init.ones_(module.weight)
                    repaired += 1
                if module.bias is not None and not torch.isfinite(module.bias).all():
                    nn.init.zeros_(module.bias)
                    repaired += 1

        return {
            "action_param_repaired_count": repaired,
            "action_param_clamped_count": clamped,
        }

    @torch.no_grad()
    def _compute_block_grad_norms(self) -> dict:
        """Aggregate gradient L2 norms per component: DiT trunk vs action side.

        "Action side" = every parameter introduced for action conditioning:
        root ``action_encoder`` / ``action_decoder`` MLPs plus per-block
        ``k_action`` / ``v_action`` / ``norm_k_action``.

        Also reports ``v_action_weight_norm_mean`` — the mean L2 norm of the
        per-block ``v_action.weight`` tensors. Since the fixed-scale init
        (``V_ACTION_INIT_SCALE``) starts this tiny, watching it grow is the
        cleanest "is the action path learning" signal — it replaces the
        collapsed-toward-zero ``alpha_action_mean`` gate metric.

        Under FSDP the raw .grad is a local shard, so reported norms are
        per-rank; they still answer the key question "is any wire broken".
        """
        dit_sum = 0.0
        action_sum = 0.0
        dit_count = 0
        action_count = 0
        v_action_w_norms: list[float] = []
        for name, p in self._dit.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            finite = torch.isfinite(g)
            if not bool(finite.all()):
                g = torch.where(finite, g, torch.zeros_like(g))
            g = g.clamp(min=-1.0e4, max=1.0e4)
            sq = float((g * g).sum().item())
            if self._is_action_side_param(name):
                action_sum += sq
                action_count += 1
            else:
                dit_sum += sq
                dit_count += 1
            if name.endswith("v_action.weight"):
                v_action_w_norms.append(float(p.detach().norm().item()))

        out = {
            "grad_l2_dit": (dit_sum ** 0.5),
            "grad_l2_action": (action_sum ** 0.5),
            "grad_params_dit": dit_count,
            "grad_params_action": action_count,
        }
        if v_action_w_norms:
            out["v_action_weight_norm_mean"] = (
                sum(v_action_w_norms) / len(v_action_w_norms)
            )
        return out

    def train(
        self,
        train_dataset,
        val_dataset=None,
    ) -> None:
        """Main training loop.

        Args:
            train_dataset: VideoLatentTrainingDataset for training.
            val_dataset: Optional VideoLatentTrainingDataset for validation.
        """
        output_dir = Path(
            getattr(self.tcfg, 'output_dir', None)
            or self.tcfg.wandb_run_name
            or "experiments/wm_train"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # DataLoader
        sampler = ResumableDistributedMixtureSampler(
            root_sizes=[len(samples) for samples in train_dataset.samples_all],
            probabilities=list(train_dataset.dataset_probs),
            num_replicas=self.world_size,
            rank=self.rank,
            batch_size=self.mcfg.batch_size,
            start_micro_step=self._global_step,
            seed=int(getattr(self.tcfg, "mixture_sampler_seed", 0)),
            samples_per_epoch=int(
                getattr(self.tcfg, "mixture_samples_per_epoch", 0)
            ),
        )
        self._sampling_state = sampler.describe()
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.mcfg.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_video_latent_batch,
        )

        max_steps = self.mcfg.max_train_steps
        # _infinite_loader handles sampler.set_epoch() at each epoch boundary
        loader_iter = _infinite_loader(dataloader)

        logger.info(
            "Starting training: %d micro-steps, batch_size=%d, grad_accum=%d, "
            "target_optimizer_updates=%d, lr=%.1e",
            max_steps,
            self.mcfg.batch_size,
            self.mcfg.grad_accum_steps,
            max_steps // max(1, int(self.mcfg.grad_accum_steps)),
            self.mcfg.learning_rate,
        )
        if self.is_main:
            logger.info(
                "Mixture sampler: seed=%d global_samples_per_epoch=%d "
                "micro_steps_per_epoch=%d start_epoch=%d start_offset=%d "
                "root_draws=%s",
                sampler.seed,
                sampler.global_samples_per_epoch,
                sampler.micro_steps_per_epoch,
                sampler.start_epoch,
                sampler.start_offset,
                dict(zip(train_dataset.dataset_keys, sampler.root_draws_per_epoch)),
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        running_loss = 0.0
        step_start_time = time.time()
        root_interval_counts = torch.zeros(
            len(train_dataset.dataset_roots),
            device=self.device,
            dtype=torch.int64,
        )

        while self._global_step < max_steps:
            batch = next(loader_iter)
            root_interval_counts += torch.bincount(
                batch["dataset_index"].to(device=self.device),
                minlength=len(train_dataset.dataset_roots),
            )

            log_dict = self.train_step(batch)
            running_loss += log_dict["loss"]
            self._global_step += 1

            sampling_metrics: dict[str, float] = {}
            if self._global_step % self.tcfg.log_interval == 0:
                global_root_counts = root_interval_counts.clone()
                if self.world_size > 1:
                    torch.distributed.all_reduce(
                        global_root_counts,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                global_draws = int(global_root_counts.sum().item())
                if self.is_main and global_draws > 0:
                    for key, count in zip(
                        train_dataset.dataset_keys,
                        global_root_counts.tolist(),
                    ):
                        sampling_metrics[f"sampling/{key}_fraction"] = (
                            float(count) / float(global_draws)
                        )
                        sampling_metrics[f"sampling/{key}_draws"] = float(count)
                root_interval_counts.zero_()

            # Logging
            if self._global_step % self.tcfg.log_interval == 0 and self.is_main:
                elapsed = time.time() - step_start_time
                avg_loss = running_loss / self.tcfg.log_interval
                steps_per_sec = self.tcfg.log_interval / elapsed
                watch_keys = [
                    "loss_main",
                    "loss_aux_action",
                    "loss_counterfactual_action",
                    "counterfactual_action_gap",
                    "counterfactual_true_mse",
                    "counterfactual_alt_mse",
                    "counterfactual_pred_diff",
                    "counterfactual_pred_diff_abs_mean",
                    "counterfactual_pred_diff_abs_max",
                    "counterfactual_alt_push_loss",
                    "counterfactual_action_delta",
                    "counterfactual_dense_action_delta",
                    "counterfactual_action_mode",
                    "counterfactual_action_active",
                    "counterfactual_action_interval",
                    "action_warmup_active",
                    "action_warmup_steps",
                    "grad_l2_action",
                    "grad_l2_dit",
                    "grad_params_action",
                    "grad_inf_count",
                    "grad_large_count",
                    "optimizer_step_skipped_nonfinite",
                    "action_param_repaired_count",
                    "action_param_clamped_count",
                    "v_action_weight_norm_mean",
                    "dense_film_scale_abs_mean",
                    "dense_film_shift_abs_mean",
                    "lr_action",
                    "lr_dit_wd",
                    "lr_dit_nowd",
                    "cfg_drop_text_rate",
                    "cfg_drop_action_rate",
                    "sigma_future",
                    "sigma_history",
                    "latent_std",
                    "generated_history_active",
                    "generated_history_eligible_fraction",
                    "generated_history_sample_fraction",
                    "generated_history_replaced_frames",
                    "generated_history_delta",
                    "generated_history_blend",
                    "generated_history_denoise_steps",
                    "generated_history_unroll_chunks",
                    "generated_history_available_chunks",
                    "gpu_allocated_gib",
                    "gpu_reserved_gib",
                ]
                watch_parts = []
                for k in watch_keys:
                    if k not in log_dict:
                        continue
                    v = log_dict[k]
                    if isinstance(v, float):
                        watch_parts.append(f"{k}={v:.6g}")
                    else:
                        watch_parts.append(f"{k}={v}")
                watch_msg = " | " + " ".join(watch_parts) if watch_parts else ""

                logger.info(
                    "Micro-step %d/%d (optimizer_updates=%d) — loss: %.4f, "
                    "micro_steps/s: %.2f, lr: %.1e%s",
                    self._global_step,
                    max_steps,
                    self._accum_count // max(1, int(self.mcfg.grad_accum_steps)),
                    avg_loss,
                    steps_per_sec,
                    self.mcfg.learning_rate,
                    watch_msg,
                )

                if self._wandb_run is not None:
                    import wandb
                    wandb.log(
                        {"loss": avg_loss, "steps_per_sec": steps_per_sec,
                         **{k: v for k, v in log_dict.items() if k != "loss"},
                         **sampling_metrics},
                        step=self._global_step,
                    )

                running_loss = 0.0
                step_start_time = time.time()

            # Checkpointing — all ranks must participate (model export uses a
            # collective full_tensor() gather). Every save writes a full
            # safetensors + per-rank optim_shards; see save_checkpoint for
            # why we no longer use DCP.
            if self.tcfg.save_interval > 0 and self._global_step % self.tcfg.save_interval == 0:
                self.save_checkpoint(self._global_step, output_dir)

            # Diagnostics — all ranks participate (FSDP forwards are collective).
            if (
                val_dataset is not None
                and self.tcfg.diagnose_interval > 0
                and self._global_step % self.tcfg.diagnose_interval == 0
            ):
                self.diagnose(val_dataset, self._global_step)

            # Validation — must be called on ALL ranks under FSDP, else collective
            # forward deadlocks (prior bug: gated by self.is_main). Internal
            # logging is already main-only.
            if (
                val_dataset is not None
                and self.tcfg.val_interval > 0
                and self._global_step % self.tcfg.val_interval == 0
            ):
                self.validate(val_dataset, self._global_step, output_dir)

        # Final checkpoint — avoid saving the same step twice when the last
        # training step already hit save_interval. A duplicate full-state
        # gather is expensive and can leave enough CUDA/NCCL memory pressure
        # to fail during process-group shutdown on 4-GPU smoke runs.
        final_save_enabled = bool(getattr(self.tcfg, "save_final_checkpoint", True))
        last_step_already_saved = (
            self.tcfg.save_interval > 0
            and self._global_step % self.tcfg.save_interval == 0
        )
        if final_save_enabled and not last_step_already_saved:
            self.save_checkpoint(self._global_step, output_dir)
        if self.is_main:
            logger.info(
                "Training complete: %d micro-steps (%d optimizer updates)",
                self._global_step,
                self._accum_count // max(1, int(self.mcfg.grad_accum_steps)),
            )

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        step: int,
        output_dir: Path | str = ".",
        save_safetensors: bool = False,
    ) -> str:
        """Save model + optimizer + step for FSDP2 resume.

        Layout:
            checkpoint-{step}/
              dit_model.safetensors      # Full model weights
              optim_shards/rank_{r}.pt   # Per-rank, name-keyed optimizer state
              meta.json                  # step + world_size + config

        The model is gathered once onto rank 0 (via full_tensor()) and
        written as safetensors — this is the authoritative weight file used
        both for resume and for inference. Optimizer state is sharded: each
        rank writes its own DTensor local state, keyed by parameter name
        (not by the int index that PyTorch's default optimizer.state_dict
        uses), so the checkpoint survives param-group and ordering changes.

        Resume requires the same world_size; we fail clearly on mismatch
        rather than silently corrupt DTensor placements.

        ALL ranks must call this method — the model export is a collective.

        `save_safetensors` is retained for API compatibility with earlier
        versions; the safetensors export is always performed.
        """
        output_dir = Path(output_dir)
        ckpt_dir = output_dir / f"checkpoint-{step}"
        optim_dir = ckpt_dir / "optim_shards"
        if self.is_main:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            optim_dir.mkdir(parents=True, exist_ok=True)
        if self.world_size > 1:
            torch.distributed.barrier()

        # --- 1. Full model weights as safetensors (rank-0 gather) -------------
        # The model is always exported — it's the authoritative weight copy
        # used for both inference and resume. Cost: ~24 GB CPU RAM on rank 0
        # during gather, freed immediately after write.
        self._export_safetensors(ckpt_dir)

        # --- 2. Per-rank optimizer shard -------------------------------------
        # Save optimizer state keyed by parameter name (not by the int index
        # that PyTorch's `optimizer.state_dict()` uses internally). The int
        # indexing breaks whenever param_group structure changes, parameters
        # are reordered, or LoRA adapters are added — none of which should
        # invalidate the underlying per-param Adam momenta. Keying by name
        # makes resume tolerate any such reshaping: the loader looks each
        # name up in the current model and drops state for params that no
        # longer exist. Group hyperparams (lr, betas, weight_decay) are saved
        # alongside for audit. DTensor values are preserved per-rank
        # (FSDP2 stores local shards as DTensors; torch.save pickles them
        # with their placement intact).
        param_to_name = {id(p): n for n, p in self._dit.named_parameters()}
        state_by_name: dict[str, dict] = {}
        for p, st in self._optimizer.state.items():
            name = param_to_name.get(id(p))
            if name is None:
                continue
            state_by_name[name] = dict(st)

        group_hparams = []
        for group in self._optimizer.param_groups:
            names = [
                param_to_name.get(id(p))
                for p in group["params"]
                if param_to_name.get(id(p)) is not None
            ]
            hp = {k: v for k, v in group.items() if k != "params"}
            hp["param_names"] = names
            group_hparams.append(hp)

        torch.save(
            {
                "state_by_name": state_by_name,
                "group_hparams": group_hparams,
                "step": step,
                "rank": self.rank,
                "world_size": self.world_size,
            },
            optim_dir / f"rank_{self.rank}.pt",
        )

        # --- 3. LR scheduler state (rank-independent, rank 0 saves) ----------
        # Without this, LambdaLR's `last_epoch` resets to 0 on resume and the
        # warmup runs a second time.
        if self.is_main and self._lr_scheduler is not None:
            # LambdaLR.state_dict() omits the `lr_lambdas` callable, so the
            # remaining entries (last_epoch, _step_count, base_lrs, _last_lr)
            # are all picklable. Use torch.save for consistency with the
            # optimizer shard format.
            torch.save(
                self._lr_scheduler.state_dict(),
                ckpt_dir / "lr_scheduler.pt",
            )

        # --- 4. Meta ---------------------------------------------------------
        if self.is_main:
            meta = {
                "checkpoint_contract_version": 2,
                "step": step,
                "world_size": self.world_size,
                "format": "safetensors+optim_shards",
                "safetensors_exported": True,  # always true in this layout
                "lr_scheduler_saved": self._lr_scheduler is not None,
                # Keep the historical string fields so existing audit tools do
                # not break. New tooling should consume the structured fields.
                "model_config": str(self.mcfg),
                "training_config": str(self.tcfg),
                "model_config_dict": asdict(self.mcfg),
                "training_config_dict": asdict(self.tcfg),
                "sampling_state": self._sampling_state,
            }
            with open(ckpt_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(
                "Checkpoint saved: %s (step %d, world_size=%d)",
                ckpt_dir, step, self.world_size,
            )

        if self.world_size > 1:
            torch.distributed.barrier()
        return str(ckpt_dir)

    def _export_safetensors(self, ckpt_dir: Path) -> None:
        """Gather full FSDP weights and write safetensors on rank 0 only."""
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, StateDictOptions,
        )

        # Use DCP's FSDP2-aware state-dict path instead of calling
        # DTensor.full_tensor() parameter-by-parameter. The latter routes
        # through allgather_into_tensor_coalesced on PyTorch 2.6/NCCL and is
        # unsupported in the current cluster build.
        full_state_dict = get_model_state_dict(
            self._dit,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                strict=False,
            ),
        )

        if self.is_main:
            full_state_dict = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in full_state_dict.items()
            }
            try:
                from safetensors.torch import save_file
                save_file(full_state_dict, str(ckpt_dir / "dit_model.safetensors"))
            except ImportError:
                torch.save(full_state_dict, ckpt_dir / "dit_model.pt")
        del full_state_dict

    def _load_optimizer_shard(self, data: dict) -> None:
        """Restore per-param Adam state from a name-keyed shard payload.

        `data["state_by_name"]` maps parameter name to its Adam state
        (exp_avg / exp_avg_sq / step / ...). We look each name up in the
        current `self._dit.named_parameters()` and assign directly into
        `self._optimizer.state[param]`. This decouples resume from the
        optimizer's internal index-based format, so changing param groups
        (e.g. adding a differential-LR group) or reordering parameters does
        not invalidate the checkpoint. Params present in the shard but not
        in the current model are dropped; params present now but missing
        from the shard silently start at Adam's default empty state.
        """
        state_by_name = data.get("state_by_name")
        if state_by_name is None:
            raise RuntimeError(
                "Optimizer shard missing 'state_by_name'. Re-save with the "
                "current trainer before resuming."
            )

        name_to_param = {n: p for n, p in self._dit.named_parameters()}
        restored, unmatched, recast = 0, [], 0
        for name, st in state_by_name.items():
            p = name_to_param.get(name)
            if p is None:
                unmatched.append(name)
                continue
            # Move each Adam momentum tensor (exp_avg, exp_avg_sq) to the
            # current param's device AND dtype. ``load_checkpoint`` reads the
            # shard with ``map_location="cpu"`` to avoid OOM during load, so
            # tensors arrive on CPU regardless of where they were saved from.
            # FSDP2 params live on the rank's GPU (DTensor-wrapped), and
            # multi-tensor AdamW refuses to operate across a CPU/CUDA split.
            # Dtype cast is also needed for pre-mp_policy checkpoints (bf16
            # master params) being resumed into an fp32-master run.
            # Scalars like ``step`` are not tensors and stay as-is.
            cast_st = {}
            for k, v in st.items():
                if torch.is_tensor(v) and (
                    v.device != p.device or v.dtype != p.dtype
                ):
                    cast_st[k] = v.to(device=p.device, dtype=p.dtype)
                    recast += 1
                else:
                    cast_st[k] = v
            self._optimizer.state[p] = cast_st
            restored += 1
        if self.is_main:
            logger.info(
                "Optimizer restored by name: %d params (shard had %d; "
                "%d unmatched in current model; %d tensor entries recast)",
                restored, len(state_by_name), len(unmatched), recast,
            )
            if unmatched:
                logger.info("First unmatched param names: %s", unmatched[:5])

    def load_pretrained_weights(self, path: str | Path) -> None:
        """Load model weights BEFORE FSDP wrapping (for fresh-start fine-tuning).

        Call this after _init_dit() but before _init_fsdp(). Optimizer state
        is NOT loaded — use load_checkpoint() after FSDP wrapping for that.
        """
        ckpt_dir = Path(path)
        st_path = ckpt_dir / "dit_model.safetensors"
        pt_path = ckpt_dir / "dit_model.pt"

        if st_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(st_path))
        elif pt_path.exists():
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model checkpoint found in {ckpt_dir}")

        current = self._dit.state_dict()
        compatible_state = {}
        skipped_shape = []
        expanded_shape = []
        unexpected = []
        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(key)
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                expanded = self._expand_warmstart_tensor(key, value, current[key])
                if expanded is None:
                    skipped_shape.append(key)
                    continue
                compatible_state[key] = expanded
                expanded_shape.append(key)
                continue
            compatible_state[key] = value

        illegal_unexpected = [k for k in unexpected if not self._is_warmstart_optional_param(k)]
        illegal_shape = [k for k in skipped_shape if not self._is_warmstart_optional_param(k)]
        if illegal_unexpected or illegal_shape:
            raise RuntimeError(
                "Checkpoint architecture mismatch outside action-side modules: "
                f"unexpected={illegal_unexpected[:10]}, shape_mismatch={illegal_shape[:10]}"
            )

        missing, unexpected_after = self._dit.load_state_dict(compatible_state, strict=False)
        illegal_missing = [k for k in missing if not self._is_warmstart_optional_param(k)]
        illegal_unexpected_after = [
            k for k in unexpected_after if not self._is_warmstart_optional_param(k)
        ]
        if illegal_missing or illegal_unexpected_after:
            raise RuntimeError(
                "Checkpoint load left non-action-side incompatibilities: "
                f"missing={illegal_missing[:10]}, unexpected={illegal_unexpected_after[:10]}"
            )
        logger.info(
            "Loaded model weights from %s (%d compatible keys, %d warmstart-optional missing, "
            "%d warmstart-optional unexpected, %d compatible tensors expanded, "
            "%d warmstart-optional shape-skipped)",
            ckpt_dir,
            len(compatible_state),
            len(missing),
            len(unexpected),
            len(expanded_shape),
            len(skipped_shape),
        )

    def load_checkpoint(self, path: str | Path) -> int:
        """Resume from a safetensors + optim_shards checkpoint.

        Must be called AFTER _init_fsdp() and _init_optimizer(). Loads:
          1. Full model weights from `dit_model.safetensors` via
             `set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)`
             — rank 0 holds the full tensors, others pass an empty dict and
             receive their DTensor shard via NCCL broadcast.
          2. This rank's optimizer shard from `optim_shards/rank_{r}.pt`, in
             the name-keyed format written by save_checkpoint. Resume
             requires the same world_size (DTensor shard layout can't be
             re-split); we fail clearly if they mismatch.

        Returns the step number to resume from.
        """
        ckpt_dir = Path(path)
        st_path = ckpt_dir / "dit_model.safetensors"
        optim_dir = ckpt_dir / "optim_shards"
        meta_path = ckpt_dir / "meta.json"

        # --- 1. Model weights ------------------------------------------------
        if not st_path.exists():
            raise FileNotFoundError(f"No dit_model.safetensors in {ckpt_dir}")

        if self.is_main:
            from safetensors.torch import load_file
            full_sd = load_file(str(st_path))
        else:
            full_sd = {}

        dit_sd = full_sd

        if self.world_size > 1:
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict, StateDictOptions,
            )
            # DiT: FSDP2-wrapped, use the DCP collective path.
            set_model_state_dict(
                self._dit,
                model_state_dict=dit_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    cpu_offload=True,
                    strict=True,
                ),
            )
        else:
            self._dit.load_state_dict(dit_sd, strict=True)
        if self.is_main:
            logger.info(
                "Resumed model weights from %s (DiT: %d keys)",
                st_path, len(dit_sd),
            )

        # --- 2. Optimizer shard for this rank --------------------------------
        shard_path = optim_dir / f"rank_{self.rank}.pt"
        if not shard_path.exists():
            raise FileNotFoundError(
                f"No optimizer shard at {shard_path}. This checkpoint was "
                "not saved by the current trainer."
            )
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        saved_ws = int(data.get("world_size", -1))
        if saved_ws != self.world_size:
            raise RuntimeError(
                f"Optimizer shard world_size={saved_ws} != current "
                f"{self.world_size}. DTensor shard layout can't be re-split; "
                f"restart the job with world_size={saved_ws}."
            )
        self._load_optimizer_shard(data)

        # --- 3. LR scheduler state ------------------------------------------
        # Rank-independent: same state on all ranks, but every rank loads it
        # from disk (cheap, a few scalars). Missing file is tolerated for
        # backward-compat with pre-fix checkpoints — in that case we
        # reconstruct last_epoch from step / grad_accum_steps so warmup
        # doesn't repeat even on old checkpoints.
        sched_path = ckpt_dir / "lr_scheduler.pt"
        if self._lr_scheduler is not None:
            if sched_path.exists():
                sched_state = torch.load(
                    sched_path, map_location="cpu", weights_only=False
                )
                self._lr_scheduler.load_state_dict(sched_state)
                if self.is_main:
                    logger.info(
                        "LR scheduler restored: last_epoch=%d",
                        self._lr_scheduler.last_epoch,
                    )
            else:
                # Pre-fix checkpoint: reconstruct last_epoch from step.
                # scheduler.step() is called once per optimizer step, and
                # one optimizer step = grad_accum_steps train_steps, so the
                # scheduler has taken step // grad_accum_steps ticks.
                meta_step = 0
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta_step = json.load(f).get("step", 0)
                opt_steps = meta_step // max(1, self.mcfg.grad_accum_steps)
                self._lr_scheduler.last_epoch = opt_steps
                # Refresh cached LR so the first log line after resume shows
                # the correct post-warmup value instead of the base LR.
                for grp, base in zip(
                    self._lr_scheduler.optimizer.param_groups,
                    self._lr_scheduler.base_lrs,
                ):
                    grp["lr"] = base * self._lr_scheduler.lr_lambdas[0](opt_steps)
                if self.is_main:
                    logger.info(
                        "LR scheduler state missing; reconstructed "
                        "last_epoch=%d from step=%d, grad_accum=%d",
                        opt_steps, meta_step, self.mcfg.grad_accum_steps,
                    )

        # --- 4. Step counter -------------------------------------------------
        step = 0
        if meta_path.exists():
            with open(meta_path) as f:
                step = json.load(f).get("step", 0)
        self._global_step = step
        self._accum_count = step
        if self.is_main:
            logger.info("Resuming from step %d", step)
        return step

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validation_refs(
        val_dataset,
        num_samples: int,
        seed: int,
        *,
        future_latents_required: int | None = None,
        window_fraction: float | None = None,
    ):
        if hasattr(val_dataset, "root_balanced_validation_refs"):
            return val_dataset.root_balanced_validation_refs(
                int(num_samples),
                seed=int(seed),
                future_latents_required=future_latents_required,
                window_fraction=window_fraction,
            )
        count = min(int(num_samples), len(val_dataset))
        return [int(x) for x in np.linspace(0, len(val_dataset) - 1, count, dtype=int)]

    @staticmethod
    def _validation_sample(val_dataset, ref):
        if isinstance(ref, tuple):
            return val_dataset.get_source_sample(ref[0], ref[1])
        return val_dataset[int(ref)]

    @staticmethod
    def _validation_description(val_dataset, ref) -> dict:
        if isinstance(ref, tuple) and hasattr(val_dataset, "describe_source_sample"):
            return val_dataset.describe_source_sample(ref[0], ref[1])
        return {"flat_index": int(ref)}

    def _write_validation_selection(
        self,
        output_dir: Path,
        *,
        name: str,
        seed: int,
        val_dataset,
        refs,
        metadata: dict | None = None,
    ) -> None:
        if not self.is_main:
            return
        selection_path = output_dir / "validation" / f"selection_{name}.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "coachworld_fixed_root_balanced_validation_selection",
            "name": str(name),
            "seed": int(seed),
            "metadata": dict(metadata or {}),
            "samples": [
                self._validation_description(val_dataset, ref) for ref in refs
            ],
        }
        if selection_path.exists():
            existing = json.loads(selection_path.read_text())
            if existing != payload:
                raise RuntimeError(
                    f"fixed validation selection changed during one run: {selection_path}"
                )
            return
        selection_path.write_text(json.dumps(payload, indent=2) + "\n")

    @torch.no_grad()
    def _validate_rgb_metrics(
        self,
        val_dataset,
        step: int,
        output_dir: Path,
    ) -> dict:
        """Run fixed held-out closed loops and score decoded future RGB.

        DiT inference remains collective under FSDP. Only rank 0 retains the
        generated latents, then loads the VAE and LPIPS after all DiT forwards
        have finished. This keeps VAE memory out of the high-water DiT phase.
        """
        from coachworld.world_model.base import WorldModelInput
        from coachworld.world_model.wan_world_model import WanWorldModel

        seed = int(getattr(self.tcfg, "val_metrics_seed", 0))
        num_samples = int(getattr(self.tcfg, "val_metrics_num_samples", 0))
        inference_steps = int(getattr(self.tcfg, "val_metrics_inference_steps", 25))
        validation_protocol = resolve_closed_loop_validation_protocol(
            mode=str(
                getattr(
                    self.tcfg,
                    "val_metrics_rollout_mode",
                    "fixed_chunks",
                )
            ),
            fixed_chunks=int(
                getattr(self.tcfg, "val_metrics_closed_loop_chunks", 1)
            ),
        )
        full_episode = validation_protocol.full_episode
        configured_rollout_chunks = validation_protocol.fixed_chunks
        history_frames = int(self.mcfg.history_length)
        future_frames = int(self.mcfg.pred_frames)
        history_dilation = int(getattr(self.tcfg, "history_eval_dilation", 2))
        if (
            num_samples <= 0
            or inference_steps <= 0
        ):
            raise ValueError(
                "closed-loop RGB validation requires positive sample and "
                "inference-step counts"
            )
        if not hasattr(val_dataset, "get_episode_window"):
            raise TypeError("closed-loop RGB validation requires a video-latent dataset")
        refs = self._validation_refs(
            val_dataset,
            num_samples,
            seed,
            future_latents_required=(
                future_frames
                if full_episode
                else int(configured_rollout_chunks) * future_frames
            ),
            window_fraction=0.0 if full_episode else None,
        )
        self._write_validation_selection(
            output_dir,
            name="closed_loop",
            seed=seed,
            val_dataset=val_dataset,
            refs=refs,
            metadata={
                "rollout_mode": validation_protocol.mode,
                "rollout_chunks": configured_rollout_chunks,
                "future_latents_per_chunk": future_frames,
                "history_dilation": history_dilation,
            },
        )

        validator = WanWorldModel(
            self.mcfg,
            device=str(self.device),
            text_encoder_device=str(self.device),
            num_cameras=int(self.tcfg.num_cameras),
        )
        validator._dit = self._dit
        validator._text_encoder = self._text_encoder
        validator._scheduler = FlowMatchScheduler(
            shift=self.mcfg.flow_matching_shift,
            num_train_timesteps=1000,
            extra_one_step=True,
        )
        # Reuse the trainer's bounded text cache instead of re-encoding all
        # fixed validation prompts at every metric checkpoint.
        validator._prepare_text_embedding = (
            lambda instruction: self._get_text_embedding([str(instruction)])
        )

        rng_state = torch.random.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        )
        generated: list[
            tuple[dict, torch.Tensor, torch.Tensor, list[int]]
        ] = []
        for ordinal, ref in enumerate(refs):
            description = self._validation_description(val_dataset, ref)
            dataset_index = int(description["dataset_index"])
            entry_index = int(description["episode_entry_index"])
            first_future_latent = (
                int(description["first_future_frame"]) // self.VAE_TEMPORAL_STRIDE
            )
            initial_current_latent = first_future_latent - 1
            num_latent_frames = int(description["num_latent_frames"])
            remaining_episode_latents = num_latent_frames - first_future_latent
            if remaining_episode_latents <= 0:
                raise RuntimeError(
                    "closed-loop selection has no future episode latents: "
                    f"{description}"
                )
            sample_rollout_chunks = (
                math.ceil(remaining_episode_latents / future_frames)
                if full_episode
                else int(configured_rollout_chunks)
            )
            predicted_by_id: dict[int, torch.Tensor] = {}
            pred_future_items: list[torch.Tensor] = []
            gt_future_items: list[torch.Tensor] = []
            chunk_latent_lengths: list[int] = []
            current_anchor: torch.Tensor | None = None

            for chunk_index in range(sample_rollout_chunks):
                sample = val_dataset.get_episode_window(
                    dataset_index,
                    entry_index,
                    first_future_frame=first_future_latent * self.VAE_TEMPORAL_STRIDE,
                    history_dilation=history_dilation,
                    history_collapsed=False,
                    allow_future_padding=full_episode,
                )
                latent = sample["latent"].permute(1, 0, 2, 3).contiguous()
                if latent.shape[1] != history_frames + future_frames:
                    raise RuntimeError(
                        "closed-loop latent window mismatch: "
                        f"got {tuple(latent.shape)}, expected T="
                        f"{history_frames + future_frames}"
                    )
                history_ids = [int(x) for x in sample["history_latent_ids"].tolist()]
                future_ids = [int(x) for x in sample["future_latent_ids"].tolist()]
                chunk_take = (
                    min(
                        future_frames,
                        num_latent_frames - first_future_latent,
                    )
                    if full_episode
                    else future_frames
                )
                expected_future_ids = list(
                    range(first_future_latent, first_future_latent + chunk_take)
                )
                if future_ids[:chunk_take] != expected_future_ids:
                    raise RuntimeError(
                        "closed-loop future ids are not contiguous at the real "
                        f"episode positions: got={future_ids}, "
                        f"expected_prefix={expected_future_ids}"
                    )
                history_latent = latent[:, :history_frames].clone()
                if chunk_index == 0:
                    if history_ids[-1] != initial_current_latent:
                        raise RuntimeError(
                            "sparse history must end at the latest observed latent: "
                            f"history={history_ids}, current={initial_current_latent}"
                        )
                    current_anchor = history_latent[:, -1].detach().cpu()
                for position, latent_id in enumerate(history_ids):
                    if latent_id in predicted_by_id:
                        history_latent[:, position] = predicted_by_id[latent_id].to(
                            device=history_latent.device,
                            dtype=history_latent.dtype,
                        )
                    elif latent_id > initial_current_latent:
                        raise RuntimeError(
                            f"closed-loop history latent {latent_id} was neither observed "
                            "before rollout nor generated by an earlier chunk"
                        )

                sample_seed = seed + ordinal * 1009 + chunk_index
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                output = validator.step(
                    WorldModelInput(
                        current_obs=history_latent,
                        action_sequence=sample["action"].cpu().numpy(),
                        instruction=str(sample["text"]),
                        domain_id=int(sample["domain_id"]),
                        embodiment_id=int(sample["embodiment_id"]),
                        camera_setup_id=int(sample["camera_setup_id"]),
                        viewmats=sample.get("viewmats"),
                        Ks=sample.get("Ks"),
                        eef_uv=sample.get("eef_uv"),
                        eef_depth=sample.get("eef_depth"),
                        eef_valid=sample.get("eef_valid"),
                        eef_image_hw=sample.get("eef_image_hw"),
                        eef_gripper=sample.get("eef_gripper"),
                    ),
                    num_steps=inference_steps,
                    decode_frames=False,
                )
                pred_future = output.predicted_obs[:, history_frames:].detach().cpu()
                gt_future = latent[:, history_frames:].detach().cpu()
                for position, latent_id in enumerate(future_ids[:chunk_take]):
                    predicted_by_id[latent_id] = pred_future[:, position]
                    pred_future_items.append(pred_future[:, position])
                    gt_future_items.append(gt_future[:, position])
                chunk_latent_lengths.append(chunk_take)
                first_future_latent += chunk_take

            if self.is_main:
                if current_anchor is None:
                    raise RuntimeError("closed-loop validation produced no current anchor")
                pred_rollout = torch.stack(
                    [current_anchor, *pred_future_items], dim=1
                ).to(dtype=torch.bfloat16)
                gt_rollout = torch.stack(
                    [current_anchor, *gt_future_items], dim=1
                ).to(dtype=torch.bfloat16)
                generated.append(
                    (
                        description,
                        pred_rollout,
                        gt_rollout,
                        chunk_latent_lengths,
                    )
                )

        if dist.is_initialized():
            dist.barrier()

        metrics: dict = {}
        if self.is_main:
            from coachworld.evaluator.training_rgb_metrics import (
                RGBMetricAccumulator,
                init_lpips_model,
            )
            from coachworld.evaluator.video_io import save_h264_mp4
            from coachworld.wan import Wan2_2_VAE

            metric_dir = output_dir / "validation" / f"step_{step}" / "closed_loop"
            video_dir = metric_dir / "videos"
            gt_cache_dir = output_dir / "validation" / "gt_closed_loop_rgb_cache"
            metric_dir.mkdir(parents=True, exist_ok=True)
            gt_cache_dir.mkdir(parents=True, exist_ok=True)
            if bool(getattr(self.tcfg, "val_metrics_save_videos", True)):
                video_dir.mkdir(parents=True, exist_ok=True)

            vae = None
            lpips_model = None
            try:
                torch.cuda.empty_cache()
                vae = Wan2_2_VAE(
                    vae_pth=str(self.mcfg.vae_checkpoint),
                    device=str(self.device),
                )
                validator._vae = vae
                lpips_device = str(self.device)
                lpips_model = init_lpips_model(
                    str(getattr(self.tcfg, "val_metrics_lpips_net", "alex")),
                    lpips_device,
                )
                lpips_batch_size = int(
                    getattr(self.tcfg, "val_metrics_lpips_batch_size", 4)
                )
                aggregate = RGBMetricAccumulator()
                by_dataset: dict[str, RGBMetricAccumulator] = {}
                by_rollout_chunk: dict[int, RGBMetricAccumulator] = {}
                rows = []
                for ordinal, (
                    description,
                    pred_latent,
                    gt_latent,
                    chunk_latent_lengths,
                ) in enumerate(generated):
                    dataset_key = str(description.get("dataset_key", "validation"))
                    safe_key = "".join(
                        char if char.isalnum() or char in "-_" else "_"
                        for char in dataset_key
                    )
                    cache_name = (
                        f"d{int(description.get('dataset_index', 0)):02d}_"
                        f"ep{int(description.get('episode_id', ordinal))}_"
                        f"f{int(description.get('first_future_frame', 0))}_"
                        f"c{len(chunk_latent_lengths)}_"
                        f"n{sum(chunk_latent_lengths)}_d{history_dilation}.npy"
                    )
                    gt_cache_path = gt_cache_dir / cache_name
                    if gt_cache_path.exists():
                        gt_frames = np.load(gt_cache_path, allow_pickle=False)
                    else:
                        decoded_gt = validator.decode(gt_latent)
                        if decoded_gt is None:
                            raise RuntimeError("WAN VAE returned no GT validation frames")
                        gt_frames = decoded_gt[1:]
                        np.save(gt_cache_path, gt_frames, allow_pickle=False)

                    decoded_pred = validator.decode(pred_latent)
                    if decoded_pred is None:
                        raise RuntimeError("WAN VAE returned no predicted validation frames")
                    pred_frames = decoded_pred[1:]
                    expected_raw_frames = (
                        sum(chunk_latent_lengths) * self.VAE_TEMPORAL_STRIDE
                    )
                    if len(pred_frames) != expected_raw_frames:
                        raise RuntimeError(
                            "decoded closed-loop length mismatch: "
                            f"got {len(pred_frames)} frames, expected "
                            f"{expected_raw_frames} from "
                            f"{len(chunk_latent_lengths)} chunks"
                        )
                    if pred_frames.shape != gt_frames.shape:
                        raise RuntimeError(
                            f"decoded future RGB mismatch: pred={pred_frames.shape}, "
                            f"gt={gt_frames.shape}"
                        )
                    sample_accumulator = RGBMetricAccumulator()
                    chunk_metrics = []
                    start_frame = 0
                    for chunk_index, chunk_latents in enumerate(
                        chunk_latent_lengths
                    ):
                        end_frame = (
                            start_frame
                            + int(chunk_latents) * self.VAE_TEMPORAL_STRIDE
                        )
                        chunk_accumulator = RGBMetricAccumulator()
                        chunk_summary = chunk_accumulator.update(
                            pred_frames[start_frame:end_frame],
                            gt_frames[start_frame:end_frame],
                            lpips_model=lpips_model,
                            lpips_device=lpips_device,
                            lpips_batch_size=lpips_batch_size,
                        )
                        by_rollout_chunk.setdefault(
                            chunk_index + 1, RGBMetricAccumulator()
                        ).merge(chunk_accumulator)
                        sample_accumulator.merge(chunk_accumulator)
                        chunk_metrics.append(
                            {"chunk": chunk_index + 1, **chunk_summary}
                        )
                        start_frame = end_frame
                    sample_metrics = sample_accumulator.summary()
                    aggregate.merge(sample_accumulator)
                    bucket = by_dataset.setdefault(dataset_key, RGBMetricAccumulator())
                    bucket.merge(sample_accumulator)
                    video_path = None
                    if bool(getattr(self.tcfg, "val_metrics_save_videos", True)):
                        comparison = np.concatenate([gt_frames, pred_frames], axis=2)
                        video_path = video_dir / (
                            f"{ordinal:02d}_{safe_key}_ep"
                            f"{int(description.get('episode_id', ordinal))}.mp4"
                        )
                        save_h264_mp4(
                            comparison,
                            video_path,
                            fps=float(getattr(self.tcfg, "val_metrics_fps", 5.0)),
                        )
                    rows.append(
                        {
                            **description,
                            "rollout_mode": validation_protocol.mode,
                            "rollout_chunks": len(chunk_latent_lengths),
                            "rollout_future_latents": sum(
                                chunk_latent_lengths
                            ),
                            "metrics": sample_metrics,
                            "chunk_metrics": chunk_metrics,
                            "comparison_video": str(video_path) if video_path else None,
                            "layout": "GT_left__prediction_right" if video_path else None,
                        }
                    )

                metrics = aggregate.summary()
                payload = {
                    "kind": "coachworld_periodic_closed_loop_rgb_validation",
                    "step": int(step),
                    "reference": "wan_vae_decoded_closed_loop_vs_heldout_gt",
                    "history_latents": history_frames,
                    "future_latents_per_chunk": future_frames,
                    "rollout_mode": validation_protocol.mode,
                    "configured_rollout_chunks": configured_rollout_chunks,
                    "history_dilation": history_dilation,
                    "seed": seed,
                    "inference_steps": inference_steps,
                    "aggregate": metrics,
                    "by_dataset": {
                        key: value.summary() for key, value in by_dataset.items()
                    },
                    "by_rollout_chunk": {
                        str(key): value.summary()
                        for key, value in sorted(by_rollout_chunk.items())
                    },
                    "samples": rows,
                }
                (metric_dir / "metrics.json").write_text(
                    json.dumps(payload, indent=2) + "\n"
                )
            finally:
                validator._vae = None
                del lpips_model
                del vae
                gc.collect()
                torch.cuda.empty_cache()

            logger.info(
                "Closed-loop validation at step %d: PSNR=%.3f SSIM=%.4f "
                "LPIPS=%.4f (%d samples, %d future frames, mode=%s)",
                step,
                float(metrics["psnr"]),
                float(metrics["ssim"]),
                float(metrics["lpips"]),
                int(metrics["samples"]),
                int(metrics["frames"]),
                "full_episode"
                if full_episode
                else f"{configured_rollout_chunks}_chunks",
            )
            if self._wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "val_closed_loop/psnr": float(metrics["psnr"]),
                        "val_closed_loop/ssim": float(metrics["ssim"]),
                        "val_closed_loop/lpips": float(metrics["lpips"]),
                        "val_closed_loop/mse": float(metrics["mse"]),
                    },
                    step=step,
                )

        if dist.is_initialized():
            dist.barrier()
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, self.device)
        return metrics

    @torch.no_grad()
    def validate(self, val_dataset, step: int, output_dir: Path | str = ".") -> dict:
        """Generate sample videos from validation set and log.

        Uses a few samples from val_dataset, runs DiT inference,
        and saves comparison images/videos.
        """
        output_dir = Path(output_dir)
        val_dir = output_dir / "validation" / f"step_{step}"
        val_dir.mkdir(parents=True, exist_ok=True)

        self._dit.eval()

        validation_seed = int(getattr(self.tcfg, "val_metrics_seed", 0))
        refs = self._validation_refs(
            val_dataset,
            int(self.tcfg.val_video_num),
            validation_seed,
        )
        self._write_validation_selection(
            output_dir,
            name="loss",
            seed=validation_seed,
            val_dataset=val_dataset,
            refs=refs,
        )

        # Seed RNG for reproducible validation loss (noise + timestep sampling)
        rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)

        val_losses = []
        for ref in refs:
            sample = self._validation_sample(val_dataset, ref)
            batch = collate_video_latent_batch([sample])

            loss, _ = self.compute_loss(batch)
            val_losses.append(loss.item())

        # Restore RNG state
        torch.random.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        result = {"val_loss": float(avg_val_loss)}
        if self.is_main:
            logger.info("Validation at step %d: avg_loss=%.4f", step, avg_val_loss)
            if self._wandb_run is not None:
                import wandb
                wandb.log({"val_loss": avg_val_loss}, step=step)

        metrics_enabled = bool(getattr(self.tcfg, "val_metrics_enabled", False))
        metrics_interval = int(getattr(self.tcfg, "val_metrics_interval", 0))
        if metrics_enabled and metrics_interval > 0 and step % metrics_interval == 0:
            rgb_metrics = self._validate_rgb_metrics(val_dataset, step, output_dir)
            if self.is_main:
                result.update({f"val_rgb_{key}": value for key, value in rgb_metrics.items()})

        self._dit.train()
        return result

    # ------------------------------------------------------------------
    # Training-time diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def diagnose(self, val_dataset, step: int) -> dict:
        """Lightweight training-time diagnostic metrics.

        Runs ~6 extra forward passes to answer:
          1. Is action conditioning wired & effective?   (action_sensitivity)
          2. Is history/current conditioning effective?  (history_effect)
          3. Does the model learn an unconditional branch for CFG? (cfg_gap)
          4. Are VAE latents in a healthy distribution?   (latent stats)
          5. Rough rollout quality (5-step denoise)      (rollout_mse)

        Per-block gradient norms are logged separately inside train_step.

        MUST be called by all ranks (forwards are FSDP-collective).
        """
        if val_dataset is None or len(val_dataset) == 0:
            return {}

        self._dit.eval()
        try:
            # Use a single deterministic sample so metrics are comparable across steps
            idx = step % len(val_dataset)
            sample = val_dataset[int(idx)]
            batch = collate_video_latent_batch([sample])

            latent_full = batch["latent"].to(self.device, self.dtype).permute(0, 2, 1, 3, 4)
            B, C, T_full, H, W = latent_full.shape

            # Slice to DiT input window — same logic as compute_loss.
            full_window = full_window_enabled(self.tcfg) or full_window_enabled(
                self.mcfg
            )
            if full_window:
                start = 0
                end = T_full
                latent = latent_full
                history_frames = min(max(1, int(self.mcfg.history_length)), T_full - 1)
            else:
                T_model = self.mcfg.pred_frames
                start = max(0, self.mcfg.history_length - 1)
                end = min(start + T_model, T_full)
                latent = latent_full[:, :, start:end, :, :]
                history_frames = 1
            T = latent.shape[2]

            viewmats = None
            Ks = None
            if bool(getattr(self.mcfg, "prope_enabled", False)):
                if "viewmats" not in batch or "Ks" not in batch:
                    raise RuntimeError(
                        "PRoPE is enabled but diagnose batch has no viewmats/Ks."
                    )
                viewmats = batch["viewmats"][:, start:end].to(
                    device=self.device, dtype=self.dtype
                )
                Ks = batch["Ks"][:, start:end].to(device=self.device, dtype=self.dtype)
            eef_projection_kwargs = self._eef_projection_kwargs(batch, start, end)

            # Mid-noise sigma for diagnostic forwards
            sigma_val = torch.tensor([0.5], device=self.device, dtype=self.dtype)
            sigma_bc = sigma_val.view(1, 1, 1, 1, 1)

            g = torch.Generator(device=self.device).manual_seed(step + 1)
            noise = torch.randn(
                latent.shape, device=self.device, dtype=self.dtype, generator=g,
            )
            noisy_latent = (1.0 - sigma_bc) * latent + sigma_bc * noise
            if bool(
                getattr(self.tcfg, "gt_condition_replacement", False)
                or getattr(self.mcfg, "gt_condition_replacement", False)
            ):
                noisy_latent[:, :, :history_frames] = latent[:, :, :history_frames]
            else:
                sigma_h = float(self.tcfg.history_noise_max) * 0.5
                noisy_latent[:, :, :history_frames] = (
                    (1.0 - sigma_h) * latent[:, :, :history_frames]
                    + sigma_h * noise[:, :, :history_frames]
                )
            condition_mask = None
            if bool(
                getattr(self.tcfg, "condition_mask_enabled", False)
                or getattr(self.mcfg, "condition_mask_enabled", False)
            ):
                condition_mask = torch.zeros(
                    1, 1, T, H, W, device=self.device, dtype=self.dtype
                )
                condition_mask[:, :, :history_frames, :, :] = 1.0

            # Map sigma back to a training-space timestep for the model input
            t_idx = int(sigma_val.item() * (len(self._scheduler.timesteps) - 1))
            timestep = self._scheduler.timesteps[t_idx:t_idx + 1].to(
                dtype=self.dtype, device=self.device,
            )

            pt, ph, pw = self.PATCH_SIZE
            seq_len = (T // pt) * (H // ph) * (W // pw)

            text_emb = self._get_text_embedding(batch["text"])
            # Null text: tiny noise (not strict zeros) — matches training-time
            # convention and avoids the bf16 flash_attn NaN when zero_text
            # creates fully-identical K/V across all 512 text positions.
            zero_text = torch.randn_like(text_emb) * 0.01

            actions = batch["action"]
            action_domain_ids = self._batch_action_domain_ids(
                batch,
                action_present=isinstance(actions, torch.Tensor),
            )
            action_rate = str(getattr(self.tcfg, "action_condition_timestep_rate", "raw"))
            a_start = start * self.VAE_TEMPORAL_STRIDE
            dense_len = num_video_frames_for_latent_window(
                T, self.VAE_TEMPORAL_STRIDE
            )
            a_dense_end = min(a_start + dense_len, actions.shape[1])
            actions_dense_sliced = actions[:, a_start:a_dense_end, ...].contiguous()
            action_dense_cond = self._prepare_batch_actions(actions_dense_sliced, T)
            if action_rate == "latent":
                actions_sliced = actions[
                    :, a_start:a_dense_end:self.VAE_TEMPORAL_STRIDE, ...
                ][:, :T, ...].contiguous()
                action_cond = self._align_latent_rate_actions(
                    actions_sliced.to(device=self.device, dtype=self.dtype), T
                )
            else:
                action_cond = self._prepare_batch_actions(actions_dense_sliced, T)
                if action_dense_cond is None:
                    action_dense_cond = action_cond
            dense_zero_first_chunk = False
            dense_zero_prefix_chunks = 0
            if (
                action_cond is not None
                and bool(
                    getattr(self.tcfg, "future_only_action_conditioning", False)
                    or getattr(self.mcfg, "future_only_action_conditioning", False)
                )
            ):
                action_cond, action_mask = apply_kv_future_only_action_mask(
                    action_cond,
                    history_frames=history_frames,
                    action_rate=action_rate,
                    vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
                )
                dense_zero_first_chunk = action_mask.dense_zero_first_chunk
                dense_zero_prefix_chunks = action_mask.dense_zero_prefix_chunks

            eef_spatial = self._eef_spatial_condition(
                batch,
                start,
                end,
                latent_height=H,
                latent_width=W,
            )
            cond_with_action = self._concat_condition_channels(
                condition_mask,
                eef_spatial,
            )
            cond_without_action = self._concat_condition_channels(
                condition_mask,
                None if eef_spatial is None else torch.zeros_like(eef_spatial),
            )

            def _fwd(x, t_emb, a_seq):
                """Forward with optional action. a_seq=None → text-only branch
                (uses context_lens to drop action tokens from cross-attn K/V)."""
                dense_seq = action_dense_cond if a_seq is not None else None
                branch_cond = cond_with_action if a_seq is not None else cond_without_action
                branch_eef = eef_projection_kwargs if a_seq is not None else {}
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self._dit(
                        [x[i] for i in range(x.shape[0])],
                        timestep, t_emb, seq_len,
                        cond_concat=branch_cond,
                        action_seq=a_seq,
                        action_dense_seq=dense_seq,
                        action_dense_zero_first_chunk=dense_zero_first_chunk,
                        action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                        action_domain_ids=action_domain_ids if a_seq is not None else None,
                        viewmats=viewmats,
                        Ks=Ks,
                        **branch_eef,
                    )
                return torch.stack(out).to(self.dtype)

            # 1) Action sensitivity: real vs no-action prediction.
            # The no-action path passes action_seq=None so action tokens are
            # fully excluded from cross-attn (context_lens = L_text only),
            # matching inference-time CFG behavior exactly.
            pred_with_action = _fwd(noisy_latent, text_emb, action_cond)
            pred_no_action = _fwd(noisy_latent, text_emb, None)
            action_sensitivity = F.mse_loss(pred_with_action, pred_no_action).item()

            # 2) CFG gap: full (text+action) vs text-only-zeroed.
            pred_no_text = _fwd(noisy_latent, zero_text, action_cond)
            cfg_gap = F.mse_loss(pred_with_action, pred_no_text).item()

            # 3) History effect: change the condition/history frames inside
            # the same DiT window and check whether the prediction shifts.
            # In full-window mode T_full == T, so this must not rely
            # on extra frames outside the current window.
            shuffled_latent = noisy_latent.clone()
            sigma_h_diag = float(self.tcfg.history_noise_max) * 0.5
            if T > history_frames:
                src_start = history_frames
                src_end = min(T, src_start + history_frames)
                alt_current = latent[:, :, src_start:src_end, :, :]
                if alt_current.shape[2] < history_frames:
                    pad = alt_current[:, :, -1:, :, :].expand(
                        -1, -1, history_frames - alt_current.shape[2], -1, -1
                    )
                    alt_current = torch.cat([alt_current, pad], dim=2)
                alt_current = alt_current[:, :, :history_frames, :, :]
                alt_noise = noise[:, :, :history_frames]
                alt_current = (1 - sigma_h_diag) * alt_current + sigma_h_diag * alt_noise
                shuffled_latent[:, :, :history_frames] = alt_current
            else:
                shuffled_latent[:, :, :history_frames] = noise[:, :, :history_frames]
            pred_alt_hist = _fwd(shuffled_latent, text_emb, action_cond)
            history_effect = F.mse_loss(pred_with_action, pred_alt_hist).item()

            # 4) Latent stats
            latent_mean = latent_full.mean().item()
            latent_std = latent_full.std().item()
            latent_max = latent_full.abs().max().item()

            # 5) Mini rollout: 5-step denoise from mid-noise, compare latent MSE
            rollout_latent = noisy_latent.clone()
            self._scheduler.set_timesteps(5, device=self.device)
            mini_ts = self._scheduler.timesteps
            for ti in mini_ts:
                t_in = ti.unsqueeze(0).to(self.dtype)
                with torch.autocast("cuda", dtype=self.dtype):
                    out = self._dit(
                        [rollout_latent[i] for i in range(rollout_latent.shape[0])],
                        t_in, text_emb, seq_len,
                        cond_concat=cond_with_action,
                        action_seq=action_cond,
                        action_dense_seq=action_dense_cond,
                        action_dense_zero_first_chunk=dense_zero_first_chunk,
                        action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                        action_domain_ids=action_domain_ids,
                        viewmats=viewmats,
                        Ks=Ks,
                        **eef_projection_kwargs,
                    )
                pred_step = torch.stack(out).to(torch.float32)
                rollout_latent = self._scheduler.step(
                    pred_step, ti, rollout_latent.to(torch.float32),
                )
                if bool(
                    getattr(self.tcfg, "gt_condition_replacement", False)
                    or getattr(self.mcfg, "gt_condition_replacement", False)
                ):
                    rollout_latent[:, :, :history_frames] = latent[:, :, :history_frames].to(
                        rollout_latent.dtype
                    )
                rollout_latent = rollout_latent.to(self.dtype)
            # Restore scheduler to the full-training grid so next train_step's
            # sigma sampling is unchanged.
            self._scheduler.set_timesteps(1000, device=self.device)
            rollout_mse = F.mse_loss(rollout_latent, latent).item()

            metrics = {
                "diag/action_sensitivity": action_sensitivity,
                "diag/cfg_gap": cfg_gap,
                "diag/history_effect": history_effect,
                "diag/latent_mean": latent_mean,
                "diag/latent_std": latent_std,
                "diag/latent_max_abs": latent_max,
                "diag/rollout_mse": rollout_mse,
            }

            # Color-coded warnings on main rank only
            if self.is_main:
                warnings = []
                if action_sensitivity < 0.01:
                    warnings.append(f"action_sensitivity={action_sensitivity:.4f} (<0.01 RED)")
                if cfg_gap < 0.005:
                    warnings.append(f"cfg_gap={cfg_gap:.4f} (<0.005 RED)")
                if history_effect == history_effect and history_effect < 0.01:  # not NaN
                    warnings.append(f"history_effect={history_effect:.4f} (<0.01 RED)")
                if latent_std > 3.0:
                    warnings.append(f"latent_std={latent_std:.2f} (>3 exploding)")
                if rollout_mse > 2.0:
                    warnings.append(f"rollout_mse={rollout_mse:.2f} (>2 train collapse)")

                if warnings:
                    logger.warning(
                        "[diagnose step=%d] ⚠ %s | action=%.4f cfg=%.4f "
                        "hist=%.4f rollout_mse=%.3f latent(μ=%.2f σ=%.2f max=%.2f)",
                        step, "; ".join(warnings), action_sensitivity, cfg_gap,
                        history_effect, rollout_mse, latent_mean, latent_std,
                        latent_max,
                    )
                else:
                    logger.info(
                        "[diagnose step=%d] action=%.4f cfg=%.4f hist=%.4f "
                        "rollout_mse=%.3f latent(μ=%.2f σ=%.2f max=%.2f)",
                        step, action_sensitivity, cfg_gap, history_effect,
                        rollout_mse, latent_mean, latent_std, latent_max,
                    )

                if self._wandb_run is not None:
                    import wandb
                    wandb.log(metrics, step=step)

            return metrics
        finally:
            self._dit.train()
