"""Track B world model: WAN 2.2 DiT with action conditioning.

Wraps the WAN 2.2 pretrained video model (finetuned with action conditioning)
as a BaseWorldModel. Operates in WAN VAE pixel-latent space.

The underlying model comes from third_party/yy-wan-training/:
  - DiT backbone: wan.modules.model_action_cond.WanModelAction
  - VAE: wan.modules.vae2_2.Wan2_2_VAE
  - Scheduler: wan.utils.fm.FlowMatchScheduler

Reference inference pipeline: third_party/yy-wan-training/wan_i2v_wrapper.py
Reference training loop: third_party/yy-wan-training/cosmos_predict2/models/wan_warped_model.py

Key conventions:
  - DiT forward: dit(x=[list of (C_in, F, H, W)], t=timestep, text_embedding=emb,
                     seq_len=int, action_seq=action_tensor_or_None)
    * action_seq=None → text-only path (action tokens excluded from cross-attn)
    * action_seq=tensor → dedicated action K/V conditioning, parallel to text K/V
  - VAE encode: vae.encode([tensor of (3, T, H, W)]) -> [tensor of (48, T_latent, H//16, W//16)]
  - VAE decode: vae.decode([tensor of (48, T, H, W)]) -> [tensor of (3, T_video, H*16, W*16)]
  - Rectified flow: noisy = (1-sigma)*clean + sigma*noise, target = noise - clean
  - Scheduler step: prev_sample = sample + pred * (sigma_next - sigma_cur)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from coachworld.config import WorldModelConfig
from coachworld.data.action_normalization import ActionNormalizer
from coachworld.world_model.action_contract import (
    apply_kv_future_only_action_mask,
    full_window_enabled,
    validate_action_timing_contract,
)
from coachworld.world_model.action_adapter import make_robot_action_config, prepare_action_tensor
from coachworld.world_model.base import BaseWorldModel, WorldModelInput, WorldModelOutput
from coachworld.world_model.eef_spatial_condition import (
    EEF_SPATIAL_CHANNELS_PER_SLOT,
    rasterize_eef_spatial_condition,
)
from coachworld.world_model.multi_camera import split_camera_latents
from coachworld.world_model.text_encoder import T5TextEncoder, make_null_text_embedding
from coachworld.wan import FlowMatchScheduler, Wan2_2_VAE, WanModelAction

logger = logging.getLogger(__name__)


class WanWorldModel(BaseWorldModel):
    """WAN 2.2 action-conditioned video world model (Track B).

    Latent space: WAN VAE pixel latents.
      - VAE z_dim = 48 channels for WAN 2.2 TI2V
      - Temporal compression: 4x (17 frames -> 5 latent frames)
      - Spatial compression: 16x (192x320 -> 12x20)

    The DiT takes action conditioning via its ``action_seq`` argument:
    a root-level ActionEncoder projects the robot condition sequence to
    DiT-dim tokens, which enter every block through a dedicated action K/V
    branch parallel to the pretrained text K/V branch.
    """

    # WAN 2.2 fixed constants
    VAE_TEMPORAL_STRIDE = 4
    VAE_SPATIAL_STRIDE = 16
    VAE_Z_DIM = 48
    PATCH_SIZE = (1, 2, 2)

    def __init__(
        self,
        config: WorldModelConfig,
        device: str = "cuda:0",
        text_encoder_device: Optional[str] = None,
        num_cameras: Optional[int] = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32
        validate_action_timing_contract(
            action_schema=getattr(config, "action_schema", "fixed"),
            action_rate=getattr(config, "action_condition_timestep_rate", "raw"),
        )
        config_num_cameras = int(getattr(config, "num_cameras", 3))
        self.num_cameras = int(num_cameras) if num_cameras is not None else config_num_cameras
        if self.num_cameras != config_num_cameras:
            raise ValueError(
                "WanWorldModel num_cameras mismatch: "
                f"constructor={self.num_cameras}, config={config_num_cameras}"
            )
        self.latent_height_per_view = int(getattr(config, "latent_height_per_view", 12))
        self.latent_width = int(getattr(config, "latent_width", 20))
        if self.num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {self.num_cameras}")
        if self.latent_height_per_view <= 0 or self.latent_width <= 0:
            raise ValueError(
                "latent geometry must be positive, got "
                f"latent_height_per_view={self.latent_height_per_view}, "
                f"latent_width={self.latent_width}"
            )

        # Lazy-loaded heavy components
        self._dit = None
        self._vae = None
        self._scheduler = None
        self._action_normalizer: Optional[ActionNormalizer] = None

        # T5 text encoder (separate device to save GPU memory if needed).
        # checkpoint + tokenizer are bundled inside the WAN model directory.
        self._text_encoder = T5TextEncoder(
            wan_model_dir=config.checkpoint,
            device=text_encoder_device or device,
            dtype=self.dtype,
        )

        t_future = max(1, config.pred_frames - 1)
        latent_h = self.num_cameras * self.latent_height_per_view
        latent_w = self.latent_width
        self._robot_action_config = make_robot_action_config(
            action_dim=config.action_dim,
            latent_C=48,
            latent_T=t_future,
            latent_H=latent_h,
            latent_W=latent_w,
            latent_height_per_view=self.latent_height_per_view,
            vae_spatial_stride=self.VAE_SPATIAL_STRIDE,
            vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
            action_schema=str(getattr(config, "action_schema", "fixed")),
            max_arm_slots=int(getattr(config, "max_arm_slots", 2)),
            action_max_time_steps=int(getattr(config, "action_max_time_steps", 512)),
            action_kv_enabled=bool(getattr(config, "action_kv_enabled", True)),
            action_v_init_scale=float(getattr(config, "action_v_init_scale", 0.1)),
            action_num_domains=int(getattr(config, "action_num_domains", 1)),
            action_domain_prompt_tokens=int(getattr(config, "action_domain_prompt_tokens", 0)),
            domain_aware_action_projection_enabled=bool(
                getattr(config, "domain_aware_action_projection_enabled", False)
            ),
            domain_aware_group_projection_enabled=bool(
                getattr(config, "domain_aware_group_projection_enabled", False)
            ),
            num_cameras=self.num_cameras,
            multi_view_position_mode=str(
                getattr(config, "multi_view_position_mode", "global")
            ),
            camera_id_embedding_enabled=bool(
                getattr(config, "camera_id_embedding_enabled", False)
            ),
            eef_projection_kv_enabled=bool(
                getattr(config, "eef_projection_kv_enabled", False)
            ),
        )
        self._robot_action_config["dense_action_film_enabled"] = bool(
            config.dense_action_film_enabled
        )
        self._robot_action_config["action_adaln_modulation_enabled"] = bool(
            getattr(config, "action_adaln_modulation_enabled", False)
        )
        self._robot_action_config["action_adaln_scale"] = float(
            getattr(config, "action_adaln_scale", 1.0)
        )
        self._robot_action_config["action_control_scale"] = float(
            getattr(config, "action_control_scale", 0.05)
        )
        self._robot_action_config["action_dense_rank"] = int(
            getattr(config, "action_dense_rank", 128)
        )
        self._robot_action_config["action_dense_layernorm_enabled"] = bool(
            getattr(config, "action_dense_layernorm_enabled", False)
        )
        self._robot_action_config["action_dense_relative_chunks"] = bool(
            getattr(config, "action_dense_relative_chunks", False)
        )
        self._robot_action_config["eef_projection_kv_enabled"] = bool(
            getattr(config, "eef_projection_kv_enabled", False)
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    @property
    def dit(self):
        if self._dit is None:
            self._load_models()
        return self._dit

    @property
    def vae(self):
        if self._vae is None:
            self._load_models()
        return self._vae

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._load_models()
        return self._scheduler

    def set_action_normalizer(self, normalizer: ActionNormalizer) -> None:
        self._action_normalizer = normalizer

    def _load_models(self) -> None:
        # WanModelAction, Wan2_2_VAE, FlowMatchScheduler imported at module level from coachworld.wan

        logger.info("Loading WAN 2.2 DiT from %s", self.config.checkpoint)
        # Pass action_config so the root-level ActionEncoder is created with
        # the right action_dim (Franka=7, AGI Bot=14). Without it the default
        # (7) would silently mismatch checkpoints trained with a different dim.
        self._dit = WanModelAction.from_pretrained(
            self.config.checkpoint,
            action_config=self._robot_action_config,
            device_map=None, low_cpu_mem_usage=False,
        )
        # Refresh V1.1 warm-start: copy freshly-loaded pretrained text K/V
        # into action K/V so action attention shares the pretrained pattern.
        # (At inference time this is mostly moot — finetuned checkpoints will
        # overwrite action K/V anyway via the subsequent load_state_dict call
        # in e.g. inference_smoke_test.py — but it's cheap and defensive.)
        self._dit.post_load_warm_start_action_kv()
        if bool(getattr(self.config, "prope_enabled", False)):
            self._dit.enable_prope(
                zero_init=bool(getattr(self.config, "prope_zero_init", True))
            )
        extra_input_channels = 1 if self.config.condition_mask_enabled else 0
        if bool(getattr(self.config, "eef_spatial_conditioning_enabled", False)):
            extra_input_channels += (
                int(self.config.max_arm_slots) * EEF_SPATIAL_CHANNELS_PER_SLOT
            )
        if extra_input_channels:
            self._dit.expand_input_channels(
                extra_channels=extra_input_channels,
                zero_init=True,
            )
        self._dit = self._dit.to(device=self.device, dtype=self.dtype)
        self._dit.eval()
        self._dit.requires_grad_(False)

        # Defensive assert: the new cross-attn design lives on a root-level
        # ActionEncoder. If this trips, action_config was silently ignored.
        assert hasattr(self._dit, "action_encoder"), (
            "WanModelAction has no .action_encoder — action_config path is "
            "broken. Inference will produce video with zero action sensitivity."
        )

        logger.info("Loading WAN 2.2 VAE from %s", self.config.vae_checkpoint)
        self._vae = Wan2_2_VAE(vae_pth=self.config.vae_checkpoint, device=self.device)

        self._scheduler = FlowMatchScheduler(
            shift=self.config.flow_matching_shift,
            num_train_timesteps=1000,
            extra_one_step=True,
        )
        logger.info(
            "WAN world model ready (device=%s, dtype=%s, steps=%d)",
            self.device, self.dtype, self.config.num_inference_steps,
        )

    # ------------------------------------------------------------------
    # BaseWorldModel interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        wm_input: WorldModelInput,
        num_steps: Optional[int] = None,
        decode_frames: bool = True,
    ) -> WorldModelOutput:
        """Predict future frames via rectified-flow denoising.

        Args:
            wm_input: Structured input (observation, actions, instruction).
            num_steps: Override inference steps. Use 5-10 for fast diagnosis,
                full (50) for high-quality generation / data collection.
            decode_frames: If False, skip VAE decode (faster when only
                latent-space scoring is needed).
        """
        current_latent = wm_input.current_obs.to(device=self.device, dtype=self.dtype)
        text_emb = self._prepare_text_embedding(wm_input.instruction)

        if current_latent.dim() == 3:
            current_latent = current_latent.unsqueeze(1)  # (C, h, w) -> (C, 1, h, w)
        elif current_latent.dim() != 4:
            raise ValueError(
                "WanWorldModel.step expected current_obs with shape "
                "(C,H,W) or (C,T,H,W), got "
                f"{tuple(current_latent.shape)}"
            )

        C, T_cond_in, h, w = current_latent.shape
        expected_c = self.VAE_Z_DIM
        if C != expected_c:
            raise ValueError(
                f"WanWorldModel expected {expected_c} latent channels "
                f"(cameras are stacked along height, not channels), got {C}. "
                "Use height-stacked multi-camera latents or construct "
                "WanWorldModel(num_cameras=1) for single-camera checkpoints."
            )
        full_window = full_window_enabled(self.config)
        if full_window:
            history_frames = min(max(1, T_cond_in), self.config.history_length)
            current_latent = current_latent[:, :history_frames, :, :]
            T_latent = history_frames + self.config.pred_frames
        else:
            # Short-window i2v mode: use one anchor frame. If a caller passes a
            # history stack, the last slice is the current frame.
            history_frames = 1
            if T_cond_in > 1:
                current_latent = current_latent[:, -1:, :, :]
            T_latent = self.config.pred_frames

        action_seq = self._prepare_action_cond(
            wm_input.action_sequence,
            num_latent_frames=T_latent,
        )
        action_dense_seq = self._prepare_dense_action_cond(
            wm_input.action_sequence,
            num_latent_frames=T_latent,
        )
        action_domain_ids = self._prepare_action_domain_ids(wm_input, action_seq)
        dense_zero_first_chunk = False
        dense_zero_prefix_chunks = 0
        if action_seq is not None and self.config.future_only_action_conditioning:
            action_seq, action_mask = apply_kv_future_only_action_mask(
                action_seq,
                history_frames=history_frames,
                action_rate=str(getattr(self.config, "action_condition_timestep_rate", "raw")),
                vae_temporal_stride=self.VAE_TEMPORAL_STRIDE,
            )
            dense_zero_first_chunk = action_mask.dense_zero_first_chunk
            dense_zero_prefix_chunks = action_mask.dense_zero_prefix_chunks

        viewmats = None
        Ks = None
        if bool(getattr(self.config, "prope_enabled", False)):
            if wm_input.viewmats is None or wm_input.Ks is None:
                raise ValueError(
                    "world_model.prope_enabled=true requires WorldModelInput.viewmats/Ks"
                )
            viewmats = wm_input.viewmats.to(device=self.device, dtype=self.dtype)
            Ks = wm_input.Ks.to(device=self.device, dtype=self.dtype)
            if viewmats.dim() == 4:
                viewmats = viewmats.unsqueeze(0)
            if Ks.dim() == 4:
                Ks = Ks.unsqueeze(0)
            if viewmats.shape != (1, T_latent, self.num_cameras, 4, 4):
                raise ValueError(
                    "WanWorldModel PRoPE expected viewmats shape "
                    f"(1,{T_latent},{self.num_cameras},4,4), got {tuple(viewmats.shape)}"
                )
            if Ks.shape != (1, T_latent, self.num_cameras, 3, 3):
                raise ValueError(
                    "WanWorldModel PRoPE expected Ks shape "
                    f"(1,{T_latent},{self.num_cameras},3,3), got {tuple(Ks.shape)}"
                )
        eef_projection_kwargs = {}
        eef_gripper = None
        eef_projection_required = bool(
            getattr(self.config, "eef_projection_kv_enabled", False)
            or getattr(self.config, "eef_spatial_conditioning_enabled", False)
        )
        if eef_projection_required:
            if (
                wm_input.eef_uv is None
                or wm_input.eef_depth is None
                or wm_input.eef_valid is None
                or wm_input.eef_image_hw is None
            ):
                raise ValueError(
                    "world_model.eef_projection_kv_enabled=true requires "
                    "WorldModelInput eef_uv/eef_depth/eef_valid/eef_image_hw"
                )
            eef_uv = wm_input.eef_uv.to(device=self.device, dtype=self.dtype)
            eef_depth = wm_input.eef_depth.to(device=self.device, dtype=self.dtype)
            eef_valid = wm_input.eef_valid.to(device=self.device)
            eef_image_hw = wm_input.eef_image_hw.to(device=self.device)
            if eef_uv.dim() == 3:
                eef_uv = eef_uv.unsqueeze(0)
                eef_depth = eef_depth.unsqueeze(0)
                eef_valid = eef_valid.unsqueeze(0)
                eef_image_hw = eef_image_hw.unsqueeze(0)
            elif eef_uv.dim() == 4 and eef_uv.shape[0] == T_latent and eef_uv.shape[1] == self.num_cameras:
                eef_uv = eef_uv.unsqueeze(0)
                eef_depth = eef_depth.unsqueeze(0)
                eef_valid = eef_valid.unsqueeze(0)
                eef_image_hw = eef_image_hw.unsqueeze(0)
            elif eef_uv.dim() not in {4, 5}:
                raise ValueError(
                    "expected eef_uv shape (F,V,2), (F,V,S,2), "
                    f"(1,F,V,2), or (1,F,V,S,2); got {tuple(eef_uv.shape)}"
                )
            if eef_uv.dim() == 4:
                expected_uv = (1, T_latent, self.num_cameras, 2)
                expected_scalar = (1, T_latent, self.num_cameras)
            else:
                if eef_uv.shape[-1] != 2:
                    raise ValueError(f"expected eef_uv last dim 2, got {tuple(eef_uv.shape)}")
                expected_uv = tuple(eef_uv.shape)
                expected_scalar = tuple(eef_uv.shape[:-1])
                if expected_uv[:3] != (1, T_latent, self.num_cameras):
                    raise ValueError(
                        "expected eef_uv leading shape "
                        f"(1,{T_latent},{self.num_cameras},S,2), got {tuple(eef_uv.shape)}"
                    )
            if tuple(eef_uv.shape) != expected_uv:
                raise ValueError(f"expected eef_uv shape {expected_uv}, got {tuple(eef_uv.shape)}")
            if tuple(eef_image_hw.shape) not in {expected_uv, (1, T_latent, self.num_cameras, 2)}:
                raise ValueError(
                    f"expected eef_image_hw shape {expected_uv} or "
                    f"{(1, T_latent, self.num_cameras, 2)}, got {tuple(eef_image_hw.shape)}"
                )
            if tuple(eef_depth.shape) != expected_scalar:
                raise ValueError(
                    f"expected eef_depth shape {expected_scalar}, got {tuple(eef_depth.shape)}"
                )
            if tuple(eef_valid.shape) != expected_scalar:
                raise ValueError(
                    f"expected eef_valid shape {expected_scalar}, got {tuple(eef_valid.shape)}"
                )
            eef_projection_kwargs = {
                "eef_uv": eef_uv,
                "eef_depth": eef_depth,
                "eef_valid": eef_valid,
                "eef_image_hw": eef_image_hw,
            }
            if not bool(getattr(self.config, "eef_projection_kv_enabled", False)):
                eef_projection_kwargs = {}
            if bool(getattr(self.config, "eef_spatial_conditioning_enabled", False)):
                if wm_input.eef_gripper is None:
                    raise ValueError(
                        "eef_spatial_conditioning_enabled=true requires "
                        "WorldModelInput.eef_gripper"
                    )
                eef_gripper = wm_input.eef_gripper.to(
                    device=self.device,
                    dtype=self.dtype,
                )
                if eef_gripper.dim() == 2:
                    eef_gripper = eef_gripper.unsqueeze(0)
                expected_gripper = (
                    1,
                    T_latent,
                    expected_scalar[-1] if len(expected_scalar) == 4 else 1,
                )
                if tuple(eef_gripper.shape) != expected_gripper:
                    raise ValueError(
                        f"expected eef_gripper shape {expected_gripper}, "
                        f"got {tuple(eef_gripper.shape)}"
                    )

        # Inpainting: keep condition/history frames clean, denoise the rest.
        # All mask arithmetic in float32 to avoid bf16 precision loss.
        condition_z = current_latent.float()  # (C, T_cond, h, w), fp32

        # Initial noise. Frame 0 is inpainted to the clean current frame;
        # future frames start from standard Gaussian noise.
        noise_base = torch.randn(
            C, T_latent, h, w, device=self.device, dtype=torch.float32
        )
        noise = noise_base

        # mask: condition frames = 0 (keep clean), future = 1 (denoise)
        m = torch.ones(1, T_latent, 1, 1, device=self.device, dtype=torch.float32)
        m[:, :history_frames, :, :] = 0.0

        condition_padded = torch.zeros_like(noise)
        condition_padded[:, :history_frames, :, :] = condition_z
        latent = (1.0 - m) * condition_padded + m * noise
        condition_mask = None
        if self.config.condition_mask_enabled:
            condition_mask = torch.zeros(
                1, 1, T_latent, h, w, device=self.device, dtype=self.dtype
            )
            condition_mask[:, :, :history_frames, :, :] = 1.0
        eef_spatial = None
        if bool(getattr(self.config, "eef_spatial_conditioning_enabled", False)):
            eef_spatial = rasterize_eef_spatial_condition(
                uv=eef_uv,
                depth=eef_depth,
                valid=eef_valid,
                image_hw=eef_image_hw,
                gripper=eef_gripper,
                latent_height_per_view=self.latent_height_per_view,
                latent_width=self.latent_width,
                sigma_px=float(self.config.eef_spatial_sigma_px),
                depth_scale=float(self.config.eef_spatial_depth_scale),
            )
            if tuple(eef_spatial.shape[2:]) != (T_latent, h, w):
                raise ValueError(
                    "EEF spatial raster does not match latent geometry: "
                    f"raster={tuple(eef_spatial.shape)}, latent={(T_latent, h, w)}"
                )
        cond_concat = condition_mask if eef_spatial is None else (
            eef_spatial
            if condition_mask is None
            else torch.cat([condition_mask, eef_spatial], dim=1)
        )
        cond_without_action = condition_mask if eef_spatial is None else (
            torch.zeros_like(eef_spatial)
            if condition_mask is None
            else torch.cat([condition_mask, torch.zeros_like(eef_spatial)], dim=1)
        )

        # Scheduler setup — respect num_steps override for fast diagnosis
        effective_steps = num_steps or self.config.num_inference_steps
        self.scheduler.set_timesteps(effective_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Positional encoding seq_len
        patch_t, patch_h, patch_w = self.PATCH_SIZE
        seq_len = (T_latent // patch_t) * (h // patch_h) * (w // patch_w)

        # 3-branch CFG contexts:
        #   uncond: zero text, no action  (context = zero text tokens only)
        #   text:   real text, no action  (context = text tokens only)
        #   full:   real text + action    (context = [text; action_tokens])
        # This decomposition lets us weight text vs action independently:
        #   pred = uncond + w_text * (text - uncond) + w_act * (full - text)
        # Set w_act=0 to disable action guidance (text-only CFG).
        # Set w_text=0 to disable text guidance (action-only CFG, rare).
        context_cond = text_emb
        # Null text for CFG uncond branch: tiny noise (not strict zeros) —
        # zeros passed through the pretrained self.text_embedding MLP produce
        # an identical bias vector at all 512 positions, degenerating
        # cross-attn K/V and occasionally NaN'ing bf16 flash_attn. Must match
        # training-time null-text convention in trainer.compute_loss.
        context_null = torch.randn_like(context_cond) * 0.01

        # Guidance ramps: linearly increase from min to max across denoising
        # steps (Ctrl-World style). We use the same ramp for both text and
        # action weights — they both benefit from more guidance at low-noise
        # steps where fine detail matters.
        guidance_min = getattr(self.config, "guidance_scale_min", self.config.guidance_scale)
        guidance_max = getattr(self.config, "guidance_scale_max", self.config.guidance_scale)

        for step_idx, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).to(self.device)
            latent_dit = latent.to(self.dtype)

            # Linear ramp for both weights
            frac = step_idx / max(len(timesteps) - 1, 1)
            w_guidance = guidance_min + frac * (guidance_max - guidance_min)

            with torch.autocast("cuda", dtype=self.dtype):
                # Branch 1: full conditioning (text + action)
                pred_full = self.dit(
                    [latent_dit], t_tensor, context_cond, seq_len,
                    cond_concat=cond_concat,
                    action_seq=action_seq,
                    action_dense_seq=action_dense_seq,
                    action_dense_zero_first_chunk=dense_zero_first_chunk,
                    action_dense_zero_prefix_chunks=dense_zero_prefix_chunks,
                    action_domain_ids=action_domain_ids,
                    viewmats=viewmats,
                    Ks=Ks,
                    **eef_projection_kwargs,
                )[0]

                if w_guidance > 1.0:
                    # Branch 2: text-only (no action — action_seq=None)
                    pred_text = self.dit(
                        [latent_dit], t_tensor, context_cond, seq_len,
                        cond_concat=cond_without_action,
                        action_seq=None,
                        action_dense_seq=None,
                        viewmats=viewmats,
                        Ks=Ks,
                    )[0]
                    # Branch 3: fully unconditional (zero text, no action)
                    pred_uncond = self.dit(
                        [latent_dit], t_tensor, context_null, seq_len,
                        cond_concat=cond_without_action,
                        action_seq=None,
                        action_dense_seq=None,
                        viewmats=viewmats,
                        Ks=Ks,
                    )[0]

                    # Decomposed CFG: same net weight w_guidance applied to both
                    # the text-alone delta and the action-alone delta. Keeping
                    # them equal matches single-guidance-scale Ctrl-World
                    # behavior while exposing independent knobs for ablation.
                    pred = (
                        pred_uncond
                        + w_guidance * (pred_text - pred_uncond)
                        + w_guidance * (pred_full - pred_text)
                    )
                else:
                    pred = pred_full

            # Euler step in float32 for numerical stability
            latent = self.scheduler.step(pred.float(), t, latent)

            # Re-apply condition-frame inpainting (float32).
            latent = (1.0 - m) * condition_padded + m * latent

        # Convert final latent to model dtype for decode / downstream
        latent = latent.to(self.dtype)

        decoded = self.decode(latent) if decode_frames else None
        # decode() returns (T, H, W, 3) ndarray; convert to list per type hint
        if decoded is not None:
            predicted_frames = [decoded[i] for i in range(len(decoded))]
        else:
            predicted_frames = None
        next_latent = latent[:, -1, :, :]  # (C, h, w)

        return WorldModelOutput(
            predicted_obs=latent,
            predicted_frames=predicted_frames,
            next_latent=next_latent,
        )

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """Encode RGB frames to WAN VAE latent space.

        Args:
            frames: (N, H, W, 3) uint8 RGB.

        Returns:
            Latent: (C, T_latent, h, w) where C=48 for Wan2.2 TI2V,
            T_latent = 1 + (N-1)//4, h=H//16, w=W//16.
        """
        # (N, H, W, 3) uint8 -> (3, N, H, W) float in [-1, 1]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
        x = x / 127.5 - 1.0
        x = x.permute(1, 0, 2, 3)  # (3, N, H, W)
        x = x.to(device=self.device, dtype=self.dtype)
        # VAE.encode expects list of (3, T, H, W) tensors
        latents = self.vae.encode([x])
        return latents[0]  # (48, T_latent, H//16, W//16)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> Optional[np.ndarray]:
        """Decode WAN VAE latents to RGB.

        Args:
            latents: (C, T, H, W) latent tensor.

        Returns:
            (N, H, W, 3) uint8 RGB frames, or None on failure.
            Multi-camera latents are split by view before VAE decode and
            stacked vertically in RGB space. This matches how the latents were
            extracted and avoids decoding camera seams as one continuous image.
        """
        latents = latents.to(device=self.device, dtype=self.dtype)
        if self.num_cameras > 1:
            if latents.dim() != 4:
                raise ValueError(
                    "WanWorldModel.decode expected latents with shape "
                    f"(C,T,H,W), got {tuple(latents.shape)}"
                )
            h_total = int(latents.shape[2])
            if h_total % int(self.num_cameras) != 0:
                raise ValueError(
                    f"Latent height {h_total} is not divisible by "
                    f"num_cameras={self.num_cameras}; cannot per-camera decode."
                )
            camera_latents = split_camera_latents(latents, self.num_cameras)
            decoded = self.vae.decode(camera_latents)
        else:
            decoded = self.vae.decode([latents])
        if decoded is None:
            return None
        if self.num_cameras > 1:
            if len(decoded) != self.num_cameras:
                raise RuntimeError(
                    f"WAN VAE returned {len(decoded)} decoded views, "
                    f"expected {self.num_cameras}"
                )
            view_frames = [
                ((view.float().permute(1, 2, 3, 0) + 1.0) * 127.5)
                .clamp(0, 255)
                .byte()
                .cpu()
                .numpy()
                for view in decoded
            ]
            n = min(int(frames.shape[0]) for frames in view_frames)
            out = np.stack(
                [
                    np.concatenate([frames[i] for frames in view_frames], axis=0)
                    for i in range(n)
                ],
                axis=0,
            )
        else:
            # decoded[0]: (3, T_out, H_out, W_out) in [-1, 1]
            out = decoded[0].float().permute(1, 2, 3, 0)  # (T, H, W, 3)
            out = ((out + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
        return out

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self._action_normalizer is not None:
            return self._action_normalizer.normalize(action)
        return action

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        if self._action_normalizer is not None:
            return self._action_normalizer.denormalize(action_norm)
        return action_norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_action_condition(self, action: torch.Tensor) -> torch.Tensor:
        """Apply the configured action schema to a prepared (B,T,D) tensor."""
        schema = str(getattr(self.config, "action_schema", "fixed"))
        if schema == "fixed":
            if action.dim() != 3:
                raise ValueError(f"fixed action condition must be 3D, got {tuple(action.shape)}")
            return action
        if schema == "arm_slot":
            slots = int(getattr(self.config, "max_arm_slots", 2))
            action_dim = int(getattr(self.config, "action_dim", action.shape[-1] - 1))
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

    def _prepare_action_domain_ids(
        self,
        wm_input: WorldModelInput,
        action_seq: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if action_seq is None:
            return None
        domain_features_enabled = bool(
            getattr(self.config, "domain_aware_action_projection_enabled", False)
            or getattr(self.config, "domain_aware_group_projection_enabled", False)
            or int(getattr(self.config, "action_domain_prompt_tokens", 0)) > 0
        )
        if wm_input.domain_id is None:
            if domain_features_enabled:
                raise ValueError(
                    "WorldModelInput.domain_id is required when domain-aware "
                    "action projection or domain prompts are enabled"
                )
            return None
        domain_id = int(wm_input.domain_id)
        num_domains = int(getattr(self.config, "action_num_domains", 1))
        if domain_id < 0 or domain_id >= num_domains:
            raise ValueError(
                f"WorldModelInput.domain_id={domain_id} outside [0,{num_domains})"
            )
        return torch.tensor([domain_id], device=self.device, dtype=torch.long)

    def _prepare_action_cond(
        self,
        action_sequence: np.ndarray,
        num_latent_frames: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Convert raw action array to the tensor format DiT expects.

        The DiT's root-level ActionEncoder consumes a fixed-vector tensor
        ``(1,T,D)`` or arm-slot tensor ``(1,T,S,D+1)`` and produces action
        tokens for the dedicated action K/V branch. This function only handles
        temporal alignment; encoding is done inside ``WanModelAction._forward``.
        """
        if action_sequence is None or len(action_sequence) == 0:
            return None
        action_norm = self.normalize_action(action_sequence)
        if str(getattr(self.config, "action_condition_timestep_rate", "raw")) == "latent":
            target_len = int(num_latent_frames or self.config.pred_frames)
            raw_len = (target_len - 1) * self.VAE_TEMPORAL_STRIDE + 1
            if action_norm.shape[0] >= raw_len:
                action_norm = action_norm[:raw_len:self.VAE_TEMPORAL_STRIDE]
            if action_norm.shape[0] < target_len:
                pad = np.tile(action_norm[-1:], (target_len - action_norm.shape[0], 1))
                action_norm = np.concatenate([action_norm, pad], axis=0)
            elif action_norm.shape[0] > target_len:
                action_norm = action_norm[:target_len]
            action = torch.from_numpy(action_norm).to(
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            return self._format_action_condition(action)

        T_latent = int(num_latent_frames or self.config.pred_frames)
        num_video_frames = (T_latent - 1) * self.VAE_TEMPORAL_STRIDE + 1
        action = prepare_action_tensor(
            action_norm,
            num_video_frames,
            self.device,
            dtype=torch.float32,
        )
        if action is None:
            return None
        return self._format_action_condition(action)

    def _prepare_dense_action_cond(
        self,
        action_sequence: np.ndarray,
        num_latent_frames: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Prepare raw/video-rate action for dense latent-frame chunks."""
        if action_sequence is None or len(action_sequence) == 0:
            return None
        action_norm = self.normalize_action(action_sequence)
        T_latent = int(num_latent_frames or self.config.pred_frames)
        num_video_frames = (T_latent - 1) * self.VAE_TEMPORAL_STRIDE + 1
        if (
            str(getattr(self.config, "action_condition_timestep_rate", "raw"))
            == "latent"
            and action_norm.shape[0] <= T_latent
        ):
            if action_norm.shape[0] < T_latent:
                pad = np.tile(action_norm[-1:], (T_latent - action_norm.shape[0], 1))
                action_norm = np.concatenate([action_norm, pad], axis=0)
            action = torch.from_numpy(action_norm[:T_latent]).to(
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            return self._format_action_condition(action)
        action = prepare_action_tensor(
            action_norm,
            num_video_frames,
            self.device,
            dtype=torch.float32,
        )
        if action is None:
            return None
        return self._format_action_condition(action)

    def _prepare_text_embedding(self, instruction: str) -> torch.Tensor:
        """Encode instruction text to embedding for DiT cross-attention.

        The DiT expects text_embedding of shape (B, L, D) where D=4096
        (matching UMT5-XXL / T5 embedding dimension used in WAN 2.2).

        Empty instructions use the null-text embedding used by training and
        CFG branches. If text conditioning is enabled and T5 encoding fails,
        the error is raised; inference must not silently drop text.
        """
        if not bool(getattr(self.config, "text_conditioning_enabled", True)):
            return make_null_text_embedding(device=str(self.device), dtype=self.dtype)
        if not instruction:
            return make_null_text_embedding(device=str(self.device), dtype=self.dtype)

        emb = self._text_encoder.encode_single(instruction)
        return emb.to(device=self.device, dtype=self.dtype)
