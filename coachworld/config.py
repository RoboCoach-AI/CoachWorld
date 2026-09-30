"""Hierarchical YAML-based configuration for CoachWorld.

All settings live in YAML files under configs/.
CLI overrides are supported via dotted paths: --world_model.action_dim=14
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf, SCMode


# ---------------------------------------------------------------------------
# Per-module config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    backbone: str = "wan2.2"
    checkpoint: str = ""
    vae_checkpoint: str = ""
    action_dim: int = 7
    text_dim: int = 4096  # Text embedding dimension (4096 for T5-XXL / UMT5)
    text_conditioning_enabled: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 2.0
    guidance_scale_min: float = 2.0  # Guidance ramp: start value
    guidance_scale_max: float = 3.0  # Guidance ramp: end value (linearly interpolated)
    pred_frames: int = 5
    history_length: int = 6
    # Training
    flow_matching_shift: float = 3.0
    mixed_precision: str = "bf16"
    fsdp_enabled: bool = True
    fsdp_shard_size: int = 8
    gradient_checkpoint: bool = True
    learning_rate: float = 5e-5
    # Multiplier applied to learning_rate for ActionEncoder params (root-level
    # MLP that feeds cross-attn K/V). 1.0 = uniform. >1 speeds up the encoder,
    # useful because its final linear is zero-init so it contributes ~0 early.
    action_lr_mult: float = 1.0
    max_train_steps: int = 0
    batch_size: int = 2
    grad_accum_steps: int = 2
    # Simulator conditioning switches. These keep the flow target clean and
    # make action/state a condition rather than part of the source.
    condition_mask_enabled: bool = False
    gt_condition_replacement: bool = False
    future_only_action_conditioning: bool = False
    action_schema: str = "fixed"  # fixed | arm_slot
    max_arm_slots: int = 2
    action_max_time_steps: int = 512
    action_kv_enabled: bool = True
    action_v_init_scale: float = 0.1
    action_num_domains: int = 1
    action_domain_prompt_tokens: int = 0
    domain_aware_action_projection_enabled: bool = False
    domain_aware_group_projection_enabled: bool = False
    dense_action_film_enabled: bool = False
    action_adaln_modulation_enabled: bool = False
    action_adaln_scale: float = 1.0
    action_dense_rank: int = 128
    action_dense_relative_chunks: bool = False
    action_control_scale: float = 0.05
    action_dense_layernorm_enabled: bool = False
    eef_projection_kv_enabled: bool = False
    # Camera-aligned EEF control raster concatenated with the WAN latent.
    # Each arm slot contributes occupancy, camera-depth, and gripper channels.
    eef_spatial_conditioning_enabled: bool = False
    eef_spatial_sigma_px: float = 16.0
    eef_spatial_depth_scale: float = 2.0
    full_window: bool = False
    action_condition_timestep_rate: str = "raw"  # raw | latent
    history_selector: str = "recent"  # recent | sparse | first_recent | first_offset_recent
    history_offsets: list = field(default_factory=list)
    history_dilations: list = field(default_factory=lambda: [1, 2])
    history_collapse_prob: float = 0.0
    history_eval_dilation: int = 2
    history_rollout_dilation: int = 2
    # Multi-view token topology. The default preserves the historical
    # height-stacked latent behavior. Set multi_view_position_mode=view_local
    # and camera_id_embedding_enabled=true for PE1.
    multi_view_position_mode: str = "global"  # global | view_local
    camera_id_embedding_enabled: bool = False
    prope_enabled: bool = False
    prope_zero_init: bool = True
    prope_spatial_stride: int = 16
    # Latent geometry consumed by action-side modules. These are Wan-VAE latent
    # spatial sizes per camera/view, not RGB sizes. For 192x320 RGB this is
    # 12x20; for 384x640 it is 24x40.
    num_cameras: int = 3
    latent_height_per_view: int = 12
    latent_width: int = 20

    # Experimental minWM-style causal Wan backend. These fields are ignored by
    # the current action-conditioned WanModelAction path and are only used when
    # ``world_model.backbone == "wan2.2_causal"``.
    causal_dim: int = 128
    causal_ffn_dim: int = 256
    causal_num_heads: int = 4
    causal_num_layers: int = 2
    causal_freq_dim: int = 64
    causal_text_tokens: int = 16
    causal_num_frame_per_block: int = 2
    causal_local_attn_size: int = -1
    causal_sink_size: int = 0
    causal_prope_enabled: bool = False
    causal_prope_zero_init: bool = True
    causal_preserve_zero_head: bool = False
    causal_self_forcing_enabled: bool = False
    causal_self_forcing_start_step: int = 0
    causal_self_forcing_history_frames: int = 1
    causal_self_forcing_sigma: float = 0.5
    causal_self_forcing_prob: float = 1.0
    causal_proprio_head_enabled: bool = False
    causal_proprio_mid_dim: int = 512


@dataclass
class ExpertConfig:
    backbone: str = "pi0.5"  # "pi0.5" | "openvla_oft"
    backbone_checkpoint: str = ""
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])
    action_chunk_size: int = 10
    action_space: str = "cartesian_7d"  # "cartesian_7d" | "cartesian_14d" | "joint_vel"
    device: str = "cuda:0"


@dataclass
class RouterConfig:
    strategy: str = "llm_skill_card"  # "llm_skill_card" | "skill_card" | "clip" | "vlm" | "random" | "wm_ranking"
    clip_model: str = "openai/clip-vit-base-patch32"
    similarity_threshold: float = 0.3
    use_vlm_refinement: bool = False
    vlm_model: str = "qwen-vl"
    wm_ranking_num_steps: int = 10
    wm_ranking_confidence_temperature: float = 1.0


@dataclass
class EvaluatorConfig:
    vlm_type: str = "qwen-vl"  # "qwen-vl" | "openai" | "dummy"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # Local VLM model name
    api_model_name: str = "gpt-4o"  # OpenAI API model name (used by APIEvaluator)
    device: str = "auto"
    max_retries: int = 3
    eval_num_frames: int = 8
    success_weight: float = 1.0
    consistency_weight: float = 0.5


@dataclass
class CoachingConfig:
    """Legacy prototype fields retained so historical YAML still parses.

    Current CoachWorld training does not consume this section. New coaching
    orchestration must use an explicit experiment config rather than treating
    imagined teacher trajectories as demonstrations.
    """

    diagnosis_episodes: int = 100
    failure_threshold: float = 0.5
    coaching_data_per_task: int = 10
    lora_training_steps: int = 500
    lora_learning_rate: float = 1e-4
    verification_episodes: int = 50
    max_coaching_rounds: int = 5
    teacher_expert: str = "generalist"
    quality_threshold: float = 0.7  # Min reward to accept teacher demo
    max_teacher_attempts: int = 5  # Max rollout attempts per demo target
    # Rollout parameters
    max_rollout_steps: int = 20  # WM prediction steps per episode
    diagnosis_inference_steps: int = 10  # Fast WM denoising for diagnosis (5-10)
    generation_inference_steps: int = 50  # Full WM denoising for data generation
    history_length: int = 6  # Past steps for WM temporal conditioning


@dataclass
class WMTrainingConfig:
    """Training-specific configuration for world model training."""

    # Loss
    # 1.0 matches standard flow-matching conventions. With gradient_clip=1.0 this
    # means clipping acts on the true loss gradient magnitude (no hidden 10x scaling).
    loss_scale: float = 1.0

    # Optimizer / schedule
    warmup_steps: int = 0
    lr_schedule: str = "constant"  # constant | cosine
    lr_min_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Training-time condition dropout. The fixed fields are still supported;
    # optional start/end/decay fields define a linear schedule over micro-steps.
    cfg_dropout_text: float = 0.10
    cfg_dropout_action: float = 0.10
    cfg_dropout_text_start: float = -1.0
    cfg_dropout_text_end: float = -1.0
    cfg_dropout_text_decay_steps: int = 0
    cfg_dropout_action_start: float = -1.0
    cfg_dropout_action_end: float = -1.0
    cfg_dropout_action_decay_steps: int = 0

    # History/current frame noise schedule (Ctrl-World style).
    # Current frame (slice index 0) gets sigma ~ U(0, history_noise_max) during
    # training so the model learns to preserve low-noise anchor frames at inference.
    history_noise_max: float = 0.30
    # When GT replacement is disabled, retain an exact-history subset while
    # corrupting the rest. This closes the train/rollout context gap without
    # removing clean-history supervision entirely.
    history_clean_prob: float = 0.25

    # LDA-1B-style auxiliary action reconstruction loss weight. Adds a
    # small decoder head that predicts per-frame action from the last
    # block's visual tokens; the MSE between predicted and ground-truth
    # action is added to the main flow-matching loss at this weight.
    # 0.0 disables the aux loss entirely (falls back to pure flow matching).
    # 0.1 is a reasonable starting value — strong enough to force the
    # action path to carry signal, weak enough not to distort the video
    # prediction objective.
    aux_action_weight: float = 0.0
    aux_action_mask_weight: float = 0.05

    # Counterfactual action ranking loss. Runs an extra forward with shuffled
    # actions and penalizes cases where the true-action prediction is not
    # better than the counterfactual-action prediction by at least this margin.
    # Keep disabled by default; recent A800 runs showed this can provide wrong
    # gradients early in training when a counterfactual action is accidentally
    # closer to GT.
    counterfactual_action_weight: float = 0.0
    counterfactual_action_margin: float = 0.02
    counterfactual_action_mode: str = "roll"  # roll | time_reverse | sign_flip | zero
    counterfactual_action_interval: int = 1  # compute every N train steps
    action_warmup_steps: int = 0  # freeze non-action params for first N steps
    action_param_max_abs: float = 100.0  # repair/clamp action-side params above this

    # Simulator condition mask / optional GT condition replacement.
    # When enabled, a binary condition-frame mask is appended to the DiT input
    # channels and condition frames are kept clean during training. Action
    # conditioning can be zeroed on condition frames so only future frames are
    # controlled by action.
    condition_mask_enabled: bool = False
    gt_condition_replacement: bool = False
    future_only_action_conditioning: bool = False
    action_schema: str = "fixed"  # fixed | arm_slot
    max_arm_slots: int = 2
    action_max_time_steps: int = 512
    action_kv_enabled: bool = True
    action_v_init_scale: float = 0.1
    action_num_domains: int = 1
    action_domain_prompt_tokens: int = 0
    domain_aware_action_projection_enabled: bool = False
    domain_aware_group_projection_enabled: bool = False
    dense_action_film_enabled: bool = False
    action_adaln_modulation_enabled: bool = False
    action_adaln_scale: float = 1.0
    action_dense_rank: int = 128
    action_dense_relative_chunks: bool = False
    action_control_scale: float = 0.05
    action_dense_layernorm_enabled: bool = False
    eef_projection_kv_enabled: bool = False
    eef_spatial_conditioning_enabled: bool = False
    eef_spatial_sigma_px: float = 16.0
    eef_spatial_depth_scale: float = 2.0
    full_window: bool = False
    action_source: str = "commanded_action"  # commanded_action | commanded_original_action | commanded_joint_position_gripper | commanded_cartesian_velocity_gripper | commanded_joint_velocity_gripper | causal_state_plus_cartesian_velocity_gripper | causal_state_quat_cartesian_velocity_gripper | observation_state/abs_state*/abs_plus* only with COACHWORLD_ALLOW_FUTURE_STATE_ACTION_SOURCE=1
    action_condition_timestep_rate: str = "raw"  # raw | latent
    history_selector: str = "recent"  # recent | sparse | first_recent | first_offset_recent
    history_offsets: list = field(default_factory=list)
    history_dilations: list = field(default_factory=lambda: [1, 2])
    history_collapse_prob: float = 0.0
    history_eval_dilation: int = 2
    history_rollout_dilation: int = 2
    # Generated-history training for the non-causal H/F model. Selected batches
    # roll out one of the configured chunk horizons without gradients, then
    # train the following chunk from the accumulated generated sparse history.
    generated_history_enabled: bool = False
    generated_history_probability: float = 0.0
    generated_history_denoise_steps: int = 2
    generated_history_blend: float = 0.5
    generated_history_unroll_chunks: list = field(default_factory=lambda: [1])
    generated_history_start_step: int = 0
    generated_history_seed: int = 0
    generated_history_dataset_keys: list = field(default_factory=list)
    multi_view_position_mode: str = "global"  # global | view_local
    camera_id_embedding_enabled: bool = False
    prope_enabled: bool = False
    prope_zero_init: bool = True
    prope_spatial_stride: int = 16

    # Logging / checkpointing
    output_dir: str = "experiments/wm_train"
    log_interval: int = 1
    save_interval: int = 0
    save_final_checkpoint: bool = True
    val_interval: int = 0
    diagnose_interval: int = 0  # Lightweight training-time diagnostics
    # Periodic decoded-RGB validation. Metrics compare generated future frames
    # against the WAN-VAE decode of held-out GT latents.
    val_metrics_enabled: bool = False
    val_metrics_interval: int = 0
    val_metrics_num_samples: int = 0
    val_metrics_inference_steps: int = 25
    val_metrics_seed: int = 0
    val_metrics_lpips_net: str = "alex"
    val_metrics_lpips_batch_size: int = 4
    val_metrics_save_videos: bool = True
    val_metrics_fps: float = 5.0
    val_metrics_rollout_mode: str = "fixed_chunks"
    val_metrics_closed_loop_chunks: int = 1
    wandb_project: str = "coachworld-training"
    wandb_run_name: str = ""
    require_val_split: bool = False  # if False, missing val split disables validation/diagnostics
    text_encoder_device: str = "auto"  # auto | cpu | cuda | cuda:<idx>; auto=cpu on 1 GPU, local cuda under torchrun
    text_conditioning_enabled: bool = True
    text_embedding_cache_limit: int = 32  # each WAN T5 embedding is about 4 MiB on GPU

    # Dataset
    dataset_format: str = "video_latent"
    dataset_roots: list = field(default_factory=list)
    meta_info_roots: list = field(default_factory=list)
    dataset_probs: list = field(default_factory=lambda: [1.0])
    val_dataset_probs: list = field(default_factory=list)
    mixture_sampler_seed: int = 0
    # Zero uses the sum of all natural root-local windows, rounded to a global batch.
    mixture_samples_per_epoch: int = 0
    video_latent_video_keys: list = field(default_factory=list)
    video_latent_condition_view: str = "arm_slot_eef_pose"
    video_latent_sample_stride: int = 8
    video_latent_camera_conditioning: bool = False
    video_latent_camera_sidecar_roots: list = field(default_factory=list)
    video_latent_camera_intrinsics_fallback: str = "identity"  # identity | error
    video_latent_camera_extrinsics_convention: str = "world_from_camera"  # world_from_camera | camera_from_world
    # Formal training roots can opt into a fail-closed production contract.
    # This prevents an old latent root without canonical state or embedded K/T
    # from silently entering a camera-aware run.
    video_latent_require_production_contract: bool = False
    video_latent_required_target_fps: float = 0.0
    video_latent_required_target_image_hw: list = field(default_factory=list)
    video_latent_eef_projection_roots: list = field(default_factory=list)
    video_latent_eef_projection_mode: str = "sidecar"  # sidecar | online
    video_latent_eef_projection_load_heatmap: bool = False
    video_latent_eef_projection_heatmap_size: str = ""  # empty keeps sidecar size; e.g. "24x40"
    rgb_skip: int = 2  # 15Hz → 7.5Hz; WAN VAE 4x temporal compression handles the rest
    down_sample: int = 2  # state-to-video frame ratio (matches rgb_skip)
    annotation_name: str = "annotation"
    allow_empty_text: bool = False  # keep action-only trajectories with no language label
    num_cameras: int = 3
    latent_height_per_view: int = 12
    latent_width: int = 20
    val_video_num: int = 10  # number of videos to generate during validation
    causal_proprio_loss_weight: float = 0.0
    causal_proprio_mask_weight: float = 0.05
    causal_proprio_future_only: bool = True


@dataclass
class DataConfig:
    format: str = "lerobot"  # "lerobot" | "raw"
    dataset_path: str = ""
    action_norm_method: str = "quantiles"  # "quantiles" | "minmax" | "none"
    camera_keys: list = field(
        default_factory=lambda: ["observation.images.cam_high"]
    )
    num_workers: int = 4
    batch_size: int = 32


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class CoachWorldConfig:
    """Root configuration aggregating all modules."""

    experiment_name: str = "coachworld"
    output_dir: str = "experiments/"
    seed: int = 42
    device: str = "cuda:0"

    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    coaching: CoachingConfig = field(default_factory=CoachingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wm_training: WMTrainingConfig = field(default_factory=WMTrainingConfig)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[list[str]] = None,
) -> CoachWorldConfig:
    """Load config from YAML, merge CLI overrides, return typed dataclass.

    Usage:
        cfg = load_config("configs/training/coachworld.yaml")
        cfg = load_config(
            "configs/training/coachworld.yaml",
            ["world_model.action_dim=14"],
        )
    """
    schema = OmegaConf.structured(CoachWorldConfig)

    if yaml_path is not None:
        file_cfg = OmegaConf.load(yaml_path)
        cfg = OmegaConf.merge(schema, file_cfg)
    else:
        cfg = schema

    if cli_overrides:
        cli_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(
        cfg, structured_config_mode=SCMode.INSTANTIATE, resolve=True
    )


def load_config_from_argv(yaml_path: str) -> CoachWorldConfig:
    """Convenience: load YAML + treat sys.argv[1:] as dotlist overrides."""
    overrides = [a for a in sys.argv[1:] if "=" in a]
    return load_config(yaml_path, overrides)
