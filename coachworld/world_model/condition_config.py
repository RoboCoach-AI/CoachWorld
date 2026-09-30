"""Shared architecture-condition configuration contract.

Training configuration mirrors architecture fields under both
``world_model`` and ``wm_training``. This module checks that they agree.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CONDITION_CONFIG_FIELDS = (
    "condition_mask_enabled",
    "gt_condition_replacement",
    "future_only_action_conditioning",
    "text_conditioning_enabled",
    "action_schema",
    "action_dim",
    "max_arm_slots",
    "action_max_time_steps",
    "action_kv_enabled",
    "action_v_init_scale",
    "dense_action_film_enabled",
    "action_adaln_modulation_enabled",
    "action_adaln_scale",
    "action_dense_rank",
    "action_dense_relative_chunks",
    "action_control_scale",
    "action_dense_layernorm_enabled",
    "eef_projection_kv_enabled",
    "eef_spatial_conditioning_enabled",
    "eef_spatial_sigma_px",
    "eef_spatial_depth_scale",
    "action_num_domains",
    "action_domain_prompt_tokens",
    "domain_aware_action_projection_enabled",
    "domain_aware_group_projection_enabled",
    "full_window",
    "action_condition_timestep_rate",
    "history_selector",
    "history_offsets",
    "history_dilations",
    "history_collapse_prob",
    "history_eval_dilation",
    "history_rollout_dilation",
    "multi_view_position_mode",
    "camera_id_embedding_enabled",
    "prope_enabled",
    "prope_zero_init",
    "prope_spatial_stride",
    "num_cameras",
    "latent_height_per_view",
    "latent_width",
)


def condition_config_mismatches(
    model_config: Any,
    training_config: Any,
    *,
    fields: Iterable[str] = CONDITION_CONFIG_FIELDS,
) -> list[str]:
    """Return mirrored architecture fields whose values disagree."""

    mismatches: list[str] = []
    for field in fields:
        if not hasattr(model_config, field) or not hasattr(training_config, field):
            continue
        model_value = getattr(model_config, field)
        training_value = getattr(training_config, field)
        if model_value != training_value:
            mismatches.append(
                f"{field}: world_model={model_value!r}, "
                f"wm_training={training_value!r}"
            )
    return mismatches


def validate_condition_config_match(model_config: Any, training_config: Any) -> None:
    """Reject split-brain architecture settings before model construction."""

    mismatches = condition_config_mismatches(model_config, training_config)
    if mismatches:
        raise ValueError(
            "world_model and wm_training condition config mismatch. "
            "Set both namespaces explicitly for backward-compatible configs:\n  - "
            + "\n  - ".join(mismatches)
        )
