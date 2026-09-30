#!/usr/bin/env python3
"""Regression tests for mirrored world-model condition configuration."""

from __future__ import annotations

import unittest

from coachworld.config import WorldModelConfig, WMTrainingConfig, load_config
from coachworld.world_model.condition_config import (
    CONDITION_CONFIG_FIELDS,
    condition_config_mismatches,
    validate_condition_config_match,
)
from coachworld.world_model.inference_utils import (
    validate_condition_config_match as validate_inference_config,
)
from coachworld.world_model.trainer import WMTrainer


class ConditionConfigTest(unittest.TestCase):
    def test_training_and_inference_share_the_same_field_contract(self) -> None:
        self.assertIs(WMTrainer.CONDITION_CONFIG_FIELDS, CONDITION_CONFIG_FIELDS)
        self.assertIn("eef_projection_kv_enabled", CONDITION_CONFIG_FIELDS)
        self.assertIn("eef_spatial_conditioning_enabled", CONDITION_CONFIG_FIELDS)
        self.assertIn("text_conditioning_enabled", CONDITION_CONFIG_FIELDS)

    def test_eef_architecture_mismatch_fails_closed(self) -> None:
        model = WorldModelConfig(eef_projection_kv_enabled=True)
        training = WMTrainingConfig(eef_projection_kv_enabled=False)
        mismatches = condition_config_mismatches(model, training)
        self.assertTrue(
            any(row.startswith("eef_projection_kv_enabled:") for row in mismatches)
        )
        with self.assertRaisesRegex(ValueError, "eef_projection_kv_enabled"):
            validate_condition_config_match(model, training)

    def test_example_config_remains_loadable(self) -> None:
        cfg = load_config("configs/training/coachworld.yaml")
        validate_condition_config_match(cfg.world_model, cfg.wm_training)
        validate_inference_config(cfg)

    def test_legacy_defaults_remain_loadable(self) -> None:
        model = WorldModelConfig()
        training = WMTrainingConfig()
        self.assertEqual(condition_config_mismatches(model, training), [])


if __name__ == "__main__":
    unittest.main()
