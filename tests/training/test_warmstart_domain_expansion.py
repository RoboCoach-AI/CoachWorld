#!/usr/bin/env python3
"""Unit checks for model-only warm-start domain expansion."""

from __future__ import annotations

import unittest

import torch

from coachworld.world_model.trainer import WMTrainer


class WarmstartDomainExpansionTest(unittest.TestCase):
    def test_preserves_existing_domain_rows(self) -> None:
        checkpoint = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        current = torch.full((8, 3, 2), -7.0)
        expanded = WMTrainer._expand_warmstart_tensor(
            "action_encoder.slot_projection.delta_weight",
            checkpoint,
            current,
        )
        self.assertIsNotNone(expanded)
        torch.testing.assert_close(expanded[:4], checkpoint)
        torch.testing.assert_close(expanded[4:], current[4:])

    def test_rejects_non_domain_shape_change(self) -> None:
        expanded = WMTrainer._expand_warmstart_tensor(
            "action_encoder.slot_net.1.weight",
            torch.zeros(4, 3),
            torch.zeros(8, 3),
        )
        self.assertIsNone(expanded)

    def test_patch_embedding_preserves_old_channels_and_zero_suffix(self) -> None:
        checkpoint = torch.arange(2 * 3 * 1 * 2 * 2, dtype=torch.float32).reshape(
            2, 3, 1, 2, 2
        )
        current = torch.zeros(2, 5, 1, 2, 2)
        expanded = WMTrainer._expand_warmstart_tensor(
            "patch_embedding.weight",
            checkpoint,
            current,
        )
        self.assertIsNotNone(expanded)
        torch.testing.assert_close(expanded[:, :3], checkpoint)
        torch.testing.assert_close(expanded[:, 3:], torch.zeros_like(expanded[:, 3:]))


if __name__ == "__main__":
    unittest.main()
