"""Action normalization utilities.

Supports quantile normalization (q01/q99) as used by lingbot-va and EvoW.
This is the standard approach for robot action spaces where the range
varies per dimension and outliers exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ActionNormalizer:
    """Normalize / denormalize actions using quantile statistics.

    Usage:
        normalizer = ActionNormalizer.from_json("stats.json")
        action_norm = normalizer.normalize(raw_action)   # -> [-1, 1]
        action_raw  = normalizer.denormalize(action_norm) # -> original scale
    """

    def __init__(self, q01: np.ndarray, q99: np.ndarray):
        """
        Args:
            q01: 1st percentile per action dim, shape (action_dim,).
            q99: 99th percentile per action dim, shape (action_dim,).
        """
        self.q01 = np.asarray(q01, dtype=np.float64)
        self.q99 = np.asarray(q99, dtype=np.float64)
        if self.q01.shape != self.q99.shape:
            raise ValueError(f"q01/q99 shape mismatch: {self.q01.shape} != {self.q99.shape}")
        self.constant_mask = (self.q99 - self.q01) < 1e-8
        self.range = np.where(self.constant_mask, 1.0, self.q99 - self.q01)

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1]."""
        out = 2.0 * (action - self.q01) / self.range - 1.0
        if np.any(self.constant_mask):
            out[..., self.constant_mask] = 0.0
        return out

    def denormalize(self, action_norm: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to original scale."""
        out = (action_norm + 1.0) / 2.0 * self.range + self.q01
        if np.any(self.constant_mask):
            out[..., self.constant_mask] = self.q01[self.constant_mask]
        return out

    @classmethod
    def from_json(cls, path: str | Path) -> ActionNormalizer:
        """Load from a JSON file with 'q01' and 'q99' keys."""
        data = json.loads(Path(path).read_text())
        return cls(q01=data["q01"], q99=data["q99"])

    def to_json(self, path: str | Path) -> None:
        """Save normalization stats to JSON."""
        Path(path).write_text(json.dumps({
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
        }, indent=2))
