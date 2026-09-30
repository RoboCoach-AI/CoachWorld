"""Embodiment-neutral robot action bundle.

The bundle keeps arm structure explicit before flattening to a model-specific
action vector. FR3 single-arm commands and AgiBot dual-arm commands can both be
represented as `(T, arms, 7)` with an arm mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


DEFAULT_ARM_ORDER = ("left", "right")
ARM_COMMAND_DIM = 7


@dataclass
class ActionBundle:
    """Structured action commands for one or more robot arms."""

    command: np.ndarray
    arm_order: tuple[str, ...] = DEFAULT_ARM_ORDER
    arm_mask: np.ndarray | None = None
    embodiment: str = "unknown"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        command = np.asarray(self.command, dtype=np.float32)
        if command.ndim != 3:
            raise ValueError(f"command must have shape (T, arms, dim), got {command.shape}")
        if command.shape[-1] != ARM_COMMAND_DIM:
            raise ValueError(
                f"per-arm command dim must be {ARM_COMMAND_DIM}, got {command.shape[-1]}"
            )
        if command.shape[1] != len(self.arm_order):
            raise ValueError(
                f"command arm count={command.shape[1]} does not match arm_order={self.arm_order}"
            )
        self.command = command
        if self.arm_mask is None:
            self.arm_mask = np.ones(len(self.arm_order), dtype=bool)
        else:
            mask = np.asarray(self.arm_mask, dtype=bool).reshape(-1)
            if mask.shape[0] != len(self.arm_order):
                raise ValueError(
                    f"arm_mask length={mask.shape[0]} does not match arm_order={self.arm_order}"
                )
            self.arm_mask = mask

    @classmethod
    def from_single_arm(
        cls,
        action: np.ndarray,
        *,
        arm: str = "right",
        embodiment: str = "fr3",
        arm_order: tuple[str, ...] = DEFAULT_ARM_ORDER,
    ) -> "ActionBundle":
        """Create a bundle from a single-arm `(T, 7)` command sequence."""
        arr = _as_action_2d(action)
        if arm not in arm_order:
            raise ValueError(f"arm={arm!r} not in arm_order={arm_order}")
        command = np.zeros((arr.shape[0], len(arm_order), ARM_COMMAND_DIM), dtype=np.float32)
        arm_idx = arm_order.index(arm)
        command[:, arm_idx, :] = arr
        mask = np.zeros(len(arm_order), dtype=bool)
        mask[arm_idx] = True
        return cls(command, arm_order=arm_order, arm_mask=mask, embodiment=embodiment)

    @classmethod
    def from_dual_arm(
        cls,
        left: np.ndarray,
        right: np.ndarray,
        *,
        embodiment: str = "agibot",
        arm_order: tuple[str, ...] = DEFAULT_ARM_ORDER,
    ) -> "ActionBundle":
        """Create a bundle from left/right `(T, 7)` command sequences."""
        left_arr = _as_action_2d(left)
        right_arr = _as_action_2d(right)
        if left_arr.shape[0] != right_arr.shape[0]:
            raise ValueError(
                f"left/right lengths differ: {left_arr.shape[0]} vs {right_arr.shape[0]}"
            )
        parts = {"left": left_arr, "right": right_arr}
        command = np.stack([parts[arm] for arm in arm_order], axis=1).astype(np.float32)
        return cls(
            command,
            arm_order=arm_order,
            arm_mask=np.ones(len(arm_order), dtype=bool),
            embodiment=embodiment,
        )

    @classmethod
    def from_flat(
        cls,
        action: np.ndarray,
        *,
        arm_order: tuple[str, ...],
        active_arms: Iterable[str] | None = None,
        embodiment: str = "unknown",
    ) -> "ActionBundle":
        """Create a bundle from a concatenated `(T, arms*7)` action array."""
        arr = _as_2d(action)
        expected_dim = len(arm_order) * ARM_COMMAND_DIM
        if arr.shape[-1] != expected_dim:
            raise ValueError(f"flat action dim={arr.shape[-1]} does not match {expected_dim}")
        command = arr.reshape(arr.shape[0], len(arm_order), ARM_COMMAND_DIM)
        if active_arms is None:
            mask = np.ones(len(arm_order), dtype=bool)
        else:
            active = set(active_arms)
            mask = np.asarray([arm in active for arm in arm_order], dtype=bool)
        return cls(command, arm_order=arm_order, arm_mask=mask, embodiment=embodiment)

    def to_flat(
        self,
        *,
        arm_order: tuple[str, ...] | None = None,
        active_only: bool = False,
    ) -> np.ndarray:
        """Flatten to `(T, arms*7)` in a requested arm order."""
        order = arm_order or self.arm_order
        indices = [self.arm_order.index(arm) for arm in order]
        command = self.command[:, indices, :].copy()
        mask = self.arm_mask[indices]
        if active_only:
            command = command[:, mask, :]
        else:
            command[:, ~mask, :] = 0.0
        return command.reshape(command.shape[0], -1).astype(np.float32)

    def active_arms(self) -> tuple[str, ...]:
        return tuple(arm for arm, keep in zip(self.arm_order, self.arm_mask) if bool(keep))

    @property
    def horizon(self) -> int:
        return int(self.command.shape[0])


def _as_action_2d(action: np.ndarray) -> np.ndarray:
    arr = _as_2d(action)
    if arr.shape[-1] != ARM_COMMAND_DIM:
        raise ValueError(f"expected action shape (T, {ARM_COMMAND_DIM}), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _as_2d(action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D action array, got {arr.shape}")
    return arr.astype(np.float32, copy=False)
