"""Deterministic scheduling helpers for generated-history training."""

from __future__ import annotations

from collections.abc import Sequence


def _activation_hash(global_step: int, seed: int) -> int:
    value = (
        int(global_step) + 0x9E3779B9 * (int(seed) + 1)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _horizon_hash(global_step: int, seed: int) -> int:
    value = (
        int(global_step)
        + 0x85EBCA6B * (int(seed) + 1)
        + 0xC2B2AE35
    ) & 0xFFFFFFFF
    return value ^ (value >> 16)


def generated_history_step_is_active(
    *,
    global_step: int,
    enabled: bool,
    start_step: int,
    probability: float,
    seed: int,
) -> bool:
    """Return the same generated-history decision on every distributed rank."""
    if not enabled or int(global_step) < int(start_step):
        return False
    value = float(probability)
    if value <= 0.0:
        return False
    if value >= 1.0:
        return True
    return _activation_hash(global_step, seed) < int(value * (2**32))


def select_generated_history_unroll_chunks(
    *,
    global_step: int,
    configured_horizons: Sequence[int],
    available_chunks: int,
    seed: int,
) -> int:
    """Select a deterministic configured horizon supported by the batch."""
    available = int(available_chunks)
    if available <= 0:
        raise ValueError(f"generated-history batch has available_chunks={available}")

    schedule = sorted({int(value) for value in configured_horizons})
    if not schedule or any(value <= 0 for value in schedule):
        raise ValueError(
            "generated-history horizons must contain positive integers, "
            f"got {schedule}"
        )
    usable = [value for value in schedule if value <= available]
    if not usable:
        return available

    desired = schedule[_horizon_hash(global_step, seed) % len(schedule)]
    supported = [candidate for candidate in usable if candidate <= desired]
    return max(supported) if supported else min(usable)
