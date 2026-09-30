"""Configuration contract for periodic closed-loop training validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClosedLoopValidationProtocol:
    mode: str
    fixed_chunks: int | None

    @property
    def full_episode(self) -> bool:
        return self.mode == "full_episode"


def resolve_closed_loop_validation_protocol(
    *,
    mode: str,
    fixed_chunks: int,
) -> ClosedLoopValidationProtocol:
    """Validate the explicit full-episode or fixed-chunk rollout contract."""
    normalized = str(mode).strip().lower()
    if normalized not in {"full_episode", "fixed_chunks"}:
        raise ValueError(
            "val_metrics_rollout_mode must be 'full_episode' or 'fixed_chunks', "
            f"got {mode!r}"
        )
    chunks = int(fixed_chunks)
    if chunks <= 0:
        raise ValueError(
            "val_metrics_closed_loop_chunks must be positive; use "
            "val_metrics_rollout_mode='full_episode' for episode-end rollout"
        )
    return ClosedLoopValidationProtocol(
        mode=normalized,
        fixed_chunks=None if normalized == "full_episode" else chunks,
    )
