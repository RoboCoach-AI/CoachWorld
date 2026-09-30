"""Action/history causal-contract helpers for CoachWorld simulators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def validate_action_timing_contract(action_schema: str, action_rate: str) -> None:
    """Reject unsupported action-schema / timestep-rate combinations."""
    schema = str(action_schema)
    rate = str(action_rate)
    if schema == "arm_slot" and rate != "raw":
        raise ValueError(
            "arm_slot action_schema requires raw action_condition_timestep_rate. "
            f"Got action_condition_timestep_rate={rate!r}. Arm-slot conditions are "
            "video-rate tensors shaped (B,T,slots,features+mask); latent-rate "
            "padding/truncation is only defined for the old fixed-vector action path."
        )


@dataclass(frozen=True)
class FutureOnlyActionMask:
    """Mask metadata shared by K/V action tokens and dense control chunks."""

    kv_zero_prefix_actions: int
    dense_zero_prefix_chunks: int
    dense_zero_first_chunk: bool


def latent_chunk_frame_ranges(
    num_latent_frames: int,
    vae_temporal_stride: int = 4,
    start_video_frame: int = 0,
) -> list[tuple[int, int]]:
    """Return inclusive video-frame spans represented by WAN causal latents.

    WAN's causal VAE treats the first latent specially: latent 0 is anchored by
    frame 0, while later latents summarize the frames since the previous latent
    boundary. With stride 4 this is:

      latent 0 -> [0, 0]
      latent 1 -> [1, 4]
      latent 2 -> [5, 8]

    The boundary state for latent i is therefore frame ``i * stride``.
    """
    stride = int(vae_temporal_stride)
    start = int(start_video_frame)
    ranges: list[tuple[int, int]] = []
    for i in range(int(num_latent_frames)):
        if i == 0:
            ranges.append((start, start))
        else:
            ranges.append((start + (i - 1) * stride + 1, start + i * stride))
    return ranges


def latent_boundary_frame_indices(
    num_latent_frames: int,
    vae_temporal_stride: int = 4,
    start_video_frame: int = 0,
) -> list[int]:
    """Return the video-frame boundary state index for each latent token."""
    stride = int(vae_temporal_stride)
    start = int(start_video_frame)
    return [start + i * stride for i in range(int(num_latent_frames))]


def num_wan_latent_frames(
    num_video_frames: int,
    vae_temporal_stride: int = 4,
) -> int:
    """Return WAN causal VAE latent-frame count for a raw video length.

    WAN keeps frame 0 as the first latent, then emits one latent per stride
    frames. For example, 17 video frames map to 5 latent frames with stride 4:
    boundaries 0, 4, 8, 12, 16.
    """
    n = max(1, int(num_video_frames))
    stride = int(vae_temporal_stride)
    return 1 + (n - 1) // stride


def num_video_frames_for_latent_window(
    num_latent_frames: int,
    vae_temporal_stride: int = 4,
) -> int:
    """Return raw video/action frames covered by a WAN latent window."""
    n = max(1, int(num_latent_frames))
    stride = int(vae_temporal_stride)
    return 1 + (n - 1) * stride


def full_window_enabled(config) -> bool:
    """Return whether the model consumes the complete history/future window."""
    return bool(getattr(config, "full_window", False))


def sparse_history_indices(
    current_latent: int,
    history_frames: int,
    *,
    dilation: int,
    collapse_to_current: bool = False,
) -> list[int]:
    """Return a dilated sparse history ending at the current latent.

    ``dilation`` is measured in temporal-latent units. The latest observed
    latent is always retained. At episode boundaries negative indices are
    clamped to latent 0, matching rollout-buffer initialization.
    """
    current = max(0, int(current_latent))
    history = max(1, int(history_frames))
    step = int(dilation)
    if step <= 0:
        raise ValueError(f"sparse-history dilation must be positive, got {step}")
    if bool(collapse_to_current):
        return [current] * history
    return [
        max(0, current - (history - 1 - position) * step)
        for position in range(history)
    ]


def latent_window_indices_for_first_future_frame(
    first_future_frame: int,
    video_length: int,
    history_frames: int,
    future_frames: int,
    vae_temporal_stride: int = 4,
    history_selector: str = "recent",
    history_offsets: list[int] | tuple[int, ...] | None = None,
    history_dilation: int = 1,
    history_collapse: bool = False,
    allow_future_padding: bool = False,
) -> list[int]:
    """Return WAN latent ids for a history+future window.

    ``first_future_frame`` is on the extracted video/action timeline
    (post-rgb_skip). Away from episode boundaries, the latent at offset
    ``history_frames`` is exactly the first future latent. Near boundaries the
    window is clamped to valid latent ids; callers should use the returned ids,
    not recompute a separate rollout start.
    """
    selector = str(history_selector)
    if selector not in {
        "recent",
        "first_recent",
        "first_offset_recent",
        "sparse",
    }:
        raise ValueError(
            f"unsupported history_selector={history_selector!r}; "
            "expected 'recent', 'first_recent', 'first_offset_recent', or "
            "'sparse'"
        )

    history = max(1, int(history_frames))
    future = max(1, int(future_frames))
    total = history + future
    max_latent_t = num_wan_latent_frames(video_length, vae_temporal_stride)
    latent_now = int(first_future_frame) // int(vae_temporal_stride)
    latent_now = min(max(0, latent_now), max_latent_t - 1)

    if selector == "sparse":
        if not allow_future_padding:
            max_future_start = max(0, max_latent_t - future)
            latent_now = min(latent_now, max_future_start)
        current_latent = max(0, latent_now - 1)
        history_ids = sparse_history_indices(
            current_latent,
            history,
            dilation=int(history_dilation),
            collapse_to_current=bool(history_collapse),
        )
        future_ids = list(range(latent_now, min(max_latent_t, latent_now + future)))
        while len(future_ids) < future:
            future_ids.append(future_ids[-1] if future_ids else max_latent_t - 1)
        return history_ids + future_ids

    if selector == "first_recent":
        if not allow_future_padding:
            max_future_start = max(0, max_latent_t - future)
            latent_now = min(latent_now, max_future_start)
        recent_count = max(0, history - 1)
        recent_start = max(0, latent_now - recent_count)
        recent_ids = list(range(recent_start, latent_now))
        while len(recent_ids) < recent_count:
            recent_ids.insert(0, 0)
        history_ids = [0] + recent_ids[-recent_count:] if recent_count else [0]
        future_ids = list(range(latent_now, min(max_latent_t, latent_now + future)))
        while len(future_ids) < future:
            future_ids.append(future_ids[-1] if future_ids else max_latent_t - 1)
        return history_ids + future_ids

    if selector == "first_offset_recent":
        offsets = [int(x) for x in (history_offsets or [])]
        expected = history - 1
        if len(offsets) != expected:
            raise ValueError(
                "history_selector='first_offset_recent' requires exactly "
                f"history_length-1 offsets; got {len(offsets)} offsets for "
                f"history_length={history}: {offsets}"
            )
        if any(x <= 0 for x in offsets):
            raise ValueError(f"history_offsets must be positive latent offsets, got {offsets}")
        if not allow_future_padding:
            max_future_start = max(0, max_latent_t - future)
            latent_now = min(latent_now, max_future_start)
        history_ids = [0] + [max(0, latent_now - offset) for offset in offsets]
        future_ids = list(range(latent_now, min(max_latent_t, latent_now + future)))
        while len(future_ids) < future:
            future_ids.append(future_ids[-1] if future_ids else max_latent_t - 1)
        return history_ids + future_ids

    start = latent_now - history
    end = start + total
    if start < 0:
        start = 0
        end = min(total, max_latent_t)
    if end > max_latent_t:
        end = max_latent_t
        start = max(0, end - total)

    indices = list(range(start, end))
    while len(indices) < total:
        indices.append(indices[-1] if indices else 0)
    return indices


def action_start_frame_for_first_future_latent(
    first_future_latent: int,
    history_frames: int,
    vae_temporal_stride: int = 4,
) -> int:
    """Return the virtual raw-action start frame for a model window.

    This intentionally does not depend on the first history latent id. In
    first+recent windows the visual history starts with episode latent 0, but
    the action tensor still uses the contiguous causal timeline ending at the
    first future latent, with all history-aligned action chunks removed or
    zeroed by ``future_only_action_conditioning``.
    """
    first_future = int(first_future_latent)
    history = max(1, int(history_frames))
    stride = int(vae_temporal_stride)
    if first_future <= 0:
        # Wan latent 0 is a causal anchor for RGB frame 0 repeated over the
        # first action group.  Use a virtual negative start so the future-only
        # action mask lands on frame 0 after left padding.
        return -future_action_start_index(
            history,
            action_len=10**9,
            action_rate="raw",
            vae_temporal_stride=stride,
        )
    start = (first_future - history) * stride
    return start


def future_action_start_index(
    history_frames: int,
    action_len: int,
    action_rate: str = "raw",
    vae_temporal_stride: int = 4,
) -> int:
    """Return the first future action index under WAN causal timing.

    For latent-rate actions, history consumes one action token per history
    latent. For raw/video-rate actions, history latent 0 covers frame 0 and
    history latent i>0 covers ``[(i-1)*stride+1, i*stride]``. Therefore the
    first future raw action after H history latents starts at
    ``(H-1)*stride + 1``.
    """
    history = max(0, int(history_frames))
    length = max(0, int(action_len))
    if str(action_rate) == "latent":
        return min(history, length)
    if history <= 0:
        return 0
    start = (history - 1) * int(vae_temporal_stride) + 1
    return min(start, length)


def compute_future_only_action_mask(
    history_frames: int,
    action_length: int,
    action_rate: str,
    vae_temporal_stride: int = 4,
) -> FutureOnlyActionMask:
    """Compute the shared future-only mask for action conditioning.

    History latent frames are state anchors. Action conditioning aligned to those
    frames must be removed from every action path. For latent-rate actions this
    drops one K/V action token per history latent. For raw-rate actions this
    drops the raw video-frame actions covered by the history latent chunks.
    Dense FiLM/AdaLN are latent-chunk aligned, so they always zero the same
    number of leading latent chunks as the history length.
    """
    history = max(0, int(history_frames))
    length = max(0, int(action_length))
    if str(action_rate) == "latent":
        kv_zero = min(history, length)
    else:
        kv_zero = future_action_start_index(
            history_frames=history,
            action_len=length,
            action_rate="raw",
            vae_temporal_stride=vae_temporal_stride,
        )
    dense_zero = history if length > 0 else 0
    return FutureOnlyActionMask(
        kv_zero_prefix_actions=kv_zero,
        dense_zero_prefix_chunks=dense_zero,
        dense_zero_first_chunk=dense_zero > 0,
    )


def apply_kv_future_only_action_mask(
    action_seq: torch.Tensor | None,
    history_frames: int,
    action_rate: str,
    vae_temporal_stride: int = 4,
) -> tuple[torch.Tensor | None, FutureOnlyActionMask]:
    """Drop history-aligned K/V action tokens and return dense mask metadata.

    Dropping is stricter than zeroing. The action K/V projections have biases,
    so a zero token can still produce a non-zero constant K/V contribution.
    Removing the prefix tokens makes the sparse action path exactly future-only.
    """
    if action_seq is None:
        return None, FutureOnlyActionMask(0, 0, False)
    mask = compute_future_only_action_mask(
        history_frames=history_frames,
        action_length=action_seq.shape[1],
        action_rate=action_rate,
        vae_temporal_stride=vae_temporal_stride,
    )
    if mask.kv_zero_prefix_actions <= 0:
        return action_seq, mask
    if mask.kv_zero_prefix_actions >= action_seq.shape[1]:
        return None, mask
    return action_seq[:, mask.kv_zero_prefix_actions :, ...].contiguous(), mask


def zero_prefix_latent_chunks(
    tensor: torch.Tensor | None,
    n_chunks: int,
) -> torch.Tensor | None:
    """Zero leading latent-frame chunks for dense/AdaLN action tensors."""
    if tensor is None:
        return None
    n_zero = min(max(0, int(n_chunks)), tensor.shape[1])
    if n_zero <= 0:
        return tensor
    out = tensor.clone()
    out[:, :n_zero] = 0
    return out
