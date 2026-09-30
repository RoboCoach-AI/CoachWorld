"""Action/state representations shared by training and evaluation."""

from __future__ import annotations

import os

import numpy as np


OBSERVATION_STATE = "observation_state"
COMMANDED_ACTION = "commanded_action"
COMMANDED_ORIGINAL_ACTION = "commanded_original_action"
COMMANDED_JOINT_POSITION_GRIPPER = "commanded_joint_position_gripper"
COMMANDED_CARTESIAN_VELOCITY_GRIPPER = "commanded_cartesian_velocity_gripper"
COMMANDED_JOINT_VELOCITY_GRIPPER = "commanded_joint_velocity_gripper"
COMMANDED_FULL_ACTION = "commanded_full_action"
ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER = "abs_state_plus_cartesian_velocity_gripper"
ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER = "abs_state_quat_cartesian_velocity_gripper"
CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER = "causal_state_plus_cartesian_velocity_gripper"
CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER = "causal_state_quat_cartesian_velocity_gripper"
ABS_PLUS_DELTA_PLUS_GRIPPER = "abs_plus_delta_plus_gripper"
ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER = "abs_plus_chunk_delta_plus_gripper"
WAN_LATENT_TEMPORAL_STRIDE = 4

ACTION_SOURCE_DIMS = {
    OBSERVATION_STATE: 7,
    COMMANDED_ACTION: 7,
    COMMANDED_ORIGINAL_ACTION: 7,
    COMMANDED_JOINT_POSITION_GRIPPER: 8,
    COMMANDED_CARTESIAN_VELOCITY_GRIPPER: 7,
    COMMANDED_JOINT_VELOCITY_GRIPPER: 8,
    COMMANDED_FULL_ACTION: 35,
    ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER: 14,
    ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER: 15,
    CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER: 14,
    CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER: 15,
    ABS_PLUS_DELTA_PLUS_GRIPPER: 14,
    ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER: 14,
}

STATE_PREFIX_DIMS = {
    ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER: 7,
    ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER: 8,
    CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER: 7,
    CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER: 8,
}

CAUSAL_STATE_PREFIX_DIMS = {
    CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER: STATE_PREFIX_DIMS[
        CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER
    ],
    CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER: STATE_PREFIX_DIMS[
        CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER
    ],
}

FUTURE_STATE_LEAKING_ACTION_SOURCES = {
    OBSERVATION_STATE,
    ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
    ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
    ABS_PLUS_DELTA_PLUS_GRIPPER,
    ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER,
}

STATE_COMMAND_ACTION_SOURCES = {
    ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
    ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
    CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
    CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
}

COMMANDED_ACTION_SOURCES = {
    COMMANDED_ACTION,
    COMMANDED_ORIGINAL_ACTION,
    COMMANDED_JOINT_POSITION_GRIPPER,
    COMMANDED_CARTESIAN_VELOCITY_GRIPPER,
    COMMANDED_JOINT_VELOCITY_GRIPPER,
    COMMANDED_FULL_ACTION,
}


def expected_action_dim(source: str) -> int | None:
    """Return the expected action dimension for fixed-layout action sources."""
    return ACTION_SOURCE_DIMS.get(source)


def action_source_contains_future_state(source: str) -> bool:
    """Return True for sources that expose future observed state as condition."""
    return source in FUTURE_STATE_LEAKING_ACTION_SOURCES


def causal_state_prefix_dim(source: str) -> int:
    """Return held-state prefix width for window-causal state+command sources."""
    return CAUSAL_STATE_PREFIX_DIMS.get(source, 0)


def action_source_state_prefix_dim(source: str) -> int:
    """Return state/proprio prefix width for structured state+command sources."""
    return STATE_PREFIX_DIMS.get(source, 0)


def action_source_command_slice(source: str) -> tuple[int, int] | None:
    """Return ``[start, end)`` command/control dims for known action layouts."""
    dim = expected_action_dim(source)
    if dim is None:
        return None
    if source in COMMANDED_ACTION_SOURCES:
        return (0, dim)
    if source in STATE_COMMAND_ACTION_SOURCES:
        start = action_source_state_prefix_dim(source)
        return (start, dim)
    return None


def action_source_layout(source: str) -> dict[str, object]:
    """Return compact layout metadata for logging, A/B, ranking, and routing."""
    dim = expected_action_dim(source)
    state_prefix_dim = action_source_state_prefix_dim(source)
    command_slice = action_source_command_slice(source)
    return {
        "source": source,
        "dim": dim,
        "state_slice": (0, state_prefix_dim) if state_prefix_dim > 0 else None,
        "command_slice": command_slice,
        "contains_future_state": action_source_contains_future_state(source),
        "is_causal_condition": action_source_is_causal_condition(source),
    }


def action_source_is_causal_condition(source: str) -> bool:
    """Return True when the source can be used as future condition by default."""
    return not action_source_contains_future_state(source)


def validate_action_source_for_condition(source: str) -> None:
    """Reject action layouts that leak future observed state into WM condition."""
    if not action_source_contains_future_state(source):
        return
    if os.environ.get("COACHWORLD_ALLOW_FUTURE_STATE_ACTION_SOURCE") == "1":
        return
    raise RuntimeError(
        f"Refusing action_source={source!r} for WM conditioning: it exposes "
        "future observed state/proprio in prediction windows. Use a causal_state_* "
        "or commanded_* source, or set COACHWORLD_ALLOW_FUTURE_STATE_ACTION_SOURCE=1 "
        "only for a clearly named historical ablation."
    )


def causalize_action_window(
    action: np.ndarray,
    source: str,
    *,
    history_frames: int,
    vae_temporal_stride: int = WAN_LATENT_TEMPORAL_STRIDE,
) -> np.ndarray:
    """Hold proprio/state prefixes at the last observed history boundary.

    The raw 14D/15D builders intentionally still produce per-frame observed
    state so that stats and inspection tools share one layout. Causality is a
    window property: after the dataset crops a history+future window, future
    condition tokens may only see the final observed history state plus the
    commanded action stream.
    """
    prefix_dim = causal_state_prefix_dim(source)
    if prefix_dim <= 0:
        return action

    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"action window must be 2D, got shape={arr.shape}")
    if arr.shape[-1] < prefix_dim:
        raise ValueError(
            f"action dim={arr.shape[-1]} is smaller than causal prefix dim={prefix_dim}"
        )
    if len(arr) == 0:
        return arr

    history = max(1, int(history_frames))
    stride = max(1, int(vae_temporal_stride))
    anchor_index = min((history - 1) * stride, len(arr) - 1)
    causal = arr.copy()
    causal[:, :prefix_dim] = arr[anchor_index : anchor_index + 1, :prefix_dim]
    return causal.astype(np.float32, copy=False)


def action_window_from_array(
    action: np.ndarray,
    source: str,
    *,
    state_start: int,
    state_end: int,
    history_frames: int,
    vae_temporal_stride: int = WAN_LATENT_TEMPORAL_STRIDE,
    pad_to_length: bool = False,
) -> np.ndarray:
    """Clip an action array and apply the source-specific causal window rule."""
    validate_action_source_for_condition(source)
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"action array must be 2D, got shape={arr.shape}")
    if len(arr) == 0:
        raise ValueError(f"empty action array for source={source}")

    start = max(0, min(int(state_start), len(arr) - 1))
    end = max(start + 1, min(int(state_end), len(arr)))
    out = arr[start:end].astype(np.float32, copy=False)

    requested = max(1, int(state_end) - int(state_start))
    if pad_to_length and len(out) < requested:
        pad = np.repeat(out[-1:], requested - len(out), axis=0)
        out = np.concatenate([out, pad], axis=0)

    return causalize_action_window(
        out,
        source,
        history_frames=history_frames,
        vae_temporal_stride=vae_temporal_stride,
    )


def _as_2d(values, key: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{key} must be 1D or 2D, got shape={arr.shape}")
    return arr


def base_state_array(annotation: dict, source: str) -> np.ndarray:
    """Return the base 7D EEF/gripper state or commanded action array."""
    if source in {
        OBSERVATION_STATE,
        ABS_PLUS_DELTA_PLUS_GRIPPER,
        ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER,
    }:
        cart_key = "observation.state.cartesian_position"
        grip_key = "observation.state.gripper_position"
    elif source in {COMMANDED_ACTION, "commanded_cartesian_position_gripper"} and "action.cartesian_position" in annotation:
        cart_key = "action.cartesian_position"
        grip_key = "action.gripper_position"
    elif source in {COMMANDED_ACTION, "legacy"} and "observation.state.cartesian_position" in annotation:
        cart_key = "observation.state.cartesian_position"
        grip_key = "observation.state.gripper_position"
    elif "states" in annotation:
        return np.asarray(annotation["states"], dtype=np.float32)
    else:
        raise KeyError(f"Annotation missing action/state data; keys={list(annotation.keys())}")

    cart = _as_2d(annotation[cart_key], cart_key)
    grip = _as_2d(annotation[grip_key], grip_key)
    return np.concatenate([cart, grip], axis=-1).astype(np.float32)


def commanded_cartesian_velocity_gripper(annotation: dict) -> np.ndarray:
    """Return DROID commanded Cartesian velocity plus commanded gripper target."""
    if "action.cartesian_velocity" not in annotation:
        raise KeyError("annotation missing action.cartesian_velocity")
    if "action.gripper_position" not in annotation:
        raise KeyError("annotation missing action.gripper_position")
    cart_vel = _as_2d(annotation["action.cartesian_velocity"], "action.cartesian_velocity")
    grip = _as_2d(annotation["action.gripper_position"], "action.gripper_position")
    return np.concatenate([cart_vel, grip], axis=-1).astype(np.float32)


def commanded_original_action(annotation: dict) -> np.ndarray:
    """Return the compact original DROID action vector when available."""
    if "action.original" not in annotation:
        raise KeyError("annotation missing action.original")
    return _as_2d(annotation["action.original"], "action.original").astype(np.float32)


def commanded_joint_position_gripper(annotation: dict) -> np.ndarray:
    """Return DROID joint-position/gripper command."""
    if "action" in annotation:
        return _as_2d(annotation["action"], "action").astype(np.float32)
    if "action.joint_position" not in annotation:
        raise KeyError("annotation missing action.joint_position")
    if "action.gripper_position" not in annotation:
        raise KeyError("annotation missing action.gripper_position")
    joint = _as_2d(annotation["action.joint_position"], "action.joint_position")
    grip = _as_2d(annotation["action.gripper_position"], "action.gripper_position")
    return np.concatenate([joint, grip], axis=-1).astype(np.float32)


def commanded_joint_velocity_gripper(annotation: dict) -> np.ndarray:
    """Return DROID commanded joint velocity plus commanded gripper target."""
    if "action.joint_velocity" not in annotation:
        raise KeyError("annotation missing action.joint_velocity")
    if "action.gripper_position" not in annotation:
        raise KeyError("annotation missing action.gripper_position")
    joint_vel = _as_2d(annotation["action.joint_velocity"], "action.joint_velocity")
    grip = _as_2d(annotation["action.gripper_position"], "action.gripper_position")
    return np.concatenate([joint_vel, grip], axis=-1).astype(np.float32)


def commanded_full_action(annotation: dict) -> np.ndarray:
    """Return all saved DROID commanded action fields in a fixed order."""
    parts = []
    for key in [
        "action.cartesian_position",
        "action.cartesian_velocity",
        "action.gripper_position",
        "action.gripper_velocity",
        "action.joint_position",
        "action.joint_velocity",
    ]:
        if key not in annotation:
            raise KeyError(f"annotation missing {key}")
        parts.append(_as_2d(annotation[key], key))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def rpy_to_quaternion_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw radians to temporally continuous xyzw quats."""
    rpy = np.asarray(rpy, dtype=np.float32)
    roll = rpy[..., 0] * 0.5
    pitch = rpy[..., 1] * 0.5
    yaw = rpy[..., 2] * 0.5

    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    quat = np.stack([x, y, z, w], axis=-1).astype(np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-8)
    flat = quat.reshape(-1, 4)
    for i in range(1, flat.shape[0]):
        if float(np.dot(flat[i - 1], flat[i])) < 0.0:
            flat[i] *= -1.0
    return quat


def observation_state_xyz_quat_gripper(annotation: dict) -> np.ndarray:
    """Return observed EEF xyz + quaternion rotation + gripper state."""
    state = base_state_array(annotation, OBSERVATION_STATE)
    if state.shape[-1] < 7:
        raise ValueError(f"observation state must be at least 7D, got shape={state.shape}")
    xyz = state[:, :3]
    quat = rpy_to_quaternion_xyzw(state[:, 3:6])
    grip = state[:, 6:7]
    return np.concatenate([xyz, quat, grip], axis=-1).astype(np.float32)


def abs_state_plus_cartesian_velocity_gripper(annotation: dict) -> np.ndarray:
    """Return observed EEF/gripper state plus commanded Cartesian velocity."""
    state = base_state_array(annotation, OBSERVATION_STATE)
    velocity = commanded_cartesian_velocity_gripper(annotation)
    return np.concatenate([state, velocity], axis=-1).astype(np.float32)


def abs_state_quat_cartesian_velocity_gripper(annotation: dict) -> np.ndarray:
    """Return observed EEF xyz/quat/gripper state plus commanded Cartesian velocity."""
    state = observation_state_xyz_quat_gripper(annotation)
    velocity = commanded_cartesian_velocity_gripper(annotation)
    return np.concatenate([state, velocity], axis=-1).astype(np.float32)


def next_delta_state(states: np.ndarray) -> np.ndarray:
    """Return state[t+1] - state[t], with zero delta for the final row."""
    states = np.asarray(states, dtype=np.float32)
    delta = np.zeros_like(states, dtype=np.float32)
    if len(states) > 1:
        delta[:-1] = states[1:] - states[:-1]
        # DROID-style cartesian_position is 6D pose plus gripper. The last
        # three pose coordinates are angular, so raw subtraction can create
        # artificial ~2*pi jumps at the wrap boundary.
        if delta.shape[-1] >= 6:
            delta[:-1, 3:6] = (delta[:-1, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return delta


def chunk_delta_state(
    states: np.ndarray,
    stride: int = WAN_LATENT_TEMPORAL_STRIDE,
) -> np.ndarray:
    """Return state[t+stride] - state[t], clamping the target at episode end."""
    states = np.asarray(states, dtype=np.float32)
    delta = np.zeros_like(states, dtype=np.float32)
    if len(states) > 1:
        target = np.minimum(
            np.arange(len(states), dtype=np.int64) + int(stride),
            len(states) - 1,
        )
        delta = states[target] - states
        if delta.shape[-1] >= 6:
            delta[:, 3:6] = (delta[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    return delta


def action_array_from_annotation(annotation: dict, source: str) -> np.ndarray:
    """Build the configured action representation from an annotation."""
    if source == COMMANDED_ORIGINAL_ACTION:
        return commanded_original_action(annotation)
    if source == COMMANDED_JOINT_POSITION_GRIPPER:
        return commanded_joint_position_gripper(annotation)
    if source == COMMANDED_CARTESIAN_VELOCITY_GRIPPER:
        return commanded_cartesian_velocity_gripper(annotation)
    if source == COMMANDED_JOINT_VELOCITY_GRIPPER:
        return commanded_joint_velocity_gripper(annotation)
    if source == COMMANDED_FULL_ACTION:
        return commanded_full_action(annotation)
    if source in {
        ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
        CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
    }:
        return abs_state_plus_cartesian_velocity_gripper(annotation)
    if source in {
        ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
        CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
    }:
        return abs_state_quat_cartesian_velocity_gripper(annotation)
    states = base_state_array(annotation, source)
    if source == ABS_PLUS_DELTA_PLUS_GRIPPER:
        return np.concatenate([states, next_delta_state(states)], axis=-1).astype(np.float32)
    if source == ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER:
        return np.concatenate([states, chunk_delta_state(states)], axis=-1).astype(np.float32)
    return states.astype(np.float32)


def action_window_from_annotation(
    annotation: dict,
    source: str,
    *,
    state_start: int,
    state_end: int,
    history_frames: int,
    vae_temporal_stride: int = WAN_LATENT_TEMPORAL_STRIDE,
) -> np.ndarray:
    """Build a clipped, window-causal action sequence from an annotation."""
    action = action_array_from_annotation(annotation, source)
    return action_window_from_array(
        action,
        source,
        state_start=state_start,
        state_end=state_end,
        history_frames=history_frames,
        vae_temporal_stride=vae_temporal_stride,
    )


def stat_filename_for_action_source(source: str) -> str:
    if source == OBSERVATION_STATE:
        return "stat_observation_state.json"
    if source == ABS_PLUS_DELTA_PLUS_GRIPPER:
        return "stat_abs_plus_delta_plus_gripper.json"
    if source == ABS_PLUS_CHUNK_DELTA_PLUS_GRIPPER:
        return "stat_abs_plus_chunk_delta_plus_gripper.json"
    if source == COMMANDED_ORIGINAL_ACTION:
        return "stat_commanded_original_action.json"
    if source == COMMANDED_JOINT_POSITION_GRIPPER:
        return "stat_commanded_joint_position_gripper.json"
    if source == COMMANDED_CARTESIAN_VELOCITY_GRIPPER:
        return "stat_commanded_cartesian_velocity_gripper.json"
    if source == COMMANDED_JOINT_VELOCITY_GRIPPER:
        return "stat_commanded_joint_velocity_gripper.json"
    if source == COMMANDED_FULL_ACTION:
        return "stat_commanded_full_action.json"
    if source in {
        ABS_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
        CAUSAL_STATE_PLUS_CARTESIAN_VELOCITY_GRIPPER,
    }:
        return "stat_abs_state_plus_cartesian_velocity_gripper.json"
    if source in {
        ABS_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
        CAUSAL_STATE_QUAT_CARTESIAN_VELOCITY_GRIPPER,
    }:
        return "stat_abs_state_quat_cartesian_velocity_gripper.json"
    if source == COMMANDED_ACTION:
        return "stat_commanded_action.json"
    return "stat.json"
