#!/usr/bin/env python3
"""
Convert RoboMind2.0 HDF5 files with the confirmed dual-arm aligned schema
(camera_observations/color_images + master/puppet/*_align/data)
into the ctrl-world intermediate format.

Supported input schema in this script:
- camera_observations/color_images/<camera_name>
- camera_observations/depth_images/<camera_name>  (optional, ignored for ctrl-world)
- camera_observations/timestamp
- puppet/arm_left_position_align/data
- puppet/arm_right_position_align/data
- puppet/end_effector_left_pose_align/data
- puppet/end_effector_right_pose_align/data
- puppet/end_effector_left_position_align/data
- puppet/end_effector_right_position_align/data
- master/arm_left_position_align/data
- master/arm_right_position_align/data
- master/end_effector_left_pose_align/data
- master/end_effector_right_pose_align/data
- master/end_effector_left_position_align/data
- master/end_effector_right_position_align/data

The script writes:
- output_root/videos/{train|val}/{episode_id}/{view_id}.mp4
- output_root/latent_videos/{train|val}/{episode_id}/{view_id}.pt   (optional)
- output_root/annotation/{train|val}/{episode_id}.json
- output_root/meta/episodes.jsonl
- output_root/meta/info.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

try:
    import torch
    from diffusers.models import AutoencoderKLTemporalDecoder
except Exception:
    torch = None
    AutoencoderKLTemporalDecoder = None


DEFAULT_CAMERA_PRIORITY = [
    "camera_top",
    "camera_left",
    "camera_right",
    "camera_front",
    "camera_wrist_left",
    "camera_wrist_right",
]

DEFAULT_CTRLWORLD_CAMS = ["camera_top", "camera_left", "camera_right"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def list_h5_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    return sorted([p for p in path.rglob("*.hdf5") if p.is_file()])


def infer_instruction_from_path(h5_path: Path) -> str:
    parts = list(h5_path.parts)
    if "success_episodes" in parts:
        idx = parts.index("success_episodes")
        if idx - 1 >= 0:
            return parts[idx - 1]
    return h5_path.stem


def infer_split_from_path(h5_path: Path, episode_id: int, val_mod: int, val_remainder: int) -> str:
    parts = [p.lower() for p in h5_path.parts]
    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "val"
    return "val" if (val_mod > 0 and episode_id % val_mod == val_remainder) else "train"


def decode_scalar_string(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray) and x.shape == ():
        return decode_scalar_string(x.item())
    return str(x)


def entry_to_uint8_buffer(entry) -> np.ndarray:
    if isinstance(entry, bytes):
        return np.frombuffer(entry, dtype=np.uint8)
    if isinstance(entry, np.void):
        return np.frombuffer(bytes(entry), dtype=np.uint8)
    if isinstance(entry, memoryview):
        return np.frombuffer(entry.tobytes(), dtype=np.uint8)
    arr = np.asarray(entry)
    if arr.dtype == np.uint8:
        return arr.reshape(-1)
    if arr.dtype.kind in {"S", "V"}:
        return np.frombuffer(arr.tobytes(), dtype=np.uint8)
    # fallback
    try:
        return np.frombuffer(bytes(entry), dtype=np.uint8)
    except Exception:
        return arr.astype(np.uint8).reshape(-1)


def decode_color_entry(f: h5py.File, cam: str, entry, bgr_to_rgb: bool) -> np.ndarray:
    buf = entry_to_uint8_buffer(entry)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        # fallback: raw bytes with known resolution
        res = np.asarray(f["camera_color_resolution"][cam][:], dtype=np.int64)
        h, w = int(res[0]), int(res[1])
        raw = np.frombuffer(buf.tobytes(), dtype=np.uint8)
        if raw.size != h * w * 3:
            raise ValueError(f"Cannot decode color frame for {cam}: buffer size {raw.size}, expected {h*w*3}")
        img = raw.reshape(h, w, 3)
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def decode_depth_entry(f: h5py.File, cam: str, entry) -> np.ndarray:
    buf = entry_to_uint8_buffer(entry)
    depth = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if depth is None:
        res = np.asarray(f["camera_depth_resolution"][cam][:], dtype=np.int64)
        h, w = int(res[0]), int(res[1])
        raw = np.frombuffer(buf.tobytes(), dtype=np.uint8)
        if raw.size == h * w * 2:
            depth = raw.view(np.uint16).reshape(h, w)
        elif raw.size == h * w:
            depth = raw.reshape(h, w)
        else:
            raise ValueError(f"Cannot decode depth frame for {cam}: buffer size {raw.size}")
    return depth


def normalize_depth_for_video(depth: np.ndarray) -> np.ndarray:
    x = np.asarray(depth)
    if x.size == 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if x.ndim == 3 and x.shape[-1] == 3:
        return x.astype(np.uint8)
    x = x.astype(np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-12:
        gray = np.zeros_like(x, dtype=np.uint8)
    else:
        gray = np.clip((x - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def select_cameras(available: Sequence[str], requested: Optional[Sequence[str]], use_all: bool) -> List[str]:
    available_set = set(available)
    if requested:
        cams = [c for c in requested if c in available_set]
        if cams:
            return cams
    if use_all:
        return [c for c in DEFAULT_CAMERA_PRIORITY if c in available_set] + [
            c for c in available if c not in DEFAULT_CAMERA_PRIORITY
        ]
    preferred = [c for c in DEFAULT_CTRLWORLD_CAMS if c in available_set]
    if preferred:
        return preferred
    return [c for c in DEFAULT_CAMERA_PRIORITY if c in available_set] + [
        c for c in available if c not in DEFAULT_CAMERA_PRIORITY
    ]


def nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    source_ts = np.asarray(source_ts)
    target_ts = np.asarray(target_ts)
    if source_ts.ndim != 1 or target_ts.ndim != 1:
        raise ValueError("timestamps must be 1-D")
    if len(source_ts) == 0 or len(target_ts) == 0:
        return np.zeros((0,), dtype=np.int64)
    pos = np.searchsorted(source_ts, target_ts, side="left")
    pos = np.clip(pos, 0, len(source_ts) - 1)
    prev_pos = np.clip(pos - 1, 0, len(source_ts) - 1)
    choose_prev = np.abs(source_ts[prev_pos] - target_ts) <= np.abs(source_ts[pos] - target_ts)
    out = pos.copy()
    out[choose_prev] = prev_pos[choose_prev]
    return out.astype(np.int64)


def estimate_joint_velocity(joint: np.ndarray, ts: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float32)
    ts = np.asarray(ts)
    if joint.ndim != 2:
        raise ValueError("joint array must be 2-D")
    if len(joint) == 0:
        return np.zeros_like(joint)
    if len(joint) == 1:
        return np.zeros_like(joint)
    dt = np.diff(ts.astype(np.float64))
    # Heuristic: timestamps are often in ns/us/ms. Convert to seconds if possible.
    median_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0
    if median_dt > 1e8:
        scale = 1e9
    elif median_dt > 1e5:
        scale = 1e6
    elif median_dt > 1e2:
        scale = 1e3
    else:
        scale = 1.0
    dt_sec = dt / scale
    dt_sec[dt_sec <= 0] = np.nan
    vel = np.zeros_like(joint, dtype=np.float32)
    diff = np.diff(joint, axis=0) / dt_sec[:, None]
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vel[1:] = diff
    vel[0] = vel[1] if len(vel) > 1 else 0.0
    return vel


class LatentEncoder:
    def __init__(self, svd_path: str, device: str = "cuda"):
        if torch is None or AutoencoderKLTemporalDecoder is None:
            raise ImportError("torch/diffusers are required when latent extraction is enabled")
        self.device = device
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
        self.vae.eval()

    @torch.no_grad()
    def encode_video(self, frames_rgb_uint8: np.ndarray, batch_size: int = 64) -> torch.Tensor:
        # frames: [T,H,W,3] RGB uint8
        x = torch.from_numpy(frames_rgb_uint8).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        x = x.to(self.device)
        chunks = []
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size]
            latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
            chunks.append(latent)
        return torch.cat(chunks, dim=0)


class H5EpisodeReader:
    def __init__(self, h5_path: Path, bgr_to_rgb: bool = True):
        self.h5_path = h5_path
        self.bgr_to_rgb = bgr_to_rgb

    def __enter__(self):
        self.f = h5py.File(self.h5_path, "r")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.f.close()

    @property
    def available_cameras(self) -> List[str]:
        return sorted(list(self.f["camera_observations"]["color_images"].keys()))

    def load_control_block(self, arm_root: str) -> Dict[str, np.ndarray]:
        block = {}
        for key in [
            "arm_left_position_align",
            "arm_right_position_align",
            "end_effector_left_pose_align",
            "end_effector_left_position_align",
            "end_effector_right_pose_align",
            "end_effector_right_position_align",
        ]:
            grp = self.f[arm_root][key]
            block[key] = np.asarray(grp["data"][:], dtype=np.float32)
            block[key + ".timestamp"] = np.asarray(grp["timestamp"][:])
            block[key + ".is_intervene"] = np.asarray(grp["is_intervene"][:]).astype(bool)
        return block

    def build_aligned_episode(
        self,
        selected_cameras: Sequence[str],
        rgb_skip: int,
        target_size: Tuple[int, int],
    ) -> Dict[str, object]:
        f = self.f
        cam_ts = np.asarray(f["camera_observations"]["timestamp"][:])
        if cam_ts.ndim != 1:
            raise ValueError("camera_observations/timestamp must be 1-D")

        puppet = self.load_control_block("puppet")
        master = self.load_control_block("master")

        puppet_joint = np.concatenate(
            [puppet["arm_left_position_align"], puppet["arm_right_position_align"]], axis=1
        )
        puppet_cart = np.concatenate(
            [puppet["end_effector_left_pose_align"], puppet["end_effector_right_pose_align"]], axis=1
        )
        puppet_grip = np.concatenate(
            [puppet["end_effector_left_position_align"], puppet["end_effector_right_position_align"]], axis=1
        )

        master_joint = np.concatenate(
            [master["arm_left_position_align"], master["arm_right_position_align"]], axis=1
        )
        master_cart = np.concatenate(
            [master["end_effector_left_pose_align"], master["end_effector_right_pose_align"]], axis=1
        )
        master_grip = np.concatenate(
            [master["end_effector_left_position_align"], master["end_effector_right_position_align"]], axis=1
        )

        # use left-arm timestamps as the canonical source timestamps for aligned control streams
        puppet_ts = puppet["arm_left_position_align.timestamp"]
        master_ts = master["arm_left_position_align.timestamp"]

        sampled_indices = np.arange(0, len(cam_ts), max(1, rgb_skip), dtype=np.int64)
        target_ts = cam_ts[sampled_indices]
        obs_idx = nearest_indices(puppet_ts, target_ts)
        act_idx = nearest_indices(master_ts, target_ts)

        obs_joint = puppet_joint[obs_idx]
        obs_cart = puppet_cart[obs_idx]
        obs_grip = puppet_grip[obs_idx]
        act_joint = master_joint[act_idx]
        act_cart = master_cart[act_idx]
        act_grip = master_grip[act_idx]
        act_vel = estimate_joint_velocity(act_joint, target_ts)
        compact_states = np.concatenate([obs_cart, obs_grip], axis=1)

        # decode only the kept frames for selected cameras
        decoded_videos: Dict[str, np.ndarray] = {}
        raw_color_group = f["camera_observations"]["color_images"]
        for cam in selected_cameras:
            frames = []
            ds = raw_color_group[cam]
            for i in sampled_indices:
                img = decode_color_entry(f, cam, ds[int(i)], bgr_to_rgb=self.bgr_to_rgb)
                img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
                frames.append(img)
            decoded_videos[cam] = np.stack(frames, axis=0) if frames else np.zeros((0, target_size[0], target_size[1], 3), dtype=np.uint8)

        min_len = min(
            [len(target_ts), len(obs_joint), len(obs_cart), len(obs_grip), len(act_joint), len(act_cart), len(act_grip)]
            + [len(v) for v in decoded_videos.values()]
        )

        if min_len <= 0:
            raise ValueError(f"No aligned frames found in {self.h5_path}")

        for cam in list(decoded_videos.keys()):
            decoded_videos[cam] = decoded_videos[cam][:min_len]

        return {
            "camera_timestamps": target_ts[:min_len],
            "videos": decoded_videos,
            "observation.state.joint_position": obs_joint[:min_len].astype(np.float32),
            "observation.state.cartesian_position": obs_cart[:min_len].astype(np.float32),
            "observation.state.gripper_position": obs_grip[:min_len].astype(np.float32),
            "action.joint_position": act_joint[:min_len].astype(np.float32),
            "action.cartesian_position": act_cart[:min_len].astype(np.float32),
            "action.gripper_position": act_grip[:min_len].astype(np.float32),
            "action.joint_velocity": act_vel[:min_len].astype(np.float32),
            "states": compact_states[:min_len].astype(np.float32),
            "raw_length": int(len(cam_ts)),
            "selected_indices": sampled_indices[:min_len].astype(np.int64),
        }



def save_video_mp4(path: Path, frames_rgb: np.ndarray, fps: int) -> None:
    ensure_dir(path.parent)
    if len(frames_rgb) == 0:
        raise ValueError(f"No frames to save: {path}")
    h, w = frames_rgb.shape[1], frames_rgb.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {path}")
    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()



def process_single_h5(
    h5_path: Path,
    output_root: Path,
    episode_id: int,
    instruction: Optional[str],
    rgb_skip: int,
    source_fps: int,
    target_fps: int,
    size: Tuple[int, int],
    cameras: Optional[Sequence[str]],
    use_all_cameras: bool,
    pad_views_to: int,
    duplicate_mode: str,
    default_success: int,
    bgr_to_rgb: bool,
    latent_encoder: Optional[LatentEncoder],
    val_mod: int,
    val_remainder: int,
) -> Dict[str, object]:
    split = infer_split_from_path(h5_path, episode_id, val_mod, val_remainder)
    text = instruction if instruction is not None else infer_instruction_from_path(h5_path)

    with H5EpisodeReader(h5_path, bgr_to_rgb=bgr_to_rgb) as reader:
        available = reader.available_cameras
        selected = select_cameras(available, cameras, use_all_cameras)
        aligned = reader.build_aligned_episode(selected, rgb_skip=rgb_skip, target_size=size)

    videos = aligned["videos"]
    selected_names = list(videos.keys())
    if pad_views_to > 0 and len(selected_names) > 0 and len(selected_names) < pad_views_to:
        filler_name = selected_names[0] if duplicate_mode == "first" else selected_names[-1]
        filler_video = videos[filler_name]
        while len(selected_names) < pad_views_to:
            new_name = f"{filler_name}_dup{len(selected_names)}"
            videos[new_name] = filler_video.copy()
            selected_names.append(new_name)

    video_entries = []
    latent_entries = []
    video_dir = output_root / "videos" / split / str(episode_id)
    latent_dir = output_root / "latent_videos" / split / str(episode_id)
    ensure_dir(video_dir)
    if latent_encoder is not None:
        ensure_dir(latent_dir)

    for view_id, cam in enumerate(selected_names):
        frames_rgb = videos[cam]
        rel_video = Path("videos") / split / str(episode_id) / f"{view_id}.mp4"
        save_video_mp4(output_root / rel_video, frames_rgb, fps=target_fps)
        video_entries.append({"video_path": rel_video.as_posix(), "camera_name": cam})
        if latent_encoder is not None:
            rel_latent = Path("latent_videos") / split / str(episode_id) / f"{view_id}.pt"
            latent = latent_encoder.encode_video(frames_rgb)
            torch.save(latent, output_root / rel_latent)
            latent_entries.append({"latent_video_path": rel_latent.as_posix(), "camera_name": cam})

    obs_cart = aligned["observation.state.cartesian_position"].tolist()
    obs_joint = aligned["observation.state.joint_position"].tolist()
    obs_grip = aligned["observation.state.gripper_position"].tolist()
    act_cart = aligned["action.cartesian_position"].tolist()
    act_joint = aligned["action.joint_position"].tolist()
    act_grip = aligned["action.gripper_position"].tolist()
    act_vel = aligned["action.joint_velocity"].tolist()
    states = aligned["states"].tolist()

    info = {
        "texts": [text],
        "episode_id": int(episode_id),
        "success": int(default_success),
        "video_length": int(len(states)),
        "state_length": int(len(states)),
        "raw_length": int(aligned["raw_length"]),
        "videos": video_entries,
        "latent_videos": latent_entries,
        "states": states,
        "observation.state.cartesian_position": obs_cart,
        "observation.state.joint_position": obs_joint,
        "observation.state.gripper_position": obs_grip,
        "action.cartesian_position": act_cart,
        "action.joint_position": act_joint,
        "action.gripper_position": act_grip,
        "action.joint_velocity": act_vel,
        "camera_timestamps": aligned["camera_timestamps"].astype(np.int64).tolist(),
        "selected_indices": aligned["selected_indices"].astype(np.int64).tolist(),
        "selected_cameras": selected_names,
        "source_h5_path": str(h5_path),
        "source_schema": "robomind2_dual_arm_align_hdf5",
        "source_fps": int(source_fps),
        "target_fps": int(target_fps),
        "rgb_skip": int(rgb_skip),
    }

    ann_path = output_root / "annotation" / split / f"{episode_id}.json"
    write_json(ann_path, info)

    return {
        "episode_index": int(episode_id),
        "tasks": [text],
        "length": int(len(states)),
        "split": split,
        "h5_path": str(h5_path),
        "selected_cameras": selected_names,
    }



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="single .hdf5 file or a directory to recursively scan")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--instruction", type=str, default=None, help="override task text for all episodes")
    parser.add_argument("--source_fps", type=int, default=30)
    parser.add_argument("--target_fps", type=int, default=5)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--use_all_cameras", action="store_true")
    parser.add_argument("--pad_views_to", type=int, default=0)
    parser.add_argument("--duplicate_mode", type=str, choices=["first", "last"], default="last")
    parser.add_argument("--default_success", type=int, default=1)
    parser.add_argument("--episode_start", type=int, default=0)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--val_mod", type=int, default=100)
    parser.add_argument("--val_remainder", type=int, default=99)
    parser.add_argument("--bgr_to_rgb", action="store_true", default=True)
    parser.add_argument("--no_bgr_to_rgb", action="store_false", dest="bgr_to_rgb")
    parser.add_argument("--svd_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if args.target_fps <= 0 or args.source_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    rgb_skip = max(1, int(round(args.source_fps / args.target_fps)))
    size = (int(args.height), int(args.width))

    h5_files = list_h5_files(input_path)
    if not h5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {input_path}")
    if args.max_episodes is not None:
        h5_files = h5_files[: args.max_episodes]

    latent_encoder = None
    if args.svd_path is not None:
        latent_encoder = LatentEncoder(args.svd_path, device=args.device)

    meta_records = []
    for local_idx, h5_path in enumerate(h5_files):
        episode_id = args.episode_start + local_idx
        print(f"Processing episode {episode_id}: {h5_path}")
        rec = process_single_h5(
            h5_path=h5_path,
            output_root=output_root,
            episode_id=episode_id,
            instruction=args.instruction,
            rgb_skip=rgb_skip,
            source_fps=args.source_fps,
            target_fps=args.target_fps,
            size=size,
            cameras=args.cameras,
            use_all_cameras=args.use_all_cameras,
            pad_views_to=args.pad_views_to,
            duplicate_mode=args.duplicate_mode,
            default_success=args.default_success,
            bgr_to_rgb=args.bgr_to_rgb,
            latent_encoder=latent_encoder,
            val_mod=args.val_mod,
            val_remainder=args.val_remainder,
        )
        meta_records.append(rec)

    meta_dir = output_root / "meta"
    ensure_dir(meta_dir)
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps({
                "episode_index": rec["episode_index"],
                "tasks": rec["tasks"],
                "length": rec["length"],
            }, ensure_ascii=False) + "\n")

    info = {
        "source_schema": "robomind2_dual_arm_align_hdf5",
        "total_episodes": len(meta_records),
        "source_fps": int(args.source_fps),
        "target_fps": int(args.target_fps),
        "rgb_skip": int(rgb_skip),
        "size": [int(args.height), int(args.width)],
        "latent_enabled": args.svd_path is not None,
        "input": str(input_path),
        "output_root": str(output_root),
        "splits": {
            "train": sum(1 for x in meta_records if x["split"] == "train"),
            "val": sum(1 for x in meta_records if x["split"] == "val"),
        },
    }
    write_json(meta_dir / "info.json", info)
    print(f"Done. Wrote {len(meta_records)} episodes to {output_root}")


if __name__ == "__main__":
    main()
