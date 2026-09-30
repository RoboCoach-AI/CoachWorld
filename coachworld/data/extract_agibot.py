import os
import json
import argparse
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch

try:
    from diffusers.models import AutoencoderKLTemporalDecoder
except Exception:
    AutoencoderKLTemporalDecoder = None


DEFAULT_CAMERAS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_numpy_1d(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x)
    else:
        arr = np.asarray(x)
    return arr.astype(np.float32).reshape(-1)


def get_field_slice(field_desc: Dict[str, Any]) -> slice:
    indices = field_desc.get("indices", [])
    if not indices:
        return slice(0, 0)
    return slice(indices[0], indices[-1] + 1)


def extract_fields(vec: np.ndarray, feature_desc: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fd = feature_desc["field_descriptions"]

    obs_gripper = np.concatenate(
        [
            vec[get_field_slice(fd["state/left_effector/position"])],
            vec[get_field_slice(fd["state/right_effector/position"])],
        ],
        axis=0,
    )
    obs_cart = np.concatenate(
        [
            vec[get_field_slice(fd["state/end/position"])],
            vec[get_field_slice(fd["state/end/orientation"])],
        ],
        axis=0,
    )
    obs_joint = vec[get_field_slice(fd["state/joint/position"])].copy()
    obs_joint_vel = vec[get_field_slice(fd["state/joint/velocity"])].copy()

    return {
        "gripper": obs_gripper,
        "cartesian": obs_cart,
        "joint": obs_joint,
        "joint_velocity": obs_joint_vel,
        "raw": vec.copy(),
    }


def extract_action_fields(vec: np.ndarray, feature_desc: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fd = feature_desc["field_descriptions"]

    act_gripper = np.concatenate(
        [
            vec[get_field_slice(fd["action/left_effector/position"])],
            vec[get_field_slice(fd["action/right_effector/position"])],
        ],
        axis=0,
    )
    act_cart = np.concatenate(
        [
            vec[get_field_slice(fd["action/end/position"])],
            vec[get_field_slice(fd["action/end/orientation"])],
        ],
        axis=0,
    )
    act_joint = vec[get_field_slice(fd["action/joint/position"])].copy()

    return {
        "gripper": act_gripper,
        "cartesian": act_cart,
        "joint": act_joint,
        "raw": vec.copy(),
    }


def compute_joint_velocity(joints: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float32).reshape(-1)
    n = joints.shape[0]
    if n == 0:
        return joints.copy()
    vel = np.zeros_like(joints, dtype=np.float32)
    if n == 1:
        return vel
    dt = np.diff(timestamps)
    dt[dt == 0] = 1.0
    vel[1:] = (joints[1:] - joints[:-1]) / dt[:, None]
    vel[0] = vel[1]
    return vel


def read_video_cv2(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return np.stack(frames, axis=0)


def write_video_cv2(video_path: str, frames_rgb: np.ndarray, fps: float) -> None:
    if frames_rgb.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C], got {frames_rgb.shape}")
    ensure_dir(os.path.dirname(video_path))
    h, w = frames_rgb.shape[1], frames_rgb.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    for frame in frames_rgb:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


class AgiBotWorldToCtrlWorld:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_root = args.dataset_root
        self.output_root = args.output_root
        self.meta_root = os.path.join(self.dataset_root, "meta")
        self.info = self._load_info()
        self.episodes = self._load_episodes()
        self.chunk_size = int(self.info.get("chunks_size", 1000))
        self.source_fps = float(args.source_fps or self.info.get("fps", 30))
        self.target_fps = float(args.target_fps)
        self.rgb_skip = max(1, int(round(self.source_fps / self.target_fps)))
        self.size = (args.height, args.width)
        self.camera_keys = args.camera_keys if args.camera_keys else list(DEFAULT_CAMERAS)
        self.default_success = int(args.default_success)
        self.val_mod = int(args.val_mod)
        self.val_remainder = int(args.val_remainder)
        self.skip_latent = args.skip_latent or not args.svd_path
        self.vae = None
        self.device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.skip_latent:
            if AutoencoderKLTemporalDecoder is None:
                raise ImportError("diffusers is required for latent extraction. Install diffusers or use --skip_latent.")
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(args.svd_path, subfolder="vae").to(self.device)
            self.vae.eval()

        obs_desc = self.info["features"]["observation.state"]
        act_desc = self.info["features"]["action"]
        self.obs_desc = obs_desc
        self.act_desc = act_desc

    def _load_info(self) -> Dict[str, Any]:
        info_path = os.path.join(self.meta_root, "info.json")
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_episodes(self) -> List[Dict[str, Any]]:
        episodes_path = os.path.join(self.meta_root, "episodes.jsonl")
        return load_jsonl(episodes_path)

    def data_type_for_episode(self, episode_index: int) -> str:
        return "val" if episode_index % self.val_mod == self.val_remainder else "train"

    def resolve_parquet_path(self, episode_index: int) -> str:
        tmpl = self.info["data_path"]
        chunk = episode_index // self.chunk_size
        rel = tmpl.format(episode_chunk=chunk, episode_index=episode_index)
        return os.path.join(self.dataset_root, rel)

    def resolve_video_path(self, episode_index: int, video_key: str) -> str:
        tmpl = self.info["video_path"]
        chunk = episode_index // self.chunk_size
        rel = tmpl.format(episode_chunk=chunk, episode_index=episode_index, video_key=video_key)
        return os.path.join(self.dataset_root, rel)

    def build_instruction(self, episode_meta: Dict[str, Any]) -> str:
        tasks = episode_meta.get("tasks", [])
        if tasks:
            return str(tasks[0])
        return self.args.default_instruction or "AgiBot-World task"

    def maybe_limit_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.args.max_episodes is None:
            return episodes
        return episodes[: self.args.max_episodes]

    def process(self) -> None:
        ensure_dir(self.output_root)
        ensure_dir(os.path.join(self.output_root, "videos"))
        ensure_dir(os.path.join(self.output_root, "latent_videos"))
        ensure_dir(os.path.join(self.output_root, "annotation"))
        ensure_dir(os.path.join(self.output_root, "meta"))

        out_eps: List[Dict[str, Any]] = []
        episodes = self.maybe_limit_episodes(self.episodes)
        for idx, ep in enumerate(episodes):
            episode_index = int(ep["episode_index"])
            try:
                self.process_one(ep)
                out_eps.append(ep)
                if idx % 10 == 0:
                    print(f"Processed {idx + 1}/{len(episodes)} episodes")
            except Exception as e:
                print(f"Error processing episode {episode_index}: {e}")
                if self.args.stop_on_error:
                    raise

        with open(os.path.join(self.output_root, "meta", "episodes.jsonl"), "w", encoding="utf-8") as f:
            for ep in out_eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

        output_info = {
            "source_dataset": "AgiBot World",
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "rgb_skip": self.rgb_skip,
            "resize": [self.size[0], self.size[1]],
            "camera_keys": self.camera_keys,
            "total_episodes": len(out_eps),
        }
        with open(os.path.join(self.output_root, "meta", "info.json"), "w", encoding="utf-8") as f:
            json.dump(output_info, f, indent=2, ensure_ascii=False)

    def process_one(self, episode_meta: Dict[str, Any]) -> None:
        episode_index = int(episode_meta["episode_index"])
        instruction = self.build_instruction(episode_meta)
        data_type = self.data_type_for_episode(episode_index)

        parquet_path = self.resolve_parquet_path(episode_index)
        df = pd.read_parquet(parquet_path)
        if "observation.state" not in df.columns or "action" not in df.columns:
            raise KeyError("Parquet must contain 'observation.state' and 'action' columns")
        if "timestamp" in df.columns:
            timestamps = np.asarray(df["timestamp"].tolist(), dtype=np.float32).reshape(-1)
        elif "ts" in df.columns:
            timestamps = np.asarray(df["ts"].tolist(), dtype=np.float32).reshape(-1)
        else:
            timestamps = np.arange(len(df), dtype=np.float32) / self.source_fps

        obs_cart, obs_joint, obs_gripper, obs_joint_vel, raw_obs = [], [], [], [], []
        act_cart, act_joint, act_gripper, raw_act = [], [], [], []

        for i in range(len(df)):
            obs_vec = to_numpy_1d(df["observation.state"].iloc[i])
            act_vec = to_numpy_1d(df["action"].iloc[i])
            obs = extract_fields(obs_vec, self.obs_desc)
            act = extract_action_fields(act_vec, self.act_desc)
            obs_cart.append(obs["cartesian"])
            obs_joint.append(obs["joint"])
            obs_gripper.append(obs["gripper"])
            obs_joint_vel.append(obs["joint_velocity"])
            raw_obs.append(obs["raw"])
            act_cart.append(act["cartesian"])
            act_joint.append(act["joint"])
            act_gripper.append(act["gripper"])
            raw_act.append(act["raw"])

        obs_cart_arr = np.stack(obs_cart, axis=0)
        obs_joint_arr = np.stack(obs_joint, axis=0)
        obs_gripper_arr = np.stack(obs_gripper, axis=0)
        obs_joint_vel_arr = np.stack(obs_joint_vel, axis=0)
        act_cart_arr = np.stack(act_cart, axis=0)
        act_joint_arr = np.stack(act_joint, axis=0)
        act_gripper_arr = np.stack(act_gripper, axis=0)
        action_joint_vel_arr = compute_joint_velocity(act_joint_arr, timestamps)

        # downsample states/actions to match video skip
        ds_idx = np.arange(0, len(df), self.rgb_skip)
        obs_cart_ds = obs_cart_arr[ds_idx]
        obs_joint_ds = obs_joint_arr[ds_idx]
        obs_gripper_ds = obs_gripper_arr[ds_idx]
        obs_joint_vel_ds = obs_joint_vel_arr[ds_idx]
        act_cart_ds = act_cart_arr[ds_idx]
        act_joint_ds = act_joint_arr[ds_idx]
        act_gripper_ds = act_gripper_arr[ds_idx]
        action_joint_vel_ds = action_joint_vel_arr[ds_idx]
        timestamps_ds = timestamps[ds_idx]
        states = np.concatenate([obs_cart_ds, obs_gripper_ds], axis=1)

        videos_meta = []
        latent_videos_meta = []
        saved_video_lengths = []

        for video_id, camera_key in enumerate(self.camera_keys):
            video_path = self.resolve_video_path(episode_index, camera_key)
            frames = read_video_cv2(video_path)
            frames = frames[:: self.rgb_skip]
            resized = np.stack(
                [cv2.resize(frame, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA) for frame in frames],
                axis=0,
            )
            save_video_path = os.path.join(self.output_root, "videos", data_type, str(episode_index), f"{video_id}.mp4")
            write_video_cv2(save_video_path, resized, fps=self.target_fps)
            videos_meta.append({"video_path": f"videos/{data_type}/{episode_index}/{video_id}.mp4"})
            saved_video_lengths.append(len(resized))

            if not self.skip_latent:
                latent = self.encode_video_to_latent(resized)
                latent_path = os.path.join(self.output_root, "latent_videos", data_type, str(episode_index), f"{video_id}.pt")
                ensure_dir(os.path.dirname(latent_path))
                torch.save(latent, latent_path)
                latent_videos_meta.append({"latent_video_path": f"latent_videos/{data_type}/{episode_index}/{video_id}.pt"})

        video_length = min(saved_video_lengths) if saved_video_lengths else len(ds_idx)
        state_length = min(video_length, len(states))

        info = {
            "texts": [instruction],
            "episode_id": episode_index,
            "success": self.default_success,
            "video_length": int(video_length),
            "state_length": int(state_length),
            "raw_length": int(len(df)),
            "videos": videos_meta,
            "latent_videos": latent_videos_meta,
            "states": states[:state_length].tolist(),
            "observation.state.cartesian_position": obs_cart_ds[:state_length].tolist(),
            "observation.state.joint_position": obs_joint_ds[:state_length].tolist(),
            "observation.state.gripper_position": obs_gripper_ds[:state_length].tolist(),
            "observation.state.joint_velocity": obs_joint_vel_ds[:state_length].tolist(),
            "action.cartesian_position": act_cart_ds[:state_length].tolist(),
            "action.joint_position": act_joint_ds[:state_length].tolist(),
            "action.gripper_position": act_gripper_ds[:state_length].tolist(),
            "action.joint_velocity": action_joint_vel_ds[:state_length].tolist(),
            "timestamp": timestamps_ds[:state_length].tolist(),
            "source_observation.state": [arr.tolist() for arr in raw_obs],
            "source_action": [arr.tolist() for arr in raw_act],
            "source_camera_keys": self.camera_keys,
            "source_parquet_path": os.path.relpath(parquet_path, self.dataset_root).replace("\\", "/"),
        }
        ann_path = os.path.join(self.output_root, "annotation", data_type, f"{episode_index}.json")
        ensure_dir(os.path.dirname(ann_path))
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def encode_video_to_latent(self, frames_rgb: np.ndarray) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE is not initialized")
        x = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float() / 255.0
        x = x * 2 - 1
        x = x.to(self.device)
        latents = []
        batch_size = self.args.latent_batch_size
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch = x[i : i + batch_size]
                latent = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                latents.append(latent)
        return torch.cat(latents, dim=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert AgiBot World dataset to ctrl-world intermediate format")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--camera_keys", nargs="*", default=None, help="Default: top_head hand_left hand_right")
    parser.add_argument("--source_fps", type=float, default=None, help="Override source fps; default reads meta/info.json")
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--svd_path", type=str, default=None)
    parser.add_argument("--skip_latent", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--latent_batch_size", type=int, default=64)
    parser.add_argument("--default_success", type=int, default=1)
    parser.add_argument("--default_instruction", type=str, default=None)
    parser.add_argument("--val_mod", type=int, default=100)
    parser.add_argument("--val_remainder", type=int, default=99)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--stop_on_error", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    converter = AgiBotWorldToCtrlWorld(args)
    converter.process()


if __name__ == "__main__":
    main()
