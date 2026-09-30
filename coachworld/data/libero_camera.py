"""LIBERO camera utilities shared by data production and visual review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .canonical_robot_frame import libero_base_from_world


def normalize_instruction(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def transform_intrinsics(K: np.ndarray, image_transform: dict[str, Any]) -> np.ndarray:
    result = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    scale_x, scale_y = [float(x) for x in image_transform["scale_xy"]]
    pad_x, pad_y = [float(x) for x in image_transform["pad_xy"]]
    result[0, 0] *= scale_x
    result[0, 2] = result[0, 2] * scale_x + pad_x
    result[1, 1] *= scale_y
    result[1, 2] = result[1, 2] * scale_y + pad_y
    return result


def load_task_calibrations(sidecar_jsonl: Path) -> dict[str, dict[str, Any]]:
    calibrations: dict[str, dict[str, Any]] = {}
    seen_sources: set[str] = set()
    with sidecar_jsonl.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            sidecar = json.loads(line)
            if str(sidecar.get("suite", "")) != "libero_10":
                continue
            source_path = Path(sidecar["source_path"])
            if str(source_path) in seen_sources:
                continue
            seen_sources.add(str(source_path))
            with h5py.File(source_path, "r") as h5_file:
                problem_info = json.loads(str(h5_file["data"].attrs["problem_info"]))
                instruction = normalize_instruction(problem_info["language_instruction"])
                model_xml = str(h5_file[f"data/{sidecar['demo']}"].attrs["model_file"])
            candidate = {
                "instruction": instruction,
                "official_demo": str(source_path),
                "accepted_sidecar_episode_uid": str(sidecar["episode_uid"]),
                "K_accepted": np.asarray(
                    sidecar["intrinsics"]["K_opencv_display"], dtype=np.float64
                ),
                "accepted_raw_hw": tuple(
                    int(x) for x in sidecar["camera"]["raw_image_hw"]
                ),
                "T_camera_world": np.asarray(
                    sidecar["extrinsics"]["T_camera_world"], dtype=np.float64
                ),
                "T_robot_base_world": libero_base_from_world(model_xml),
            }
            previous = calibrations.get(instruction)
            if previous is not None:
                for key in ("K_accepted", "T_camera_world", "T_robot_base_world"):
                    if not np.allclose(previous[key], candidate[key], atol=1.0e-6):
                        raise ValueError(f"LIBERO calibration varies within task: {instruction} ({key})")
                continue
            calibrations[instruction] = candidate
    if len(calibrations) != 10:
        raise RuntimeError(f"expected 10 LIBERO-10 task calibrations, got {len(calibrations)}")
    return calibrations
