"""RGB metrics used by periodic world-model validation.

The reference is the WAN-VAE decode of the held-out GT latent window. This
measures DiT prediction quality without mixing in an additional raw-RGB/VAE
reconstruction error. Raw-RGB VAE-oracle quality is audited separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred)
    gt_arr = np.asarray(gt)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(f"pred/gt RGB shape mismatch: {pred_arr.shape} != {gt_arr.shape}")
    if pred_arr.ndim != 4 or pred_arr.shape[-1] != 3:
        raise ValueError(f"expected RGB videos shaped (T,H,W,3), got {pred_arr.shape}")
    if pred_arr.shape[0] == 0:
        raise ValueError("cannot score an empty RGB video")
    return pred_arr.astype(np.uint8, copy=False), gt_arr.astype(np.uint8, copy=False)


def init_lpips_model(net_name: str, device: str):
    """Initialize LPIPS without a silent metric fallback."""
    import lpips

    model = lpips.LPIPS(net=str(net_name)).to(device).eval()
    return model


def lpips_per_frame(
    model,
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> list[float]:
    import torch

    pred_arr, gt_arr = _validate_pair(pred, gt)
    values: list[float] = []
    size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, len(pred_arr), size):
            pred_t = torch.from_numpy(pred_arr[start : start + size].copy())
            gt_t = torch.from_numpy(gt_arr[start : start + size].copy())
            pred_t = pred_t.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            gt_t = gt_t.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            pred_t = pred_t / 127.5 - 1.0
            gt_t = gt_t / 127.5 - 1.0
            distance = model(pred_t, gt_t)
            values.extend(float(x) for x in distance.reshape(-1).detach().cpu().tolist())
    return values


@dataclass
class RGBMetricAccumulator:
    """Frame-weighted PSNR plus mean per-frame SSIM and LPIPS."""

    squared_error_sum: float = 0.0
    value_count: int = 0
    ssim_values: list[float] = field(default_factory=list)
    lpips_values: list[float] = field(default_factory=list)
    samples: int = 0
    frames: int = 0

    def merge(self, other: "RGBMetricAccumulator") -> None:
        self.squared_error_sum += float(other.squared_error_sum)
        self.value_count += int(other.value_count)
        self.ssim_values.extend(other.ssim_values)
        self.lpips_values.extend(other.lpips_values)
        self.samples += int(other.samples)
        self.frames += int(other.frames)

    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        *,
        lpips_model,
        lpips_device: str,
        lpips_batch_size: int,
    ) -> dict[str, Any]:
        from skimage.metrics import structural_similarity

        pred_arr, gt_arr = _validate_pair(pred, gt)
        diff = pred_arr.astype(np.float64) - gt_arr.astype(np.float64)
        squared_error_sum = float(np.square(diff).sum(dtype=np.float64))
        value_count = int(diff.size)
        ssim_values = [
            float(
                structural_similarity(
                    gt_frame,
                    pred_frame,
                    channel_axis=2,
                    data_range=255,
                )
            )
            for pred_frame, gt_frame in zip(pred_arr, gt_arr)
        ]
        lpips_values = lpips_per_frame(
            lpips_model,
            pred_arr,
            gt_arr,
            device=lpips_device,
            batch_size=lpips_batch_size,
        )

        self.squared_error_sum += squared_error_sum
        self.value_count += value_count
        self.ssim_values.extend(ssim_values)
        self.lpips_values.extend(lpips_values)
        self.samples += 1
        self.frames += int(len(pred_arr))
        return self._summary_from_parts(
            squared_error_sum=squared_error_sum,
            value_count=value_count,
            ssim_values=ssim_values,
            lpips_values=lpips_values,
            samples=1,
            frames=int(len(pred_arr)),
        )

    @staticmethod
    def _summary_from_parts(
        *,
        squared_error_sum: float,
        value_count: int,
        ssim_values: list[float],
        lpips_values: list[float],
        samples: int,
        frames: int,
    ) -> dict[str, Any]:
        mse = squared_error_sum / max(1, value_count)
        # A finite cap keeps JSON/W&B standards-compliant for a perfect pair.
        psnr = 100.0 if mse <= 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        return {
            "psnr": float(psnr),
            "ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
            "lpips": float(np.mean(lpips_values)) if lpips_values else float("nan"),
            "mse": float(mse),
            "samples": int(samples),
            "frames": int(frames),
        }

    def summary(self) -> dict[str, Any]:
        return self._summary_from_parts(
            squared_error_sum=self.squared_error_sum,
            value_count=self.value_count,
            ssim_values=self.ssim_values,
            lpips_values=self.lpips_values,
            samples=self.samples,
            frames=self.frames,
        )
