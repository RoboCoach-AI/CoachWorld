"""Abstract world model interface.

A world model takes (current observation + action sequence) and predicts
future observations. Concrete implementations wrap specific backbones
(currently the action-conditioned WAN 2.2 backend).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class WorldModelInput:
    """Structured input to a world model step."""

    current_obs: torch.Tensor  # Current observation latent: (C,H,W) or (C,T,H,W)
    action_sequence: np.ndarray  # Causal actions: fixed (T,D) or arm-slot (T,S,D+1)
    history_obs: list[torch.Tensor] = field(default_factory=list)  # Past obs latents
    history_actions: list[np.ndarray] = field(default_factory=list)  # Past actions
    instruction: str = ""  # Language task instruction
    domain_id: int | None = None  # Action-domain id from video_latent metadata.
    embodiment_id: int | None = None
    camera_setup_id: int | None = None
    viewmats: torch.Tensor | None = None  # Optional (F,V,4,4) camera<-world matrices.
    Ks: torch.Tensor | None = None  # Optional (F,V,3,3) intrinsics.
    eef_uv: torch.Tensor | None = None  # Optional (F,V,2) or (F,V,S,2) EEF projection pixels.
    eef_depth: torch.Tensor | None = None  # Optional (F,V) or (F,V,S) EEF camera depth.
    eef_valid: torch.Tensor | None = None  # Optional (F,V) or (F,V,S) visibility mask.
    eef_image_hw: torch.Tensor | None = None  # Optional (F,V,2) or (F,V,S,2) image height/width.
    eef_gripper: torch.Tensor | None = None  # Optional (F,S) canonical gripper state.


@dataclass
class WorldModelOutput:
    """Structured output from a world model step."""

    predicted_obs: torch.Tensor  # Predicted future obs latent: (T, C, H, W)
    predicted_frames: Optional[list[np.ndarray]] = None  # Decoded RGB: list of (H, W, 3)
    next_latent: Optional[torch.Tensor] = None  # Last latent for chaining


class BaseWorldModel(ABC):
    """Abstract action-conditioned world model interface."""

    @abstractmethod
    def step(
        self,
        wm_input: WorldModelInput,
        num_steps: Optional[int] = None,
        decode_frames: bool = True,
    ) -> WorldModelOutput:
        """Run one world model prediction step.

        Args:
            wm_input: Structured input (observation, actions, instruction).
            num_steps: Override denoising steps. Use 5-10 for fast diagnosis,
                full (50) for high-quality data generation. None = model default.
            decode_frames: If False, skip pixel decode (faster for latent-only scoring).
        """
        ...

    @abstractmethod
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """Encode RGB frames to the model's latent space.

        Args:
            frames: (N, H, W, 3) uint8 RGB images.

        Returns:
            Latent tensor in the implementation's native representation.
        """
        ...

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> Optional[np.ndarray]:
        """Decode latent tensors to RGB frames (if supported).

        Args:
            latents: Latent tensors in the model's native space.

        Returns:
            RGB frames (N, H, W, 3) uint8, or None when decoding is unsupported.
        """
        ...

    def score_rollout(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """Score a rollout by comparing predicted vs target latents.

        Default: MSE in latent space. Implementations may override this with a
        domain-specific metric or a decoded-video evaluator.

        Args:
            predicted: Predicted future latents from step().
            target: Ground-truth latents (from encode() on real frames).

        Returns:
            Scalar score (higher = better prediction / more likely success).
        """
        mse = torch.nn.functional.mse_loss(predicted, target).item()
        return -mse  # Negative MSE: higher is better

    @abstractmethod
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize raw action to [-1, 1] using dataset statistics."""
        ...

    @abstractmethod
    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] to raw scale."""
        ...
