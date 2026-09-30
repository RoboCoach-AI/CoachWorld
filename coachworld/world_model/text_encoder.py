"""T5 text encoder for WAN 2.2 world model.

WAN 2.2 uses UMT5-XXL (4096-dim embeddings, max 512 tokens) for text
conditioning. WAN ships its own packed T5 checkpoint (``models_t5_umt5-xxl-enc-bf16.pth``)
and a custom T5EncoderModel implementation under ``coachworld/wan/t5.py``.

This module wraps that implementation with a cache-friendly interface
producing embeddings of shape (B, text_len, text_dim).

Reference: third_party/yy-wan-training/wan/modules/t5.py (T5EncoderModel)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# WAN 2.2 text embedding constants
DEFAULT_TEXT_DIM = 4096  # UMT5-XXL hidden size
DEFAULT_TEXT_LEN = 512   # Max sequence length

DEFAULT_CHECKPOINT_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_TOKENIZER_SUBDIR = "google/umt5-xxl"


class T5TextEncoder:
    """Lazy-loaded UMT5-XXL encoder using WAN's packed checkpoint.

    Accepts either a WAN model directory (containing
    ``models_t5_umt5-xxl-enc-bf16.pth`` and ``google/umt5-xxl/`` tokenizer)
    or explicit paths to checkpoint + tokenizer.
    """

    def __init__(
        self,
        wan_model_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = DEFAULT_TEXT_LEN,
    ):
        # Resolve paths from wan_model_dir if not explicit
        if wan_model_dir is not None:
            d = Path(wan_model_dir)
            if checkpoint_path is None:
                checkpoint_path = str(d / DEFAULT_CHECKPOINT_NAME)
            if tokenizer_path is None:
                tokenizer_path = str(d / DEFAULT_TOKENIZER_SUBDIR)

        # Auto-detect from env var (WM_CHECKPOINT_DIR) if still missing
        if checkpoint_path is None:
            env_dir = os.environ.get("WAN_MODEL_DIR")
            if env_dir:
                d = Path(env_dir)
                checkpoint_path = str(d / DEFAULT_CHECKPOINT_NAME)
                tokenizer_path = tokenizer_path or str(d / DEFAULT_TOKENIZER_SUBDIR)

        if checkpoint_path is None or tokenizer_path is None:
            raise ValueError(
                "T5TextEncoder requires wan_model_dir or explicit "
                "checkpoint_path + tokenizer_path"
            )

        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_length = max_length
        self._encoder = None

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    def load(self) -> None:
        if self._encoder is not None:
            return

        from coachworld.wan.t5 import T5EncoderModel

        logger.info(
            "Loading WAN T5 encoder: ckpt=%s tokenizer=%s",
            self.checkpoint_path, self.tokenizer_path,
        )
        self._encoder = T5EncoderModel(
            text_len=self.max_length,
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
        )
        logger.info("WAN T5 encoder loaded (device=%s)", self.device)

    def unload(self) -> None:
        if self._encoder is not None:
            self._encoder.model.cpu()
            torch.cuda.empty_cache()
            logger.info("T5 text encoder offloaded to CPU")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings.

        Returns (B, max_length, text_dim) padded embeddings.
        """
        self.load()

        # WAN encoder returns list of variable-length tensors (B, L_i, D).
        # Pad each to max_length.
        per_sample = self._encoder(texts, self.device)  # list of (L_i, D)

        B = len(per_sample)
        D = per_sample[0].shape[-1]
        out = torch.zeros(B, self.max_length, D, device=self.device, dtype=self.dtype)
        for i, emb in enumerate(per_sample):
            L = min(emb.shape[0], self.max_length)
            out[i, :L] = emb[:L].to(dtype=self.dtype)
        return out

    @torch.no_grad()
    def encode_single(self, text: str) -> torch.Tensor:
        return self.encode([text])


def make_zero_text_embedding(
    batch_size: int = 1,
    text_len: int = DEFAULT_TEXT_LEN,
    text_dim: int = DEFAULT_TEXT_DIM,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Create a zero text embedding (unconditional / CFG negative)."""
    return torch.zeros(batch_size, text_len, text_dim, device=device, dtype=dtype)


def make_null_text_embedding(
    batch_size: int = 1,
    text_len: int = DEFAULT_TEXT_LEN,
    text_dim: int = DEFAULT_TEXT_DIM,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """Create the null-text embedding used by training and inference.

    Strict zeros become identical K/V tokens after WAN's pretrained text MLP
    bias terms. Small noise preserves null-text semantics while avoiding that
    degenerate all-token-identical cross-attention input.
    """
    return (
        torch.randn(batch_size, text_len, text_dim, device=device, dtype=dtype)
        * float(noise_scale)
    )
