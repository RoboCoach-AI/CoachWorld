from __future__ import annotations

import unittest

import numpy as np
import torch

from coachworld.world_model.inference_utils import decode_latents_in_chunks


class _FakeWorldModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def decode(self, latents: torch.Tensor) -> np.ndarray:
        count = 1 + 4 * (int(latents.shape[1]) - 1)
        return np.zeros((count, 2, 2, 3), dtype=np.uint8)


class ChunkedDecodeTest(unittest.TestCase):
    def test_chunked_decode_preserves_wan_frame_count(self) -> None:
        latents = torch.zeros(4, 11, 2, 2)
        decoded = decode_latents_in_chunks(_FakeWorldModel(), latents, 4)
        self.assertEqual(decoded.shape[0], 1 + 4 * (11 - 1))

    def test_chunk_size_must_leave_room_for_shared_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            decode_latents_in_chunks(
                _FakeWorldModel(), torch.zeros(4, 2, 2, 2), 1
            )


if __name__ == "__main__":
    unittest.main()
