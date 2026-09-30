from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from coachworld.data.video_latent_release import build_video_latent_release


class VideoLatentReleaseTest(unittest.TestCase):
    def _collection(self, root: Path, asset_root: Path) -> Path:
        collection = root / "collection"
        production = collection / "example_root"
        (production / "shards").mkdir(parents=True)
        asset_root.mkdir(parents=True)
        shard = asset_root / "train.latent.f16.bin"
        shard.write_bytes(b"latent")
        (production / "shards" / shard.name).symlink_to(shard)
        (production / "manifest.json").write_text("{}\n", encoding="utf-8")
        summary = {
            "complete": True,
            "missing_roots": [],
            "target_fps": 5.0,
            "target_image_hw": [512, 768],
            "window_contract": {"history_latents": 5, "future_latents": 3},
            "roots": [
                {
                    "dataset_key": "example",
                    "family": "single_real_success",
                    "output_root": str(production),
                    "splits": {
                        "train": {"episodes": 1, "hours": 1.0, "h5f3_windows": 2},
                        "val": {"episodes": 1, "hours": 0.1, "h5f3_windows": 1},
                    },
                    "linked_source_shards": 1,
                }
            ],
            "totals": {
                "train": {"episodes": 1, "hours": 1.0, "h5f3_windows": 2},
                "val": {"episodes": 1, "hours": 0.1, "h5f3_windows": 1},
            },
        }
        (collection / "collection_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (collection / "build.log").write_text("local-only\n", encoding="utf-8")
        return collection

    def test_build_hardlinks_symlink_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_root = root / "assets"
            collection = self._collection(root, asset_root)
            output = root / "release"
            result = build_video_latent_release(
                collection,
                output,
                allowed_symlink_root=asset_root,
            )
            released = output / "example_root/shards/train.latent.f16.bin"
            self.assertTrue(released.is_file())
            self.assertFalse(released.is_symlink())
            self.assertTrue(released.samefile(asset_root / "train.latent.f16.bin"))
            self.assertFalse((output / "build.log").exists())
            self.assertEqual(result["payload"]["source_symlinks_materialized"], 1)
            self.assertFalse(result["contains_symlinks"])
            self.assertEqual(result["schema_version"], 2)
            self.assertNotIn("window_contract", result)
            self.assertNotIn("h5f3_windows", result["roots"][0]["splits"]["train"])
            self.assertNotIn("h5f3_windows", result["totals"]["train"])
            self.assertEqual(
                result["roots"][0]["canonical_frame"],
                "franka_base_x_forward_y_left_z_up",
            )
            readme = (output / "README.md").read_text(encoding="utf-8")
            self.assertTrue(readme.startswith("---\nlicense: other\n"))
            self.assertIn("task:\n- video-generation", readme)
            self.assertIn("- config_name: example", readme)
            self.assertIn("path: example_root/train.index.jsonl", readme)
            self.assertIn("path: example_root/val.index.jsonl", readme)

    def test_rejects_symlink_outside_asset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset_root = root / "assets"
            collection = self._collection(root, asset_root)
            outside = root / "outside.bin"
            outside.write_bytes(b"outside")
            link = collection / "example_root/shards/train.latent.f16.bin"
            link.unlink()
            link.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes allowed asset root"):
                build_video_latent_release(
                    collection,
                    root / "release",
                    allowed_symlink_root=asset_root,
                )


if __name__ == "__main__":
    unittest.main()
