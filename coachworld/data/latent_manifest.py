"""Processing manifest helpers for pre-extracted latent datasets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


MANIFEST_NAME = "processing_manifest.json"


def sha1_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_commit(project_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def git_dirty(project_root: Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.strip())


def path_fingerprint(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "is_dir": path.is_dir(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "meta_info" / MANIFEST_NAME


def build_extraction_manifest(
    *,
    project_root: Path,
    data_path: Path,
    output_dir: Path,
    vae_path: Path,
    rgb_skip: int,
    target_size: tuple[int, int] | None,
    splits: list[str],
    video_keys: list[str],
    require_all_cameras: bool,
    decoder: str,
    resize_backend: str,
    action_grid: str,
) -> dict[str, Any]:
    script_path = project_root / "scripts" / "extract_latents.py"
    manifest = {
        "schema_version": 1,
        "kind": "coachworld_latent_extraction",
        "project_root": str(project_root.resolve()),
        "git_commit": git_commit(project_root),
        "git_dirty": git_dirty(project_root),
        "script": {
            "path": str(script_path.resolve()),
            "sha1": sha1_file(script_path),
        },
        "data_source": path_fingerprint(data_path),
        "output_dir": str(output_dir.expanduser().resolve()),
        "vae": {
            **path_fingerprint(vae_path),
            "sha1": sha1_file(vae_path),
        },
        "processing": {
            "rgb_skip": int(rgb_skip),
            "target_size": list(target_size) if target_size is not None else None,
            "splits": list(splits),
            "video_keys": list(video_keys),
            "require_all_cameras": bool(require_all_cameras),
            "decoder": str(decoder),
            "resize_backend": str(resize_backend),
            "action_grid": str(action_grid),
        },
        "environment": {
            "python": os.environ.get("VIRTUAL_ENV", ""),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    return manifest


def comparable_manifest_key(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "kind": manifest.get("kind"),
        "data_source": {
            "path": manifest.get("data_source", {}).get("path"),
            "name": manifest.get("data_source", {}).get("name"),
        },
        "vae": {
            "path": manifest.get("vae", {}).get("path"),
            "sha1": manifest.get("vae", {}).get("sha1"),
        },
        "processing": manifest.get("processing", {}),
    }


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def assert_manifest_compatible(existing: dict[str, Any], current: dict[str, Any], path: Path) -> None:
    existing_key = comparable_manifest_key(existing)
    current_key = comparable_manifest_key(current)
    if existing_key == current_key:
        return
    raise RuntimeError(
        "latent processing manifest mismatch; refusing to resume into this output_dir. "
        f"manifest={path}\n"
        f"existing={json.dumps(existing_key, indent=2)}\n"
        f"current={json.dumps(current_key, indent=2)}"
    )
