"""URDF mesh and FK sequence assets shared by calibration viewers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBALL_ROOT = PROJECT_ROOT / "CalibAll"


def iter_mesh_paths(robot: Any) -> list[Path]:
    paths: list[Path] = []
    for item in list(getattr(robot, "MESH_PATHS")):
        path = Path(str(item))
        if not path.is_absolute():
            path = CALIBALL_ROOT / path
        paths.append(path)
    return paths


def load_mesh_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )


def normalize_fk_poses(value: Any) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim == 4 and poses.shape[-2:] == (4, 4):
        poses = poses.reshape(-1, 4, 4)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"unexpected fkine_all shape: {poses.shape}")
    return poses


def resolve_link_paths(robot: Any, pose_count: int) -> list[Path]:
    paths = iter_mesh_paths(robot)
    if len(paths) == pose_count:
        return paths
    if len(paths) * 2 == pose_count:
        return paths + paths
    if pose_count * 2 == len(paths):
        return paths[:pose_count]
    raise ValueError(f"mesh path count {len(paths)} does not match link poses {pose_count}")


def build_link_geometry(
    robot: Any,
    q0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    poses = normalize_fk_poses(robot.fkine_all(q0))
    paths = resolve_link_paths(robot, len(poses))
    declared_names = list(getattr(robot, "LINK_NAMES", ()))
    if len(declared_names) != len(poses):
        declared_names = [path.stem for path in paths]
    gripper_names = {
        str(name) for name in getattr(robot, "GRIPPER_NAMES", ())
    }
    all_vertices: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    links: list[dict[str, Any]] = []
    vertex_offset = 0
    index_offset = 0
    for link_index, path in enumerate(paths):
        vertices, faces = load_mesh_vertices_faces(path)
        indices = faces.reshape(-1).astype(np.uint32) + np.uint32(vertex_offset)
        all_vertices.append(vertices.astype(np.float32))
        all_indices.append(indices)
        link_name = str(declared_names[link_index])
        links.append(
            {
                "link_index": link_index,
                "name": link_name,
                "is_gripper": link_name in gripper_names,
                "mesh_path": str(path),
                "vertex_offset": vertex_offset,
                "vertex_count": int(vertices.shape[0]),
                "index_offset": index_offset,
                "index_count": int(indices.size),
            }
        )
        vertex_offset += int(vertices.shape[0])
        index_offset += int(indices.size)
    return np.concatenate(all_vertices), np.concatenate(all_indices), links


def build_fk_sequence(robot: Any, qpos: np.ndarray, link_count: int) -> np.ndarray:
    """Return WebGL column-major FK matrices for every state and link."""

    transforms = np.empty((len(qpos), link_count, 4, 4), dtype=np.float32)
    for state_index, q in enumerate(qpos):
        poses = normalize_fk_poses(robot.fkine_all(q))
        if len(poses) != link_count:
            raise ValueError(
                f"state {state_index}: FK link count {len(poses)} "
                f"!= geometry link count {link_count}"
            )
        transforms[state_index] = poses.astype(np.float32)
    return transforms.transpose(0, 1, 3, 2).copy()
