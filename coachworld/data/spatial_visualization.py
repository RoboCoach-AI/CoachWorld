"""Spatial scene extraction for video_latent visual reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from coachworld.data.video_latent import iter_index, read_manifest


def _memmap_array(root: Path, spec: dict[str, Any], *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    dtype = np.dtype(spec["dtype"])
    path = root / "shards" / str(spec["shard"])
    if not path.exists():
        raise FileNotFoundError(path)
    if int(spec["offset_bytes"]) % dtype.itemsize:
        raise ValueError(f"{path} offset is not divisible by dtype size")
    out_shape = tuple(int(x) for x in (shape or spec["shape"]))
    count = int(np.prod(out_shape))
    start = int(spec["offset_bytes"]) // dtype.itemsize
    mm = np.memmap(path, mode="r", dtype=dtype)
    end = start + count
    if end > int(mm.shape[0]):
        raise ValueError(f"{path} spec exceeds shard bounds")
    return np.asarray(mm[start:end].reshape(out_shape))


def load_condition(root: Path, entry: dict[str, Any], condition_view: str) -> np.ndarray:
    spec = entry["model_condition_views"].get(condition_view)
    if spec is None:
        raise KeyError(f"{entry['episode_uid']} missing condition view {condition_view!r}")
    arr = np.asarray(_memmap_array(root, spec), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] < 4:
        raise ValueError(f"{entry['episode_uid']} invalid condition shape {arr.shape}")
    return arr


def load_signal_field(root: Path, entry: dict[str, Any], field_name: str) -> np.ndarray:
    signals = entry["signals"]
    field = signals["fields"].get(field_name)
    if field is None:
        raise KeyError(f"{entry['episode_uid']} missing signal field {field_name!r}")
    dtype = np.dtype(signals["dtype"])
    path = root / "shards" / str(signals["shard"])
    if not path.exists():
        raise FileNotFoundError(path)
    base = int(signals["offset_bytes"]) // dtype.itemsize
    start = base + int(field["offset_values"])
    shape = tuple(int(x) for x in field["shape"])
    count = int(np.prod(shape))
    mm = np.memmap(path, mode="r", dtype=dtype)
    end = start + count
    if end > int(mm.shape[0]):
        raise ValueError(f"{path} signal field {field_name!r} exceeds shard bounds")
    return np.asarray(mm[start:end].reshape(shape), dtype=np.float32)


def _sample_indices(length: int, max_points: int) -> np.ndarray:
    if length <= 0:
        return np.zeros((0,), dtype=np.int64)
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_points).round().astype(np.int64))


def _finite_points(points: np.ndarray) -> list[list[float]]:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {arr.shape}")
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask].astype(float).tolist()


def condition_trajectories(
    condition: np.ndarray,
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    values = condition[..., :-1]
    exists = condition[..., -1] > 0.5
    if values.shape[-1] < 3:
        raise ValueError(f"condition feature dim must be >=3, got {values.shape[-1]}")
    idx = _sample_indices(values.shape[0], max_points)
    trajectories: list[dict[str, Any]] = []
    for slot in range(values.shape[1]):
        active = exists[:, slot]
        slot_idx = idx[active[idx]]
        trajectories.append(
            {
                "slot": int(slot),
                "active_frames": int(active.sum()),
                "total_frames": int(values.shape[0]),
                "points": _finite_points(values[slot_idx, slot, :3]),
                "mask_preview": active[idx].astype(int).tolist(),
            }
        )
    return trajectories


def camera_trajectories(
    root: Path,
    entry: dict[str, Any],
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for view in entry["views"]:
        source = view.get("extrinsics_source")
        if not source:
            out.append(
                {
                    "view_id": int(view["view_id"]),
                    "name": str(view.get("name", view["view_id"])),
                    "available": False,
                    "reason": "extrinsics_source is null",
                    "points": [],
                    "rpy": [],
                }
            )
            continue
        try:
            arr = load_signal_field(root, entry, str(source))
        except KeyError:
            out.append(
                {
                    "view_id": int(view["view_id"]),
                    "name": str(view.get("name", view["view_id"])),
                    "available": False,
                    "reason": f"missing signal field {source}",
                    "points": [],
                    "rpy": [],
                }
            )
            continue
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"camera extrinsics {source!r} must be (T,>=3), got {arr.shape}")
        idx = _sample_indices(arr.shape[0], max_points)
        rpy = arr[idx, 3:6].astype(float).tolist() if arr.shape[1] >= 6 else []
        out.append(
            {
                "view_id": int(view["view_id"]),
                "name": str(view.get("name", view["view_id"])),
                "source": str(source),
                "available": True,
                "points": _finite_points(arr[idx, :3]),
                "rpy": rpy,
            }
        )
    return out


def select_entries(root: Path, split: str, samples: int, seed: int) -> list[dict[str, Any]]:
    entries = list(iter_index(root, split))
    if not entries:
        raise ValueError(f"{root}/{split}.index.jsonl has no entries")
    rng = np.random.default_rng(seed)
    order = np.arange(len(entries))
    rng.shuffle(order)
    return [entries[int(i)] for i in order[: min(samples, len(entries))]]


def scene_for_entry(
    *,
    root: Path,
    dataset_name: str,
    entry: dict[str, Any],
    condition_view: str,
    max_points: int,
) -> dict[str, Any]:
    condition = load_condition(root, entry, condition_view)
    return {
        "dataset": dataset_name,
        "root": str(root),
        "episode_uid": entry["episode_uid"],
        "episode_id": entry["episode_id"],
        "task": entry.get("task"),
        "text": entry.get("text", {}).get("primary", ""),
        "time": entry["time"],
        "views": entry["views"],
        "embodiment": entry["embodiment"],
        "domain": entry["domain"],
        "quality": entry["quality"],
        "source": entry["source"],
        "condition_view": condition_view,
        "eef": condition_trajectories(condition, max_points=max_points),
        "cameras": camera_trajectories(root, entry, max_points=max_points),
    }


def build_payload(
    *,
    datasets: list[tuple[str, Path]],
    split: str,
    samples: int,
    seed: int,
    condition_view: str,
    max_points: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "split": split,
        "samples": int(samples),
        "condition_view": condition_view,
        "max_points": int(max_points),
        "datasets": [],
    }
    for dataset_index, (name, root) in enumerate(datasets):
        manifest = read_manifest(root)
        entries = select_entries(root, split, samples, seed + dataset_index * 1009)
        payload["datasets"].append(
            {
                "name": name,
                "root": str(root),
                "manifest": {
                    "source_dataset": manifest["source_dataset"],
                    "processing": manifest["processing"],
                    "camera_schema": manifest["camera_schema"],
                },
                "scenes": [
                    scene_for_entry(
                        root=root,
                        dataset_name=name,
                        entry=entry,
                        condition_view=condition_view,
                        max_points=max_points,
                    )
                    for entry in entries
                ],
            }
        )
    return payload


def _json_script(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    text = text.replace("</", "<\\/")
    return f"<script id='scene-data' type='application/json'>{text}</script>"


def write_spatial_html(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    css = """
    body { font-family: system-ui, sans-serif; margin: 22px; background: #f5f6f8; color: #111; }
    h1 { margin: 0 0 4px; }
    .muted { color: #667; }
    .scene { background: white; border: 1px solid #d8dde6; border-radius: 8px; padding: 14px; margin: 16px 0; }
    .row { display: grid; grid-template-columns: minmax(520px, 1fr) 380px; gap: 14px; align-items: start; }
    canvas { width: 100%; height: 520px; background: #0b1020; border-radius: 6px; display: block; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; white-space: pre-wrap; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    td, th { border-top: 1px solid #e1e5ed; padding: 5px; text-align: left; vertical-align: top; }
    .legend span { display: inline-block; margin-right: 12px; }
    .chip { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
    @media (max-width: 920px) { .row { grid-template-columns: 1fr; } canvas { height: 420px; } }
    """
    js = r"""
    const payload = JSON.parse(document.getElementById("scene-data").textContent);
    const colors = {
      slot0: "#ff4d4d",
      slot1: "#4d9dff",
      cam0: "#38d67a",
      cam1: "#ffd34d",
      cam2: "#d15cff",
      axisX: "#e74c3c",
      axisY: "#2ecc71",
      axisZ: "#3498db"
    };

    function allPoints(scene) {
      const pts = [];
      for (const tr of scene.eef) for (const p of tr.points) pts.push(p);
      for (const cam of scene.cameras) for (const p of cam.points) pts.push(p);
      return pts;
    }

    function bounds(points) {
      if (!points.length) return {center:[0,0,0], scale:1};
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of points) for (let i=0; i<3; i++) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
      const center = [0,1,2].map(i => (lo[i] + hi[i]) / 2);
      const span = Math.max(hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2], 1e-3);
      return {center, scale: span};
    }

    function rotate(p, state) {
      let [x,y,z] = p;
      const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
      const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
      const x1 = cy*x + sy*z;
      const z1 = -sy*x + cy*z;
      const y1 = cp*y - sp*z1;
      const z2 = sp*y + cp*z1;
      return [x1, y1, z2];
    }

    function project(p, b, state, w, h) {
      const q = [(p[0]-b.center[0])/b.scale, (p[1]-b.center[1])/b.scale, (p[2]-b.center[2])/b.scale];
      const r = rotate(q, state);
      const zoom = Math.min(w, h) * 0.62 * state.zoom;
      return [w/2 + r[0]*zoom, h/2 - r[1]*zoom, r[2]];
    }

    function drawPolyline(ctx, pts, b, state, w, h, color, width=2) {
      if (pts.length < 1) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      for (let i=0; i<pts.length; i++) {
        const [x,y] = project(pts[i], b, state, w, h);
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.stroke();
      const first = project(pts[0], b, state, w, h);
      const last = project(pts[pts.length-1], b, state, w, h);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(first[0], first[1], 4, 0, Math.PI*2); ctx.fill();
      ctx.fillRect(last[0]-4, last[1]-4, 8, 8);
    }

    function drawAxes(ctx, b, state, w, h) {
      const origin = b.center;
      const len = b.scale * 0.25;
      const axes = [
        [[origin[0],origin[1],origin[2]], [origin[0]+len,origin[1],origin[2]], colors.axisX, "x"],
        [[origin[0],origin[1],origin[2]], [origin[0],origin[1]+len,origin[2]], colors.axisY, "y"],
        [[origin[0],origin[1],origin[2]], [origin[0],origin[1],origin[2]+len], colors.axisZ, "z"],
      ];
      ctx.font = "13px system-ui";
      for (const [a, c, color, label] of axes) {
        const pa = project(a, b, state, w, h), pc = project(c, b, state, w, h);
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pc[0], pc[1]); ctx.stroke();
        ctx.fillStyle = color; ctx.fillText(label, pc[0]+4, pc[1]+4);
      }
    }

    function drawScene(canvas, scene) {
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width*dpr));
      canvas.height = Math.max(1, Math.floor(rect.height*dpr));
      ctx.setTransform(dpr,0,0,dpr,0,0);
      const w = rect.width, h = rect.height;
      const state = canvas._state || {yaw: -0.72, pitch: 0.58, zoom: 1.0};
      canvas._state = state;
      const b = bounds(allPoints(scene));
      ctx.clearRect(0,0,w,h);
      drawAxes(ctx, b, state, w, h);
      for (const tr of scene.eef) drawPolyline(ctx, tr.points, b, state, w, h, tr.slot === 0 ? colors.slot0 : colors.slot1, 3);
      for (const cam of scene.cameras) {
        if (!cam.available) continue;
        drawPolyline(ctx, cam.points, b, state, w, h, colors["cam"+cam.view_id] || "#aaa", 2);
      }
      ctx.fillStyle = "#c9d4ff";
      ctx.font = "12px system-ui";
      ctx.fillText("drag: rotate | wheel: zoom | circle=start square=end", 12, h-14);
    }

    function attach(canvas, scene) {
      canvas._state = {yaw: -0.72, pitch: 0.58, zoom: 1.0};
      let dragging = false, last = [0,0];
      canvas.addEventListener("mousedown", e => { dragging = true; last = [e.clientX, e.clientY]; });
      window.addEventListener("mouseup", () => dragging = false);
      window.addEventListener("mousemove", e => {
        if (!dragging) return;
        const dx = e.clientX - last[0], dy = e.clientY - last[1];
        last = [e.clientX, e.clientY];
        canvas._state.yaw += dx * 0.008;
        canvas._state.pitch = Math.max(-1.45, Math.min(1.45, canvas._state.pitch + dy * 0.008));
        drawScene(canvas, scene);
      });
      canvas.addEventListener("wheel", e => {
        e.preventDefault();
        canvas._state.zoom *= Math.exp(-e.deltaY * 0.001);
        canvas._state.zoom = Math.max(0.2, Math.min(8, canvas._state.zoom));
        drawScene(canvas, scene);
      }, {passive:false});
      window.addEventListener("resize", () => drawScene(canvas, scene));
      drawScene(canvas, scene);
    }

    function escapeHtml(x) {
      return String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    function cameraRows(scene) {
      return scene.cameras.map(c => `<tr><td>${escapeHtml(c.name)}</td><td>${c.available ? "yes" : "no"}</td><td>${escapeHtml(c.source || c.reason || "")}</td></tr>`).join("");
    }

    function slotRows(scene) {
      return scene.eef.map(t => `<tr><td>slot ${t.slot}</td><td>${t.active_frames}/${t.total_frames}</td><td>${t.points.length}</td></tr>`).join("");
    }

    function render() {
      const root = document.getElementById("root");
      for (const ds of payload.datasets) {
        for (const scene of ds.scenes) {
          const id = `canvas-${root.children.length}`;
          const div = document.createElement("section");
          div.className = "scene";
          div.innerHTML = `
            <h2>${escapeHtml(ds.name)} | ${escapeHtml(scene.episode_uid)}</h2>
            <p><b>task</b> ${escapeHtml(scene.task || "")}</p>
            <p><b>text</b> ${escapeHtml(scene.text || "")}</p>
            <p><b>domain</b> <code>${escapeHtml(JSON.stringify(scene.domain))}</code></p>
            <div class="legend">
              <span><i class="chip" style="background:${colors.slot0}"></i>slot0</span>
              <span><i class="chip" style="background:${colors.slot1}"></i>slot1</span>
              <span><i class="chip" style="background:${colors.cam0}"></i>camera0</span>
              <span><i class="chip" style="background:${colors.cam1}"></i>camera1</span>
              <span><i class="chip" style="background:${colors.cam2}"></i>camera2</span>
            </div>
            <div class="row">
              <canvas id="${id}"></canvas>
              <div>
                <h3>EEF</h3>
                <table><thead><tr><th>slot</th><th>active</th><th>points</th></tr></thead><tbody>${slotRows(scene)}</tbody></table>
                <h3>Camera</h3>
                <table><thead><tr><th>view</th><th>pose</th><th>source/reason</th></tr></thead><tbody>${cameraRows(scene)}</tbody></table>
                <details><summary>metadata</summary><pre>${escapeHtml(JSON.stringify({time: scene.time, views: scene.views, embodiment: scene.embodiment, quality: scene.quality, source: scene.source}, null, 2))}</pre></details>
              </div>
            </div>`;
          root.appendChild(div);
          attach(document.getElementById(id), scene);
        }
      }
    }
    render();
    """
    doc = (
        "<!doctype html><meta charset='utf-8'>"
        "<title>CoachWorld Spatial Data Visualizer</title>"
        f"<style>{css}</style>"
        "<h1>CoachWorld Spatial Data Visualizer</h1>"
        "<p class='muted'>3D EEF trajectories from arm_slot_eef_pose; camera trajectories are shown only when extrinsics exist in signals.</p>"
        "<div id='root'></div>"
        f"{_json_script(payload)}"
        f"<script>{js}</script>"
    )
    out_path.write_text(doc, encoding="utf-8")


def write_summary(out_path: Path, payload: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
