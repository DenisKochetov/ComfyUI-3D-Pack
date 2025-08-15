import os
import time
from typing import Tuple

import numpy as np
import trimesh

from mesh_processor.gltf_ops import load_gltf_or_glb, get_all_meshes_triangles


def _flatten_trimesh(mesh_or_scene):
    if isinstance(mesh_or_scene, trimesh.Scene):
        combined = trimesh.Trimesh()
        for geom in mesh_or_scene.geometry.values():
            combined = trimesh.util.concatenate([combined, geom])
        return combined
    return mesh_or_scene


def _load_old_glb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(path)
    mesh = _flatten_trimesh(loaded)
    return mesh.vertices, mesh.faces


def _load_new_glb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    doc = load_gltf_or_glb(path)
    verts, faces, _ = get_all_meshes_triangles(doc, transform_to_global=True)
    return verts, faces


def maybe_bench_glb_loaders(path: str) -> None:
    """Benchmark old (trimesh) vs new (custom) GLB loader when env HY3D_BENCH_GLB=1."""
    if os.getenv("HY3D_BENCH_GLB", "0") != "1":
        return

    print(f"[HY3D BENCH] Benchmark enabled for {os.path.basename(path)} (HY3D_BENCH_GLB=1)")

    # warmup
    for _ in range(1):
        _load_old_glb(path)
        _load_new_glb(path)

    def time_once(fn):
        t0 = time.perf_counter()
        v, f = fn(path)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0, v.shape[0], f.shape[0]

    old_ms, old_v, old_f = time_once(_load_old_glb)
    new_ms, new_v, new_f = time_once(_load_new_glb)

    print(f"[HY3D BENCH] {os.path.basename(path)}")
    print(f"  old(trimesh): {old_ms:.2f} ms  (v={old_v}, f={old_f})")
    print(f"  new(custom) : {new_ms:.2f} ms  (v={new_v}, f={new_f})  speedup={(old_ms/new_ms) if new_ms>0 else float('inf'):.2f}x")


def load_glb_mesh(path: str) -> trimesh.Trimesh:
    """Load GLB/GLTF using selected loader (env HY3D_GLB_LOADER: new|old), and benchmark optionally."""
    maybe_bench_glb_loaders(path)
    raw_choice = os.getenv("HY3D_GLB_LOADER", "new")
    which = (raw_choice or "new").lower()
    reason = f"HY3D_GLB_LOADER={raw_choice!r}" if raw_choice is not None else "default"
    if which not in ("new", "old"):
        print(f"[HY3D LOADER] Invalid loader '{which}', defaulting to 'new' (reason: {reason})")
        which = "new"

    print(
        f"[HY3D LOADER] Using '{which}' loader for {os.path.basename(path)} (reason: {reason})"
        + ("; benchmark enabled" if os.getenv("HY3D_BENCH_GLB", "0") == "1" else "")
    )

    if which == "old":
        verts, faces = _load_old_glb(path)
    else:
        verts, faces = _load_new_glb(path)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

