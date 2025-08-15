import argparse
import glob
import os
import statistics
import time
from typing import List, Tuple


def _flatten_trimesh(mesh_or_scene):
    import trimesh

    if isinstance(mesh_or_scene, trimesh.Scene):
        combined = trimesh.Trimesh()
        for geom in mesh_or_scene.geometry.values():
            combined = trimesh.util.concatenate([combined, geom])
        return combined
    return mesh_or_scene


def old_loader(path: str) -> Tuple[int, int]:
    import trimesh

    loaded = trimesh.load(path)
    mesh = _flatten_trimesh(loaded)
    return int(mesh.vertices.shape[0]), int(mesh.faces.shape[0])


def new_loader(path: str) -> Tuple[int, int]:
    from .gltf_ops import load_gltf_or_glb, get_all_meshes_triangles

    doc = load_gltf_or_glb(path)
    verts, faces, _ = get_all_meshes_triangles(doc, transform_to_global=True)
    return int(verts.shape[0]), int(faces.shape[0])


def time_fn(fn, path: str, repeats: int, warmup: int) -> Tuple[float, List[float]]:
    # warmup
    for _ in range(max(0, warmup)):
        fn(path)

    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(path)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    mean = statistics.mean(times)
    return mean, times


def collect_paths(args) -> List[str]:
    if args.paths:
        return [p for p in args.paths if os.path.isfile(p)]
    pattern = os.path.join(args.dir, args.pattern)
    return sorted(glob.glob(pattern))


def main():
    parser = argparse.ArgumentParser(description="Benchmark GLB loading: trimesh vs custom glTF loader")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=str, help="Directory with GLB/GLTF files")
    group.add_argument("--paths", nargs="*", help="Explicit list of GLB/GLTF files")
    parser.add_argument("--pattern", type=str, default="*.glb", help="Glob pattern when using --dir (default: *.glb)")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed runs per file (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per file (default: 1)")
    args = parser.parse_args()

    paths = collect_paths(args)
    if not paths:
        print("No files found to benchmark.")
        return

    print(f"Benchmarking {len(paths)} file(s), repeats={args.repeats}, warmup={args.warmup}\n")

    rows = []
    for path in paths:
        try:
            # Validate both pipelines return some geometry
            old_v, old_f = old_loader(path)
            new_v, new_f = new_loader(path)
        except Exception as e:
            print(f"[SKIP] {path}: failed to load in validation step: {e}")
            continue

        old_mean, old_times = time_fn(old_loader, path, repeats=args.repeats, warmup=args.warmup)
        new_mean, new_times = time_fn(new_loader, path, repeats=args.repeats, warmup=args.warmup)

        speedup = old_mean / new_mean if new_mean > 0 else float('inf')

        print(f"{os.path.basename(path)}")
        print(f"  old(trimesh): {old_mean:.2f} ms  (v={old_v}, f={old_f})")
        print(f"  new(custom) : {new_mean:.2f} ms  (v={new_v}, f={new_f})  speedup={speedup:.2f}x\n")
        rows.append((old_mean, new_mean))

    if rows:
        old_avg = statistics.mean(r[0] for r in rows)
        new_avg = statistics.mean(r[1] for r in rows)
        print("Overall:")
        print(f"  old avg: {old_avg:.2f} ms")
        print(f"  new avg: {new_avg:.2f} ms")
        if new_avg > 0:
            print(f"  speedup: {old_avg / new_avg:.2f}x")


if __name__ == "__main__":
    main()

