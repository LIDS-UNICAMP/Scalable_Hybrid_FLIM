# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""_docker_pyift_worker.py — pyift image processor that runs INSIDE the Docker container.

This script is NOT meant to be executed on the host.
It is copied into the Docker image and invoked by PyiftLoader when the fallback
path is triggered.

Protocol
--------
Input  : /job/paths.json   — JSON array of absolute image paths inside /workspace
Output : /job/results.npz  — NumPy archive, keys "arr_0", "arr_1", ..., "arr_N-1"
                              Each array is float32 (H, W, 3) LABNorm2 in [0, 1].
Exit   : 0 on success, 1 on any error (stderr has details).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    job_dir = Path("/job")
    paths_file   = job_dir / "paths.json"
    results_file = job_dir / "results.npz"

    # ── Read input ────────────────────────────────────────────────────────────
    if not paths_file.is_file():
        print(f"[worker] ERROR: paths file not found: {paths_file}", file=sys.stderr)
        sys.exit(1)

    with open(paths_file, "r") as f:
        paths: list[str] = json.load(f)

    if not paths:
        print("[worker] ERROR: paths list is empty", file=sys.stderr)
        sys.exit(1)

    print(f"[worker] Received {len(paths)} paths", flush=True)

    # ── Import pyift ──────────────────────────────────────────────────────────
    try:
        import pyift.pyift as ift
    except ImportError as exc:
        print(f"[worker] ERROR: cannot import pyift.pyift — {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Process images ────────────────────────────────────────────────────────
    arrays: dict[str, np.ndarray] = {}

    for i, path_str in enumerate(paths):
        path = Path(path_str)
        if not path.is_file():
            print(f"[worker] ERROR: image not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            image  = ift.ReadImageByExt(str(path))
            mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
            arr    = mimage.AsNumPy().squeeze()   # (H, W, 3), float in [0, 1]
            arrays[f"arr_{i}"] = arr.astype(np.float32)
        except Exception as exc:
            print(f"[worker] ERROR processing {path}: {exc}", file=sys.stderr)
            sys.exit(1)

        if (i + 1) % 100 == 0:
            print(f"[worker] Processed {i + 1}/{len(paths)}", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    np.savez_compressed(str(results_file), **arrays)
    print(f"[worker] Saved {len(arrays)} arrays → {results_file}", flush=True)


if __name__ == "__main__":
    main()
