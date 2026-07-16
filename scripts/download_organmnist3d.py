# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""
Download OrganMNIST3D and convert to 2D PNG slices for this pipeline.

Each 3D volume  (D × H × W, grayscale uint8) is sliced along all three axes
so the pipeline sees ordinary RGB images.  Labels come from the medmnist
metadata and map to human-readable organ names.

Output layout (ImageFolder-compatible, matches LejepaDataset):

    data/organmnist3d/
        bladder/       000000_ax10.png  000001_ax14.png  ...
        femur_l/       ...
        femur_r/       ...
        heart/         ...
        kidney_l/      ...
        kidney_r/      ...
        liver/         ...
        lung_l/        ...
        lung_r/        ...
        pancreas/      ...
        spleen/        ...

Usage:
    python scripts/download_organmnist3d.py
    python scripts/download_organmnist3d.py --out data/organmnist3d --size 64 --axis axial
    python scripts/download_organmnist3d.py --slices middle   # only the centre slice per axis

Axes extracted (default: all three, tripling the dataset size):
    axial     → slice along depth (axis 0)
    coronal   → slice along height (axis 1)
    sagittal  → slice along width  (axis 2)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# ── class names (MedMNIST v2 label order for OrganMNIST3D) ──────────────────
ORGAN_NAMES = [
    "bladder",
    "femur_l",
    "femur_r",
    "heart",
    "kidney_l",
    "kidney_r",
    "liver",
    "lung_l",
    "lung_r",
    "pancreas",
    "spleen",
]


def _to_rgb(arr2d: np.ndarray) -> Image.Image:
    """Convert a 2-D uint8 grayscale array to a 3-channel PIL image."""
    rgb = np.stack([arr2d, arr2d, arr2d], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _slice_volume(
    volume: np.ndarray,
    axes: list[int],
    slices: str,
) -> list[tuple[str, np.ndarray]]:
    """
    Return a list of (tag, 2D array) for the requested axes / slice strategy.

    Args:
        volume:  Shape (D, H, W), numpy uint8.
        axes:    Which axes to slice (0=axial, 1=coronal, 2=sagittal).
        slices:  'all' → every slice; 'middle' → only the centre one.
    """
    axis_tag = {0: "ax", 1: "co", 2: "sa"}
    results: list[tuple[str, np.ndarray]] = []

    for ax in axes:
        depth = volume.shape[ax]
        if slices == "middle":
            indices = [depth // 2]
        else:          # all
            indices = list(range(depth))

        for i in indices:
            sl = np.take(volume, i, axis=ax)  # shape (H, W) or (D, W) or (D, H)
            results.append((f"{axis_tag[ax]}{i:02d}", sl))

    return results


def download_and_export(
    out_dir: str | Path,
    size: int,
    axes: list[int],
    slices: str,
    splits: list[str],
) -> None:
    try:
        import medmnist
        from medmnist import OrganMNIST3D
    except ImportError:
        raise SystemExit(
            "medmnist is not installed.  Run:  pip install medmnist"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create class subdirs
    for name in ORGAN_NAMES:
        (out_dir / name).mkdir(exist_ok=True)

    total_saved = 0

    for split in splits:
        print(f"\n=== split: {split} ===")
        ds = OrganMNIST3D(split=split, download=True, size=size)

        # medmnist datasets expose .imgs (N, D, H, W) and .labels (N, 1)
        imgs: np.ndarray = ds.imgs        # uint8, 0-255
        labels: np.ndarray = ds.labels.flatten().astype(int)

        print(f"  volumes: {len(imgs)}  shape: {imgs.shape[1:]}")

        for vol_idx, (volume, label) in enumerate(zip(imgs, labels)):
            organ = ORGAN_NAMES[label]
            tag_base = f"{split}_{vol_idx:06d}"

            for slice_tag, arr in _slice_volume(volume, axes, slices):
                fname = f"{tag_base}_{slice_tag}.png"
                fpath = out_dir / organ / fname
                if not fpath.exists():
                    _to_rgb(arr).save(fpath)
                    total_saved += 1

            if (vol_idx + 1) % 200 == 0:
                print(f"  processed {vol_idx + 1}/{len(imgs)} volumes …")

        print(f"  done — running total saved: {total_saved}")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Output dir  : {out_dir.resolve()}")
    print(f"Total images: {total_saved}")
    print("\nClass breakdown:")
    for name in ORGAN_NAMES:
        n = len(list((out_dir / name).glob("*.png")))
        print(f"  {name:<12} {n:>6} images")
    print(f"\nSet  data_dir: data/organmnist3d  in your config to use this dataset.")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OrganMNIST3D and export as 2D PNG slices."
    )
    parser.add_argument(
        "--out", default="data/organmnist3d",
        help="Output root directory (default: data/organmnist3d)",
    )
    parser.add_argument(
        "--size", type=int, default=28, choices=[28, 64],
        help="Volume resolution: 28 (default) or 64 (MedMNIST+)",
    )
    parser.add_argument(
        "--axis", default="all",
        choices=["all", "axial", "coronal", "sagittal"],
        help="Which axis to slice.  'all' extracts axial+coronal+sagittal.",
    )
    parser.add_argument(
        "--slices", default="all", choices=["all", "middle"],
        help="'all' → every slice per axis (default);  'middle' → centre only.",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which MedMNIST splits to include (default: all three).",
    )
    args = parser.parse_args()

    axis_map = {
        "all":      [0, 1, 2],
        "axial":    [0],
        "coronal":  [1],
        "sagittal": [2],
    }

    download_and_export(
        out_dir=args.out,
        size=args.size,
        axes=axis_map[args.axis],
        slices=args.slices,
        splits=args.splits,
    )


if __name__ == "__main__":
    main()
