"""Object-patch coverage per image, to size the object-masking ratio.

For each 200x200 mask: area-pool to a 12x12 grid (= 16x16 patches of the
image resized to 192, what a ViT/16 sees); a patch is "object" if >=25% of
its pixels are in the mask. n_obj = #object patches, coverage = n_obj / 144.

Then applies the decision rule:
    ratio = max(0.20, TARGET / median(n_obj))
    if p5(n_obj) < 40: n_masked = max(ratio * n_obj, FLOOR)

Usage: python scripts/measure_object_coverage.py [--crop | --zoom MARGIN]
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data/to_modules/new_split_parasito")
DATASETS = ["helminth-eggs", "helminth-larvae", "protozoan-cysts"]
GRID, PATCH, THR = 12, 16, 0.25
TARGET, BASE_RATIO, FLOOR = 12, 0.20, 8


def zoom_to_object(m: torch.Tensor, margin: float) -> torch.Tensor:
    """Square crop around the mask bbox (side = max bbox side * (1 + 2*margin)),
    zero-padded past the border, resized to GRID*PATCH."""
    ys, xs = torch.nonzero(m, as_tuple=True)
    if len(ys) == 0:
        return F.interpolate(m[None, None], size=GRID * PATCH, mode="area")[0, 0]
    cy, cx = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
    side = int(np.ceil(max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1 + 2 * margin)))
    y0, x0 = int(round(float(cy) - side / 2)), int(round(float(cx) - side / 2))
    pad = side
    mp = F.pad(m, (pad, pad, pad, pad))
    c = mp[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    return F.interpolate(c[None, None], size=GRID * PATCH, mode="area")[0, 0]


def n_obj_of(path: str, crop: bool, zoom: float | None = None) -> int:
    m = torch.from_numpy((np.array(Image.open(path).convert("L")) > 0).astype(np.float32))
    if zoom is not None:
        m = zoom_to_object(m, zoom)
    elif crop:  # center crop 192 -> exact 16x16 patches
        o = (m.shape[0] - GRID * PATCH) // 2
        m = m[o:o + GRID * PATCH, o:o + GRID * PATCH]
    frac = F.adaptive_avg_pool2d(m[None, None], GRID)[0, 0]
    return int((frac >= THR).sum())


def report(name: str, n: np.ndarray, out: list[str]) -> None:
    cov = n / GRID**2
    p5, med = np.percentile(n, 5), np.median(n)
    ratio = max(BASE_RATIO, TARGET / med) if med > 0 else float("nan")
    masked = ratio * n
    if p5 < 40:
        masked = np.maximum(masked, FLOOR)
    masked = np.minimum(np.round(masked), n)  # can't mask more than exists
    hist, edges = np.histogram(n, bins=[0, 10, 20, 30, 40, 50, 60, 72, 90, 110, 145])
    out += [
        f"\n## {name}  (N={len(n)})",
        f"coverage: mean={cov.mean():.3f} median={np.median(cov):.3f} "
        f"p5={np.percentile(cov, 5):.3f} p95={np.percentile(cov, 95):.3f}",
        f"n_obj:    mean={n.mean():.1f} median={med:.0f} p5={p5:.0f} p95={np.percentile(n, 95):.0f} "
        f"min={n.min()} max={n.max()}",
        f"n_obj < 20: {(n < 20).sum()} ({(n < 20).mean():.1%})   n_obj == 0: {(n == 0).sum()}",
        "histogram n_obj:",
        *[f"  [{a:>3},{b:>3}) {c:>6}  {'#' * int(60 * c / max(hist.max(), 1))}"
          for a, b, c in zip(edges[:-1], edges[1:], hist)],
        f"decision: ratio_objeto = max(0.20, 12/{med:.0f}) = {ratio:.3f}"
        + (f"; p5={p5:.0f} < 40 -> floor {FLOOR} patches" if p5 < 40 else "; p5>=40 -> no floor"),
        f"  masked patches: median={np.median(masked):.0f} p5={np.percentile(masked, 5):.0f}; "
        f">=12: {(masked >= TARGET).mean():.1%}; n_obj<{FLOOR} (floor unreachable): {(n < FLOOR).sum()}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", action="store_true", help="center-crop 192 instead of area-pooling 200")
    ap.add_argument("--zoom", type=float, default=None,
                    help="square crop around the object bbox with this margin per side (0.1 = 10%%), resized to 192")
    args = ap.parse_args()
    mode = (f"zoom to object bbox, margin {args.zoom:.0%}/side, resize 192" if args.zoom is not None
            else "center-crop 192" if args.crop else "area-pool 200->12x12")
    out = [f"# Object coverage ({mode}, thr={THR})"]
    alln = []
    for ds in DATASETS:
        paths = sorted(glob.glob(os.path.join(ROOT, ds, "masks", "*.png")))
        n = np.array([n_obj_of(p, args.crop, args.zoom) for p in paths])
        labels = np.array([int(os.path.basename(p).split("_")[0]) for p in paths])
        report(ds, n, out)
        out.append("  per class (median n_obj / #<20 / N): " + ", ".join(
            f"c{c}: {np.median(n[labels == c]):.0f}/{(n[labels == c] < 20).sum()}/{(labels == c).sum()}"
            for c in np.unique(labels)))
        alln.append(n)
    report("ALL", np.concatenate(alln), out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
