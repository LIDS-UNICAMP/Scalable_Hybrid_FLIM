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
"""
Object-centred crops of LAB[0, 1] images, guided by the segmentation mask (Especificação 2).

  base crop     deterministic, precomputed once and cached on disk: object bbox + ``margin``
                per side, square by the larger side, centred on the object, resized to
                ``image_size`` (image bilinear, mask nearest). The window never upsamples more
                than ``max_upscale``: if the square is too small the context grows instead.
  elongated     bbox ratio r = long/short > ``elongated.ratio``: "pad" keeps the aspect (long
                side -> S, short side -> S/r rounded up to a multiple of ``patch``, the rest
                filled by reflection) or "pca" rotates the object to its principal axis first.
                ``strategy: auto`` keeps the one with the larger median n_obj on the train split.
  augmentation  per sample (train only): margin jitter (p) or a RandomResizedCrop on the base
                crop that keeps the whole object and enough object patches, then flips, rot90
                and a free rotation. The mask follows every transform (nearest).

No black anywhere: every border the image opens is filled by reflection. The mask is 0 outside
the real image content (mirrored copies of the object are not the object), so n_obj only
counts real object patches. Crops are cut from the ORIGINAL resolution image, never from an
already resized one. n_obj = number of patches of the floor grid with >= ``object_threshold``
of their pixels on the object (Especificação 1 §4, ``object_patches``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from src.losses.masked_object_mse import object_patches

logger = logging.getLogger(__name__)

DEFAULT_CROP = {
    "enabled": True,
    "margin": 0.10,
    "min_bbox": 32,
    "max_upscale": 2.0,
    "patch": 16,
    "object_threshold": 0.25,
    "elongated": {"ratio": 2.0, "strategy": "auto"},
    "rrc": {"scale": [0.70, 1.00], "ratio": [0.85, 1.18], "min_keep": 0.60, "min_obj": 12, "max_tries": 10},
    "margin_jitter": {"p": 0.30, "range": [0.05, 0.25]},
    "hflip": 0.5,
    "vflip": 0.5,
    "rot90": 0.5,
    "rotation": {"p": 0.3, "deg": 15},
    "cache_dir": "artifacts/object_crops",
}


def _uniform(low: float, high: float) -> float:
    return float(torch.empty(1).uniform_(low, high))


def _rand() -> float:
    return float(torch.rand(1))


# ── loading / geometry of the original ───────────────────────────────────────

def load_mask(path: str) -> np.ndarray:
    m = np.array(Image.open(path))
    return (m[..., 0] if m.ndim == 3 else m) > 0


def load_lab(path: str) -> np.ndarray:
    from src.data_modules.datasets.dataset import ift_lab_loader  # pyift only where it is needed
    return np.ascontiguousarray(ift_lab_loader(path), dtype=np.float32)


def bbox_stats(mask: np.ndarray) -> Tuple[int, float]:
    """(larger bbox side in px, r = long/short); (0, 1) for an empty mask."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0, 1.0
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    return int(max(h, w)), max(h, w) / min(h, w)


def _principal_angle(pts: np.ndarray) -> float:
    vals, vecs = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
    v = vecs[:, np.argmax(vals)]
    return math.atan2(v[1], v[0])


def n_obj(mask: np.ndarray, patch: int, threshold: float) -> int:
    return int(object_patches(torch.from_numpy(mask)[None, None].float(), patch, threshold).sum())


def n_obj_batch(masks: List[np.ndarray], patch: int, threshold: float) -> List[int]:
    """One ``object_patches`` call for many masks (a tiny op per image burns CPU on torch threads)."""
    if not masks:
        return []
    obj = object_patches(torch.from_numpy(np.stack(masks))[:, None].float(), patch, threshold)
    return obj.flatten(1).sum(1).tolist()


def base_crop(img: Optional[np.ndarray], mask: np.ndarray, margin: float, mode: str, size: int,
              max_upscale: float, patch: int = 16):
    """
    Crop ``img`` [H,W,C] float32 (any C; None = mask only) around the object of ``mask`` [H,W] bool.

    ``mode``: square | pad | pca | whole. Returns (image [S,S,C] or None, mask [S,S] uint8,
    side of the window in original pixels, so the upscale is ``size / side``).
    """
    m8 = mask.astype(np.uint8)
    if mode == "whole":
        out = None if img is None else cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        return out, cv2.resize(m8, (size, size), interpolation=cv2.INTER_NEAREST), float(max(mask.shape))

    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    cen = pts.mean(0)
    R = np.eye(2)
    if mode == "pca":                       # principal axis -> horizontal
        a = -_principal_angle(pts)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = (pts - cen) @ R.T
    lo, hi = q.min(0) - 0.5, q.max(0) + 0.5   # pixel edges of the object in the frame
    ext, qc = hi - lo, (lo + hi) / 2

    side = max(ext.max() * (1 + 2 * margin), size / max_upscale)
    k = size / side
    short = size
    if mode == "pad":                       # ceil, so the rounding never cuts the margin
        short = min(size, patch * math.ceil(k * ext.min() * (1 + 2 * margin) / patch))
    out_w, out_h = (size, short) if ext[0] >= ext[1] else (short, size)

    A = k * R
    t = np.array([(out_w - 1) / 2, (out_h - 1) / 2]) - k * qc - A @ cen
    M = np.hstack([A, t[:, None]]).astype(np.float32)
    mo = cv2.warpAffine(m8, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = None if img is None else cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_REFLECT_101)
    if (out_w, out_h) != (size, size):      # aspect kept: fill the rest by reflection
        ph, pw = size - out_h, size - out_w
        pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
        mo = np.pad(mo, pads)
        if out is not None:
            out = np.pad(out, pads + ((0, 0),), mode="reflect")
    return out, mo, float(side)


# ── per-sample augmentation (on the base crop) ───────────────────────────────

def random_resized_crop(img: np.ndarray, mask: np.ndarray, side: float, n_base: int, cfg: dict):
    """RandomResizedCrop that keeps the whole object and enough object patches; base crop on failure."""
    c, size = cfg["rrc"], mask.shape[0]
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return img, mask
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    # upscale vs. the original = (size / side) * (size / window side) <= max_upscale
    min_side = size * (size / side) / cfg["max_upscale"] - 1e-6
    log_r = (math.log(c["ratio"][0]), math.log(c["ratio"][1]))
    for _ in range(c["max_tries"]):
        s, r = _uniform(*c["scale"]), math.exp(_uniform(*log_r))
        w, h = size * math.sqrt(s * r), size * math.sqrt(s / r)
        if w > size or h > size or min(w, h) < min_side:
            continue
        x_lo, x_hi = max(0.0, bx1 - w), min(float(bx0), size - w)   # object bbox inside the window
        y_lo, y_hi = max(0.0, by1 - h), min(float(by0), size - h)
        if x_lo > x_hi or y_lo > y_hi:
            continue
        x0, y0 = _uniform(x_lo, x_hi), _uniform(y_lo, y_hi)
        kx, ky = size / w, size / h
        M = np.float32([[kx, 0, kx * (0.5 - x0) - 0.5], [0, ky, ky * (0.5 - y0) - 0.5]])
        mo = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        n = n_obj(mo, cfg["patch"], cfg["object_threshold"])
        if n >= c["min_keep"] * n_base and n >= c["min_obj"]:
            out = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            return out, mo
    return img, mask


def geometry(img: np.ndarray, mask: np.ndarray, cfg: dict):
    """hflip, vflip, rot90 and a free rotation (reflection fill), skipped if it would cut the object."""
    if _rand() < cfg["hflip"]:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if _rand() < cfg["vflip"]:
        img, mask = img[::-1], mask[::-1]
    if _rand() < cfg["rot90"]:
        k = int(torch.randint(1, 4, (1,)))
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

    rot = cfg["rotation"]
    if _rand() < rot["p"]:
        size = mask.shape[0]
        M = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), _uniform(-rot["deg"], rot["deg"]), 1.0)
        pts = np.argwhere(mask)[:, ::-1].astype(np.int32)
        if len(pts):
            hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
            moved = hull @ M[:, :2].T + M[:, 2]
            if moved.min() < -0.5 or moved.max() > size - 0.5:
                return img, mask
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


# ── dataset ──────────────────────────────────────────────────────────────────

class ObjectCropDataset(Dataset):
    """
    ``(views [1,3,S,S] LAB [0,1], label, mask [1,S,S] float {0,1})``.

    ``entries[i]`` = {cache, mode, side, n_obj}; ``augment`` turns on the per-sample Spec 2
    augmentation (train), otherwise the cached base crop is returned as is.
    """

    def __init__(self, samples: List[str], labels: List[int], entries: List[dict], masks_dir: str,
                 cfg: dict, image_size: int, augment: bool) -> None:
        self.samples, self.labels, self.entries = samples, labels, entries
        self.masks_dir, self.cfg, self.image_size, self.augment = masks_dir, cfg, image_size, augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int, Tensor]:
        e, cfg = self.entries[idx], self.cfg
        jit = cfg["margin_jitter"]
        if self.augment and e["mode"] != "whole" and _rand() < jit["p"]:
            path = self.samples[idx]
            mask = load_mask(os.path.join(self.masks_dir, os.path.basename(path)))
            img, m, _ = base_crop(load_lab(path), mask, _uniform(*jit["range"]), e["mode"],
                                  self.image_size, cfg["max_upscale"], cfg["patch"])
        else:
            with np.load(e["cache"]) as d:
                img, m = d["image"].astype(np.float32), d["mask"]
            if self.augment and e["mode"] != "whole":
                img, m = random_resized_crop(img, m, e["side"], e["n_obj"], cfg)
        if self.augment:
            img, m = geometry(img, m, cfg)
        views = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).clamp(0.0, 1.0)[None]
        return views, self.labels[idx], torch.from_numpy(np.ascontiguousarray(m)).float()[None]


# ── precompute + wiring ──────────────────────────────────────────────────────

def _params_hash(cfg: dict, image_size: int) -> str:
    key = {"S": image_size, "margin": cfg["margin"], "max_upscale": cfg["max_upscale"],
           "patch": cfg["patch"], "thr": cfg["object_threshold"], "ratio": cfg["elongated"]["ratio"]}
    return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def _label(path: str) -> int:
    return int(os.path.basename(path).split("_")[0]) - 1


def resolve_strategy(train_paths: List[str], masks_dir: str, cfg: dict, image_size: int,
                     out_json: str) -> dict:
    """pad vs pca on the elongated train images: the larger median n_obj wins (``auto``)."""
    el = cfg["elongated"]
    if os.path.isfile(out_json):
        with open(out_json) as f:
            return json.load(f)
    crops = {"pad": [], "pca": []}
    for p in train_paths:
        mask = load_mask(os.path.join(masks_dir, os.path.basename(p)))
        side, r = bbox_stats(mask)
        if side < cfg["min_bbox"] or r <= el["ratio"]:
            continue
        for mode in crops:
            crops[mode].append(base_crop(None, mask, cfg["margin"], mode, image_size,
                                         cfg["max_upscale"], cfg["patch"])[1])
    counts = {k: n_obj_batch(v, cfg["patch"], cfg["object_threshold"]) for k, v in crops.items()}
    med = {k: (float(np.median(v)) if v else None) for k, v in counts.items()}
    choice = "pad" if med["pca"] is None or med["pad"] >= med["pca"] else "pca"
    info = {"strategy": choice, "median_n_obj": med, "n_elongated": len(counts["pad"])}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(info, f, indent=2)
    return info


def _precompute(paths: List[str], masks_dir: str, cfg: dict, image_size: int, root: str,
                strategy: str, keep_small: bool) -> Tuple[List[str], List[dict]]:
    """Cache the base crop of every image; returns (kept paths, entries). Small bboxes: whole image or dropped."""
    kept, entries, pending = [], [], []

    def flush() -> None:  # n_obj of the new crops in one call, then write them
        for (e, img, m, extra), n in zip(pending, n_obj_batch([x[2] for x in pending], cfg["patch"],
                                                               cfg["object_threshold"])):
            e["n_obj"] = n
            np.savez(e["cache"], image=img.astype(np.float16), mask=m, side=e["side"], n_obj=n, **extra)
        pending.clear()

    for p in paths:
        mask = load_mask(os.path.join(masks_dir, os.path.basename(p)))
        side, r = bbox_stats(mask)
        if side < cfg["min_bbox"]:
            if not keep_small:
                continue
            mode = "whole"
        else:
            mode = strategy if r > cfg["elongated"]["ratio"] else "square"
        cache = os.path.join(root, f"{os.path.splitext(os.path.basename(p))[0]}_{mode}.npz")
        e = {"cache": cache, "mode": mode}
        if os.path.isfile(cache):
            with np.load(cache) as d:
                e.update(side=float(d["side"]), n_obj=int(d["n_obj"]))
        else:
            img, m, win = base_crop(load_lab(p), mask, cfg["margin"], mode, image_size,
                                    cfg["max_upscale"], cfg["patch"])
            e["side"] = win
            pending.append((e, img, m, {"scale": win / image_size, "bbox_side": side, "r": r, "mode": mode}))
            if len(pending) == 256:
                flush()
        kept.append(p)
        entries.append(e)
    flush()
    return kept, entries


def attach_object_crops(dm, dataset: str, masks_dir: str, crop_cfg: dict, image_size: int) -> None:
    """
    After ``dm.setup()``: replace ``dm.ds_train/ds_val/ds_test`` by object-crop datasets.

    train gets the per-sample augmentation, val/test the cached base crop; ``dm.ds_train_plain``
    is train with the base crop only. Images with bbox < ``min_bbox`` are dropped from train
    and kept in val/test as the whole image resized.
    """
    cfg = {**DEFAULT_CROP, **(crop_cfg or {})}
    root = os.path.join(cfg["cache_dir"], dataset, _params_hash(cfg, image_size))
    os.makedirs(root, exist_ok=True)
    t0 = time.time()

    train_paths = list(dm.ds_train.base.samples)
    strategy = cfg["elongated"]["strategy"]
    if strategy == "auto":
        info = resolve_strategy(train_paths, masks_dir, cfg, image_size,
                                os.path.join(root, f"strategy_split{dm.split}_pct{dm.percentage}.json"))
        strategy = info["strategy"]
        logger.info("[object_crop] %s split%s pct%s elongated=%d median n_obj pad=%s pca=%s -> %s",
                    dataset, dm.split, dm.percentage, info["n_elongated"],
                    info["median_n_obj"]["pad"], info["median_n_obj"]["pca"], strategy)

    built = {}
    for name in ("train", "val", "test"):
        paths = list(getattr(dm, f"ds_{name}").base.samples)
        kept, entries = _precompute(paths, masks_dir, cfg, image_size, root, strategy,
                                    keep_small=name != "train")
        built[name] = (kept, [_label(p) for p in kept], entries)
        logger.info("[object_crop] %s %s: %d images (%d dropped)", dataset, name, len(kept),
                    len(paths) - len(kept))
    logger.info("[object_crop] base crops ready in %.1fs (cache %s)", time.time() - t0, root)

    def make(name: str, augment: bool) -> ObjectCropDataset:
        return ObjectCropDataset(*built[name], masks_dir, cfg, image_size, augment)

    dm.ds_train, dm.ds_train_plain = make("train", True), make("train", False)
    dm.ds_val, dm.ds_test = make("val", False), make("test", False)
    # trainer.fit(datamodule=dm) calls dm.setup() again, which would rebuild the uncropped
    # datasets over these (E1/E3 crash on the missing mask, E2 would silently train uncropped).
    dm.setup = lambda stage=None: None


# ── preview (Especificação 2 §5) ─────────────────────────────────────────────

def _tile(img_rgb: np.ndarray, mask: np.ndarray, cfg: dict, title: str) -> Image.Image:
    """RGB [S,S,3] in [0,1] with the mask tinted red, the floor patch grid and object patches in green."""
    from PIL import ImageDraw

    rgb = (np.clip(img_rgb, 0, 1) * 255).astype(np.float32)
    rgb[mask > 0] = 0.65 * rgb[mask > 0] + 0.35 * np.array([255, 0, 0])
    im = Image.fromarray(rgb.astype(np.uint8))
    d = ImageDraw.Draw(im)
    p, size = cfg["patch"], mask.shape[0]
    obj = object_patches(torch.from_numpy(np.ascontiguousarray(mask))[None, None].float(), p,
                         cfg["object_threshold"])[0].numpy()
    g = size // p
    for i in range(g + 1):
        d.line([(i * p, 0), (i * p, g * p)], fill=(90, 90, 90))
        d.line([(0, i * p), (g * p, i * p)], fill=(90, 90, 90))
    for y, x in zip(*np.nonzero(obj)):
        d.rectangle([x * p, y * p, x * p + p, y * p + p], outline=(0, 255, 0))
    canvas = Image.new("RGB", (size, size + 14), (255, 255, 255))
    canvas.paste(im, (0, 14))
    ImageDraw.Draw(canvas).text((2, 1), f"{title} n_obj={int(obj.sum())}", fill=(0, 0, 0))
    return canvas


def _grid(tiles: List[Image.Image], cols: int) -> Image.Image:
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * w, (i // cols) * h))
    return out


def _preview(datasets: List[str], split: int, pct: int, out_dir: str, image_size: int) -> None:
    from config import get_single_parasite_paths
    from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited
    from src.modules.autoencoder_flim_module import _dataset_short_to_parasite_name

    os.makedirs(out_dir, exist_ok=True)
    cfg = DEFAULT_CROP
    torch.manual_seed(0)
    for ds in datasets:
        parasite = _dataset_short_to_parasite_name(ds)
        masks_dir = get_single_parasite_paths(parasite, split, pct)[0]["masks_dir"]
        dm = ParasiteLejepaDataModuleSplited(parasite_name=parasite, split=split, percentage=pct,
                                             image_size=image_size, V_train=1, V_eval=1,
                                             num_workers=0, loader="ift_lab", imagenet_norm=False)
        dm.setup()
        n_train0 = len(dm.ds_train.base)
        t0 = time.time()
        attach_object_crops(dm, ds, masks_dir, cfg, image_size)
        print(f"[{ds}] attach {time.time() - t0:.1f}s | train {len(dm.ds_train)}/{n_train0} "
              f"(dropped {n_train0 - len(dm.ds_train)}) | val {len(dm.ds_val)} | test {len(dm.ds_test)}")

        tr = dm.ds_train
        n_jit = cfg["margin_jitter"]

        def both(path: str):  # LAB and RGB stacked, so the display goes through the same geometry
            rgb = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            return np.concatenate([load_lab(path), rgb], 2), load_mask(os.path.join(masks_dir, os.path.basename(path)))

        # 1. base crops (original | base crop) for a few train images
        tiles = []
        for i in np.linspace(0, len(tr) - 1, 6).astype(int):
            x, m = both(tr.samples[i])
            mode = tr.entries[i]["mode"]
            b, bm, side = base_crop(x, m, cfg["margin"], mode, image_size, cfg["max_upscale"], cfg["patch"])
            tiles += [_tile(cv2.resize(x[..., 3:], (image_size, image_size)),
                            cv2.resize(m.astype(np.uint8), (image_size, image_size),
                                       interpolation=cv2.INTER_NEAREST), cfg, "orig"),
                      _tile(b[..., 3:], bm, cfg, f"{mode} x{image_size / side:.2f}")]
        _grid(tiles, 4).save(os.path.join(out_dir, f"{ds}_base.png"))

        # 2. square vs pad vs pca on elongated images
        elong = [i for i, e in enumerate(tr.entries)
                 if bbox_stats(load_mask(os.path.join(masks_dir, os.path.basename(tr.samples[i]))))[1]
                 > cfg["elongated"]["ratio"]][:5]
        tiles = []
        for i in elong:
            x, m = both(tr.samples[i])
            for mode in ("square", "pad", "pca"):
                b, bm, _ = base_crop(x, m, cfg["margin"], mode, image_size, cfg["max_upscale"], cfg["patch"])
                tiles.append(_tile(b[..., 3:], bm, cfg, mode))
        if tiles:
            _grid(tiles, 3).save(os.path.join(out_dir, f"{ds}_pad_vs_pca.png"))

        # 3. augmented samples of the same images (two images: one capped at 2x, the largest object)
        sides = [e["side"] for e in tr.entries]
        for tag, i in (("small", int(np.argmin(sides))), ("large", int(np.argmax(sides)))):
            x, m = both(tr.samples[i])
            e = tr.entries[i]
            b, bm, side = base_crop(x, m, cfg["margin"], e["mode"], image_size, cfg["max_upscale"], cfg["patch"])
            tiles = [_tile(b[..., 3:], bm, cfg, "base")]
            for _ in range(11):
                if _rand() < n_jit["p"]:
                    a, am, _ = base_crop(x, m, _uniform(*n_jit["range"]), e["mode"], image_size,
                                         cfg["max_upscale"], cfg["patch"])
                    t = "jitter"
                else:
                    a, am = random_resized_crop(b, bm, side, e["n_obj"], cfg)
                    t = "rrc" if a is not b else "rrc-fail"
                a, am = geometry(a, am, cfg)
                tiles.append(_tile(a[..., 3:], am, cfg, t))
            _grid(tiles, 6).save(os.path.join(out_dir, f"{ds}_augment_{tag}.png"))

        # 4. checks over real augmented samples: no black (L = 0) pixel, object never touching the frame
        black = edge = 0
        n = 300
        for k in range(n):
            v, _, mk = tr[k % len(tr)]
            black += int(bool((v[0, 0] <= 1e-3).any()))
            mk = mk[0]
            edge += int(bool(mk[0].any() or mk[-1].any() or mk[:, 0].any() or mk[:, -1].any()))
        loader = torch.utils.data.DataLoader(tr, batch_size=32, num_workers=16, shuffle=True)
        t0 = time.time()
        for j, (v, y, mk) in enumerate(loader):
            if j == 0:
                shapes = (tuple(v.shape), tuple(y.shape), tuple(mk.shape))
        print(f"[{ds}] dataloader 1 epoch ({len(tr)} imgs, 16 workers, bs 32) {time.time() - t0:.1f}s "
              f"| batch {shapes} | samples with L=0 pixels {black}/{n} | object touching frame {edge}/{n}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Object-centred crop augmentation (Especificação 2).")
    parser.add_argument("--preview", action="store_true", help="write preview PNGs into --out")
    parser.add_argument("--datasets", nargs="+", default=["eggs", "larvae", "protozoan"])
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--percentage", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=200)
    parser.add_argument("--out", default="preview_crops")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.preview:
        _preview(args.datasets, args.split, args.percentage, args.out, args.image_size)
