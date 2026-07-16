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

"""tsne_flim.py — t-SNE visualization of FLIM encoder embeddings on test splits.

For each available FLIM model (eggs/train{1,2,3}, larvae/train{1,2,3},
protozoan/train{1,2,3}), loads test-split samples from every incremental
split JSON descriptor, extracts encoder embeddings, applies t-SNE, and saves
one Matplotlib figure per (model × split × percentage).

No training is performed.  Inference only.

Output: tsne_analisys/{problem}/split{N}/perc{pct}/{train_id}.png

Usage:
    python -m src.analysis.tsne_flim
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive — no GUI required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from src.analysis.pyift_strategy import PyiftLoader

# ── Project root ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parent.parent.parent   # src/analysis/tsne_flim.py → project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load models.py directly — bypasses src/models/__init__.py to avoid
#    import-chain side-effects from other model submodules.
def _load_module_from_file(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_models_mod = _load_module_from_file(
    "src_models_models", _ROOT / "src" / "models" / "models.py"
)
Encoder                      = _models_mod.Encoder
parse_architecture           = _models_mod.parse_architecture
get_actual_channels_from_weights = _models_mod.get_actual_channels_from_weights
override_arch_channels       = _models_mod.override_arch_channels
load_FLIM_encoder            = _models_mod.load_FLIM_encoder

# config.py lives at the project root — safe to import directly.
from config import get_single_parasite_paths

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths & constants ─────────────────────────────────────────────────────────
FLIM_BASE   = _ROOT / "data" / "to_mateus" / "model" / "ch24_32_48_a0.5_f5"
OUTPUT_BASE = _ROOT / "tsne_analisys"

PROBLEMS = ["eggs", "larvae", "protozoan"]

# Maps FLIM model problem folder → full parasite name used in the data registry.
PROBLEM_MAP: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

# Number of classes per dataset — mirrors src/utils/constant.py.
DATASET_NUM_CLASSES: dict[str, int] = {
    "helminth-eggs":    9,
    "helminth-larvae":  2,
    "protozoan-cysts":  7,
}

SPLITS      = [1, 2, 3]
PERCENTAGES = [1, 5, 25, 50, 75, 100]

IMAGE_SIZE  = 200
BATCH_SIZE  = 32
# num_workers=0 avoids macOS multiprocessing issues with C extensions.
NUM_WORKERS = 0

TSNE_RANDOM_STATE = 42

# Qualitative palette — visually distinct across up to 20 classes.
_PALETTE = [
    "#E6194B", "#3CB44B", "#4363D8", "#F58231", "#911EB4",
    "#42D4F4", "#F032E6", "#BFEF45", "#FABED4", "#469990",
    "#DCBEFF", "#9A6324", "#C8B400", "#800000", "#AAFFC3",
    "#808000", "#FFD8B1", "#000075", "#A9A9A9", "#6B6B6B",
]


# ── Image transform (mirrors lejepa_dataset._build_test) ─────────────────────

def _build_test_transform(image_size: int) -> v2.Compose:
    """Deterministic resize + centre-crop + normalise — identical to the one
    used throughout the project (src/data_modules/datasets/lejepa_dataset.py)."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(image_size),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Lightweight test dataset ──────────────────────────────────────────────────

class _FLIMTestDataset(Dataset):
    """Reads a split JSON and loads test images through a PyiftLoader strategy.

    Loading modes (determined by the provided *loader*):
    - loader.mode == "local"  → native pyift, lazy per __getitem__.
    - loader.mode == "docker" → all images preloaded in __init__ via a single
                                Docker batch call to avoid per-sample overhead.
    - loader.mode == "pil"    → PIL RGB, lazy per __getitem__.

    Label is derived from the filename prefix (1-indexed → 0-indexed),
    matching the DatasetParasite convention.
    """

    def __init__(
        self,
        parasite_name: str,
        split: int,
        percentage: int,
        transform,
        loader: PyiftLoader,
    ) -> None:
        class_info = get_single_parasite_paths(parasite_name, split, percentage)
        info       = class_info[0]
        json_path  = Path(info["split_json"])
        images_dir = Path(info["images_dir"])

        if not json_path.is_file():
            raise FileNotFoundError(f"Split JSON not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "test" not in data:
            raise KeyError(f"'test' field missing in {json_path}")

        filenames = data["test"]
        if not filenames:
            raise ValueError(f"'test' field is empty in {json_path}")

        self.samples   = [images_dir / fname for fname in filenames]
        self.transform = transform
        self._loader   = loader

        # Docker mode: preload the entire test set in one batch call so we
        # never spawn a container per sample inside __getitem__.
        self._cache: Optional[list[np.ndarray]] = None
        if loader.mode == "docker":
            log.info(
                "  Pre-loading %d test images via Docker…", len(self.samples)
            )
            self._cache = loader.load_batch([str(p) for p in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path  = self.samples[idx]
        label = int(path.name.split("_")[0]) - 1   # 1-indexed → 0-indexed

        if self._cache is not None:
            # Docker pre-loaded: use cached (H, W, 3) float32 array.
            arr = self._cache[idx]
            img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        elif self._loader.mode == "local":
            arr = self._loader.load_batch([str(path)])[0]
            img = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        else:
            # PIL fallback — returns PIL Image; v2.ToImage() handles conversion.
            img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        return img, label


# ── Device selection (Mac-aware — no CUDA assumed) ────────────────────────────

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Selected device: %s", device)
    return device


# ── FLIM model discovery ──────────────────────────────────────────────────────

def discover_flim_models() -> list[dict]:
    """Return one dict per available FLIM model directory.

    Keys: problem, train_id, arch_json, weights_path.
    """
    models: list[dict] = []
    for problem in PROBLEMS:
        problem_dir = FLIM_BASE / problem
        if not problem_dir.is_dir():
            log.warning("FLIM problem dir not found: %s — skipped", problem_dir)
            continue
        for train_dir in sorted(problem_dir.iterdir()):
            if not train_dir.is_dir():
                continue
            arch_json    = train_dir / "architecture.json"
            weights_path = train_dir / "models"
            if not arch_json.is_file():
                log.warning("Missing architecture.json in %s — skipped", train_dir)
                continue
            if not weights_path.is_dir():
                log.warning("Missing models/ dir in %s — skipped", train_dir)
                continue
            models.append({
                "problem":      problem,
                "train_id":     train_dir.name,
                "arch_json":    str(arch_json),
                "weights_path": str(weights_path),
            })
    return models


# ── Encoder loading ───────────────────────────────────────────────────────────

def load_flim_encoder(
    arch_json: str,
    weights_path: str,
    device: torch.device,
) -> nn.Module:
    """Parse arch JSON, build Encoder, load FLIM weights, freeze, set eval."""
    arch     = parse_architecture(arch_json)
    channels = get_actual_channels_from_weights(weights_path, arch, in_channels=3)
    arch     = override_arch_channels(arch, channels)
    encoder  = Encoder(arch, in_channels=3)
    load_FLIM_encoder(encoder, arch_json, weights_path, channels)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    encoder.to(device)
    return encoder


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run encoder + global average pool over all batches.

    Returns:
        embeddings – float32 ndarray of shape (N, D)
        labels     – int64  ndarray of shape (N,)
    """
    pool     = nn.AdaptiveAvgPool2d(1).to(device)
    all_emb: list[np.ndarray] = []
    all_lbl: list[int]        = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        out    = encoder(inputs)
        out    = pool(out).flatten(1).cpu().numpy()
        all_emb.append(out)
        if isinstance(labels, torch.Tensor):
            all_lbl.extend(labels.tolist())
        else:
            all_lbl.extend(list(labels))

    return np.concatenate(all_emb, axis=0), np.array(all_lbl, dtype=np.int64)


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Fit 2-D t-SNE on *embeddings*.  Returns projection or None on failure."""
    n = len(embeddings)
    if n < 5:
        log.warning("Too few samples (%d) for t-SNE — skipped", n)
        return None
    perplexity = min(30.0, max(5.0, n / 4.0))
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=TSNE_RANDOM_STATE,
            max_iter=1000,
        )
        return tsne.fit_transform(embeddings)
    except Exception as exc:
        log.error("t-SNE failed: %s", exc)
        return None


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Save a t-SNE scatter plot (no legend, no show, no imshow).

    Class names are placed at the spatial median of each point cloud so the
    plot is self-annotating without a legend.
    """
    unique_labels = sorted(set(labels.tolist()))

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("t-SNE dim 1", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", fontsize=10)

    for i, lbl in enumerate(unique_labels):
        mask  = labels == lbl
        color = _PALETTE[i % len(_PALETTE)]

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=14,
            alpha=0.75,
            linewidths=0,
        )

        # Median centroid — robust to cluster outliers.
        cx = float(np.median(coords[mask, 0]))
        cy = float(np.median(coords[mask, 1]))
        ax.text(
            cx, cy,
            f"cls {lbl}",
            fontsize=8,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec=color,
                alpha=0.65,
                linewidth=0.9,
            ),
        )

    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    try:
        rel = save_path.relative_to(_ROOT)
    except ValueError:
        rel = save_path
    log.info("Saved → %s", rel)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device    = _select_device()
    transform = _build_test_transform(IMAGE_SIZE)

    # Resolve the image-loading backend once — local pyift → Docker → PIL.
    loader = PyiftLoader(allow_pil_fallback=True)
    log.info("Image loader mode: %s", loader.mode)

    flim_models = discover_flim_models()
    if not flim_models:
        log.error("No FLIM models found under %s — aborting.", FLIM_BASE)
        return
    log.info("Discovered %d FLIM models.", len(flim_models))

    for model_meta in tqdm(flim_models, desc="FLIM models", unit="model"):
        problem       = model_meta["problem"]
        train_id      = model_meta["train_id"]
        arch_json     = model_meta["arch_json"]
        weights_path  = model_meta["weights_path"]
        parasite_name = PROBLEM_MAP[problem]

        log.info("═══ Model: %s/%s  (dataset: %s)", problem, train_id, parasite_name)

        try:
            encoder = load_flim_encoder(arch_json, weights_path, device)
        except Exception as exc:
            log.error(
                "Cannot load encoder %s/%s: %s — skipped", problem, train_id, exc
            )
            continue

        for split in tqdm(SPLITS, desc=f"  splits ({train_id})", leave=False):
            for pct in tqdm(
                PERCENTAGES, desc=f"    perc (split{split})", leave=False
            ):
                log.info("  split=%d  pct=%d%%  →  %s", split, pct, parasite_name)

                # ── Build test dataset ────────────────────────────────────────
                try:
                    test_ds = _FLIMTestDataset(
                        parasite_name=parasite_name,
                        split=split,
                        percentage=pct,
                        transform=transform,
                        loader=loader,
                    )
                except FileNotFoundError as exc:
                    log.warning("Split JSON not found: %s — skipped", exc)
                    continue
                except (KeyError, ValueError) as exc:
                    log.warning("Invalid split JSON (split=%d, pct=%d): %s — skipped",
                                split, pct, exc)
                    continue
                except Exception as exc:
                    log.error("Dataset error: %s — skipped", exc)
                    continue

                if len(test_ds) == 0:
                    log.warning(
                        "Empty test set (split=%d, pct=%d, %s) — skipped",
                        split, pct, parasite_name,
                    )
                    continue

                test_loader = DataLoader(
                    test_ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                )

                # ── Extract embeddings ────────────────────────────────────────
                try:
                    embeddings, labels = extract_embeddings(
                        encoder, test_loader, device
                    )
                except Exception as exc:
                    log.error("Embedding extraction failed: %s — skipped", exc)
                    continue

                # ── t-SNE ─────────────────────────────────────────────────────
                projection = run_tsne(embeddings)
                if projection is None:
                    continue

                # ── Save plot ─────────────────────────────────────────────────
                save_path = (
                    OUTPUT_BASE
                    / problem
                    / f"split{split}"
                    / f"perc{pct}"
                    / f"{train_id}.png"
                )
                title = (
                    f"t-SNE  ·  {parasite_name}"
                    f"  ·  split {split}  ·  {pct}% train  ·  {train_id}"
                )
                try:
                    plot_tsne(
                        coords=projection,
                        labels=labels,
                        title=title,
                        save_path=save_path,
                    )
                except Exception as exc:
                    log.error("Plot failed (%s): %s", save_path.name, exc)

    log.info("Done.  All outputs under: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
