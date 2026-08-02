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

"""evaluate.py — SVM evaluation of pretrained LeJEPA Line encoders.

For each (run_id, run_name) in MY_EXPERIMENTS (imported from constant.py):
  1. Finds the best checkpoint under logs/flim-ssl/<run_id>/checkpoints/.
  2. Loads the LejepaLineModule and extracts the FLIM encoder (conv1/conv2/conv3).
  3. Trains an SVM on the training split (same percentage as model training).
  4. Evaluates on the test split.
  5. Writes all results (including failures) to a CSV file using pandas.

Usage:
    python -m src.utils.evaluate
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics
from src.modules.lejepa_line_module import LejepaLineModule

DATASET_NUM_CLASSES: dict[str, int] = {
    "helminth-eggs": 9,
    "helminth-larvae": 2,
    "protozoan-cysts": 7,
    "parasito": 9,
}

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Checkpoint search order: newest experiments first, legacy fallback second.
_LOGS_DIRS = [
    os.path.join(_ROOT, "logs", "flim-ssl"),
    #os.path.join(_ROOT, "logs", "flim-ssl_old"),
]


# ─── Available-experiment resolution ──────────────────────────────────────────


def get_local_run_ids() -> set[str]:
    """Return run IDs that have at least one .ckpt file under any logs dir."""
    local_ids: set[str] = set()
    for logs_dir in _LOGS_DIRS:
        if not os.path.isdir(logs_dir):
            continue
        for entry in os.scandir(logs_dir):
            if not entry.is_dir():
                continue
            ckpt_dir = os.path.join(entry.path, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            if glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True):
                local_ids.add(entry.name)
    return local_ids


def resolve_available_experiments(update_wandb: bool = False) -> dict[str, str]:
    """Return {run_id: run_name} for all W&B runs that also have local weights.

    By default loads W&B metadata from the local cache
    (``configs/wandb_update/ids_wandb.json``).  Pass ``update_wandb=True``
    to refresh the cache from the live W&B API first.

    Args:
        update_wandb: When ``True``, fetch fresh metadata from W&B and
                      overwrite the local cache before resolving.
    """
    from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
    from src.utils.wandb_cache import get_runs_dict_cached  # noqa: PLC0415
    print("[FILTER] Fetching W&B run list...")
    # Fetch without deduplication so we can prefer run_ids that have local
    # checkpoints when the same experiment name has multiple W&B runs.
    wandb_runs_all = get_runs_dict_cached(
        ENTITY, PROJECT, deduplicate=False, update=update_wandb
    )

    local_ids = get_local_run_ids()

    # Deduplicate: for each unique run name keep the run_id that has a local
    # checkpoint.  If none (or multiple) have a checkpoint, fall back to the
    # most-recently-created one.
    from src.utils.wandb_cache import load_cache, CACHE_FILE  # noqa: PLC0415
    _cache = load_cache(CACHE_FILE) or {}
    _meta: dict[str, dict] = _cache.get("runs", {})

    best: dict[str, tuple[str, str, bool]] = {}  # name → (run_id, created_at, has_local)
    for rid, name in wandb_runs_all.items():
        created_at = _meta.get(rid, {}).get("created_at", "")
        has_local = rid in local_ids
        if name not in best:
            best[name] = (rid, created_at, has_local)
        else:
            prev_rid, prev_ts, prev_has = best[name]
            # Prefer run_ids with local checkpoint; break ties by newest created_at
            if (has_local and not prev_has) or (has_local == prev_has and created_at > prev_ts):
                best[name] = (rid, created_at, has_local)

    wandb_runs = {rid: name for name, (rid, _, _) in best.items()}

    available = {rid: name for rid, name in wandb_runs.items() if rid in local_ids}
    missing_local = {rid: name for rid, name in wandb_runs.items() if rid not in local_ids}

    print(f"[FILTER] W&B runs total        : {len(wandb_runs_all)}")
    print(f"[FILTER] After dedup           : {len(wandb_runs)}")
    print(f"[FILTER] With local weights    : {len(available)}")
    if missing_local:
        print(f"[FILTER] Skipping — no ckpt   : {len(missing_local)} run(s)")
        for rid in sorted(missing_local):
            print(f"           skip  {rid}  ({missing_local[rid]})")

    return available

# ─── Configuration ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 200

# ─── Experiment name parsing ──────────────────────────────────────────────────

_CANONICAL_RE = re.compile(
    r"^lejepa_line_([a-z\-]+)_split_(\d+)_pct_(\d+)_model_"
    r"(xavier|random|he|flim|trunc_normal)"
    # Sufixo opcional de variante (batch size / config de treino), documentado no
    # README como `[_bs<batch>]` — ex.: `_bs256`, `_bs256_mc2g8l`, `_bs150_mc8g8l_3lproj`.
    r"(?:_(.+))?$"
)
_OLD_RE = re.compile(r"^line_p(\d+)_(xavier|random|he|flim|trunc_normal)$")


@dataclass
class ExperimentInfo:
    dataset_name: str
    split_id: int
    percentage: int
    initialization_type: str
    path_dataset: Optional[str]  # None → legacy combined-parasito mode


def parse_experiment_name(run_name: str) -> ExperimentInfo:
    """Parse dataset, split, percentage, and init from a run name.

    Supports:
    - Canonical: ``lejepa_line_{dataset}_split_{N}_pct_{pct}_model_{init}``
    - Legacy:    ``line_p{pct}_{init}``  (→ parasito combined, split=1)
    """
    m = _CANONICAL_RE.match(run_name)
    if m:
        dataset = m.group(1)
        return ExperimentInfo(
            dataset_name=dataset,
            split_id=int(m.group(2)),
            percentage=int(m.group(3)),
            initialization_type=m.group(4),
            path_dataset=dataset,
        )
    m = _OLD_RE.match(run_name)
    if m:
        return ExperimentInfo(
            dataset_name="parasito",
            split_id=1,
            percentage=int(m.group(1)),
            initialization_type=m.group(2),
            path_dataset=None,
        )
    raise ValueError(f"Cannot parse experiment name: '{run_name}'")


# ─── Checkpoint helpers ────────────────────────────────────────────────────────


def find_best_checkpoint(run_id: str) -> str:
    """Return the lowest-loss checkpoint for *run_id*.

    Searches ``logs/flim-ssl`` first, then ``logs/flim-ssl_old``.
    Within each run, picks the best checkpoint: the constant ``best.ckpt``
    written by the current config, or — for legacy runs — the ``best-*``
    checkpoint with the lowest loss value encoded in the filename; falls back
    to ``last.ckpt`` if no best exists.
    """
    for logs_dir in _LOGS_DIRS:
        ckpt_dir = os.path.join(logs_dir, run_id, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        # New layout: a single constant best.ckpt directly in checkpoints/.
        candidates = glob.glob(os.path.join(ckpt_dir, "best.ckpt"))
        # Legacy layout: best-<epoch=…-loss=…>/<file>.ckpt subdirectories.
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, "best-*", "*.ckpt"))
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, "last.ckpt"))
        if candidates:
            def _loss(p: str) -> float:
                hit = re.search(r"loss=([0-9.]+)\.ckpt$", p)
                return float(hit.group(1)) if hit else float("inf")
            return min(candidates, key=_loss)
    raise FileNotFoundError(
        f"No checkpoint found for run_id={run_id!r} "
        f"in: {[os.path.relpath(d, _ROOT) for d in _LOGS_DIRS]}"
    )


# ─── Feature pooling ──────────────────────────────────────────────────────────

# Global average pooling applied to the conv3 feature map before flattening.
# Matches the convention used by the classification heads (``MLPHead.forward``,
# ``TwoLayerSigmoidHead``) and by ``src/evaluate/svm_classification_flim.py``,
# so the SVM sees the same ``channels[-1]``-d embedding the heads see instead of
# a spatially flattened map.
_POOL = nn.AdaptiveAvgPool2d(1)


def _encode_pooled(model, inputs: torch.Tensor) -> torch.Tensor:
    """Run conv1→conv2→conv3, global-average-pool and flatten to ``[B, C]``."""
    out = model.conv1(inputs)
    out = model.conv2(out)
    out = model.conv3(out)
    return _POOL(out).flatten(start_dim=1).detach().cpu()


# ─── SVM training ─────────────────────────────────────────────────────────────


@torch.no_grad()
def train_svm(model, dataloader, max_iter: int = 10000, C: float = 1e2, degree: int = 3):
    """Fit a linear SVM on frozen encoder features.

    Features are the conv3 feature map reduced by ``AdaptiveAvgPool2d(1)`` and
    flattened, i.e. one value per output channel (``channels[-1]``-d).

    *dataloader* must yield ``(inputs, one_hot_labels)`` where one_hot_labels
    has shape ``(B, num_classes)``.  Labels returned by the fitted SVM are
    1-indexed (matching the ``+1`` applied here).
    """
    print("[INFO] Initializing SVM")
    clf = svm.SVC(
        max_iter=max_iter,
        C=C,
        degree=degree,
        gamma="auto",
        coef0=0,
        decision_function_shape="ovo",
        kernel="linear",
    )

    model.eval()
    model.to(DEVICE)

    all_feats = torch.Tensor([])
    all_y = torch.Tensor([]).long()

    print("[INFO] Preparing data for SVM")
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(DEVICE)
        all_feats = torch.cat((all_feats, _encode_pooled(model, inputs)))

        labels_np = np.argmax(labels.cpu().numpy(), axis=1) + 1  # 1-indexed
        all_y = torch.cat((all_y, torch.from_numpy(labels_np).long()))

    _stop = threading.Event()

    def _progress():
        with tqdm(desc="SVM fit", unit="s", bar_format="{desc}: {elapsed} [{postfix}]") as pbar:
            while not _stop.wait(1.0):
                pbar.update(1)
            pbar.set_postfix_str("done")

    _thread = threading.Thread(target=_progress, daemon=True)
    _thread.start()
    clf.fit(all_feats, all_y)
    _stop.set()
    _thread.join()
    return clf


# ─── One-hot label wrapper ─────────────────────────────────────────────────────


class _OneHotDataset(Dataset):
    """Wraps DatasetParasite and converts integer labels to one-hot vectors.

    Required because ``train_svm`` expects ``(B, num_classes)`` labels and
    applies ``np.argmax(..., axis=1)`` to recover class indices.
    """

    def __init__(self, base: Dataset, num_classes: int) -> None:
        self.base = base
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        one_hot = F.one_hot(torch.tensor(label), self.num_classes).float()
        return img, one_hot


# ─── Feature extraction ────────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(model, dataloader):
    """Extract conv1→conv2→conv3 features from *model* for every batch.

    The conv3 feature map is reduced by ``AdaptiveAvgPool2d(1)`` and flattened,
    so each sample yields one value per output channel (``channels[-1]``-d) —
    the same embedding the classification heads consume.

    Args:
        model:      FLIM Encoder with ``.conv1``, ``.conv2``, ``.conv3``.
        dataloader: Yields ``(inputs, int_labels)``.

    Returns:
        features: ``np.ndarray`` of shape ``(N, channels[-1])`` — pooled conv3.
        y_true:   ``np.ndarray`` of shape ``(N,)``  — 0-indexed class labels.
    """
    model.eval()
    model.to(DEVICE)

    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting test features"):
        inputs = inputs.to(DEVICE)
        all_feats.append(_encode_pooled(model, inputs).numpy())

        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


# ─── Main evaluation loop ──────────────────────────────────────────────────────


def main() -> None:
    transform = _build_test(IMAGE_SIZE)
    rows: list[dict] = []

    for run_id, run_name in resolve_available_experiments().items():
        print(f"\n{'=' * 70}")
        print(f"  Run : {run_name}  (id={run_id})")
        print(f"{'=' * 70}")

        base_row: dict = {"wandb_run_id": run_id, "experiment_name": run_name}

        try:
            info = parse_experiment_name(run_name)
            base_row.update({
                "dataset_name": info.dataset_name,
                "split_id": info.split_id,
                "percentage": info.percentage,
                "initialization_type": info.initialization_type,
            })

            num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)

            # ── Load encoder from best checkpoint ──────────────────────────
            ckpt_path = find_best_checkpoint(run_id)
            print(f"  [INFO] Checkpoint : {os.path.relpath(ckpt_path, _ROOT)}")

            module = LejepaLineModule.load_from_checkpoint(
                ckpt_path, map_location=DEVICE
            )
            module.eval()
            encoder = module.model.encoder

            for p in encoder.parameters():
                p.requires_grad_(False)

            # ── Train SVM on training split ────────────────────────────────
            print(
                f"  [INFO] Building train dataloader "
                f"(split={info.split_id}, pct={info.percentage})"
            )
            train_base = DatasetParasite(
                set_name="train",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            clf = train_svm(encoder, train_loader)

            # ── Evaluate on test split ─────────────────────────────────────
            print(
                f"  [INFO] Building test dataloader  "
                f"(split={info.split_id}, pct={info.percentage})"
            )
            test_ds = DatasetParasite(
                set_name="test",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            feats, y_true = extract_features(encoder, test_loader)

            # SVM trained with 1-indexed labels → convert predictions back to 0-indexed
            y_pred = clf.predict(feats) - 1

            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                num_classes=num_classes,
            )

            rows.append({**base_row, **metrics, "status": "ok", "error": ""})
            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                **base_row,
                "kappa": float("nan"),
                "acc": float("nan"),
                "f1": float("nan"),
                "status": "error",
                "error": str(exc),
            })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    _results_dir = os.path.join(_ROOT, "results")
    os.makedirs(_results_dir, exist_ok=True)
    csv_path = os.path.join(_results_dir, "svm_results.csv")

    _meta = ["wandb_run_id", "experiment_name", "dataset_name",
             "split_id", "percentage", "initialization_type"]
    _metrics = ["kappa", "acc", "f1"]
    _extra = ["status", "error"]
    _col_order = _meta + _metrics + _extra

    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
