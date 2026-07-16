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

"""svm_distillation.py — SVM evaluation for distilled FLIM CNN student models.

Scans ``artifacts/distillation/`` for completed runs (each having a
``run_metadata.json`` and at least one ``.ckpt`` checkpoint), loads each
student model, extracts raw encoder embeddings (no projection head), trains a
linear SVM on the training split, and evaluates on the test split.

Reuses from the existing evaluate infrastructure:
    * ``DatasetParasite`` — dataset loading (same parasite/split/pct as training)
    * ``_OneHotDataset``  — label wrapper for SVM training
    * ``compute_metrics`` — kappa / acc / f1
    * ``train_svm`` logic — adapted for ``student.encode()`` instead of conv layers

The ``train_svm`` from ``src.utils.evaluate`` accesses ``.conv1/.conv2/.conv3``
directly (FLIM-specific interface), so a distillation-compatible variant is
provided here that calls ``student.encode()``.

Usage::

    python -m src.evaluate.svm_distillation
    python -m src.evaluate.svm_distillation --wandb-update
    python -m src.evaluate.svm_distillation --run distillation_eggs_split1_pct1_modeldirect
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import svm as sk_svm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics
from src.modules.distillation_module import DistillationModule

_ARTIFACTS_DIR = os.path.join(_ROOT, "artifacts", "distillation")
_RESULTS_DIR   = os.path.join(_ROOT, "results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 200  # student input size

_DATASET_NUM_CLASSES: dict[str, int] = {
    "eggs":      9,
    "larvae":    2,
    "protozoan": 7,
}

_DATASET_PARASITE_NAME: dict[str, str] = {
    "eggs":      "helminth-eggs_split_2",
    "larvae":    "helminth-larvae_split_2",
    "protozoan": "protozoan-cysts_split_2",
}


# ── Dataset helpers ────────────────────────────────────────────────────────────


class _OneHotDataset(Dataset):
    """Wraps DatasetParasite and converts integer labels to one-hot vectors."""

    def __init__(self, base: Dataset, num_classes: int) -> None:
        self.base = base
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        one_hot = F.one_hot(torch.tensor(label), self.num_classes).float()
        return img, one_hot


# ── Feature extraction for distilled student ───────────────────────────────────


@torch.no_grad()
def extract_features_distillation(
    student_model,
    dataloader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw student encoder embeddings (global avg pool output).

    Unlike the FLIM SVM pipeline (which accesses .conv1/.conv2/.conv3 spatial
    tensors), this function uses ``student_model.encode()`` to obtain the
    pooled 1-D embedding, which is what the distillation training optimises.

    Args:
        student_model: ``LeJEPAFLIMModel`` instance.
        dataloader:    Yields ``(inputs, int_labels)``.

    Returns:
        features: ``np.ndarray`` shape ``(N, embed_dim)``
        y_true:   ``np.ndarray`` shape ``(N,)`` — 0-indexed labels.
    """
    student_model.eval()
    student_model.to(DEVICE)

    all_feats:  list[np.ndarray] = []
    all_labels: list[int]        = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting features"):
        feats = student_model.encode(inputs.to(DEVICE)).detach().cpu().numpy()
        all_feats.append(feats)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


def train_svm_distillation(
    student_model,
    dataloader: DataLoader,
    max_iter: int = 10_000,
    C: float = 1e2,
) -> object:
    """Fit a linear SVM on distilled student embeddings (1-indexed labels).

    Args:
        student_model: ``LeJEPAFLIMModel`` with ``encode()`` interface.
        dataloader:    Yields ``(inputs, one_hot_labels)`` — labels shape ``(B, C)``.
        max_iter:      SVM solver iteration cap.
        C:             SVM regularisation parameter.

    Returns:
        Fitted ``sklearn.svm.SVC`` (predictions are 1-indexed).
    """
    clf = sk_svm.SVC(
        max_iter=max_iter,
        C=C,
        gamma="auto",
        decision_function_shape="ovo",
        kernel="linear",
    )

    student_model.eval()
    student_model.to(DEVICE)

    all_feats: list[np.ndarray] = []
    all_y: list[int] = []

    print("[INFO] Extracting train features for SVM...")
    for inputs, labels in tqdm(dataloader, desc="  Train features"):
        feats = student_model.encode(inputs.to(DEVICE)).detach().cpu().numpy()
        all_feats.append(feats)
        y_np = np.argmax(labels.cpu().numpy(), axis=1) + 1  # 1-indexed
        all_y.extend(y_np.tolist())

    X = np.concatenate(all_feats, axis=0)
    y = np.array(all_y, dtype=np.int64)

    _stop = threading.Event()

    def _progress():
        with tqdm(desc="SVM fit", unit="s", bar_format="{desc}: {elapsed} [{postfix}]") as pbar:
            while not _stop.wait(1.0):
                pbar.update(1)
            pbar.set_postfix_str("done")

    _thr = threading.Thread(target=_progress, daemon=True)
    _thr.start()
    clf.fit(X, y)
    _stop.set()
    _thr.join()
    return clf


# ── Checkpoint / run discovery ─────────────────────────────────────────────────


def _ckpt_progress(path: str) -> tuple[int, int]:
    """(epoch, global_step) gravados dentro de um checkpoint Lightning.

    Usado para escolher o checkpoint MAIS treinado em vez de confiar no nome
    ("best.ckpt" pode ser um arquivo stale de epoch 0 deixado por um re-run que
    crashou — ver analysis_flim_distill/INVESTIGATION_training_problems_nonorm.md).
    """
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return (int(ck.get("epoch", -1)), int(ck.get("global_step", -1)))
    except Exception:
        return (-1, -1)


def find_distillation_runs(artifacts_dir: str, run_filter: str | None = None) -> list[dict]:
    """Scan artifacts/distillation/ and return metadata dicts for valid runs.

    A run is considered valid if it has a ``run_metadata.json`` and at least
    one ``.ckpt`` checkpoint file.

    Args:
        artifacts_dir: Root of the distillation artifacts.
        run_filter:    If provided, only include runs whose name contains this
                       string (substring match).

    Returns:
        List of metadata dicts with an extra ``"_ckpt_path"`` key.
    """
    if not os.path.isdir(artifacts_dir):
        return []

    runs = []
    for entry in sorted(os.scandir(artifacts_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        if run_filter and run_filter not in entry.name:
            continue

        meta_path = os.path.join(entry.path, "run_metadata.json")
        if not os.path.isfile(meta_path):
            continue

        ckpts = glob.glob(os.path.join(entry.path, "checkpoints", "**", "*.ckpt"), recursive=True)
        if not ckpts:
            continue

        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        # Seleção de checkpoint: pega o MAIS TREINADO, nunca pelo nome do arquivo.
        #   1) confia em meta["best_checkpoint"] (gravado no fim do treino) se existir;
        #   2) senão: maior epoch → maior global_step → menor val/loss → mtime.
        # O nome "best.ckpt" NÃO é confiável: re-runs/crash deixam um best.ckpt de
        # epoch 0 ao lado de um best-v1.ckpt treinado (epoch 99).
        import re
        best_ckpt_meta = meta.get("best_checkpoint")
        if best_ckpt_meta and os.path.isfile(best_ckpt_meta):
            meta["_ckpt_path"] = best_ckpt_meta
        else:
            best_ckpts = [c for c in ckpts if os.path.basename(c).startswith("best")]
            last_ckpts = [c for c in ckpts if os.path.basename(c).startswith("last")]
            group = best_ckpts or last_ckpts
            if not group:
                continue

            def _pick(p: str):
                ep, gs = _ckpt_progress(p)
                hit = re.search(r"loss=([0-9.]+)\.ckpt$", os.path.basename(p))
                loss = float(hit.group(1)) if hit else float("inf")
                return (ep, gs, -loss, os.path.getmtime(p))

            meta["_ckpt_path"] = max(group, key=_pick)

        meta["_run_dir"] = entry.path
        runs.append(meta)

    return runs


# ── Main evaluation loop ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVM evaluation of distilled FLIM CNN student models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--wandb-update", action="store_true",
        help="Log per-experiment SVM metrics to W&B.",
    )
    parser.add_argument(
        "--run", default=None,
        help="Restrict to a single run name (substring match).",
    )
    parser.add_argument(
        "--artifacts-dir", default=_ARTIFACTS_DIR,
        help="Root of distillation artifacts (default: artifacts/distillation/).",
    )
    args = parser.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    transform = _build_test(IMAGE_SIZE)

    # ── Discover runs ──────────────────────────────────────────────────────
    runs = find_distillation_runs(args.artifacts_dir, run_filter=args.run)

    if not runs:
        print(f"[WARN] No valid distillation runs found under: {args.artifacts_dir}")
        print("       Train at least one run with distillation_ray.py first.")
        return

    print(f"\n{'=' * 70}")
    print(f"[FILTER] Found {len(runs)} distillation run(s) to evaluate.")
    print(f"{'=' * 70}")

    rows: list[dict] = []

    for meta in runs:
        run_name = meta.get("run_name", os.path.basename(meta["_run_dir"]))
        ckpt_path = meta["_ckpt_path"]

        print(f"\n{'=' * 70}")
        print(f"  Run  : {run_name}")
        print(f"  Ckpt : {os.path.relpath(ckpt_path, _ROOT)}")
        print(f"{'=' * 70}")

        dataset   = meta.get("dataset", "")
        split     = meta.get("split", 1)
        pct       = meta.get("percentage", 100)
        dist_type = meta.get("distillation_type", "direct")
        enc_init  = meta.get("encoder_init", "trunc_normal")
        emb_dim   = meta.get("student_embed_dim", "?")

        base_row = {
            "run_name":          run_name,
            "dataset":           dataset,
            "split":             split,
            "percentage":        pct,
            "distillation_type": dist_type,
            "encoder_init":      enc_init,
            "student_embed_dim": emb_dim,
            "teacher_model":     meta.get("teacher_model", "ijepa"),
            "ckpt_path":         ckpt_path,
        }

        num_classes = _DATASET_NUM_CLASSES.get(dataset, 9)
        parasite_name = _DATASET_PARASITE_NAME.get(dataset, dataset)

        try:
            # ── Load student model from checkpoint ─────────────────────────
            module = DistillationModule.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )
            module.eval()
            student = module.student
            for p in student.parameters():
                p.requires_grad_(False)

            actual_emb_dim = student.embed_dim
            print(f"  [INFO] Student embed_dim = {actual_emb_dim}")

            # ── Train SVM on training split ────────────────────────────────
            train_base = DatasetParasite(
                set_name="train",
                split=split,
                percentage=pct,
                transform=transform,
                loader="ift_lab",
                path_dataset=parasite_name,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=32,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            clf = train_svm_distillation(student, train_loader)

            # ── Evaluate on test split ────────────────────────────────────
            test_ds = DatasetParasite(
                set_name="test",
                split=split,
                percentage=pct,
                transform=transform,
                loader="ift_lab",
                path_dataset=parasite_name,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            feats, y_true = extract_features_distillation(student, test_loader)

            # SVM trained with 1-indexed labels → convert predictions to 0-indexed
            y_pred = clf.predict(feats) - 1

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
            rows.append({**base_row, **metrics, "status": "ok", "error": ""})

            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )

            # ── Optional W&B logging ───────────────────────────────────────
            if args.wandb_update:
                try:
                    import wandb  # noqa: PLC0415
                    from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
                    wandb_run = wandb.init(
                        project=PROJECT,
                        entity=ENTITY,
                        name=f"svm_distil_{run_name}",
                        config={**base_row, "num_classes": num_classes},
                        reinit=True,
                    )
                    wandb.log({f"svm/{k}": v for k, v in metrics.items()})
                    wandb_run.finish()
                except Exception as _we:
                    print(f"  [WARN] W&B logging failed: {_we}")

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                **base_row,
                "kappa": float("nan"),
                "acc":   float("nan"),
                "f1":    float("nan"),
                "status": "error",
                "error":  str(exc),
            })

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(_RESULTS_DIR, "svm_distillation_results.csv")
    _meta_cols = [
        "run_name", "dataset", "split", "percentage",
        "distillation_type", "encoder_init", "student_embed_dim",
        "teacher_model",
    ]
    _metric_cols = ["kappa", "acc", "f1"]
    _extra_cols  = ["status", "error", "ckpt_path"]
    _col_order   = _meta_cols + _metric_cols + _extra_cols

    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to : {csv_path}")
    print(f"       Runs evaluated   : {len(rows)}")
    print(f"{'=' * 70}")

    # ── W&B summary upload ─────────────────────────────────────────────────
    if args.wandb_update:
        try:
            import wandb  # noqa: PLC0415
            from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415

            df_ok = df[df["status"] == "ok"]
            summary: dict = {}
            for m in _metric_cols:
                if m in df_ok.columns and not df_ok[m].isna().all():
                    summary[f"summary/mean_{m}"] = float(df_ok[m].mean())
                    summary[f"summary/std_{m}"]  = float(df_ok[m].std())

            print(f"\n[W&B] Uploading summary to project '{PROJECT}'...")
            srun = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name="svm_distillation_summary",
                job_type="evaluation_summary",
                reinit=True,
            )
            wandb.log(summary)
            wandb.log({
                "svm_distil/results_table": wandb.Table(
                    dataframe=df_ok[_meta_cols + _metric_cols].reset_index(drop=True)
                )
            })
            artifact = wandb.Artifact("svm_distillation_results", type="evaluation_results")
            artifact.add_file(csv_path, name="svm_distillation_results.csv")
            srun.log_artifact(artifact)
            srun.finish()
            print("[W&B] Upload complete.")
        except Exception as _we:
            print(f"[WARN] W&B summary upload failed: {_we}")


if __name__ == "__main__":
    main()
