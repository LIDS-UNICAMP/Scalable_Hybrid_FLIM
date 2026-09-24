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

"""unified_eval.py — Unified evaluation pipeline for LEJEPA encoders.

Evaluates encoder representations via:
  1. SVM    — linear SVM on frozen encoder embeddings
  2. MLP unfreeze — inference with fine-tuned MLP (unfrozen encoder)
  3. MLP freeze   — inference with fine-tuned MLP (frozen encoder)

Artifact layout::

    artifacts/
      SVM/
        {dataset}/lejepa_pct_{pct}/
          predictions_{run_name}.csv
          metrics_{model_type}_{init}.csv
          svm_results.csv
      MLP/
        {dataset}/lejepa_pct_{pct}/
          predictions_{run_name}_{freeze|unfreeze}.csv
          metrics_{model_type}_{init}.csv
          mlp_results.csv
      plots/
        {dataset}/lejepa_pct_{pct}/
          {dataset}_pct{pct}_composite.{pdf,png}
          {dataset}_{method}_{metric}_vs_pct.{pdf,png}

Usage::

    python -m src.evaluate.unified_eval --model svm --dataset eggs
    python -m src.evaluate.unified_eval --model all --dataset all
    python -m src.evaluate.unified_eval --model svm --dataset protozoan --dry-run
    python -m src.evaluate.unified_eval --model mlp_freeze --dataset larvae --pct 25
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.constants import DATASET_NUM_CLASSES, IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from eval.eval_plotter import plot_composite, plot_metric_vs_pct
from eval.wandb_resolver import (
    DATASET_SHORT_TO_FULL,
    FinetuneRunInfo,
    SSLRunInfo,
    resolve_finetune_runs,
    resolve_ssl_runs,
)
from methods.lejepa.lejepa_line_module import LejepaLineModule
import eval.svm as _ev  # módulo, para rebindar EMBED_MODE em main()
from eval.svm import (
    _OneHotDataset,
    cli_kwargs,
    extract_features,
    find_best_checkpoint,
    train_svm,
)

# Raiz do repositorio: antes vinha de src.utils.evaluate._ROOT, hoje de core.constants.
_ROOT = PROJECT_ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASETS_ALL = ["eggs", "protozoan", "larvae"]
MODELS_ALL = ["svm", "mlp_unfreeze", "mlp_freeze"]  # execution order per SDD 5.2

log = logging.getLogger(__name__)


# ── Artifact path helper ───────────────────────────────────────────────────────

def _artifact_dir(output_root: Path, method_prefix: str, dataset_short: str, pct: int) -> Path:
    """Return canonical artifact directory for a (method, dataset, pct) group."""
    return output_root / method_prefix / dataset_short / f"lejepa_pct_{pct}"


# ── Prediction CSV helper ──────────────────────────────────────────────────────

def _save_predictions(
    out_dir: Path,
    filename: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: dict,
) -> Path:
    """Write per-sample predictions CSV with traceability metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "sample_id":       np.arange(len(y_true)),
        "dataset":         metadata.get("dataset", ""),
        "split":           metadata.get("split", ""),
        "true_label":      y_true,
        "predicted_label": y_pred,
        "run_name":        metadata.get("run_name", ""),
        "wandb_run_name":  metadata.get("wandb_run_name", ""),
        "model_type":      metadata.get("model_type", ""),
        "freeze_status":   metadata.get("freeze_status", "n/a"),
        "pretrained_pct":  metadata.get("pretrained_pct", ""),
        "init":            metadata.get("init", ""),
        "ckpt_source":     metadata.get("ckpt_source", ""),
    })
    csv_path = out_dir / filename
    df.to_csv(csv_path, index=False)
    return csv_path


# ── SVM evaluation ─────────────────────────────────────────────────────────────

def run_svm(
    ssl_runs: list[SSLRunInfo],
    output_root: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run SVM evaluation on all SSL encoder runs.

    For each run:
      1. Loads the encoder from the best checkpoint.
      2. Fits a linear SVM on the training split.
      3. Evaluates on the test split.
      4. Saves per-sample predictions CSV.

    Returns:
        DataFrame with per-run results (one row per run).
    """
    # Contrato do protocolo original: LAB[0,1] raw, sem Normalize(ImageNet) —
    # ver artifacts/plots/comparacao_flim_protocolo_original/pipelines.md.
    # ponytail: hardcoded em vez de flag — o braço SVM ajusta e avalia com o
    # mesmo transform, então não há peso treinado exigindo a norma ligada.
    transform = build_test(IMAGE_SIZE, imagenet_norm=False)
    rows: list[dict] = []

    for run_info in ssl_runs:
        print(f"\n{'─' * 68}")
        print(
            f"  [SVM] {run_info.run_name}\n"
            f"        dataset={run_info.dataset_short}  split={run_info.split_id}"
            f"  pct={run_info.pct}  init={run_info.init}"
        )

        base_row = {
            "wandb_run_id":  run_info.run_id,
            "run_name":      run_info.run_name,
            "model_type":    "SVM",
            "dataset":       run_info.dataset_name,
            "dataset_short": run_info.dataset_short,
            "split":         run_info.split_id,
            "pretrained_pct": run_info.pct,
            "init":          run_info.init,
            "freeze_status": "n/a",
            "ckpt_source":   run_info.ckpt_path,
        }

        if dry_run:
            print("  [DRY-RUN] Skipping actual evaluation")
            rows.append({**base_row, "kappa": None, "acc": None, "f1": None,
                         "status": "dry_run", "error": ""})
            continue

        try:
            # Load encoder
            module = LejepaLineModule.load_from_checkpoint(
                run_info.ckpt_path, map_location=DEVICE,
            )
            module.eval()
            encoder = module.model.encoder
            for p in encoder.parameters():
                p.requires_grad_(False)

            path_dataset = (
                run_info.dataset_name if run_info.dataset_name != "parasito" else None
            )

            # Fit SVM on training split
            train_ds = ParasiteDataset(
                set_name="train",
                split=run_info.split_id,
                percentage=run_info.pct,
                transform=transform,
                loader="ift_lab",
                path_dataset=path_dataset,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_ds, run_info.num_classes),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            clf = train_svm(encoder, train_loader)

            # Evaluate on test split
            test_ds = ParasiteDataset(
                set_name="test",
                split=run_info.split_id,
                percentage=run_info.pct,
                transform=transform,
                loader="ift_lab",
                path_dataset=path_dataset,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )

            feats, y_true = extract_features(encoder, test_loader)
            y_pred = clf.predict(feats) - 1  # SVM is 1-indexed → convert to 0-indexed

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred,
                                      num_classes=run_info.num_classes)

            # Save predictions CSV
            out_dir = _artifact_dir(output_root, "SVM", run_info.dataset_short, run_info.pct)
            safe_name = run_info.run_name.replace("/", "_")
            pred_csv = _save_predictions(
                out_dir=out_dir,
                filename=f"predictions_{safe_name}.csv",
                y_true=y_true,
                y_pred=y_pred,
                metadata={**base_row},
            )

            rows.append({**base_row, **metrics, "status": "ok", "error": ""})
            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )
            print(f"  [SAVED]  {pred_csv.relative_to(output_root)}")

            del module, encoder

        except Exception as exc:
            log.exception("SVM failed for %s", run_info.run_name)
            print(f"  [ERROR] {exc}")
            rows.append({**base_row, "kappa": float("nan"), "acc": float("nan"),
                         "f1": float("nan"), "status": "error", "error": str(exc)})

    return pd.DataFrame(rows)


# ── MLP inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _mlp_inference(model, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run inference with a ClassificationModel. Returns (y_true, y_pred)."""
    model.eval()
    model.to(DEVICE)
    all_labels: list[int] = []
    all_preds: list[int] = []

    for imgs, labels in tqdm(test_loader, desc="  Inference", leave=False):
        imgs = imgs.to(DEVICE)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        lbl = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        all_labels.extend(lbl.tolist() if hasattr(lbl, "tolist") else list(lbl))

    return np.array(all_labels, dtype=np.int64), np.array(all_preds, dtype=np.int64)


def run_mlp(
    finetune_runs: list[FinetuneRunInfo],
    output_root: Path,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run MLP inference for all fine-tune runs (freeze and/or unfreeze).

    For each run:
      1. Loads the ClassificationModel architecture from the SSL encoder checkpoint.
      2. Loads the fine-tuned weights from results/mlp_weights/.
      3. Runs inference on the test split.
      4. Saves per-sample predictions CSV.

    Returns:
        DataFrame with per-run results.
    """
    from eval.mlp import load_classification_model  # noqa: PLC0415

    # NÃO desligar imagenet_norm aqui: ao contrário do braço SVM, este caminho é
    # só inferência com pesos já ajustados, e o fine-tune rodou com
    # ParasiteLejepaDataModuleSplited(imagenet_norm=True) (o default, que nenhum
    # config sobrescreve). Desligar criaria descasamento treino/teste.
    transform = build_test(IMAGE_SIZE)
    rows: list[dict] = []

    for run_info in finetune_runs:
        if run_info.skip_reason is not None:
            continue  # already reported in resolver

        mode_str = "freeze" if run_info.freeze else "unfreeze"
        model_type = f"MLP_{mode_str}"

        print(f"\n{'─' * 68}")
        print(
            f"  [{model_type.upper()}] {run_info.encoder_run_name}\n"
            f"        dataset={run_info.dataset_short}  split={run_info.split_id}"
            f"  pct={run_info.pct}  init={run_info.init}"
        )

        base_row = {
            "wandb_run_id":   run_info.wandb_run_id,
            "wandb_run_name": run_info.wandb_run_name,
            "run_name":       run_info.encoder_run_name,
            "model_type":     model_type,
            "dataset":        run_info.dataset_name,
            "dataset_short":  run_info.dataset_short,
            "split":          run_info.split_id,
            "pretrained_pct": run_info.pct,
            "init":           run_info.init,
            "freeze_status":  mode_str,
            "ckpt_source":    run_info.weights_path,
        }

        if dry_run:
            print("  [DRY-RUN] Skipping actual evaluation")
            rows.append({**base_row, "kappa": None, "acc": None, "f1": None,
                         "status": "dry_run", "error": ""})
            continue

        try:
            # Build model from SSL checkpoint architecture, then override weights
            model, ssl_ckpt_path = load_classification_model(
                run_info.encoder_run_id,
                run_info.num_classes,
            )
            print(f"  [INFO] SSL ckpt : {os.path.relpath(ssl_ckpt_path, _ROOT)}")
            print(f"  [INFO] FT  weights : {os.path.relpath(run_info.weights_path, _ROOT)}")

            if not os.path.isfile(run_info.weights_path):
                raise FileNotFoundError(
                    f"Fine-tuned weights not found: {run_info.weights_path}"
                )
            fine_tuned_state = torch.load(run_info.weights_path, map_location="cpu",
                                          weights_only=True)
            print(f"  [INFO] Loading state_dict ({len(fine_tuned_state)} keys, strict=True)")
            model.load_state_dict(fine_tuned_state, strict=True)

            path_dataset = (
                run_info.dataset_name if run_info.dataset_name != "parasito" else None
            )

            test_ds = ParasiteDataset(
                set_name="test",
                split=run_info.split_id,
                percentage=run_info.pct,
                transform=transform,
                loader="ift_lab",
                path_dataset=path_dataset,
            )
            print(f"  [INFO] Test set size : {len(test_ds)} samples  (split={run_info.split_id}  pct={run_info.pct})")
            if len(test_ds) == 0:
                raise ValueError(
                    f"Empty test dataset for split={run_info.split_id} pct={run_info.pct} "
                    f"dataset={run_info.dataset_name}"
                )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )

            y_true, y_pred = _mlp_inference(model, test_loader)
            metrics = compute_metrics(y_true=y_true, y_pred=y_pred,
                                      num_classes=run_info.num_classes)

            # Save predictions CSV
            out_dir = _artifact_dir(output_root, "MLP", run_info.dataset_short, run_info.pct)
            safe_name = run_info.encoder_run_name.replace("/", "_")
            pred_csv = _save_predictions(
                out_dir=out_dir,
                filename=f"predictions_{safe_name}_{mode_str}.csv",
                y_true=y_true,
                y_pred=y_pred,
                metadata={**base_row},
            )

            rows.append({**base_row, **metrics, "status": "ok", "error": ""})
            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )
            print(f"  [SAVED]  {pred_csv.relative_to(output_root)}")

            del model

        except Exception as exc:
            log.exception("MLP inference failed for %s", run_info.encoder_run_name)
            print(f"  [ERROR] {exc}")
            rows.append({**base_row, "kappa": float("nan"), "acc": float("nan"),
                         "f1": float("nan"), "status": "error", "error": str(exc)})

    return pd.DataFrame(rows)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate_and_save(
    df: pd.DataFrame,
    output_root: Path,
) -> pd.DataFrame:
    """Aggregate metrics across splits and save per-group metric CSVs.

    Groups by (model_type, dataset_short, pretrained_pct, init) and computes
    mean ± std across splits.  Saves one small CSV per group inside the
    appropriate artifact directory.

    Args:
        df:          Raw per-run results (may contain SVM and/or MLP rows).
        output_root: Root artifacts directory.

    Returns:
        Aggregated DataFrame (one row per group).
    """
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()

    df_ok = df[df["status"] == "ok"].copy()
    if df_ok.empty:
        print("[AGGREGATE] No successful rows — nothing to aggregate.")
        return pd.DataFrame()

    agg_rows: list[dict] = []
    group_cols = ["model_type", "dataset_short", "pretrained_pct", "init"]

    for group_keys, group_df in df_ok.groupby(group_cols, dropna=False):
        model_type, dataset_short, pct, init = group_keys
        method_prefix = "SVM" if model_type == "SVM" else "MLP"

        row: dict = {
            "model_type":    model_type,
            "dataset_short": dataset_short,
            "pretrained_pct": pct,
            "init":          init,
            "n_splits":      len(group_df),
        }

        for metric in ["kappa", "acc", "f1"]:
            vals = group_df[metric].dropna().values
            row[metric]            = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            row[f"{metric}_std"]   = float(np.std(vals))  if len(vals) > 1 else 0.0

        if row["n_splits"] < 3:
            print(
                f"[AGGREGATE] WARN: {model_type} {dataset_short} pct={pct} init={init} "
                f"has only {row['n_splits']}/3 split(s) — results will be partial"
            )

        agg_rows.append(row)

        # Save group-level metrics CSV inside the artifact directory
        out_dir = _artifact_dir(output_root, method_prefix, str(dataset_short), int(pct))
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = out_dir / f"metrics_{model_type}_{init}.csv"
        pd.DataFrame([row]).to_csv(metrics_csv, index=False)

    agg_df = pd.DataFrame(agg_rows)

    # Save one aggregated CSV per method prefix — merge with existing so that
    # running --dataset protozoan does not erase eggs/larvae rows.
    for prefix in ("SVM", "MLP"):
        sub = agg_df[agg_df["model_type"].str.startswith(prefix)] if not agg_df.empty else pd.DataFrame()
        if not sub.empty:
            agg_csv = output_root / prefix / f"{prefix.lower()}_aggregated.csv"
            agg_csv.parent.mkdir(parents=True, exist_ok=True)
            if agg_csv.exists():
                existing = pd.read_csv(agg_csv)
                merge_keys = ["model_type", "dataset_short", "pretrained_pct", "init"]
                # Drop rows from existing that are being updated by current run
                mask = existing.set_index(merge_keys).index.isin(sub.set_index(merge_keys).index)
                existing = existing[~mask]
                sub = pd.concat([existing, sub], ignore_index=True)
                print(f"[AGGREGATE] Merged with existing {agg_csv.name} ({len(existing)} kept + {len(agg_df)} new)")
            sub.to_csv(agg_csv, index=False)
            print(f"[AGGREGATE] {agg_csv.relative_to(output_root)}")

    return agg_df


# ── Plot generation ────────────────────────────────────────────────────────────

def generate_plots(agg_df: pd.DataFrame, output_root: Path) -> None:
    """Generate composite and line plots from aggregated metrics.

    Per (dataset, pct): composite bar chart with kappa, accuracy, F1.
    Per (dataset, method): kappa vs. percentage line plot (plot_svm_results style).

    Args:
        agg_df:      Output of aggregate_and_save().
        output_root: Root artifacts directory.
    """
    if agg_df.empty:
        print("[PLOT] No aggregated data — skipping plots.")
        return

    plots_root = output_root / "plots"

    # Map model_type → method label for plotting
    agg_df = agg_df.copy()
    agg_df["method"] = agg_df["model_type"]  # "SVM", "MLP_freeze", "MLP_unfreeze"

    datasets = agg_df["dataset_short"].unique()
    pcts = sorted(agg_df["pretrained_pct"].unique())

    print(f"\n[PLOT] Generating composite plots for {len(datasets)} dataset(s) × {len(pcts)} pct(s)...")

    for dataset_short in datasets:
        ds_df = agg_df[agg_df["dataset_short"] == dataset_short]

        # ── Composite plot per (dataset, pct) ─────────────────────────────
        for pct in pcts:
            pct_df = ds_df[ds_df["pretrained_pct"] == pct]
            if pct_df.empty:
                continue
            out_dir = plots_root / dataset_short / f"lejepa_pct_{pct}"
            plot_composite(
                df=pct_df,
                dataset=dataset_short,
                pct=int(pct),
                out_dir=out_dir,
            )

        # ── Line plot per (dataset, method, metric) ────────────────────────
        for method in agg_df["method"].unique():
            method_df = ds_df[ds_df["method"] == method]
            if method_df.empty:
                continue
            line_out_dir = plots_root / dataset_short / "all_pcts"
            for metric in ["kappa", "acc", "f1"]:
                plot_metric_vs_pct(
                    df=method_df,
                    dataset=dataset_short,
                    method=method,
                    metric=metric,
                    out_dir=line_out_dir,
                )


# ── Metadata manifest ──────────────────────────────────────────────────────────

def _save_run_manifest(
    svm_runs: list[SSLRunInfo],
    finetune_runs: list[FinetuneRunInfo],
    output_root: Path,
) -> None:
    """Save a manifest of all discovered runs (usable + skipped) for traceability."""
    manifest_rows: list[dict] = []

    for r in svm_runs:
        manifest_rows.append({
            "phase": "SVM",
            "run_name": r.run_name,
            "wandb_run_id": r.run_id,
            "dataset_short": r.dataset_short,
            "split": r.split_id,
            "pct": r.pct,
            "init": r.init,
            "skip_reason": "",
        })

    for r in finetune_runs:
        manifest_rows.append({
            "phase": f"MLP_{'freeze' if r.freeze else 'unfreeze'}",
            "run_name": r.encoder_run_name or r.wandb_run_name,
            "wandb_run_id": r.wandb_run_id,
            "dataset_short": r.dataset_short,
            "split": r.split_id,
            "pct": r.pct,
            "init": r.init,
            "skip_reason": r.skip_reason or "",
        })

    if manifest_rows:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "run_manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"[MANIFEST] {manifest_path}")


# ── Resume helpers ─────────────────────────────────────────────────────────────

def _load_done_svm(svm_csv: Path) -> set[str]:
    """Return set of wandb_run_ids already successfully evaluated as SVM."""
    if not svm_csv.exists():
        return set()
    df = pd.read_csv(svm_csv)
    if df.empty or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "ok", "wandb_run_id"].astype(str).tolist())


def _load_done_mlp(mlp_csv: Path) -> set[tuple[str, str]]:
    """Return set of (run_name, model_type) pairs already evaluated as MLP."""
    if not mlp_csv.exists():
        return set()
    df = pd.read_csv(mlp_csv)
    if df.empty or "status" not in df.columns:
        return set()
    ok = df[df["status"] == "ok"]
    return set(zip(ok["run_name"].astype(str), ok["model_type"].astype(str)))


def _merge_and_save_csv(new_df: pd.DataFrame, csv_path: Path, merge_key: str = "wandb_run_id") -> None:
    """Append *new_df* to *csv_path*, replacing existing rows with the same key."""
    if csv_path.exists() and not new_df.empty:
        existing = pd.read_csv(csv_path)
        mask = existing[merge_key].astype(str).isin(new_df[merge_key].astype(str))
        existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(csv_path, index=False)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(
    model: str = "all",
    mode: Optional[str] = None,
    dataset: str = "all",
    entity: Optional[str] = None,
    project: Optional[str] = None,
    output_root: str = str(Path(_ROOT) / "artifacts"),
    dry_run: bool = False,
    verbose: bool = True,
    split: Optional[int] = None,
    pct: Optional[int] = None,
    init: Optional[str] = None,
    max_runs: Optional[int] = None,
    embed_mode: str = "avgpool2d",
    resume: bool = False,
    restart: bool = False,
    resume_id: Optional[str] = None,
) -> None:
    """Unified evaluation pipeline for LEJEPA encoders.

    Cada parametro abaixo era uma flag do argparse; o default e o mesmo de antes.

    Args:
        model:       svm | mlp_freeze | mlp_unfreeze | all (legado; `mode` tem
                     precedencia quando informado).
        mode:        svm | freeze | unfreeze | all. Tem precedencia sobre `model`.
                     'freeze'/'unfreeze' mapeiam para mlp_freeze/mlp_unfreeze.
        dataset:     eggs | protozoan | larvae | all. Filtro de dataset.
        entity:      W&B entity (default: de get_names_wandb).
        project:     W&B project (default: de get_names_wandb).
        output_root: Diretorio raiz de todos os artefatos.
        dry_run:     Descobre e lista runs, mas pula a avaliacao de fato.
        verbose:     Verbosidade repassada ao resolver.
        split:       Restringe a um indice de split.
        pct:         Restringe a um percentual de pre-treino.
        init:        flim | he | xavier | random | trunc_normal | normal.
                     'trunc_normal' seleciona so runs trunc_normal_std;
                     'normal' seleciona todos os nao-trunc_normal
                     (xavier/he/random/flim). Omitir inclui todos.
        max_runs:    Teto de runs utilizaveis (para debug).
        embed_mode:  avgpool2d | flatten. Como o mapa conv3 vira embedding do SVM.
                     'avgpool2d' = AdaptiveAvgPool2d(1) → 48-d (default).
                     'flatten' = mapa conv3 achatado → 27648-d (protocolo original).
        resume:      Acrescenta as planilhas ja existentes em artifacts/;
                     experimentos com status=ok sao pulados.
        restart:     Limpa resultados anteriores do escopo do filtro atual e
                     regenera do zero. Tem precedencia sobre `resume`.
        resume_id:   Restringe a execucao a um W&B run id (run do encoder SSL).
                     No SVM casa com o run_id do encoder; no MLP casa com
                     encoder_run_id ou com o wandb_run_id do fine-tune.
    """
    logging.basicConfig(level=logging.WARNING)

    _ev.EMBED_MODE = embed_mode  # antes de qualquer extract_features()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Resolve effective evaluation mode ──────────────────────────────────────
    # --mode (new) takes precedence over --model (legacy)
    if mode is not None:
        _mode_map = {"svm": "svm", "freeze": "mlp_freeze",
                     "unfreeze": "mlp_unfreeze", "all": "all"}
        effective_model = _mode_map[mode]
        print(f"[CLI] --mode {mode!r} → effective model={effective_model!r}")
    else:
        effective_model = model

    dataset_filter = None if dataset == "all" else dataset

    # ── Resolve init filter ────────────────────────────────────────────────────
    # "normal" is a meta-value meaning "any init that is not trunc_normal"
    _raw_init = init
    _exclude_trunc_norm = (_raw_init == "normal")
    init_filter_for_resolver = None if (_raw_init is None or _raw_init == "normal") else _raw_init

    # ── Parameter precedence logging ───────────────────────────────────────────
    if restart and resume:
        print("[WARN] --restart and --resume both given; --restart takes precedence — "
              "running all in-scope experiments and replacing their rows")
    if resume_id:
        print(f"[RESUME_ID] Execution restricted to W&B run id: {resume_id!r}")
    if resume and not restart:
        print("[RESUME] Will skip already-completed experiments and append new results")
    if restart:
        print("[RESTART] Prior results for the current scope will be replaced")

    # Execution order per SDD 5.2: svm → mlp_unfreeze → mlp_freeze
    if effective_model == "all":
        ordered_modes = ["svm", "mlp_unfreeze", "mlp_freeze"]
    else:
        ordered_modes = [effective_model]

    resolver_kwargs = dict(
        dataset_filter=dataset_filter,
        pct_filter=pct,
        init_filter=init_filter_for_resolver,
        split_filter=split,
        verbose=verbose,
    )
    if entity:
        resolver_kwargs["entity"] = entity
    if project:
        resolver_kwargs["project"] = project

    all_results: list[pd.DataFrame] = []
    svm_runs_discovered: list[SSLRunInfo] = []
    finetune_runs_discovered: list[FinetuneRunInfo] = []

    svm_csv = output_root / "SVM" / "svm_results.csv"
    mlp_csv = output_root / "MLP" / "mlp_results.csv"

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: SVM
    # ══════════════════════════════════════════════════════════════════════════
    if "svm" in ordered_modes:
        print(f"\n{'═' * 70}")
        print("  PHASE 1 — SVM EVALUATION")
        print(f"{'═' * 70}")

        ssl_runs = resolve_ssl_runs(**resolver_kwargs)

        # Post-filter: "normal" init means exclude trunc_norm
        if _exclude_trunc_norm:
            ssl_runs = [r for r in ssl_runs if r.init != "trunc_normal"]
            print(f"[INIT] --init normal: excluded trunc_norm runs, {len(ssl_runs)} remaining")

        # --resume_id: restrict to a single encoder run
        if resume_id:
            ssl_runs = [r for r in ssl_runs if r.run_id == resume_id]
            print(f"[RESUME_ID] SVM: {len(ssl_runs)} run(s) match run_id={resume_id!r}")

        svm_runs_discovered = ssl_runs

        # --resume: skip already-done runs (unless --restart overrides)
        if resume and not restart:
            done_ids = _load_done_svm(svm_csv)
            before = len(ssl_runs)
            ssl_runs = [r for r in ssl_runs if r.run_id not in done_ids]
            skipped_count = before - len(ssl_runs)
            if skipped_count:
                print(f"[RESUME] SVM: skipping {skipped_count} already-completed run(s), "
                      f"{len(ssl_runs)} remaining")

        if max_runs is not None:
            ssl_runs = ssl_runs[: max_runs]

        print(f"[SVM] Evaluating {len(ssl_runs)} SSL encoder run(s)...")
        svm_df = run_svm(ssl_runs, output_root, dry_run=dry_run)

        if not svm_df.empty:
            svm_csv.parent.mkdir(parents=True, exist_ok=True)
            if resume or restart:
                # Merge: replace rows for current run_ids, keep all others
                _merge_and_save_csv(svm_df, svm_csv, merge_key="wandb_run_id")
            else:
                svm_df.to_csv(svm_csv, index=False)
            print(f"\n[SVM] Full results → {svm_csv.relative_to(output_root)}")
            all_results.append(svm_df)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: MLP (unfreeze first, then freeze — SDD 5.2)
    # ══════════════════════════════════════════════════════════════════════════
    mlp_modes = [m for m in ordered_modes if m.startswith("mlp")]
    if mlp_modes:
        phase_num = 2 if "svm" in ordered_modes else 1
        print(f"\n{'═' * 70}")
        print(f"  PHASE {phase_num} — MLP EVALUATION")
        print(f"{'═' * 70}")

        # Determine freeze filter
        if len(mlp_modes) == 1:
            freeze_filter: Optional[bool] = (
                False if mlp_modes[0] == "mlp_unfreeze" else True
            )
        else:
            freeze_filter = None

        ft_kwargs = {**resolver_kwargs, "freeze_filter": freeze_filter}
        finetune_runs = resolve_finetune_runs(**ft_kwargs)

        # Post-filter: "normal" init means exclude trunc_norm
        if _exclude_trunc_norm:
            finetune_runs = [r for r in finetune_runs if r.init != "trunc_normal"]
            print(f"[INIT] --init normal: excluded trunc_norm MLP runs, {len(finetune_runs)} remaining")

        # --resume_id: restrict to runs whose encoder matches the given id
        if resume_id:
            finetune_runs = [
                r for r in finetune_runs
                if r.encoder_run_id == resume_id or r.wandb_run_id == resume_id
            ]
            print(f"[RESUME_ID] MLP: {len(finetune_runs)} run(s) match run_id={resume_id!r}")

        finetune_runs_discovered = finetune_runs

        # --resume: skip already-done (run_name, model_type) pairs
        if resume and not restart:
            done_mlp = _load_done_mlp(mlp_csv)
            before = len([r for r in finetune_runs if r.skip_reason is None])
            finetune_runs = [
                r for r in finetune_runs
                if r.skip_reason is not None
                or (r.encoder_run_name, f"MLP_{'freeze' if r.freeze else 'unfreeze'}") not in done_mlp
            ]
            after = len([r for r in finetune_runs if r.skip_reason is None])
            skipped_count = before - after
            if skipped_count:
                print(f"[RESUME] MLP: skipping {skipped_count} already-completed run(s), "
                      f"{after} remaining")

        # Enforce execution order: unfreeze before freeze (SDD 5.2)
        usable = [r for r in finetune_runs if r.skip_reason is None]
        skipped = [r for r in finetune_runs if r.skip_reason is not None]
        unfreeze_first = sorted(usable, key=lambda r: (r.freeze, r.dataset_short, r.pct, r.split_id))

        if max_runs is not None:
            unfreeze_first = unfreeze_first[: max_runs]

        ordered_runs = unfreeze_first + skipped

        n_usable = len(unfreeze_first)
        print(f"[MLP] Evaluating {n_usable} fine-tune run(s)...")
        mlp_df = run_mlp(ordered_runs, output_root, dry_run=dry_run)

        if not mlp_df.empty:
            mlp_csv.parent.mkdir(parents=True, exist_ok=True)
            if resume or restart:
                # Merge: for MLP, identity key is (run_name, model_type) — use run_name column
                _merge_and_save_csv(mlp_df, mlp_csv, merge_key="run_name")
            else:
                mlp_df.to_csv(mlp_csv, index=False)
            print(f"\n[MLP] Full results → {mlp_csv.relative_to(output_root)}")
            all_results.append(mlp_df)

    # ══════════════════════════════════════════════════════════════════════════
    # Aggregation, plotting, manifest
    # ══════════════════════════════════════════════════════════════════════════
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        print(f"\n{'═' * 70}")
        print("  AGGREGATING METRICS")
        print(f"{'═' * 70}")
        agg_df = aggregate_and_save(combined, output_root)

        print(f"\n{'═' * 70}")
        print("  GENERATING PLOTS")
        print(f"{'═' * 70}")
        generate_plots(agg_df, output_root)
    else:
        agg_df = pd.DataFrame()
        print("\n[INFO] No results to aggregate (dry-run or all runs skipped).")

    # Save run manifest for traceability
    _save_run_manifest(svm_runs_discovered, finetune_runs_discovered, output_root)

    print(f"\n{'═' * 70}")
    print(f"[DONE] All artifacts saved under: {output_root}")
    print(f"{'═' * 70}")

    # Brief execution summary
    if not agg_df.empty:
        ok_counts = combined[combined["status"] == "ok"]["model_type"].value_counts().to_dict()  # type: ignore[union-attr]
        err_counts = combined[combined["status"] == "error"]["model_type"].value_counts().to_dict()  # type: ignore[union-attr]
        print("\n=== EXECUTION SUMMARY ===")
        for mt in sorted(set(list(ok_counts) + list(err_counts))):
            print(f"  {mt:20s}  ok={ok_counts.get(mt, 0):3d}  err={err_counts.get(mt, 0):3d}")


if __name__ == "__main__":
    import sys
    main(**cli_kwargs(sys.argv[1:]))
