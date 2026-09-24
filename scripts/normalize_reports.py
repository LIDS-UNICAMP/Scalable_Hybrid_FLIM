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

"""normalize_reports.py — Normalize legacy report CSVs into the artifacts schema.

Reads:
  - data/reports_felipe/svm/*.csv       (FLIM SVM, no LeJEPA pretraining)
  - artifacts/SVM/*/metrics_SVM_*.csv   (LeJEPA-pretrained SVM)
  - results/ijepa_svm_aggregated.csv    (I-JEPA SVM)

Writes:
  - artifacts/normalized/svm_flim_aggregated.csv      (normalized Felipe SVM)
  - artifacts/normalized/unified_svm_comparison.csv   (all 3 methods)

Generates plots (one per dataset × metric):
  - artifacts/plots/comparison/{dataset}_{metric}_comparison.{pdf,png}

Usage:
    python scripts/normalize_reports.py
    python scripts/normalize_reports.py --partial   # skip missing sources instead of failing
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless-safe matplotlib

# scripts/ é o sys.path[0] quando se roda `python scripts/normalize_reports.py`;
# tem de vir antes do bloco abaixo, que usa PROJECT_ROOT.
from constants import (  # noqa: E402
    ARTIFACTS_NORMALIZED_DIR,
    ARTIFACTS_PLOTS_DIR,
    CANONICAL_COLS as _CANONICAL_COLS,
    DATASET_ALIASES as DATASET_MAP,
    DATASET_LONG_TO_SHORT as _LONG_DATASET_MAP,
    METRICS,
    PROJECT_ROOT as _ROOT_STR,
    REPORTS_FELIPE_SVM_DIR,
    RESULTS_DIR,
    UNIFIED_SVM_COMPARISON_CSV,
)

# Allow running directly: python scripts/normalize_reports.py
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

import numpy as np
import pandas as pd

_ROOT = Path(_ROOT_STR)
_RESULTS = Path(RESULTS_DIR)

# ── Constants ──────────────────────────────────────────────────────────────────

_FELIPE_SVM_RE = re.compile(
    r"^report_svm_([a-z]+)_split(\d+)_perc(\d+)\.csv$"
)


# ── Source 1: reports_felipe/svm ──────────────────────────────────────────────

def normalize_felipe_svm() -> pd.DataFrame:
    """Read all reports_felipe/svm CSVs, aggregate across splits.

    Filename pattern: report_svm_{dataset}_split{N}_perc{pct}.csv
    Metrics: test_cohen_kappa, test_accuracy, test_f1_weighted
    Returns canonical DataFrame with method=SVM_FLIM, init=flim.
    """
    svm_dir = Path(REPORTS_FELIPE_SVM_DIR)
    raw_rows: list[dict] = []

    for csv_file in sorted(svm_dir.glob("*.csv")):
        m = _FELIPE_SVM_RE.match(csv_file.name)
        if not m:
            continue
        dataset_raw, split_str, pct_str = m.group(1), m.group(2), m.group(3)
        dataset_short = DATASET_MAP.get(dataset_raw)
        if dataset_short is None:
            print(f"  [WARN] Unknown dataset '{dataset_raw}' in {csv_file.name}, skipping")
            continue

        df = pd.read_csv(csv_file)
        # SVM reports may have multiple rows (e.g. different encoder_mode); take frozen
        df = df[df["encoder_mode"] == "frozen"]
        if df.empty:
            continue

        row = df.iloc[0]
        try:
            raw_rows.append({
                "dataset_short": dataset_short,
                "split": int(split_str),
                "pretrained_pct": int(pct_str),
                "kappa": float(row["test_cohen_kappa"]),
                "acc":   float(row["test_accuracy"]),
                "f1":    float(row["test_f1_weighted"]),
            })
        except (ValueError, KeyError) as exc:
            print(f"  [WARN] Skipping {csv_file.name}: {exc}")

    if not raw_rows:
        raise RuntimeError(
            f"No valid reports_felipe/svm CSV files found under {svm_dir}"
        )

    raw_df = pd.DataFrame(raw_rows)

    # Aggregate mean ± std across splits
    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in raw_df.groupby(
        ["dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":        "SVM_FLIM",
            "init":          "flim",
            "dataset_short": dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":      n,
            "kappa":         grp["kappa"].mean(),
            "kappa_std":     grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":           grp["acc"].mean(),
            "acc_std":       grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":            grp["f1"].mean(),
            "f1_std":        grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_FLIM  : {len(result)} rows from {len(raw_rows)} split-level files")
    return result


# ── Source 2: artifacts/SVM metrics (already aggregated) ─────────────────────

def load_lejepa_svm_metrics() -> pd.DataFrame:
    """Read all metrics_SVM_*.csv from artifacts/SVM/.

    Schema: model_type, dataset_short, pretrained_pct, init,
            n_splits, kappa, kappa_std, acc, acc_std, f1, f1_std
    Returns canonical DataFrame with method=SVM_lejepa_view.

    Nota: a curva se chamava `SVM_LeJEPA` (legenda "LeJEPA (59.504)"); foi
    renomeada para `SVM_lejepa_view` / legenda `lejepa_view`. Consumidores do CSV
    unificado (plot_comparison_flim.py, plot_parameters_vs_metrics.py,
    statistics/tools/wilcoxon_*.py) já usam a chave nova.
    """
    artifacts_svm = _ROOT / "artifacts" / "SVM"
    rows: list[dict] = []

    for metrics_csv in sorted(artifacts_svm.glob("*/*/metrics_SVM_*.csv")):
        df = pd.read_csv(metrics_csv)
        if df.empty:
            continue
        row = df.iloc[0]
        try:
            rows.append({
                "method":        "SVM_lejepa_view",
                "init":          str(row["init"]),
                "dataset_short": str(row["dataset_short"]),
                "pretrained_pct": int(row["pretrained_pct"]),
                "n_splits":      int(row["n_splits"]),
                "kappa":         float(row["kappa"]),
                "kappa_std":     float(row["kappa_std"]),
                "acc":           float(row["acc"]),
                "acc_std":       float(row["acc_std"]),
                "f1":            float(row["f1"]),
                "f1_std":        float(row["f1_std"]),
            })
        except (ValueError, KeyError) as exc:
            print(f"  [WARN] Skipping {metrics_csv}: {exc}")

    if not rows:
        return None  # caller handles missing gracefully

    result = pd.DataFrame(rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_lejepa_view: {len(result)} rows from artifacts/SVM/")
    return result


# ── Source 3: ijepa_svm_aggregated.csv ────────────────────────────────────────

def normalize_ijepa_svm() -> pd.DataFrame:
    """Normalize results/ijepa_svm_aggregated.csv to canonical schema.

    Source columns: dataset, dataset_short, pretrained_pct, n_splits,
                    embedding, kappa, kappa_std, acc, acc_std, f1, f1_std
    Returns canonical DataFrame with method=SVM_IJEPA, init=ijepa.
    """
    src = _RESULTS / "ijepa_svm_aggregated.csv"
    if not src.exists():
        raise FileNotFoundError(f"ijepa_svm_aggregated.csv not found at {src}")

    df = pd.read_csv(src)

    result = pd.DataFrame({
        "method":        "SVM_IJEPA",
        "init":          "ijepa",
        "dataset_short": df["dataset_short"],
        "pretrained_pct": df["pretrained_pct"].astype(int),
        "n_splits":      df["n_splits"].astype(int),
        "kappa":         df["kappa"],
        "kappa_std":     df["kappa_std"],
        "acc":           df["acc"],
        "acc_std":       df["acc_std"],
        "f1":            df["f1"],
        "f1_std":        df["f1_std"],
    })
    print(f"[NORM] SVM_IJEPA : {len(result)} rows from results/ijepa_svm_aggregated.csv")
    return result[_CANONICAL_COLS]


# ── Source 4: svm_distillation_conv_results.csv ───────────────────────────────

def normalize_distillation_conv_svm() -> pd.DataFrame | None:
    """Normalize results/svm_distillation_conv_results.csv to canonical schema.

    Aggregates mean ± std across splits per (dataset, pretrained_pct).
    Returns canonical DataFrame with method=SVM_Distillation_Conv, init=trunc_normal.
    Returns None if the file does not exist yet.
    """
    src = _RESULTS / "svm_distillation_conv_results.csv"
    if not src.exists():
        print(f"[SKIP] svm_distillation_conv_results.csv not found — skipping distillation conv")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_distillation_conv_results.csv has no ok rows")
        return None

    df["dataset_short"] = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distillation_Conv",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distillation_Conv: {len(result)} rows from results/svm_distillation_conv_results.csv")
    return result


# ── Source 4b: svm_distillation_conv_flim_frozen_results.csv ──────────────────

def _normalize_distill_flim_frozen() -> pd.DataFrame | None:
    """Normalize results/svm_distillation_conv_flim_frozen_results.csv.

    Keeps the per-checkpoint method names from the CSV
    (SVM_Distill_1x1BN_flim_frozen_eval_knn / _eval_loss), aggregating mean ± std
    across splits per (method, dataset, pretrained_pct). Skips non-ok / NaN rows
    (partial runs are fine — whatever is available gets plotted). Returns None if
    the file is absent.
    """
    src = _RESULTS / "svm_distillation_conv_flim_frozen_results.csv"
    if not src.exists():
        print("[SKIP] svm_distillation_conv_flim_frozen_results.csv not found — "
              "rode svm_distillation_conv.py --run-filter 1x1_BN2d_1280_flim_frozen "
              "--output-csv svm_distillation_conv_flim_frozen_results primeiro")
        return None

    df = pd.read_csv(src)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = df[df["kappa"].notna()].copy()  # ignore NaN
    if df.empty:
        print("[SKIP] svm_distillation_conv_flim_frozen_results.csv has no usable rows")
        return None

    df["dataset_short"] = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (method, dataset_short, pct), grp in df.groupby(
        ["method", "dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":         method,
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] flim_frozen (eval_knn/eval_loss): {len(result)} rows "
          f"from results/svm_distillation_conv_flim_frozen_results.csv")
    return result


# ── Source 5: svm_distill_proj1280_results.csv ────────────────────────────────

def _normalize_distill_proj1280() -> pd.DataFrame | None:
    """Normalize results/svm_distill_proj1280_results.csv — embedding [B,1280].

    method=SVM_Distill_Proj1280, init=trunc_normal.
    Returns None se o arquivo ainda não existir.
    """
    src = _RESULTS / "svm_distill_proj1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_distill_proj1280_results.csv não encontrado — rode svm_distill_with_projection.py primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_distill_proj1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_Proj1280",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_Proj1280: {len(result)} rows from results/svm_distill_proj1280_results.csv")
    return result


# ── Source 6: svm_proj1280_3x3_BN2d_results.csv ──────────────────────────────

def _normalize_distill_3x3bn() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_3x3_BN2d_results.csv — proj head 3x3 BN2d.

    method=SVM_Distill_3x3BN, init=trunc_normal.
    Returns None se o arquivo ainda não existir.
    """
    src = _RESULTS / "svm_proj1280_3x3_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 3x3_BN2d_1280_one_layer primeiro")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "trunc_normal")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv sem rows ok (trunc_normal)")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_3x3BN",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_3x3BN: {len(result)} rows from results/svm_proj1280_3x3_BN2d_results.csv")
    return result


# ── Source 7: svm_proj1280_1x1_BN2d_results.csv ──────────────────────────────

def _normalize_distill_1x1bn() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_1x1_BN2d_results.csv — proj head 1x1 BN2d.

    method=SVM_Distill_1x1BN, init=trunc_normal.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 1x1_BN2d_1280_one_layer --output-csv svm_proj1280_1x1_BN2d_results primeiro")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "trunc_normal")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv sem rows ok (trunc_normal)")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN: {len(result)} rows from results/svm_proj1280_1x1_BN2d_results.csv")
    return result


def _normalize_distill_1x1bn_nonorm() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv.

    Proj head 1x1 BN2d (126k), FLIM init, treinado/avaliado SEM ImageNet norm.
    method=SVM_Distill_1x1BN_nonorm, init=flim. Linha separada para comparar
    "com norm" (SVM_Distill_1x1BN) vs "sem norm".
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv não encontrado "
              "— rode svm_distill_with_projection.py --run-filter 1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm "
              "--no-imagenet-norm --only-ok primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] ...no_imagenet_norm... sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN_nonorm",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN_nonorm: {len(result)} rows")
    return result


# ── Source 8b: svm_2l_1x1_init_flim_256_1280_results.csv ────────────────────

def _normalize_distill_2l_400k_flim() -> pd.DataFrame | None:
    """Normalize results/svm_2l_1x1_init_flim_256_1280_results.csv.

    method=SVM_Distill_2l400K_flim, init=flim.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_2l_1x1_init_flim_256_1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_results.csv não encontrado")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K_flim: {len(result)} rows from results/svm_2l_1x1_init_flim_256_1280_results.csv")
    return result


# ── Source 8c: svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv ─────────────

def _normalize_distill_2l_400k_flim_nonorm() -> pd.DataFrame | None:
    """Normalize results/svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv.

    method=SVM_Distill_2l400K_flim_nonorm, init=flim. Backbone init-FLIM 400k
    treinado com --no-imagenet-norm, avaliado COM a projetora ativa (proj_kd →
    1280d) — mesma régua do baseline trunc SVM_Distill_2l400K, corrigindo o
    ruler mismatch (antes era o encoder 48d puro).
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv"
    if not src.exists():
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv não encontrado")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K_flim_nonorm",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K_flim_nonorm: {len(result)} rows from results/svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv")
    return result


# ── Source 7b: svm_1x1_BN2d_1280_one_layer_flim_init_results.csv ─────────────

def _normalize_distill_1x1bn_flim() -> pd.DataFrame | None:
    """Normalize encoder_init=flim rows from results/svm_proj1280_1x1_BN2d_results.csv.

    method=SVM_Distill_1x1BN_flim, init=flim.
    Reads the same source as _normalize_distill_1x1bn() but filters encoder_init=flim,
    avoiding dependence on a separate split file that rsync can overwrite.
    Returns None se o arquivo não existir ou não tiver rows flim ok.
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv não encontrado (1x1BN flim)")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "flim")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv sem rows ok com encoder_init=flim")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN_flim: {len(result)} rows from results/svm_proj1280_1x1_BN2d_results.csv (flim only)")
    return result


# ── Source 7c: svm_3x3_BN2d_1280_one_layer_flim_init_results.csv ─────────────

def _normalize_distill_3x3bn_flim() -> pd.DataFrame | None:
    """Normalize encoder_init=flim rows from results/svm_proj1280_3x3_BN2d_results.csv.

    method=SVM_Distill_3x3BN_flim, init=flim.
    Reads the same source as _normalize_distill_3x3bn() but filters encoder_init=flim,
    avoiding dependence on a separate split file that rsync can overwrite.
    Returns None se o arquivo não existir ou não tiver rows flim ok.
    """
    src = _RESULTS / "svm_proj1280_3x3_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv não encontrado (3x3BN flim)")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "flim")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv sem rows ok com encoder_init=flim")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_3x3BN_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_3x3BN_flim: {len(result)} rows from results/svm_proj1280_3x3_BN2d_results.csv (flim only)")
    return result


# ── Source 8: svm_proj1280_2l_1x1_BN2d_256_1280_results.csv ──────────────────

def _normalize_distill_2l_400k() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv — proj head 2l 1x1 BN2d (48→256→1280, ~400K).

    method=SVM_Distill_2l400K, init=trunc_normal.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_proj1280_2l_1x1_BN2d_256_1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_2l_1x1_BN2d_256_1280_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 2l_1x1_BN2d_256_1280 primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_2l_1x1_BN2d_256_1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K: {len(result)} rows from results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv")
    return result


# ── Source 9: svm_flim_residual_eggs.csv ─────────────────────────────────────

def _normalize_flim_residual() -> pd.DataFrame | None:
    """Normalize results/svm_flim_residual_eggs.csv — FLIM residual encoders (eggs).

    Keeps the per-variant method names already present in the CSV
    (SVM_FLIMResidual_1_3 / SVM_FLIMResidual_2_3), aggregating mean ± std across
    splits per (method, dataset, pretrained_pct). Maps dataset_name
    (helminth-eggs) to the canonical short name (eggs). Skips non-ok / NaN rows.
    Returns None if the file is absent.
    """
    src = _RESULTS / "svm_flim_residual_eggs.csv"
    if not src.exists():
        print("[SKIP] svm_flim_residual_eggs.csv não encontrado — "
              "rode svm_flim_residual.py --flim_residual_assessment primeiro")
        return None

    df = pd.read_csv(src)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = df[df["kappa"].notna()].copy()  # ignore NaN
    if df.empty:
        print("[SKIP] svm_flim_residual_eggs.csv has no usable rows")
        return None

    df["dataset_short"]  = df["dataset_name"].map(
        lambda d: _LONG_DATASET_MAP.get(d, d)
    )
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (method, dataset_short, pct), grp in df.groupby(
        ["method", "dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":         method,
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] FLIM residual (1_3/2_3): {len(result)} rows "
          f"from results/svm_flim_residual_eggs.csv")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def _safe(fn, partial: bool):
    """Call fn(); on failure return None if partial=True, else re-raise."""
    try:
        result = fn()
        return result
    except Exception as exc:
        if partial:
            print(f"[SKIP] {fn.__name__}: {exc}")
            return None
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize SVM result CSVs into unified_svm_comparison.csv.",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Skip missing data sources instead of failing. "
             "Useful when artifacts/SVM/ or other files are not available locally.",
    )
    args = parser.parse_args()
    partial = args.partial

    if partial:
        print("[MODE] --partial: missing sources will be skipped.\n")

    from src.evaluate.eval_plotter import plot_method_comparison  # noqa: PLC0415

    out_dir = Path(ARTIFACTS_NORMALIZED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(ARTIFACTS_PLOTS_DIR) / "comparison"

    # ── Normalize each source (all optional in --partial mode) ────────────────
    flim_df           = _safe(normalize_felipe_svm, partial)
    lejepa_df         = _safe(load_lejepa_svm_metrics, partial)
    ijepa_df          = _safe(normalize_ijepa_svm, partial)
    distil_conv_df    = _safe(normalize_distillation_conv_svm, partial)
    distil_flim_frozen_df = _safe(_normalize_distill_flim_frozen, partial)
    distil_proj_df    = _safe(_normalize_distill_proj1280, partial)
    distil_3x3bn_df   = _safe(_normalize_distill_3x3bn, partial)
    distil_1x1bn_df      = _safe(_normalize_distill_1x1bn, partial)
    distil_1x1bn_nonorm_df = _safe(_normalize_distill_1x1bn_nonorm, partial)
    distil_1x1bn_flim_df = _safe(_normalize_distill_1x1bn_flim, partial)
    distil_3x3bn_flim_df = _safe(_normalize_distill_3x3bn_flim, partial)
    distil_2l400k_df      = _safe(_normalize_distill_2l_400k, partial)
    distil_2l400k_flim_df = _safe(_normalize_distill_2l_400k_flim, partial)
    distil_2l400k_flim_nonorm_df = _safe(_normalize_distill_2l_400k_flim_nonorm, partial)
    flim_residual_df  = _safe(_normalize_flim_residual, partial)

    # ── Save normalized FLIM SVM aggregated (if available) ────────────────────
    if flim_df is not None:
        flim_path = out_dir / "svm_flim_aggregated.csv"
        flim_df.to_csv(flim_path, index=False)
        print(f"[SAVE] {flim_path.relative_to(_ROOT)}  ({len(flim_df)} rows)")

    # ── Build and save unified comparison ─────────────────────────────────────
    dfs = [df for df in [flim_df, lejepa_df, ijepa_df] if df is not None]
    if distil_conv_df is not None:
        dfs.append(distil_conv_df)
    if distil_flim_frozen_df is not None:
        dfs.append(distil_flim_frozen_df)
    if distil_proj_df is not None:
        dfs.append(distil_proj_df)
    if distil_3x3bn_df is not None:
        dfs.append(distil_3x3bn_df)
    if distil_1x1bn_df is not None:
        dfs.append(distil_1x1bn_df)
    if distil_1x1bn_nonorm_df is not None:
        dfs.append(distil_1x1bn_nonorm_df)
    if distil_1x1bn_flim_df is not None:
        dfs.append(distil_1x1bn_flim_df)
    if distil_3x3bn_flim_df is not None:
        dfs.append(distil_3x3bn_flim_df)
    if distil_2l400k_df is not None:
        dfs.append(distil_2l400k_df)
    if distil_2l400k_flim_df is not None:
        dfs.append(distil_2l400k_flim_df)
    if distil_2l400k_flim_nonorm_df is not None:
        dfs.append(distil_2l400k_flim_nonorm_df)
    if flim_residual_df is not None:
        dfs.append(flim_residual_df)

    if not dfs:
        print("[ERROR] No data sources available. Nothing to save.")
        return

    unified = pd.concat(dfs, ignore_index=True)
    unified_path = Path(UNIFIED_SVM_COMPARISON_CSV)
    unified.to_csv(unified_path, index=False)
    print(f"[SAVE] {unified_path.relative_to(_ROOT)}  ({len(unified)} rows)")

    # ── Count validation ───────────────────────────────────────────────────────
    print("\n[VALIDATE] Row counts per method:")
    for method, grp in unified.groupby("method"):
        print(f"  {method}: {len(grp)} rows")

    # ── Generate comparison plots ──────────────────────────────────────────────
    print("\n[PLOTS] Generating comparison plots ...")
    datasets = sorted(unified["dataset_short"].unique())
    for dataset in datasets:
        ds_df = unified[unified["dataset_short"] == dataset]
        for metric in METRICS:
            plot_method_comparison(ds_df, dataset, metric, plots_dir)

    print(
        f"\n[DONE] Outputs written to:\n"
        f"  {out_dir.relative_to(_ROOT)}/\n"
        f"  {plots_dir.relative_to(_ROOT)}/"
    )


if __name__ == "__main__":
    main()
