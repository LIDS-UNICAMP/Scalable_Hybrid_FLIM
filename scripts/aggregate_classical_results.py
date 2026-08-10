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

"""aggregate_classical_results.py — Normalise and aggregate
results/classical_classifiers_results.csv into the format expected by
src/evaluate/eval_plotter.py and scripts/plot_comparison_flim.py.

Input  : results/classical_classifiers_results.csv   (one row per split)
Output : results/classical_classifiers_aggregated.csv (one row per group,
         with mean ± std across splits)

Column mapping applied
  dataset_name      → dataset_short  (helminth-eggs→eggs, etc.)
  percentage        → pretrained_pct
  initialization_type → init
  classifier_variant or classifier → method

Output columns (in order):
  method, init, dataset_short, pretrained_pct, n_splits,
  kappa, kappa_std, acc, acc_std, f1, f1_std

Usage:
    python scripts/aggregate_classical_results.py
    python scripts/aggregate_classical_results.py --input results/other.csv
    python scripts/aggregate_classical_results.py --init trunc_normal
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from constants import (
    CANONICAL_COLS as _OUT_COLS,
    DATASET_ALIASES as _DATASET_SHORT,
    METRICS as _METRICS,
    PROJECT_ROOT,
    RESULTS_DIR,
)

# ── Repo root ──────────────────────────────────────────────────────────────────
_ROOT = Path(PROJECT_ROOT)
sys.path.insert(0, str(_ROOT))

_RESULTS_DIR = Path(RESULTS_DIR)


def _method_label(row: pd.Series) -> str:
    """Return the method key used as the legend identifier in plots."""
    variant = str(row.get("classifier_variant", "")).strip()
    return variant if variant else str(row.get("classifier", "unknown"))


def aggregate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and aggregate over splits.

    Only rows with status=="ok" are aggregated.  Rows where status is
    "skipped_gp_too_large" are dropped silently.  Error rows are reported
    but not aggregated.
    """
    df = df_raw.copy()

    # ── Report non-ok rows ────────────────────────────────────────────────
    n_errors = (df["status"] == "error").sum()
    n_skipped = (df["status"] == "skipped_gp_too_large").sum()
    if n_errors:
        print(f"  [WARN] {n_errors} error row(s) excluded from aggregation.")
    if n_skipped:
        print(f"  [INFO] {n_skipped} GP-skipped row(s) excluded.")

    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("  [WARN] No ok rows found — output will be empty.")
        return pd.DataFrame(columns=_OUT_COLS)

    # ── Column normalisation ──────────────────────────────────────────────
    df["dataset_short"] = (
        df["dataset_name"]
        .map(lambda x: _DATASET_SHORT.get(str(x).strip(), str(x).strip()))
    )
    df["pretrained_pct"] = df["percentage"].astype(int)
    df["init"] = df["initialization_type"].astype(str).str.strip()
    df["method"] = df.apply(_method_label, axis=1)

    # ── Aggregate by (method, init, dataset_short, pretrained_pct) ────────
    group_keys = ["method", "init", "dataset_short", "pretrained_pct"]
    agg_rows: list[dict] = []

    for keys, group in df.groupby(group_keys, sort=True):
        method, init, dataset_short, pretrained_pct = keys
        row: dict = {
            "method":         method,
            "init":           init,
            "dataset_short":  dataset_short,
            "pretrained_pct": pretrained_pct,
            "n_splits":       len(group),
        }
        for m in _METRICS:
            vals = group[m].dropna()
            row[m]           = float(vals.mean()) if not vals.empty else float("nan")
            row[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)

    out = pd.DataFrame(agg_rows, columns=_OUT_COLS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate classical_classifiers_results.csv for plotting."
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to raw CSV (default: results/classical_classifiers_results.csv)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for aggregated CSV (default: results/classical_classifiers_aggregated.csv)",
    )
    parser.add_argument(
        "--init", default=None,
        help=(
            "Filter to a single initialization_type before aggregating "
            "(e.g. --init trunc_normal).  Default: keep all."
        ),
    )
    args = parser.parse_args()

    in_path  = Path(args.input)  if args.input  else _RESULTS_DIR / "classical_classifiers_results.csv"
    out_path = Path(args.output) if args.output else _RESULTS_DIR / "classical_classifiers_aggregated.csv"

    print(f"\n{'=' * 70}")
    print(f"[AGGREGATE] Reading : {in_path}")
    print(f"{'=' * 70}")

    if not in_path.exists():
        print(f"[ERROR] File not found: {in_path}")
        sys.exit(1)

    df_raw = pd.read_csv(in_path)
    df_raw = df_raw.dropna(subset=["classifier", "dataset_name"]).copy()
    print(f"  Rows loaded   : {len(df_raw)}")
    print(f"  Classifiers   : {sorted(df_raw['classifier'].dropna().unique()) if 'classifier' in df_raw.columns else 'n/a'}")
    print(f"  Datasets      : {sorted(df_raw['dataset_name'].dropna().unique()) if 'dataset_name' in df_raw.columns else 'n/a'}")
    print(f"  Init types    : {sorted(df_raw['initialization_type'].dropna().unique()) if 'initialization_type' in df_raw.columns else 'n/a'}")
    print(f"  Percentages   : {sorted(df_raw['percentage'].dropna().unique()) if 'percentage' in df_raw.columns else 'n/a'}")

    if args.init:
        before = len(df_raw)
        df_raw = df_raw[df_raw["initialization_type"] == args.init].copy()
        print(f"\n  [FILTER] init='{args.init}' → {before} → {len(df_raw)} rows")

    df_agg = aggregate(df_raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] Aggregated rows : {len(df_agg)}")
    print(f"[DONE] Saved to        : {out_path}")
    print(f"{'=' * 70}")

    if not df_agg.empty:
        print("\n  Preview (first 10 rows):")
        print(df_agg.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
