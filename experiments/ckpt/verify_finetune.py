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

"""verify_finetune_weights.py — Audit local fine-tune weights vs YAML configs vs W&B.

Checks:
  1. Every YAML config (freeze + unfreeze) has a local model_best.pth.
  2. W&B has at least one fine-tune run per (encoder_run_id, mode).
  3. When multiple W&B runs exist for the same key, only the LATEST is used
     (by run.created_at) — flags AMBIGUOUS entries.
  4. Reports missing, duplicate, and ambiguous cases in a clean table.

Usage (da raiz do repositorio; o argparse morreu no refactor, os filtros sao
defaults nomeados de ``verify``)::

    conda run -n scalable_FLIM python -m experiments.ckpt.verify_finetune
    conda run -n scalable_FLIM python -c "from experiments.ckpt.verify_finetune \
        import verify; raise SystemExit(verify(dataset_filter='protozoan'))"
    conda run -n scalable_FLIM python -c "from experiments.ckpt.verify_finetune \
        import verify; raise SystemExit(verify(mode_filter='freeze'))"
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
import wandb

# Import only the lightweight constants — avoid pulling the full dataset stack.
# PROJECT_ROOT saiu junto com o sys.path.insert que ele alimentava: rodado como
# `python -m experiments.ckpt.verify_finetune` da raiz, a raiz ja e sys.path[0].
from core.constants import DATASETS
from experiments.constants import (
    DATASET_ALIASES as DATASET_FULL_TO_SHORT,
    MLP_CONFIGS_DIR,
    RESULTS_DIR,
    WANDB_ENTITY as ENTITY,
    WANDB_PROJECT as PROJECT,
)

_WANDB_FILTER = {
    "$or": [
        {"display_name": {"$regex": "^X_finetune_"}},
        {"display_name": {"$regex": "^ray_freeze_"}},
        {"display_name": {"$regex": "^ray_unfreeze_"}},
    ]
}

_RUNS_CACHE: dict[tuple[str, str], list] = {}


def _fetch_project_runs(entity: str = ENTITY, project: str = PROJECT, per_page: int = 500) -> list:
    key = (entity, project)
    if key in _RUNS_CACHE:
        return _RUNS_CACHE[key]
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters=_WANDB_FILTER, per_page=per_page))
    _RUNS_CACHE[key] = runs
    return runs

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_ROOT = Path(RESULTS_DIR) / "mlp_weights"
YAML_ROOT = Path(MLP_CONFIGS_DIR)

MODES = ("freeze", "unfreeze")

# Prefixes used by ray_mlp.py and mlp.py when logging to W&B
_FINETUNE_PREFIXES = ("X_finetune_", "ray_freeze_", "ray_unfreeze_")


# ── YAML loading ──────────────────────────────────────────────────────────────

def _load_yaml_configs(mode: str, dataset_filter: str | None) -> list[dict]:
    """Return list of dicts, one per YAML config for (mode, dataset_filter)."""
    base = YAML_ROOT / mode
    if not base.exists():
        return []
    configs = []
    for path in sorted(base.rglob("*.yaml")):
        with path.open() as f:
            cfg = yaml.safe_load(f)
        cfg["_yaml_path"] = str(path)
        cfg["_mode"] = mode
        ds_full = cfg.get("dataset_name", "")
        ds_short = DATASET_FULL_TO_SHORT.get(ds_full, ds_full)
        cfg["_dataset_short"] = ds_short
        if dataset_filter and ds_short != dataset_filter:
            continue
        configs.append(cfg)
    return configs


# ── W&B fine-tune run grouping ────────────────────────────────────────────────

def _is_finetune_run(name: str) -> bool:
    return any(name.startswith(p) for p in _FINETUNE_PREFIXES)


def _infer_mode_from_name(name: str) -> str | None:
    """Infer freeze/unfreeze from run name."""
    lower = name.lower()
    if "unfreeze" in lower:
        return "unfreeze"
    if "freeze" in lower:
        return "freeze"
    return None


def _infer_mode_from_config(cfg: dict) -> str | None:
    raw = cfg.get("freeze_encoder")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return "freeze" if raw else "unfreeze"
    if isinstance(raw, int):
        return "freeze" if raw else "unfreeze"
    if isinstance(raw, str):
        return "freeze" if raw.lower() in ("true", "1", "yes") else "unfreeze"
    return None


def _infer_mode_from_tags(tags: list[str]) -> str | None:
    tags_lower = [t.lower() for t in (tags or [])]
    if "freeze" in tags_lower and "unfreeze" not in tags_lower:
        return "freeze"
    if "unfreeze" in tags_lower:
        return "unfreeze"
    return None


def _encoder_id_from_run(run) -> str:
    """Best-effort encoder run_id from a fine-tune W&B run object."""
    cfg = dict(run.config) if run.config else {}
    enc = cfg.get("wandb_run_id", "") or cfg.get("run_id", "")
    if not enc:
        # Fallback: last segment of run name (X_finetune_..._<enc_id>)
        enc = run.name.rsplit("_", 1)[-1]
    return enc


def build_wandb_index(all_runs: list) -> dict[tuple[str, str], list]:
    """Group W&B finetune runs by (encoder_run_id, mode).

    Returns dict: (encoder_id, mode) → list of run objects, sorted newest-first.
    """
    index: dict[tuple[str, str], list] = {}
    for run in all_runs:
        if not run.name or not _is_finetune_run(run.name):
            continue

        cfg = dict(run.config) if run.config else {}
        mode = (
            _infer_mode_from_config(cfg)
            or _infer_mode_from_tags(run.tags)
            or _infer_mode_from_name(run.name)
        )
        if mode is None:
            continue

        enc_id = _encoder_id_from_run(run)
        if not enc_id:
            continue

        key = (enc_id, mode)
        index.setdefault(key, []).append(run)

    # Sort each group newest-first
    for key in index:
        index[key].sort(
            key=lambda r: r.created_at or "",
            reverse=True,
        )
    return index


# ── Verification ──────────────────────────────────────────────────────────────

def _weight_mtime(path: Path) -> str:
    if path.exists():
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    return "—"


def verify(
    dataset_filter: str | None = None,
    mode_filter: str | None = None,
    verbose: bool = False,
) -> int:
    """Run the full audit. Returns number of problems found.

    dataset_filter  short name, ou "parasito" (o dataset agregado: nao esta em
                    DATASETS, mas aparece nos YAML). None = todos.
    mode_filter     "freeze" | "unfreeze". None = os dois.
    verbose         extra W&B debug output.
    """
    if dataset_filter is not None and dataset_filter not in (*DATASETS, "parasito"):
        raise SystemExit(f"verify: dataset_filter invalido: {dataset_filter!r} "
                         f"(esperado um de {[*DATASETS, 'parasito']})")
    if mode_filter is not None and mode_filter not in MODES:
        raise SystemExit(f"verify: mode_filter invalido: {mode_filter!r} "
                         f"(esperado um de {list(MODES)})")

    modes = [mode_filter] if mode_filter else list(MODES)

    # ── Load all YAML configs ─────────────────────────────────────────────
    all_configs: list[dict] = []
    for mode in modes:
        all_configs.extend(_load_yaml_configs(mode, dataset_filter))

    if not all_configs:
        print("[ERROR] No YAML configs found for the given filters.")
        return 1

    print(f"\n{'═'*72}")
    print(f"  FINETUNE WEIGHT AUDIT")
    print(f"  dataset={dataset_filter or 'all'}  mode={mode_filter or 'all'}")
    print(f"  {len(all_configs)} YAML configs to verify")
    print(f"{'═'*72}\n")

    # ── Fetch W&B runs once ───────────────────────────────────────────────
    print("[W&B] Fetching runs from W&B (cached)...")
    try:
        all_runs = _fetch_project_runs(entity=ENTITY, project=PROJECT)
    except Exception as exc:
        print(f"[ERROR] Cannot reach W&B: {exc}")
        all_runs = []

    wandb_index = build_wandb_index(all_runs)

    if verbose:
        print(f"[W&B] Indexed {len(wandb_index)} (encoder_id, mode) pairs with finetune runs")

    # ── Audit each config ─────────────────────────────────────────────────
    rows_ok: list[dict] = []
    rows_warn: list[dict] = []
    rows_missing: list[dict] = []

    for cfg in all_configs:
        run_id = cfg.get("run_id", "?")
        mode = cfg["_mode"]
        ds = cfg["_dataset_short"]
        split = cfg.get("split", cfg.get("split_id", "?"))
        pct = cfg.get("percentage", "?")
        init = cfg.get("initialization_type", "?")

        weight_path = WEIGHTS_ROOT / mode / run_id / "model_best.pth"
        has_weight = weight_path.exists()
        mtime = _weight_mtime(weight_path)

        # W&B lookup
        key = (run_id, mode)
        wb_runs = wandb_index.get(key, [])
        n_wb = len(wb_runs)
        latest_wb = wb_runs[0] if wb_runs else None
        latest_created = (latest_wb.created_at or "?")[:10] if latest_wb else "—"
        latest_state = latest_wb.state if latest_wb else "—"

        # Find the latest FINISHED run (skip crashed/running as "latest good")
        latest_finished_wb = next(
            (r for r in wb_runs if r.state == "finished"), None
        )

        latest_crashed = (
            has_weight
            and latest_wb is not None
            and latest_wb.state != "finished"
            and latest_finished_wb is not None  # there IS a finished one, but it's older
        )

        row = {
            "run_id": run_id,
            "mode": mode,
            "dataset": ds,
            "split": split,
            "pct": pct,
            "init": init,
            "has_weight": has_weight,
            "weight_mtime": mtime,
            "n_wb_runs": n_wb,
            "wb_latest_date": latest_created,
            "wb_latest_state": latest_state,
            "wb_latest_finished_date": (latest_finished_wb.created_at or "?")[:10] if latest_finished_wb else "—",
            "ambiguous": n_wb > 1,
            "latest_crashed": latest_crashed,
        }

        if not has_weight:
            rows_missing.append(row)
        else:
            rows_ok.append(row)

    # Split ok rows into clean / ambiguous / crashed_latest for display
    rows_clean = [r for r in rows_ok if not r["ambiguous"]]
    rows_crashed = [r for r in rows_ok if r["latest_crashed"]]
    rows_ambiguous = [r for r in rows_ok if r["ambiguous"] and not r["latest_crashed"]]

    # ── Print results ─────────────────────────────────────────────────────
    col = "{:<12} {:<10} {:<10} {:<6} {:<5} {:<8} {:<17} {:<6} {:<12} {:<10}"
    header = col.format(
        "run_id", "mode", "dataset", "split", "pct", "init",
        "weight_mtime", "n_wb", "wb_latest", "wb_state"
    )
    sep = "─" * 105

    def _sort_key(r):
        return (r["dataset"], r["mode"], r["pct"], r["split"])

    if rows_clean:
        print(f"{'✓ OK (single W&B run, finished)':─<72}")
        print(header)
        print(sep)
        for r in sorted(rows_clean, key=_sort_key):
            print(col.format(
                r["run_id"], r["mode"], r["dataset"],
                str(r["split"]), str(r["pct"]), r["init"],
                r["weight_mtime"], r["n_wb_runs"],
                r["wb_latest_date"], r["wb_latest_state"],
            ))

    if rows_crashed:
        print(f"\n{'✗ LATEST W&B RUN CRASHED — weight may be from older run':─<72}")
        print("  The local model_best.pth may have been overwritten by a crashed run.")
        print("  Consider rerunning to get a clean weight.\n")
        print(header)
        print(sep)
        for r in sorted(rows_crashed, key=_sort_key):
            print(col.format(
                r["run_id"], r["mode"], r["dataset"],
                str(r["split"]), str(r["pct"]), r["init"],
                r["weight_mtime"], r["n_wb_runs"],
                r["wb_latest_date"], r["wb_latest_state"],
            ))
            wb_runs = wandb_index.get((r["run_id"], r["mode"]), [])
            for i, wr in enumerate(wb_runs):
                marker = "← LATEST (CRASHED)" if i == 0 else (
                    "← latest finished" if wr.state == "finished" and i == next(
                        (j for j, x in enumerate(wb_runs) if x.state == "finished"), -1) else f"  [{i}]"
                )
                print(f"    {marker}  wb_run_id={wr.id}  name={wr.name[:55]}  "
                      f"created={str(wr.created_at)[:10]}  state={wr.state}")

    if rows_ambiguous:
        print(f"\n{'⚠ AMBIGUOUS (multiple W&B runs, latest=finished — using latest)':─<72}")
        print(header)
        print(sep)
        for r in sorted(rows_ambiguous, key=_sort_key):
            print(col.format(
                r["run_id"], r["mode"], r["dataset"],
                str(r["split"]), str(r["pct"]), r["init"],
                r["weight_mtime"], r["n_wb_runs"],
                r["wb_latest_date"], r["wb_latest_state"],
            ))

    if rows_missing:
        print(f"\n{'✗ MISSING WEIGHTS':─<72}")
        print(header)
        print(sep)
        for r in sorted(rows_missing, key=_sort_key):
            wb_info = "no_wb_run" if r["n_wb_runs"] == 0 else f"{r['n_wb_runs']}_wb_run(s)"
            print(col.format(
                r["run_id"], r["mode"], r["dataset"],
                str(r["split"]), str(r["pct"]), r["init"],
                "MISSING", wb_info,
                r["wb_latest_date"], r["wb_latest_state"],
            ))

    # ── Summary ───────────────────────────────────────────────────────────
    n_critical = len(rows_missing) + len(rows_crashed)
    print(f"\n{'═'*72}")
    print(f"  SUMMARY")
    print(f"  Total configs      : {len(all_configs)}")
    print(f"  OK (clean)         : {len(rows_clean)}")
    print(f"  OK (multi-run)     : {len(rows_ambiguous)}  ← multiple W&B runs, latest=finished")
    print(f"  Latest W&B crashed : {len(rows_crashed)}  ← weight may be stale/partial")
    print(f"  Missing weights    : {len(rows_missing)}")
    print(f"{'═'*72}\n")

    if n_critical == 0:
        print("All weights present and latest W&B runs finished. Ready for unified_eval.")
    else:
        if rows_crashed:
            print("Rerun fine-tune for crashed experiments (weight may be incomplete):")
            for r in rows_crashed:
                print(f"  conda run -n scalable_FLIM python -m src.evaluate.ray_mlp \\")
                print(f"      --mode {r['mode']} --num-gpus 2 --gpus-per-trial 0.2 "
                      f"--cpus-per-trial 2 \\")
                print(f"      --output-dir results/ray_finetune --experiment {r['run_id']}")
        if rows_missing:
            print("\nRun fine-tune for missing weights:")
            for r in rows_missing:
                print(f"  conda run -n scalable_FLIM python -m src.evaluate.ray_mlp \\")
                print(f"      --mode {r['mode']} --num-gpus 2 --gpus-per-trial 0.2 "
                      f"--cpus-per-trial 2 \\")
                print(f"      --output-dir results/ray_finetune --experiment {r['run_id']}")

    return n_critical


if __name__ == "__main__":
    raise SystemExit(verify())
