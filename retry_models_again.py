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

"""retry_models_again.py — Stage-aware audit and retry for LeJEPA/MLP/SVM experiments.

Stage Model
-----------
  A : MLP config YAML generation  (configs/evaluate/mlp/{mode}/{dataset}/…/{run_id}.yaml)
  B : LeJEPA SSL fine-tune        (logs/flim-ssl/{run_id}/checkpoints/*.ckpt)
  C : MLP freeze evaluation       (artifacts/MLP/mlp_results.csv  or  results/mlp_weights/)
  D : MLP unfreeze evaluation     (artifacts/MLP/mlp_results.csv  or  results/mlp_weights/)
  E : SVM evaluation              (artifacts/SVM/svm_results.csv)

Workflow
--------
  1. Build the full expected matrix (3 datasets × 3 splits × 6 pcts × 4 inits = 216 runs).
  2. Resolve SSL run_id for each entry from ids_wandb.json, with fallback to run log.
  3. Check every stage for every run using actual artifact evidence.
  4. Cross-check run_manifest.csv skip_reason for additional failure context.
  5. Classify each stage as: ok | missing | broken | ssl_failed | unknown.
  6. Emit a pre-retry audit report.
  7. Retry only missing/broken stages:
       - Stage A : create YAML config in place.
       - Stage B : print the fit command (long training — manual execution required).
       - Stage C/D : submit to Ray GPU-slot queue (reuses ray_mlp_queue.py).
       - Stage E : run src.evaluate.svm via subprocess (processes whole experiment set).
  8. Emit a post-retry report and write a machine-usable retry plan CSV.

Usage
-----
    # Full audit + retry (2 GPUs, 10 slots/GPU)
    python retry_models_again.py

    # Audit only — no execution
    python retry_models_again.py --report-only

    # Dry-run — show queue plan without submitting
    python retry_models_again.py --dry-run

    # Override manifest path
    python retry_models_again.py --manifest artifacts/run_manifest.csv

    # Save full report to file
    python retry_models_again.py --report-only --report-out results/audit_report.txt

Constraints
-----------
  * Does NOT rewrite training/evaluation logic — reuses existing src/evaluate/ scripts.
  * Ignores __pycache__/, *.pyc, *.so, *.git/, and other irrelevant compiled artefacts.
  * Does NOT modify configs/wandb_update/ids_wandb.json unless a mismatch is proven.
  * Follows canonical naming: lejepa_line_{dataset}_split_{N}_pct_{P}_model_{init}.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import yaml

# ── Project root ───────────────────────────────────────────────────────────────

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── File paths ─────────────────────────────────────────────────────────────────

_MANIFEST_PATH      = os.path.join(_ROOT, "artifacts", "run_manifest.csv")
_RUN_LOG_PATH       = os.path.join(_ROOT, "logs", "run_experiments_run_ids.log")
_IDS_WANDB_PATH     = os.path.join(_ROOT, "configs", "wandb_update", "ids_wandb.json")
_LOGS_SSL_DIR       = os.path.join(_ROOT, "logs", "flim-ssl")
_CONFIGS_MLP_DIR    = os.path.join(_ROOT, "configs", "evaluate", "mlp")
_RESULTS_DIR        = os.path.join(_ROOT, "results")

# Ordered sources for MLP result confirmation
_MLP_RESULTS_PATHS: list[str] = [
    os.path.join(_ROOT, "artifacts", "MLP", "mlp_results.csv"),
    os.path.join(_ROOT, "results", "mlp_results.csv"),
    os.path.join(_ROOT, "results", "ray_mlp_results.csv"),
    os.path.join(_ROOT, "results", "ray_mlp_queue_results.csv"),
]
_SVM_RESULTS_PATHS: list[str] = [
    os.path.join(_ROOT, "artifacts", "SVM", "svm_results.csv"),
    os.path.join(_ROOT, "results", "svm_results.csv"),
]
_MLP_WEIGHTS_DIR    = os.path.join(_ROOT, "results", "mlp_weights")

# Legacy artifact roots (weight-file based, kept for cross-check)
_MLP_WEIGHT_ROOTS: list[tuple[str, str]] = [
    (os.path.join(_ROOT, "results", "ray_finetune"),        "resolved_local"),
    (os.path.join(_ROOT, "results", "ray_finetune_queue"), "resolved_local"),
    (os.path.join(_ROOT, "artifacts_view", "MLP"),         "resolved_artifacts_view"),
]

# ── Constants ─────────────────────────────────────────────────────────────────

_DATASET_SHORT_TO_FULL: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}
_DATASET_FULL_TO_SHORT: dict[str, str] = {v: k for k, v in _DATASET_SHORT_TO_FULL.items()}

_EXPECTED_DATASETS = ["eggs", "larvae", "protozoan"]
_EXPECTED_SPLITS   = [1, 2, 3]
_EXPECTED_PCTS     = [1, 5, 25, 50, 75, 100]
_EXPECTED_INITS    = ["xavier", "random", "he", "flim"]
_MLP_MODES         = ["freeze", "unfreeze"]

# MLP YAML defaults — mirrors generate_mlp_configs.py
_MLP_YAML_DEFAULTS = {
    "max_epochs":   1000,
    "patience":     50,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   32,
    "hidden_dim":   256,
    "dropout":      0.3,
}

# Stage status tags
_OK          = "ok"
_MISSING     = "missing"
_BROKEN      = "broken"
_SSL_FAILED  = "ssl_failed"
_SKIP_ERROR  = "skip_error"
_UNKNOWN     = "unknown"

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExpectedRun:
    dataset_short: str
    dataset_full:  str
    split:         int
    pct:           int
    init:          str

    @property
    def canonical_name(self) -> str:
        return (f"lejepa_line_{self.dataset_full}"
                f"_split_{self.split}_pct_{self.pct}_model_{self.init}")


@dataclass
class StageResult:
    status:  str    = _UNKNOWN   # ok | missing | broken | ssl_failed | skip_error | unknown
    reason:  str    = ""         # human-readable explanation
    detail:  str    = ""         # path / extra context


@dataclass
class StageAuditEntry:
    run:          ExpectedRun
    ssl_run_id:   Optional[str]  = None
    # Manifest rows (for skip_reason context)
    manifest_svm:      list[dict] = field(default_factory=list)
    manifest_mlp_freeze: list[dict] = field(default_factory=list)
    manifest_mlp_unfreeze: list[dict] = field(default_factory=list)
    # Stage results
    stage_b: StageResult = field(default_factory=StageResult)  # LeJEPA checkpoint
    stage_a_freeze: StageResult = field(default_factory=StageResult)   # YAML freeze
    stage_a_unfreeze: StageResult = field(default_factory=StageResult) # YAML unfreeze
    stage_c: StageResult = field(default_factory=StageResult)  # MLP freeze
    stage_d: StageResult = field(default_factory=StageResult)  # MLP unfreeze
    stage_e: StageResult = field(default_factory=StageResult)  # SVM


@dataclass
class RetryOutcome:
    run:            ExpectedRun
    mode:           str          # "freeze" | "unfreeze" | "svm" | "lejepa"
    ssl_run_id:     str
    stage:          str          # "C" | "D" | "E" | "B"
    reason_before:  str
    yaml_path:      str          = ""
    yaml_created:   bool         = False
    ray_result:     dict         = field(default_factory=dict)
    weight_found_after: str      = ""


# ── Helper: build expected matrix ─────────────────────────────────────────────


def build_expected_matrix() -> list[ExpectedRun]:
    runs: list[ExpectedRun] = []
    for ds in _EXPECTED_DATASETS:
        ds_full = _DATASET_SHORT_TO_FULL[ds]
        for sp in _EXPECTED_SPLITS:
            for pct in _EXPECTED_PCTS:
                for init in _EXPECTED_INITS:
                    runs.append(ExpectedRun(ds, ds_full, sp, pct, init))
    return runs


# ── Helper: load SSL run IDs ──────────────────────────────────────────────────


def load_ids_wandb(path: str = _IDS_WANDB_PATH) -> dict[str, str]:
    """Return {canonical_experiment_name: run_id} from ids_wandb.json.

    Only considers runs with state='finished' or non-empty summary.
    """
    result: dict[str, str] = {}
    if not os.path.isfile(path):
        return result
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", {})
    for run_id, info in runs.items():
        name = info.get("name", "").strip()
        if name:
            # Only overwrite if this entry looks more complete
            if name not in result or info.get("state") == "finished":
                result[name] = run_id
    return result


def load_ssl_run_log(log_path: str = _RUN_LOG_PATH) -> dict[str, tuple[str, str]]:
    """Return {canonical_name: (run_id, status)} from run log.

    For duplicates, keeps the last OK entry, falling back to last FAILED.
    Used as fallback when ids_wandb.json does not cover an experiment.
    """
    entries: dict[str, tuple[str, str]] = {}
    if not os.path.isfile(log_path):
        return entries
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            exp_name = parts[1]
            run_id   = parts[2]
            status   = parts[5]
            if status == "OK":
                entries[exp_name] = (run_id, "ok")
            elif exp_name not in entries:
                entries[exp_name] = (run_id, "failed")
    return entries


# ── Helper: load manifest ─────────────────────────────────────────────────────


def load_manifest(manifest_path: str = _MANIFEST_PATH) -> dict[str, list[dict]]:
    """Return {'{phase}:{ds}:{split}:{pct}:{init}': [row, ...]}."""
    result: dict[str, list[dict]] = {}
    if not os.path.isfile(manifest_path):
        return result
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "").strip()
            ds    = row.get("dataset_short", "").strip()
            split = row.get("split", "").strip()
            pct   = row.get("pct", "").strip()
            init  = row.get("init", "").strip()
            key   = f"{phase}:{ds}:{split}:{pct}:{init}"
            result.setdefault(key, []).append({
                "phase":       phase,
                "dataset_short": ds,
                "split":       split,
                "pct":         pct,
                "init":        init,
                "skip_reason": row.get("skip_reason", "").strip(),
                "wandb_run_id": row.get("wandb_run_id", "").strip(),
                "run_name":    row.get("run_name", "").strip(),
            })
    return result


def _manifest_key(phase: str, ds: str, split: int, pct: int, init: str) -> str:
    return f"{phase}:{ds}:{split}:{pct}:{init}"


# ── Helper: load MLP / SVM result sets ───────────────────────────────────────


def load_mlp_results() -> set[tuple[str, str]]:
    """Return set of (ssl_run_id, mode) for completed MLP runs.

    mode is 'freeze' or 'unfreeze'.  Checks all known result CSVs in order.
    SSL run_id is extracted from the 'wandb_run_name' or 'run_name' column
    (last 8-char segment matching the W&B id pattern).
    Also checks results/mlp_weights/{mode}/{run_id}/model_best.pth as
    supplementary evidence.
    """
    _RUN_ID_RE = re.compile(r"_([a-z0-9]{8})$")
    completed: set[tuple[str, str]] = set()

    for csv_path in _MLP_RESULTS_PATHS:
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "").strip()
                if status != "ok":
                    continue
                # Determine mode from freeze_status column
                freeze_status = row.get("freeze_status", "").strip()
                if freeze_status in ("freeze", "unfreeze"):
                    mode = freeze_status
                elif str(row.get("freeze_encoder", "")).lower() in ("true", "1"):
                    mode = "freeze"
                else:
                    mode = "unfreeze"
                # Extract ssl_run_id from run name columns
                for col in ("wandb_run_name", "run_name", "ckpt_source"):
                    val = row.get(col, "")
                    m = _RUN_ID_RE.search(val)
                    if m:
                        completed.add((m.group(1), mode))
                        break

    # Supplement with local weight files
    for mode in ("freeze", "unfreeze"):
        weights_root = os.path.join(_MLP_WEIGHTS_DIR, mode)
        if os.path.isdir(weights_root):
            for entry in os.scandir(weights_root):
                if entry.is_dir():
                    pth = os.path.join(entry.path, "model_best.pth")
                    if os.path.isfile(pth):
                        completed.add((entry.name, mode))

    # Legacy: ray_finetune weight roots
    for root, _ in _MLP_WEIGHT_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            for mode in ("freeze", "unfreeze"):
                weights_dir = os.path.join(entry.path, "weights")
                pth = os.path.join(weights_dir, "model_final.pth")
                if os.path.isfile(pth):
                    # run dir name may contain ssl_run_id
                    m = re.search(r"([a-z0-9]{8})$", entry.name)
                    if m:
                        # Both modes considered complete if weight present
                        for _m in ("freeze", "unfreeze"):
                            completed.add((m.group(1), _m))
                    break

    return completed


def load_svm_results() -> set[str]:
    """Return set of ssl_run_ids for completed SVM evaluations."""
    completed: set[str] = set()
    for csv_path in _SVM_RESULTS_PATHS:
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status", "").strip() == "ok":
                    run_id = row.get("wandb_run_id", "").strip()
                    if run_id:
                        completed.add(run_id)
    return completed


# ── Helper: stage checks ──────────────────────────────────────────────────────


def check_lejepa_ckpt(run_id: str) -> tuple[bool, str]:
    """Return (exists, ckpt_path).  Ignores __pycache__ and compiled files."""
    ckpt_dir = os.path.join(_LOGS_SSL_DIR, run_id, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False, ""
    matches = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not matches:
        matches = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    # Filter out __pycache__ and compiled artefacts
    matches = [p for p in matches
               if "__pycache__" not in p
               and not p.endswith((".pyc", ".pyo", ".so"))]
    if matches:
        return True, matches[0]
    return False, ""


def _yaml_path_for(ssl_run_id: str, exp_name: str, mode: str) -> str:
    m = re.match(
        r"^lejepa_line_([a-z\-]+)_split_(\d+)_pct_(\d+)_model_(xavier|random|he|flim)$",
        exp_name,
    )
    if not m:
        raise ValueError(f"Cannot parse canonical name: {exp_name!r}")
    dataset, split, pct = m.group(1), m.group(2), m.group(3)
    return os.path.join(
        _CONFIGS_MLP_DIR, mode, dataset, f"split_{split}", f"pct_{pct}",
        f"{ssl_run_id}.yaml",
    )


def check_yaml_exists(ssl_run_id: str, exp_name: str, mode: str) -> tuple[bool, str]:
    """Return (exists, yaml_path)."""
    try:
        path = _yaml_path_for(ssl_run_id, exp_name, mode)
    except ValueError:
        return False, ""
    return os.path.isfile(path), path


def ensure_mlp_yaml(ssl_run_id: str, run: ExpectedRun, mode: str) -> tuple[str, bool]:
    """Return (yaml_path, was_created).  Creates YAML if absent."""
    exp_name = run.canonical_name
    path = _yaml_path_for(ssl_run_id, exp_name, mode)
    if os.path.isfile(path):
        return path, False
    content = {
        "run_id":              ssl_run_id,
        "experiment_name":     exp_name,
        "dataset_name":        run.dataset_full,
        "split":               run.split,
        "percentage":          run.pct,
        "initialization_type": run.init,
        "freeze_encoder":      (mode == "freeze"),
        **_MLP_YAML_DEFAULTS,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)
    return path, True


# ── Helper: skip_reason classifier ───────────────────────────────────────────


def classify_skip_reason(
    skip_reason: str,
    ckpt_exists: bool,
    stage: str,
) -> tuple[str, str]:
    """Return (status_tag, human_explanation) for a manifest skip_reason.

    Distinguishes:
      - real missing upstream dependency
      - downstream never launched
      - stale manifest (artifact now exists)
      - name/parse error
      - ambiguous / needs manual inspection
    """
    if not skip_reason:
        if ckpt_exists:
            return _OK, "manifest row present, no skip_reason, checkpoint confirmed"
        return _OK, "manifest row present, no skip_reason"

    if skip_reason.startswith("unknown_dataset"):
        return _SKIP_ERROR, f"dataset parse error in manifest: {skip_reason}"

    if skip_reason == "missing_weights":
        if ckpt_exists:
            return _MISSING, "manifest says missing_weights but checkpoint exists — downstream never launched"
        return _SSL_FAILED, "manifest says missing_weights and no checkpoint found"

    if skip_reason.startswith("missing"):
        return _MISSING, f"upstream dependency missing: {skip_reason}"

    return _UNKNOWN, f"unrecognised skip_reason: {skip_reason!r} — manual inspection required"


# ── Audit engine ──────────────────────────────────────────────────────────────


def audit(
    manifest_path: str = _MANIFEST_PATH,
) -> list[StageAuditEntry]:
    """Run full stage-aware audit over the expected experiment matrix.

    Returns one StageAuditEntry per expected experiment.
    """
    # ── Load data sources ─────────────────────────────────────────────────
    ids_wandb  = load_ids_wandb(_IDS_WANDB_PATH)
    run_log    = load_ssl_run_log(_RUN_LOG_PATH)
    manifest   = load_manifest(manifest_path)
    mlp_done   = load_mlp_results()
    svm_done   = load_svm_results()
    matrix     = build_expected_matrix()

    entries: list[StageAuditEntry] = []

    for run in matrix:
        entry = StageAuditEntry(run=run)
        canon = run.canonical_name

        # ── Resolve SSL run_id ────────────────────────────────────────────
        if canon in ids_wandb:
            entry.ssl_run_id = ids_wandb[canon]
        elif canon in run_log:
            rid, rl_status = run_log[canon]
            entry.ssl_run_id = rid
        # else: None — experiment was never trained

        ssl_id = entry.ssl_run_id

        # ── Load manifest rows for context ────────────────────────────────
        entry.manifest_svm = manifest.get(
            _manifest_key("SVM", run.dataset_short, run.split, run.pct, run.init), [])
        entry.manifest_mlp_freeze = manifest.get(
            _manifest_key("MLP_freeze", run.dataset_short, run.split, run.pct, run.init), [])
        entry.manifest_mlp_unfreeze = manifest.get(
            _manifest_key("MLP_unfreeze", run.dataset_short, run.split, run.pct, run.init), [])

        # ── Stage B: LeJEPA checkpoint ────────────────────────────────────
        if ssl_id is None:
            entry.stage_b = StageResult(
                _MISSING, "no SSL run_id found in ids_wandb.json or run log",
                "experiment may have never been trained",
            )
        else:
            ckpt_ok, ckpt_path = check_lejepa_ckpt(ssl_id)
            if ckpt_ok:
                entry.stage_b = StageResult(_OK, "checkpoint found", ckpt_path)
            else:
                # Check run log for failure context
                log_status = run_log.get(canon, (None, None))[1]
                if log_status == "failed":
                    entry.stage_b = StageResult(
                        _SSL_FAILED, "run log shows FAILED status",
                        f"run_id={ssl_id}, no checkpoint in logs/flim-ssl/",
                    )
                elif ssl_id in ids_wandb.values():
                    # In W&B but no local checkpoint → rsync gap
                    entry.stage_b = StageResult(
                        _MISSING, "W&B entry exists but no local checkpoint",
                        f"run_id={ssl_id} — checkpoint may need rsync",
                    )
                else:
                    entry.stage_b = StageResult(
                        _MISSING, "no checkpoint found",
                        f"run_id={ssl_id}, logs/flim-ssl/{ssl_id}/checkpoints/ absent or empty",
                    )

        ckpt_exists = (entry.stage_b.status == _OK)

        # ── Stages A + C/D/E require ssl_id and checkpoint ────────────────
        for mode, a_attr, cd_attr, phase_attr in (
            ("freeze",   "stage_a_freeze",   "stage_c", "manifest_mlp_freeze"),
            ("unfreeze", "stage_a_unfreeze",  "stage_d", "manifest_mlp_unfreeze"),
        ):
            if ssl_id is None:
                setattr(entry, a_attr,  StageResult(_MISSING, "no SSL run_id", ""))
                setattr(entry, cd_attr, StageResult(_MISSING, "no SSL run_id — Stage B missing", ""))
                continue

            # Stage A: YAML config existence
            yaml_ok, yaml_path = check_yaml_exists(ssl_id, canon, mode)
            if yaml_ok:
                setattr(entry, a_attr, StageResult(_OK, "YAML config found", yaml_path))
            else:
                setattr(entry, a_attr, StageResult(
                    _MISSING, "YAML config absent",
                    f"configs/evaluate/mlp/{mode}/{run.dataset_full}/split_{run.split}/pct_{run.pct}/{ssl_id}.yaml",
                ))

            # Stage C / D: MLP results
            mlp_key = (ssl_id, mode)
            manifest_rows = getattr(entry, phase_attr)
            skip_reason = manifest_rows[-1]["skip_reason"] if manifest_rows else ""

            if mlp_key in mlp_done:
                setattr(entry, cd_attr, StageResult(_OK, "found in MLP results CSV", f"ssl_run_id={ssl_id}, mode={mode}"))
            elif not ckpt_exists:
                setattr(entry, cd_attr, StageResult(
                    _MISSING, "Stage B missing — cannot run MLP",
                    f"ssl_run_id={ssl_id}, no checkpoint",
                ))
            else:
                # Checkpoint exists but MLP result absent — investigate manifest
                if manifest_rows:
                    status_tag, explanation = classify_skip_reason(
                        skip_reason, ckpt_exists, f"Stage C/D [{mode}]"
                    )
                    if status_tag == _OK:
                        # Manifest says ok but results CSV is absent — rsync gap
                        setattr(entry, cd_attr, StageResult(
                            _MISSING,
                            f"manifest row present (no skip_reason) but not in results CSV",
                            f"ssl_run_id={ssl_id}, mode={mode}",
                        ))
                    else:
                        setattr(entry, cd_attr, StageResult(status_tag, explanation, skip_reason))
                else:
                    # Not in manifest at all → downstream never launched
                    setattr(entry, cd_attr, StageResult(
                        _MISSING, "not in manifest and not in results CSV — downstream never launched",
                        f"ssl_run_id={ssl_id}, mode={mode}",
                    ))

        # ── Stage E: SVM ──────────────────────────────────────────────────
        if ssl_id is None:
            entry.stage_e = StageResult(_MISSING, "no SSL run_id", "")
        elif ssl_id in svm_done:
            entry.stage_e = StageResult(_OK, "found in SVM results CSV", f"ssl_run_id={ssl_id}")
        elif not ckpt_exists:
            entry.stage_e = StageResult(
                _MISSING, "Stage B missing — cannot run SVM",
                f"ssl_run_id={ssl_id}",
            )
        else:
            # Checkpoint exists but SVM not in results
            svm_manifest = entry.manifest_svm
            if svm_manifest:
                svm_skip = svm_manifest[-1]["skip_reason"]
                status_tag, explanation = classify_skip_reason(svm_skip, ckpt_exists, "Stage E")
                if status_tag == _OK:
                    entry.stage_e = StageResult(
                        _MISSING,
                        "manifest row present (no skip_reason) but not in SVM results CSV",
                        f"ssl_run_id={ssl_id}",
                    )
                else:
                    entry.stage_e = StageResult(status_tag, explanation, svm_skip)
            else:
                entry.stage_e = StageResult(
                    _MISSING, "not in manifest and not in SVM results CSV — SVM never launched",
                    f"ssl_run_id={ssl_id}",
                )

        entries.append(entry)

    return entries


# ── Report helpers ────────────────────────────────────────────────────────────


def _hdr(title: str, width: int = 76) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "  (none)"
    all_rows = [headers] + rows
    widths   = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    sep      = "  ".join("-" * w for w in widths)
    lines    = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), sep]
    for row in rows:
        lines.append("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
    return "\n".join(lines)


def _stage_tag(stage: str, status: str) -> str:
    icons = {_OK: "OK", _MISSING: "MISS", _BROKEN: "BRKN",
             _SSL_FAILED: "SFAIL", _SKIP_ERROR: "SERR", _UNKNOWN: "UNK"}
    return icons.get(status, status)


# ── Audit report ──────────────────────────────────────────────────────────────


def emit_audit_report(
    entries: list[StageAuditEntry],
    manifest_path: str,
) -> str:
    ts    = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    lines = [_hdr("RETRY MODELS — STAGE-AWARE AUDIT REPORT")]
    lines.append(f"  Timestamp   : {ts}")
    lines.append(f"  Manifest    : {os.path.relpath(manifest_path, _ROOT)}")
    lines.append(f"  ids_wandb   : {os.path.relpath(_IDS_WANDB_PATH, _ROOT)}")
    lines.append(f"  Run log     : {os.path.relpath(_RUN_LOG_PATH, _ROOT)}")

    # ── 1. Executive summary ───────────────────────────────────────────────
    b_ctr  = Counter(e.stage_b.status  for e in entries)
    ac_ctr = Counter(e.stage_a_freeze.status  for e in entries)
    ad_ctr = Counter(e.stage_a_unfreeze.status for e in entries)
    c_ctr  = Counter(e.stage_c.status  for e in entries)
    d_ctr  = Counter(e.stage_d.status  for e in entries)
    e_ctr  = Counter(e.stage_e.status  for e in entries)

    lines.append(_hdr("1. EXECUTIVE SUMMARY (counts per stage)"))
    tbl = [
        ["B (LeJEPA ckpt)",    str(b_ctr.get(_OK,0)),  str(b_ctr.get(_MISSING,0)), str(b_ctr.get(_SSL_FAILED,0)), str(b_ctr.get(_UNKNOWN,0))],
        ["A freeze (YAML)",    str(ac_ctr.get(_OK,0)), str(ac_ctr.get(_MISSING,0)), "-", "-"],
        ["A unfreeze (YAML)",  str(ad_ctr.get(_OK,0)), str(ad_ctr.get(_MISSING,0)), "-", "-"],
        ["C (MLP freeze)",     str(c_ctr.get(_OK,0)),  str(c_ctr.get(_MISSING,0)), str(c_ctr.get(_SSL_FAILED,0)), str(c_ctr.get(_UNKNOWN,0))],
        ["D (MLP unfreeze)",   str(d_ctr.get(_OK,0)),  str(d_ctr.get(_MISSING,0)), str(d_ctr.get(_SSL_FAILED,0)), str(d_ctr.get(_UNKNOWN,0))],
        ["E (SVM)",            str(e_ctr.get(_OK,0)),  str(e_ctr.get(_MISSING,0)), str(e_ctr.get(_SSL_FAILED,0)), str(e_ctr.get(_UNKNOWN,0))],
    ]
    lines.append(_table(tbl, ["stage", "ok", "missing", "ssl_failed/error", "unknown"]))

    fully_done = sum(
        1 for e in entries
        if all(
            getattr(e, st).status == _OK
            for st in ("stage_b", "stage_a_freeze", "stage_a_unfreeze",
                       "stage_c", "stage_d", "stage_e")
        )
    )
    lines.append(f"\n  Total expected SSL runs : {len(entries)}")
    lines.append(f"  Fully complete (all stages ok) : {fully_done}")
    lines.append(f"  At least one stage missing     : {len(entries) - fully_done}")

    # ── 2. Missing by stage ────────────────────────────────────────────────
    lines.append(_hdr("2. MISSING BY STAGE"))

    def _missing_rows(attr: str) -> list[list[str]]:
        rows = []
        for e in entries:
            s = getattr(e, attr)
            if s.status != _OK:
                rows.append([
                    e.run.dataset_short, str(e.run.split), str(e.run.pct),
                    e.run.init, e.ssl_run_id or "n/a",
                    _stage_tag(attr, s.status), s.reason[:60],
                ])
        return rows

    for stage_label, attr in (
        ("B — LeJEPA checkpoint",  "stage_b"),
        ("A — MLP YAML (freeze)",  "stage_a_freeze"),
        ("A — MLP YAML (unfreeze)", "stage_a_unfreeze"),
        ("C — MLP freeze",         "stage_c"),
        ("D — MLP unfreeze",       "stage_d"),
        ("E — SVM",                "stage_e"),
    ):
        rows = _missing_rows(attr)
        if rows:
            lines.append(f"\n  Stage {stage_label}: {len(rows)} incomplete")
            lines.append(_table(rows, ["dataset", "split", "pct", "init", "ssl_run_id", "tag", "reason"]))

    # ── 3. skip_reason analysis ────────────────────────────────────────────
    lines.append(_hdr("3. skip_reason ANALYSIS (manifest entries with non-empty skip_reason)"))
    skip_rows: list[list[str]] = []
    seen: set[str] = set()
    for e in entries:
        for phase, rows_list in (
            ("SVM",         e.manifest_svm),
            ("MLP_freeze",  e.manifest_mlp_freeze),
            ("MLP_unfreeze", e.manifest_mlp_unfreeze),
        ):
            for row in rows_list:
                sr = row.get("skip_reason", "")
                if not sr:
                    continue
                key = f"{phase}:{row.get('wandb_run_id','')}"
                if key in seen:
                    continue
                seen.add(key)
                # Classify
                ckpt_ok = (e.stage_b.status == _OK)
                tag, expl = classify_skip_reason(sr, ckpt_ok, phase)
                skip_rows.append([
                    e.run.canonical_name[:40],
                    phase, row.get("wandb_run_id",""),
                    sr[:30], tag, expl[:50],
                ])
    if skip_rows:
        lines.append(_table(skip_rows,
            ["experiment", "phase", "run_id", "skip_reason", "classification", "explanation"]))
    else:
        lines.append("  No non-empty skip_reason entries found in manifest.")

    # ── 4. Retry plan overview ────────────────────────────────────────────
    lines.append(_hdr("4. RETRY PLAN OVERVIEW"))
    plan_rows: list[list[str]] = []
    for e in entries:
        ssl_id = e.ssl_run_id or "n/a"
        for stage, stage_attr, mode_label, action_tmpl in (
            ("B", "stage_b",       "lejepa",   "python src/main.py fit [requires manual config]"),
            ("C", "stage_c",       "freeze",   f"python -m src.evaluate.mlp --config <yaml>"),
            ("D", "stage_d",       "unfreeze",  "python -m src.evaluate.mlp --config <yaml>"),
            ("E", "stage_e",       "svm",       "python -m src.evaluate.svm"),
        ):
            s = getattr(e, stage_attr)
            if s.status == _OK:
                continue
            upstream_ok = "yes" if (stage != "B" and e.stage_b.status == _OK) else (
                "n/a" if stage == "B" else "no"
            )
            plan_rows.append([
                e.run.canonical_name[:44], ssl_id, stage, mode_label,
                s.reason[:40], upstream_ok,
            ])
    if plan_rows:
        lines.append(f"  {len(plan_rows)} stage-level retry actions identified.")
        lines.append(_table(plan_rows,
            ["experiment", "ssl_run_id", "stage", "mode", "reason", "upstream_ok"]))
    else:
        lines.append("  All stages complete — no retry actions required.")

    # ── 5. Retry removal plan ─────────────────────────────────────────────
    lines.append(_hdr("5. RETRY REMOVAL PLAN (entries that are now complete)"))
    removal_rows: list[list[str]] = []
    for e in entries:
        # Identify manifest rows that had skip_reason but artifact now exists
        for phase, rows_list, stage_attr in (
            ("SVM",         e.manifest_svm,         "stage_e"),
            ("MLP_freeze",  e.manifest_mlp_freeze,  "stage_c"),
            ("MLP_unfreeze", e.manifest_mlp_unfreeze, "stage_d"),
        ):
            stage_ok = (getattr(e, stage_attr).status == _OK)
            for row in rows_list:
                sr = row.get("skip_reason", "")
                if sr and stage_ok:
                    removal_rows.append([
                        e.run.canonical_name[:44], phase,
                        row.get("wandb_run_id", ""),
                        sr[:30], "REMOVE — artifact now confirmed complete",
                    ])
    if removal_rows:
        lines.append(f"  {len(removal_rows)} stale retry entries can be removed.")
        lines.append(_table(removal_rows,
            ["experiment", "phase", "run_id", "skip_reason", "action"]))
    else:
        lines.append("  No stale retry entries detected.")

    return "\n".join(lines)


# ── Machine-friendly retry plan CSV ──────────────────────────────────────────


def write_retry_plan_csv(
    entries: list[StageAuditEntry],
    out_path: Optional[str] = None,
) -> str:
    """Write retry plan CSV.  Returns the path written."""
    if out_path is None:
        out_path = os.path.join(_RESULTS_DIR, "retry_plan.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        "experiment_name", "ssl_run_id", "stage", "mode",
        "reason", "upstream_ok", "action",
    ]
    rows: list[dict] = []

    for e in entries:
        ssl_id = e.ssl_run_id or ""
        for stage, stage_attr, mode_label, action_fn in (
            ("B", "stage_b",    "lejepa",
             lambda e, ssl_id: "python src/main.py fit [manual]"),
            ("C", "stage_c",    "freeze",
             lambda e, ssl_id: (
                 f"python -m src.evaluate.mlp --config "
                 f"{_yaml_path_for(ssl_id, e.run.canonical_name, 'freeze')}"
                 if ssl_id else "needs ssl_run_id"
             )),
            ("D", "stage_d",    "unfreeze",
             lambda e, ssl_id: (
                 f"python -m src.evaluate.mlp --config "
                 f"{_yaml_path_for(ssl_id, e.run.canonical_name, 'unfreeze')}"
                 if ssl_id else "needs ssl_run_id"
             )),
            ("E", "stage_e",    "svm",
             lambda e, ssl_id: "python -m src.evaluate.svm"),
        ):
            s = getattr(e, stage_attr)
            if s.status == _OK:
                continue
            upstream_ok = (stage != "B" and e.stage_b.status == _OK)
            try:
                action = action_fn(e, ssl_id) if ssl_id else "needs ssl_run_id"
            except Exception:
                action = "see audit report"
            rows.append({
                "experiment_name": e.run.canonical_name,
                "ssl_run_id":      ssl_id,
                "stage":           stage,
                "mode":            mode_label,
                "reason":          s.reason,
                "upstream_ok":     str(upstream_ok),
                "action":          action,
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ── Ray queue helpers (MLP retry — reuses ray_mlp_queue.py) ──────────────────


def _import_ray_queue():
    """Lazy import of Ray queue components to avoid hard dependency."""
    try:
        from src.evaluate.ray_mlp_queue import (
            ExecutionState,
            run_queue,
            _RESULTS_DIR as _RDIR,
        )
        from src.evaluate.ray_mlp import build_ray_run_name
        return run_queue, ExecutionState, build_ray_run_name, _RDIR
    except ImportError as exc:
        raise ImportError(
            "ray or ray_mlp_queue dependencies not available. "
            "Install with: pip install 'ray[tune]'"
        ) from exc


def build_mlp_retry_configs(
    entries: list[StageAuditEntry],
) -> tuple[list[RetryOutcome], list[dict]]:
    """Prepare YAML files and config dicts for MLP retry (Stages C/D)."""
    outcomes: list[RetryOutcome] = []
    configs:  list[dict]         = []

    for entry in entries:
        ssl_run_id = entry.ssl_run_id
        if not ssl_run_id or entry.stage_b.status != _OK:
            continue

        for mode, stage_attr, stage_label in (
            ("freeze",   "stage_c", "C"),
            ("unfreeze", "stage_d", "D"),
        ):
            s = getattr(entry, stage_attr)
            if s.status == _OK:
                continue

            oc = RetryOutcome(
                run=entry.run, mode=mode, ssl_run_id=ssl_run_id,
                stage=stage_label, reason_before=s.reason,
            )
            try:
                yaml_path, created = ensure_mlp_yaml(ssl_run_id, entry.run, mode)
                oc.yaml_path    = yaml_path
                oc.yaml_created = created
                tag = "Created" if created else "Exists "
                print(f"  [YAML] {tag}: {os.path.relpath(yaml_path, _ROOT)}")
            except Exception as exc:
                print(f"  [ERROR] YAML creation failed for "
                      f"{entry.run.canonical_name} [{mode}]: {exc}")
                oc.ray_result = {"status": "yaml_error", "error": str(exc)}
                outcomes.append(oc)
                continue

            with open(yaml_path) as fh:
                cfg = yaml.safe_load(fh)
            cfg["config_path"] = yaml_path
            outcomes.append(oc)
            configs.append(cfg)

    return outcomes, configs


# ── SVM retry (Stage E) ───────────────────────────────────────────────────────


def needs_svm_retry(entries: list[StageAuditEntry]) -> list[StageAuditEntry]:
    """Return entries where Stage E is incomplete but Stage B is ok."""
    return [
        e for e in entries
        if e.stage_e.status != _OK and e.stage_b.status == _OK
    ]


def run_svm_retry(dry_run: bool = False) -> dict:
    """Run src.evaluate.svm as a subprocess to fill missing SVM results.

    SVM script iterates all available experiments; it will naturally skip
    already-completed ones via the W&B check.  We invoke the full script
    rather than duplicating its iteration logic.
    """
    cmd = [sys.executable, "-m", "src.evaluate.svm"]
    print(f"\n[retry-svm] Command: {' '.join(cmd)}")
    if dry_run:
        print("[retry-svm] --dry-run: skipping SVM execution.")
        return {"status": "dry_run"}
    try:
        result = subprocess.run(
            cmd, cwd=_ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return {"status": "ok"}
    except subprocess.CalledProcessError as exc:
        output = (exc.output or "")[-2000:]
        print(f"[retry-svm] FAILED (exit {exc.returncode})\n{output}")
        return {"status": "error", "returncode": exc.returncode, "output": output}


# ── Execute retries ───────────────────────────────────────────────────────────


def execute_retries(
    entries: list[StageAuditEntry],
    gpu_ids: list[int],
    max_concurrent_per_gpu: int,
    cpus_per_experiment: int,
    state_file: str,
    dry_run: bool = False,
    report_only: bool = False,
    log_to_wandb: bool = False,
    update_wandb: bool = False,
) -> tuple[list[RetryOutcome], dict]:
    """Build configs, dispatch MLP via Ray queue and SVM via subprocess.

    Returns (mlp_outcomes, svm_result_dict).
    """
    print("\n[retry] Preparing MLP YAML configs …")
    outcomes, configs = build_mlp_retry_configs(entries)

    svm_needed = needs_svm_retry(entries)
    print(f"[retry] {len(outcomes)} MLP stage(s) to retry, "
          f"{len(svm_needed)} SVM stage(s) pending.")

    if report_only:
        print("[retry] --report-only: skipping execution.")
        return outcomes, {}

    # ── SVM retry (Stage E) ───────────────────────────────────────────────
    svm_result: dict = {}
    if svm_needed:
        print(f"\n[retry-svm] {len(svm_needed)} experiments need SVM evaluation.")
        for e in svm_needed[:5]:
            print(f"  → {e.run.canonical_name}  ssl={e.ssl_run_id}")
        if len(svm_needed) > 5:
            print(f"  … and {len(svm_needed) - 5} more.")
        svm_result = run_svm_retry(dry_run=dry_run)
    else:
        print("[retry-svm] All SVM stages complete.")

    # ── MLP retry via Ray queue (Stages C/D) ─────────────────────────────
    if not configs:
        print("[retry-mlp] No MLP configs to run.")
        return outcomes, svm_result

    if dry_run:
        print(f"\n[retry-mlp] --dry-run: would submit {len(configs)} config(s) to Ray queue.")
        print(f"  GPU profile: {len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots/GPU")
        for cfg in configs:
            mode = "freeze" if cfg.get("freeze_encoder") else "unfreeze"
            print(f"  → {cfg.get('experiment_name','?')} [{mode}] run_id={cfg.get('run_id','?')}")
        return outcomes, svm_result

    run_queue, ExecutionState, build_ray_run_name, _RDIR = _import_ray_queue()
    state = ExecutionState(state_file)

    # Inject local-only SSL run_ids not covered by W&B
    try:
        from src.utils.evaluate import resolve_available_experiments, get_local_run_ids
        available_experiments = resolve_available_experiments(update_wandb=update_wandb)
    except Exception as exc:
        print(f"  [retry][WARN] resolve_available_experiments failed: {exc}")
        available_experiments = {}

    local_ids: Optional[set] = None
    for entry in entries:
        rid = entry.ssl_run_id
        if not rid or rid in available_experiments:
            continue
        if local_ids is None:
            try:
                local_ids = get_local_run_ids()
            except Exception:
                local_ids = set()
                if os.path.isdir(_LOGS_SSL_DIR):
                    for _e in os.scandir(_LOGS_SSL_DIR):
                        if _e.is_dir():
                            ckpt_dir = os.path.join(_e.path, "checkpoints")
                            if os.path.isdir(ckpt_dir) and glob.glob(
                                os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True
                            ):
                                local_ids.add(_e.name)
        if rid in local_ids:
            available_experiments[rid] = entry.run.canonical_name

    print(f"\n[retry-mlp] Submitting {len(configs)} MLP experiment(s) to Ray GPU-slot queue …")
    print(f"  GPU profile : {len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots/GPU")
    print(f"  CPUs/job    : {cpus_per_experiment}")
    print(f"  State file  : {os.path.relpath(state_file, _ROOT)}")

    ray_results: list[dict] = run_queue(
        configs=configs,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=max_concurrent_per_gpu,
        cpus_per_experiment=cpus_per_experiment,
        state=state,
        fail_fast=False,
        log_to_wandb=log_to_wandb,
        ray_address=None,
        available_experiments=available_experiments,
    )

    results_by_key: dict[str, dict] = {}
    for r in ray_results:
        results_by_key[r.get("ray_run_name", "")] = r

    for oc in outcomes:
        if not oc.yaml_path or oc.ray_result.get("status") == "yaml_error":
            continue
        with open(oc.yaml_path) as fh:
            cfg = yaml.safe_load(fh)
        rname = build_ray_run_name(cfg)
        oc.ray_result = results_by_key.get(rname, {})
        if oc.ray_result.get("status") == "ok":
            pth = os.path.join(_MLP_WEIGHTS_DIR, oc.mode, oc.ssl_run_id, "model_best.pth")
            if os.path.isfile(pth):
                oc.weight_found_after = pth

    import pandas as pd
    csv_path = os.path.join(_RESULTS_DIR, "retry_models_again_results.csv")
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    pd.DataFrame(ray_results).to_csv(csv_path, index=False)
    print(f"\n[retry-mlp] Results CSV → {os.path.relpath(csv_path, _ROOT)}")

    return outcomes, svm_result


# ── Post-retry report ─────────────────────────────────────────────────────────


def emit_post_retry_report(
    mlp_outcomes: list[RetryOutcome],
    svm_result: dict,
    manifest_path: str,
) -> str:
    ts    = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    lines = [_hdr("RETRY MODELS — POST-RETRY REPORT")]
    lines.append(f"  Timestamp : {ts}")

    def _oc_status(oc: RetryOutcome) -> str:
        r = oc.ray_result
        if not r:
            return "not_attempted"
        return r.get("status", "unknown")

    n_att     = sum(1 for o in mlp_outcomes if _oc_status(o) != "not_attempted")
    n_rec     = sum(1 for o in mlp_outcomes if _oc_status(o) == "ok" and o.weight_found_after)
    n_fail    = sum(1 for o in mlp_outcomes if _oc_status(o) not in ("not_attempted", "ok"))
    n_skip    = sum(1 for o in mlp_outcomes if _oc_status(o) == "not_attempted")

    lines.append(_hdr("6. MLP RETRY OUTCOMES"))
    outcome_rows: list[list[str]] = []
    for oc in mlp_outcomes:
        st = _oc_status(oc)
        if st == "ok" and oc.weight_found_after:
            label = "RECOVERED"
        elif st == "not_attempted":
            label = "NOT-RUN"
        else:
            label = f"FAILED({st})"
        err = oc.ray_result.get("error", "") if oc.ray_result else ""
        outcome_rows.append([
            oc.run.canonical_name[:44], oc.mode, oc.stage,
            label, "YES" if oc.weight_found_after else "NO",
            str(err)[:40],
        ])
    lines.append(_table(outcome_rows,
        ["run_name", "mode", "stage", "outcome", "weight_after", "error"]))
    lines.append(f"\n  Attempted  : {n_att}")
    lines.append(f"  Recovered  : {n_rec}")
    lines.append(f"  Still failing : {n_fail}")
    lines.append(f"  Not attempted : {n_skip}")

    lines.append(_hdr("7. SVM RETRY OUTCOME"))
    if not svm_result:
        lines.append("  SVM retry was not triggered (all SVM stages complete or report-only mode).")
    else:
        lines.append(f"  Status: {svm_result.get('status', 'unknown')}")
        if svm_result.get("output"):
            lines.append(f"  Output (tail):\n{svm_result['output'][-500:]}")

    if n_fail:
        lines.append(_hdr("8. MLP FAILURE DETAILS"))
        for oc in mlp_outcomes:
            st = _oc_status(oc)
            if st not in ("not_attempted", "ok"):
                lines.append(f"\n  Run   : {oc.run.canonical_name}")
                lines.append(f"  Mode  : {oc.mode} (Stage {oc.stage})")
                lines.append(f"  YAML  : {os.path.relpath(oc.yaml_path, _ROOT) if oc.yaml_path else 'n/a'}")
                lines.append(f"  Status: {st}")
                err = oc.ray_result.get("error", "") if oc.ray_result else ""
                if err:
                    lines.append(f"  Error : {err[:400]}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-aware audit and retry for LeJEPA / MLP / SVM experiments.\n"
            "Checks all 5 pipeline stages and retries only what is truly missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__ or ""),
    )
    parser.add_argument(
        "--manifest", default=_MANIFEST_PATH,
        help="Path to run_manifest.csv (default: artifacts/run_manifest.csv)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2, metavar="N",
        help="Number of GPUs for the MLP retry queue (default: 2).",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu", type=int, default=10, metavar="N",
        help="Max MLP experiments per GPU slot (default: 10).",
    )
    parser.add_argument(
        "--cpus-per-experiment", type=int, default=4, metavar="N",
        help="CPU cores per MLP experiment (default: 4).",
    )
    parser.add_argument(
        "--state-file", default=os.path.join(_ROOT, "results", "retry_again_queue_state.json"),
        help="JSON file for Ray resume state.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print queue plan without executing.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Audit and classify only; do not execute retries.",
    )
    parser.add_argument(
        "--report-out", default=None, metavar="PATH",
        help="Write full report to this file (in addition to stdout).",
    )
    parser.add_argument(
        "--retry-plan-csv", default=None, metavar="PATH",
        help="Write machine-usable retry plan CSV (default: results/retry_plan.csv).",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable W&B metric logging for retried MLP experiments.",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    gpu_ids       = list(range(args.num_gpus))

    n_expected = (len(_EXPECTED_DATASETS) * len(_EXPECTED_SPLITS)
                  * len(_EXPECTED_PCTS)   * len(_EXPECTED_INITS))
    print(f"\n[audit] Manifest         : {os.path.relpath(manifest_path, _ROOT)}")
    print(f"[audit] ids_wandb.json   : {os.path.relpath(_IDS_WANDB_PATH, _ROOT)}")
    print(f"[audit] Expected matrix  : {len(_EXPECTED_DATASETS)} datasets × "
          f"{len(_EXPECTED_SPLITS)} splits × {len(_EXPECTED_PCTS)} pcts × "
          f"{len(_EXPECTED_INITS)} inits = {n_expected} SSL runs")

    entries = audit(manifest_path)

    audit_report = emit_audit_report(entries, manifest_path)
    print(audit_report)

    # Write machine-friendly retry plan CSV
    plan_csv = write_retry_plan_csv(entries, args.retry_plan_csv)
    print(f"\n[plan] Retry plan CSV → {os.path.relpath(plan_csv, _ROOT)}")

    # Stage B summary (manual action required)
    b_missing = [e for e in entries if e.stage_b.status != _OK]
    if b_missing:
        print(f"\n[info] {len(b_missing)} LeJEPA checkpoints missing (Stage B).")
        print(f"[info] These require manual training via: python src/main.py fit ...")
        print(f"[info] See retry_plan.csv for the full list.")

    # Execute retries (Stages C/D via Ray, Stage E via subprocess)
    print(f"\n[retry] Building retry list …")
    n_mlp = sum(
        1 for e in entries
        for s in ("stage_c", "stage_d")
        if getattr(e, s).status != _OK and e.stage_b.status == _OK
    )
    n_svm = len(needs_svm_retry(entries))
    print(f"[retry] MLP stages needing retry: {n_mlp} (Stages C/D)")
    print(f"[retry] SVM stages needing retry: {n_svm} (Stage E)")

    mlp_outcomes, svm_result = execute_retries(
        entries,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        state_file=args.state_file,
        dry_run=args.dry_run,
        report_only=args.report_only,
        log_to_wandb=args.wandb,
        update_wandb=False,  # do not auto-update W&B cache
    )

    post_report = emit_post_retry_report(mlp_outcomes, svm_result, manifest_path)
    print(post_report)

    full_report = audit_report + "\n" + post_report
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n[report] Written to: {args.report_out}")

    def _oc_status(oc: RetryOutcome) -> str:
        return (oc.ray_result or {}).get("status", "not_attempted")

    n_failed = sum(
        1 for o in mlp_outcomes
        if _oc_status(o) not in ("not_attempted", "ok")
    )
    svm_failed = svm_result.get("status") not in (None, "", "ok", "dry_run")
    if n_failed or svm_failed:
        print(f"\n[WARN] {n_failed} MLP retry(ies) and SVM={svm_result.get('status','')} still failing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
