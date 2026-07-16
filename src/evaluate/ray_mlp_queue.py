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

"""ray_mlp_queue.py — GPU slot-based packed experiment queue.

Runs multiple experiments **concurrently on the same GPU** using slot-based
scheduling.  Each GPU has a configurable number of slots (default: 10).
Experiments are dispatched as soon as a slot is free on any GPU, with explicit
``CUDA_VISIBLE_DEVICES`` pinning so every task uses only its assigned GPU.

Default profile (2× NVIDIA L40S, 128 CPUs):
  • 10 concurrent experiments per GPU  (20 total running at once)
  • 4 CPUs per experiment
  • ~1.5 GB VRAM per experiment  <<  46 GB available per GPU

Scheduling model::

    GPU 0 ──► slots [0..9]  ──► up to 10 experiments simultaneously
    GPU 1 ──► slots [0..9]  ──► up to 10 experiments simultaneously

    New experiment → placed on GPU with fewest running experiments (< max).
    When an experiment finishes → slot freed, next pending job starts immediately.

Usage::

    # All eligible configs, default packing (10/GPU, both freeze and unfreeze)
    python -m src.evaluate.ray_mlp_queue

    # Custom packing density
    python -m src.evaluate.ray_mlp_queue --max-concurrent-per-gpu 5 --cpus-per-experiment 4

    # Preview queue without executing
    python -m src.evaluate.ray_mlp_queue --dry-run

    # Resume interrupted run
    python -m src.evaluate.ray_mlp_queue --resume

    # Enable W&B logging
    python -m src.evaluate.ray_mlp_queue --wandb

    # Stop after first failure
    python -m src.evaluate.ray_mlp_queue --fail-fast
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

import pandas as pd
import yaml

# ─── Project root ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ─── Ray ──────────────────────────────────────────────────────────────────────
try:
    import ray
except ImportError as _e:
    raise ImportError("ray is not installed. Run: pip install 'ray[tune]'") from _e

from src.evaluate.mlp import (
    _RESULTS_DIR,
    get_experiment_configs,
    run_from_yaml,
)
from src.evaluate.ray_mlp import build_ray_run_name
from src.utils.evaluate import resolve_available_experiments
from src.utils.get_names_wandb import build_finetune_name_dict

# ─── Logging ──────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)


# ─── Durable state (resume) ───────────────────────────────────────────────────


class ExecutionState:
    """Persists per-experiment outcomes to JSON so a rerun can skip completed ones."""

    def __init__(self, state_file: str) -> None:
        self.state_file = state_file
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self._state = json.load(f)
            _log(f"Resume state loaded from {self.state_file} "
                 f"({len(self._state)} record(s))")

    def is_completed(self, ray_run_name: str) -> bool:
        return self._state.get(ray_run_name, {}).get("status") == "ok"

    def mark(self, ray_run_name: str, result: dict) -> None:
        self._state[ray_run_name] = {
            "status": result.get("status", "unknown"),
            "timestamp": time.time(),
            "kappa": result.get("kappa"),
            "acc": result.get("acc"),
            "f1": result.get("f1"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def completed_names(self) -> list[str]:
        return [k for k, v in self._state.items() if v.get("status") == "ok"]


# ─── GPU slot scheduler ───────────────────────────────────────────────────────


class GpuSlotScheduler:
    """Tracks per-GPU concurrency slots and assigns experiments to the least-loaded GPU.

    No GPU is treated as exclusively owned.  A GPU is available while its
    running count is below ``max_per_gpu``.  When a slot is freed, the next
    pending experiment can be placed there immediately.
    """

    def __init__(self, gpu_ids: list[int], max_per_gpu: int) -> None:
        self.max_per_gpu = max_per_gpu
        self._running: dict[int, int] = {gid: 0 for gid in gpu_ids}

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._running)

    def pick_gpu(self) -> Optional[int]:
        """Return the GPU with the fewest running experiments that still has a free slot."""
        candidates = [
            (count, gid)
            for gid, count in self._running.items()
            if count < self.max_per_gpu
        ]
        if not candidates:
            return None
        return min(candidates)[1]

    def acquire(self, gpu_id: int) -> None:
        self._running[gpu_id] += 1

    def release(self, gpu_id: int) -> None:
        self._running[gpu_id] = max(0, self._running[gpu_id] - 1)

    def total_running(self) -> int:
        return sum(self._running.values())

    def has_free_slot(self) -> bool:
        return any(c < self.max_per_gpu for c in self._running.values())

    def status_line(self) -> str:
        parts = [
            f"GPU {gid}: {self._running[gid]}/{self.max_per_gpu}"
            for gid in self.gpu_ids
        ]
        return " | ".join(parts)


# ─── Ray remote experiment task ───────────────────────────────────────────────


@ray.remote
def run_experiment(
    cfg: dict,
    gpu_id: int,
    finetune_names: dict,
    available_experiments: dict,
    queue_index: int = -1,
    total: int = -1,
    log_to_wandb: bool = True,
) -> dict:
    """Fine-tune one experiment on a specific GPU.

    Delegates all training and W&B logging to ``mlp.run_from_yaml`` — the
    same code path as ``python -m src.evaluate.mlp``.  The only Ray-specific
    work here is pinning ``CUDA_VISIBLE_DEVICES`` before any CUDA context is
    created.

    Args:
        cfg:                   Experiment config dict (from ``get_experiment_configs``).
        gpu_id:                Physical GPU index to bind (sets CUDA_VISIBLE_DEVICES).
        finetune_names:        ``{run_id: wandb_name}`` pre-computed in main process.
        available_experiments: ``{run_id: run_name}`` pre-computed in main process.
        queue_index:           Global submission index (for logging).
        total:                 Total experiments in the queue (for logging).
        log_to_wandb:          Whether to log metrics and upload to W&B.

    Returns:
        Result dict: ray_run_name, gpu_id, status, kappa, acc, f1, error, ...
    """
    # ── Pin to assigned GPU before any CUDA init ──────────────────────────────
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import sys as _sys
    if _ROOT not in _sys.path:
        _sys.path.insert(0, _ROOT)

    from src.evaluate.mlp import run_from_yaml
    from src.evaluate.ray_mlp import build_ray_run_name

    ray_run_name = build_ray_run_name(cfg)
    rows: list[dict] = []
    report: dict = {
        "executed": [],
        "skipped_missing_weights": [],
        "skipped_errors": [],
        "collision_resolved": [],
    }

    # Protozoan-cysts FLIM checkpoints were trained with 30 conv2 channels
    # (FLIM kernel selection produced 30 filters, not the default 32).
    dataset_name = cfg.get("dataset_name", "")
    init_type = cfg.get("initialization_type", "")
    if "protozoan" in dataset_name and init_type == "flim":
        conv2_channels = 30
    else:
        conv2_channels = None

    run_from_yaml(
        cfg["config_path"], rows, finetune_names, report, available_experiments,
        log_to_wandb=log_to_wandb,
        conv2_channels=conv2_channels,
    )

    result: dict = rows[0] if rows else {
        "status": "error",
        "error": "run_from_yaml returned no result",
        "kappa": float("nan"),
        "acc": float("nan"),
        "f1": float("nan"),
    }
    result.update({
        "ray_run_name": ray_run_name,
        "gpu_id": gpu_id,
        "queue_index": queue_index,
    })
    return result


# ─── Queue orchestrator ───────────────────────────────────────────────────────


def run_queue(
    configs: list[dict],
    gpu_ids: list[int],
    max_concurrent_per_gpu: int,
    cpus_per_experiment: int,
    state: ExecutionState,
    fail_fast: bool,
    log_to_wandb: bool,
    ray_address: Optional[str],
    available_experiments: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Dispatch experiments using GPU slot-based scheduling.

    A new experiment is submitted as soon as any GPU has a free slot.
    The GPU with the fewest currently running experiments is preferred
    (least-loaded assignment).

    Args:
        available_experiments: Optional pre-built ``{run_id: run_name}`` dict.
            When provided, skips the W&B API call and uses this dict directly.
            Useful for run_ids that have local checkpoints but no W&B entry.

    Returns list of result dicts (one per completed experiment).
    """
    if not configs:
        _log("No experiments to run.", "WARN")
        return []

    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # ── Pre-compute W&B name dicts once in the main process ───────────────────
    # These are serialised into each Ray worker so workers don't call W&B API.
    if available_experiments is None:
        _log("Resolving available experiments (W&B + local checkpoints)…")
        available_experiments = resolve_available_experiments()
    else:
        _log(f"Using caller-supplied available_experiments ({len(available_experiments)} run(s))")
    _log(f"Available: {len(available_experiments)} run(s)")
    finetune_names = build_finetune_name_dict(experiments=available_experiments)

    # ── Init Ray ──────────────────────────────────────────────────────────────
    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "log_to_driver": True}
    if ray_address:
        ray_kwargs["address"] = ray_address
    else:
        total_cpus = cpus_per_experiment * max_concurrent_per_gpu * len(gpu_ids)
        ray_kwargs["num_cpus"] = total_cpus
        ray_kwargs["num_gpus"] = 0  # GPU managed manually via CUDA_VISIBLE_DEVICES

    if not ray.is_initialized():
        ray.init(**ray_kwargs)

    _log("Ray initialised.")

    scheduler = GpuSlotScheduler(gpu_ids=gpu_ids, max_per_gpu=max_concurrent_per_gpu)
    total = len(configs)
    pending_queue = list(configs)
    futures: dict[Any, tuple[int, dict, int]] = {}  # future → (gpu_id, cfg, idx)
    rows: list[dict] = []
    global_index = 0
    completed = 0
    had_failure = False

    _log(
        f"Queue initialized — {total} experiment(s) | "
        f"{len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots = "
        f"{len(gpu_ids) * max_concurrent_per_gpu} max concurrent"
    )
    _log("=" * 70)

    def _submit_next() -> bool:
        nonlocal global_index
        if not pending_queue or (fail_fast and had_failure):
            return False
        gpu_id = scheduler.pick_gpu()
        if gpu_id is None:
            return False

        cfg = pending_queue.pop(0)
        idx = global_index
        global_index += 1
        rname = build_ray_run_name(cfg)

        fut = run_experiment.options(num_cpus=cpus_per_experiment).remote(
            cfg,
            gpu_id=gpu_id,
            finetune_names=finetune_names,
            available_experiments=available_experiments,
            queue_index=idx,
            total=total,
            log_to_wandb=log_to_wandb,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, cfg, idx)

        _log(
            f"[{idx + 1:>4}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  pending={len(pending_queue)}  {rname}"
        )
        return True

    # Initial burst: fill all slots
    while pending_queue and scheduler.has_free_slot():
        _submit_next()

    # Collect-and-dispatch loop
    while futures:
        done_refs, _ = ray.wait(list(futures), num_returns=1, timeout=None)
        ref = done_refs[0]
        gpu_id, cfg, idx = futures.pop(ref)
        scheduler.release(gpu_id)
        completed += 1

        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {
                "ray_run_name": build_ray_run_name(cfg),
                "gpu_id": gpu_id,
                "queue_index": idx,
                "status": "ray_error",
                "error": str(exc),
            }

        rows.append(result)
        state.mark(result.get("ray_run_name", "unknown"), result)

        status = result.get("status", "?")
        rname = result.get("ray_run_name", "?")
        kappa = result.get("kappa", float("nan"))
        acc = result.get("acc", float("nan"))
        f1 = result.get("f1", float("nan"))

        if status == "ok":
            _log(
                f"[{completed:>4}/{total}] OK      gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  running={scheduler.total_running()}  "
                f"pending={len(pending_queue)}\n"
                f"{'':>18}kappa={kappa:.4f}  acc={acc:.4f}  f1={f1:.4f}  {rname}"
            )
        else:
            _log(
                f"[{completed:>4}/{total}] ERROR   gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  {rname}: {result.get('error', '')}",
                "WARN",
            )
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] Dropping all pending experiments.", "WARN")
                pending_queue.clear()

        # Fill freed slot immediately
        while pending_queue and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()
    return rows


# ─── CSV ──────────────────────────────────────────────────────────────────────

_META_COLS = [
    "ray_run_name", "gpu_id", "queue_index",
    "run_id", "experiment_name", "dataset_name", "split_id",
    "percentage", "initialization_type", "freeze_encoder", "config_path",
]
_METRIC_COLS = ["kappa", "acc", "f1"]
_EXTRA_COLS = ["status", "error", "weights_path", "metrics_path"]
_COL_ORDER = _META_COLS + _METRIC_COLS + _EXTRA_COLS


def _save_csv(rows: list[dict], csv_path: str) -> None:
    df = pd.DataFrame(rows)
    if not df.empty:
        remaining = [c for c in df.columns if c not in _COL_ORDER]
        df = df.reindex(columns=_COL_ORDER + remaining)
    df.to_csv(csv_path, index=False)
    _log(f"Results CSV → {csv_path}")


# ─── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(
    rows: list[dict],
    skipped: list[str],
    t_start: float,
    gpu_ids: list[int],
    max_concurrent_per_gpu: int,
) -> None:
    elapsed = time.time() - t_start
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok

    gpu_stats: dict[int, dict] = {gid: {"ok": 0, "error": 0} for gid in gpu_ids}
    for r in rows:
        gid = r.get("gpu_id")
        if gid in gpu_stats:
            gpu_stats[gid]["ok" if r.get("status") == "ok" else "error"] += 1

    print(f"\n{'=' * 70}")
    print("QUEUE SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Succeeded      : {n_ok}")
    print(f"  Failed         : {n_err}")
    print(f"  Skipped (done) : {len(skipped)}")
    print(f"  Total runtime  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"\n  GPU profile    : {len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots/GPU")
    print("  Per-GPU results:")
    for gid in sorted(gpu_stats):
        s = gpu_stats[gid]
        print(f"    GPU {gid}: {s['ok']} ok, {s['error']} error")
    if skipped:
        print(f"\n  Skipped (already completed — {len(skipped)}):")
        for name in skipped[:10]:
            print(f"    - {name}")
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('ray_run_name', '?')}: {r.get('error', '')}")
    print(f"{'=' * 70}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        description=(
            "GPU slot-based packed experiment queue. "
            "Runs multiple experiments concurrently on each GPU "
            "(default: 10 per GPU, 4 CPUs each)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2, metavar="N",
        help="Number of GPUs available (CUDA device IDs 0..N-1).",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu", type=int, default=10, metavar="N",
        help="Maximum experiments running simultaneously on each GPU.",
    )
    parser.add_argument(
        "--cpus-per-experiment", type=int, default=4, metavar="N",
        help="CPU cores reserved per experiment (DataLoader workers).",
    )
    parser.add_argument(
        "--mode", choices=["freeze", "unfreeze", "all"], default="all",
        help="Which encoder-freeze mode configs to include.",
    )
    parser.add_argument(
        "--state-file", type=str,
        default=os.path.join(_ROOT, "results", "queue_state.json"),
        help="JSON file for resume state.",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already recorded as 'ok'.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop dispatching after the first failure.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B metric logging for each experiment.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the queue plan and exit without running.")
    parser.add_argument(
        "--experiment-filter", type=str, default=None, metavar="PATTERN",
        help="Only run experiments whose run_id or name contains PATTERN.",
    )
    parser.add_argument(
        "--ray-address", type=str, default=None, metavar="ADDRESS",
        help="Ray cluster address. Omit to start a local Ray instance.",
    )
    parser.add_argument(
        "--dataset-group",
        choices=["all", "protozoan", "eggs", "larvae"],
        default="all",
        help="Filter experiments to a single dataset group.",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO")
    parser.add_argument(
        "--wandb-update", action="store_true",
        help=(
            "Refresh the local W&B metadata cache (ids_wandb.json) from the "
            "live W&B API before running.  By default the cache is used as-is."
        ),
    )
    args = parser.parse_args()

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    t_start = time.time()
    gpu_ids = list(range(args.num_gpus))

    # ── Resolve W&B metadata once (avoids duplicate API calls) ────────────────
    _log("=" * 70)
    _log("Resolving available experiments (W&B cache ∩ local checkpoints)…")
    available_experiments = resolve_available_experiments(update_wandb=args.wandb_update)
    _log(f"Available: {len(available_experiments)} run(s)")

    # ── Discover configs ───────────────────────────────────────────────────────
    _log("Discovering experiment configs…")
    all_configs = get_experiment_configs(
        mode=args.mode, available_experiments=available_experiments
    )
    configs = [
        c for c in all_configs
        if not c.get("experiment_name", "").startswith("finetune_")
    ]

    # ── Dataset-group filter ───────────────────────────────────────────────────
    _DATASET_GROUP_MAP = {
        "protozoan": "protozoan-cysts",
        "eggs": "helminth-eggs",
        "larvae": "helminth-larvae",
    }
    if args.dataset_group != "all":
        target_dataset = _DATASET_GROUP_MAP.get(args.dataset_group, args.dataset_group)
        configs = [c for c in configs if c.get("dataset_name", "") == target_dataset]
        _log(f"Dataset-group filter '{args.dataset_group}' → {len(configs)} experiment(s)")

    if args.experiment_filter:
        needle = args.experiment_filter.lower()
        configs = [
            c for c in configs
            if needle in c.get("run_id", "").lower()
            or needle in c.get("experiment_name", "").lower()
        ]
        if not configs:
            _log(f"No experiment matched --experiment-filter '{args.experiment_filter}'.", "WARN")
            return

    # ── Resume ────────────────────────────────────────────────────────────────
    state = ExecutionState(args.state_file)
    skipped: list[str] = []
    if args.resume:
        before = len(configs)
        to_run, to_skip = [], []
        for cfg in configs:
            (to_skip if state.is_completed(build_ray_run_name(cfg)) else to_run).append(cfg)
        skipped = [build_ray_run_name(c) for c in to_skip]
        configs = to_run
        _log(f"Resume: {len(skipped)} skipped, {len(configs)} remaining (of {before})")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        max_concurrent = args.max_concurrent_per_gpu * args.num_gpus
        print(f"\n{'=' * 70}")
        print("DRY RUN — GPU slot profile")
        print(f"{'=' * 70}")
        print(f"  GPUs                   : {args.num_gpus}  (IDs: {gpu_ids})")
        print(f"  Slots per GPU          : {args.max_concurrent_per_gpu}")
        print(f"  Max concurrent total   : {max_concurrent}")
        print(f"  CPUs per experiment    : {args.cpus_per_experiment}")
        print(f"  Experiments queued     : {len(configs)}")
        print(f"  W&B logging            : {args.wandb}")
        print()
        for i, cfg in enumerate(configs[:30]):
            freeze = "freeze" if cfg.get("freeze_encoder", True) else "unfreeze"
            print(f"  {i + 1:>3}. [{freeze}] {build_ray_run_name(cfg)}")
        if len(configs) > 30:
            print(f"  … and {len(configs) - 30} more")
        print(f"{'=' * 70}\n")
        return

    if not configs:
        _log("No experiments to run. Exiting.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    rows = run_queue(
        configs=configs,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        state=state,
        fail_fast=args.fail_fast,
        log_to_wandb=args.wandb,
        ray_address=args.ray_address,
        available_experiments=available_experiments,
    )

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(_RESULTS_DIR, "ray_mlp_queue_results.csv")
    _save_csv(rows, csv_path)

    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu)


if __name__ == "__main__":
    main()
