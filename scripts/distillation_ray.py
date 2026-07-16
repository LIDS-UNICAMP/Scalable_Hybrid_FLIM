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

"""distillation_ray.py — Scalable Ray-based launcher for I-JEPA → FLIM distillation.

Generates a full experiment grid over:
    datasets  × splits × percentages × distillation_types × inits

and schedules training jobs using a GPU slot-based Ray scheduler (same design
as ``scripts/run_ssl_ray.py``).

Each experiment runs as a subprocess call to::

    python -m src.modules.distillation_module <args>

Experiment naming convention::

    distillation_<dataset>_split<N>_pct<P>_model<type>

Example runs::

    # Full grid (108 runs across 3 datasets × 3 splits × 6 pcts × 2 types)
    python scripts/distillation_ray.py --num-gpus 1 --max-concurrent-per-gpu 3

    # Direct only, 1 GPU, debug
    python scripts/distillation_ray.py \\
        --inits trunc_normal --distillation-types direct \\
        --datasets eggs --splits 1 --percentages 1 \\
        --num-gpus 1 --max-concurrent-per-gpu 1

    # Dry-run to inspect the queue without executing
    python scripts/distillation_ray.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import ray
except ImportError as _e:
    raise ImportError("ray is not installed. Run: pip install 'ray[tune]'") from _e

# ── Constants ──────────────────────────────────────────────────────────────────

_ALL_DATASETS      = ["eggs", "larvae", "protozoan"]
_ALL_SPLITS        = [1, 2, 3]
_ALL_PCTS          = [1, 5, 25, 50, 75, 100]
_ALL_DIST_TYPES    = ["direct", "hybrid"]
_ALL_INITS         = ["trunc_normal"]

_NUM_CLASSES: dict[str, int] = {
    "eggs":      9,
    "larvae":    2,
    "protozoan": 7,
}

_DEFAULT_DATA_ROOT = os.path.join(
    _ROOT, "data", "to_modules", "new_split_parasito"
)

# FLIM architecture JSON paths per dataset (relative to project root)
_ARCH_BASE: dict[str, str] = {
    "eggs":      os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "eggs"),
    "larvae":    os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "larvae"),
    "protozoan": os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_30_48_a0.5_f5", "protozoan"),
}

# Parasite directory names (for split JSON validation)
_PARASITE_DIR: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

_MANIFEST_PATH = os.path.join(_ROOT, "artifacts", "distillation", "run_manifest.csv")
_MANIFEST_FIELDS = [
    "run_name", "dataset", "split", "percentage", "distillation_type",
    "encoder_init", "status", "skip_reason", "checkpoint_path",
    "embedding_path", "svm_metrics_path",
]

# ── Logging ────────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)


# ── Path resolution ────────────────────────────────────────────────────────────

def _arch_json(dataset: str, split: int) -> str:
    return os.path.join(_ARCH_BASE[dataset], f"train{split}", "architecture.json")


def _split_json(dataset: str, split: int, pct: int) -> str:
    parasite_dir = _PARASITE_DIR[dataset]
    return os.path.join(
        _DEFAULT_DATA_ROOT,
        parasite_dir,
        "splits_incremental",
        f"split{split}",
        f"data_descriptor_perc{pct}.json",
    )


def _run_name(dataset: str, split: int, pct: int, dist_type: str) -> str:
    return f"distillation_{dataset}_split{split}_pct{pct}_model{dist_type}"


# ── Pre-flight validation ──────────────────────────────────────────────────────

def validate_experiment(dataset: str, split: int, pct: int) -> tuple[bool, str]:
    """Return (valid, reason) — reason is empty string when valid."""
    arch = _arch_json(dataset, split)
    if not os.path.isfile(arch):
        return False, f"arch JSON not found: {os.path.relpath(arch, _ROOT)}"

    sjson = _split_json(dataset, split, pct)
    if not os.path.isfile(sjson):
        return False, f"split JSON not found: {os.path.relpath(sjson, _ROOT)}"

    return True, ""


def validate_output_dir() -> tuple[bool, str]:
    """Check that the artifacts directory is writable."""
    out = os.path.join(_ROOT, "artifacts", "distillation")
    try:
        os.makedirs(out, exist_ok=True)
        test = os.path.join(out, ".write_test")
        with open(test, "w") as fh:
            fh.write("ok")
        os.remove(test)
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ── Experiment grid builder ────────────────────────────────────────────────────

def build_experiment_grid(
    datasets:    list[str],
    splits:      list[int],
    percentages: list[int],
    dist_types:  list[str],
    inits:       list[str],
    alpha:       float,
    temperature: float,
    wandb_update: bool,
) -> tuple[list[dict], list[dict]]:
    """Build the full grid and validate each combination.

    Returns:
        (valid_experiments, skipped_records)
        where each element is a dict describing one run.
    """
    valid: list[dict] = []
    skipped: list[dict] = []

    for dataset in datasets:
        for split in splits:
            for pct in percentages:
                for dist_type in dist_types:
                    for init in inits:
                        run_name = _run_name(dataset, split, pct, dist_type)
                        ok, reason = validate_experiment(dataset, split, pct)

                        base = {
                            "run_name":          run_name,
                            "dataset":           dataset,
                            "split":             split,
                            "pct":               pct,
                            "distillation_type": dist_type,
                            "encoder_init":      init,
                            "alpha":             alpha,
                            "temperature":       temperature,
                            "arch_json":         _arch_json(dataset, split),
                            "wandb_update":      wandb_update,
                            "num_classes":       _NUM_CLASSES[dataset],
                            "key":               run_name,
                        }

                        if not ok:
                            _log(
                                f"[SKIP] {run_name}  reason: {reason}", "WARN"
                            )
                            skipped.append({
                                **base,
                                "status":      "skipped",
                                "skip_reason": reason,
                                "checkpoint_path": "",
                                "embedding_path":  "",
                                "svm_metrics_path": "",
                            })
                        else:
                            valid.append(base)

    return valid, skipped


# ── GPU slot scheduler (identical pattern to run_ssl_ray.py) ──────────────────

class GpuSlotScheduler:
    def __init__(self, gpu_ids: list[int], max_per_gpu: int) -> None:
        self.max_per_gpu = max_per_gpu
        self._running: dict[int, int] = {gid: 0 for gid in gpu_ids}

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._running)

    def pick_gpu(self) -> Optional[int]:
        candidates = [
            (count, gid)
            for gid, count in self._running.items()
            if count < self.max_per_gpu
        ]
        return min(candidates)[1] if candidates else None

    def acquire(self, gpu_id: int) -> None:
        self._running[gpu_id] += 1

    def release(self, gpu_id: int) -> None:
        self._running[gpu_id] = max(0, self._running[gpu_id] - 1)

    def total_running(self) -> int:
        return sum(self._running.values())

    def has_free_slot(self) -> bool:
        return any(c < self.max_per_gpu for c in self._running.values())

    def status_line(self) -> str:
        return " | ".join(
            f"GPU {gid}: {self._running[gid]}/{self.max_per_gpu}"
            for gid in self.gpu_ids
        )


# ── Ray remote task ────────────────────────────────────────────────────────────

@ray.remote
def run_distillation_experiment(
    exp:          dict,
    gpu_id:       int,
    project_root: str,
    queue_index:  int = -1,
    total:        int = -1,
    cpus:         int = 4,
) -> dict:
    """Run one distillation training experiment as a subprocess on the assigned GPU."""
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"]       = str(cpus)

    cmd = [
        _sys.executable, "-m", "src.modules.distillation_module",
        "--dataset",           exp["dataset"],
        "--split",             str(exp["split"]),
        "--percentage",        str(exp["pct"]),
        "--distillation-type", exp["distillation_type"],
        "--encoder-init",      exp["encoder_init"],
        "--arch-json",         exp["arch_json"],
        "--run-name",          exp["run_name"],
        "--alpha",             str(exp["alpha"]),
        "--temperature",       str(exp["temperature"]),
        "--num-classes",       str(exp["num_classes"]),
    ]

    if exp.get("wandb_update"):
        cmd.append("--wandb")

    result: dict = {
        "key":         exp["key"],
        "run_name":    exp["run_name"],
        "gpu_id":      gpu_id,
        "queue_index": queue_index,
        "dataset":     exp["dataset"],
        "split":       exp["split"],
        "pct":         exp["pct"],
        "distillation_type": exp["distillation_type"],
        "encoder_init": exp["encoder_init"],
    }

    # Pre-flight: verify arch_json exists
    if not _os.path.isfile(exp["arch_json"]):
        result["status"]    = "error"
        result["returncode"] = -1
        result["error"]     = f"PREFLIGHT FAIL — arch JSON not found: {exp['arch_json']}"
        return result

    try:
        proc = _sp.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            stderr = (proc.stderr or "").strip()
            if len(stderr) > 2000:
                stderr = stderr[:1000] + "\n...[truncated]...\n" + stderr[-1000:]
            result["status"] = "error"
            result["error"]  = f"returncode={proc.returncode}\n{stderr}"
    except Exception as exc:
        result["status"]     = "error"
        result["returncode"] = -1
        result["error"]      = str(exc)

    return result


# ── Queue orchestrator ─────────────────────────────────────────────────────────

def run_queue(
    experiments:            list[dict],
    gpu_ids:                list[int],
    max_concurrent_per_gpu: int,
    cpus_per_experiment:    int,
    fail_fast:              bool,
    ray_address:            Optional[str],
) -> list[dict]:
    if not experiments:
        _log("No experiments to run.", "WARN")
        return []

    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "log_to_driver": True}
    if ray_address:
        ray_kwargs["address"] = ray_address
    else:
        total_cpus = cpus_per_experiment * max_concurrent_per_gpu * len(gpu_ids)
        ray_kwargs["num_cpus"] = total_cpus
        ray_kwargs["num_gpus"] = 0

    if not ray.is_initialized():
        ray.init(**ray_kwargs)
    _log("Ray initialised.")

    scheduler    = GpuSlotScheduler(gpu_ids=gpu_ids, max_per_gpu=max_concurrent_per_gpu)
    total        = len(experiments)
    pending      = list(experiments)
    futures: dict[Any, tuple[int, dict, int]] = {}
    rows:    list[dict] = []
    global_index = 0
    completed    = 0
    had_failure  = False

    _log(
        f"Queue: {total} experiment(s) | "
        f"{len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots = "
        f"{len(gpu_ids) * max_concurrent_per_gpu} max concurrent"
    )
    _log("=" * 70)

    def _submit_next() -> bool:
        nonlocal global_index
        if not pending or (fail_fast and had_failure):
            return False
        gpu_id = scheduler.pick_gpu()
        if gpu_id is None:
            return False
        exp = pending.pop(0)
        idx = global_index
        global_index += 1
        fut = run_distillation_experiment.options(num_cpus=cpus_per_experiment).remote(
            exp, gpu_id=gpu_id, project_root=_ROOT,
            queue_index=idx, total=total, cpus=cpus_per_experiment,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, exp, idx)
        _log(
            f"[{idx + 1:>3}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  pending={len(pending)}  {exp['key']}"
        )
        return True

    while pending and scheduler.has_free_slot():
        _submit_next()

    while futures:
        done_refs, _ = ray.wait(list(futures), num_returns=1, timeout=None)
        ref          = done_refs[0]
        gpu_id, exp, idx = futures.pop(ref)
        scheduler.release(gpu_id)
        completed += 1

        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {
                "key": exp["key"], "gpu_id": gpu_id,
                "queue_index": idx, "status": "ray_error", "error": str(exc),
                "run_name": exp["run_name"],
            }

        rows.append(result)

        if result.get("status") == "ok":
            _log(
                f"[{completed:>3}/{total}] OK     gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  {result['key']}"
            )
        else:
            _log(
                f"[{completed:>3}/{total}] ERROR  gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  {result['key']}: "
                f"{result.get('error', '')}",
                "WARN",
            )
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] Dropping remaining pending experiments.", "WARN")
                pending.clear()

        while pending and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()
    return rows


# ── Manifest writing ───────────────────────────────────────────────────────────

def _write_manifest(rows_ok: list[dict], skipped: list[dict]) -> None:
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    all_records: list[dict] = []

    for r in rows_ok:
        run_name = r.get("run_name", r.get("key", ""))
        ckpt_dir = os.path.join(_ROOT, "artifacts", "distillation", run_name, "checkpoints")
        all_records.append({
            "run_name":          run_name,
            "dataset":           r.get("dataset", ""),
            "split":             r.get("split", ""),
            "percentage":        r.get("pct", ""),
            "distillation_type": r.get("distillation_type", ""),
            "encoder_init":      r.get("encoder_init", ""),
            "status":            r.get("status", "unknown"),
            "skip_reason":       r.get("error", ""),
            "checkpoint_path":   ckpt_dir if r.get("status") == "ok" else "",
            "embedding_path":    "",
            "svm_metrics_path":  "",
        })

    for s in skipped:
        all_records.append({
            "run_name":          s.get("run_name", ""),
            "dataset":           s.get("dataset", ""),
            "split":             s.get("split", ""),
            "percentage":        s.get("pct", ""),
            "distillation_type": s.get("distillation_type", ""),
            "encoder_init":      s.get("encoder_init", ""),
            "status":            "skipped",
            "skip_reason":       s.get("skip_reason", ""),
            "checkpoint_path":   "",
            "embedding_path":    "",
            "svm_metrics_path":  "",
        })

    with open(_MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    _log(f"Manifest written: {_MANIFEST_PATH}  ({len(all_records)} record(s))")


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(
    rows:     list[dict],
    skipped:  list[dict],
    t_start:  float,
    gpu_ids:  list[int],
    max_per_gpu: int,
) -> None:
    elapsed = time.time() - t_start
    n_ok  = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok

    gpu_stats: dict[int, dict] = {gid: {"ok": 0, "error": 0} for gid in gpu_ids}
    for r in rows:
        gid = r.get("gpu_id")
        if gid in gpu_stats:
            gpu_stats[gid]["ok" if r.get("status") == "ok" else "error"] += 1

    print(f"\n{'=' * 70}")
    print("QUEUE SUMMARY — Distillation experiments")
    print(f"{'=' * 70}")
    print(f"  Succeeded      : {n_ok}")
    print(f"  Failed         : {n_err}")
    print(f"  Skipped        : {len(skipped)}")
    print(f"  Total runtime  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"\n  GPU profile    : {len(gpu_ids)} GPU(s) × {max_per_gpu} slots/GPU")
    print("  Per-GPU results:")
    for gid in sorted(gpu_stats):
        s = gpu_stats[gid]
        print(f"    GPU {gid}: {s['ok']} ok, {s['error']} error")
    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for s in skipped[:10]:
            reason = s.get("skip_reason", "")
            print(f"    - {s.get('run_name', '?')}: {reason}")
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('key', '?')}: {str(r.get('error', ''))[:200]}")
    print(f"  Manifest       : {_MANIFEST_PATH}")
    print(f"{'=' * 70}\n")


# ── Dry-run printer ────────────────────────────────────────────────────────────

def _print_dry_run(
    experiments: list[dict],
    skipped:     list[dict],
    gpu_ids:     list[int],
    max_per_gpu: int,
    cpus_per:    int,
) -> None:
    total = len(experiments) + len(skipped)
    max_concurrent = max_per_gpu * len(gpu_ids)

    print(f"\n{'=' * 70}")
    print("DRY RUN — Distillation experiment queue")
    print(f"{'=' * 70}")
    print(f"  GPUs                  : {len(gpu_ids)}  (IDs: {gpu_ids})")
    print(f"  Slots per GPU         : {max_per_gpu}")
    print(f"  Max concurrent total  : {max_concurrent}")
    print(f"  CPUs per experiment   : {cpus_per}")
    print(f"  Experiments queued    : {len(experiments)}")
    print(f"  Skipped (invalid)     : {len(skipped)}")
    print(f"  Total grid size       : {total}")
    print()

    # Group by dataset
    datasets_in_queue = sorted({e["dataset"] for e in experiments})
    for ds in datasets_in_queue:
        exps = [e for e in experiments if e["dataset"] == ds]
        print(f"  {ds} ({len(exps)} runs):")
        for e in exps[:6]:
            print(
                f"    split={e['split']}  pct={e['pct']:>3}%  "
                f"type={e['distillation_type']:<8}  init={e['encoder_init']}"
                f"  → {e['run_name']}"
            )
        if len(exps) > 6:
            print(f"    … and {len(exps) - 6} more")

    if skipped:
        print(f"\n  Skipped runs ({len(skipped)}):")
        for s in skipped[:5]:
            print(f"    [{s['run_name']}]  {s.get('skip_reason', '')}")
        if len(skipped) > 5:
            print(f"    … and {len(skipped) - 5} more")

    print(f"{'=' * 70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scalable Ray-based distillation launcher: trains FLIM CNN students "
            "via knowledge distillation from a frozen I-JEPA teacher."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Grid parameters
    parser.add_argument(
        "--datasets", nargs="+", choices=_ALL_DATASETS, default=_ALL_DATASETS,
        help="Parasite datasets to include in the grid.",
    )
    parser.add_argument(
        "--splits", nargs="+", type=int, choices=_ALL_SPLITS, default=_ALL_SPLITS,
        help="Split indices to include.",
    )
    parser.add_argument(
        "--percentages", nargs="+", type=int, choices=_ALL_PCTS, default=_ALL_PCTS,
        help="Training percentage subsets to include.",
    )
    parser.add_argument(
        "--distillation-types", nargs="+", choices=_ALL_DIST_TYPES, default=_ALL_DIST_TYPES,
        help="Distillation strategies: 'direct' or 'hybrid'.",
    )
    parser.add_argument(
        "--inits", nargs="+", choices=_ALL_INITS, default=_ALL_INITS,
        help="Student encoder initialisation strategies.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="In hybrid mode: weight on L_SSL. L = alpha*L_SSL + (1-alpha)*L_KD.",
    )
    parser.add_argument(
        "--temperature", type=float, default=4.0,
        help="Softmax temperature T for KL divergence (both modes).",
    )

    # Infrastructure
    parser.add_argument(
        "--num-gpus", type=int, default=1, metavar="N",
        help="Number of GPUs available (CUDA device IDs 0..N-1).",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu", type=int, default=3, metavar="N",
        help="Maximum experiments running simultaneously on each GPU.",
    )
    parser.add_argument(
        "--cpus-per-experiment", type=int, default=4, metavar="N",
        help="CPU cores (OMP_NUM_THREADS) per experiment subprocess.",
    )
    parser.add_argument(
        "--ray-address", type=str, default=None, metavar="ADDRESS",
        help="Ray cluster address. Omit to start a local Ray instance.",
    )

    # Experiment flags
    parser.add_argument(
        "--wandb-update", action="store_true",
        help="Pass --wandb to each training subprocess for W&B logging.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop dispatching after the first failure.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the queue plan and exit without running.",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO",
    )

    args = parser.parse_args()

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    t_start  = time.time()
    gpu_ids  = list(range(args.num_gpus))

    # ── Validate output directory ──────────────────────────────────────────
    ok, reason = validate_output_dir()
    if not ok:
        _log(f"Output directory is not writable: {reason}", "WARN")

    # ── Build experiment grid ──────────────────────────────────────────────
    _log("Building experiment grid and validating paths...")
    experiments, skipped = build_experiment_grid(
        datasets=args.datasets,
        splits=args.splits,
        percentages=args.percentages,
        dist_types=args.distillation_types,
        inits=args.inits,
        alpha=args.alpha,
        temperature=args.temperature,
        wandb_update=args.wandb_update,
    )

    _log(
        f"Grid: {len(experiments)} valid runs, {len(skipped)} skipped. "
        f"Max possible: "
        f"{len(args.datasets)}×{len(args.splits)}×{len(args.percentages)}"
        f"×{len(args.distillation_types)} = "
        f"{len(args.datasets)*len(args.splits)*len(args.percentages)*len(args.distillation_types)} runs."
    )

    # ── Dry run ───────────────────────────────────────────────────────────
    if args.dry_run:
        _print_dry_run(
            experiments, skipped, gpu_ids,
            args.max_concurrent_per_gpu, args.cpus_per_experiment,
        )
        _write_manifest([], skipped)
        return

    if not experiments:
        _log("No valid experiments to run. Check data paths and grid parameters.", "WARN")
        _write_manifest([], skipped)
        return

    # ── Run ───────────────────────────────────────────────────────────────
    rows = run_queue(
        experiments=experiments,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        fail_fast=args.fail_fast,
        ray_address=args.ray_address,
    )

    _write_manifest(rows, skipped)
    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu)


if __name__ == "__main__":
    main()
