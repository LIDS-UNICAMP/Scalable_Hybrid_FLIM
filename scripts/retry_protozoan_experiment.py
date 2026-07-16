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

"""retry_protozoan_experiment.py — Re-train protozoan-cysts SSL experiments.

Re-trains protozoan-cysts LeJEPA SSL pre-training experiments using the
30-channel conv2 architecture (matching FLIM kernel selection for protozoan),
so all initializations are architecturally comparable.

Runs the same commands as scripts/run_protozoan_ssl.sh, but dispatched via
the same GPU slot-based Ray scheduler as src/evaluate/ray_mlp_queue.py.

Default profile (2× GPUs):
  • 10 concurrent experiments per GPU  (20 total running at once)
  • 4 CPUs per experiment
  • Runs xavier, random, he only (flim already trained correctly)

Scheduling model::

    GPU 0 ──► slots [0..9]  ──► up to 10 experiments simultaneously
    GPU 1 ──► slots [0..9]  ──► up to 10 experiments simultaneously

    New experiment → placed on GPU with fewest running experiments (< max).
    When an experiment finishes → slot freed, next pending job starts immediately.

Usage::

    # Defaults: xavier+random+he, 2 GPUs, 10 slots/GPU
    python scripts/retry_protozoan_experiment.py

    # Include flim too
    python scripts/retry_protozoan_experiment.py --inits xavier random he flim

    # Preview queue
    python scripts/retry_protozoan_experiment.py --dry-run

    # Resume interrupted run
    python scripts/retry_protozoan_experiment.py --resume

    # Custom GPU/concurrency
    python scripts/retry_protozoan_experiment.py --num-gpus 4 --max-concurrent-per-gpu 5

    # Only a specific split
    python scripts/retry_protozoan_experiment.py --splits 1 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Optional

# ─── Project root ─────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ─── Ensure 30-channel arch JSONs exist in THIS repo copy ─────────────────────

# Embedded 30-channel protozoan architecture (ch24_30_48_a0.5_f5).
# Identical for all 3 splits — no dependency on source files needed.
_PROTOZOAN_30CH_ARCH = {
    "stdev_factor": 0.01,
    "nlayers": 3,
    "apply_intrinsic_atrous": False,
    "layer1": {
        "conv": {
            "kernel_size": [5, 5, 0],
            "nkernels_per_marker": 24,
            "dilation_rate": [1, 1, 0],
            "nkernels_per_image": 24,
            "noutput_channels": 24,
        },
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
    "layer2": {
        "conv": {
            "kernel_size": [5, 5, 0],
            "nkernels_per_marker": 30,
            "dilation_rate": [1, 1, 0],
            "nkernels_per_image": 30,
            "noutput_channels": 30,
        },
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
    "layer3": {
        "conv": {
            "kernel_size": [5, 5, 0],
            "nkernels_per_marker": 48,
            "dilation_rate": [1, 1, 0],
            "nkernels_per_image": 48,
            "noutput_channels": 48,
        },
        "relu": True,
        "pooling": {"type": "max_pool", "size": [3, 3, 0], "stride": 2},
    },
}


def _ensure_protozoan_arch_jsons() -> None:
    """Create 30-channel protozoan architecture JSONs if they don't already exist.

    Uses an embedded template — no dependency on ch24_32_48_a0.5_f5 source files.
    This runs at startup so the files are always present in the current repo copy,
    regardless of whether ``git pull`` / rsync has been done on another machine.
    """
    dst_base = os.path.join(_ROOT, "data", "to_mateus", "model",
                            "ch24_30_48_a0.5_f5", "protozoan")

    for split_n in [1, 2, 3]:
        dst_dir  = os.path.join(dst_base, f"train{split_n}")
        dst_json = os.path.join(dst_dir, "architecture.json")

        if os.path.exists(dst_json):
            continue  # already present

        os.makedirs(dst_dir, exist_ok=True)
        with open(dst_json, "w") as f:
            json.dump(_PROTOZOAN_30CH_ARCH, f, indent=2)

        print(f"[setup] Created arch JSON: {os.path.relpath(dst_json, _ROOT)}", flush=True)


def _ensure_protozoan_model_configs() -> None:
    """Create/update xavier/random/he protozoan model YAML configs.

    Always writes absolute arch_json paths so configs work regardless of CWD
    or which machine the script runs on. Overwrites any existing config that
    still contains a relative (non-absolute) arch_json path.
    """
    configs_dir = os.path.join(_ROOT, "configs", "model")
    arch_base   = os.path.join(_ROOT, "data", "to_mateus", "model",
                               "ch24_30_48_a0.5_f5", "protozoan")

    for split_n in [1, 2, 3]:
        arch_json_abs = os.path.join(arch_base, f"train{split_n}", "architecture.json")
        for init in ["xavier", "random", "he"]:
            fname = os.path.join(configs_dir,
                                 f"lejepa_line_{init}_protozoan_train{split_n}.yaml")

            # Skip only if the file already has the correct absolute path
            if os.path.exists(fname):
                with open(fname) as fh:
                    existing = fh.read()
                if arch_json_abs in existing:
                    continue  # already correct — nothing to do
                # Falls through: overwrite with absolute path

            content = (
                f"# LeJEPA Line — encoder_init: {init} — protozoan-cysts / split_{split_n}\n"
                f"# Uses 30-channel conv2 architecture (matching FLIM kernel selection).\n"
                f"model:\n"
                f"  class_path: src.modules.lejepa_line_module.LejepaLineModule\n"
                f"  init_args:\n"
                f"    arch_json: {arch_json_abs}\n"
                f"    encoder_init: {init}\n"
                f"    flim_weights_path: null\n"
                f"    in_channels: 3\n"
                f"    proj_dim: 256\n"
                f"    proj_hidden: 2048\n"
                f"    lam: 0.05\n"
                f"    sigreg_type: real\n"
                f"    lr: 3.0e-3\n"
                f"    weight_decay: 5.0e-2\n"
                f"    warmup_epochs: 10\n"
            )
            with open(fname, "w") as f:
                f.write(content)
            print(f"[setup] Wrote model config (abs path): {os.path.relpath(fname, _ROOT)}", flush=True)

# ─── Ray ──────────────────────────────────────────────────────────────────────
try:
    import ray
except ImportError as _e:
    raise ImportError("ray is not installed.  Run: pip install 'ray[tune]'") from _e

# ─── Experiment definition ────────────────────────────────────────────────────

_SPLITS = [1, 2, 3]
_PCTS   = [1, 5, 25, 50, 75, 100]
_INITS  = ["xavier", "random", "he", "flim"]

_STATE_FILE_DEFAULT = os.path.join(_ROOT, "results", "protozoan_ssl_state.json")
_RESULTS_DIR        = os.path.join(_ROOT, "results")


def _model_config(init: str, split_n: int) -> str:
    if init == "flim":
        return f"configs/model/lejepa_line_flim_protozoan_train{split_n}.yaml"
    return f"configs/model/lejepa_line_{init}_protozoan_train{split_n}.yaml"


def build_experiments(
    splits: list[int] = _SPLITS,
    pcts:   list[int] = _PCTS,
    inits:  list[str] = _INITS,
) -> list[dict]:
    """Return list of experiment dicts, one per (split, pct, init)."""
    exps = []
    for split_n in splits:
        for pct in pcts:
            for init in inits:
                exps.append({
                    "split":       split_n,
                    "pct":         pct,
                    "init":        init,
                    "data_config": f"configs/data/percentage/protozoan-cysts_split_{split_n}/{pct}/lejepa_line.yaml",
                    "model_config": _model_config(init, split_n),
                    "key":         f"protozoan_split{split_n}_pct{pct}_{init}",
                })
    return exps


# ─── Logging ──────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)


# ─── Durable state (resume) ───────────────────────────────────────────────────


class ExecutionState:
    """Persists per-experiment outcomes so a rerun can skip completed ones."""

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

    def is_completed(self, key: str) -> bool:
        return self._state.get(key, {}).get("status") == "ok"

    def mark(self, key: str, result: dict) -> None:
        self._state[key] = {
            "status":    result.get("status", "unknown"),
            "timestamp": time.time(),
            "returncode": result.get("returncode"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def completed_keys(self) -> list[str]:
        return [k for k, v in self._state.items() if v.get("status") == "ok"]


# ─── GPU slot scheduler ───────────────────────────────────────────────────────


class GpuSlotScheduler:
    """Tracks per-GPU concurrency slots; assigns experiments to least-loaded GPU."""

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
        return " | ".join(
            f"GPU {gid}: {self._running[gid]}/{self.max_per_gpu}"
            for gid in self.gpu_ids
        )


# ─── Ray remote training task ─────────────────────────────────────────────────


@ray.remote
def run_ssl_experiment(
    exp:           dict,
    gpu_id:        int,
    project_root:  str,
    queue_index:   int = -1,
    total:         int = -1,
    cpus:          int = 4,
) -> dict:
    """Run one SSL pre-training experiment as a subprocess on a specific GPU.

    Pins ``CUDA_VISIBLE_DEVICES`` before launching the child process so that
    the training script uses only the assigned GPU.

    Args:
        exp:           Experiment dict (split, pct, init, data_config, model_config, key).
        gpu_id:        Physical GPU index to bind.
        project_root:  Absolute path to the project root (passed explicitly so
                       it resolves correctly inside Ray workers).
        queue_index:   Position in the global queue (for logging).
        total:         Total experiments in the queue (for logging).
        cpus:          OMP_NUM_THREADS for the subprocess.

    Returns:
        Result dict: key, gpu_id, status, returncode, error.
    """
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"]       = str(cpus)

    # Absolute arch_json path — overrides whatever is in the YAML config.
    # This avoids any CWD-relative path issues with jsonargparse/parse_architecture.
    arch_json_abs = _os.path.join(
        project_root, "data", "to_mateus", "model",
        "ch24_30_48_a0.5_f5", "protozoan",
        f"train{exp['split']}", "architecture.json",
    )

    run_name = (
        f"lejepa_line_protozoan-cysts"
        f"_split_{exp['split']}"
        f"_pct_{exp['pct']}"
        f"_model_{exp['init']}"
    )

    cmd = [
        _sys.executable, "src/main.py", "fit",
        "--config", "configs/default.yaml",
        "--config", exp["data_config"],
        "--config", exp["model_config"],
        f"--model.init_args.arch_json={arch_json_abs}",
        f"--trainer.logger.init_args.name={run_name}",
        "--trainer.accelerator=gpu",
        "--trainer.devices=1",
    ]

    # Pre-flight: verify arch JSON exists before launching subprocess
    if not _os.path.exists(arch_json_abs):
        return {
            "key": exp["key"], "gpu_id": gpu_id, "queue_index": queue_index,
            "status": "error", "returncode": -1,
            "error": f"PREFLIGHT FAIL — arch JSON not found: {arch_json_abs}",
        }

    result: dict = {"key": exp["key"], "gpu_id": gpu_id, "queue_index": queue_index}
    try:
        proc = _sp.run(
            cmd,
            cwd=project_root,       # explicit root — avoids __file__ in Ray worker
            capture_output=True,    # capture stderr so errors are visible in logs
            text=True,
        )
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            # Show first 1000 + last 1000 chars of stderr for diagnosis
            stderr = (proc.stderr or "").strip()
            if len(stderr) > 2000:
                stderr_excerpt = (
                    stderr[:1000] + "\n...[truncated]...\n" + stderr[-1000:]
                )
            else:
                stderr_excerpt = stderr
            result["status"] = "error"
            result["error"]  = f"returncode={proc.returncode}\n{stderr_excerpt}"
    except Exception as exc:
        result["status"]     = "error"
        result["returncode"] = -1
        result["error"]      = str(exc)

    return result


# ─── Queue orchestrator ───────────────────────────────────────────────────────


def run_queue(
    experiments:            list[dict],
    gpu_ids:                list[int],
    max_concurrent_per_gpu: int,
    cpus_per_experiment:    int,
    state:                  ExecutionState,
    fail_fast:              bool,
    ray_address:            Optional[str],
) -> list[dict]:
    """Dispatch SSL training experiments using GPU slot-based scheduling."""
    if not experiments:
        _log("No experiments to run.", "WARN")
        return []

    os.makedirs(_RESULTS_DIR, exist_ok=True)

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

    scheduler    = GpuSlotScheduler(gpu_ids=gpu_ids, max_per_gpu=max_concurrent_per_gpu)
    total        = len(experiments)
    pending      = list(experiments)
    futures: dict[Any, tuple[int, dict, int]] = {}  # future → (gpu_id, exp, idx)
    rows: list[dict] = []
    global_index = 0
    completed    = 0
    had_failure  = False

    _log(
        f"Queue initialized — {total} experiment(s) | "
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

        fut = run_ssl_experiment.options(num_cpus=cpus_per_experiment).remote(
            exp,
            gpu_id=gpu_id,
            project_root=_ROOT,
            queue_index=idx,
            total=total,
            cpus=cpus_per_experiment,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, exp, idx)

        _log(
            f"[{idx + 1:>3}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  pending={len(pending)}  {exp['key']}"
        )
        return True

    # Initial burst: fill all slots
    while pending and scheduler.has_free_slot():
        _submit_next()

    # Collect-and-dispatch loop
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
                "key":       exp["key"],
                "gpu_id":    gpu_id,
                "queue_index": idx,
                "status":    "ray_error",
                "error":     str(exc),
            }

        rows.append(result)
        state.mark(result["key"], result)

        status = result.get("status", "?")
        if status == "ok":
            _log(
                f"[{completed:>3}/{total}] OK      gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  running={scheduler.total_running()}  "
                f"pending={len(pending)}  {result['key']}"
            )
        else:
            _log(
                f"[{completed:>3}/{total}] ERROR   gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  {result['key']}: "
                f"{result.get('error', '')}",
                "WARN",
            )
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] Dropping all pending experiments.", "WARN")
                pending.clear()

        while pending and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()
    return rows


# ─── Summary ──────────────────────────────────────────────────────────────────


def _print_summary(
    rows:                   list[dict],
    skipped:                list[str],
    t_start:                float,
    gpu_ids:                list[int],
    max_concurrent_per_gpu: int,
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
    print("QUEUE SUMMARY — protozoan SSL re-training")
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
                print(f"    - {r.get('key', '?')}: {r.get('error', '')}")
    print(f"{'=' * 70}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-train protozoan-cysts SSL experiments with 30-channel architecture.\n"
            "Equivalent to run_protozoan_ssl.sh but dispatched via GPU slot-based Ray scheduler."
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
        help="CPU cores (OMP_NUM_THREADS) per experiment.",
    )
    parser.add_argument(
        "--inits", nargs="+",
        choices=_INITS, default=["xavier", "random", "he"],
        help="Which initializations to run (flim excluded by default — already trained).",
    )
    parser.add_argument(
        "--splits", nargs="+", type=int,
        choices=_SPLITS, default=_SPLITS,
        help="Which data splits to include.",
    )
    parser.add_argument(
        "--pcts", nargs="+", type=int,
        choices=_PCTS, default=_PCTS,
        help="Which training data percentages to include.",
    )
    parser.add_argument(
        "--state-file", type=str, default=_STATE_FILE_DEFAULT,
        help="JSON file for resume state.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip experiments already recorded as 'ok' in the state file.",
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
        "--ray-address", type=str, default=None, metavar="ADDRESS",
        help="Ray cluster address. Omit to start a local Ray instance.",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO",
    )
    args = parser.parse_args()

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    # ── Ensure arch JSONs and model configs exist in THIS repo copy ───────────
    _ensure_protozoan_arch_jsons()
    _ensure_protozoan_model_configs()

    t_start  = time.time()
    gpu_ids  = list(range(args.num_gpus))

    # ── Build experiment list ─────────────────────────────────────────────────
    experiments = build_experiments(
        splits=args.splits,
        pcts=args.pcts,
        inits=args.inits,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    state   = ExecutionState(args.state_file)
    skipped: list[str] = []
    if args.resume:
        before   = len(experiments)
        to_run   = [e for e in experiments if not state.is_completed(e["key"])]
        skipped  = [e["key"] for e in experiments if state.is_completed(e["key"])]
        experiments = to_run
        _log(f"Resume: {len(skipped)} skipped, {len(experiments)} remaining (of {before})")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        max_concurrent = args.max_concurrent_per_gpu * args.num_gpus
        print(f"\n{'=' * 70}")
        print("DRY RUN — protozoan SSL re-training queue")
        print(f"{'=' * 70}")
        print(f"  GPUs                   : {args.num_gpus}  (IDs: {gpu_ids})")
        print(f"  Slots per GPU          : {args.max_concurrent_per_gpu}")
        print(f"  Max concurrent total   : {max_concurrent}")
        print(f"  CPUs per experiment    : {args.cpus_per_experiment}")
        print(f"  Initializations        : {args.inits}")
        print(f"  Splits                 : {args.splits}")
        print(f"  Percentages            : {args.pcts}")
        print(f"  Experiments queued     : {len(experiments)}")
        print()
        for i, exp in enumerate(experiments):
            cmd_preview = (
                f"python src/main.py fit "
                f"--config {exp['data_config']} "
                f"--config {exp['model_config']}"
            )
            print(f"  {i + 1:>3}. {exp['key']}")
            print(f"       {cmd_preview}")
        print(f"{'=' * 70}\n")
        return

    if not experiments:
        _log("No experiments to run. Exiting.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    rows = run_queue(
        experiments=experiments,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        state=state,
        fail_fast=args.fail_fast,
        ray_address=args.ray_address,
    )

    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu)


if __name__ == "__main__":
    main()
