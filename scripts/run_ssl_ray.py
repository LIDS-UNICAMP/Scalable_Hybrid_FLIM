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

"""retry_ssl_missing.py — Re-train all missing LeJEPA SSL experiments.

Dynamically discovers every expected experiment that is either absent from
the W&B registry or registered but without a local checkpoint, then submits
all of them via the same GPU slot-based Ray scheduler used by
``scripts/retry_protozoan_experiment.py``.

Supports all three datasets (helminth-eggs, helminth-larvae, protozoan-cysts)
and all four initialisations (xavier, random, he, flim).

Config mapping per dataset / init:
    helminth-eggs   flim   → configs/model/lejepa_line_flim_eggs_train{N}.yaml
    helminth-eggs   other  → configs/model/lejepa_line_{init}.yaml
    helminth-larvae flim   → configs/model/lejepa_line_flim_larvae_train{N}.yaml
    helminth-larvae other  → configs/model/lejepa_line_{init}.yaml
    protozoan-cysts flim   → configs/model/lejepa_line_flim_protozoan_train{N}.yaml
    protozoan-cysts other  → configs/model/lejepa_line_{init}_protozoan_train{N}.yaml

Usage::

    # Discover missing and run them
    python scripts/retry_ssl_missing.py --num-gpus 2 --max-concurrent-per-gpu 12 --cpus-per-experiment 4

    # Preview queue without running
    python scripts/retry_ssl_missing.py --dry-run

    # Resume an interrupted run (skips experiments already recorded as ok)
    python scripts/retry_ssl_missing.py --resume

    # Restrict to specific datasets or inits
    python scripts/retry_ssl_missing.py --datasets helminth-eggs --inits flim he
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

# ─── Constantes compartilhadas ────────────────────────────────────────────────
# Rodando como ``python scripts/run_ssl_ray.py``, scripts/ é o sys.path[0] —
# constants.py importa direto. Este arquivo fala a chave LONGA de dataset
# ("helminth-eggs"); DATASET_LONG_TO_SHORT faz a ponte para a chave curta que
# constants.py usa nos dicionários de caminho.
from constants import (
    ARCH_JSON_FILENAME,
    CUDA_ENV_VAR,
    DATASET_LONG_TO_SHORT,
    DATASETS_LONG as _ALL_DATASETS,
    DEFAULT_CONFIG_YAML,
    DEFAULT_CPUS_PER_EXPERIMENT,
    FLIM_ARCH_BASE,
    LOG_LEVEL_DEFAULT,
    LOG_LEVELS as _LOG_LEVELS,
    LOG_TIME_FMT,
    OMP_ENV_VAR,
    PROJECT_ROOT as _ROOT,
    RAY_INIT_KWARGS,
    RESULTS_DIR,
    SEP_WIDTH,
    STDERR_TRUNCATE_HEAD,
    STDERR_TRUNCATE_MAX_CHARS,
    STDERR_TRUNCATE_TAIL,
    train_dir,
)

if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ─── Ray ──────────────────────────────────────────────────────────────────────
try:
    import ray
except ImportError as _e:
    raise ImportError("ray is not installed.  Run: pip install 'ray[tune]'") from _e

# ─── Constants ────────────────────────────────────────────────────────────────

# Local: a lista de inits deste launcher tem 5 entradas (as outras grades têm
# vocabulários diferentes de init — não unificar).
_ALL_INITS    = ["xavier", "random", "he", "flim", "trunc_normal"]

_STATE_FILE_DEFAULT = os.path.join(RESULTS_DIR, "ssl_missing_state.json")

# Arch JSON base paths per dataset (used to build absolute paths for flim
# override) — é FLIM_ARCH_BASE, só que rechaveado para o nome longo do dataset.
_ARCH_BASE: dict[str, str] = {
    long_name: FLIM_ARCH_BASE[short] for long_name, short in DATASET_LONG_TO_SHORT.items()
}


# ─── Config resolution ────────────────────────────────────────────────────────

def _model_config(dataset: str, init: str, split: int) -> str:
    alias = DATASET_LONG_TO_SHORT[dataset]
    if init == "flim":
        return f"configs/model/lejepa_line_flim_{alias}_train{split}.yaml"
    if dataset == "protozoan-cysts":
        # protozoan uses per-split configs for all non-flim inits
        return f"configs/model/lejepa_line_{init}_protozoan_train{split}.yaml"
    # helminth-eggs and helminth-larvae use the generic per-init config
    return f"configs/model/lejepa_line_{init}.yaml"


def _data_config(dataset: str, split: int, pct: int) -> str:
    return f"configs/data/percentage/{dataset}_split_{split}/{pct}/lejepa_line.yaml"


def _arch_json_abs(dataset: str, split: int) -> Optional[str]:
    """Return absolute arch_json path for flim override, or None if not needed."""
    base = _ARCH_BASE.get(dataset)
    if base is None:
        return None
    return os.path.join(base, train_dir(split), ARCH_JSON_FILENAME)


# ─── Hyperparameter introspection ─────────────────────────────────────────────

def _read_init_args(rel_path: str) -> dict:
    """Best-effort read of a config YAML's ``init_args`` (returns {} on failure).

    Lazily imports PyYAML (available in the scalable_FLIM env); if unavailable,
    falls back to a minimal line parser that handles the flat ``key: value``
    init_args blocks these configs use.
    """
    abs_path = os.path.join(_ROOT, rel_path)
    try:
        import yaml  # noqa: PLC0415
        with open(abs_path) as f:
            cfg = yaml.safe_load(f) or {}
        for top in ("data", "model"):
            section = cfg.get(top)
            if isinstance(section, dict):
                return section.get("init_args", {}) or {}
        return {}
    except ModuleNotFoundError:
        pass
    except Exception:
        return {}

    # ── Fallback: regex line parser (no PyYAML) ──
    import re  # noqa: PLC0415
    out: dict = {}
    try:
        with open(abs_path) as f:
            in_init = False
            for line in f:
                if re.match(r"\s*init_args:\s*$", line):
                    in_init = True
                    continue
                if in_init:
                    m = re.match(r"\s{6,}([A-Za-z_]\w*):\s*([^#\n]+)", line)
                    if not m:
                        continue
                    key, raw = m.group(1), m.group(2).strip()
                    try:
                        out[key] = json.loads(raw)
                    except Exception:
                        out[key] = raw
    except Exception:
        return {}
    return out


def _experiment_hparams(exp: dict) -> dict:
    """Resolve effective hyperparameters (YAML values + CLI overrides)."""
    data_args  = _read_init_args(exp["data_config"])
    model_args = _read_init_args(exp["model_config"])

    batch_size  = exp.get("batch_size") or data_args.get("batch_size", "-")
    num_workers = exp.get("num_workers")
    if num_workers is None:
        num_workers = data_args.get("num_workers", "-")

    # CLI --multicrop overrides win over YAML for display.
    if exp.get("multicrop"):
        n_global = exp.get("n_global", "-")
        n_local  = exp.get("n_local", "-")
    else:
        n_global = data_args.get("n_global", "-")
        n_local  = data_args.get("n_local", "-")

    return {
        "n_global":     n_global,
        "n_local":      n_local,
        "V_train":      data_args.get("V_train", "-"),
        "V_eval":       data_args.get("V_eval", "-"),
        "batch_size":   batch_size,
        "image_size":   data_args.get("image_size", "-"),
        "lr":           model_args.get("lr", "-"),
        "weight_decay": model_args.get("weight_decay", "-"),
        "encoder_init": model_args.get("encoder_init", "-"),
        "num_workers":  num_workers,
    }


def _log_experiment_hparams(exp: dict) -> None:
    """Log resolved hyperparameters for one experiment."""
    h = _experiment_hparams(exp)
    _log(
        "        hparams: "
        f"n_global={h['n_global']} n_local={h['n_local']} "
        f"V_train={h['V_train']} V_eval={h['V_eval']} "
        f"batch_size={h['batch_size']} image_size={h['image_size']} "
        f"lr={h['lr']} weight_decay={h['weight_decay']} "
        f"init={h['encoder_init']} num_workers={h['num_workers']}"
    )


# ─── Missing experiment discovery ────────────────────────────────────────────

def find_missing_experiments(
    datasets: list[str],
    inits:    list[str],
    pcts:     Optional[list[int]] = None,
    splits:   Optional[list[int]] = None,
    force:    bool = False,
) -> list[dict]:
    """Return experiment dicts for all SSL runs that are missing locally."""
    from check_experiments._common import (  # noqa: PLC0415
        all_expected, has_ssl_checkpoint, load_ids_wandb,
    )

    name_to_id   = load_ids_wandb()
    all_exps     = all_expected()
    missing: list[dict] = []

    for exp in all_exps:
        if exp.dataset not in datasets:
            continue
        if exp.init not in inits:
            continue
        if pcts is not None and exp.pct not in pcts:
            continue
        if splits is not None and exp.split not in splits:
            continue

        if not force:
            run_id = name_to_id.get(exp.canonical_name)
            if run_id is not None and has_ssl_checkpoint(run_id):
                continue  # already present — skip

        model_cfg = _model_config(exp.dataset, exp.init, exp.split)
        data_cfg  = _data_config(exp.dataset, exp.split, exp.pct)
        arch_abs  = _arch_json_abs(exp.dataset, exp.split) if exp.init == "flim" else None

        # Validate configs exist before queueing
        if not os.path.isfile(os.path.join(_ROOT, model_cfg)):
            print(
                f"[WARN] Model config not found, skipping: {model_cfg}  "
                f"({exp.canonical_name})", flush=True,
            )
            continue
        if not os.path.isfile(os.path.join(_ROOT, data_cfg)):
            print(
                f"[WARN] Data config not found, skipping: {data_cfg}  "
                f"({exp.canonical_name})", flush=True,
            )
            continue
        if arch_abs is not None and not os.path.isfile(arch_abs):
            print(
                f"[WARN] Arch JSON not found, skipping: "
                f"{os.path.relpath(arch_abs, _ROOT)}  ({exp.canonical_name})", flush=True,
            )
            continue

        missing.append({
            "dataset":        exp.dataset,
            "split":          exp.split,
            "pct":            exp.pct,
            "init":           exp.init,
            "canonical_name": exp.canonical_name,
            "model_config":   model_cfg,
            "data_config":    data_cfg,
            "arch_json_abs":  arch_abs,
            "key":            (
                f"{exp.dataset}_split{exp.split}_pct{exp.pct}_{exp.init}"
            ),
        })

    return missing


# ─── Logging ──────────────────────────────────────────────────────────────────

_current_log_level = _LOG_LEVELS[LOG_LEVEL_DEFAULT]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime(LOG_TIME_FMT)
        print(f"[{ts}][{level}] {msg}", flush=True)


# ─── Durable state (resume) ───────────────────────────────────────────────────

class ExecutionState:
    """Persists per-experiment outcomes so a re-run can skip completed ones."""

    def __init__(self, state_file: str) -> None:
        self.state_file = state_file
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self._state = json.load(f)
            _log(f"Resume state loaded ({len(self._state)} record(s))")

    def is_completed(self, key: str) -> bool:
        return self._state.get(key, {}).get("status") == "ok"

    def mark(self, key: str, result: dict) -> None:
        self._state[key] = {
            "status":     result.get("status", "unknown"),
            "timestamp":  time.time(),
            "returncode": result.get("returncode"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file)), exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    def completed_keys(self) -> list[str]:
        return [k for k, v in self._state.items() if v.get("status") == "ok"]


# ─── GPU slot scheduler ───────────────────────────────────────────────────────

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


# ─── Ray remote training task ─────────────────────────────────────────────────

@ray.remote
def run_ssl_experiment(
    exp:          dict,
    gpu_id:       int,
    project_root: str,
    queue_index:  int = -1,
    total:        int = -1,
    cpus:         int = DEFAULT_CPUS_PER_EXPERIMENT,
) -> dict:
    """Run one SSL training experiment as a subprocess on the assigned GPU."""
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ[CUDA_ENV_VAR] = str(gpu_id)
    _os.environ[OMP_ENV_VAR]  = str(cpus)

    cmd = [
        _sys.executable, "src/main.py", "fit",
        "--config", DEFAULT_CONFIG_YAML,
        "--config", exp["data_config"],
        "--config", exp["model_config"],
        f"--trainer.logger.init_args.name={exp['canonical_name']}",
        "--trainer.accelerator=gpu",
        "--trainer.devices=1",
        "--trainer.precision=16-mixed",
    ]

    # Override arch_json with absolute path for flim experiments to avoid
    # CWD-relative path issues inside Ray workers (same technique as
    # retry_protozoan_experiment.py).
    if exp.get("arch_json_abs"):
        cmd.append(f"--model.init_args.arch_json={exp['arch_json_abs']}")

    # Collect WandB tags so multiple variants (bs / 3lproj) can coexist —
    # they are emitted as a single tags arg below (LightningCLI keeps only the
    # last repeated list arg, so we must combine them).
    wandb_tags: list[str] = []

    if exp.get("batch_size"):
        cmd.append(f"--data.init_args.batch_size={exp['batch_size']}")
        # Append a bs<N> tag so the run is filterable in the WandB UI
        wandb_tags.append(f"bs{exp['batch_size']}")

    if exp.get("num_workers") is not None:
        cmd.append(f"--data.init_args.num_workers={exp['num_workers']}")

    # Multi-crop (global/local) overrides. Only emitted when --multicrop is set,
    # so default runs are byte-for-byte identical to before.
    if exp.get("multicrop"):
        cmd.append("--data.init_args.multicrop=true")
        cmd.append(f"--data.init_args.n_global={exp['n_global']}")
        cmd.append(f"--data.init_args.n_local={exp['n_local']}")
        if exp.get("global_size") is not None:
            cmd.append(f"--data.init_args.global_size={exp['global_size']}")
        if exp.get("local_size") is not None:
            cmd.append(f"--data.init_args.local_size={exp['local_size']}")

    # 3-Layer Projector variant. Only emitted when --three_layer_projector is
    # set, so default runs are byte-for-byte identical to before. The 3lproj
    # tag makes the run filterable in the WandB UI.
    if exp.get("three_layer_projector"):
        cmd.append("--model.init_args.three_layer_projector=true")
        wandb_tags.append("3lproj")

    if wandb_tags:
        _tags_str = ",".join(f"\"{t}\"" for t in wandb_tags)
        cmd.append(f"--trainer.logger.init_args.tags=[{_tags_str}]")

    # Pre-flight: verify arch JSON exists before launching subprocess
    if exp.get("arch_json_abs") and not _os.path.exists(exp["arch_json_abs"]):
        return {
            "key": exp["key"], "gpu_id": gpu_id, "queue_index": queue_index,
            "status": "error", "returncode": -1,
            "error": f"PREFLIGHT FAIL — arch JSON not found: {exp['arch_json_abs']}",
        }

    result: dict = {"key": exp["key"], "gpu_id": gpu_id, "queue_index": queue_index}
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
            if len(stderr) > STDERR_TRUNCATE_MAX_CHARS:
                stderr = (stderr[:STDERR_TRUNCATE_HEAD] + "\n...[truncated]...\n"
                          + stderr[-STDERR_TRUNCATE_TAIL:])
            result["status"] = "error"
            result["error"]  = f"returncode={proc.returncode}\n{stderr}"
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
    if not experiments:
        _log("No experiments to run.", "WARN")
        return []

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # num_gpus=0 vem de RAY_INIT_KWARGS: a GPU é fixada na mão via CUDA_VISIBLE_DEVICES.
    ray_kwargs: dict[str, Any] = dict(RAY_INIT_KWARGS)
    if ray_address:
        # Ray recusa num_cpus/num_gpus ao conectar num cluster já existente.
        ray_kwargs.pop("num_gpus")
        ray_kwargs["address"] = ray_address
    else:
        total_cpus = cpus_per_experiment * max_concurrent_per_gpu * len(gpu_ids)
        ray_kwargs["num_cpus"] = total_cpus

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
    _log("=" * SEP_WIDTH)

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
            exp, gpu_id=gpu_id, project_root=_ROOT,
            queue_index=idx, total=total, cpus=cpus_per_experiment,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, exp, idx)
        _log(
            f"[{idx + 1:>3}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  pending={len(pending)}  {exp['key']}"
        )
        _log_experiment_hparams(exp)
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
            }

        rows.append(result)
        state.mark(result["key"], result)

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


# ─── Summary ──────────────────────────────────────────────────────────────────

def _print_summary(
    rows: list[dict], skipped: list[str], t_start: float,
    gpu_ids: list[int], max_per_gpu: int,
) -> None:
    elapsed = time.time() - t_start
    n_ok  = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok

    gpu_stats: dict[int, dict] = {gid: {"ok": 0, "error": 0} for gid in gpu_ids}
    for r in rows:
        gid = r.get("gpu_id")
        if gid in gpu_stats:
            gpu_stats[gid]["ok" if r.get("status") == "ok" else "error"] += 1

    print(f"\n{'=' * SEP_WIDTH}")
    print("QUEUE SUMMARY — missing SSL re-training")
    print(f"{'=' * SEP_WIDTH}")
    print(f"  Succeeded      : {n_ok}")
    print(f"  Failed         : {n_err}")
    print(f"  Skipped (done) : {len(skipped)}")
    print(f"  Total runtime  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"\n  GPU profile    : {len(gpu_ids)} GPU(s) × {max_per_gpu} slots/GPU")
    print("  Per-GPU results:")
    for gid in sorted(gpu_stats):
        s = gpu_stats[gid]
        print(f"    GPU {gid}: {s['ok']} ok, {s['error']} error")
    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for name in skipped[:10]:
            print(f"    - {name}")
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('key', '?')}: {r.get('error', '')[:200]}")
    print(f"{'=' * SEP_WIDTH}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-train all missing LeJEPA SSL experiments via GPU slot-based Ray scheduler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-gpus", type=int, default=2, metavar="N",
        help="Number of GPUs available (CUDA device IDs 0..N-1). "
             "Ignored when --gpu-ids is given.",
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", type=int, default=None, metavar="ID",
        help="Explicit physical GPU IDs to use, e.g. --gpu-ids 2 3. "
             "Each experiment runs with CUDA_VISIBLE_DEVICES set to one of "
             "these IDs. Overrides --num-gpus.",
    )
    parser.add_argument(
        "--max-concurrent-per-gpu", type=int, default=10, metavar="N",
        help="Maximum experiments running simultaneously on each GPU.",
    )
    parser.add_argument(
        "--cpus-per-experiment", type=int, default=DEFAULT_CPUS_PER_EXPERIMENT, metavar="N",
        help="CPU cores (OMP_NUM_THREADS) per experiment.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=_ALL_DATASETS, default=_ALL_DATASETS,
        help="Restrict to specific datasets.",
    )
    parser.add_argument(
        "--inits", nargs="+", choices=_ALL_INITS, default=_ALL_INITS,
        help="Restrict to specific initialisations.",
    )
    parser.add_argument(
        "--pcts", nargs="+", type=int, default=None, metavar="PCT",
        help="Restrict to specific percentages, e.g. --pcts 100. Default: all.",
    )
    parser.add_argument(
        "--splits", nargs="+", type=int, default=None, metavar="SPLIT",
        help="Restrict to specific splits/folds, e.g. --splits 1. Default: all.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="Override batch_size in the data config (e.g. 256). Default: use YAML value.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, metavar="N",
        help="Override DataLoader num_workers in the data config (e.g. 16). "
             "Default: use YAML value (8). Keep <= --cpus-per-experiment "
             "headroom; this machine has 96 cores.",
    )
    parser.add_argument(
        "--multicrop", action="store_true",
        help="Enable global/local multi-crop views in training (n_global + "
             "n_local crops of different scales, same output size). Default off "
             "→ uses the YAML V_train identical-view scheme.",
    )
    parser.add_argument(
        "--n-global", type=int, default=2, metavar="N",
        help="Number of global crops (only with --multicrop).",
    )
    parser.add_argument(
        "--n-local", type=int, default=6, metavar="N",
        help="Number of local crops (only with --multicrop).",
    )
    parser.add_argument(
        "--global-size", type=int, default=None, metavar="N",
        help="Output spatial size for all crops (only with --multicrop). "
             "Default: data config image_size.",
    )
    parser.add_argument(
        "--local-size", type=int, default=None, metavar="N",
        help="Local crop param passed through (only with --multicrop).",
    )
    parser.add_argument(
        "--three_layer_projector", action="store_true",
        help="Use the 3-Layer Projector variant (Linear→BN→ReLU ×2 + Linear, "
             "all dim 48) in place of the default ProjectionHead. Changes the "
             "model → distinct WandB run identity (suffix _3lproj).",
    )
    parser.add_argument(
        "--ignore-existing", action="store_true",
        help="Queue experiments even if a checkpoint already exists locally. "
             "Use when running the same setup with different hyperparameters "
             "(e.g. --batch-size) — produces a new WandB run, does not "
             "overwrite the original.",
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
        "--log-level", choices=list(_LOG_LEVELS), default=LOG_LEVEL_DEFAULT,
    )
    args = parser.parse_args()

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    t_start = time.time()
    gpu_ids = args.gpu_ids if args.gpu_ids else list(range(args.num_gpus))

    # ── Discover missing experiments ──────────────────────────────────────────
    _log("Scanning for missing SSL experiments...")
    experiments = find_missing_experiments(
        datasets=args.datasets,
        inits=args.inits,
        pcts=args.pcts,
        splits=args.splits,
        force=args.ignore_existing,
    )
    if args.batch_size is not None:
        for exp in experiments:
            exp["batch_size"] = args.batch_size
            # Suffix the WandB run name so it is distinct from the default run.
            # e.g. lejepa_line_helminth-eggs_split_1_pct_100_model_trunc_normal_bs256
            exp["canonical_name"] = f"{exp['canonical_name']}_bs{args.batch_size}"
    if args.num_workers is not None:
        # DataLoader workers don't change run identity, so no name suffix.
        for exp in experiments:
            exp["num_workers"] = args.num_workers
    if args.multicrop:
        for exp in experiments:
            exp["multicrop"]    = True
            exp["n_global"]     = args.n_global
            exp["n_local"]      = args.n_local
            exp["global_size"]  = args.global_size
            exp["local_size"]   = args.local_size
            # Multi-crop changes view sampling → distinct WandB run identity.
            exp["canonical_name"] = (
                f"{exp['canonical_name']}_mc{args.n_global}g{args.n_local}l"
            )
    if args.three_layer_projector:
        for exp in experiments:
            exp["three_layer_projector"] = True
            # Changing the projector changes the model → distinct WandB run
            # identity. The _3lproj marker makes the run identifiable.
            exp["canonical_name"] = f"{exp['canonical_name']}_3lproj"
    _log(f"Found {len(experiments)} missing experiment(s).")

    if not experiments:
        _log("Nothing to do — all SSL experiments are present.")
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    state   = ExecutionState(args.state_file)
    skipped: list[str] = []
    if args.resume:
        before      = len(experiments)
        to_run      = [e for e in experiments if not state.is_completed(e["key"])]
        skipped     = [e["key"] for e in experiments if state.is_completed(e["key"])]
        experiments = to_run
        _log(f"Resume: {len(skipped)} skipped, {len(experiments)} remaining (of {before})")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        max_concurrent = args.max_concurrent_per_gpu * len(gpu_ids)
        print(f"\n{'=' * SEP_WIDTH}")
        print("DRY RUN — missing SSL re-training queue")
        print(f"{'=' * SEP_WIDTH}")
        print(f"  GPUs                   : {len(gpu_ids)}  (IDs: {gpu_ids})")
        print(f"  Slots per GPU          : {args.max_concurrent_per_gpu}")
        print(f"  Max concurrent total   : {max_concurrent}")
        print(f"  CPUs per experiment    : {args.cpus_per_experiment}")
        print(f"  Datasets               : {args.datasets}")
        print(f"  Initialisations        : {args.inits}")
        print(f"  Experiments queued     : {len(experiments)}")
        if skipped:
            print(f"  Skipped (state file)   : {len(skipped)}")
        print()
        for i, exp in enumerate(experiments):
            arch_note = f"  arch_json override: {os.path.relpath(exp['arch_json_abs'], _ROOT)}" if exp.get("arch_json_abs") else ""
            h = _experiment_hparams(exp)
            print(f"  {i + 1:>3}. {exp['key']}")
            print(f"       run   : {exp['canonical_name']}")
            print(f"       model : {exp['model_config']}")
            print(f"       data  : {exp['data_config']}")
            print(
                f"       hparams: n_global={h['n_global']} n_local={h['n_local']} "
                f"V_train={h['V_train']} V_eval={h['V_eval']} "
                f"batch_size={h['batch_size']} image_size={h['image_size']} "
                f"lr={h['lr']} weight_decay={h['weight_decay']} "
                f"init={h['encoder_init']} num_workers={h['num_workers']}"
            )
            if arch_note:
                print(f"       {arch_note}")
        print(f"{'=' * SEP_WIDTH}\n")
        return

    if not experiments:
        _log("No experiments to run after applying resume filter. Exiting.")
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
