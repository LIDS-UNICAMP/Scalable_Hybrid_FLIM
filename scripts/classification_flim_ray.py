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
"""classification_flim_ray.py — Ray launcher for Experiment 3: FLIM-init encoder +
2-layer Sigmoid classification head, direct supervised training (no SSL, no distillation).

Head: AdaptiveAvgPool2d(1) -> flatten -> Linear(48,24) -> Sigmoid -> Linear(24,n_classes) -> Softmax.
Trained module: src.modules.classification_flim_module.ClassificationFlimModule.

Mirrors scripts/distillation_conv_ray.py's grid/scheduler/manifest machinery, but
stripped of the distillation-specific axes (proj_type, distillation_type, teacher) —
this experiment has exactly one architecture per dataset, always FLIM-initialised.

Example runs::

    # Full grid (all datasets/splits/percentages)
    python scripts/classification_flim_ray.py \\
        --num-gpus 1 --max-concurrent-per-gpu 1 --wandb-update

    # Exp.3 grid used by the FLIM distillation experiment series (1% and 75% only)
    python scripts/classification_flim_ray.py \\
        --datasets eggs larvae protozoan --splits 1 2 3 --percentages 1 75 \\
        --run-prefix "sigmoid2l_v1_" \\
        --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --wandb-update --skip-existing --check-wandb

    # Variante Softplus da cabeça (...->Sigmoid->Linear(24,C)->Softplus->Softmax);
    # única diferença em relação ao grupo relu2l. Runs ganham o sufixo '_softplus2l'.
    python scripts/classification_flim_ray.py \\
        --datasets eggs larvae protozoan --splits 1 2 3 --percentages 1 75 \\
        --output-softplus \\
        --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --wandb-update --skip-existing --check-wandb

    # Dry-run
    python scripts/classification_flim_ray.py --dry-run

    # Run both freeze/unfreeze variants (two separate invocations — run_name gets
    # a '_frozen' suffix so the two grids never collide on artifacts/W&B/manifest):
    python scripts/classification_flim_ray.py --datasets eggs larvae protozoan \\
        --splits 1 2 3 --percentages 1 75 --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --wandb-update --skip-existing --check-wandb
    python scripts/classification_flim_ray.py --datasets eggs larvae protozoan \\
        --splits 1 2 3 --percentages 1 75 --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --freeze-encoder --wandb-update --skip-existing --check-wandb

--skip-existing / --check-wandb / --retry follow the exact same semantics as in
distillation_conv_ray.py — see that file's module docstring for details.
"""
from __future__ import annotations

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

_ALL_DATASETS = ["eggs", "larvae", "protozoan"]
_ALL_SPLITS   = [1, 2, 3]
_ALL_PCTS     = [1, 5, 25, 50, 75, 100]

_NUM_CLASSES: dict[str, int] = {
    "eggs":      9,
    "larvae":    2,
    "protozoan": 7,
}

_DEFAULT_DATA_ROOT = os.path.join(_ROOT, "data", "to_modules", "new_split_parasito")

# Always FLIM-init: arch + weights both come from the ch24_32_48 directory (the
# one with a models/ subdirectory), matching the convention already used by
# distillation_conv_ray.py's _ARCH_BASE_FLIM. For protozoan, ClassificationFlimModule
# ignores --arch-json and rebuilds the architecture from the hardcoded
# PROTOZOAN_FLIM_ARCH dict (ch24_30_48 has no trained FLIM weights on disk).
_ARCH_BASE_FLIM: dict[str, str] = {
    "eggs":      os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "eggs"),
    "larvae":    os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "larvae"),
    "protozoan": os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "protozoan"),
}

_PARASITE_DIR: dict[str, str] = {
    "eggs":      "helminth-eggs",
    "larvae":    "helminth-larvae",
    "protozoan": "protozoan-cysts",
}

_MANIFEST_PATH       = os.path.join(_ROOT, "artifacts", "classification_flim", "run_manifest.csv")
_MANIFEST_RETRY_PATH = os.path.join(_ROOT, "artifacts", "classification_flim", "run_manifest_retry.csv")
_MANIFEST_FIELDS = [
    "run_name", "dataset", "split", "percentage",
    "status", "skip_reason", "checkpoint_path",
]

# ── Logging ────────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)


# ── W&B / local state check (identical semantics to distillation_conv_ray.py) ──

_WANDB_FAILED_STATES = {"crashed", "failed", "killed"}


def _query_wandb_state(run_name: str, entity: str, project: str) -> tuple[str, str]:
    try:
        import wandb as _wandb  # noqa: PLC0415
        api = _wandb.Api(timeout=15)
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name}, order="-created_at")
        for run in runs:
            if run.state in _WANDB_FAILED_STATES:
                return "failed", run.id
            if run.state == "finished":
                return "finished", run.id
            if run.state == "running":
                return "running", run.id
            return "failed", run.id
        return "not_found", ""
    except Exception as exc:
        _log(f"W&B API error for '{run_name}': {exc} — treating as unknown (will re-queue).", "WARN")
        return "error", ""


def _should_skip_experiment(run_name: str, check_wandb: bool, wandb_entity: str, wandb_project: str) -> tuple[bool, str]:
    if check_wandb:
        wb_state, wb_id = _query_wandb_state(run_name, wandb_entity, wandb_project)
        if wb_state == "running":
            return True, f"wandb:running({wb_id})"
        if wb_state == "finished":
            return True, f"wandb:finished({wb_id})"
        if wb_state == "failed":
            local_meta = os.path.join(_ROOT, "artifacts", "classification_flim", run_name, "run_metadata.json")
            if os.path.isfile(local_meta):
                try:
                    with open(local_meta, encoding="utf-8") as fh:
                        _m = json.load(fh)
                    if _m.get("status") == "ok":
                        return True, f"wandb:crashed+local:ok({wb_id})"
                except Exception:
                    pass
            return False, f"wandb:failed({wb_id})"
        if wb_state == "error":
            return False, "wandb:api_error"
        # not_found → fall through to local check

    meta_path = os.path.join(_ROOT, "artifacts", "classification_flim", run_name, "run_metadata.json")
    if not os.path.isfile(meta_path):
        return False, "no_metadata"
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        _log(f"Could not read metadata for {run_name}: {exc}", "WARN")
        return False, "metadata_read_error"
    if meta.get("status") == "ok":
        return True, "local:ok"
    return False, f"local:{meta.get('status', 'unknown')}"


# ── Path resolution ────────────────────────────────────────────────────────────

def _arch_json(dataset: str, split: int) -> str:
    return os.path.join(_ARCH_BASE_FLIM[dataset], f"train{split}", "architecture.json")


def _flim_weights_path(dataset: str, split: int) -> str:
    return os.path.join(_ARCH_BASE_FLIM[dataset], f"train{split}", "models")


def _split_json(dataset: str, split: int, pct: int) -> str:
    parasite_dir = _PARASITE_DIR[dataset]
    return os.path.join(
        _DEFAULT_DATA_ROOT, parasite_dir, "splits_incremental",
        f"split{split}", f"data_descriptor_perc{pct}.json",
    )


def _run_name(dataset: str, split: int, pct: int, run_prefix: str = "", freeze_encoder: bool = False,
              output_relu: bool = False, output_softplus: bool = False) -> str:
    if output_softplus:
        head_tag = "softplus2l"
    elif output_relu:
        head_tag = "relu2l"
    else:
        head_tag = "sigmoid2l"
    name = f"classhead_{dataset}_split{split}_pct{pct}_{head_tag}"
    if freeze_encoder:
        name += "_frozen"
    return f"{run_prefix}{name}" if run_prefix else name


# ── Pre-flight validation ──────────────────────────────────────────────────────

def validate_experiment(dataset: str, split: int, pct: int) -> tuple[bool, str]:
    arch = _arch_json(dataset, split)
    if not os.path.isfile(arch):
        return False, f"arch JSON not found: {os.path.relpath(arch, _ROOT)}"
    weights = _flim_weights_path(dataset, split)
    if not os.path.isdir(weights):
        return False, f"FLIM weights dir not found: {os.path.relpath(weights, _ROOT)}"
    sjson = _split_json(dataset, split, pct)
    if not os.path.isfile(sjson):
        return False, f"split JSON not found: {os.path.relpath(sjson, _ROOT)}"
    return True, ""


def validate_output_dir() -> tuple[bool, str]:
    out = os.path.join(_ROOT, "artifacts", "classification_flim")
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
    datasets: list[str],
    splits: list[int],
    percentages: list[int],
    wandb_update: bool,
    skip_existing: bool = False,
    check_wandb: bool = False,
    wandb_entity: str = "ophira-ai",
    wandb_project: str = "flim-ssl",
    num_workers: int = 4,
    no_imagenet_norm: bool = False,
    run_prefix: str = "",
    freeze_encoder: bool = False,
    output_relu: bool = False,
    output_softplus: bool = False,
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    skipped: list[dict] = []

    for dataset in datasets:
        for split in splits:
            for pct in percentages:
                run_name = _run_name(dataset, split, pct, run_prefix, freeze_encoder, output_relu, output_softplus)
                ok, reason = validate_experiment(dataset, split, pct)

                base = {
                    "run_name":          run_name,
                    "dataset":           dataset,
                    "split":             split,
                    "pct":               pct,
                    "arch_json":         _arch_json(dataset, split),
                    "flim_weights_path": _flim_weights_path(dataset, split),
                    "wandb_update":      wandb_update,
                    "num_classes":       _NUM_CLASSES[dataset],
                    "num_workers":       num_workers,
                    "no_imagenet_norm":  no_imagenet_norm,
                    "freeze_encoder":    freeze_encoder,
                    "output_relu":       output_relu,
                    "output_softplus":   output_softplus,
                    "key":               run_name,
                }

                if not ok:
                    _log(f"[SKIP] {run_name}  reason: {reason}", "WARN")
                    skipped.append({**base, "status": "skipped", "skip_reason": reason, "checkpoint_path": ""})
                    continue

                if skip_existing:
                    skip, reason = _should_skip_experiment(run_name, check_wandb, wandb_entity, wandb_project)
                    if skip:
                        _log(f"[SKIP] {run_name}  ({reason})")
                        skipped.append({**base, "status": "skipped_existing", "skip_reason": reason, "checkpoint_path": ""})
                        continue
                    _log(f"[QUEUE] {run_name}  ({reason})")

                valid.append(base)

    return valid, skipped


# ── GPU slot scheduler (identical to distillation_conv_ray.py) ─────────────────

class GpuSlotScheduler:
    """Concurrency slots per GPU, optionally capped by a per-GPU experiment quota.

    ``max_per_gpu`` limits how many experiments run *at the same time* on a GPU.
    ``quota`` (``--gpu-experiments``) limits how many experiments that GPU receives
    *in total* over the whole queue; a GPU whose quota is exhausted is never picked
    again. Without ``quota`` the queue is drained greedily (whoever frees up first
    takes the next experiment).
    """

    def __init__(self, gpu_ids: list[int], max_per_gpu: "int | dict[int, int]",
                 quota: Optional[dict[int, int]] = None) -> None:
        if isinstance(max_per_gpu, int):
            self._limits: dict[int, int] = {gid: max_per_gpu for gid in gpu_ids}
        else:
            self._limits = dict(max_per_gpu)
        self._running: dict[int, int] = {gid: 0 for gid in gpu_ids}
        self._quota: Optional[dict[int, int]] = dict(quota) if quota else None
        self._assigned: dict[int, int] = {gid: 0 for gid in gpu_ids}

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._running)

    def _available(self, gid: int) -> bool:
        if self._running[gid] >= self._limits[gid]:
            return False
        if self._quota is not None and self._assigned[gid] >= self._quota.get(gid, 0):
            return False
        return True

    def pick_gpu(self) -> Optional[int]:
        candidates = [(self._running[gid], gid) for gid in self._running if self._available(gid)]
        return min(candidates)[1] if candidates else None

    def acquire(self, gpu_id: int) -> None:
        self._running[gpu_id] += 1
        self._assigned[gpu_id] += 1

    def release(self, gpu_id: int) -> None:
        self._running[gpu_id] = max(0, self._running[gpu_id] - 1)

    def has_free_slot(self) -> bool:
        return any(self._available(gid) for gid in self._running)

    def status_line(self) -> str:
        if self._quota is not None:
            return " | ".join(
                f"GPU {gid}: {self._running[gid]}/{self._limits[gid]} "
                f"({self._assigned[gid]}/{self._quota.get(gid, 0)})"
                for gid in self.gpu_ids
            )
        return " | ".join(f"GPU {gid}: {self._running[gid]}/{self._limits[gid]}" for gid in self.gpu_ids)


# ── Ray remote task ────────────────────────────────────────────────────────────

@ray.remote
def run_classification_experiment(
    exp: dict, gpu_id: int, project_root: str,
    queue_index: int = -1, total: int = -1, cpus: int = 4,
) -> dict:
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"]       = str(cpus)
    _os.environ["WANDB_CONSOLE"] = "off"
    _os.environ["WANDB_MODE"]    = "online"

    cmd = [
        _sys.executable, "-m", "src.modules.classification_flim_module",
        "--dataset",            exp["dataset"],
        "--split",              str(exp["split"]),
        "--percentage",         str(exp["pct"]),
        "--arch-json",          exp["arch_json"],
        "--flim-weights-path",  exp["flim_weights_path"],
        "--run-name",           exp["run_name"],
        "--num-classes",        str(exp["num_classes"]),
        "--num-workers",        str(exp.get("num_workers", 4)),
    ]
    if exp.get("no_imagenet_norm"):
        cmd.append("--no-imagenet-norm")
    if exp.get("freeze_encoder"):
        cmd.append("--freeze-encoder")
    if exp.get("output_relu"):
        cmd.append("--output-relu")
    if exp.get("output_softplus"):
        cmd.append("--output-softplus")
    if exp.get("wandb_update"):
        cmd.append("--wandb")

    result: dict = {
        "key": exp["key"], "run_name": exp["run_name"], "gpu_id": gpu_id,
        "queue_index": queue_index, "dataset": exp["dataset"],
        "split": exp["split"], "pct": exp["pct"],
    }

    if not _os.path.isfile(exp["arch_json"]):
        result["status"] = "error"
        result["returncode"] = -1
        result["error"] = f"PREFLIGHT FAIL — arch JSON not found: {exp['arch_json']}"
        return result

    try:
        proc = _sp.run(cmd, cwd=project_root, stdout=None, stderr=_sp.PIPE, text=True)
        result["returncode"] = proc.returncode
        if proc.returncode == 0:
            result["status"] = "ok"
        else:
            stderr = (proc.stderr or "").strip()
            if len(stderr) > 2000:
                stderr = stderr[:1000] + "\n...[truncated]...\n" + stderr[-1000:]
            result["status"] = "error"
            result["error"] = f"returncode={proc.returncode}\n{stderr}"
    except Exception as exc:
        result["status"] = "error"
        result["returncode"] = -1
        result["error"] = str(exc)

    return result


# ── Queue orchestrator ─────────────────────────────────────────────────────────

def run_queue(
    experiments: list[dict], gpu_ids: list[int], max_concurrent_per_gpu: int,
    cpus_per_experiment: int, fail_fast: bool, ray_address: Optional[str],
    slots_per_gpu: Optional[dict[int, int]] = None,
    quota_per_gpu: Optional[dict[int, int]] = None,
) -> list[dict]:
    if not experiments:
        _log("No experiments to run.", "WARN")
        return []

    if slots_per_gpu:
        scheduler_arg: "int | dict[int, int]" = slots_per_gpu
        total_slots = sum(slots_per_gpu.values())
    else:
        scheduler_arg = max_concurrent_per_gpu
        total_slots = max_concurrent_per_gpu * len(gpu_ids)
    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "log_to_driver": True}
    if ray_address:
        ray_kwargs["address"] = ray_address
    else:
        ray_kwargs["num_cpus"] = cpus_per_experiment * total_slots
        ray_kwargs["num_gpus"] = 0

    if not ray.is_initialized():
        ray.init(**ray_kwargs)
    _log("Ray initialised.")

    scheduler = GpuSlotScheduler(gpu_ids=gpu_ids, max_per_gpu=scheduler_arg, quota=quota_per_gpu)
    total = len(experiments)
    pending = list(experiments)
    futures: dict[Any, tuple[int, dict, int]] = {}
    rows: list[dict] = []
    global_index = 0
    completed = 0
    had_failure = False

    _log(f"Queue: {total} experiment(s) | {len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots = {total_slots} max concurrent")
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
        fut = run_classification_experiment.options(num_cpus=cpus_per_experiment).remote(
            exp, gpu_id=gpu_id, project_root=_ROOT, queue_index=idx, total=total, cpus=cpus_per_experiment,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, exp, idx)
        _log(f"[{idx + 1:>3}/{total}] SUBMIT  gpu={gpu_id}  [{scheduler.status_line()}]  pending={len(pending)}  {exp['key']}")
        return True

    while pending and scheduler.has_free_slot():
        _submit_next()

    while futures:
        done_refs, _ = ray.wait(list(futures), num_returns=1, timeout=None)
        ref = done_refs[0]
        gpu_id, exp, idx = futures.pop(ref)
        scheduler.release(gpu_id)
        completed += 1

        try:
            result = ray.get(ref)
        except Exception as exc:
            result = {"key": exp["key"], "gpu_id": gpu_id, "queue_index": idx,
                      "status": "ray_error", "error": str(exc), "run_name": exp["run_name"]}

        rows.append(result)

        if result.get("status") == "ok":
            _log(f"[{completed:>3}/{total}] OK     gpu={gpu_id}  [{scheduler.status_line()}]  {result['key']}")
        else:
            _log(f"[{completed:>3}/{total}] ERROR  gpu={gpu_id}  [{scheduler.status_line()}]  {result['key']}: {result.get('error', '')}", "WARN")
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] Dropping remaining pending experiments.", "WARN")
                pending.clear()

        while pending and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()
    return rows


# ── Manifest ───────────────────────────────────────────────────────────────────

def _write_manifest(rows_ok: list[dict], skipped: list[dict], retry: bool = False) -> None:
    path = _MANIFEST_RETRY_PATH if retry else _MANIFEST_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_records: list[dict] = []

    for r in rows_ok:
        run_name = r.get("run_name", r.get("key", ""))
        ckpt_dir = os.path.join(_ROOT, "artifacts", "classification_flim", run_name, "checkpoints")
        all_records.append({
            "run_name":        run_name,
            "dataset":         r.get("dataset", ""),
            "split":           r.get("split", ""),
            "percentage":      r.get("pct", ""),
            "status":          r.get("status", "unknown"),
            "skip_reason":     r.get("error", ""),
            "checkpoint_path": ckpt_dir if r.get("status") == "ok" else "",
        })

    for s in skipped:
        all_records.append({
            "run_name":        s.get("run_name", ""),
            "dataset":         s.get("dataset", ""),
            "split":           s.get("split", ""),
            "percentage":      s.get("pct", ""),
            "status":          "skipped",
            "skip_reason":     s.get("skip_reason", ""),
            "checkpoint_path": "",
        })

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    _log(f"Manifest written: {path}  ({len(all_records)} record(s))")


# ── Summary / dry-run printing ──────────────────────────────────────────────────

def _print_summary(rows: list[dict], skipped: list[dict], t_start: float, gpu_ids: list[int], max_per_gpu: int, retry: bool = False) -> None:
    elapsed = time.time() - t_start
    n_ok  = sum(1 for r in rows if r.get("status") == "ok")
    n_err = len(rows) - n_ok
    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    n_skip_invalid  = len(skipped) - n_skip_existing

    title = "RETRY SUMMARY" if retry else "QUEUE SUMMARY"
    manifest = _MANIFEST_RETRY_PATH if retry else _MANIFEST_PATH

    print(f"\n{'=' * 70}")
    print(f"{title} — classification_flim (FLIM-init + 2-layer Sigmoid head) experiments")
    print(f"{'=' * 70}")
    print(f"  Succeeded          : {n_ok}")
    print(f"  Failed             : {n_err}")
    print(f"  Skipped (existing) : {n_skip_existing}")
    print(f"  Skipped (invalid)  : {n_skip_invalid}")
    print(f"  Total runtime      : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"  GPU profile        : {len(gpu_ids)} GPU(s) × {max_per_gpu} slots/GPU")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('key', '?')}: {str(r.get('error', ''))[:200]}")
    print(f"  Manifest           : {manifest}")
    print(f"{'=' * 70}\n")


def _print_dry_run(experiments: list[dict], skipped: list[dict], gpu_ids: list[int], max_per_gpu: int, cpus_per: int, retry: bool = False,
                   slots_per_gpu: Optional[dict[int, int]] = None,
                   quota_per_gpu: Optional[dict[int, int]] = None) -> None:
    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    n_skip_invalid  = len(skipped) - n_skip_existing
    total = len(experiments) + len(skipped)
    if slots_per_gpu:
        slots_desc = ", ".join(f"GPU{gid}={n}" for gid, n in slots_per_gpu.items())
        max_concurrent = sum(slots_per_gpu.values())
    else:
        slots_desc = str(max_per_gpu)
        max_concurrent = max_per_gpu * len(gpu_ids)
    title = "DRY RUN [RETRY]" if retry else "DRY RUN"

    print(f"\n{'=' * 70}")
    print(f"{title} — classification_flim experiment queue")
    print(f"{'=' * 70}")
    print(f"  GPUs                  : {len(gpu_ids)}  (IDs: {gpu_ids})")
    print(f"  Slots per GPU         : {slots_desc}")
    print(f"  Max concurrent total  : {max_concurrent}")
    if quota_per_gpu:
        print(f"  Experiments per GPU   : " + ", ".join(f"GPU{gid}={n}" for gid, n in quota_per_gpu.items()))
    print(f"  CPUs per experiment   : {cpus_per}")
    print(f"  Experiments queued    : {len(experiments)}")
    print(f"  Skipped (existing)    : {n_skip_existing}")
    print(f"  Skipped (invalid)     : {n_skip_invalid}")
    print(f"  Total grid size       : {total}")
    print()

    datasets_in_queue = sorted({e["dataset"] for e in experiments})
    for ds in datasets_in_queue:
        exps = [e for e in experiments if e["dataset"] == ds]
        print(f"  {ds} ({len(exps)} runs):")
        for e in exps[:6]:
            print(f"    split={e['split']}  pct={e['pct']:>3}%  → {e['run_name']}")
        if len(exps) > 6:
            print(f"    … and {len(exps) - 6} more")

    if n_skip_existing:
        existing = [s for s in skipped if s.get("status") == "skipped_existing"]
        print(f"\n  Skipped-existing ({n_skip_existing}) — will NOT re-run:")
        for s in existing[:5]:
            print(f"    [{s['run_name']}]  {s.get('skip_reason', '')}")
        if len(existing) > 5:
            print(f"    … and {len(existing) - 5} more")

    if n_skip_invalid:
        invalid = [s for s in skipped if s.get("status") == "skipped"]
        print(f"\n  Skipped-invalid ({n_skip_invalid}) — missing data files:")
        for s in invalid[:5]:
            print(f"    [{s['run_name']}]  {s.get('skip_reason', '')}")
        if len(invalid) > 5:
            print(f"    … and {len(invalid) - 5} more")

    print(f"{'=' * 70}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Ray launcher for Experiment 3: FLIM-init encoder + 2-layer Sigmoid "
            "classification head (Linear(48,24)->Sigmoid->Linear(24,n_classes)->Softmax), "
            "trained directly (no SSL, no distillation)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--datasets", nargs="+", choices=_ALL_DATASETS, default=_ALL_DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, choices=_ALL_SPLITS, default=_ALL_SPLITS)
    parser.add_argument("--percentages", nargs="+", type=int, choices=_ALL_PCTS, default=_ALL_PCTS)
    parser.add_argument("--num-gpus", type=int, default=1, metavar="N",
                        help="Usa as GPUs 0..N-1. Ignorado quando --gpus é passado.")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, metavar="ID",
                        help=(
                            "IDs físicos das GPUs a usar, ex: --gpus 0 2 3. "
                            "Só essas recebem experimentos; sobrescreve --num-gpus."
                        ))
    parser.add_argument("--gpu-experiments", nargs="+", type=int, default=None, metavar="N",
                        help=(
                            "Quantos experimentos no TOTAL cada GPU recebe, na mesma ordem de "
                            "--gpus. Ex: --gpus 0 2 3 --gpu-experiments 4 6 8. Um 0 deixa a GPU "
                            "de fora. A soma deve bater com o nº de experimentos da grade. "
                            "Diferente de --gpu-slots, que é quantos rodam SIMULTANEAMENTE."
                        ))
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=1, metavar="N",
                        help="I-JEPA is not used here (no teacher), but keep 1 unless you have "
                             "verified headroom — FLIM encoders + this head are tiny (~60K params).")
    parser.add_argument("--gpu-slots", type=str, default=None, metavar="S0,S1,...",
                        help=(
                            "Slots por GPU individualmente, separados por vírgula. "
                            "Ex: '0,0,0,8' com --num-gpus 4 → só a GPU física 3 recebe 8 slots. "
                            "Sobrescreve --max-concurrent-per-gpu quando especificado."
                        ))
    parser.add_argument("--cpus-per-experiment", type=int, default=4, metavar="N")
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--wandb-update", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--check-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="ophira-ai")
    parser.add_argument("--wandb-project", default="flim-ssl")
    parser.add_argument("--num-workers", type=int, default=4, metavar="N")
    parser.add_argument("--no-imagenet-norm", action="store_true", default=False,
                        help="Disable ImageNet RGB Normalize on ift_lab inputs (correct for FLIM init).")
    parser.add_argument("--run-prefix", type=str, default="", metavar="PREFIX",
                        help="Prepended to every run_name / W&B display name / artifacts path.")
    parser.add_argument("--output-relu", action="store_true", default=False,
                        help="Insert a ReLU between Linear(24,C) and the Softmax; runs are named "
                             "..._relu2l instead of ..._sigmoid2l.")
    parser.add_argument("--output-softplus", action="store_true", default=False,
                        help="Insert a Softplus between Linear(24,C) and the Softmax — head becomes "
                             "...->Sigmoid->Linear(24,C)->Softplus->Softmax. Smooth counterpart of "
                             "--output-relu (no dead-gradient region); runs are named ..._softplus2l "
                             "instead of ..._sigmoid2l. Mutually exclusive with --output-relu.")
    parser.add_argument("--freeze-encoder", action="store_true", default=False,
                        help="Freeze the FLIM encoder; train only the classification head. "
                             "Run names get a '_frozen' suffix so they don't collide with the "
                             "default (unfrozen, end-to-end) grid — launch two separate invocations "
                             "(with/without this flag) to run both.")

    args = parser.parse_args()

    if args.output_relu and args.output_softplus:
        parser.error(
            "--output-relu e --output-softplus são mutuamente exclusivos: a cabeça só admite "
            "UMA não-linearidade antes do Softmax. Escolha uma (ou nenhuma, para a "
            "..._sigmoid2l padrão) e faça duas invocações separadas se quiser as duas grades."
        )

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    t_start = time.time()
    gpu_ids = list(args.gpus) if args.gpus else list(range(args.num_gpus))
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"--gpus tem IDs repetidos: {gpu_ids}")
    _log(f"[GPUS] Usando as GPUs físicas {gpu_ids}")

    # ── Slots por GPU: --gpu-slots sobrescreve --max-concurrent-per-gpu ──────
    slots_per_gpu: Optional[dict[int, int]] = None
    if args.gpu_slots:
        raw = [s.strip() for s in args.gpu_slots.split(",")]
        if len(raw) != len(gpu_ids):
            raise ValueError(
                f"--gpu-slots tem {len(raw)} valores mas há {len(gpu_ids)} GPU(s) {gpu_ids}. "
                f"Devem ser iguais. Ex: --gpus 0 2 3 --gpu-slots 1,1,1"
            )
        slots_per_gpu = {gid: int(s) for gid, s in zip(gpu_ids, raw)}
        _log("[PER-GPU SLOTS] " + ", ".join(f"GPU{gid}={s}" for gid, s in slots_per_gpu.items()))

    # ── Cota de experimentos por GPU: --gpu-experiments ──────────────────────
    quota_per_gpu: Optional[dict[int, int]] = None
    if args.gpu_experiments:
        if len(args.gpu_experiments) != len(gpu_ids):
            raise ValueError(
                f"--gpu-experiments tem {len(args.gpu_experiments)} valores mas há "
                f"{len(gpu_ids)} GPU(s) {gpu_ids}. Devem ser iguais e na mesma ordem. "
                f"Ex: --gpus 0 2 3 --gpu-experiments 4 6 8"
            )
        if any(n < 0 for n in args.gpu_experiments):
            raise ValueError(f"--gpu-experiments não aceita valores negativos: {args.gpu_experiments}")
        quota_per_gpu = {gid: n for gid, n in zip(gpu_ids, args.gpu_experiments)}
        _log("[PER-GPU QUOTA] " + ", ".join(f"GPU{gid}={n}" for gid, n in quota_per_gpu.items()))
        if slots_per_gpu is None:
            # Sem --gpu-slots explícito, a cota também vira concorrência: os N experimentos
            # de cada GPU sobem todos de uma vez. Use --gpu-slots para serializar.
            slots_per_gpu = dict(quota_per_gpu)
            _log("[PER-GPU SLOTS] (= cota) " + ", ".join(f"GPU{gid}={n}" for gid, n in slots_per_gpu.items()))

    if args.retry:
        _log("[RETRY] Mode active — will re-run failed/missing experiments.")
    if args.skip_existing:
        wandb_note = " + W&B cross-check" if args.check_wandb else ""
        _log(f"[SKIP-EXISTING] Checking local metadata{wandb_note} before queuing.")

    ok, reason = validate_output_dir()
    if not ok:
        _log(f"Output directory is not writable: {reason}", "WARN")

    _log("Building experiment grid and validating paths...")
    experiments, skipped = build_experiment_grid(
        datasets=args.datasets, splits=args.splits, percentages=args.percentages,
        wandb_update=args.wandb_update, skip_existing=args.skip_existing,
        check_wandb=args.check_wandb, wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project, num_workers=args.num_workers,
        no_imagenet_norm=args.no_imagenet_norm, run_prefix=args.run_prefix,
        freeze_encoder=args.freeze_encoder, output_relu=args.output_relu,
        output_softplus=args.output_softplus,
    )

    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    _log(
        f"Grid: {len(experiments)} to run, {n_skip_existing} skipped-existing, "
        f"{len(skipped) - n_skip_existing} skipped-invalid. "
        f"Max possible: {len(args.datasets)}×{len(args.splits)}×{len(args.percentages)} = "
        f"{len(args.datasets) * len(args.splits) * len(args.percentages)} runs."
    )

    if quota_per_gpu is not None and experiments:
        total_quota = sum(quota_per_gpu.values())
        if total_quota != len(experiments):
            raise ValueError(
                f"--gpu-experiments soma {total_quota} mas há {len(experiments)} experimento(s) "
                f"a rodar. As cotas devem somar exatamente isso, senão sobram experimentos sem "
                f"GPU designada (ou cotas sem uso). GPUs: {gpu_ids}."
            )

    if args.dry_run:
        _print_dry_run(experiments, skipped, gpu_ids, args.max_concurrent_per_gpu, args.cpus_per_experiment,
                       retry=args.retry, slots_per_gpu=slots_per_gpu, quota_per_gpu=quota_per_gpu)
        _write_manifest([], skipped, retry=args.retry)
        return

    if not experiments:
        _log("No valid experiments to run. Check data paths and grid parameters.", "WARN")
        _write_manifest([], skipped, retry=args.retry)
        return

    rows = run_queue(
        experiments=experiments, gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        fail_fast=args.fail_fast, ray_address=args.ray_address,
        slots_per_gpu=slots_per_gpu, quota_per_gpu=quota_per_gpu,
    )

    _write_manifest(rows, skipped, retry=args.retry)
    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu, retry=args.retry)


if __name__ == "__main__":
    main()
