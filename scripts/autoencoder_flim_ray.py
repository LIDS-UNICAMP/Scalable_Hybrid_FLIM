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
"""
Ray queue for the unsupervised AutoEncoder grid (Experiment Day 5).

Fans the 3 datasets x {5%, 75%} x 3 splits = 18 runs over the available GPUs, one
subprocess per run (``python -m src.modules.autoencoder_flim_module``). Same queue
semantics as ``classification_flim_ray.py``: per-GPU concurrency slots, greedy drain,
``--skip-existing`` / ``--check-wandb`` resume, manifest, ``--dry-run``.

    # the 18 runs of the experiment, 4 GPUs
    python scripts/autoencoder_flim_ray.py \\
        --num-gpus 4 --max-concurrent-per-gpu 1 \\
        --wandb-update --skip-existing --check-wandb

    # see the plan without running anything
    python scripts/autoencoder_flim_ray.py --dry-run

    # re-queue only what failed
    python scripts/autoencoder_flim_ray.py --retry --skip-existing --check-wandb --wandb-update

Architecture per dataset: eggs/larvae ``ch24_32_48``, protozoan ``ch24_30_48``. The FLIM
**weights** always come from the ``ch24_32_48_a0.5_f5`` tree — it is the only one with a
``models/`` subdirectory — and for protozoan the real kernel counts there are 24/30/48,
which ``get_actual_channels_from_weights`` recovers at load time. This mirrors the
convention already used by ``classification_flim_ray.py``.
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
_ALL_SPLITS = [1, 2, 3]
# The brief fixes the grid at 5% and 75%; the others stay reachable for follow-ups.
_ALL_PCTS = [1, 5, 25, 50, 75, 100]
_DEFAULT_PCTS = [5, 75]

_NUM_CLASSES: dict[str, int] = {"eggs": 9, "larvae": 2, "protozoan": 7}

_ARCH_TAG: dict[str, str] = {
    "eggs": "ch24_32_48",
    "larvae": "ch24_32_48",
    "protozoan": "ch24_30_48",
}

_MODEL_ROOT = os.path.join(_ROOT, "data", "to_mateus", "model")

# arch JSON: the directory that declares the *correct* widths (protozoan -> 30).
# weights:   always ch24_32_48, the only tree with models/ on disk.
_ARCH_JSON_BASE: dict[str, str] = {
    "eggs": os.path.join(_MODEL_ROOT, "ch24_32_48_a0.5_f5", "eggs"),
    "larvae": os.path.join(_MODEL_ROOT, "ch24_32_48_a0.5_f5", "larvae"),
    "protozoan": os.path.join(_MODEL_ROOT, "ch24_30_48_a0.5_f5", "protozoan"),
}
_WEIGHTS_BASE: dict[str, str] = {
    "eggs": os.path.join(_MODEL_ROOT, "ch24_32_48_a0.5_f5", "eggs"),
    "larvae": os.path.join(_MODEL_ROOT, "ch24_32_48_a0.5_f5", "larvae"),
    "protozoan": os.path.join(_MODEL_ROOT, "ch24_32_48_a0.5_f5", "protozoan"),
}

_PARASITE_DIR: dict[str, str] = {
    "eggs": "helminth-eggs",
    "larvae": "helminth-larvae",
    "protozoan": "protozoan-cysts",
}
_DEFAULT_DATA_ROOT = os.path.join(_ROOT, "data", "to_modules", "new_split_parasito")

# Substring the Group 3 evaluator matches on (--run-filter autoencoder_resnet_init_flim).
_ARTIFACT_SUBDIR = "autoencoder_resnet_init_flim"
_ARTIFACT_ROOT = os.path.join(_ROOT, "artifacts", _ARTIFACT_SUBDIR)

_MANIFEST_PATH = os.path.join(_ARTIFACT_ROOT, "run_manifest.csv")
_MANIFEST_RETRY_PATH = os.path.join(_ARTIFACT_ROOT, "run_manifest_retry.csv")
_MANIFEST_FIELDS = [
    "run_name", "dataset", "split", "percentage", "architecture",
    "status", "skip_reason", "checkpoint_path",
]

_DEFAULT_WANDB_PROJECT = "journal_02_2026_hybrid_FLIM"


# ── Logging ────────────────────────────────────────────────────────────────────

_LOG_LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2}
_current_log_level = _LOG_LEVELS["INFO"]


def _log(msg: str, level: str = "INFO") -> None:
    if _LOG_LEVELS.get(level, 1) >= _current_log_level:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][{level}] {msg}", flush=True)


# ── W&B / local state check ────────────────────────────────────────────────────

_WANDB_FAILED_STATES = {"crashed", "failed", "killed"}


def _query_wandb_state(run_name: str, entity: str, project: str) -> tuple[str, str]:
    try:
        import wandb as _wandb  # noqa: PLC0415

        api = _wandb.Api(timeout=15)
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name},
                        order="-created_at")
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


def _meta_path(run_name: str) -> str:
    return os.path.join(_ARTIFACT_ROOT, run_name, "run_metadata.json")


def _should_skip_experiment(run_name: str, check_wandb: bool,
                            wandb_entity: str, wandb_project: str) -> tuple[bool, str]:
    if check_wandb:
        wb_state, wb_id = _query_wandb_state(run_name, wandb_entity, wandb_project)
        if wb_state == "running":
            return True, f"wandb:running({wb_id})"
        if wb_state == "finished":
            return True, f"wandb:finished({wb_id})"
        if wb_state == "failed":
            if os.path.isfile(_meta_path(run_name)):
                try:
                    with open(_meta_path(run_name), encoding="utf-8") as fh:
                        if json.load(fh).get("status") == "ok":
                            return True, f"wandb:crashed+local:ok({wb_id})"
                except Exception:
                    pass
            return False, f"wandb:failed({wb_id})"
        if wb_state == "error":
            return False, "wandb:api_error"
        # not_found → fall through to the local check

    path = _meta_path(run_name)
    if not os.path.isfile(path):
        return False, "no_metadata"
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:
        _log(f"Could not read metadata for {run_name}: {exc}", "WARN")
        return False, "metadata_read_error"
    if meta.get("status") == "ok":
        return True, "local:ok"
    return False, f"local:{meta.get('status', 'unknown')}"


# ── Path resolution ────────────────────────────────────────────────────────────

def _arch_json(dataset: str, split: int) -> str:
    return os.path.join(_ARCH_JSON_BASE[dataset], f"train{split}", "architecture.json")


def _flim_weights_path(dataset: str, split: int) -> str:
    return os.path.join(_WEIGHTS_BASE[dataset], f"train{split}", "models")


def _split_json(dataset: str, split: int, pct: int) -> str:
    return os.path.join(
        _DEFAULT_DATA_ROOT, _PARASITE_DIR[dataset], "splits_incremental",
        f"split{split}", f"data_descriptor_perc{pct}.json",
    )


def _run_name(dataset: str, split: int, pct: int, run_prefix: str = "") -> str:
    return f"{run_prefix}encoder_decoder_FLIM_{dataset}_split{split}_pct{pct}"


# ── Pre-flight validation ──────────────────────────────────────────────────────

def validate_experiment(dataset: str, split: int, pct: int) -> tuple[bool, str]:
    arch = _arch_json(dataset, split)
    if not os.path.isfile(arch):
        return False, f"arch JSON not found: {arch}"

    weights = _flim_weights_path(dataset, split)
    if not os.path.isdir(weights):
        return False, f"FLIM weights dir not found: {weights}"
    for n in (1, 2, 3):
        for fname in (f"conv{n}-kernels.npy", f"conv{n}-bias.txt"):
            if not os.path.isfile(os.path.join(weights, fname)):
                return False, f"missing FLIM weight file: {os.path.join(weights, fname)}"

    split_json = _split_json(dataset, split, pct)
    if not os.path.isfile(split_json):
        return False, f"split JSON not found: {split_json}"
    return True, ""


def validate_output_dir() -> tuple[bool, str]:
    try:
        os.makedirs(_ARTIFACT_ROOT, exist_ok=True)
    except Exception as exc:
        return False, f"cannot create {_ARTIFACT_ROOT}: {exc}"
    if not os.access(_ARTIFACT_ROOT, os.W_OK):
        return False, f"not writable: {_ARTIFACT_ROOT}"
    return True, ""


# ── Experiment grid builder ────────────────────────────────────────────────────

def build_experiment_grid(
    datasets: list[str], splits: list[int], pcts: list[int],
    num_workers: int, max_epochs: int, warmup_epochs: int, batch_size: int,
    svm_probe_every: int, log_recon_every: int,
    no_imagenet_norm: bool, wandb_update: bool, wandb_project: str,
    run_prefix: str, skip_existing: bool, check_wandb: bool, wandb_entity: str,
) -> tuple[list[dict], list[dict]]:
    experiments: list[dict] = []
    skipped: list[dict] = []

    for dataset in datasets:
        for pct in pcts:
            for split in splits:
                run_name = _run_name(dataset, split, pct, run_prefix)
                key = f"{dataset}/split{split}/pct{pct}"

                ok, reason = validate_experiment(dataset, split, pct)
                if not ok:
                    _log(f"SKIP (preflight) {key}: {reason}", "WARN")
                    skipped.append({"key": key, "run_name": run_name, "dataset": dataset,
                                    "split": split, "pct": pct, "reason": f"preflight:{reason}"})
                    continue

                if skip_existing:
                    should_skip, why = _should_skip_experiment(
                        run_name, check_wandb, wandb_entity, wandb_project)
                    if should_skip:
                        _log(f"SKIP (done) {key}: {why}")
                        skipped.append({"key": key, "run_name": run_name, "dataset": dataset,
                                        "split": split, "pct": pct, "reason": why})
                        continue

                experiments.append({
                    "key": key,
                    "run_name": run_name,
                    "dataset": dataset,
                    "split": split,
                    "pct": pct,
                    "architecture": _ARCH_TAG[dataset],
                    "num_classes": _NUM_CLASSES[dataset],
                    "arch_json": _arch_json(dataset, split),
                    "flim_weights_path": _flim_weights_path(dataset, split),
                    "num_workers": num_workers,
                    "max_epochs": max_epochs,
                    "warmup_epochs": warmup_epochs,
                    "batch_size": batch_size,
                    "svm_probe_every": svm_probe_every,
                    "log_recon_every": log_recon_every,
                    "no_imagenet_norm": no_imagenet_norm,
                    "wandb_update": wandb_update,
                    "wandb_project": wandb_project,
                    "wandb_entity": wandb_entity,
                })

    return experiments, skipped


# ── GPU slot scheduler ─────────────────────────────────────────────────────────

class GpuSlotScheduler:
    """Concurrency slots per GPU, optionally capped by a per-GPU experiment quota."""

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
        return " | ".join(f"GPU {gid}: {self._running[gid]}/{self._limits[gid]}"
                          for gid in self.gpu_ids)


# ── Ray remote task ────────────────────────────────────────────────────────────

@ray.remote
def run_autoencoder_experiment(
    exp: dict, gpu_id: int, project_root: str,
    queue_index: int = -1, total: int = -1, cpus: int = 4,
) -> dict:
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"] = str(cpus)
    _os.environ["WANDB_CONSOLE"] = "off"
    _os.environ["WANDB_MODE"] = "online"

    cmd = [
        _sys.executable, "-m", "src.modules.autoencoder_flim_module",
        "--dataset", exp["dataset"],
        "--split", str(exp["split"]),
        "--percentage", str(exp["pct"]),
        "--arch-json", exp["arch_json"],
        "--flim-weights-path", exp["flim_weights_path"],
        "--recon-loss", "bce_logits",
        "--run-name", exp["run_name"],
        "--max-epochs", str(exp["max_epochs"]),
        "--warmup-epochs", str(exp["warmup_epochs"]),
        "--batch-size", str(exp["batch_size"]),
        "--num-workers", str(exp.get("num_workers", 4)),
        "--svm-probe-every", str(exp["svm_probe_every"]),
        "--log-recon-every", str(exp["log_recon_every"]),
    ]
    if exp.get("no_imagenet_norm"):
        cmd.append("--no-imagenet-norm")
    if exp.get("wandb_update"):
        cmd += ["--wandb",
                "--wandb-project", exp["wandb_project"],
                "--wandb-entity", exp["wandb_entity"]]

    result: dict = {
        "key": exp["key"], "run_name": exp["run_name"], "gpu_id": gpu_id,
        "queue_index": queue_index, "dataset": exp["dataset"],
        "split": exp["split"], "pct": exp["pct"], "architecture": exp["architecture"],
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

    _log(f"Queue: {total} experiment(s) | {len(gpu_ids)} GPU(s) x "
         f"{max_concurrent_per_gpu} slots = {total_slots} max concurrent")
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
        fut = run_autoencoder_experiment.options(num_cpus=cpus_per_experiment).remote(
            exp, gpu_id=gpu_id, project_root=_ROOT, queue_index=idx,
            total=total, cpus=cpus_per_experiment,
        )
        scheduler.acquire(gpu_id)
        futures[fut] = (gpu_id, exp, idx)
        _log(f"[{idx + 1:>3}/{total}] SUBMIT  gpu={gpu_id}  [{scheduler.status_line()}]  "
             f"pending={len(pending)}  {exp['key']}")
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
                      "status": "ray_error", "error": str(exc), "run_name": exp["run_name"],
                      "dataset": exp["dataset"], "split": exp["split"], "pct": exp["pct"],
                      "architecture": exp["architecture"]}

        rows.append(result)

        if result.get("status") == "ok":
            _log(f"[{completed:>3}/{total}] OK     gpu={gpu_id}  "
                 f"[{scheduler.status_line()}]  {result['key']}")
        else:
            _log(f"[{completed:>3}/{total}] ERROR  gpu={gpu_id}  [{scheduler.status_line()}]  "
                 f"{result['key']}: {result.get('error', '')}", "WARN")
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] Dropping remaining pending experiments.", "WARN")
                pending.clear()

        while pending and scheduler.has_free_slot():
            _submit_next()

    ray.shutdown()
    return rows


# ── Manifest ───────────────────────────────────────────────────────────────────

def _write_manifest(rows: list[dict], skipped: list[dict], retry: bool = False) -> None:
    path = _MANIFEST_RETRY_PATH if retry else _MANIFEST_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records: list[dict] = []

    for r in rows:
        run_name = r.get("run_name", r.get("key", ""))
        ckpt = os.path.join(_ARTIFACT_ROOT, run_name, "checkpoints", "best_kappa.ckpt")
        records.append({
            "run_name": run_name,
            "dataset": r.get("dataset", ""),
            "split": r.get("split", ""),
            "percentage": r.get("pct", ""),
            "architecture": r.get("architecture", ""),
            "status": r.get("status", ""),
            "skip_reason": "",
            "checkpoint_path": ckpt if os.path.isfile(ckpt) else "",
        })

    for s in skipped:
        run_name = s.get("run_name", "")
        ckpt = os.path.join(_ARTIFACT_ROOT, run_name, "checkpoints", "best_kappa.ckpt")
        records.append({
            "run_name": run_name,
            "dataset": s.get("dataset", ""),
            "split": s.get("split", ""),
            "percentage": s.get("pct", ""),
            "architecture": _ARCH_TAG.get(s.get("dataset", ""), ""),
            "status": "skipped",
            "skip_reason": s.get("reason", ""),
            "checkpoint_path": ckpt if os.path.isfile(ckpt) else "",
        })

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    _log(f"Manifest written: {path} ({len(records)} rows)")


# ── Summary / dry-run ──────────────────────────────────────────────────────────

def _best_kappa(run_name: str) -> str:
    """Read best val/svm_kappa and the FLIM baseline back out of run_metadata.json."""
    path = _meta_path(run_name)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        best = meta.get("best_val_svm_kappa")
        base = (meta.get("baseline_flim_svm") or {}).get("kappa")
        if best is None:
            return ""
        if base is None:
            return f"{best:.4f}"
        return f"{best:.4f} (flim {base:.4f}, delta {best - base:+.4f})"
    except Exception:
        return ""


def _print_summary(rows: list[dict], skipped: list[dict], t_start: float,
                   gpu_ids: list[int], max_per_gpu: int, retry: bool = False) -> None:
    elapsed = time.time() - t_start
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]

    print("\n" + "=" * 78)
    print(f"  AutoEncoder queue {'(retry) ' if retry else ''}finished in "
          f"{elapsed / 60:.1f} min on GPUs {gpu_ids} ({max_per_gpu}/GPU)")
    print("=" * 78)
    print(f"  ok: {len(ok)}   error: {len(bad)}   skipped: {len(skipped)}")

    if ok:
        print("\n  Best val/svm_kappa per run (vs the untouched FLIM encoder):")
        for r in sorted(ok, key=lambda x: x.get("key", "")):
            print(f"    {r['key']:<34} {_best_kappa(r.get('run_name', ''))}")

    if bad:
        print("\n  Failures:")
        for r in bad:
            err = (r.get("error", "") or "").splitlines()
            print(f"    {r.get('key', '?'):<34} {err[0] if err else r.get('status', '')}")
    print("=" * 78 + "\n")


def _print_dry_run(experiments: list[dict], skipped: list[dict],
                   gpu_ids: list[int], max_per_gpu: int, cpus_per: int,
                   retry: bool = False) -> None:
    total_slots = max_per_gpu * len(gpu_ids)
    print("\n" + "=" * 78)
    print(f"  DRY-RUN {'(retry) ' if retry else ''}— nothing will be executed")
    print("=" * 78)
    print(f"  GPUs           : {gpu_ids}  ({max_per_gpu} concurrent/GPU "
          f"= {total_slots} slots, {cpus_per} CPU each)")
    print(f"  artifacts      : {_ARTIFACT_ROOT}")
    print(f"  to run         : {len(experiments)}")
    print(f"  skipped        : {len(skipped)}")
    if experiments:
        print("\n  Queue:")
        for i, e in enumerate(experiments):
            print(f"    [{i + 1:>3}] {e['run_name']:<44} {e['architecture']}  "
                  f"{e['num_classes']}cls  {e['max_epochs']}ep")
        print("\n  Sample command:")
        e = experiments[0]
        print(f"    python -m src.modules.autoencoder_flim_module \\\n"
              f"      --dataset {e['dataset']} --split {e['split']} --percentage {e['pct']} \\\n"
              f"      --arch-json {e['arch_json']} \\\n"
              f"      --flim-weights-path {e['flim_weights_path']} \\\n"
              f"      --recon-loss bce_logits --run-name {e['run_name']}")
    if skipped:
        print("\n  Skipped:")
        for s in skipped:
            print(f"    {s['key']:<34} {s['reason']}")
    print("=" * 78 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    global _current_log_level
    import argparse

    parser = argparse.ArgumentParser(
        description="Ray queue for the unsupervised AutoEncoder grid (FLIM encoder + ResNet decoder).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", nargs="+", choices=_ALL_DATASETS, default=_ALL_DATASETS)
    parser.add_argument("--splits", nargs="+", type=int, choices=_ALL_SPLITS, default=_ALL_SPLITS)
    parser.add_argument("--percentages", nargs="+", type=int, choices=_ALL_PCTS,
                        default=_DEFAULT_PCTS,
                        help="Default is the experiment grid: 5 and 75.")
    parser.add_argument("--num-gpus", type=int, default=4, metavar="N",
                        help="Use GPUs 0..N-1. Overridden by --gpus.")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, metavar="ID",
                        help="Explicit GPU ids, e.g. --gpus 0 1 3.")
    parser.add_argument("--gpu-experiments", nargs="+", type=int, default=None, metavar="N",
                        help="Total experiment quota per GPU, aligned with the GPU list.")
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=1, metavar="N")
    parser.add_argument("--gpu-slots", type=str, default=None, metavar="S0,S1,...",
                        help="Per-GPU concurrency slots, aligned with the GPU list.")
    parser.add_argument("--cpus-per-experiment", type=int, default=4, metavar="N")
    parser.add_argument("--ray-address", type=str, default=None)

    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, metavar="N")
    parser.add_argument("--svm-probe-every", type=int, default=1,
                        help="Run the one-vs-one SVM probe every N validation epochs.")
    parser.add_argument("--log-recon-every", type=int, default=10,
                        help="Log the LAB->RGB reconstruction preview every N epochs.")
    parser.add_argument("--no-imagenet-norm", action="store_true", default=False,
                        help="Disable ImageNet Normalize on the LAB input (and on the target).")

    parser.add_argument("--wandb-update", action="store_true",
                        help="Log runs to W&B (required for the experiment's signature).")
    parser.add_argument("--wandb-entity", default="ophira-ai")
    parser.add_argument("--wandb-project", default=_DEFAULT_WANDB_PROJECT)
    parser.add_argument("--run-prefix", type=str, default="", metavar="PREFIX")

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--check-wandb", action="store_true")
    parser.add_argument("--retry", action="store_true",
                        help="Queue only runs whose local metadata is missing or not ok.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN"], default="INFO")

    args = parser.parse_args()
    _current_log_level = _LOG_LEVELS[args.log_level]

    gpu_ids = args.gpus if args.gpus else list(range(args.num_gpus))
    if not gpu_ids:
        _log("No GPUs selected.", "WARN")
        sys.exit(1)

    slots_per_gpu: Optional[dict[int, int]] = None
    if args.gpu_slots:
        try:
            slots = [int(s) for s in args.gpu_slots.split(",")]
        except ValueError:
            _log(f"--gpu-slots must be a comma-separated list of ints: {args.gpu_slots}", "WARN")
            sys.exit(1)
        if len(slots) != len(gpu_ids):
            _log(f"--gpu-slots has {len(slots)} entries but {len(gpu_ids)} GPUs selected.", "WARN")
            sys.exit(1)
        slots_per_gpu = dict(zip(gpu_ids, slots))

    quota_per_gpu: Optional[dict[int, int]] = None
    if args.gpu_experiments:
        if len(args.gpu_experiments) != len(gpu_ids):
            _log(f"--gpu-experiments has {len(args.gpu_experiments)} entries but "
                 f"{len(gpu_ids)} GPUs selected.", "WARN")
            sys.exit(1)
        quota_per_gpu = dict(zip(gpu_ids, args.gpu_experiments))

    ok, reason = validate_output_dir()
    if not ok:
        _log(reason, "WARN")
        sys.exit(1)

    if args.wandb_update and args.wandb_project != _DEFAULT_WANDB_PROJECT:
        _log(f"W&B project is '{args.wandb_project}', not the experiment board "
             f"'{_DEFAULT_WANDB_PROJECT}'. These runs must not land in the old workspace.",
             "WARN")

    skip_existing = args.skip_existing or args.retry
    experiments, skipped = build_experiment_grid(
        datasets=args.datasets, splits=args.splits, pcts=args.percentages,
        num_workers=args.num_workers, max_epochs=args.max_epochs,
        warmup_epochs=args.warmup_epochs, batch_size=args.batch_size,
        svm_probe_every=args.svm_probe_every, log_recon_every=args.log_recon_every,
        no_imagenet_norm=args.no_imagenet_norm, wandb_update=args.wandb_update,
        wandb_project=args.wandb_project, run_prefix=args.run_prefix,
        skip_existing=skip_existing, check_wandb=args.check_wandb,
        wandb_entity=args.wandb_entity,
    )

    if args.dry_run:
        _print_dry_run(experiments, skipped, gpu_ids, args.max_concurrent_per_gpu,
                       args.cpus_per_experiment, retry=args.retry)
        return

    if not experiments:
        _log("Nothing to run — every experiment was skipped.")
        _write_manifest([], skipped, retry=args.retry)
        return

    t_start = time.time()
    rows = run_queue(
        experiments=experiments, gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        fail_fast=args.fail_fast, ray_address=args.ray_address,
        slots_per_gpu=slots_per_gpu, quota_per_gpu=quota_per_gpu,
    )

    _write_manifest(rows, skipped, retry=args.retry)
    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu, retry=args.retry)

    if any(r.get("status") != "ok" for r in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
