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

"""distillation_conv_ray.py — Ray launcher for next_layers distillation experiments.

Conv projection head variant: 48 → 128 → 256 → 512 → 1280 (1×1 convs + GAP).
Run name convention: distillation_<dataset>_split<N>_pct<P>_next_layers_<type>

Example runs::

    # Full grid — direct only
    python scripts/distillation_conv_ray.py \\
        --num-gpus 1 --max-concurrent-per-gpu 3 \\
        --distillation-types direct --wandb-update

    # Single experiment (debug)
    python scripts/distillation_conv_ray.py \\
        --datasets protozoan --splits 3 --percentages 100 \\
        --distillation-types direct \\
        --num-gpus 1 --max-concurrent-per-gpu 1

    # Dry-run
    python scripts/distillation_conv_ray.py --dry-run

    # Retry after partial failure (skip already-finished experiments)
    python scripts/distillation_conv_ray.py \\
        --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --retry --skip-existing

    # Retry with W&B cross-check (skips experiments finished locally AND on W&B)
    python scripts/distillation_conv_ray.py \\
        --num-gpus 1 --max-concurrent-per-gpu 1 \\
        --retry --skip-existing --check-wandb \\
        --wandb-entity ophira-ai --wandb-project flim-ssl

Flags --retry and --skip-existing
----------------------------------
--skip-existing
    Before queuing each experiment, checks whether it already completed
    successfully by reading ``artifacts/distillation/<run_name>/run_metadata.json``
    and verifying ``"status": "ok"``.  If the file exists and the status is ok,
    the experiment is added to the skipped list instead of the queue.
    Use this when re-running after a partial failure (e.g. OOM) to avoid
    wasting time repeating experiments that already finished correctly.

--check-wandb
    Extends --skip-existing with a W&B API cross-check.  For each experiment
    that has a local ``status: ok``, also queries the W&B run by display name
    and verifies its state is ``"finished"``.  An experiment is only skipped if
    BOTH the local metadata and W&B agree it completed.  Useful when a run may
    have been interrupted after writing the metadata but before W&B synced.
    Requires the ``wandb`` package and a valid API key (WANDB_API_KEY env var
    or prior ``wandb login``).  W&B failures are non-fatal: if the API call
    errors out the experiment is NOT skipped (safe default).

--retry
    Marks the entire launcher invocation as a retry run.  Concretely:
      - Logs show ``[RETRY]`` prefix so it is easy to distinguish in output.
      - The manifest CSV is written to ``run_manifest_conv_retry.csv`` instead
        of ``run_manifest_conv.csv``, preserving the original run record.
    Does not change run names or checkpoint paths — retried experiments
    overwrite the previous artifacts for that run name.
    Intended to be combined with --skip-existing so only genuinely failed or
    missing experiments are re-executed.
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
_ALL_DIST_TYPES    = ["direct", "direct_cosine", "hybrid"]
_ALL_INITS         = ["trunc_normal"]

# ── Projection head variants ───────────────────────────────────────────────────
# conv_next_layers : 4× 1×1 convs 48→128→256→512→1280 + GAP  (original)
# 3x3_bn2d_1280   : single Conv2d(3×3) 48→1280 + BN2d + GELU + GAP
# 1x1_bn2d_1280   : single Conv2d(1×1) 48→1280 + BN2d + GELU + GAP
# 2l_1x1_bn2d_256_1280     : two Conv2d(1×1) 48→256→1280 + BN2d + GELU (~402k), random init
# 2l_1x1_init_flim_256_1280: same two-layer head but FLIM encoder init forced ON
# 3x3_init_flim_1280       : 3x3_bn2d_1280 head (~615k, "600k") but FLIM encoder init forced ON
# next_layers_init_flim    : conv_next_layers head (~889k, "800k") but FLIM encoder init forced ON
# 1x1_init_flim_frozen     : 1x1 head (~123k, "123k") with FLIM encoder init FROZEN — only the
#                            projection trains. Student fed LAB[0,1] (--no-imagenet-norm), teacher
#                            fed ImageNet-norm (--teacher-imagenet-norm). Checkpoint by val/knn_kappa
#                            on the 1280d projection (--knn-probe projection).
_ALL_PROJ_TYPES    = ["conv_next_layers", "3x3_bn2d_1280", "1x1_bn2d_1280", "2l_1x1_bn2d_256_1280", "2l_1x1_init_flim_256_1280", "3x3_init_flim_1280", "next_layers_init_flim", "1x1_init_flim_frozen"]

# Proj types that always train the FLIM-initialised encoder (encoder_init forced to "flim")
_FLIM_INIT_PROJ_TYPES = {"2l_1x1_init_flim_256_1280", "3x3_init_flim_1280", "next_layers_init_flim", "1x1_init_flim_frozen"}

_NUM_CLASSES: dict[str, int] = {
    "eggs":      9,
    "larvae":    2,
    "protozoan": 7,
}

_DEFAULT_DATA_ROOT = os.path.join(
    _ROOT, "data", "to_modules", "new_split_parasito"
)

_ARCH_BASE: dict[str, str] = {
    "eggs":      os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "eggs"),
    "larvae":    os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "larvae"),
    "protozoan": os.path.join(_ROOT, "data", "to_mateus", "model", "ch24_30_48_a0.5_f5", "protozoan"),
}

# Paths used when use_flim_init=True — points every dataset to the directory
# that actually contains a models/ subdirectory with trained FLIM weights.
# For protozoan the weights live in ch24_32_48 (layer2=32ch), not ch24_30_48.
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

_MANIFEST_PATH         = os.path.join(_ROOT, "artifacts", "distillation", "run_manifest_conv.csv")
_MANIFEST_RETRY_PATH   = os.path.join(_ROOT, "artifacts", "distillation", "run_manifest_conv_retry.csv")
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


# ── W&B / local state check ────────────────────────────────────────────────────

# Possible return values of _query_experiment_state:
#   "finished"  → completed successfully    → SKIP
#   "running"   → currently in progress     → SKIP  (do not interrupt)
#   "failed"    → crashed / failed / killed → QUEUE (re-run)
#   "not_found" → no W&B record             → fall back to local
#   "error"     → W&B API error             → QUEUE (safe default)

_WANDB_FAILED_STATES = {"crashed", "failed", "killed"}
_WANDB_SKIP_STATES   = {"finished", "running"}


def _query_wandb_state(
    run_name: str,
    entity:   str,
    project:  str,
) -> tuple[str, str]:
    """Query the most recent W&B run with display_name == run_name.

    Returns (state, run_id) where state is one of:
        "finished", "running", "failed", "not_found", "error"
    """
    try:
        import wandb as _wandb  # noqa: PLC0415
        api  = _wandb.Api(timeout=15)
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": run_name},
            order="-created_at",
        )
        for run in runs:
            if run.state in _WANDB_FAILED_STATES:
                return "failed",   run.id
            if run.state == "finished":
                return "finished", run.id
            if run.state == "running":
                return "running",  run.id
            # any other state (e.g. "waiting") — treat as failed
            return "failed", run.id
        return "not_found", ""
    except Exception as exc:
        _log(
            f"W&B API error for '{run_name}': {exc} — treating as unknown (will re-queue).",
            "WARN",
        )
        return "error", ""


def _should_skip_experiment(
    run_name:     str,
    check_wandb:  bool,
    wandb_entity: str,
    wandb_project: str,
) -> tuple[bool, str]:
    """Decide whether to skip (not re-queue) an experiment.

    Decision logic (in priority order):

    1. W&B state = "running"   → SKIP   (already in progress)
    2. W&B state = "finished"  → SKIP   (completed successfully)
    3. W&B state = "failed"    → QUEUE  (re-run)
    4. W&B state = "not_found" → check local metadata as fallback
    5. W&B state = "error"     → QUEUE  (safe default on API failure)
    6. No --check-wandb        → check local metadata only

    Local metadata fallback:
        status "ok"    → SKIP
        anything else  → QUEUE
    """
    # ── Step 1: W&B (authoritative) ───────────────────────────────────────
    if check_wandb:
        wb_state, wb_id = _query_wandb_state(run_name, wandb_entity, wandb_project)

        if wb_state == "running":
            return True,  f"wandb:running({wb_id})"

        if wb_state == "finished":
            return True,  f"wandb:finished({wb_id})"

        if wb_state == "failed":
            # Se W&B crashou mas local está ok, treino terminou — não re-rodar
            local_meta = os.path.join(
                _ROOT, "artifacts", "distillation", run_name, "run_metadata.json"
            )
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

        # wb_state == "not_found" → fall through to local check

    # ── Step 2: local metadata fallback ───────────────────────────────────
    meta_path = os.path.join(
        _ROOT, "artifacts", "distillation", run_name, "run_metadata.json"
    )
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

def _arch_json(dataset: str, split: int, use_flim_init: bool = False) -> str:
    base = _ARCH_BASE_FLIM[dataset] if use_flim_init else _ARCH_BASE[dataset]
    return os.path.join(base, f"train{split}", "architecture.json")


def _flim_weights_path(dataset: str, split: int) -> str:
    return os.path.join(_ARCH_BASE_FLIM[dataset], f"train{split}", "models")


def _split_json(dataset: str, split: int, pct: int) -> str:
    parasite_dir = _PARASITE_DIR[dataset]
    return os.path.join(
        _DEFAULT_DATA_ROOT,
        parasite_dir,
        "splits_incremental",
        f"split{split}",
        f"data_descriptor_perc{pct}.json",
    )


def _run_name(
    dataset: str, split: int, pct: int, dist_type: str,
    proj_type: str = "conv_next_layers", use_flim_init: bool = False,
    no_imagenet_norm: bool = False,
) -> str:
    suffix = "_flim_init" if use_flim_init else ""
    if proj_type == "3x3_bn2d_1280":
        name = f"distillation_{dataset}_split{split}_pct{pct}_3x3_BN2d_1280_one_layer{suffix}"
    elif proj_type == "1x1_bn2d_1280":
        name = f"distillation_{dataset}_split{split}_pct{pct}_1x1_BN2d_1280_one_layer{suffix}"
    elif proj_type == "2l_1x1_bn2d_256_1280":
        name = f"distillation_{dataset}_split{split}_pct{pct}_2l_1x1_BN2d_256_1280{suffix}"
    elif proj_type == "2l_1x1_init_flim_256_1280":
        # FLIM init is intrinsic to this variant; the name already encodes it,
        # so no extra "_flim_init" suffix is appended.
        name = f"distillation_{dataset}_split{split}_pct{pct}_2l_1x1_init_flim_256_1280"
    elif proj_type == "3x3_init_flim_1280":
        # 600k variant (3x3 head) with FLIM init intrinsic — name encodes init_flim.
        name = f"distillation_{dataset}_split{split}_pct{pct}_3x3_BN2d_1280_one_layer_init_flim"
    elif proj_type == "next_layers_init_flim":
        # 800k variant (conv_next_layers head) with FLIM init intrinsic — name encodes init_flim.
        name = f"distillation_{dataset}_split{split}_pct{pct}_next_layers_init_flim_{dist_type}"
    elif proj_type == "1x1_init_flim_frozen":
        # 123k variant: FLIM encoder FROZEN, only the 1×1 projection trains.
        # Norm split (student LAB / teacher ImageNet) is forced in the subprocess,
        # so the name does not carry a _no_imagenet_norm suffix.
        return f"distillation_{dataset}_split{split}_pct{pct}_1x1_BN2d_1280_flim_frozen"
    else:
        name = f"distillation_{dataset}_split{split}_pct{pct}_next_layers_{dist_type}{suffix}"
    if no_imagenet_norm:
        name += "_no_imagenet_norm"
    return name


# ── Pre-flight validation ──────────────────────────────────────────────────────

def validate_experiment(dataset: str, split: int, pct: int) -> tuple[bool, str]:
    arch = _arch_json(dataset, split)
    if not os.path.isfile(arch):
        return False, f"arch JSON not found: {os.path.relpath(arch, _ROOT)}"

    sjson = _split_json(dataset, split, pct)
    if not os.path.isfile(sjson):
        return False, f"split JSON not found: {os.path.relpath(sjson, _ROOT)}"

    return True, ""


def validate_output_dir() -> tuple[bool, str]:
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
    datasets:      list[str],
    splits:        list[int],
    percentages:   list[int],
    dist_types:    list[str],
    inits:         list[str],
    alpha:         float,
    temperature:   float,
    wandb_update:  bool,
    skip_existing: bool = False,
    check_wandb:   bool = False,
    wandb_entity:  str  = "ophira-ai",
    wandb_project: str  = "flim-ssl",
    proj_type:     str  = "conv_next_layers",
    use_flim_init: bool = False,
    num_workers:   int  = 4,
    no_imagenet_norm: bool = False,
    run_prefix:    str  = "",
) -> tuple[list[dict], list[dict]]:
    valid: list[dict] = []
    skipped: list[dict] = []

    for dataset in datasets:
        for split in splits:
            for pct in percentages:
                for dist_type in dist_types:
                    for init in inits:
                        # Some proj types (e.g. 2l_1x1_init_flim_256_1280) always train
                        # the FLIM-initialised encoder regardless of the global --flim-init flag.
                        eff_flim_init = use_flim_init or proj_type in _FLIM_INIT_PROJ_TYPES
                        actual_init = "flim" if eff_flim_init else init
                        run_name = _run_name(dataset, split, pct, dist_type, proj_type, eff_flim_init, no_imagenet_norm=no_imagenet_norm)
                        if run_prefix:
                            run_name = f"{run_prefix}{run_name}"
                        ok, reason = validate_experiment(dataset, split, pct)

                        base = {
                            "run_name":          run_name,
                            "dataset":           dataset,
                            "split":             split,
                            "pct":               pct,
                            "distillation_type": dist_type,
                            "encoder_init":      actual_init,
                            "alpha":             alpha,
                            "temperature":       temperature,
                            "arch_json":         _arch_json(dataset, split, eff_flim_init),
                            "wandb_update":      wandb_update,
                            "num_classes":       _NUM_CLASSES[dataset],
                            "proj_type":         proj_type,
                            "flim_weights_path": _flim_weights_path(dataset, split) if eff_flim_init else None,
                            "num_workers":       num_workers,
                            "no_imagenet_norm":  no_imagenet_norm,
                            "key":               run_name,
                        }

                        if not ok:
                            _log(f"[SKIP] {run_name}  reason: {reason}", "WARN")
                            skipped.append({
                                **base,
                                "status":       "skipped",
                                "skip_reason":  reason,
                                "checkpoint_path":  "",
                                "embedding_path":   "",
                                "svm_metrics_path": "",
                            })
                            continue

                        # ── skip-existing check ────────────────────────────
                        if skip_existing:
                            skip, reason = _should_skip_experiment(
                                run_name,
                                check_wandb=check_wandb,
                                wandb_entity=wandb_entity,
                                wandb_project=wandb_project,
                            )
                            if skip:
                                _log(f"[SKIP] {run_name}  ({reason})")
                                skipped.append({
                                    **base,
                                    "status":           "skipped_existing",
                                    "skip_reason":      reason,
                                    "checkpoint_path":  "",
                                    "embedding_path":   "",
                                    "svm_metrics_path": "",
                                })
                                continue
                            else:
                                _log(f"[QUEUE] {run_name}  ({reason})")

                        valid.append(base)

    return valid, skipped


# ── GPU slot scheduler ─────────────────────────────────────────────────────────

class GpuSlotScheduler:
    def __init__(
        self,
        gpu_ids:     list[int],
        max_per_gpu: "int | dict[int, int]",
    ) -> None:
        # Aceita int uniforme OU dict por GPU: {0: 7, 1: 3}
        if isinstance(max_per_gpu, int):
            self._limits: dict[int, int] = {gid: max_per_gpu for gid in gpu_ids}
        else:
            self._limits = dict(max_per_gpu)
        self._running: dict[int, int] = {gid: 0 for gid in gpu_ids}

    @property
    def gpu_ids(self) -> list[int]:
        return sorted(self._running)

    def pick_gpu(self) -> Optional[int]:
        candidates = [
            (count, gid)
            for gid, count in self._running.items()
            if count < self._limits[gid]
        ]
        return min(candidates)[1] if candidates else None

    def acquire(self, gpu_id: int) -> None:
        self._running[gpu_id] += 1

    def release(self, gpu_id: int) -> None:
        self._running[gpu_id] = max(0, self._running[gpu_id] - 1)

    def total_running(self) -> int:
        return sum(self._running.values())

    def has_free_slot(self) -> bool:
        return any(self._running[gid] < self._limits[gid] for gid in self._running)

    def total_slots(self) -> int:
        return sum(self._limits.values())

    def status_line(self) -> str:
        return " | ".join(
            f"GPU {gid}: {self._running[gid]}/{self._limits[gid]}"
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
    """Run one next_layers distillation experiment as a subprocess."""
    import os as _os
    import subprocess as _sp
    import sys as _sys

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"]       = str(cpus)
    # W&B: força modo online e desabilita captura de console (capture_output=True é pipe)
    _os.environ["WANDB_CONSOLE"] = "off"
    _os.environ["WANDB_MODE"]    = "online"

    # ── Select module based on proj_type flag ─────────────────────────────
    proj_type  = exp.get("proj_type", "conv_next_layers")
    frozen     = proj_type == "1x1_init_flim_frozen"
    if proj_type in ("3x3_bn2d_1280", "1x1_bn2d_1280", "3x3_init_flim_1280", "1x1_init_flim_frozen"):
        module_name = "src.modules.distillation_onelayer_module"
    elif proj_type in ("2l_1x1_bn2d_256_1280", "2l_1x1_init_flim_256_1280"):
        module_name = "src.modules.distillation_twolayer_module"
    else:
        module_name = "src.modules.distillation_conv_module"

    cmd = [
        _sys.executable, "-m", module_name,
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
    if proj_type in ("1x1_bn2d_1280", "1x1_init_flim_frozen"):
        cmd += ["--proj-kernel", "1"]
    if exp.get("flim_weights_path"):
        cmd += ["--flim-weights-path", exp["flim_weights_path"]]

    cmd += ["--num-workers", str(exp.get("num_workers", 4))]
    # Sinaliza o tipo do experimento no W&B (tag/config) — ex.: 1x1_init_flim_frozen.
    cmd += ["--proj-type", proj_type]
    # Frozen variant forces the decoupled-norm + freeze + projection-probe regime
    # regardless of the launcher's global --no-imagenet-norm flag.
    if exp.get("no_imagenet_norm") or frozen:
        cmd.append("--no-imagenet-norm")
    if frozen:
        cmd += ["--freeze-encoder", "--teacher-imagenet-norm",
                "--knn-probe", "projection"]

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

    if not _os.path.isfile(exp["arch_json"]):
        result["status"]     = "error"
        result["returncode"] = -1
        result["error"]      = f"PREFLIGHT FAIL — arch JSON not found: {exp['arch_json']}"
        return result

    try:
        proc = _sp.run(
            cmd,
            cwd=project_root,
            stdout=None,
            stderr=_sp.PIPE,
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
    slots_per_gpu:          Optional[dict[int, int]] = None,
) -> list[dict]:
    if not experiments:
        _log("No experiments to run.", "WARN")
        return []

    # Se --gpu-slots foi passado, usa por GPU; senão usa uniform max_concurrent_per_gpu
    if slots_per_gpu:
        scheduler_arg = slots_per_gpu
        total_slots   = sum(slots_per_gpu.values())
    else:
        scheduler_arg = max_concurrent_per_gpu
        total_slots   = max_concurrent_per_gpu * len(gpu_ids)

    ray_kwargs: dict[str, Any] = {"ignore_reinit_error": True, "log_to_driver": True}
    if ray_address:
        ray_kwargs["address"] = ray_address
    else:
        total_cpus = cpus_per_experiment * total_slots
        ray_kwargs["num_cpus"] = total_cpus
        ray_kwargs["num_gpus"] = 0

    if not ray.is_initialized():
        ray.init(**ray_kwargs)
    _log("Ray initialised.")

    scheduler    = GpuSlotScheduler(gpu_ids=gpu_ids, max_per_gpu=scheduler_arg)
    total        = len(experiments)
    pending      = list(experiments)
    futures: dict[Any, tuple[int, dict, int]] = {}
    rows:    list[dict] = []
    global_index = 0
    completed    = 0
    had_failure  = False

    if slots_per_gpu:
        slots_desc = "  ".join(f"GPU{gid}={s}" for gid, s in sorted(slots_per_gpu.items()))
        _log(f"Queue: {total} experiment(s) | {slots_desc} = {total_slots} max concurrent")
    else:
        _log(
            f"Queue: {total} experiment(s) | "
            f"{len(gpu_ids)} GPU(s) × {max_concurrent_per_gpu} slots = "
            f"{total_slots} max concurrent"
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


# ── Manifest ───────────────────────────────────────────────────────────────────

def _write_manifest(rows_ok: list[dict], skipped: list[dict], retry: bool = False) -> None:
    path = _MANIFEST_RETRY_PATH if retry else _MANIFEST_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_records)

    _log(f"Manifest written: {path}  ({len(all_records)} record(s))")


# ── Summary ────────────────────────────────────────────────────────────────────

def _print_summary(
    rows:        list[dict],
    skipped:       list[dict],
    t_start:       float,
    gpu_ids:       list[int],
    max_per_gpu:   int,
    retry:         bool = False,
    slots_per_gpu: Optional[dict[int, int]] = None,
) -> None:
    elapsed  = time.time() - t_start
    n_ok     = sum(1 for r in rows if r.get("status") == "ok")
    n_err    = len(rows) - n_ok
    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    n_skip_invalid  = len(skipped) - n_skip_existing

    gpu_stats: dict[int, dict] = {gid: {"ok": 0, "error": 0} for gid in gpu_ids}
    for r in rows:
        gid = r.get("gpu_id")
        if gid in gpu_stats:
            gpu_stats[gid]["ok" if r.get("status") == "ok" else "error"] += 1

    title = "RETRY SUMMARY" if retry else "QUEUE SUMMARY"
    manifest = _MANIFEST_RETRY_PATH if retry else _MANIFEST_PATH

    print(f"\n{'=' * 70}")
    print(f"{title} — next_layers distillation experiments")
    print(f"{'=' * 70}")
    print(f"  Succeeded           : {n_ok}")
    print(f"  Failed              : {n_err}")
    print(f"  Skipped (existing)  : {n_skip_existing}")
    print(f"  Skipped (invalid)   : {n_skip_invalid}")
    print(f"  Total runtime       : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    if slots_per_gpu:
        slots_desc = ", ".join(f"GPU{gid}={s}" for gid, s in sorted(slots_per_gpu.items()))
        print(f"\n  GPU profile    : per-GPU slots  [{slots_desc}]")
    else:
        print(f"\n  GPU profile    : {len(gpu_ids)} GPU(s) × {max_per_gpu} slots/GPU")
    print("  Per-GPU results:")
    for gid in sorted(gpu_stats):
        s = gpu_stats[gid]
        print(f"    GPU {gid}: {s['ok']} ok, {s['error']} error")
    existing = [s for s in skipped if s.get("status") == "skipped_existing"]
    if existing:
        print(f"\n  Skipped-existing ({len(existing)}):")
        for s in existing[:10]:
            print(f"    - {s.get('run_name', '?')}  ({s.get('skip_reason', '')})")
        if len(existing) > 10:
            print(f"    … and {len(existing) - 10} more")
    invalid = [s for s in skipped if s.get("status") == "skipped"]
    if invalid:
        print(f"\n  Skipped-invalid ({len(invalid)}):")
        for s in invalid[:5]:
            print(f"    - {s.get('run_name', '?')}: {s.get('skip_reason', '')}")
        if len(invalid) > 5:
            print(f"    … and {len(invalid) - 5} more")
    if n_err:
        print("\n  Failed experiments:")
        for r in rows:
            if r.get("status") != "ok":
                print(f"    - {r.get('key', '?')}: {str(r.get('error', ''))[:200]}")
    print(f"  Manifest       : {manifest}")
    print(f"{'=' * 70}\n")


def _print_dry_run(
    experiments:   list[dict],
    skipped:       list[dict],
    gpu_ids:       list[int],
    max_per_gpu:   int,
    cpus_per:      int,
    retry:         bool = False,
    slots_per_gpu: Optional[dict[int, int]] = None,
) -> None:
    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    n_skip_invalid  = len(skipped) - n_skip_existing
    total           = len(experiments) + len(skipped)
    if slots_per_gpu:
        max_concurrent = sum(slots_per_gpu.values())
    else:
        max_concurrent = max_per_gpu * len(gpu_ids)
    title = "DRY RUN [RETRY]" if retry else "DRY RUN"

    print(f"\n{'=' * 70}")
    print(f"{title} — next_layers distillation experiment queue")
    print(f"{'=' * 70}")
    print(f"  GPUs                  : {len(gpu_ids)}  (IDs: {gpu_ids})")
    if slots_per_gpu:
        for gid, s in sorted(slots_per_gpu.items()):
            print(f"  Slots GPU {gid}           : {s}")
    else:
        print(f"  Slots per GPU         : {max_per_gpu}")
    print(f"  Max concurrent total  : {max_concurrent}")
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
            print(
                f"    split={e['split']}  pct={e['pct']:>3}%  "
                f"type={e['distillation_type']:<8}  init={e['encoder_init']}"
                f"  → {e['run_name']}"
            )
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
    parser = argparse.ArgumentParser(
        description=(
            "Ray launcher for next_layers distillation: FLIM CNN with conv projection head "
            "(48→128→256→512→1280 via 1×1 convs + AdaptiveAvgPool)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datasets", nargs="+", choices=_ALL_DATASETS, default=_ALL_DATASETS,
    )
    parser.add_argument(
        "--splits", nargs="+", type=int, choices=_ALL_SPLITS, default=_ALL_SPLITS,
    )
    parser.add_argument(
        "--percentages", nargs="+", type=int, choices=_ALL_PCTS, default=_ALL_PCTS,
    )
    parser.add_argument(
        "--distillation-types", nargs="+", choices=_ALL_DIST_TYPES, default=_ALL_DIST_TYPES,
    )
    parser.add_argument(
        "--inits", nargs="+", choices=_ALL_INITS, default=_ALL_INITS,
    )
    parser.add_argument("--alpha",       type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--num-gpus",               type=int, default=1, metavar="N")
    parser.add_argument("--max-concurrent-per-gpu", type=int, default=3, metavar="N",
                        help="Slots uniformes por GPU. Ignorado se --gpu-slots for passado.")
    parser.add_argument("--gpu-slots",              type=str, default=None, metavar="S0,S1,...",
                        help=(
                            "Slots por GPU individualmente, separados por vírgula. "
                            "Ex: '7,3' → GPU0=7 slots, GPU1=3 slots. "
                            "Sobrescreve --max-concurrent-per-gpu quando especificado."
                        ))
    parser.add_argument("--cpus-per-experiment",    type=int, default=4, metavar="N")
    parser.add_argument("--ray-address", type=str, default=None)
    parser.add_argument("--wandb-update",  action="store_true")
    parser.add_argument("--fail-fast",     action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--log-level",     choices=["DEBUG", "INFO", "WARN"], default="INFO")
    # ── Retry / skip-existing flags ────────────────────────────────────────
    parser.add_argument(
        "--retry",
        action="store_true",
        help=(
            "Mark this invocation as a retry run. Logs show [RETRY] prefix and "
            "the manifest is written to run_manifest_conv_retry.csv instead of "
            "run_manifest_conv.csv.  Combine with --skip-existing to avoid "
            "re-running experiments that already completed successfully."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip experiments that already have a local run_metadata.json with "
            "status=ok.  Avoids repeating work after a partial failure (e.g. OOM). "
            "Add --check-wandb to also cross-check the W&B run state."
        ),
    )
    parser.add_argument(
        "--check-wandb",
        action="store_true",
        help=(
            "When --skip-existing is active, also query the W&B API to verify the "
            "run state is 'finished'.  An experiment is only skipped if BOTH the "
            "local metadata and W&B agree it completed.  W&B API errors are "
            "non-fatal: the experiment will be re-queued instead of silently skipped."
        ),
    )
    parser.add_argument("--wandb-entity",  default="ophira-ai")
    parser.add_argument("--wandb-project", default="flim-ssl")
    # ── Projection head variant ────────────────────────────────────────────
    parser.add_argument(
        "--proj-type",
        choices=_ALL_PROJ_TYPES,
        default="conv_next_layers",
        help=(
            "Projection head architecture. "
            "'conv_next_layers': 4× 1×1 convs 48→128→256→512→1280 (original). "
            "'3x3_bn2d_1280': single Conv2d(3×3) 48→1280 + BN2d + GELU. "
            "'1x1_bn2d_1280': single Conv2d(1×1) 48→1280 + BN2d + GELU. "
            "'2l_1x1_bn2d_256_1280': two Conv2d(1×1) 48→256→1280 + BN2d + GELU (~402k total), random init. "
            "'2l_1x1_init_flim_256_1280': same two-layer head but FLIM encoder init forced ON "
            "(protozoan loads ch24_32_48 weights via PROTOZOAN_FLIM_ARCH). "
            "'3x3_init_flim_1280': 3x3 head (~615k, '600k') with FLIM encoder init forced ON. "
            "'next_layers_init_flim': conv_next_layers head (~889k, '800k') with FLIM encoder init forced ON. "
            "'1x1_init_flim_frozen': 1x1 head (~123k) with the FLIM encoder FROZEN — only the projection "
            "trains; student LAB[0,1], teacher ImageNet-norm, checkpoint by val/knn_kappa on the 1280d projection. "
            "Run names: ..._next_layers_<type>, ..._3x3_BN2d_1280_one_layer, "
            "..._1x1_BN2d_1280_one_layer, ..._2l_1x1_BN2d_256_1280, ..._2l_1x1_init_flim_256_1280, "
            "..._3x3_BN2d_1280_one_layer_init_flim, ..._next_layers_init_flim_<type>."
        ),
    )
    # ── FLIM backbone initialization ───────────────────────────────────────
    parser.add_argument(
        "--flim-init",
        action="store_true",
        default=False,
        help=(
            "Initialize backbone encoder with pre-trained FLIM weights instead of "
            "trunc_normal. For protozoan uses ch24_32_48 (PROTOZOAN_FLIM_ARCH), "
            "others use ch24_32_48 from _ARCH_BASE_FLIM. "
            "Run names get a '_flim_init' suffix."
        ),
    )
    # ── DataLoader / normalization passthrough ─────────────────────────────
    parser.add_argument(
        "--num-workers", type=int, default=4, metavar="N",
        help="DataLoader workers forwarded to each training subprocess (--num-workers).",
    )
    parser.add_argument(
        "--no-imagenet-norm", action="store_true", default=False,
        help=(
            "Forward --no-imagenet-norm to each subprocess: disables the ImageNet RGB "
            "Normalize on ift_lab inputs (correct for FLIM init — input stays LAB[0,1])."
        ),
    )
    # ── Run naming ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--run-prefix", type=str, default="", metavar="PREFIX",
        help=(
            "Prepended to every run_name (and therefore to the W&B display name and "
            "the local artifacts/distillation/<run_name>/ path). Use this to mark a "
            "batch as a distinct new set of experiments in W&B, e.g. "
            "--run-prefix 'cosine_v1_' -> 'cosine_v1_distillation_eggs_split1_pct1_...'."
        ),
    )

    args = parser.parse_args()

    global _current_log_level
    _current_log_level = _LOG_LEVELS[args.log_level]

    t_start  = time.time()
    gpu_ids  = list(range(args.num_gpus))

    # ── Slots por GPU: --gpu-slots sobrescreve --max-concurrent-per-gpu ──────
    if args.gpu_slots:
        raw = [s.strip() for s in args.gpu_slots.split(",")]
        if len(raw) != len(gpu_ids):
            raise ValueError(
                f"--gpu-slots tem {len(raw)} valores mas --num-gpus={len(gpu_ids)}. "
                f"Devem ser iguais. Ex: --num-gpus 2 --gpu-slots 7,3"
            )
        slots_per_gpu: Optional[dict[int, int]] = {gid: int(s) for gid, s in enumerate(raw)}
        _log(f"[PER-GPU SLOTS] " + ", ".join(f"GPU{gid}={s}" for gid, s in slots_per_gpu.items()))
    else:
        slots_per_gpu = None

    if args.retry:
        _log("[RETRY] Mode active — will re-run failed/missing experiments.")
    if args.skip_existing:
        wandb_note = " + W&B cross-check" if args.check_wandb else ""
        _log(f"[SKIP-EXISTING] Checking local metadata{wandb_note} before queuing.")

    ok, reason = validate_output_dir()
    if not ok:
        _log(f"Output directory is not writable: {reason}", "WARN")

    if args.flim_init:
        _log("[FLIM-INIT] Backbone will be initialized with pre-trained FLIM weights.")

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
        skip_existing=args.skip_existing,
        check_wandb=args.check_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        proj_type=args.proj_type,
        use_flim_init=args.flim_init,
        num_workers=args.num_workers,
        no_imagenet_norm=args.no_imagenet_norm,
        run_prefix=args.run_prefix,
    )

    n_skip_existing = sum(1 for s in skipped if s.get("status") == "skipped_existing")
    _log(
        f"Grid: {len(experiments)} to run, {n_skip_existing} skipped-existing, "
        f"{len(skipped) - n_skip_existing} skipped-invalid. "
        f"Max possible: "
        f"{len(args.datasets)}×{len(args.splits)}×{len(args.percentages)}"
        f"×{len(args.distillation_types)} = "
        f"{len(args.datasets)*len(args.splits)*len(args.percentages)*len(args.distillation_types)} runs."
    )

    if args.dry_run:
        _print_dry_run(
            experiments, skipped, gpu_ids,
            args.max_concurrent_per_gpu, args.cpus_per_experiment,
            retry=args.retry, slots_per_gpu=slots_per_gpu,
        )
        _write_manifest([], skipped, retry=args.retry)
        return

    if not experiments:
        _log("No valid experiments to run. Check data paths and grid parameters.", "WARN")
        _write_manifest([], skipped, retry=args.retry)
        return

    rows = run_queue(
        experiments=experiments,
        gpu_ids=gpu_ids,
        max_concurrent_per_gpu=args.max_concurrent_per_gpu,
        cpus_per_experiment=args.cpus_per_experiment,
        fail_fast=args.fail_fast,
        ray_address=args.ray_address,
        slots_per_gpu=slots_per_gpu,
    )

    _write_manifest(rows, skipped, retry=args.retry)
    _print_summary(rows, skipped, t_start, gpu_ids, args.max_concurrent_per_gpu,
                   retry=args.retry, slots_per_gpu=slots_per_gpu)


if __name__ == "__main__":
    main()
