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
Ray queue for the unsupervised AutoEncoder grid (Experiment Day 5), two-stage protocol.

The grid is 3 datasets x {5%, 75%} x 3 splits = **18 cells**, and each cell is trained
twice — stage 1 then stage 2 — so the experiment is **36 runs**, one subprocess each
(``python -m src.modules.autoencoder_flim_module``):

    stage 1  ``--freeze-encoder``   FLIM encoder frozen, the ResNet decoder learns to
                                    invert it. Checkpoint selected on val/recon_loss
                                    (min) -> ``best_recon.ckpt``.
    stage 2  ``--init-ckpt <s1>``   both halves unfrozen, warm-started from stage 1.
                                    Checkpoint selected on val/svm_kappa (max) ->
                                    ``best_kappa.ckpt``.

The 18 cells do **not** collapse to 9 stage-1 runs shared by both percentages: the
``percentage`` argument selects a different ``data_descriptor_perc{p}.json``
(``config.py:193-201``) that repartitions train **and** validation — validation is the
complement of train, so pct5 and pct75 disagree on both halves. A stage-1 checkpoint is
therefore only valid for the percentage it was fit and model-selected on. 18 cells,
36 runs.

Same queue semantics as ``classification_flim_ray.py``: per-GPU concurrency slots, greedy
drain, ``--skip-existing`` / ``--check-wandb`` resume, manifest, ``--dry-run``.

**Input normalisation — the default changed.** The FLIM kernels were estimated on LAB
images in [0, 1], so feeding the encoder an ImageNet-normalised input evaluates those
kernels off the distribution they were built for; the report measures ~0.33 of kappa lost
that way (``docs/relatorio_kappa_estagio1_estagio2.md`` §8.1). This queue therefore runs
**LAB [0, 1] by default** and passes ``--no-imagenet-norm`` to every child explicitly, so
the flag is visible in the logged command line. ``--imagenet-norm`` opts back in and
reproduces the legacy behaviour. This is a *local* decision for the FLIM-init autoencoder
grid only: the distillation and I-JEPA arms still **require** ImageNet normalisation,
because their teacher was trained with it, and nothing here should be copied there.

Because the two normalisations are different experiments that must not share a name, the
LAB default appends a ``_lab`` marker to the run name (see ``_run_name``). Without it,
``--skip-existing`` would match the 36 legacy ImageNet-normalised runs already on disk and
the grid would exit in seconds looking like a success.

    # both stages chained inside one queue entry per cell — 18 entries, 36 runs
    # (LAB [0, 1] input; run names end in ``_lab``)
    python scripts/autoencoder_flim_ray.py --stage both \\
        --num-gpus 4 --max-concurrent-per-gpu 1 \\
        --wandb-update --skip-existing --check-wandb

    # or as two explicit waves (wave 2 only queues cells whose stage-1 ckpt exists)
    python scripts/autoencoder_flim_ray.py --stage 1 --num-gpus 4 --wandb-update
    python scripts/autoencoder_flim_ray.py --stage 2 --num-gpus 4 --wandb-update

    # legacy arm: ImageNet Normalize on the LAB input, legacy run names (no ``_lab``)
    python scripts/autoencoder_flim_ray.py --stage both --imagenet-norm \\
        --num-gpus 4 --wandb-update

    # see the plan (run names + the exact child command lines) without running anything
    python scripts/autoencoder_flim_ray.py --stage both --dry-run

    # re-queue only what failed
    python scripts/autoencoder_flim_ray.py --stage both --retry \\
        --skip-existing --check-wandb --wandb-update

Every child's stderr is persisted to ``<run_dir>/child_stderr.log`` **always**, not only
when the subprocess fails, and any warning line in it is echoed into this queue's log even
on a clean exit — that is how the truncated-SVM ``ConvergenceWarning`` stayed invisible for
36 runs (§8.8).

W&B note: the child module now prefixes its stage-1/stage-2 metric keys with the stage
(``stage1/…`` / ``stage2/…``), so dashboard panels built on the old flat keys must be
repointed.

``--stage both`` is the cheaper schedule: the two waves impose a global barrier (every
cell waits on the slowest stage-1 run of the whole grid) that ``both`` removes, because
each cell's stage 2 depends only on its own stage 1.

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
import shlex
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
    "run_name", "dataset", "split", "percentage", "stage", "architecture",
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


_STAGE_SUFFIX: dict[int, str] = {1: "stage1_frozen", 2: "stage2_fine_tune"}

# Stage 1 selects its checkpoint on val/recon_loss (its encoder never moves, so the kappa
# curve is fit noise); stage 2 selects on val/svm_kappa.
_STAGE_CKPT: dict[int, str] = {1: "best_recon.ckpt", 2: "best_kappa.ckpt"}


# The input normalisation is part of the experiment's identity, not a detail: the 36 runs
# already on disk were trained with ImageNet Normalize and carry the bare name. If the LAB
# default reused that name, `--skip-existing` (and `--check-wandb`, which matches on the
# same string) would report the whole grid as done without training anything. The `_lab`
# marker is what keeps the two arms apart on disk and on the dashboard.
_LAB_MARKER = "_lab"


def _run_name(dataset: str, split: int, pct: int, stage: int = 1, run_prefix: str = "",
              imagenet_norm: bool = True) -> str:
    """Run name for a cell. ``imagenet_norm=True`` reproduces the legacy name exactly."""
    return (f"{run_prefix}ae_resnet_flim_{dataset}_split{split}_pct{pct}"
            f"_{_STAGE_SUFFIX[stage]}{'' if imagenet_norm else _LAB_MARKER}")


def _run_dir(run_name: str) -> str:
    return os.path.join(_ARTIFACT_ROOT, run_name)


def _stage_ckpt(dataset: str, split: int, pct: int, stage: int, run_prefix: str = "",
                imagenet_norm: bool = True) -> str:
    """Checkpoint a given stage writes — this is what the next wave consumes."""
    return os.path.join(
        _run_dir(_run_name(dataset, split, pct, stage, run_prefix, imagenet_norm)),
        "checkpoints", _STAGE_CKPT[stage],
    )


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


# ── Child command builder ──────────────────────────────────────────────────────

def _child_cmd_tail(exp: dict, stage: int, run_name: str, init_ckpt: str) -> list[str]:
    """Argv for one child run, *without* the interpreter (the worker prepends its own).

    Built on the driver and carried in the stage plan so that ``--dry-run`` prints the
    literal argv the worker will execute — there is no second, hand-mirrored copy of this
    command to drift out of sync.
    """
    cmd = [
        "-m", "src.modules.autoencoder_flim_module",
        "--dataset", exp["dataset"],
        "--split", str(exp["split"]),
        "--percentage", str(exp["pct"]),
        "--arch-json", exp["arch_json"],
        "--flim-weights-path", exp["flim_weights_path"],
        "--recon-loss", "bce_logits",
        "--run-name", run_name,
        "--max-epochs", str(exp["max_epochs"]),
        "--warmup-epochs", str(exp["warmup_epochs"]),
        "--patience", str(exp["patience"]),
        "--batch-size", str(exp["batch_size"]),
        "--num-workers", str(exp.get("num_workers", 4)),
        "--svm-probe-every", str(exp["svm_probe_every"]),
        "--log-recon-every", str(exp["log_recon_every"]),
    ]

    if stage == 1:
        cmd.append("--freeze-encoder")
    else:
        cmd += ["--init-ckpt", init_ckpt]

    # Passed in both directions rather than relying on the child's default: the flag is
    # the whole point of this arm, so it belongs in the logged command line where anyone
    # reading a queue log can see which normalisation produced the numbers.
    cmd.append("--imagenet-norm" if exp.get("imagenet_norm") else "--no-imagenet-norm")

    if exp.get("wandb_update"):
        cmd += ["--wandb",
                "--wandb-project", exp["wandb_project"],
                "--wandb-entity", exp["wandb_entity"]]
    return cmd


# ── Experiment grid builder ────────────────────────────────────────────────────

def build_experiment_grid(
    datasets: list[str], splits: list[int], pcts: list[int],
    num_workers: int, max_epochs: int, warmup_epochs: int, patience: int, batch_size: int,
    svm_probe_every: int, log_recon_every: int,
    imagenet_norm: bool, wandb_update: bool, wandb_project: str,
    run_prefix: str, skip_existing: bool, check_wandb: bool, wandb_entity: str,
    stage: str = "1",
) -> tuple[list[dict], list[dict]]:
    experiments: list[dict] = []
    skipped: list[dict] = []

    stages = [1, 2] if str(stage) == "both" else [int(stage)]

    for dataset in datasets:
        for pct in pcts:
            for split in splits:
                key = f"{dataset}/split{split}/pct{pct}"

                # The data on disk is the same for both stages, so validate the cell once.
                ok, reason = validate_experiment(dataset, split, pct)
                if not ok:
                    _log(f"SKIP (preflight) {key}: {reason}", "WARN")
                    skipped.append({"key": key, "stage": stages[0],
                                    "run_name": _run_name(dataset, split, pct, stages[0],
                                                          run_prefix, imagenet_norm),
                                    "dataset": dataset, "split": split, "pct": pct,
                                    "reason": f"preflight:{reason}"})
                    continue

                # Everything the child needs that does not depend on the stage. Built once
                # per cell so the stage plan and the experiment entry cannot disagree.
                common = {
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
                    "patience": patience,
                    "batch_size": batch_size,
                    "svm_probe_every": svm_probe_every,
                    "log_recon_every": log_recon_every,
                    "imagenet_norm": imagenet_norm,
                    "wandb_update": wandb_update,
                    "wandb_project": wandb_project,
                    "wandb_entity": wandb_entity,
                }

                # One queue entry carries the whole stage chain for this cell. Chaining
                # inside a single Ray task is what removes the barrier between the two
                # waves: cell A can already be fine-tuning while cell B is still frozen.
                # Running `--stage 1` and then `--stage 2` as separate invocations would
                # make every cell wait on the slowest stage-1 run of the entire grid.
                plan: list[dict] = []
                for st in stages:
                    run_name = _run_name(dataset, split, pct, st, run_prefix, imagenet_norm)
                    init_ckpt = (_stage_ckpt(dataset, split, pct, 1, run_prefix, imagenet_norm)
                                 if st == 2 else "")

                    # Only a stage-2-only invocation can check the checkpoint now. In
                    # `both` mode stage 1 has not run yet, so the file is expected to be
                    # missing at build time — the Ray task re-checks it after stage 1.
                    if len(stages) == 1 and st == 2 and not os.path.isfile(init_ckpt):
                        _log(f"SKIP (no stage-1 ckpt) {key}: {init_ckpt}", "WARN")
                        skipped.append({"key": key, "stage": st, "run_name": run_name,
                                        "dataset": dataset, "split": split, "pct": pct,
                                        "reason": f"missing stage-1 ckpt:{init_ckpt}"})
                        continue

                    # Resume is per stage: a finished stage 1 must not force its stage 2
                    # to be re-run, nor the other way round.
                    if skip_existing:
                        should_skip, why = _should_skip_experiment(
                            run_name, check_wandb, wandb_entity, wandb_project)
                        if should_skip:
                            _log(f"SKIP (done) {key} stage{st}: {why}")
                            skipped.append({"key": key, "stage": st, "run_name": run_name,
                                            "dataset": dataset, "split": split, "pct": pct,
                                            "reason": why})
                            continue

                    plan.append({
                        "stage": st, "run_name": run_name, "init_ckpt": init_ckpt,
                        "run_dir": _run_dir(run_name),
                        "cmd_tail": _child_cmd_tail(common, st, run_name, init_ckpt),
                    })

                if not plan:
                    continue

                experiments.append({
                    # A single-stage entry says which stage it is; a chained one does not,
                    # because it covers both.
                    "key": key if len(plan) > 1 else f"{key}/stage{plan[0]['stage']}",
                    "stage_plan": plan,
                    "run_name": plan[0]["run_name"],
                    "stage": plan[0]["stage"],
                    "init_ckpt": plan[0]["init_ckpt"],
                    **common,
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
    from collections import deque as _deque

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _os.environ["OMP_NUM_THREADS"] = str(cpus)
    _os.environ["WANDB_CONSOLE"] = "off"
    _os.environ["WANDB_MODE"] = "online"

    # One queue entry = one cell = one or two sequential runs on the same GPU slot.
    plan = exp.get("stage_plan") or [{"stage": exp.get("stage", 1),
                                      "run_name": exp["run_name"],
                                      "init_ckpt": exp.get("init_ckpt", ""),
                                      "run_dir": exp.get("run_dir", ""),
                                      "cmd_tail": exp.get("cmd_tail", [])}]

    result: dict = {
        "key": exp["key"], "run_name": exp["run_name"], "gpu_id": gpu_id,
        "queue_index": queue_index, "dataset": exp["dataset"],
        "split": exp["split"], "pct": exp["pct"], "architecture": exp["architecture"],
        "stage": plan[0]["stage"], "runs": [],
    }

    if not _os.path.isfile(exp["arch_json"]):
        result["status"] = "error"
        result["returncode"] = -1
        result["error"] = f"PREFLIGHT FAIL — arch JSON not found: {exp['arch_json']}"
        return result

    for step in plan:
        run_row: dict = {"stage": step["stage"], "run_name": step["run_name"]}

        # The argv was built on the driver (`_child_cmd_tail`) and travels inside the plan,
        # so the worker never re-derives it and `--dry-run` shows the real thing.
        if not step.get("cmd_tail") or not step.get("run_dir"):
            run_row["status"] = "error"
            run_row["returncode"] = -1
            run_row["error"] = "malformed stage plan: missing cmd_tail/run_dir"
            result["runs"].append(run_row)
            break

        # In `both` mode this path only exists once stage 1 has written it, which is why
        # it is checked here and not at build time.
        if step["stage"] == 2 and not _os.path.isfile(step["init_ckpt"]):
            run_row["status"] = "error"
            run_row["returncode"] = -1
            run_row["error"] = f"stage-1 checkpoint not found: {step['init_ckpt']}"
            result["runs"].append(run_row)
            break

        cmd = [_sys.executable] + list(step["cmd_tail"])

        # stderr is written to disk on *every* run, not only on failure. The previous
        # version piped it and threw it away on returncode 0, which is how 36 runs' worth
        # of sklearn ConvergenceWarnings — the truncated SVM probe behind the kappa
        # numbers — were emitted and never seen. stdout stays inherited, as before: Ray
        # forwards it to the driver.
        _os.makedirs(step["run_dir"], exist_ok=True)
        log_path = _os.path.join(step["run_dir"], "child_stderr.log")
        run_row["stderr_log"] = log_path

        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as log_fh:
                proc = _sp.run(cmd, cwd=project_root, stdout=None, stderr=log_fh, text=True)
            run_row["returncode"] = proc.returncode

            # Re-read the file rather than buffering the stream: only the last 40 lines and
            # at most 10 distinct warning messages are ever held in memory, whatever the
            # child printed.
            tail = _deque(maxlen=40)
            warn_hits = 0
            distinct: list[str] = []
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    # "Warning" subsumes ConvergenceWarning / RuntimeWarning / UserWarning.
                    if "Warning" in line:
                        warn_hits += 1
                        msg = line.strip()
                        if msg not in distinct and len(distinct) < 10:
                            distinct.append(msg)
                    tail.append(line)

            if warn_hits:
                # Surfaced even on a clean exit: a ConvergenceWarning here means the kappa
                # this run reported came out of a solver that never converged.
                run_row["warnings"] = warn_hits
                print(f"[queue][{step['run_name']}] {warn_hits} warning line(s) in child "
                      f"stderr -> {log_path}", flush=True)
                for msg in distinct:
                    print(f"[queue][{step['run_name']}]   {msg}", flush=True)

            if proc.returncode == 0:
                run_row["status"] = "ok"
            else:
                run_row["status"] = "error"
                run_row["error"] = (f"returncode={proc.returncode} (full stderr: {log_path})\n"
                                    + "".join(tail).strip())
        except Exception as exc:
            run_row["status"] = "error"
            run_row["returncode"] = -1
            run_row["error"] = str(exc)

        result["runs"].append(run_row)
        # No point fine-tuning from a stage that died — it left no checkpoint to start from.
        if run_row["status"] != "ok":
            break

    failed = [r for r in result["runs"] if r["status"] != "ok"]
    result["status"] = "ok" if (not failed and len(result["runs"]) == len(plan)) else "error"
    result["returncode"] = failed[0]["returncode"] if failed else 0
    if failed:
        result["error"] = f"stage {failed[0]['stage']}: {failed[0].get('error', '')}"

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
            # No "runs" key here — the task died before reporting any. Carry the entry's
            # first stage so the manifest row is not silently labelled stage 1.
            result = {"key": exp["key"], "gpu_id": gpu_id, "queue_index": idx,
                      "status": "ray_error", "error": str(exc), "run_name": exp["run_name"],
                      "dataset": exp["dataset"], "split": exp["split"], "pct": exp["pct"],
                      "architecture": exp["architecture"], "stage": exp.get("stage", 1)}

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

    # One row per *run*, not per queue entry: a chained cell produced two W&B runs and two
    # checkpoints, and the manifest is what downstream evaluation joins against.
    for r in rows:
        steps = r.get("runs") or [{"stage": r.get("stage", 1),
                                   "run_name": r.get("run_name", r.get("key", "")),
                                   "status": r.get("status", "")}]
        for step in steps:
            stage = step.get("stage", 1)
            run_name = step.get("run_name", "")
            ckpt = os.path.join(_run_dir(run_name), "checkpoints", _STAGE_CKPT[stage])
            records.append({
                "run_name": run_name,
                "dataset": r.get("dataset", ""),
                "split": r.get("split", ""),
                "percentage": r.get("pct", ""),
                "stage": stage,
                "architecture": r.get("architecture", ""),
                "status": step.get("status", r.get("status", "")),
                "skip_reason": "",
                "checkpoint_path": ckpt if os.path.isfile(ckpt) else "",
            })

    for s in skipped:
        run_name = s.get("run_name", "")
        stage = s.get("stage", 1)
        ckpt = os.path.join(_run_dir(run_name), "checkpoints", _STAGE_CKPT[stage])
        records.append({
            "run_name": run_name,
            "dataset": s.get("dataset", ""),
            "split": s.get("split", ""),
            "percentage": s.get("pct", ""),
            "stage": stage,
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
            # For a chained cell the number that matters is the last stage's — stage 1's
            # kappa is a frozen-encoder probe, not the result of the protocol.
            last = (r.get("runs") or [{"run_name": r.get("run_name", "")}])[-1]
            print(f"    {r['key']:<34} {_best_kappa(last.get('run_name', ''))}")

    if bad:
        print("\n  Failures:")
        for r in bad:
            err = (r.get("error", "") or "").splitlines()
            print(f"    {r.get('key', '?'):<34} {err[0] if err else r.get('status', '')}")
    print("=" * 78 + "\n")


def _print_dry_run(experiments: list[dict], skipped: list[dict],
                   gpu_ids: list[int], max_per_gpu: int, cpus_per: int,
                   imagenet_norm: bool = False, retry: bool = False) -> None:
    total_slots = max_per_gpu * len(gpu_ids)
    print("\n" + "=" * 78)
    print(f"  DRY-RUN {'(retry) ' if retry else ''}— nothing will be executed")
    print("=" * 78)
    print(f"  GPUs           : {gpu_ids}  ({max_per_gpu} concurrent/GPU "
          f"= {total_slots} slots, {cpus_per} CPU each)")
    print(f"  artifacts      : {_ARTIFACT_ROOT}")
    print(f"  input norm     : {'imagenet' if imagenet_norm else 'lab[0,1] (default)'}")
    print(f"  run-name form  : ...{_STAGE_SUFFIX[1]}{'' if imagenet_norm else _LAB_MARKER}")
    print(f"  to run         : {len(experiments)}")
    print(f"  skipped        : {len(skipped)}")
    if experiments:
        print("\n  Queue:")
        for i, e in enumerate(experiments):
            plan = e.get("stage_plan") or [{"stage": e.get("stage", 1)}]
            chain = "->".join(f"s{s['stage']}" for s in plan)
            print(f"    [{i + 1:>3}] {e['key']:<34} {chain:<7} {e['architecture']}  "
                  f"{e['num_classes']}cls  {e['max_epochs']}ep")
            for s in plan:
                exists = " [DIR EXISTS]" if os.path.isdir(_run_dir(s["run_name"])) else ""
                print(f"          {s['run_name']}{exists}")
        # The literal argv the worker will run — same list, not a hand-written mirror.
        print("\n  Child command lines (first queue entry, one per stage):")
        for s in experiments[0].get("stage_plan", []):
            argv = " ".join(shlex.quote(a) for a in s.get("cmd_tail", []))
            print(f"    [s{s['stage']}] python {argv}")
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
    parser.add_argument("--stage", choices=["1", "2", "both"], required=True,
                        help="1 = frozen FLIM encoder, decoder learns to invert it. "
                             "2 = both unfrozen, starting from the stage-1 checkpoint "
                             "(skips cells whose stage-1 ckpt is absent). "
                             "both = chain 1 then 2 per cell in one queue entry, so no "
                             "cell waits on any other cell's stage 1.")
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

    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50, metavar="N",
                        help="EarlyStopping patience on val/svm_kappa, passed to each run.")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, metavar="N")
    parser.add_argument("--svm-probe-every", type=int, default=1,
                        help="Run the one-vs-one SVM probe every N validation epochs.")
    parser.add_argument("--log-recon-every", type=int, default=10,
                        help="Log the LAB->RGB reconstruction preview every N epochs.")
    parser.add_argument("--imagenet-norm", action="store_true", default=False,
                        help="Opt back in to ImageNet Normalize on the LAB input. OFF by "
                             "default: the FLIM kernels were estimated on LAB in [0, 1] and "
                             "normalising costs ~0.33 of kappa here (report §8.1). Runs "
                             "with the default carry a '_lab' suffix in the run name, so "
                             "they never collide with the legacy ImageNet-normalised runs. "
                             "The distillation / I-JEPA arms still require normalisation — "
                             "this default is local to this grid.")

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
        warmup_epochs=args.warmup_epochs, patience=args.patience, batch_size=args.batch_size,
        svm_probe_every=args.svm_probe_every, log_recon_every=args.log_recon_every,
        imagenet_norm=args.imagenet_norm, wandb_update=args.wandb_update,
        wandb_project=args.wandb_project, run_prefix=args.run_prefix,
        skip_existing=skip_existing, check_wandb=args.check_wandb,
        wandb_entity=args.wandb_entity, stage=args.stage,
    )

    if args.dry_run:
        _print_dry_run(experiments, skipped, gpu_ids, args.max_concurrent_per_gpu,
                       args.cpus_per_experiment, imagenet_norm=args.imagenet_norm,
                       retry=args.retry)
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
