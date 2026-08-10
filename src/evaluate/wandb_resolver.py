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

"""wandb_resolver.py — W&B metadata resolution for SSL and fine-tune runs.

Queries the W&B project and returns structured metadata objects for:
  - SSL pretrained encoder runs (used for SVM evaluation)
  - MLP fine-tune runs (X_finetune_ naming convention) for MLP inference

The resolution strategy follows SDD Section 5.3 and 5.4:
  1. Explicit structured config field  (freeze_encoder in run.config)
  2. wandb tags                         (run.tags)
  3. Run name pattern                   (X_finetune_* / heuristic)
  4. Fallback → skip with reason logged
"""
from __future__ import annotations

import glob as _glob
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml as _yaml

import wandb

from src.evaluate.constants import DATASET_NUM_CLASSES
from src.utils.evaluate import (
    find_best_checkpoint,
    get_local_run_ids,
    parse_experiment_name,
    resolve_available_experiments,
)
from src.utils.get_names_wandb import ENTITY, PROJECT

# ── Dataset name mappings ──────────────────────────────────────────────────────

DATASET_FULL_TO_SHORT: dict[str, str] = {
    "helminth-eggs":   "eggs",
    "helminth-larvae": "larvae",
    "protozoan-cysts": "protozoan",
    "parasito":        "parasito",
}

DATASET_SHORT_TO_FULL: dict[str, str] = {v: k for k, v in DATASET_FULL_TO_SHORT.items()}

# ── Regex patterns ─────────────────────────────────────────────────────────────

# X_finetune_lejepa_line_{dataset}_split_{N}_model_{init}[_{run_id}]
_FINETUNE_NAME_RE = re.compile(
    r"^X_finetune_lejepa_line_([a-z\-]+)_split_(\d+)_model_(xavier|random|he|flim|trunc_normal)(?:_([a-z0-9]+))?$"
)


# ── Shared W&B fetch (one round-trip for both SSL and fine-tune) ───────────────

# Module-level cache: (entity, project) → list of W&B run objects
_RUNS_CACHE: dict[tuple[str, str], list] = {}

_WANDB_FILTER = {
    "$or": [
        {"display_name": {"$regex": "^lejepa_line_"}},
        {"display_name": {"$regex": "^X_finetune_"}},
        {"display_name": {"$regex": "^ray_freeze_"}},
        {"display_name": {"$regex": "^ray_unfreeze_"}},
    ]
}


def _fetch_project_runs(
    entity: str = ENTITY,
    project: str = PROJECT,
    per_page: int = 500,
) -> list:
    """Fetch all relevant W&B runs once and cache the result.

    Uses a server-side filter so only ``lejepa_line_*`` (SSL encoder) and
    ``X_finetune_*`` (fine-tune) runs are transferred.  ``per_page=500``
    reduces the number of paginated HTTP requests to typically 1–2 for most
    project sizes.

    Subsequent calls with the same (entity, project) return the cached list
    immediately — no extra network round-trip.
    """
    key = (entity, project)
    if key in _RUNS_CACHE:
        return _RUNS_CACHE[key]

    api = wandb.Api()
    runs = list(api.runs(
        f"{entity}/{project}",
        filters=_WANDB_FILTER,
        per_page=per_page,
    ))
    _RUNS_CACHE[key] = runs
    return runs


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SSLRunInfo:
    """Metadata for an SSL pretrained encoder run with a local checkpoint."""
    run_id: str
    run_name: str
    dataset_name: str       # full name, e.g. "helminth-eggs"
    dataset_short: str      # short name, e.g. "eggs"
    split_id: int
    pct: int
    init: str
    ckpt_path: str
    num_classes: int


@dataclass
class FinetuneRunInfo:
    """Metadata for an MLP fine-tune W&B run."""
    wandb_run_id: str       # W&B run ID of the fine-tune run
    wandb_run_name: str     # e.g. X_finetune_lejepa_line_helminth-eggs_split_1_model_flim
    encoder_run_id: str     # W&B run ID of the SSL pretrained encoder
    encoder_run_name: str   # canonical SSL run name
    dataset_name: str       # full name
    dataset_short: str      # short name
    split_id: int
    pct: int
    init: str
    freeze: bool
    weights_path: str       # local path to model_best.pth
    num_classes: int
    skip_reason: Optional[str] = None  # set when this run should be skipped


# ── YAML-based fallback for offline SSL runs ───────────────────────────────────

def _yaml_ssl_fallback(
    already_known: set[str],
    local_ids: set[str],
) -> dict[str, str]:
    """Return {run_id: experiment_name} for encoder runs found in MLP YAML configs
    that have local checkpoints but are NOT in W&B (e.g. trained without logging).

    Scans configs/evaluate/mlp/freeze/**/*.yaml — only the 'freeze' split is needed
    since each encoder run_id appears in both freeze and unfreeze with the same
    experiment_name.
    """
    config_root = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "evaluate", "mlp", "freeze")
    config_root = os.path.normpath(config_root)
    if not os.path.isdir(config_root):
        return {}

    fallback: dict[str, str] = {}
    for yaml_path in _glob.glob(os.path.join(config_root, "**", "*.yaml"), recursive=True):
        try:
            with open(yaml_path) as fh:
                cfg = _yaml.safe_load(fh)
        except Exception:
            continue
        run_id = cfg.get("run_id", "")
        exp_name = cfg.get("experiment_name", "")
        if run_id and exp_name and run_id not in already_known and run_id in local_ids:
            fallback[run_id] = exp_name

    return fallback


# ── SSL run resolution ─────────────────────────────────────────────────────────

def resolve_ssl_runs(
    dataset_filter: Optional[str] = None,
    pct_filter: Optional[int] = None,
    init_filter: Optional[str] = None,
    split_filter: Optional[int] = None,
    entity: str = ENTITY,
    project: str = PROJECT,
    verbose: bool = True,
) -> list[SSLRunInfo]:
    """Return SSL pretrained encoder runs that have local checkpoints.

    Wraps ``resolve_available_experiments()`` from src.utils.evaluate and
    applies optional filters before returning structured SSLRunInfo objects.

    Args:
        dataset_filter: Short dataset name ("eggs", "protozoan", "larvae") or None.
        pct_filter:     Pretraining percentage or None.
        init_filter:    Initialization type or None.
        split_filter:   Split index or None.
        entity:         W&B entity.
        project:        W&B project.
        verbose:        Print progress messages.

    Returns:
        List of SSLRunInfo objects for usable runs.
    """
    if verbose:
        print("[wandb_resolver] Resolving SSL encoder runs (W&B ∩ local checkpoints)...")

    # Use shared fetch (server-side filtered, cached) instead of a separate full pull
    all_runs = _fetch_project_runs(entity, project)
    local_ids = get_local_run_ids()

    # Build {run_id: run_name} for canonical SSL runs that have local checkpoints
    available: dict[str, str] = {
        r.id: r.name
        for r in all_runs
        if r.name and r.name.startswith("lejepa_line_") and r.id in local_ids
    }

    if verbose:
        n_wandb = sum(1 for r in all_runs if r.name and r.name.startswith("lejepa_line_"))
        print(
            f"[wandb_resolver] SSL — W&B: {n_wandb} canonical runs, "
            f"{len(available)} with local checkpoints"
        )

    # Fallback: include runs that have local checkpoints + YAML MLP configs but no W&B entry
    yaml_extra = _yaml_ssl_fallback(already_known=set(available.keys()), local_ids=local_ids)
    if yaml_extra:
        if verbose:
            print(f"[wandb_resolver] SSL — YAML fallback: +{len(yaml_extra)} offline run(s) added")
        available.update(yaml_extra)

    results: list[SSLRunInfo] = []
    n_skipped = 0

    for run_id, run_name in sorted(available.items()):
        try:
            info = parse_experiment_name(run_name)
        except ValueError:
            if verbose:
                print(f"  [SKIP] Cannot parse: {run_name!r}")
            n_skipped += 1
            continue

        dataset_short = DATASET_FULL_TO_SHORT.get(info.dataset_name, "")
        if not dataset_short:
            if verbose:
                print(f"  [SKIP] Unknown dataset {info.dataset_name!r} in {run_name}")
            n_skipped += 1
            continue

        # Apply optional filters
        if dataset_filter is not None and dataset_short != dataset_filter:
            continue
        if pct_filter is not None and info.percentage != pct_filter:
            continue
        if init_filter is not None and info.initialization_type != init_filter:
            continue
        if split_filter is not None and info.split_id != split_filter:
            continue

        try:
            ckpt_path = find_best_checkpoint(run_id)
        except FileNotFoundError:
            if verbose:
                print(f"  [SKIP] No checkpoint for {run_id} ({run_name})")
            n_skipped += 1
            continue

        num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)

        results.append(SSLRunInfo(
            run_id=run_id,
            run_name=run_name,
            dataset_name=info.dataset_name,
            dataset_short=dataset_short,
            split_id=info.split_id,
            pct=info.percentage,
            init=info.initialization_type,
            ckpt_path=ckpt_path,
            num_classes=num_classes,
        ))

    if verbose:
        print(f"[wandb_resolver] SSL runs: {len(results)} usable, {n_skipped} skipped")

    return results


# ── Fine-tune run resolution ───────────────────────────────────────────────────

def resolve_finetune_runs(
    dataset_filter: Optional[str] = None,
    freeze_filter: Optional[bool] = None,
    pct_filter: Optional[int] = None,
    init_filter: Optional[str] = None,
    split_filter: Optional[int] = None,
    entity: str = ENTITY,
    project: str = PROJECT,
    verbose: bool = True,
) -> list[FinetuneRunInfo]:
    """Return fine-tune runs from W&B matching the ``X_finetune_`` naming convention.

    Prioritises runs matching ``X_finetune_*`` (SDD 5.3) and skips the rest.
    Freeze / unfreeze detection follows the priority order in SDD 5.4:
      1. Explicit config field ``freeze_encoder``
      2. W&B tags
      3. Run name pattern
      4. Skip with reason logged

    Checks for local fine-tuned weights at::
        results/mlp_weights/{freeze|unfreeze}/{encoder_run_id}/model_best.pth

    Args:
        dataset_filter: Short dataset name or None.
        freeze_filter:  True=freeze only, False=unfreeze only, None=both.
        pct_filter:     Percentage filter or None.
        init_filter:    Initializer filter or None.
        split_filter:   Split filter or None.
        entity:         W&B entity.
        project:        W&B project.
        verbose:        Print progress messages.

    Returns:
        List of FinetuneRunInfo objects.  Runs that cannot be resolved
        have ``skip_reason`` set (non-None).
    """
    from src.utils.evaluate import _ROOT  # noqa: PLC0415

    if verbose:
        print("[wandb_resolver] Resolving fine-tune runs from W&B (X_finetune_ filter)...")

    try:
        # Shared fetch — hits network only once per (entity, project) pair
        all_runs = _fetch_project_runs(entity, project)
    except Exception as exc:
        print(f"[wandb_resolver] ERROR: cannot reach W&B — {exc}")
        return []

    # Partition: finetune runs — X_finetune_ (new) and ray_freeze_/ray_unfreeze_ (legacy)
    ft_runs = [
        r for r in all_runs
        if r.name and (
            r.name.startswith("X_finetune_")
            or r.name.startswith("ray_freeze_")
            or r.name.startswith("ray_unfreeze_")
        )
    ]

    if verbose:
        print(f"[wandb_resolver] Found {len(ft_runs)} finetune run(s) in W&B")

    # Build SSL run lookup {ssl_run_id → parsed ExperimentInfo} from cached W&B data.
    # Used as authoritative fallback for pct/split/init/dataset when finetune config is sparse.
    _ssl_meta: dict[str, object] = {}
    for _r in all_runs:
        if _r.name and _r.name.startswith("lejepa_line_"):
            try:
                _ssl_meta[_r.id] = parse_experiment_name(_r.name)
            except ValueError:
                pass

    results: list[FinetuneRunInfo] = []
    # Dedup key: (encoder_run_id, freeze, dataset_short, split_id, pct, init)
    # Prevents duplicate entries when multiple W&B runs map to the same encoder+mode.
    _seen: set[tuple] = set()

    _mlp_root = os.path.join(_ROOT, "results", "mlp_weights")

    for run in ft_runs:
        run_name: str = run.name
        run_id: str = run.id
        cfg: dict = dict(run.config) if run.config else {}

        # ── Extract all metadata first (needed for freeze fallback + entry creation) ──
        encoder_run_id: str = cfg.get("wandb_run_id", "")
        encoder_run_name: str = cfg.get("experiment_name", "")
        dataset_name: str = cfg.get("dataset_name", "")
        # YAML uses "split"; W&B config may serialise as "split" or "split_id"
        split_id: Optional[int] = cfg.get("split_id") or cfg.get("split") or None
        pct: Optional[int] = cfg.get("percentage", None)
        init: str = cfg.get("initialization_type", "")

        # Name regex as fallback for fields absent from config
        m = _FINETUNE_NAME_RE.match(run_name)
        if m:
            if not dataset_name:
                dataset_name = m.group(1)
            if split_id is None:
                split_id = int(m.group(2))
            if not init:
                init = m.group(3)
        elif verbose and (not dataset_name or split_id is None or not init):
            # Regex didn't match (e.g. name has _pct_{N}_ component) — fields
            # will be resolved via config or encoder_run_name fallbacks.
            missing = [f for f, v in [("dataset", dataset_name), ("split", split_id), ("init", init)] if not v]
            print(f"  [DEBUG] {run_name}: regex no match, resolving {missing} via fallbacks")

        # Parse encoder run name as secondary fallback for pct/split/init
        if (pct is None or split_id is None or not init) and encoder_run_name:
            try:
                _enc_info = parse_experiment_name(encoder_run_name)
                if pct is None:
                    pct = _enc_info.percentage
                if split_id is None:
                    split_id = _enc_info.split_id
                if not init:
                    init = _enc_info.initialization_type
                if not dataset_name:
                    dataset_name = _enc_info.dataset_name
            except ValueError:
                pass

        # SSL W&B lookup: authoritative fallback using the encoder run_id from config
        # or the name suffix. Covers runs where finetune config was not fully logged.
        _enc_id_for_lookup = encoder_run_id or run_name.rsplit("_", 1)[-1]
        if (pct is None or split_id is None or not init or not dataset_name) and _enc_id_for_lookup:
            _ssl_info = _ssl_meta.get(_enc_id_for_lookup)
            if _ssl_info is not None:
                if pct is None:
                    pct = _ssl_info.percentage
                if split_id is None:
                    split_id = _ssl_info.split_id
                if not init:
                    init = _ssl_info.initialization_type
                if not dataset_name:
                    dataset_name = _ssl_info.dataset_name

        dataset_short = DATASET_FULL_TO_SHORT.get(dataset_name, "")
        num_classes = DATASET_NUM_CLASSES.get(dataset_name, 9)

        # Candidate encoder run_id for filesystem lookup (already computed above).
        _enc_id = _enc_id_for_lookup

        # ── Freeze / unfreeze identification (SDD 5.4 priority order) ────────

        # 1. Explicit config field (handle bool, int, and str serialisations)
        freeze: Optional[bool] = None
        if "freeze_encoder" in cfg:
            raw = cfg["freeze_encoder"]
            if isinstance(raw, bool):
                freeze = raw
            elif isinstance(raw, int):
                freeze = bool(raw)
            elif isinstance(raw, str):
                freeze = raw.lower() in ("true", "1", "yes")

        # 2. W&B tags
        if freeze is None and run.tags:
            tags_lower = [t.lower() for t in run.tags]
            if "freeze" in tags_lower and "unfreeze" not in tags_lower:
                freeze = True
            elif "unfreeze" in tags_lower:
                freeze = False

        # 3. Run name pattern
        if freeze is None:
            name_lower = run_name.lower()
            if "unfreeze" in name_lower:
                freeze = False
            elif "freeze" in name_lower:
                freeze = True

        # 4. Filesystem fallback — infer from which weight directories exist locally
        if freeze is None and _enc_id:
            _has_freeze = os.path.exists(
                os.path.join(_mlp_root, "freeze", _enc_id, "model_best.pth"))
            _has_unfreeze = os.path.exists(
                os.path.join(_mlp_root, "unfreeze", _enc_id, "model_best.pth"))

            if _has_freeze and _has_unfreeze:
                # Both modes present — emit one complete entry per mode, then skip
                # the normal single-entry flow below.
                if pct is None or not dataset_short or split_id is None:
                    if verbose:
                        print(
                            f"  [SKIP] {run_name}: both modes found but metadata incomplete "
                            f"(dataset={dataset_short!r} pct={pct} split={split_id})"
                        )
                    continue
                if verbose:
                    print(f"  [INFO] {run_name}: both modes inferred from local weights")
                for _fval, _mdir in ((True, "freeze"), (False, "unfreeze")):
                    if freeze_filter is not None and freeze_filter != _fval:
                        continue
                    if dataset_filter is not None and dataset_short != dataset_filter:
                        continue
                    if pct_filter is not None and pct != pct_filter:
                        continue
                    if init_filter is not None and init != init_filter:
                        continue
                    if (split_filter is not None and split_id is not None
                            and split_id != split_filter):
                        continue
                    _key = (_enc_id, _fval, dataset_short, split_id, pct, init)
                    if _key in _seen:
                        continue
                    _seen.add(_key)
                    results.append(FinetuneRunInfo(
                        wandb_run_id=run_id, wandb_run_name=run_name,
                        encoder_run_id=_enc_id, encoder_run_name=encoder_run_name,
                        dataset_name=dataset_name, dataset_short=dataset_short,
                        split_id=split_id, pct=pct, init=init,
                        freeze=_fval,
                        weights_path=os.path.join(_mlp_root, _mdir, _enc_id, "model_best.pth"),
                        num_classes=num_classes,
                        skip_reason=None,
                    ))
                continue  # handled — skip normal single-entry flow

            elif _has_freeze:
                freeze = True
                if verbose:
                    print(f"  [INFO] {run_name}: freeze inferred from local weights")
            elif _has_unfreeze:
                freeze = False
                if verbose:
                    print(f"  [INFO] {run_name}: unfreeze inferred from local weights")

        # 5. Skip — cannot determine mode by any means
        if freeze is None:
            if verbose:
                print(f"  [SKIP] {run_name}: cannot determine freeze/unfreeze mode")
            results.append(FinetuneRunInfo(
                wandb_run_id=run_id, wandb_run_name=run_name,
                encoder_run_id="", encoder_run_name="",
                dataset_name="", dataset_short="",
                split_id=0, pct=0, init="", freeze=False,
                weights_path="", num_classes=0,
                skip_reason="cannot_determine_freeze_mode",
            ))
            continue

        # ── Per-run resolution summary ────────────────────────────────────────
        if verbose:
            mode_str = "freeze" if freeze else ("unfreeze" if freeze is False else "?")
            print(
                f"  [RESOLVED] {run_name[:60]}\n"
                f"             enc={_enc_id or '?'}  ds={dataset_short or '?'}  "
                f"split={split_id}  pct={pct}  init={init or '?'}  mode={mode_str}"
            )
            if not encoder_run_id:
                print(f"  [WARN] {run_name}: encoder_run_id missing from config — using name suffix")
            if split_id is None:
                print(f"  [WARN] {run_name}: split_id unresolved — will default to 0")

        # ── Apply freeze filter ────────────────────────────────────────────────
        if freeze_filter is not None and freeze != freeze_filter:
            continue

        # ── Dataset validation ────────────────────────────────────────────────
        if not dataset_short:
            if verbose:
                print(f"  [SKIP] {run_name}: unknown dataset {dataset_name!r}")
            results.append(FinetuneRunInfo(
                wandb_run_id=run_id, wandb_run_name=run_name,
                encoder_run_id=encoder_run_id, encoder_run_name=encoder_run_name,
                dataset_name=dataset_name, dataset_short="",
                split_id=split_id or 0, pct=pct or 0, init=init, freeze=freeze,
                weights_path="", num_classes=0,
                skip_reason=f"unknown_dataset:{dataset_name}",
            ))
            continue

        # ── Pct validation ────────────────────────────────────────────────────
        if pct is None:
            if verbose:
                print(f"  [SKIP] {run_name}: cannot resolve pretraining percentage")
            results.append(FinetuneRunInfo(
                wandb_run_id=run_id, wandb_run_name=run_name,
                encoder_run_id=encoder_run_id, encoder_run_name=encoder_run_name,
                dataset_name=dataset_name, dataset_short=dataset_short,
                split_id=split_id or 0, pct=0, init=init, freeze=freeze,
                weights_path="", num_classes=0,
                skip_reason="cannot_resolve_pct",
            ))
            continue

        # ── Apply optional filters ────────────────────────────────────────────
        if dataset_filter is not None and dataset_short != dataset_filter:
            continue
        if pct_filter is not None and pct != pct_filter:
            continue
        if init_filter is not None and init != init_filter:
            continue
        if split_filter is not None and split_id is not None and split_id != split_filter:
            continue

        # ── Local weights check ───────────────────────────────────────────────
        mode_dir = "freeze" if freeze else "unfreeze"
        weights_path = os.path.join(_mlp_root, mode_dir, _enc_id, "model_best.pth")

        if not os.path.exists(weights_path):
            if verbose:
                print(
                    f"  [SKIP] {run_name}: missing weights at "
                    f"results/mlp_weights/{mode_dir}/{_enc_id}/model_best.pth"
                )
            results.append(FinetuneRunInfo(
                wandb_run_id=run_id, wandb_run_name=run_name,
                encoder_run_id=_enc_id, encoder_run_name=encoder_run_name,
                dataset_name=dataset_name, dataset_short=dataset_short,
                split_id=split_id or 0, pct=pct, init=init, freeze=freeze,
                weights_path=weights_path, num_classes=num_classes,
                skip_reason="missing_weights",
            ))
            continue

        _key = (_enc_id, freeze, dataset_short, split_id, pct, init)
        if _key in _seen:
            continue
        _seen.add(_key)
        results.append(FinetuneRunInfo(
            wandb_run_id=run_id, wandb_run_name=run_name,
            encoder_run_id=_enc_id, encoder_run_name=encoder_run_name,
            dataset_name=dataset_name, dataset_short=dataset_short,
            split_id=split_id or 0, pct=pct, init=init, freeze=freeze,
            weights_path=weights_path, num_classes=num_classes,
            skip_reason=None,
        ))

    usable = [r for r in results if r.skip_reason is None]
    skipped = [r for r in results if r.skip_reason is not None]

    if verbose:
        print(
            f"[wandb_resolver] Fine-tune runs: {len(usable)} usable, "
            f"{len(skipped)} skipped (see above for reasons)"
        )

    return results
