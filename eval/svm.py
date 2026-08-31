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

"""svm.py — o SVM do projeto: features do encoder, fit e a sonda de linha de base.

Este arquivo funde duas origens e vira o LAR UNICO do protocolo de SVM:

* `src/utils/evaluate.py` (todo o corpo, linhas 59-615) — resolucao de runs,
  parsing de nome de experimento, busca de checkpoint, `_encode_pooled`,
  `fit_svm`, `train_svm` e os tres extratores de feature. Aquele arquivo nao
  tinha casa na arvore nova; o comentario `# ponytail:` de
  `methods/autoencoder/autoencoder_flim_module.py:119` ja apontava para ca.
* `src/evaluate/svm.py` (o `main()`, linha 58) — a sonda em si, agora
  `run_svm_probe()`, com defaults nomeados no lugar do argparse.

REGRA PERMANENTE: existe UM `fit_svm` e UMA `compute_metrics`. `compute_metrics`
mora em `core/metrics.py` e e importada aqui; `fit_svm` mora aqui e e importada
por todo o resto de `eval/`. Nenhum dos dois e reimplementado em lugar nenhum.

`LejepaLineModule` e `wandb` sao importados DENTRO de `run_svm_probe`: sao os
unicos simbolos pesados do arquivo e `methods/autoencoder/autoencoder_flim_module.py`
importa `_encode_pooled` daqui — deixa-los no topo faria `methods` puxar
`methods` de volta por tabela.
"""
from __future__ import annotations

import glob
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.constants import DATASET_NUM_CLASSES, IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from core.wandb import ENTITY, PROJECT, cached_history

_ROOT = PROJECT_ROOT

_RESULTS_DIR = os.path.join(_ROOT, "results")


# ─── CLI minima ───────────────────────────────────────────────────────────────


def cli_kwargs(argv: list[str]) -> dict:
    """Traduz `--chave valor`, `--chave=valor` e `--flag` em kwargs.

    Substitui o argparse dos entrypoints de `eval/` SEM virar um segundo
    framework de CLI: nao ha parser, registry, decorator nem classe base — so
    esta funcao, e ela e a unica do pacote. O contrato e o que
    `experiments/ray/runners/eval.py:build_cmd` ja monta hoje:
    `python -m eval.mlp --config <path> --ckpt-selection=best [--wandb]`.

    Digito puro vira `int`: os parsers de origem declaravam `type=int` em
    `--split`, `--pct` e `--max-runs` (src/evaluate/unified_eval.py:672-685) e
    sem isso a comparacao com o eixo da grade falharia calada.

    Teto conhecido: um valor que comece com `-` (numero negativo) precisa da
    forma `--chave=-1`, porque a forma com espaco o leria como outra flag — e
    nesse caso ele fica string, nao int.
    """
    out: dict = {}
    i = 0
    while i < len(argv):
        key, sep, value = argv[i].lstrip("-").partition("=")
        if not sep:
            # Varios valores seguidos viram lista: e o `nargs="+"` dos parsers de
            # origem (src/evaluate/eval_growth_stages.py:332-346).
            values: list[str] = []
            while i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 1
                values.append(argv[i])
            if not values:
                value = True
            else:
                value = values[0] if len(values) == 1 else values
        if isinstance(value, list):
            value = [int(v) if v.isdigit() else v for v in value]
        elif isinstance(value, str) and value.isdigit():
            value = int(value)
        out[key.replace("-", "_")] = value
        i += 1
    return out

# Checkpoint search order: newest experiments first, legacy fallback second.
_LOGS_DIRS = [
    os.path.join(_ROOT, "logs", "flim-ssl"),
    #os.path.join(_ROOT, "logs", "flim-ssl_old"),
]


# ─── Available-experiment resolution ──────────────────────────────────────────


def get_local_run_ids() -> set[str]:
    """Return run IDs that have at least one .ckpt file under any logs dir."""
    local_ids: set[str] = set()
    for logs_dir in _LOGS_DIRS:
        if not os.path.isdir(logs_dir):
            continue
        for entry in os.scandir(logs_dir):
            if not entry.is_dir():
                continue
            ckpt_dir = os.path.join(entry.path, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue
            if glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True):
                local_ids.add(entry.name)
    return local_ids


def resolve_available_experiments(update_wandb: bool = False) -> dict[str, str]:
    """Return {run_id: run_name} for all W&B runs that also have local weights.

    By default loads W&B metadata from the local cache
    (``configs/wandb_update/ids_wandb.json``).  Pass ``update_wandb=True``
    to refresh the cache from the live W&B API first.

    Args:
        update_wandb: When ``True``, fetch fresh metadata from W&B and
                      overwrite the local cache before resolving.
    """
    print("[FILTER] Fetching W&B run list...")
    # Fetch without deduplication so we can prefer run_ids that have local
    # checkpoints when the same experiment name has multiple W&B runs.
    #
    # Origem: src/utils/evaluate.py:99-113 abria o cache DUAS vezes — uma por
    # `get_runs_dict_cached` (nomes) e outra por `load_cache` (created_at).
    # `core.wandb.cached_history` ja devolve `{run_id: {"name", "created_at",
    # ...}}`, entao a segunda leitura de disco sai; nenhum numero muda.
    _meta: dict[str, dict] = cached_history(
        ENTITY, PROJECT, deduplicate=False, update=update_wandb
    )
    wandb_runs_all = {rid: info["name"] for rid, info in _meta.items()}

    local_ids = get_local_run_ids()

    # Deduplicate: for each unique run name keep the run_id that has a local
    # checkpoint.  If none (or multiple) have a checkpoint, fall back to the
    # most-recently-created one.

    best: dict[str, tuple[str, str, bool]] = {}  # name → (run_id, created_at, has_local)
    for rid, name in wandb_runs_all.items():
        created_at = _meta.get(rid, {}).get("created_at", "")
        has_local = rid in local_ids
        if name not in best:
            best[name] = (rid, created_at, has_local)
        else:
            prev_rid, prev_ts, prev_has = best[name]
            # Prefer run_ids with local checkpoint; break ties by newest created_at
            if (has_local and not prev_has) or (has_local == prev_has and created_at > prev_ts):
                best[name] = (rid, created_at, has_local)

    wandb_runs = {rid: name for name, (rid, _, _) in best.items()}

    available = {rid: name for rid, name in wandb_runs.items() if rid in local_ids}
    missing_local = {rid: name for rid, name in wandb_runs.items() if rid not in local_ids}

    print(f"[FILTER] W&B runs total        : {len(wandb_runs_all)}")
    print(f"[FILTER] After dedup           : {len(wandb_runs)}")
    print(f"[FILTER] With local weights    : {len(available)}")
    if missing_local:
        print(f"[FILTER] Skipping — no ckpt   : {len(missing_local)} run(s)")
        for rid in sorted(missing_local):
            print(f"           skip  {rid}  ({missing_local[rid]})")

    return available

# ─── Configuration ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Experiment name parsing ──────────────────────────────────────────────────

_CANONICAL_RE = re.compile(
    r"^lejepa_line_([a-z\-]+)_split_(\d+)_pct_(\d+)_model_"
    r"(xavier|random|he|flim|trunc_normal)"
    # Sufixo opcional de variante (batch size / config de treino), documentado no
    # README como `[_bs<batch>]` — ex.: `_bs256`, `_bs256_mc2g8l`, `_bs150_mc8g8l_3lproj`.
    r"(?:_(.+))?$"
)
_OLD_RE = re.compile(r"^line_p(\d+)_(xavier|random|he|flim|trunc_normal)$")


@dataclass
class ExperimentInfo:
    dataset_name: str
    split_id: int
    percentage: int
    initialization_type: str
    path_dataset: Optional[str]  # None → legacy combined-parasito mode


def parse_experiment_name(run_name: str) -> ExperimentInfo:
    """Parse dataset, split, percentage, and init from a run name.

    Supports:
    - Canonical: ``lejepa_line_{dataset}_split_{N}_pct_{pct}_model_{init}``
    - Legacy:    ``line_p{pct}_{init}``  (→ parasito combined, split=1)
    """
    m = _CANONICAL_RE.match(run_name)
    if m:
        dataset = m.group(1)
        return ExperimentInfo(
            dataset_name=dataset,
            split_id=int(m.group(2)),
            percentage=int(m.group(3)),
            initialization_type=m.group(4),
            path_dataset=dataset,
        )
    m = _OLD_RE.match(run_name)
    if m:
        return ExperimentInfo(
            dataset_name="parasito",
            split_id=1,
            percentage=int(m.group(1)),
            initialization_type=m.group(2),
            path_dataset=None,
        )
    raise ValueError(f"Cannot parse experiment name: '{run_name}'")


# ─── Checkpoint helpers ────────────────────────────────────────────────────────


def find_best_checkpoint(run_id: str) -> str:
    """Return the lowest-loss checkpoint for *run_id*.

    Searches ``logs/flim-ssl`` first, then ``logs/flim-ssl_old``.
    Within each run, picks the best checkpoint: the constant ``best.ckpt``
    written by the current config, or — for legacy runs — the ``best-*``
    checkpoint with the lowest loss value encoded in the filename; falls back
    to ``last.ckpt`` if no best exists.
    """
    for logs_dir in _LOGS_DIRS:
        ckpt_dir = os.path.join(logs_dir, run_id, "checkpoints")
        if not os.path.isdir(ckpt_dir):
            continue
        # New layout: a single constant best.ckpt directly in checkpoints/.
        candidates = glob.glob(os.path.join(ckpt_dir, "best.ckpt"))
        # Legacy layout: best-<epoch=…-loss=…>/<file>.ckpt subdirectories.
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, "best-*", "*.ckpt"))
        if not candidates:
            candidates = glob.glob(os.path.join(ckpt_dir, "last.ckpt"))
        if candidates:
            def _loss(p: str) -> float:
                hit = re.search(r"loss=([0-9.]+)\.ckpt$", p)
                return float(hit.group(1)) if hit else float("inf")
            return min(candidates, key=_loss)
    raise FileNotFoundError(
        f"No checkpoint found for run_id={run_id!r} "
        f"in: {[os.path.relpath(d, _ROOT) for d in _LOGS_DIRS]}"
    )


# ─── Feature pooling ──────────────────────────────────────────────────────────

# Global average pooling applied to the conv3 feature map before flattening.
# Matches the convention used by the classification heads (``MLPHead.forward``,
# ``TwoLayerSigmoidHead``) and by ``src/evaluate/svm_classification_flim.py``,
# so the SVM sees the same ``channels[-1]``-d embedding the heads see instead of
# a spatially flattened map.

# Como o mapa conv3 é reduzido: ``"avgpool2d"`` (GAP, ``channels[-1]``-d) ou
# ``"flatten"`` (mapa raw achatado, ``C*H*W``-d — o protocolo original).  É só o
# default: quem quer o outro modo passa ``embed_mode=`` / ``mode=`` na chamada.
DEFAULT_EMBED_MODE = "avgpool2d"

# Alias mantido para os chamadores que ainda rebindam o módulo
# (``src.utils.evaluate.EMBED_MODE = ...``) para trocar o default de um processo
# inteiro.  Teto: rebindar não é thread-safe e não viaja para workers em
# processos separados — o parâmetro explícito não tem nenhum dos dois problemas.
EMBED_MODE = DEFAULT_EMBED_MODE

EMBED_MODES = ("avgpool2d", "flatten")


def _encode_pooled(model, inputs: torch.Tensor, mode: str | None = None) -> torch.Tensor:
    """Percorre conv1→…→conv``n_layers``, faz GAP e achata para ``[B, C]``.

    Args:
        model:  Encoder com ``.conv1``…``.conv{n}``, onde ``n`` vem de
                ``model.n_layers`` (default 3) — "conv3" não é mais o último.
        inputs: Batch ``[B, C, H, W]`` já no device do modelo.
        mode:   ``"avgpool2d"`` (default) ou ``"flatten"``; ``None`` usa
                ``EMBED_MODE`` (o default do módulo).  Com ``"flatten"`` o
                pooling é pulado e o mapa conv3 raw é achatado direto,
                devolvendo ``[B, C*H*W]``.
    """
    mode = mode or EMBED_MODE
    if mode not in EMBED_MODES:
        raise ValueError(
            f"embed mode inválido: {mode!r}; use um de {EMBED_MODES}"
        )
    out = inputs
    for n in range(1, getattr(model, "n_layers", 3) + 1):
        out = getattr(model, f"conv{n}")(out)
    if mode == "flatten":
        return out.flatten(start_dim=1).detach().cpu()
    return F.adaptive_avg_pool2d(out, 1).flatten(start_dim=1).detach().cpu()


# ─── SVM solver diagnostics ───────────────────────────────────────────────────

# Values written by every SVM arm when the fit itself never happened (the cell
# raised before ``fit``), so error rows keep the same columns as ok rows.
SVM_DIAG_MISSING: dict = {
    "svm_protocol":    "",
    "svm_fit_status":  -1,
    "svm_n_iter_max":  -1,
    "svm_n_iter_sum":  -1,
    "svm_n_sv":        -1,
    "svm_fit_seconds": float("nan"),
}


def _svc_step(estimator):
    """Return the ``SVC`` inside *estimator* — itself, or the step of a Pipeline.

    ``fit_status_`` / ``n_iter_`` / ``support_`` live on the ``SVC``, never on
    the wrapping ``Pipeline``.
    """
    steps = getattr(estimator, "named_steps", None)
    if steps is None:
        return estimator
    for step in steps.values():
        if isinstance(step, svm.SVC):
            return step
    return estimator


def svm_protocol(estimator) -> str:
    """One-line, auditable description of the SVM protocol of *estimator*.

    Derived from the estimator itself (never hardcoded), so the string cannot
    drift from the code.  Recorded in every result CSV so two experiment arms
    can be checked for pairing without reading the source — the scaler is
    deliberately NOT unified across arms, only reported.
    """
    svc   = _svc_step(estimator)
    steps = getattr(estimator, "named_steps", None) or {}
    scaler = "standard" if any(isinstance(s, StandardScaler) for s in steps.values()) else "none"
    return (
        f"scaler={scaler};kernel={svc.kernel};C={svc.C:g};"
        f"{svc.decision_function_shape};max_iter={svc.max_iter}"
    )


def fit_svm_with_diagnostics(estimator, X, y, tag: str = "SVM") -> dict:
    """Fit *estimator* on ``(X, y)`` and record how the libsvm solver terminated.

    Persisting this is the whole point: ``fit_status_=1`` (solver hit the
    iteration cap) went undetected for months because nothing in the repo ever
    saved it.  A non-zero status is logged as a WARNING.

    The returned dict is also stashed on the estimator as ``fit_diagnostics_``
    so callers that only receive the fitted classifier back can still write the
    columns without a signature change.

    Returns:
        dict with the ``svm_*`` columns (see ``SVM_DIAG_MISSING``).
    """
    _t0 = time.perf_counter()
    estimator.fit(X, y)
    elapsed = time.perf_counter() - _t0

    svc = _svc_step(estimator)
    n_iter = np.asarray(getattr(svc, "n_iter_", []))
    diag = {
        "svm_protocol":    svm_protocol(estimator),
        "svm_fit_status":  int(getattr(svc, "fit_status_", -1)),
        "svm_n_iter_max":  int(n_iter.max()) if n_iter.size else -1,
        "svm_n_iter_sum":  int(n_iter.sum()) if n_iter.size else -1,
        "svm_n_sv":        int(len(getattr(svc, "support_", ()))),
        "svm_fit_seconds": round(elapsed, 3),
    }

    msg = (
        f"[{tag}] {diag['svm_protocol']}  fit_status={diag['svm_fit_status']}  "
        f"n_iter(max/sum)={diag['svm_n_iter_max']}/{diag['svm_n_iter_sum']}  "
        f"n_sv={diag['svm_n_sv']}  fit={diag['svm_fit_seconds']:.1f}s"
    )
    if diag["svm_fit_status"] != 0:
        logging.warning("%s  <- SOLVER DID NOT CONVERGE (fit_status != 0)", msg)
        print(f"  [WARN] {msg}  <- SOLVER DID NOT CONVERGE")
    else:
        print(f"  {msg}")

    estimator.fit_diagnostics_ = diag
    return diag


# ─── SVM training ─────────────────────────────────────────────────────────────


def fit_svm(X, y, *, max_iter: int = -1, C: float = 1e2, scaler: bool = False,
            tag: str = "SVM"):
    """Build the repo's canonical linear SVM, fit it on ``(X, y)`` and return it.

    Every SVM in this repository uses exactly these hyperparameters, so they
    live here once instead of being retyped per call site.  ``degree`` and
    ``coef0`` are inert under ``kernel="linear"`` but are pinned anyway so the
    estimator repr matches the historical CSVs.

    *scaler* is the ONLY knob beyond the solver cap and ``C``: two arms (the
    1280-d projection ones) genuinely need ``StandardScaler`` to converge.  That
    asymmetry is deliberate — it is recorded per row by ``svm_protocol`` and
    must NOT be unified across arms.

    This function takes an already-built feature matrix; it does not extract
    features and it does not touch label indexing.  The ``+1`` / 0-indexed
    convention stays the caller's business, on purpose.

    Args:
        X:        Feature matrix ``(N, D)``.
        y:        Label vector ``(N,)`` — whatever indexing the caller chose.
        max_iter: Solver cap; ``-1`` = unbounded (default).
        C:        Regularisation parameter.
        scaler:   Wrap the SVC in ``Pipeline([StandardScaler, SVC])``.
        tag:      Prefix for the diagnostics line printed by
                  ``fit_svm_with_diagnostics``.

    Returns:
        The fitted estimator (``SVC`` or ``Pipeline``); solver diagnostics are
        attached as ``fit_diagnostics_``.
    """
    clf = svm.SVC(
        # Unbounded solver: results are deliberately NOT comparable with the
        # CSVs produced under the old max_iter cap.
        max_iter=max_iter,
        C=C,
        degree=3,
        gamma="auto",
        coef0=0,
        decision_function_shape="ovo",
        kernel="linear",
    )
    estimator = Pipeline([("scaler", StandardScaler()), ("svm", clf)]) if scaler else clf

    # Barra de progresso viva durante o fit: libsvm não reporta nada e um fit
    # longo parece travado.  Fica aqui para não ser reescrita em cada arm.
    _stop = threading.Event()

    def _progress():
        with tqdm(desc="SVM fit", unit="s", bar_format="{desc}: {elapsed} [{postfix}]") as pbar:
            while not _stop.wait(1.0):
                pbar.update(1)
            pbar.set_postfix_str("done")

    _thread = threading.Thread(target=_progress, daemon=True)
    _thread.start()
    try:
        fit_svm_with_diagnostics(estimator, X, y, tag=tag)
    finally:
        _stop.set()
        _thread.join()
    return estimator


@torch.no_grad()
def train_svm(model, dataloader, max_iter: int = -1, C: float = 1e2, degree: int = 3,
              embed_mode: str | None = None):
    """Fit a linear SVM on frozen encoder features.

    Features are the conv3 feature map reduced by ``AdaptiveAvgPool2d(1)`` and
    flattened, i.e. one value per output channel (``channels[-1]``-d).  Pass
    ``embed_mode="flatten"`` for the raw ``C*H*W``-d map instead; ``None``
    (default) uses the module default ``EMBED_MODE``.

    *dataloader* must yield ``(inputs, one_hot_labels)`` where one_hot_labels
    has shape ``(B, num_classes)``.  Labels returned by the fitted SVM are
    1-indexed (matching the ``+1`` applied here).

    ``max_iter`` defaults to ``-1`` (unbounded solver); pass the old cap
    explicitly only to reproduce a historical CSV.  Solver diagnostics are left
    on the returned classifier as ``fit_diagnostics_``.
    """
    print("[INFO] Initializing SVM")
    # ponytail: *degree* continua na assinatura só por compatibilidade — o
    # kernel é linear, então libsvm ignora o grau, ``fit_svm`` fixa degree=3 e
    # nenhum call site passa outro valor.  Teto: se algum dia alguém quiser
    # kernel polinomial, isto vira um parâmetro de verdade em ``fit_svm``.
    del degree

    model.eval()
    model.to(DEVICE)

    all_feats = torch.Tensor([])
    all_y = torch.Tensor([]).long()

    print("[INFO] Preparing data for SVM")
    for inputs, labels in tqdm(dataloader, desc="  Extracting train features"):
        inputs = inputs.to(DEVICE)
        all_feats = torch.cat((all_feats, _encode_pooled(model, inputs, embed_mode)))

        labels_np = np.argmax(labels.cpu().numpy(), axis=1) + 1  # 1-indexed
        all_y = torch.cat((all_y, torch.from_numpy(labels_np).long()))

    return fit_svm(all_feats, all_y, max_iter=max_iter, C=C, tag="SVM_FLIM")


# ─── One-hot label wrapper ─────────────────────────────────────────────────────


class _OneHotDataset(Dataset):
    """Wraps DatasetParasite and converts integer labels to one-hot vectors.

    Required because ``train_svm`` expects ``(B, num_classes)`` labels and
    applies ``np.argmax(..., axis=1)`` to recover class indices.
    """

    def __init__(self, base: Dataset, num_classes: int) -> None:
        self.base = base
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        one_hot = F.one_hot(torch.tensor(label), self.num_classes).float()
        return img, one_hot


# ─── Feature extraction ────────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(model, dataloader, embed_mode: str | None = None):
    """Extract conv1→conv2→conv3 features from *model* for every batch.

    The conv3 feature map is reduced by ``AdaptiveAvgPool2d(1)`` and flattened,
    so each sample yields one value per output channel (``channels[-1]``-d) —
    the same embedding the classification heads consume.

    Args:
        model:      FLIM Encoder with ``.conv1``, ``.conv2``, ``.conv3``.
        dataloader: Yields ``(inputs, int_labels)``.
        embed_mode: ``"avgpool2d"`` or ``"flatten"``; ``None`` (default) uses
                    the module default ``EMBED_MODE``.

    Returns:
        features: ``np.ndarray`` of shape ``(N, channels[-1])`` — pooled conv3
                  (``(N, C*H*W)`` under ``embed_mode="flatten"``).
        y_true:   ``np.ndarray`` of shape ``(N,)``  — 0-indexed class labels.
    """
    model.eval()
    model.to(DEVICE)

    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting test features"):
        inputs = inputs.to(DEVICE)
        all_feats.append(_encode_pooled(model, inputs, embed_mode).numpy())

        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


@torch.no_grad()
def extract_features_encode(model, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled embeddings via ``model.encode()`` for every batch.

    Serves every model that exposes the ``encode()`` interface (the distilled
    students, among others) — nothing here is distillation-specific.  The
    embedding returned is the pooled 1-D vector that training optimises.

    The file's other extractor, ``extract_features`` (above), serves the
    conv1→conv2→conv3 family instead: it reaches into the spatial feature maps
    and pools them itself.  The two cannot be merged because the model
    interfaces genuinely differ — one has ``.encode()``, the other has
    ``.conv1``/``.conv2``/``.conv3``.

    Args:
        model:      Any model with an ``encode()`` method.
        dataloader: Yields ``(inputs, int_labels)``.

    Returns:
        features: ``np.ndarray`` of shape ``(N, embed_dim)``.
        y_true:   ``np.ndarray`` of shape ``(N,)`` — 0-indexed class labels.
    """
    model.eval()
    model.to(DEVICE)

    all_feats:  list[np.ndarray] = []
    all_labels: list[int]        = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting features"):
        feats = model.encode(inputs.to(DEVICE)).detach().cpu().numpy()
        all_feats.append(feats)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


@torch.no_grad()
def extract_proj_features(student, proj_kd, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``proj_kd(student.encoder(x))`` embeddings over *dataloader*.

    A projeção treinável fica ativa, ao contrário de ``extract_features_encode``
    (que usa ``student.encode`` e devolve o embedding raw do encoder).

    Args:
        student:    ``LeJEPAFLIMModel`` (usa-se ``.encoder``, não ``.encode``).
        proj_kd:    Cabeça de projeção da destilação.
        dataloader: Yields ``(inputs, int_labels)``.

    Returns:
        features: ``np.ndarray`` of shape ``(N, proj_dim)``.
        y_true:   ``np.ndarray`` of shape ``(N,)`` — 0-indexed class labels.
    """
    # ponytail: as duas cópias originais só divergiam no texto da barra do tqdm
    # (``leave=False`` em uma delas); adotei uma descrição só.  Teto: se algum
    # dia o texto da barra importar para alguém, vira parâmetro.
    student.eval()
    proj_kd.eval()
    student.to(DEVICE)
    proj_kd.to(DEVICE)

    all_feats:  list[np.ndarray] = []
    all_labels: list[int]        = []

    for inputs, labels in tqdm(dataloader, desc="  Extracting proj features"):
        emb = proj_kd(student.encoder(inputs.to(DEVICE))).detach().cpu().numpy()
        all_feats.append(emb)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)

    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)



# ─── Modo JOB: um YAML, um checkpoint, uma linha ──────────────────────────────
#
# O YAML e escrito pelo runner (`experiments/ray/runners/eval.py`) e ja traz o
# escopo resolvido — nada aqui varre disco nem consulta o W&B. Contrato fechado:
#
#   ckpt / run_name / dataset_name / dataset_short / split / percentage /
#   method / source / stage_label (so em growth_stages) / results_csv
#
# O SVM e o de sempre (`fit_svm`) e a metrica e a de sempre (`compute_metrics`):
# este bloco so escolhe o encoder e o dataloader.

_JOB_FIELDS = ["method", "run_name", "source", "stage_label", "dataset_short",
               "dataset_name", "split", "percentage", "dim", "kappa", "acc",
               "f1", "ckpt_epoch", "ckpt", "status", "error"]

# Rodar o mesmo job duas vezes SOBRESCREVE a linha: quem colide nesta chave sai
# do CSV velho antes do concat.
_JOB_KEY = ["method", "run_name", "stage_label", "dataset_short", "split",
            "percentage"]


def _upsert_row(csv_path: str, row: dict) -> None:
    """Grava *row* em *csv_path* substituindo a linha de mesma `_JOB_KEY`.

    Mesmo padrao delete-then-concat de `src/evaluate/eval_growth_stages.py:315`
    (hoje `eval/growth_stages.py:_wandb_val`): `astype(str)` porque o CSV relido
    volta com int64 onde o DataFrame novo tem int nativo, e o `isin` de
    MultiIndex compara por tipo.

    O `fillna("")` e o que o padrao de origem nao precisava: la nenhuma coluna da
    chave vem vazia, aqui `stage_label` vem — e o CSV relido devolve NaN onde foi
    escrita string vazia, entao sem isso um job de destilacao nunca casaria
    consigo mesmo e o CSV so cresceria.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    new = pd.DataFrame([row], columns=_JOB_FIELDS)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        old = pd.read_csv(csv_path)
        key = pd.MultiIndex.from_frame(new[_JOB_KEY].fillna("").astype(str))
        old = old[~pd.MultiIndex.from_frame(
            old[_JOB_KEY].fillna("").astype(str)).isin(key)]
        new = pd.concat([old, new], ignore_index=True)[_JOB_FIELDS]
    new.to_csv(csv_path, index=False)


def _ckpt_epoch(ckpt_path: str) -> int:
    """A epoca gravada no checkpoint, sem trazer os tensores para a RAM.

    `mmap=True` porque os ckpts de destilacao tem 7,2 GB (o teacher I-JEPA de
    630M viaja dentro) e aqui so se le uma chave escalar. Um ckpt que nao abra
    assim vale -1: a epoca e metadado da linha, nao pode derrubar a avaliacao.
    """
    try:
        blob = torch.load(ckpt_path, map_location="cpu", mmap=True,
                          weights_only=False)
        return int(blob.get("epoch", -1))
    except Exception:
        return -1


def _run_svm_job(config: str) -> str:
    """Avalia UM checkpoint descrito por *config* e faz upsert de UMA linha.

    O rotulo do SVM e 0-indexed nas duas pontas (treino e teste vem do mesmo
    extrator), entao o par `+1`/`-1` do caminho legado nao aparece aqui — nao ha
    nada para desfazer, e `compute_metrics` ja quer 0-indexed.

    Args:
        config: Caminho do YAML do job — absoluto, ou relativo a raiz do repo.

    Returns:
        O `results_csv` do job.
    """
    import yaml  # noqa: PLC0415

    cfg_path = config if os.path.isabs(config) else os.path.join(_ROOT, config)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"--config: arquivo inexistente: {cfg_path}")
    with open(cfg_path) as fh:
        job = yaml.safe_load(fh) or {}

    ckpt = job["ckpt"]
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(_ROOT, ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"{cfg_path}: 'ckpt' inexistente em disco: {ckpt_path}"
        )

    source       = job["source"]
    dataset_name = job["dataset_name"]
    split        = int(job["split"])
    percentage   = int(job["percentage"])
    csv_path     = job["results_csv"]
    csv_path     = csv_path if os.path.isabs(csv_path) else os.path.join(_ROOT, csv_path)
    num_classes  = DATASET_NUM_CLASSES[dataset_name]

    row: dict = {
        "method":       job["method"],
        "run_name":     job["run_name"],
        "source":       source,
        # Ausente quando source=distillation: a chave do upsert e a mesma coluna
        # nos dois casos, entao vira string vazia em vez de sumir.
        "stage_label":  job.get("stage_label", ""),
        "dataset_short": job["dataset_short"],
        "dataset_name": dataset_name,
        "split":        split,
        "percentage":   percentage,
        "dim":          "", "kappa": "", "acc": "", "f1": "",
        "ckpt_epoch":   "",
        "ckpt":         ckpt,
        "status":       "ok", "error": "",
    }

    print(f"[JOB] {row['method']} | {row['run_name']} | {source} "
          f"| {dataset_name} split{split} pct{percentage}")
    print(f"[JOB] ckpt : {ckpt_path}")

    try:
        if source == "growth_stages":
            # Import tardio: `eval/growth_stages.py` importa deste modulo no topo
            # (`import eval.svm as _ev`), entao no topo daqui seria ciclo.
            from eval.growth_stages import _encoder  # noqa: PLC0415

            encoder, _head, epoch = _encoder({"ckpt": ckpt_path}, DEVICE)
            # Os dois contratos da grade SPiFiL, iguais aos de
            # `eval/growth_stages.py` (`_EMBED_MODE` e o `_loader`): flatten e
            # LAB[0,1], que e o que os bracos viram no treino.
            imagenet_norm = False
            def _feats(loader):
                return extract_features(encoder, loader, "flatten")
        elif source == "distillation":
            from eval.svm_variants.svm_distillation_conv import (  # noqa: PLC0415
                _load_student_from_ckpt,
            )

            student = _load_student_from_ckpt(ckpt_path, DEVICE)
            epoch = _ckpt_epoch(ckpt_path)
            imagenet_norm = True
            def _feats(loader):
                return extract_features_encode(student, loader)
        else:
            raise ValueError(
                f"{cfg_path}: source={source!r} — use 'distillation' ou "
                f"'growth_stages'"
            )

        row["ckpt_epoch"] = epoch
        transform = build_test(IMAGE_SIZE, imagenet_norm=imagenet_norm)

        def _loader(set_name: str) -> DataLoader:
            return DataLoader(
                ParasiteDataset(
                    set_name=set_name, split=split, percentage=percentage,
                    transform=transform, loader="ift_lab",
                    path_dataset=dataset_name,
                ),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )

        X_train, y_train = _feats(_loader("train"))
        clf = fit_svm(X_train, y_train, tag=row["method"])

        feats, y_true = _feats(_loader("test"))
        y_pred = clf.predict(feats)

        metrics = compute_metrics(y_true=y_true, y_pred=y_pred,
                                  num_classes=num_classes)
        row.update({"dim": int(feats.shape[1]), **metrics})
        print(f"  [RESULT] kappa={metrics['kappa']:.4f}  "
              f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}")
    except Exception as exc:
        # A linha de erro e escrita ANTES do re-raise: o CSV registra o que
        # falhou, e o processo ainda sai != 0 para o runner ver.
        row.update({"status": "error", "error": str(exc)})
        _upsert_row(csv_path, row)
        print(f"  [ERROR] {exc}")
        raise

    _upsert_row(csv_path, row)
    print(f"[DONE] {csv_path}")
    return csv_path


# ─── Sonda SVM ────────────────────────────────────────────────────────────────
#
# Origem: src/evaluate/svm.py:58 (`main()` com argparse). A unica flag do parser
# era `--wandb-update`; virou o parametro `wandb_update` com o mesmo default.


def run_svm_probe(wandb_update: bool = False, config: str | None = None,
                  ckpt_selection: str = "best") -> str:
    """Avalia com SVM todo encoder LeJEPA Line que tenha peso local e run no W&B.

    Com `config`, nada disso acontece: o YAML descreve UM checkpoint e a funcao
    delega para `_run_svm_job`, que escreve UMA linha no `results_csv` do job.
    Sem `config`, o comportamento e o de sempre — a varredura por cache do W&B.

    Args:
        wandb_update:   Atualiza o cache local de metadados do W&B
                        (`configs/wandb_update/ids_wandb.json`) antes de rodar.
                        Default `False` — usa o cache como esta, igual ao
                        `--wandb-update` do argparse de origem.
        config:         Caminho do YAML de job (`--config`). `None` (default) =
                        modo de varredura.
        ckpt_selection: Aceito so para `experiments/ray/runners/eval.py:build_cmd`
                        poder passar `--ckpt-selection=best` a todos os
                        avaliadores sem excecao. No modo JOB o checkpoint ja vem
                        resolvido no YAML, e no modo de varredura quem escolhe e
                        `find_best_checkpoint`: nos dois casos nao ha nada a
                        selecionar aqui.

    Returns:
        Caminho do CSV escrito — `results/svm_results.csv` no modo de varredura,
        o `results_csv` do job no modo JOB.
    """
    del ckpt_selection

    if config is not None:
        return _run_svm_job(config)

    # Import tardio de proposito: sao os dois simbolos pesados do arquivo e
    # `methods/autoencoder/autoencoder_flim_module.py:119` importa `_encode_pooled`
    # deste modulo — subi-los para o topo faria `methods` fechar ciclo consigo.
    import wandb  # noqa: PLC0415
    from methods.lejepa.lejepa_line_module import LejepaLineModule  # noqa: PLC0415

    transform = build_test(IMAGE_SIZE)
    rows: list[dict] = []
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("[FILTER] Resolving available experiments (local weights ∩ W&B)...")
    print(f"{'=' * 70}")
    available_experiments = resolve_available_experiments(update_wandb=wandb_update)
    print(f"[FILTER] Will evaluate {len(available_experiments)} experiment(s).\n")

    for run_id, run_name in tqdm(available_experiments.items(), desc="runs"):
        print(f"\n{'=' * 70}")
        print(f"  Run : {run_name}  (id={run_id})")
        print(f"{'=' * 70}")

        base_row: dict = {"wandb_run_id": run_id, "experiment_name": run_name}

        try:
            info = parse_experiment_name(run_name)
            base_row.update({
                "dataset_name": info.dataset_name,
                "split_id": info.split_id,
                "percentage": info.percentage,
                "initialization_type": info.initialization_type,
            })

            num_classes = DATASET_NUM_CLASSES.get(info.dataset_name, 9)

            ckpt_path = find_best_checkpoint(run_id)
            print(f"  [INFO] Checkpoint : {os.path.relpath(ckpt_path, _ROOT)}")

            module = LejepaLineModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            module.eval()
            encoder = module.model.encoder
            for p in encoder.parameters():
                p.requires_grad_(False)

            train_base = ParasiteDataset(
                set_name="train",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            clf = train_svm(encoder, train_loader)

            test_ds = ParasiteDataset(
                set_name="test",
                split=info.split_id,
                percentage=info.percentage,
                transform=transform,
                loader="ift_lab",
                path_dataset=info.path_dataset,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            feats, y_true = extract_features(encoder, test_loader)
            y_pred = clf.predict(feats) - 1

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
            rows.append({**base_row, **metrics, "status": "ok", "error": ""})
            print(
                f"  [RESULT] kappa={metrics['kappa']:.4f}  "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
            )

            wandb_run = wandb.init(
                project=PROJECT,
                entity=ENTITY,
                name=f"svm_{run_name}",
                config={**base_row, "num_classes": num_classes},
                reinit=True,
            )
            wandb.log({f"svm/{k}": v for k, v in metrics.items()})
            wandb_run.finish()

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                **base_row,
                "kappa": float("nan"),
                "acc": float("nan"),
                "f1": float("nan"),
                "status": "error",
                "error": str(exc),
            })

    csv_path = os.path.join(_RESULTS_DIR, "svm_results.csv")
    _meta = ["wandb_run_id", "experiment_name", "dataset_name",
             "split_id", "percentage", "initialization_type"]
    _metrics = ["kappa", "acc", "f1"]
    _extra = ["status", "error"]
    _col_order = _meta + _metrics + _extra

    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to: {csv_path}")
    print(f"{'=' * 70}")

    # ── Upload CSV + summary metrics to W&B ──────────────────────────────────
    df_ok = df[df["status"] == "ok"]
    summary_metrics: dict = {}
    for m in _metrics:
        if m in df_ok.columns and not df_ok[m].isna().all():
            summary_metrics[f"summary/mean_{m}"] = float(df_ok[m].mean())
            summary_metrics[f"summary/std_{m}"]  = float(df_ok[m].std())

    print(f"\n[W&B] Uploading CSV and summary to project '{PROJECT}'...")
    for k, v in summary_metrics.items():
        print(f"  {k}: {v:.4f}")

    summary_run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        name="svm_summary",
        job_type="evaluation_summary",
        reinit=True,
    )
    wandb.log(summary_metrics)
    # Log full results as a W&B Table
    wandb.log({"svm/results_table": wandb.Table(dataframe=df_ok[_meta + _metrics].reset_index(drop=True))})
    # Upload CSV as an artifact
    artifact = wandb.Artifact("svm_results", type="evaluation_results")
    artifact.add_file(csv_path, name="svm_results.csv")
    summary_run.log_artifact(artifact)
    summary_run.finish()
    print("[W&B] Upload complete.")

    return csv_path


if __name__ == "__main__":
    import sys

    run_svm_probe(**cli_kwargs(sys.argv[1:]))
