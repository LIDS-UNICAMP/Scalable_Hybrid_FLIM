# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""svm_classification_flim.py — SVM evaluation dos encoders treinados pelo
Experimento 3 (`scripts/classification_flim_ray.py`).

Sonda linear (SVM) sobre o encoder FLIM **depois** do treino supervisionado, para
separar "a representação é boa?" de "a cabeça conseguiu otimizar?". Isso é o
controle que faltava no diagnóstico do grupo `_relu2l`, onde vários runs
colapsaram para predição constante com o encoder livre.

Reutiliza integralmente o pipeline de `svm_distillation.py` — só troca:
  * LeJEPAFLIMModel      →  encoder de ClassificationFlimModule (via adaptador)
  * artifacts/distillation → artifacts/classification_flim
  * filtro de runs        →  glob sobre os diretórios de run (`--pattern`)

Embedding usado: `AdaptiveAvgPool2d(1)` sobre a saída do encoder → ``[B, 48]``
(``[B, 30]`` no protozoan não se aplica: a última conv é sempre 48). É exatamente
a entrada que a cabeça de classificação recebe, sem a cabeça.

Métricas: gravadas nas **duas convenções**, com nomes explícitos —
`test_accuracy` é global (micro) e `test_f1_weighted` é ponderado, comparáveis com
`data/reports_felipe/svm/`; `test_accuracy_balanced` e `test_f1_macro` são as
variantes macro. Ver a ressalva C3 em `A_reports/july/check_sanity.md`.

Usage::

    python -m eval.svm_variants.svm_classification_flim
    python -m eval.svm_variants.svm_classification_flim --pattern 'classhead_*_softplus2l'
    python -m eval.svm_variants.svm_classification_flim --pattern 'classhead_*_relu2l' --out results/svm_relu2l_results.csv
    python -m eval.svm_variants.svm_classification_flim --run classhead_eggs_split1_pct75_softplus2l --wandb-update
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from torch.utils.data import DataLoader


# ── Reutiliza o pipeline de SVM já existente ──────────────────────────────────
from eval.svm_variants.svm_distillation import (  # noqa: E402
    _OneHotDataset,
    train_svm_distillation,
    _RESULTS_DIR,
    DEVICE,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
)
from core.constants import IMAGE_SIZE, PROJECT_ROOT  # noqa: E402
from eval.svm import cli_kwargs, extract_features_encode  # noqa: E402
from core.data.parasite_dataset import ParasiteDataset  # noqa: E402
from core.data.transforms import build_test  # noqa: E402
from methods.classification import ClassificationFlimModule  # noqa: E402

_ROOT = PROJECT_ROOT

_ARTIFACTS_DIR = os.path.join(_ROOT, "artifacts", "classification_flim")
_DEFAULT_PATTERN = "classhead_*_softplus2l"


class _EncoderProbe(nn.Module):
    """Adaptador: expõe ``encode()`` sobre o encoder de ``ClassificationFlimModule``.

    ``extract_features_encode`` e ``train_svm_distillation`` esperam um objeto
    com ``.encode(x) -> [B, D]``; o encoder FLIM devolve um mapa espacial
    ``[B, C, H, W]``. Este adaptador aplica o mesmo ``AdaptiveAvgPool2d(1)`` que a
    cabeça de classificação usa, de modo que o SVM enxerga exatamente as features
    que a cabeça enxergava.
    """

    def __init__(self, module: ClassificationFlimModule) -> None:
        super().__init__()
        self.encoder = module.model.encoder
        self.pool = nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.pool(self.encoder(x))
        return feats.flatten(1)


def find_classification_runs(artifacts_dir: str, pattern: str,
                             run_filter: str | None = None) -> list[dict]:
    """Descobre runs válidos (com metadata e best_kappa.ckpt) sob ``artifacts_dir``.

    Args:
        artifacts_dir: raiz dos artefatos (artifacts/classification_flim).
        pattern:       glob dos diretórios de run.
        run_filter:    se dado, mantém só runs cujo nome contém esta substring.

    Returns:
        Lista de dicts do ``run_metadata.json`` acrescidos de ``_run_dir`` e ``_ckpt_path``.
    """
    runs: list[dict] = []
    for run_dir in sorted(glob.glob(os.path.join(artifacts_dir, pattern))):
        if not os.path.isdir(run_dir):
            continue
        run_name = os.path.basename(run_dir)
        if run_filter and run_filter not in run_name:
            continue

        meta_path = os.path.join(run_dir, "run_metadata.json")
        ckpt_path = os.path.join(run_dir, "checkpoints", "best_kappa.ckpt")
        if not os.path.exists(meta_path):
            print(f"  [SKIP] sem run_metadata.json: {run_name}")
            continue
        if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) == 0:
            print(f"  [SKIP] sem/vazio best_kappa.ckpt: {run_name}")
            continue

        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["_run_dir"] = run_dir
        meta["_ckpt_path"] = ckpt_path
        meta.setdefault("run_name", run_name)
        runs.append(meta)
    return runs


def _head_variant(meta: dict) -> str:
    if meta.get("output_softplus"):
        return "softplus2l"
    if meta.get("output_relu"):
        return "relu2l"
    return "sigmoid2l"


def _metrics_both_conventions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas nas duas convenções, com nomes explícitos (ver C3 do sanity check)."""
    return {
        "test_accuracy":          float(accuracy_score(y_true, y_pred)),
        "test_f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "test_cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        "test_accuracy_balanced": float(balanced_accuracy_score(y_true, y_pred)),
        "test_f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main(
    pattern: str = _DEFAULT_PATTERN,
    run: str | None = None,
    out: str | None = None,
    artifacts_dir: str = _ARTIFACTS_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    no_imagenet_norm: bool = False,
    wandb_update: bool = False,
) -> int:
    """SVM linear sobre o encoder FLIM dos runs de classification_flim.

    Args:
        pattern:          Glob dos diretorios de run (`--pattern`).
        run:              Restringe a um run, por substring (`--run`).
        out:              CSV de saida (`--out`); default derivado da variante
                          de cabeca encontrada.
        artifacts_dir:    Raiz dos artefatos (`--artifacts-dir`).
        batch_size:       Batch dos dois loaders (`--batch-size`).
        num_workers:      Workers dos dois loaders (`--num-workers`).
        no_imagenet_norm: Desliga a Normalize ImageNet no teste
                          (`--no-imagenet-norm`).
        wandb_update:     Loga as metricas por experimento no W&B
                          (`--wandb-update`).
    """

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    transform = build_test(IMAGE_SIZE, imagenet_norm=not no_imagenet_norm)

    runs = find_classification_runs(artifacts_dir, pattern, run_filter=run)
    if not runs:
        print(f"[WARN] Nenhum run válido em {artifacts_dir} com pattern '{pattern}'.")
        return 1

    print(f"\n{'=' * 70}")
    print(f"[FILTER] {len(runs)} run(s) para avaliar  |  device={DEVICE}")
    print(f"{'=' * 70}")

    rows: list[dict] = []

    for meta in runs:
        run_name = meta["run_name"]
        ckpt_path = meta["_ckpt_path"]
        dataset = meta.get("dataset", "")
        split = int(meta.get("split", 1))
        pct = int(meta.get("percentage", 100))
        variant = _head_variant(meta)

        print(f"\n{'=' * 70}")
        print(f"  Run  : {run_name}")
        print(f"  Ckpt : {os.path.relpath(ckpt_path, _ROOT)}")
        print(f"{'=' * 70}")

        base_row = {
            "experiment":     f"svm_{variant}",
            "run_name":       run_name,
            "dataset":        dataset,
            "split":          split,
            "percentage":     pct,
            # O SVM sempre sonda features congeladas; este campo descreve a sonda.
            "encoder_mode":   "frozen",
            "init_method":    "flim",
            "head_variant":   variant,
            # Como o encoder foi OBTIDO durante o treino supervisionado:
            "train_freeze_encoder": bool(meta.get("freeze_encoder", False)),
            "ckpt_path":      ckpt_path,
        }

        num_classes = _DATASET_NUM_CLASSES.get(dataset, 9)
        parasite_name = _DATASET_PARASITE_NAME.get(dataset, dataset)

        try:
            module = ClassificationFlimModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            probe = _EncoderProbe(module).eval().to(DEVICE)
            for p in probe.parameters():
                p.requires_grad_(False)

            train_base = ParasiteDataset(
                set_name="train", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
            clf = train_svm_distillation(probe, train_loader)

            test_ds = ParasiteDataset(
                set_name="test", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )
            feats, y_true = extract_features_encode(probe, test_loader)

            # O SVM é treinado com rótulos 1-indexed → volta para 0-indexed.
            y_pred = clf.predict(feats) - 1

            metrics = _metrics_both_conventions(y_true, y_pred)
            rows.append({**base_row, **metrics, "status": "ok", "error": ""})

            print(f"  [RESULT] acc={metrics['test_accuracy']:.4f}  "
                  f"f1_w={metrics['test_f1_weighted']:.4f}  "
                  f"kappa={metrics['test_cohen_kappa']:.4f}  "
                  f"(acc_bal={metrics['test_accuracy_balanced']:.4f})")

            if wandb_update:
                try:
                    import wandb  # noqa: PLC0415
                    from core.wandb import ENTITY, PROJECT  # noqa: PLC0415
                    wandb_run = wandb.init(
                        project=PROJECT, entity=ENTITY,
                        name=f"svm_{variant}_{run_name}",
                        config={**base_row, "num_classes": num_classes},
                        reinit=True,
                    )
                    wandb.log({f"svm/{k}": v for k, v in metrics.items()})
                    wandb_run.finish()
                except Exception as _we:
                    print(f"  [WARN] W&B logging falhou: {_we}")

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            rows.append({
                **base_row,
                "test_accuracy": float("nan"), "test_f1_weighted": float("nan"),
                "test_cohen_kappa": float("nan"), "test_accuracy_balanced": float("nan"),
                "test_f1_macro": float("nan"),
                "status": "error", "error": str(exc),
            })

    df = pd.DataFrame(rows)
    if out:
        csv_path = out if os.path.isabs(out) else os.path.join(_ROOT, out)
    else:
        variants = sorted(set(df["head_variant"]))
        tag = variants[0] if len(variants) == 1 else "classification_flim"
        csv_path = os.path.join(_RESULTS_DIR, f"svm_{tag}_results.csv")

    _col_order = [
        "experiment", "run_name", "dataset", "split", "percentage",
        "encoder_mode", "init_method", "head_variant", "train_freeze_encoder",
        "test_accuracy", "test_f1_weighted", "test_cohen_kappa",
        "test_accuracy_balanced", "test_f1_macro",
        "status", "error", "ckpt_path",
    ]
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df = df.sort_values(["dataset", "split", "percentage"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)

    n_ok = int((df["status"] == "ok").sum())
    print(f"\n{'=' * 70}")
    print(f"[DONE] {n_ok}/{len(df)} runs OK  →  {os.path.relpath(csv_path, _ROOT)}")
    print(f"{'=' * 70}")
    return 0 if n_ok else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(**cli_kwargs(sys.argv[1:])))
