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

    python -m src.evaluate.svm_classification_flim
    python -m src.evaluate.svm_classification_flim --pattern 'classhead_*_softplus2l'
    python -m src.evaluate.svm_classification_flim --pattern 'classhead_*_relu2l' --out results/svm_relu2l_results.csv
    python -m src.evaluate.svm_classification_flim --run classhead_eggs_split1_pct75_softplus2l --wandb-update
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

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

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reutiliza o pipeline de SVM já existente ──────────────────────────────────
from src.evaluate.svm_distillation import (  # noqa: E402
    _OneHotDataset,
    extract_features_distillation,
    train_svm_distillation,
    _RESULTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
)
from src.data_modules.datasets.dataset import DatasetParasite  # noqa: E402
from src.data_modules.datasets.lejepa_dataset import _build_test  # noqa: E402
from src.modules.classification_flim_module import ClassificationFlimModule  # noqa: E402

_ARTIFACTS_DIR = os.path.join(_ROOT, "artifacts", "classification_flim")
_DEFAULT_PATTERN = "classhead_*_softplus2l"


class _EncoderProbe(nn.Module):
    """Adaptador: expõe ``encode()`` sobre o encoder de ``ClassificationFlimModule``.

    ``extract_features_distillation`` e ``train_svm_distillation`` esperam um objeto
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SVM linear sobre o encoder FLIM dos runs de classification_flim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pattern", default=_DEFAULT_PATTERN,
                        help="Glob dos diretórios de run sob artifacts/classification_flim/.")
    parser.add_argument("--run", default=None,
                        help="Restringe a um run (match por substring).")
    parser.add_argument("--out", default=None,
                        help="CSV de saída. Default: results/svm_<tag>_results.csv, "
                             "onde <tag> vem da variante de cabeça encontrada.")
    parser.add_argument("--artifacts-dir", default=_ARTIFACTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-imagenet-norm", action="store_true", default=False,
                        help="Desliga a Normalize ImageNet no transform de teste. "
                             "O treino do Experimento 3 usa imagenet_norm=True.")
    parser.add_argument("--wandb-update", action="store_true",
                        help="Loga as métricas de SVM por experimento no W&B.")
    args = parser.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    transform = _build_test(IMAGE_SIZE, imagenet_norm=not args.no_imagenet_norm)

    runs = find_classification_runs(args.artifacts_dir, args.pattern, run_filter=args.run)
    if not runs:
        print(f"[WARN] Nenhum run válido em {args.artifacts_dir} com pattern '{args.pattern}'.")
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

            train_base = DatasetParasite(
                set_name="train", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_base, num_classes),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            clf = train_svm_distillation(probe, train_loader)

            test_ds = DatasetParasite(
                set_name="test", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            feats, y_true = extract_features_distillation(probe, test_loader)

            # O SVM é treinado com rótulos 1-indexed → volta para 0-indexed.
            y_pred = clf.predict(feats) - 1

            metrics = _metrics_both_conventions(y_true, y_pred)
            rows.append({**base_row, **metrics, "status": "ok", "error": ""})

            print(f"  [RESULT] acc={metrics['test_accuracy']:.4f}  "
                  f"f1_w={metrics['test_f1_weighted']:.4f}  "
                  f"kappa={metrics['test_cohen_kappa']:.4f}  "
                  f"(acc_bal={metrics['test_accuracy_balanced']:.4f})")

            if args.wandb_update:
                try:
                    import wandb  # noqa: PLC0415
                    from src.utils.get_names_wandb import ENTITY, PROJECT  # noqa: PLC0415
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
    if args.out:
        csv_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
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
    raise SystemExit(main())
