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
"""svm_flim_residual.py — SVM evaluation of residual FLIM encoders (eggs only).

Mirrors ``eval/svm.py`` but evaluates two hand-built residual FLIM
encoders (output-side skip-concat after conv3) instead of trained SSL
checkpoints:

* ``1_3``: concat(conv3_out[48], conv1_out[24]) → 72 ch (method ``SVM_FLIMResidual_1_3``)
* ``2_3``: concat(conv3_out[48], conv2_out[32]) → 80 ch (method ``SVM_FLIMResidual_2_3``)

conv1/conv2/conv3 all load the pretrained FLIM kernels unchanged — no new or
retrained weights anywhere, so the encoder is identical across percentages.
The percentage axis instead controls how much of the **downstream training
split** is used to fit the SVM (the same fixed encoder is reused); the test
split is always the full, constant test set (unaffected by percentage) — see
``data/to_modules/new_split_parasito/<dataset>/splits_incremental/``. This
mirrors the existing repo pattern of a per-percentage data-efficiency curve,
without requiring any SSL/distillation training step.

Evaluated on the **eggs** dataset only, splits 1/2/3, percentages
1/5/25/50/75/100 (36 SVM fits total: 2 variants x 3 splits x 6 percentages).

Usage:
    python -m eval.svm_variants.svm_flim_residual --flim_residual_assessment
"""
from __future__ import annotations

import os

import pandas as pd
from torch.utils.data import DataLoader

from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from flim.flim_residual_encoder import build_flim_residual_encoder
from core.constants import IMAGE_SIZE
from eval.svm import (
    DEVICE,
    _OneHotDataset,
    _ROOT,
    cli_kwargs,
    extract_features,
    train_svm,
)

_RESULTS_DIR = os.path.join(_ROOT, "results")

# ─── Registry of the two residual variants ────────────────────────────────────
FLIM_RESIDUAL_VARIANTS: dict[str, dict] = {
    "1_3": {"method": "SVM_FLIMResidual_1_3", "init": "flim_residual_1_3"},
    "2_3": {"method": "SVM_FLIMResidual_2_3", "init": "flim_residual_2_3"},
}

# Fixed evaluation scope.
_DATASET_NAME = "helminth-eggs"
_PATH_DATASET = "helminth-eggs"
_NUM_CLASSES = 9
_SPLITS = [1, 2, 3]
_PERCENTAGES = [1, 5, 25, 50, 75, 100]

# Pretrained FLIM weights for eggs (arch ch24_32_48).
_FLIM_ROOT = os.path.join(
    _ROOT, "data", "to_mateus", "model", "ch24_32_48_a0.5_f5", "eggs"
)


def _weights_and_arch(split: int) -> tuple[str, str]:
    train_dir = os.path.join(_FLIM_ROOT, f"train{split}")
    weights_path = os.path.join(train_dir, "models")
    arch_json = os.path.join(train_dir, "architecture.json")
    return weights_path, arch_json


def _build_encoder(mode: str, split: int):
    """Build the (fixed, deterministic, percentage-independent) residual encoder."""
    weights_path, arch_json = _weights_and_arch(split)
    encoder = build_flim_residual_encoder(
        mode=mode,
        arch_json_path=arch_json,
        weights_path=weights_path,
        in_channels=3,
    )
    encoder.eval()
    encoder.to(DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def _evaluate_variant(encoder, split: int, percentage: int, transform) -> dict:
    """Fit an SVM on *encoder* features at *percentage* of the train split.

    The encoder itself never changes with *percentage* (no training) — only
    how much of the downstream train split feeds the SVM. The test split is
    always the same fixed set regardless of percentage (see module docstring).
    """
    train_base = ParasiteDataset(
        set_name="train",
        split=split,
        percentage=percentage,
        transform=transform,
        loader="ift_lab",
        path_dataset=_PATH_DATASET,
    )
    train_loader = DataLoader(
        _OneHotDataset(train_base, _NUM_CLASSES),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
    )
    clf = train_svm(encoder, train_loader)

    test_ds = ParasiteDataset(
        set_name="test",
        split=split,
        percentage=percentage,
        transform=transform,
        loader="ift_lab",
        path_dataset=_PATH_DATASET,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
    )
    feats, y_true = extract_features(encoder, test_loader)
    y_pred = clf.predict(feats) - 1

    return compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=_NUM_CLASSES)


def run_flim_residual_assessment() -> str:
    """Evaluate both residual variants over eggs splits x percentages; write the CSV."""
    transform = build_test(IMAGE_SIZE)
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    rows: list[dict] = []

    for mode, spec in FLIM_RESIDUAL_VARIANTS.items():
        for split in _SPLITS:
            print(f"\n{'=' * 70}")
            print(f"  Variant : {spec['method']}  |  split={split}  |  building encoder (once)")
            print(f"{'=' * 70}")
            encoder = _build_encoder(mode, split)

            for percentage in _PERCENTAGES:
                experiment_name = (
                    f"flim_residual_{mode}_{_DATASET_NAME}_split_{split}_pct_{percentage}"
                )
                print(f"  -- percentage={percentage}")

                base_row = {
                    "experiment_name": experiment_name,
                    "dataset_name": _DATASET_NAME,
                    "split_id": split,
                    "percentage": percentage,
                    "initialization_type": spec["init"],
                    "method": spec["method"],
                }
                try:
                    metrics = _evaluate_variant(encoder, split, percentage, transform)
                    rows.append({**base_row, **metrics, "status": "ok", "error": ""})
                    print(
                        f"     [RESULT] kappa={metrics['kappa']:.4f}  "
                        f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"     [ERROR] {exc}")
                    rows.append({
                        **base_row,
                        "kappa": float("nan"),
                        "acc": float("nan"),
                        "f1": float("nan"),
                        "status": "error",
                        "error": str(exc),
                    })

    csv_path = os.path.join(_RESULTS_DIR, "svm_flim_residual_eggs.csv")
    _col_order = [
        "experiment_name", "dataset_name", "split_id", "percentage",
        "initialization_type", "method", "kappa", "acc", "f1", "status", "error",
    ]
    df = pd.DataFrame(rows)
    remaining = [c for c in df.columns if c not in _col_order]
    df = df.reindex(columns=_col_order + remaining)
    df.to_csv(csv_path, index=False)
    return csv_path


def main(flim_residual_assessment: bool = False) -> None:
    """Roda a avaliacao SVM das duas variantes residuais FLIM em eggs.

    Args:
        flim_residual_assessment: Continua obrigatorio, como o
            `--flim_residual_assessment` do argparse de origem: sem ele o
            script nao tinha nada a fazer e o parser abortava.
    """
    if not flim_residual_assessment:
        raise SystemExit("nothing to do: pass --flim_residual_assessment")

    csv_path = run_flim_residual_assessment()

    print(f"\n{'=' * 70}")
    print(f"[DONE] Results saved to: {csv_path}")
    print("[NEXT] Normalize + plot (handled by the other agent):")
    print("       python scripts/normalize_reports.py")
    print("       python scripts/plot_comparison_flim.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import sys

    main(**cli_kwargs(sys.argv[1:]))
