"""eval_sigmoid2l_test.py — Compute TEST metrics for the sigmoid2l_ classification runs.

The ClassificationFlimModule only logs *validation* metrics during training
(there is no test_step), so the trained sigmoid2l_ checkpoints under
``artifacts/classification_flim/`` have no test_accuracy / test_cohen_kappa on
disk. This script fills that gap: it loads each best checkpoint, runs it over
the **test** split (mirroring validation_step, but on test_dataloader) using the
exact training transform, and writes one report CSV.

It does NOT train or modify anything — pure inference over existing checkpoints.

Output columns match data/reports_felipe/{svm,flim_mlp} for easy comparison:
    experiment, dataset, split, percentage, encoder_mode, init_method,
    test_accuracy, test_f1_weighted, test_cohen_kappa

Usage:
    python scripts/eval_sigmoid2l_test.py
    python scripts/eval_sigmoid2l_test.py --pattern 'sigmoid2l_classhead_*'  --out results/sigmoid2l_test_results.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_modules.parasite_data_module_lejepa_splited import (  # noqa: E402
    ParasiteLejepaDataModuleSplited,
)
from src.metrics.classification import compute_metrics  # noqa: E402
from src.modules.classification_flim_module import (  # noqa: E402
    ClassificationFlimModule,
    _dataset_short_to_parasite_name,
)

_ARTIFACTS = os.path.join(_ROOT, "artifacts", "classification_flim")


def _first_view(views):
    if isinstance(views, (list, tuple)):
        return views[0]
    if views.ndim == 5:
        return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
    return views


@torch.no_grad()
def _evaluate_run(run_dir: str, device: torch.device, image_size: int) -> dict | None:
    meta_path = os.path.join(run_dir, "run_metadata.json")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_kappa.ckpt")
    run_name = os.path.basename(run_dir)

    if not os.path.exists(meta_path):
        print(f"  [SKIP] no run_metadata.json: {run_name}")
        return None
    if not os.path.exists(ckpt_path) or os.path.getsize(ckpt_path) == 0:
        print(f"  [SKIP] no/empty best_kappa.ckpt: {run_name}")
        return None

    meta = json.load(open(meta_path))
    dataset = meta["dataset"]
    split = int(meta["split"])
    pct = int(meta["percentage"])
    num_classes = int(meta.get("num_classes", 9))
    freeze_encoder = bool(meta.get("freeze_encoder", False))
    # Training used imagenet_norm=True unless --no-imagenet-norm was passed.
    imagenet_norm = not bool(meta.get("no_imagenet_norm", False))
    encoder_mode = "frozen" if freeze_encoder else "unfrozen"

    print(f"  [RUN ] {run_name}  ({dataset} split{split} pct{pct} {encoder_mode})")

    module = ClassificationFlimModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval().to(device)

    dm = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(dataset),
        split=split, percentage=pct,
        image_size=image_size, V_train=1, V_eval=1,
        batch_size=32, num_workers=4, pin_memory=True, persistent_workers=False,
        loader="ift_lab", imagenet_norm=imagenet_norm,
    )
    dm.setup("test")

    preds, labels = [], []
    for batch in dm.test_dataloader():
        views, y = batch
        x = _first_view(views).to(device)
        probs = module(x)
        preds.append(probs.argmax(dim=1).cpu())
        labels.append(y.cpu())

    y_pred = torch.cat(preds)
    y_true = torch.cat(labels)
    m = compute_metrics(y_true, y_pred, num_classes=num_classes)

    print(f"         -> test_acc={m['acc']:.4f}  test_kappa={m['kappa']:.4f}  test_f1={m['f1']:.4f}")
    return {
        "experiment": run_name,
        "dataset": dataset,
        "split": split,
        "percentage": pct,
        "encoder_mode": encoder_mode,
        "init_method": "flim",
        "test_accuracy": float(m["acc"]),
        "test_f1_weighted": float(m["f1"]),
        "test_cohen_kappa": float(m["kappa"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute TEST metrics for sigmoid2l_ checkpoints.")
    ap.add_argument("--pattern", default="sigmoid2l_classhead_*",
                    help="glob of run dirs under artifacts/classification_flim/")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "sigmoid2l_test_results.csv"))
    ap.add_argument("--image-size", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(d for d in glob.glob(os.path.join(_ARTIFACTS, args.pattern)) if os.path.isdir(d))
    print(f"[eval_sigmoid2l_test] device={device}  runs matched={len(run_dirs)}")

    rows = []
    for rd in run_dirs:
        try:
            r = _evaluate_run(rd, device, args.image_size)
            if r:
                rows.append(r)
        except Exception as exc:  # keep going on failure
            print(f"  [ERROR] {os.path.basename(rd)}: {exc}")

    if not rows:
        print("[eval_sigmoid2l_test] No runs evaluated.")
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["dataset", "split", "percentage", "encoder_mode"])
    df.to_csv(args.out, index=False)
    print(f"\n[eval_sigmoid2l_test] Wrote {len(df)} rows -> {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
