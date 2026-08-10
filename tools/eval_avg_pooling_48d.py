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
"""eval_avg_pooling_48d.py — 48-d + LAB[0,1] raw + solver convergido.

EXPERIMENTO VERSIONADO (nao e script descartavel). Re-executavel.

Pergunta: com **48 dimensoes**, quanto as correcoes de normalizacao e de solver
recuperam de kappa sozinhas? Ou seja, o ganho medido com o mapa conv3 achatado
(27.648-d) vinha da entrada e do solver, ou vinha da dimensionalidade?

Esta rodada adota DUAS das tres divergencias entre o avaliador oficial e o
diagnostico, mantendo a dimensao do lado oficial:

  slot       | oficial (src/utils/evaluate.py) | aqui
  -----------|---------------------------------|---------------------------
  features   | AdaptiveAvgPool2d(1) -> 48-d    | IDEM (48-d, nao muda)
  entrada    | _build_test(200, True)          | _build_test(200, False)
  solver     | max_iter=10000                  | max_iter=-1 (convergido)

O avaliador oficial e IMPORTADO, nunca reescrito:
  * train_svm        src/utils/evaluate.py:259   (o parametro max_iter ja existe)
  * extract_features src/utils/evaluate.py:337
  * _encode_pooled   src/utils/evaluate.py:247   (e o que fixa as 48-d)
  * compute_metrics  src/metrics/classification.py:31
  * rotulos          1-indexed no fit, predict(feats)-1  (src/evaluate/svm.py:135)

Encoder: FLIM raw, sem checkpoint — mesmo caminho que o estagio 1 congela
(src/modules/autoencoder_flim_module.py:155-168).

Uso:
    python tools/eval_avg_pooling_48d.py --dry-run
    OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
      conda run -n scalable_FLIM --no-capture-output \
      python tools/eval_avg_pooling_48d.py --splits 1 2 3 --percentages 1 5 25 50 75 100
"""

import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch.utils.data import DataLoader                                 # noqa: E402

from src.data_modules.datasets.dataset import DatasetParasite           # noqa: E402
from src.data_modules.datasets.lejepa_dataset import _build_test        # noqa: E402
from src.metrics.classification import compute_metrics                  # noqa: E402
from src.modules.autoencoder_flim_module import (                       # noqa: E402
    NUM_CLASSES, AutoEncoderFlimModule,
)
from src.evaluate.constants import IMAGE_SIZE                           # noqa: E402
from src.utils.evaluate import (                                        # noqa: E402
    DEVICE, _OneHotDataset, extract_features, train_svm,
)
from autoencoder_flim_ray import _arch_json, _flim_weights_path         # noqa: E402

_PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae",
             "protozoan": "protozoan-cysts"}

# Onde vivem os dados das curvas ja plotadas — o CSV novo fica ao lado.
_CURVE_DIR = os.path.join(_ROOT, "artifacts", "plots",
                          "comparacao_flim_protocolo_original")
_OUT_CSV = os.path.join(_CURVE_DIR, "eval_48d_norm_off.csv")

_METHOD = "SVM_FLIM_48d_labcru_conv"
_FIELDS = ["method", "dataset", "split", "percentage", "n_train", "n_test",
           "dim", "imagenet_norm", "max_iter", "fit_status", "n_iter_max",
           "kappa", "acc", "f1", "acc_raw", "fit_s"]


def _build_encoder(dataset: str, split: int):
    """Encoder FLIM raw — o mesmo que o estagio 1 congela, sem checkpoint."""
    mod = AutoEncoderFlimModule(
        arch_json=_arch_json(dataset, split),
        dataset=dataset,
        flim_weights_path=_flim_weights_path(dataset, split),
        num_classes=NUM_CLASSES[dataset],
        image_size=IMAGE_SIZE,
        imagenet_norm=True,      # so afeta o alvo da reconstrucao; irrelevante aqui
        freeze_encoder_flag=True,
    )
    enc = mod.model.encoder.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _loader(dataset: str, split: int, pct: int, set_name: str,
            num_workers: int, one_hot: int = 0):
    """Dataloader identico ao do avaliador oficial (evaluate.py:413-449),
    trocando SO imagenet_norm=False."""
    base = DatasetParasite(
        set_name=set_name, split=split, percentage=pct,
        transform=_build_test(IMAGE_SIZE, imagenet_norm=False),
        loader="ift_lab", path_dataset=_PARASITE[dataset],
    )
    ds = _OneHotDataset(base, one_hot) if one_hot else base
    return DataLoader(ds, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="protozoan",
                    choices=["protozoan", "eggs", "larvae"])
    ap.add_argument("--splits", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--percentages", nargs="+", type=int,
                    default=[1, 5, 25, 50, 75, 100])
    ap.add_argument("--max-iter", type=int, default=-1,
                    help="-1 = convergido. Um teto positivo fica declarado no CSV.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default=_OUT_CSV)
    ap.add_argument("--dry-run", action="store_true",
                    help="Valida arch JSON, pesos e splits; nao ajusta SVM nenhum.")
    args = ap.parse_args()

    ds = args.dataset
    n_cls = NUM_CLASSES[ds]
    print(f"[CONTRATO] features=AdaptiveAvgPool2d(1) -> 48-d (evaluate.py:244,247-252)")
    print(f"[CONTRATO] entrada =_build_test({IMAGE_SIZE}, imagenet_norm=False)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter})")
    print(f"[CONTRATO] dataset ={ds} ({n_cls} classes) | device={DEVICE}\n")

    if args.dry_run:
        ok = True
        for sp in args.splits:
            aj, wp = _arch_json(ds, sp), _flim_weights_path(ds, sp)
            for path in (aj, wp):
                flag = os.path.exists(path)
                ok &= flag
                print(f"  [{'OK ' if flag else 'FALTA'}] {path}")
            for pct in args.percentages:
                try:
                    n = len(DatasetParasite(set_name="train", split=sp, percentage=pct,
                                            transform=None, loader="ift_lab",
                                            path_dataset=_PARASITE[ds]))
                    print(f"  [OK ] split{sp} pct{pct:<3d} train={n}")
                except Exception as exc:                       # noqa: BLE001
                    ok = False
                    print(f"  [FALTA] split{sp} pct{pct}: {exc}")
        print(f"\n[DRY-RUN] {'tudo presente' if ok else 'FALTAM ARQUIVOS'}")
        return

    rows = []
    for sp in args.splits:
        enc = _build_encoder(ds, sp)
        test_loader = _loader(ds, sp, args.percentages[0], "test", args.num_workers)
        feats_te, y_true = extract_features(enc, test_loader)
        print(f"[INFO] split{sp}: test n={len(y_true)} dim={feats_te.shape[1]}")

        for pct in args.percentages:
            tr_loader = _loader(ds, sp, pct, "train", args.num_workers, one_hot=n_cls)
            n_train = len(tr_loader.dataset)

            t0 = time.perf_counter()
            clf = train_svm(enc, tr_loader, max_iter=args.max_iter)
            fit_s = time.perf_counter() - t0

            y_pred = clf.predict(feats_te) - 1          # svm.py:135
            m = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_cls)
            acc_raw = float((y_pred == y_true).mean())
            n_iter = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else -1

            rows.append({
                "method": _METHOD, "dataset": ds, "split": sp, "percentage": pct,
                "n_train": n_train, "n_test": len(y_true), "dim": feats_te.shape[1],
                "imagenet_norm": False, "max_iter": args.max_iter,
                "fit_status": int(clf.fit_status_), "n_iter_max": n_iter,
                "kappa": float(m["kappa"]), "acc": float(m["acc"]),
                "f1": float(m["f1"]), "acc_raw": acc_raw, "fit_s": round(fit_s, 1),
            })
            print(f"  [RESULT] sp{sp} pct{pct:<3d} n_train={n_train:<5d} "
                  f"kappa={m['kappa']:+.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} "
                  f"acc_raw={acc_raw:.4f} fit_status={clf.fit_status_} "
                  f"n_iter={n_iter} ({fit_s:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {args.out}  ({len(rows)} linhas)")

    print("\n=== media por percentage (3 splits) ===")
    print(f"{'pct':>5} {'kappa':>9} {'acc':>8} {'f1':>8} {'acc_raw':>8}")
    for pct in args.percentages:
        sel = [r for r in rows if r["percentage"] == pct]
        if sel:
            print(f"{pct:>5} {np.mean([r['kappa'] for r in sel]):>+9.4f} "
                  f"{np.mean([r['acc'] for r in sel]):>8.4f} "
                  f"{np.mean([r['f1'] for r in sel]):>8.4f} "
                  f"{np.mean([r['acc_raw'] for r in sel]):>8.4f}")


if __name__ == "__main__":
    main()
