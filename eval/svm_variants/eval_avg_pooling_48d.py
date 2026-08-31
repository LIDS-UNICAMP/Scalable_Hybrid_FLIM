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
  entrada    | build_test(200, True)          | build_test(200, False)
  solver     | max_iter=10000                  | max_iter=-1 (convergido)

O avaliador oficial e IMPORTADO, nunca reescrito:
  * train_svm        src/utils/evaluate.py:259   (o parametro max_iter ja existe)
  * extract_features src/utils/evaluate.py:337
  * _encode_pooled   src/utils/evaluate.py:247   (e o que fixa as 48-d)
  * compute_metrics  src/metrics/classification.py:31
  * rotulos          1-indexed no fit, predict(feats)-1  (eval/svm.py:135)

Encoder: FLIM raw, sem checkpoint — mesmo caminho que o estagio 1 congela
(src/modules/autoencoder_flim_module.py:155-168).

Uso:
    python -m eval.svm_variants.eval_avg_pooling_48d --dry-run
    OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
      conda run -n scalable_FLIM --no-capture-output \
      python -m eval.svm_variants.eval_avg_pooling_48d --splits 1 2 3 --percentages 1 5 25 50 75 100
"""

import csv
from types import SimpleNamespace
import os
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")


from torch.utils.data import DataLoader                                 # noqa: E402

from core.data.parasite_dataset import ParasiteDataset           # noqa: E402
from core.data.transforms import build_test        # noqa: E402
from core.metrics import compute_metrics                  # noqa: E402
from methods.autoencoder import AutoEncoderFlimModule  # noqa: E402
from core.constants import IMAGE_SIZE, NUM_CLASSES, PROJECT_ROOT                           # noqa: E402
from eval.svm import (                                        # noqa: E402
    DEVICE, _OneHotDataset, cli_kwargs, extract_features, train_svm,
)
# `experiments.ray.paths.arch_json` recebe o nome LONGO do dataset
# (paths.py:342 faz `DATASET_LONG_TO_SHORT[dataset]`), enquanto o
# `_arch_json` de origem (scripts/autoencoder_flim_ray.py:266) indexava
# a base direto pela chave CURTA. `PARASITE_DIR` e a traducao canonica.
from experiments.ray.paths import arch_json, flim_weights_path         # noqa: E402
from experiments.constants import PARASITE_DIR  # noqa: E402

_ROOT = PROJECT_ROOT

_PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae",
             "protozoan": "protozoan-cysts"}

# Onde vivem os dados das curvas ja plotadas — o CSV novo fica ao lado.
_CURVE_DIR = os.path.join(_ROOT, "artifacts", "plots",
                          "comparacao_flim_protocolo_original")
_OUT_CSV = os.path.join(_CURVE_DIR, "eval_48d_norm_off.csv")

_METHOD = "SVM_FLIM_48d_lab_raw_conv"
_FIELDS = ["method", "dataset", "split", "percentage", "n_train", "n_test",
           "dim", "imagenet_norm", "max_iter", "fit_status", "n_iter_max",
           "kappa", "acc", "f1", "acc_raw", "fit_s"]


def _build_encoder(dataset: str, split: int):
    """Encoder FLIM raw — o mesmo que o estagio 1 congela, sem checkpoint."""
    module = AutoEncoderFlimModule(
        arch_json=arch_json(PARASITE_DIR[dataset], split),
        dataset=dataset,
        flim_weights_path=flim_weights_path(PARASITE_DIR[dataset], split),
        num_classes=NUM_CLASSES[dataset],
        image_size=IMAGE_SIZE,
        imagenet_norm=True,      # so afeta o alvo da reconstrucao; irrelevante aqui
        freeze_encoder_flag=True,
    )
    encoder = module.model.encoder.to(DEVICE).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def _loader(dataset: str, split: int, percentage: int, set_name: str,
            num_workers: int, one_hot: int = 0):
    """Dataloader identico ao do avaliador oficial (evaluate.py:413-449),
    trocando SO imagenet_norm=False."""
    base = ParasiteDataset(
        set_name=set_name, split=split, percentage=percentage,
        transform=build_test(IMAGE_SIZE, imagenet_norm=False),
        loader="ift_lab", path_dataset=_PARASITE[dataset],
    )
    wrapped = _OneHotDataset(base, one_hot) if one_hot else base
    return DataLoader(wrapped, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main(
    dataset: str = "protozoan",
    splits: list[int] | None = None,
    percentages: list[int] | None = None,
    max_iter: int = -1,
    num_workers: int = 8,
    out: str = _OUT_CSV,
    dry_run: bool = False,
) -> None:
    """Um parametro por flag do argparse antigo, com o mesmo default.

    Os dois `nargs="+"` saem como `None` na assinatura para nao deixar lista
    mutavel de default; o valor de origem e restaurado logo abaixo. O que os
    `choices=` validavam agora e responsabilidade de quem chama.
    """
    # `nargs="+"` do argparse sempre devolvia lista; `cli_kwargs` devolve escalar
    # quando so um valor e passado. A normalizacao mora aqui, na propria funcao.
    splits = [1, 2, 3] if splits is None else (
        [splits] if isinstance(splits, int) else splits)
    percentages = [1, 5, 25, 50, 75, 100] if percentages is None else (
        [percentages] if isinstance(percentages, int) else percentages)
    # Shim de uma linha: o corpo abaixo e o da origem, que lia `args.<flag>`.
    # `locals()` na primeira linha viva e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    dataset = args.dataset
    num_classes = NUM_CLASSES[dataset]
    print(f"[CONTRATO] features=AdaptiveAvgPool2d(1) -> 48-d (evaluate.py:244,247-252)")
    print(f"[CONTRATO] entrada =build_test({IMAGE_SIZE}, imagenet_norm=False)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter})")
    print(f"[CONTRATO] dataset ={dataset} ({num_classes} classes) "
          f"| device={DEVICE}\n")

    if args.dry_run:
        ok = True
        for split in args.splits:
            arch_json_path = arch_json(PARASITE_DIR[dataset], split)
            weights_path = flim_weights_path(PARASITE_DIR[dataset], split)
            for path in (arch_json_path, weights_path):
                flag = os.path.exists(path)
                ok &= flag
                print(f"  [{'OK ' if flag else 'FALTA'}] {path}")
            for percentage in args.percentages:
                try:
                    n_train = len(ParasiteDataset(
                        set_name="train", split=split, percentage=percentage,
                        transform=None, loader="ift_lab",
                        path_dataset=_PARASITE[dataset]))
                    print(f"  [OK ] split{split} percentage{percentage:<3d} "
                          f"train={n_train}")
                except Exception as exc:                       # noqa: BLE001
                    ok = False
                    print(f"  [FALTA] split{split} "
                          f"percentage{percentage}: {exc}")
        print(f"\n[DRY-RUN] {'tudo presente' if ok else 'FALTAM ARQUIVOS'}")
        return

    rows = []
    for split in args.splits:
        encoder = _build_encoder(dataset, split)
        test_loader = _loader(dataset, split, args.percentages[0], "test",
                              args.num_workers)
        test_features, y_true = extract_features(encoder, test_loader)
        print(f"[INFO] split{split}: test n={len(y_true)} "
              f"dim={test_features.shape[1]}")

        for percentage in args.percentages:
            train_loader = _loader(dataset, split, percentage, "train",
                                   args.num_workers, one_hot=num_classes)
            n_train = len(train_loader.dataset)

            start_time = time.perf_counter()
            classifier = train_svm(encoder, train_loader,
                                   max_iter=args.max_iter)
            fit_seconds = time.perf_counter() - start_time

            y_pred = classifier.predict(test_features) - 1     # svm.py:135
            metrics = compute_metrics(y_true=y_true, y_pred=y_pred,
                                      num_classes=num_classes)
            acc_raw = float((y_pred == y_true).mean())
            n_iter = (int(np.max(classifier.n_iter_))
                      if hasattr(classifier, "n_iter_") else -1)

            rows.append({
                "method": _METHOD, "dataset": dataset, "split": split,
                "percentage": percentage, "n_train": n_train,
                "n_test": len(y_true), "dim": test_features.shape[1],
                "imagenet_norm": False, "max_iter": args.max_iter,
                "fit_status": int(classifier.fit_status_), "n_iter_max": n_iter,
                "kappa": float(metrics["kappa"]), "acc": float(metrics["acc"]),
                "f1": float(metrics["f1"]), "acc_raw": acc_raw,
                "fit_s": round(fit_seconds, 1),
            })
            print(f"  [RESULT] split{split} percentage{percentage:<3d} "
                  f"n_train={n_train:<5d} kappa={metrics['kappa']:+.4f} "
                  f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} "
                  f"acc_raw={acc_raw:.4f} "
                  f"fit_status={classifier.fit_status_} "
                  f"n_iter={n_iter} ({fit_seconds:.1f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {args.out}  ({len(rows)} linhas)")

    print("\n=== media por percentage (3 splits) ===")
    print(f"{'percentage':>10} {'kappa':>9} {'acc':>8} {'f1':>8} {'acc_raw':>8}")
    for percentage in args.percentages:
        selected = [row for row in rows if row["percentage"] == percentage]
        if selected:
            print(f"{percentage:>10} "
                  f"{np.mean([row['kappa'] for row in selected]):>+9.4f} "
                  f"{np.mean([row['acc'] for row in selected]):>8.4f} "
                  f"{np.mean([row['f1'] for row in selected]):>8.4f} "
                  f"{np.mean([row['acc_raw'] for row in selected]):>8.4f}")


if __name__ == "__main__":
    import sys

    main(**cli_kwargs(sys.argv[1:]))
