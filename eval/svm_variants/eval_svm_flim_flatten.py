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
"""eval_svm_flim_flatten.py — conv3 achatado 27.648-d + LAB[0,1] cru + solver convergido.

EXPERIMENTO VERSIONADO (nao e script descartavel). Re-executavel.

Este e o eval da curva LARANJA de
`artifacts/plots/comparacao_flim_protocolo_original/comparacao_kappa_registrado_vs_reproducao.png`.
O codigo original eram dois scripts `tools/diag_*.py` (`diag_flim_gap_vs_flatten.py`
para protozoan, `diag_g2_flatten_curve.py` para eggs/larvae) que a regra
`tools/diag_*.py` do .gitignore apagou sem passar pelo historico. Sobraram so as
saidas, em `dados_brutos/g1/pct*.json` e `dados_brutos/g2_{eggs,larvae}.json`.
Este arquivo reconstroi o eval a partir do protocolo declarado no README daquela
pasta e se verifica contra aqueles JSONs (`--compare`).

Pergunta: o encoder FLIM em disco reproduz a curva de kappa do CSV externo
(`data/reports_felipe/svm/`) quando avaliado com o protocolo original?

E o irmao de `eval_avg_pooling_48d.py`. Os dois compartilham encoder, dados, SVM e
metrica; divergem em UMA linha — como o mapa conv3 vira vetor:

  slot       | oficial (src/utils/evaluate.py) | eval_avg_pooling_48d | aqui
  -----------|---------------------------------|-------------------|------------------
  features   | AdaptiveAvgPool2d(1) -> 48-d    | IDEM (48-d)       | flatten -> 27.648-d
  entrada    | build_test(200, True)          | (200, False)      | (200, False)
  solver     | max_iter=10000                  | max_iter=-1       | max_iter=-1

As 27.648 dimensoes sao 48 x 24 x 24 — o mapa conv3 inteiro, sem pooling. A troca
e feita passando `embed_mode="flatten"` nas chamadas do avaliador oficial, que
repassam para `_encode_pooled`. Nenhuma funcao e reescrita, nenhum estado global
e alterado.

O avaliador oficial e IMPORTADO, nunca reescrito:
  * train_svm        src/utils/evaluate.py       (max_iter e embed_mode ja sao parametros)
  * extract_features src/utils/evaluate.py       (idem embed_mode)
  * _encode_pooled   src/utils/evaluate.py       (e quem le o embed_mode)
  * compute_metrics  src/metrics/classification.py:31
  * rotulos          1-indexed no fit, predict(feats)-1  (eval/svm.py:135)

Encoder: FLIM cru, sem checkpoint — mesmo caminho que o estagio 1 congela
(src/modules/autoencoder_flim_module.py:155-168).

Uso:
    python -m eval.svm_variants.eval_svm_flim_flatten --dry-run
    OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
      conda run -n scalable_FLIM --no-capture-output \
      python -m eval.svm_variants.eval_svm_flim_flatten --datasets larvae eggs protozoan --compare
"""

import csv
from types import SimpleNamespace
import glob
import json
import os
import re
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
from experiments.ray.paths import arch_json, flim_weights_path  # noqa: E402
from experiments.constants import PARASITE_DIR  # noqa: E402

_ROOT = PROJECT_ROOT

_PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae",
             "protozoan": "protozoan-cysts"}

# Onde vivem os dados das curvas ja plotadas — o CSV novo fica ao lado, e os
# JSONs de referencia (a saida dos diag_* apagados) ficam em dados_brutos/.
_CURVE_DIR = os.path.join(_ROOT, "artifacts", "plots",
                          "comparacao_flim_protocolo_original")
_OUT_CSV = os.path.join(_CURVE_DIR, "eval_svm_flim_flatten.csv")
_REF_DIR = os.path.join(_CURVE_DIR, "dados_brutos")

_METHOD = "SVM_FLIM_flatten27648_lab_raw"
# Passado explicitamente para extract_features/train_svm — e a unica divergencia
# em relacao a eval_avg_pooling_48d.py.
_EMBED_MODE = "flatten"
_FIELDS = ["method", "dataset", "split", "percentage", "n_train", "n_test",
           "dim", "imagenet_norm", "max_iter", "fit_status", "n_iter_max",
           "n_iter_sum", "n_sv", "kappa", "acc", "f1", "acc_raw", "fit_s",
           "extract_s"]


def _reference() -> dict:
    """kappa por celula (dataset, split, pct) dos JSONs dos diag_* apagados.

    Os dois esquemas divergem: g1 traz feat/norm/solver/pool_padding e o pct no
    nome do arquivo; g2 traz `pct` e nada de configuracao (por construcao da
    missao). O filtro e o mesmo de build_comparacao_csv.py.
    """
    ref = {}
    for path in sorted(glob.glob(os.path.join(_REF_DIR, "g1", "pct*.json"))):
        pct = int(re.search(r"pct(\d+)\.json$", path).group(1))
        for r in json.load(open(path)):
            if (r.get("feat"), r.get("norm"), r.get("solver")) != ("B", "N2", "S2"):
                continue
            if r.get("pool_padding", 0) != 0 or r["dim"] != 27648:
                continue
            ref[(r["dataset"], r["split"], pct)] = r["kappa"]
    for name in ("g2_eggs.json", "g2_larvae.json"):
        path = os.path.join(_REF_DIR, name)
        if os.path.exists(path):
            for r in json.load(open(path)):
                if r["dim"] == 27648:
                    ref[(r["dataset"], r["split"], r["pct"])] = r["kappa"]
    return ref


def _build_encoder(dataset: str, split: int):
    """Encoder FLIM cru — o mesmo que o estagio 1 congela, sem checkpoint."""
    mod = AutoEncoderFlimModule(
        arch_json=arch_json(PARASITE_DIR[dataset], split),
        dataset=dataset,
        flim_weights_path=flim_weights_path(PARASITE_DIR[dataset], split),
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
    base = ParasiteDataset(
        set_name=set_name, split=split, percentage=pct,
        transform=build_test(IMAGE_SIZE, imagenet_norm=False),
        loader="ift_lab", path_dataset=_PARASITE[dataset],
    )
    ds = _OneHotDataset(base, one_hot) if one_hot else base
    return DataLoader(ds, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main(
    datasets: list[str] | None = None,
    splits: list[int] | None = None,
    percentages: list[int] | None = None,
    max_iter: int = -1,
    num_workers: int = 8,
    out: str = _OUT_CSV,
    compare: bool = False,
    dry_run: bool = False,
) -> None:
    """Um parametro por flag do argparse antigo, com o mesmo default.

    Os tres `nargs="+"` saem como `None` na assinatura para nao deixar lista
    mutavel de default; o valor de origem e restaurado logo abaixo. O que os
    `choices=` validavam agora e responsabilidade de quem chama.
    """
    # `nargs="+"` do argparse sempre devolvia lista; `cli_kwargs` devolve escalar
    # quando so um valor e passado. A normalizacao mora aqui, na propria funcao.
    datasets = ["protozoan", "eggs", "larvae"] if datasets is None else (
        [datasets] if isinstance(datasets, str) else datasets)
    splits = [1, 2, 3] if splits is None else (
        [splits] if isinstance(splits, int) else splits)
    percentages = [1, 5, 25, 50, 75, 100] if percentages is None else (
        [percentages] if isinstance(percentages, int) else percentages)
    # Shim de uma linha: o corpo abaixo e o da origem, que lia `args.<flag>`.
    # `locals()` na primeira linha viva e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    # A UNICA divergencia em relacao a eval_avg_pooling_48d.py: o modo de embedding
    # vai explicito em cada chamada do avaliador oficial (nada de estado global).
    print(f"[CONTRATO] features=embed_mode='{_EMBED_MODE}' -> flatten(conv3) = 48*24*24")
    print(f"[CONTRATO] entrada =build_test({IMAGE_SIZE}, imagenet_norm=False)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter})")
    print(f"[CONTRATO] datasets={args.datasets} | device={DEVICE}\n")

    if args.dry_run:
        ok = True
        for ds in args.datasets:
            for sp in args.splits:
                aj, wp = arch_json(PARASITE_DIR[ds], sp), flim_weights_path(PARASITE_DIR[ds], sp)
                for path in (aj, wp):
                    flag = os.path.exists(path)
                    ok &= flag
                    print(f"  [{'OK ' if flag else 'FALTA'}] {path}")
                for pct in args.percentages:
                    try:
                        n = len(ParasiteDataset(set_name="train", split=sp, percentage=pct,
                                                transform=None, loader="ift_lab",
                                                path_dataset=_PARASITE[ds]))
                        print(f"  [OK ] {ds} split{sp} pct{pct:<3d} train={n}")
                    except Exception as exc:                   # noqa: BLE001
                        ok = False
                        print(f"  [FALTA] {ds} split{sp} pct{pct}: {exc}")
        ref = _reference()
        print(f"  [REF] {len(ref)} celulas de referencia em {_REF_DIR}")
        print(f"\n[DRY-RUN] {'tudo presente' if ok else 'FALTAM ARQUIVOS'}")
        return

    ref = _reference() if args.compare else {}
    rows = []
    for ds in args.datasets:
        n_cls = NUM_CLASSES[ds]
        for sp in args.splits:
            enc = _build_encoder(ds, sp)
            test_loader = _loader(ds, sp, args.percentages[0], "test", args.num_workers)
            t0 = time.perf_counter()
            feats_te, y_true = extract_features(enc, test_loader, embed_mode=_EMBED_MODE)
            extract_s = time.perf_counter() - t0
            print(f"[INFO] {ds} split{sp}: test n={len(y_true)} dim={feats_te.shape[1]} "
                  f"({extract_s:.1f}s)")

            for pct in args.percentages:
                tr_loader = _loader(ds, sp, pct, "train", args.num_workers, one_hot=n_cls)
                n_train = len(tr_loader.dataset)

                t0 = time.perf_counter()
                clf = train_svm(enc, tr_loader, max_iter=args.max_iter,
                                embed_mode=_EMBED_MODE)
                fit_s = time.perf_counter() - t0

                y_pred = clf.predict(feats_te) - 1          # svm.py:135
                m = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_cls)
                acc_raw = float((y_pred == y_true).mean())
                diag = clf.fit_diagnostics_

                rows.append({
                    "method": _METHOD, "dataset": ds, "split": sp, "percentage": pct,
                    "n_train": n_train, "n_test": len(y_true), "dim": feats_te.shape[1],
                    "imagenet_norm": False, "max_iter": args.max_iter,
                    "fit_status": diag["svm_fit_status"],
                    "n_iter_max": diag["svm_n_iter_max"],
                    "n_iter_sum": diag["svm_n_iter_sum"], "n_sv": diag["svm_n_sv"],
                    "kappa": float(m["kappa"]), "acc": float(m["acc"]),
                    "f1": float(m["f1"]), "acc_raw": acc_raw,
                    "fit_s": round(fit_s, 1), "extract_s": round(extract_s, 1),
                })
                tail = ""
                if (ds, sp, pct) in ref:
                    dk = float(m["kappa"]) - ref[(ds, sp, pct)]
                    tail = f" | ref={ref[(ds, sp, pct)]:+.4f} dk={dk:+.4f}"
                print(f"  [RESULT] {ds} sp{sp} pct{pct:<3d} n_train={n_train:<5d} "
                      f"kappa={m['kappa']:+.4f} acc={m['acc']:.4f} f1={m['f1']:.4f} "
                      f"acc_raw={acc_raw:.4f} fit_status={diag['svm_fit_status']} "
                      f"n_iter={diag['svm_n_iter_max']} n_sv={diag['svm_n_sv']} "
                      f"({fit_s:.1f}s){tail}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {args.out}  ({len(rows)} linhas)")

    for ds in args.datasets:
        print(f"\n=== {ds}: media por percentage ({len(args.splits)} splits) ===")
        print(f"{'pct':>5} {'kappa':>9} {'acc':>8} {'f1':>8} {'acc_raw':>8}")
        for pct in args.percentages:
            sel = [r for r in rows if r["dataset"] == ds and r["percentage"] == pct]
            if sel:
                print(f"{pct:>5} {np.mean([r['kappa'] for r in sel]):>+9.4f} "
                      f"{np.mean([r['acc'] for r in sel]):>8.4f} "
                      f"{np.mean([r['f1'] for r in sel]):>8.4f} "
                      f"{np.mean([r['acc_raw'] for r in sel]):>8.4f}")

    if ref:
        dks = [r["kappa"] - ref[(r["dataset"], r["split"], r["percentage"])]
               for r in rows if (r["dataset"], r["split"], r["percentage"]) in ref]
        if dks:
            a = np.abs(dks)
            print(f"\n=== vs dados_brutos ({len(dks)} celulas) ===")
            print(f"MAE(kappa)={a.mean():.6f}  max|dk|={a.max():.6f}  "
                  f"identicas(<1e-6)={int((a < 1e-6).sum())}/{len(dks)}")


if __name__ == "__main__":
    import sys

    main(**cli_kwargs(sys.argv[1:]))
