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
"""eval_autoencoder.py — SVM sobre o encoder do autoencoder TREINADO, um braco por rodada.

EXPERIMENTO VERSIONADO (nao e script descartavel). Re-executavel.

Pergunta: depois de reconstruir LAB[0,1] sem rotulo nenhum, o encoder do
autoencoder entrega mais kappa do que o FLIM cru que ele inicializou?

E o terceiro irmao de `eval_avg_pooling_48d.py` e `eval_svm_flim_flatten.py`.
Os tres compartilham dados, SVM e metrica; divergem em QUEM e o encoder:

  slot     | eval_avg_pooling_48d | eval_svm_flim_flatten | aqui
  ---------|----------------------|-----------------------|--------------------
  encoder  | FLIM cru, sem ckpt   | FLIM cru, sem ckpt     | ckpt do autoencoder
  features | GAP 48-d             | flatten 27.648-d       | segue o braco
  entrada  | build_test(200,F)   | IDEM                   | IDEM
  solver   | max_iter=-1          | IDEM                   | IDEM

BRACOS DE PESO. O braco e a identidade do run, gravada no proprio nome do
diretorio por `_run_name` (scripts/autoencoder_flim_ray.py:302-308):

  --weights lab       runs `..._lab`       embed_mode=avgpool2d  ->    48-d
  --weights lab_flat  runs `..._lab_flat`  embed_mode=flatten    -> 27.648-d

Os dois marcadores sao ortogonais: `_lab` diz que a entrada e LAB[0,1] cru
(`imagenet_norm=False`, o default do treino) e `_flat` diz que o SVM viu o mapa
conv3 inteiro. Como so existem em disco os bracos `_lab` e `_lab_flat`, a
entrada e sempre LAB cru aqui e o `--weights` decide so o `embed_mode` — que
NAO esta no checkpoint, so no nome do run e no `run_metadata.json`.

O avaliador oficial e IMPORTADO, nunca reescrito:
  * train_svm        src/utils/evaluate.py   (max_iter e embed_mode ja sao parametros)
  * extract_features src/utils/evaluate.py   (idem embed_mode)
  * fit_svm          src/utils/evaluate.py   (a UNICA SVC do repositorio)
  * compute_metrics  src/metrics/classification.py:31
  * rotulos          1-indexed no fit, predict(feats)-1  (eval/svm.py:135)

Encoder: `AutoEncoderFlimModule.load_from_checkpoint(...)` — o mesmo caminho que
o estagio 2 usa para partir do estagio 1 (autoencoder_flim_module.py:830). O
checkpoint carrega todos os hiperparametros do construtor, entao nenhum kwarg
precisa ser reconstruido a mao.

RESSALVA: o probe de treino (`_svm_probe`, autoencoder_flim_module.py:431)
pontua na **validacao**; este avaliador pontua no **teste**, como todo o resto de
`eval/svm_variants/`. Os dois numeros nao sao comparaveis entre si — compare cada
linha daqui com a curva FLIM cru dos irmaos, nao com `stage{N}/svm_kappa`.

Uso:
    python -m eval.svm_variants.eval_autoencoder --weights lab --dry-run

    OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
      conda run -n scalable_FLIM --no-capture-output \
      python -m eval.svm_variants.eval_autoencoder --weights lab \
        --out results/eval_autoencoder_lab.csv
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

_ROOT = PROJECT_ROOT

_PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae",
             "protozoan": "protozoan-cysts"}

# O manifesto que a fila de treino escreve (autoencoder_flim_ray.py:181) e a
# fonte de verdade da grade: run_name, dataset, split, percentage, stage, status
# e o checkpoint escolhido por estagio (best_recon.ckpt / best_kappa.ckpt).
_MANIFEST = os.path.join(_ROOT, "artifacts", "autoencoder_resnet_init_flim",
                         "run_manifest.csv")

# Sufixo do nome do run -> embed_mode com que aquele braco foi treinado e tem de
# ser avaliado (autoencoder_flim_ray.py:293,299).
_WEIGHTS = {"lab": "avgpool2d", "lab_flat": "flatten"}
_METHOD = {"lab": "SVM_AE_FLIM_48d_lab_raw",
           "lab_flat": "SVM_AE_FLIM_flatten27648_lab_raw"}

_FIELDS = ["method", "run_name", "stage", "dataset", "split", "percentage",
           "n_train", "n_test", "dim", "imagenet_norm", "max_iter",
           "fit_status", "n_iter_max", "n_iter_sum", "n_sv", "kappa", "acc",
           "f1", "acc_raw", "fit_s", "extract_s"]


def _runs(manifest: str, weights: str, datasets, splits, percentages, stages):
    """Linhas `ok` do manifesto que pertencem a este braco, ja filtradas.

    `endswith` separa os dois bracos sozinho: `_lab_flat` nao termina em `_lab`.
    """
    suffix = "_" + weights
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows
            if r["run_name"].endswith(suffix)
            and r["status"] == "ok"
            and (datasets is None or r["dataset"] in datasets)
            and (splits is None or int(r["split"]) in splits)
            and (percentages is None or int(r["percentage"]) in percentages)
            and (stages is None or int(r["stage"]) in stages)]


def _build_encoder(ckpt_path: str):
    """Encoder do autoencoder treinado, congelado para inferencia.

    `load_from_checkpoint` re-roda `__init__` (relendo os kernels FLIM do disco)
    e so entao aplica o state_dict — o mesmo contrato do estagio 2
    (autoencoder_flim_module.py:822-830). Os hiperparametros do construtor
    inteiros vem do proprio checkpoint, entao nao ha kwargs a reconstruir.
    """
    module = AutoEncoderFlimModule.load_from_checkpoint(ckpt_path,
                                                        map_location=DEVICE)
    enc = module.model.encoder.to(DEVICE).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _loader(dataset: str, split: int, pct: int, set_name: str,
            num_workers: int, one_hot: int = 0):
    """Dataloader identico ao dos irmaos (eval_svm_flim_flatten.py:161-172).

    `imagenet_norm=False` nao e escolha deste script: e o que os bracos `_lab`
    viram no treino (autoencoder_flim_ray.py:995-1002).
    """
    base = ParasiteDataset(
        set_name=set_name, split=split, percentage=pct,
        transform=build_test(IMAGE_SIZE, imagenet_norm=False),
        loader="ift_lab", path_dataset=_PARASITE[dataset],
    )
    ds = _OneHotDataset(base, one_hot) if one_hot else base
    return DataLoader(ds, batch_size=32, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main(
    weights: str | None = None,
    out: str | None = None,
    manifest: str = _MANIFEST,
    datasets: list[str] | None = None,
    splits: list[int] | None = None,
    percentages: list[int] | None = None,
    stages: list[int] | None = None,
    max_iter: int = -1,
    num_workers: int = 8,
    dry_run: bool = False,
) -> None:
    """Um parametro por flag do argparse antigo, com o mesmo default.

    `weights` nao tem default porque era `required=True`; sem ele a funcao
    aborta como o parser abortava. Os quatro `nargs="+"` mantem o default
    `None` da origem — quem consome (`_runs`) ja trata `None` como "tudo".
    """
    if weights not in _WEIGHTS:
        raise SystemExit(
            f"--weights e obrigatorio e deve ser um de {sorted(_WEIGHTS)}; "
            f"recebi {weights!r}")
    # `nargs="+"` do argparse sempre devolvia lista; `cli_kwargs` devolve escalar
    # quando so um valor e passado. A normalizacao mora aqui, na propria funcao.
    datasets = [datasets] if isinstance(datasets, str) else datasets
    splits = [splits] if isinstance(splits, int) else splits
    percentages = [percentages] if isinstance(percentages, int) else percentages
    stages = [stages] if isinstance(stages, int) else stages
    # Shim de uma linha: o corpo abaixo e o da origem, que lia `args.<flag>`.
    # `locals()` na primeira linha viva e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    embed_mode = _WEIGHTS[args.weights]
    out = args.out or os.path.join(_ROOT, "results",
                                   f"eval_autoencoder_{args.weights}.csv")
    runs = _runs(args.manifest, args.weights, args.datasets, args.splits,
                 args.percentages, args.stages)

    print(f"[CONTRATO] weights ={args.weights} -> embed_mode='{embed_mode}'")
    print(f"[CONTRATO] encoder =load_from_checkpoint(<run>/checkpoints/*.ckpt)")
    print(f"[CONTRATO] entrada =build_test({IMAGE_SIZE}, imagenet_norm=False)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter})")
    print(f"[CONTRATO] {len(runs)} runs | device={DEVICE} | out={out}\n")

    if args.dry_run:
        ok = True
        for r in runs:
            flag = os.path.exists(r["checkpoint_path"])
            ok &= flag
            print(f"  [{'OK ' if flag else 'FALTA'}] {r['run_name']}  "
                  f"-> {os.path.basename(r['checkpoint_path'])}")
        print(f"\n[DRY-RUN] {len(runs)} celulas | "
              f"{'todos os checkpoints presentes' if ok else 'FALTAM CHECKPOINTS'}")
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = []
    # Escrita incremental: uma rodada do braco flatten leva horas e o CSV nao
    # pode morrer junto com ela (eval_svm_flim_flatten.py:271-276 escreve so no
    # fim e a rodada interrompida nao deixou arquivo).
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()

        for r in runs:
            ds, sp, pct = r["dataset"], int(r["split"]), int(r["percentage"])
            n_cls = NUM_CLASSES[ds]
            enc = _build_encoder(r["checkpoint_path"])

            test_loader = _loader(ds, sp, pct, "test", args.num_workers)
            t0 = time.perf_counter()
            feats_te, y_true = extract_features(enc, test_loader,
                                                embed_mode=embed_mode)
            extract_s = time.perf_counter() - t0
            print(f"[INFO] {r['run_name']}: test n={len(y_true)} "
                  f"dim={feats_te.shape[1]} ({extract_s:.1f}s)")

            tr_loader = _loader(ds, sp, pct, "train", args.num_workers,
                                one_hot=n_cls)
            n_train = len(tr_loader.dataset)

            t0 = time.perf_counter()
            clf = train_svm(enc, tr_loader, max_iter=args.max_iter,
                            embed_mode=embed_mode)
            fit_s = time.perf_counter() - t0

            y_pred = clf.predict(feats_te) - 1          # svm.py:135
            m = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_cls)
            acc_raw = float((y_pred == y_true).mean())
            diag = clf.fit_diagnostics_

            row = {
                "method": _METHOD[args.weights], "run_name": r["run_name"],
                "stage": int(r["stage"]), "dataset": ds, "split": sp,
                "percentage": pct, "n_train": n_train, "n_test": len(y_true),
                "dim": feats_te.shape[1], "imagenet_norm": False,
                "max_iter": args.max_iter,
                "fit_status": diag["svm_fit_status"],
                "n_iter_max": diag["svm_n_iter_max"],
                "n_iter_sum": diag["svm_n_iter_sum"], "n_sv": diag["svm_n_sv"],
                "kappa": float(m["kappa"]), "acc": float(m["acc"]),
                "f1": float(m["f1"]), "acc_raw": acc_raw,
                "fit_s": round(fit_s, 1), "extract_s": round(extract_s, 1),
            }
            writer.writerow(row)
            fh.flush()
            rows.append(row)

            print(f"  [RESULT] {r['run_name']} n_train={n_train:<5d} "
                  f"kappa={m['kappa']:+.4f} acc={m['acc']:.4f} "
                  f"f1={m['f1']:.4f} acc_raw={acc_raw:.4f} "
                  f"fit_status={diag['svm_fit_status']} "
                  f"n_iter={diag['svm_n_iter_max']} n_sv={diag['svm_n_sv']} "
                  f"({fit_s:.1f}s)")

    print(f"\n[OK] {out}  ({len(rows)} linhas)")

    for stage in sorted({r["stage"] for r in rows}):
        print(f"\n=== stage {stage}: media por (dataset, percentage) ===")
        print(f"{'dataset':>10} {'pct':>5} {'kappa':>9} {'acc':>8} {'f1':>8} "
              f"{'acc_raw':>8}")
        for ds in sorted({r["dataset"] for r in rows}):
            for pct in sorted({r["percentage"] for r in rows}):
                sel = [r for r in rows if r["stage"] == stage
                       and r["dataset"] == ds and r["percentage"] == pct]
                if sel:
                    print(f"{ds:>10} {pct:>5} "
                          f"{np.mean([r['kappa'] for r in sel]):>+9.4f} "
                          f"{np.mean([r['acc'] for r in sel]):>8.4f} "
                          f"{np.mean([r['f1'] for r in sel]):>8.4f} "
                          f"{np.mean([r['acc_raw'] for r in sel]):>8.4f}")


if __name__ == "__main__":
    import sys

    main(**cli_kwargs(sys.argv[1:]))
