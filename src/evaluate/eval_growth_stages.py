"""eval_growth_stages.py — SVM de TESTE sobre os encoders das grades de crescimento SPiFiL.

EXPERIMENTO VERSIONADO (nao e script descartavel). Re-executavel.

Pergunta: ao longo da escada `stage1 -> stage2 -> round1_stage3 -> round1_stage4
-> round2_stage3 -> round2_stage4`, cada camada nova enxertada pelo SPiFiL
entrega mais kappa **no teste**?

E o irmao de `eval_autoencoder.py` para a grade de crescimento. O fluxo (ordem
das chamadas, `-1` no predict, escrita incremental, formato do CSV) e copiado
dele; o que muda e so DE ONDE vem a grade e QUANTOS rotulos de estagio existem:

  slot     | eval_autoencoder            | aqui
  ---------|-----------------------------|--------------------------------------
  grade    | run_manifest.csv            | glob de artifacts/spifil_growth/<family>
  estagios | 1, 2                        | 6 rotulos (round0..2 x stage1..4)
  encoder  | ckpt do autoencoder         | IDEM
  features | segue o braco               | sempre flatten (channels[-1]*24*24)
  entrada  | _build_test(200, F)         | IDEM (o `_loader` dele, importado)
  solver   | max_iter=-1                 | IDEM

O avaliador oficial e IMPORTADO, nunca reescrito — e o mesmo SVM unico do
repositorio, com os mesmos hiperparametros (`tools/check_refactor_equivalence.py`
trava isso):
  * train_svm        src/utils/evaluate.py:437  (extrai features de treino e ajusta)
  * extract_features src/utils/evaluate.py:505
  * fit_svm          src/utils/evaluate.py:372  (a UNICA SVC: C=1e2, linear, ovo)
  * compute_metrics  src/metrics/classification.py:31
  * rotulos          1-indexed no fit, predict(feats)-1  (src/evaluate/svm.py:135)

TRES ARMADILHAS ja pagas aqui, para quem for mexer:

  1. O rotulo do estagio e o NOME DO DIRETORIO, nunca o campo `stage` do
     metadado — esse vale 1/2/3, confunde `stage2` com `round1_stage4` e nao
     tem numero para o `round<R>_head`.
  2. Nos estagios 3 e 4 os hparams do checkpoint guardam caminho RELATIVO para
     `<braco>/round{r}_grow/`. Sem override absoluto, so carrega com cwd na raiz.
  3. `EMBED_MODE` e global de modulo e o default e `avgpool2d`. Todos os runs
     de todas as grades treinaram com `flatten`; sem o rebind daria 48-d e o
     numero nao teria relacao nenhuma com o treino.

RESSALVA: o `flim_ref_svm_kappa` / `best_svm_kappa` do `run_metadata.json`
pontuam na **validacao**; este avaliador pontua no **teste**. Nao devem bater —
se batessem, era sinal de conjunto trocado.

Uso::

    python -m src.evaluate.eval_growth_stages --pct 5 --dry-run

    python -m src.evaluate.eval_growth_stages --pct 5 \\
      --family g5_in_feature g5_in_image --dry-run

    OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE \\
      python -m src.evaluate.eval_growth_stages --pct 5 --device cuda:0

    python -m src.evaluate.eval_growth_stages --pct 50 --dataset larvae \\
      --stage round2_stage4 --skip-existing
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Tres niveis: src/evaluate/<este arquivo> -> src/evaluate -> src -> raiz.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tqdm import tqdm                                                   # noqa: E402

import src.utils.evaluate as _ev                                        # noqa: E402
from src.evaluate.eval_autoencoder import _loader                       # noqa: E402
from src.metrics.classification import compute_metrics                  # noqa: E402
from src.modules.autoencoder_flim_module import (                       # noqa: E402
    NUM_CLASSES, AutoEncoderFlimModule,
)
from src.utils.evaluate import extract_features, train_svm              # noqa: E402

# Raiz das grades: cada familia e um subdiretorio daqui, escolhido por --family.
_GRID = os.path.join(_ROOT, "artifacts", "spifil_growth")
_METHOD = "SVM_SPiFiL_growth_flatten_labcru"

# Todos os runs de todas as grades treinaram com flatten
# (run_metadata.json:"embed_mode").
_EMBED_MODE = "flatten"

# `<dataset>_split<N>_pct<P>` e `[round<R>_]stage<S>` ou `round<R>_head`.
_ARM = re.compile(r"^(eggs|larvae|protozoan)_split(\d+)_pct(\d+)$")
# O `head` (encoder destravado + Head supervisionada) fecha a rodada no lugar do par
# 3+4, entao ordena como estagio 5 dela: depois do 4, antes da rodada seguinte. E dai o
# `or 5` de quem le o grupo 2, que vem vazio no head.
_STAGE = re.compile(r"^(?:round(\d+)_)?(?:stage(\d)|head)$")

_FIELDS = ["method", "run_name", "stage_label", "round", "dataset", "split",
           "percentage", "n_train", "n_test", "dim", "embed_mode",
           "imagenet_norm", "max_iter", "fit_status", "n_iter_max",
           "n_iter_sum", "n_sv", "kappa", "acc", "f1", "acc_raw", "extract_s",
           "fit_s", "predict_s", "ckpt_epoch", "ckpt"]


def _jobs(pct: int, families, stages, datasets, splits):
    """Os trabalhos das grades pedidas, em ordem cronologica de estagio.

    A existencia de `checkpoints/best_kappa.ckpt` e o proprio filtro: os
    diretorios `*_grow` (que so guardam a arquitetura enxertada pelo SPiFiL) nao
    tem checkpoint nenhum, e um `-not -path '*_grow*'` descartaria a grade
    inteira, porque `spifil_growth` ja casa com `_grow`.
    """
    out = []
    for fam in families:
        for ckpt in glob.glob(os.path.join(_GRID, fam, "*", "*", "checkpoints",
                                           "best_kappa.ckpt")):
            stage_dir = os.path.dirname(os.path.dirname(ckpt))
            arm = _ARM.match(os.path.basename(os.path.dirname(stage_dir)))
            st = _STAGE.match(os.path.basename(stage_dir))
            if not arm or not st:
                continue
            job = {"ckpt": ckpt, "stage_dir": stage_dir, "family": fam,
                   "stage_label": os.path.basename(stage_dir),
                   "round": int(st[1] or 0), "stage_num": int(st[2] or 5),
                   "dataset": arm[1], "split": int(arm[2]),
                   "percentage": int(arm[3])}
            if (job["percentage"] == pct
                    and (stages is None or job["stage_label"] in stages)
                    and (datasets is None or job["dataset"] in datasets)
                    and (splits is None or job["split"] in splits)):
                out.append(job)
    return sorted(out, key=lambda j: (j["family"], j["round"], j["stage_num"],
                                      j["dataset"], j["split"]))


def _encoder(job: dict, device):
    """Encoder do estagio, congelado para inferencia — e a epoca do checkpoint.

    `load_from_checkpoint` re-roda `__init__` (relendo os kernels FLIM do disco)
    e so entao aplica o state_dict, entao os caminhos guardados nos hparams
    precisam existir. Nos estagios 3 e 4 eles apontam para
    `<braco>/round{r}_grow/` de forma RELATIVA: sao reancorados em `_ROOT` aqui,
    para o script rodar de qualquer cwd.
    """
    blob = torch.load(job["ckpt"], map_location="cpu", weights_only=False)
    hp = blob["hyper_parameters"]

    override = {k: os.path.join(_ROOT, hp[k])
                for k in ("arch_json", "flim_weights_path")
                if hp.get(k) and not os.path.isabs(hp[k])}

    module = AutoEncoderFlimModule.load_from_checkpoint(
        job["ckpt"], map_location=device, **override)
    enc = module.model.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, int(blob.get("epoch", -1))


def _done(out: str):
    """Chaves `(familia, stage_label, dataset, split)` ja no CSV, para retomar.

    A familia nao e coluna (coluna nova desalinharia o append contra os CSVs de
    26 campos ja escritos): sai do `ckpt`, que e
    `artifacts/spifil_growth/<FAMILY>/<braco>/<estagio>/checkpoints/...`. CSV
    velho, so de grid4, resolve para "grid4" sozinho.
    """
    if not os.path.exists(out):
        return set()
    with open(out, newline="") as fh:
        return {(r["ckpt"].split(os.sep)[2], r["stage_label"], r["dataset"],
                 int(r["split"])) for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pct", required=True, type=int, choices=[5, 50],
                    help="Porcentagem de treino do SVM. Define o CSV de saida — "
                         "um por porcentagem, para pct5 e pct50 rodarem em "
                         "paralelo sem disputar o mesmo arquivo. O TESTE e "
                         "byte-identico entre as duas.")
    ap.add_argument("--family", nargs="+", default=["grid4"],
                    help="Grades de artifacts/spifil_growth/ a varrer. A "
                         "familia e o basename do --work-dir do "
                         "scripts/spifil_growth_loop.py (`_exp()`, linha 122): "
                         "grid3, grid4, g5_in_feature, g5_in_image. Sem "
                         "`choices` travado — grade nova entra sem editar isto.")
    ap.add_argument("--stage", nargs="+", default=None,
                    choices=["stage1", "stage2", "round1_stage3",
                             "round1_stage4", "round2_stage3", "round2_stage4",
                             "round1_head", "round2_head"],
                    help="Default: todos os rotulos presentes no grid.")
    ap.add_argument("--dataset", nargs="+", default=None,
                    choices=["eggs", "larvae", "protozoan"])
    ap.add_argument("--split", nargs="+", type=int, default=None,
                    choices=[1, 2, 3])
    ap.add_argument("--device", default=None,
                    help="cuda:N ou cpu (default: cuda se disponivel).")
    ap.add_argument("--csv", "--out", dest="out", default=None,
                    help="Override do CSV (default: results/eval_growth_stages"
                         "[_<familia>...]_pct<PCT>.csv; a familia so entra no "
                         "nome quando nao e a grid4, para o caminho de hoje "
                         "ficar identico). `--out` e alias historico.")
    ap.add_argument("--max-iter", type=int, default=-1,
                    help="-1 = convergido. Um teto positivo fica declarado no CSV.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Pula linha ja presente no CSV e acrescenta o resto.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Lista os trabalhos selecionados e sai.")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    # Os dois globais que governam o avaliador oficial. Rebindar e o contrato do
    # modulo (evaluate.py:245-248) — e o que o proprio treino faz em
    # autoencoder_flim_module.py:909.
    _ev.EMBED_MODE = _EMBED_MODE
    _ev.DEVICE = device

    # Sem `choices`, quem valida e o disco — e erra com a lista do que existe.
    for fam in args.family:
        if not os.path.isdir(os.path.join(_GRID, fam)):
            ap.error(f"--family {fam!r}: nao existe {os.path.join(_GRID, fam)}. "
                     f"Grades em disco: {' '.join(sorted(os.listdir(_GRID)))}")

    tag = "" if args.family == ["grid4"] else "_" + "_".join(args.family)
    out = args.out or os.path.join(_ROOT, "results",
                                   f"eval_growth_stages{tag}_pct{args.pct}.csv")
    jobs = _jobs(args.pct, args.family, args.stage, args.dataset, args.split)

    print(f"[CONTRATO] embed_mode='{_EMBED_MODE}' (o do treino; default do "
          f"modulo seria '{_ev.DEFAULT_EMBED_MODE}')")
    print(f"[CONTRATO] rotulo  = nome do diretorio, nao o campo 'stage' do metadado")
    print(f"[CONTRATO] encoder = load_from_checkpoint(<estagio>/checkpoints/best_kappa.ckpt)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter}) -> fit_svm")
    print(f"[CONTRATO] {len(jobs)} trabalhos | family={' '.join(args.family)} "
          f"| pct={args.pct} | device={device} | out={out}\n")

    if args.dry_run:
        for j in jobs:
            print(f"  [{'OK ' if os.path.exists(j['ckpt']) else 'FALTA'}] "
                  f"{j['family'] + '  ' if len(args.family) > 1 else ''}"
                  f"{j['stage_label']:>13}  {j['dataset']:>9} split{j['split']}")
        print(f"\n[DRY-RUN] {len(jobs)} trabalhos")
        return

    skip = _done(out) if args.skip_existing else set()
    jobs = [j for j in jobs
            if (j["family"], j["stage_label"], j["dataset"], j["split"])
            not in skip]
    if skip:
        print(f"[SKIP] {len(skip)} linhas ja no CSV; restam {len(jobs)}\n")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # `getsize > 0`, nao so `exists`: um CSV vazio (tmux morto antes da primeira
    # linha) entraria em modo append e as linhas sairiam SEM cabecalho.
    append = args.skip_existing and os.path.exists(out) and os.path.getsize(out) > 0
    rows = []
    with open(out, "a" if append else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        if not append:
            writer.writeheader()

        bar = tqdm(jobs, desc="growth stages", unit="job")
        for j in bar:
            ds, sp = j["dataset"], j["split"]
            # A familia entra no run_name so quando nao e grid4: assim a grid4
            # mantem a string exata de hoje e as outras casam com o nome de run
            # do laco de treino (spifil_growth_loop.py:233).
            fam = "" if j["family"] == "grid4" else f"{j['family']}_"
            bar.set_postfix_str(
                f"{j['family'] + ' ' if len(args.family) > 1 else ''}"
                f"{j['stage_label']} {ds} split{sp}")
            n_cls = NUM_CLASSES[ds]
            enc, epoch = _encoder(j, device)

            t0 = time.perf_counter()
            feats_te, y_true = extract_features(enc,
                                                _loader(ds, sp, args.pct, "test",
                                                        args.num_workers))
            extract_s = time.perf_counter() - t0

            tr_loader = _loader(ds, sp, args.pct, "train", args.num_workers,
                                one_hot=n_cls)
            n_train = len(tr_loader.dataset)

            t0 = time.perf_counter()
            clf = train_svm(enc, tr_loader, max_iter=args.max_iter)
            fit_s = time.perf_counter() - t0

            # O `-1` desfaz o `+1` que train_svm aplica: o dataset e 1-indexed,
            # compute_metrics espera 0-indexed (svm.py:135).
            t0 = time.perf_counter()
            y_pred = clf.predict(feats_te) - 1
            predict_s = time.perf_counter() - t0

            m = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=n_cls)
            acc_raw = float((y_pred == y_true).mean())
            diag = clf.fit_diagnostics_

            row = {
                "method": _METHOD,
                "run_name": f"spifil_growth_{fam}{ds}_split{sp}_"
                            f"pct{args.pct}_{j['stage_label']}",
                "stage_label": j["stage_label"], "round": j["round"],
                "dataset": ds, "split": sp, "percentage": args.pct,
                "n_train": n_train, "n_test": len(y_true),
                "dim": feats_te.shape[1], "embed_mode": _EMBED_MODE,
                "imagenet_norm": False, "max_iter": args.max_iter,
                "fit_status": diag["svm_fit_status"],
                "n_iter_max": diag["svm_n_iter_max"],
                "n_iter_sum": diag["svm_n_iter_sum"], "n_sv": diag["svm_n_sv"],
                "kappa": float(m["kappa"]), "acc": float(m["acc"]),
                "f1": float(m["f1"]), "acc_raw": acc_raw,
                "extract_s": round(extract_s, 1), "fit_s": round(fit_s, 1),
                "predict_s": round(predict_s, 1), "ckpt_epoch": epoch,
                "ckpt": os.path.relpath(j["ckpt"], _ROOT),
            }
            writer.writerow(row)
            fh.flush()
            rows.append(row)

            tqdm.write(f"  [RESULT] {row['run_name']} dim={row['dim']} "
                       f"n_train={n_train:<5d} kappa={m['kappa']:+.4f} "
                       f"acc={m['acc']:.4f} f1={m['f1']:.4f} "
                       f"fit_status={diag['svm_fit_status']} "
                       f"n_sv={diag['svm_n_sv']} epoch={epoch} "
                       f"({extract_s:.1f}s/{fit_s:.1f}s/{predict_s:.1f}s)")

    print(f"\n[OK] {out}  ({len(rows)} linhas novas)")

    for label in sorted({r["stage_label"] for r in rows},
                        key=lambda s: (int(_STAGE.match(s)[1] or 0),
                                       int(_STAGE.match(s)[2] or 5))):
        sel = [r for r in rows if r["stage_label"] == label]
        print(f"{label:>13}  n={len(sel):<3d} "
              f"kappa={np.mean([r['kappa'] for r in sel]):+.4f} "
              f"acc={np.mean([r['acc'] for r in sel]):.4f} "
              f"f1={np.mean([r['f1'] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
