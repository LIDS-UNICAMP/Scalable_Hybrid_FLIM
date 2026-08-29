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

    # VALIDACAO (nao passa por checkpoint: o numero ja existe no W&B)
    python -m src.evaluate.eval_growth_stages --pct 5 --fetch-wandb \\
      --family g5_random_in_feature g5_head_larvae
"""

import argparse
import csv
import glob
import json
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
           "fit_s", "predict_s", "ckpt_epoch", "ckpt",
           # No FIM de proposito: coluna no meio desalinharia o append de
           # `--skip-existing` contra os CSVs de 26 campos ja escritos. Vazias
           # em quem nao tem Head (`restval=""` do DictWriter).
           "head_kappa", "head_acc", "head_f1"]

# O CSV de VALIDACAO e outro artefato: outra medida (a sonda do proprio treino, no
# conjunto de validacao), outro schema. `kappa`/`acc`/`f1` sem prefixo de proposito —
# o plot resolve a coluna com `metric.rsplit("_", 1)[-1]`.
_VAL_METHOD = "WANDB_probe_svm_val"
_VAL_FIELDS = ["method", "run_name", "stage_label", "round", "dataset", "split",
               "percentage", "eval_split", "family", "run_id", "run_state",
               "stage_epoch", "kappa", "acc", "f1", "val_recon_loss",
               "head_kappa", "head_acc", "flim_ref_kappa", "ckpt"]
# Rodar duas vezes SOBRESCREVE: quem colide nesta chave sai do CSV velho antes do concat.
_VAL_KEY = ["family", "run_name", "stage_label", "split", "percentage", "eval_split"]


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
    """Encoder e Head do estagio, congelados — e a epoca do checkpoint.

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
    # `self.head = Head(...) if head_finetune else None` (autoencoder_flim_module.py:311),
    # entao `is None` ja E o teste de `head_finetune`. Fica na CPU: e um Linear
    # de 48 entradas, e assim consome `feats_te` sem copiar nada para a GPU.
    head = module.head.cpu().eval() if module.head is not None else None
    return enc, head, int(blob.get("epoch", -1))


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


def _summary(rows) -> None:
    """Media por rotulo de estagio, na ordem cronologica. `float()` porque as linhas
    podem vir do `DictWriter` (ja float) ou de um `DictReader` do CSV (string)."""
    for label in sorted({r["stage_label"] for r in rows},
                        key=lambda s: (int(_STAGE.match(s)[1] or 0),
                                       int(_STAGE.match(s)[2] or 5))):
        sel = [r for r in rows if r["stage_label"] == label]
        print(f"{label:>13}  n={len(sel):<3d} "
              f"kappa={np.mean([float(r['kappa']) for r in sel]):+.4f} "
              f"acc={np.mean([float(r['acc']) for r in sel]):.4f} "
              f"f1={np.mean([float(r['f1']) for r in sel]):.4f}")


def _wandb_val(args, out: str) -> None:
    """Baixa os runs das familias pedidas e (re)escreve o CSV de VALIDACAO por estagio.

    O download bruto e o `fetch()` do plot — o unico downloader do repo, com o retry de
    rede ja pago. `PROJECT` la e constante de modulo, entao `--wandb-entity/--wandb-project`
    entram por rebind, do mesmo jeito que `_ev.EMBED_MODE` no `main()`.

    ARMADILHA: `probe/flim_ref_svm_kappa` so existe no SUMMARY do run — e logado uma vez,
    em `on_fit_start`, e nao sobrevive ao `scan_history()` que alimenta os CSVs por-run.
    Sai da listagem, que ja traz o summary junto: nenhum request a mais por run.

    ARMADILHA 2: a familia vem do `RUN_RE`, nunca do `config["stage"]` — aquele campo vale
    2 num `round1_head` e nem tem numero para o head.
    """
    import pandas as pd
    import wandb

    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    import plot_partial_train_spifil_hybrid as _pt

    project = f"{args.wandb_entity}/{args.wandb_project}"
    _pt.PROJECT = project
    _pt.fetch(families=set(args.family))

    flim_ref, seen_fams = {}, set()
    for run in wandb.Api(timeout=60).runs(project):
        m = _pt.RUN_RE.match(run.name)
        fam = (m["family"] or "grid4") if m else None
        if fam in args.family:
            seen_fams.add(fam)
            flim_ref[run.name] = run.summary.get("probe/flim_ref_svm_kappa")

    # Falhar alto: nunca plotar incompleto calado.
    for fam in args.family:
        if fam not in seen_fams:
            raise SystemExit(f"[FALHA] --family {fam!r}: 0 runs em {project}.")
    disk = set()
    for fam in args.family:
        for meta_path in _pt.grid_dir(fam).glob("*/*/run_metadata.json"):
            meta = json.loads(meta_path.read_text())
            if meta.get("percentage") == args.pct:
                disk.add(meta["run_name"])
    faltam = sorted(disk - flim_ref.keys())
    if faltam and not args.allow_missing_runs:
        raise SystemExit(f"[FALHA] {len(faltam)} braco(s) em disco sem run no W&B: "
                         f"{' '.join(faltam)}\n  Use --allow-missing-runs para ignorar.")

    def cell(row, key):
        """Coluna ausente e coluna vazia sao a mesma coisa aqui: string vazia."""
        v = row.get(key)
        return "" if v is None or pd.isna(v) else v

    rows = []
    for path in tqdm(sorted(_pt.CSV_DIR.glob("spifil_growth_*.csv")),
                     desc="val stages", unit="run"):
        m = _pt.RUN_RE.match(path.stem)
        if not m or (m["family"] or "grid4") not in args.family:
            continue
        label, st = m["label"], _STAGE.match(m["label"])
        if (not st or int(m["pct"]) != args.pct
                or (args.stage and label not in args.stage)
                or (args.dataset and m["dataset"] not in args.dataset)
                or (args.split and int(m["split"]) not in args.split)):
            continue
        hist = pd.read_csv(path)
        # `scan_history()` traz todo log point; a maioria nao tem a sonda. `stage_epoch`
        # e o contador zerado no inicio do estagio — `epoch` e o global do Lightning e
        # abre num offset diferente por split no braco com Head.
        hist = hist[hist["probe/svm_kappa"].notna() & hist["stage_epoch"].notna()]
        if hist.empty:
            continue
        hist = hist.sort_values("_step")
        pick = (hist.loc[hist["probe/svm_kappa"].idxmax()] if args.stage_agg == "best"
                else hist.iloc[-1])
        fam = m["family"] or "grid4"
        arm = f"{m['dataset']}_split{m['split']}_pct{m['pct']}"
        rows.append({
            "method": _VAL_METHOD, "run_name": pick["run_name"], "stage_label": label,
            "round": int(st[1] or 0), "dataset": m["dataset"],
            "split": int(m["split"]), "percentage": int(m["pct"]),
            "eval_split": "val", "family": fam, "run_id": pick["run_id"],
            "run_state": pick["run_state"], "stage_epoch": int(pick["stage_epoch"]),
            "kappa": cell(pick, "probe/svm_kappa"), "acc": cell(pick, "probe/svm_acc"),
            "f1": cell(pick, "probe/svm_f1"),
            "val_recon_loss": cell(pick, "probe/val_recon_loss"),
            "head_kappa": cell(pick, "probe/head_kappa"),
            "head_acc": cell(pick, "probe/head_acc"),
            "flim_ref_kappa": "" if flim_ref.get(pick["run_name"]) is None
                              else flim_ref[pick["run_name"]],
            # So por compat com o `.str[2]` do plot, que le a familia do caminho.
            "ckpt": f"artifacts/spifil_growth/{fam}/{arm}/{label}/checkpoints/"
                    f"best_kappa.ckpt",
        })

    new = pd.DataFrame(rows, columns=_VAL_FIELDS)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        old = pd.read_csv(out)
        # `astype(str)` porque o CSV relido volta com int64 onde o DataFrame novo tem int
        # nativo, e o `isin` de MultiIndex compara por tipo.
        key = pd.MultiIndex.from_frame(new[_VAL_KEY].astype(str))
        old = old[~pd.MultiIndex.from_frame(old[_VAL_KEY].astype(str)).isin(key)]
        new = pd.concat([old, new], ignore_index=True)[_VAL_FIELDS]
    new.to_csv(out, index=False)
    print(f"\n[OK] {out}  ({len(rows)} linhas de validacao)")


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
                         "grid3, grid4, g5_in_feature, g5_in_image, g5_random, "
                         "g5_random_in_feature, g5_head, g5_head_larvae. Sem "
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
    ap.add_argument("--eval-split", choices=("test", "val"), default="test",
                    help="Qual medicao alimenta o eixo dos estagios. `test` (default) = "
                         "este avaliador, rodando o SVM sobre os checkpoints. `val` = CSV "
                         "de validacao vindo do W&B (`probe/svm_*`), que nao passa por "
                         "checkpoint nenhum. Entra no nome do CSV como infixo `_val`.")
    ap.add_argument("--fetch-wandb", action="store_true",
                    help="Baixa os runs das familias pedidas e (re)escreve o CSV de "
                         "validacao antes de qualquer uso. Implica --eval-split val.")
    ap.add_argument("--wandb-entity", default="ophira-ai")
    ap.add_argument("--wandb-project", default="phd_thesis_grid4")
    ap.add_argument("--stage-agg", choices=("best", "last"), default="best",
                    help="Como colapsar a curva de um estagio num ponto. `best` = epoca "
                         "de maior probe/svm_kappa (e o que o best_kappa.ckpt guardou); "
                         "`last` = ultima stage_epoch.")
    ap.add_argument("--allow-missing-runs", action="store_true",
                    help="Desliga a falha alta quando um braco presente em disco nao tem "
                         "run correspondente no W&B.")
    args = ap.parse_args()

    # --fetch-wandb so sabe produzir o CSV de validacao: forcar aqui mantem o nome do
    # arquivo coerente com o que foi de fato medido.
    if args.fetch_wandb:
        args.eval_split = "val"

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
    kind = "_val" if args.eval_split == "val" else ""
    out = args.out or os.path.join(_ROOT, "results",
                                   f"eval_growth_stages{kind}{tag}_pct{args.pct}.csv")

    # A validacao nao vem de checkpoint: nao ha nada para avaliar aqui, so o CSV do W&B.
    if args.eval_split == "val":
        if args.fetch_wandb:
            _wandb_val(args, out)
        if not os.path.exists(out):
            raise SystemExit(f"[FALHA] {out} nao existe. Rode com --fetch-wandb.")
        with open(out, newline="") as fh:
            vrows = list(csv.DictReader(fh))
        falta = [c for c in _VAL_FIELDS if not vrows or c not in vrows[0]]
        if falta:
            raise SystemExit(f"[FALHA] {out}: schema de validacao sem {falta}")
        print(f"[OK] {out}  ({len(vrows)} linhas, eval_split=val, "
              f"stage_agg={args.stage_agg})")
        _summary(vrows)
        return

    jobs = _jobs(args.pct, args.family, args.stage, args.dataset, args.split)

    print(f"[CONTRATO] embed_mode='{_EMBED_MODE}' (o do treino; default do "
          f"modulo seria '{_ev.DEFAULT_EMBED_MODE}')")
    print(f"[CONTRATO] rotulo  = nome do diretorio, nao o campo 'stage' do metadado")
    print(f"[CONTRATO] encoder = load_from_checkpoint(<estagio>/checkpoints/best_kappa.ckpt)")
    print(f"[CONTRATO] solver  = train_svm(..., max_iter={args.max_iter}) -> fit_svm")
    print(f"[CONTRATO] {len(jobs)} trabalhos | family={' '.join(args.family)} "
          f"| pct={args.pct} | device={device} | out={out}\n")

    # Zero trabalho e erro, nao CSV vazio: sem isto um filtro que nao casa nada
    # so vira arquivo de cabecalho e o NaN aparece la na frente, na tabela.
    if not jobs:
        ap.error(f"nenhum estagio casou o filtro: family={' '.join(args.family)} "
                 f"pct={args.pct} stage={args.stage} dataset={args.dataset} "
                 f"split={args.split}")

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
            enc, head, epoch = _encoder(j, device)

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
            # A Head no MESMO passe do SVM: sob `flatten`, `feats_te` E o mapa do
            # encoder achatado, entao (N, C, H*W, 1) e o mesmo tensor de volta e o
            # AdaptiveAvgPool2d(1) da Head media sobre H*W igual. Zero forward extra.
            # (Sob `avgpool2d` o dim ja e C, o reshape vira (N, C, 1, 1) e a pool
            # deixa passar: correto nos dois modos.) Rotulo 0-indexed direto — o `-1`
            # do SVM desfaz um `+1` que so o SVM aplica.
            if head is not None:
                ch = head.fc.in_features
                fmap = torch.from_numpy(feats_te).view(-1, ch,
                                                       feats_te.shape[1] // ch, 1)
                with torch.no_grad():
                    hm = compute_metrics(y_true=y_true, y_pred=head(fmap),
                                         num_classes=n_cls)
                row |= {f"head_{k}": float(v) for k, v in hm.items()}

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
    _summary(rows)


if __name__ == "__main__":
    main()
