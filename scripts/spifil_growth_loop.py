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
"""spifil_growth_loop.py — cresce o autoencoder FLIM uma camada SPiFiL por vez.

Um comando so para o protocolo inteiro, agora em GRADE: dataset x split x
percentual. A ordem dentro de cada celula e fixa e nao e configuravel:

    stage1  encoder congelado, so o decoder aprende
    stage2  tudo solto, ponto a ponto, partindo do best de stage1
    rodada r:
      grow      SPiFiL corta UMA camada nova do backbone JA TREINADO (sem backprop)
      stage3    encoder congelado de novo, com a camada nova no lugar
      stage4    tudo solto, o modelo crescido inteiro
    repete ate o kappa parar de melhorar

Com `--unfrozen-after-stage-two` a rodada perde o stage3: cada camada nova
entra e o treino segue com tudo destravado, so stage4. Os estagios 1 e 2 e a
regra de parada (que sempre leu o kappa do stage4) nao mudam.

Com `--head-finetune` a rodada troca stage3+stage4 por um estagio so,
`round{r}_head`: fine-tune SUPERVISIONADO com a Head do treinador (GAP ->
dropout -> linear) sobre a camada nova, tudo destravado. As duas flags sao
mutuamente exclusivas e, sem nenhuma delas, o fluxo acima segue identico.

O grow da rodada r come o checkpoint do stage4 da rodada r-1 (e o stage2 na
rodada 1). Isso e o ponto do protocolo: a camada tem que sair do backbone
treinado, nao do backbone de tres camadas original.

## Braco = unidade de paralelismo

Cada (dataset, split, percentual) e um **braco**, com a cadeia sequencial
inteira acima. Como cada estagio come o checkpoint do anterior, nao ha o que
paralelizar DENTRO de um braco — entao o que roda concorrente sao os bracos, um
por slot, cada um dono de uma GPU do comeco ao fim. Um braco que falha nao
derruba os irmaos; o processo so sai != 0 no fim.

## Por que subprocesso, e nao import

Cada estagio e um processo do SO separado de proposito — run W&B propria, pasta
de checkpoint propria, `last.ckpt` proprio para retomar de queda. Chamar o
LightningModule em processo colapsaria os tres.

**Quase nada viaja por variavel de ambiente.** Tudo que o filho precisa entra
como argumento de linha de comando explicito — inclusive a GPU (`--gpu` no
treinador, `--device cuda:N` no grow) — para que `--dry-run` mostre o plano
completo e nada dependa do shell de quem chamou.

A UNICA excecao e `OMP_NUM_THREADS`, e ela e obrigatoria: o OpenBLAS que o numpy
e o sklearn usam le esse limite no import, antes de qualquer flag do argparse,
entao nao existe forma de fixa-lo por CLI. Sem ele cada treinador abre um pool
OpenMP do tamanho da maquina (~121 threads medidos), e 18 bracos levaram um box
de 96 nucleos a load 835 sem fechar uma unica epoca. Ela e injetada no `env=` de
CADA subprocesso — nunca em `os.environ`, que as threads deste laco compartilham
— e o valor sai do `--cpus-per-experiment`, o mesmo flag que os outros
lancadores do repo ja expoem.

## Selecao de modelo

O unico sinal e o Cohen kappa do SVM sondando o gargalo — em TODOS os estagios,
congelado ou nao. O laco le `best_val_svm_kappa` do `run_metadata.json` de cada
estagio; arquivo faltando ou valor nulo mata AQUELE braco em vez de chutar
numero.

## Uso

    # grade inteira: 3 datasets x 3 splits x 2 percentuais, 4 GPUs x 4 bracos
    python scripts/spifil_growth_loop.py --work-dir artifacts/spifil_growth/grid \\
        --num-gpus 4 --max-concurrent-per-gpu 4 --max-rounds 4

    # ver o plano sem rodar nada
    python scripts/spifil_growth_loop.py --work-dir artifacts/spifil_growth/grid --dry-run

    # um braco so, com arquitetura de partida escolhida na mao
    python scripts/spifil_growth_loop.py --datasets eggs --splits 1 --percentages 50 \\
        --arch-json <arch.json> --flim-weights-path <dir> \\
        --work-dir artifacts/spifil_growth/one
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

from constants import (ARCH_JSON_FILENAME, BEST_CKPT_FILENAME, CHECKPOINTS_SUBDIR,
                       CONV1_KERNELS, DATASETS, DEFAULT_CONFIG_YAML,
                       FLIM_ARCH_BASE, FLIM_WEIGHTS_BASE, GROW_EXHAUSTED,
                       GROW_SCRIPT, GROWTH_DEFAULT_PCTS, KAPPA_KEY,
                       KAPPA_TOLERANCE, DEFAULT_CPUS_PER_EXPERIMENT,
                       MODELS_SUBDIR, OMP_ENV_VAR, PERCENTAGES, PROJECT_ROOT,
                       RUN_METADATA_FILENAME, RUN_NAME_PREFIX, SPLITS,
                       TRAINER_MODULE, train_dir)


def _exp(args) -> str:
    """Nome curto do experimento: o basename do `--work-dir`.

    Sem isto, dois lacos com flags diferentes (in-feature contra in-image contra
    finetune) sobem com nome de run IDENTICO e ficam indistinguiveis no board do
    W&B. O work-dir ja e unico por braco, entao ele e a etiqueta natural — nao
    precisa de flag nova nem de variavel de ambiente.
    """
    return os.path.basename(os.path.normpath(args.work_dir))

# O orcamento de epocas e propriedade do protocolo de treino, nao da linha de
# tmux que por acaso lanca a grade: `max_epochs` e a `patience` do EarlyStopping
# saem de configs/default.yaml para ninguem digitar numero diferente por engano.
# ponytail: duas chaves de um arquivo conhecido, sem camada de config e sem
# default de reserva — chave faltando ou renomeada tem que levantar
# StopIteration/KeyError aqui, no import, em vez de treinar em silencio com
# outro orcamento.
with open(DEFAULT_CONFIG_YAML, encoding="utf-8") as _fh:
    _TRAINER_CFG = yaml.safe_load(_fh)["trainer"]
MAX_EPOCHS: int = _TRAINER_CFG["max_epochs"]
PATIENCE: int = next(cb["init_args"]["patience"]
                     for cb in _TRAINER_CFG["callbacks"]
                     if cb["class_path"].endswith("EarlyStopping"))

# Repassados verbatim a TODO filho treinador. Default None: quem manda o default
# e o argparse do treinador, nao este arquivo — duplicar aqui e garantir deriva.
# `max-epochs` e `patience` NAO estao aqui: vem do yaml e vao sempre explicitos.
# `embed-mode` e a excecao deliberada: tem default proprio (flatten) porque e uma
# escolha DESTE experimento, e por isso sai sempre explicito no comando filho.
TRAINER_FWD = ["warmup-epochs", "batch-size", "lr", "num-workers", "image-size",
               "embed-mode", "svm-probe-every", "log-every-n-steps", "seed",
               "wandb-project", "wandb-entity"]

# `--image-size` e `--seed` vao para os dois lados: o corte SPiFiL tem que ver a
# imagem no mesmo tamanho e com a mesma semente que o treino. `--device` tambem
# nao esta aqui: e derivado da GPU do braco.
GROW_FWD = ["out-channels", "kernel-size", "pool-stride", "n-superpixels",
            "n-images", "image-size", "seed"]


def should_stop(kappas: list[float], tolerance: float, patience: int) -> bool:
    """True when the last `patience` rounds all failed to beat the best kappa before them by more than `tolerance`."""
    fails = 0
    for r in range(1, len(kappas)):
        fails = 0 if kappas[r] > max(kappas[:r]) + tolerance else fails + 1
    return fails >= patience


def _tag(arm: tuple[str, int, int]) -> str:
    """Nome do braco: entra na pasta e no nome da run W&B."""
    dataset, split, pct = arm
    return f"{dataset}_split{split}_pct{pct}"


def _stage_dir(args: argparse.Namespace, arm: tuple[str, int, int], name: str) -> str:
    """`<work-dir>/<braco>/<estagio>` — uma pasta por braco, os estagios dentro dela."""
    return os.path.join(args.work_dir, _tag(arm), name)


def _paths(args: argparse.Namespace, arm: tuple[str, int, int]) -> tuple[str, str]:
    """arch JSON e diretorio de pesos FLIM de partida do braco.

    Sao DUAS arvores diferentes de proposito: o protozoan tem `ch24_30_48` na
    arquitetura e `ch24_32_48` nos pesos (constants.py:48-50 explica). E por isso
    que fixar um par na linha de comando nao serve para a grade inteira — os
    overrides so valem quando ha um braco so.
    """
    dataset, split, _ = arm
    arch = os.path.join(FLIM_ARCH_BASE[dataset], train_dir(split), ARCH_JSON_FILENAME)
    weights = os.path.join(FLIM_WEIGHTS_BASE[dataset], train_dir(split), MODELS_SUBDIR)
    return args.arch_json or arch, args.flim_weights_path or weights


def _fwd(args: argparse.Namespace, names: list[str]) -> list[str]:
    """So repassa o que o usuario realmente pediu — None significa 'usa teu default'."""
    out: list[str] = []
    for name in names:
        value = getattr(args, name.replace("-", "_"))
        if value is not None:
            out += [f"--{name}", str(value)]
    return out


def _trainer_cmd(args: argparse.Namespace, arm: tuple[str, int, int], gpu: int, name: str,
                 arch: str, weights: str, freeze: bool, init_ckpt: str,
                 head: bool) -> list[str]:
    dataset, split, pct = arm
    cmd = [sys.executable, "-m", TRAINER_MODULE,
           "--dataset", dataset,
           "--split", str(split),
           "--percentage", str(pct),
           "--arch-json", arch,
           "--flim-weights-path", weights,
           "--run-name", f"{RUN_NAME_PREFIX}_{_exp(args)}_{_tag(arm)}_{name}",
           "--output-dir", _stage_dir(args, arm, name),
           # Pinning de GPU por flag, nunca por variavel de ambiente: assim o
           # --dry-run mostra em que GPU cada comando cai e nada depende do shell.
           "--gpu", str(gpu),
           # configs/default.yaml e a fonte unica do orcamento — sempre explicito,
           # para que nenhum filho herde um numero diferente sem querer.
           "--max-epochs", str(MAX_EPOCHS),
           "--patience", str(PATIENCE)]
    if freeze:
        cmd.append("--freeze-encoder")
    # So o estagio `round{r}_head` pede a Head: os estagios 1 e 2 sao de
    # reconstrucao e a flag nao pode vazar para eles.
    if head:
        cmd.append("--head-finetune")
    if init_ckpt:
        cmd += ["--init-ckpt", init_ckpt]
    if args.wandb:
        cmd.append("--wandb")
    if args.freeze_spifil_layer:
        cmd.append("--freeze-spifil-layer")
    if args.imagenet_norm is not None:
        cmd.append("--imagenet-norm" if args.imagenet_norm else "--no-imagenet-norm")
    return cmd + _fwd(args, TRAINER_FWD)


def _grow_cmd(args: argparse.Namespace, arm: tuple[str, int, int], gpu: int, ckpt: str,
              arch: str, weights: str, out: str) -> list[str]:
    dataset, split, pct = arm
    cmd = [sys.executable, GROW_SCRIPT,
           "--ckpt", ckpt,
           "--arch-json", arch,
           "--flim-weights-path", weights,
           "--dataset", dataset,
           "--split", str(split),
           "--percentage", str(pct),
           "--out", out,
           "--device", f"cuda:{gpu}"]
    if args.spifil_in_feature:
        cmd.append("--spifil-in-feature")
    if args.spifil_in_image:
        cmd.append("--spifil-in-image")
    if args.impurities is not None:
        cmd.append("--impurities" if args.impurities else "--no-impurities")
    if args.one_per_class is not None:
        cmd.append("--one-per-class" if args.one_per_class else "--no-one-per-class")
    # Booleana nao cabe em GROW_FWD: `_fwd` emitiria `--random-layer False`.
    if args.random_layer:
        cmd.append("--random-layer")
    return cmd + _fwd(args, GROW_FWD)


def _die(message: str) -> None:
    """Mata SO o braco corrente. `run_arm` pega isso; a grade continua sem ele."""
    raise RuntimeError(message)


def _kappa(out_dir: str) -> float:
    """`best_val_svm_kappa` do estagio. Faltando ou nulo mata o braco, nao vira zero."""
    path = os.path.join(out_dir, RUN_METADATA_FILENAME)
    if not os.path.isfile(path):
        _die(f"{path} nao existe — o estagio terminou sem metadata")
    with open(path, encoding="utf-8") as f:
        value = json.load(f).get(KAPPA_KEY)
    if value is None:
        _die(f"{path} tem {KAPPA_KEY}=null — a sonda SVM nunca rodou neste estagio")
    return float(value)


def run_arm(args: argparse.Namespace, arm: tuple[str, int, int], gpu: int) -> dict:
    """A cadeia sequencial inteira de UM braco, presa a uma GPU do inicio ao fim."""
    tag = _tag(arm)
    plan: list[list[str]] = []
    rows: list[tuple[str, str, float]] = []
    result = {"tag": tag, "gpu": gpu, "plan": plan, "rows": rows, "failed": False,
              "reason": f"--max-rounds {args.max_rounds} atingido"}

    def run(cmd: list[str]) -> int:
        if args.dry_run:
            plan.append(cmd)
            return 0
        print(f"[loop][{tag}] $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        env = {**os.environ, OMP_ENV_VAR: str(args.cpus_per_experiment)}
        return subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False).returncode

    def train(name: str, arch: str, weights: str, freeze: bool = False,
              init_ckpt: str = "", head: bool = False) -> str:
        """
        @Mateus Oliveira
        monta todo o comando com os argumentos, e dai se nao for. para carregar pesos do treino anterior,
        ele nao carrega. 
        """
        rc = run(_trainer_cmd(args, arm, gpu, name, arch, weights, freeze, init_ckpt, head))
        if rc != 0:
            _die(f"{name} saiu com codigo {rc}")
        return os.path.join(_stage_dir(args, arm, name), CHECKPOINTS_SUBDIR, BEST_CKPT_FILENAME)

    arch, weights = _paths(args, arm)
    try:
        """
        @Mateus Oliveira
        nessa parte do code ele trabalha a ideia de dar o comando e ir carregando os pesos ao longo do processo.
        os comandos sao criados para rodar via cmd.
        """
        ckpt = train("stage1", arch, weights, freeze=True)
        ckpt = train("stage2", arch, weights, init_ckpt=ckpt)

        kappas: list[float] = []
        if not args.dry_run: # so para ver se funciona.
            rows += [("—", "stage1", _kappa(_stage_dir(args, arm, "stage1"))),
                     ("—", "stage2", _kappa(_stage_dir(args, arm, "stage2")))]
            kappas = [rows[-1][2]]

        for r in range(1, args.max_rounds + 1):
            grow_dir = _stage_dir(args, arm, f"round{r}_grow")
            rc = run(_grow_cmd(args, arm, gpu, ckpt, arch, weights, grow_dir))
            if rc == GROW_EXHAUSTED:
                result["reason"] = (f"spifil_grow recusou na rodada {r}: orcamento de "
                                    "covariancia esgotado (N <= D)")
                break
            if rc != 0:
                _die(f"round{r}_grow saiu com codigo {rc}")
            arch, weights = os.path.join(grow_dir, ARCH_JSON_FILENAME), grow_dir

            if not (args.unfrozen_after_stage_two or args.head_finetune):
                ckpt = train(f"round{r}_stage3", arch, weights, freeze=True, init_ckpt=ckpt)
            # O estagio final da rodada. Com --head-finetune ele e supervisionado e
            # se chama `round{r}_head` de proposito: nome distinto de stage3/stage4
            # para que pasta e run W&B nao se misturem com as dos outros bracos.
            stage = "head" if args.head_finetune else "stage4"
            ckpt = train(f"round{r}_{stage}", arch, weights, init_ckpt=ckpt,
                         head=args.head_finetune)
            if args.dry_run:
                continue

            if not (args.unfrozen_after_stage_two or args.head_finetune):
                rows.append((str(r), "stage3", _kappa(_stage_dir(args, arm, f"round{r}_stage3"))))
            rows.append((str(r), stage, _kappa(_stage_dir(args, arm, f"round{r}_{stage}"))))
            # RESSALVA, deixada por escrito em vez de resolvida: com --head-finetune o
            # kappa do stage2 vem da sonda SVM sobre o gargalo e o do round{r}_head vem
            # da Head supervisionada. O should_stop abaixo compara os dois como se
            # fossem a mesma metrica — nao sao. Quem for ler estes numeros depois tem
            # que saber que a comparacao entre rodada e stage2 e entre metricas
            # diferentes; entre rodadas, ai sim, e a mesma metrica.
            kappas.append(rows[-1][2])
            if should_stop(kappas, args.kappa_tolerance, args.rounds_patience):
                result["reason"] = (f"kappa estagnou: {args.rounds_patience} rodada(s) seguidas "
                                    f"sem ganho > {args.kappa_tolerance} sobre o melhor anterior")
                break
    except RuntimeError as exc:
        # A falha para aqui de proposito: dentro do braco ela e fatal (o proximo
        # estagio comeria um checkpoint que nao existe), mas os irmaos rodando nas
        # outras GPUs nao tem nada a ver com isso. O processo sai != 0 no fim.
        print(f"[loop][{tag}] FALHOU: {exc}", file=sys.stderr, flush=True)
        result["failed"] = True
        result["reason"] = f"falha: {exc}"
    return result


def _summary(results: list[dict], skipped: list[tuple[str, list[str]]]) -> None:
    for r in results:
        print(f"\n[loop] {r['tag']}  (gpu {r['gpu']})")
        print(f"  {'round':<8}{'stage':<10}{'kappa':>9}{'delta vs melhor':>18}")
        best = None
        for round_label, stage, kappa in r["rows"]:
            delta = "     —" if best is None else f"{kappa - best:+.4f}"
            print(f"  {round_label:<8}{stage:<10}{kappa:>9.4f}{delta:>18}")
            best = kappa if best is None else max(best, kappa)
        print(f"  fim do laco: {r['reason']}")
    failed = sum(r["failed"] for r in results)
    print(f"\n[loop] {len(results) - failed} braco(s) terminaram, {failed} falharam, "
          f"{len(skipped)} pulados no preflight")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Protocolo de crescimento SPiFiL em grade: um braco por "
                    "dataset x split x percentual, stage1/2 e depois grow+stage3+stage4.")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--splits", nargs="+", type=int, choices=SPLITS, default=SPLITS)
    p.add_argument("--percentages", nargs="+", type=int, choices=PERCENTAGES,
                   default=GROWTH_DEFAULT_PCTS)
    p.add_argument("--work-dir", required=True,
                   help="uma subpasta por braco nasce aqui, e uma por estagio dentro dela")
    p.add_argument("--arch-json",
                   help="override do architecture.json de partida; so com UM braco na grade")
    p.add_argument("--flim-weights-path",
                   help="override do dir de pesos FLIM de partida; so com UM braco na grade")
    p.add_argument("--num-gpus", type=int, default=4, metavar="N",
                   help="usa as GPUs 0..N-1. --gpus tem precedencia.")
    p.add_argument("--gpus", nargs="+", type=int, default=None, metavar="ID",
                   help="ids explicitos de GPU, ex.: --gpus 0 1 3")
    p.add_argument("--max-concurrent-per-gpu", type=int, default=4, metavar="N",
                   help="bracos simultaneos por GPU")
    p.add_argument("--max-rounds", type=int, default=4)
    p.add_argument("--kappa-tolerance", type=float, default=KAPPA_TOLERANCE)
    p.add_argument("--rounds-patience", type=int, default=1,
                   help="rodadas seguidas sem ganho > tolerancia antes de parar")
    # Mutuamente exclusivas: as duas reescrevem a rodada, e de formas diferentes.
    round_mode = p.add_mutually_exclusive_group()
    round_mode.add_argument("--unfrozen-after-stage-two", action="store_true", default=False,
                            help="pula o stage3 de cada rodada: depois do stage2 a camada nova "
                                 "entra e o treino segue com tudo destravado, so stage4. Sem a "
                                 "flag o fluxo grow+stage3+stage4 continua identico.")
    round_mode.add_argument("--head-finetune", action="store_true", default=False,
                            help="troca stage3+stage4 por um unico `round{r}_head`: fine-tune "
                                 "SUPERVISIONADO com a Head do treinador, tudo destravado, "
                                 "selecao por probe/head_kappa. Sem a flag o fluxo de "
                                 "reconstrucao continua identico.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="imprime todos os comandos de todos os bracos e sai 0, sem rodar nada")
    # Repasses ao treinador. Default None de proposito: ver TRAINER_FWD.
    p.add_argument("--warmup-epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--embed-mode", choices=["avgpool2d", "flatten"], default="flatten",
                   help="Espaco em que o kappa e medido — SO avaliacao: o `_encode_pooled` nao entra na loss nem no gradiente. Este experimento mede em `flatten` (mapa da ultima camada achatado), dai o default divergir do `avgpool2d` do treinador.")
    p.add_argument("--svm-probe-every", type=int)
    p.add_argument("--log-every-n-steps", type=int)
    p.add_argument("--cpus-per-experiment", type=int, default=DEFAULT_CPUS_PER_EXPERIMENT,
                   metavar="N",
                   help="OMP_NUM_THREADS de cada subprocesso. Sem isto o pool OpenMP de "
                        "cada filho nasce do tamanho da maquina e a grade afoga o box.")
    p.add_argument("--seed", type=int)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project")
    p.add_argument("--wandb-entity")
    p.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--freeze-spifil-layer", action="store_true", default=False)
    # Repasses ao grow.
    p.add_argument("--out-channels", type=int)
    p.add_argument("--kernel-size", type=int)
    p.add_argument("--pool-stride", type=int)
    p.add_argument("--n-superpixels", type=int)
    p.add_argument("--n-images", type=int)
    # Mutuamente exclusivas: o argparse ja recusa o par e imprime o porque.
    spifil_in = p.add_mutually_exclusive_group()
    spifil_in.add_argument("--spifil-in-feature", action="store_true", default=False,
                           help="recalcula os superpixels em cima da ultima camada")
    spifil_in.add_argument("--spifil-in-image", action="store_true", default=False,
                           help="superpixel na imagem original, sementes reprojetadas")
    p.add_argument("--impurities", action=argparse.BooleanOptionalAction, default=None,
                   help="--impurities segmenta o quadro inteiro (sem mascara): fora do parasita "
                        "e impureza. DEFAULT do grow: --no-impurities, com mascara")
    p.add_argument("--one-per-class", action=argparse.BooleanOptionalAction, default=None,
                   help="uma unica imagem por classe")
    # Controle do crescimento: mesma camada, pesos sorteados. Repassada ao grow, que segue
    # calculando o superpixel e o orcamento normalmente e so troca os VALORES no fim — e o
    # que garante que a familia aleatoria cresca nos MESMOS bracos que a SPiFiL.
    p.add_argument("--random-layer", action="store_true", default=False,
                   help="camada nova com pesos aleatorios de mesmo shape (ablacao de controle)")
    args = p.parse_args()

    gpu_ids = args.gpus if args.gpus else list(range(args.num_gpus))
    if not gpu_ids:
        p.error("nenhuma GPU selecionada: use --num-gpus >= 1 ou --gpus")

    arms = [(d, s, pct) for d in args.datasets for s in args.splits for pct in args.percentages]
    if (args.arch_json or args.flim_weights_path) and len(arms) > 1:
        p.error(f"--arch-json/--flim-weights-path descrevem UM ponto de partida, e a grade "
                f"tem {len(arms)} bracos — um override nao pode significar dois caminhos ao "
                f"mesmo tempo. Restrinja com --datasets/--splits/--percentages, ou deixe o "
                f"constants.py resolver os caminhos por braco.")

    # Preflight, como em autoencoder_flim_ray.py:594: os 9 pares dataset/split
    # existem hoje, entao um braco que nao passa aqui e problema real de caminho —
    # ele fica de fora e os outros seguem, em vez de morrer 40 minutos depois.
    ready: list[tuple[str, int, int]] = []
    skipped: list[tuple[str, list[str]]] = []
    for arm in arms:
        arch, weights = _paths(args, arm)
        missing = [f for f in (arch, os.path.join(weights, CONV1_KERNELS))
                   if not os.path.isfile(f)]
        if missing:
            skipped.append((_tag(arm), missing))
        else:
            ready.append(arm)

    slots = len(gpu_ids) * args.max_concurrent_per_gpu
    one_stage_round = args.unfrozen_after_stage_two or args.head_finetune
    round_stages = ("head" if args.head_finetune else
                    "stage4" if args.unfrozen_after_stage_two else "stage3+stage4")
    per_arm = 2 + (1 if one_stage_round else 2) * args.max_rounds
    print(f"[loop] {len(ready)} braco(s) prontos, {len(skipped)} pulados no preflight")
    print(f"[loop] {slots} slots = {len(gpu_ids)} gpu(s) {gpu_ids} x "
          f"{args.max_concurrent_per_gpu} braco(s) por gpu")
    print(f"[loop] configs/default.yaml amarra: --max-epochs {MAX_EPOCHS} --patience {PATIENCE}")
    print(f"[loop] no maximo {len(ready) * per_arm} processos de treino "
          f"({per_arm} por braco: stage1, stage2 e ate {args.max_rounds} rodada(s) "
          f"de {round_stages})")
    for tag, missing in skipped:
        print(f"[loop] PREFLIGHT pulou {tag}: nao existe " + ", ".join(missing))

    # O braco e a unidade agendada, nao o estagio: a cadeia dele e estritamente
    # sequencial. As threads so esperam subprocess.run, entao nao ha GIL no meio.
    jobs = [(arm, gpu_ids[slot % len(gpu_ids)]) for slot, arm in enumerate(ready)]
    with ThreadPoolExecutor(max_workers=slots) as pool:
        results = list(pool.map(lambda job: run_arm(args, *job), jobs))

    if args.dry_run:
        total = sum(len(r["plan"]) for r in results)
        print(f"\n[loop] plano com {total} comandos em {len(results)} braco(s) "
              f"(nada foi executado):")
        for r in results:
            print(f"\n  === {r['tag']}  (gpu {r['gpu']}) ===")
            for cmd in r["plan"]:
                print("    " + " ".join(shlex.quote(c) for c in cmd))
        return

    _summary(results, skipped)
    if any(r["failed"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
