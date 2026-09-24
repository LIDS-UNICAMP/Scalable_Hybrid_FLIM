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

"""Runner `growth` — o protocolo de crescimento SPiFiL, multi-rodada, um braco por task.

Cresce o autoencoder FLIM uma camada SPiFiL por vez. A ordem dentro de cada
braco e fixa e nao e configuravel:

    stage1  encoder congelado, so o decoder aprende
    stage2  tudo solto, ponto a ponto, partindo do best de stage1
    rodada r:
      grow      SPiFiL corta UMA camada nova do backbone JA TREINADO (sem backprop)
      stage3    encoder congelado de novo, com a camada nova no lugar
      stage4    tudo solto, o modelo crescido inteiro
    repete ate o kappa parar de melhorar

Com `unfrozen_after_stage_two` a rodada perde o stage3: cada camada nova entra e
o treino segue com tudo destravado, so stage4. Os estagios 1 e 2 e a regra de
parada (que sempre leu o kappa do stage4) nao mudam.

Com `head_finetune` a rodada troca stage3+stage4 por um estagio so,
`round{r}_head`: fine-tune SUPERVISIONADO com a Head do treinador (GAP ->
dropout -> linear) sobre a camada nova, tudo destravado. As duas chaves sao
mutuamente exclusivas e, sem nenhuma delas, o fluxo acima segue identico.

O grow da rodada r come o checkpoint do stage4 da rodada r-1 (e o stage2 na
rodada 1). Isso e o ponto do protocolo: a camada tem que sair do backbone
treinado, nao do backbone de tres camadas original.

## Braco = unidade de paralelismo, e uma GPU do comeco ao fim

Cada (dataset, split, percentual) e um **braco**, com a cadeia sequencial
inteira acima. Como cada estagio come o checkpoint do anterior, nao ha o que
paralelizar DENTRO de um braco — entao o que roda concorrente sao os bracos, um
por slot, cada um dono de uma GPU do comeco ao fim. Um braco que falha nao
derruba os irmaos: `run_arm` captura a falha dele e devolve `failed=True`.

O contrato "1 braco = 1 GPU" sobreviveu a troca de executor
(`ThreadPoolExecutor` -> Ray) por construcao, e nao por convencao: `gpu_id` e
**parametro** de `run_arm`, fixado uma vez pelo lancador quando ele pega o slot
no `GpuSlotScheduler`, e a funcao inteira — os N estagios e as N chamadas de
`grow` — roda dentro dessa unica invocacao. Nao ha ponto no meio da cadeia onde
outra GPU possa entrar. Como `run_growth` e `@ray.remote` SEM `num_gpus`
(mesmo desenho de `experiments/ray/runners/train.py:123`), o Ray nao reagenda
nem migra nada: quem fixa a GPU e o `CUDA_VISIBLE_DEVICES` do env de cada filho
e o `device=cuda:{gpu_id}` do `grow`.

## Por que o treino continua em subprocesso — e por que o `grow` nao

Cada ESTAGIO de treino e um processo do SO separado de proposito — run W&B
propria, pasta de checkpoint propria, `last.ckpt` proprio para retomar de queda.
Chamar o LightningModule em processo colapsaria os tres. Por isso todo estagio
passa por `train.build_cmd`, o MESMO comando do runner `train`: este arquivo nao
monta comando proprio, ele acrescenta ao comando de `train.py` os `--<k>=<v>` que
descrevem o estagio (congelado ou nao, de qual checkpoint parte, para qual pasta
escreve).

O `grow`, ao contrario, virou chamada EM PROCESSO: `flim.spifil.grow` e uma
funcao pura de biblioteca, e o codigo de saida `GROW_EXHAUSTED` que antes vinha
do `returncode` de um subprocesso agora e o valor de retorno dela. Isso move uma
responsabilidade para este arquivo: `grow` nao abre mais checkpoint nenhum, quem
carrega o encoder e o chamador — aqui — porque quem sabe abrir esse checkpoint e
o `AutoEncoderFlimModule`, que mora em `methods/`, e `flim/` nunca importa
`methods/`.

**Quase nada viaja por variavel de ambiente.** Tudo que o filho precisa entra
como argumento de linha de comando explicito, para que o `--dry-run` mostre o
plano completo e nada dependa do shell de quem chamou.

A UNICA excecao e `OMP_NUM_THREADS`, e ela e obrigatoria: o OpenBLAS que o numpy
e o sklearn usam le esse limite no import, antes de qualquer flag, entao nao
existe forma de fixa-lo por CLI. Sem ele cada treinador abre um pool OpenMP do
tamanho da maquina (~121 threads medidos), e 18 bracos levaram um box de 96
nucleos a load 835 sem fechar uma unica epoca. Ela e injetada no `env=` de CADA
subprocesso — nunca em `os.environ`, que o worker Ray reusa entre tasks — e o
valor sai de `resources.cpus_per_experiment`.

## Selecao de modelo

O unico sinal e o Cohen kappa do SVM sondando o gargalo — em TODOS os estagios,
congelado ou nao. O laco le `best_val_svm_kappa` do `run_metadata.json` de cada
estagio; arquivo faltando ou valor nulo mata AQUELE braco em vez de chutar
numero.

## runner_args

`max_rounds`, `kappa_tolerance`, `rounds_patience` e `embed_mode` sao as quatro
chaves que o schema aceita hoje (`experiments/ray/schema.py:50`). As outras que
este arquivo le — os knobs do `grow` e as duas que mudam a forma da rodada —
existiam como flag no laco de hoje e continuam com os mesmos defaults, mas
`RUNNER_ARGS["growth"]` ainda NAO as lista: um YAML que as escreva e recusado no
schema antes de chegar aqui. A lista completa esta no relatorio do R8.
"""

from __future__ import annotations

import json
import os

import ray

from core.constants import (ARCH_JSON_FILENAME, CHECKPOINTS_SUBDIR,
                            DEFAULT_CPUS_PER_EXPERIMENT, FLIM_ARCH_BASE,
                            FLIM_WEIGHTS_BASE, KAPPA_TOLERANCE, MODELS_SUBDIR,
                            OMP_ENV_VAR, RUN_METADATA_FILENAME)
from experiments.constants import CUDA_ENV_VAR, DATASET_LONG_TO_SHORT
from experiments.ray.runners.train import _run, build_cmd
from experiments.ray.schema import Experiment
from flim.constants import GROW_EXHAUSTED
from flim.spifil import grow
from methods.autoencoder import AutoEncoderFlimModule

# Nome do checkpoint que o `ModelCheckpoint` do braco de crescimento escreve, e
# que o estagio seguinte come. Origem: scripts/constants.py:178. Vive aqui
# porque este e o unico arquivo do repo que monta esse caminho por convencao.
BEST_CKPT_FILENAME: str = "best_kappa.ckpt"

# Chave do `run_metadata.json` que decide a parada. Origem: scripts/constants.py:244.
KAPPA_KEY: str = "best_val_svm_kappa"

# Defaults de `runner_args`, byte a byte os do argparse de hoje:
# max_rounds  scripts/spifil_growth_loop.py:403
# rounds_patience  :405
# embed_mode  :426 — divergencia DELIBERADA do `avgpool2d` do treinador: o kappa
#   deste experimento e medido no mapa da ultima camada achatado, e por isso sai
#   sempre explicito no comando filho, nunca por omissao.
# kappa_tolerance  :404, que le core.constants.KAPPA_TOLERANCE (0.01).
DEFAULT_MAX_ROUNDS: int = 4
DEFAULT_ROUNDS_PATIENCE: int = 1
DEFAULT_EMBED_MODE: str = "flatten"

# Os knobs que sao repassados a `flim.spifil.grow` com o nome identico. Ausente
# em `runner_args` significa "usa o default do grow" — a mesma semantica do
# `_fwd` de scripts/spifil_growth_loop.py:189-196, sem o baile do None.
# `device` NAO esta aqui: e derivado da GPU do braco.
GROW_KNOBS: tuple[str, ...] = (
    "out_channels", "kernel_size", "pool_stride", "n_superpixels", "n_images",
    "image_size", "imagenet_norm", "spifil_in_image", "spifil_in_feature",
    "impurities", "one_per_class", "random_layer", "random_layer_classic",
    "seed",
)


def should_stop(kappas: list[float], tolerance: float, patience: int) -> bool:
    """True when the last `patience` rounds all failed to beat the best kappa before them by more than `tolerance`."""
    fails = 0
    for r in range(1, len(kappas)):
        fails = 0 if kappas[r] > max(kappas[:r]) + tolerance else fails + 1
    return fails >= patience


def _short(point: dict) -> str:
    """Chave curta do dataset. A grade fala `helminth-eggs`, o resto do protocolo fala `eggs`."""
    return DATASET_LONG_TO_SHORT[point["dataset"]]


def _tag(point: dict) -> str:
    """Nome do braco: entra na pasta e no nome da run W&B."""
    return f"{_short(point)}_split{point['split']}_pct{point['percentage']}"


def _stage_dir(exp: Experiment, point: dict, name: str) -> str:
    """`<work_dir>/<braco>/<estagio>` — uma pasta por braco, os estagios dentro dela."""
    return os.path.join(exp.output.work_dir, _tag(point), name)


def _paths(point: dict) -> tuple[str, str]:
    """arch JSON e diretorio de pesos FLIM de PARTIDA do braco.

    Sao DUAS arvores diferentes de proposito: o protozoan tem `ch24_30_48` na
    arquitetura e `ch24_32_48` nos pesos (core/constants.py:145-162 explica).
    Da rodada 1 em diante os dois passam a apontar para a pasta do `grow`.
    """
    dataset, split = _short(point), point["split"]
    arch = os.path.join(FLIM_ARCH_BASE[dataset], f"train{split}", ARCH_JSON_FILENAME)
    weights = os.path.join(FLIM_WEIGHTS_BASE[dataset], f"train{split}", MODELS_SUBDIR)
    return arch, weights


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


def stage_cmd(exp: Experiment, point: dict, name: str, arch: str, weights: str,
              freeze: bool = False, init_ckpt: str = "", head: bool = False) -> list[str]:
    """O comando do runner `train`, mais os `--<k>=<v>` que descrevem ESTE estagio.

    Nao ha comando proprio aqui: a base e `train.build_cmd`, a mesma de todo
    ponto da grade, e o que se acrescenta sao overrides do LightningCLI — que
    resolve chave a chave e deixa o ULTIMO vencer. Por isso o nome da run e o
    `default_root_dir` podem ser reescritos por estagio mesmo ja vindo no comando
    base.
    """
    stage_dir = _stage_dir(exp, point, name)
    cmd = build_cmd(exp, point) + [
        f"--model.init_args.arch_json={arch}",
        f"--model.init_args.flim_weights_path={weights}",
        # Espaco em que o kappa e medido — SO avaliacao. Sempre explicito: e uma
        # escolha DESTE experimento, nao um default herdado do treinador.
        f"--model.init_args.embed_mode={exp.runner_args.get('embed_mode', DEFAULT_EMBED_MODE)}",
        f"--trainer.logger.init_args.name={exp.name}_{_tag(point)}_{name}",
        f"--trainer.logger.init_args.save_dir={stage_dir}",
        f"--trainer.default_root_dir={stage_dir}",
    ]
    if freeze:
        cmd.append("--model.init_args.freeze_encoder_flag=true")
    # So o estagio `round{r}_head` pede a Head: os estagios 1 e 2 sao de
    # reconstrucao e a chave nao pode vazar para eles.
    if head:
        cmd.append("--model.init_args.head_finetune=true")
    if init_ckpt:
        cmd.append(f"--model.init_args.init_ckpt={init_ckpt}")
    return cmd


def _grow_once(exp: Experiment, point: dict, gpu_id: int, ckpt: str, arch: str,
               weights: str, out: str) -> int:
    """Uma rodada de `flim.spifil.grow`, com o encoder carregado AQUI.

    `grow` nao abre checkpoint: o contrato de `flim/spifil.py:696` pede o
    `nn.Module` pronto. Com `random_layer_classic` o `grow` sai antes de tocar no
    backbone, entao a economia de minutos por braco que o caminho classico
    documenta so existe se o chamador tambem NAO carregar — e por isso o `if`
    abaixo, e nao um load incondicional.
    """
    knobs = {k: v for k, v in exp.runner_args.items() if k in GROW_KNOBS}
    device = f"cuda:{gpu_id}"
    encoder = None
    if not knobs.get("random_layer_classic", False):
        encoder = AutoEncoderFlimModule.load_from_checkpoint(
            ckpt, map_location=device).model.encoder
    return grow(encoder=encoder, ckpt=ckpt, arch_json=arch,
                flim_weights_path=weights, dataset=_short(point),
                split=point["split"], percentage=point["percentage"], out=out,
                device=device, **knobs)


def run_arm(exp: Experiment, point: dict, gpu_id: int,
            cpus: int = DEFAULT_CPUS_PER_EXPERIMENT, dry_run: bool = False) -> dict:
    """A cadeia sequencial inteira de UM braco, presa a `gpu_id` do inicio ao fim."""
    tag = _tag(point)
    max_rounds = exp.runner_args.get("max_rounds", DEFAULT_MAX_ROUNDS)
    tolerance = exp.runner_args.get("kappa_tolerance", KAPPA_TOLERANCE)
    patience = exp.runner_args.get("rounds_patience", DEFAULT_ROUNDS_PATIENCE)
    head_finetune = exp.runner_args.get("head_finetune", False)
    unfrozen = exp.runner_args.get("unfrozen_after_stage_two", False)

    plan: list[list[str]] = []
    rows: list[tuple[str, str, float]] = []
    result = {"tag": tag, "gpu_id": gpu_id, "plan": plan, "rows": rows,
              "failed": False, "reason": f"max_rounds {max_rounds} atingido"}

    def train(name: str, arch: str, weights: str, freeze: bool = False,
              init_ckpt: str = "", head: bool = False) -> str:
        cmd = stage_cmd(exp, point, name, arch, weights, freeze, init_ckpt, head)
        if dry_run:
            plan.append(cmd)
        else:
            # env do FILHO via subprocess(env=...), NUNCA os.environ do worker: o
            # worker Ray e reusado entre bracos e o valor vazaria para o proximo.
            env = {**os.environ, CUDA_ENV_VAR: str(gpu_id), OMP_ENV_VAR: str(cpus)}
            rc = _run(cmd, env=env, name=f"{tag}_{name}", gpu_id=gpu_id)["returncode"]
            if rc != 0:
                _die(f"{name} saiu com codigo {rc}")
        return os.path.join(_stage_dir(exp, point, name), CHECKPOINTS_SUBDIR,
                            BEST_CKPT_FILENAME)

    arch, weights = _paths(point)
    try:
        ckpt = train("stage1", arch, weights, freeze=True)
        ckpt = train("stage2", arch, weights, init_ckpt=ckpt)

        kappas: list[float] = []
        if not dry_run:
            rows += [("—", "stage1", _kappa(_stage_dir(exp, point, "stage1"))),
                     ("—", "stage2", _kappa(_stage_dir(exp, point, "stage2")))]
            kappas = [rows[-1][2]]

        for r in range(1, max_rounds + 1):
            grow_dir = _stage_dir(exp, point, f"round{r}_grow")
            if dry_run:
                plan.append(["<in-process>", "flim.spifil.grow", f"--out={grow_dir}",
                             f"--device=cuda:{gpu_id}", f"--ckpt={ckpt}"])
            elif _grow_once(exp, point, gpu_id, ckpt, arch, weights,
                            grow_dir) == GROW_EXHAUSTED:
                result["reason"] = (f"spifil grow recusou na rodada {r}: orcamento de "
                                    "covariancia esgotado (N <= D)")
                break
            arch, weights = os.path.join(grow_dir, ARCH_JSON_FILENAME), grow_dir

            if not (unfrozen or head_finetune):
                ckpt = train(f"round{r}_stage3", arch, weights, freeze=True, init_ckpt=ckpt)
            # O estagio final da rodada. Com head_finetune ele e supervisionado e
            # se chama `round{r}_head` de proposito: nome distinto de stage3/stage4
            # para que pasta e run W&B nao se misturem com as dos outros bracos.
            stage = "head" if head_finetune else "stage4"
            ckpt = train(f"round{r}_{stage}", arch, weights, init_ckpt=ckpt,
                         head=head_finetune)
            if dry_run:
                continue

            if not (unfrozen or head_finetune):
                rows.append((str(r), "stage3",
                             _kappa(_stage_dir(exp, point, f"round{r}_stage3"))))
            rows.append((str(r), stage,
                         _kappa(_stage_dir(exp, point, f"round{r}_{stage}"))))
            # RESSALVA, deixada por escrito em vez de resolvida: com head_finetune o
            # kappa do stage2 vem da sonda SVM sobre o gargalo e o do round{r}_head vem
            # da Head supervisionada. O should_stop abaixo compara os dois como se
            # fossem a mesma metrica — nao sao. Quem for ler estes numeros depois tem
            # que saber que a comparacao entre rodada e stage2 e entre metricas
            # diferentes; entre rodadas, ai sim, e a mesma metrica.
            kappas.append(rows[-1][2])
            if should_stop(kappas, tolerance, patience):
                result["reason"] = (f"kappa estagnou: {patience} rodada(s) seguidas "
                                    f"sem ganho > {tolerance} sobre o melhor anterior")
                break
    except RuntimeError as exc:
        # A falha para aqui de proposito: dentro do braco ela e fatal (o proximo
        # estagio comeria um checkpoint que nao existe), mas os irmaos rodando nas
        # outras GPUs nao tem nada a ver com isso.
        result["failed"] = True
        result["reason"] = f"falha: {exc}"
    result["status"] = "error" if result["failed"] else "ok"
    return result


@ray.remote  # sem num_gpus: Ray nao gerencia GPU, o pinning e CUDA_VISIBLE_DEVICES
def run_growth(exp: Experiment, point: dict, gpu_id: int,
               cpus: int = DEFAULT_CPUS_PER_EXPERIMENT) -> dict:
    """Roda o braco inteiro na `gpu_id`. Uma task = um braco = uma GPU, sem migracao."""
    return run_arm(exp, point, gpu_id, cpus)
