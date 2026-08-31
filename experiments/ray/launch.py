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

"""launch.py — a entrada unica do launcher Ray.

    python -m experiments.ray.launch <caminho.yaml> [--dry-run] [--fail-fast]
                                     [--gpu-ids 0 1] [--set chave=valor ...]

Substitui os seis lancadores de `scripts/*_ray.py` + `scripts/spifil_growth_loop.py`.
Este arquivo NAO sabe nada de metodo, de dataset nem de hparam: ele le UM
experiment YAML, expande a grade com `experiments.ray.grid`, e entrega ponto por
ponto ao runner declarado em `runner:`. Trocar `method:` no YAML nao muda uma
linha aqui.

## O argparse, que e a unica excecao do refactor

A regra do documento e "nenhum entrypoint novo nasce com argparse"; a excecao
esta escrita em spec_refactor.md:246-248 e vale SO para as quatro flags abaixo
mais o caminho posicional do YAML. Nenhuma delas pode ser chave do YAML porque
todas descrevem a INVOCACAO, nao o experimento:

* `--dry-run`   e a pergunta "o que voce rodaria?" feita ao MESMO YAML que vai
                rodar depois; como chave, o YAML mentiria sobre si mesmo e o
                usuario teria que edita-lo duas vezes por corrida.
* `--fail-fast` e politica de reacao a falha desta corrida (abortar a fila x
                deixar os irmaos terminarem), nao propriedade do experimento.
* `--gpu-ids`   e o estado da MAQUINA no minuto da submissao (quem esta livre),
                nao do experimento; `resources.gpu_ids` continua no YAML e esta
                flag apenas o sobrepoe.
* `--set`       e o escape pontual para uma corrida (repetir so um percentual,
                por exemplo) sem sujar o YAML versionado com uma edicao que se
                esquece de desfazer.

Precedencia, em uma frase: flag de CLI > YAML > default do schema. Ela e obtida
sem `if` espalhado — a flag e aplicada SOBRE o mapeamento cru do YAML, e o
`Experiment` ja nasce com o valor final (`_effective_yaml`).

## Pre-voo antes de tudo

`experiments.ray.preflight.preflight(caminho)` roda OBRIGATORIAMENTE antes de
`ray.init()` e antes de qualquer import pesado (por isso `ray`, `grid` e os
runners sao importados dentro das funcoes, nunca no topo: importar modulo de
modelo puxa torch e pode alocar CUDA, exatamente o que o pre-voo existe para
denunciar). O pre-voo levanta `SystemExit(1)` sozinho quando acha erro; se o
modulo nao existir, este arquivo falha alto em vez de seguir sem checagem.

## GPU: o desenho que nao muda

`ray.init(num_gpus=0)`, task `@ray.remote` SEM `num_gpus`, e cada task recebe um
`gpu_id` e fixa `CUDA_VISIBLE_DEVICES` no env do FILHO via `subprocess(env=...)`.
Quem escolhe a GPU e o `GpuSlotScheduler`, nunca o Ray — declarar `num_gpus=1`
tiraria do usuario a escolha de QUAL GPU e impediria mais de um job por GPU.
A concorrencia sai de `scheduler.total_slots()` e nunca de
`len(gpu_ids) * max_per_gpu`: com `max_per_gpu` como dict essa conta e TypeError.

## "Ja rodou?" tem UMA resposta

O launcher pergunta so a `experiments.ray.skip`, e nem isso diretamente: quem
chama `should_skip` e o `grid.build` (experiments/ray/grid.py:80-88), que ja
devolve os pulados com o motivo por extenso. Este arquivo NAO abre um
`ExecutionState` para os runners `train`/`growth` — duas fontes respondendo a
mesma pergunta seria bug, e o campo `skip:` do YAML e a fonte declarada. O
`ExecutionState` continua vivo dentro de `runners/eval.py`, que tem fila e
orquestracao proprias.

## O log

O `tee` sai da linha de comando: o launcher grava `<log_dir>/<name>_<timestamp>.log`
sozinho, duplicando stdout E stderr do driver (o proprio e o dos workers, que
chegam por `log_to_driver=True`) — o `2>&1` do comando antigo tambem esta aqui
dentro. Em `--dry-run` nao ha log e nao ha `work_dir`: dry-run nao escreve nada
em disco.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import asdict

import yaml

from experiments.constants import LOG_TIME_FMT, SEP_WIDTH
from experiments.ray import schema

# Sufixo do arquivo de log, igual ao `date +%Y%m%d_%H%M%S` que estava no `tee`
# da linha de comando de hoje (spec_refactor.md:1743).
LOG_STAMP_FMT: str = "%Y%m%d_%H%M%S"


def _log(msg: str, level: str = "INFO") -> None:
    """Uma linha de estado por evento: submetido, terminou, falhou, pulado."""
    print(f"[{time.strftime(LOG_TIME_FMT)}][{level}] {msg}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse(argv: list[str] | None) -> argparse.Namespace:
    """As quatro flags autorizadas e o caminho do YAML. Nao acrescente uma quinta."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.ray.launch",
        description="Entrada unica do launcher Ray: recebe UM experiment YAML.",
    )
    parser.add_argument("config", help="caminho do experiment YAML")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="imprime cada comando montado e sai sem executar nada",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="na primeira falha, descarta a fila pendente",
    )
    parser.add_argument(
        "--gpu-ids", nargs="+", type=int, metavar="ID",
        help="sobrepoe resources.gpu_ids do YAML",
    )
    parser.add_argument(
        "--set", action="append", default=[], dest="sets", metavar="CHAVE=VALOR",
        help="sobrepoe uma chave do YAML por caminho pontilhado, ex. "
             "--set grid.percentages=[5,50]",
    )
    return parser.parse_args(argv)


def _assign(data: dict, dotted: str, value) -> None:
    """Escreve `value` no caminho pontilhado, criando os mapeamentos que faltarem."""
    keys = dotted.split(".")
    node = data
    for depth, key in enumerate(keys[:-1]):
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        if not isinstance(child, dict):
            prefix = ".".join(keys[: depth + 1])
            raise SystemExit(
                f"--set {dotted}: '{prefix}' nao e mapeamento no YAML, e "
                f"{type(child).__name__}; nao ha onde descer"
            )
        node = child
    node[keys[-1]] = value


def _effective_yaml(
    path: str, sets: list[str], gpu_ids: list[int] | None, tmp_dir: str
) -> str:
    """Caminho do YAML que o schema E o pre-voo vao ler — o mesmo para os dois.

    Sem `--set` e sem `--gpu-ids` e o proprio arquivo do usuario, intocado. Com
    qualquer um dos dois, o YAML ja sobreposto e gravado no diretorio temporario
    e e ELE que os dois leem: o pre-voo tem que checar a grade que VAI rodar, nao
    a que esta no disco. Um `--set grid.percentages=[5]` que o pre-voo nao visse
    faria a corrida morrer por um caminho ausente de um percentual excluido.

    O tipo do lado direito sai de `yaml.safe_load`, o mesmo parser do arquivo:
    `5` vira int, `[5,50]` vira lista, `true` vira bool e `mlp` continua str.
    Um parser so, sem tabela de conversao propria.
    """
    if not sets and not gpu_ids:
        return path
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(
            f"{path}: topo do YAML nao e mapeamento; --set/--gpu-ids nao tem onde escrever"
        )
    for assignment in sets:
        dotted, sep, raw = assignment.partition("=")
        if not sep or not dotted.strip():
            raise SystemExit(f"--set: esperava chave=valor, veio {assignment!r}")
        _assign(data, dotted.strip(), yaml.safe_load(raw))
    if gpu_ids:
        _assign(data, "resources.gpu_ids", list(gpu_ids))
    out = os.path.join(tmp_dir, os.path.basename(path))
    with open(out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return out


# ─── Pre-voo ──────────────────────────────────────────────────────────────────


def _preflight(yaml_path: str) -> None:
    """Pre-voo obrigatorio. Sem ele, nenhum job e submetido — nem com --dry-run.

    O import fica aqui dentro, e nao no topo, pelo mesmo motivo dos runners: o
    pre-voo tem que rodar ANTES dos imports pesados do launcher.
    """
    try:
        from experiments.ray.preflight import preflight
    except ImportError as exc:
        raise SystemExit(
            "launch: nao consegui importar experiments.ray.preflight.preflight "
            f"({exc}).\n"
            "       O pre-voo e obrigatorio antes de ray.init(); nenhum job e "
            "submetido sem ele.\n"
            "       Crie experiments/ray/preflight.py com "
            "preflight(exp_path: str, root: str = '.') -> list[dict]."
        ) from None
    # O pre-voo imprime o relatorio e levanta SystemExit(1) sozinho quando acha
    # erro. A lista de pontos que ele devolve NAO e usada: quem monta a fila e o
    # grid.build, que e o unico dono da grade neste pacote.
    preflight(yaml_path)


# ─── Log ──────────────────────────────────────────────────────────────────────


class _Tee:
    """Um fluxo em dois lugares: o terminal e o arquivo de log do launcher."""

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        self._stream.flush()
        self._handle.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        # Ray pergunta o fd do fluxo do driver; devolver o do terminal mantem o
        # comportamento de antes. O que passar por fd cru nao entra no arquivo.
        return self._stream.fileno()


def _open_log(exp: schema.Experiment):
    """`<log_dir>/<name>_<timestamp>.log`, criado pelo launcher, sem `tee`."""
    os.makedirs(exp.output.log_dir, exist_ok=True)
    path = os.path.join(
        exp.output.log_dir, f"{exp.name}_{time.strftime(LOG_STAMP_FMT)}.log"
    )
    return path, open(path, "w", encoding="utf-8", buffering=1)


# ─── Comandos de um ponto ─────────────────────────────────────────────────────


def _cmds(exp: schema.Experiment, point: dict) -> list[list[str]]:
    """Os comandos de UM ponto — os mesmos que o caminho normal executa.

    `train` e um comando; `growth` e a cadeia de estagios do braco, montada pelo
    proprio `run_arm` em modo dry-run. Nao ha segunda montagem em lugar nenhum:
    dois caminhos fariam o dry-run testar codigo que nao roda em producao.
    """
    from experiments.ray.runners import train as train_runner

    if exp.runner == "train":
        return [train_runner.build_cmd(exp, point)]

    from experiments.ray.runners import growth as growth_runner

    # gpu_ids[0] aparece so no `--device=cuda:` da linha ficticia do grow; a GPU
    # real de cada braco e a que o escalonador entrega em tempo de execucao.
    return growth_runner.run_arm(
        exp,
        point,
        exp.resources.gpu_ids[0],
        exp.resources.cpus_per_experiment,
        dry_run=True,
    )["plan"]


# ─── --dry-run ────────────────────────────────────────────────────────────────


def _plan(exp: schema.Experiment) -> int:
    """Imprime tudo que seria submetido e sai. Nada de Ray, nada de disco."""
    if exp.runner == "eval":
        # O runner eval tem plano proprio (glob de configs), nao grade cartesiana.
        from experiments.ray.runners import eval as eval_runner

        eval_runner.run(exp, dry_run=True)
        return 0

    from experiments.ray import grid

    cells, skipped = grid.build(exp)
    print("=" * SEP_WIDTH)
    print(
        f"DRY RUN — {exp.name} (runner {exp.runner}) — "
        f"{len(cells)} ponto(s), {len(skipped)} pulado(s)"
    )
    print("=" * SEP_WIDTH)
    for entry in skipped:
        print(f"  SKIP  {entry['run_name']}  ({entry['reason']})")
    total = 0
    for cell in cells:
        print(f"  {cell.run_name}")
        for cmd in _cmds(exp, asdict(cell)):
            print(f"    {' '.join(cmd)}")
            total += 1
    print("=" * SEP_WIDTH)
    print(f"{len(cells)} ponto(s), {total} comando(s). Nada foi executado.")
    return 0


# ─── Execucao ─────────────────────────────────────────────────────────────────


def _queue(exp: schema.Experiment, cells: list, *, fail_fast: bool) -> list[dict]:
    """A fila: um slot de GPU por ponto, submetido assim que um slot vagar."""
    import ray

    from experiments.ray.gpu_slot_scheduler import GpuSlotScheduler
    # `_ray_init_kwargs` ja resolve `num_gpus=0` local x `address` do cluster e
    # ja dimensiona `num_cpus` por `total_slots()`. E o unico lugar do pacote que
    # sabe disso — reusar e melhor que uma segunda copia aqui.
    from experiments.ray.runners.eval import _ray_init_kwargs

    if exp.runner == "train":
        from experiments.ray.runners.train import run_train as remote
    else:
        from experiments.ray.runners.growth import run_growth as remote

    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs(exp))

    scheduler = GpuSlotScheduler(
        gpu_ids=exp.resources.gpu_ids,
        max_per_gpu=exp.resources.max_per_gpu,
        quota=exp.resources.max_total_per_gpu,
    )
    queue = [asdict(cell) for cell in cells]
    total = len(queue)
    futures: dict = {}
    rows: list[dict] = []
    submitted = completed = 0
    had_failure = False

    _log(
        f"{total} ponto(s) | {len(exp.resources.gpu_ids)} GPU(s) | "
        f"{scheduler.total_slots()} concorrentes"
    )

    def submit() -> bool:
        nonlocal submitted
        if not queue or (fail_fast and had_failure):
            return False
        gpu_id = scheduler.pick_gpu()
        if gpu_id is None:
            return False
        point = queue.pop(0)
        future = remote.options(
            num_cpus=exp.resources.cpus_per_experiment
        ).remote(exp, point, gpu_id, exp.resources.cpus_per_experiment)
        scheduler.acquire(gpu_id)
        futures[future] = (gpu_id, point)
        submitted += 1
        _log(
            f"[{submitted:>4}/{total}] SUBMIT  gpu={gpu_id}  "
            f"[{scheduler.status_line()}]  fila={len(queue)}  {point['run_name']}"
        )
        return True

    while queue and scheduler.has_free_slot():
        submit()

    while futures:
        done, _ = ray.wait(list(futures), num_returns=1, timeout=None)
        future = done[0]
        gpu_id, point = futures.pop(future)
        try:
            result = ray.get(future)
        except Exception as exc:  # noqa: BLE001
            result = {
                "gpu_id": gpu_id,
                "status": "ray_error",
                "returncode": None,
                "error": str(exc),
            }
        finally:
            # release nunca so no caminho feliz: um ponto travado nao pode
            # segurar o slot da GPU pelo resto da corrida.
            scheduler.release(gpu_id)

        completed += 1
        # `run_arm` devolve `tag`, `run_train` devolve `run_name`; o nome
        # canonico do ponto vem do grid e serve para os dois.
        result.setdefault("run_name", point["run_name"])
        rows.append(result)

        if result.get("status") == "ok":
            _log(
                f"[{completed:>4}/{total}] OK      gpu={gpu_id}  "
                f"[{scheduler.status_line()}]  fila={len(queue)}  {result['run_name']}"
            )
        else:
            detail = result.get("error") or result.get("reason", "")
            _log(
                f"[{completed:>4}/{total}] "
                f"{str(result.get('status', '?')).upper():<9} gpu={gpu_id}  "
                f"{result['run_name']}: {detail}",
                "WARN",
            )
            had_failure = True
            if fail_fast:
                _log("[FAIL-FAST] descartando a fila pendente.", "WARN")
                queue.clear()

        while queue and scheduler.has_free_slot():
            submit()

    ray.shutdown()
    n_ok = sum(1 for row in rows if row.get("status") == "ok")
    _log(f"Fim — {n_ok} ok, {len(rows) - n_ok} falha(s).")
    return rows


def _execute(exp: schema.Experiment, *, fail_fast: bool) -> int:
    """Roda de verdade. Devolve o codigo de saida do processo."""
    if exp.runner == "eval":
        from experiments.ray.runners import eval as eval_runner

        return _exit_code(eval_runner.run(exp, fail_fast=fail_fast))

    from experiments.ray import grid

    ok, why = grid.validate_output_dir(exp.output.work_dir)
    if not ok:
        raise SystemExit(
            f"launch: output.work_dir nao e gravavel ({exp.output.work_dir}): {why}"
        )

    cells, skipped = grid.build(exp)
    for entry in skipped:
        _log(f"SKIP    {entry['run_name']}  ({entry['reason']})")
    if not cells:
        _log(f"Nada a fazer ({len(skipped)} pulado(s)).")
        return 0
    return _exit_code(_queue(exp, cells, fail_fast=fail_fast))


def _exit_code(rows: list[dict]) -> int:
    """0 so quando nenhum ponto falhou; qualquer falha e saida 1."""
    return 1 if any(row.get("status") != "ok" for row in rows) else 0


# ─── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)

    # O diretorio temporario segura o YAML ja sobreposto ate o pre-voo terminar
    # de le-lo; depois disso so o `Experiment` importa.
    with tempfile.TemporaryDirectory(prefix="ray_launch_") as tmp_dir:
        yaml_path = _effective_yaml(args.config, args.sets, args.gpu_ids, tmp_dir)
        try:
            exp = schema.load(yaml_path)
        except (OSError, ValueError) as exc:
            # O YAML sobreposto vive no tmp; o usuario tem que ler o nome DELE.
            raise SystemExit(
                f"launch: {str(exc).replace(yaml_path, args.config)}"
            ) from None

        if args.dry_run:
            _preflight(yaml_path)
            return _plan(exp)

        log_path, handle = _open_log(exp)
        stdout, stderr = sys.stdout, sys.stderr
        # O `2>&1 | tee` de antes capturava os dois fluxos; o launcher tambem.
        # O banner do Ray e os warnings saem por stderr, e eles fazem parte do log.
        sys.stdout, sys.stderr = _Tee(stdout, handle), _Tee(stderr, handle)
        try:
            _log(f"experiment {args.config} -> {exp.name} "
                 f"(method {exp.method}, runner {exp.runner})")
            if args.sets or args.gpu_ids:
                _log(f"sobreposto por CLI: --set {args.sets} --gpu-ids {args.gpu_ids}")
            _log(f"log em {log_path}")
            _preflight(yaml_path)
            return _execute(exp, fail_fast=args.fail_fast)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
