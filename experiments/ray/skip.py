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

"""skip.py — a unica resposta para "esta celula ja rodou?".

Cinco flags diziam isso antes, cada uma no seu launcher: `--resume`,
`--ignore-existing`, `--skip-existing`, `--check-wandb` e `--retry`. Agora e um
campo so, `Experiment.skip` (schema.py:204), com tres valores:

* ``none``  — nunca pula; roda a grade inteira (era `--ignore-existing`/`--retry`).
* ``state`` — pula quem tem `run_metadata.json` com ``status: ok`` no
  `output.work_dir` (era `--skip-existing` sem `--check-wandb`, e tambem o
  `--resume` que lia um `ExecutionState` JSON a parte).
* ``wandb`` — o W&B decide, com o metadata local como desempate (era
  `--skip-existing --check-wandb`).

Nao ha camada de compatibilidade com os nomes antigos: eles morreram, o
MIGRATION.md e quem os traduz.

O acesso ao W&B e o de `core.wandb.cached_history` — nao ha um segundo caminho.
As 3 copias de `_query_wandb_state` abriam `wandb.Api(timeout=15)` e faziam UMA
chamada POR CELULA; aqui o snapshot inteiro do projeto vem numa ida so, indexado
por `display_name`, e o resto da grade le do indice.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

from core.constants import RUN_METADATA_FILENAME
from core.wandb import CACHE_FILE, ENTITY, PROJECT, cached_history
from experiments.constants import WANDB_ENTITY, WANDB_PROJECT
from experiments.ray.schema import SKIP_MODES


def query_wandb_state(
    run_name: str, entity: str | None = None, project: str | None = None
) -> tuple[str, str]:
    """Estado do run mais recente com esse `display_name`, e o id dele.

    O estado e um de: ``finished``, ``running``, ``failed``, ``not_found``
    (nao ha registro no W&B) ou ``error`` (o W&B nao respondeu).
    """
    index = _index(entity or WANDB_ENTITY, project or WANDB_PROJECT)
    if index is None:
        return "error", ""
    return index.get(run_name, ("not_found", ""))


def should_skip(
    run_name: str,
    skip: str,
    work_dir: str,
    entity: str | None = None,
    project: str | None = None,
) -> tuple[bool, str]:
    """(pular, motivo) para uma celula. O motivo vai inteiro para o manifesto.

    Ordem de decisao no modo ``wandb``: running e finished pulam; failed volta
    para a fila, a menos que o metadata local diga que o treino terminou
    (o W&B perde o fim do run quando o processo morre depois de salvar); erro de
    API volta para a fila, que e o default seguro; sem registro, cai no local.
    """
    if skip not in SKIP_MODES:
        raise ValueError(f"skip: {skip!r} invalido; validos: {list(SKIP_MODES)}")

    if skip == "none":
        return False, "skip:none"

    if skip == "wandb":
        state, run_id = query_wandb_state(run_name, entity, project)
        if state == "running":
            return True, f"wandb:running({run_id})"
        if state == "finished":
            return True, f"wandb:finished({run_id})"
        if state == "failed":
            if _local(work_dir, run_name)[0]:
                return True, f"wandb:crashed+local:ok({run_id})"
            return False, f"wandb:failed({run_id})"
        if state == "error":
            return False, "wandb:api_error"
        # not_found: cai no metadata local

    return _local(work_dir, run_name)


def _local(work_dir: str, run_name: str) -> tuple[bool, str]:
    """Metadata local do run: so ``status: ok`` conta como feito."""
    path = os.path.join(work_dir, run_name, RUN_METADATA_FILENAME)
    if not os.path.isfile(path):
        return False, "no_metadata"
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return False, "metadata_read_error"
    if meta.get("status") == "ok":
        return True, "local:ok"
    return False, f"local:{meta.get('status', 'unknown')}"


def _cache_path(entity: str, project: str) -> str:
    """Cache por projeto. O par default fica no caminho historico, para nao
    invalidar o `ids_wandb.json` que os relatorios ja leem."""
    if (entity, project) == (ENTITY, PROJECT):
        return CACHE_FILE
    slug = f"{entity}_{project}".replace("/", "_")
    return os.path.join(os.path.dirname(CACHE_FILE), f"ids_wandb_{slug}.json")


@lru_cache(maxsize=None)
def _index(entity: str, project: str) -> dict[str, tuple[str, str]] | None:
    """``{display_name: (estado, run_id)}``, uma ida ao W&B por processo.

    `deduplicate=True` mantem so o run mais recente de cada nome, que e o
    `order="-created_at"` + primeiro-que-casa das copias. `None` (e nao dict
    vazio) quando a consulta falha: projeto sem run nenhum e uma resposta,
    W&B fora do ar e outra, e as duas levam a decisoes diferentes.
    """
    # O cache do core.wandb tem UM caminho fixo (configs/wandb_update/ids_wandb.json),
    # que nao depende do projeto: consultar outro projeto sobrescreveria o cache do
    # projeto default, que e versionado e lido pelos relatorios. Derivamos um arquivo
    # por (entity, project); o default continua no caminho historico.
    try:
        runs = cached_history(entity, project, deduplicate=True, update=True,
                              cache_path=_cache_path(entity, project))
    except Exception as exc:
        print(f"[skip] W&B indisponivel ({exc}); nada sera pulado por wandb.", flush=True)
        return None
    return {
        info["name"]: (_state(info.get("state", "")), run_id)
        for run_id, info in runs.items()
    }


def _state(raw: str) -> str:
    """Estado cru do W&B -> vocabulario desta decisao.

    Sem `WANDB_FAILED_STATES`: as 3 copias testavam esse conjunto primeiro e
    depois caiam num `return "failed"` para todo o resto
    (distillation_conv_ray.py:219-227), e como crashed/failed/killed nunca sao
    `finished` nem `running`, o teste nunca mudou o resultado.
    """
    if raw in ("finished", "running"):
        return raw
    return "failed"
