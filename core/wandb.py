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

"""core/wandb.py — cache local de metadados de runs do W&B e nomes de fine-tuning.

Funde `src/utils/wandb_cache.py` e `src/utils/get_names_wandb.py` em duas funcoes:

* `cached_history` — le/atualiza o snapshot em `configs/wandb_update/ids_wandb.json`.
* `resolve_run`    — converte nomes canonicos de experimento em nomes de run de
  fine-tuning, resolvendo colisao com sufixo de `run_id`.

O formato do JSON em disco e o mesmo de sempre::

    {"entity": ..., "project": ..., "updated_at": ..., "runs": {run_id: {...}}}

Nao ha camada de retry: uma chamada ao W&B que falha propaga a excecao, como antes.
"""
from __future__ import annotations

import datetime
import json
import os
import re
from collections import Counter

# Origem: src/utils/get_names_wandb.py:26-27. Migra para core/constants.py
# (WANDB_ENTITY / WANDB_PROJECT) quando aquele arquivo existir.
ENTITY = "ophira-ai"
PROJECT = "flim-ssl"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_ROOT, "configs", "wandb_update")
CACHE_FILE = os.path.join(CACHE_DIR, "ids_wandb.json")

# Nome canonico de experimento SSL; o segmento pct_* e descartado de proposito.
_CANONICAL_RE = re.compile(
    r"^lejepa_line_([a-z\-]+)_split_(\d+)_pct_\d+_model_(xavier|random|he|flim|trunc_normal)$"
)


def cached_history(
    entity: str = ENTITY,
    project: str = PROJECT,
    *,
    deduplicate: bool = False,
    update: bool = False,
    cache_path: str = CACHE_FILE,
) -> dict[str, dict]:
    """Retorna ``{run_id: metadata}`` a partir do cache local do W&B.

    `metadata` traz `name`, `created_at`, `state`, `config` e `summary` — o mesmo
    dicionario que ja esta gravado no JSON. Para o antigo ``{run_id: run_name}``
    basta ``{rid: m["name"] for rid, m in cached_history(...).items()}``.

    Comportamento:

    * ``update=False`` (padrao): le `cache_path`; se o arquivo nao existir, busca
      no W&B e grava uma copia nova.
    * ``update=True``: sempre busca no W&B e sobrescreve `cache_path`.
    * ``deduplicate=True``: mantem so o run mais recente (por `created_at`) de
      cada nome. Strings ISO-8601 comparam lexicograficamente.
    """
    cached = None if update else _load(cache_path)

    if cached is None:
        if not update:
            print(
                f"[W&B cache] No cache at "
                f"{os.path.relpath(cache_path, _ROOT)} — fetching from W&B …"
            )
        runs = _fetch_and_save(entity, project, cache_path)
    else:
        runs = cached.get("runs", {})
        print(
            f"[W&B cache] Using local cache "
            f"({len(runs)} runs, updated {cached.get('updated_at', 'unknown')})"
        )

    if not deduplicate:
        return runs

    best: dict[str, str] = {}  # name -> run_id mais recente
    for rid, info in runs.items():
        name = info["name"]
        cur = best.get(name)
        if cur is None or info.get("created_at", "") > runs[cur].get("created_at", ""):
            best[name] = rid
    return {rid: runs[rid] for rid in best.values()}


def resolve_run(
    experiments: dict[str, str] | None = None,
    *,
    entity: str = ENTITY,
    project: str = PROJECT,
    update: bool = False,
) -> dict[str, str]:
    """Retorna ``{run_id: nome_de_run_de_finetune}``, seguro contra colisao.

    O nome sai no padrao::

        finetune_lejepa_line_<dataset>_split_<N>_model_<init>

    Quando dois `run_id` caem no mesmo nome base (por exemplo `pct` diferente), o
    `run_id` e anexado para desambiguar::

        finetune_lejepa_line_helminth-eggs_split_1_model_xavier_7rkcbbnk

    Runs com nome fora do padrao canonico (formato legado `line_p*`) ficam de fora
    em silencio.

    Args:
        experiments: ``{run_id: run_name}``. Sem ele, usa o cache local
                     (`update=True` busca no W&B antes).
    """
    if experiments is None:
        experiments = {
            rid: info["name"]
            for rid, info in cached_history(entity, project, update=update).items()
        }

    bases = {
        rid: f"finetune_lejepa_line_{m.group(1)}_split_{m.group(2)}_model_{m.group(3)}"
        for rid, name in experiments.items()
        if (m := _CANONICAL_RE.match(name))
    }
    counts = Counter(bases.values())
    return {
        rid: f"{base}_{rid}" if counts[base] > 1 else base
        for rid, base in bases.items()
    }


def _load(cache_path: str) -> dict | None:
    """Le o JSON do cache; devolve None se o arquivo nao existir."""
    if not os.path.isfile(cache_path):
        return None
    with open(cache_path, encoding="utf-8") as fh:
        return json.load(fh)


def _fetch_and_save(entity: str, project: str, cache_path: str) -> dict[str, dict]:
    """Busca todos os runs no W&B, grava em `cache_path` e devolve `{run_id: meta}`.

    So `name`, `created_at` e `state` sao lidos — `config` e `summary` ficam vazios
    porque cada acesso a eles dispara uma chamada lazy por run, que domina a
    latencia em projetos com 1000+ runs. `per_page=1000` minimiza os round-trips.
    """
    import wandb  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    print(f"[W&B cache] Refreshing from {entity}/{project} …")
    runs = {
        run.id: {
            "name":       run.name,
            "created_at": run.created_at or "",
            "state":      getattr(run, "state", ""),
            "config":     {},
            "summary":    {},
        }
        for run in tqdm(
            wandb.Api().runs(f"{entity}/{project}", per_page=1000),
            desc=f"Fetching {entity}/{project}", unit="run",
        )
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "entity": entity,
                "project": project,
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "runs": runs,
            },
            fh,
            indent=2,
        )
    print(
        f"[W&B cache] Saved {len(runs)} runs → "
        f"{os.path.relpath(cache_path, _ROOT)}"
    )
    return runs
