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
"""check_distill_conv_status.py — Status completo dos experimentos next_layers_direct.

Cruza 4 fontes de informação:
  1. Processo do SO  — ps aux (está rodando agora no sistema?)
  2. W&B             — estado do run na API (finished/running/failed/crashed)
  3. Metadata local  — artifacts/distillation/<run>/run_metadata.json
  4. Checkpoint      — artifacts/distillation/<run>/checkpoints/last.ckpt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import wandb

_ENTITY  = "ophira-ai"
_PROJECT = "flim-ssl"

_DATASETS  = ["eggs", "larvae", "protozoan"]
_SPLITS    = [1, 2, 3]
_PCTS      = [1, 5, 25, 50, 75, 100]
_DIST_TYPE = "direct"

_WANDB_FAILED = {"crashed", "failed", "killed"}


def run_name(dataset, split, pct):
    return f"distillation_{dataset}_split{split}_pct{pct}_next_layers_{_DIST_TYPE}"


# ── Fonte 1: processos do SO ───────────────────────────────────────────────────

def _get_running_processes() -> set[str]:
    """Retorna run_names que têm um processo python ativo no SO agora."""
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return set()

    active = set()
    for line in out.splitlines():
        if "distillation_conv_module" not in line or "--run-name" not in line:
            continue
        parts = line.split("--run-name")
        if len(parts) < 2:
            continue
        rn = parts[1].strip().split()[0]
        active.add(rn)
    return active


# ── Fonte 2: W&B ──────────────────────────────────────────────────────────────

def _get_wandb_states() -> dict[str, tuple[str, str]]:
    """Retorna {run_name: (state, run_id)} para runs next_layers_direct."""
    print("Consultando W&B...", flush=True)
    api = wandb.Api(timeout=20)
    all_runs = api.runs(
        f"{_ENTITY}/{_PROJECT}",
        filters={"display_name": {"$regex": "next_layers_direct"}},
        order="-created_at",
    )
    states: dict[str, tuple[str, str]] = {}
    for run in all_runs:
        name = run.display_name
        if name not in states:
            states[name] = (run.state, run.id)
    return states


# ── Fonte 3: metadata local ────────────────────────────────────────────────────

def _get_local_status(rn: str) -> str:
    """Retorna 'ok', 'error', ou 'missing'."""
    path = os.path.join(_ROOT, "artifacts", "distillation", rn, "run_metadata.json")
    if not os.path.isfile(path):
        return "missing"
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta.get("status", "unknown")
    except Exception:
        return "unreadable"


# ── Fonte 4: checkpoint ────────────────────────────────────────────────────────

def _has_checkpoint(rn: str) -> bool:
    ckpt_dir = os.path.join(_ROOT, "artifacts", "distillation", rn, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    return any(
        f.endswith(".ckpt")
        for f in os.listdir(ckpt_dir)
    )


# ── Decisão final ─────────────────────────────────────────────────────────────

def _classify(
    rn:         str,
    proc_set:   set[str],
    wb_states:  dict[str, tuple[str, str]],
) -> dict:
    local_status = _get_local_status(rn)
    has_ckpt     = _has_checkpoint(rn)
    in_proc      = rn in proc_set
    wb_state, wb_id = wb_states.get(rn, ("not_found", ""))

    # Decisão de categoria (em ordem de prioridade)
    if wb_state == "finished" and local_status == "ok":
        category = "done"
    elif wb_state == "finished" and local_status != "ok":
        category = "done_wandb_only"       # W&B ok mas metadata local diverge
    elif in_proc or wb_state == "running":
        category = "running"
    elif wb_state in _WANDB_FAILED:
        category = "failed"
    elif local_status == "ok":
        category = "done_local_only"       # ok local mas sem W&B record
    else:
        category = "missing"

    return {
        "run_name":     rn,
        "category":     category,
        "in_process":   in_proc,
        "wb_state":     wb_state,
        "wb_id":        wb_id,
        "local_status": local_status,
        "has_ckpt":     has_ckpt,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    proc_set  = _get_running_processes()
    wb_states = _get_wandb_states()

    results = []
    for dataset in _DATASETS:
        for split in _SPLITS:
            for pct in _PCTS:
                rn = run_name(dataset, split, pct)
                results.append(_classify(rn, proc_set, wb_states))

    done           = [r for r in results if r["category"] in ("done", "done_wandb_only", "done_local_only")]
    running        = [r for r in results if r["category"] == "running"]
    failed         = [r for r in results if r["category"] == "failed"]
    missing        = [r for r in results if r["category"] == "missing"]
    divergent      = [r for r in results if r["category"] in ("done_wandb_only", "done_local_only")]

    total = len(results)

    print()
    print("=" * 72)
    print("STATUS COMPLETO — distillation next_layers_direct")
    print("=" * 72)
    print(f"  Total esperado  : {total}")
    print(f"  Concluídos      : {len(done)}")
    print(f"  Em progresso    : {len(running)}")
    print(f"  Falharam        : {len(failed)}")
    print(f"  Nunca rodaram   : {len(missing)}")
    print(f"  Faltam terminar : {total - len(done)}")

    if running:
        print()
        print(f"─── EM PROGRESSO ({len(running)}) ───────────────────────────────────────")
        for r in running:
            proc_tag = "[proc=YES]" if r["in_process"] else "[proc=NO ]"
            ckpt_tag = "[ckpt]" if r["has_ckpt"] else "[     ]"
            print(f"  {proc_tag} {ckpt_tag}  wb={r['wb_state']:<10}  local={r['local_status']:<10}  {r['run_name']}")

    if failed:
        print()
        print(f"─── FALHARAM ({len(failed)}) ─────────────────────────────────────────────")
        for r in failed:
            proc_tag = "[proc=YES]" if r["in_process"] else "[proc=NO ]"
            ckpt_tag = "[ckpt]" if r["has_ckpt"] else "[     ]"
            print(f"  {proc_tag} {ckpt_tag}  wb={r['wb_state']:<10}  local={r['local_status']:<10}  {r['run_name']}")

    if missing:
        print()
        print(f"─── NUNCA RODARAM ({len(missing)}) ──────────────────────────────────────")
        for r in missing:
            print(f"  {r['run_name']}")

    if divergent:
        print()
        print(f"─── DIVERGENTES local≠W&B ({len(divergent)}) — verificar manualmente ───")
        for r in divergent:
            print(f"  category={r['category']}  wb={r['wb_state']}  local={r['local_status']}  {r['run_name']}")

    print()
    print(f"  Faltam terminar: {total - len(done)} de {total}")
    print("=" * 72)


if __name__ == "__main__":
    main()
