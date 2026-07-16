"""report_distill_disk_wandb.py — Relatório de disco + W&B dos experimentos de distillation.

Para cada experimento em ``artifacts/distillation/<run>/``:
  * espaço total ocupado pelo diretório (inclui wandb/ logs/ checkpoints/)
  * checkpoint *best*  — caminho + tamanho
  * checkpoint *last*  — caminho + tamanho
  * status local       — run_metadata.json
  * cruzamento W&B     — state, run_id, melhor val/loss

Saída (por default dentro de artifacts/distillation/):
  * tabela impressa no terminal (uma linha por experimento)
  * artifacts/distillation/distill_disk_wandb_report.csv  — relatório completo
  * artifacts/distillation/distill_divergences.csv        — só divergências W&B×local

A coluna ``category`` consolida o cruzamento: done / divergent / failed / missing.
Use o CSV de divergências para verificar possíveis inconsistências entre o estado
do run no W&B (finished/failed/crashed/not_found) e o status local (run_metadata.json).

Uso:
    python scripts/report_distill_disk_wandb.py                 # com W&B
    python scripts/report_distill_disk_wandb.py --no-wandb      # só disco (offline)
    python scripts/report_distill_disk_wandb.py --csv outro.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_ARTIFACTS = os.path.join(_ROOT, "artifacts", "distillation")
_RESULTS   = os.path.join(_ROOT, "results")

_ENTITY  = "ophira-ai"
_PROJECT = "flim-ssl"


# ── Helpers de disco ────────────────────────────────────────────────────────────

def _dir_size_bytes(path: str) -> int:
    """Soma recursiva do tamanho de todos os arquivos sob ``path``."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _human(n: int) -> str:
    """Bytes → string legível (KiB/MiB/GiB)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{size:.1f} TiB"


def _loss_from_name(path: str) -> float:
    """Extrai val/loss do nome do checkpoint best (menor = melhor)."""
    hit = re.search(r"loss=([0-9.]+)\.ckpt$", path)
    return float(hit.group(1)) if hit else float("inf")


def _find_checkpoints(run_dir: str) -> tuple[str | None, str | None]:
    """Retorna (best_ckpt_path, last_ckpt_path) ou None se ausente.

    O PL salva o best com 'val/loss' no nome, criando um subdir
    'best-epoch=NNN-val/loss=X.ckpt' — por isso busca recursiva.
    """
    ckpt_root = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        return None, None
    all_ckpts = glob.glob(os.path.join(ckpt_root, "**", "*.ckpt"), recursive=True)
    last = next((c for c in all_ckpts if os.path.basename(c) == "last.ckpt"), None)
    bests = [c for c in all_ckpts if os.path.basename(c) != "last.ckpt"]
    best = min(bests, key=_loss_from_name) if bests else None
    return best, last


def _file_size(path: str | None) -> int:
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _local_meta(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


# ── Categoria consolidada ───────────────────────────────────────────────────────

def _category(wandb_state: str, local_status: str) -> str:
    """Classifica o experimento cruzando W&B × status local.

    Prioridade:
      * "done"      : W&B finished E local ok
      * "skipped"   : W&B pulado (--no-wandb) → done se local ok, senão missing
      * "failed"    : W&B failed/crashed E local não-ok
      * "divergent" : W&B e local discordam (XOR de "ok")
      * "missing"   : restante (local ausente/erro, sem falha/divergência)
    """
    local_ok = local_status == "ok"
    if wandb_state == "skipped":
        # W&B não foi consultado — nada a comparar, não marca divergente.
        return "done" if local_ok else "missing"
    if wandb_state == "finished" and local_ok:
        return "done"
    if wandb_state in {"failed", "crashed"} and not local_ok:
        return "failed"
    if (wandb_state == "finished") != local_ok:   # XOR: discordância W&B × local
        return "divergent"
    return "missing"


# ── W&B ───────────────────────────────────────────────────────────────────────

def _fetch_wandb() -> dict[str, dict]:
    """Retorna {display_name: {state, id, best_val_loss}} para runs de distillation."""
    import wandb
    print("Consultando W&B (pode demorar)…", flush=True)
    api = wandb.Api(timeout=30)
    runs = api.runs(
        f"{_ENTITY}/{_PROJECT}",
        filters={"display_name": {"$regex": "^distillation_"}},
        order="-created_at",
    )
    out: dict[str, dict] = {}
    for run in runs:
        name = run.display_name
        if name in out:                      # mantém o mais recente (order=-created_at)
            continue
        summary = run.summary or {}
        best = summary.get("val/loss", summary.get("val_loss", ""))
        out[name] = {"state": run.state, "id": run.id, "best_val_loss": best}
    print(f"  {len(out)} runs distillation_* encontrados no W&B.", flush=True)
    return out


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", default=_ARTIFACTS)
    ap.add_argument("--csv", default=os.path.join(_ARTIFACTS, "distill_disk_wandb_report.csv"),
                    help="CSV completo (default: dentro de artifacts/distillation/).")
    ap.add_argument("--divergences-csv",
                    default=os.path.join(_ARTIFACTS, "distill_divergences.csv"),
                    help="CSV só com experimentos divergentes W&B×local (default: artifacts/distillation/).")
    ap.add_argument("--no-wandb", action="store_true", help="Pula o cruzamento com W&B.")
    args = ap.parse_args()

    if not os.path.isdir(args.artifacts_dir):
        print(f"[ERRO] {args.artifacts_dir} não existe.")
        sys.exit(1)

    run_dirs = sorted(
        e.path for e in os.scandir(args.artifacts_dir) if e.is_dir()
    )

    wb: dict[str, dict] = {}
    if not args.no_wandb:
        try:
            wb = _fetch_wandb()
        except Exception as exc:
            print(f"[WARN] W&B indisponível ({exc}). Seguindo só com disco.", flush=True)

    rows: list[dict] = []
    for rd in run_dirs:
        name = os.path.basename(rd)
        meta = _local_meta(rd)
        best, last = _find_checkpoints(rd)
        dir_b  = _dir_size_bytes(rd)
        best_b = _file_size(best)
        last_b = _file_size(last)
        wbi = wb.get(name, {})
        wandb_state  = wbi.get("state", "not_found" if wb else "skipped")
        local_status = meta.get("status", "missing")

        rows.append({
            "run_name":           name,
            "dataset":            meta.get("dataset", ""),
            "split":              meta.get("split", ""),
            "percentage":         meta.get("percentage", ""),
            "proj_head":          meta.get("proj_head", ""),
            "distillation_type":  meta.get("distillation_type", ""),
            "encoder_init":       meta.get("encoder_init", ""),
            "local_status":       local_status,
            "category":           _category(wandb_state, local_status),
            "dir_size_bytes":     dir_b,
            "dir_size_human":     _human(dir_b),
            "best_ckpt_path":     os.path.relpath(best, _ROOT) if best else "",
            "best_ckpt_bytes":    best_b,
            "best_ckpt_human":    _human(best_b) if best else "",
            "last_ckpt_path":     os.path.relpath(last, _ROOT) if last else "",
            "last_ckpt_bytes":    last_b,
            "last_ckpt_human":    _human(last_b) if last else "",
            "wandb_state":        wandb_state,
            "wandb_id":           wbi.get("id", ""),
            "wandb_best_val_loss": wbi.get("best_val_loss", ""),
        })

    # ── CSV completo ───────────────────────────────────────────────────────────
    fields = list(rows[0].keys()) if rows else []
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ── CSV só com divergências W&B × local ──────────────────────────────────────
    div_rows = [dict(r, divergence_reason=f"wandb={r['wandb_state']} but local={r['local_status']}")
                for r in rows if r["category"] == "divergent"]
    os.makedirs(os.path.dirname(os.path.abspath(args.divergences_csv)), exist_ok=True)
    with open(args.divergences_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["divergence_reason"])
        w.writeheader()
        w.writerows(div_rows)

    # ── Tabela no terminal ─────────────────────────────────────────────────────
    total_b = sum(r["dir_size_bytes"] for r in rows)
    print()
    print("=" * 120)
    print(f"{'EXPERIMENTO':<58} {'TAMANHO':>9}  {'BEST':>9}  {'LAST':>9}  {'WANDB':<10} {'STATUS':<8} {'CATEGORIA':<10}")
    print("=" * 120)
    for r in rows:
        print(
            f"{r['run_name']:<58} "
            f"{r['dir_size_human']:>9}  "
            f"{(r['best_ckpt_human'] or '—'):>9}  "
            f"{(r['last_ckpt_human'] or '—'):>9}  "
            f"{r['wandb_state']:<10} "
            f"{r['local_status']:<8} "
            f"{r['category']:<10}"
        )
    print("=" * 120)
    n_best = sum(1 for r in rows if r["best_ckpt_path"])
    n_last = sum(1 for r in rows if r["last_ckpt_path"])
    print(f"  Experimentos      : {len(rows)}")
    print(f"  Espaço total      : {_human(total_b)}  ({total_b:,} bytes)")
    print(f"  Com best ckpt     : {n_best}")
    print(f"  Com last ckpt     : {n_last}")
    cats = {c: sum(1 for r in rows if r["category"] == c) for c in ("done", "divergent", "failed", "missing")}
    print(
        f"  Por categoria     : done={cats['done']} divergent={cats['divergent']} "
        f"failed={cats['failed']} missing={cats['missing']}"
    )
    print(f"  CSV completo      : {os.path.relpath(args.csv, _ROOT)}")
    print(f"  CSV divergências  : {os.path.relpath(args.divergences_csv, _ROOT)}  ({len(div_rows)} linhas)")
    print("=" * 120)


if __name__ == "__main__":
    main()
