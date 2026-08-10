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
"""report_distill_divergences.py — Filtra divergências entre W&B e status local.

Lê o relatório gerado por ``scripts/report_distill_disk_wandb.py`` e escreve um
segundo CSV contendo APENAS os experimentos divergentes — aqueles em que o estado
do W&B e o status local discordam.

Regra de divergência (idêntica à do relatório principal):
    divergente  ⇔  (wandb_state == "finished")  XOR  (local_status == "ok")

Ou seja: o W&B diz "finished" mas o status local não é "ok", OU o status local é
"ok" mas o W&B não está "finished" (failed / crashed / not_found).

Exceção: linhas com ``wandb_state == "skipped"`` (modo --no-wandb, W&B não
consultado) NÃO são divergentes — não há o que comparar.

Saída:
  * CSV em results/distill_divergences.csv (mesmas colunas + "divergence_reason")
  * resumo no terminal: total de divergências + agrupamento por motivo

Uso:
    python scripts/report_distill_divergences.py
    python scripts/report_distill_divergences.py --in outro.csv --out saida.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

from constants import PROJECT_ROOT as _ROOT, RESULTS_DIR as _RESULTS

sys.path.insert(0, _ROOT)


def _is_divergent(wandb_state: str, local_status: str) -> bool:
    """(wandb=='finished') XOR (local=='ok'); 'skipped' nunca é divergente."""
    if wandb_state == "skipped":
        return False
    return (wandb_state == "finished") != (local_status == "ok")


def _reason(wandb_state: str, local_status: str) -> str:
    """Descreve o mismatch, ex.: 'wandb=failed but local=ok'."""
    return f"wandb={wandb_state or 'missing'} but local={local_status or 'missing'}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_csv",
                    default=os.path.join(_RESULTS, "distill_disk_wandb_report.csv"))
    ap.add_argument("--out", dest="out_csv",
                    default=os.path.join(_RESULTS, "distill_divergences.csv"))
    args = ap.parse_args()

    if not os.path.isfile(args.in_csv):
        print(f"[ERRO] CSV de entrada não encontrado: {args.in_csv}")
        print("       Rode primeiro: python scripts/report_distill_disk_wandb.py")
        sys.exit(1)

    with open(args.in_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    divergent: list[dict] = []
    for r in rows:
        wstate = r.get("wandb_state", "")
        lstatus = r.get("local_status", "")
        if _is_divergent(wstate, lstatus):
            r = dict(r)
            r["divergence_reason"] = _reason(wstate, lstatus)
            divergent.append(r)

    out_fields = fields + ["divergence_reason"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        w.writerows(divergent)

    # ── Resumo no terminal ──────────────────────────────────────────────────────
    breakdown = Counter(r["divergence_reason"] for r in divergent)
    print()
    print("=" * 70)
    print(f"  Divergências encontradas : {len(divergent)} de {len(rows)} experimentos")
    print("-" * 70)
    for reason, n in sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>4}  {reason}")
    print("-" * 70)
    print(f"  CSV : {os.path.relpath(args.out_csv, _ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
