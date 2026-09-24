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

"""check_ssl.py — Validate which LeJEPA SSL experiments exist.

For each expected experiment (DATASETS × SPLITS × PCTS × INITS) the
script checks:
  1. Whether the run is registered in configs/wandb_update/ids_wandb.json.
  2. Whether a local checkpoint (.ckpt) exists under logs/flim-ssl/<run_id>/.

Missing experiments are listed explicitly and grouped by dataset.

Usage:
    python -m analysis.checks.check_ssl
    # os antigos flags sao parametros de check_ssl():
    #   update_wandb, json, fail_on_missing
"""
from __future__ import annotations

import json as _json
import sys
from collections import defaultdict
from types import SimpleNamespace

from analysis.checks._common import (
    Experiment,
    all_expected,
    has_ssl_checkpoint,
    load_ids_wandb,
    update_wandb_cache,
)


def check_ssl(update_wandb: bool = False, json: bool = False,
              fail_on_missing: bool = False) -> int:
    """Check which LeJEPA SSL experiments have local checkpoints."""
    # O corpo abaixo continua lendo `args.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    if args.update_wandb:
        print("[W&B] Refreshing metadata cache...")
        update_wandb_cache()

    name_to_id = load_ids_wandb()
    experiments = all_expected()

    found:              list[tuple[Experiment, str]] = []
    missing_registry:   list[Experiment]             = []  # not in ids_wandb.json
    missing_checkpoint: list[tuple[Experiment, str]] = []  # registered but no local .ckpt

    for exp in experiments:
        run_id = name_to_id.get(exp.canonical_name)
        if run_id is None:
            missing_registry.append(exp)
        elif not has_ssl_checkpoint(run_id):
            missing_checkpoint.append((exp, run_id))
        else:
            found.append((exp, run_id))

    total   = len(experiments)
    n_found = len(found)
    n_miss  = len(missing_registry) + len(missing_checkpoint)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  SSL CHECKPOINT INTEGRITY REPORT")
    print(sep)
    print(f"  Expected                 : {total}")
    print(f"  Found (registry + ckpt)  : {n_found}")
    print(f"  Missing total            : {n_miss}")
    print(f"    Not in W&B registry    : {len(missing_registry)}")
    print(f"    In registry, no ckpt   : {len(missing_checkpoint)}")
    print(sep)

    if missing_registry:
        by_ds: dict[str, list[Experiment]] = defaultdict(list)
        for exp in missing_registry:
            by_ds[exp.dataset].append(exp)
        print(f"\n[MISSING — not in W&B registry] ({len(missing_registry)})")
        for dataset in sorted(by_ds):
            print(f"\n  Dataset: {dataset}")
            for exp in sorted(by_ds[dataset], key=lambda e: (e.split, e.pct, e.init)):
                print(f"    split={exp.split}  pct={exp.pct:>3}  init={exp.init:<8}  {exp.canonical_name}")

    if missing_checkpoint:
        by_ds2: dict[str, list[tuple[Experiment, str]]] = defaultdict(list)
        for exp, rid in missing_checkpoint:
            by_ds2[exp.dataset].append((exp, rid))
        print(f"\n[MISSING — registered but no local checkpoint] ({len(missing_checkpoint)})")
        for dataset in sorted(by_ds2):
            print(f"\n  Dataset: {dataset}")
            for exp, rid in sorted(by_ds2[dataset], key=lambda x: (x[0].split, x[0].pct, x[0].init)):
                print(
                    f"    split={exp.split}  pct={exp.pct:>3}  init={exp.init:<8}"
                    f"  {exp.canonical_name}  (run_id={rid})"
                )

    if n_miss == 0:
        print("\n  All SSL experiments present.")

    print(f"\n{sep}\n")

    if args.json:
        report = {
            "total": total,
            "found": n_found,
            "missing_total": n_miss,
            "missing_registry": [e.canonical_name for e in missing_registry],
            "missing_checkpoint": [
                {"canonical_name": e.canonical_name, "run_id": rid}
                for e, rid in missing_checkpoint
            ],
        }
        print(_json.dumps(report, indent=2))

    return 1 if (args.fail_on_missing and n_miss > 0) else 0


if __name__ == "__main__":
    sys.exit(check_ssl())
