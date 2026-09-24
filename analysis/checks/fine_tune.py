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

"""fine_tune.py — Validate which MLP fine-tune weight files exist.

For each expected experiment that has a completed SSL checkpoint the
script checks whether results/mlp_weights/{mode}/{run_id}/model_best.pth
exists for both freeze and unfreeze modes.

SVM is excluded: it does not produce saved weight files.

Usage:
    python -m analysis.checks.fine_tune
    # os antigos flags sao parametros de fine_tune():
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
    has_mlp_weight,
    has_ssl_checkpoint,
    load_ids_wandb,
    update_wandb_cache,
)

_MODES = ["freeze", "unfreeze"]


def fine_tune(update_wandb: bool = False, json: bool = False,
              fail_on_missing: bool = False) -> int:
    """Check which MLP fine-tune weight files exist."""
    # O corpo abaixo continua lendo `args.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    if args.update_wandb:
        print("[W&B] Refreshing metadata cache...")
        update_wandb_cache()

    name_to_id  = load_ids_wandb()
    experiments = all_expected()

    # Per mode: lists of (Experiment, run_id) or just Experiment
    found:          dict[str, list[tuple[Experiment, str]]] = {m: [] for m in _MODES}
    missing_weight: dict[str, list[tuple[Experiment, str]]] = {m: [] for m in _MODES}
    no_ssl:         dict[str, list[Experiment]]             = {m: [] for m in _MODES}

    for exp in experiments:
        run_id    = name_to_id.get(exp.canonical_name)
        ssl_ready = run_id is not None and has_ssl_checkpoint(run_id)
        for mode in _MODES:
            if not ssl_ready:
                no_ssl[mode].append(exp)
            elif has_mlp_weight(run_id, mode):
                found[mode].append((exp, run_id))
            else:
                missing_weight[mode].append((exp, run_id))

    total    = len(experiments)
    has_miss = False

    sep = "=" * 70
    print(f"\n{sep}")
    print("  MLP FINE-TUNE WEIGHTS INTEGRITY REPORT")
    print(sep)

    for mode in _MODES:
        n_no_ssl   = len(no_ssl[mode])
        n_eligible = total - n_no_ssl
        n_found    = len(found[mode])
        n_missing  = len(missing_weight[mode])

        print(f"\n  Mode: {mode.upper()}")
        print(f"    SSL available (eligible) : {n_eligible}")
        print(f"    Weights found            : {n_found}")
        print(f"    Weights missing          : {n_missing}")
        print(f"    Skipped (no SSL ckpt)    : {n_no_ssl}")

        if missing_weight[mode]:
            has_miss = True
            by_ds: dict[str, list[tuple[Experiment, str]]] = defaultdict(list)
            for exp, rid in missing_weight[mode]:
                by_ds[exp.dataset].append((exp, rid))
            print(f"\n  [MISSING {mode} weights] ({n_missing})")
            for dataset in sorted(by_ds):
                print(f"\n    Dataset: {dataset}")
                for exp, rid in sorted(by_ds[dataset], key=lambda x: (x[0].split, x[0].pct, x[0].init)):
                    print(
                        f"      split={exp.split}  pct={exp.pct:>3}  init={exp.init:<8}"
                        f"  {exp.canonical_name}  (run_id={rid})"
                    )

    print(f"\n{sep}\n")

    if not has_miss:
        print("  All MLP fine-tune weights present.\n")

    if args.json:
        report = {
            mode: {
                "found": len(found[mode]),
                "missing": [
                    {"canonical_name": e.canonical_name, "run_id": rid}
                    for e, rid in missing_weight[mode]
                ],
                "no_ssl": [e.canonical_name for e in no_ssl[mode]],
            }
            for mode in _MODES
        }
        print(_json.dumps(report, indent=2))

    return 1 if (args.fail_on_missing and has_miss) else 0


if __name__ == "__main__":
    sys.exit(fine_tune())
