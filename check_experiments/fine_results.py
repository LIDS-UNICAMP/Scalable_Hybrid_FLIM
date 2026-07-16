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

"""fine_results.py — Validate which evaluation result entries exist.

For each expected experiment that has a completed SSL checkpoint the
script checks whether a successful (status=ok) result row exists in:
  - artifacts/MLP/mlp_results.csv  (for MLP freeze and MLP unfreeze)
  - artifacts/SVM/svm_results.csv  (for SVM)

Usage:
    python -m check_experiments.fine_results
    python -m check_experiments.fine_results --update_wandb
    python -m check_experiments.fine_results --json
    python -m check_experiments.fine_results --fail-on-missing
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from check_experiments._common import (
    Experiment,
    all_expected,
    has_ssl_checkpoint,
    load_ids_wandb,
    load_mlp_ok_set,
    load_svm_ok_set,
    make_arg_parser,
    update_wandb_cache,
)

_CATEGORIES = ["freeze", "unfreeze", "svm"]
_LABELS     = {"freeze": "MLP FREEZE", "unfreeze": "MLP UNFREEZE", "svm": "SVM"}


def main() -> int:
    parser = make_arg_parser("Check which MLP/SVM evaluation results exist.")
    args = parser.parse_args()

    if args.update_wandb:
        print("[W&B] Refreshing metadata cache...")
        update_wandb_cache()

    name_to_id  = load_ids_wandb()
    experiments = all_expected()
    mlp_ok      = load_mlp_ok_set()   # {(ssl_run_id, freeze_status)}
    svm_ok      = load_svm_ok_set()   # {ssl_run_id}

    found:   dict[str, list[tuple[Experiment, str]]] = {c: [] for c in _CATEGORIES}
    missing: dict[str, list[tuple[Experiment, str]]] = {c: [] for c in _CATEGORIES}
    no_ssl:  dict[str, list[Experiment]]             = {c: [] for c in _CATEGORIES}

    for exp in experiments:
        run_id    = name_to_id.get(exp.canonical_name)
        ssl_ready = run_id is not None and has_ssl_checkpoint(run_id)

        if not ssl_ready:
            for c in _CATEGORIES:
                no_ssl[c].append(exp)
            continue

        # MLP freeze / unfreeze
        for mode in ("freeze", "unfreeze"):
            if (run_id, mode) in mlp_ok:
                found[mode].append((exp, run_id))
            else:
                missing[mode].append((exp, run_id))

        # SVM
        if run_id in svm_ok:
            found["svm"].append((exp, run_id))
        else:
            missing["svm"].append((exp, run_id))

    total    = len(experiments)
    has_miss = False

    sep = "=" * 70
    print(f"\n{sep}")
    print("  EVALUATION RESULTS INTEGRITY REPORT")
    print(sep)

    for cat in _CATEGORIES:
        n_no_ssl   = len(no_ssl[cat])
        n_eligible = total - n_no_ssl
        n_found    = len(found[cat])
        n_missing  = len(missing[cat])
        label      = _LABELS[cat]

        print(f"\n  Mode: {label}")
        print(f"    SSL available (eligible) : {n_eligible}")
        print(f"    Results found            : {n_found}")
        print(f"    Results missing          : {n_missing}")
        print(f"    Skipped (no SSL ckpt)    : {n_no_ssl}")

        if missing[cat]:
            has_miss = True
            by_ds: dict[str, list[tuple[Experiment, str]]] = defaultdict(list)
            for exp, rid in missing[cat]:
                by_ds[exp.dataset].append((exp, rid))
            print(f"\n  [MISSING {label} results] ({n_missing})")
            for dataset in sorted(by_ds):
                print(f"\n    Dataset: {dataset}")
                for exp, rid in sorted(by_ds[dataset], key=lambda x: (x[0].split, x[0].pct, x[0].init)):
                    print(
                        f"      split={exp.split}  pct={exp.pct:>3}  init={exp.init:<8}"
                        f"  {exp.canonical_name}  (run_id={rid})"
                    )

    print(f"\n{sep}\n")

    if not has_miss:
        print("  All evaluation results present.\n")

    if args.json:
        report = {
            cat: {
                "found": len(found[cat]),
                "missing": [
                    {"canonical_name": e.canonical_name, "run_id": rid}
                    for e, rid in missing[cat]
                ],
                "no_ssl": [e.canonical_name for e in no_ssl[cat]],
            }
            for cat in _CATEGORIES
        }
        print(json.dumps(report, indent=2))

    return 1 if (args.fail_on_missing and has_miss) else 0


if __name__ == "__main__":
    sys.exit(main())
