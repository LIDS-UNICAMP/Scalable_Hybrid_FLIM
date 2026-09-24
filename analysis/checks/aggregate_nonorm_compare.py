#!/usr/bin/env python3
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
"""Aggregate OLD (buggy double-norm) vs NEW (--no-imagenet-norm) FLIM-init 126k distillation.

Two rulers:
  - encoder48: SVM on the 48-dim FLIM encoder (headline; isolates the encoder, where the bug lives)
  - proj1280 : SVM on the 1280-dim projection head output

For each ruler builds mean-over-splits kappa per dataset x pct, with Δκ = NEW - OLD.
Robust to missing CSVs (prints what it has)."""
import csv, os, collections, statistics

R = "results"
CSVS = {
    ("encoder48", "OLD"): "svm_old_1x1_encoder48_results.csv",
    ("encoder48", "NEW"): "svm_nonorm_1x1_encoder48_results.csv",
    ("proj1280",  "OLD"): "svm_proj1280_1x1_BN2d_results.csv",
    ("proj1280",  "NEW"): "svm_nonorm_1x1_proj1280_results.csv",
}
# OLD proj1280 CSV mixes trunc + flim rows -> keep only flim_init runs (one-layer 126k)
def keep_row(ruler, ver, row):
    rn = row["run_name"]
    if "no_imagenet_norm" in rn:           # NEW rows
        return ver == "NEW"
    # OLD: must be flim_init one-layer 126k, not the no-norm ones
    if ver == "OLD":
        return rn.endswith("_flim_init") or rn.endswith("_flim_init_v1") or "_flim_init" in rn and "no_imagenet" not in rn
    return False

def load(path):
    if not os.path.exists(os.path.join(R, path)):
        return None
    out = []
    with open(os.path.join(R, path)) as f:
        for row in csv.DictReader(f):
            out.append(row)
    return out

DATASETS = ["eggs", "larvae", "protozoan"]
PCTS = [1, 5, 25, 50, 75, 100]

def build(ruler):
    # (dataset,pct,ver) -> list of kappa
    acc = collections.defaultdict(list)
    have = {}
    for ver in ("OLD", "NEW"):
        rows = load(CSVS[(ruler, ver)])
        have[ver] = rows is not None
        if rows is None:
            continue
        for row in rows:
            if row.get("status") != "ok":
                continue
            if not keep_row(ruler, ver, row):
                continue
            try:
                ds = row["dataset"]; pct = int(row["percentage"]); k = float(row["kappa"])
            except (KeyError, ValueError):
                continue
            acc[(ds, pct, ver)].append(k)
    return acc, have

def fmt(x):
    return f"{x:5.3f}" if x is not None else "  -  "

# O corpo de topo virou funcao: como modulo de pacote, ele rodava a analise inteira
# no `import analysis.checks.aggregate_nonorm_compare`. So le CSV e imprime, nao escreve.
def aggregate_nonorm_compare():
    print("=" * 78)
    for ruler in ("encoder48", "proj1280"):
        acc, have = build(ruler)
        print(f"\n### RULER: {ruler}   (OLD csv present={have.get('OLD')}, NEW csv present={have.get('NEW')})")
        print(f"{'dataset':9} {'pct':>4} | {'OLD κ':>6} {'NEW κ':>6} {'Δκ':>7} | nO nN")
        print("-" * 60)
        dataset_deltas = collections.defaultdict(list)
        for ds in DATASETS:
            for pct in PCTS:
                o = acc.get((ds, pct, "OLD")); n = acc.get((ds, pct, "NEW"))
                om = statistics.mean(o) if o else None
                nm = statistics.mean(n) if n else None
                d = (nm - om) if (om is not None and nm is not None) else None
                if d is not None:
                    dataset_deltas[ds].append(d)
                print(f"{ds:9} {pct:>4} | {fmt(om):>6} {fmt(nm):>6} "
                      f"{('%+0.3f'%d) if d is not None else '   -   ':>7} | "
                      f"{len(o) if o else 0:>2} {len(n) if n else 0:>2}")
        print("-" * 60)
        for ds in DATASETS:
            dd = dataset_deltas[ds]
            if dd:
                print(f"  mean Δκ {ds:9}: {statistics.mean(dd):+0.3f}  (over {len(dd)} pcts, NEW-OLD)")
    print("=" * 78)


if __name__ == "__main__":
    aggregate_nonorm_compare()
