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

"""Fair, leakage-free train/val/test splits + nested incremental training subsets.

Reads ``<dataset_dir>/manifest.json`` (written by ``tools/dataset_std``) and writes, next
to it, the directory shape the FLIM pipeline already expects::

    splits/split<i>.json                                    {"train","validation","test"}
    splits_incremental/split<i>/data_descriptor_perc<p>.json
    splits_incremental/split<i>_distribution.png
    split_report.json

Every JSON holds flat, sorted lists of bare filenames.

Fairness notes:
  * No preprocessing leakage: this module computes NO statistics of any kind over the
    images (no mean/std/normalisation). Splits are index-only.
  * The original vendor split is deliberately discarded and re-split; its old membership
    survives only as the ``orig_split`` covariate and is reported for visibility.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are produced, never shown

import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

DEFAULT_CFG = {
    "strategy": "stratified_group",
    "group_by": "group",
    "stratify_by": "label",
    "n_splits": 3,
    "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
    "percentages": [1, 5, 10, 25, 50, 75, 100],
}
SUBSETS = ("train", "validation", "test")


def _key(item, name, default_field):
    """Config may name a covariate (e.g. group_by: patient_id); fall back to the manifest field."""
    return str((item.get("covariates") or {}).get(name, item[default_field]))


def _build_groups(items, cfg):
    """-> [(group_key, members, class_label)]. Ungrouped strategies make each item its own group."""
    grouped = cfg["strategy"] in ("group", "stratified_group")
    buckets = defaultdict(list)
    for it in items:
        buckets[_key(it, cfg["group_by"], "group") if grouped else it["file"]].append(it)
    groups = []
    for gkey in sorted(buckets):
        members = buckets[gkey]
        cls = Counter(_key(m, cfg["stratify_by"], "label") for m in members).most_common(1)[0][0]
        groups.append((gkey, members, cls))
    return groups


def _assign(groups, cfg, rng, warnings):
    """Assign WHOLE groups to train/validation/test, stratified by the group-level class."""
    stratified = cfg["strategy"] in ("stratified", "stratified_group")
    ratios = cfg["ratios"]
    by_class = defaultdict(list)
    for g in groups:
        by_class[g[2] if stratified else "__all__"].append(g)

    assignment = {s: [] for s in SUBSETS}
    for cls in sorted(by_class):
        pool = list(by_class[cls])
        rng.shuffle(pool)
        n = len(pool)
        # --- rare-class guard: never silently produce an empty cell -------------------
        if n == 1:
            warnings.append(f"class '{cls}': only 1 group -> train only, NO test sample")
            assignment["train"] += pool
            continue
        if n == 2:
            warnings.append(f"class '{cls}': only 2 groups -> 1 test / 1 train, empty validation")
            assignment["test"].append(pool[0])
            assignment["train"].append(pool[1])
            continue
        n_test = max(1, round(n * ratios["test"]))
        n_val = max(1, round(n * ratios["val"]))
        while n - n_test - n_val < 1:  # train must keep at least one group too
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        if n < 10:
            warnings.append(
                f"class '{cls}': only {n} groups -> test has {n_test}; metric will be unstable"
            )
        assignment["test"] += pool[:n_test]
        assignment["validation"] += pool[n_test:n_test + n_val]
        assignment["train"] += pool[n_test + n_val:]
    return assignment


def _counts(assignment):
    per_sub = {s: Counter() for s in SUBSETS}
    for sub, groups in assignment.items():
        for _, members, cls in groups:
            per_sub[sub][cls] += len(members)
    return per_sub


def _check_drift(per_sub, cfg, tol=0.15):
    """Realised class proportions must stay near the configured ratios; fail loudly if not."""
    want = {"train": cfg["ratios"]["train"], "validation": cfg["ratios"]["val"], "test": cfg["ratios"]["test"]}
    totals = Counter()
    for sub in SUBSETS:
        totals.update(per_sub[sub])
    for cls, total in totals.items():
        if total < 20:  # too thin for a proportion to carry information
            continue
        for sub, target in want.items():
            frac = per_sub[sub][cls] / total
            if abs(frac - target) > tol:
                raise AssertionError(
                    f"class '{cls}' drifted in {sub}: {frac:.3f} realised vs {target:.3f} configured"
                )


def _covariates(assignment):
    rep = defaultdict(lambda: defaultdict(Counter))
    for sub, groups in assignment.items():
        for _, members, _ in groups:
            for m in members:
                for k, v in (m.get("covariates") or {}).items():
                    rep[k][sub][str(v)] += 1
    return {k: {s: dict(c) for s, c in subs.items()} for k, subs in rep.items()}


def _incremental(train_groups, percentages, rng):
    """Nested per-class prefixes: shuffle each class's train files ONCE, take growing prefixes."""
    by_class = defaultdict(list)
    for _, members, cls in train_groups:
        by_class[cls] += [m["file"] for m in members]
    for cls in by_class:
        by_class[cls].sort()
        rng.shuffle(by_class[cls])
    out = {}
    for p in percentages:
        files = []
        for pool in by_class.values():
            k = min(len(pool), max(1, math.ceil(len(pool) * p / 100)))  # ceil -> class never vanishes
            files += pool[:k]
        out[p] = sorted(files)
    return out


def _files(groups):
    return sorted(m["file"] for _, members, _ in groups for m in members)


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def _plot(path, percentages, per_perc, title):
    classes = sorted({c for counts in per_perc.values() for c in counts})
    x = list(range(len(percentages)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        ax.plot(x, [per_perc[p].get(cls, 0) for p in percentages], marker="o", linewidth=1, label=cls)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in percentages])
    ax.set_xlabel("training percentage")
    ax.set_ylabel("images in train")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(classes) <= 15:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def verify_splits(dataset_dir, cfg):
    """Re-load what was written and assert every leakage/nesting invariant. Raises on failure."""
    dataset_dir = Path(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    # group integrity only means something when the strategy is group-aware
    group_of = ({it["file"]: _key(it, cfg["group_by"], "group") for it in manifest["items"]}
                if cfg["strategy"] in ("group", "stratified_group") else {})
    percentages = list(cfg["percentages"])

    for i in range(1, cfg["n_splits"] + 1):
        split = json.loads((dataset_dir / "splits" / f"split{i}.json").read_text())
        sets = {s: set(split[s]) for s in SUBSETS}
        for a in SUBSETS:
            for b in SUBSETS:
                if a < b:
                    assert not sets[a] & sets[b], f"split{i}: files shared by {a} and {b}"
        seen = {}
        for sub in SUBSETS:
            for f in split[sub]:
                g = group_of.get(f)
                assert g is None or seen.setdefault(g, sub) == sub, \
                    f"split{i}: group '{g}' crosses subsets"

        prev = None
        for p in percentages:
            inc = json.loads(
                (dataset_dir / "splits_incremental" / f"split{i}" / f"data_descriptor_perc{p}.json").read_text()
            )
            assert inc["validation"] == split["validation"], f"split{i} perc{p}: validation changed"
            assert inc["test"] == split["test"], f"split{i} perc{p}: test changed"
            tr = set(inc["train"])
            assert not tr & sets["test"], f"split{i} perc{p}: test leaked into incremental train"
            assert not tr & sets["validation"], f"split{i} perc{p}: validation leaked into train"
            assert tr <= sets["train"], f"split{i} perc{p}: train not a subset of the full train"
            if prev is not None:
                assert prev <= tr, f"split{i} perc{p}: nesting broken"
            prev = tr
        assert prev == sets["train"], f"split{i}: perc100 train != split train"


def make_splits(dataset_dir, split_cfg=None, seed=42):
    """Write splits/, splits_incremental/ and split_report.json under dataset_dir. Returns the report."""
    dataset_dir = Path(dataset_dir)
    cfg = {**DEFAULT_CFG, **(split_cfg or {})}
    if cfg["strategy"] not in ("random", "stratified", "group", "stratified_group"):
        raise ValueError(f"unknown split strategy: {cfg['strategy']}")
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    percentages = list(cfg["percentages"])
    groups = _build_groups(manifest["items"], cfg)

    report = {
        "dataset": manifest.get("dataset", dataset_dir.name),
        "seed": seed,
        "strategy": cfg["strategy"],
        "ratios": cfg["ratios"],
        "n_items": len(manifest["items"]),
        "n_groups": len(groups),
        "splits": {},
    }

    bar = tqdm(total=cfg["n_splits"] * len(percentages), desc=f"splits[{report['dataset']}]")
    for i in range(1, cfg["n_splits"] + 1):
        rng = random.Random(seed + i)  # independent but reproducible per split
        warnings = []
        assignment = _assign(groups, cfg, rng, warnings)
        per_sub = _counts(assignment)
        if cfg["strategy"] in ("stratified", "stratified_group"):
            _check_drift(per_sub, cfg)
        for w in warnings:
            print(f"[WARN] split{i}: {w}")

        split_json = {s: _files(assignment[s]) for s in SUBSETS}
        _write_json(dataset_dir / "splits" / f"split{i}.json", split_json)

        inc = _incremental(assignment["train"], percentages, rng)
        cls_of = {m["file"]: cls for _, members, cls in assignment["train"] for m in members}
        per_perc = {}
        for p in percentages:
            _write_json(
                dataset_dir / "splits_incremental" / f"split{i}" / f"data_descriptor_perc{p}.json",
                {"train": inc[p], "validation": split_json["validation"], "test": split_json["test"]},
            )
            per_perc[p] = dict(Counter(cls_of[f] for f in inc[p]))
            bar.update(1)

        _plot(
            dataset_dir / "splits_incremental" / f"split{i}_distribution.png",
            percentages,
            per_perc,
            f"{report['dataset']} — split{i}: train class distribution",
        )
        report["splits"][f"split{i}"] = {
            "n_groups": {s: len(assignment[s]) for s in SUBSETS},
            "counts": {s: dict(per_sub[s]) for s in SUBSETS},
            "covariates": _covariates(assignment),
            "percentages": {str(p): {"n_train": len(inc[p]), "counts": per_perc[p]} for p in percentages},
            "warnings": warnings,
        }
    bar.close()

    _write_json(dataset_dir / "split_report.json", report)
    verify_splits(dataset_dir, cfg)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", required=True, help="organized/<name> containing manifest.json")
    ap.add_argument("--config", help="YAML holding a 'split:' block")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split_cfg = None
    if args.config:
        split_cfg = (yaml.safe_load(Path(args.config).read_text()) or {}).get("split")
    rep = make_splits(args.dataset_dir, split_cfg, args.seed)
    print(f"[OK] {rep['dataset']}: {rep['n_items']} items, {rep['n_groups']} groups, "
          f"{len(rep['splits'])} splits -> {args.dataset_dir}")


if __name__ == "__main__":
    main()
