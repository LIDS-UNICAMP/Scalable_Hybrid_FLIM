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

"""organize_raw_dataset.py — Le o YAML, padroniza cada dataset e gera os splits.

    python tools/organize_raw_dataset.py -c configs/organize_datasets.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from dataset_std.base import REGISTRY
from dataset_std import aid, coconut_trees, fer2013, flowers, food101  # noqa: F401
from dataset_std import medmnist, parasites_ref, remote_sensing        # noqa: F401
from make_splits import make_splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--only", nargs="*", help="roda so estes datasets (por name)")
    ap.add_argument("--force", action="store_true", help="reprocessa mesmo ja organizado")
    args = ap.parse_args()

    doc = yaml.safe_load(Path(args.config).read_text())
    defaults = doc.get("defaults", {})
    split_cfg = doc["split"]
    out_root = Path(doc["out_root"])

    for entry in doc["datasets"]:
        if not entry.get("enabled") or (args.only and entry["name"] not in args.only):
            continue
        cfg = {**defaults, **entry, "out_root": str(out_root), "force": args.force}
        cfg["out"] = str(out_root / cfg["name"])
        std = REGISTRY[cfg.get("standardizer", cfg["name"])](cfg)
        out = std.run()
        make_splits(out, split_cfg, cfg.get("seed", 42))
        summarize(out)


def summarize(out: Path):
    """D2: contagem por classe e por split, em cada percentual."""
    rep = json.loads((out / "split_report.json").read_text())
    print(f"\n=== {rep['dataset']}  ({rep['n_items']} imagens, {rep['n_groups']} grupos) ===")
    for sname, sp in rep["splits"].items():
        ng = sp["n_groups"]
        print(f"  {sname}: train {ng['train']} | val {ng['validation']} | test {ng['test']}")
        classes = sorted(sp["counts"]["train"])
        head = "    {:<6}".format("perc") + "".join(f"{c[:14]:>16}" for c in classes)
        print(head)
        for pc, blk in sp["percentages"].items():
            row = "    {:<6}".format(pc + "%")
            row += "".join(f"{blk['counts'].get(c, 0):>16}" for c in classes)
            print(row)
        for sub in ("validation", "test"):
            row = "    {:<6}".format(sub[:5])
            row += "".join(f"{sp['counts'][sub].get(c, 0):>16}" for c in classes)
            print(row)
        for w in sp["warnings"]:
            print(f"    [WARN] {w}")


if __name__ == "__main__":
    main()
