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
"""Amostras de treino e de avaliacao de cada dataset, lidas do dataset_<nome>.yaml (§1, §4)."""
import json
import random
from functools import lru_cache
from pathlib import Path

import yaml
from spifil.data import Sample, parse_class_label

from ga_search.constants import CONFIG_DIR


def load_dataset_cfg(name: str, config_dir=CONFIG_DIR) -> dict:
    """Le config/dataset_<name>.yaml e troca {root} nos caminhos."""
    cfg = yaml.safe_load((Path(config_dir) / f"dataset_{name}.yaml").read_text(encoding="utf-8"))
    for key in ("images_dir", "masks_dir", "split_json"):
        if cfg.get(key):
            cfg[key] = cfg[key].replace("{root}", cfg["root"])
    return cfg


@lru_cache(maxsize=None)
def _split(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _samples(ds_cfg, pct, split, key) -> list[Sample]:
    names = sorted(_split(ds_cfg["split_json"].format(split=split, pct=pct))[key])
    images, masks = Path(ds_cfg["images_dir"]), ds_cfg["has_mask"] and Path(ds_cfg["masks_dir"])
    return [Sample(name=Path(n).stem, image_path=images / n, mask_path=masks / n if masks else None,
                   label=parse_class_label(n)) for n in names]


def n_classes_eff(ds_cfg, paper_classes: bool) -> int:
    return len(ds_cfg["paper_classes"]) if paper_classes else ds_cfg["n_classes"]


def _train(ds_cfg, pct, split, paper_classes) -> list[Sample]:
    chosen = _samples(ds_cfg, pct, split, ds_cfg["train_key"])
    if paper_classes:
        keep = {i for i, c in ds_cfg["classes"].items() if c in ds_cfg["paper_classes"]}
        chosen = [s for s in chosen if s.label in keep]
    return chosen


def n_available(ds_cfg, pct, split, paper_classes: bool = False) -> int:
    """Imagens de treino da pct, ja filtradas como em train_samples."""
    return len(_train(ds_cfg, pct, split, paper_classes))


def train_samples(ds_cfg, pct, split, *, one_per_class: bool, n_images: int | None,
                  paper_classes: bool, seed: int) -> list[Sample]:
    """Uma imagem sorteada por classe, ou passo uniforme como o flim/spifil.py."""
    chosen = _train(ds_cfg, pct, split, paper_classes)
    if not one_per_class:
        return chosen[:: max(1, len(chosen) // n_images)][:n_images]
    rng = random.Random(seed)
    return [rng.choice([s for s in chosen if s.label == c]) for c in sorted({s.label for s in chosen})]


def eval_samples(ds_cfg, pct, split) -> list[Sample]:
    """Conjunto de avaliacao: a lista `eval.source` (validation) do JSON da pct. Nunca o teste."""
    # sem filtro de paper_classes de proposito: conjunto fixo do split, comparavel entre individuos
    return _samples(ds_cfg, pct, split, ds_cfg["eval"]["source"])
