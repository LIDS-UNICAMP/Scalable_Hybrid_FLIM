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

"""Dataset dos parasitos lidos a partir dos JSONs de split.

Movido de src/data_modules/datasets/dataset.py (classe DatasetParasite).
Leitura, rotulo e split identicos a origem.
"""

import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import get_parasito_split_paths, get_single_parasite_paths
from core.data.loaders import ift_lab_loader, pil_loader

logger = logging.getLogger(__name__)


class ParasiteDataset(Dataset):
    """
    Dataset que agrega as 3 classes do new_split_parasito
    (helminth-eggs_split_2, helminth-larvae_split_2, protozoan-cysts_split_2).

    Le os JSONs de split de cada classe e combina todos os samples.
    O label vem do prefixo do filename (1-indexed, convertido para 0-indexed).

    Args:
        set_name: One of ``"train"``, ``"validation"``, or ``"test"``.
        split: Fold index.
        percentage: Percentage of training data (None = full split).
        transform: Optional image transform.
        loader: ``"ift_lab"`` (default) or ``"pil"``.
        path_dataset: When provided, restricts loading to this single
            parasite name (e.g. ``"helminth-eggs_split_2"``).  Legacy multi-parasite
            behavior is preserved when *path_dataset* is ``None``.
    """
    def __init__(self, set_name, split, percentage, transform=None, loader="ift_lab",
                 path_dataset=None):
        self.set_name = set_name
        self.split = split
        self.percentage = percentage
        self.transform = transform
        self.loader = loader
        self.path_dataset = path_dataset

        if path_dataset is not None:
            # Single-parasite mode: load only the requested parasite.
            class_infos = get_single_parasite_paths(path_dataset, split, percentage)
            logger.info(
                "[ParasiteDataset] single-parasite mode | parasite=%s | split=%s | "
                "percentage=%s | set=%s",
                path_dataset, split, percentage, set_name,
            )
            for ci in class_infos:
                logger.info(
                    "  images_dir        : %s", ci["images_dir"]
                )
                logger.info(
                    "  split_json        : %s", ci["split_json"]
                )
        else:
            # Legacy multi-parasite mode: aggregate all registered parasites.
            class_infos = get_parasito_split_paths(split, percentage)

        self.samples = []
        for cls_info in class_infos:
            json_path = cls_info["split_json"]
            images_dir = cls_info["images_dir"]

            if not os.path.isfile(json_path):
                raise FileNotFoundError(f"Split JSON not found: {json_path}")

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            filenames = data[self.set_name]
            for fname in filenames:
                self.samples.append(os.path.join(images_dir, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img_name = os.path.basename(path)
        label = int(img_name.split("_")[0]) - 1

        if self.loader == "ift_lab":
            img = ift_lab_loader(path)
            # ift_lab_loader devolve numpy (H, W, 3) float em [0,1].
            # As transforms do torchvision esperam PIL Image ou Tensor (C, H, W),
            # entao a conversao acontece aqui, antes de qualquer transform.
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        else:
            img = pil_loader(path)

        if self.transform:
            img = self.transform(img)

        return img, label
