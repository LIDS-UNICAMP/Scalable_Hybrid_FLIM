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

import logging
import os
import json
from PIL import Image

import numpy as np
import torch
import pyift.pyift as ift

from config import get_parasito_split_paths, get_single_parasite_paths

from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


def pil_loader(image_path):
    with open(image_path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def ift_lab_loader(image_path):
    """Load image as LABNorm2 using pyift (same as FLIM training pipeline).
    Returns a PIL-compatible numpy array with 3 channels in [0, 1] range."""
    image = ift.ReadImageByExt(str(image_path))
    mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
    lab_image = mimage.AsNumPy().squeeze()  # (H, W, 3), float in [0, 1]
    return lab_image


class DataModuleParasite:
    """
    Gerencia DataLoaders de train/val/test usando DatasetParasite.

    Uso:
      dm = DataModuleParasite(split=1, percentage=100, batch_size=32)
      dm.setup()
      train_loader = dm.train_dataloader()
      val_loader   = dm.val_dataloader()
      test_loader  = dm.test_dataloader()
    """
    def __init__(self, split, percentage, batch_size=32,
                 num_workers=4, transform=None, loader="pil"):
        self.split = split
        self.percentage = percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.loader = loader

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        self.train_dataset = DatasetParasite("train", split=self.split, percentage=self.percentage, transform=self.transform, loader=self.loader)
        self.val_dataset = DatasetParasite("validation", split=self.split, percentage=self.percentage, transform=self.transform, loader=self.loader)
        self.test_dataset = DatasetParasite("test", split=self.split, percentage=self.percentage, transform=self.transform, loader=self.loader)

    def _ensure_datasets(self):
        if self.train_dataset is None or self.val_dataset is None or self.test_dataset is None:
            self.setup()

    def train_dataloader(self, batch_size=None, shuffle=True):
        self._ensure_datasets()
        bs = batch_size or self.batch_size
        return DataLoader(self.train_dataset, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers)

    def val_dataloader(self, batch_size=None, shuffle=False):
        self._ensure_datasets()
        bs = batch_size or self.batch_size
        return DataLoader(self.val_dataset, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self, batch_size=None, shuffle=False):
        self._ensure_datasets()
        bs = batch_size or self.batch_size
        return DataLoader(self.test_dataset, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def get_dataloaders(self):
        return {
            "train": self.train_dataloader(),
            "val": self.val_dataloader(),
            "test": self.test_dataloader(),
        }


class DatasetParasite(Dataset):
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
                "[DatasetParasite] single-parasite mode | parasite=%s | split=%s | "
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
            # ift_lab_loader returns numpy (H, W, 3) float in [0,1].
            # torchvision transforms expect PIL Image or Tensor (C, H, W),
            # so we convert here before applying any transform.
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() #new line
        else:
            img = pil_loader(path)

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    import sys
    from torchvision.transforms import v2

    ROOT = "/mnt/arquivos_linux/LIBRARY/PHD_planning/hawk/scalable_FLIM_self_supervised"

    SPLIT      = 1
    PERCENTAGE = None
    BATCH_SIZE = 4

    # v2 transforms aceita tanto PIL Image quanto torch.Tensor (C,H,W),
    # eliminando o conflito entre pil_loader e ift_lab_loader.
    transform = v2.Compose([
        v2.Resize((200, 200)),
        v2.ToDtype(torch.float32, scale=False),  # mantém valores em [0,1]
    ])

    print("\n--- DatasetParasite ---")
    for set_name in ("train", "validation", "test"):
        ds = DatasetParasite(
            set_name=set_name,
            split=SPLIT,
            percentage=PERCENTAGE,
            transform=transform,
            loader="ift_lab",
        )
        img, label = ds[0]
        print(f"  [{set_name:10s}]  len={len(ds)}  img={tuple(img.shape)}  label={label}")

    print("\n--- DataModuleParasite ---")
    dm = DataModuleParasite(
        split=SPLIT,
        percentage=PERCENTAGE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform=transform,
        loader="ift_lab",
    )
    for split_name, loader in dm.get_dataloaders().items():
        imgs, labels = next(iter(loader))
        print(f"  [{split_name:10s}]  batch={tuple(imgs.shape)}  labels={labels.tolist()}")

    print("\nAll OK!")
