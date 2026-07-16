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

"""
Test script for DatasetParasite and DataModuleParasite.

Run from the project root:
    python scripts/test_dataset.py

What this script tests
----------------------
1. config helpers  – get_dataset_paths / get_split_path_incremental
2. DatasetParasite – all three splits (training / validation / test)
3. DataModuleParasite – get_dataloaders(), iterating one batch from each split
4. ParasiteDataModule (Lightning) – setup() + all three dataloaders
"""

import sys
import os

# Make sure the project root is on sys.path so `config` and `src` are importable.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torchvision import transforms

from config import get_dataset_paths, get_split_path_incremental
from src.data_modules.datasets.dataset import DatasetParasite, DataModuleParasite
from src.data_modules.parasite import ParasiteDataModule

# ---------------------------------------------------------------------------
# Settings – change these to match the dataset you want to test.
# ---------------------------------------------------------------------------
DATASET_NAME = "to_modules"
SPLIT = 0
PERCENTAGE = 100
BATCH_SIZE = 4

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def test_config_helpers():
    section("1. config helpers")
    paths = get_dataset_paths(DATASET_NAME)
    print(f"  images dir  : {paths['images']}")
    print(f"  splits dir  : {paths['splits']}")

    split_json = get_split_path_incremental(DATASET_NAME, SPLIT, PERCENTAGE)
    print(f"  split JSON  : {split_json}")
    assert os.path.isfile(split_json), f"Split JSON not found: {split_json}"
    print("  [OK] split JSON exists")


def test_dataset_parasite():
    section("2. DatasetParasite")
    for set_name in ("training", "validation", "test"):
        ds = DatasetParasite(
            dataset_name=DATASET_NAME,
            set=set_name,
            split=SPLIT,
            percentage=PERCENTAGE,
            transform=transform,
            dataset_name=DATASET_NAME,
        )
        img, label = ds[0]
        print(f"  [{set_name:10s}] len={len(ds):3d}  "
              f"img shape={tuple(img.shape)}  label={label}  dtype={img.dtype}")
    print("  [OK] DatasetParasite works for all splits")


def test_datamodule_parasite():
    section("3. DataModuleParasite (plain wrapper)")
    dm = DataModuleParasite(
        dataset_name=DATASET_NAME,
        split=SPLIT,
        percentage=PERCENTAGE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform=transform,
        loader="pil",
    )
    loaders = dm.get_dataloaders()
    for split_name, loader in loaders.items():
        imgs, labels = next(iter(loader))
        print(f"  [{split_name:10s}] batch imgs={tuple(imgs.shape)}  "
              f"labels={labels.tolist()}")
    print("  [OK] DataModuleParasite iterates without error")


def test_parasite_lightning():
    section("4. ParasiteDataModule (pl.LightningDataModule)")
    dm = ParasiteDataModule(
        dataset_name=DATASET_NAME,
        split=SPLIT,
        percentage=PERCENTAGE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        transform=transform,
        loader="pil",
        ssl_mode=False,
        pin_memory=False,
    )
    dm.setup()
    for name, loader in [
        ("train", dm.train_dataloader()),
        ("val",   dm.val_dataloader()),
        ("test",  dm.test_dataloader()),
    ]:
        imgs, labels = next(iter(loader))
        print(f"  [{name:5s}] batch imgs={tuple(imgs.shape)}  "
              f"labels={labels.tolist()}")
    print("  [OK] ParasiteDataModule (Lightning) works")


if __name__ == "__main__":
    test_config_helpers()
    test_dataset_parasite()
    test_datamodule_parasite()
    test_parasite_lightning()
    section("All tests passed!")
