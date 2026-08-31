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

"""FolderDataModule: datamodule de pasta generica (ex-LejepaDataModule)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

from core.data.folder_dataset import FolderDataset
from core.data.multi_crop_dataset import MultiCropDataset


class FolderDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for FLIM RGB image datasets.

    Supports two modes:
    - ``ssl_mode=False``: Returns ``(image_tensor, label)`` for supervised fine-tuning.
    - ``ssl_mode=True``: Returns ``(List[Tensor], label)`` multi-crop views for SSL.

    Args:
        data_dir: Root of the dataset (ImageFolder-style subdirs, or flat for SSL).
        batch_size: Batch size for all dataloaders.
        image_size: Spatial resolution for training crops.
        num_workers: DataLoader workers.
        val_split: Fraction of data used for validation.
        ssl_mode: If True, returns MultiCropDataset views.
        n_global: Global crop count (ssl_mode only).
        n_local: Local crop count (ssl_mode only).
        global_size: Global crop size (ssl_mode only).
        local_size: Local crop size (ssl_mode only).
        pin_memory: Whether to pin memory in DataLoaders.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        image_size: int = 224,
        num_workers: int = 8,
        val_split: float = 0.1,
        ssl_mode: bool = True,
        n_global: int = 2,
        n_local: int = 6,
        global_size: int = 224,
        local_size: int = 96,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        # Monta o dataset so para indexar e obter a lista completa de amostras
        index_ds = FolderDataset(
            data_dir=self.hparams.data_dir,
            image_size=self.hparams.image_size,
        )

        n_val = max(1, math.floor(len(index_ds) * self.hparams.val_split))
        n_train = len(index_ds) - n_val

        train_base, val_base = random_split(
            index_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_samples = [index_ds.samples[i] for i in train_base.indices]
        val_samples = [index_ds.samples[i] for i in val_base.indices]

        if self.hparams.ssl_mode:
            train_ds = FolderDataset(
                data_dir=self.hparams.data_dir,
                image_size=self.hparams.global_size,
            )
            train_ds.samples = train_samples
            self._train_dataset = MultiCropDataset(
                dataset=train_ds,
                n_global=self.hparams.n_global,
                n_local=self.hparams.n_local,
                global_size=self.hparams.global_size,
                local_size=self.hparams.local_size,
            )
        else:
            train_ds = FolderDataset(
                data_dir=self.hparams.data_dir,
                image_size=self.hparams.image_size,
            )
            train_ds.samples = train_samples
            self._train_dataset = train_ds

        val_ds = FolderDataset(
            data_dir=self.hparams.data_dir,
            image_size=self.hparams.image_size,
        )
        val_ds.samples = val_samples
        self._val_dataset = val_ds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
