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

from __future__ import annotations

from typing import Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.multicrop_dataset import MultiCropDataset


class ParasiteDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the Parasite dataset.

    Wraps DatasetParasite (which reads split JSONs via config helpers) and
    exposes standard Lightning train/val/test dataloaders.

    Supports two modes:
    - ``ssl_mode=False`` (default): returns ``(image_tensor, label)`` for
      supervised or fine-tuning workflows.
    - ``ssl_mode=True``: wraps the training set with MultiCropDataset to
      return multi-crop views for self-supervised learning.

    Args:
        dataset_name: Key passed to ``get_dataset_paths`` / ``get_split_path_incremental``.
        split: Split identifier (e.g. 0, 1, … or a string name).
        percentage: Percentage of labelled data used in the incremental split.
        batch_size: Batch size for all dataloaders.
        num_workers: DataLoader worker processes.
        transform: torchvision transform applied to every sample.
        loader: ``"pil"`` (default) or ``"ift_lab"`` image loader.
        ssl_mode: If True, wraps the train dataset with MultiCropDataset.
        n_global: Number of global crops (ssl_mode only).
        n_local: Number of local crops (ssl_mode only).
        global_size: Global crop spatial size (ssl_mode only).
        local_size: Local crop spatial size (ssl_mode only).
        pin_memory: Whether to pin memory in DataLoaders.
        use_mask: If True, loads the corresponding mask from the dataset's masks_dir
          and applies it to the image (image = image * mask) before transforms.
    """

    def __init__(
        self,
        dataset_name: str,
        split: int | str,
        percentage: int | float,
        batch_size: int = 32,
        num_workers: int = 4,
        transform=None,
        loader: str = "pil",
        ssl_mode: bool = False,
        n_global: int = 2,
        n_local: int = 6,
        global_size: int = 224,
        local_size: int = 96,
        pin_memory: bool = True,
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        # NOTE: dataset_name is kept for backwards compatibility with existing configs,
        # but DatasetParasite currently always uses the 'parasito' registry via config.get_parasito_split_paths.
        self.save_hyperparameters(ignore=["transform"])
        self.transform = transform

    def setup(self, stage: Optional[str] = None) -> None:
        hp = self.hparams

        if hp.ssl_mode:
            train_ds = DatasetParasite(
                set_name="train",
                split=hp.split,
                percentage=hp.percentage,
                transform=self.transform,
                loader=hp.loader,
            )
            self._train_dataset = MultiCropDataset(
                dataset=train_ds,  # MultiCropDataset re-opens images from dataset.samples
                n_global=hp.n_global,
                n_local=hp.n_local,
                global_size=hp.global_size,
                local_size=hp.local_size,
            )
        else:
            self._train_dataset = DatasetParasite(
                set_name="train",
                split=hp.split,
                percentage=hp.percentage,
                transform=self.transform,
                loader=hp.loader,
            )

        self._val_dataset = DatasetParasite(
            set_name="validation",
            split=hp.split,
            percentage=hp.percentage,
            transform=self.transform,
            loader=hp.loader,
        )

        self._test_dataset = DatasetParasite(
            set_name="test",
            split=hp.split,
            percentage=hp.percentage,
            transform=self.transform,
            loader=hp.loader,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
