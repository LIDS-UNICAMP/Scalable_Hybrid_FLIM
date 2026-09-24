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

from __future__ import annotations

from typing import Optional, Union

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_aug, _build_test
from src.data_modules.datasets.parasite_lejepa import ParasiteLejepaMultiViewDataset


class ParasiteLejepaDataModule(pl.LightningDataModule):
    """
    DataModule que cria:
      - DatasetParasite(set_name='train'/'val'/'test', ...)
      - Wrapper ParasiteLejepaMultiViewDataset para entregar multiview [B,V,3,H,W]
    """

    def __init__(
        self,
        *,
        split: Union[int, str],
        percentage: float,
        image_size: int = 128,
        V_train: int = 2,
        V_eval: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        loader: str = "ift_lab",
        use_mask: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.percentage = percentage
        self.image_size = image_size
        self.V_train = V_train
        self.V_eval = V_eval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.loader = loader
        self.use_mask = use_mask

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:

        aug = _build_aug(self.image_size)
        test = _build_test(self.image_size)

        # IMPORTANTÍSSIMO: transform=None aqui, para o wrapper controlar o multi-view
        base_train = DatasetParasite(
            set_name="train",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
        )
        base_val = DatasetParasite(
            set_name="validation",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
        )
        base_test = DatasetParasite(
            set_name="test",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
        )

        self.ds_train = ParasiteLejepaMultiViewDataset(
            parasite_dataset=base_train,
            V=self.V_train,
            image_size=self.image_size,
            aug_transform=aug,
            test_transform=test,
        )
        self.ds_val = ParasiteLejepaMultiViewDataset(
            parasite_dataset=base_val,
            V=self.V_eval,
            image_size=self.image_size,
            aug_transform=aug,
            test_transform=test,
        )
        self.ds_test = ParasiteLejepaMultiViewDataset(
            parasite_dataset=base_test,
            V=self.V_eval,
            image_size=self.image_size,
            aug_transform=aug,
            test_transform=test,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )