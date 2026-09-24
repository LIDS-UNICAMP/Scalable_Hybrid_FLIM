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

"""Single-parasite LeJEPA DataModule.

Pipeline
--------
ParasiteLejepaDataModuleSplited
    → DataLoader
    → ParasiteLejepaMultiViewDataset
    → DatasetParasite (single-parasite mode via path_dataset)

YAML base directory
-------------------
configs/data/percentage/<parasite>/<percentage>/lejepa_line.yaml

Where <parasite> is one of:
  - helminth-eggs_split_2
  - helminth-larvae_split_2
  - protozoan-cysts_split_2

And <percentage> is one of: 1, 5, 25, 50, 75, 100

Usage in YAML
-------------
data:
  class_path: src.data_modules.parasite_data_module_lejepa_splited.ParasiteLejepaDataModuleSplited
  init_args:
    parasite_name: helminth-eggs_split_2
    split: 1
    percentage: 1
    batch_size: 15
    num_workers: 4
    loader: ift_lab
    image_size: 200
    V_train: 12
    V_eval: 1
    pin_memory: true
    persistent_workers: true
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_aug, _build_test
from src.data_modules.datasets.parasite_lejepa import ParasiteLejepaMultiViewDataset
from src.transforms.multicrop import MultiCropTransform

logger = logging.getLogger(__name__)


class ParasiteLejepaDataModuleSplited(pl.LightningDataModule):
    """DataModule that trains on a **single** parasite type.

    Wiring order::

        ParasiteLejepaDataModuleSplited
            → DataLoader
            → ParasiteLejepaMultiViewDataset
            → DatasetParasite(path_dataset=parasite_name)

    Args:
        parasite_name: One of ``"helminth-eggs_split_2"``, ``"helminth-larvae_split_2"``,
            or ``"protozoan-cysts_split_2"``.  This value is forwarded to
            :class:`DatasetParasite` as ``path_dataset``, which selects the
            single-parasite branch and skips the legacy multi-parasite
            aggregation.
        split: Fold index (integer) or a custom string identifier.
        percentage: Percentage of labelled training data.
            Use ``None`` for the full (non-incremental) split.
        image_size: Spatial resolution for crops (pixels).
        V_train: Number of augmented views per sample during training.
        V_eval: Number of views per sample during validation/test.
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        pin_memory: Whether to pin DataLoader memory.
        persistent_workers: Whether to keep worker processes alive between
            epochs.
        loader: Image loader backend — ``"ift_lab"`` (default) or ``"pil"``.
        use_mask: Reserved for future mask support (not yet active).
    """

    def __init__(
        self,
        *,
        parasite_name: str,
        split: Union[int, str],
        percentage: float,
        image_size: int = 200,
        V_train: int = 12,
        V_eval: int = 1,
        batch_size: int = 20,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        loader: str = "ift_lab",
        use_mask: bool = False,
        imagenet_norm: bool = True,
        multicrop: bool = False,
        n_global: int = 2,
        n_local: int = 6,
        global_size: Optional[int] = None,
        local_size: int = 96,
        global_scale: tuple[float, float] = (0.3, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.3),
    ) -> None:
        super().__init__()
        self.imagenet_norm = imagenet_norm
        self.parasite_name = parasite_name
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

        # Multi-crop (global/local). Quando multicrop=True, o treino usa
        # n_global + n_local views (escalas distintas, mesmo tamanho de saída)
        # em vez de V_train views idênticas. Default off → comportamento original.
        self.multicrop = multicrop
        self.n_global = n_global
        self.n_local = n_local
        self.global_size = global_size if global_size is not None else image_size
        self.local_size = local_size
        self.global_scale = global_scale
        self.local_scale = local_scale

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        logger.info(
            "[ParasiteLejepaDataModuleSplited] setup | parasite=%s | split=%s | "
            "percentage=%s | image_size=%s | V_train=%s",
            self.parasite_name, self.split, self.percentage,
            self.image_size, self.V_train,
        )

        aug = _build_aug(self.image_size, imagenet_norm=self.imagenet_norm)
        test = _build_test(self.image_size, imagenet_norm=self.imagenet_norm)

        # Multi-crop transform (treino) + nº efetivo de views por amostra.
        multicrop_tf = None
        v_train = self.V_train
        if self.multicrop:
            multicrop_tf = MultiCropTransform(
                n_global=self.n_global,
                n_local=self.n_local,
                global_size=self.global_size,
                local_size=self.local_size,
                global_scale=self.global_scale,
                local_scale=self.local_scale,
            )
            v_train = self.n_global + self.n_local
            logger.info(
                "  multicrop ON | n_global=%d n_local=%d global_size=%d "
                "local_scale=%s global_scale=%s → V_train=%d",
                self.n_global, self.n_local, self.global_size,
                self.local_scale, self.global_scale, v_train,
            )

        # transform=None so ParasiteLejepaMultiViewDataset controls the pipeline.
        base_train = DatasetParasite(
            set_name="train",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
            path_dataset=self.parasite_name,
        )
        base_val = DatasetParasite(
            set_name="validation",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
            path_dataset=self.parasite_name,
        )
        base_test = DatasetParasite(
            set_name="test",
            split=self.split,
            percentage=self.percentage,
            transform=None,
            loader=self.loader,
            path_dataset=self.parasite_name,
        )

        logger.info(
            "  train=%d  val=%d  test=%d samples",
            len(base_train), len(base_val), len(base_test),
        )

        self.ds_train = ParasiteLejepaMultiViewDataset(
            parasite_dataset=base_train,
            V=v_train,
            image_size=self.image_size,
            aug_transform=aug,
            test_transform=test,
            multicrop_transform=multicrop_tf,
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
        assert self.ds_train is not None, "Call setup() before train_dataloader()"
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None, "Call setup() before val_dataloader()"
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None, "Call setup() before test_dataloader()"
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )
