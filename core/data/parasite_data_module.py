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

"""DataModule unico do dataset de parasitas.

Funde os tres datamodules antigos numa classe so; a diferenca entre eles
virou parametro do ``__init__``:

===============================  =========================================
Classe antiga                    Parametros equivalentes
===============================  =========================================
ParasiteLejepaDataModuleSplited  multi_view=True, parasite_name="<nome>"
ParasiteLejepaDataModule         multi_view=True, parasite_name=None
ParasiteDataModule (supervisao)  multi_view=False, drop_last=True,
                                 persistent_workers=False
===============================  =========================================

Encadeamento com multi_view=True (LeJEPA / SSL multiview)::

    ParasiteDataModule -> DataLoader -> MultiViewDataset -> ParasiteDataset

Encadeamento com multi_view=False (supervisionado / fine-tuning)::

    ParasiteDataModule -> DataLoader -> [MultiCropDataset] -> ParasiteDataset
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from core.data.multi_crop_dataset import MultiCropDataset
from core.data.multi_crop_transform import MultiCropTransform
from core.data.multi_view_dataset import MultiViewDataset
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_aug, build_test

logger = logging.getLogger(__name__)


class ParasiteDataModule(pl.LightningDataModule):
    """DataModule do dataset de parasitas, em modo multiview ou supervisionado.

    Args:
        split: Indice do fold (int) ou identificador string.
        percentage: Percentual de dados rotulados de treino. ``None`` = split cheio.
        parasite_name: Nome de um unico parasita (``"helminth-eggs_split_2"``,
            ``"helminth-larvae_split_2"``, ``"protozoan-cysts_split_2"``),
            repassado a :class:`ParasiteDataset` como ``path_dataset``.
            ``None`` mantem o modo legado que agrega as tres classes.
        multi_view: ``True`` entrega ``[V, 3, H, W]`` via :class:`MultiViewDataset`
            (pipeline LeJEPA). ``False`` entrega ``(imagem, label)`` cru,
            aplicando ``transform``.
        transform: Transform torchvision aplicada por amostra. Usada apenas
            quando ``multi_view=False``; no modo multiview o wrapper controla
            o pipeline e o dataset base recebe ``transform=None``.
        image_size: Resolucao espacial dos crops (pixels), modo multiview.
        V_train: Numero de views por amostra no treino (ignorado com multicrop).
        V_eval: Numero de views por amostra em validacao/teste.
        batch_size: Amostras por batch.
        num_workers: Processos worker do DataLoader.
        pin_memory: Fixa a memoria do DataLoader.
        persistent_workers: Mantem os workers vivos entre epocas.
        drop_last: Descarta o ultimo batch incompleto do treino.
        loader: Backend de leitura de imagem: ``"ift_lab"`` ou ``"pil"``.
        use_mask: Reservado para suporte a mascara (ainda inativo).
        imagenet_norm: Aplica a normalizacao ImageNet nas transforms. Deve ficar
            desligado para entrada FLIM (LAB em [0,1]).
        multicrop: Liga o multi-crop global/local no treino. Com
            ``multi_view=True`` usa :class:`MultiCropTransform` dentro do
            wrapper; com ``multi_view=False`` envolve o treino em
            :class:`MultiCropDataset` (o antigo ``ssl_mode``).
        n_global: Numero de crops globais (multicrop).
        n_local: Numero de crops locais (multicrop).
        global_size: Tamanho do crop global. ``None`` usa ``image_size`` no modo
            multiview e 224 no modo supervisionado.
        local_size: Tamanho do crop local.
        global_scale: Faixa de escala dos crops globais.
        local_scale: Faixa de escala dos crops locais.
    """

    def __init__(
        self,
        *,
        split: Union[int, str],
        percentage: float,
        parasite_name: Optional[str] = None,
        multi_view: bool = True,
        transform: Any = None,
        image_size: int = 200,
        V_train: int = 12,
        V_eval: int = 1,
        batch_size: int = 20,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = False,
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
        self.split = split
        self.percentage = percentage
        self.parasite_name = parasite_name
        self.multi_view = multi_view
        self.transform = transform
        self.image_size = image_size
        self.V_train = V_train
        self.V_eval = V_eval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.loader = loader
        self.use_mask = use_mask
        self.imagenet_norm = imagenet_norm

        # Multi-crop (global/local). Com multicrop=True o treino usa
        # n_global + n_local views (escalas distintas, mesmo tamanho de saida)
        # em vez de V_train views identicas. Default off -> comportamento original.
        self.multicrop = multicrop
        self.n_global = n_global
        self.n_local = n_local
        # Default historico do crop global: image_size no modo multiview,
        # 224 no modo supervisionado.
        self.global_size = global_size if global_size is not None else (
            image_size if multi_view else 224
        )
        self.local_size = local_size
        self.global_scale = global_scale
        self.local_scale = local_scale

        self.ds_train: Optional[Dataset] = None
        self.ds_val: Optional[Dataset] = None
        self.ds_test: Optional[Dataset] = None

    def _base(self, set_name: str, transform: Any) -> Dataset:
        return ParasiteDataset(
            set_name=set_name,
            split=self.split,
            percentage=self.percentage,
            transform=transform,
            loader=self.loader,
            path_dataset=self.parasite_name,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        logger.info(
            "[ParasiteDataModule] setup | parasite=%s | split=%s | percentage=%s | "
            "multi_view=%s | image_size=%s | V_train=%s",
            self.parasite_name, self.split, self.percentage,
            self.multi_view, self.image_size, self.V_train,
        )

        if not self.multi_view:
            # Modo supervisionado: o transform do usuario vai direto no dataset.
            train: Dataset = self._base("train", self.transform)
            if self.multicrop:
                # MultiCropDataset reabre as imagens a partir de dataset.samples.
                train = MultiCropDataset(
                    dataset=train,
                    n_global=self.n_global,
                    n_local=self.n_local,
                    global_size=self.global_size,
                    local_size=self.local_size,
                    global_scale=self.global_scale,
                    local_scale=self.local_scale,
                )
            self.ds_train = train
            self.ds_val = self._base("validation", self.transform)
            self.ds_test = self._base("test", self.transform)
            return

        aug = build_aug(self.image_size, imagenet_norm=self.imagenet_norm)
        test = build_test(self.image_size, imagenet_norm=self.imagenet_norm)

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
                "local_scale=%s global_scale=%s -> V_train=%d",
                self.n_global, self.n_local, self.global_size,
                self.local_scale, self.global_scale, v_train,
            )

        # transform=None no dataset base para o wrapper controlar o pipeline.
        views = {"train": v_train, "validation": self.V_eval, "test": self.V_eval}
        self.ds_train, self.ds_val, self.ds_test = [
            MultiViewDataset(
                parasite_dataset=self._base(set_name, None),
                V=v,
                image_size=self.image_size,
                aug_transform=aug,
                test_transform=test,
                multicrop_transform=multicrop_tf if set_name == "train" else None,
            )
            for set_name, v in views.items()
        ]

        logger.info(
            "  train=%d  val=%d  test=%d samples",
            len(self.ds_train), len(self.ds_val), len(self.ds_test),
        )

    def _dl(self, ds: Optional[Dataset], *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        assert ds is not None, "chame setup() antes dos dataloaders"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dl(self.ds_train, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        return self._dl(self.ds_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dl(self.ds_test, shuffle=False)
