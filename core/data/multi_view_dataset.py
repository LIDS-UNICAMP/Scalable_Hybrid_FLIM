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

"""Wrapper que transforma um dataset de imagem unica em dataset de V views.

Movido de src/data_modules/datasets/parasite_lejepa.py
(classe ParasiteLejepaMultiViewDataset). Nao absorve MultiCropDataset:
sao datasets diferentes e continuam separados.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from core.data.transforms import build_aug, build_test


class MultiViewDataset(Dataset):
    """
    Wrapper que:
      1) Le samples e labels via ParasiteDataset (sem mexer nele)
      2) Converte cada amostra em V views usando o mesmo esquema do FolderDataset

    Retorno:
      views: Tensor [V, 3, H, W]
      label: int
    """

    def __init__(
        self,
        *,
        parasite_dataset: Dataset,
        V: int = 2,
        image_size: int = 128,
        aug_transform: Optional[Any] = None,
        test_transform: Optional[Any] = None,
        multicrop_transform: Optional[Any] = None,
    ) -> None:
        """
        Args:
            parasite_dataset: instancia pronta do ParasiteDataset (ou compativel),
                             com transform=None (recomendado) para nao duplicar pipeline.
            V: numero de views por amostra.
            image_size: tamanho base (usado se voce nao passar aug_transform/test_transform).
            aug_transform: se None, usa build_aug(image_size)
            test_transform: se None, usa build_test(image_size)
            multicrop_transform: se fornecido (e V>1), gera as views via multi-crop
                global/local (uma chamada -> lista de n_global+n_local views) em vez
                de aplicar ``aug_transform`` V vezes. None mantem o comportamento
                original de single-view repetido.
        """
        self.base = parasite_dataset
        self.V = int(V)

        # Reusa exatamente a ideia do FolderDataset:
        # - V>1 => aug
        # - V==1 => test
        self.aug = aug_transform or build_aug(image_size)
        self.test = test_transform or build_test(image_size)
        self.multicrop = multicrop_transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def _ensure_pil_rgb(self, img: Any) -> Image.Image:
        """
        ParasiteDataset pode devolver:
          - PIL.Image (loader='pil')
          - torch.Tensor [C,H,W] float (loader='ift_lab')
        Aqui garantimos PIL RGB para alimentar o pipeline de views (que abre PIL RGB).
        """
        if isinstance(img, Image.Image):
            return img.convert("RGB")

        if isinstance(img, torch.Tensor):
            # Esperado: [C,H,W], float em [0,1] ou [0,255]
            if img.ndim != 3:
                raise ValueError(f"Esperava Tensor [C,H,W], recebi shape={tuple(img.shape)}")
            pil = to_pil_image(img)
            return pil.convert("RGB")

        raise TypeError(f"Tipo de imagem nao suportado: {type(img)}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]

        # Multi-crop (global/local) - so no treino (V>1) e quando habilitado.
        # MultiCropTransform recebe PIL e devolve uma lista de n_global+n_local
        # views, todas no mesmo tamanho de saida, entao o stack e seguro.
        if self.multicrop is not None and self.V > 1:
            img_pil = self._ensure_pil_rgb(img)
            views = torch.stack(self.multicrop(img_pil))  # [n_global+n_local,3,H,W]
            return views, int(label)

        transform = self.aug if self.V > 1 else self.test
        views = torch.stack([transform(img) for _ in range(self.V)])  # [V,3,H,W]
        return views, int(label)
