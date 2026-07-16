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

from typing import List, Tuple, Any

import torch
from torch.utils.data import Dataset

from src.transforms.multicrop import MultiCropTransform


class MultiCropDataset(Dataset):
    """
    Wraps FLIMDataset to return multiple augmented views per image.

    Each call to `__getitem__` returns a list of tensors (crops) instead of
    a single tensor. The first `n_global` crops are large-scale views; the
    remaining `n_local` crops are small-scale views.

    Used by: LeJEPA (2 global + N local).

    Args:
        dataset: Underlying FLIMDataset (transform will be replaced).
        n_global: Number of global crops.
        n_local: Number of local crops.
        global_size: Spatial size of global crops.
        local_size: Spatial size of local crops.
        global_scale: Scale range for global crops.
        local_scale: Scale range for local crops.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_global: int = 2,
        n_local: int = 6,
        global_size: int = 224,
        local_size: int = 96,
        global_scale: tuple[float, float] = (0.3, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.3),
    ) -> None:
        self.dataset = dataset

        self.multicrop = MultiCropTransform(
            n_global=n_global,
            n_local=n_local,
            global_size=global_size,
            local_size=local_size,
            global_scale=global_scale,
            local_scale=local_scale,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _to_pil_rgb(self, img: Any):
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image

        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            # Expect [C,H,W] float in [0,1] or [0,255]
            t = img
            if t.ndim != 3:
                raise ValueError(f"Expected image Tensor [C,H,W], got {tuple(t.shape)}")
            if t.dtype.is_floating_point:
                # to_pil_image expects [0,1] for float
                t = t.clamp(0, 1)
            else:
                t = t.to(torch.uint8)
            return to_pil_image(t).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        # Prefer going through the underlying dataset's __getitem__ so any
        # preprocessing (e.g. mask application) happens there.
        img, label = self.dataset[idx]
        pil_img = self._to_pil_rgb(img)
        views = self.multicrop(pil_img)
        return views, int(label)
