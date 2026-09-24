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

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


def _build_aug(image_size: int, imagenet_norm: bool = True) -> v2.Compose:
    steps = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),  # Convert to uint8 first for solarize compatibility
        v2.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    # ImageNet RGB Normalize is wrong for FLIM-init (input is LAB[0,1]). Skip when disabled.
    if imagenet_norm:
        steps.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(steps)


def _build_test(image_size: int, imagenet_norm: bool = True) -> v2.Compose:
    steps = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),  # Convert to uint8 for consistency
        v2.Resize(image_size),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if imagenet_norm:
        steps.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(steps)


class LejepaDataset(Dataset):
    """
    ImageFolder-style dataset for FLIM RGB images with multi-view support.

    Directory layout::

        data_dir/
            class_a/  image1.png  image2.tif  ...
            class_b/  ...

    Flat directory (no subdirs) is also accepted — all images get label 0
    and the dataset operates in label-free (SSL) mode.

    Args:
        data_dir: Root directory.
        V: Number of views per sample.
           - V=1 → applies test/val transform, returns Tensor [3, H, W].
           - V>1 → applies augmentation V times, returns stacked Tensor [V, 3, H, W].
        image_size: Spatial resolution for crops.
        extensions: Accepted file extensions.
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        data_dir: str,
        V: int = 1,
        image_size: int = 128,
        extensions: Optional[set] = None,
    ) -> None:
        self.V = V
        self.aug = _build_aug(image_size)
        self.test = _build_test(image_size)
        self.extensions = extensions or self.EXTENSIONS

        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self._build_index(Path(data_dir))

    def _build_index(self, root: Path) -> None:
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if subdirs:
            self.classes = [d.name for d in subdirs]
            for label, subdir in enumerate(subdirs):
                for path in sorted(subdir.iterdir()):
                    if path.suffix.lower() in self.extensions:
                        self.samples.append((path, label))
        else:
            self.classes = ["unlabeled"]
            for path in sorted(root.iterdir()):
                if path.suffix.lower() in self.extensions:
                    self.samples.append((path, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), label


if __name__ == "__main__":
    import sys
    from collections import Counter

    # ── edit these ────────────────────────────────────────
    DATA_DIR   = "data/flim"
    V          = 2        # views per sample: 1=val transform, >1=aug
    IMAGE_SIZE = 128
    N_INSPECT  = 3        # how many samples to print
    # ──────────────────────────────────────────────────────

    print(f"\n=== LejepaDataset smoke test ===")
    print(f"  data_dir   : {DATA_DIR}")
    print(f"  V          : {V}")
    print(f"  image_size : {IMAGE_SIZE}\n")

    ds = LejepaDataset(data_dir=DATA_DIR, V=V, image_size=IMAGE_SIZE)

    print(f"Classes  : {ds.classes}")
    print(f"Samples  : {len(ds)}")

    if len(ds) == 0:
        print("ERROR: No images found. Check DATA_DIR and file extensions.")
        sys.exit(1)

    for i in range(min(N_INSPECT, len(ds))):
        tensor, label = ds[i]
        print(f"  [{i}] shape={tuple(tensor.shape)}  label={label}  "
              f"min={tensor.min():.3f}  max={tensor.max():.3f}")

    dist = Counter(label for _, label in ds.samples)
    print(f"\nClass distribution: { {ds.classes[k]: v for k, v in sorted(dist.items())} }")
    print("\nOK")
