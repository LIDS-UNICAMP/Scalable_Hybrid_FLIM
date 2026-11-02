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

"""Pipelines de augmentation e de teste. Nomes publicos: eram _build_aug/_build_test."""

from __future__ import annotations

import torch
from torchvision.transforms import v2


def build_aug(image_size: int, imagenet_norm: bool = True) -> v2.Compose:
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


def _rot90(img: torch.Tensor) -> torch.Tensor:
    """0/90/180/270 graus ao acaso: sem interpolacao e sem borda nova (imagem quadrada)."""
    return torch.rot90(img, int(torch.randint(4, ())), dims=(-2, -1))


def _strong(img: torch.Tensor) -> torch.Tensor:
    """Rotacao livre com borda refletida + jitter LAB + blur/ruido leves (A18). Sem escala/cutout.
    LAB[0,1]: L/100, a/b/256 -> L*[0.9,1.1]+-5 vira +-0.05, a/b +-4 vira +-4/256."""
    h, w = img.shape[-2:]
    pad = (h + 3) // 4  # > (sqrt2-1)/2 do lado: os cantos da rotacao caem dentro do reflexo
    big = torch.nn.functional.pad(img[None], (pad,) * 4, mode="reflect")[0]
    img = v2.functional.center_crop(v2.functional.rotate(big, float(torch.rand(()) * 360),
                                    v2.InterpolationMode.BILINEAR), [h, w])
    if torch.rand(()) < 0.8:
        u = lambda lo, hi: float(torch.empty(()).uniform_(lo, hi))  # noqa: E731
        img = torch.stack([img[0] * u(0.9, 1.1) + u(-0.05, 0.05),
                           img[1] + u(-4, 4) / 256, img[2] + u(-4, 4) / 256])
    if torch.rand(()) < 0.3:
        img = v2.functional.gaussian_blur(img, [5, 5], [float(torch.empty(()).uniform_(0.1, 1.2))])
    if torch.rand(()) < 0.3:
        img = img + float(torch.rand(())) * 0.02 * torch.randn_like(img)
    return img.clamp(0, 1)


def build_test(image_size: int, imagenet_norm: bool = True, geo_aug: bool = False,
               strong_aug: bool = False) -> v2.Compose:
    steps = [
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),  # Convert to uint8 for consistency
        v2.Resize(image_size),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
    ]
    # Flip H/V + rotacao de 90 graus: o parasito nao tem orientacao canonica na lamina.
    # Nada de cor, blur, crop ou distorcao: cor, textura e morfologia sao a classe.
    if geo_aug:
        steps += [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.Lambda(_rot90)]
    if strong_aug:
        steps.append(v2.Lambda(_strong))
    if imagenet_norm:
        steps.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(steps)
