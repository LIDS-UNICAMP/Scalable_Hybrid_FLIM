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

"""Augmentation de treino em lote: LAB[0,1], [B,3,H,W], sorteio por imagem.

``TrainAug(x)`` devolve ``(suja, limpa)``. A geometria vale para as duas; cor, blur, ruido,
oclusao e mistura so para a suja. Entrada = suja e alvo = limpa da restauracao de verdade,
ao contrario de ``strong_aug``, em que o alvo sai da entrada ja aumentada.
Sem chave ligada, ou em eval, devolve ``(x, x)``.

Chaves (0 desliga; ``p`` e a chance de cada familia em cada imagem):
  hflip, vflip  liga o flip
  rotate        graus, +-; borda refletida
  crop          fracao maxima da area cortada
  scale         zoom +-
  l_jitter      k: L*(1+-2k)+-k
  ab_jitter     deslocamento +- de a e de b, separados
  stain         k: ganho 1+-k e giro +-k rad do croma, em torno do ponto sem cor
  blur          sigma maximo
  noise         desvio maximo
  cutout        lado do quadrado, fracao do lado da imagem, preenchido com a media
  erasing       area maxima de um retangulo, preenchido com ruido
  mixup         alpha da Beta; a imagem da vez fica com peso >= 0.5
  cutmix        area maxima de um retangulo vindo do vizinho do lote
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torchvision.transforms import v2

KEYS = ("hflip", "vflip", "rotate", "crop", "scale", "l_jitter", "ab_jitter", "stain",
        "blur", "noise", "cutout", "erasing", "mixup", "cutmix")
GEO_KEYS = KEYS[:5]
AB_NEUTRAL = (0.467, 0.533)  # a, b de qualquer cinza no ift_lab_loader (medido de 0 a 255)


def _u(x: Tensor, k: float, c: int = 1) -> Tensor:
    return (2 * torch.rand(x.shape[0], c, 1, 1, device=x.device) - 1) * k


def _box(x: Tensor, bh: Tensor, bw: Tensor) -> Tensor:
    """Retangulo por imagem; ``bh``/``bw`` sao fracoes do lado, [B,1,1,1]."""
    h, w = x.shape[-2:]
    yy = ((torch.arange(h, device=x.device) + 0.5) / h)[:, None]
    xx = (torch.arange(w, device=x.device) + 0.5) / w
    return (((yy - _u(x, 0.5) - 0.5).abs() < bh / 2)
            & ((xx - _u(x, 0.5) - 0.5).abs() < bw / 2))


class TrainAug(nn.Module):
    def __init__(self, p: float = 0.5, **keys: float) -> None:
        super().__init__()
        if set(keys) - set(KEYS):
            raise ValueError(f"train_aug: chaves invalidas {sorted(set(keys) - set(KEYS))}")
        self.p = p
        self.cfg = {k: keys.get(k, 0) for k in KEYS}
        self.on = any(self.cfg.values())

    def _hit(self, x: Tensor) -> Tensor:
        return torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p

    def geo(self, x: Tensor) -> Tensor:
        c = self.cfg
        if c["hflip"]:
            x = torch.where(self._hit(x), x.flip(-1), x)
        if c["vflip"]:
            x = torch.where(self._hit(x), x.flip(-2), x)
        if not (c["rotate"] or c["crop"] or c["scale"]):
            return x
        ang = _u(x, math.radians(c["rotate"])) * self._hit(x)
        side = (1 - (_u(x, 0.5) + 0.5) * c["crop"] * self._hit(x)).sqrt()
        zoom = side / (1 + _u(x, c["scale"]) * self._hit(x))
        cos, sin = (ang.cos() * zoom).flatten(), (ang.sin() * zoom).flatten()
        tx, ty = ((1 - side) * _u(x, 1)).flatten(), ((1 - side) * _u(x, 1)).flatten()
        theta = torch.stack([cos, -sin, tx, sin, cos, ty], 1).view(-1, 2, 3)
        grid = nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
        return nn.functional.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    def corrupt(self, x: Tensor) -> Tensor:
        c = self.cfg
        lum, ab = x[:, :1], x[:, 1:]
        if c["l_jitter"]:
            k = c["l_jitter"]
            lum = torch.where(self._hit(x), lum * (1 + _u(x, 2 * k)) + _u(x, k), lum)
        if c["ab_jitter"]:
            ab = torch.where(self._hit(x), ab + _u(x, c["ab_jitter"], 2), ab)
        if c["stain"]:
            neutral = x.new_tensor(AB_NEUTRAL)[:, None, None]
            da, db = (ab - neutral).split(1, 1)
            turn, gain = _u(x, c["stain"]), 1 + _u(x, c["stain"])
            rotated = torch.cat([turn.cos() * da - turn.sin() * db,
                                 turn.sin() * da + turn.cos() * db], 1)
            ab = torch.where(self._hit(x), gain * rotated + neutral, ab)
        x = torch.cat([lum, ab], 1)
        if c["blur"]:
            sigma = float(torch.empty(()).uniform_(0.1, c["blur"]))
            k = 2 * math.ceil(3 * sigma) + 1
            x = torch.where(self._hit(x), v2.functional.gaussian_blur(x, [k, k], [sigma, sigma]), x)
        if c["noise"]:
            x = torch.where(self._hit(x), x + (_u(x, 0.5) + 0.5) * c["noise"] * torch.randn_like(x), x)
        if c["cutout"]:
            side = x.new_full((1, 1, 1, 1), c["cutout"])
            x = torch.where(_box(x, side, side) & self._hit(x), x.mean((2, 3), keepdim=True), x)
        if c["erasing"]:
            area = (0.02 + (_u(x, 0.5) + 0.5) * (c["erasing"] - 0.02)).clamp_min(0.02)
            ratio = _u(x, math.log(3)).exp()
            box = _box(x, (area * ratio).sqrt(), (area / ratio).sqrt())
            x = torch.where(box & self._hit(x), torch.rand_like(x), x)
        if c["mixup"]:
            lam = torch.distributions.Beta(c["mixup"], c["mixup"]).sample((x.shape[0], 1, 1, 1)).to(x)
            lam = torch.maximum(lam, 1 - lam)
            x = torch.where(self._hit(x), lam * x + (1 - lam) * x.roll(1, 0), x)
        if c["cutmix"]:
            side = ((_u(x, 0.5) + 0.5) * c["cutmix"]).sqrt()
            x = torch.where(_box(x, side, side) & self._hit(x), x.roll(1, 0), x)
        return x.clamp(0, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.on or not self.training:
            return x, x
        clean = self.geo(x)
        return self.corrupt(clean), clean


if __name__ == "__main__":
    strength = dict(hflip=True, vflip=True, rotate=180, crop=0.4, scale=0.05, l_jitter=0.05,
                    ab_jitter=0.016, stain=0.1, blur=1.2, noise=0.02, cutout=0.2, erasing=0.2,
                    mixup=1.0, cutmix=0.3)
    assert set(strength) == set(KEYS)
    torch.manual_seed(0)
    x = torch.rand(8, 3, 64, 64)

    dirty, clean = TrainAug()(x)
    assert dirty is x and clean is x
    print("desligado: tensor igual")

    dirty, clean = TrainAug(p=1, **strength).eval()(x)
    assert dirty is x and clean is x
    print("eval com tudo ligado: tensor igual")

    for key, value in strength.items():
        ref = x.clone()
        dirty, clean = TrainAug(p=1, **{key: value})(x)
        assert torch.equal(x, ref) and dirty.shape == x.shape and 0 <= dirty.min() <= dirty.max() <= 1
        if key in GEO_KEYS:
            assert not torch.equal(clean, x) and torch.equal(dirty, clean.clamp(0, 1))
            print(f"{key:10s} altera os dois igual, delta medio {(clean - x).abs().mean():.4f}")
        else:
            assert torch.equal(clean, x) and not torch.equal(dirty, x)
            print(f"{key:10s} so a entrada, alvo limpo, delta medio {(dirty - x).abs().mean():.4f}")

    gray = torch.ones(2, 3, 8, 8) * torch.tensor([0.5, *AB_NEUTRAL])[:, None, None]
    assert torch.allclose(TrainAug(p=1, stain=0.3)(gray)[0], gray, atol=1e-6)
    print("stain: cinza continua cinza")

    try:
        TrainAug(ab_jiter=0.1)
    except ValueError as err:
        print("chave errada recusada:", err)
    else:
        raise AssertionError("chave errada passou")
    print(f"sem parametros no state_dict: {not TrainAug(rotate=1).state_dict()}")
