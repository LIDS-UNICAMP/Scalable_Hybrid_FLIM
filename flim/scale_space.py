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
"""scale_space.py — espaco de escalas do SIFT num lote (N, C, H, W): Gaussiana, DoG e LoG.

Piramide de Lowe 2004: LEVELS niveis por oitava, LEVELS + 3 imagens, SIGMA0 sobre a imagem que ja
vem com INPUT_SIGMA, e a oitava seguinte nasce do nivel LEVELS decimado por 2. Sem a oitava -1 e
sem ajuste quadratico: a semente vira pixel inteiro de qualquer jeito.

Extremo = maximo ou minimo dos 26 vizinhos em (x, y, sigma), com |resposta| >= contraste e razao de
curvaturas abaixo de EDGE_RATIO. O mesmo detector serve DoG e LoG. Como D ~ (k - 1) sigma^2 lap L, o
limiar do LoG e o do DoG dividido por (k - 1): os dois cortam a mesma coisa.

`view_pool` e o `patch_pool` da grade para candidatos com escala. Cada linha traz uma chave
(tipo, oitava, nivel): RAW e o patch cru, GAUSS e DOG saem do nivel da piramide do mapa, REGION
pega uma regiao d vezes maior, d = sigma / SIGMA0, e a reduz ao kernel. Oitava e nivel valem nos
pixels do mapa da camada: a piramide de uma camada funda e a do mapa dela, com oitava cortada
ao que o mapa comporta.

Uma linha e padronizada no proprio mapa, com media e desvio das sementes de referencia nesse mesmo
mapa, e so depois escrita no z-score comum: DoG tem media ~0, e o z-score do mapa cru a afogaria
numa constante. Para RAW isso e a identidade, por isso o patch cru nao muda.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from flim.spifil import rescale

SIGMA0, LEVELS, INPUT_SIGMA = 1.6, 3, 0.5
K = 2 ** (1 / LEVELS)
MIN_SIDE = 16            # menor lado de uma oitava
CONTRAST = 0.02 / LEVELS  # o SIFT de `eval/spf_grade.py`: contrastThreshold 0,02, dividido como no OpenCV
EDGE_RATIO = 10.0        # Lowe 2004, secao 4.1
BORDER = 2               # meio kernel: o patch do extremo cabe na oitava
RAW, GAUSS, DOG, REGION = range(4)
VIEW_ELEMENTS = 2**23    # floats de um nivel no lote de `view_pool`


@dataclass
class Keypoints:
    """Extremos de um lote, do mais forte ao mais fraco, em pixels da imagem de entrada."""
    image: Tensor
    y: Tensor
    x: Tensor
    octave: Tensor
    level: Tensor
    response: Tensor


def sigma_of(octave: int, level: int) -> float:
    """Sigma do nivel, em pixels da entrada."""
    return SIGMA0 * 2 ** (octave + level / LEVELS)


def octaves_of(side: int) -> int:
    return max(1, int(math.log2(side / MIN_SIDE)) + 1)


def blur(x: Tensor, sigma: float) -> Tensor:
    """Gaussiana separavel canal a canal, borda replicada."""
    if sigma <= 0:
        return x
    radius = max(1, math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    g = torch.exp(-t**2 / (2 * sigma**2))
    g = g / g.sum()
    n, c, h, w = x.shape
    y = F.conv2d(F.pad(x.reshape(n * c, 1, h, w), (radius, radius, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="replicate"), g.view(1, 1, -1, 1))
    return y.view(n, c, h, w)


def sync(x: Tensor) -> float:
    """Relogio que espera a GPU: sem isso o tempo por nivel mede so o lancamento."""
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    return time.perf_counter()


def note(records: dict | None, kind: str, octave: int, level: int, shape: tuple, **add) -> None:
    """Acumula o registro de um nivel: sigma, oitava, nivel, resolucao, tempo e candidatos."""
    if records is not None:
        row = records.setdefault((kind, octave, level), {
            "kind": kind, "octave": octave, "level": level, "sigma": round(sigma_of(octave, level), 3), "height": shape[0],
            "width": shape[1], "seconds": 0.0, "extrema": 0, "candidates": 0, "kept": 0})
        for name, value in add.items():
            row[name] += value


def gaussian_pyramid(x: Tensor, records: dict | None = None, octaves: int = 99) -> list:
    """L(x, y, sigma): uma pilha (LEVELS + 3, N, C, h, w) por oitava."""
    out, start = [], sync(x)
    base = blur(x, (SIGMA0**2 - INPUT_SIGMA**2) ** 0.5)
    for octave in range(min(octaves, octaves_of(min(x.shape[-2:])))):
        stack = [base]
        for level in range(LEVELS + 3):
            if level:
                stack.append(blur(stack[-1], SIGMA0 * K ** (level - 1) * (K**2 - 1) ** 0.5))
            now = sync(x)
            note(records, "gauss", octave, level, tuple(base.shape[-2:]), seconds=now - start)
            start = now
        out.append(torch.stack(stack))
        base = stack[LEVELS][..., ::2, ::2]
    return out


def dog_pyramid(pyramid: list) -> list:
    """D(x, y, sigma) = L(x, y, k sigma) - L(x, y, sigma), LEVELS + 2 niveis por oitava."""
    return [stack[1:] - stack[:-1] for stack in pyramid]


def log_pyramid(pyramid: list) -> list:
    """sigma^2 lap L nos pixels da oitava, que e o mesmo valor nos pixels da entrada."""
    out = []
    for stack in pyramid:
        s, n, c, h, w = stack[:-1].shape
        kernel = stack.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
        lap = F.conv2d(F.pad(stack[:-1].reshape(s * n * c, 1, h, w), (1, 1, 1, 1), mode="replicate"), kernel)
        out.append(lap.view(s, n, c, h, w) * (SIGMA0 * K ** torch.arange(s, dtype=stack.dtype, device=stack.device)).view(-1, 1, 1, 1, 1) ** 2)
    return out


def extrema(stack: Tensor, threshold: float, edge_ratio: float | None = EDGE_RATIO) -> tuple:
    """Pilha (S, N, h, w) -> mapas booleanos: extremos 3D e os que passam contraste e borda."""
    v = stack.permute(1, 0, 2, 3)[:, None]
    top = F.max_pool3d(v, 3, 1, 1)[:, 0].permute(1, 0, 2, 3)
    low = -F.max_pool3d(-v, 3, 1, 1)[:, 0].permute(1, 0, 2, 3)
    raw = (stack == top) | (stack == low)
    raw[[0, -1]] = False
    raw[..., :BORDER, :] = raw[..., -BORDER:, :] = False
    raw[..., :BORDER] = raw[..., -BORDER:] = False
    hit = raw & (stack.abs() >= threshold)
    if edge_ratio is not None:
        c = stack[..., 1:-1, 1:-1]
        dxx = stack[..., 1:-1, 2:] + stack[..., 1:-1, :-2] - 2 * c
        dyy = stack[..., 2:, 1:-1] + stack[..., :-2, 1:-1] - 2 * c
        dxy = (stack[..., 2:, 2:] - stack[..., 2:, :-2] - stack[..., :-2, 2:] + stack[..., :-2, :-2]) / 4
        det = dxx * dyy - dxy**2
        hit[..., 1:-1, 1:-1] &= (det > 0) & ((dxx + dyy) ** 2 * edge_ratio < (edge_ratio + 1) ** 2 * det)
    return raw, hit


def detect(maps: list, threshold: float, kind: str = "dog", records: dict | None = None,
           edge_ratio: float | None = EDGE_RATIO) -> Keypoints:
    """Extremos de uma piramide de 1 canal, todas as oitavas juntas."""
    found = []
    for octave, stack in enumerate(maps):
        start = sync(stack)
        raw, hit = extrema(stack[:, :, 0], threshold, edge_ratio)
        level, image, y, x = hit.nonzero(as_tuple=True)
        found.append((image, y * 2**octave, x * 2**octave, torch.full_like(y, octave), level, stack[:, :, 0][level, image, y, x]))
        spent = (sync(stack) - start) / max(LEVELS, 1)
        for i in range(1, len(stack) - 1):
            note(records, kind, octave, i, tuple(stack.shape[-2:]), seconds=spent, extrema=int(raw[i].sum()), candidates=int(hit[i].sum()))
    points = [torch.cat(part) for part in zip(*found)]
    order = points[-1].abs().argsort(descending=True, stable=True)
    return Keypoints(*[p[order] for p in points])


def scale_space(x: Tensor, records: dict | None = None) -> dict:
    """Extremos DoG e LoG de um lote de 1 canal."""
    pyramid = gaussian_pyramid(x, records)
    return {"dog": detect(dog_pyramid(pyramid), CONTRAST, "dog", records),
            "log": detect(log_pyramid(pyramid), CONTRAST / (K - 1), "log", records)}


def sample_patches(maps: Tensor, owner: Tensor, yx: Tensor, kernel: int, spacing: float = 1.0) -> Tensor:
    """Lote (B, C, H, W) -> (n, C * k * k) canal primeiro, como `extract_patches`; bilinear, fora do mapa vale 0."""
    h, w = maps.shape[-2:]
    step = (torch.arange(kernel, dtype=yx.dtype, device=yx.device) - kernel // 2) * spacing
    y, x = yx[:, :1] + step, yx[:, 1:] + step
    y0, x0 = y.floor(), x.floor()
    out = 0
    for dy, wy in ((0, 1 - (y - y0)), (1, y - y0)):
        for dx, wx in ((0, 1 - (x - x0)), (1, x - x0)):
            yy, xx = (y0 + dy).long(), (x0 + dx).long()
            weight = (wy * ((yy >= 0) & (yy < h)))[:, :, None] * (wx * ((xx >= 0) & (xx < w)))[:, None, :]
            found = maps[owner[:, None, None], :, yy.clamp(0, h - 1)[:, :, None], xx.clamp(0, w - 1)[:, None, :]]
            out = out + found * weight[..., None]
    return out.permute(0, 3, 1, 2).flatten(1)


class Maps:
    """Os mapas de escala de um lote (B, C, h, w), feitos so quando alguma chave pede."""

    def __init__(self, feature: Tensor, octaves: int) -> None:
        self.feature, self.octaves = feature, octaves

    @cached_property
    def pyramid(self) -> list:
        return gaussian_pyramid(self.feature, octaves=self.octaves)

    @cached_property
    def dog(self) -> list:
        return dog_pyramid(self.pyramid)

    def patches(self, key: tuple, owner: Tensor, yx: Tensor, kernel: int) -> Tensor:
        """Um patch por posicao `yx`, em pixels do mapa, no mapa da chave."""
        kind, octave, level = key
        if kind == RAW:
            return sample_patches(self.feature, owner, yx, kernel)
        if kind == REGION:
            d = max(1.0, min(2 ** (octave + level / LEVELS), min(self.feature.shape[-2:]) / (2 * kernel)))
            return sample_patches(blur(self.feature, INPUT_SIGMA * (d**2 - 1) ** 0.5), owner, yx, kernel, d)
        stacks = self.pyramid if kind == GAUSS else self.dog
        octave = min(octave, len(stacks) - 1)
        return sample_patches(stacks[octave][level], owner, yx / 2**octave, kernel)


def located(seeds: list, grid: tuple, device: torch.device) -> tuple:
    """Dono e posicao de cada semente de um lote, no grid do mapa."""
    found = [rescale(s, grid).coords for s in seeds]
    owner = torch.arange(len(found)).repeat_interleave(torch.tensor([len(c) for c in found]))
    return owner.to(device), torch.cat(found).to(device, torch.float32)


@torch.no_grad()
def view_pool(images: list, seeds: list, keys: list, ref_seeds: list, encoder, kernel: int,
              mean: Tensor, stdev: Tensor, floor: float, flat: float) -> Tensor:
    """Uma linha por semente, em RAM, escrita no espaco de `mean` e `stdev`. Ver o docstring do modulo.

    Janela sem informacao no proprio mapa sai zerada, e o `Ctx.keep` da grade a descarta. Em lotes:
    com a GPU dividida entre processos, cada sincronia por imagem custa mais que o calculo.
    """
    every = np.concatenate(keys)
    names = [tuple(k) for k in np.unique(every, axis=0).tolist()]
    top = 1 + max((k[1] for k in names if k[0] in (GAUSS, DOG)), default=0)
    sums = {k: [0, 0.0, 0.0] for k in names if k[0] != RAW}
    shape = encoder(images[0][None]).shape
    batch = max(1, VIEW_ELEMENTS // math.prod(shape[1:]))
    parts = []
    for start in range(0, len(images), batch):
        part = slice(start, start + batch)
        feature = encoder(torch.stack(images[part]))
        grid, key = tuple(feature.shape[2:]), np.concatenate(keys[part])
        owner, at = located(seeds[part], grid, feature.device)
        ref_owner, ref_at = located(ref_seeds[part], grid, feature.device)
        out, maps = feature.new_zeros(len(key), mean.numel()), Maps(feature, top)
        for name in names:
            index = torch.from_numpy(np.flatnonzero((key == np.array(name)).all(1))).to(feature.device)
            if name[0] == RAW:
                out[index] = maps.patches(name, owner[index], at[index], kernel)
                continue
            rows = maps.patches(name, torch.cat([ref_owner, owner[index]]), torch.cat([ref_at, at[index]]), kernel)
            base = rows[:len(ref_at)].double()
            total = sums[name]
            total[0] += len(base)
            total[1] = total[1] + base.sum(0)
            total[2] = total[2] + base.pow(2).sum(0)
            out[index] = rows[len(ref_at):]
        parts.append(out.cpu())
    rows = torch.cat(parts)
    for name, (n, total, squares) in sums.items():
        mine = torch.from_numpy((every == np.array(name)).all(1))
        here = total.cpu() / n
        spread = (squares.cpu() / n - here**2).clamp_min(0).sqrt() + floor
        z = (rows[mine].double() - here) / spread
        live = (rows[mine].std(1) >= flat)[:, None]
        rows[mine] = torch.where(live, mean + stdev * z.float(), torch.zeros(()))
    return rows


if __name__ == "__main__":
    from spifil.config import LayerSpec
    from spifil.patches import extract_patches
    from spifil.types import Seeds

    side = 201
    impulse = torch.zeros(1, 1, side, side)
    impulse[0, 0, side // 2, side // 2] = 1
    pyramid = gaussian_pyramid(impulse)
    grid_y = (torch.arange(side, dtype=torch.float32) - side // 2)[:, None] ** 2
    for level in range(LEVELS + 3):
        spread = float((pyramid[0][level, 0, 0] * grid_y).sum()) ** 0.5
        want = (sigma_of(0, level) ** 2 - INPUT_SIGMA**2) ** 0.5
        assert abs(spread - want) < 0.02 * want, (level, spread, want)
    print(f"piramide: impulso de lado {side}, oitavas {[tuple(p.shape[-2:]) for p in pyramid]}, desvio medido = sigma nos {LEVELS + 3} niveis")

    side = 400
    yy, xx = torch.meshgrid(torch.arange(side).float(), torch.arange(side).float(), indexing="ij")

    def blob(y: int, x: int, t: float, amp: float) -> Tensor:
        return amp * torch.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * t**2))

    blobs = [(60, 60, 3.0), (60, 200, 6.0), (200, 100, 12.0), (280, 280, 24.0)]
    image = sum(blob(y, x, t, 0.8) for y, x, t in blobs)[None, None]
    records = {}
    found = scale_space(image, records)
    for name, points in found.items():
        for y, x, t in blobs:
            near = ((points.y - y) ** 2 + (points.x - x) ** 2).float().sqrt() <= max(2.0, t / 2)
            assert near.any(), (name, y, x, t)
            first = near.nonzero()[0, 0]
            sigma = sigma_of(int(points.octave[first]), int(points.level[first]))
            ratio = math.log2(sigma / t)
            assert abs(ratio) <= 0.75, (name, t, sigma)
            print(f"{name}: blob t={t:<4} em ({y},{x}) -> ({int(points.y[first])},{int(points.x[first])}), sigma {sigma:.2f}, log2(sigma/t) {ratio:+.2f}")

    weak = scale_space(blob(200, 200, 6.0, 0.5 * CONTRAST)[None, None])
    flat = scale_space(torch.full((1, 1, side, side), 0.4))
    assert all(len(p.y) == 0 for p in (*weak.values(), *flat.values()))
    print("blob abaixo do contraste e imagem chapada: 0 extremos em DoG e LoG")

    step = blur((xx >= side // 2).float()[None, None] * 0.6 + 0.2, 1.0)
    stacks = dog_pyramid(gaussian_pyramid(step))
    loose = detect(stacks, CONTRAST, edge_ratio=None)
    tight = detect(stacks, CONTRAST)
    assert len(loose.y) > 10 * max(len(tight.y), 1)
    print(f"degrau reto: {len(loose.y)} extremos sem o filtro de borda, {len(tight.y)} com ele")

    pyramid = gaussian_pyramid(image)
    d, g = dog_pyramid(pyramid)[1], log_pyramid(pyramid)[1]
    corr = float(torch.corrcoef(torch.stack([d.flatten(), g.flatten()]))[0, 1])
    gain = float((d * g).sum() / (g * g).sum())
    assert corr > 0.97 and abs(gain / (K - 1) - 1) < 0.3, (corr, gain)
    print(f"DoG ~ (k-1) LoG: correlacao {corr:.4f}, ganho {gain:.3f} contra k-1 = {K - 1:.3f}")

    for row in list(records.values())[:3] + [r for r in records.values() if r["kind"] != "gauss"][:3]:
        print("registro:", row)

    torch.manual_seed(0)
    feature = torch.rand(3, 50, 60)
    smooth = blur(feature[None], 1.0)
    y, x, d = 25.0, 31.0, 1.5
    grid = torch.stack(((x + d * (torch.arange(5.0) - 2))[None].expand(5, 5), (y + d * (torch.arange(5.0) - 2))[:, None].expand(5, 5)), -1)
    grid = grid / torch.tensor([59.0, 49.0]) * 2 - 1
    twin = F.grid_sample(smooth, grid[None], align_corners=True)[0].flatten()
    assert torch.allclose(sample_patches(smooth, torch.zeros(1, dtype=torch.long), torch.tensor([[y, x]]), 5, d)[0], twin, atol=1e-5)
    print("sample_patches com espacamento fracionario = grid_sample bilinear")
    yx = torch.tensor([[0, 0], [10, 20], [49, 59], [25, 31]])
    seeds = Seeds.from_coords(yx, 0, (50, 60))
    want = extract_patches(feature, seeds, LayerSpec(kernel_size=5, out_channels=1)).feats
    assert torch.allclose(sample_patches(feature[None], torch.zeros(4, dtype=torch.long), yx.float(), 5), want, atol=1e-5)
    print("sample_patches em coordenada inteira = extract_patches do SPiFiL, com a borda zerada")

    images = [torch.rand(3, 64, 64) for _ in range(3)]
    ref = [Seeds.from_coords(torch.randint(4, 60, (40, 2)), 0, (64, 64)) for _ in images]
    plain = [Seeds.from_coords(torch.randint(4, 60, (6, 2)), 0, (64, 64)) for _ in images]
    mixed = [np.array([[RAW, 0, 0], [RAW, 0, 0], [GAUSS, 0, 1], [DOG, 0, 2], [DOG, 1, 1], [REGION, 1, 0]]) for _ in images]
    span = LayerSpec(kernel_size=5, out_channels=1)
    feats = torch.cat([extract_patches(i, s, span).feats for i, s in zip(images, ref)])
    mean, stdev = feats.mean(0), feats.std(0, unbiased=False) + 0.01
    rows = view_pool(images, plain, mixed, ref, torch.nn.Identity(), 5, mean, stdev, 0.01, 1e-3)
    raw = torch.cat([extract_patches(i, s, span).feats[:2] for i, s in zip(images, plain)])
    assert rows.shape == (18, 75) and torch.allclose(rows[torch.tensor([0, 1, 6, 7, 12, 13])], raw, atol=1e-5)
    z = (rows[3::6] - mean) / stdev
    assert 0.3 < float(z.std()) < 3 and abs(float(z.mean())) < 1
    print(f"view_pool: RAW identico ao patch cru; DoG em z-score proprio, media {float(z.mean()):+.3f}, desvio {float(z.std()):.3f}")
