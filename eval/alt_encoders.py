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
"""alt_encoders.py — P6: encoders alternativos sem backprop, na mesma leitura e no mesmo SVM do SPiFiL.

Um job = (dataset, split, tamanho), `source: alt_encoders`; celula = (encoder, config, sorteio). Imagens
pelo `load_images` de `eval/l1_fast.py` (cache, `no_crop`); leitura GeM ou media por canal + `pw_whiten`;
SVM e metricas pelo `score_full` de `eval/spf_grade.py`. So treino e validacao entram.

- `scattering` (regime A, sem dado): Bruna e Mallat (2013), Morlet como no kymatio, ordens 0, 1 e 2,
  modulo e passa-baixa, em FFT. Um job por J; o sorteio nao muda nada, roda so o primeiro.
- `softhebb` (regime C): Journe et al. (2023). Por camada, BN sem afim, conv sem vies, softmax entre canais
  com anti-Hebb nos perdedores e dw = y (x - u w), passo normalizado por neuronio e lr de cada neuronio
  lr * |‖w‖ - 1|^0.5. A correlacao y.x sai de `conv2d_weight`: e a regra local, nenhuma perda e
  derivada. `steps` atualizacoes por camada, lotes sorteados das imagens de treino, sem rotulo; camadas
  anteriores congeladas. Ativacao triangulo (p = 1). lr e temperatura sairam da convergencia (norma,
  redundancia, vencedores) em imagens de treino de Eggs_subgroup, nunca de kappa.
- `pcanet` (regime A): Chan et al. (2015), 2 estagios de filtros PCA de patches de treino (media de cada
  banda removida), hash binario e histograma global: L1 * 2^L2 dimensoes, sem blocos por causa do teto
  de 2000-d.
O aleatorio (regime D) nao roda aqui: e o `xavier`/`random_z` de `svm_disc_md_ms_ctl{,2}`.

`hebb_init`: o mesmo SoftHebb partindo de `random` (kaiming, igual a `softhebb`) ou de um banco SPiFiL de
`eval/spf_grade.py` (`fps`, `crus@geodesic`), montado camada a camada nos mapas da propria rede, nas fontes
do sorteio (`amostra.fontes`) e com norma inicial raiz de 2. Cada celula roda com 0 passos (lr 0: so as
estatisticas do BN, nos mesmos lotes) e com cada duracao de `steps`; `restricted` repete esses inits com cada
filtro preso a bola de raio `radius` x |w0| em torno do inicial (projecao apos cada passo). Por camada saem o
cosseno de cada filtro com o inicial, a distancia de Procrustes, o vizinho mais proximo, o posto efetivo, a
correlacao das ativacoes (`flim/kernel_redundancy.py`) em imagens de treino e a frequencia de vitoria, ao
lado do kappa. Kernels (inicial, final, vitorias) em `<csv>/hebb/<job>/`; figura por job em `<csv>/figs/`.
`homeo` (Fila 2, desligada por padrao), por variante nomeada, so na competicao (softmax e vencedor), nunca na
saida da rede: E04 `gain`/`cap`: vies -gain*log(W*f) da frequencia de vitoria f (media movel `decay`), cortado
em +-cap, que puxa sem igualar; E05 `window`/`patience`/`boost`: kernel com zero vitorias em `patience`
janelas seguidas ganha +boost por janela ate vencer numa janela, e ai o vies volta a 0. Raro que vence em toda
janela nunca e tocado. Celulas `<init>_w<largura>_h<variante>` sobre `homeo_inits`.
"""
from __future__ import annotations

import csv
import json
import math
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from spifil.config import LayerSpec

from core.constants import DATASET_NUM_CLASSES
from eval.l1_fast import load_images
from eval import spf_grade
from eval.l1_sem_rotulo import patch_pool, train_pool
from eval.spf_grade import IMPURITY, MAX_DIM, METHODS, STDEV_FLOOR, Ctx, score_full, stored_sources
from eval.svm import DEVICE, cli_kwargs
from flim.kernel_redundancy import activation_sample, effective_rank

PIXELS = 2**25       # pixels x canais por lote
FIELDS = ["dataset", "split", "percentage", "image_size", "encoder", "regime", "config", "seed", "layer",
          "readout", "feat_dim", "dim", "params", "flops", "local_updates", "learn_s", "extract_s", "ms_per_img", "gpu_mib",
          "rss_mib", "n_train", "n_val", "val_kappa", "val_acc", "val_f1", "val_accuracy", "imp_recall",
          "par_recall", "imp_kappa", "par_kappa", "fit_status", "fit_s", "diag", "status", "error"]
HEBB = ["dataset", "split", "percentage", "image_size", "encoder", "regime", "config", "init", "width", "seed",
        "steps", "layer", "readout", "feat_dim", "dim", "cos_init_mean", "cos_init_median", "cos_init_frac90",
        "radius", "procrustes", "cos_nearest_mean", "erank", "corr_mean", "corr_nearest_mean", "norm_mean",
        "win_eff", "win_rare", "win_never", "local_updates",
        "learn_s", "extract_s", "gpu_mib", "n_train", "n_val", "val_kappa", "val_acc", "val_f1", "val_accuracy",
        "fit_status", "fit_s", "status", "error"]
PROBE_IMAGES = 16    # imagens de treino da correlacao das ativacoes
POOL = {"avg_pw": lambda z: z.mean((2, 3)),
        "gem_pw": lambda z: z.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1 / 3)}  # o "gem" de `eval/svm.py`


# ─── Scattering ───────────────────────────────────────────────────────────────

def gabor(m: int, sigma: float, theta: float, xi: float, slant: float = 1.0) -> np.ndarray:
    """Gabor periodizada numa grade m x m, centrada em 0 (kymatio `gabor_2d`)."""
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    curv = rot @ np.diag([1, slant**2]) @ rot.T / (2 * sigma**2)
    out = np.zeros((m, m), np.complex128)
    for ex in range(-2, 3):
        for ey in range(-2, 3):
            xx, yy = np.mgrid[ex * m:(ex + 1) * m, ey * m:(ey + 1) * m]
            out += np.exp(-(curv[0, 0] * xx**2 + 2 * curv[0, 1] * xx * yy + curv[1, 1] * yy**2)
                          + 1j * xi * (xx * np.cos(theta) + yy * np.sin(theta)))
    return out * slant / (2 * np.pi * sigma**2)


def morlet_hat(m: int, sigma: float, theta: float, xi: float, slant: float) -> np.ndarray:
    wave, env = gabor(m, sigma, theta, xi, slant), gabor(m, sigma, theta, 0, slant)
    return np.real(np.fft.fft2(wave - wave.sum() / env.sum() * env))


def periodize(f: torch.Tensor, res: int) -> torch.Tensor:
    """O filtro de Fourier na grade 2^res vezes menor (kymatio `periodize_filter_fft`)."""
    k, m = 2**res, f.shape[-1]
    n = m // k
    f = f.clone()
    f[n // 2:n // 2 + m - n] = 0
    f[:, n // 2:n // 2 + m - n] = 0
    return f.reshape(k, n, k, n).sum((0, 2))


def sub(x: torch.Tensor, k: int) -> torch.Tensor:
    """Subamostra por k no espaco, feito no espectro."""
    m = x.shape[-1] // k
    return x.reshape(*x.shape[:-2], k, m, k, m).mean((-4, -2))


class Scattering(nn.Module):
    """S0, S1 e S2 de cada canal: C (1 + J L + L^2 J (J - 1) / 2) mapas a N / 2^J."""

    def __init__(self, size: int, J: int, L: int = 8) -> None:
        super().__init__()
        self.J, self.L, self.size = J, L, size
        self.m = -(-(size + 2**(J + 1)) // 2**J) * 2**J
        self.flops = 0
        psi = [[morlet_hat(self.m, 0.8 * 2**j, (L - L // 2 - 1 - t) * np.pi / L, 3 / 4 * np.pi / 2**j, 4 / L)
                for t in range(L)] for j in range(J)]
        phi = torch.tensor(np.real(np.fft.fft2(gabor(self.m, 0.8 * 2**(J - 1), 0, 0))), dtype=torch.float32)
        for j in range(J):
            bank = torch.tensor(np.stack(psi[j]), dtype=torch.float32)
            for res in range(j + 1):
                self.register_buffer(f"psi{j}_{res}", torch.stack([periodize(b, res) for b in bank]))
            self.register_buffer(f"phi{j}", periodize(phi, j))

    def _fft(self, z: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        self.flops += 5 * z.numel() * math.log2(z.shape[-1] ** 2)
        return torch.fft.ifft2(z) if inverse else torch.fft.fft2(z)

    def _low(self, zh: torch.Tensor, res: int) -> torch.Tensor:
        self.flops += 2 * zh.numel()
        return self._fft(sub(zh * getattr(self, f"phi{res}"), 2**(self.J - res)), True).real

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Em x ja m x m, periodico: (B, C, K, m / 2^J, m / 2^J)."""
        J, L = self.J, self.L
        xh = self._fft(x)
        out = [self._low(xh, 0)[:, :, None]]
        for j1 in range(J):
            u1 = self._fft(sub(xh[:, :, None] * getattr(self, f"psi{j1}_0"), 2**j1), True).abs()
            u1h = self._fft(u1)
            self.flops += 5 * u1.numel()
            out.append(self._low(u1h, j1))
            for j2 in range(j1 + 1, J):
                u2 = self._fft(sub(u1h[:, :, :, None] * getattr(self, f"psi{j2}_{j1}"), 2**(j2 - j1)), True).abs()
                self.flops += 7 * u2.numel()
                out.append(self._low(self._fft(u2), j2).flatten(2, 3))
        return torch.cat(out, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.m - self.size) // 2
        s = self.transform(F.pad(x, (pad, self.m - self.size - pad) * 2, mode="reflect"))
        lo, hi = pad // 2**self.J, -(-(pad + self.size) // 2**self.J)
        return s[..., lo:hi, lo:hi].flatten(1, 2)


# ─── SoftHebb ─────────────────────────────────────────────────────────────────

class HebbBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, pool: int, power: float = 1.0) -> None:
        super().__init__()
        self.power = power
        self.bn = nn.BatchNorm2d(cin, affine=False, momentum=None)
        self.conv = nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.pool = nn.MaxPool2d(3, stride=pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.conv(self.bn(x))
        t = F.relu(u - u.mean(1, keepdim=True))
        return self.pool(t if self.power == 1 else t ** self.power)


@torch.no_grad()
def hebb_step(block: HebbBlock, h: torch.Tensor, lr: float, beta: float, bias: torch.Tensor | None = None,
              norm: str = "neuron") -> torch.Tensor:
    """Uma atualizacao SoftHebb; devolve quantas vezes cada neuronio venceu. `bias` so entra na competicao.
    `norm`: Δw dividido pelo max de cada neuronio ("neuron") ou do tensor inteiro ("tensor", como a referencia)."""
    z = block.bn(h)
    w = block.conv.weight
    u = F.conv2d(z, w, padding=w.shape[-1] // 2)
    race = u if bias is None else u + bias.view(1, -1, 1, 1)
    win = race.argmax(1, keepdim=True)
    y = -torch.softmax(beta * race, 1)
    y.scatter_(1, win, -y.gather(1, win))
    delta = torch.nn.grad.conv2d_weight(z, w.shape, y, padding=w.shape[-1] // 2) - (y * u).sum((0, 2, 3))[:, None, None, None] * w
    delta /= (delta.flatten(1).abs().amax(1).clamp_min(1e-30)[:, None, None, None] if norm == "neuron"
              else delta.abs().amax().clamp_min(1e-30))
    w += (lr * ((w.flatten(1).norm(dim=1) - 1).abs() + 1e-10).sqrt())[:, None, None, None] * delta
    return torch.bincount(win.flatten(), minlength=len(w))


@torch.no_grad()
def learn_softhebb(spec: dict, width: int, x: torch.Tensor, to_float, seed: int, init=None) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pick = torch.randperm(len(x), generator=gen)[:spec["images"]]
    blocks, diag, updates, cin = [], [], 0, x.shape[1]
    layer = lambda key, i, default=None: (lambda v: v[min(i, len(v) - 1)] if isinstance(v, list) else v)(spec.get(key, default))
    for i in range(spec["depth"]):
        block = HebbBlock(cin, width, spec["kernel"], spec["pool"], layer("triangle_power", i, 1.0)).to(DEVICE).train()
        if init:
            block.conv.weight.copy_(init(blocks, width))
        if spec.get("init_radius"):
            w = block.conv.weight
            w *= spec["init_radius"] / w.flatten(1).norm(dim=1).clamp_min(1e-12)[:, None, None, None]
        block.start = block.conv.weight.clone()
        wins, homeo, bias = 0, spec.get("homeo"), None
        if homeo:
            ema, recent, idle, boost = (torch.full((width,), 1 / width, device=DEVICE), torch.zeros(width, device=DEVICE),
                                        torch.zeros(width, device=DEVICE), torch.zeros(width, device=DEVICE))
            bias, boosted = torch.zeros(width, device=DEVICE), torch.zeros(width, dtype=torch.bool, device=DEVICE)
        for step in range(spec["steps"]):
            h = to_float(x[pick[torch.randint(len(pick), (spec["batch"],), generator=gen)]].to(DEVICE))
            for b in blocks:
                h = b(h)
            count = hebb_step(block, h, layer("lr", i), layer("inv_temp", i), bias, spec.get("delta_norm", "neuron"))
            wins = wins + count
            updates += 1
            if spec.get("radius"):
                d = block.conv.weight - block.start
                cap = spec["radius"] * block.start.flatten(1).norm(dim=1) / d.flatten(1).norm(dim=1).clamp_min(1e-12)
                block.conv.weight.copy_(block.start + d * cap.clamp(max=1)[:, None, None, None])
            if homeo:
                ema = homeo.get("decay", 0.99) * ema + (1 - homeo.get("decay", 0.99)) * count / count.sum()
                recent += count
                if homeo.get("window") and (step + 1) % homeo["window"] == 0:
                    idle = torch.where(recent == 0, idle + 1, 0)
                    boost = torch.where(idle >= homeo["patience"], boost + homeo["boost"], 0)
                    boosted |= boost > 0
                    recent.zero_()
                bias = boost + (-homeo["gain"] * (ema * width).log()).clamp(-homeo["cap"], homeo["cap"]) if homeo.get("gain") else boost
        block.boosted = int(boosted.sum()) if homeo else 0
        block.wins = wins
        block.eval()
        blocks.append(block)
        norm = block.conv.weight.flatten(1).norm(dim=1)
        unit = F.normalize(block.conv.weight.flatten(1), dim=1)
        share = wins.float() / wins.sum()
        diag.append({"norm_mean": round(norm.mean().item(), 3), "norm_max": round(norm.max().item(), 3),
                     "cos_max": round((unit @ unit.T).fill_diagonal_(0).abs().amax(1).mean().item(), 3),
                     "winners": round((wins > 0).float().mean().item(), 3),
                     "eff_winners": round(float(torch.exp(-(share * share.clamp_min(1e-12).log()).sum())), 1)})
        cin = width
    return blocks, {"epochs": round(spec["steps"] * spec["batch"] / len(pick), 2), "layers": diag}, updates


# ─── PCANet ───────────────────────────────────────────────────────────────────

def pca_rows(rows: torch.Tensor, bands: int, n: int) -> tuple:
    """Os n autovetores maiores dos patches com a media de cada banda removida."""
    rows = rows.double().view(len(rows), bands, -1)
    rows = (rows - rows.mean(2, keepdim=True)).flatten(1)
    values, vectors = torch.linalg.eigh(rows.T @ rows / len(rows))
    return vectors[:, -n:].flip(1).T.float(), float(values[-n:].sum() / values.sum())


def sample_patches(maps: torch.Tensor, kernel: int, per: int, gen: torch.Generator) -> torch.Tensor:
    cols = F.unfold(maps, kernel, padding=kernel // 2, stride=kernel // 2 + 1)
    pick = torch.randint(cols.shape[-1], (len(maps), per), generator=gen).to(cols.device)
    return cols.gather(2, pick[:, None].expand(-1, cols.shape[1], -1)).transpose(1, 2).flatten(0, 1).cpu()


class PCANet(nn.Module):
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, l1, l2 = self.w1.shape[-1], len(self.w1), len(self.w2)
        o1 = F.conv2d(x, self.w1, padding=k // 2)
        o2 = F.conv2d(o1.flatten(0, 1)[:, None], self.w2, padding=k // 2)
        code = ((o2 > 0).long() << torch.arange(l2, device=x.device)[:, None, None]).sum(1)
        code = code.view(len(x), l1, -1) + (torch.arange(len(x) * l1, device=x.device) << l2).view(len(x), l1, 1)
        hist = torch.bincount(code.flatten(), minlength=len(x) * l1 << l2).float() / code.shape[-1]
        return hist.view(len(x), l1 << l2, 1, 1)


@torch.no_grad()
def learn_pcanet(spec: dict, x: torch.Tensor, to_float, seed: int) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(x), generator=gen)[:spec["images"]]
    k, (l1, l2), per = spec["kernel"], spec["filters"], spec["patches_per_image"]
    step = max(1, PIXELS // (l1 * x.shape[-1] ** 2))
    batches = [to_float(x[pick[i:i + step]].to(DEVICE)) for i in range(0, len(pick), step)]
    w1, ev1 = pca_rows(torch.cat([sample_patches(b, k, per, gen) for b in batches]), x.shape[1], l1)
    w1 = w1.view(l1, x.shape[1], k, k).to(DEVICE)
    rows = [sample_patches(F.conv2d(b, w1, padding=k // 2).flatten(0, 1)[:, None], k, per, gen) for b in batches]
    w2, ev2 = pca_rows(torch.cat(rows), 1, l2)
    return [PCANet(w1, w2.view(l2, 1, k, k).to(DEVICE))], {"explained": [round(ev1, 3), round(ev2, 3)]}


# ─── Celulas ──────────────────────────────────────────────────────────────────

def cells(spec: dict) -> list:
    """(encoder, config, sorteio), barato primeiro."""
    out = [("scattering", f"J{j}_L{spec['scattering']['L']}", spec["draws"][0])
           for j in spec.get("scattering", {}).get("J", [])]
    for seed in spec["draws"]:
        if "pcanet" in spec:
            p = spec["pcanet"]
            out.append(("pcanet", f"L{p['filters'][0]}-{p['filters'][1]}_k{p['kernel']}", seed))
        if "softhebb" in spec:
            h = spec["softhebb"]
            out += [("softhebb", f"w{w}x{h['depth']}_k{h['kernel']}", seed) for w in h["widths"]]
        if "hebb_init" in spec:
            h = spec["hebb_init"]
            out += [("hebb_init", f"{arm}_w{w}", seed) for w in h["widths"] for arm in h["inits"]]
            out += [("hebb_init", f"{arm}_w{w}_r{h['radius']}", seed) for w in h["widths"] for arm in h.get("restricted", [])]
            out += [("hebb_init", f"{arm}_w{w}_c{c}", seed) for w in h["widths"] for arm in h["inits"] for c in h.get("calib", [])]
            out += [("hebb_init", f"{arm}_w{w}_h{name}", seed) for w in h["widths"] for arm in h.get("homeo_inits", [])
                    for name in h.get("homeo", {})]
    return out


def build(job: dict, encoder: str, config: str, seed: int, train: torch.Tensor, to_float) -> tuple:
    """Os blocos do encoder, o diagnostico, as atualizacoes locais e o tempo de aprendizado."""
    spec, start = job["amostra"][encoder], time.perf_counter()
    if encoder == "scattering":
        blocks, diag, updates = [Scattering(job["image_size"], int(config[1:].split("_")[0]), spec["L"]).to(DEVICE)], {}, 0
    elif encoder == "pcanet":
        (blocks, diag), updates = learn_pcanet(spec, train, to_float, seed), 0
    else:
        blocks, diag, updates = learn_softhebb(spec, int(config[1:].split("x")[0]), train, to_float, seed)
    return blocks, diag, updates, time.perf_counter() - start


@torch.no_grad()
def embed(blocks: list, x: torch.Tensor, readout: str) -> list:
    """Um vetor por camada; `gem_tta_pw` e a media nas 8 rotacoes/flips, como `gem_tta`."""
    if readout == "gem_tta_pw":
        views = [embed(blocks, torch.rot90(v, k, (2, 3)), "gem_pw") for v in (x, x.flip(3)) for k in range(4)]
        return [torch.stack(v).mean(0) for v in zip(*views)]
    out = []
    for b in blocks:
        x = b(x)
        out.append(POOL[readout](x).cpu())
    return out


def extract(blocks: list, x: torch.Tensor, to_float, readouts: list, width: int) -> dict:
    step = max(1, PIXELS // (width * x.shape[-1] ** 2))
    parts = {r: [] for r in readouts}
    for i in range(0, len(x), step):
        batch = to_float(x[i:i + step].to(DEVICE))
        for r in readouts:
            parts[r].append(embed(blocks, batch, r))
    return {r: [torch.cat(layer).double().numpy() for layer in zip(*p)] for r, p in parts.items()}


def cost(blocks: list, size: int, channels: int) -> tuple:
    """Parametros ajustados e FLOPs de uma imagem: conv pelo `FlopCounterMode`, FFT pela conta do Scattering."""
    probe = torch.rand(1, channels, size, size, device=DEVICE)
    for b in blocks:
        if isinstance(b, Scattering):
            b.flops = 0
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        for b in blocks:
            probe = b(probe)
    fft = sum(b.flops for b in blocks if isinstance(b, Scattering))
    params = sum(m.weight.numel() for b in blocks for m in b.modules() if isinstance(m, nn.Conv2d))
    params += sum(b.w1.numel() + b.w2.numel() for b in blocks if isinstance(b, PCANet))
    return params, int(counter.get_total_flops() + fft)


def run_cell(job: dict, images: dict, to_float, cell: tuple, base: dict, write) -> None:
    encoder, config, seed = cell
    spec = job["amostra"][encoder]
    row = base | {"encoder": encoder, "config": config, "seed": seed,
                  "regime": {"softhebb": "C"}.get(encoder, "A")}
    torch.cuda.reset_peak_memory_stats()
    blocks, diag, updates, learn_s = build(job, encoder, config, seed, images["train"][0], to_float)
    params, flops = cost(blocks, job["image_size"], images["train"][0].shape[1])
    width = (3 * spec["L"] ** 2 if encoder == "scattering" else math.prod(spec["filters"]) if encoder == "pcanet"
             else int(config[1:].split("x")[0]))
    start = time.perf_counter()
    feats = {name: extract(blocks, images[name][0], to_float, spec["readouts"], width) for name in ("train", "validation")}
    extract_s = time.perf_counter() - start
    row |= {"params": params, "flops": flops, "local_updates": updates, "learn_s": round(learn_s, 2),
            "extract_s": round(extract_s, 2), "ms_per_img": round(1000 * extract_s / (row["n_train"] + row["n_val"]), 3),
            "gpu_mib": torch.cuda.max_memory_allocated() >> 20,
            "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10, "diag": json.dumps(diag)}
    del blocks
    torch.cuda.empty_cache()
    score_reads(job, images, feats, spec["readouts"], row, f"alt {job['stem']} {encoder} {config} s{seed}", write)


def score_reads(job: dict, images: dict, feats: dict, readouts: list, row: dict, tag: str, write,
                layers: list | None = None) -> None:
    """Um SVM por camada (`L`) e um nas ultimas camadas concatenadas abaixo de MAX_DIM (`cat`)."""
    y = images["train"][1], images["validation"][1]
    classes, impurity = DATASET_NUM_CLASSES[job["dataset_name"]], IMPURITY.get(job["dataset_name"])
    for readout in readouts:
        train, val = feats["train"][readout], feats["validation"][readout]
        reads = [(n + 1, "L", train[n], val[n]) for n in range(len(train))]
        fit = [n for n in range(len(train)) if sum(train[m].shape[1] for m in range(n, len(train))) < MAX_DIM]
        if len(train) > 1 and fit and fit[0] < len(train) - 1:
            reads.append((len(train), "cat", np.concatenate(train[fit[0]:], 1), np.concatenate(val[fit[0]:], 1)))
        for layer, kind, a, b in reads:
            out = row | {"layer": layer, "readout": readout if kind == "L" else f"cat_{readout}", "feat_dim": a.shape[1]}
            if layers and kind == "L":
                out |= layers[layer - 1]
            try:
                assert a.shape[1] < MAX_DIM, f"leitura de {a.shape[1]}-d passa de {MAX_DIM}"
                write(out | score_full(a, y[0], b, y[1], classes, tag, impurity))
            except Exception as exc:  # noqa: BLE001
                write(out | {"dim": a.shape[1], "status": "error", "error": repr(exc)[:500]})


def spifil_init(lab: list, srcs: dict, arm: str, seed: int, kernel: int):
    """O braco `metodo@pontos` de `spf_grade.chain` nos mapas da rede ate aqui: direcoes no z-score, norma raiz de 2."""
    method, _, point = arm.partition("@")
    src, ref_seeds = srcs[point or "medoid"], srcs["medoid"].seeds

    def init(blocks: list, width: int) -> torch.Tensor:
        encoder, layer = nn.Sequential(*blocks).eval(), LayerSpec(kernel_size=kernel, out_channels=width)
        ref = patch_pool(lab, ref_seeds, encoder, layer).feats
        feats = ref if src.seeds is ref_seeds else patch_pool(lab, src.seeds, encoder, layer).feats
        u = METHODS[method](Ctx(feats, ref.mean(0), ref.std(0, unbiased=False) + STDEV_FLOOR, src, seed + len(blocks)), width)[0]
        assert len(u) == width, f"{arm}: {len(u)} de {width} filtros"
        return (F.normalize(torch.as_tensor(u, dtype=torch.float32), dim=1) * math.sqrt(2)).view(width, -1, kernel, kernel)
    return init


def procrustes(a: torch.Tensor, b: torch.Tensor) -> float:
    """Disparidade de `scipy.spatial.procrustes` (filtros como pontos), via QR: o SVD fica n x n."""
    a, b = a - a.mean(0), b - b.mean(0)
    r1, r2 = torch.linalg.qr((a / a.norm()).T.double())[1], torch.linalg.qr((b / b.norm()).T.double())[1]
    return float(1 - torch.linalg.svdvals(r1 @ r2.T).sum() ** 2)


@torch.no_grad()
def layer_measures(blocks: list, x: torch.Tensor, to_float) -> list:
    """Por camada: o banco contra o inicial, a diversidade dos pesos e a das ativacoes (triangulo, antes do pool)."""
    h, out = to_float(x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:PROBE_IMAGES]].to(DEVICE)), []
    for block in blocks:
        w0, w1 = (F.normalize(w.flatten(1), dim=1) for w in (block.start, block.conv.weight))
        own, share = (w0 * w1).sum(1), block.wins.float() / block.wins.sum()
        sample = activation_sample(list(block.bn(h)), LayerSpec(kernel_size=block.conv.kernel_size[0], out_channels=len(w1)))
        act = sample @ block.conv.weight.flatten(1).T
        corr = torch.corrcoef(F.relu(act - act.mean(1, keepdim=True)).T).nan_to_num()
        off = ~torch.eye(len(w1), dtype=torch.bool, device=w1.device)
        out.append({"cos_init_mean": float(own.mean()), "cos_init_median": float(own.median()),
                    "cos_init_frac90": float((own > 0.9).float().mean()), "procrustes": procrustes(w0, w1),
                    "cos_nearest_mean": float((w1 @ w1.T).masked_fill(~off, -1).amax(1).mean()),
                    "erank": effective_rank(w1), "corr_mean": float(corr[off].mean()),
                    "corr_nearest_mean": float(corr.masked_fill(~off, -1).amax(1).mean()),
                    "norm_mean": float(block.conv.weight.flatten(1).norm(dim=1).mean()),
                    "win_eff": float(torch.exp(-(share * share.clamp_min(1e-12).log()).sum())),
                    "win_rare": int((share < 0.1 / len(share)).sum()), "win_never": int((block.wins == 0).sum()),
                    "boosted": getattr(block, "boosted", 0)})
        h = block(h)
    return out


def run_hebb(job: dict, images: dict, to_float, cell: tuple, base: dict, write, banks: dict) -> None:
    """Uma (init, largura, sorteio): a rede com 0 passos e com `steps`, mesmos lotes, mesmo SVM."""
    _, config, seed = cell
    spec = job["amostra"]["hebb_init"]
    arm, width = config.rsplit("_w", 1)
    width, _, tag = width.partition("_")
    radius = float(tag[1:]) if tag.startswith("r") else None
    width, homeo, init = int(width), spec["homeo"][tag[1:]] if tag.startswith("h") else None, None
    out = Path(job["results_csv"]).parent / "hebb" / job["stem"]
    out.mkdir(parents=True, exist_ok=True)
    if arm != "random":
        if seed not in banks:
            sys.modules["__main__"].Source = spf_grade.Source  # as fontes foram salvas por `python -m eval.spf_grade`
            job["draw"] = seed
            pool = train_pool(job)
            banks.clear()
            banks[seed] = stored_sources(job, random.Random(seed).sample(pool, min(spec["bank_images"], len(pool))))[:2]
        init = spifil_init(*banks[seed], arm, seed, spec["kernel"])
    calib = int(tag[1:]) if tag.startswith("c") else spec.get("calib_steps", durations(spec)[-1])
    for steps in [0] * (not tag or tag.startswith("c")) + durations(spec) * (not tag.startswith("c")):
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        blocks, _, updates = learn_softhebb(spec | {"steps": steps or calib, "lr": spec["lr"] if steps else 0.0,
                                                    "radius": radius, "homeo": homeo}, width, images["train"][0], to_float, seed, init)
        learn_s, layers = time.perf_counter() - start, layer_measures(blocks, images["train"][0], to_float)
        for n, b in enumerate(blocks, 1):
            stem = out / f"{config}_s{seed}_p{steps}_L{n}"
            np.save(f"{stem}.npy", b.conv.weight.detach().cpu().numpy())
            np.save(f"{stem}_wins.npy", b.wins.cpu().numpy())
            if steps:
                np.save(f"{stem}_init.npy", b.start.cpu().numpy())
        start = time.perf_counter()
        feats = {name: extract(blocks, images[name][0], to_float, spec["readouts"], width) for name in ("train", "validation")}
        row = base | {"encoder": "hebb_init", "regime": "C" if steps else "A" if arm != "random" else "D",
                      "config": config, "init": arm, "width": width, "seed": seed, "steps": steps, "radius": radius,
                      "local_updates": updates if steps else 0, "learn_s": round(learn_s, 2),
                      "extract_s": round(time.perf_counter() - start, 2), "gpu_mib": torch.cuda.max_memory_allocated() >> 20}
        del blocks
        torch.cuda.empty_cache()
        score_reads(job, images, feats, spec["readouts"], row, f"hebb {job['stem']} {config} s{seed} p{steps}", write, layers)


def durations(spec: dict) -> list:
    return spec["steps"] if isinstance(spec["steps"], list) else [spec["steps"]]


def figure(job: dict, path: Path) -> None:
    """Camada 1 antes e depois por init (largura menor, 1o sorteio) e o delta kappa por init, largura e leitura."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    spec = job["amostra"]["hebb_init"]
    if not spec["inits"]:
        return  # so variantes `homeo`: o par sem homeostase e a figura estao na Fila 1
    with open(path, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["status"] == "ok" and r["layer"] == str(spec["depth"])]
    kappa, last = {}, durations(spec)[-1]
    for r in rows:
        if r["readout"] == spec["readouts"][0]:
            kappa.setdefault((r["config"], int(r["steps"])), []).append(float(r["val_kappa"]))
    inits, width, seed = spec["inits"], min(spec["widths"]), job["amostra"]["draws"][0]
    fig = plt.figure(figsize=(13, 2.6 * len(inits)))
    grid = fig.add_gridspec(len(inits), 3, width_ratios=[1, 1, 1.6])

    def mosaic(w: np.ndarray) -> np.ndarray:
        w = w - w.min((1, 2, 3), keepdims=True)
        w = (w / w.max((1, 2, 3), keepdims=True).clip(1e-12)).transpose(0, 2, 3, 1)
        side, k = math.ceil(math.sqrt(len(w))), w.shape[1]
        tile = np.ones((side * (k + 1), side * (k + 1), 3))
        for i, kernel in enumerate(w):
            tile[i // side * (k + 1):i // side * (k + 1) + k, i % side * (k + 1):i % side * (k + 1) + k] = kernel
        return tile

    for i, arm in enumerate(inits):
        for j, steps in enumerate((0, last)):
            ax = fig.add_subplot(grid[i, j])
            file = path.parent / "hebb" / job["stem"] / f"{arm}_w{width}_s{seed}_p{steps}_L1.npy"
            if file.exists():
                ax.imshow(mosaic(np.load(file)), interpolation="nearest")
            ax.set_title(f"{arm}, L1, {'antes' if j == 0 else f'{steps} passos'}", fontsize=9)
            ax.axis("off")
    ax = fig.add_subplot(grid[:, 2])
    labels, deltas = [], []
    for config, steps in sorted(k for k in kappa if k[1]):
        before = kappa.get((config.rsplit("_", 1)[0] if config.count("_w") and "_" in config.rsplit("_w", 1)[1] else config, 0))
        if before:
            labels.append(f"{config} p{steps}")
            deltas.append(np.mean(kappa[(config, steps)]) - np.mean(before))
    ax.barh(labels, deltas, color=["#3b7dd8" if d >= 0 else "#d8573b" for d in deltas])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(f"delta kappa de validacao (L{spec['depth']}, {spec['readouts'][0]}): passos menos 0")
    ax.tick_params(labelsize=8)
    fig.suptitle(f"{job['stem']}: SoftHebb por inicializacao", fontsize=11)
    fig.tight_layout()
    out = path.parent / "figs" / path.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=130)
    plt.close(fig)


def expected(job: dict, cell: tuple) -> int:
    spec = job["amostra"][cell[0]]
    if cell[0] == "hebb_init":
        tag = cell[1].rsplit("_w", 1)[1].partition("_")[2]
        return (1 if tag.startswith("c") else len(durations(spec)) + (not tag)) * len(spec["readouts"]) * (2 * spec["depth"] - 1)
    return len(spec["readouts"]) * (2 * spec["depth"] - 1 if cell[0] == "softhebb" else 1)


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    torch.set_num_threads(2)
    with open(config, encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    path = Path(job["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if path.exists():
        with open(path, newline="") as handle:
            for r in csv.DictReader(handle):
                key = (r["encoder"], r["config"], int(r["seed"]))
                seen[key] = seen.get(key, 0) + (r["status"] == "ok")
    todo = [c for c in cells(job["amostra"]) if seen.get(c, 0) < expected(job, c)]
    hebb = "hebb_init" in job["amostra"]
    if not todo:
        if hebb:
            figure(job, path)
        return
    images, to_float = load_images(job)
    base = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"],
            "image_size": job["image_size"], "n_train": len(images["train"][1]),
            "n_val": len(images["validation"][1]), "status": "ok"}
    with open(path, "a", newline="") as handle:
        fields = HEBB + ["boosted"] * bool(job["amostra"]["hebb_init"].get("homeo")) if hebb else FIELDS
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def write(row: dict) -> None:
            writer.writerow(row)
            handle.flush()

        banks = {}
        for cell in todo:
            try:
                if cell[0] == "hebb_init":
                    run_hebb(job, images, to_float, cell, base, write, banks)
                else:
                    run_cell(job, images, to_float, cell, base, write)
            except Exception as exc:  # noqa: BLE001
                write(base | {"encoder": cell[0], "config": cell[1], "seed": cell[2], "status": "error",
                              "error": repr(exc)[:500]})
            torch.cuda.empty_cache()
    if hebb:
        figure(job, path)


def check() -> None:
    """Scattering numa textura sintetica (ruido suavizado + mancha): energia, Littlewood-Paley e translacao."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415
    torch.set_num_threads(2)
    torch.manual_seed(0)
    size, J = 128, 4
    net = Scattering(size, J)
    m = net.m
    yy, xx = np.mgrid[:m, :m]
    x = gaussian_filter(np.random.default_rng(0).standard_normal((m, m)), 2, mode="wrap")
    x = torch.tensor(x + 0.5 * np.exp(-((xx - m / 2) ** 2 + (yy - m / 3) ** 2) / 200))[None, None].float()
    s = net.transform(x)
    energy = float((s**2).sum() * 4**J / (x**2).sum())
    flips = [getattr(net, f"psi{j}_0") for j in range(J)]
    lp = net.phi0**2 + sum((f**2 + f.flip(1, 2).roll((1, 1), (1, 2)) ** 2).sum(0) / 2 for f in flips)
    exact = float((net.transform(torch.roll(x, (2**J,) * 2, (2, 3))) - torch.roll(s, (1, 1), (3, 4))).norm() / s.norm())
    print(f"J={J}: {s.shape[2]} caminhos, energia |S|^2/|x|^2 = {energy:.3f}; "
          f"Littlewood-Paley {lp.min():.3f}..{lp.max():.3f}; covariancia exata a 2^J px: {exact:.1e}")
    for shift in (1, 2, 4):
        moved = torch.roll(x, (shift, shift), (2, 3))
        dx = float((moved - x).norm() / x.norm())
        ds = float((net.transform(moved) - s).norm() / s.norm())
        print(f"translacao {shift} px: |dx|/|x| = {dx:.3f}, |dS|/|S| = {ds:.4f}")
        if shift == 1:
            assert ds < dx / 5, (ds, dx)
    assert 0.7 < energy < 1.1 and exact < 1e-4, (energy, exact)
    out = Scattering(200, 4).to(DEVICE)(torch.rand(2, 3, 200, 200, device=DEVICE))
    assert out.shape[1] == 3 * (1 + 4 * 8 + 64 * 6) and torch.isfinite(out).all(), out.shape
    w1 = torch.linalg.qr(torch.randn(75, 8))[0].T.reshape(8, 3, 5, 5)
    hist = PCANet(w1, torch.randn(7, 1, 5, 5))(torch.rand(2, 3, 32, 32))
    assert hist.shape == (2, 8 * 128, 1, 1) and torch.allclose(hist.sum(1), torch.full((2, 1, 1), 8.0))
    block = HebbBlock(3, 16, 5, 2)
    x = torch.rand(64, 3, 32, 32)
    for _ in range(200):
        wins = hebb_step(block, x, 0.05, 4.0)
    norm = block.conv.weight.flatten(1).norm(dim=1)
    print(f"SoftHebb: |w| {norm.min():.3f}..{norm.max():.3f}, vencedores {int((wins > 0).sum())}/16")
    print("ok")


if __name__ == "__main__":
    kwargs = cli_kwargs(sys.argv[1:])
    check() if kwargs.pop("check", False) else main(**kwargs)
