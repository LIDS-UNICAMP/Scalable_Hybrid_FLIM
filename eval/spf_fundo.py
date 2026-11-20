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
"""spf_fundo.py — F1 a F4: o banco de filtros olha o objeto ou o fundo, e aguenta transformacao?

Toda funcao recebe um banco (`banco(pasta)`), entao o mesmo codigo mede SPiFiL e aleatorio nas mesmas
imagens perturbadas. `aleatorio` cria o gemeo `random_z` de um banco salvo, com a dobra de `spf_grade`.

F1 `f1_fundo`: mexe no fundo e preserva o objeto. F2 `f2_periferia`: mexe no anel externo, sem mascara.
F3 `f3_estabilidade`: translacao, rotacao, escala, iluminacao e ruido; mede keypoints, cada filtro e os
mapas, com o mapa original levado pela mesma transformacao. F4 `f4_energia`: energia dentro e fora do
objeto. F1 e F4 usam o label so aqui, para medir: nunca escolhem filtro. Sem label (flowers), nao rodam.

Duas perguntas separadas, em CSVs separados e sem nota combinada. `robustez_transformacao.csv`: a imagem
inteira muda. `robustez_fundo.csv`: so uma regiao muda, com a mesma operacao em `inteira` e `objeto` como
controle. `separacao.csv` tira do fundo a fragilidade geral: `parte_fundo` = mudanca com o fundo / mudanca
com a imagem inteira, a comparar com `area`, a fracao da imagem mexida.

Mudanca na leitura oficial (GeM + `pw_whiten`): `rel` = deslocamento / espalhamento natural das imagens
limpas, `cos` = cosseno com a limpa. `kappa` = um `fit_svm` no treino limpo, previsto na validacao
perturbada. So treino e validacao: o teste nem sobe. Nada sobrescreve: a pasta do `stem` nao pode existir.

`python -m eval.spf_fundo --config <yaml>`, com stem, job (YAML de `spf_grade`), bancos ({nome: pasta}),
aleatorio ({nome, seed, pasta}), n_train, n_val, out.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from spifil.config import LayerSpec
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode, v2

from core.constants import DATASET_NUM_CLASSES
from core.data.parasite_dataset import ParasiteDataset, image_cache_path
from core.data.transforms import LAB_BLACK, build_test
from eval.l1_fast import fit_pending
from eval.l1_sem_rotulo import patch_pool, train_pool
from eval.spf_grade import MASK_DIR, STDEV_FLOOR, control, keypoints, mask_of, pools, sources
from eval.svm import DEVICE, cli_kwargs, pw_whiten
from experiments.ray.runners.growth import _paths
from flim import spifil
from flim.arch import parse_architecture
from flim.build import build

BATCH = 8
OPS = ("ruido", "borrado", "chapado", "troca", "preto")
RINGS = (0.1, 0.2, 0.3)      # fracao de cada lado do conteudo
NOISE = 40                   # desvio do ruido de F1 e F2, em niveis de 0 a 255
BLUR = 8.0
TRANSFORMS = {"translacao": (4, 8, 16, 32), "rotacao": (5, 15, 30, 90), "escala": (0.8, 0.9, 1.1, 1.25),
              "iluminacao": (0.7, 0.85, 1.15, 1.3), "ruido": (5, 10, 20, 40)}
GEOMETRIC = ("translacao", "rotacao", "escala")
TOP = 50                     # keypoints mais fortes por imagem
RADIUS = 3                   # pixels de tolerancia na repetibilidade
BORDER = 8                   # pixels fora da conta, perto da borda do conteudo
STABLE = 0.5                 # correlacao a partir da qual um filtro conta como estavel
FLAT_VAR = 1e-8


@dataclass
class Amostra:
    """Imagens uint8 de treino e validacao; `obj` e None quando a base nao tem label."""
    dataset: str
    x: torch.Tensor
    y: np.ndarray
    v: torch.Tensor
    vy: np.ndarray
    obj: torch.Tensor | None
    content: torch.Tensor
    to_float: object
    size: int
    classes: int


def imagens(job: dict, name: str, n: int) -> tuple:
    """`n` imagens sorteadas de um conjunto, sem subir os outros: `load_images` leria o teste junto."""
    size, no_crop = job["image_size"], job["no_crop"]
    test = build_test(size, imagenet_norm=False, no_crop=no_crop)
    data = ParasiteDataset(set_name=name, split=job["split"], percentage=job["percentage"],
                           transform=v2.Compose(test.transforms[:-1]), loader="ift_lab",
                           path_dataset=job["dataset_name"],
                           cache_dir=image_cache_path(job["image_cache_dir"], job["dataset_name"], size, no_crop))
    picks = np.random.default_rng(0).permutation(len(data))[:n].tolist()
    pairs = list(DataLoader(Subset(data, picks), batch_size=BATCH, num_workers=2))
    return (torch.cat([x.as_subclass(torch.Tensor) for x, _ in pairs]), torch.cat([y for _, y in pairs]).numpy(),
            [Path(data.samples[i]).name for i in picks], test.transforms[-1])


def conteudo(x: torch.Tensor) -> torch.Tensor:
    """Tira as faixas do letterbox: linha ou coluna inteira da cor de preenchimento."""
    pad = (x == torch.tensor(LAB_BLACK, dtype=x.dtype)[:, None, None]).all(1)
    return ~(pad.all(2, keepdim=True) | pad.all(1, keepdim=True))


def amostra(job: dict, n_train: int, n_val: int) -> Amostra:
    x, y, _, to_float = imagens(job, "train", n_train)
    v, vy, names, _ = imagens(job, "validation", n_val)
    content, dataset, size = conteudo(v), job["dataset_name"], job["image_size"]
    obj = torch.stack([mask_of(dataset, n, size) for n in names]) & content if dataset in MASK_DIR else None
    return Amostra(dataset, x, y, v, vy, obj, content, to_float, size, DATASET_NUM_CLASSES[dataset])


def banco(pasta: str) -> torch.nn.Module:
    return build(str(Path(pasta) / "architecture.json"), "flim", str(pasta)).to(DEVICE).eval()


def aleatorio(job: dict, ref: str, out: str, seed: int) -> None:
    """Gemeo `random_z` de `ref`: mesmas larguras, e a dobra de `chain` nas sementes medoide do sorteio."""
    spec, ref = job["amostra"], Path(ref)
    kernel = spec["kernels"][0]
    span = 2 * (kernel // 2) + 1
    widths = [np.load(ref / f"conv{n + 1}-kernels.npy").shape[1] for n in range(len(glob.glob(str(ref / "conv*-kernels.npy"))))]
    arch_json, weights = _paths({"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]})
    arch = {k: v for k, v in parse_architecture(arch_json).items() if not k.startswith("layer")} | {"nlayers": 0}
    pool = train_pool(job)
    lab, found, *_ = sources(job, random.Random(seed).sample(pool, min(spec["images"], len(pool))))
    encoder, source, out = torch.nn.Identity(), Path(weights), Path(out)
    for n, width in enumerate(widths):
        rows = patch_pool(lab, found["medoid"].seeds, encoder, LayerSpec(kernel_size=kernel, out_channels=width)).feats
        mean, stdev = rows.mean(0), rows.std(0, unbiased=False) + STDEV_FLOOR
        u = control("random_z", SimpleNamespace(seed=seed + n, feats=rows), width, n)
        weight = F.normalize(torch.as_tensor(u, dtype=torch.float32), dim=1) / stdev
        layer = out / f"L{n + 1}"
        arch = spifil.grown_arch(arch, kernel, width, pools(spec)[n])
        spifil.write_weights(layer, source, arch, n + 1, SimpleNamespace(conv=SimpleNamespace(
            weight=weight.view(width, -1, span, span), bias=-(weight * mean).sum(1))))
        source = layer
        encoder = banco(layer)
    for path in source.iterdir():
        path.replace(out / path.name)


@torch.no_grad()
def mapas(modelo, x: torch.Tensor, to_float):
    """Lote a lote, o mapa depois de cada camada."""
    for i in range(0, len(x), BATCH):
        out, found = to_float(x[i:i + BATCH].to(DEVICE)), []
        for n in range(1, modelo.n_layers + 1):
            out = getattr(modelo, f"conv{n}")(out)
            found.append(out)
        yield i, found


def camadas(modelo, x: torch.Tensor, a: Amostra) -> list:
    """GeM de cada camada, a leitura de `_encode_pooled`."""
    parts = [[] for _ in range(modelo.n_layers)]
    for _, found in mapas(modelo, x, a.to_float):
        for part, z in zip(parts, found):
            part.append(F.adaptive_avg_pool2d(z.clamp(min=1e-6).pow(3), 1).pow(1 / 3).flatten(1).cpu())
    return [torch.cat(p).numpy() for p in parts]


def perturba(x: torch.Tensor, region: torch.Tensor, op: str, obj: torch.Tensor | None = None) -> torch.Tensor:
    """Troca so os pixels de `region`. Em `troca`, objeto da imagem vizinha vira a media do fundo dela."""
    f, area = x.float(), region[:, None].float()
    if op == "ruido":
        new = f + NOISE * torch.randn(f.shape, generator=torch.Generator().manual_seed(0))
    elif op == "borrado":
        new = v2.functional.gaussian_blur(f, 4 * int(BLUR) - 1, BLUR)
    elif op == "chapado":
        new = ((f * area).sum((2, 3), keepdim=True) / area.sum((2, 3), keepdim=True).clamp_min(1)).expand_as(f)
    elif op == "preto":
        new = torch.tensor(LAB_BLACK, dtype=f.dtype)[None, :, None, None].expand_as(f)
    else:
        new = f.roll(1, 0)
        if obj is not None:
            other = obj.roll(1, 0)[:, None]
            rest = (~other).float()
            new = torch.where(other, (new * rest).sum((2, 3), keepdim=True) / rest.sum((2, 3), keepdim=True).clamp_min(1), new)
    return torch.where(region[:, None], new.clamp(0, 255).round(), f).to(torch.uint8)


def anel(content: torch.Tensor, frac: float) -> torch.Tensor:
    """O conteudo menos a caixa central: `frac` de cada lado. Periferia, sem supor onde esta o objeto."""
    out = content.clone()
    for ring, alive in zip(out, content):
        ys, xs = alive.any(1).nonzero()[:, 0], alive.any(0).nonzero()[:, 0]
        dy, dx = int(frac * len(ys)), int(frac * len(xs))
        ring[ys[0] + dy:ys[-1] + 1 - dy, xs[0] + dx:xs[-1] + 1 - dx] = False
    return out


def warp(z: torch.Tensor, kind: str, level: float, size: int, fill=0.0) -> torch.Tensor:
    """A transformacao geometrica em qualquer grade: `level` e dado em pixels da imagem."""
    if kind not in GEOMETRIC:
        return z
    shift = level * z.shape[-1] / size if kind == "translacao" else 0
    return v2.functional.affine(z, angle=level if kind == "rotacao" else 0.0, translate=[shift, shift],
                                scale=level if kind == "escala" else 1.0, shear=[0.0, 0.0],
                                interpolation=InterpolationMode.BILINEAR, fill=fill)


def transforma(x: torch.Tensor, content: torch.Tensor, kind: str, level: float, size: int) -> torch.Tensor:
    """Imagem inteira. Iluminacao escala so a banda L; ruido e iluminacao poupam o letterbox."""
    if kind in GEOMETRIC:
        return warp(x, kind, level, size, list(map(float, LAB_BLACK)))
    f = x.float()
    if kind == "ruido":
        new = f + level * torch.randn(f.shape, generator=torch.Generator().manual_seed(0))
    else:
        new = torch.cat([f[:, :1] * level, f[:, 1:]], 1)
    return torch.where(content[:, None], new.clamp(0, 255).round(), f).to(torch.uint8)


def medir(modelo, nome: str, a: Amostra, conds: dict) -> list:
    """Uma linha por camada e condicao. `conds`: chave -> (imagens, campos da linha)."""
    train, clean = camadas(modelo, a.x, a), camadas(modelo, a.v, a)
    feats = {k: camadas(modelo, imgs, a) for k, (imgs, _) in conds.items()}
    pending, fitted, rows = [], [], []
    for n in range(modelo.n_layers):
        evals = {"limpo": clean[n]} | {k: feats[k][n] for k in conds}
        pending.append(({"camada": n + 1}, (train[n], a.y, evals, dict.fromkeys(evals, a.vy), a.classes,
                                            f"spf_fundo {a.dataset} {nome} L{n + 1}")))
    fit_pending(pending, fitted.append)
    for row in fitted:
        if "error" in row:
            raise RuntimeError(row["error"])
        n = row["camada"] - 1
        z, c, *others = pw_whiten(train[n], clean[n], *[feats[k][n] for k in conds])
        spread = np.linalg.norm(c - c.mean(0), axis=1).mean()
        for (key, (_, fields)), o in zip(conds.items(), others):
            rows.append({"dataset": a.dataset, "banco": nome, "camada": n + 1, **fields,
                         "rel": np.linalg.norm(o - c, axis=1).mean() / spread, "cos": (o * c).sum(1).mean(),
                         "kappa": row[f"{key}_kappa"], "kappa_limpo": row["limpo_kappa"],
                         "dkappa": row[f"{key}_kappa"] - row["limpo_kappa"]})
    return rows


def regioes(a: Amostra, nomes: dict) -> dict:
    """Cada operacao em cada regiao; `area` = fracao do conteudo mexida."""
    total = a.content.sum((1, 2)).float()
    out = {}
    for alvo, region in nomes.items():
        area = float((region.sum((1, 2)) / total).mean())
        extra = {} if a.obj is None else {"obj_na_regiao": float(((region & a.obj).sum((1, 2)) / region.sum((1, 2)).clamp_min(1)).mean())}
        for op in OPS:
            out[f"{op}:{alvo}"] = (perturba(a.v, region, op, a.obj), {"op": op, "alvo": alvo, "area": area} | extra)
    return out


def f1_conds(a: Amostra) -> dict:
    return regioes(a, {"fundo": a.content & ~a.obj, "objeto": a.obj, "inteira": a.content})


def f2_conds(a: Amostra) -> dict:
    return regioes(a, {f"anel{f}": anel(a.content, f) for f in RINGS} | {"inteira": a.content})


def f3_conds(a: Amostra) -> dict:
    return {f"{k}:{v}": (transforma(a.v, a.content, k, v, a.size), {"op": k, "nivel": v})
            for k, levels in TRANSFORMS.items() for v in levels}


def f1_fundo(modelo, nome: str, a: Amostra, conds: dict) -> list:
    return [r | {"teste": "F1"} for r in medir(modelo, nome, a, conds)]


def f2_periferia(modelo, nome: str, a: Amostra, conds: dict) -> list:
    return [r | {"teste": "F2"} for r in medir(modelo, nome, a, conds)]


def valido(content: torch.Tensor, kind: str, level: float, size: int, shape: tuple, margin: int) -> torch.Tensor:
    """Onde as duas imagens tem conteudo, longe da borda dele. `content` e [B, H, W]; devolve na grade `shape`."""
    grid = F.interpolate(content[:, None].float(), shape, mode="nearest")
    both = torch.minimum(grid, warp(grid, kind, level, size))
    return (1 - F.max_pool2d(1 - both, 2 * margin + 1, 1, margin))[:, 0] > 0.5


def repete(first: torch.Tensor, second: torch.Tensor, kind: str, level: float, size: int, ok: torch.Tensor, radius: int) -> tuple:
    """Pontos de `second` com par em `first` depois da transformacao, e o que daria por acaso."""
    near = F.max_pool2d(first[:, None].float(), 2 * radius + 1, 1, radius)
    near = warp(near, kind, level, size)[:, 0] > 0.5
    shown = (second & ok).sum((1, 2)).clamp_min(1)
    return ((near & second & ok).sum((1, 2)) / shown).mean().item(), ((near & ok).sum((1, 2)) / ok.sum((1, 2)).clamp_min(1)).mean().item()


def picos(energy: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """Maximos locais mais fortes do mapa de energia, 1 a cada 100 posicoes, no minimo 5."""
    live = torch.where((F.max_pool2d(energy[:, None], 5, 1, 2)[:, 0] == energy) & ok, energy, torch.zeros_like(energy))
    values, where = live.flatten(1).topk(max(5, energy[0].numel() // 100), 1)
    return torch.zeros_like(live.flatten(1), dtype=torch.bool).scatter(1, where, values > 0).view_as(live)


def pearson(p: torch.Tensor, q: torch.Tensor, ok: torch.Tensor) -> tuple:
    """Correlacao em `ok` ao longo das posicoes, por imagem e filtro. `p`, `q`: [B, C, h, w]."""
    w = ok[:, None].float()
    n = w.sum((2, 3), keepdim=True).clamp_min(1)
    dp, dq = (p - (p * w).sum((2, 3), keepdim=True) / n) * w, (q - (q * w).sum((2, 3), keepdim=True) / n) * w
    vp, vq = dp.pow(2).sum((2, 3)), dq.pow(2).sum((2, 3))
    corr = (dp * dq).sum((2, 3)) / (vp * vq).sqrt().clamp_min(1e-12)
    return corr.masked_fill((vp < FLAT_VAR) | (vq < FLAT_VAR), float("nan")), dp, dq


@torch.no_grad()
def alinhado(modelo, a: Amostra, imgs: torch.Tensor, kind: str, level: float) -> list:
    """Mapa da imagem transformada contra o mapa original levado pela mesma transformacao."""
    filters, whole, hits, chance = ([[] for _ in range(modelo.n_layers)] for _ in range(4))
    for (i, before), (_, after) in zip(mapas(modelo, a.v, a.to_float), mapas(modelo, imgs, a.to_float)):
        content = a.content[i:i + BATCH].to(DEVICE)
        for n, (p, q) in enumerate(zip(before, after)):
            ok = valido(content, kind, level, a.size, q.shape[-2:], 1)
            corr, dp, dq = pearson(warp(p, kind, level, a.size), q, ok)
            filters[n].append(corr.cpu())
            whole[n].append(((dp * dq).sum((1, 2, 3)) / (dp.pow(2).sum((1, 2, 3)) * dq.pow(2).sum((1, 2, 3))).sqrt().clamp_min(1e-12)).cpu())
            hit, luck = repete(picos(p.pow(2).mean(1), valido(content, "ruido", 0, a.size, q.shape[-2:], 1)),
                               picos(q.pow(2).mean(1), ok), kind, level, a.size, ok, 1)
            hits[n].append(hit)
            chance[n].append(luck)
    rows = []
    for n in range(modelo.n_layers):
        per_filter = torch.nanmean(torch.cat(filters[n]), 0).numpy()
        alive = per_filter[~np.isnan(per_filter)]
        rows.append({"camada": n + 1, "mapa_corr": float(torch.cat(whole[n]).mean()), "filtro_med": float(np.median(alive)),
                     "filtro_p10": float(np.percentile(alive, 10)), "filtro_estavel": float((alive > STABLE).mean()),
                     "filtro_mortos": int(np.isnan(per_filter).sum()), "repet_picos": float(np.mean(hits[n])),
                     "repet_acaso": float(np.mean(chance[n]))})
    return rows


def f3_estabilidade(modelo, nome: str, a: Amostra, conds: dict) -> list:
    """Mudanca na leitura, mais a estabilidade de cada filtro, dos mapas e dos picos de energia."""
    rows = {(r["camada"], r["op"], r["nivel"]): r | {"teste": "F3"} for r in medir(modelo, nome, a, conds)}
    for imgs, fields in conds.values():
        for row in alinhado(modelo, a, imgs, fields["op"], fields["nivel"]):
            rows[row["camada"], fields["op"], fields["nivel"]] |= row
    return list(rows.values())


def f3_keypoints(a: Amostra, conds: dict) -> list:
    """Repetibilidade SIFT: depende da imagem, nao do banco, entao sai uma vez so."""
    def mark(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], *x.shape[2:], dtype=torch.bool)
        for image, plane in zip(x[:, 0].numpy(), out):
            points = keypoints(image)[0][:TOP]
            plane[points[:, 0], points[:, 1]] = True
        return out

    first, rows = mark(a.v), []
    for imgs, fields in conds.values():
        kind, level = fields["op"], fields["nivel"]
        ok = valido(a.content, kind, level, a.size, (a.size, a.size), BORDER)
        hit, luck = repete(first, mark(imgs), kind, level, a.size, ok, RADIUS)
        rows.append({"dataset": a.dataset, "banco": "sift", "camada": 0, "teste": "F3", **fields,
                     "repet_sift": hit, "repet_acaso": luck})
    return rows


@torch.no_grad()
def f4_energia(modelo, nome: str, a: Amostra) -> list:
    """Quanto da energia cai no objeto, contra a area dele. Densidade = energia por posicao."""
    share, area, ratio, outside = ([[] for _ in range(modelo.n_layers)] for _ in range(4))
    for i, found in mapas(modelo, a.v, a.to_float):
        obj, content = (m[i:i + BATCH, None].to(DEVICE).float() for m in (a.obj, a.content))
        for n, z in enumerate(found):
            inside, alive = (F.interpolate(m, z.shape[-2:], mode="nearest") for m in (obj, content))
            rest = alive * (1 - inside)
            energy = z.pow(2)
            e_in, e_out = (energy * inside).sum((2, 3)), (energy * rest).sum((2, 3))
            a_in, a_out = inside.sum((2, 3)), rest.sum((2, 3))
            seen = (a_in[:, 0] > 0) & (a_out[:, 0] > 0)
            d_in, d_out = e_in / a_in.clamp_min(1), e_out / a_out.clamp_min(1)
            share[n].append((e_in.sum(1) / (e_in + e_out).sum(1).clamp_min(1e-12))[seen].cpu())
            area[n].append((a_in / (a_in + a_out).clamp_min(1))[seen, 0].cpu())
            ratio[n].append((d_in.sum(1) / d_out.sum(1).clamp_min(1e-12))[seen].cpu())
            outside[n].append((d_out > d_in).float().mean(1)[seen].cpu())
    return [{"dataset": a.dataset, "banco": nome, "camada": n + 1, "imagens": len(torch.cat(share[n])),
             "energia_objeto": float(torch.cat(share[n]).mean()), "area_objeto": float(torch.cat(area[n]).mean()),
             "densidade_objeto_fundo": float(torch.cat(ratio[n]).median()),
             "filtros_mais_no_fundo": float(torch.cat(outside[n]).mean())} for n in range(modelo.n_layers)]


def separacao(rows: list) -> list:
    """Quanto da mudanca da imagem inteira vem so da regiao, a comparar com a area dela."""
    whole = {(r["banco"], r["camada"], r["op"], r["teste"]): r for r in rows if r["alvo"] == "inteira"}
    return [{k: r.get(k, "") for k in ("dataset", "banco", "camada", "teste", "op", "alvo", "area", "obj_na_regiao")} |
            {"parte_rel": r["rel"] / whole[r["banco"], r["camada"], r["op"], r["teste"]]["rel"]}
            for r in rows if r["alvo"] != "inteira"]


def salva(path: Path, rows: list) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "x", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    torch.set_num_threads(2)
    with open(config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(cfg["job"], encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    out = Path(cfg["out"]) / cfg["stem"]
    out.mkdir(parents=True)
    bancos, twin = dict(cfg["bancos"]), cfg["aleatorio"]
    if not (Path(twin["pasta"]) / "architecture.json").exists():
        aleatorio(job, next(iter(bancos.values())), twin["pasta"], twin["seed"])
    bancos[twin["nome"]] = twin["pasta"]
    a = amostra(job, cfg["n_train"], cfg["n_val"])
    c1, c2, c3 = (f1_conds(a) if a.obj is not None else {}), f2_conds(a), f3_conds(a)
    fundo, trans, energia = [], f3_keypoints(a, c3), []
    for nome, pasta in bancos.items():
        modelo = banco(pasta)
        fundo += f2_periferia(modelo, nome, a, c2)
        trans += f3_estabilidade(modelo, nome, a, c3)
        if a.obj is not None:
            fundo += f1_fundo(modelo, nome, a, c1)
            energia += f4_energia(modelo, nome, a)
    salva(out / "robustez_fundo.csv", fundo)
    salva(out / "robustez_transformacao.csv", trans)
    salva(out / "separacao.csv", separacao(fundo))
    if energia:
        salva(out / "energia_objeto.csv", energia)
    (out / "run.json").write_text(json.dumps(cfg | {
        "n_val_real": len(a.vy), "torch": torch.__version__, "hash_banco": {
            nome: hashlib.sha256(b"".join(Path(p).read_bytes() for p in sorted(glob.glob(f"{pasta}/conv*-kernels.npy")))).hexdigest()
            for nome, pasta in bancos.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
