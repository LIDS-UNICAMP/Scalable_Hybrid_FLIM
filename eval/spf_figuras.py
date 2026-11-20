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
"""spf_figuras.py — as 15 figuras do pedido multiescala, em Matplotlib.

1 piramide gaussiana, 2 DoG, 3 LoG, 4 keypoints, 5 seletores, 6 patches, 7 filtros, 8 similaridade,
9 correlacao de ativacoes, 10 mapas por camada, 11 desempenho x profundidade, 12 delta por dataset,
13 intervalos, 14 comparacao com aleatorio, 15 estabilidade.

1 a 6 olham uma imagem de treino; 7 a 10 comparam os bancos do YAML de `eval/spf_fundo.py`, com
imagens de validacao; 11 a 14 leem os CSVs de `eval/spf_tabela.py` (so validacao); 15 le a pasta de
`eval/spf_fundo.py`, com transformacao e fundo em linhas separadas. Salva em `<figs>/<stem>/`, sem
sobrescrever, e copia AMOSTRAS em `<amostras>/<stem>/`.

`python -m eval.spf_figuras --config <yaml>`: o YAML de `spf_fundo` mais figs, amostras, rodadas, bracos.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from spifil.color import LabNorm
from spifil.superpixels.disf import DISF

from eval import spf_tabela
from eval.l1_amostras import kernel_stats
from eval.l1_sem_rotulo import train_pool
from eval.spf_fundo import OPS, TRANSFORMS, amostra, banco, camadas, mapas
from eval.spf_grade import image_points
from eval.svm import cli_kwargs
from flim import spifil
from flim.seed_points import SeedPoints

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = (BLUE, ORANGE, AQUA, YELLOW)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OCTAVES, LEVELS, SIGMA0 = 3, 5, 1.6
K = 2 ** (1 / (LEVELS - 2))
SELECTORS = ("medoid", "centroid", "geodesic", "sift_sp", "sift_nms")
PATCH = 15        # janela mostrada em volta da semente
PER_ROW = 12
GRID_FILTERS = 64
TOP = 50
AMOSTRAS = (1, 4, 7, 10, 12, 15)
READOUT = "L"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.titlesize": 10, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
                     "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def salva(fig, out: Path, number: int, name: str) -> None:
    path = out / f"fig{number:02d}_{name}.png"
    assert not path.exists(), f"nao sobrescreve {path}"
    fig.savefig(path)
    plt.close(fig)


def limpa(ax, title: str = "") -> None:
    ax.set(xticks=[], yticks=[])
    ax.grid(False)
    ax.set_title(title, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def espaco(gray: np.ndarray) -> list:
    """Oitavas de L = G * I: lista de (oitava, nivel, sigma, mapa). Cada oitava parte da metade da anterior."""
    out, base = [], gray.astype(np.float32) / 255
    for octave in range(OCTAVES):
        for level in range(LEVELS):
            sigma = SIGMA0 * K**level
            out.append((octave, level, sigma * 2**octave, cv2.GaussianBlur(base, (0, 0), sigma)))
        base = cv2.resize(out[-2][3], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    return out


def fig_piramide(pyramid: list, out: Path) -> None:
    fig, axes = plt.subplots(OCTAVES, LEVELS, figsize=(LEVELS * 1.7, OCTAVES * 1.9))
    for (octave, level, sigma, image) in pyramid:
        axes[octave, level].imshow(image, cmap="gray")
        limpa(axes[octave, level], f"oitava {octave}, nível {level}\nσ={sigma:.1f}, {image.shape[0]} px")
    fig.suptitle("Espaço de escalas gaussiano", x=0.02, ha="left")
    salva(fig, out, 1, "piramide")


def fig_dog(pyramid: list, out: Path) -> None:
    fig, axes = plt.subplots(OCTAVES, LEVELS - 1, figsize=((LEVELS - 1) * 1.8, OCTAVES * 1.9))
    for first, second in zip(pyramid, pyramid[1:]):
        if first[0] == second[0]:
            image = second[3] - first[3]
            top = max(np.abs(image).max(), 1e-6)
            axes[first[0], first[1]].imshow(image, cmap="RdBu_r", vmin=-top, vmax=top)
            limpa(axes[first[0], first[1]], f"oitava {first[0]}: σ {first[2]:.1f} → {second[2]:.1f}")
    fig.suptitle("Diferença de gaussianas: L(kσ) − L(σ)", x=0.02, ha="left")
    salva(fig, out, 2, "dog")


def fig_log(gray: np.ndarray, out: Path) -> None:
    sigmas = [SIGMA0 * K**i for i in range(0, 3 * (LEVELS - 2), 2)]
    fig, axes = plt.subplots(1, len(sigmas), figsize=(len(sigmas) * 1.9, 2.3))
    for ax, sigma in zip(axes, sigmas):
        image = sigma**2 * cv2.Laplacian(cv2.GaussianBlur(gray.astype(np.float32) / 255, (0, 0), sigma), cv2.CV_32F)
        top = max(np.abs(image).max(), 1e-6)
        ax.imshow(image, cmap="RdBu_r", vmin=-top, vmax=top)
        limpa(ax, f"σ={sigma:.1f}")
    fig.suptitle("LoG normalizado: σ²∇²L", x=0.02, ha="left", y=1.05)
    salva(fig, out, 3, "log")


def fig_keypoints(gray: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(gray, cmap="gray")
    for point in sorted(cv2.SIFT_create(contrastThreshold=0.02).detect(gray, None), key=lambda p: -p.response)[:TOP]:
        ax.add_patch(plt.Circle(point.pt, point.size / 2, fill=False, color=YELLOW, lw=1))
    limpa(ax)
    ax.set_title(f"{TOP} keypoints SIFT mais fortes; círculo = escala", loc="left")
    salva(fig, out, 4, "keypoints")


def fig_seletores(gray: np.ndarray, found: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(SELECTORS), figsize=(len(SELECTORS) * 2.1, 2.5))
    for ax, name in zip(axes, SELECTORS):
        points = found[name][0]
        ax.imshow(gray, cmap="gray")
        ax.scatter(points[:, 1], points[:, 0], s=12, color=YELLOW, edgecolor=INK, linewidth=0.3)
        limpa(ax, f"{name} ({len(points)})")
    fig.suptitle("Onde cada seletor põe os pontos, sem rótulo", x=0.02, ha="left", y=1.04)
    salva(fig, out, 5, "seletores")


def fig_patches(gray: np.ndarray, found: dict, out: Path) -> None:
    half = PATCH // 2
    padded = np.pad(gray, half, mode="edge")
    fig, axes = plt.subplots(len(SELECTORS), PER_ROW, figsize=(PER_ROW * 0.7, len(SELECTORS) * 0.8))
    for row, name in zip(axes, SELECTORS):
        points = found[name][0]
        picks = points[np.linspace(0, len(points) - 1, PER_ROW).round().astype(int)]
        for ax, (y, x) in zip(row, picks):
            ax.imshow(padded[y:y + PATCH, x:x + PATCH], cmap="gray", vmin=0, vmax=255)
            limpa(ax)
        row[0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"Janelas de {PATCH} px em volta dos pontos de cada seletor", x=0.02, ha="left")
    salva(fig, out, 6, "patches")


def kernels(modelo) -> torch.Tensor:
    return modelo.conv1[0].weight.detach().cpu()


def fig_filtros(bancos: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(bancos), figsize=(4 * len(bancos), 4.2))
    side = int(GRID_FILTERS**0.5)
    for ax, (nome, modelo) in zip(axes, bancos.items()):
        w = kernels(modelo)[:GRID_FILTERS, 0].numpy()
        k = w.shape[-1]
        sheet = np.full((side * (k + 1), side * (k + 1)), np.nan)
        for i, tile in enumerate(w):
            r, c = divmod(i, side)
            sheet[r * (k + 1):r * (k + 1) + k, c * (k + 1):c * (k + 1) + k] = tile / max(np.abs(tile).max(), 1e-9)
        ax.imshow(sheet, cmap="RdBu_r", vmin=-1, vmax=1)
        limpa(ax, f"{nome}: {GRID_FILTERS} de {len(kernels(modelo))} filtros, banda L, camada 1")
    salva(fig, out, 7, "filtros")


def fig_similaridade(bancos: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(bancos) + 1, figsize=(3.6 * (len(bancos) + 1), 3.4))
    for i, (ax, (nome, modelo)) in enumerate(zip(axes, bancos.items())):
        unit = torch.nn.functional.normalize(kernels(modelo).flatten(1), dim=1).double().numpy()
        cos = unit @ unit.T
        stats = kernel_stats(unit)
        ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
        limpa(ax, f"{nome}: |cos| médio {stats['cos_mean']:.2f}, posto efetivo {stats['erank']:.0f}")
        axes[-1].hist(np.abs(cos[np.tril_indices(len(cos), -1)]), bins=40, range=(0, 1), histtype="step", color=SERIES[i], label=nome, density=True, lw=2)
    axes[-1].set(xlabel="|cosseno| entre filtros", ylabel="densidade")
    axes[-1].legend()
    salva(fig, out, 8, "similaridade")


def fig_correlacao(bancos: dict, a, out: Path) -> None:
    fig, axes = plt.subplots(1, len(bancos) + 1, figsize=(3.6 * (len(bancos) + 1), 3.4))
    for i, (ax, (nome, modelo)) in enumerate(zip(axes, bancos.items())):
        corr = np.nan_to_num(np.corrcoef(camadas(modelo, a.v, a)[0].T))
        off = np.abs(corr[np.tril_indices(len(corr), -1)])
        ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        limpa(ax, f"{nome}: |corr| média {off.mean():.2f}")
        axes[-1].hist(off, bins=40, range=(0, 1), histtype="step", color=SERIES[i], label=nome, density=True, lw=2)
    axes[-1].set(xlabel="|correlação| entre ativações (GeM, camada 1)", ylabel="densidade")
    axes[-1].legend()
    salva(fig, out, 9, "correlacao_ativacoes")


def fig_mapas(bancos: dict, a, out: Path) -> None:
    depth = next(iter(bancos.values())).n_layers
    fig, axes = plt.subplots(len(bancos), depth + 1, figsize=((depth + 1) * 2.1, len(bancos) * 2.3))
    for row, (nome, modelo) in zip(axes, bancos.items()):
        _, found = next(mapas(modelo, a.v[:1], a.to_float))
        row[0].imshow(a.v[0, 0], cmap="gray")
        limpa(row[0], f"{nome}: entrada")
        for n, (ax, z) in enumerate(zip(row[1:], found)):
            ax.imshow(z[0].pow(2).mean(0).cpu(), cmap="Blues")
            limpa(ax, f"energia, camada {n + 1}, {z.shape[-1]} px")
    salva(fig, out, 10, "mapas_camadas")


def desempenho(cfg: dict) -> tuple:
    """Linhas e deltas da leitura por camada, na largura pedida, com o fundo real."""
    data = spf_tabela.load(cfg["rodadas"])
    data = data[(data.fundo == "real") & (data.readout == READOUT) & (data.plan == cfg["plano"])]
    return data, spf_tabela.deltas(data)


def fig_profundidade(data: pd.DataFrame, cfg: dict, out: Path) -> None:
    names = sorted(data.dataset.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        part = data[data.dataset == name]
        best = part[part.family == "controle"].groupby("arm").val_kappa.mean().idxmax()
        for color, arm in zip(SERIES, [*cfg["bracos"], best]):
            curve = part[part.arm == arm].groupby("layer").val_kappa.mean()
            ax.plot(curve.index, curve.values, marker="o", ms=5, color=color, label=arm)
        ax.set(title=name, xlabel="camada", xticks=sorted(part.layer.unique()))
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("kappa de validação")
    salva(fig, out, 11, "profundidade")


def fig_delta(table: pd.DataFrame, cfg: dict, out: Path) -> None:
    arm, layer = cfg["bracos"][0], table.layer.max()
    part = table[(table.arm == arm) & (table.layer == layer)].groupby("dataset").delta.mean().sort_values()
    fig, ax = plt.subplots(figsize=(5.5, 0.4 * len(part) + 1.3))
    ax.barh(part.index, part.values, color=BLUE, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set(title=f"Δ kappa de {arm} contra o melhor aleatório, camada {layer}", xlabel="Δ kappa (validação)")
    salva(fig, out, 12, "delta_dataset")


def fig_intervalos(table: pd.DataFrame, cfg: dict, out: Path) -> None:
    layer = table.layer.max()
    names = sorted(table.dataset.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 0.5 * len(cfg["bracos"]) + 1.6), sharey=True, squeeze=False)
    for ax, name in zip(axes[0], names):
        for i, (arm, color) in enumerate(zip(cfg["bracos"], SERIES)):
            values = table[(table.dataset == name) & (table.arm == arm) & (table.layer == layer)].delta.to_numpy()
            if not len(values):
                continue
            mean, lo, hi = spf_tabela.interval(values)
            ax.plot([lo, hi], [i, i], color=color)
            ax.plot(mean, i, "o", color=color, ms=7)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set(title=name, xlabel="Δ kappa, IC 95% (t, por split)", yticks=range(len(cfg["bracos"])), yticklabels=cfg["bracos"])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    salva(fig, out, 13, "intervalos")


def fig_aleatorio(table: pd.DataFrame, cfg: dict, out: Path) -> None:
    part = table[table.layer == table.layer.max()]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    for arm, color in zip(cfg["bracos"], SERIES):
        mine = part[part.arm == arm]
        ax.scatter(mine.control_kappa, mine.val_kappa, s=36, color=color, edgecolor="#fcfcfb", linewidth=1, label=arm)
    low, high = part[["control_kappa", "val_kappa"]].min().min(), part[["control_kappa", "val_kappa"]].max().max()
    ax.plot([low, high], [low, high], color=MUTED, lw=1)
    ax.set(xlabel="kappa do melhor aleatório", ylabel="kappa do método", title="Acima da diagonal, o método vence (1 ponto = base, split)")
    ax.legend()
    salva(fig, out, 14, "contra_aleatorio")


def fig_estabilidade(pasta: Path, nomes: list, out: Path) -> None:
    trans, parts = pd.read_csv(pasta / "robustez_transformacao.csv"), pd.read_csv(pasta / "separacao.csv")
    layer = trans.camada.max()
    fig, axes = plt.subplots(2, len(TRANSFORMS), figsize=(2.5 * len(TRANSFORMS), 5.4), gridspec_kw={"hspace": 0.55})
    for ax, kind in zip(axes[0], TRANSFORMS):
        for nome, color in zip(nomes, SERIES):
            curve = trans[(trans.banco == nome) & (trans.camada == layer) & (trans.op == kind)].sort_values("nivel")
            ax.plot(curve.nivel, curve.rel, marker="o", ms=5, color=color, label=nome)
        ax.set(title=kind, xlabel="nível")
    axes[0, 0].set_ylabel("mudança relativa da leitura")
    axes[0, 0].legend(fontsize=7)
    fig.text(0.02, 0.94, f"Robustez a transformação (imagem inteira muda), camada {layer}", fontweight="bold")
    alvos = sorted(parts.alvo.unique())[:len(TRANSFORMS)]
    for ax in axes[1]:
        ax.set_visible(False)
    for ax, alvo in zip(axes[1], alvos):
        ax.set_visible(True)
        part = parts[(parts.alvo == alvo) & (parts.camada == layer)]
        for i, (nome, color) in enumerate(zip(nomes, SERIES)):
            mine = part[part.banco == nome].set_index("op").reindex(OPS)
            ax.bar(np.arange(len(OPS)) + (i - 0.5) * 0.38, mine.parte_rel, width=0.34, color=color, label=nome)
        ax.axhline(part.area.mean(), color=MUTED, lw=1, ls="--")
        ax.set_xticks(range(len(OPS)), OPS, rotation=45, ha="right")
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{alvo} (tracejado = área)")
    axes[1, 0].set_ylabel("região / inteira")
    fig.text(0.02, 0.46, "Robustez ao fundo (só a região muda)", fontweight="bold")
    salva(fig, out, 15, "estabilidade")


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    torch.set_num_threads(2)
    with open(config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(cfg["job"], encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    out = Path(cfg["figs"]) / cfg["stem"]
    out.mkdir(parents=True)
    sample = train_pool(job)[:1]
    data = spifil.Cropped(sample, job["image_size"], use_mask=False, no_crop=job["no_crop"])
    ready = spifil.preparation.prepare_sample(data, 0, superpixels=DISF(), seed_extractor=SeedPoints("medoid"),
                                              n_superpixels=job["superpixels"][0], color=LabNorm())
    gray = (ready.features[0].clip(0, 1) * 255).astype(np.uint8)
    found = image_points(ready, job["superpixels"][0])
    pyramid = espaco(gray)
    fig_piramide(pyramid, out)
    fig_dog(pyramid, out)
    fig_log(gray, out)
    fig_keypoints(gray, out)
    fig_seletores(gray, found, out)
    fig_patches(gray, found, out)
    pastas = {**cfg["bancos"], cfg["aleatorio"]["nome"]: cfg["aleatorio"]["pasta"]}
    bancos = {nome: banco(p) for nome, p in pastas.items()}
    a = amostra(job, 1, cfg["n_val"])
    fig_filtros(bancos, out)
    fig_similaridade(bancos, out)
    fig_correlacao(bancos, a, out)
    fig_mapas(bancos, a, out)
    rows, table = desempenho(cfg)
    fig_profundidade(rows, cfg, out)
    fig_delta(table, cfg, out)
    fig_intervalos(table, cfg, out)
    fig_aleatorio(table, cfg, out)
    fig_estabilidade(Path(cfg["out"]) / cfg["stem"], list(pastas), out)
    picked = Path(cfg["amostras"]) / cfg["stem"]
    picked.mkdir(parents=True)
    for number in AMOSTRAS:
        for path in out.glob(f"fig{number:02d}_*.png"):
            shutil.copy2(path, picked / path.name)


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
