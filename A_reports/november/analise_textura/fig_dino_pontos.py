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
"""Figura dino_registers_pontos.png: pontos de A (norma alta), B (receita) e D (norma baixa) sobre a imagem de 200 px,
2 imagens por base, e patches 5x5 cortados de cada braco. Os pontos sao os mesmos de `spf_grade.add_dino`."""
import __main__
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import eval.spf_grade as g
from core.data.transforms import letterbox

__main__.Source = g.Source  # o cache `fontes` foi gravado com `python -m eval.spf_grade`
DIR = Path("/exps/moliveira/fontes_filtros/dino_pontos")
OUT = Path(__file__).parent / "dino_registers_pontos.png"
BASES = {"aid": "fps", "remote_sensing": "fps", "flowers": "fps"}
COR = {"A": "#d62728", "B": "#1f77b4", "D": "#2ca02c"}
NOME = {"A": "A: norma alta (registradores)", "B": "B: pontos da receita", "D": "D: norma mais baixa"}


def main() -> None:
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 4, wspace=0.08, hspace=0.25)
    for r, (base, point) in enumerate(BASES.items()):
        pool = g.train_pool({"dataset_name": base, "split": 1, "percentage": 50})
        drawn = random.Random(42).sample(pool, min(500, len(pool)))
        lab, found, cut, _ = torch.load(f"/exps/moliveira/fontes_filtros/receita_tta/{base}_split1_d42.pt", weights_only=False)
        job = {"amostra": {"dino": {"dir": str(DIR), "corte": 150}}, "dataset_name": base, "draw": 42}
        g.add_dino(job, drawn[:2], lab[:2], found, cut)
        src = found["medoid" if point == "fps" else point]
        rows = g.coords_of(src)
        crops = {k: [] for k in "ABD"}
        for c in range(2):
            rgb = letterbox(torch.from_numpy(np.asarray(Image.open(drawn[c].image_path).convert("RGB"))).permute(2, 0, 1),
                            200, (0, 0, 0)).permute(1, 2, 0).numpy()
            pts = {"A": found["dino_alto"].seeds[c].coords.numpy(), "B": rows[c].numpy(), "D": found["dino_baixo"].seeds[c].coords.numpy()}
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(rgb)
            for k, p in pts.items():
                ax.scatter(p[:, 1], p[:, 0], s=22, c=COR[k], edgecolors="white", linewidths=0.5, label=f"{NOME[k]} ({len(p)})")
                pad = np.pad(rgb, ((2, 2), (2, 2), (0, 0)))
                crops[k] += [pad[y:y + 5, x:x + 5] for y, x in p[:4]]
            ax.set_title(f"{base}, imagem {c + 1}", fontsize=13)
            ax.legend(fontsize=8, loc="lower left", framealpha=0.8)
            ax.axis("off")
        sub = grid[r, 2:].subgridspec(3, 8, wspace=0.1, hspace=0.3)
        for i, k in enumerate("ABD"):
            for j, patch in enumerate(crops[k][:8]):
                ax = fig.add_subplot(sub[i, j])
                ax.imshow(patch, interpolation="nearest")
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color(COR[k]), s.set_linewidth(2)
                if j == 0:
                    ax.set_ylabel(k, fontsize=13, color=COR[k], rotation=0, labelpad=12)
    fig.suptitle("Onde o SPiFiL corta os patches 5x5: registradores do DINOv2-giant (A), receita (B) e norma baixa (D)",
                 fontsize=15)
    fig.text(0.5, 0.01, "A: tokens de norma > 150 no ultimo bloco. D: a mesma contagem, com as menores normas. "
             "B: medoides dos superpixels (fps). Direita: patches 5x5 ampliados.",
             ha="center", fontsize=11)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
