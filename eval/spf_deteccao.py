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
"""spf_deteccao.py — onde o SPiFiL sem backprop poe os pontos e onde ele acende, em 4 imagens de treino.

`python -m eval.spf_deteccao <base>`. Bancos de `spf_grade.chain` (split 1, 50%, 200 px, 50 superpixels, 64
filtros, 3 camadas, sorteio 42, 500 imagens); pontos de `spf_grade.sources`, com o corte de fundo das 500;
energia de `spf_fundo.mapas`. A mascara nao escolhe nada: so desenha o contorno e mede a fracao dentro.
Pontos medem contra a area na grade inteira; energia, contra a area fora da borda de 8 px e das faixas lisas.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from spifil.color import LabNorm

from core.data.transforms import letterbox
from eval import spf_grade
from eval.l1_sem_rotulo import train_pool
from eval.spf_fundo import mapas
from eval.spf_grade import MASK_DIR, SIFT, coords_of, mask_of, saliency_grid, segmenter, sources
from flim import spifil
from flim.seed_points import SeedPoints
from saliency.batch import content_box

OUT = Path("/exps/moliveira/results_sol/figures/deteccao")
SIZE, SHOWN, BORDER, LAYER, WIDTH, DRAW = 200, 4, 8, 3, 64, 42
POINTS = {"sift_nms": "sift_nms", "medoid": "medoide (padrão)", "obj": "obj"}
ARMS = {"fps": "fps", "fpsw@mapa_peso": "fpsw@mapa_peso", "crus@sift": "crus@sift", "km_w": "km_w",
        "zca2": "zca2", "spifil": "spifil", "random_z": "aleatório (régua)"}
BLACK = ("helminth_eggs", "helminth_larvae", "protozoan_cysts")   # fundo ja preto: `content_box` deixaria so o objeto
MASK_DIR |= dict.fromkeys(BLACK, "label")
REGION = f"content_box + borda {BORDER} px"
LINE = "#00e5ff"


def job_of(base: str) -> dict:
    return {"stem": f"{base}_deteccao", "dataset_name": base, "split": 1, "percentage": 50, "image_size": SIZE,
            "no_crop": True, "image_cache_dir": "/exps/moliveira/image_cache", "superpixels": [50],
            "grow": {"impurities": True, "superpixels": "disf", "budget_check": False},
            "amostra": {"images": 500, "depth": LAYER, "pool": 2, "saliency": {"map": "gmr", "cuts": []}},
            "work_dir": f"/exps/moliveira/tmp/deteccao/{base}", "draw": DRAW}


def rgb_grid(path: Path, box: bool) -> tuple:
    """RGB na grade do letterbox, a regiao medida (sem faixas lisas de `content_box`, se `box`, e sem 8 px de borda) e o nome dela."""
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    f = SIZE / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    alive = np.zeros(img.shape[:2], np.uint8)
    alive[content_box(img) if box else ()] = 255
    rgb = letterbox(torch.from_numpy(img).permute(2, 0, 1), SIZE, (0, 0, 0)).permute(1, 2, 0).numpy()
    alive = letterbox(torch.from_numpy(alive)[None], SIZE, (0,))[0].numpy()
    kernel = np.ones((2 * BORDER + 1,) * 2, np.uint8)
    inner = cv2.erode(alive, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 127
    if not inner.any():   # tira de Larvae raw: 16 px de lado
        return rgb, alive > 127, "content_box, sem borda (tira fina)"
    return rgb, inner, REGION if box else f"sem content_box, só borda {BORDER} px"


def share(mask: np.ndarray | None, yx: np.ndarray) -> float:
    return float(mask[yx[:, 0], yx[:, 1]].mean()) if mask is not None and len(yx) else float("nan")


def frame(ax, image: np.ndarray, mask: np.ndarray | None, title: str, **kw) -> None:
    ax.imshow(image, **kw)
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors=LINE, linewidths=1)
    ax.set(xticks=[], yticks=[])
    ax.set_title(title, fontsize=9)


def banks(job: dict, lab: list, found: dict, cut: float, x: torch.Tensor) -> dict:
    """O encoder da camada 3 de cada braco: `chain` constroi e descarta, entao o `build` dele e espiado."""
    built, out = spf_grade.build, {}
    images = {"train": (x, None), "validation": (x, None)}
    for arm in ARMS:
        got = []
        spf_grade.build = lambda *a, got=got: got.append(built(*a)) or got[-1]
        spf_grade.chain(job, images, lambda t: t, lab, found, cut, arm, (WIDTH,) * LAYER, 5, DRAW, {}, lambda *_: None)
        out[arm] = got[-1]
    spf_grade.build = built
    return out


def main(base: str) -> None:
    torch.set_num_threads(2)
    job = job_of(base)
    Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
    pool = train_pool(job)
    four = [pool[i] for i in np.linspace(0, len(pool) - 1, SHOWN).round().astype(int)]
    lab, found, cut, _ = sources(job, random.Random(DRAW).sample(pool, min(job["amostra"]["images"], len(pool))))
    spf_grade.background_cut = lambda _: cut   # `obj` das 4 com o mesmo corte que os bancos viram
    lab4, shown, *_ = sources(job | {"amostra": {}}, four)
    x4 = torch.stack(lab4)
    models = banks(job, lab, found, cut, x4.cpu())
    data = spifil.Cropped(four, SIZE, use_mask=False, no_crop=True)
    energy = {}
    for arm, model in models.items():
        _, maps = next(mapas(model, x4, lambda t: t))
        e = F.interpolate(maps[LAYER - 1].pow(2).sum(1, keepdim=True), (SIZE, SIZE), mode="bilinear", align_corners=False)
        energy[arm] = e[:, 0].cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    fa, axa = plt.subplots(SHOWN, 6, figsize=(6 * 2.7, SHOWN * 3.2), gridspec_kw={"hspace": 0.45, "wspace": 0.08, "left": 0.01, "right": 0.99})
    fb, axb = plt.subplots(SHOWN, 1 + len(ARMS), figsize=((1 + len(ARMS)) * 2.7, SHOWN * 3.2),
                           gridspec_kw={"hspace": 0.45, "wspace": 0.08, "left": 0.01, "right": 0.99})
    for i, sample in enumerate(four):
        name = sample.image_path.name
        rgb, valid, region = rgb_grid(sample.image_path, base not in BLACK)
        mask = mask_of(base, name, SIZE).numpy() if base in MASK_DIR else None
        area = float(mask.mean()) if mask is not None else float("nan")
        head = f"{name}\nobjeto: {area:.0%} da imagem" if mask is not None else f"{name}\n(sem máscara)"
        head += "" if region == REGION else f"\nenergia: {region}"

        def note(panel: str, frac: float, where: float = area, part: str = "grade inteira") -> str:
            rows.append({"base": base, "imagem": name, "painel": panel, "fracao_dentro": frac, "area_objeto": where,
                         "regiao": part})
            return "" if np.isnan(frac) else f" · {frac:.0%} dentro"

        frame(axa[i, 0], rgb, mask, head)
        frame(axb[i, 0], rgb, mask, head)
        ready = spifil.preparation.prepare_sample(data, i, superpixels=segmenter(job), seed_extractor=SeedPoints("medoid"),
                                                  n_superpixels=job["superpixels"][0], color=LabNorm())
        kps = SIFT.detect((ready.features[0].clip(0, 1) * 255).astype(np.uint8), None)
        yx = np.array([k.pt[::-1] for k in kps]).round().astype(int).clip(0, SIZE - 1).reshape(-1, 2)
        frame(axa[i, 1], rgb, mask, f"SIFT, todos ({len(kps)}){note('sift_todos', share(mask, yx))}")
        for k in kps:
            axa[i, 1].add_patch(plt.Circle(k.pt, k.size / 2, fill=False, color="#ffd400", lw=0.6))
        for j, (point, label) in enumerate(POINTS.items(), 2):
            yx = coords_of(shown[point])[i].numpy()
            frame(axa[i, j], rgb, mask, f"{label} ({len(yx)}){note(point, share(mask, yx))}")
            axa[i, j].scatter(yx[:, 1], yx[:, 0], s=14, color="#ffd400", edgecolor="black", linewidth=0.4)
        frame(axa[i, 5], saliency_grid(sample.image_path, SIZE, {"map": "gmr", "cuts": []}, base)["peso"], mask,
              "mapa gmr (saliência)", cmap="magma", vmin=0, vmax=1)
        for j, (arm, label) in enumerate(ARMS.items(), 1):
            e = energy[arm][i]
            lo, hi = np.percentile(e[valid], [1, 99])
            seen = np.ma.masked_where(~valid, ((e - lo) / max(hi - lo, 1e-12)).clip(0, 1))
            text = ""
            if mask is not None:
                inner = valid & mask
                frac, where = float(e[inner].sum() / max(e[valid].sum(), 1e-12)), float(inner.sum() / valid.sum())
                note(f"energia_{arm}", frac, where, region)
                text = f"\nenergia dentro {frac:.0%} (área {where:.0%})"
            frame(axb[i, j], seen, mask, f"{label}{text}", cmap="inferno", vmin=0, vmax=1)
            axb[i, j].set_facecolor("#9a9a9a")
    cut = (f"só a borda de {BORDER} px em cinza, sem content_box (fundo já preto)" if base in BLACK
           else f"borda de {BORDER} px e faixas lisas em cinza")
    tail = "contorno = máscara, só para medir; nada usa máscara na escolha"
    fa.suptitle(f"{base}: pontos sem rótulo e saliência ({tail})", x=0.01, ha="left", fontsize=12)
    fb.suptitle(f"{base}: energia na camada {LAYER} (soma dos quadrados por pixel), {WIDTH} filtros, sem backprop; "
                f"normalizada por imagem (p1–p99), {cut} ({tail})",
                x=0.01, ha="left", fontsize=11)
    fa.savefig(OUT / f"{base}_deteccao.png", dpi=110, bbox_inches="tight")
    fb.savefig(OUT / f"{base}_ativacao.png", dpi=110, bbox_inches="tight")
    with open(OUT / f"{base}_fracoes.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
