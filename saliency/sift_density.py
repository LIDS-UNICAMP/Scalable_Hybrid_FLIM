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
"""sift_density.py — densidade de keypoints SIFT como mapa de "onde esta o objeto", sem rotulo.

`python -m saliency.sift_density experiments/saliency/lote.yaml experiments/saliency/lote_helminth.yaml`.
Mesmas imagens, recorte (`content_box`) e lado do `gmr` de `batch`; detector igual ao de `spf_grade`
(`SIFT`, banda L do `LabNorm`). Cada keypoint vira uma gaussiana de pico 1 e sigma `SIGMA_K * size`:
`sift_a` conta, `sift_b` pesa pela `response`, `sift_c` pesa pela `response` com sigma fixo
`FIXED * lado`, `sift_d` = `sift_b * gmr`. Tudo normalizado para [0,1]. A mascara so mede (`evaluate`
e a fracao de keypoints dentro do objeto); nada dela volta para o mapa. Saida em `OUT` e figuras em `FIG`.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from spifil.color import LabNorm
from threadpoolctl import threadpool_limits

from saliency.batch import content_box, heat, markdown, plan, read_image
from saliency.fusion import evaluate

SIGMA_K = 0.5   # `size` do OpenCV e o diametro da vizinhanca: sigma = raio
FIXED = 0.05    # sigma fixo de `sift_c`, em fracao do lado maior
OUT = Path("/exps/moliveira/saliency_sift")
FIG = Path("A_reports/november/multi_dataset/sift_densidade")
LABEL = {"helminth_eggs": "label", "helminth_larvae": "label", "protozoan_cysts": "label"}  # fora do MASK_DIR
NAMES = ["sift_a", "sift_b", "sift_c", "sift_d", "gmr", "B", "D"]
COLUMNS = ["ap", "iou", "dice", "precision", "recall"]


def unit(m: np.ndarray) -> np.ndarray:
    return ((m - m.min()) / np.ptp(m) if np.ptp(m) > 0 else np.zeros_like(m)).astype(np.float32)


def splat(shape: tuple, yx: np.ndarray, sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Soma exata de gaussianas separaveis: sum_k w_k g_k(y) g_k(x)^T = Gy^T diag(w) Gx."""
    gy = np.exp(-0.5 * ((np.arange(shape[0])[None] - yx[:, :1]) / sigma[:, None]) ** 2)
    gx = np.exp(-0.5 * ((np.arange(shape[1])[None] - yx[:, 1:]) / sigma[:, None]) ** 2)
    return unit((gy.T * w).astype(np.float32) @ gx.astype(np.float32))


def sift_maps(img: np.ndarray, gmr: np.ndarray) -> tuple:
    """As 4 variantes e os keypoints (y, x, size, response) de uma imagem RGB uint8."""
    from eval.spf_grade import SIFT
    found = SIFT.detect((LabNorm()(img)[0].clip(0, 1) * 255).astype(np.uint8), None)
    kp = np.array([(k.pt[1], k.pt[0], k.size, k.response) for k in found], np.float64).reshape(-1, 4)
    if not len(kp):
        zero = np.zeros(img.shape[:2], np.float32)
        return dict.fromkeys(["sift_a", "sift_b", "sift_c", "sift_d"], zero), kp
    yx, sigma, resp = kp[:, :2], np.maximum(SIGMA_K * kp[:, 2], 0.5), kp[:, 3]
    b = splat(img.shape[:2], yx, sigma, resp)
    return {"sift_a": splat(img.shape[:2], yx, sigma, np.ones(len(kp))), "sift_b": b,
            "sift_c": splat(img.shape[:2], yx, np.full(len(kp), FIXED * max(img.shape[:2])), resp),
            "sift_d": unit(b * gmr)}, kp


def score(task: dict) -> list:
    cv2.setNumThreads(1)
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.simplefilter("ignore")
        frame = read_image(task["path"], task["max_side"])
        box = content_box(frame)
        img = np.ascontiguousarray(frame[box])
        stored = np.load(Path(task["out"]) / "mapas.npz")
        if stored["gmr"].shape != img.shape[:2]:  # `content_box` mudou depois do lote (larvas finas)
            return [{"dataset": task["dataset"], "image": Path(task["out"]).name, "skip": True}]
        start = time.perf_counter()
        maps, kp = sift_maps(img, stored["gmr"])
        seconds = time.perf_counter() - start
        where = OUT / task["dataset"] / Path(task["out"]).name
        where.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"{where}.npz", kp=kp, **maps)
        tag = {"dataset": task["dataset"], "image": Path(task["out"]).name, "h": img.shape[0], "w": img.shape[1],
               "n_kp": len(kp), "seconds": seconds}
        if not task["mask"]:
            return [tag]
        gt = cv2.resize(cv2.imread(task["mask"], cv2.IMREAD_GRAYSCALE), frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)[box] > 0
        if not gt.any() or gt.all():
            return [tag]
        at = kp[:, :2].round().astype(int).clip(0, np.array(gt.shape) - 1)
        inside = gt[at[:, 0], at[:, 1]]
        tag |= {"area": gt.mean(), "kp_in": inside.mean() if len(kp) else np.nan,
                "kp_in_resp": (kp[:, 3] * inside).sum() / kp[:, 3].sum() if len(kp) else np.nan}
        rows = []
        for name in NAMES:
            got = evaluate(maps[name] if name.startswith("sift") else stored[name], gt, ())
            rows.append(tag | {"name": name, "ap": got["ap"]} | got["thresholds"].get("otsu", dict.fromkeys(COLUMNS[1:], 0.0)))
        return rows


def tasks(configs: list) -> list:
    from core.constants import ORGANIZED_ROOT
    out = []
    for config in configs:
        with open(config, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        for task in plan(cfg):
            if task["dataset"] in LABEL:
                task["mask"] = str(Path(ORGANIZED_ROOT) / task["dataset"] / LABEL[task["dataset"]] / Path(task["path"]).name)
            if not (Path(task["out"]) / "mapas.npz").exists():
                raise FileNotFoundError(f"amostra diferente da de batch: falta {task['out']}")
            out.append(task)
    return out


def figure(table: pd.DataFrame, todo: list) -> None:
    """Por base, 3 linhas: melhor, mediano e pior AP de `sift_b` onde ha mascara; os 3 primeiros onde nao ha."""
    FIG.mkdir(parents=True, exist_ok=True)
    tile, by = 200, {(t["dataset"], Path(t["out"]).name): t for t in todo}
    for dataset, part in table.groupby("dataset"):
        b = part[part.name == "sift_b"]
        if len(b):
            order = b.sort_values("ap", ascending=False).image.tolist()
            pick = [order[0], order[len(order) // 2], order[-1]]
        else:
            pick = sorted(part.image.unique())[:3]
        rows = []
        for image in pick:
            folder = Path(by[dataset, image]["out"])
            img = cv2.imread(str(folder / "original.png"))
            z = np.load(OUT / dataset / f"{image}.npz")
            kps = [cv2.KeyPoint(float(x), float(y), float(s)) for y, x, s, _ in z["kp"]]
            dots = cv2.drawKeypoints(img, kps, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            gmr = np.load(folder / "mapas.npz")["gmr"]
            cells = [dots] + [np.ascontiguousarray(heat(np.ascontiguousarray(img[..., ::-1]), m)[..., ::-1]) for m in (z["sift_b"], gmr, z["sift_d"])]
            rows.append(cv2.hconcat([cv2.resize(c, (tile, tile), interpolation=cv2.INTER_AREA) for c in cells]))
        sheet = cv2.vconcat(rows)
        for i, text in enumerate(["original + SIFT", "densidade (b)", "gmr", "produto (d)"]):
            cv2.putText(sheet, text, (i * tile + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(FIG / f"{dataset}.png"), sheet)


def main(configs: list) -> None:
    todo = tasks(configs)
    print(len(todo), "imagens", flush=True)
    with mp.get_context("spawn").Pool(3) as pool:
        rows = [r for part in pool.imap_unordered(score, todo) for r in part]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sift.csv", index=False)
    labelled = table[table.name.notna()]
    print(markdown(labelled.groupby(["dataset", "name"])[COLUMNS].mean().unstack(1)))
    print(markdown(labelled[labelled.name == "gmr"].groupby("dataset")[["area", "kp_in", "kp_in_resp"]].mean()))
    print("s/imagem", table.drop_duplicates(["dataset", "image"]).groupby("dataset")[["seconds", "n_kp", "h", "w"]].mean())
    figure(table, todo)


if __name__ == "__main__":
    main(sys.argv[1:])
