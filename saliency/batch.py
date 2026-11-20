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
"""batch.py — os 11 mapas, a selecao sem rotulo e as fusoes numa amostra de imagens de treino.

`python -m saliency.batch experiments/saliency/lote.yaml`. Candidatos: A melhor mapa pelo score, B
fusao Top-K (`weighted_mean`), C fusao regional, D media simples, e cada mapa fixo. Tudo e refeito
em 4 imagens mexidas para medir a estabilidade, com a escolha refeita em cada uma. Por imagem, em
<out>/<base>/<nome>/: original, mapas e heatmaps em PNG, mapas.npz, mascaras por limiar, scores.csv
(score, rank, Top-K) e candidatos.csv; em <out>, os mesmos CSV juntos e resumo.md.

O label so entra em limiares.csv e nas colunas ap, iou, dice, precision, recall, que medem depois;
`choose` le so Q e custo. Saliente quer dizer que se destaca, nao que e o objeto. Imagem com lado
acima de `max_side` e reduzida antes de tudo (Larvae chega a 2528 px). As faixas pretas e brancas
do recorte saem antes dos mapas (`content_box`): o prior de borda leria a faixa como fundo.
"""
from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time
import warnings
import zlib
from functools import partial
from itertools import zip_longest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from saliency.fusion import (best_single, binarize, evaluate, map_features, quality_scores, regional_fusion,
                             simple_mean, top_k, weighted_mean)
from saliency.maps_a import MAPS_A
from saliency.maps_b import MAPS_B

REFERENCE = 400   # lado maior das imagens dos artigos; os defaults valem nesse tamanho
MARGIN = 0.03     # perda de Q aceita em troca de custo
LABEL_COLUMNS = ["ap", "iou", "dice", "precision", "recall"]
CANDIDATES = {"A": "melhor mapa", "B": "fusão Top-K", "C": "fusão regional", "D": "média simples"}


def segments(h: int, w: int) -> int:
    """200 superpixels do GMR, com pelo menos 64 pixels em cada (Protozoan raw desce a 48 px)."""
    return min(200, h * w // 64)


def scaled(h: int, w: int) -> dict:
    """Os 11 mapas com tudo que mede pixel proporcional ao lado; o que ja e fracao fica como esta."""
    side = max(h, w)
    s, inner_bms, inner_ss = side / REFERENCE, min(side, 400) / 400, min(side, 256) / 256
    region = {"seg_k": 50 * s**2, "seg_sigma": 0.5 * s, "seg_min_size": max(round(200 * s**2), 5),
              "border_px": max(round(15 * s), 2)}
    change = {
        "region_contrast": region, "region_contrast_tpami": region,
        "gbvs": {"scales": (min(300 / side, 1), min(150 / side, 1))},   # 600 px no artigo
        "bms": {"dilate_1": 1 + 2 * round(inner_bms), "dilate_2": 1 + 2 * round(5 * inner_bms), "blur_sigma": 20 * inner_bms},
        "mser": {"blur_sigma": max(2 * s, 0.5)},
        "selective_search": {"ks": tuple(max(round(k * inner_ss**2), 5) for k in (50, 100))},
        "gmr": {"n_segments": segments(h, w)},
    }
    return {name: partial(f, **change.get(name, {})) for name, f in (MAPS_A | MAPS_B).items()}


def content_box(img: np.ndarray) -> tuple:
    """Fatias da imagem sem as faixas lisas de borda. Decide pelos pixels saturados, sem rotulo."""
    alive = ~((img.max(2) <= 2) | (img.min(2) >= 253))

    def span(share: np.ndarray, least: int = 16) -> slice:
        """O maior trecho vivo: a costura entre faixa preta e branca tambem parece viva."""
        edges = np.flatnonzero(np.diff(np.r_[0, (share > 0.05).astype(np.int8), 0])).reshape(-1, 2)
        if not len(edges):
            return slice(None)
        start, stop = edges[np.argmax(edges[:, 1] - edges[:, 0])]
        # 2 px a mais nos lados cortados: a transicao da faixa para a imagem tem 1 a 2 px
        start, stop = start + 2 * (start > 0), stop - 2 * (stop < len(share))
        return slice(start, stop) if stop - start >= least else slice(None)

    box = [span(alive.mean(1)), span(alive.mean(0))]
    # Larva fina deixa 9 a 15 px vivos e o corte de 16 desiste. A tira fina so vale se for imagem:
    # o recorte quase todo vivo e encostando na borda no outro eixo. Objeto sobre fundo ja apagado
    # (helminth, cysts) deixa preto dentro do recorte ou flutua no meio, e fica como estava.
    for axis, share in enumerate((alive.mean(1), alive.mean(0))):
        thin, other = span(share, 8), box[1 - axis]
        inside = alive[thin, other] if axis == 0 else alive[other, thin]
        edge = other.start in (None, 0) or other.stop in (None, alive.shape[1 - axis])
        if box[axis] == slice(None) and thin != slice(None) and edge and inside.mean() > 0.95:
            box[axis] = thin
    return tuple(box)


def read_image(path: str, max_side: int) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    f = max_side / max(img.shape[:2])
    return img if f >= 1 else cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def heat(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    colour = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return cv2.addWeighted(img, 0.5, colour, 0.5, 0)


def save(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[..., ::-1] if rgb.ndim == 3 else rgb)


def candidates(maps: dict, img: np.ndarray, scores: dict, k: int) -> tuple:
    """Top-K e os quatro candidatos, com o custo proprio de B e C."""
    chosen = top_k(scores, k, maps)
    use = {name: maps[name] for name in chosen}
    start = time.perf_counter()
    b = weighted_mean(use, scores)
    middle = time.perf_counter()
    c = regional_fusion(use, img, scores, segments(*img.shape[:2]))[0]
    end = time.perf_counter()
    out = {"A": best_single(maps, scores), "B": b, "C": c, "D": simple_mean(maps)}
    return chosen, {name: np.clip(m, 0, 1) for name, m in out.items()}, middle - start, end - middle


def run_image(task: dict) -> str | None:
    """Uma imagem inteira; devolve o erro em vez de derrubar o lote."""
    out = Path(task["out"])
    if (out / "candidatos.csv").exists():
        return None
    try:
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore")
            process(task, out)
    except Exception as exc:  # noqa: BLE001
        return f"{task['path']}: {exc!r}"
    return None


def process(task: dict, out: Path) -> None:
    cv2.setNumThreads(1)
    k = task["top_k"]
    frame = read_image(task["path"], task["max_side"])
    box = content_box(frame)
    img = np.ascontiguousarray(frame[box])
    h, w = img.shape[:2]
    funcs, seconds, maps = scaled(h, w), {}, {}
    for name, f in funcs.items():
        start = time.perf_counter()
        maps[name] = f(img)
        seconds[name] = time.perf_counter() - start
    memo, spent = {}, [0.0]

    def remake(name: str, moved: np.ndarray) -> np.ndarray:
        """O pipeline inteiro numa imagem mexida, uma vez so por imagem; ali o score nao tem stability."""
        key = zlib.crc32(moved.tobytes())
        if key not in memo:
            start = time.perf_counter()
            again = {n: f(moved) for n, f in funcs.items()}
            memo[key] = again | candidates(again, moved, quality_scores(map_features(again, moved)), k)[1]
            spent[0] += time.perf_counter() - start
        return memo[key][name]

    start = time.perf_counter()
    feats = map_features(maps, img, remake)
    total = time.perf_counter() - start
    scores = quality_scores(feats)
    chosen, cands, cost_b, cost_c = candidates(maps, img, scores, k)
    every = maps | cands
    final = pd.DataFrame(map_features(every, img, remake)).T
    final["q"] = pd.Series(quality_scores(final.to_dict("index")))

    base = sum(seconds.values())
    full = base + total
    final["seconds"] = pd.Series(seconds | {"A": full, "B": full + cost_b, "C": full + cost_c, "D": base})
    table = pd.DataFrame(feats).T
    table["score"] = pd.Series(scores)
    table["rank"] = table.score.rank(ascending=False).astype(int)
    table["top_k"] = table.index.isin(chosen)
    table["selected"] = table.index == max(scores, key=scores.get)
    table["seconds"] = pd.Series(seconds)

    save(out / "original.png", img)
    np.savez_compressed(out / "mapas.npz", **every)
    for name, m in every.items():
        save(out / "mapas" / f"{name}.png", (m * 255).astype(np.uint8))
        save(out / "calor" / f"{name}.png", heat(img, m))
    for name, m in cands.items():
        for cut, mask in binarize(m).items():
            save(out / "mascaras" / f"{name}_{cut}.png", mask.astype(np.uint8) * 255)
    tag = {"dataset": task["dataset"], "image": out.name, "label_class": task["label"], "h": h, "w": w}

    if task["mask"]:
        gt = cv2.resize(cv2.imread(task["mask"], cv2.IMREAD_GRAYSCALE), frame.shape[1::-1], interpolation=cv2.INTER_NEAREST)[box] > 0
        rows = []
        if gt.any() and not gt.all():
            for name, m in every.items():
                got = evaluate(m, gt)
                final.loc[name, "ap"] = got["ap"]
                final.loc[name, ["iou", "dice", "precision", "recall"]] = pd.Series(got["thresholds"].get("otsu", {}), dtype=float)
                rows += [tag | {"name": name, "cut": cut} | v for cut, v in got["thresholds"].items()]
        pd.DataFrame(rows).to_csv(out / "limiares.csv", index=False)
    table.assign(**tag).rename_axis("name").reset_index().to_csv(out / "scores.csv", index=False)
    final.assign(**tag).rename_axis("name").reset_index().to_csv(out / "candidatos.csv", index=False)


def plan(cfg: dict) -> list:
    """Tarefas por base: uma imagem de cada classe por vez, para a amostra cobrir as classes."""
    from core.constants import ORGANIZED_ROOT
    from eval.spf_grade import MASK_DIR
    from eval.l1_sem_rotulo import train_pool
    tasks = []
    for dataset in cfg["datasets"]:
        pool = train_pool({"dataset_name": dataset, "split": cfg["split"], "percentage": cfg["percentage"]})
        groups = {}
        for sample in pool:
            groups.setdefault(sample.label, []).append(sample)
        rng = random.Random(cfg["seed"])
        for key in sorted(groups):
            rng.shuffle(groups[key])
        turns = [s for row in zip_longest(*(groups[key] for key in sorted(groups))) for s in row if s]
        for sample in turns[:cfg["images"]]:
            name = sample.image_path.name
            mask = Path(ORGANIZED_ROOT) / dataset / MASK_DIR[dataset] / name if dataset in MASK_DIR else None
            tasks.append({"dataset": dataset, "path": str(sample.image_path), "label": sample.label,
                          "mask": str(mask) if mask and mask.exists() else None, "top_k": cfg["top_k"],
                          "max_side": cfg["max_side"], "out": str(Path(cfg["out"]) / dataset / Path(name).stem)})
    return tasks


def choose(table: pd.DataFrame) -> tuple:
    """(melhor Q, o mais barato a MARGIN dele). So Q e custo: sem label. A e o maximo por imagem do
    proprio score, logo fica acima de todo mapa fixo por construcao: a margem mede perda, nao ruido."""
    means = table.groupby("name")[["q", "seconds"]].mean()
    return means.q.idxmax(), means[means.q >= means.q.max() - MARGIN].seconds.idxmin()


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    frame = frame.reset_index()
    head = "| " + " | ".join(map(str, frame.columns)) + " |\n|" + "---|" * frame.shape[1]
    rows = ["| " + " | ".join(f"{v:.{digits}f}".replace(".", ",") if isinstance(v, float) else str(v) for v in r) + " |"
            for r in frame.itertuples(index=False)]
    return "\n".join([head, *rows])


def summary(cfg: dict) -> None:
    """Junta os CSV por imagem e escreve resumo.md."""
    root = Path(cfg["out"])
    joined = {}
    for name in ("scores", "candidatos", "limiares"):
        parts = [pd.read_csv(p) for p in root.glob(f"*/*/{name}.csv") if p.stat().st_size > 1]
        joined[name] = pd.concat(parts, ignore_index=True)
        joined[name].to_csv(root / f"{name}.csv", index=False)
    table = joined["candidatos"]
    table["coerencia"] = 1 - table.fragmentation
    measures = ["q", "stability", "continuity", "coerencia", "seconds"]
    text = [f"{table.groupby('dataset').image.nunique().sum()} imagens.", "", "## Interno, sem label", ""]
    four = table[table.name.isin(CANDIDATES)]
    text += [markdown(four.groupby("name")[measures].mean()), "", markdown(four.groupby(["dataset", "name"])[measures].mean()), ""]
    text += ["## Mapas fixos", "", markdown(table[~table.name.isin(CANDIDATES)].groupby("name")[measures].mean().sort_values("q", ascending=False)), ""]
    picked = joined["scores"]
    text += ["## Vezes selecionado e no Top-K", "", markdown(picked.groupby(["dataset", "name"])[["selected", "top_k"]].mean().unstack(0).selected), "",
             markdown(picked.groupby(["dataset", "name"]).top_k.mean().unstack(0)), ""]
    rows = [{"base": "todas"} | dict(zip(["melhor Q", "mais barato na margem"], choose(table)))]
    rows += [{"base": d} | dict(zip(["melhor Q", "mais barato na margem"], choose(g))) for d, g in table.groupby("dataset")]
    text += ["## Escolha pelo critério interno", "", markdown(pd.DataFrame(rows).set_index("base")), ""]
    labelled = table[table.ap.notna()]
    text += ["## Com label: só avaliação", "", markdown(labelled[labelled.name.isin(CANDIDATES)].groupby("name")[LABEL_COLUMNS].mean()), "",
             markdown(labelled[labelled.name.isin(CANDIDATES)].groupby(["dataset", "name"])[LABEL_COLUMNS].mean()), "",
             markdown(labelled[~labelled.name.isin(CANDIDATES)].groupby("name")[LABEL_COLUMNS].mean().sort_values("ap", ascending=False)), ""]
    curve = joined["limiares"]
    curve = curve[curve.name.isin(CANDIDATES) & curve.cut.str.startswith("t")]
    text += ["## Precisão e revocação por limiar (com label)", "", markdown(curve.groupby(["cut", "name"])[["precision", "recall"]].mean().unstack(1)), ""]
    (root / "resumo.md").write_text("\n".join(text), encoding="utf-8")
    montage(table, cfg)


def montage(table: pd.DataFrame, cfg: dict) -> None:
    """Por base: o melhor, o mediano e o pior caso de C, por AP onde ha label, por Q onde nao ha."""
    Path(cfg["figures"]).mkdir(parents=True, exist_ok=True)
    tile = 200
    for dataset, part in table[table.name == "C"].groupby("dataset"):
        by = "ap" if part.ap.notna().any() else "q"
        order = part.dropna(subset=[by]).sort_values(by, ascending=False).image.tolist()
        rows = []
        for image in (order[0], order[len(order) // 2], order[-1]):
            folder = Path(cfg["out"]) / dataset / image
            cells = [folder / "original.png"] + [folder / "calor" / f"{name}.png" for name in CANDIDATES]
            rows.append(cv2.hconcat([cv2.resize(cv2.imread(str(c)), (tile, tile), interpolation=cv2.INTER_AREA) for c in cells]))
        sheet = cv2.vconcat(rows)
        for i, text in enumerate(["original", *CANDIDATES]):
            cv2.putText(sheet, text, (i * tile + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(Path(cfg["figures"]) / f"{dataset}.png"), sheet)


def main(config: str) -> None:
    with open(config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    tasks = plan(cfg)
    print(len(tasks), "imagens", flush=True)
    with mp.get_context("spawn").Pool(cfg["workers"]) as pool:
        for i, error in enumerate(pool.imap_unordered(run_image, tasks), 1):
            if error:
                print("ERRO", error, flush=True)
            if i % 10 == 0:
                print(time.strftime("%H:%M:%S"), i, "de", len(tasks), flush=True)
    summary(cfg)


if __name__ == "__main__":
    main(sys.argv[1])
