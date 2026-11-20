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
"""spf_grade.py — SPiFiL contra aleatorio sem backprop: cada comparacao isola o conteudo do kernel.

Um job = (dataset, split, tamanho, fatia), `source: spf_grade`. Celula = (braco, larguras, kernel,
sorteio). `pool` e um numero ou a lista por camada. Todo braco entrega so DIRECOES: linhas de norma 1 no espaco z-score. A dobra e a mesma
para todos: K = u / desvio, b = -K . media, com media e desvio das sementes medoide das mesmas
imagens, nos mapas do proprio braco. Quem muda entre bracos e so o conteudo de u.

Braco = `metodo@pontos` (sem `@`, medoide). Pontos: medoid, centroid, geodesic (`flim/seed_points`),
sift_sp (keypoint mais forte de cada superpixel), sift (os mais fortes da imagem), sift_nms (NMS
espacial antes do corte), uniao (medoide + sift_sp), obj e fundo (medoides de variancia local alta
e baixa), sal (superpixel que contrasta com a borda), l1obj (energia da propria L1). Tudo isso sem
mascara. Superpixel, sementes, patches e `grow` vem do SPiFiL; nada dele muda.

Com `amostra.saliency` ({map, cuts}), um mapa de `saliency/` entra sem rotulo: `mapa_<corte>` e
`nms_<corte>` deixam so as sementes medoide e sift_nms acima do corte; `mapa_peso` e `nms_peso`
sorteiam os patches com peso na saliencia, e `fpsw` multiplica a distancia do FPS por ela. Com
`saliency.cache`, o mapa sai de `<cache>/<base>/<tamanho>/<nome>.npy` em vez de ser calculado. Com `saliency.pesos`
({t, g, foco}), o peso e w = clip((S - t) / (1 - t), 0, 1) ** g: `mapa_w<t>_<g>` pesa as sementes medoide e
`foco_w<t>_<g>`, `foco_<corte>` sorteiam, dos `foco` superpixels da imagem, tantos pontos quanto o medoide tem.
Com `pesos.gauss` (lista de k), o peso vem de uma gaussiana com o centro e a covariancia da propria saliencia,
alargada k vezes: forte no objeto, caindo suave ate a borda, sem corte (`mapa_gauss<k>`, `foco_gauss<k>`).

Com `amostra.combo` ({words}), o ponto `combo` e a uniao de geodesic, centroid, sift_nms, obj e `dic`
(o dicionario SIFT: um keypoint real por palavra, sem rotulo); `combo_sg`, `_sc`, `_sn`, `_so`, `_sd`
tiram um ingrediente. Com `amostra.entre_camadas`, cada mapa e branqueado por canal, com shrinkage, e
isso dobra na conv seguinte: o kernel nao e branqueado; a leitura sem `pw_whiten` sai como `<nome>_b`.
Com `amostra.soma`, entra a leitura `soma`: as camadas somadas depois de L2 em cada uma.

Com `amostra.agrupamento` ({eps, rows}), o job inteiro, controles inclusive, troca o z-score comum pelo do
proprio patch, banda a banda, e a dobra por K = u / desvio medio de cada banda, b = 0: ver `PatchCtx`. Grupo vira
kernel em `bank`, com medoide de verdade; `gr` e `gr_m` agrupam por Louvain (`graph_groups`), `hib` e `hib_m`
por patch e descritor juntos (so `@sift*`). Superpixel SLIC vem de `l1_grow.superpixels: slic`.

Com `amostra.multiescala`, entram os pontos de `flim/scale_space` (um candidato por ponto, sem rotulo):
`s1` e `s2` sao o centro geodesico com o patch do nivel de Gauss ou DoG sorteado na 1a oitava; `s3` o patch
cru no extremo DoG, `s4` o mapa DoG e `s5` a regiao proporcional a escala, nas MESMAS posicoes; `s6` o centro
geodesico com regioes de 1, 2 e 4 kernels, que `trio` leva juntas; `s7` o patch cru no extremo LoG; `s8` junta
geodesico, sift_nms, s3, s7, s5 e s6, e `quota` da a mesma cota a cada um. Extremos: ate 1 por superpixel,
com o NMS de `sift_nms`. Esses mapas sao por camada: ver `flim/scale_space.py`.

SOL, tudo desligado por padrao. Variante viaja no nome do braco, `metodo@pontos~variante`: `amostra.norm`
([n0, global, canal, mad]: N2 um escalar, N3 um por banda, N7 mediana e MAD por dimensao), `amostra.ativacao`
([relu, abs, leaky, gelu, crelu, crelu_h]: A2, A3, GELU, A1 com o dobro de canais, A1 com metade das direcoes e o mesmo
numero de canais) e `amostra.entre_camadas` em lista (o lambda do whitening, `~w<lambda>`). F2-F6 e `kmed` sao
metodos (ver `functional`). `amostra.plano_tol` em lista (`~pl<tol>`, 0 = sem tag) tira o candidato cuja janela
LAB da L1 varia menos que `tol` (`local_std`), em toda camada. `amostra.leituras` troca a leitura GeM por uma lista de `POOLS`; `gem` sai como `L` e `cat`.
`vlad<K>` (so na ultima camada): codebook k-means sem rotulo no treino, sai como `vlad<K>_eq` (PCA a dimensao de `cat`) e
`vlad<K>` cru se abaixo de MAX_DIM.

ORACULO DE DIAGNOSTICO, nunca modelo: `m_obj`, `m_fundo` e `m_nms` filtram as sementes pelo label, e
`fundo: zerado` apaga o fundo pelo label em tudo que le imagem. Servem so para medir quanto o
fundo custa. As linhas saem com `family: oraculo`, ou `fundo: zerado`. O `_diag.csv` traz a fracao
de sementes em objeto por tipo de ponto; o label so entra ali, nessas duas portas e nos flags.

Largura real diferente da pedida (PCA acima do posto, grupos sem numero fixo) chama os controles
`twin_controls` nessas larguras exatas: o par sempre tem a mesma largura real. Leitura: GeM +
whitening + SVM em cada camada (`L`) e nas camadas concatenadas abaixo de 2000-d (`cat`).
So treino e validacao: o teste nem sobe. O SVM roda em threads, sem `svm.lock`: quem limita a
carga e `max_per_gpu` x `svm_workers` do YAML.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import random
import resource
import sys
import tempfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from pathlib import Path
from types import SimpleNamespace

import cv2
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from skimage.filters import threshold_otsu
from skimage.io import imread
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from spifil.color import LabNorm
from spifil.config import LayerSpec
from spifil.superpixels.disf import DISF
from spifil.superpixels.slic import SLIC
from spifil.types import Seeds
from torch.nn import init

from core.constants import DATASET_NUM_CLASSES, ORGANIZED_ROOT
from core.data.loaders import ift_lab_loader
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import LAB_BLACK, build_test, letterbox
from core.metrics import compute_metrics
from core.svm import fit_svm
from eval import l1_fast, l1_sem_rotulo
from eval.l1_amostras import Spy, features, grow_layer, kernel_stats
from eval.l1_fast import COPY_COS, load_images
from eval.l1_sem_rotulo import (CLUSTER_ROWS, FLAT, K_GRID, MIN_GROUP, QUADRATIC_ROWS, cos_prune, fps,
                                patch_pool, qr_pivot, reduced, representatives, train_pool, uniform)
from eval.svm import DEVICE, cli_kwargs, pw_whiten
from experiments.ray.runners.growth import _paths, _short
from experiments.ray.schema import check_encdeep
from flim import spifil
from flim.arch import parse_architecture
from flim.build import build
from flim.filter_diversity import greedy, k_medoids, low_redundancy
from flim.scale_space import DOG, GAUSS, LEVELS, RAW, REGION, scale_space, view_pool
from flim.seed_points import SINGLE, SeedPoints
from saliency.batch import content_box, scaled
from saliency.fusion import binarize

l1_sem_rotulo.MAX_WIDTH = 4096   # o teto de 256 grupos era da grade antiga
SURVIVORS = os.environ.get("SPF_SURVIVORS", "/exps/moliveira/queues/spf_sobreviventes.json")
READOUT = "gem_pw"
MAX_DIM = 2000       # leitura abaixo disso
STDEV_FLOOR = 0.01   # o `stdev_factor` de `build_filters`
ZCA_EPS = 1e-2       # o eps relativo do `GrowOneLayer.whiten`
EIGH_MAX = 8192      # acima disso, so os LOW_RANK maiores autovetores
LOW_RANK = 1024
BG_KERNEL = 5
BG_STEP = 8
L1_BANK = 64         # banco provisorio de `l1obj`
RING = 2             # espessura da borda do conteudo em `sal`
WEIGHT_FLOOR = 1e-3  # saliencia zero ainda entra no sorteio, por ultimo
SHRINK = 0.1         # o shrinkage de `pw_whiten`
WHITEN_ROWS = 8192   # patches por bloco na GPU
COMBO = {"geo": "geodesic", "cen": "centroid", "nms": "sift_nms", "obj": "obj", "dic": "dic"}
DIC_WEIGHT = 1e6     # poucas linhas do dicionario: sem isso o corte de CLUSTER_ROWS as perde
SCALE_CHUNK = 100    # imagens por lote na deteccao de extremos
TRIO = (0, 1, 2)     # `s6`: regioes de 1, 2 e 4 kernels de lado
EXTRA = ["val_accuracy", "gpu_mib", "rss_mib"]  # so com `multiescala`: CSV antigo nao ganha coluna
LEI = ["lei", "lei_args", "depth_req", "pools", "params", "macs", "maps", "rf", "width_req", "short", "short_why"]  # so com `leis`
SOL = ["raw_dim"]    # so com `leituras`: dimensao antes da projecao
PROBE_ROWS = 4096    # janelas de treino em que F2-F6 medem a resposta
MAD = 1.4826         # MAD vira desvio na normal
LEAK = 0.1
SCALES = (0.75, 0.5)  # `ms`: escalas da imagem alem da original
COV_DIM = 48         # `cov`: canais apos a PCA, 48 * 49 / 2 = 1176-d
COV_FIT = 512        # imagens de treino que fitam a PCA de canais
VLAD_PTS = 32        # `vlad<K>`: posicoes sorteadas por imagem de treino que fitam o codebook
CLUSTER = ["width_req", "short", "pts_mean", "pts_min", "live_rows", "used_rows", "flat_patches", "flat_bands",
           "comm_natural", "resolution", "size_min", "size_med", "size_max", "disp", "kernel_bands",
           "kernel_drops"]  # so com `agrupamento` ou `saliency.pesos`
NEIGHBORS = 10       # vizinhos de cada no em `graph_groups`
MEDOID_ROWS = 1024   # linhas por bloco de distancias em `medoid`
IMPURITY = {"Eggs": 8, "Protozoan": 6, "Larvae": 1, "Eggs_subgroup": 8, "Protozoan_subgroup": 6, "Larvae_subgroup": 1}
# `config.py` nega `masks_dir` de proposito: a mascara nao entra no modelo. Porta so do diagnostico.
MASK_DIR = {"Eggs": "raw/label", "Protozoan": "raw/label", "Larvae": "raw/label", "Eggs_subgroup": "masks",
            "Protozoan_subgroup": "masks", "Larvae_subgroup": "masks"}
cv2.setNumThreads(int(os.environ.get("OPENCV_FOR_THREADS_NUM", 1)))  # sem isso o OpenCV abre um thread por nucleo em cada job: carga passou de 200
SIFT = cv2.SIFT_create(contrastThreshold=0.02)
FIELDS = ["dataset", "split", "percentage", "image_size", "n_superpixels", "fundo", "arm", "family", "plan",
          "kernel", "seed", "layer", "readout", "width", "widths", "images", "candidates", "bg_src",
          "bg_sel", "groups", "top_group", "noise_frac", "copies_frac", "cos_mean", "erank", "dim",
          "n_train", "n_val", "val_kappa", "val_acc", "val_f1", "imp_recall", "par_recall", "imp_kappa",
          "par_kappa", "fit_status", "select_s", "extract_s", "fit_s", "status", "error"]
DIAG = ["dataset", "split", "image_size", "fundo", "seed", "point", "rows", "per_image", "obj_frac",
        "obj_frac_imp", "obj_frac_par", "obj_recall", "area_frac", "area_imp", "area_par", "bg_frac"]


@dataclass
class Source:
    """Sementes de cada imagem; `desc`, `bg` e `rows` por linha do pool, na ordem das sementes."""
    seeds: list
    desc: np.ndarray | None = None
    bg: np.ndarray | None = None
    rows: np.ndarray | None = None
    obj: np.ndarray | None = None
    weight: np.ndarray | None = None
    keys: list | None = None        # por imagem, (tipo, oitava, nivel) de cada semente: `flim/scale_space`
    group: np.ndarray | None = None  # por linha: ponto em `trio`, ingrediente em `quota`


@lru_cache(maxsize=None)
def mask_of(dataset: str, name: str, size: int) -> torch.Tensor:
    """O label na grade do letterbox. So para diagnostico."""
    mask = imread(Path(ORGANIZED_ROOT) / dataset / MASK_DIR[dataset] / name)
    mask = mask[..., 0] if mask.ndim == 3 else mask
    return letterbox(torch.from_numpy((mask > 0).astype(np.uint8) * 255)[None], size, (0,))[0] > 127


class Oracle(Spy):
    """Tira do ranking oficial as sementes do lado errado do label. So para diagnostico."""
    side = None
    place = ("", 0)

    def run(self, learn) -> None:
        if Oracle.side is not None:
            state = learn.state[learn.n_layers - 1]
            patches, keep = state.patches, torch.zeros(len(state.patches), dtype=torch.bool)
            images, rows = patches.image_ids.cpu(), patches.seed_rows.cpu()
            for i in images.unique().tolist():
                mine = images == i
                seeds = state.seeds[i]
                dataset, size = Oracle.place
                at = seeds.coords[rows[mine]] * size // torch.tensor(seeds.grid)
                inside = mask_of(dataset, learn.data[i].image_path.name, size)[at[:, 0], at[:, 1]]
                keep[mine] = inside == Oracle.side
            state.ranks = state.ranks * keep.to(state.ranks.device)
        super().run(learn)


spifil.GrowOneLayer = Oracle


def keypoints(gray: np.ndarray) -> tuple:
    """Pixels distintos com keypoint, do mais forte ao mais fraco, e o descritor de cada um."""
    found, desc = SIFT.detectAndCompute(gray, None)
    if not found:
        return np.zeros((0, 2), int), np.zeros((0, 128), np.float32)
    order = np.argsort([-k.response for k in found], kind="stable")
    points = np.array([found[i].pt[::-1] for i in order]).round().astype(int)
    points = points.clip(0, np.array(gray.shape) - 1)
    first = np.sort(np.unique(points, axis=0, return_index=True)[1])
    return points[first], desc[order][first]


def nms(points: np.ndarray, radius: float, top: int) -> np.ndarray:
    """Na ordem de forca, fica quem esta a mais de `radius` pixels de todo mantido."""
    kept = []
    for i, point in enumerate(points):
        if all(((points[j] - point) ** 2).sum() >= radius**2 for j in kept):
            kept.append(i)
            if len(kept) == top:
                break
    return np.array(kept, int)


def image_points(prepared, top: int) -> dict:
    """Candidatos de uma imagem por tipo de ponto: `(coordenadas, descritores ou None)`."""
    labels, feats = prepared.superpixel_labels, prepared.features
    out = {mode: (SeedPoints(mode)(labels, feats), None) for mode in SINGLE}
    points, desc = keypoints((feats[0].clip(0, 1) * 255).astype(np.uint8))
    strongest = np.sort(np.unique(labels[points[:, 0], points[:, 1]], return_index=True)[1])
    spaced = nms(points, (labels.size / top) ** 0.5 / 2, top)
    out |= {"sift_sp": (points[strongest], desc[strongest]), "sift": (points[:top], desc[:top]),
            "sift_nms": (points[spaced], desc[spaced])}
    out["uniao"] = (np.unique(np.concatenate([out["medoid"][0], points[strongest]]), axis=0), None)
    return out


def local_std(rows: torch.Tensor) -> torch.Tensor:
    """Variacao dentro do patch, somada nas 3 bandas LAB."""
    return rows.view(len(rows), 3, -1).std(2).pow(2).sum(1).sqrt()


def background_cut(lab: list) -> float:
    """Otsu no log da variacao local de janelas densas: separa fundo de borda, sem mascara."""
    rows = F.unfold(torch.stack(lab), BG_KERNEL, stride=BG_STEP).transpose(1, 2).flatten(0, 1)
    spread = local_std(rows).cpu().numpy()
    return float(np.exp(threshold_otsu(np.log(spread[spread > FLAT]))))


def split_otsu(values: np.ndarray) -> np.ndarray:
    """Metade de cima pelo Otsu da propria imagem; valores iguais nao separam nada."""
    return values > threshold_otsu(values) if np.ptp(values) > 0 else np.ones(len(values), bool)


def salient(prepared, points: np.ndarray) -> np.ndarray:
    """Sem mascara: superpixel cuja cor media foge da cor da borda do conteudo (prior de fundo)."""
    feats, labels = prepared.features, prepared.superpixel_labels
    content = feats[0] > 1e-3
    if not content.any():
        return np.ones(len(points), bool)
    ys, xs = np.flatnonzero(content.any(1)), np.flatnonzero(content.any(0))
    ring = np.zeros_like(content)
    ring[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] = True
    ring[ys[0] + RING:ys[-1] + 1 - RING, xs[0] + RING:xs[-1] + 1 - RING] = False
    ring &= content
    reference = np.median(feats[:, ring if ring.any() else content], axis=1)
    flat = labels.ravel()
    sums = np.stack([np.bincount(flat, band.ravel()) for band in feats], 1)
    color = sums / np.maximum(np.bincount(flat), 1)[:, None]
    return split_otsu(np.linalg.norm(color[labels[points[:, 0], points[:, 1]]] - reference, axis=1))


def stored_map(spec: dict, dataset: str, size: int, name: str, shape: tuple) -> np.ndarray:
    """Mapa pronto do cache, na mesma regiao de conteudo que o `gmr` receberia; so leitura."""
    where = Path(spec["cache"]) / dataset / str(size) / f"{name}.npy"
    if not where.exists():
        raise FileNotFoundError(f"saliency.cache: falta {where}")
    m = np.load(where)
    if m.shape != shape:
        raise ValueError(f"saliency.cache: {where} tem forma {m.shape}, o conteudo da imagem tem {shape}")
    return m.astype(np.float32)


def saliency_grid(path: Path, size: int, spec: dict, dataset: str) -> dict:
    """Mapa na grade do letterbox e suas mascaras, feitos sem as faixas de borda; sem rotulo."""
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    f = size / max(h, w)
    img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    box = content_box(img)
    inner = np.ascontiguousarray(img[box])
    m = stored_map(spec, dataset, size, path.stem, inner.shape[:2]) if "cache" in spec else scaled(*inner.shape[:2])[spec["map"]](inner)
    cuts = binarize(m)

    def grid(x: np.ndarray) -> np.ndarray:
        frame = np.zeros(img.shape[:2], np.float32)
        frame[box] = x
        return letterbox(torch.from_numpy(frame)[None], size, (0,))[0].numpy()

    return {"peso": grid(m)} | {c: grid(cuts.get(c, np.zeros_like(m))) > 0.5 for c in spec["cuts"]}


def soft(s: np.ndarray, t: float, g: float) -> np.ndarray:
    """`t` = 0 e peso puro, `t` maior zera o fundo, `g` grande chega perto de mascara rigida."""
    return np.clip((s - t) / (1 - t), 0, 1) ** g


def gauss(s: np.ndarray, k: float) -> np.ndarray:
    """Gaussiana ajustada aos momentos de `s` (centro e covariancia), com a covariancia multiplicada por k^2."""
    p = np.indices(s.shape).reshape(2, -1).T.astype(np.float64)
    w = s.reshape(-1).astype(np.float64) / max(float(s.sum()), 1e-12)
    d = p - w @ p
    cov = (d * w[:, None]).T @ d + np.eye(2)
    m = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
    return np.exp(-m / (2 * k * k)).reshape(s.shape).astype(np.float32)


def add_weights(job: dict, data, lab: list, found: dict, grids: list, masks: list | None, edge: float) -> None:
    """`mapa_w*` e `foco_*`, ver o docstring. Pontos sem peso nao entram: a imagem pode ficar com menos."""
    spec = job["amostra"]["saliency"]["pesos"]
    medoid, grid = found["medoid"], tuple(lab[0].shape[1:])
    layer = LayerSpec(kernel_size=BG_KERNEL, out_channels=1)
    dense = [spifil.preparation.prepare_sample(
        data, i, superpixels=segmenter(job), seed_extractor=SeedPoints("medoid"), n_superpixels=spec["foco"],
        color=LabNorm()).seeds.coords for i in range(len(lab))]
    rng = np.random.default_rng(job["draw"])
    tickets = [rng.exponential(size=len(c)) for c in dense]

    def on(key: str) -> list:
        return [g[key][c[:, 0].numpy(), c[:, 1].numpy()].astype(np.float32) for g, c in zip(grids, dense)]

    def focused(weights: list) -> Source:
        """Sorteio sem reposicao proporcional ao peso (Efraimidis & Spirakis), o mesmo bilhete em todo peso."""
        coords = []
        for c, ticket, w, seeds in zip(dense, tickets, weights, medoid.seeds):
            live = np.flatnonzero(w > 0)
            coords.append(c[live[np.argsort(ticket[live] / w[live])[:len(seeds)]]])
        seeds = [Seeds.from_coords(c, 0, grid) for c in coords]
        pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
        obj = np.concatenate([m[c[:, 0], c[:, 1]].numpy() for m, c in zip(masks, coords)]) if masks else None
        return Source(seeds, None, (local_std(pool.feats) < edge).numpy(), None, obj)

    sal = np.concatenate([g["peso"][s.coords[:, 0].numpy(), s.coords[:, 1].numpy()] for g, s in zip(grids, medoid.seeds)])
    near = on("peso")
    for t, g in itertools.product(spec["t"], spec["g"]):
        w = soft(sal, t, g)
        found[f"mapa_w{t:g}_{g:g}"] = replace(medoid, weight=w, rows=w > 0 if t else None)
        found[f"foco_w{t:g}_{g:g}"] = focused([soft(s, t, g) for s in near])
    for cut in job["amostra"]["saliency"]["cuts"]:
        found[f"foco_{cut}"] = focused(on(cut))
    for k in spec.get("gauss", []):
        maps = [gauss(gr["peso"], k) for gr in grids]
        w = np.concatenate([x[s.coords[:, 0].numpy(), s.coords[:, 1].numpy()] for x, s in zip(maps, medoid.seeds)])
        found[f"mapa_gauss{k:g}"] = replace(medoid, weight=w)
        found[f"foco_gauss{k:g}"] = focused([x[c[:, 0].numpy(), c[:, 1].numpy()] for x, c in zip(maps, dense)])


def add_saliency(job: dict, samples: list, found: dict, masks: list | None, data, lab: list, edge: float) -> None:
    """Os pontos `mapa_*` e `nms_*`, e `m_nms`, o oraculo de `crus@nms_*`."""
    spec = job["amostra"]["saliency"]
    grids = [saliency_grid(s.image_path, job["image_size"], spec, job["dataset_name"]) for s in samples]
    for tag, base in (("mapa", found["medoid"]), ("nms", found["sift_nms"])):
        def at(key: str, seeds: list = base.seeds) -> np.ndarray:
            return np.concatenate([g[key][s.coords[:, 0].numpy(), s.coords[:, 1].numpy()] for g, s in zip(grids, seeds)])
        found[f"{tag}_peso"] = replace(base, weight=at("peso"))
        found |= {f"{tag}_{cut}": replace(base, rows=at(cut)) for cut in spec["cuts"]}
    if masks:
        found["m_nms"] = replace(found["sift_nms"], rows=found["sift_nms"].obj)
    if "pesos" in spec:
        add_weights(job, data, lab, found, grids, masks, edge)


def coords_of(src: Source) -> list:
    """As coordenadas de cada imagem, ja com o filtro `rows`."""
    sizes = [len(s) for s in src.seeds]
    rows = np.ones(sum(sizes), bool) if src.rows is None else src.rows
    return [s.coords[torch.from_numpy(m)] for s, m in zip(src.seeds, np.split(rows, np.cumsum(sizes)[:-1]))]


def add_combo(job: dict, lab: list, found: dict, masks: list | None, cut: float) -> None:
    """`dic`, `combo` e os `combo_s*`, que tiram um ingrediente; a mascara so entra no diagnostico."""
    x = unit(found["sift"].desc.astype(np.float64))
    words = np.zeros(len(x), bool)
    words[representatives(x, fixed_groups(x, job["amostra"]["combo"]["words"], 0))[0]] = True
    found["dic"] = replace(found["sift"], rows=words)
    coords = {key: coords_of(found[name]) for key, name in COMBO.items()}
    layer, grid = LayerSpec(kernel_size=BG_KERNEL, out_channels=1), tuple(lab[0].shape[1:])
    for name, left in [("combo", list(COMBO))] + [(f"combo_s{key[0]}", [k for k in COMBO if k != key]) for key in COMBO]:
        seeds = [Seeds.from_coords(torch.cat(c).unique(dim=0), 0, grid) for c in zip(*[coords[k] for k in left])]
        pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
        obj = np.concatenate([m[s.coords[:, 0], s.coords[:, 1]].numpy() for m, s in zip(masks, seeds)]) if masks else None
        weight = None
        if "dic" in left:
            weight = np.concatenate([np.where((s.coords[:, None] == d[None]).all(-1).any(1).numpy(), DIC_WEIGHT, 1.0)
                                     for s, d in zip(seeds, coords["dic"])])
        found[name] = Source(seeds, None, (local_std(pool.feats) < cut).numpy(), None, obj, weight)


def scale_points(lab: list, top: int) -> tuple:
    """Por imagem, extremos DoG e LoG do mais forte ao mais fraco, um por pixel, com o NMS de `sift_nms`."""
    stack, records, out = torch.stack([x[:1] for x in lab]), {}, {"dog": [], "log": []}
    radius = (stack[0].numel() / top) ** 0.5 / 2
    for part in stack.split(SCALE_CHUNK):
        for name, found in scale_space(part, records).items():
            owner = found.image.cpu().numpy()
            yx = torch.stack([found.y, found.x], 1).cpu().numpy()
            scale = torch.stack([found.octave, found.level], 1).cpu().numpy()
            for i in range(len(part)):
                points, where = yx[owner == i], scale[owner == i]
                if len(points):
                    first = np.sort(np.unique(points, axis=0, return_index=True)[1])
                    kept = first[nms(points[first], radius, top)]
                    points, where = points[kept], where[kept]
                out[name].append((torch.from_numpy(points), where))
                for octave, level in where.tolist():
                    records[name, octave, level]["kept"] += 1
    return out, records


def add_scale(job: dict, samples: list, lab: list, found: dict, masks: list | None, cut: float) -> None:
    """Os pontos `s1` a `s8`, ver o docstring, e o registro das escalas; a mascara so entra no diagnostico."""
    grid, rng = tuple(lab[0].shape[1:]), np.random.default_rng(job["draw"])
    points, records = scale_points(lab, job["superpixels"][0])
    layer = LayerSpec(kernel_size=BG_KERNEL, out_channels=1)

    def source(coords: list, keys: list, group: np.ndarray | None = None) -> Source:
        seeds = [Seeds.from_coords(c, 0, grid) for c in coords]
        pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
        obj = np.concatenate([m[s.coords[:, 0], s.coords[:, 1]].numpy() for m, s in zip(masks, seeds)]) if masks else None
        return Source(seeds, None, (local_std(pool.feats) < cut).numpy(), None, obj, keys=keys, group=group)

    def tagged(kind: int, scales: list) -> list:
        return [np.column_stack([np.full(len(s), kind), s]) for s in scales]

    def plain(coords: list) -> list:
        return tagged(RAW, [np.zeros((len(c), 2), int) for c in coords])

    geo, sift_at = coords_of(found["geodesic"]), coords_of(found["sift_nms"])
    dog_at, dog_scale = map(list, zip(*points["dog"]))
    log_at = [c for c, _ in points["log"]]
    drawn = [np.column_stack([np.zeros(len(c), int), rng.integers(LEVELS + 1, size=len(c))]) for c in geo]
    picks = [rng.permutation(len(c))[:max(1, CLUSTER_ROWS // (len(TRIO) * len(lab)))] for c in geo]
    trio_at = [c[p].repeat_interleave(len(TRIO), 0) for c, p in zip(geo, picks)]
    trio_keys = tagged(REGION, [np.tile([[octave, 0] for octave in TRIO], (len(p), 1)) for p in picks])
    fixed = {"s1": (geo, tagged(GAUSS, drawn)), "s2": (geo, tagged(DOG, drawn)), "s3": (dog_at, plain(dog_at)),
             "s4": (dog_at, tagged(DOG, dog_scale)), "s5": (dog_at, tagged(REGION, dog_scale)), "s7": (log_at, plain(log_at))}
    found |= {name: source(*pair) for name, pair in fixed.items()}
    found["s6"] = source(trio_at, trio_keys, np.repeat(np.arange(sum(map(len, picks))), len(TRIO)))
    parts = [(geo, plain(geo)), (sift_at, plain(sift_at)), fixed["s3"], fixed["s7"], fixed["s5"], (trio_at, trio_keys)]
    found["s8"] = source([torch.cat([p[0][i] for p in parts]) for i in range(len(lab))],
                         [np.concatenate([p[1][i] for p in parts]) for i in range(len(lab))],
                         np.concatenate([np.repeat(np.arange(len(parts)), [len(p[0][i]) for p in parts]) for i in range(len(lab))]))
    where = Path(job["results_csv"]).parent / "ms_escala" / f"{job['stem']}.csv"
    where.parent.mkdir(exist_ok=True)
    head = {"dataset": job["dataset_name"], "split": job["split"], "seed": job["draw"], "images": len(lab)}
    with open(where, "a", newline="") as handle:
        sheet = csv.DictWriter(handle, [*head, *next(iter(records.values()))])
        if handle.tell() == 0:
            sheet.writeheader()
        sheet.writerows(head | r for r in records.values())


@torch.no_grad()
def l1_objectness(lab: list, seeds: list, feats: torch.Tensor) -> np.ndarray:
    """Sem mascara: energia de um banco L1 provisorio na semente, cortada por imagem."""
    mean, stdev = feats.mean(0), feats.std(0, unbiased=False) + STDEV_FLOOR
    live = feats[feats.std(1) >= FLAT][::max(1, len(feats) // L1_BANK)][:L1_BANK]
    weight = F.normalize((live - mean) / stdev, dim=1) / stdev
    bias = -(weight * mean).sum(1)
    weight = weight.view(len(live), 3, BG_KERNEL, BG_KERNEL).to(DEVICE)
    out = []
    for image, seed in zip(lab, seeds):
        energy = F.relu(F.conv2d(image[None], weight, bias.to(DEVICE), padding=BG_KERNEL // 2)).mean(1)[0]
        out.append(split_otsu(energy[seed.coords[:, 0], seed.coords[:, 1]].cpu().numpy()))
    return np.concatenate(out)


def segmenter(job: dict):
    return SLIC() if job["grow"].get("superpixels") == "slic" else DISF()


def sources(job: dict, samples: list) -> tuple:
    """Imagens LAB, os candidatos de cada tipo de ponto, o corte de fundo e o diagnostico."""
    size, no_crop, n = job["image_size"], job["no_crop"], job["superpixels"][0]
    data = spifil.Cropped(samples, size, use_mask=False, no_crop=no_crop)
    lab = spifil.encoder_features(samples, torch.nn.Identity(), size, False, DEVICE, no_crop)
    ready = [spifil.preparation.prepare_sample(
        data, i, superpixels=segmenter(job), seed_extractor=SeedPoints("medoid"), n_superpixels=n,
        color=LabNorm()) for i in range(len(samples))]
    parts = [image_points(p, n) for p in ready]
    cut, found = background_cut(lab), {}
    dataset = job["dataset_name"]
    masks = [mask_of(dataset, s.image_path.name, size) for s in samples] if dataset in MASK_DIR else None
    layer = LayerSpec(kernel_size=BG_KERNEL, out_channels=1)
    for name in parts[0]:
        seeds = [Seeds.from_coords(torch.as_tensor(p[name][0]), 0, tuple(lab[0].shape[1:])) for p in parts]
        pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
        desc = np.concatenate([p[name][1] for p in parts]) if parts[0][name][1] is not None else None
        obj = np.concatenate([m[s.coords[:, 0], s.coords[:, 1]].numpy() for m, s in zip(masks, seeds)]) if masks else None
        found[name] = Source(seeds, desc, (local_std(pool.feats) < cut).numpy(), None, obj)
        if name == "medoid":
            medoid_feats = pool.feats
    medoid = found["medoid"]
    rows = {"obj": ~medoid.bg, "fundo": medoid.bg,
            "sal": np.concatenate([salient(p, s.coords.numpy()) for p, s in zip(ready, medoid.seeds)]),
            "l1obj": l1_objectness(lab, medoid.seeds, medoid_feats)}
    if masks:
        rows |= {"m_obj": medoid.obj, "m_fundo": ~medoid.obj}
    found |= {name: Source(medoid.seeds, None, medoid.bg, keep, medoid.obj) for name, keep in rows.items()}
    if "saliency" in job["amostra"]:
        add_saliency(job, samples, found, masks, data, lab, cut)
    if "combo" in job["amostra"]:
        add_combo(job, lab, found, masks, cut)
    if job["amostra"].get("multiescala"):
        add_scale(job, samples, lab, found, masks, cut)
    if "ijepa" in job["amostra"]:
        add_ijepa(job, samples, lab, found, cut)
    return lab, found, cut, diagnosis(job, samples, found, masks)


def add_dino(job: dict, samples: list, lab: list, found: dict, cut: float) -> None:
    """`amostra.dino` ({dir, corte}): `dino_alto` nos tokens de norma alta do DINOv2-giant (`corte` >= 1 e a norma,
    < 1 a fracao do topo; ao menos 1 por imagem), `dino_baixo` nos de norma mais baixa e `dino_rand` em pixels
    sorteados, os dois com a mesma contagem de `dino_alto` por imagem. Normas de `dino_pontos.py`, fora do cache `fontes`."""
    spec, grid, rng = job["amostra"]["dino"], tuple(lab[0].shape[1:]), np.random.default_rng(job["draw"])
    layer, coords = LayerSpec(kernel_size=BG_KERNEL, out_channels=1), {"dino_alto": [], "dino_baixo": [], "dino_rand": []}
    for s in samples:
        got = np.load(Path(spec["dir"]) / job["dataset_name"] / f"{s.image_path.name}.npz")
        norm, (top, left, ch, cw) = got["norm"], got["box"]
        gh, gw = norm.shape
        order = np.argsort(-norm, axis=None)
        k = max(1, int((norm > spec["corte"]).sum()) if spec["corte"] >= 1 else round(spec["corte"] * norm.size))
        yx = lambda idx: np.column_stack([top + (idx // gw + 0.5) * ch / gh, left + (idx % gw + 0.5) * cw / gw]).astype(np.int64)
        coords["dino_alto"].append(yx(order[:k]))
        coords["dino_baixo"].append(yx(order[::-1][:k]))
        coords["dino_rand"].append(np.column_stack([top + rng.integers(0, ch, k), left + rng.integers(0, cw, k)]))
    for name, c in coords.items():
        seeds = [Seeds.from_coords(torch.from_numpy(np.unique(x, axis=0)), 0, grid) for x in c]
        pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
        found[name] = Source(seeds, None, (local_std(pool.feats) < cut).numpy(), None, None)


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def rgb_box(path: Path, size: int) -> torch.Tensor:
    """RGB uint8 (3, size, size) na mesma geometria do letterbox do LAB (lado maior = size, borda preta)."""
    img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    f = size / max(h, w)
    img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return letterbox(torch.from_numpy(img).permute(2, 0, 1), size, (0, 0, 0))


@torch.no_grad()
def add_ijepa(job: dict, samples: list, lab: list, found: dict, cut: float) -> None:
    """`amostra.ijepa` ({modelo, entrada, pesos}): grade densa sem superpixel, um candidato no centro de cada token.
    `grade` e so a grade (FPS no patch, como `fps@grade`); `ijepa` e a mesma grade com o token do I-JEPA congelado
    (L2) como descritor, para `fpsd@ijepa` (FPS no espaco do I-JEPA). `ijepa_w<t>_<g>` pesa por soft(gmr, t, g),
    para `fpsdw`. O I-JEPA so escolhe os pontos: o recorte e a dobra sao os de sempre, e a rede roda sem ele."""
    from benchmark.lista import LOADERS  # tardio: so quem pede `ijepa` carrega timm e o ViT
    spec, size, grid = job["amostra"]["ijepa"], job["image_size"], tuple(lab[0].shape[1:])
    side_in = spec.get("entrada", 224)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    net = LOADERS[spec.get("modelo", "ijepa_vith14_1k")]()[0].to(DEVICE).eval().half()
    mean = torch.tensor(IMAGENET_MEAN, device=DEVICE)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=DEVICE)[:, None, None]
    descs = []
    for start in range(0, len(samples), 32):
        x = torch.stack([rgb_box(s.image_path, size) for s in samples[start:start + 32]]).to(DEVICE).float() / 255
        x = F.interpolate(x, size=(side_in, side_in), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        tok = net.tokens(((x - mean) / std).half()).float()
        descs += list(F.normalize(tok, dim=2).cpu().numpy())
    del net
    with open(Path(job["work_dir"]) / "ijepa_pico.tsv", "a") as fh:  # pico do ViT na selecao (gpu_mib do CSV zera depois)
        fh.write(f"{job['dataset_name']}\t{job['split']}\t{torch.cuda.max_memory_allocated() >> 20}\t{time.perf_counter() - t0:.1f}\n")
    torch.cuda.empty_cache()
    g = round(len(descs[0]) ** 0.5)
    centers = ((np.arange(g) + 0.5) * size / g).astype(np.int64)
    yx = np.stack(np.meshgrid(centers, centers, indexing="ij"), -1).reshape(-1, 2)
    seeds = [Seeds.from_coords(torch.from_numpy(yx), 0, grid) for _ in samples]
    layer = LayerSpec(kernel_size=BG_KERNEL, out_channels=1)
    bg = (local_std(patch_pool(lab, seeds, torch.nn.Identity(), layer).feats) < cut).numpy()
    found["grade"] = Source(seeds, None, bg, None, None)
    found["ijepa"] = Source(seeds, np.concatenate(descs), bg, None, None)
    if spec.get("pesos"):
        sal = job["amostra"]["saliency"]
        maps = [saliency_grid(s.image_path, size, sal, job["dataset_name"])["peso"] for s in samples]
        at = np.concatenate([m[yx[:, 0], yx[:, 1]] for m in maps])
        for t, k in spec["pesos"]:
            found[f"ijepa_w{t}_{k}"] = replace(found["ijepa"], weight=soft(at, t, k))


def stored_sources(job: dict, samples: list) -> tuple:
    """Com `amostra.fontes`, `sources` roda uma vez por sorteio: ele nao depende do braco, e custa ~0,5 s por imagem."""
    if "fontes" not in job["amostra"]:
        return sources(job, samples)
    pct = "" if job["percentage"] == 50 else f"_pct{job['percentage']}"  # as fontes vem do treino da porcentagem; 50 fica sem sufixo para os caches antigos valerem
    path = Path(job["amostra"]["fontes"]) / f"{job['dataset_name']}_split{job['split']}{pct}_d{job['draw']}.pt"
    if path.exists():
        lab, *rest = torch.load(path, weights_only=False)
        return [x.to(DEVICE) for x in lab], *rest
    out = sources(job, samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save([[x.cpu() for x in out[0]], *out[1:]], temp)
    os.replace(temp, path)
    return out


def diagnosis(job: dict, samples: list, found: dict, masks: list | None) -> list:
    """Fracao de sementes em objeto por tipo de ponto, ao lado da area do objeto."""
    labels = np.array([s.label - 1 for s in samples])
    impurity = IMPURITY.get(job["dataset_name"])
    area = np.array([float(m.float().mean()) for m in masks]) if masks else None
    out = []
    for name, src in found.items():
        owner = np.repeat(np.arange(len(samples)), [len(s) for s in src.seeds])
        keep = np.ones(len(owner), bool) if src.rows is None else src.rows
        row = {"point": name, "rows": int(keep.sum()), "per_image": keep.sum() / len(samples),
               "bg_frac": float(src.bg[keep].mean())}
        if masks:
            hit, mine = src.obj[keep], labels[owner[keep]] == impurity
            row |= {"obj_frac": float(hit.mean()), "area_frac": float(area.mean()),
                    "obj_recall": float(hit.sum() / max(src.obj.sum(), 1))}
            if impurity is not None:
                row |= {"obj_frac_imp": float(hit[mine].mean()), "obj_frac_par": float(hit[~mine].mean()),
                        "area_imp": float(area[labels == impurity].mean()),
                        "area_par": float(area[labels != impurity].mean())}
        out.append(row)
    return out


def zero_background(job: dict, images: dict) -> dict:
    """Fundo preto pelo label em toda leitura de imagem. So para diagnostico."""
    size, dataset = job["image_size"], job["dataset_name"]
    black = torch.tensor(LAB_BLACK, dtype=torch.uint8).view(1, 3, 1, 1)
    out = {}
    for name, (x, y) in images.items():
        paths = ParasiteDataset(set_name=name, split=job["split"], percentage=job["percentage"],
                                path_dataset=job["dataset_name"]).samples
        with ThreadPoolExecutor(8) as pool:
            found = list(pool.map(lambda p: mask_of(dataset, os.path.basename(p), size), paths))
        out[name] = torch.where(torch.stack(found)[:, None], x, black), y
    mask_of.cache_clear()
    load, test = spifil.Cropped.load_image, build_test(size, imagenet_norm=False, no_crop=job["no_crop"])

    def load_black(self, index: int) -> np.ndarray:
        image = load(self, index).copy()
        image[~mask_of(dataset, self[index].image_path.name, size).numpy()] = 0
        return image

    @torch.no_grad()
    def features_black(samples, encoder, size, imagenet_norm, device, no_crop=False):
        fill = torch.tensor(LAB_BLACK, dtype=torch.float32).view(3, 1, 1) / 255
        reply = []
        for sample in samples:
            lab = np.ascontiguousarray(ift_lab_loader(sample.image_path))
            x = test(torch.from_numpy(lab).permute(2, 0, 1).float())
            x = torch.where(mask_of(dataset, sample.image_path.name, size)[None], x, fill)
            reply.append(encoder(x[None].to(device))[0])
        return reply

    spifil.Cropped.load_image = load_black
    spifil.encoder_features = features_black
    return out


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


class Ctx:
    """Os candidatos de uma camada no espaco z-score comum."""

    def __init__(self, feats: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, src: Source,
                 seed: int) -> None:
        self.feats, self.mean, self.stdev, self.seed = feats, mean, stdev, seed
        self.src = src
        self.audit, self.cap = {}, CLUSTER_ROWS
        self.drop = None

    def live(self, rows: np.ndarray) -> np.ndarray:
        keep = rows.std(1) >= FLAT
        if self.drop is not None:  # `~pl<tol>`: o teste acima nao ve o preto (L=0, a=b=0,5 dao desvio entre bandas)
            self.audit["flat_patches"] = int((keep & self.drop).sum())
            keep &= ~self.drop
        return keep

    @cached_property
    def keep(self) -> np.ndarray:
        """Sem janela chapada nem repetida, ate CLUSTER_ROWS linhas, em ordem sorteada."""
        rows = self.feats.numpy()
        z = (rows - self.mean.numpy()) / self.stdev.numpy()
        hashed = z @ np.random.default_rng(0).standard_normal(z.shape[1]).astype(np.float32)
        keep = np.flatnonzero(self.live(rows))
        keep = keep[np.unique(hashed[keep], return_index=True)[1]]
        self.audit["live_rows"] = len(keep)
        rng = np.random.default_rng(self.seed)
        if self.src.keys is not None:  # `s3`, `s4` e `s5` sorteiam as mesmas linhas, mesmo se uma linha cair em um deles
            order = rng.permutation(len(rows))
            return order[np.isin(order, keep)][:self.cap]
        if self.src.weight is None:
            return rng.permutation(keep)[:self.cap]
        # sorteio sem reposicao proporcional ao peso (Efraimidis & Spirakis)
        weight = self.src.weight if self.src.rows is None else self.src.weight[self.src.rows]
        ticket = rng.exponential(size=len(keep)) / weight[keep].clip(WEIGHT_FLOOR)
        return keep[np.argsort(ticket)][:self.cap]

    @cached_property
    def z(self) -> np.ndarray:
        rows = self.feats[self.keep].double()
        return unit(((rows - self.mean.double()) / self.stdev.double()).numpy())

    @cached_property
    def desc(self) -> np.ndarray:
        return unit(self.src.desc[self.src_rows].astype(np.float64))

    @cached_property
    def src_rows(self) -> np.ndarray:
        return self.keep if self.src.rows is None else np.flatnonzero(self.src.rows)[self.keep]

    @cached_property
    def spectrum(self) -> tuple:
        """Autovalores, autovetores e a variancia media dos candidatos no espaco z-score."""
        x = (self.feats.to(DEVICE) - self.mean.to(DEVICE)) / self.stdev.to(DEVICE)
        total = float(x.pow(2).sum() / (len(x) * x.shape[1]))
        if x.shape[1] <= EIGH_MAX:
            values, vectors = torch.linalg.eigh(x.T @ x / len(x))
            return values.clamp_min(0).flip(0), vectors.flip(1), total
        _, values, vectors = torch.svd_lowrank(x, q=min(LOW_RANK, *x.shape), niter=2)
        return values**2 / len(x), vectors, total

    def white(self, rows: np.ndarray, eps: float = ZCA_EPS) -> np.ndarray:
        """ZCA simetrica; fora dos autovetores guardados vale so o `eps`."""
        values, vectors, total = self.spectrum
        x, gap = torch.as_tensor(rows, dtype=torch.float32, device=DEVICE), eps * total
        gain = (values + gap) ** -0.5 - gap**-0.5
        return ((x @ vectors * gain) @ vectors.T + x * gap**-0.5).double().cpu().numpy()


def band_sd(rows: torch.Tensor, area: int) -> torch.Tensor:
    return rows.view(len(rows), -1, area).std(2, unbiased=False)


def band_z(rows: torch.Tensor, area: int, eps: float) -> torch.Tensor:
    """(P - media) / (desvio + eps), com media e desvio do proprio patch em cada banda."""
    bands = rows.view(len(rows), -1, area)
    return ((bands - bands.mean(2, keepdim=True)) / (band_sd(rows, area) + eps)[..., None]).flatten(1)


class PatchCtx(Ctx):
    """`amostra.agrupamento`: patch em Z-score proprio, banda a banda; sai so quem esta chapado em todas as bandas."""

    def __init__(self, *args, eps: float, cap: int, area: int) -> None:
        super().__init__(*args)
        self.eps, self.cap, self.area = eps, cap, area

    def live(self, rows: np.ndarray) -> np.ndarray:
        flat = (band_sd(torch.from_numpy(rows), self.area) < self.eps).numpy()
        self.audit |= {"flat_patches": int(flat.all(1).sum()), "flat_bands": "-".join(map(str, flat.sum(0)))}
        return ~flat.all(1)

    @cached_property
    def z(self) -> np.ndarray:
        return unit(band_z(self.feats[self.keep].double(), self.area, self.eps).numpy())

    def rez(self, rows: np.ndarray) -> tuple:
        """Kernels de norma 1 em Z-score de novo, medidos na escala de um patch; e as bandas chapadas de cada um."""
        scaled = torch.as_tensor(rows) * rows.shape[1] ** 0.5
        return unit(band_z(scaled, self.area, self.eps).numpy()), (band_sd(scaled, self.area) < self.eps).numpy()

    def spread(self, ref: torch.Tensor) -> torch.Tensor:
        """Desvio medio de cada banda nos patches vivos. A conv nao divide pelo desvio de cada janela: o kernel tem
        media zero em cada banda, entao a media local cai sozinha, e o desvio local vira este, igual para todo pixel."""
        sd = band_sd(ref, self.area)
        return (torch.nanmean(sd.where(sd >= self.eps, torch.nan), 0).nan_to_num(0) + self.eps).repeat_interleave(self.area)


def seen(ctx: Ctx, picks: np.ndarray, info: dict | None = None) -> tuple:
    """Linhas escolhidas, com a fracao delas que caiu em fundo."""
    return ctx.z[picks], (info or {}) | {"bg_sel": float(ctx.src.bg[ctx.src_rows[picks]].mean())}


def rowwise(pick):
    """Um seletor de `eval/l1_sem_rotulo.py` visto como metodo desta grade."""
    return lambda ctx, width: seen(ctx, *pick(ctx.z, width, ctx.seed))


def fps_weighted(ctx: Ctx, width: int, space: str = "z") -> tuple:
    """`fps` com a saliencia multiplicando a distancia: longe do banco e saliente. `fpsdw`: o mesmo no descritor."""
    weight, x = ctx.src.weight[ctx.src_rows], getattr(ctx, space)
    picks, near = [0], np.abs(x @ x[0])
    for _ in range(min(width, len(x)) - 1):
        picks.append(int(((1 - near) * weight).argmax()))
        near = np.maximum(near, np.abs(x @ x[picks[-1]]))
    return seen(ctx, np.array(picks))


def trio(ctx: Ctx, width: int) -> tuple:
    """Posicoes sorteadas, cada uma com todas as escalas: a mesma regiao entra inteira no banco."""
    group = ctx.src.group[ctx.src_rows]
    order = np.random.default_rng(ctx.seed).permutation(group.max() + 1)
    return seen(ctx, np.argsort(order[group], kind="stable")[:width])


def quota(ctx: Ctx, width: int) -> tuple:
    """`crus` com a mesma cota por ingrediente: na ordem sorteada, um de cada por vez."""
    group, turn = ctx.src.group[ctx.src_rows], np.zeros(len(ctx.keep), int)
    for name in np.unique(group):
        turn[group == name] = np.arange((group == name).sum())
    return seen(ctx, np.argsort(turn, kind="stable")[:width])


def signs(u: np.ndarray) -> np.ndarray:
    return np.concatenate([u, -u])


def zca2(ctx: Ctx, width: int, eps: float = ZCA_EPS) -> tuple:
    """Patch branqueado `d`: a conv mede d . W z(x), logo a direcao em z e W d. Com os dois sinais."""
    n = min(width // 2, len(ctx.z))
    return signs(ctx.white(unit(ctx.white(ctx.z[:n], eps)), eps)), {"bg_sel": float(ctx.src.bg[ctx.src_rows[:n]].mean())}


def pca(ctx: Ctx, width: int) -> tuple:
    return ctx.spectrum[1][:, :width].T.double().cpu().numpy(), {}


def pca2(ctx: Ctx, width: int) -> tuple:
    return signs(pca(ctx, width // 2)[0]), {}


def sil_groups(x: np.ndarray, seed: int) -> np.ndarray:
    """k-means com k pelo silhouette, sem classe nem numero de classes."""
    fits = [MiniBatchKMeans(k, n_init=3, random_state=seed).fit_predict(x) for k in K_GRID if k < len(x)]
    scores = [silhouette_score(x, g, sample_size=QUADRATIC_ROWS, random_state=seed) for g in fits]
    return fits[int(np.argmax(scores))]


def fixed_groups(x: np.ndarray, width: int, seed: int) -> np.ndarray:
    return MiniBatchKMeans(min(width, len(x)), n_init=3, random_state=seed).fit_predict(x)


def graph_groups(x: np.ndarray, width: int, seed: int) -> tuple:
    """Louvain no grafo de vizinhos, sem matriz N x N; a resolucao dobra ate passar da largura."""
    near = kneighbors_graph(x, NEIGHBORS, include_self=False)
    graph = nx.from_scipy_sparse_array(near.maximum(near.T))
    found = natural = nx.community.louvain_communities(graph, resolution=1, seed=seed)
    resolution = 1
    while len(found) <= width < len(x):
        resolution *= 2
        found = nx.community.louvain_communities(graph, resolution=resolution, seed=seed)
    groups = np.empty(len(x), int)
    for name, members in enumerate(found):
        groups[list(members)] = name
    return groups, {"comm_natural": len(natural), "resolution": resolution}


def medoid(x: np.ndarray) -> int:
    """A linha real de menor distancia media aos membros do grupo."""
    rows = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    return int(torch.cat([torch.cdist(part, rows).mean(1) for part in rows.split(MEDOID_ROWS)]).argmin())


def bank(ctx: PatchCtx, x: np.ndarray, groups: np.ndarray, mean: bool, width: int = 0) -> tuple:
    """Um kernel por grupo, do maior ao menor, ate `width`; o de variancia quase nula da lugar ao proximo grupo."""
    names, sizes = np.unique(groups[groups >= 0], return_counts=True)
    kernels, members, picks, bands, drops = [], [], [], 0, 0
    for name in names[np.argsort(-sizes, kind="stable")]:
        mine = np.flatnonzero(groups == name)
        pick = mine if mean else mine[[medoid(x[mine])]]
        kernel, flat = ctx.rez(ctx.z[pick].mean(0, keepdims=True))
        bands = bands + flat[0]
        if flat.all():
            drops += 1
            continue
        kernels.append(kernel[0])
        members.append(mine)
        picks.append(pick)
        if len(kernels) == width:
            break
    used = [len(m) for m in members]
    return np.stack(kernels), {
        "groups": len(names), "top_group": sizes.max() / len(groups), "noise_frac": float((groups < 0).mean()),
        "bg_sel": float(ctx.src.bg[ctx.src_rows[np.concatenate(picks)]].mean()),
        "size_min": min(used), "size_med": float(np.median(used)), "size_max": max(used),
        "disp": float(np.mean([np.linalg.norm(x[m] - x[m].mean(0), axis=1).mean() for m in members])),
        "kernel_bands": "-".join(map(str, bands)), "kernel_drops": drops}


def graph_arm(ctx: PatchCtx, width: int, mean: bool) -> tuple:
    x = reduced(ctx.z, ctx.seed)
    groups, info = graph_groups(x, width, ctx.seed)
    rows, more = bank(ctx, x, groups, mean, width)
    return rows, more | info


def grouped(ctx: Ctx, x: np.ndarray, groups: np.ndarray, mean: bool) -> tuple:
    """Representante real de cada grupo, ou a media do grupo como kernel."""
    if isinstance(ctx, PatchCtx):
        return bank(ctx, x, groups, mean)
    picks, info = representatives(x, groups)
    if not mean:
        return seen(ctx, picks, info)
    return np.stack([ctx.z[groups == groups[i]].mean(0) for i in picks]), info


def hybrid(ctx: Ctx, width: int, mean: bool) -> tuple:
    """Patch reduzido e descritor, cada um de norma 1, no mesmo K-means; o kernel sai do patch."""
    x = np.hstack([unit(reduced(ctx.z, ctx.seed)), ctx.desc])
    return grouped(ctx, x, fixed_groups(x, width, ctx.seed), mean)


def zca_km2(ctx: Ctx, width: int) -> tuple:
    """Coates & Ng: k-means no espaco branco, centroide como filtro, com os dois sinais."""
    x = unit(ctx.white(ctx.z))
    fit = MiniBatchKMeans(min(width // 2, len(x)), n_init=3, random_state=ctx.seed).fit(x)
    return signs(ctx.white(unit(fit.cluster_centers_))), {}


def responses(ctx: Ctx, rows: torch.Tensor) -> torch.Tensor:
    """relu(u . z(x)) de cada candidato nas janelas `rows`: o que a conv dobrada daria."""
    assert not isinstance(ctx, PatchCtx), "F2-F6 medem a resposta no z-score comum"
    z = (rows.to(DEVICE) - ctx.mean.to(DEVICE)) / ctx.stdev.to(DEVICE)
    return F.relu(z @ torch.as_tensor(ctx.z, dtype=torch.float32, device=DEVICE).T)


def centered(a: torch.Tensor) -> torch.Tensor:
    """Cada coluna sem media e com norma 1, entre janelas: coluna constante vira 0."""
    return F.normalize(a - a.mean(0), dim=0)


def functional(ctx: Ctx, width: int, kind: str) -> tuple:
    """F2-F6 nas respostas a PROBE_ROWS janelas medoide de treino; candidato que nunca responde fica de fora."""
    probe = np.random.default_rng(ctx.seed).permutation(len(ctx.ref))[:PROBE_ROWS]
    act = responses(ctx, ctx.ref[probe])
    live = torch.nonzero(act.std(0) > 0)[:, 0]
    act, n, mine = act[:, live], min(width, len(live)), live.cpu().numpy()
    corr = centered(act).T @ centered(act)
    if kind == "f2":    # FPS em 1 - correlacao
        picks = greedy(corr, n, 0, torch.maximum)
    elif kind == "f3":  # a gulosa de `activation`
        picks = low_redundancy(corr, n, 0)
    elif kind == "f4":  # cobertura: cada janela quer um filtro de cosseno alto com ela (facility location)
        z = F.normalize((ctx.ref[probe].to(DEVICE) - ctx.mean.to(DEVICE)) / ctx.stdev.to(DEVICE), dim=1)
        cos = F.relu(z @ torch.as_tensor(ctx.z[mine], dtype=torch.float32, device=DEVICE).T)
        covered, taken, picks = torch.zeros_like(cos[:, :1]), torch.zeros(len(live), dtype=torch.bool, device=DEVICE), []
        for _ in range(n):
            picks.append(int((cos - covered).clamp_min(0).sum(0).masked_fill(taken, -1).argmax()))
            taken[picks[-1]] = True
            covered = torch.maximum(covered, cos[:, picks[-1]:picks[-1] + 1])
        picks = torch.tensor(picks, device=DEVICE)
    elif kind == "f5":  # FPS na soma das distancias visual (1 - |cos|) e funcional (1 - correlacao)
        u = torch.as_tensor(ctx.z[mine], dtype=torch.float32, device=DEVICE)
        picks = greedy((u @ u.T).abs() + corr - 2, n, 0, torch.maximum)
    else:  # f6: a gulosa de f3 mais (1 - estabilidade), a correlacao da resposta na mesma janela em meia escala
        stable = (centered(act) * centered(responses(ctx, ctx.half()[probe])[:, live])).sum(0)
        picks = greedy(F.relu(corr).square() + (1 - stable)[None], n, int(stable.argmax()), torch.add)
    return seen(ctx, mine[picks.cpu().numpy()])


def kmed(ctx: Ctx, width: int) -> tuple:
    """K-medoids no z dos patches, comecando do FPS euclidiano: `kmedoids_visual` de `filter_diversity`."""
    z = torch.as_tensor(ctx.z, dtype=torch.float32, device=DEVICE)
    dist = torch.cdist(z, z)
    return seen(ctx, k_medoids(dist, greedy(-dist, min(width, len(z)), 0, torch.maximum)).cpu().numpy())


METHODS = {
    "crus": rowwise(uniform), "fps": rowwise(fps), "qr": rowwise(qr_pivot), "cos": rowwise(cos_prune),
    "fpsw": fps_weighted, "trio": trio, "quota": quota,
    "zca2": zca2, "zca2h": lambda ctx, width: zca2(ctx, width, 10 * ZCA_EPS), "pca": pca, "pca2": pca2,
    "zca_km2": zca_km2,
    "km_w": lambda c, w: grouped(c, c.z, fixed_groups(reduced(c.z, c.seed), w, c.seed), False),
    "km_w_m": lambda c, w: grouped(c, c.z, fixed_groups(reduced(c.z, c.seed), w, c.seed), True),
    "km_sil": lambda c, w: grouped(c, c.z, sil_groups(reduced(c.z, c.seed), c.seed), False),
    "km_sil_m": lambda c, w: grouped(c, c.z, sil_groups(reduced(c.z, c.seed), c.seed), True),
    "hdbscan": lambda c, w: grouped(c, c.z, HDBSCAN(min_cluster_size=MIN_GROUP).fit_predict(reduced(c.z, c.seed)), False),
    "hdbscan_m": lambda c, w: grouped(c, c.z, HDBSCAN(min_cluster_size=MIN_GROUP).fit_predict(reduced(c.z, c.seed)), True),
    "fpsd": lambda c, w: seen(c, *fps(c.desc, w, c.seed)),
    "fpsdw": lambda c, w: fps_weighted(c, w, "desc"),
    "cosd": lambda c, w: seen(c, *cos_prune(c.desc, w, c.seed)),
    "bovw_w": lambda c, w: grouped(c, c.desc, fixed_groups(c.desc, w, c.seed), False),
    "bovw_sil": lambda c, w: grouped(c, c.desc, sil_groups(c.desc, c.seed), False),
    "bovw_w_m": lambda c, w: grouped(c, c.desc, fixed_groups(c.desc, w, c.seed), True),
    "bovw_sil_m": lambda c, w: grouped(c, c.desc, sil_groups(c.desc, c.seed), True),
    "gr": lambda c, w: graph_arm(c, w, False), "gr_m": lambda c, w: graph_arm(c, w, True),
    "hib": lambda c, w: hybrid(c, w, False), "hib_m": lambda c, w: hybrid(c, w, True),
    **{f"f{i}": lambda c, w, k=f"f{i}": functional(c, w, k) for i in range(2, 7)}, "kmed": kmed,
}
FREE = {"km_sil", "km_sil_m", "hdbscan", "hdbscan_m", "bovw_sil", "bovw_sil_m"}
RANDOM = {"xavier": init.xavier_uniform_, "he": init.kaiming_normal_, "orth": init.orthogonal_,
          "tnorm": lambda w: init.trunc_normal_(w, std=1.0, a=-2.0, b=2.0), "random_z": init.normal_}


def control(arm: str, ctx: Ctx, width: int, layer: int) -> np.ndarray:
    """Direcoes sorteadas. As iid so diferem na forma da distribuicao: a escala cai na dobra."""
    torch.manual_seed(ctx.seed + 1000 * layer + zlib.crc32(arm.encode()) % 100_000)
    d = ctx.feats.shape[1]
    if arm in RANDOM:
        return RANDOM[arm](torch.empty(width, d)).double().numpy()
    half = arm.endswith("2")
    u = unit(init.normal_(torch.empty(width // 2 if half else width, d)).double().numpy())
    if arm.startswith("whrand"):
        u = ctx.white(u)
    return signs(u) if half else u


CONTROLS = {*RANDOM, "whrand", "whrand2", "rz2"}


def score_full(train, train_y, val, val_y, num_classes: int, tag: str, impurity: int | None, white: bool = True,
               keep: bool = False) -> dict:
    """A leitura de `eval/l1_fast.py`, mais o que a impureza e as demais classes recebem."""
    z, other = pw_whiten(train, val) if white else (normalize(train), normalize(val))
    clf = fit_svm(z, train_y, tag=tag)
    pred = clf.predict(other)
    metrics = compute_metrics(y_true=val_y, y_pred=pred, num_classes=num_classes)
    row = {"fit_status": clf.fit_diagnostics_["svm_fit_status"], "fit_s": clf.fit_diagnostics_["svm_fit_seconds"],
           "dim": z.shape[1], "val_kappa": metrics["kappa"], "val_acc": metrics["acc"], "val_f1": metrics["f1"],
           "val_accuracy": float((pred == val_y).mean())}  # `acc` do projeto e a balanceada; esta e a fracao de acertos
    if keep:  # `encdeep.erros`: a predicao sai para o CSV lateral, o recall por classe para a linha
        row |= {"pred": pred, "class_recall": ";".join(f"{(pred[val_y == c] == c).mean():.4f}" for c in np.unique(val_y))}
    if impurity is not None:
        mine = val_y == impurity
        row |= {"imp_recall": float((pred[mine] == impurity).mean()),
                "par_recall": float(np.mean([(pred[val_y == c] == c).mean() for c in np.unique(val_y[~mine])])),
                "imp_kappa": compute_metrics(y_true=mine.astype(np.int64), y_pred=(pred == impurity).astype(np.int64), num_classes=2)["kappa"],
                "par_kappa": compute_metrics(y_true=val_y[~mine], y_pred=pred[~mine], num_classes=num_classes)["kappa"]}
    return row


def map_whitener(rows: torch.Tensor, area: int, shrink: float = SHRINK) -> tuple:
    """Media e ZCA dos canais do mapa, nas sementes (centro de cada patch), com shrinkage."""
    center = rows.view(len(rows), -1, area)[:, :, area // 2].double()
    values, vectors = torch.linalg.eigh(torch.cov(center.T))
    values = values.clamp_min(0)
    return center.mean(0).float(), ((vectors * (values + shrink * values.mean()) ** -0.5) @ vectors.T).float()


def renorm(ref: torch.Tensor, area: int, tags: list, mean: torch.Tensor, stdev: torch.Tensor) -> tuple:
    """`amostra.norm`: N2 um escalar para tudo, N3 um por banda, N7 mediana e MAD por dimensao; sem tag, N0."""
    if "global" in tags:
        return torch.full_like(mean, float(ref.mean())), torch.full_like(stdev, float(ref.std(unbiased=False)) + STDEV_FLOOR)
    if "canal" in tags:
        bands = ref.view(len(ref), -1, area)
        return bands.mean((0, 2)).repeat_interleave(area), bands.std((0, 2), correction=0).repeat_interleave(area) + STDEV_FLOOR
    if "mad" in tags:
        middle = ref.median(0).values
        return middle, MAD * (ref - middle).abs().median(0).values + STDEV_FLOOR
    return mean, stdev


class Abs(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.abs()


ACTS = {"abs": Abs, "leaky": lambda: torch.nn.LeakyReLU(LEAK), "gelu": torch.nn.GELU}


def _sa(x: torch.Tensor, alpha: float, scales=(1,)) -> torch.Tensor:
    """X + a*softmax(n(X)n(K)^T/sqrt(d))V, canais com layer norm; K, V no mapa reduzido por cada escala, media."""
    b, c, h, w = x.shape
    q = F.layer_norm(x.flatten(2).transpose(1, 2), (c,))
    outs = []
    for s in scales:
        v = F.avg_pool2d(x, s) if s > 1 and min(h, w) >= s else x
        v = v.flatten(2).transpose(1, 2)
        a = torch.softmax(q @ F.layer_norm(v, (c,)).transpose(1, 2) / c**0.5, -1)
        outs.append(a @ v)
    return x + alpha * torch.stack(outs).mean(0).transpose(1, 2).reshape(x.shape)


def _z(m: torch.Tensor) -> torch.Tensor:
    return (m - m.mean((2, 3), keepdim=True)) / (m.std((2, 3), keepdim=True) + 1e-6)


def _nonlocal_g(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Non-local com nucleo gaussiano exp(-|qi-qj|^2/2c) em canais com layer norm; residuo alpha."""
    c = x.shape[1]
    t = x.flatten(2).transpose(1, 2)
    q = F.layer_norm(t, (c,))
    a = torch.softmax(-torch.cdist(q, q).pow(2) / (2 * c), -1)
    return x + alpha * (a @ t).transpose(1, 2).reshape(x.shape)


def _simam(x: torch.Tensor, alpha: float, lam: float = 1e-4) -> torch.Tensor:
    d = (x - x.mean((2, 3), keepdim=True)).pow(2)
    v = d.sum((2, 3), keepdim=True) / (x.shape[2] * x.shape[3] - 1)
    return x * torch.sigmoid(d / (4 * (v + lam)) + 0.5)


def _gc(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """GCNet sem parametros: descritor global pesado pelo softmax espacial da media dos canais, somado em todo ponto."""
    a = torch.softmax(_z(x.mean(1, keepdim=True)).flatten(2), -1)
    return x + alpha * (x.flatten(2) * a).sum(2)[..., None, None]


SS_TAUS = torch.logspace(-2.5, 0, 16).tolist()  # temperaturas candidatas do `ss`


def _ss(x: torch.Tensor, alpha: float, iters: int = 2, target: float = 0.5) -> torch.Tensor:
    """Self-self sem treino (SCLIP/ClearCLIP/GEM, 2024): cosseno entre tokens, diagonal mascarada, sem residuo, `iters` passos.
    tau por imagem, sem rotulo: a da grade cuja entropia media da atencao (/log(N-1)) fica mais perto de `target`."""
    t = x.flatten(2).transpose(1, 2)
    q = F.normalize(t, dim=2)
    s = (q @ q.transpose(1, 2)).diagonal_scatter(torch.full(t.shape[:2], -torch.inf, device=x.device), dim1=1, dim2=2)
    ent = torch.stack([torch.special.entr(torch.softmax(s / tau, -1)).sum(-1).mean(-1) for tau in SS_TAUS], 1)
    tau = torch.tensor(SS_TAUS, device=x.device)[(ent / math.log(t.shape[1] - 1) - target).abs().argmin(1)]
    a = torch.softmax(s / tau[:, None, None], -1)
    for _ in range(iters):
        t = a @ t
    return t.transpose(1, 2).reshape(x.shape)


class Dilated(torch.nn.Module):
    """Modo (b) `dilat`: a mesma conv mais alpha * media dela nas dilatacoes 2, 4 e 8; mesmo tamanho de mapa."""

    def __init__(self, conv: torch.nn.Conv2d, alpha: float):
        super().__init__()
        self.conv, self.alpha = conv, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, w, b = self.conv.kernel_size[0] // 2, self.conv.weight, self.conv.bias
        far = torch.stack([F.conv2d(x, w, b, padding=d * k, dilation=d) for d in (2, 4, 8)]).mean(0)
        return self.conv(x) + self.alpha * far


# `amostra.atencao`: sem parametros, so torch; modo (a) aplica no mapa da camada lida, antes do GeM
ATTN = {"sa": _sa, "espacial": lambda x, alpha: x * torch.sigmoid(_z(x.mean(1, keepdim=True)) + _z(x.amax(1, keepdim=True))),
        "simam": _simam, "nonlocal_g": _nonlocal_g, "multiescala": lambda x, alpha: _sa(x, alpha, (1, 2, 4)),
        "fourier": lambda x, alpha: x + alpha * torch.fft.fft2(x, norm="ortho").real, "gc": _gc, "ss": _ss}  # FNet: Re(FFT2) espacial


class Attn(torch.nn.Module):
    def __init__(self, name: str, alpha: float):
        super().__init__()
        self.name, self.alpha = name, alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ATTN[self.name](x, self.alpha)


def attach(block: torch.nn.Sequential, name: str, alpha: float) -> None:
    """Modo (b): o mecanismo entra no bloco conv{k}; `dilat` troca a conv, os outros vao depois do pool."""
    if name == "dilat":
        block[0] = Dilated(block[0], alpha)
    else:
        block.append(Attn(name, alpha))


def att_layers(spec: dict) -> set:
    """Posicoes de `amostra.atencao.camadas`; 'final' e a ultima camada (depth)."""
    return {spec["depth"] if p == "final" else p for p in spec.get("atencao", {}).get("camadas", [])}


def half_pool(lab: list, seeds: list, encoder, layer: LayerSpec, fold: tuple | None, span: int) -> torch.Tensor:
    """As janelas de `ref` com a imagem em meia escala, nas mesmas sementes: a outra escala de F6."""
    small = [F.interpolate(x[None], scale_factor=0.5, mode="bilinear", antialias=True)[0] for x in lab]
    rows = patch_pool(small, seeds, encoder, layer).feats
    return whitened(rows, *fold, span**2) if fold else rows


def gem(x: torch.Tensor, p: float = 3) -> torch.Tensor:
    return F.adaptive_avg_pool2d(x.clamp(min=1e-6).pow(p), 1).pow(1 / p).flatten(start_dim=1)


def covariance(out: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """R9: covariancia dos canais projetados, com shrinkage e raiz da matriz (normalizacao de potencia)."""
    x = torch.einsum("ck,bchw->bkhw", basis, out).flatten(2)
    x = x - x.mean(2, keepdim=True)
    cov = x @ x.transpose(1, 2) / x.shape[2]
    eye = torch.eye(len(basis.T), device=out.device)
    cov = cov + SHRINK * cov.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * eye
    values, vectors = torch.linalg.eigh(cov)
    root = (vectors * values.clamp_min(0).sqrt()[:, None]) @ vectors.transpose(1, 2)
    i, j = torch.triu_indices(len(eye), len(eye))
    return root[:, i, j]


def isqrt_cov(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """iSQRT-COV (Li et al. CVPR 2018): covariancia de x (b, k, n), raiz por Newton-Schulz, sinal*raiz e L2; triangulo superior."""
    x = x - x.mean(2, keepdim=True)
    cov = x @ x.transpose(1, 2) / x.shape[2]
    eye = torch.eye(cov.shape[1], device=x.device).expand_as(cov)
    tr = cov.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-12)[:, None, None]
    y, z = cov / tr, eye
    for _ in range(iters):
        t = 0.5 * (3 * eye - z @ y)
        y, z = y @ t, t @ z
    i, j = torch.triu_indices(cov.shape[1], cov.shape[1])
    v = (y * tr.sqrt())[:, i, j]
    return F.normalize(v.sign() * v.abs().sqrt(), dim=1)


def shifted_gram(x: torch.Tensor, shifts=(1, 2, 4)) -> torch.Tensor:
    """Gram deslocado (Berger & Memisevic, ICLR 2017): F(p) F(p+d)^T medio, d na horizontal e na vertical, por deslocamento."""
    def pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bihw,bjhw->bij", a, b) / (a.shape[2] * a.shape[3])

    out = [((pair(x[..., :-d], x[..., d:]) + pair(x[..., :-d, :], x[..., d:, :])) / 2).flatten(1)
           for d in shifts if d < min(x.shape[2:])]
    v = torch.cat(out, 1)
    return F.normalize(v.sign() * v.abs().sqrt(), dim=1)


def fisher(x: torch.Tensor, w: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Fisher vector (Cimpoi et al. CVPR 2015) de x (b, n, d) na GMM diagonal: 1a e 2a ordem, sinal*raiz e L2."""
    z = (x[:, :, None] - mu) / var.sqrt()
    q = torch.softmax(torch.log(w) - 0.5 * (z.pow(2).sum(-1) + torch.log(var).sum(-1)), -1)[..., None]
    n = x.shape[1]
    v = torch.cat([((q * z).sum(1) / (n * w.sqrt()[:, None])).flatten(1),
                   ((q * (z.pow(2) - 1)).sum(1) / (n * (2 * w).sqrt()[:, None])).flatten(1)], 1)
    return F.normalize(v.sign() * v.abs().sqrt(), dim=1)


SECOND = {"cov60": 60, "mscp": 15, "gram": 24, "fisher": 32}  # canais apos a PCA sem rotulo; so na ultima camada; 1830, 1830, 1728, 1024-d
FV_K, FV_PTS = 16, 50000  # `fisher`: gaussianas e pixels de treino que fitam a GMM
MSCP_LAYERS = 4  # MSCP (He et al., IEEE TGRS 2018): as 4 ultimas camadas na grade da ultima, empilhadas


POOLS = {"gem": lambda pre, out: gem(out), "gap": lambda pre, out: out.mean((2, 3)),
         "gmp": lambda pre, out: out.amax((2, 3)),
         "spp": lambda pre, out: torch.cat([F.adaptive_avg_pool2d(out, s).flatten(1) for s in (1, 2, 4)], 1),
         "pol": lambda pre, out: torch.cat([gem(pre), gem(-pre)], 1)}  # A5: as duas polaridades antes da ativacao


def vlad(out: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """VLAD: residuos somados no centroide mais proximo, intra-normalizados, raiz com sinal e L2."""
    x = out.flatten(2).transpose(1, 2)
    hit = F.one_hot((centers.pow(2).sum(1) - 2 * x @ centers.T).argmin(2), len(centers)).to(x.dtype)
    v = F.normalize(hit.transpose(1, 2) @ x - hit.sum(1)[..., None] * centers, dim=2).flatten(1)
    return F.normalize(v.sign() * v.abs().sqrt(), dim=1)


@torch.no_grad()
def readouts(encoder, images: dict, to_float, size: int, names: list, tta: bool = False) -> dict:
    """`amostra.leituras` num forward so: `POOLS`, `gem_p<p>`, `ms` (GeM em 1, 0,75 e 0,5 da imagem) e `cov`.
    Lote 4x menor que o de `features`: o mapa antes e depois da ativacao ficam juntos. Acima de MAX_DIM, PCA fitada no treino. Devolve nome -> (treino, validacao, dimensao antes da PCA)."""
    step, last = max(1, l1_fast.BATCH_PIXELS // size**2 // 4), getattr(encoder, f"conv{encoder.n_layers}")

    def maps(x: torch.Tensor) -> tuple:
        for k in range(1, encoder.n_layers):
            x = getattr(encoder, f"conv{k}")(x)
        pre = last[0](x)
        return pre, last[1:](pre.clone())

    def batches(x: torch.Tensor):
        for i in range(0, len(x), step):
            yield to_float(x[i:i + step].to(DEVICE))

    basis = None
    if "cov" in names:
        fit, total, outer, count = images["train"][0][::max(1, len(images["train"][0]) // COV_FIT)][:COV_FIT], 0, 0, 0
        for b in batches(fit):
            pix = maps(b)[1].transpose(0, 1).flatten(1).double()
            total, outer, count = total + pix.sum(1), outer + pix @ pix.T, count + pix.shape[1]
        mean = total / count
        basis = torch.linalg.eigh(outer / count - torch.outer(mean, mean))[1].flip(1)[:, :COV_DIM].float()

    def tail(x: torch.Tensor) -> list:  # mapas das MSCP_LAYERS ultimas camadas, a ultima por ultimo
        maps_ = []
        for k in range(1, encoder.n_layers + 1):
            x = getattr(encoder, f"conv{k}")(x)
            maps_.append(x)
        return maps_[-MSCP_LAYERS:]

    pcs, gmm, sub = {}, None, images["train"][0][::max(1, len(images["train"][0]) // COV_FIT)][:COV_FIT]
    if SECOND.keys() & set(names):  # SECOND: media e eixos da PCA de pixels por camada, so no treino e sem rotulo
        acc = {}
        for b in batches(sub):
            for k, m in enumerate(tail(b)):
                pix = m.transpose(0, 1).flatten(1).double()
                s, o, c = acc.get(k, (0, 0, 0))
                acc[k] = (s + pix.sum(1), o + pix @ pix.T, c + pix.shape[1])
        for k, (s, o, c) in acc.items():
            pcs[k - len(acc)] = (s / c).float(), torch.linalg.eigh(o / c - torch.outer(s / c, s / c))[1].flip(1).float()

    def second(r: str, maps_: list) -> torch.Tensor:
        dim = SECOND[r]
        proj = [torch.einsum("ck,bchw->bkhw", pcs[k][1][:, :dim], m - pcs[k][0][:, None, None])
                for k, m in zip(range(-len(maps_), 0), maps_)]
        if r == "fisher":
            return fisher(proj[-1].flatten(2).transpose(1, 2), *gmm)
        if r == "gram":
            return shifted_gram(proj[-1])
        if r == "cov60":
            return isqrt_cov(proj[-1].flatten(2))
        side = proj[-1].shape[2:]
        return isqrt_cov(torch.cat([F.adaptive_avg_pool2d(p, side) for p in proj], 1).flatten(2))

    if "fisher" in names:  # GMM diagonal so nos pixels de treino projetados, sem rotulo
        mean, axes = pcs[-1]
        pts = torch.cat([torch.einsum("ck,bchw->bkhw", axes[:, :SECOND["fisher"]], tail(b)[-1] - mean[:, None, None])
                         .flatten(2).transpose(1, 2).reshape(-1, SECOND["fisher"]).cpu() for b in batches(sub)])
        pts = pts[torch.randperm(len(pts), generator=torch.Generator().manual_seed(0))[:FV_PTS]].double().numpy()
        g = GaussianMixture(FV_K, covariance_type="diag", random_state=0).fit(pts)
        gmm = [torch.as_tensor(v, dtype=torch.float32, device=DEVICE) for v in (g.weights_, g.means_, g.covariances_)]
    books = {}
    if any(r.startswith("vlad") for r in names):  # codebook so no treino, sem rotulo
        gen, pts = torch.Generator().manual_seed(0), []
        for b in batches(images["train"][0]):
            pts += [x[torch.randperm(len(x), generator=gen)[:VLAD_PTS]] for x in maps(b)[1].flatten(2).transpose(1, 2).cpu()]
        pts = torch.cat(pts).numpy()
        books = {r: torch.from_numpy(MiniBatchKMeans(int(r[4:]), random_state=0, n_init=3).fit(pts).cluster_centers_).to(DEVICE)
                 for r in names if r.startswith("vlad")}
    out = {}
    for split in ("train", "validation"):
        got = {r: [] for r in names}
        for whole in batches(images[split][0]):
            views = [torch.rot90(v, k, (2, 3)) for v in (whole, whole.flip(3)) for k in range(4)] if tta else [whole]
            mark = {r: len(got[r]) for r in names}
            for batch in views:
                pre, post = maps(batch)
                deep_maps = tail(batch) if pcs else None
                for r in names:
                    if r in SECOND:
                        got[r].append(second(r, deep_maps).cpu())
                    elif r in POOLS:
                        got[r].append(POOLS[r](pre, post).cpu())
                    elif r.startswith("gem_p"):
                        got[r].append(gem(post, float(r[5:])).cpu())
                    elif r == "ms":
                        small = [maps(F.interpolate(batch, scale_factor=s, mode="bilinear", antialias=True))[1] for s in SCALES]
                        got[r].append(torch.cat([gem(post), *map(gem, small)], 1).cpu())
                    elif r in books:
                        got[r].append(vlad(post, books[r]).cpu())
                    else:
                        got[r].append(covariance(post, basis).cpu())
            if tta:  # `amostra.leituras_tta`: media das 8 vistas rot90 x flip, como `gem_tta`
                for r in names:
                    got[r][mark[r]:] = [torch.stack(got[r][mark[r]:]).mean(0)]
        out[split] = {r: torch.cat(v).numpy() for r, v in got.items()}
    reply = {}
    for r in names:
        train, val = out["train"][r], out["validation"][r]
        reply[r] = (*pca_fit(train, val, MAX_DIM - 1), train.shape[1]) if train.shape[1] >= MAX_DIM and r not in books else (train, val, train.shape[1])
    return reply


def pca_fit(train: np.ndarray, val: np.ndarray, dim: int) -> tuple:
    """PCA fitada so no treino, aplicada nos dois."""
    pca = PCA(min(dim, *train.shape), random_state=0).fit(train)
    return pca.transform(train), pca.transform(val)


@torch.no_grad()
def whitened(rows: torch.Tensor, shift: torch.Tensor, gain: torch.Tensor, area: int) -> torch.Tensor:
    """Patches canal-primeiro como sairiam do mapa branqueado."""
    shift, gain = shift.to(DEVICE), gain.to(DEVICE)
    return torch.cat([torch.einsum("dc,ncp->ndp", gain, part.to(DEVICE).view(len(part), -1, area) - shift[:, None]).flatten(1).cpu()
                      for part in rows.split(WHITEN_ROWS)])


def folded(weight: torch.Tensor, bias: torch.Tensor, shift: torch.Tensor, gain: torch.Tensor, area: int) -> tuple:
    """A conv sobre o mapa branqueado, escrita sobre o mapa cru: a rede segue conv, ReLU e pool."""
    kernel = weight.view(len(weight), -1, area)
    return torch.einsum("ocp,cd->odp", kernel, gain).flatten(1), bias - torch.einsum("ocp,c->o", kernel, gain @ shift)


def pools(spec: dict) -> list:
    """O pool de cada camada. MaxPool 3 sem padding a 200 px: pool 2 em tudo acaba com o mapa na 7a."""
    return spec["pool"] if isinstance(spec["pool"], list) else [spec["pool"]] * spec["depth"]


LAW_VERSION = "v1"
WMIN, WMAX = 32, 1024


def p2(x: float) -> int:
    """Potencia de 2 mais proxima em log2 (meio exato vai para o par, `round`), presa em [32, 1024]."""
    return min(WMAX, max(WMIN, 2 ** round(math.log2(x))))


def law_channels(spec: dict, kernel: int = 5, scale: float = 1.0) -> list:
    """Lei de construcao -> largura de cada camada. A geometrica, B posicao relativa, C orcamento sobre A ou B.

    O arredondamento a potencia de 2 vem por ultimo, depois de `scale`: em C, s escala a forma continua.
    """
    lei, depth = spec["lei"], spec["depth"]
    if lei == "W":  # larguras explicitas "a-b-c": leis da literatura ja traduzidas (restricao_5M.md)
        return [int(x) for x in str(spec["w"]).split("-")]
    if lei == "A":
        return [p2(scale * max(spec["w0"] * spec["r"] ** i, spec.get("wmin", WMIN))) for i in range(depth)]
    if lei == "B":
        f = {"lin": lambda t: t, "fast": math.sqrt, "late": lambda t: t * t}[spec["f"]]
        a, b = math.log2(spec["w0"]), math.log2(spec["wL"])
        return [p2(scale * 2 ** (a + (b - a) * f(i / max(depth - 1, 1)))) for i in range(depth)]
    if lei == "C":  # s em passos de 2^(1/8); empate fica com o menor s
        shapes = [law_channels(spec["forma"] | {"depth": depth}, kernel, 2 ** (j / 8)) for j in range(-40, 41)]
        return min(shapes, key=lambda ws: abs(conv_cost(ws, kernel)[0] - spec["P"]))
    raise NotImplementedError("D: crescimento por dados (posto efetivo, cobertura, redundancia, ganho marginal) — fora do P0")


def lei_kernels(spec: dict, kernel: int, depth: int) -> list:
    """`lei.k` = kernel de cada camada, "3-3-3-5-5-5"; sem ele, o kernel da celula em todas (como antes)."""
    k = spec.get("lei", {}).get("k")
    return [int(x) for x in str(k).split("-")] if k else [kernel] * depth


def conv_cost(widths: list, kernel: int | list, size: int = 200, pool: list | None = None) -> tuple:
    """Parametros de conv (sem bias), MACs, mapa de saida e campo receptivo; conv `same`, MaxPool 3 stride p sem padding."""
    params = macs = 0
    cin, side, rf, jump, maps = 3, size, 1, 1, []
    for n, c in enumerate(widths):
        k = kernel[n] if isinstance(kernel, list) else kernel
        params += cin * c * k**2
        macs += cin * c * k**2 * side**2
        rf += (k - 1) * jump
        if pool and pool[n] > 1:
            side, rf, jump = (side - 3) // pool[n] + 1, rf + 2 * jump, jump * pool[n]
        cin = c
        maps.append(side)
    return params, macs, maps, rf


def law_name(spec: dict) -> str:
    """A lei como texto compacto e estavel: entra no CSV e no nome dele."""
    return "_".join(f"{k}{law_name(v) if isinstance(v, dict) else v}" for k, v in sorted(spec.items()))


def spifil_rows(job: dict, ctx: Ctx, n: int, width: int, kernel: int, seed: int, chainer: dict,
                out: Path, cut: float, **extra) -> tuple:
    """O seletor oficial, por classe; a largura pedida e cortada por igual, sem resto do allocator."""
    classes = DATASET_NUM_CLASSES[job["dataset_name"]]
    per = -(-width // classes)
    k = max(job["amostra"].get("spifil_k", 3), -(-3 * per // job["superpixels"][0]))
    depth = job["amostra"]["depth"]
    plan = {"channels": [0] * depth, "kernels": [kernel] * depth, "pools": pools(job["amostra"])}
    grow_layer(job | {"amostra": plan}, n, k, per * classes, seed, chainer["encoder"], chainer["ckpt"],
               chainer["arch"], chainer["weights"], out, **extra)
    chosen = Spy.learn.state[Spy.learn.n_layers].selected
    feats = chosen.feats.detach().cpu()
    if len(feats) > width:
        feats = feats[torch.linspace(0, len(feats) - 1, width).round().long()]
    Spy.learn = None
    info = {"images": k * classes}
    if n == 0:
        info["bg_sel"] = float((local_std(feats) < cut).float().mean())
    return ((feats - ctx.mean) / ctx.stdev).double().numpy(), info


def chain(job: dict, images: dict, to_float, lab: list, srcs: dict, cut: float, arm: str, plan: tuple,
          kernel: int, seed: int, base: dict, submit) -> tuple:  # noqa: PLR0913
    """Uma rede sem gradiente, camada a camada; devolve as larguras reais."""
    spec = job["amostra"]
    name, *tags = arm.split("~")
    method, _, point = name.partition("@")
    shrink = next((float(t[1:]) for t in tags if t[0] == "w"), SHRINK)
    plano, drop = next((float(t[2:]) for t in tags if t.startswith("pl")), 0.0), None
    signed = {"crelu", "crelu_h"} & set(tags)
    ab = next((t[3:].rpartition("_")[::2] for t in tags if t.startswith("ab_")), None)  # `~ab_<mec>_<pos>`: modo (b)
    ab_at = ab and (spec["depth"] if ab[1] == "final" else int(ab[1]))
    between, mix = spec.get("entre_camadas", False), spec.get("agrupamento")
    explicit = spec.get("whiten_between", False)  # o whitening vira flim.arch.Whiten no forward, nao dobra no conv
    assert between or not explicit, "whiten_between pede entre_camadas"
    assert not (between and method.startswith("spifil")), "o `grow` le o mapa cru, nao o branqueado"
    assert not (mix and method.startswith("spifil")), "o `grow` dobra no z-score comum"
    src, ref_seeds = srcs[point or "medoid"], srcs["medoid"].seeds
    place = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    arch_json, weights = _paths(place)
    arch = parse_architecture(arch_json)
    arch = {k: v for k, v in arch.items() if not k.startswith("layer")} | {"nlayers": 0}
    chainer = {"encoder": None, "ckpt": "", "arch": arch_json, "weights": weights}
    encoder, source, widths, kept = torch.nn.Identity(), Path(weights), [], []
    one = spec.get("uma_camada")  # `montar_camada`: L1..L(k-1) prontos em encoder_dir, so a camada k nasce
    if one and one["k"] > 1:
        source = Path(one["encoder_dir"])
        arch = json.loads((source / "architecture.json").read_text())
        assert arch["nlayers"] == one["k"] - 1, f"{source}: nlayers {arch['nlayers']} != k-1"
        chainer = {"encoder": None, "ckpt": str(source), "arch": str(source / "architecture.json"), "weights": str(source)}
        encoder = chainer["encoder"] = build(chainer["arch"], "flim", str(source)).to(DEVICE).eval()
    l1_fast.BATCH_PIXELS = 2**27 // max(max(wanted(arm, plan)), 64)
    family = "controle" if method in CONTROLS else "oraculo" if point.startswith("m_") else "metodo"
    row0 = base | {"arm": arm, "family": family, "seed": seed, "plan": "-".join(map(str, plan)),
                   "kernel": kernel, "images": len(lab)}
    points = [len(c) for c in coords_of(src)]
    row0 |= {"pts_mean": float(np.mean(points)), "pts_min": min(points)}
    Oracle.side, Oracle.place = {"m_obj": True, "m_fundo": False}.get(point), (job["dataset_name"], job["image_size"])
    impurity = IMPURITY.get(job["dataset_name"])
    tag = f"spf {job['stem']} {arm} {row0['plan']} k{kernel} s{seed}"
    y = images["train"][1], images["validation"][1]
    deep = spec.get("encdeep")
    if deep:
        from eval import encdeep  # tardio: encdeep -> flim.local_plasticity -> eval.alt_encoders -> este modulo
    ks, ds, ps = encdeep.geometry(spec, tags, pools(spec)) if deep else (lei_kernels(spec, kernel, len(plan)), [1] * len(plan), pools(spec))
    if deep:
        row0 |= encdeep.geo_log(tags, job["image_size"], ks, ds, ps)
        trunk, cell = min(deep.get("camadas", [4, 5])) - 1, Path(job["work_dir"]) / "encdeep" / f"{arm}_k{kernel}_s{seed}"
    torch.cuda.reset_peak_memory_stats()
    with tempfile.TemporaryDirectory(dir=job["work_dir"]) as tmp:
        for n, target in enumerate(plan):
            if one and n + 1 < one["k"]:
                continue
            start = time.perf_counter()
            span = 2 * (ks[n] // 2) + 1
            layer = LayerSpec(kernel_size=ks[n], out_channels=max(target, 1), dilation=ds[n])
            ref = patch_pool(lab, ref_seeds, encoder, layer).feats
            plain = "nw" in tags and n + 1 in deep.get("camadas", [4, 5])  # H0/H2: sem whitening nas camadas encdeep
            fold = map_whitener(ref, span**2, shrink) if between and n and not plain else None
            if fold:
                ref = whitened(ref, *fold, span**2)
            mean, stdev = ref.mean(0), ref.std(0, unbiased=False) + STDEV_FLOOR
            if tags:
                mean, stdev = renorm(ref, span**2, tags, mean, stdev)
            want = target // 2 if "crelu_h" in tags else target
            if src.keys is not None:
                feats = view_pool(lab, src.seeds, src.keys, ref_seeds, encoder, span, mean, stdev, STDEV_FLOOR, FLAT)
            else:
                feats = ref if src.seeds is ref_seeds else patch_pool(lab, src.seeds, encoder, layer).feats
            if fold and src.seeds is not ref_seeds:
                feats = whitened(feats, *fold, span**2)
            if src.rows is not None:
                feats = feats[torch.from_numpy(src.rows)]
            if plano and n == 0:  # a janela LAB crua da semente; a mesma semente sai de toda camada
                drop = (local_std(feats) < plano).numpy()
            ctx = (PatchCtx(feats, mean, stdev, src, seed + n, eps=mix["eps"], cap=mix["rows"], area=span**2) if mix
                   else Ctx(feats, mean, stdev, src, seed + n))
            ctx.drop = drop
            ctx.ref, ctx.half = ref, lambda: half_pool(lab, ref_seeds, encoder, layer, fold, span)
            if method in CONTROLS:
                u, info = control(method, ctx, want, n), {}
                if mix:
                    u = ctx.rez(unit(u))[0]
            elif method.startswith("spifil"):
                extra = {"redundancy_cos": COPY_COS, "redundancy_replace": True} if method == "spifil_dd" else {}
                u, info = spifil_rows(job, ctx, n, want, kernel, seed, chainer, Path(tmp) / f"g{n + 1}", cut, **extra)
            else:
                u, info = METHODS[method](ctx, want)
            if signed:  # A1: u e -u viram ReLU(x) e ReLU(-x)
                u = signs(np.asarray(u))
            u = F.normalize(torch.as_tensor(u, dtype=torch.float32), dim=1)
            mech, minfo = deep and encdeep.deep_mechanism(spec, n, tags), {}
            if mech:  # §6.1: W e b_v aprendidos em v-space; a dobra abaixo e a de sempre
                with encdeep.no_autograd():
                    u, b_v, minfo = mech(u=u, encoder=encoder, fold=fold, mean=mean, stdev=stdev, k=ks[n], dil=ds[n],
                                         train=images["train"][0], to_float=to_float, seed=seed + n)
            if mix:
                mean, stdev = torch.zeros_like(mean), ctx.spread(ref)
            weight = u / stdev
            bias = -(weight * mean).sum(1)
            if mech:
                bias = bias + b_v
            if fold and not explicit:
                weight, bias = folded(weight, bias, *fold, span**2)
            out = Path(tmp) / f"L{n + 1}"
            arch = spifil.grown_arch(arch, ks[n], len(u), ps[n], ds[n])
            spifil.write_weights(out, source, arch, n + 1, SimpleNamespace(conv=SimpleNamespace(
                weight=weight.view(len(u), -1, span, span), bias=bias)))
            if fold and explicit:
                np.savez(out / f"conv{n + 1}-whiten.npz", shift=fold[0].cpu().numpy(), gain=fold[1].cpu().numpy())
            if deep and n + 1 >= trunk:  # §14.3: L1..L(trunk) sao os mesmos bytes em toda camada seguinte
                got = encdeep.trunk_hash(out, trunk)
                assert n + 1 == trunk or got == row0.get("trunk_hash", got), "encdeep: o tronco mudou"
                row0["trunk_hash"] = got
            if deep and deep.get("salvar"):
                encdeep.persist(job, arm, kernel, seed, n, out, {"fold": fold, "mean": mean, "stdev": stdev})
            source, widths = out, widths + [len(u)]
            chainer = {"encoder": None, "ckpt": str(out), "arch": str(out / "architecture.json"), "weights": str(out)}
            encoder = chainer["encoder"] = build(chainer["arch"], "flim", str(out)).to(DEVICE).eval()
            for t in set(tags) & set(ACTS):  # A2, A3: troca o ReLU de toda camada
                for k in range(1, n + 2):
                    getattr(encoder, f"conv{k}")[1] = ACTS[t]()
            if ab and n + 1 >= ab_at:  # modo (b): o encoder e relido a cada camada; a seguinte recorta do mapa atendido
                attach(getattr(encoder, f"conv{ab_at}"), ab[0], spec["atencao"].get("alpha", 0.5))
            if one:
                import shutil
                shutil.copytree(out, one["out"], dirs_exist_ok=True)
                return tuple(widths)
            stats = kernel_stats(u.double().numpy())
            select_s = round(time.perf_counter() - start, 2)
            start = time.perf_counter()
            if "leituras" in spec:
                kept.append(readouts(encoder, images, to_float, job["image_size"],
                                     [r for r in spec["leituras"] if (not (r.startswith("vlad") or r in SECOND) or n + 1 == len(plan))
                                      and not (r in SECOND and family == "controle")],  # o aleatorio fica sem mecanismo
                                     tta=bool(spec.get("leituras_tta")) and family != "controle"))
            else:  # `amostra.tta`: GeM medio nas 8 vistas rot90 x flip, mesmo encoder; sai como L_tta, cat_tta
                names = [READOUT] + ["gem_tta_pw"] * bool(spec.get("tta"))
                got = [features(encoder, images[name][0], to_float, job["image_size"], names) for name in ("train", "validation")]
                kept.append({k: [g[r] for g in got] for k, r in zip(("gem", "tta"), names)})
            att, att_s = spec.get("atencao"), {}
            if att and att.get("modo", "a") == "a" and "leituras" not in spec and n + 1 in att_layers(spec):  # modo (a): mesmo encoder, Attn so na leitura desta camada
                last = getattr(encoder, f"conv{n + 1}")
                pos = "final" if n + 1 == spec["depth"] and "final" in att["camadas"] else f"L{n + 1}"
                for m in att["mecanismos"]:
                    tick = time.perf_counter()
                    last.append(Attn(m, att.get("alpha", 0.5)))
                    got = [features(encoder, images[name][0], to_float, job["image_size"], names) for name in ("train", "validation")]
                    del last[-1]
                    for k, r in zip(("att", "tta_att"), names):  # com `amostra.tta`, a TTA tambem passa pelo Attn
                        kept[-1][f"{k}_{m}_{pos}"] = [g[r] for g in got]
                        att_s[f"{k}_{m}_{pos}"] = round(time.perf_counter() - tick, 2)
            if deep:
                minfo["deploy_params"] = sum(p.numel() for p in encoder.parameters())
            row = row0 | stats | info | ctx.audit | minfo | {
                "layer": n + 1, "width": len(u), "widths": "-".join(map(str, widths)),
                "width_req": target, "short": int(len(u) != target),
                "short_why": "" if len(u) >= target else f"realized=available ({len(feats)} candidatos)", "used_rows": len(ctx.keep) if "live_rows" in ctx.audit else "",
                "candidates": len(feats), "bg_src": float(src.bg.mean()) if src.rows is None else float(src.bg[src.rows].mean()),
                "select_s": select_s, "extract_s": round(time.perf_counter() - start, 2),
                "gpu_mib": torch.cuda.max_memory_allocated() >> 20, "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 10}
            classes = DATASET_NUM_CLASSES[job["dataset_name"]]

            def read(name: str, train: np.ndarray, val: np.ndarray, raw: int = 0) -> None:
                keep = (True, True) if deep and deep.get("erros") and name.startswith("L") else ()  # so as camadas isoladas
                submit(row | {"readout": name, "raw_dim": raw or train.shape[1]}, (train, y[0], val, y[1], classes, tag, impurity, *keep))
                if between:
                    submit(row | {"readout": f"{name}_b", "raw_dim": raw or train.shape[1]},
                           (train, y[0], val, y[1], classes, tag, impurity, False))

            vlads = {r: kept[-1].pop(r) for r in [*kept[-1]] if r.startswith("vlad")}
            atts = {r: kept[-1].pop(r) for r in att_s}
            seconds = {r: kept[-1].pop(r) for r in [*kept[-1]] if r in SECOND}  # so L_<r>, sem cat
            ref, cats = {}, {}  # `amostra.eq`: `ms` reduzido a dimensao de `gem` (L_ms_eq, cat_ms_eq)
            for r in kept[-1]:
                end = "" if r == "gem" else f"_{r}"
                read("L" + end, *kept[-1][r])
                ref[r, "L"] = kept[-1][r][0].shape[1]
                eq = deep and r == "gem" and deep.get("dim_eq")  # §12: L e cat na mesma dimensao, PCA so no treino
                if eq and ref[r, "L"] > eq:
                    read(f"L_eq{eq}", *pca_fit(*kept[-1][r][:2], eq), ref[r, "L"])
                if r == "ms" and spec.get("eq") and ("gem", "L") in ref:
                    read("L_ms_eq", *pca_fit(*kept[-1][r][:2], ref["gem", "L"]), kept[-1][r][2])
                if n:
                    fit, total = [], 0
                    for pair in reversed([k[r] for k in kept]):
                        total += pair[0].shape[1]
                        if total >= MAX_DIM:
                            break
                        fit.insert(0, pair)
                    if len(fit) > 1:
                        both = np.concatenate([p[0] for p in fit], 1), np.concatenate([p[1] for p in fit], 1)
                        read("cat" + end, *both)
                        cats[r] = both
                        ref[r, "cat"] = both[0].shape[1]
                        if eq:
                            read(f"cat_eq{eq}", *pca_fit(*both, eq), both[0].shape[1])
                        if r == "ms" and spec.get("eq") and ("gem", "cat") in ref:
                            read("cat_ms_eq", *pca_fit(*both, ref["gem", "cat"]), both[0].shape[1])
            for r, (train, val, raw) in vlads.items():  # a dimensao de `cat`, ou de `L` sem `cat`
                read(f"{r}_eq", *pca_fit(train, val, ref.get(("gem", "cat"), ref.get(("gem", "L"), MAX_DIM - 1))), raw)
                if raw < MAX_DIM:
                    read(r, train, val)
            for r, (train, val, raw) in seconds.items():
                read(f"L_{r}", train, val, raw)
                if spec.get("somar") and "gem" in cats:  # cat+<r>: soma, nao troca; cada bloco com o seu `pw_whiten` (shrinkage + L2), metade da dimensao
                    a, b = (pw_whiten(*pca_fit(*x, x[0].shape[1] // 2)) for x in (cats["gem"], (train, val)))
                    read(f"cat+{r}", np.concatenate([a[0], b[0]], 1), np.concatenate([a[1], b[1]], 1))
            if n and spec.get("soma") and "gem" in kept[0] and len({k["gem"][0].shape[1] for k in kept}) == 1:
                read("soma", sum(normalize(k["gem"][0]) for k in kept), sum(normalize(k["gem"][1]) for k in kept))
            for r, pair in atts.items():  # L_att_<mec>_<pos> e cat_att_<mec>_<pos>: cat com as camadas anteriores sem atencao
                row = row | {"extract_s": att_s[r]}
                read(f"L_{r}", *pair)
                fit, total = [pair], pair[0].shape[1]
                plain = "tta" if r.startswith("tta_") else "gem"  # cat_tta_att: as camadas anteriores tambem com TTA
                for k in reversed(kept[:-1]):
                    total += k[plain][0].shape[1]
                    if total >= MAX_DIM:
                        break
                    fit.insert(0, k[plain])
                if len(fit) > 1:
                    read(f"cat_{r}", np.concatenate([p[0] for p in fit], 1), np.concatenate([p[1] for p in fit], 1))
        if spec.get("salvar_encoder"):  # o encoder inteiro em formato FLIM, para o `init_flim_dir` do runner growth
            import shutil
            shutil.copytree(source, Path(spec["salvar_encoder"]) / arm
                            / f"{job['dataset_name']}_split{job['split']}_pct{job['percentage']}", dirs_exist_ok=True)
    return tuple(widths)


def arms_of(spec: dict) -> tuple:
    """Os bracos da rodada; `survivors` le o que a rodada anterior congelou."""
    if "survivors" not in spec:
        return spec["methods"], spec["controls"]
    with open(SURVIVORS, encoding="utf-8") as handle:
        frozen = json.load(handle)[spec["survivors"]]
    return frozen["methods"][:spec["top"]], frozen["controls"]


def tails(spec: dict) -> list:
    """Os sufixos `~variante` de `norm`, `ativacao`, `entre_camadas` e `plano_tol` em lista; sem essas chaves, so ""."""
    lam = spec.get("entre_camadas")
    axes = [spec.get("norm", ["n0"]), spec.get("ativacao", ["relu"]),
            [f"w{float(x):g}" for x in lam] if isinstance(lam, list) else ["n0"],
            [f"pl{float(x):g}" if x else "n0" for x in spec.get("plano_tol", [0])]]
    att = spec.get("atencao", {})
    if att.get("modo") == "b":  # um braco por mecanismo x posicao, mais o braco sem atencao
        axes.append(["n0"] + [f"ab_{m}_{p}" for m in att["mecanismos"] for p in att["camadas"]])
    return ["".join(f"~{t}" for t in combo if t not in ("n0", "relu")) for combo in itertools.product(*axes)]


def method_of(arm: str) -> str:
    return arm.split("~")[0].partition("@")[0]


def wanted(arm: str, plan: tuple) -> tuple:
    """`crelu` entrega o dobro de canais de proposito: nao e largura curta."""
    return tuple(2 * w for w in plan) if "crelu" in arm.split("~") else plan


def cells(spec: dict) -> list:
    """Barato primeiro; cada sorteio inteiro antes do seguinte, para o DISF rodar uma vez so."""
    methods, controls = arms_of(spec)
    depth, todo = spec["depth"], []
    ends = lambda control: [""]
    if "encdeep" in spec:
        from eval import encdeep  # tardio, ver `chain`
        ends = lambda control: encdeep.ends(spec, control)
    for seed in spec["draws"]:
        for kernel in spec["kernels"]:
            for index, width in enumerate(spec["widths"]):
                for arm, end in itertools.product(controls + methods, tails(spec)):
                    if arm in controls and "~ab_" in end:  # o aleatorio fica sem mecanismo
                        continue
                    for deep in ends(arm in controls):
                        if arm.partition("@")[0] not in FREE:
                            plan = tuple(law_channels(spec["lei"], kernel)) if "lei" in spec else (width,) * depth
                            todo.append((arm + end + deep, plan, kernel, seed))
                        elif index == 0:
                            todo.append((arm + end + deep, (0,) * depth, kernel, seed))
    return todo


def bank_pool(spec: dict, pool_all: list, seed: int) -> list:
    """`amostra.por_classe` = k: k imagens de treino por classe (regime B), ou o mesmo total sem olhar a classe (A)."""
    if "por_classe" not in spec:
        return random.Random(seed).sample(pool_all, min(spec["images"], len(pool_all)))
    rng, groups = random.Random(seed), {}
    for s in pool_all:
        groups.setdefault(s.label, []).append(s)
    by_class = [s for c in sorted(groups) for s in rng.sample(groups[c], min(spec["por_classe"], len(groups[c])))]
    return by_class if spec["regime"] == "B" else random.Random(seed).sample(pool_all, len(by_class))


def montar_camada(encoder_dir: str | None, k: int, cfg: dict, out_dir: str) -> str:
    """So a camada k sobre L1..L(k-1) de `encoder_dir` (formato FLIM); grava L1..Lk direto em out_dir.

    cfg: dataset, split, percentage, out_channels, kernel_size, pool_stride, selecao (fps|random_z|<braco>);
    opcionais: softhebb (a config `hebb` do encdeep), n_imagens (500), seed (42), entre_camadas, image_size (200),
    whiten_between (com entre_camadas: o whitening sai como `conv{k}-whiten.npz` + pesos sem dobra, flim.arch.Whiten),
    superpixels (50), image_cache_dir, work_dir. Com k=1, encoder_dir e ignorado.
    `branches` [[kernel, largura], ...]: camada Inception — um SPiFiL por ramo (banco de patches, FPS e SoftHebb
    proprios, sobre as mesmas features de L1..L(k-1)) em out_dir/ramo<i>; out_dir recebe os ramos como um conv KxK
    (K = maior kernel, os menores com zeros em volta) e `conv.branches` no architecture.json (flim.arch.Branches).
    """
    seed, arm = cfg.get("seed", 42), cfg["selecao"]
    amostra = {"depth": k, "pool": [1] * (k - 1) + [cfg["pool_stride"]], "lei": {"k": "-".join([str(cfg["kernel_size"])] * k)},
               "images": cfg.get("n_imagens", 500), "entre_camadas": cfg.get("entre_camadas", False),
               "whiten_between": cfg.get("whiten_between", False),
               "uma_camada": {"encoder_dir": encoder_dir, "k": k, "out": out_dir}}
    if cfg.get("softhebb"):
        amostra["encdeep"] = {"kernels": [cfg["kernel_size"]] * k, "camadas": [k], "hebb": cfg["softhebb"]}
        arm += "~hebb"
    job = {"stem": f"uma_camada_L{k}", "dataset_name": cfg["dataset"], "split": cfg["split"], "percentage": cfg["percentage"],
           "image_size": cfg.get("image_size", 200), "no_crop": True, "superpixels": [cfg.get("superpixels", 50)],
           "image_cache_dir": cfg.get("image_cache_dir", "/exps/moliveira/image_cache"),
           "grow": {"impurities": True, "superpixels": "disf", "budget_check": False},
           "work_dir": cfg.get("work_dir", str(Path(out_dir).parent)), "amostra": amostra, "draw": seed}
    Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
    images, to_float = load_images(job)
    *ready, _ = stored_sources(job, bank_pool(amostra, train_pool(job), seed))
    if "branches" not in cfg:
        chain(job, images, to_float, *ready, arm, (0,) * (k - 1) + (cfg["out_channels"],), cfg["kernel_size"], seed, {}, lambda *a: None)
        return out_dir
    from flim.weights import get_bias, get_weights
    parts, ramos, ws, bs = [], [], [], []
    for i, (kb, wb) in enumerate(cfg["branches"]):
        part = Path(out_dir) / f"ramo{i}"
        amostra |= {"lei": {"k": "-".join([str(kb)] * k)}, "uma_camada": amostra["uma_camada"] | {"out": str(part)}}
        if "encdeep" in amostra:
            amostra["encdeep"]["kernels"] = [kb] * k
        chain(job, images, to_float, *ready, arm, (0,) * (k - 1) + (wb,), kb, seed, {}, lambda *a: None)
        n = json.loads((part / "architecture.json").read_text())[f"layer{k}"]["conv"]["noutput_channels"]
        flat = np.load(part / f"conv{k}-kernels.npy")
        ws.append(torch.from_numpy(get_weights(part / f"conv{k}-kernels.npy", (kb, kb), flat.shape[0] // kb**2)).float())
        bs.append(torch.from_numpy(get_bias(part / f"conv{k}-bias.txt")))
        parts, ramos = parts + [part], ramos + [{"kernel_size": kb, "noutput_channels": n}]
    big = max(kb for kb, _ in cfg["branches"])
    weight = torch.cat([F.pad(w, [(big - w.shape[-1]) // 2] * 4) for w in ws])
    arch = json.loads((parts[0] / "architecture.json").read_text())
    arch[f"layer{k}"]["conv"] |= {"kernel_size": [big, big, 0], "noutput_channels": len(weight), "nkernels_per_marker": len(weight),
                                  "nkernels_per_image": len(weight), "branches": ramos}
    spifil.write_weights(Path(out_dir), parts[0], arch, k, SimpleNamespace(conv=SimpleNamespace(weight=weight, bias=torch.cat(bs))))
    return out_dir


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    torch.set_num_threads(2)
    with open(config, encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    spec = job["amostra"]
    if isinstance(spec.get("por_classe"), list):
        for k, regime in itertools.product(spec["por_classe"], "BA"):
            run(job | {"amostra": spec | {"por_classe": k, "regime": regime},
                       "results_csv": job["results_csv"].replace(".csv", f"_pc{k}{regime}.csv")})
        return
    for lei in spec.get("leis", []):  # cada lei e UMA arquitetura, com o proprio CSV: a lei entra na identidade da celula
        depth, pool = lei["depth"], pools(spec)
        run(job | {"amostra": {k: v for k, v in spec.items() if k != "leis"} | {
            "lei": lei, "depth": depth, "widths": [0], "pool": (pool + [1] * depth)[:depth]},
            "results_csv": job["results_csv"].replace(".csv", f"_lei{law_name(lei)}.csv")})
    if "leis" not in spec:
        run(job)


def run(job: dict) -> None:
    spec = job["amostra"]
    path = Path(job["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
    expected = (2 * spec["depth"] - 1 + (spec["depth"] - 1) * bool(spec.get("soma"))) * (1 + bool(spec.get("entre_camadas")))
    if "leituras" in spec:  # `cat` depende da dimensao: conta so as linhas `L`
        vlads = sum(r.startswith("vlad") or r in SECOND for r in spec["leituras"])  # `vlad<K>_eq` uma vez, na ultima camada
        expected = (spec["depth"] * (len(spec["leituras"]) - vlads) + vlads) * (1 + bool(spec.get("entre_camadas")))
    elif spec.get("tta"):
        expected *= 2
    if spec.get("atencao", {}).get("modo", "a") == "a" and "atencao" in spec:  # L_att e cat_att por mecanismo e posicao (cat so com camada anterior)
        expected += len(spec["atencao"]["mecanismos"]) * sum(1 + (p > 1) for p in att_layers(spec)) * (1 + bool(spec.get("entre_camadas"))) * (1 + bool(spec.get("tta")))
    if "encdeep" in spec:
        from eval import encdeep  # tardio, ver `chain`
        check_encdeep(spec)
        expected += (spec["depth"] - 1) * (1 + bool(spec.get("entre_camadas"))) * bool(spec["encdeep"].get("dim_eq"))  # cat_eq
    seen_rows, twins = {}, []
    if path.exists():
        with open(path, newline="") as handle:
            for r in csv.DictReader(handle):
                if r["status"] == "ok":
                    key = (r["arm"], tuple(map(int, r["plan"].split("-"))), int(r["kernel"]), int(r["seed"]))
                    seen_rows.setdefault(key, []).append(r)
    done = {key for key, rows in seen_rows.items() if len(rows) >= expected}
    shard, shards = job.get("shard", [0, 1])
    todo = cells(spec)[shard::shards]
    for key in done:
        got = tuple(map(int, seen_rows[key][-1]["widths"].split("-")))
        if method_of(key[0]) not in CONTROLS and len(got) == spec["depth"] and got != wanted(*key[:2]):
            twins += [(c + "".join(f"~{x}" for x in key[0].split("~")[1:] if not x.startswith("ab_")), got, key[2], key[3]) for c in spec["twin_controls"]]
    todo = [c for c in todo + twins if c not in done]
    if not todo:
        return
    if spec.get("avaliar") == "test":  # o mesmo encoder (so treino o constroi) pontuado no teste; CSV proprio no YAML
        sets, l1_fast.SETS = l1_fast.SETS, {"test": "test"}  # l1_amostras tira o teste do load_images; so aqui sobe
        try:
            images, to_float = load_images(job)
        finally:
            l1_fast.SETS = sets
        images["validation"] = images.pop("test")
    else:
        images, to_float = load_images(job)
    fundo = spec.get("fundo", "real")
    if fundo == "zerado":
        images = zero_background(job, images)
    place = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    pool_all = train_pool(job)
    base = place | {"image_size": job["image_size"], "n_superpixels": job["superpixels"][0], "fundo": fundo,
                    "n_train": len(images["train"][1]), "n_val": len(images["validation"][1]), "status": "ok"}
    if "lei" in spec:
        params, macs, maps, rf = conv_cost(cells(spec)[0][1], lei_kernels(spec, spec["kernels"][0], spec["depth"]), job["image_size"], spec["pool"])
        base |= {"lei": f"{spec['lei']['lei']}.{LAW_VERSION}", "lei_args": law_name(spec["lei"]), "depth_req": spec["depth"],
                 "pools": "-".join(map(str, spec["pool"])), "params": params, "macs": macs, "maps": "-".join(map(str, maps)), "rf": rf}
    if "por_classe" in spec:
        base |= {"regime": spec["regime"], "por_classe": spec["por_classe"]}
    guard = threading.Lock()
    with open(path, "a", newline="") as handle, ThreadPoolExecutor(spec["svm_workers"]) as fits:
        mix = "agrupamento" in spec or "pesos" in spec.get("saliency", {})
        sol, deep = "leituras" in spec, "encdeep" in spec
        writer = csv.DictWriter(handle, FIELDS + EXTRA * (mix or bool(spec.get("multiescala")) or sol or deep) + CLUSTER * mix
                                + SOL * (sol or deep) + (encdeep.FIELDS if deep else [])
                                + ["regime", "por_classe"] * ("por_classe" in spec)
                                + [f for f in LEI if not (mix and f in CLUSTER)] * ("lei" in spec),
                                extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def write(row: dict) -> None:
            with guard:
                writer.writerow(row)
                handle.flush()

        ids = encdeep.val_ids(job) if spec.get("encdeep", {}).get("erros") else None

        def fit(row: dict, args: tuple) -> None:
            try:
                got = score_full(*args)
                if "pred" in got:
                    encdeep.save_errors(Path(str(path).replace(".csv", "_erros.csv")), row, ids, args[3], got.pop("pred"), guard)
                write(row | got)
            except Exception as exc:  # noqa: BLE001
                write(row | {"status": "error", "error": repr(exc)[:500]})

        def submit(row: dict, args: tuple) -> None:
            fits.submit(fit, row, args)

        cache = {}
        while todo:
            arm, plan, kernel, seed = cell = todo.pop(0)
            if seed not in cache:
                drawn = bank_pool(spec, pool_all, seed)
                job["draw"] = seed
                *ready, diag = stored_sources(job, drawn)
                if "dino" in spec:
                    add_dino(job, drawn, *ready[:3])
                cache = {seed: ready}
                if shard == 0:
                    with open(str(path).replace(".csv", "_diag.csv"), "a", newline="") as side:
                        sheet = csv.DictWriter(side, DIAG)
                        if side.tell() == 0:
                            sheet.writeheader()
                        sheet.writerows({"dataset": job["dataset_name"], "split": job["split"], "fundo": fundo,
                                         "image_size": job["image_size"], "seed": seed} | r for r in diag)
            done.add(cell)
            try:
                got = chain(job, images, to_float, *cache[seed], arm, plan, kernel, seed, base, submit)
            except Exception as exc:  # noqa: BLE001
                write(base | {"arm": arm, "plan": "-".join(map(str, plan)), "kernel": kernel, "seed": seed,
                              "layer": 0, "readout": "", "status": "error", "error": repr(exc)[:500]})
                torch.cuda.empty_cache()
                continue
            if method_of(arm) not in CONTROLS and got != wanted(arm, plan):
                twin = [(c + "".join(f"~{x}" for x in arm.split("~")[1:] if not x.startswith("ab_")), got, kernel, seed) for c in spec["twin_controls"]]
                todo[:0] = [c for c in twin if c not in done and c not in todo]
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
