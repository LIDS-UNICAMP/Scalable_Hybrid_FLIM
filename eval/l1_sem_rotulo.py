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
"""l1_sem_rotulo.py — banco de kernels que nunca le a classe, nem o numero de classes.

Um job = (dataset, split), `source: l1_sem_rotulo`. As imagens saem do treino por sorteio, os
patches pelas funcoes do SPiFiL (superpixel, medoide, `extract_patches`), e um metodo classico
escolhe as linhas. `build_filters` dobra as escolhidas em conv, como no SPiFiL. A camada seguinte
repete a escolha sobre os mapas da anterior. Sem backprop; leitura e SVM de `eval/l1_amostras.py`.

Metodos. Sem numero de grupos: hdbscan, meanshift, affinity, kmeans_sil, gmm_bic, gs (Gram-Schmidt
com limiar). Por largura pedida: uniform (patches sorteados, o piso), fps (distancia maxima em
|cos|), qr (pivotado; para no posto, D = 75 na L1), dpp (MAP guloso, nucleo RBF, para o nucleo
linear daria o mesmo que qr), cos_prune (`prune` de flim/kernel_redundancy.py). Aleatorio: rand,
a dobra do uniform com direcao sorteada.

`modo: desbalanceio` mede a redundancia contra o peso da maior classe. A classe so entra para
sortear a amostra e na analise; o banco nao a recebe.
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.linalg import qr
from sklearn.cluster import HDBSCAN, AffinityPropagation, MeanShift, MiniBatchKMeans, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from spifil.color import LabNorm
from spifil.config import LayerSpec
from spifil.nn.builder import build_filters
from spifil.patches import extract_patches
from spifil.superpixels.disf import DISF
from spifil.types import PatchSet, Seeds

from eval.l1_amostras import bank_stats, cover_sample, coverage, kernel_stats, read
from eval.l1_fast import fit_pending, load_images
from eval.svm import DEVICE, cli_kwargs
from experiments.ray.runners.growth import _paths, _short
from flim import spifil
from flim.arch import parse_architecture
from flim.build import build
from flim.kernel_redundancy import activation_sample, measure, prune
from flim.seed_points import SeedPoints

CLUSTER_ROWS = 5000   # O(N^2) so numa amostra
QUADRATIC_ROWS = 2000  # meanshift e affinity
CLUSTER_DIMS = 64     # acima disso os agrupadores veem PCA; a escolha continua sendo de linhas reais
K_GRID = [8, 16, 32, 64, 128, 256]
MIN_GROUP = 10
GS_TAU = 0.3          # sobra minima da norma depois de projetar nos ja escolhidos
PRUNE_COS = 0.9
MAX_WIDTH = 256
FLAT = 1e-3
FIELDS = ["dataset", "split", "percentage", "image_size", "n_superpixels", "arm", "k", "width_req",
          "seed", "layer", "readout", "width", "widths", "images", "candidates", "distinct",
          "flat_frac", "groups", "top_group", "noise_frac", "copies", "copies_frac", "cos_max",
          "cos_mean", "erank", "cover_all", "cover_c70", "dim", "n_train", "n_val", "val_kappa",
          "val_acc", "val_f1", "fit_status", "select_s", "extract_s", "fit_s", "lock_wait_s",
          "status", "error"]
IMBALANCE = ["dataset", "split", "arm", "seed", "dominant", "share", "top_share", "imbalance",
             "n_classes", "images", "width", "cos_mean", "cos_max", "cos_nearest_mean",
             "corr_nearest_mean", "corr_max", "effective_rank"]


def train_pool(job: dict) -> list:
    """Todas as amostras de treino; `Sample.label` existe, quem escolhe o banco nao o le."""
    point = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    return spifil.train_samples(_short(point), job["split"], job["percentage"], 10**9,
                                one_per_class=False, seed=0)


def prepared(job: dict, samples: list) -> tuple[list, list]:
    """Imagens LAB como o encoder as ve e sementes medoide sem rotulo (classe 0 em todas)."""
    size, no_crop = job["image_size"], job["no_crop"]
    data = spifil.Cropped(samples, size, use_mask=False, no_crop=no_crop)
    images = spifil.encoder_features(samples, torch.nn.Identity(), size, False, DEVICE, no_crop)
    seeds = []
    for index in range(len(samples)):
        found = spifil.preparation.prepare_sample(
            data, index, superpixels=DISF(), seed_extractor=SeedPoints("medoid"),
            n_superpixels=job["superpixels"][0], color=LabNorm()).seeds
        seeds.append(Seeds.from_coords(found.coords, 0, found.grid))
    return images, seeds


@torch.no_grad()
def patch_pool(images: list, seeds: list, encoder, spec) -> PatchSet:
    """Uma linha por semente, nos mapas da ultima camada pronta; fica em RAM, nao na GPU."""
    parts = []
    for index, (image, seed) in enumerate(zip(images, seeds)):
        feature = encoder(image[None])[0]
        found = extract_patches(feature, spifil.rescale(seed, tuple(feature.shape[1:])), spec, image_id=index)
        parts.append(PatchSet(found.feats.cpu(), found.labels.cpu(), found.seed_rows.cpu(), found.image_ids.cpu()))
    return PatchSet.cat(parts)


def normalized(pool: PatchSet, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    """z-score por dimensao, sem janela chapada nem repetida, norma 1; ate CLUSTER_ROWS linhas."""
    rows = pool.feats.double().numpy()
    flat = rows.std(1) < FLAT
    z = (rows - rows.mean(0)) / (rows.std(0) + 0.01)
    hashed = (z @ np.random.default_rng(0).standard_normal(z.shape[1])).round(6)
    keep = np.flatnonzero(~flat)
    keep = keep[np.unique(hashed[keep], return_index=True)[1]]
    keep = rng.permutation(keep)[:CLUSTER_ROWS]
    z = z[keep]
    return z / np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12), keep, float(flat.mean())


def representatives(z: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    """A linha real mais perto da media de cada grupo, maiores primeiro; -1 e ruido."""
    named = groups[groups >= 0]
    sizes = Counter(named.tolist())
    picks = []
    for group, _ in sizes.most_common(MAX_WIDTH):
        idx = np.flatnonzero(groups == group)
        picks.append(idx[(z[idx] @ z[idx].mean(0)).argmax()])
    return np.array(picks), {"groups": len(sizes), "top_group": max(sizes.values()) / len(groups),
                             "noise_frac": float((groups < 0).mean())}


def reduced(z: np.ndarray, seed: int) -> np.ndarray:
    return PCA(CLUSTER_DIMS, random_state=seed).fit_transform(z) if z.shape[1] > CLUSTER_DIMS else z


def hdbscan(z, width, seed):
    return representatives(z, HDBSCAN(min_cluster_size=MIN_GROUP).fit_predict(reduced(z, seed)))


def meanshift(z, width, seed):
    x = reduced(z[:QUADRATIC_ROWS], seed)
    band = estimate_bandwidth(x, quantile=0.1, n_samples=500, random_state=seed)
    return representatives(z[:QUADRATIC_ROWS], MeanShift(bandwidth=band, bin_seeding=True).fit_predict(x))


def affinity(z, width, seed):
    x = reduced(z[:QUADRATIC_ROWS], seed)
    return representatives(z[:QUADRATIC_ROWS], AffinityPropagation(damping=0.9, random_state=seed).fit_predict(x))


def kmeans_sil(z, width, seed):
    x = reduced(z, seed)
    fits = [MiniBatchKMeans(k, n_init=3, random_state=seed).fit_predict(x) for k in K_GRID]
    scores = [silhouette_score(x, groups, sample_size=QUADRATIC_ROWS, random_state=seed) for groups in fits]
    return representatives(z, fits[int(np.argmax(scores))])


def gmm_bic(z, width, seed):
    x = reduced(z, seed)
    fits = [GaussianMixture(k, covariance_type="diag", max_iter=30, random_state=seed).fit(x) for k in K_GRID]
    return representatives(z, fits[int(np.argmin([g.bic(x) for g in fits]))].predict(x))


def uniform(z, width, seed):
    return np.arange(min(width, len(z))), {}


def fps(z, width, seed):
    picks, near = [0], np.abs(z @ z[0])
    for _ in range(min(width, len(z)) - 1):
        picks.append(int(near.argmin()))
        near = np.maximum(near, np.abs(z @ z[picks[-1]]))
    return np.array(picks), {}


def _pivots(z):
    r, order = qr(z.T, mode="r", pivoting=True)
    return order, np.abs(np.diag(r))


def qr_pivot(z, width, seed):
    order, diag = _pivots(z)
    return order[:min(width, int((diag > 1e-8).sum()))], {}


def gs(z, width, seed):
    order, diag = _pivots(z)
    return order[:min(MAX_WIDTH, int((diag >= GS_TAU).sum()))], {}


def dpp(z, width, seed):
    """MAP guloso de Chen et al. 2018; o nucleo linear daria o mesmo pivo do QR."""
    gram = z @ z.T
    dist = (2 - 2 * gram).clip(0)
    kernel = np.exp(-dist / np.median(dist))
    m = min(width, len(z))
    chol, gain, picks = np.zeros((m, len(z))), np.diag(kernel).copy(), []
    for t in range(m):
        j = int(gain.argmax())
        picks.append(j)
        chol[t] = (kernel[j] - chol[:t].T @ chol[:t, j]) / np.sqrt(max(gain[j], 1e-12))
        gain -= chol[t] ** 2
        gain[j] = -np.inf
    return np.array(picks), {}


def kmeans(z, width, seed):
    """Com largura dada: so para a medida de desbalanceio, que precisa do mesmo tamanho de banco."""
    return representatives(z, MiniBatchKMeans(width, n_init=3, random_state=seed).fit_predict(reduced(z, seed)))


def cos_prune(z, width, seed):
    """O `prune` do projeto, com uma classe so: entram os primeiros, repoe da sobra."""
    hit = torch.from_numpy(z @ z.T > PRUNE_COS)
    n = len(z)
    kept, _, _ = prune(hit, list(range(min(width, n))), list(range(min(width, n), n)), [0] * n, {0: width}, True)
    return np.array(kept), {}


METHODS = {"hdbscan": hdbscan, "meanshift": meanshift, "affinity": affinity, "kmeans_sil": kmeans_sil,
           "gmm_bic": gmm_bic, "gs": gs, "uniform": uniform, "fps": fps, "qr": qr_pivot, "dpp": dpp,
           "cos_prune": cos_prune, "rand": uniform, "kmeans": kmeans}
FREE = {"hdbscan", "meanshift", "affinity", "kmeans_sil", "gmm_bic", "gs"}


def layer_arch(arch: dict, spec: dict, n: int, width: int) -> dict:
    return spifil.grown_arch(arch, spec["kernels"][n], width, spec["pools"][n])


def chain(job: dict, images: dict, to_float, sample: dict, seed: int, method: str, width: int,
          lab: list, seeds: list, base: dict, write) -> list:
    """L1..Ldepth de um metodo: cada camada escolhe nos mapas da anterior; devolve os SVMs a fazer."""
    spec = job["amostra"]
    point = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"]}
    source = Path(_paths(point)[1])
    arch = parse_architecture(_paths(point)[0])
    arch = {k: v for k, v in arch.items() if not k.startswith("layer")} | {"nlayers": 0}
    encoder, widths, pending = torch.nn.Identity(), [], []
    tag = f"semrotulo {job['stem']} {method} w{width} s{seed}"
    row = base | {"arm": method, "width_req": width, "seed": seed, "k": 0}
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory(dir=job["work_dir"]) as tmp:
        try:
            for n in range(spec["depth"]):
                start = time.perf_counter()
                target = spec["channels"][n] or width
                layer = LayerSpec(kernel_size=spec["kernels"][n], out_channels=target)
                pool = patch_pool(lab, seeds, encoder, layer)
                z, keep, flat = normalized(pool, rng)
                picks, info = METHODS[method](z, target, seed)
                bank = build_filters(pool.select(torch.as_tensor(keep[picks])), layer)
                weight, bias = bank.weight, bank.bias
                if method == "rand":
                    noise = torch.randn(weight.flatten(1).shape, generator=torch.Generator().manual_seed(seed + n))
                    w = F.normalize(noise, dim=1) / bank.stdev
                    weight, bias = w.view_as(weight), -(w * bank.mean).sum(1)
                out = Path(tmp) / f"L{n + 1}"
                arch = layer_arch(arch, spec, n, len(weight))
                spifil.write_weights(out, source, arch, n + 1, SimpleNamespace(
                    conv=SimpleNamespace(weight=weight, bias=bias)))
                source, widths = out, widths + [len(weight)]
                select_s = round(time.perf_counter() - start, 2)
                encoder = build(str(out / "architecture.json"), "flim", str(out)).to(DEVICE).eval()
                stats = bank_stats(out, n + 1) | {
                    "layer": n + 1, "widths": "-".join(map(str, widths)), "images": len(lab),
                    "candidates": len(pool), "distinct": len(z), "flat_frac": flat, "select_s": select_s} | info
                if n == 0:
                    stats |= coverage(weight.flatten(1).to(DEVICE), bias.to(DEVICE), bank.mean.to(DEVICE),
                                      bank.stdev.to(DEVICE), sample)
                pending += read(job, images, to_float, encoder, row | stats, tag)
        except Exception as exc:  # noqa: BLE001
            write(row | {"layer": 0, "readout": "", "status": "error", "error": repr(exc)[:500]})
    return pending


def run_bank(job: dict, path: Path) -> None:
    spec = job["amostra"]
    done = set()
    if path.exists():
        with open(path, newline="") as handle:
            done = {(r["arm"], int(r["width_req"]), int(r["seed"])) for r in csv.DictReader(handle)
                    if r["status"] == "ok" and int(r["layer"]) == spec["depth"]}
    todo = [(seed, m, w) for seed in spec["draws"] for m in spec["methods"]
            for w in ([0] if m in FREE else spec["widths"]) if (m, w, seed) not in done]
    if not todo:
        return
    images, to_float = load_images(job)
    sample = cover_sample(images, to_float)
    base = {"dataset": job["dataset_name"], "split": job["split"], "percentage": job["percentage"],
            "image_size": job["image_size"], "n_superpixels": job["superpixels"][0],
            "n_train": len(images["train"][1]), "n_val": len(images["validation"][1]), "status": "ok"}
    pool_all = train_pool(job)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def write(row: dict) -> None:
            writer.writerow(row)
            handle.flush()

        cache = {}
        for seed, method, width in todo:
            if seed not in cache:
                cache = {seed: prepared(job, random.Random(seed).sample(pool_all, min(spec["images"], len(pool_all))))}
            pending = chain(job, images, to_float, sample, seed, method, width, *cache[seed], base, write)
            fit_pending(pending, write)
            if method in FREE and pending and "rand" in spec["methods"]:
                # sem largura pedida: o par aleatorio nasce na largura que o metodo achou
                got = pending[0][0]["width"]
                if ("rand", got, seed) not in done and got not in spec["widths"]:
                    fit_pending(chain(job, images, to_float, sample, seed, "rand", got, *cache[seed], base, write), write)
                    done.add(("rand", got, seed))


def run_imbalance(job: dict, path: Path) -> None:
    """Mesmo tamanho de banco e de amostra: so muda o peso da classe dominante."""
    spec = job["amostra"]
    everything = train_pool(job)
    by_class = {}
    for item in everything:
        by_class.setdefault(item.label, []).append(item)
    images, to_float = load_images(job)
    x = images["validation"][0]
    reference = list(to_float(x[::max(1, len(x) // 50)].to(DEVICE)))
    width = spec["widths"][0]
    layer = LayerSpec(kernel_size=spec["kernels"][0], out_channels=width)
    windows_ref = activation_sample(reference, layer)
    draws = [(None, None, seed) for seed in spec["draws"]]
    draws += [(c, p, seed) for c in sorted(by_class) for p in spec.get("shares", []) for seed in spec["draws"]]
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, IMBALANCE)
        if handle.tell() == 0:
            writer.writeheader()
        for dominant, share, seed in draws:
            rng = random.Random(seed)
            if dominant is None:
                chosen = rng.sample(everything, min(spec["images"], len(everything)))
            else:
                top = round(share * spec["images"])
                rest = (spec["images"] - top) // (len(by_class) - 1)
                chosen = [s for c, items in by_class.items()
                          for s in rng.sample(items, min(top if c == dominant else rest, len(items)))]
            counts = np.array(list(Counter(s.label for s in chosen).values()), dtype=float)
            p = counts / counts.sum()
            lab, seeds = prepared(job, chosen)
            pool = patch_pool(lab, seeds, torch.nn.Identity(), layer)
            z, keep, _ = normalized(pool, np.random.default_rng(seed))
            for arm in spec["methods"]:
                picks, _ = METHODS[arm](z, width, seed)
                selected = pool.select(torch.as_tensor(keep[picks]))
                kernels = build_filters(selected, layer).weight.flatten(1).double().numpy()
                stats = kernel_stats(kernels) | measure(selected, layer, windows_ref.cpu())
                writer.writerow({
                    "dataset": job["dataset_name"], "split": job["split"], "arm": arm, "seed": seed,
                    "dominant": dominant, "share": share, "top_share": p.max(),
                    "imbalance": 1 + (p * np.log(p)).sum() / np.log(len(by_class)),
                    "n_classes": len(by_class), "images": len(chosen)} | {
                        k: stats[k] for k in IMBALANCE[11:]} | {"width": stats["width"]})
                handle.flush()


def main(config: str, ckpt_selection: str = "best") -> None:
    del ckpt_selection
    with open(config, encoding="utf-8") as handle:
        job = yaml.safe_load(handle)
    path = Path(job["results_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(job["work_dir"]).mkdir(parents=True, exist_ok=True)
    (run_imbalance if job["amostra"].get("modo") == "desbalanceio" else run_bank)(job, path)


if __name__ == "__main__":
    main(**cli_kwargs(sys.argv[1:]))
