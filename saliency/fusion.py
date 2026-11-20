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
"""Escolha e fusao de mapas de saliencia sem rotulo: dict nome -> mapa HxW em [0,1] + imagem RGB.

Metodo A: map_features -> quality_scores -> top_k. Metodo B: weighted_mean, consensus_fusion,
regional_fusion, pca_fusion, nmf_fusion, cluster_fusion, search_fusion; baselines best_single e
simple_mean; fuse roda tudo. Nada disso recebe mascara: evaluate e a unica funcao que ve gt, so
mede depois, e nenhum resultado dela volta para a escolha. Saliente nao quer dizer alvo: todas
as caracteristicas descrevem a forma do mapa, nao o que ele cobre.
"""

import inspect
import time
import warnings

import numpy as np
from scipy import ndimage, sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import differential_evolution
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist, squareform
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.segmentation import slic
from skimage.transform import AffineTransform, warp
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

# Peso negativo: quanto menor a caracteristica, melhor.
DEFAULT_QUALITY = {
    "spatial_entropy": -1.0,
    "region_contrast": 1.0,
    "concentration": 1.0,
    "continuity": 1.0,
    "fragmentation": -1.0,
    "scale_consistency": 1.0,
    "stability": 1.0,
    "agreement": 1.0,
    "separation": 1.0,
}
SMALL_WARPS = ((0.0, 1.0, (0.02, -0.02)), (3.0, 1.0, (0.0, 0.0)), (0.0, 0.95, (0.0, 0.0)))


def _unit(x):
    x = np.asarray(x, np.float32)
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-8 else np.zeros_like(x)


def _corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 1e-12 else 0.0


def _stack(maps):
    return np.stack([m.ravel() for m in maps.values()]).astype(np.float64)


def _corr_matrix(maps):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.nan_to_num(np.corrcoef(_stack(maps)))
    np.fill_diagonal(r, 1.0)
    return r


def _high(m):
    return m > threshold_otsu(m) if np.ptp(m) > 1e-6 else np.zeros(m.shape, bool)


def _check(maps, img):
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"imagem precisa ser uint8 HxWx3, veio {img.dtype} {img.shape}")
    for name, m in maps.items():
        if m.shape != img.shape[:2] or not np.isfinite(m).all() or m.min() < 0 or m.max() > 1:
            raise ValueError(f"mapa {name} precisa ser HxW finito em [0,1]")


def independence(maps, power=2.0):
    """Peso por mapa: k copias exatas valem 1/k cada, ou seja, um voto para o grupo todo."""
    r = np.clip(_corr_matrix(maps), 0, 1) ** power
    return 1.0 / r.sum(1)


def _single_features(m, lab):
    """Tudo em [0,1]. spatial_entropy e exp(H)/N: fracao efetiva da imagem que o mapa ocupa."""
    if np.ptp(m) < 1e-6:
        return {
            "spatial_entropy": 1.0, "region_contrast": 0.0, "concentration": 0.0, "continuity": 0.0,
            "fragmentation": 1.0, "scale_consistency": 0.0, "separation": 0.0,
        }
    h, w = m.shape
    side = min(h, w)
    m = m.astype(np.float64)
    p = m / m.sum()
    nz = p[p > 0]
    entropy = np.exp(-(nz * np.log(nz)).sum()) / m.size

    py, px = p.sum(1), p.sum(0)
    yy, xx = np.linspace(0, 1, h), np.linspace(0, 1, w)
    spread = py @ (yy - py @ yy) ** 2 + px @ (xx - px @ xx) ** 2

    flat = lab.reshape(-1, 3)
    inside = m.ravel() @ flat / m.sum()
    outside = (1 - m).ravel() @ flat / max((1 - m).sum(), 1e-8)

    x = m - m.mean()
    moran = ((x[:, :-1] * x[:, 1:]).mean() + (x[:-1] * x[1:]).mean()) / (2 * (x * x).mean())

    pieces, n = ndimage.label(_high(m))
    area = np.bincount(pieces.ravel())[1:] if n else np.ones(1)
    smooth = ndimage.gaussian_filter(m, 0.01 * side)
    high = _high(smooth)
    gap = smooth[high].mean() - smooth[~high].mean() if high.any() and not high.all() else 0.0
    scales = [_corr(m, ndimage.gaussian_filter(m, s * side)) for s in (0.01, 0.02, 0.04)]
    return {
        "spatial_entropy": float(entropy),
        "region_contrast": float(min(np.linalg.norm(inside - outside) / 100, 1.0)),
        "concentration": float(max(1 - spread * 6, 0.0)),
        "continuity": float(np.clip(moran, 0, 1)),
        "fragmentation": float(1 - ((area / area.sum()) ** 2).sum()),
        "scale_consistency": float(np.clip(np.mean(scales), 0, 1)),
        "separation": float(gap),
    }


def _moves(img, seed=0):
    """As imagens mexidas, iguais para todo mapa: (movimento, imagem, pixels que seguem visiveis) e o ruido."""
    h, w = img.shape[:2]
    centre = AffineTransform(translation=(-w / 2, -h / 2)).params
    moves = []
    for angle, scale, (dy, dx) in SMALL_WARPS:
        move = AffineTransform(rotation=np.deg2rad(angle), scale=scale).params
        back_to = AffineTransform(translation=(w / 2 + dx * w, h / 2 + dy * h)).params
        t = AffineTransform(matrix=back_to @ move @ centre)
        moved = warp(img, t.inverse, mode="edge", preserve_range=True).astype(np.uint8)
        moves.append((t, moved, warp(np.ones((h, w)), t, cval=0.0) > 0.999))
    noise = np.random.default_rng(seed).normal(0, 4, img.shape)
    return moves, np.clip(img + noise, 0, 255).astype(np.uint8)


def _stability(m, remake, moves, noisy):
    """Refaz o mapa em imagens levemente mexidas, desfaz o movimento e correlaciona com m."""
    values = []
    for t, moved, seen in moves:
        back = warp(remake(moved), t, mode="edge", preserve_range=True)
        values.append(_corr(m[seen], back[seen]))
    values.append(_corr(m, remake(noisy)))
    return float(np.clip(np.mean(values), 0, 1))


def map_features(maps, img, remake=None):
    """Caracteristicas por mapa. remake(nome, img) -> mapa refaz o algoritmo para stability."""
    _check(maps, img)
    lab = rgb2lab(img)
    feats = {name: _single_features(m, lab) for name, m in maps.items()}
    if len(maps) > 1:
        r, ind = np.clip(_corr_matrix(maps), 0, 1), independence(maps)
        for i, name in enumerate(maps):
            others = np.arange(len(maps)) != i
            feats[name]["agreement"] = float(r[i, others] @ ind[others] / ind[others].sum())
    if remake is None:
        warnings.warn("sem remake: stability fica fora do score")
    else:
        moves, noisy = _moves(img)
        for name, m in maps.items():
            feats[name]["stability"] = _stability(m, lambda x: remake(name, x), moves, noisy)
    return feats


def quality_scores(feats, quality=DEFAULT_QUALITY):
    """quality: dict caracteristica -> peso, ou funcao dict de caracteristicas -> float."""
    if callable(quality):
        return {name: float(quality(f)) for name, f in feats.items()}
    scores = {}
    for name, f in feats.items():
        used = {k: w for k, w in quality.items() if k in f}
        total = sum(w * f[k] if w > 0 else -w * (1 - f[k]) for k, w in used.items())
        scores[name] = total / sum(abs(w) for w in used.values())
    return scores


def top_k(scores, k, maps=None, redundancy=0.5):
    """Maiores scores. Com maps, cada novo escolhido paga redundancy x correlacao com os ja escolhidos."""
    names = sorted(scores, key=scores.get, reverse=True)
    if maps is None:
        return names[:k]
    r = dict(zip(maps, np.clip(_corr_matrix(maps), 0, 1)))
    index = {name: i for i, name in enumerate(maps)}
    chosen = names[:1]
    while len(chosen) < min(k, len(names)):
        rest = [n for n in names if n not in chosen]
        chosen.append(max(rest, key=lambda n: scores[n] - redundancy * max(r[n][index[c]] for c in chosen)))
    return chosen


def _weights(maps, scores, temperature=0.1):
    s = np.array([scores[name] for name in maps])
    w = np.exp((s - s.max()) / temperature) * independence(maps)
    return w / w.sum()


def _mix(maps, w):
    return np.tensordot(w, np.stack(list(maps.values())), axes=1).astype(np.float32)


def simple_mean(maps):
    return np.mean(list(maps.values()), axis=0).astype(np.float32)


def best_single(maps, scores):
    return maps[max(scores, key=scores.get)]


def weighted_mean(maps, scores):
    return _mix(maps, _weights(maps, scores))


def consensus_fusion(maps, scores):
    """Fracao ponderada dos mapas que marcam o pixel como alto no proprio limiar de Otsu."""
    votes = {name: _high(m).astype(np.float32) for name, m in maps.items()}
    return _mix(votes, _weights(maps, scores))


def cluster_fusion(maps, scores, link=0.2):
    """Junta mapas com correlacao acima de 1-link num grupo e faz um voto por grupo."""
    d = 1 - _corr_matrix(maps)
    groups = fcluster(linkage(squareform((d + d.T) / 2, checks=False), "average"), link, "distance")
    names = np.array(list(maps))
    means = {g: simple_mean({n: maps[n] for n in names[groups == g]}) for g in np.unique(groups)}
    best = {g: max(scores[n] for n in names[groups == g]) for g in means}
    s = np.array(list(best.values()))
    w = np.exp((s - s.max()) / 0.1)
    return _mix(means, w / w.sum())


def pca_fusion(maps):
    """Primeira componente dos mapas ja descontados pela redundancia."""
    x = _stack(maps).T * independence(maps)
    pca = PCA(1, svd_solver="full").fit(x)
    sign = 1.0 if pca.components_[0].sum() >= 0 else -1.0
    return _unit(sign * pca.transform(x)[:, 0]).reshape(next(iter(maps.values())).shape)


def _candidate_score(c, lab, x, r_weights, quality):
    """Score de um mapa fundido: sem stability, porque ele nao tem algoritmo para refazer."""
    f = _single_features(c, lab)
    corr = [max(_corr(c, row), 0.0) for row in x]
    f["agreement"] = float(np.dot(corr, r_weights) / r_weights.sum())
    return quality_scores({"c": f}, quality)["c"]


def nmf_fusion(maps, img, quality=DEFAULT_QUALITY, n_components=3):
    """Quebra os mapas em camadas nao negativas e fica com a de melhor score, sem rotulo."""
    x = _stack(maps)
    ind = independence(maps)
    shape = next(iter(maps.values())).shape
    model = NMF(min(n_components, len(maps)), init="nndsvda", max_iter=400, random_state=0)
    layers = model.fit_transform(x.T * ind[None])
    lab = rgb2lab(img)
    options = [_unit(layer).reshape(shape) for layer in layers.T]
    return max(options, key=lambda c: _candidate_score(c, lab, x, ind, quality))


def search_fusion(maps, img, quality=DEFAULT_QUALITY, seed=0, maxiter=10, popsize=4):
    """Evolucao diferencial nos pesos, sem gradiente. Maximiza o proprio score: ele e so um proxy."""
    x = _stack(maps)
    ind = independence(maps)
    lab = rgb2lab(img)

    def mixed(w):
        return _mix(maps, w / max(w.sum(), 1e-8))

    found = differential_evolution(
        lambda w: -_candidate_score(mixed(w), lab, x, ind, quality),
        [(0.0, 1.0)] * len(maps), maxiter=maxiter, popsize=popsize, seed=seed, polish=False, tol=0.0,
    )
    return mixed(found.x)


def _border_link(colour, edges, labels, sigma=10.0, floor=3.0):
    """Quanto cada superpixel se liga a borda da imagem (Zhu et al., CVPR 2014). Hipotese, nao verdade."""
    d = np.linalg.norm(colour[edges[:, 0]] - colour[edges[:, 1]], axis=1)
    d = np.where(d < floor, 1e-6, d)
    n = len(colour)
    geo = dijkstra(sparse.csr_matrix((d, (edges[:, 0], edges[:, 1])), shape=(n, n)), directed=False)
    near = np.exp(-(geo**2) / (2 * sigma**2))
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    link = near[:, border].sum(1) / np.sqrt(near.sum(1))
    return 1 - np.exp(-(link**2) / 2)


def regional_fusion(maps, img, scores, n_segments=200, n_groups=6, background=0.5, coherence=1.0, temperature=0.1):
    """Cada grupo de superpixels segue seus proprios pesos.

    Agrupa superpixels SLIC por cor, textura, contraste, saliencia e posicao. No grupo, um mapa
    ganha peso se respeita os superpixels (pouca variancia dentro deles) e se concorda com os
    demais. Depois reduz o que se liga a borda (background) e suaviza no grafo de vizinhos
    (coherence). Devolve (mapa, detalhes).
    """
    lab = rgb2lab(img)
    labels = slic(img, n_segments=n_segments, compactness=10, start_label=0, channel_axis=-1)
    labels = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
    area = np.bincount(labels.ravel()).astype(np.float64)
    n = len(area)

    def per_region(x):
        return np.bincount(labels.ravel(), x.ravel()) / area

    colour = np.stack([per_region(lab[..., c]) for c in range(3)], 1)
    light = lab[..., 0]
    variance = ndimage.uniform_filter(light**2, 5) - ndimage.uniform_filter(light, 5) ** 2
    texture = np.stack([per_region(np.sqrt(variance.clip(0))), per_region(ndimage.gaussian_gradient_magnitude(light, 1))], 1)
    contrast = cdist(colour, colour) @ (area / area.sum())
    saliency = np.stack([per_region(m) for m in maps.values()], 1)
    within = np.stack([per_region(m.astype(np.float64) ** 2) for m in maps.values()], 1) - saliency**2
    yy, xx = np.mgrid[: labels.shape[0], : labels.shape[1]]
    place = np.stack([per_region(yy), per_region(xx)], 1)

    z = np.column_stack([colour, texture, contrast, saliency, place])
    z = (z - z.mean(0)) / (z.std(0) + 1e-8)
    groups = KMeans(min(n_groups, n), n_init=3, random_state=0).fit_predict(z)

    base = _weights(maps, scores, temperature)
    weights = np.tile(base, (groups.max() + 1, 1))
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() < 3:
            continue
        a = area[idx] / area[idx].sum()
        s = saliency[idx]
        between = a @ (s - a @ s) ** 2
        held = between / (between + a @ within[idx] + 1e-8)
        local = np.empty(len(base))
        for i in range(len(base)):
            rest = (s @ base - base[i] * s[:, i]) / max(1 - base[i], 1e-8)
            local[i] = (held[i] + max(_corr(s[:, i], rest), 0.0)) / 2
        w = base * np.exp((local - local.max()) / temperature)
        weights[g] = w / w.sum()

    a, b = np.r_[labels[:, :-1].ravel(), labels[:-1].ravel()], np.r_[labels[:, 1:].ravel(), labels[1:].ravel()]
    edges = np.unique(np.sort(np.c_[a, b][a != b], axis=1), axis=0)
    fused = (weights[groups] * saliency).sum(1) * (1 - background * _border_link(colour, edges, labels))

    close = np.exp(-(((colour[edges[:, 0]] - colour[edges[:, 1]]) ** 2).sum(1)) / (2 * 10.0**2))
    adjacency = sparse.coo_matrix((close, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency = adjacency + adjacency.T
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    fused = spsolve((sparse.identity(n) + coherence * laplacian).tocsc(), fused)
    return _unit(fused)[labels], {"regions": labels, "groups": groups, "weights": weights}


def fuse(maps, img, remake=None, quality=DEFAULT_QUALITY, k=None, n_segments=200, threads=2):
    """Roda A e B. Com k, as fusoes usam so os Top-K; as baselines usam sempre todos os mapas."""
    if min(k or len(maps), len(maps)) < 2:
        raise ValueError("fusao precisa de ao menos 2 mapas")
    with threadpool_limits(threads):
        feats = map_features(maps, img, remake)
        scores = quality_scores(feats, quality)
        chosen = top_k(scores, k, maps) if k else list(maps)
        use = {name: maps[name] for name in chosen}
        regional, detail = regional_fusion(use, img, scores, n_segments)
        fused = {
            "best_single": best_single(maps, scores),
            "simple_mean": simple_mean(maps),
            "weighted_mean": weighted_mean(use, scores),
            "consensus": consensus_fusion(use, scores),
            "cluster": cluster_fusion(use, scores),
            "regional": regional,
            "pca": pca_fusion(use),
            "nmf": nmf_fusion(use, img, quality),
            "search": search_fusion(use, img, quality),
        }
    return {"features": feats, "scores": scores, "chosen": chosen, "fused": fused, "regional": detail}


def binarize(m, fixed=(0.25, 0.5, 0.75)):
    """Mascaras por varios limiares: fixos, Otsu, triangulo e 2x a media (Achanta et al., 2009)."""
    out = {f"t{t:.2f}": m > t for t in fixed}
    if np.ptp(m) > 1e-6:
        out["otsu"] = m > threshold_otsu(m)
        out["triangle"] = m > threshold_triangle(m)
    out["2mean"] = m > min(2 * m.mean(), 1.0)
    return out


def evaluate(m, gt, fixed=tuple(np.round(np.arange(0.05, 1, 0.05), 2))):
    """So medida, depois da fusao. Limiares fixos dao os pontos da curva precisao-revocacao."""
    gt = gt.astype(bool)
    rows = {}
    for name, b in binarize(m, fixed).items():
        tp, fp, fn = (b & gt).sum(), (b & ~gt).sum(), (~b & gt).sum()
        rows[name] = {
            "iou": tp / max(tp + fp + fn, 1),
            "dice": 2 * tp / max(2 * tp + fp + fn, 1),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
    return {"thresholds": rows, "ap": float(average_precision_score(gt.ravel(), m.ravel()))}


def _ellipse(side, cy, cx, ry, rx):
    yy, xx = np.mgrid[:side, :side]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1


def _blur(mask, rng, sigma=3.0, noise=0.05):
    return np.clip(ndimage.gaussian_filter(mask.astype(np.float32), sigma) + rng.normal(0, noise, mask.shape), 0, 1).astype(np.float32)


def _paint(side, rng, gt, distractor=None):
    img = np.empty((side, side, 3), np.float32)
    img[:] = (120, 130, 110)
    img[gt] = (190, 90, 60)
    if distractor is not None:
        img[distractor] = (90, 150, 170)
    return np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


def _scene(seed, side=160):
    """Um mapa bom, varios ruins e 4 copias do mesmo mapa errado."""
    rng = np.random.default_rng(seed)
    cy, cx = rng.integers(60, 100, 2)
    gt = _ellipse(side, cy, cx, 28, 36)
    wrong = _ellipse(side, (cy + 80) % side, (cx + 80) % side, 18, 18)
    half = gt & (np.arange(side)[None] < cx)
    maps = {
        "good": _blur(gt, rng),
        "half": _blur(half, rng),
        "shifted": _blur(np.roll(gt, 12, 1), rng),
        "noise": rng.random((side, side), dtype=np.float32),
        "inverse": 1 - _blur(gt, rng),
        "dot": _blur(_ellipse(side, cy, cx, 3, 3), rng, 1.0, 0.0),
    }
    maps.update({f"copy{i}": _blur(wrong, rng, noise=0.01) for i in range(4)})
    return maps, _paint(side, rng, gt, wrong), gt


def _split_scene(seed, side=160):
    """Cada algoritmo so presta numa metade da imagem: nenhum mapa sozinho acerta tudo."""
    rng = np.random.default_rng(seed)
    gt = _ellipse(side, 50, 40, 22, 26) | _ellipse(side, 110, 120, 22, 26)
    left = np.arange(side)[None] < side // 2
    good = _blur(gt, rng)
    junk = rng.random((side, side), dtype=np.float32)
    maps = {"left": np.where(left, good, junk), "right": np.where(left, junk, good)}
    img = _paint(side, rng, gt)
    img[:, side // 2 :] = np.clip(img[:, side // 2 :].astype(np.int16) + (30, -20, -10), 0, 255)
    return maps, img, gt


def _mean_over(scene, seeds):
    iou, ap, seconds = {}, {}, 0.0
    for seed in seeds:
        maps, img, gt = scene(seed)
        start = time.time()
        out = fuse(maps, img)
        seconds += time.time() - start
        for name, m in out["fused"].items():
            e = evaluate(m, gt)
            iou.setdefault(name, []).append(e["thresholds"]["otsu"]["iou"] if "otsu" in e["thresholds"] else 0.0)
            ap.setdefault(name, []).append(e["ap"])
    print(f"{scene.__name__}: {seconds / len(seeds):.1f}s por imagem")
    for name in iou:
        print(f"  {name:14s} IoU {np.mean(iou[name]):.3f}  AP {np.mean(ap[name]):.3f}")
    return {name: np.mean(v) for name, v in iou.items()}


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    for f in (map_features, quality_scores, top_k, weighted_mean, consensus_fusion, regional_fusion, fuse):
        assert not {"gt", "mask", "label"} & set(inspect.signature(f).parameters)

    maps, img, gt = _scene(0)
    maps["flat"] = np.full(gt.shape, 0.5, np.float32)
    feats = map_features(maps, img)
    scores = quality_scores(feats)
    print("scores:", {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda kv: -kv[1])})
    print("independence:", dict(zip(maps, independence(maps).round(2))))
    print("top3:", top_k(scores, 3), "| sem redundancia:", top_k(scores, 3, maps))
    assert scores["flat"] == min(scores.values())
    assert scores["good"] > scores["noise"]
    assert sum(n.startswith("copy") for n in top_k(scores, 3, maps)) <= 1
    assert quality_scores(feats, lambda f: f["separation"])["flat"] == 0.0

    rng = np.random.default_rng(1)
    method = lambda x: _unit(np.linalg.norm(ndimage.gaussian_filter(x.astype(np.float32), (2, 2, 0)) - x.mean((0, 1)), axis=2))
    pair = {"steady": method(img), "shaky": rng.random(gt.shape, dtype=np.float32)}
    stable = map_features(pair, img, lambda name, x: method(x) if name == "steady" else rng.random(x.shape[:2], dtype=np.float32))
    print("stability:", {k: round(v["stability"], 3) for k, v in stable.items()})
    assert stable["steady"]["stability"] > 0.8 > 0.1 > stable["shaky"]["stability"]

    assert set(binarize(maps["good"])) >= {"t0.50", "otsu", "triangle", "2mean"}

    seeds = range(6)
    table = _mean_over(_scene, seeds)
    assert table["weighted_mean"] > table["simple_mean"]
    table = _mean_over(_split_scene, seeds)
    assert table["regional"] > max(table["simple_mean"], table["best_single"])
    maps, img, gt = _split_scene(0)
    _, detail = regional_fusion(maps, img, quality_scores(map_features(maps, img)))
    print("pesos por grupo (left, right):", detail["weights"].round(2).tolist())
    assert np.ptp(detail["weights"][:, 0]) > 0.5
    print("ok")
