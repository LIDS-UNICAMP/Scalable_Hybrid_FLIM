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

"""
Mapas de saliencia classicos, sem backprop, sem rotulo e sem pre-treino.

Cada metodo e f(img_rgb_uint8, *, parametros) -> float32 HxW em [0, 1], na resolucao
original. Uma imagem sem contraste devolve zeros. Para mudar parametros:
functools.partial(MAPS_B["bms"], step=16). Saliente aqui nao quer dizer objeto.

`approx` diz, por metodo, o que nao e o algoritmo publicado e por que (None = sem desvio
conhecido). Check: python -m saliency.maps_b
"""

import heapq
import time

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb, slic


def _norm(m):
    m = np.asarray(m, np.float32)
    lo, hi = float(m.min()), float(m.max())
    return (m - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(m)


def _shrink(img, max_side):
    scale = max_side / max(img.shape[:2]) if max_side else 1.0
    if scale >= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _back(m, img):
    h, w = img.shape[:2]
    return _norm(m if m.shape == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR))


def _adjacent_pairs(labels):
    pairs = np.concatenate([
        np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], 1),
        np.stack([labels[:-1].ravel(), labels[1:].ravel()], 1),
    ])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(np.sort(pairs, 1), axis=0)


def _surrounded(mask):
    """Componentes (8-conexas) de `mask` que nao tocam a borda da imagem."""
    _, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return mask & ~np.isin(lab, border)


def bms(img, *, step=8, dilate_1=3, dilate_2=11, blur_sigma=20.0, max_side=400):
    """Boolean Map Saliency (Zhang & Sclaroff 2013), constantes do codigo jimmie33/BMS."""
    small = _shrink(img, max_side)
    lab = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2Lab), 3)
    kernel = np.ones((3, 3), np.uint8)
    acc = np.zeros(small.shape[:2], np.float32)
    for c in range(3):
        ch = lab[..., c]
        for t in np.arange(float(ch.min()), float(ch.max()), step):
            bm = ch > t
            for m in (_surrounded(bm), _surrounded(~bm)):
                a = cv2.dilate(m.astype(np.uint8), kernel, iterations=(dilate_1 - 1) // 2)
                a = a.astype(np.float32)
                n = np.linalg.norm(a)
                acc += a / n if n > 0 else a
    acc = cv2.dilate(_norm(acc), kernel, iterations=(dilate_2 - 1) // 2)
    ksize = min(int(blur_sigma) * 4 + 1, 51)
    return _back(cv2.GaussianBlur(acc, (ksize, ksize), blur_sigma), img)


def mser(img, *, delta=5, min_area_frac=1e-4, max_area_frac=0.5, blur_sigma=2.0):
    """
    MSER (Matas 2002) em cinza, nas duas polaridades.

    Mapa: cada regiao da 1 voto a cada pixel seu; regioes aninhadas somam, entao o valor
    conta quantas regioes estaveis cobrem o pixel. O voto e suavizado por uma gaussiana.
    max_area_frac tira as regioes de fundo; acima disso uma regiao nao vota.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    area = gray.size
    det = cv2.MSER_create(delta=delta, min_area=max(int(min_area_frac * area), 1),
                          max_area=int(max_area_frac * area))
    votes = np.zeros(gray.shape, np.float32)
    for pts in det.detectRegions(gray)[0]:
        votes[pts[:, 1], pts[:, 0]] += 1
    return _norm(cv2.GaussianBlur(votes, (0, 0), blur_sigma) if blur_sigma else votes)


def edge_density(img, *, low=None, high=None, window_frac=0.1):
    """Canny + fracao de pixels de borda numa janela. low/high None: Otsu e Otsu/2."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if high is None:
        high = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    low = high / 2 if low is None else low
    edges = cv2.Canny(gray, low, high).astype(np.float32) / 255
    win = max(int(window_frac * min(gray.shape)), 3)
    return _norm(cv2.boxFilter(edges, -1, (win, win), borderType=cv2.BORDER_REFLECT))


def _region_hists(feat, labels, n):
    """Histogramas de cor (25 bins) e textura (8 orientacoes x 10 bins) por regiao, L1."""
    def hist(q, bins):
        return np.bincount(labels.ravel() * bins + q.ravel(), minlength=n * bins).reshape(n, bins)

    color, texture = [], []
    for c in range(3):
        ch = feat[..., c]
        color.append(hist(ch.astype(np.int64) * 25 // 256, 25))
        g = cv2.GaussianBlur(ch.astype(np.float32), (0, 0), 1.0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        for theta in np.arange(8) * np.pi / 4:
            d = np.cos(theta) * gx + np.sin(theta) * gy
            q = np.clip((_norm(d) * 10).astype(np.int64), 0, 9)
            texture.append(hist(q, 10))
    color, texture = np.hstack(color).astype(np.float32), np.hstack(texture).astype(np.float32)
    return (color / color.sum(1, keepdims=True), texture / texture.sum(1, keepdims=True))


def _hierarchy(labels, color, texture, use_color):
    """Agrupamento hierarquico guloso; devolve as caixas, da regiao inicial 0 ate a raiz."""
    n = int(labels.max()) + 1
    h, w = labels.shape
    im_size = h * w
    color, texture = list(color), list(texture)
    size = np.bincount(labels.ravel(), minlength=n).tolist()
    ys, xs = np.indices(labels.shape)
    flat = labels.ravel()
    box = np.stack([np.full(n, h), np.full(n, w), np.zeros(n, int), np.zeros(n, int)], 1)
    np.minimum.at(box[:, 0], flat, ys.ravel())
    np.minimum.at(box[:, 1], flat, xs.ravel())
    np.maximum.at(box[:, 2], flat, ys.ravel() + 1)
    np.maximum.at(box[:, 3], flat, xs.ravel() + 1)
    box = [tuple(b) for b in box.tolist()]
    neigh = [set() for _ in range(n)]
    for i, j in _adjacent_pairs(labels).tolist():
        neigh[i].add(j)
        neigh[j].add(i)

    def union(i, j):
        a, b = box[i], box[j]
        return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])

    def sim(i, j):
        u = union(i, j)
        s = np.minimum(texture[i], texture[j]).sum() + 1 - (size[i] + size[j]) / im_size
        s += 1 - ((u[2] - u[0]) * (u[3] - u[1]) - size[i] - size[j]) / im_size
        return s + np.minimum(color[i], color[j]).sum() if use_color else s

    heap = [(-sim(i, j), i, j) for i in range(n) for j in neigh[i] if i < j]
    heapq.heapify(heap)
    alive = set(range(n))
    while heap:
        _, i, j = heapq.heappop(heap)
        if i not in alive or j not in alive:
            continue
        t = len(size)
        wi, wj = size[i], size[j]
        color.append((wi * color[i] + wj * color[j]) / (wi + wj))
        texture.append((wi * texture[i] + wj * texture[j]) / (wi + wj))
        box.append(union(i, j))
        size.append(wi + wj)
        new = (neigh[i] | neigh[j]) - {i, j}
        neigh.append(new)
        alive -= {i, j}
        alive.add(t)
        for k in new:
            neigh[k] -= {i, j}
            neigh[k].add(t)
            heapq.heappush(heap, (-sim(t, k), k, t))
    return box


def selective_search(img, *, color_spaces=("hsv", "lab"), ks=(50, 100), sigma=0.8,
                     max_box_frac=0.5, max_side=256):
    """
    Selective Search (Uijlings 2013) sobre felzenszwalb do skimage, modo "fast" do artigo:
    cores x k x {C+T+S+F, T+S+F}.

    Mapa: cada caixa soma 1/posicao, com posicao 1 na raiz da hierarquia. Caixas acima de
    max_box_frac da imagem nao votam.
    """
    small = _shrink(img, max_side)
    h, w = small.shape[:2]
    acc = np.zeros((h, w), np.float32)
    code = {"hsv": cv2.COLOR_RGB2HSV_FULL, "lab": cv2.COLOR_RGB2Lab}
    for space in color_spaces:
        feat = cv2.cvtColor(small, code[space])
        for k in ks:
            labels = felzenszwalb(feat, scale=k, sigma=sigma, min_size=k)
            color, texture = _region_hists(feat, labels, int(labels.max()) + 1)
            for use_color in (True, False):
                boxes = _hierarchy(labels, color, texture, use_color)
                for pos, (y0, x0, y1, x1) in enumerate(reversed(boxes), 1):
                    if (y1 - y0) * (x1 - x0) <= max_box_frac * h * w:
                        acc[y0:y1, x0:x1] += 1.0 / pos
    return _back(acc, img)


def gmr(img, *, n_segments=200, compactness=10.0, alpha=0.99, inv_sigma2=10.0):
    """SLIC + Graph-Based Manifold Ranking (Yang 2013): fundo pelas 4 bordas, depois frente."""
    labels = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(img.shape[:2])
    n = int(labels.max()) + 1
    if n < 3:
        return np.zeros(img.shape[:2], np.float32)

    lab = rgb2lab(img).reshape(-1, 3)
    count = np.bincount(labels.ravel(), minlength=n)
    mean = np.stack([np.bincount(labels.ravel(), lab[:, c], n) / count for c in range(3)], 1)

    adj = np.zeros((n, n), bool)
    i, j = _adjacent_pairs(labels).T
    adj[i, j] = adj[j, i] = True
    graph = adj | (adj.astype(np.int32) @ adj.astype(np.int32) > 0)  # vizinhos e vizinhos deles
    sides = [labels[0], labels[-1], labels[:, 0], labels[:, -1]]
    ring = np.unique(np.concatenate(sides))
    graph[np.ix_(ring, ring)] = True  # bordas formam um laco fechado
    np.fill_diagonal(graph, False)

    dist = cdist(mean, mean)
    dmin, dmax = dist[graph].min(), dist[graph].max()
    if dmax - dmin < 1e-9:
        return np.zeros(img.shape[:2], np.float32)
    weights = graph * np.exp(-inv_sigma2 * (dist - dmin) / (dmax - dmin))
    rank = np.linalg.inv(np.diag(weights.sum(1)) - alpha * weights)
    np.fill_diagonal(rank, 0)

    background = np.ones(n, np.float32)
    for side in sides:
        y = np.zeros(n)
        y[np.unique(side)] = 1
        background *= 1 - _norm(rank @ y)
    return _norm(_norm(rank @ (background >= background.mean()))[labels])


MAPS_B = {
    "bms": bms,
    "mser": mser,
    "edge_density": edge_density,
    "selective_search": selective_search,
    "gmr": gmr,
}

approx = {
    "bms": "CIE Lab sem o branqueamento de cor, que no codigo jimmie33/BMS e opcional. Sem "
           "abertura morfologica, como nesse codigo. Sem o deslocamento aleatorio da borda "
           "dele, para ser deterministico.",
    "mser": "MSER detecta regioes e nao define mapa: a soma de votos, a gaussiana e o corte "
            "por area sao convencao nossa. O OpenCV nao expoe a estabilidade de cada regiao, "
            "entao todas pesam igual.",
    "edge_density": None,
    "selective_search": "cv2.ximgproc ausente (opencv-python-headless, sem contrib): "
                        "implementacao propria sobre skimage felzenszwalb. Textura com "
                        "derivadas Sobel de gaussiana, sem os bins exatos do artigo. Trocamos "
                        "o peso aleatorio RND x posicao por 1/posicao, para ser deterministico. "
                        "O metodo gera caixas: o mapa de votos e convencao nossa. Roda em "
                        "max_side=256 por custo do agrupamento em Python.",
    "gmr": None,
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:240, :320]
    disc = (yy - 90) ** 2 + (xx - 210) ** 2 <= 45 ** 2
    synth = np.full((240, 320, 3), (60, 70, 80), np.float32)
    synth[disc] = (220, 60, 40)
    synth = np.clip(synth + rng.normal(0, 8, synth.shape), 0, 255).astype(np.uint8)
    flat = np.full((64, 96, 3), 128, np.uint8)
    for name, f in MAPS_B.items():
        t = time.perf_counter()
        s = f(synth)
        t = time.perf_counter() - t
        assert s.shape == disc.shape and s.dtype == np.float32 and 0 <= s.min() <= s.max() <= 1
        z = f(flat)
        assert z.shape == flat.shape[:2] and z.dtype == np.float32 and not z.any(), name
        print(f"{name:17s} dentro={s[disc].mean():.3f} fora={s[~disc].mean():.3f} {t:.2f}s")
