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
Classic label-free saliency maps, written from the original papers.

Every method is f(img_rgb_uint8, **params) -> float32 HxW in [0, 1], at the input
resolution. Keyword defaults are the published values. MAPS_A maps name -> function and
approx maps name -> why it is not the published algorithm (None = it is). A flat image
returns all zeros. High saliency means "stands out", not "is the object".
"""
from functools import lru_cache, partial

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.color import rgb2lab
from skimage.segmentation import felzenszwalb


def _unit(x):
    """Min-max to float32 [0, 1]; a constant map becomes zeros."""
    x = np.asarray(x, dtype=np.float64)
    span = x.max() - x.min()
    return ((x - x.min()) / span if span > 0 else np.zeros_like(x)).astype(np.float32)


def spectral_residual(img, size=64, avg=3, sigma=8.0):
    """Hou & Zhang 2007, eqs. 3-8: inverse FFT of (log amplitude - local mean) with the phase kept.

    Grayscale, as in the paper. 'Height (or width) equals 64 px' is read as the long side.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.min() == gray.max():
        return np.zeros(gray.shape, np.float32)
    h, w = gray.shape
    f = size / max(h, w)
    small = cv2.resize(gray, (max(round(w * f), 1), max(round(h * f), 1)), interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small.astype(np.float64))
    log_amp = np.log(np.abs(spectrum) + 1e-12)
    residual = log_amp - cv2.blur(log_amp, (avg, avg), borderType=cv2.BORDER_REPLICATE)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), sigma)
    return _unit(cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR))


def frequency_tuned(img, kernel=(1, 4, 6, 4, 1)):
    """Achanta et al. 2009, eq. 8: Lab distance between the image mean and the binomial-blurred image."""
    lab = rgb2lab(img)
    k = np.asarray(kernel, dtype=np.float64)
    k /= k.sum()
    return _unit(np.linalg.norm(cv2.sepFilter2D(lab, -1, k, k) - lab.mean((0, 1)), axis=2))


def _quantize(img, levels, coverage, min_colors, max_colors):
    """Cheng sec. 3.1/3.3: RGB bins, keep the frequent ones, fold the rest into the closest kept bin.

    Returns the HxW palette index, palette Lab colors (n, 3) and their frequencies (n,).
    """
    h, w, _ = img.shape
    q = (img.reshape(-1, 3) / 255.0 * (levels - 1e-4)).astype(np.int64)
    bins, inverse, counts = np.unique(q @ [levels**2, levels, 1], return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    n = np.searchsorted(np.cumsum(counts[order]), coverage * h * w) + 1
    n = min(max(n, min_colors), max_colors, len(bins))
    keep = order[:n]
    cells = np.stack([bins // levels**2, bins // levels % levels, bins % levels], axis=1)
    to_palette = cdist(cells, cells[keep], "sqeuclidean").argmin(1)
    to_palette[keep] = np.arange(n)
    palette = to_palette[inverse]
    freq = np.bincount(palette, minlength=n)
    rgb = np.stack([np.bincount(palette, img.reshape(-1, 3)[:, c], n) for c in range(3)], axis=1)
    lab = rgb2lab((rgb / freq[:, None] / 255.0)[None])[0]
    return palette.reshape(h, w), lab, freq / (h * w)


def _smooth_colors(sal, dist, fraction):
    """Cheng eq. 4: replace each color's saliency by a linear-weight mean over its m nearest colors."""
    n = len(sal)
    if n < 2:
        return sal
    m = min(n, max(2, round(n * fraction)))
    nearest = np.argsort(dist, axis=1)[:, :m]
    d = np.take_along_axis(dist, nearest, axis=1)
    total = d.sum(1, keepdims=True)
    return ((total - d) * sal[nearest]).sum(1) / ((m - 1) * np.maximum(total[:, 0], 1e-12))


def histogram_contrast(img, levels=12, coverage=0.95, smooth=0.25, min_colors=10, max_colors=256):
    """Cheng et al. 2015, eqs. 3-4: contrast of each color to the image color histogram.

    min_colors/max_colors come from the authors' CmSaliencyRC.cpp, not the paper; without
    min_colors the 95% rule erases any object under 5% of the image.
    """
    palette, lab, freq = _quantize(img, levels, coverage, min_colors, max_colors)
    dist = cdist(lab, lab)
    return _unit(_smooth_colors(dist @ freq, dist, smooth)[palette])


def region_contrast(img, sigma_s2=0.4, seg_k=50, seg_sigma=0.5, seg_min_size=200, levels=12, coverage=0.95,
                    min_colors=10, max_colors=256, center_prior=False, refine=False, border_px=15,
                    border_thresh=0.15, refine_smooth=0.1):
    """Cheng et al. 2015, eqs. 5-7: color contrast between segments, weighted by distance and area.

    The defaults are eq. 7 with w_s = 1. center_prior adds w_s(r_k), which assumes centered objects;
    refine adds the border removal and color smoothing of sec. 4.3. The paper gives no segmentation
    values, so seg_* are from the authors' CmSaliencyRC.cpp. D_s is Euclidean as the paper states;
    that code squares it.
    """
    h, w, _ = img.shape
    palette, lab, _ = _quantize(img, levels, coverage, min_colors, max_colors)
    n = len(lab)
    # skimage divides scale by 255, so this gives k = seg_k on raw Lab values, as in the authors' code.
    seg = felzenszwalb(rgb2lab(img), scale=seg_k * 255.0, sigma=seg_sigma, min_size=seg_min_size)
    regions = seg.max() + 1
    hist = np.bincount((seg * n + palette).ravel(), minlength=regions * n).reshape(regions, n).astype(np.float64)
    area = hist.sum(1)
    hist /= area[:, None]
    color_dist = cdist(lab, lab)
    region_dist = hist @ color_dist @ hist.T
    np.fill_diagonal(region_dist, 0.0)

    y, x = np.mgrid[:h, :w]
    y, x = y / h, x / w
    centroids = np.stack([np.bincount(seg.ravel(), c.ravel()) / area for c in (y, x)], axis=1)
    sal = (np.exp(-cdist(centroids, centroids) / sigma_s2) * region_dist) @ area
    if center_prior:
        d_center = np.bincount(seg.ravel(), np.hypot(y - 0.5, x - 0.5).ravel()) / area
        sal = sal * np.exp(-9.0 * d_center**2)
    if refine:
        ring = np.ones((h, w), bool)
        ring[border_px:h - border_px, border_px:w - border_px] = False
        is_border = np.bincount(seg[ring], minlength=regions) / area > border_thresh
        sal = np.where(is_border, 0.0, _unit(sal).astype(np.float64))
        color_sal = np.bincount(palette.ravel(), sal[seg].ravel(), n) / np.bincount(palette.ravel(), minlength=n)
        pixels = _smooth_colors(color_sal, color_dist, refine_smooth)[palette]
        pixels[is_border[seg]] = 0.0
        sal = np.bincount(seg.ravel(), pixels.ravel()) / area
    return _unit(sal[seg])


def _gbvs_features(gray, scales, angles, contrast_frac, gabor_sigma, gabor_lambda, gabor_gamma, size):
    """Sec. 3.1: per scale, 4 Gabor orientation maps, a local-variance map and luminance."""
    ksize = (2 * int(3 * gabor_sigma) + 1,) * 2
    for s in scales:
        g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        maps = []
        for angle in angles:
            theta = np.deg2rad(angle)
            even, odd = (cv2.getGaborKernel(ksize, gabor_sigma, theta, gabor_lambda, gabor_gamma, psi, cv2.CV_64F)
                         for psi in (0.0, np.pi / 2))
            maps.append(np.hypot(cv2.filter2D(g, -1, even - even.mean()), cv2.filter2D(g, -1, odd - odd.mean())))
        win = (max(round(contrast_frac * g.shape[1]), 3),) * 2
        maps.append(np.maximum(cv2.blur(g * g, win) - cv2.blur(g, win) ** 2, 0.0))
        maps.append(g)
        yield from (cv2.resize(m, size, interpolation=cv2.INTER_AREA) for m in maps)


@lru_cache(maxsize=4)
def _falloff(mh, mw, sigma):
    yy, xx = np.mgrid[:mh, :mw]
    nodes = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return np.exp(-cdist(nodes, nodes, "sqeuclidean") / (2 * sigma**2))


def gbvs(img, scales=(0.5, 0.25), angles=(0, 45, 90, 135), map_long=37, sigma_frac=5 / 37, contrast_frac=80 / 600,
         norm_iters=1, log_eps=1e-3, gabor_sigma=2.0, gabor_lambda=7.0, gabor_gamma=0.5):
    """Harel et al. 2006: per feature map, activation chain (w1), normalization chain (w2), then sum.

    sigma is the paper's graph (ii), 5 on a 25x37 map. Color-blind, because the paper's features
    are grayscale, and the graph itself favors the image center.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape
    if gray.min() == gray.max():
        return np.zeros((h, w), np.float32)
    mw, mh = max(round(w * map_long / max(h, w)), 2), max(round(h * map_long / max(h, w)), 2)
    falloff = _falloff(mh, mw, sigma_frac * mw)

    total = np.zeros(mh * mw)
    for m in _gbvs_features(gray, scales, angles, contrast_frac, gabor_sigma, gabor_lambda, gabor_gamma, (mw, mh)):
        if m.max() - m.min() < 1e-12:
            continue  # a flat map has no edges in w1; w2 would only return the graph's own center bump
        log_m = np.log(m.ravel() / m.max() + log_eps)
        # Both chains are reversible, so the stationary vector has a closed form: same vector as the
        # paper's power iteration (and as a direct solve, to 1e-12), and it holds on periodic chains too.
        # in blocks of rows that fit in cache: the same row sums, a fraction of the memory traffic
        act = np.concatenate([(np.abs(log_m[i:i + 128, None] - log_m) * falloff[i:i + 128]).sum(1)
                              for i in range(0, len(log_m), 128)])
        act /= act.sum()
        for _ in range(norm_iters):
            act = act * (falloff @ act)
            act /= act.sum()
        total += act
    return _unit(cv2.resize(total.reshape(mh, mw), (w, h), interpolation=cv2.INTER_LINEAR))


MAPS_A = {
    "spectral_residual": spectral_residual,
    "frequency_tuned": frequency_tuned,
    "histogram_contrast": histogram_contrast,
    "region_contrast": region_contrast,
    "region_contrast_tpami": partial(region_contrast, center_prior=True, refine=True),
    "gbvs": gbvs,
}

# None = follows the paper. Choices the paper leaves open are listed in the function's docstring.
approx = {
    "spectral_residual": None,
    "frequency_tuned": None,
    "histogram_contrast": None,
    "region_contrast": None,
    "region_contrast_tpami": "The paper never publishes its border threshold (only that it misses 2% on its own "
                             "dataset), so border_thresh=0.15 is our guess, not tuned on any label.",
    "gbvs": "The paper gives no Gabor parameters and no rule for zeros under the log, so gabor_* and log_eps are ours; "
            "its 80x80 window (600x400 images) is scaled with image width.",
}
