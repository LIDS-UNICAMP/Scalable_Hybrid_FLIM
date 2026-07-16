"""embedding_analysis.py — quantitative embedding-space metrics comparing
FLIM-initialized vs trunc_normal-initialized distillation students.

For each (dataset ds in {eggs, larvae}, scale in {126k one-layer, 400k two-layer}):
  * Extract 48-dim student embeddings (encoder -> global avg pool) for the test
    split, for both the trained flim and trunc students, plus an UNTRAINED
    student carrying the same init (for representational-drift CKA).
  * Compute: linear CKA(flim, trunc), effective rank / 90-99% PCA comps,
    Fisher discriminant ratio, silhouette, drift CKA(init, trained),
    mean abs off-diagonal feature correlation.
  * Save JSON of all metrics and t-SNE / PCA scatter PNGs.

Reuses the checkpoint loader (_load_student_and_proj) and data path from
src/evaluate/svm_distill_with_projection.py. Never instantiates the I-JEPA
teacher (only student.encoder is needed).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

_ROOT = "/dados/home/moliveira/scalable_FLIM_self_supervised"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reuse repo infrastructure ────────────────────────────────────────────────
from src.evaluate.svm_distill_with_projection import _load_student_and_proj
from src.evaluate.svm_distillation import (
    IMAGE_SIZE,
    _DATASET_PARASITE_NAME,
)
from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    load_FLIM_encoder,
    load_FLIM_encoder_from_arch_dict,
    PROTOZOAN_FLIM_ARCH,
    init_weights_trunc_normal,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUT_DIR = os.path.join(_ROOT, "analysis_flim_distill")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ART = os.path.join(_ROOT, "artifacts", "distillation")


def _ckpt(run_dir: str) -> str:
    """Return the best*.ckpt under <run_dir>/checkpoints (prefer best-* over last)."""
    cdir = os.path.join(ART, run_dir, "checkpoints")
    hits = []
    for root, _dirs, files in os.walk(cdir):
        for f in files:
            if f.endswith(".ckpt"):
                hits.append(os.path.join(root, f))
    best = [h for h in hits if "best" in os.path.basename(h) or "best" in h]
    chosen = best if best else hits
    if not chosen:
        raise FileNotFoundError(f"no ckpt under {cdir}")
    return sorted(chosen)[0]


# (ds, scale) -> {init: run_dir}
RUNS = {
    ("eggs", "126k"): {
        "trunc": "distillation_eggs_split1_pct100_1x1_BN2d_1280_one_layer",
        "flim":  "distillation_eggs_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
    },
    ("eggs", "400k"): {
        "trunc": "distillation_eggs_split1_pct100_2l_1x1_BN2d_256_1280",
        "flim":  "distillation_eggs_split1_pct100_2l_1x1_init_flim_256_1280",
    },
    ("larvae", "126k"): {
        "trunc": "distillation_larvae_split1_pct100_1x1_BN2d_1280_one_layer",
        "flim":  "distillation_larvae_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
    },
    ("larvae", "400k"): {
        "trunc": "distillation_larvae_split1_pct100_2l_1x1_BN2d_256_1280",
        "flim":  "distillation_larvae_split1_pct100_2l_1x1_init_flim_256_1280",
    },
}


# ── Untrained student (same init as the run, no teacher) ─────────────────────

def build_untrained_student(ckpt_path: str, dataset: str) -> LeJEPAFLIMModel:
    """Replicates the module init logic (onelayer/twolayer) to obtain a fresh
    UNTRAINED student carrying the same init weights (flim or trunc_normal).
    Never instantiates the I-JEPA teacher."""
    ck = torch.load(ckpt_path, map_location="cpu")
    hp = ck.get("hyper_parameters", {})
    arch_json = hp["arch_json"]
    in_channels = hp.get("in_channels", 3)
    proj_dim = hp.get("proj_dim", 256)
    proj_hidden = hp.get("proj_hidden", 2048)
    encoder_init = hp.get("encoder_init", "trunc_normal")
    flim_weights_path = hp.get("flim_weights_path", None)

    if encoder_init == "flim" and dataset == "protozoan":
        arch = PROTOZOAN_FLIM_ARCH
    else:
        arch = parse_architecture(arch_json)

    if encoder_init == "flim":
        channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
        arch = override_arch_channels(arch, channels)
    else:
        channels = get_channels_from_arch(arch, in_channels)

    student = LeJEPAFLIMModel(
        arch=arch, in_channels=in_channels,
        proj_dim=proj_dim, proj_hidden=proj_hidden,
    )

    if encoder_init == "trunc_normal":
        # Reproduce the trained run's stochastic init deterministically.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        init_weights_trunc_normal(student.encoder)
    elif encoder_init == "flim":
        if dataset == "protozoan":
            load_FLIM_encoder_from_arch_dict(student, arch, flim_weights_path, channels)
        else:
            load_FLIM_encoder(student, arch_json, flim_weights_path, channels)
    else:
        raise ValueError(f"unexpected encoder_init={encoder_init}")
    return student.to(DEVICE).eval()


# ── Feature extraction: 48-dim encoder embedding ─────────────────────────────

@torch.no_grad()
def encode_48(student: LeJEPAFLIMModel, loader: DataLoader) -> np.ndarray:
    student.eval().to(DEVICE)
    feats = []
    for x, _y in loader:
        emb = student.encode(x.to(DEVICE))  # [B, 48]
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def collect_labels(loader: DataLoader) -> np.ndarray:
    ys = []
    for _x, y in loader:
        ys.extend(y.tolist() if isinstance(y, torch.Tensor) else list(y))
    return np.array(ys, dtype=np.int64)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via centered linear-kernel HSIC. X,Y: [N, d]. Returns [0,1]."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    Kxc = _center(Kx)
    Kyc = _center(Ky)
    hsic_xy = np.sum(Kxc * Kyc)
    hsic_xx = np.sum(Kxc * Kxc)
    hsic_yy = np.sum(Kyc * Kyc)
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:
        return float("nan")
    return float(hsic_xy / denom)


def effective_rank_and_pca(Z: np.ndarray) -> dict:
    """Effective rank (participation ratio) from eigenvalues of centered cov,
    plus number of PCA components for 90% / 99% variance."""
    Zc = Z - Z.mean(0, keepdims=True)
    n = Zc.shape[0]
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    s = eig.sum()
    eff_rank = float((s ** 2) / (np.sum(eig ** 2) + 1e-12)) if s > 0 else 0.0
    eig_sorted = np.sort(eig)[::-1]
    cumvar = np.cumsum(eig_sorted) / (s + 1e-12)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n99 = int(np.searchsorted(cumvar, 0.99) + 1)
    return {
        "effective_rank": eff_rank,
        "pca_comps_90": n90,
        "pca_comps_99": n99,
        "ambient_dim": int(Z.shape[1]),
    }


def fisher_ratio(Z: np.ndarray, y: np.ndarray) -> float:
    """trace(between-class scatter) / trace(within-class scatter)."""
    classes = np.unique(y)
    mu = Z.mean(0)
    Sb_tr = 0.0
    Sw_tr = 0.0
    for c in classes:
        Zc = Z[y == c]
        nc = Zc.shape[0]
        muc = Zc.mean(0)
        Sb_tr += nc * np.sum((muc - mu) ** 2)
        Sw_tr += np.sum((Zc - muc) ** 2)
    if Sw_tr <= 0:
        return float("nan")
    return float(Sb_tr / Sw_tr)


def mean_abs_offdiag_corr(Z: np.ndarray) -> float:
    """Mean absolute off-diagonal feature correlation (redundancy)."""
    Zc = Z - Z.mean(0, keepdims=True)
    std = Zc.std(0)
    keep = std > 1e-8
    Zc = Zc[:, keep]
    if Zc.shape[1] < 2:
        return float("nan")
    corr = np.corrcoef(Zc, rowvar=False)
    d = corr.shape[0]
    off = corr[~np.eye(d, dtype=bool)]
    return float(np.mean(np.abs(off)))


def silhouette(Z: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(silhouette_score(Z, y))
    except Exception:
        return float("nan")


# ── Plots ────────────────────────────────────────────────────────────────────

def scatter_pair(Z_flim, Z_trunc, y, ds, scale, method):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, Z, title in [
        (axes[0], Z_flim, f"FLIM init"),
        (axes[1], Z_trunc, f"trunc_normal init"),
    ]:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=14, alpha=0.75)
        ax.set_title(f"{method.upper()} — {title}")
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
    fig.suptitle(f"{ds} {scale} — 48-dim student embedding ({method})", fontsize=13)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("class y")
    out = os.path.join(PLOT_DIR, f"{ds}_{scale}_{method}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def make_tsne(Z):
    n = Z.shape[0]
    perp = float(min(30, max(5, (n - 1) // 3)))
    ts = TSNE(n_components=2, init="pca", perplexity=perp,
              random_state=SEED, learning_rate="auto")
    return ts.fit_transform(Z)


def make_pca(Z):
    return PCA(n_components=2, random_state=SEED).fit_transform(Z)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    transform = _build_test(IMAGE_SIZE)
    results = {}
    plot_paths = []

    for (ds, scale), inits in RUNS.items():
        key = f"{ds}_{scale}"
        print(f"\n{'='*70}\n{key}\n{'='*70}", flush=True)
        parasite = _DATASET_PARASITE_NAME[ds]

        # Test loader (same path as svm_distill_with_projection): integer labels.
        test_ds = DatasetParasite(
            set_name="test", split=1, percentage=100,
            transform=transform, loader="ift_lab", path_dataset=parasite,
        )
        loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
        y = collect_labels(loader)
        print(f"  N test images = {len(y)}  classes={sorted(np.unique(y).tolist())}", flush=True)

        Z = {}          # trained embeddings
        Z_init = {}     # untrained-init embeddings
        for init in ("flim", "trunc"):
            ckpt = _ckpt(inits[init])
            print(f"  [{init}] ckpt = {os.path.relpath(ckpt, _ROOT)}", flush=True)
            student, _proj = _load_student_and_proj(ckpt, DEVICE)
            for p in student.parameters():
                p.requires_grad_(False)
            Z[init] = encode_48(student, loader)
            del student, _proj
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # Untrained student with same init
            untrained = build_untrained_student(ckpt, ds)
            Z_init[init] = encode_48(untrained, loader)
            del untrained
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print(f"       Z_trained {Z[init].shape}  Z_init {Z_init[init].shape}", flush=True)

        # ── Cross-representation CKA (flim vs trunc, trained) ────────────────
        cka_flim_trunc = linear_cka(Z["flim"], Z["trunc"])

        entry = {
            "n_test": int(len(y)),
            "n_classes": int(len(np.unique(y))),
            "cka_flim_vs_trunc_trained": cka_flim_trunc,
        }
        for init in ("flim", "trunc"):
            er = effective_rank_and_pca(Z[init])
            entry[init] = {
                **er,
                "fisher_ratio": fisher_ratio(Z[init], y),
                "silhouette": silhouette(Z[init], y),
                "mean_abs_offdiag_corr": mean_abs_offdiag_corr(Z[init]),
                "drift_cka_init_vs_trained": linear_cka(Z_init[init], Z[init]),
                "init_effective_rank": effective_rank_and_pca(Z_init[init])["effective_rank"],
            }
        results[key] = entry
        print(json.dumps(entry, indent=2), flush=True)

        # ── Plots ────────────────────────────────────────────────────────────
        try:
            p1 = scatter_pair(make_pca(Z["flim"]), make_pca(Z["trunc"]), y, ds, scale, "pca")
            p2 = scatter_pair(make_tsne(Z["flim"]), make_tsne(Z["trunc"]), y, ds, scale, "tsne")
            plot_paths += [p1, p2]
            print(f"  plots: {p1}\n         {p2}", flush=True)
        except Exception as e:
            print(f"  [WARN] plotting failed: {e}", flush=True)

    out_json = os.path.join(OUT_DIR, "embedding_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] wrote {out_json}")
    print(f"[DONE] {len(plot_paths)} plots under {PLOT_DIR}")


if __name__ == "__main__":
    main()
