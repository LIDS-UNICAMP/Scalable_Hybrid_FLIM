# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""ruler_mismatch.py — Resolve the eval-representation ("ruler") mismatch.

The published comparison scored FLIM distilled students on the 48-dim ENCODER
(svm_distillation_conv → SVM_Distillation_Conv, embed_dim=48) but scored the
trunc/random 400k students on the 1280-dim PROJECTION HEAD
(svm_distill_with_projection → SVM_Distill_Proj1280, embed_dim=1280). Those are
different representations, so "random beats FLIM at 400k" may be an artifact.

This script re-scores ALL FOUR (init × scale) distilled students on BOTH rulers,
for ds in {eggs, larvae}, split=1, pct=100, using each ruler's native SVM
protocol (48-dim: linear C=1e2; 1280-dim: StandardScaler+linear C=1e2).
"""
from __future__ import annotations
import json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader

_ROOT = "/dados/home/moliveira/scalable_FLIM_self_supervised"
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from src.evaluate.svm_distillation import (
    _OneHotDataset, extract_features_distillation, train_svm_distillation,
    IMAGE_SIZE, _DATASET_NUM_CLASSES, _DATASET_PARASITE_NAME,
)
from src.evaluate.svm_distill_with_projection import (
    _load_student_and_proj, _extract_proj, _train_svm_proj,
)
from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(_ROOT, "analysis_flim_distill")
ART = os.path.join(_ROOT, "artifacts", "distillation")
DATASETS = ["eggs", "larvae"]

CELLS = {  # (scale, init) -> run-dir template
    ("126k", "flim"):  "distillation_{ds}_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
    ("126k", "trunc"): "distillation_{ds}_split1_pct100_1x1_BN2d_1280_one_layer",
    ("400k", "flim"):  "distillation_{ds}_split1_pct100_2l_1x1_init_flim_256_1280",
    ("400k", "trunc"): "distillation_{ds}_split1_pct100_2l_1x1_BN2d_256_1280",
}


def _ckpt(run_dir: str) -> str:
    import re
    cdir = os.path.join(ART, run_dir, "checkpoints")
    hits = [os.path.join(r, f) for r, _d, fs in os.walk(cdir) for f in fs if f.endswith(".ckpt")]
    best = [h for h in hits if "best" in h] or hits
    if not best: raise FileNotFoundError(cdir)
    def _loss(p):
        m = re.search(r"loss=([0-9.]+)\.ckpt$", p); return float(m.group(1)) if m else float("inf")
    return min(best, key=_loss)


def _loaders(ds):
    parasite, ncls = _DATASET_PARASITE_NAME[ds], _DATASET_NUM_CLASSES[ds]
    t = _build_test(IMAGE_SIZE)  # the (pipeline) transform the students were trained/eval'd with
    tr = DatasetParasite("train", split=1, percentage=100, transform=t, loader="ift_lab", path_dataset=parasite)
    te = DatasetParasite("test", split=1, percentage=100, transform=t, loader="ift_lab", path_dataset=parasite)
    tr_oh = DataLoader(_OneHotDataset(tr, ncls), batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    te_pl = DataLoader(te, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    return tr_oh, te_pl, ncls


def eval_48(student, tr_oh, te_pl, ncls):
    clf = train_svm_distillation(student, tr_oh)
    feats, y = extract_features_distillation(student, te_pl)
    m = compute_metrics(y_true=y, y_pred=clf.predict(feats) - 1, num_classes=ncls)
    return float(m["kappa"]), float(m["acc"])


def eval_1280(student, proj, tr_oh, te_pl, ncls):
    clf = _train_svm_proj(student, proj, tr_oh)
    feats, y = _extract_proj(student, proj, te_pl)
    m = compute_metrics(y_true=y, y_pred=clf.predict(feats) - 1, num_classes=ncls)
    return float(m["kappa"]), float(m["acc"])


def main():
    res = {}
    for ds in DATASETS:
        print(f"\n{'='*60}\n{ds}\n{'='*60}", flush=True)
        tr_oh, te_pl, ncls = _loaders(ds)
        res[ds] = {}
        for (scale, init), tmpl in CELLS.items():
            run = tmpl.format(ds=ds)
            ck = _ckpt(run)
            student, proj = _load_student_and_proj(ck, DEVICE)
            student.eval(); proj.eval()
            for p in list(student.parameters()) + list(proj.parameters()): p.requires_grad_(False)
            k48, a48 = eval_48(student, tr_oh, te_pl, ncls)
            k1280, a1280 = eval_1280(student, proj, tr_oh, te_pl, ncls)
            res[ds][f"{scale}_{init}"] = {"kappa_48enc": k48, "acc_48enc": a48,
                                          "kappa_1280proj": k1280, "acc_1280proj": a1280}
            print(f"  {scale}_{init}: 48enc κ={k48:.3f}  1280proj κ={k1280:.3f}", flush=True)
            del student, proj
            if DEVICE.type == "cuda": torch.cuda.empty_cache()
        with open(os.path.join(OUT, "ruler_mismatch.json"), "w") as f:
            json.dump(res, f, indent=2)

    # tables
    print("\n\n### 48-dim ENCODER ruler (κ)\n\n| ds | 126k flim | 126k trunc | 400k flim | 400k trunc |")
    print("|---|---|---|---|---|")
    for ds in DATASETS:
        r = res[ds]
        print(f"| {ds} | {r['126k_flim']['kappa_48enc']:.3f} | {r['126k_trunc']['kappa_48enc']:.3f} "
              f"| {r['400k_flim']['kappa_48enc']:.3f} | {r['400k_trunc']['kappa_48enc']:.3f} |")
    print("\n### 1280-dim PROJECTION ruler (κ)\n\n| ds | 126k flim | 126k trunc | 400k flim | 400k trunc |")
    print("|---|---|---|---|---|")
    for ds in DATASETS:
        r = res[ds]
        print(f"| {ds} | {r['126k_flim']['kappa_1280proj']:.3f} | {r['126k_trunc']['kappa_1280proj']:.3f} "
              f"| {r['400k_flim']['kappa_1280proj']:.3f} | {r['400k_trunc']['kappa_1280proj']:.3f} |")


if __name__ == "__main__":
    main()
