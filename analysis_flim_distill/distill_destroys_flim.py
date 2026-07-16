"""distill_destroys_flim.py — Decompose WHY distilling a good FLIM CNN toward
I-JEPA makes its standalone SVM kappa WORSE.

For ds in {eggs, larvae, protozoan}, split=1, pct=100, we fit a LINEAR SVM on
48-dim encoder embeddings (student.encode) on the TRAIN split and evaluate on the
TEST split, using EXACTLY the protocol of src/evaluate/svm_distillation_conv.py
(reused train_svm_distillation + extract_features_distillation: linear SVC,
C=1e2, gamma='auto', ovo, 1-indexed labels). Numbers are thus directly
comparable to results/*.csv.

Six encoders are evaluated:
  1. FLIM_untrained_pipelinenorm — untrained FLIM, fed through the distillation
     pipeline transform (ift_lab LAB + v2.Normalize ImageNet-RGB → the buggy path).
  2. FLIM_untrained_lab01        — untrained FLIM, ift_lab LAB in [0,1], NO Normalize.
  3. FLIM_untrained_markernorm   — untrained FLIM with FLIM per-marker patch
     normalization ((patch-mean)/stdev per conv layer), input in LAB[0,1].
  4. FLIM_distilled_126k         — distilled student (one-layer head, flim_init).
  5. FLIM_distilled_400k         — distilled student (2-layer head, flim_init).
  6. TRUNC_distilled_400k        — distilled student (2-layer head, trunc_normal init).

Encoders 4-6 use the SAME (pipeline/buggy) transform they were trained with.
Encoder 4/5 FLIM kappa must reproduce results/svm_*_results.csv within ~0.05,
else we STOP and report a pipeline mismatch.

Also computes per-conv-layer filter-drift between UNTRAINED FLIM (enc 1) and the
DISTILLED FLIM encoders (enc 4 and enc 5): mean cosine similarity of matched
filters and relative L2 change ||W_d-W_i||/||W_i||.
"""
from __future__ import annotations

import json
import os
import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = "/dados/home/moliveira/scalable_FLIM_self_supervised"
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from torchvision.transforms import v2

from src.evaluate.svm_distillation import (
    _OneHotDataset,
    extract_features_distillation,
    train_svm_distillation,
    IMAGE_SIZE,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
)
from src.evaluate.svm_distill_with_projection import _load_student_and_proj
from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
    load_FLIM_encoder,
    load_FLIM_encoder_from_arch_dict,
    PROTOZOAN_FLIM_ARCH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(_ROOT, "analysis_flim_distill")
ART = os.path.join(_ROOT, "artifacts", "distillation")

DATASETS = ["eggs", "larvae", "protozoan"]

# (ds) -> {role: run_dir}
RUNS = {
    "eggs": {
        "126k":  "distillation_eggs_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
        "400k":  "distillation_eggs_split1_pct100_2l_1x1_init_flim_256_1280",
        "trunc": "distillation_eggs_split1_pct100_2l_1x1_BN2d_256_1280",
    },
    "larvae": {
        "126k":  "distillation_larvae_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
        "400k":  "distillation_larvae_split1_pct100_2l_1x1_init_flim_256_1280",
        "trunc": "distillation_larvae_split1_pct100_2l_1x1_BN2d_256_1280",
    },
    "protozoan": {
        "126k":  "distillation_protozoan_split1_pct100_1x1_BN2d_1280_one_layer_flim_init",
        "400k":  "distillation_protozoan_split1_pct100_2l_1x1_init_flim_256_1280",
        "trunc": "distillation_protozoan_split1_pct100_2l_1x1_BN2d_256_1280",
    },
}

# Reference kappa from results CSVs for the verification gate (FLIM distilled).
REF_KAPPA_400K = {  # results/svm_2l_1x1_init_flim_256_1280_results.csv (split1 pct100)
    "eggs": 0.7151, "larvae": 0.8989, "protozoan": None,
}
REF_KAPPA_126K = {  # results/svm_*1x1_BN2d_1280_one_layer*flim* (split1 pct100)
    "eggs": None, "larvae": None, "protozoan": None,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ── Transforms ───────────────────────────────────────────────────────────────

def transform_pipeline():
    """Exactly the distillation pipeline test transform (ift_lab LAB + ImageNet Normalize)."""
    return _build_test(IMAGE_SIZE)


def transform_lab01():
    """ift_lab LAB in [0,1], resize/centercrop, NO ImageNet Normalize."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(IMAGE_SIZE),
        v2.CenterCrop(IMAGE_SIZE),
        v2.ToDtype(torch.float32, scale=True),  # back to [0,1]
    ])


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def _ckpt(run_dir: str) -> str:
    import re
    cdir = os.path.join(ART, run_dir, "checkpoints")
    hits = []
    for root, _d, files in os.walk(cdir):
        for f in files:
            if f.endswith(".ckpt"):
                hits.append(os.path.join(root, f))
    best = [h for h in hits if "best" in h]
    chosen = best if best else hits
    if not chosen:
        raise FileNotFoundError(f"no ckpt under {cdir}")

    def _loss(p):
        m = re.search(r"loss=([0-9.]+)\.ckpt$", p)
        return float(m.group(1)) if m else float("inf")
    return min(chosen, key=_loss)


# ── Build untrained FLIM encoder (same construction as the distilled run) ─────

def _build_flim_arch_channels(ds: str, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    hp = ck["hyper_parameters"]
    arch_json = hp["arch_json"]
    in_channels = hp.get("in_channels", 3)
    proj_dim = hp.get("proj_dim", 256)
    proj_hidden = hp.get("proj_hidden", 2048)
    flim_weights_path = hp["flim_weights_path"]

    if ds == "protozoan":
        arch = copy.deepcopy(PROTOZOAN_FLIM_ARCH)
    else:
        arch = parse_architecture(arch_json)
    channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
    arch = override_arch_channels(arch, channels)
    return arch, channels, arch_json, in_channels, proj_dim, proj_hidden, flim_weights_path


def build_untrained_flim(ds: str) -> LeJEPAFLIMModel:
    """Untrained FLIM encoder loaded with the FLIM weights (the distilled run's init state)."""
    ckpt_path = _ckpt(RUNS[ds]["400k"])  # any flim run gives same FLIM init
    (arch, channels, arch_json, in_channels,
     proj_dim, proj_hidden, flim_weights_path) = _build_flim_arch_channels(ds, ckpt_path)

    student = LeJEPAFLIMModel(arch=arch, in_channels=in_channels,
                              proj_dim=proj_dim, proj_hidden=proj_hidden)
    if ds == "protozoan":
        load_FLIM_encoder_from_arch_dict(student, arch, flim_weights_path, channels)
    else:
        load_FLIM_encoder(student, arch_json, flim_weights_path, channels)
    return student.to(DEVICE).eval(), arch, flim_weights_path


# ── Marker-normalized FLIM encoder ───────────────────────────────────────────

def _read_marker_vec(path: str) -> np.ndarray:
    """conv{n}-mean/stdev.txt: a single whitespace-separated line of floats
    (length = kernel_size**2 * in_channels). No count header (unlike bias.txt)."""
    with open(path) as f:
        txt = f.read().strip().split()
    return np.array(txt, dtype=np.float32)


class MarkerNormFLIM(torch.nn.Module):
    """Reproduces the standalone-FLIM forward with per-marker patch normalization.

    For each conv layer we replicate FLIM's patch normalization: each unfolded
    patch (im2col) is normalized as (patch - mean)/stdev element-wise, then
    convolved with the FLIM kernel matrix and biased.  The mean/stdev vectors
    are ordered channel-fastest within each spatial position (the same order as
    the raw conv{n}-kernels.npy first axis used by shift_weights), so we unfold
    with torch.nn.Unfold (channel-slowest) and reorder to channel-fastest.

    The conv weight / bias / pooling structure is taken from the already-built
    untrained FLIM Encoder so kernels match exactly.
    """

    def __init__(self, encoder, arch: dict, weights_path: str, in_channels: int = 3):
        super().__init__()
        self.encoder = encoder       # src.models.models.Encoder
        self.arch = arch
        self.n_layers = arch["nlayers"]
        self.layers = []
        ch_in = in_channels
        for n in range(1, self.n_layers + 1):
            conv_desc = arch[f"layer{n}"]["conv"]
            ks = conv_desc["kernel_size"]
            kH, kW = ks[0], ks[1]
            dil = conv_desc["dilation_rate"]
            dH, dW = dil[0], dil[1]
            pad = (kH // 2 * dH, kW // 2 * dW)
            block = getattr(encoder, f"conv{n}")
            conv = block[0]  # nn.Conv2d
            # marker stats (kH*kW*ch_in,) ordered as spatial-major, channel-fastest
            mean = _read_marker_vec(os.path.join(weights_path, f"conv{n}-mean.txt"))
            std = _read_marker_vec(os.path.join(weights_path, f"conv{n}-stdev.txt"))
            expected = kH * kW * ch_in
            assert mean.size == expected, f"conv{n} mean size {mean.size} != {expected}"
            assert std.size == expected, f"conv{n} std size {std.size} != {expected}"
            # reorder FLIM channel-fastest -> torch.Unfold channel-slowest:
            #   flim index = p*ch_in + c   (p spatial, c channel)
            #   torch index = c*(kH*kW) + p
            mean = mean.reshape(kH * kW, ch_in)   # [p, c]
            std = std.reshape(kH * kW, ch_in)
            mean_t = torch.from_numpy(mean.T.reshape(-1)).float()  # [c*p] channel-slowest
            std_t = torch.from_numpy(std.T.reshape(-1)).float()
            self.layers.append(dict(
                n=n, kH=kH, kW=kW, dH=dH, dW=dW, pad=pad,
                conv=conv, block=block,
                ch_in=ch_in, ch_out=conv.weight.shape[0],
                mean=mean_t.view(1, -1, 1).to(DEVICE),
                std=std_t.view(1, -1, 1).to(DEVICE),
            ))
            ch_in = conv.weight.shape[0]
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    @torch.no_grad()
    def _conv_marker(self, x, L):
        B, C, H, W = x.shape
        unfold = F.unfold(x, kernel_size=(L["kH"], L["kW"]),
                          dilation=(L["dH"], L["dW"]), padding=L["pad"])  # [B, C*kH*kW, Lp]
        std = torch.where(L["std"] < 1e-8, torch.ones_like(L["std"]), L["std"])
        unfold = (unfold - L["mean"]) / std
        conv = L["conv"]
        Wmat = conv.weight.view(conv.weight.shape[0], -1)  # [out, C*kH*kW]
        out = Wmat @ unfold                                # [B? ] -> need batch
        # torch matmul broadcast: Wmat [out, K], unfold [B, K, Lp]
        out = torch.einsum("ok,bkl->bol", Wmat, unfold)
        out = out + conv.bias.view(1, -1, 1)
        # output spatial size (same conv padding => H,W preserved for stride1)
        Hout = (H + 2 * L["pad"][0] - L["dH"] * (L["kH"] - 1) - 1) + 1
        Wout = (W + 2 * L["pad"][1] - L["dW"] * (L["kW"] - 1) - 1) + 1
        out = out.view(B, L["ch_out"], Hout, Wout)
        return out

    @torch.no_grad()
    def encode(self, x):
        x = x.to(DEVICE)
        for L in self.layers:
            x = self._conv_marker(x, L)
            block = L["block"]
            # apply ReLU + pooling exactly as in the original block (skip conv at idx0)
            for mod in list(block)[1:]:
                x = mod(x)
        x = self.pool(x)
        return x.flatten(1)

    def eval(self):
        return self


# ── SVM eval (same protocol as svm_distillation_conv) ────────────────────────

def eval_encoder(encoder, ds: str, transform) -> dict:
    """Fit linear SVM on train-split 48-d embeddings, eval on test split.
    Reuses train_svm_distillation + extract_features_distillation (C=1e2 linear ovo)."""
    parasite = _DATASET_PARASITE_NAME[ds]
    num_classes = _DATASET_NUM_CLASSES[ds]

    train_ds = DatasetParasite(set_name="train", split=1, percentage=100,
                               transform=transform, loader="ift_lab", path_dataset=parasite)
    train_loader = DataLoader(_OneHotDataset(train_ds, num_classes),
                              batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    clf = train_svm_distillation(encoder, train_loader)

    test_ds = DatasetParasite(set_name="test", split=1, percentage=100,
                              transform=transform, loader="ift_lab", path_dataset=parasite)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)
    feats, y_true = extract_features_distillation(encoder, test_loader)
    y_pred = clf.predict(feats) - 1
    m = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
    return {"kappa": float(m["kappa"]), "acc": float(m["acc"]), "f1": float(m["f1"]),
            "n_test": int(len(y_true))}


# ── Filter-drift metrics ─────────────────────────────────────────────────────

def conv_weight(encoder, n: int) -> torch.Tensor:
    block = getattr(encoder, f"conv{n}")
    return block[0].weight.detach().cpu().float()


def filter_drift(enc_init, enc_dist, n_layers: int) -> dict:
    out = {}
    for n in range(1, n_layers + 1):
        Wi = conv_weight(enc_init, n)   # [out, in, kH, kW]
        Wd = conv_weight(enc_dist, n)
        oi = min(Wi.shape[0], Wd.shape[0])
        Wi = Wi[:oi].reshape(oi, -1)
        Wd = Wd[:oi].reshape(oi, -1)
        cos = F.cosine_similarity(Wi, Wd, dim=1)
        rel_l2 = (torch.norm(Wd - Wi, dim=1) / (torch.norm(Wi, dim=1) + 1e-12))
        out[f"conv{n}"] = {
            "mean_cosine": float(cos.mean()),
            "median_cosine": float(cos.median()),
            "mean_rel_l2": float(rel_l2.mean()),
            "n_filters": int(oi),
        }
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    results = {}
    pipe_t = transform_pipeline()
    lab_t = transform_lab01()

    for ds in DATASETS:
        print(f"\n{'='*70}\n{ds}\n{'='*70}", flush=True)
        ds_res = {}

        # Build untrained FLIM encoder (shared model object for encoders 1,2,3 + drift ref)
        untrained, arch, flim_wp = build_untrained_flim(ds)
        n_layers = arch["nlayers"]

        # 1. FLIM_untrained_pipelinenorm
        print("[1] FLIM_untrained_pipelinenorm", flush=True)
        ds_res["FLIM_untrained_pipelinenorm"] = eval_encoder(untrained, ds, pipe_t)
        print("   ", ds_res["FLIM_untrained_pipelinenorm"], flush=True)

        # 2. FLIM_untrained_lab01
        print("[2] FLIM_untrained_lab01", flush=True)
        ds_res["FLIM_untrained_lab01"] = eval_encoder(untrained, ds, lab_t)
        print("   ", ds_res["FLIM_untrained_lab01"], flush=True)

        # 3. FLIM_untrained_markernorm
        print("[3] FLIM_untrained_markernorm", flush=True)
        marker = MarkerNormFLIM(untrained.encoder, arch, flim_wp, in_channels=3)
        ds_res["FLIM_untrained_markernorm"] = eval_encoder(marker, ds, lab_t)
        print("   ", ds_res["FLIM_untrained_markernorm"], flush=True)

        # 4. FLIM_distilled_126k
        print("[4] FLIM_distilled_126k", flush=True)
        ck126 = _ckpt(RUNS[ds]["126k"])
        s126, _ = _load_student_and_proj(ck126, DEVICE)
        s126.eval()
        for p in s126.parameters():
            p.requires_grad_(False)
        ds_res["FLIM_distilled_126k"] = eval_encoder(s126, ds, pipe_t)
        print("   ", ds_res["FLIM_distilled_126k"], flush=True)

        # 5. FLIM_distilled_400k
        print("[5] FLIM_distilled_400k", flush=True)
        ck400 = _ckpt(RUNS[ds]["400k"])
        s400, _ = _load_student_and_proj(ck400, DEVICE)
        s400.eval()
        for p in s400.parameters():
            p.requires_grad_(False)
        ds_res["FLIM_distilled_400k"] = eval_encoder(s400, ds, pipe_t)
        print("   ", ds_res["FLIM_distilled_400k"], flush=True)

        # ── Verification gate against results CSV (FLIM distilled 400k) ──────
        ref = REF_KAPPA_400K.get(ds)
        got = ds_res["FLIM_distilled_400k"]["kappa"]
        if ref is not None:
            delta = abs(got - ref)
            print(f"   [VERIFY 400k] got={got:.4f} ref={ref:.4f} |Δ|={delta:.4f}", flush=True)
            ds_res["_verify_400k"] = {"got": got, "ref": ref, "abs_delta": float(delta),
                                       "pass": bool(delta <= 0.05)}
            if delta > 0.05:
                print(f"   [STOP] FLIM_distilled_400k kappa mismatch for {ds} "
                      f"(|Δ|={delta:.4f} > 0.05). Eval pipeline differs from CSV.",
                      flush=True)
                results[ds] = ds_res
                _dump(results)
                raise SystemExit(
                    f"Verification failed for {ds}: distilled 400k kappa {got:.4f} "
                    f"vs CSV {ref:.4f} (|Δ|={delta:.4f} > 0.05)."
                )

        # 6. TRUNC_distilled_400k
        print("[6] TRUNC_distilled_400k", flush=True)
        ckT = _ckpt(RUNS[ds]["trunc"])
        sT, _ = _load_student_and_proj(ckT, DEVICE)
        sT.eval()
        for p in sT.parameters():
            p.requires_grad_(False)
        ds_res["TRUNC_distilled_400k"] = eval_encoder(sT, ds, pipe_t)
        print("   ", ds_res["TRUNC_distilled_400k"], flush=True)

        # ── Filter drift: untrained FLIM vs distilled 126k / 400k ───────────
        ds_res["drift_init_vs_126k"] = filter_drift(untrained.encoder, s126.encoder, n_layers)
        ds_res["drift_init_vs_400k"] = filter_drift(untrained.encoder, s400.encoder, n_layers)
        print("   drift init->126k:", ds_res["drift_init_vs_126k"], flush=True)
        print("   drift init->400k:", ds_res["drift_init_vs_400k"], flush=True)

        results[ds] = ds_res
        _dump(results)

        del s126, s400, sT, untrained, marker
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    _dump(results)
    _print_tables(results)


def _dump(results):
    with open(os.path.join(OUT_DIR, "distill_destroys_flim.json"), "w") as f:
        json.dump(results, f, indent=2)


def _print_tables(results):
    encoders = [
        "FLIM_untrained_pipelinenorm",
        "FLIM_untrained_lab01",
        "FLIM_untrained_markernorm",
        "FLIM_distilled_126k",
        "FLIM_distilled_400k",
        "TRUNC_distilled_400k",
    ]
    print("\n\n### Kappa table (linear-SVM Cohen's kappa)\n")
    header = "| encoder | eggs | larvae | protozoan |"
    print(header)
    print("|" + "---|" * 4)
    for e in encoders:
        cells = []
        for ds in DATASETS:
            v = results.get(ds, {}).get(e, {})
            cells.append(f"{v.get('kappa', float('nan')):.3f}" if v else "n/a")
        print(f"| {e} | {' | '.join(cells)} |")

    print("\n### Accuracy table\n")
    print(header.replace("kappa", "acc"))
    print(header)
    print("|" + "---|" * 4)
    for e in encoders:
        cells = []
        for ds in DATASETS:
            v = results.get(ds, {}).get(e, {})
            cells.append(f"{v.get('acc', float('nan')):.3f}" if v else "n/a")
        print(f"| {e} | {' | '.join(cells)} |")

    print("\n### Per-layer filter drift (untrained FLIM vs distilled)\n")
    print("| ds | comparison | layer | mean_cos | mean_rel_l2 |")
    print("|---|---|---|---|---|")
    for ds in DATASETS:
        for comp in ("drift_init_vs_126k", "drift_init_vs_400k"):
            d = results.get(ds, {}).get(comp, {})
            for layer, m in d.items():
                print(f"| {ds} | {comp} | {layer} | {m['mean_cosine']:.3f} | {m['mean_rel_l2']:.3f} |")


if __name__ == "__main__":
    main()
