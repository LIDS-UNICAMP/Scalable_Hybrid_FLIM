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
"""Heatmaps "nosso melhor x melhor FM", 10 imagens de TESTE (split 1) por base, para os slides.

Mapa = evidencia do SVM linear do teste (C=100, sem scaler, ovo; core.svm.fit_svm no treino do
split 1) projetada por posicao: CAM_c(x, y) = sum_k w_{c,k} f_k(x, y), com w_c = soma das linhas
ovo a favor da classe predita (menos as linhas contra). f = saida da conv 1x1 do head antes do
pooling (nosso), tokens de patch apos a norma final (ViT) ou ultimo estagio com a LayerNorm do
pooler por posicao (ConvNeXt). Cada mapa e normalizado 0-1 por imagem.

Passos (cada um num processo, por causa dos dois pacotes ``core``):
  ours_list  repo Scalable_Hybrid_FLIM: nomes/rotulos do nosso teste
  fm         repo Scalable_Hybrid_FLIM_discovery: escolhe as 10 imagens, SVM do benchmark, mapas FM
  ours       repo Scalable_Hybrid_FLIM: SVM do teste do nosso modelo, mapas
  plot       PNG por imagem, grade.png (10x3), grade_5x2.png (5 colunas x 2 blocos) e README.md
Argumentos extras depois do passo = bases (ex.: ``fm coconut_trees``); sem nada, todas.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path("/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps")
WORK = Path("/exps/moliveira/heatmaps_work")
ORG = Path("/dados/home/moliveira/datasets/organized")
REPO = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
JACI = "/exps/moliveira/heatmaps_src/artifacts"
N_IMG, SEED = 10, 42

FMS = {"aimv2": ("apple/aimv2-3B-patch14-448", "aimv2-3B-patch14-448", "AIMv2-3B/14 (448)"),
       "dinov2": ("facebook/dinov2-giant", "dinov2-giant", "DINOv2 ViT-g/14"),
       "convnextv2": ("facebook/convnextv2-pico-1k-224", "convnextv2-pico-1k-224", "ConvNeXtV2-pico"),
       "ijepa": ("facebook/ijepa_vitg16_22k", "ijepa_vitg16_22k", "I-JEPA ViT-g/16")}

# base: (dataset da busca, dataset do benchmark, FM, checkpoint nosso, rotulo nosso, kappa teste nosso)
BASES = {
    "remote_sensing": ("remote_sensing", "remote_sensing", "aimv2",
                       f"{REPO}/artifacts/pnas_cs/remote_sensing/state/seed42/candidates/c00008/f100/encoder.pt",
                       "PNAS-CS P1 c00008 f100", 0.633),
    "coconut_trees": ("coconut_trees", "coconut_trees", "convnextv2",
                      f"{REPO}/artifacts/modehb_hybrid/coconut_trees/state/seed42/candidates/c00001/f200/encoder.pt",
                      "MO-DEHB hybrid c00001 f200 (tiny)", 0.813),
    "eggs": ("eggs", "Eggs", "dinov2",
             f"{REPO}/artifacts/modehb_hybrid/eggs/state/seed42/candidates/c00002/f200/encoder.pt",
             "MO-DEHB hybrid c00002 f200 (small)", 0.846),
    "larvae": ("larvae", "Larvae", "aimv2",
               f"{REPO}/artifacts/modehb_lemonade/larvae/state/seed42/candidates/c00003/f200/encoder.pt",
               "LEMONADE c00003 f200 (medium)", 0.858),
    "protozoan": ("protozoan", "Protozoan", "aimv2",
                  f"{JACI}/modehb_hybrid/protozoan/state/seed42/candidates/c00001/f45/encoder.pt",
                  "MO-DEHB hybrid c00001 f45", 0.798),
    "ham10000": ("ham10000", "ham10000", "aimv2",
                 f"{JACI}/modehb_hybrid/ham10000/state/seed42/candidates/c00001/f200/encoder.pt",
                 "MO-DEHB hybrid c00001 f200", 0.534),
    "aid": ("aid", "aid", "aimv2",
            f"{JACI}/modehb_hybrid_C01_2049/aid/state/seed42/candidates/c00001/f45/encoder.pt",
            "MO-DEHB hybrid c00001 f45", 0.675),
    "flowers": ("flowers", "flowers", "aimv2",
                f"{REPO}/artifacts/modehb_hybrid_C01_final/flowers/state/seed42/candidates/c00001/f200/encoder.pt",
                "MO-DEHB hybrid c00001 f200 (tiny)", 0.649),
    "coconut_trees_ijepa": ("coconut_trees", "coconut_trees", "ijepa",
                            f"{REPO}/artifacts/modehb_hybrid/coconut_trees/state/seed42/candidates/c00001/f200/encoder.pt",
                            "MO-DEHB hybrid c00001 f200 (tiny)", 0.813),
}
# por que este FM (README); padrao: o melhor FM da base no benchmark
FM_WHY = {"coconut_trees_ijepa": "segundo melhor FM da base no benchmark; versao extra porque refazer o SVM do "
                                 "melhor (ConvNeXtV2-pico, pasta coconut_trees/) leva ~8 h"}


def class_names(ds_fm):
    for f, key in (("manifest.json", "classes"), ("classes.json", None)):
        p = ORG / ds_fm / f
        if p.exists():
            c = json.load(open(p))
            c = c[key] if key else c
            if isinstance(c, dict):
                c = [v for _, v in sorted(c.items(), key=lambda kv: int(kv[0]))]
            return [x if isinstance(x, str) else x.get("name", str(x)) for x in c]
    raise FileNotFoundError(ds_fm)


def ovo_class_weights(svc, X):
    """{indice em classes_: w_c}. Binario: decision > 0 -> classes_[1]. ovo: par (i, j) > 0 vota em i.
    Conferido contra svc.predict em X (os votos reconstruidos tem que bater)."""
    coef, n = np.asarray(svc.coef_.todense() if hasattr(svc.coef_, "todense") else svc.coef_), len(svc.classes_)
    if n == 2:
        return {1: coef[0], 0: -coef[0]}
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dec = X @ coef.T + svc.intercept_
    votes = np.zeros((len(X), n))
    for k, (i, j) in enumerate(pairs):
        votes[np.arange(len(X)), np.where(dec[:, k] > 0, i, j)] += 1
    agree = (svc.classes_[votes.argmax(1)] == svc.predict(X)).mean()
    assert agree > 0.97, f"convencao ovo nao confere ({agree:.3f})"
    w = {c: np.zeros(coef.shape[1]) for c in range(n)}
    for k, (i, j) in enumerate(pairs):
        w[i] += coef[k]
        w[j] -= coef[k]
    return w


def norm01(m):
    m = np.asarray(m, dtype=np.float64)
    return (m - m.min()) / (m.max() - m.min() + 1e-12)


# ── passo 1: nomes do nosso teste ────────────────────────────────────────────

def _ours_data(ds):
    from ga_search.modehb.run_search import load_config
    from ga_search.modehb.svm_on_demand import _test_loader
    cfg = load_config(f"{REPO}/ga_search/config/modehb/search_v3.yaml", ds)
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.update(num_workers=8, seed=cfg["seed"])                     # = monitor_report.evaluate_new
    return cfg, d, _test_loader(d)


def step_ours_list():
    for base, (ds, *_rest) in BASES.items():
        _, _, te = _ours_data(ds)
        (WORK / base).mkdir(parents=True, exist_ok=True)
        json.dump({s.name: int(s.label) for s in te.dataset.samples}, open(WORK / base / "ours_test.json", "w"))
        print(base, len(te.dataset.samples), flush=True)


# ── passo 2: FM (repo discovery) ─────────────────────────────────────────────

def _select(test_names, y_fm, rng):
    """10 imagens do teste, estratificadas por classe: com <=10 classes, rodadas de 1 por classe
    (classes em ordem, imagem sorteada dentro da classe); com >10 classes, 10 classes sorteadas e
    1 imagem de cada. Escolha sem olhar predicao nem kappa."""
    by = {}
    for n, y in zip(test_names, y_fm):
        by.setdefault(int(y), []).append(n)
    pools = {c: list(rng.permutation(sorted(v))) for c, v in sorted(by.items())}
    classes = sorted(pools)
    if len(classes) > N_IMG:
        classes = sorted(rng.choice(classes, N_IMG, replace=False).tolist())
    out = []
    while len(out) < N_IMG and any(pools[c] for c in classes):
        for c in classes:
            if pools[c] and len(out) < N_IMG:
                out.append((pools[c].pop(0), c))
    return out


def _fm_load(key):
    import torch
    if key == "aimv2":
        from benchmark.aimv2 import load
        net, _ = load()
        feat = lambda x: net.forward_features(x)                                   # [B, 1024, 3072]
    elif key == "dinov2":
        from benchmark.dinov2 import load
        net, _ = load()
        feat = lambda x: net(x).last_hidden_state[:, 1:]                           # [B, 37*37, 1536]
    elif key == "ijepa":
        from benchmark.ijepa import load
        net, _ = load()
        feat = lambda x: net.tokens(x)                                             # [B, 14*14, 1408]; SVM = media
    else:
        from benchmark.convnextv2 import load
        net, _ = load()
        feat = lambda x: net.layernorm(net(x).last_hidden_state.permute(0, 2, 3, 1)).flatten(1, 2)
    net = net.half().eval().cuda()
    params = sum(p.numel() for p in net.parameters())
    return net, (lambda x: feat(x.half().cuda()).float()), params


def step_fm():
    import csv
    import timm
    import torch
    from PIL import Image
    from timm.data import create_transform, resolve_data_config
    from core.data.parasite_dataset import ParasiteDataset
    from core.svm import fit_svm
    prep = {"aimv2": "aimv2_3b_patch14_448.apple_pt", "dinov2": "vit_giant_patch14_dinov2.lvd142m",
            "convnextv2": "convnextv2_pico.fcmae_ft_in1k", "ijepa": "vit_giant_patch16_gap_224.in22k_ijepa"}
    for key in FMS:
        bases = [b for b, v in BASES.items() if v[2] == key and (not sys.argv[2:] or b in sys.argv[2:])]
        if not bases:
            continue
        net, feat, params = _fm_load(key)
        cfg = resolve_data_config(pretrained_cfg=timm.get_pretrained_cfg(prep[key]).to_dict())
        tf = create_transform(**cfg, is_training=False)
        mean, std = np.array(cfg["mean"])[:, None, None], np.array(cfg["std"])[:, None, None]
        rows_csv = {(r["dataset"], r["split"], r["pct"]): r for r in
                    csv.DictReader(open(f"results/benchmark/{FMS[key][1]}.csv"))}
        for base in bases:
            ds_fm = BASES[base][1]
            emb = np.load(f"/exps/moliveira/benchmark/emb/{FMS[key][1]}/{ds_fm}.npz")
            index = {n: i for i, n in enumerate(emb["names"])}
            samples = {s: ParasiteDataset(s, 1, 100, path_dataset=ds_fm).samples for s in ("train", "test")}
            tr = [index[os.path.basename(p)] for p in samples["train"]]
            paths = {Path(p).stem: p for p in samples["test"]}
            ours = json.load(open(WORK / base / "ours_test.json"))
            # eggs/larvae/protozoan: a busca usa pre/orig renomeado (split1_modehb = mesmo split, nomes novos);
            # o mapeamento oficial raw -> pre esta em splits/split1_modehb_mapping.csv
            mp = ORG / ds_fm / "splits" / "split1_modehb_mapping.csv"
            raw2ours = ({Path(r["old_name_raw"]).stem: Path(r["new_name_pre"]).stem
                         for r in csv.DictReader(open(mp)) if r.get("new_name_pre")} if mp.exists() else {})
            to_ours = lambda n: raw2ours.get(n, n)
            common = sorted(n for n in paths if to_ours(n) in ours)
            y_te = [int(emb["y"][index[os.path.basename(paths[n])]]) for n in common]
            sel = _select(common, y_te, np.random.default_rng(SEED))
            clf = fit_svm(emb["X"][tr], emb["y"][tr] + 1, tag=f"heatmap {key} {ds_fm}")
            svc = getattr(clf, "steps", None) and clf.steps[-1][1] or clf
            Xsel = np.stack([emb["X"][index[os.path.basename(paths[n])]] for n, _ in sel])
            W = ovo_class_weights(svc, emb["X"][tr][:3000])
            pred = clf.predict(Xsel) - 1
            maps, views = [], []
            for (n, _), p in zip(sel, pred):
                x = tf(Image.open(paths[n]).convert("RGB"))[None]
                with torch.inference_mode():
                    f = feat(x)[0].cpu().numpy()                                      # [P, D]
                ci = int(np.where(svc.classes_ == p + 1)[0][0])
                side = int(round(np.sqrt(f.shape[0])))
                maps.append(norm01((f @ W[ci]).reshape(side, side)))
                views.append(np.clip((x[0].numpy() * std + mean).transpose(1, 2, 0), 0, 1))
            r = rows_csv.get((ds_fm, "1", "100"), {})
            np.savez(WORK / base / "fm.npz", names=[n for n, _ in sel], names_ours=[to_ours(n) for n, _ in sel], y=[c for _, c in sel], pred=pred,
                     maps=np.array(maps, dtype=object), views=np.array(views), params=params,
                     model=FMS[key][0], label=FMS[key][2], kappa=float(r.get("test_kappa", "nan")),
                     paths=[paths[n] for n, _ in sel], n_common=len(common), size=cfg["input_size"][-1],
                     crop_pct=cfg["crop_pct"], classes=class_names(ds_fm))
            print(base, key, "sel", [n for n, _ in sel], "pred", pred.tolist(), flush=True)
        del net
        torch.cuda.empty_cache()


# ── passo 3: nosso modelo ────────────────────────────────────────────────────

def step_ours():
    import torch
    from pathlib import Path as P
    from core.svm import fit_svm
    from ga_search.modehb.svm_on_demand import _features
    from ga_search.progressive_data import load_data
    dev = torch.device("cuda")
    for base, (ds, ds_fm, _fm, ckpt, label, kappa) in BASES.items():
        if sys.argv[2:] and base not in sys.argv[2:] or not (WORK / base / "fm.npz").exists():
            continue
        cfg, d, te = _ours_data(ds)
        tr, _, _, _ = load_data(d, P(REPO) / cfg["output_dir"] / "svm_data")
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        enc, head = ck["encoder"].to(dev).eval(), ck["head"].to(dev).eval()
        Xtr, ytr = _features(enc, head, tr, dev)
        Xte, yte = _features(enc, head, te, dev)
        labels = np.unique(np.concatenate([ytr, yte]))                           # = evaluate_checkpoint
        remap = {int(c): i for i, c in enumerate(labels)}
        ytr0, yte0 = np.vectorize(remap.get)(ytr), np.vectorize(remap.get)(yte)
        clf = fit_svm(Xtr, ytr0, C=100.0, scaler=False, tag=f"heatmap ours {ds}")
        svc = getattr(clf, "steps", None) and clf.steps[-1][1] or clf
        W = ovo_class_weights(svc, Xtr[:3000])
        from core.metrics import compute_metrics
        k_check = compute_metrics(yte0, clf.predict(Xte), num_classes=len(labels))["kappa"]
        fm = np.load(WORK / base / "fm.npz", allow_pickle=True)
        name2i = {s.name: i for i, s in enumerate(te.dataset.samples)}
        width, block = head.width, head.block
        chan = np.array([i if i < width else (i - width) % block for i in range(head.out_dim)])
        maps, views, pred, ytrue = [], [], [], []
        for n in fm["names_ours"]:
            x, y, _ = te.dataset[name2i[str(n)]]
            with torch.no_grad():
                z = head.proj(enc(x[None].to(dev))).float()[0].cpu().numpy()     # [width, h, w]
                e = head(enc(x[None].to(dev))).float().cpu().numpy()
            p = int(clf.predict(e)[0])
            ci = int(np.where(svc.classes_ == p)[0][0])
            wch = np.bincount(chan, weights=W[ci], minlength=width)
            maps.append(norm01(np.tensordot(wch, z, axes=1)))
            v = x.numpy().transpose(1, 2, 0)
            views.append((v - v.min()) / (v.max() - v.min() + 1e-12))
            pred.append(p), ytrue.append(remap[int(y)])
        bad = [(str(n), a, int(b)) for n, a, b in zip(fm["names_ours"], ytrue, fm["y"]) if a != int(b)]
        assert not bad, f"rotulo nosso != rotulo FM: {bad}"
        np.savez(WORK / base / "ours.npz", pred=pred, y=ytrue, maps=np.array(maps, dtype=object),
                 views=np.array(views), params=sum(t.numel() for m in (enc, head) for t in m.parameters()),
                 label=label, kappa=kappa, kappa_check=k_check, ckpt=ckpt, label_ids=labels)
        print(base, "kappa teste recalculado", round(float(k_check), 4), "(tabela", kappa, ")", flush=True)
        del enc, head
        torch.cuda.empty_cache()


# ── passo 4: figuras ─────────────────────────────────────────────────────────

def _fmt_params(n):
    n = float(n)
    s = f"{n / 1e9:.1f} bi" if n >= 1e9 else f"{n / 1e6:.2f} mi" if n >= 1e6 else f"{n / 1e3:.0f} mil"
    return s.replace(".", ",")


def _short(label):
    s = str(label).split(' ')[0].replace('-3B/14', '').replace('-pico', '')
    return s if len(s) <= 8 else s[:8] + '\n' + s[8:]    # ConvNeXtV2 -> ConvNeXt / V2 (cabe na altura da linha)


def _overlay(ax, view, m, cmap):
    from scipy.ndimage import zoom
    h, w = view.shape[:2]
    mz = zoom(m.astype(float), (h / m.shape[0], w / m.shape[1]), order=1)
    ax.imshow(view)
    ax.imshow(np.clip(mz, 0, 1), cmap=cmap, alpha=0.45, vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])


def _mark(ax, ok, fs):
    ax.text(0.04, 0.96, "✓" if ok else "✗", transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="#2e7d32" if ok else "#c62828", ec="none"))


def step_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    cmap = plt.get_cmap("jet")
    for base in BASES:
        if not (WORK / base / "ours.npz").exists():
            print("pulado", base)
            continue
        fm = np.load(WORK / base / "fm.npz", allow_pickle=True)
        us = np.load(WORK / base / "ours.npz", allow_pickle=True)
        out = OUT / base
        out.mkdir(parents=True, exist_ok=True)
        cls = list(fm["classes"])
        safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")
        ours_lbl = f"Nosso: {us['label']}\n{_fmt_params(us['params'])} params, κ teste {float(us['kappa']):.3f}".replace(".", ",")
        fm_lbl = f"{fm['label']}\n{_fmt_params(fm['params'])} params, κ teste {float(fm['kappa']):.3f}".replace(".", ",")
        files = []
        origs = [np.asarray(Image.open(p).convert("RGB").resize((448, 448))) / 255.0 for p in fm["paths"]]
        for i, n in enumerate(fm["names"]):
            yt = int(fm["y"][i])
            ok_u, ok_f = int(us["pred"][i]) == int(us["y"][i]), int(fm["pred"][i]) == yt
            fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.2))
            axs[0].imshow(origs[i]), axs[0].set_xticks([]), axs[0].set_yticks([])
            axs[0].set_title(f"Original — classe: {cls[yt]}", fontsize=10)
            _overlay(axs[1], us["views"][i], us["maps"][i], cmap)
            pu = cls[int(us["pred"][i])] if int(us["pred"][i]) < len(cls) else str(us["pred"][i])
            axs[1].set_title(f"{ours_lbl}\npred: {pu} {'✓' if ok_u else '✗'}", fontsize=8)
            _overlay(axs[2], fm["views"][i], fm["maps"][i], cmap)
            axs[2].set_title(f"{fm_lbl}\npred: {cls[int(fm['pred'][i])]} {'✓' if ok_f else '✗'}", fontsize=8)
            f = out / f"img_{i + 1:02d}_{safe(cls[yt])}.png"
            fig.savefig(f, dpi=200, bbox_inches="tight")
            plt.close(fig)
            files.append(f.name)
        # grade: linhas original | nosso | FM, 10 colunas, colorbar unica 0-1
        wcm = 26.0
        fs = 7 * wcm / 12
        fig, axs = plt.subplots(3, N_IMG, figsize=(wcm / 2.54, wcm / 2.54 * 0.36),
                                gridspec_kw=dict(wspace=0.04, hspace=0.06))
        for i in range(len(fm["names"])):
            yt = int(fm["y"][i])
            axs[0, i].imshow(origs[i]), axs[0, i].set_xticks([]), axs[0, i].set_yticks([])
            name = cls[yt] if len(cls[yt]) <= 9 else cls[yt][:8] + "."
            axs[0, i].set_title(name, fontsize=fs * 0.6)
            _overlay(axs[1, i], us["views"][i], us["maps"][i], cmap)
            _mark(axs[1, i], int(us["pred"][i]) == int(us["y"][i]), fs * 0.7)
            _overlay(axs[2, i], fm["views"][i], fm["maps"][i], cmap)
            _mark(axs[2, i], int(fm["pred"][i]) == yt, fs * 0.7)
        rows = ["Original", f"Nosso\n{_fmt_params(us['params'])}", f"{_short(fm['label'])}\n{_fmt_params(fm['params'])}"]
        for r, t in enumerate(rows):
            axs[r, 0].set_ylabel(t, fontsize=fs * (0.6 if r == 2 and max(map(len, t.split(chr(10)))) > 7 else 0.8))   # FM com nome longo: fonte menor
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.ax.tick_params(labelsize=fs * 0.7)
        fig.savefig(out / "grade.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        # grade_5x2: 5 colunas x 2 blocos de [Original | Nosso | FM], desenhada ja na escala final (12 cm)
        wcm, fs, nc = 12.0, 7.0, 5
        fig, axs = plt.subplots(6, nc, figsize=(wcm / 2.54, wcm / 2.54 * 1.32),
                                gridspec_kw=dict(wspace=0.04, hspace=0.25))
        for i in range(len(fm["names"])):
            r0, c = 3 * (i // nc), i % nc
            yt = int(fm["y"][i])
            a0, a1, a2 = axs[r0, c], axs[r0 + 1, c], axs[r0 + 2, c]
            a0.imshow(origs[i]), a0.set_xticks([]), a0.set_yticks([])
            name = cls[yt] if len(cls[yt]) <= 12 else cls[yt][:11] + "."
            a0.set_title(name, fontsize=fs, pad=2)
            _overlay(a1, us["views"][i], us["maps"][i], cmap)
            _mark(a1, int(us["pred"][i]) == int(us["y"][i]), fs * 0.85)
            _overlay(a2, fm["views"][i], fm["maps"][i], cmap)
            _mark(a2, int(fm["pred"][i]) == yt, fs * 0.85)
        for r0 in (0, 3):
            for k, t in enumerate(rows):
                axs[r0 + k, 0].set_ylabel(t, fontsize=fs)
        cb = fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=fs)
        fig.savefig(out / "grade_5x2.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        with open(out / "README.md", "w") as fh:
            fh.write(f"# Heatmaps — {base}\n\n"
                     f"- **Nosso:** {us['label']} — `{us['ckpt']}` — {int(us['params']):,} params — κ teste (SVM original) "
                     f"{float(us['kappa']):.3f} (recalculado aqui: {float(us['kappa_check']):.3f}). Sem TTA (a mesma leitura do κ).\n"
                     f"- **FM:** {fm['label']} — `{fm['model']}` — {int(fm['params']):,} params — κ teste {float(fm['kappa']):.3f} "
                     f"(benchmark, split 1, 100%). Por que este FM: {FM_WHY.get(base, 'melhor FM da base no benchmark')}. Entrada {int(fm['size'])} px, crop_pct {float(fm['crop_pct'])}, sem TTA, fp16 nos mapas.\n"
                     f"- **Split:** teste do split 1. **Imagens:** {N_IMG} da interseção dos dois conjuntos de teste "
                     f"({int(fm['n_common'])} imagens), estratificadas por classe com seed {SEED} "
                     "(≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), "
                     "escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.\n"
                     "- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, "
                     "ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; "
                     "mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling "
                     "(pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final "
                     "(AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). "
                     "ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. "
                     "Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) "
                     "na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.\n"
                     + (f"- **κ recalculado ≠ tabela:** o checkpoint veio da Jaci (torch 2.7) e aqui roda torch 2.2; "
                        "a diferença vem das features, o protocolo do SVM é o mesmo.\n"
                        if abs(float(us['kappa']) - float(us['kappa_check'])) > 0.002 else "")
                     + ("- **Parasitas:** o nosso modelo vê `pre/orig` (objeto recortado, nomes renomeados; mapeamento oficial "
                        "`splits/split1_modehb_mapping.csv`, mesmo split 1); o FM vê `raw/orig` (campo inteiro). "
                        "Por isso a coluna imagem traz os dois nomes.\n"
                        if any(str(a) != str(b) for a, b in zip(fm["names"], fm["names_ours"])) else "")
                     + f"- **Script:** `{OUT / 'make_heatmaps.py'}` (passos ours_list → fm → ours → plot).\n\n"
                     "| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |\n|---|---|---|---|---|---|\n")
            for i, n in enumerate(fm["names"]):
                yt = int(fm["y"][i])
                pu = cls[int(us["pred"][i])] if int(us["pred"][i]) < len(cls) else str(us["pred"][i])
                no = str(fm["names_ours"][i])
                fh.write(f"| {i + 1} | {n}{'' if no == str(n) else ' / ' + no} | {cls[yt]} | {pu} {'✓' if int(us['pred'][i]) == int(us['y'][i]) else '✗'} | "
                         f"{cls[int(fm['pred'][i])]} {'✓' if int(fm['pred'][i]) == yt else '✗'} | {files[i]} |\n")
        print("pronto", base, flush=True)


if __name__ == "__main__":
    {"ours_list": step_ours_list, "fm": step_fm, "ours": step_ours, "plot": step_plot}[sys.argv[1]]()
