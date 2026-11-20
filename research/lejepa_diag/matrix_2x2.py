"""Matriz 2x2 do diagnostico LeJEPA: {GAP oficial, COV oficial} x {P0 antes, P1 depois}.

Usa a biblioteca comum (extract / readouts / svm_eval) sobre os pares do
reports/lejepa_diagnostics/data/checkpoint_manifest.json (fonte (a): o MESMO encoder.pt guarda
``anchor`` = P0 e ``ssl_state`` = P1; o modulo ``encoder``/``head`` salvo e o blend WiSE).

* Dados: ``config_resolved.yaml`` de cada busca -> ``progressive_data.load_data`` (mesmo caminho
  do Evaluator: LAB [0,1] do IFT, build_test 224). Treino completo (RS 682, larvae 1118 < max_train
  4000 -> sem subamostra, igual ao fitness); metricas so na VALIDACAO. Teste nunca e lido.
* GAP = ``head(encoder(x))`` do checkpoint (EmbeddingHead; o pooling do genoma pode ser
  gap/gem/moments: e o embedding oficial). COV = ``pooling_head.CovReadout`` oficial
  (k=64, PCA ajustada so no treino, seed da busca = 42, concatenado com o head).
* SVM = ``core.svm.fit_svm`` C=100, reajustada por representacao, sem scaler. A coluna
  ``COV+scaler`` (StandardScaler) e a paridade com o fitness historico (svm_fitness scaler=standard).
* eval(), no_grad(), fp32 estrito (sem autocast/TF32), CPU. Threads BLAS do libsvm = 1
  (igual a ``svm_blas_threads`` default da busca).

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 nice -n 10 python -m research.lejepa_diag.matrix_2x2 \
        --dataset remote_sensing
    python -m research.lejepa_diag.matrix_2x2 --dataset remote_sensing --plots-only
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.lejepa_diag import analysis as A                 # noqa: E402
from research.lejepa_diag import readouts as R                 # noqa: E402
from research.lejepa_diag.extract import MemMeter, extract, load_images  # noqa: E402
from research.lejepa_diag.svm_eval import evaluate_svm, public  # noqa: E402

DIAG = REPO / "reports/lejepa_diagnostics"
MANIFEST = DIAG / "data/checkpoint_manifest.json"
METRICS = DIAG / "data/metrics.csv"
DELTAS = DIAG / "data/delta_kappa.csv"
READOUTS = ("GAP", "COV", "COV+scaler")


# ─── carga ────────────────────────────────────────────────────────────────────

def load_pair_state(path, which: str):
    """(encoder, head) de um encoder.pt MO-DEHB com os pesos ``ck[which]`` (anchor=P0, ssl_state=P1).
    Usavel como ``ckpt_loader`` (assinatura fn(path) via functools.partial)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    enc, head = copy.deepcopy(ck["encoder"]), copy.deepcopy(ck["head"])
    enc.load_state_dict(ck[which]["encoder"])
    head.load_state_dict(ck[which]["head"])
    return enc.float().eval(), head.float().eval()


def load_anchor(path):
    return load_pair_state(path, "anchor")


def load_ssl_state(path):
    return load_pair_state(path, "ssl_state")


def load_split(config_path: str, dataset: str, cache_dir: Path):
    from ga_search.modehb.run_search import load_config
    from ga_search.progressive_data import load_data
    cfg = load_config(config_path, dataset)
    d = {k: v for k, v in cfg["data"].items() if k not in ("ram_cache", "val_max_samples", "drop_last")}
    d.setdefault("seed", cfg["seed"])
    d["num_workers"] = 0
    tr, va, _sel, ident = load_data(d, cache_dir)
    Xtr, ytr, _ = load_images(tr)
    Xva, yva, _ = load_images(va)
    return (Xtr, ytr), (Xva, yva), ident, cfg


def _blas1():
    import scipy.linalg  # noqa: F401
    import sklearn.svm  # noqa: F401
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1, user_api="blas")


def short_pair(pid: str) -> str:
    _ds, search, cand = pid.split(":")
    return f"{search.replace('modehb_', '')} {cand}"


# ─── matriz ───────────────────────────────────────────────────────────────────

def run_pair(pair, data, args):
    (Xtr, ytr), (Xva, yva) = data
    rows = []
    for model, fn in (("P0", load_anchor), ("P1", load_ssl_state)):
        enc, head = fn(pair["checkpoint_file"])
        gc.collect()
        etr = extract(enc, head, (Xtr, ytr), device="cpu", batch_size=args.batch_size, layers=False)
        eva = extract(enc, head, (Xva, yva), device="cpu", batch_size=args.batch_size, layers=False)
        gc.collect()
        t0 = time.perf_counter()
        with MemMeter(torch.device("cpu")) as mm:
            ctr, cva, cr = R.cov_readout(enc, head, Xtr, Xva, "cpu", k=args.cov_k, seed=args.seed,
                                         batch_size=args.batch_size)
        cov_s = time.perf_counter() - t0
        gap_mem = max(etr.peak_mem_bytes, eva.peak_mem_bytes)
        feats = {"GAP": (etr.embedding, eva.embedding, False, etr.time_s + eva.time_s, gap_mem),
                 "COV": (ctr, cva, False, cov_s, mm.peak_bytes),
                 "COV+scaler": (ctr, cva, True, cov_s, mm.peak_bytes)}
        ref = pair["val_kappa_P0_alpha0"] if model == "P0" else pair["val_kappa_P1_alpha1"]
        for ro, (ftr, fva, sc, ts, mem) in feats.items():
            with _blas1():
                res = public(evaluate_svm(ftr, ytr, fva, yva, scaler=sc, C=args.C))
            rows.append({
                "dataset": pair["dataset"], "pair_id": pair["pair_id"], "par": short_pair(pair["pair_id"]),
                "checkpoint": model, "ckpt_key": "anchor" if model == "P0" else "ssl_state",
                "readout": ro, "head_pooling": pair["genome_globals"].get("pooling"),
                "dim": res["dim"], "kappa": res["kappa"], "macro_f1": res["macro_f1"],
                "balanced_acc": res["balanced_acc"], "t_extract_s": ts,
                "peak_mem_bytes": int(mem), "mem_kind": "cpu_rss_peak_delta",
                "feat_bytes_fp32": int((len(ftr) + len(fva)) * ftr.shape[1] * 4),
                "erank_val": A.effective_rank(fva), "n_train": res["n_train"], "n_val": res["n_val"],
                "scaler": res["scaler"], "C": res["C"], "svm_fit_s": res.get("fit_s"),
                "svm_fit_status": res.get("svm_fit_status"),
                "fitness_val_kappa_ref": ref if ro == "COV+scaler" else np.nan,
                "cov_k": args.cov_k, "cov_seed": args.seed, "split": pair["split"],
                "checkpoint_file": pair["checkpoint_file"], "md5": pair.get("md5"),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            print(f"  {pair['pair_id']} {model} {ro:11s} dim={res['dim']:6d} kappa={res['kappa']:.4f} "
                  f"f1={res['macro_f1']:.4f} bacc={res['balanced_acc']:.4f} t={ts:.1f}s"
                  + (f"  (fitness ref {ref:.4f})" if ro == "COV+scaler" else ""), flush=True)
        del enc, head, etr, eva, ctr, cva, cr
        gc.collect()
    return rows


def write_rows(rows, pair_id):
    import pandas as pd
    new = pd.DataFrame(rows)
    if METRICS.exists():
        old = pd.read_csv(METRICS)
        old = old[old["pair_id"] != pair_id]
        new = pd.concat([old, new], ignore_index=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    tmp = METRICS.with_suffix(".csv.tmp")
    new.to_csv(tmp, index=False)
    os.replace(tmp, METRICS)


def compute_deltas(df):
    import pandas as pd
    piv = df.pivot_table(index=["dataset", "pair_id", "par"], columns=["readout", "checkpoint"],
                         values="kappa").reset_index()
    out = pd.DataFrame({"dataset": piv["dataset"], "pair_id": piv["pair_id"], "par": piv["par"]})
    for ro in READOUTS:
        if (ro, "P0") in piv and (ro, "P1") in piv:
            out[f"kappa_P0_{ro}"] = piv[(ro, "P0")].values
            out[f"kappa_P1_{ro}"] = piv[(ro, "P1")].values
            out[f"dkappa_{ro}"] = piv[(ro, "P1")].values - piv[(ro, "P0")].values
    out["dkappa_COV_minus_GAP"] = out["dkappa_COV"] - out["dkappa_GAP"]
    return out


# ─── figuras ──────────────────────────────────────────────────────────────────

C_P0, C_P1 = "#8c8a85", "#6250d6"                      # antes (neutro) / depois (violeta)
C_RO = {"GAP": "#2a78d6", "COV": "#eb6834", "COV+scaler": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    return plt


def _save(fig, base: Path, table):
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    table.to_csv(base.with_suffix(".csv"), index=False)


def make_figures(dataset: str):
    import pandas as pd
    plt = _style()
    df = pd.read_csv(METRICS)
    df = df[df["dataset"] == dataset].copy()
    out = DIAG / "figures" / dataset
    pars = list(dict.fromkeys(df["par"]))
    ds_label = {"remote_sensing": "remote_sensing (split 1, 100%)", "larvae": "larvae (split 1, 100%)"}.get(dataset, dataset)
    sub = f"{ds_label} · κ de VALIDAÇÃO · SVM linear C=100 · P0 = anchor (antes do LeJEPA), P1 = ssl_state (depois)"

    # 01 κ antes/depois: um painel por readout, pares no eixo x
    t1 = df[["dataset", "par", "pair_id", "readout", "checkpoint", "kappa"]].sort_values(["readout", "par", "checkpoint"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x = np.arange(len(pars) + 1)
    w = 0.36
    for ax, ro in zip(axes, READOUTS):
        d = df[df["readout"] == ro]
        for j, (m, col) in enumerate((("P0", C_P0), ("P1", C_P1))):
            v = [float(d[(d["par"] == p) & (d["checkpoint"] == m)]["kappa"].iloc[0]) for p in pars]
            v.append(float(np.mean(v)))
            bars = ax.bar(x + (j - 0.5) * w, v, w * 0.94, color=col, label=f"{m} ({'antes' if m == 'P0' else 'depois'})")
            for xi, vi in zip(x + (j - 0.5) * w, v):
                ax.text(xi, vi + 0.005, f"{vi:.3f}", ha="center", va="bottom", fontsize=7, color=INK2, rotation=90)
        ax.set_xticks(x, pars + ["média"], rotation=25, ha="right", fontsize=8)
        ttl = {"GAP": "GAP oficial = head(encoder(x))", "COV": "COV oficial (CovReadout k=64)",
               "COV+scaler": "COV + StandardScaler (paridade fitness)"}[ro]
        ax.set_title(ttl, fontsize=10)
        ax.axvline(len(pars) - 0.5, color=GRID, lw=1)
    lo = max(0.0, df["kappa"].min() - 0.08)
    axes[0].set_ylim(lo, df["kappa"].max() + 0.07)
    axes[0].set_ylabel("κ de Cohen (validação)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.99, 0.905))
    fig.suptitle(f"01 · κ antes × depois do LeJEPA por readout — {dataset}", fontsize=12, fontweight="bold", y=0.995)
    fig.text(0.5, 0.925, sub, ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, out / "01_kappa_before_after", t1)
    plt.close(fig)

    # 02 Δκ por readout
    dl = compute_deltas(df)
    t2 = dl.copy()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    w = 0.26
    for j, ro in enumerate(READOUTS):
        v = [float(dl[dl["par"] == p][f"dkappa_{ro}"].iloc[0]) for p in pars]
        v.append(float(np.mean(v)))
        xs = x + (j - 1) * w
        ax.bar(xs, v, w * 0.92, color=C_RO[ro], label=f"Δκ {ro}")
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + (0.003 if vi >= 0 else -0.003), f"{vi:+.3f}", ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=7, color=INK2)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(len(pars) - 0.5, color=GRID, lw=1)
    ax.set_xticks(x, pars + ["média"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Δκ = κ(P1) − κ(P0)  (validação)")
    lo_, hi_ = ax.get_ylim()
    ax.set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"02 · Δκ do LeJEPA por readout — {dataset}", fontsize=12)
    fig.text(0.5, -0.02, sub + " · Δκ > 0: o readout aproveita melhor P1", ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    _save(fig, out / "02_delta_kappa", t2)
    plt.close(fig)

    # 07 κ, F1, bacc por readout (média dos pares; pontos = pares)
    t7 = df.groupby(["readout", "checkpoint"])[["kappa", "macro_f1", "balanced_acc"]].agg(["mean", "std", "min", "max"])
    t7.columns = ["_".join(c) for c in t7.columns]
    t7 = t7.reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    xr = np.arange(len(READOUTS))
    w = 0.36
    for ax, (met, lab) in zip(axes, (("kappa", "κ de Cohen"), ("macro_f1", "F1 macro"), ("balanced_acc", "acurácia balanceada"))):
        for j, (m, col) in enumerate((("P0", C_P0), ("P1", C_P1))):
            means = [df[(df["readout"] == ro) & (df["checkpoint"] == m)][met].mean() for ro in READOUTS]
            xs = xr + (j - 0.5) * w
            ax.bar(xs, means, w * 0.94, color=col, label=f"{m} ({'antes' if m == 'P0' else 'depois'})", alpha=0.9)
            for xi, ro in zip(xs, READOUTS):
                pts = df[(df["readout"] == ro) & (df["checkpoint"] == m)][met].values
                ax.scatter(np.full(len(pts), xi), pts, s=14, color="white", edgecolor=INK, linewidth=0.7, zorder=3)
        ax.set_xticks(xr, READOUTS)
        ax.set_title(lab + " (validação)", fontsize=10)
    lo = max(0.0, min(df[["kappa", "macro_f1", "balanced_acc"]].min()) - 0.06)
    hi = min(1.0, max(df[["kappa", "macro_f1", "balanced_acc"]].max()) + 0.04)
    axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel("métrica (validação)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.99, 0.9))
    fig.suptitle(f"07 · Comparação de readouts — {dataset} (barras = média de {len(pars)} pares; pontos = cada par)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.92, sub, ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    _save(fig, out / "07_readout_comparison", t7)
    plt.close(fig)

    # 08 custo: tempo, dimensão, memória (média sobre pares × {P0,P1}; pontos = cada medição)
    cost = df[df["readout"] != "COV+scaler"].copy()    # COV+scaler reusa as features do COV
    cost["peak_mem_MB"] = cost["peak_mem_bytes"] / 2 ** 20
    cost["feat_MB"] = cost["feat_bytes_fp32"] / 2 ** 20
    t8 = cost.groupby("readout")[["t_extract_s", "dim", "peak_mem_MB", "feat_MB"]].agg(["mean", "min", "max"])
    t8.columns = ["_".join(c) for c in t8.columns]
    t8 = t8.reset_index()
    ros = ["GAP", "COV"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    specs = (("t_extract_s", "tempo de extração (s, CPU 16 threads)\ntreino + validação", False),
             ("dim", "dimensão do embedding\n(escala log)", True),
             ("peak_mem_MB", "pico de memória na extração (MB)\nΔ RSS do processo*", False),
             ("feat_MB", "memória das features fp32 (MB)\ntreino + validação", False))
    for ax, (col, lab, logy) in zip(axes, specs):
        means = [cost[cost["readout"] == ro][col].mean() for ro in ros]
        ax.bar(np.arange(2), means, 0.55, color=[C_RO[r] for r in ros])
        for i, ro in enumerate(ros):
            pts = cost[cost["readout"] == ro][col].values
            ax.scatter(np.full(len(pts), i) + np.linspace(-0.12, 0.12, len(pts)), pts, s=14, color="white",
                       edgecolor=INK, linewidth=0.7, zorder=3)
            ax.text(i + 0.3, means[i], f"{means[i]:,.0f}" if means[i] >= 10 else f"{means[i]:.2f}",
                    ha="left", va="center", fontsize=8, color=INK)
        ax.set_xticks(np.arange(2), ["GAP (head)", "COV (k=64)"])
        ax.set_xlim(-0.5, 1.85)
        if logy:
            ax.set_yscale("log")
        ax.set_title(lab, fontsize=10)
    fig.suptitle(f"08 · Custo computacional por readout — {dataset} (média de {len(pars)} pares × P0/P1; pontos = medições)",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.9, f"{ds_label} · CPU, fp32, lote 8 · COV inclui ajuste da PCA no treino + embedding · "
             "COV+scaler usa as mesmas features do COV · *Δ RSS subestima quando o alocador reaproveita memória já reservada",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    _save(fig, out / "08_computational_cost", t8)
    plt.close(fig)
    return dl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cov-k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42, help="seed do CovReadout (= seed da busca)")
    ap.add_argument("--C", type=float, default=100.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--pair-filter", default=None)
    ap.add_argument("--max-train", type=int, default=None, help="so smoke")
    ap.add_argument("--max-val", type=int, default=None, help="so smoke")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="smoke: nao grava metrics.csv")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    if not args.plots_only:
        man = json.loads(Path(args.manifest).read_text())
        pairs = [p for p in man["pairs"] if p["dataset"] == args.dataset
                 and p["source_type"] == "modehb_candidate_anchor_vs_ssl_state"
                 and (not args.pair_filter or args.pair_filter in p["pair_id"])]
        cache = {}
        for pair in pairs:
            if pair["config"] not in cache:
                cache.clear()
                t0 = time.perf_counter()
                tr, va, ident, _cfg = load_split(pair["config"], args.dataset,
                                                 Path(os.environ.get("LEJEPA_DIAG_CACHE", "/tmp/lejepa_diag_data")))
                print(f"[data] {pair['config']}: train {tuple(tr[0].shape)} val {tuple(va[0].shape)} "
                      f"split_sha {ident['sha256'][:12]} ({time.perf_counter() - t0:.0f}s)", flush=True)
                cache[pair["config"]] = (tr, va)
            tr, va = cache[pair["config"]]
            if args.max_train:
                tr = (tr[0][:args.max_train], tr[1][:args.max_train])
            if args.max_val:
                va = (va[0][:args.max_val], va[1][:args.max_val])
            t0 = time.perf_counter()
            rows = run_pair(pair, (tr, va), args)
            if not args.no_write:
                write_rows(rows, pair["pair_id"])
            print(f"[ok] {pair['pair_id']} em {time.perf_counter() - t0:.0f}s", flush=True)
    if args.no_write:
        return
    dl = make_figures(args.dataset)
    import pandas as pd
    allm = pd.read_csv(METRICS)
    compute_deltas(allm).to_csv(DELTAS, index=False)
    print(dl.to_string())


if __name__ == "__main__":
    main()
