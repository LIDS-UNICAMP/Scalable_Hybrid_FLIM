"""plot_splits_classes.py -- para cada base: como ficam treino / validação / teste em cada % do treino (1..100)
e a frequência de cada classe. Usa EXATAMENTE a mesma regra do --train-pct do eval_backbone_head.py
(load_splits): descritor splits_incremental ∩ treino da busca -> casa sem prefixo -> amostra estratificada (ADAPTAÇÃO).
Saída: <out>/<base>/ga_splits_classes.png + splits_classes.tex + splits_classes.md (uma figura por base)."""
import argparse, json, sys, types
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

H = Path(__file__).resolve().parent; R = H.parents[2]; A = R / "artifacts"
sys.path.insert(0, str(R)); sys.path.insert(0, str(H))
import eval_backbone_head as ebh  # noqa: E402

PCTS = [1, 5, 10, 25, 50, 75, 100]
CK = {  # um encoder de cada base só para achar a config de dados da busca (split1)
    "flowers": A / "modehb_hybrid/flowers/state/seed42/candidates/c00064/f5/encoder.pt",
    "remote_sensing": A / "modehb_lemonade_seeded/remote_sensing/state/seed42/candidates/c00002/f200/encoder.pt",
    "ham10000": A / "modehb_hybrid_C01_2049/ham10000/state/seed42/candidates/c00001/f100/encoder.pt",
    "aid": A / "modehb_hybrid/aid/state/seed42/candidates/c00091/f45/encoder.pt",
    "coconut_trees": A / "modehb_hybrid/coconut_trees/state/seed42/candidates/c00015/f5/encoder.pt",
    "eggs": A / "modehb_lemonade/eggs/state/seed42/candidates/c00032/f15/encoder.pt",
    "larvae": A / "modehb_lemonade/larvae/state/seed42/candidates/c00078/f15/encoder.pt",
    "protozoan": A / "modehb_hybrid_C01_2049/protozoan/state/seed42/candidates/c00001/f100/encoder.pt",
    "eggs_raw": A / "modehb_hybrid/eggs_raw/state/seed42/candidates/c00027/f5/encoder.pt",
    "larvae_raw": A / "modehb_hybrid/larvae_raw/state/seed42/candidates/c00023/f15/encoder.pt",
    "protozoan_raw": A / "modehb_hybrid/protozoan_raw/state/seed42/candidates/c00001/f100/encoder.pt",
}
C_TR, C_VA, C_TE = "#2166ac", "#f4a582", "#b2182b"


def counts(base):
    import yaml
    ck = CK[base]
    cfg = ebh._find_up(ck, ["growth_state.json", "config_resolved.yaml"])
    d = (json.loads(cfg.read_text())["config_full"]["run"]["data"] if cfg.name == "growth_state.json"
         else yaml.safe_load(cfg.read_text())["data"])
    names = {int(k): v for k, v in (d["dataset_config"].get("classes") or {}).items()}
    out, adapt = {}, False
    for p in [1, 5, 10, 25, 50, 75, 100]:
        a = types.SimpleNamespace(train_pct=(p if p != 100 else None), data_cache=None, max_images=None)
        import io, contextlib
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                sp = ebh.load_splits("modehb", ck, a)
        except FileNotFoundError:
            continue  # sem descritor dessa % (ex.: ham10000 10%)
        adapt |= "ADAPTAÇÃO" in buf.getvalue()
        sp = sp[0] if isinstance(sp, tuple) else sp
        out[p] = {s: Counter(_lab(ds, i) for i in ds.keep) for s, ds in sp.items() if s in ("train", "validation", "test")}
    return out, names, adapt


def _lab(ds, i):
    b = ds.base
    while not hasattr(b, "samples"):
        b = b.base
    return b.samples[i].label


def plot(base, out, names, adapt, dst):
    global PCTS
    PCTS = [p for p in [1, 5, 10, 25, 50, 75, 100] if p in out]
    labs = sorted(set().union(*[out[100][s].keys() for s in out[100]]))
    nm = lambda l: str(names.get(l, l))
    tr = {p: sum(out[p]["train"].values()) for p in PCTS}
    va, te = sum(out[100]["validation"].values()), sum(out[100]["test"].values())
    many = True  # layout único: barras empilhadas ordenadas + tabela em linha própria (sem sobreposição)
    fig = plt.figure(figsize=(15, 6.6 + 0.3 * (len(labs) + 2)))
    gs = fig.add_gridspec(2 if many else 1, 2, width_ratios=[1, 1.35],
                          height_ratios=[5.2, 0.3 * (len(labs) + 2)])
    # (a) totais por %
    ax = fig.add_subplot(gs[0, 0]); x = np.arange(len(PCTS)); w = 0.27
    for k, (lab, vals, c) in enumerate((("treino", [tr[p] for p in PCTS], C_TR),
                                        (f"validação (fixa: {va})", [va] * len(PCTS), C_VA),
                                        (f"teste (fixo: {te})", [te] * len(PCTS), C_TE))):
        b = ax.bar(x + (k - 1) * w, vals, w, color=c, label=lab, edgecolor="white")
        if k == 0: ax.bar_label(b, fontsize=8, padding=1)
    ax.set_xticks(x, [f"{p}%" for p in PCTS]); ax.set_ylabel("nº de imagens"); ax.set_xlabel("% do treino usado na SVM")
    ax.set_title("(a) imagens por conjunto em cada %", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False); ax.margins(y=0.15)
    # (b) por classe
    ttl = "(b) imagens por classe (treino 100% / validação / teste)"
    trc = [out[100]["train"][l] for l in labs]; vac = [out[100]["validation"][l] for l in labs]; tec = [out[100]["test"][l] for l in labs]
    imb = max(trc) / max(1, min(trc))
    if not many:
        ax2 = fig.add_subplot(gs[0, 1]); y = np.arange(len(labs)); h = 0.27
        for k, (lab, v, c) in enumerate((("treino", trc, C_TR), ("validação", vac, C_VA), ("teste", tec, C_TE))):
            b = ax2.barh(y + (k - 1) * h, v, h, color=c, label=lab, edgecolor="white"); ax2.bar_label(b, fontsize=7.5, padding=2)
        ax2.set_yticks(y, [nm(l) for l in labs]); ax2.invert_yaxis(); ax2.set_xlabel("nº de imagens")
        ax2.spines[["top", "right"]].set_visible(False); ax2.margins(x=0.12)
        ax2.set_title(ttl + f"\ndesbalanceamento no treino (maior/menor classe) = {imb:.1f}×", loc="left", fontsize=11)
        ax2.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        axt = fig.add_subplot(gs[0, 0]) if False else None
    else:
        ax2 = fig.add_subplot(gs[0, 1]); o = np.argsort(trc)[::-1]
        ax2.bar(range(len(labs)), [trc[i] for i in o], color=C_TR, label="treino")
        ax2.bar(range(len(labs)), [vac[i] for i in o], bottom=[trc[i] for i in o], color=C_VA, label="validação")
        ax2.bar(range(len(labs)), [tec[i] for i in o], bottom=[trc[i] + vac[i] for i in o], color=C_TE, label="teste")
        ax2.set_xticks(range(len(labs)), [nm(labs[i]) for i in o], rotation=90, fontsize=7)
        ax2.set_ylabel("nº de imagens"); ax2.spines[["top", "right"]].set_visible(False); ax2.legend(frameon=False, fontsize=9)
        ax2.set_title(ttl + f" — ordenado\ndesbalanceamento no treino (maior/menor classe) = {imb:.1f}×", loc="left", fontsize=11)
    # (c) tabela classe × % (treino) + val + teste
    axt = fig.add_subplot(gs[1, :]) if many else None
    rows = [[nm(l)] + [str(out[p]["train"][l]) for p in PCTS] + [str(out[100]["validation"][l]), str(out[100]["test"][l]),
            f"{100 * (trc[i]) / max(1, sum(trc)):.1f}%"] for i, l in enumerate(labs)]
    rows.append(["TOTAL"] + [str(tr[p]) for p in PCTS] + [str(va), str(te), "100%"])
    cols = ["classe"] + [f"tr {p}%" for p in PCTS] + ["val", "teste", "% do treino"]
    if many:
        axt.axis("off"); t = axt.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.15)
        _shade(t, rows, trc)
    else:  # tabela embaixo da figura, também
        fig.subplots_adjust(bottom=0.36)
        axt = fig.add_axes([0.06, 0.02, 0.9, 0.24]); axt.axis("off")
        t = axt.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.2); _shade(t, rows, trc)
    nota = ("val e teste são fixos (os da busca, split1); só o treino da SVM muda com a %."
            + (" ADAPTAÇÃO: o descritor splits_incremental não casa com o split da busca → amostra estratificada"
               " semeada de p% do treino de cada classe (mín. 1 por classe)." if adapt else ""))
    fig.suptitle(f"{base} — treino / validação / teste por % e por classe (split1)", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925 if many else 0.915, nota, ha="center", fontsize=8.5, color="#444")
    if not many:
        fig.tight_layout(rect=[0, 0.28, 1, 0.90])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.91])
    dst.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst / "ga_splits_classes.png", dpi=150); plt.close(fig)
    # LaTeX + MD
    tex = ["\\begin{table}[h]\\centering\\small", f"\\caption{{{base}: imagens por classe e por \\% do treino (split1)}}",
           "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}\\toprule", " & ".join(c.replace("%", "\\%") for c in cols) + " \\\\\\midrule"]
    tex += [" & ".join(c.replace("%", "\\%").replace("_", "\\_") for c in r) + " \\\\" for r in rows[:-1]]
    tex += ["\\midrule " + " & ".join(c.replace("%", "\\%") for c in rows[-1]) + " \\\\\\bottomrule", "\\end{tabular}\\end{table}"]
    (dst / "splits_classes.tex").write_text("\n".join(tex) + "\n")
    md = [f"# {base} — treino/val/teste por % e por classe (split1)", "", nota, "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows]
    (dst / "splits_classes.md").write_text("\n".join(md) + "\n")
    return imb


def _shade(t, rows, trc):
    mx = max(trc) or 1
    for i in range(1, len(rows)):
        for j in range(1, len(PCTS) + 1):
            if i <= len(trc):
                v = trc[i - 1] / mx
                t[i, j].set_facecolor((1 - 0.55 * v, 1 - 0.35 * v, 1))
        if i == len(rows):
            for j in range(len(rows[0])):
                t[i, j].set_text_props(fontweight="bold")
    for j in range(len(rows[0])):
        t[0, j].set_text_props(fontweight="bold"); t[0, j].set_facecolor("#e8e8e8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(H.parent / "visualize_GA_copia"))
    ap.add_argument("bases", nargs="*")
    a = ap.parse_args()
    for b in (a.bases or list(CK)):
        try:
            o, n, ad = counts(b)
            imb = plot(b, o, n, ad, Path(a.out) / b)
            print(f"{b}: ok  treino100={sum(o[100]['train'].values())} val={sum(o[100]['validation'].values())} "
                  f"teste={sum(o[100]['test'].values())} classes={len(o[100]['train'])} desbal={imb:.1f}x adapt={ad}", flush=True)
        except Exception as e:
            print(f"{b}: ERRO {type(e).__name__}: {e}", flush=True)
