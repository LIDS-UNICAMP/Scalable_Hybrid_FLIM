"""Analise de picos/oscilacao/convergencia nas curvas W&B (AE local + SSL) dos melhores modelos do GA.

Criterio (por serie x, na ordem do treino):
  x0 = 1o valor; janela inicial = primeiros max(3, 10%) pontos (excluindo x0);
  pico_ini/x0 = max(janela)/x0  (>1 => a perda SUBIU logo apos a init);  pico/mediana = max(janela)/mediana(x)
  pico grande = pico_ini/x0 > 2  (dobrou a partir do ponto de partida);  moderado 1.2-2;  ausente <= 1.2
  oscilacao: cv_diff = std(diff(x))/|mediana(x)|;  reversoes = fracao de trocas de sinal de diff(x)
  convergencia: inclinacao relativa do ultimo 30% (regressao linear, por ponto, / |mediana do trecho|);
     final/min (ultimo valor / minimo global);  converge = final <= 1.10*min OU inclinacao <= 0;  diverge = final > 1.5*min e inclinacao > 0
"""
import json, os, pickle, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
BASES = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
sel = json.load(open(f"{D}/models_selected.json"))
extra = json.load(open(f"{D}/models_extra.json"))
_cache = {}
def load(rid):
    if rid not in _cache: _cache[rid] = pickle.load(open(f"{D}/cache/{rid}.pkl", "rb"))
    return _cache[rid]

def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    n = len(x)
    if n < 3: return None
    x0 = x[0]; w = max(3, int(math.ceil(0.1 * n))); win = x[1:1 + w]
    pk = win.max(); med = np.median(x)
    d = np.diff(x); sgn = np.sign(d[d != 0])
    rev = float((sgn[1:] != sgn[:-1]).mean()) if len(sgn) > 1 else 0.0
    t = x[int(0.7 * n):]
    slope = np.polyfit(np.arange(len(t)), t, 1)[0] / (abs(np.median(t)) + 1e-12) if len(t) >= 3 else float("nan")
    fin, mn = x[-1], x.min()
    conv = (fin <= 1.10 * mn) or (slope <= 0)
    div = (fin > 1.5 * mn) and slope > 0
    return dict(n=n, x0=x0, peak_init=pk / x0 if x0 else np.nan, peak_med=pk / med if med else np.nan,
                peak_idx=int(1 + np.argmax(win)), cv_diff=float(d.std() / (abs(med) + 1e-12)), rev=rev,
                slope=float(slope), fin_min=fin / mn if mn > 0 else np.nan, fin_x0=fin / x0 if x0 else np.nan,
                conv=bool(conv), div=bool(div))

def ae_blocks(h, E):
    """segmenta ae/recon_loss por bloco (E epocas por bloco); devolve lista de dicts com loss, gnorm bloco, gnorm decoder, junction."""
    h = h.sort_values("_step")
    gcols = [c for c in h.columns if c.startswith("ae/grad_norm/new_block")]
    blocks, cur, done = [], None, 0
    for _, r in h.iterrows():
        if "ae/epoch" in h and pd.notna(r.get("ae/epoch")):
            if cur is not None: cur["ep_end"].append(len(cur["loss"]))
            done += 1
            if E and done >= E:
                blocks.append(cur); cur, done = None, 0
        if pd.notna(r.get("ae/recon_loss", np.nan)):
            if cur is None: cur = dict(loss=[], g=[], gd=[], ep_end=[])
            cur["loss"].append(r["ae/recon_loss"])
            gv = [r[c] for c in gcols if pd.notna(r[c])]
            cur["g"].append(math.sqrt(sum(v * v for v in gv)) if gv else np.nan)
            cur["gd"].append(r.get("ae/grad_norm/decoder", np.nan))
    if cur is not None: blocks.append(cur)
    return [b for b in blocks if b and b["loss"]]

def ssl_series(h):
    h = h.sort_values("_step")
    out = {}
    for k in ["loss_total", "loss_inv", "loss_sigreg"]:
        if f"ssl/{k}" in h: out[k] = h[f"ssl/{k}"].dropna().tolist()
        if f"epoch/ssl/{k}" in h: out["ep_" + k] = h[f"epoch/ssl/{k}"].dropna().tolist()
    gc = [c for c in h.columns if c.startswith("grad_norm/encoder/")]
    if gc:
        g = h[gc].dropna(how="all")
        out["gnorm"] = np.sqrt((g.fillna(0) ** 2).sum(1)).tolist()
    out["method"] = h["ssl/method"].dropna().iloc[0] if "ssl/method" in h and h["ssl/method"].notna().any() else None
    return out

def fmt(s, key="peak_init"):
    return "-" if s is None else f"{s[key]:.2f}"

def flag(r):
    if r is None or not np.isfinite(r): return "-"
    return "GRANDE" if r > 2 else ("moderado" if r > 1.2 else "nao")

rows, figdata = [], {}
for m in sel:
    cdir = os.path.dirname(os.path.dirname(m["path"]))
    chain = list({c["fid"]: c for c in extra["chains"].get(cdir, []) if c["fid"] <= m["fid"]}.values())
    chain.sort(key=lambda c: c["fid"])
    if not chain: continue
    first = load(chain[0]["wandb"]); cfg = first["config"]
    r0 = json.load(open(os.path.join(cdir, f"f{int(chain[0]['fid'])}", "result.json")))
    E = r0.get("ae_epochs_per_layer") or 0
    origins = sorted(set(L.get("prov", {}).get("origin", "?") for L in r0.get("layers", [])))
    blocks = ae_blocks(first["history"], E) if E else []
    # AE: por bloco
    ae_stats = []
    for bi, b in enumerate(blocks):
        s = stats(b["loss"]); sg = stats(b["g"])
        nwarm = int(round((0.5 if cfg.get("ae_schedule") == "decoder_first" else 0.0) * E))
        junc = np.nan
        if nwarm and len(b["ep_end"]) >= nwarm:
            j = b["ep_end"][nwarm - 1]; L = b["loss"]
            if 3 <= j < len(L) - 1: junc = max(L[j:j + 3]) / np.mean(L[j - 3:j])
        L = np.asarray(b["loss"]); rec = np.nan
        if s and s["peak_init"] > 1.2:
            p = s["peak_idx"]; after = np.where(L[p:] <= L[0])[0]; rec = int(after[0]) if len(after) else -1
        ae_stats.append(dict(block=bi, s=s, sg=sg, junc=junc, rec=rec, x0=L[0], frozen0=nwarm > 0))
    # SSL: concatena a cadeia de fidelidades (step-level) e por epoca
    ssl_cat = {"loss_total": [], "ep_loss_total": [], "loss_inv": [], "loss_sigreg": [], "gnorm": [], "bounds": [], "ep_bounds": []}
    method = None; promo_jumps = []
    for c in chain:
        ss = ssl_series(load(c["wandb"])["history"]); method = method or ss.get("method")
        if ssl_cat["ep_loss_total"] and ss.get("ep_loss_total"):
            promo_jumps.append(ss["ep_loss_total"][0] / ssl_cat["ep_loss_total"][-1])
        ssl_cat["bounds"].append(len(ssl_cat["loss_total"])); ssl_cat["ep_bounds"].append(len(ssl_cat["ep_loss_total"]))
        for k in ["loss_total", "ep_loss_total", "loss_inv", "loss_sigreg", "gnorm"]:
            ssl_cat[k] += ss.get(k, [])
    s_first = ssl_series(first["history"])
    sst = stats(s_first.get("loss_total", []))           # pico no inicio do SSL (run f5, step-level)
    sst_all = stats(ssl_cat["ep_loss_total"]) if len(ssl_cat["ep_loss_total"]) >= 3 else stats(ssl_cat["loss_total"])
    s_inv = stats(s_first.get("loss_inv", [])); s_sig = stats(s_first.get("loss_sigreg", []))
    s_g = stats(s_first.get("gnorm", []))
    kap = [(c["fid"], c["kappa"]) for c in chain]; er = [(c["fid"], c["erank"]) for c in chain]
    row = dict(dataset=m["dataset"], search=m["search"].replace("modehb_", ""), cand=m["cand"], fid=m["fid"], stratum=m["stratum"],
               params=m["params"], kappa=m["kappa"], wandb=m["wandb"], why=m["why"], table_id=m.get("table_id"),
               init=cfg.get("init"), origins="+".join(origins), protect=cfg.get("protect"), ae_sched=cfg.get("ae_schedule"),
               ae_target=cfg.get("ae_target"), ssl_method=method, E=E, n_blocks=len(blocks),
               ae_peak_max=max([a["s"]["peak_init"] for a in ae_stats if a["s"]] or [np.nan]),
               ae_peakmed_max=max([a["s"]["peak_med"] for a in ae_stats if a["s"]] or [np.nan]),
               ae_junc_max=np.nanmax([a["junc"] for a in ae_stats]) if ae_stats and np.isfinite([a["junc"] for a in ae_stats]).any() else np.nan,
               ae_n_big=sum(1 for a in ae_stats if a["s"] and a["s"]["peak_init"] > 2), ae_nb=len(ae_stats),
               ae_big_blocks=",".join(str(a["block"]) for a in ae_stats if a["s"] and a["s"]["peak_init"] > 2),
               ae_peak_step=next((a["s"]["peak_idx"] for a in sorted(ae_stats, key=lambda a: -(a["s"]["peak_init"] if a["s"] else 0)) if a["s"]), None),
               ae_rec_steps=max([a["rec"] for a in ae_stats if np.isfinite(a["rec"])] or [np.nan]),
               ae_x0_of_peakblock=next((a["x0"] for a in sorted(ae_stats, key=lambda a: -(a["s"]["peak_init"] if a["s"] else 0)) if a["s"]), np.nan),
               ae_steps_per_block=int(np.median([len(b["loss"]) for b in blocks])) if blocks else 0,
               ae_gpeak_max=max([a["sg"]["peak_init"] for a in ae_stats if a["sg"]] or [np.nan]),
               ae_cv=np.nanmean([a["s"]["cv_diff"] for a in ae_stats if a["s"]] or [np.nan]),
               ae_rev=np.nanmean([a["s"]["rev"] for a in ae_stats if a["s"]] or [np.nan]),
               ae_finx0=np.nanmean([a["s"]["fin_x0"] for a in ae_stats if a["s"]] or [np.nan]),
               ae_conv=all(a["s"]["conv"] for a in ae_stats if a["s"]) if ae_stats else None,
               ae_div=any(a["s"]["div"] for a in ae_stats if a["s"]) if ae_stats else None,
               ssl_peak=sst["peak_init"] if sst else np.nan, ssl_peakmed=sst["peak_med"] if sst else np.nan,
               ssl_peak_idx=sst["peak_idx"] if sst else None, ssl_n=sst["n"] if sst else 0,
               inv_peak=s_inv["peak_init"] if s_inv else np.nan, sig_peak=s_sig["peak_init"] if s_sig else np.nan,
               gn_peak=s_g["peak_init"] if s_g else np.nan,
               ssl_cv=sst_all["cv_diff"] if sst_all else np.nan, ssl_rev=sst_all["rev"] if sst_all else np.nan,
               ssl_slope=sst_all["slope"] if sst_all else np.nan, ssl_finmin=sst_all["fin_min"] if sst_all else np.nan,
               ssl_finx0=sst_all["fin_x0"] if sst_all else np.nan,
               ssl_conv=sst_all["conv"] if sst_all else None, ssl_div=sst_all["div"] if sst_all else None,
               ssl_n_ep=len(ssl_cat["ep_loss_total"]), promo_jump_max=max(promo_jumps) if promo_jumps else np.nan,
               kappa_chain=" > ".join(f"f{int(f)}:{k:.3f}" for f, k in kap if k is not None),
               erank_chain=" > ".join(f"{e:.1f}" for f, e in er if e is not None))
    rows.append(row)
    figdata[(m["dataset"], m["cand"], m["search"], m["fid"])] = dict(row=row, blocks=blocks, ssl=ssl_cat, kap=kap)

df = pd.DataFrame(rows)
df.to_csv(f"{D}/metricas_por_modelo.csv", index=False)

# ---- controle random_he x spifil (RS, buscas v3/v3_old, f5)
ctrl = []
for c in extra["control"]:
    d = load(c["wandb"]); cfg = d["config"]
    r0 = json.load(open(c["path"])); E = r0.get("ae_epochs_per_layer") or 0
    bl = ae_blocks(d["history"], E) if E else []
    st = [stats(b["loss"]) for b in bl]; st = [s for s in st if s]
    ss = stats(ssl_series(d["history"]).get("loss_total", []))
    sse = stats(ssl_series(d["history"]).get("ep_loss_total", []))
    ctrl.append(dict(origin=c["origin"], kappa=c["kappa"], ae_peak=max([s["peak_init"] for s in st] or [np.nan]),
                     ae_finx0=np.mean([s["fin_x0"] for s in st]) if st else np.nan,
                     ae_cv=np.mean([s["cv_diff"] for s in st]) if st else np.nan,
                     ssl_peak=ss["peak_init"] if ss else np.nan, ssl_n=ss["n"] if ss else 0,
                     ssl_finx0=sse["fin_x0"] if sse else np.nan, method=ssl_series(d["history"]).get("method")))
cdf = pd.DataFrame(ctrl); cdf.to_csv(f"{D}/controle_random_vs_spifil_rs.csv", index=False)

# ---- crescimento (sem curva: so inicio/fim do AE em result.json)
import glob
grow = []
for f in sorted(glob.glob(f"{ROOT}/research/spifil_b0b4/growth/**/result.json", recursive=True)):
    d = json.load(open(f))
    for a in d.get("ae_logs") or []:
        grow.append(dict(node=f.split("growth/")[1].replace("/result.json", ""), block=a["block"], init=d.get("init_method"),
                         params=d.get("params"), kappa=d.get("kappa_val_orig") or d.get("quality"), ini=a["initial_recon_loss"],
                         fin=a["final_recon_loss"], ratio=a["recon_ratio"], stab_rel=a["stability"] / a["final_recon_loss"] if a["final_recon_loss"] else np.nan,
                         ssl_final=d.get("ssl_final_loss"), skipped=a.get("skipped_steps")))
gdf = pd.DataFrame(grow); gdf.to_csv(f"{D}/crescimento_ae_resumo.csv", index=False)

# ---- figuras: 3 melhores candidatos distintos por base
os.makedirs(f"{D}/figures", exist_ok=True)
for ds in BASES:
    keys = sorted([k for k in figdata if k[0] == ds], key=lambda k: -figdata[k]["row"]["kappa"])
    seen, top = set(), []
    for k in keys:
        if (k[1], k[2]) in seen: continue
        seen.add((k[1], k[2])); top.append(k)
        if len(top) == 3: break
    if not top: continue
    fig, ax = plt.subplots(len(top), 3, figsize=(16, 3.6 * len(top)), squeeze=False)
    for i, k in enumerate(top):
        fd = figdata[k]; r = fd["row"]
        a = ax[i, 0]
        for bi, b in enumerate(fd["blocks"]):
            a.plot(np.array(b["loss"]) / b["loss"][0], lw=1, label=f"bloco {bi}")
            for e in b["ep_end"][:-1]: a.axvline(e, color="gray", lw=0.4, ls=":")
        a.axhline(1, color="k", lw=0.6, ls="--"); a.axhline(2, color="r", lw=0.6, ls="--")
        a.set_yscale("log"); a.set_title(f"AE local (recon / recon inicial) — {r['ae_target']}, {r['ae_sched']}", fontsize=9)
        a.set_xlabel("passo"); a.legend(fontsize=7, ncol=2)
        if not fd["blocks"]: a.text(0.5, 0.5, "sem AE logado", transform=a.transAxes, ha="center")
        a = ax[i, 1]; s = fd["ssl"]
        x = np.array(s["loss_total"])
        if len(x):
            a.plot(x / x[0], color="C0", lw=1.2, label="loss total")
            for kk, col in [("loss_inv", "C1"), ("loss_sigreg", "C2")]:
                if len(s[kk]) == len(x) and len(x): a.plot(np.array(s[kk]) / (s[kk][0] or 1), color=col, lw=0.9, label=kk.replace("loss_", ""))
            for bd in s["bounds"][1:]: a.axvline(bd, color="purple", lw=0.8, ls="--")
            a.axhline(1, color="k", lw=0.6, ls="--"); a.axhline(2, color="r", lw=0.6, ls="--")
        a.set_yscale("log"); a.set_title(f"SSL {r['ssl_method']} (normalizado pelo 1o valor); | = promocao de fidelidade", fontsize=9)
        a.set_xlabel("ponto logado (cadeia f5→f%d)" % r["fid"]); a.legend(fontsize=7)
        a = ax[i, 2]
        if len(s["gnorm"]): a.plot(np.array(s["gnorm"]) / s["gnorm"][0], color="C3", lw=1, label="SSL ||g|| encoder")
        for bi, b in enumerate(fd["blocks"]):
            g = np.array(b["g"], float)
            if np.isfinite(g).any() and g[0] > 0: a.plot(np.arange(len(g)) / max(1, len(g)) * max(1, len(s["gnorm"])), g / g[0], lw=0.6, alpha=0.6, label=f"AE ||g|| bloco {bi}")
        a.axhline(1, color="k", lw=0.6, ls="--"); a.axhline(2, color="r", lw=0.6, ls="--"); a.set_yscale("log")
        a2 = a.twinx(); kp = [kk for kk in fd["kap"] if kk[1] is not None]
        if kp:
            xs = [s["bounds"][j] + (s["bounds"][j + 1] if j + 1 < len(s["bounds"]) else len(s["gnorm"]) or 1) - s["bounds"][j] for j in range(len(kp))]
            a2.plot(xs, [kk[1] for kk in kp], "ko-", ms=4, label="κ val (fim de cada fid)"); a2.set_ylabel("κ val")
        a.set_title("normas de gradiente (rel. ao inicio) e κ val", fontsize=9); a.legend(fontsize=6, loc="upper left")
        ax[i, 0].set_ylabel(f"{r['search']} {r['cand']} f{r['fid']}\n{r['stratum']} {r['params']/1e6:.2f}M κ={r['kappa']:.3f}", fontsize=8)
    fig.suptitle(f"{ds}: curvas de treino dos 3 melhores (por κ de validacao do GA) — linhas: 1×(init) e 2×(pico grande)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{D}/figures/curvas_{ds}.png", dpi=110); plt.close(fig)

# controle: figura de distribuicao
if len(cdf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for j, col in enumerate(["ae_peak", "ssl_peak"]):
        data = [cdf[cdf.origin == o][col].dropna().values for o in ["spifil", "random_he"]]
        ax[j].boxplot(data, labels=["SPiFiL", "random_he"]); ax[j].axhline(2, color="r", ls="--", lw=0.7); ax[j].axhline(1, color="k", ls="--", lw=0.6)
        ax[j].set_title({"ae_peak": "AE: max pico inicial / valor inicial (pior bloco)", "ssl_peak": "SSL: pico inicial / valor inicial"}[col], fontsize=9)
    fig.suptitle("Controle remote_sensing (buscas v3/v3_old, f5): init SPiFiL x aleatoria He", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{D}/figures/controle_spifil_vs_random_rs.png", dpi=110); plt.close(fig)
print(df[["dataset", "search", "cand", "fid", "kappa", "init", "origins", "ae_target", "ae_sched", "ssl_method", "ae_peak_max", "ae_junc_max", "ae_finx0", "ae_conv", "ssl_peak", "ssl_n", "ssl_finx0", "ssl_conv", "ssl_div", "promo_jump_max"]].round(3).to_string())
cdf["big"] = cdf.ae_peak > 2
print(cdf.groupby("origin")[["big"]].mean())
print(cdf.groupby("origin")[["ae_peak", "ae_finx0", "ae_cv", "ssl_peak", "ssl_finx0", "kappa"]].median().round(3), cdf.groupby("origin").size())
print(gdf.round(3).to_string())
