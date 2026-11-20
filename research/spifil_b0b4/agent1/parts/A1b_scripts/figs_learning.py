"""A1b — figuras C (aprendizagem) e quantitativas 2/4/7 a partir de A1b_data.csv, A1b_tests_all.csv,
A1b_orig_cov_tests.csv e A1b_seed_curves_long.csv. Paleta: referência validada do skill dataviz."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

P = "/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/agent1/parts"
F = f"{P}/figs"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "lines.linewidth": 2})
STRATA = ["tiny", "small", "medium", "huge"]

d = pd.read_csv(f"{P}/A1b_data.csv")
d = d[d.status == "ok"]
c100 = d[(d.fitness_sig == "C100_kw1") & d.wise_alpha.notna()].copy()

# ---- F1: frequência de alpha por dataset (C100)
c100["alpha_bin"] = c100.wise_alpha.map(lambda a: "0" if a == 0 else ("1" if a == 1 else "0<α<1"))
ct = pd.crosstab(c100.dataset, c100.alpha_bin, normalize="index")[["0", "0<α<1", "1"]]
n = c100.groupby("dataset").size()
fig, ax = plt.subplots(figsize=(7, 3.6))
left = np.zeros(len(ct))
for i, col in enumerate(ct.columns):
    ax.barh(range(len(ct)), ct[col], left=left, color=S[i], edgecolor=SURF, linewidth=2, label=f"α = {col}")
    left += ct[col].values
ax.set_yticks(range(len(ct)), [f"{k} (n={n[k]})" for k in ct.index])
ax.set_xlabel("fração dos jobs (fitness C=100, kw=1)")
ax.set_title("WiSE: α escolhido na validação (COV) — α=0 é o SPiFiL(+AE) sem SSL", loc="left")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35), frameon=False)
fig.tight_layout(); fig.savefig(f"{F}/A1b_F1_wise_alpha_freq.png", dpi=150); plt.close(fig)

# ---- F2: kappa(alpha) - kappa(0) por dataset
fig, ax = plt.subplots(figsize=(7.5, 3.8))
ds = sorted(c100.dataset.unique())
alphas = ["0.25", "0.5", "0.75", "1.0"]
w = 0.18
for j, a in enumerate(alphas):
    vals = [(c100[c100.dataset == k][f"wise_k_{a}"] - c100[c100.dataset == k]["wise_k_0.0"]).dropna() for k in ds]
    pos = np.arange(len(ds)) + (j - 1.5) * w
    ax.scatter(pos, [v.median() for v in vals], color=S[j], s=30, zorder=3, label=f"α={a}")
    ax.vlines(pos, [v.quantile(.25) for v in vals], [v.quantile(.75) for v in vals], color=S[j], lw=2)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(ds)), ds, rotation=25, ha="right")
ax.set_ylabel("κ_val(α) − κ_val(α=0)  (mediana, IQR)")
ax.set_title("Efeito do LeJEPA+SIGReg sobre os pesos SPiFiL, por blend WiSE", loc="left")
ax.legend(ncol=4, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{F}/A1b_F2_kappa_by_alpha.png", dpi=150); plt.close(fig)

# ---- F3: linhagens de calibração: ganho do SSL puro (alpha=1) e do WiSE vs fidelidade
cal = d[(d.op_type == "calibration") & (d.fitness_sig == "C100_kw1")].copy()
cal["ssl_gain"] = cal["wise_k_1.0"] - cal["wise_k_0.0"]
cal["wise_gain"] = cal["fitness_kappa"] - cal["wise_k_0.0"]
fids = [5, 15, 45, 100, 200]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, col, ttl in ((axs[0], "ssl_gain", "SSL puro: κ(α=1) − κ(α=0)"), (axs[1], "wise_gain", "WiSE escolhido: κ(α*) − κ(α=0)")):
    for (k, dsn, cid), g in cal.groupby(["kind", "dataset", "cand_id"]):
        g = g.sort_values("fidelity")
        ax.plot(g.fidelity, g[col], color="#c9c8c2", lw=1, zorder=1)
    m = cal.groupby("fidelity")[col].median()
    q1, q3 = cal.groupby("fidelity")[col].quantile(.25), cal.groupby("fidelity")[col].quantile(.75)
    ax.fill_between(m.index, q1, q3, color=S[0], alpha=.18, lw=0)
    ax.plot(m.index, m.values, color=S[0], marker="o", ms=5, zorder=3, label="mediana (IQR)")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xscale("log"); ax.set_xticks(fids, [str(f) for f in fids]); ax.set_xlabel("fidelidade (épocas)")
    ax.set_title(ttl, loc="left")
axs[0].set_ylabel("Δκ val (COV)"); axs[0].legend(frameon=False)
fig.suptitle(f"Mesma linhagem f5→f200 (calibração, {cal.groupby(['kind','dataset','cand_id']).ngroups} linhagens, C=100)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{F}/A1b_F3_fidelity_lineages.png", dpi=150); plt.close(fig)

# ---- F4: sementes SPiFiL-pequeno L6 (LeJEPA) — val kappa − L0
cur = pd.read_csv(f"{P}/A1b_seed_curves_long.csv")
cur = cur[cur.stage == "L6"].dropna(subset=["L0"])
cur["fam"] = cur.run.str.split("_").str[0].str.lower()
fams = ["aid", "eggs", "ham10000", "larvae", "protozoan"]
fig, axs = plt.subplots(1, 5, figsize=(12, 3), sharey=True)
for ax, f, col in zip(axs, fams, S):
    for r, g in cur[cur.fam == f].groupby("run"):
        ax.plot(g.epoch, g.val_kappa - g.L0, color=col, lw=1.2, alpha=.8)
    ax.axhline(0, color=INK2, lw=1); ax.set_title(f, loc="left"); ax.set_xlabel("época SSL")
    ax.set_xlim(0, min(200, cur[cur.fam == f].epoch.max()))
axs[0].set_ylabel("κ_val(L6) − κ_val(L0 SPiFiL)")
fig.suptitle("Sementes SPiFiL-pequeno + LeJEPA/SIGReg (splits 1–3, pct 50/100): ganho sobre o SPiFiL puro", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{F}/A1b_F4_seed_curves.png", dpi=150); plt.close(fig)

# ---- F5: promoções por estrato (pool f5 → promovidos)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
runs = [("artifacts/modehb_lemonade", "H-LEM C=100"), ("artifacts/modehb_hybrid_C01_final", "H-MODEHB C=0,1"),
        ("artifacts/modehb_lemonade_C01_final", "H-LEM C=0,1")]
for ax, (run, lab) in zip(axs, runs):
    g = d[(d.run == run) & (d.dataset == "remote_sensing")]
    prom = set(g[(g.op_type == "promotion") & (g.op_from_fidelity == 5)].cand_id)
    pool = g[(g.fidelity == 5) & (g.op_type != "calibration")].drop_duplicates("cand_id")
    k = len(prom)
    thr = pool.fitness_kappa.sort_values(ascending=False).iloc[k - 1]
    for i, s in enumerate(STRATA):
        p = pool[pool.stratum == s]
        isp = p.cand_id.isin(prom)
        x = i + np.random.default_rng(0).uniform(-.15, .15, len(p))
        ax.scatter(x[~isp.values], p.fitness_kappa[~isp], s=14, color="#c9c8c2", label="não promovido" if i == 0 else None)
        ax.scatter(x[isp.values], p.fitness_kappa[isp], s=22, color=S[1], edgecolor=SURF, lw=.8, label="promovido" if i == 0 else None)
    ax.axhline(thr, color=S[0], lw=1.2, ls="--", label=f"corte top-{k} global")
    ax.set_xticks(range(4), STRATA); ax.set_title(f"{lab}: f5→f15", loc="left")
axs[0].set_ylabel("κ val (COV) em f5"); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("remote_sensing: quem é promovido × estrato (promoção protegida ≥25%/estrato)", x=0.01, ha="left")
fig.tight_layout(); fig.savefig(f"{F}/A1b_F5_promotions_strata.png", dpi=150); plt.close(fig)

# ---- F6: ORIG x COV no teste
T = pd.read_csv(f"{P}/A1b_orig_cov_tests.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for i, s in enumerate(STRATA):
    t = T[T.stratum == s]
    axs[0].scatter(t.test_orig_kappa, t.test_cov_kappa, s=14, color=S[i], label=f"{s} (n={len(t)})", edgecolor=SURF, lw=.5)
    axs[1].scatter(t.params, t.gap, s=14, color=S[i], edgecolor=SURF, lw=.5)
t = T[T.stratum.str.startswith("P", na=False)]
axs[0].scatter(t.test_orig_kappa, t.test_cov_kappa, s=18, color=S[6], marker="D", label=f"PNAS P0–P4 (n={len(t)})")
axs[1].scatter(t.params, t.gap, s=18, color=S[6], marker="D")
lim = [0, 1]
axs[0].plot(lim, lim, color=INK2, lw=1); axs[0].set_xlabel("κ teste ORIG (1280, sem scaler)"); axs[0].set_ylabel("κ teste COV (cov_head + scaler)")
axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("ORIG × COV (mesmo checkpoint)", loc="left")
axs[1].set_xscale("log"); axs[1].axhline(0, color=INK2, lw=1)
rho = spearmanr(T.params, T.gap)[0]
axs[1].set_xlabel("parâmetros"); axs[1].set_ylabel("κ COV − κ ORIG"); axs[1].set_title(f"gap cresce com o porte (Spearman ρ={rho:.2f})", loc="left")
fig.tight_layout(); fig.savefig(f"{F}/A1b_F6_orig_vs_cov.png", dpi=150); plt.close(fig)

# ---- F7: val(COV) prediz teste? rho por (busca, dataset)
Tt = pd.read_csv(f"{P}/A1b_tests_all.csv").dropna(subset=["test_orig_kappa", "test_val_kappa_logged"])
rows = []
for (run, dsn), g in Tt.groupby(["run", "dataset"]):
    if len(g) < 4:
        continue
    gc = g.dropna(subset=["test_cov_kappa"])
    rows.append((f"{run.split('/')[-1].replace('modehb_','')}/{dsn}", len(g),
                 spearmanr(g.test_val_kappa_logged, g.test_orig_kappa)[0],
                 spearmanr(gc.test_val_kappa_logged, gc.test_cov_kappa)[0]))
R = pd.DataFrame(rows, columns=["grp", "n", "rho_orig", "rho_cov"]).sort_values("rho_cov")
fig, ax = plt.subplots(figsize=(7.5, 5.6))
y = np.arange(len(R))
ax.hlines(y, R.rho_orig, R.rho_cov, color="#c9c8c2", lw=1.5)
ax.scatter(R.rho_cov, y, color=S[0], s=30, zorder=3, label="val(COV) × teste COV (mesma régua)")
ax.scatter(R.rho_orig, y, color=S[1], s=30, zorder=3, label="val(COV) × teste ORIG (régua reportada)")
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(y, [f"{g} (n={k})" for g, k in zip(R.grp, R.n)]); ax.set_xlabel("Spearman ρ")
ax.set_title("Validação do fitness ordena o teste COV, não o ORIG", loc="right")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=1, fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/A1b_F7_val_vs_test_rank.png", dpi=150); plt.close(fig)
print("ok")
