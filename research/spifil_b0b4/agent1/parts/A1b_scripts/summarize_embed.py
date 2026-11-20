"""A1b — resume A1b_embed_diag.jsonl (diagnóstico CPU treino→val) em CSV + figuras F8/F9."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/agent1/parts"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
R = [json.loads(l) for l in open(f"{P}/A1b_embed_diag.jsonl")]
rows = []
for r in R:
    p = {k: v["kappa"] for k, v in r["probes"].items()}
    nb = r["n_blocks"]
    blk_gap = [p.get(f"gap_b{i}_scaler") for i in range(nb)]
    blk_gem = [p.get(f"gem_b{i}_scaler") for i in range(nb)]
    blk_cov = [p.get(f"cov_b{i}_scaler") for i in range(nb)]
    gh, gl = r["geom_head"], r["geom_lastgap"]
    rows.append({
        "dataset": r["dataset"], "cand_id": r["cand_id"], "fidelity": r["fidelity"], "variant": r["variant"],
        "checkpoint": r["checkpoint"], "params": r["params"], "pooling": r["pooling"], "n_blocks": nb,
        "block_channels": "-".join(map(str, r["block_channels"])), "embed_dim": r["embed_dim"],
        "n_train": r["n_train"], "n_val": r["n_val"], "wise_alpha_saved": r["wise_alpha_saved"],
        "head_W_rank": r["head_W_rank"], "head_W_cond": r["head_W_cond"],
        "h_norm_mean": gh["norm_mean"], "h_mean_over_spread": gh["mean_norm_over_spread"],
        "h_dimstd_max_over_med": gh["dim_std_max_over_median"], "h_frac_dead": gh["frac_dead_dims"],
        "h_eff_rank_pr": gh["eff_rank_pr"], "h_rankme": gh["rankme"], "h_num_rank": gh["num_rank"],
        "h_top1_var": gh["top1_var_frac"], "h_dims90": gh["n_dims_90var"], "h_dims99": gh["n_dims_99var"],
        "h_cond": gh["cond_nonzero"],
        "g_eff_rank_pr": gl["eff_rank_pr"], "g_top1_var": gl["top1_var_frac"], "g_dims90": gl["n_dims_90var"],
        "g_dim": gl["dim"], "g_rankme": gl["rankme"],
        "cov_eff_rank_pr": (r.get("geom_cov") or {}).get("eff_rank_pr"), "cov_dim": r.get("cov_dim"),
        **{f"k_{k}": v for k, v in p.items() if not (k.startswith("gap_b") or k.startswith("gem_b") or k.startswith("cov_b"))},
        "k_best_block_gap": max(b for b in blk_gap if b is not None), "best_block_gap_idx": int(np.nanargmax(blk_gap)),
        "k_best_block_gem": max(b for b in blk_gem if b is not None),
        "k_best_block_cov": max([b for b in blk_cov if b is not None] or [np.nan]),
        "block_gap_kappas": ";".join(f"{b:.3f}" for b in blk_gap), "block_gem_kappas": ";".join(f"{b:.3f}" for b in blk_gem),
        "block_cov_kappas": ";".join(f"{b:.3f}" for b in blk_cov if b is not None),
        "svm_iter_orig": r["probes"]["orig_noscaler"]["n_iter_max"], "svm_iter_orig_sc": r["probes"]["orig_scaler"]["n_iter_max"],
    })
D = pd.DataFrame(rows)
D.to_csv(f"{P}/A1b_embed_diag.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60)
cols = ["dataset", "cand_id", "fidelity", "variant", "params", "pooling", "block_channels", "k_orig_noscaler", "k_orig_scaler",
        "k_orig_pnl2_scaler", "k_orig_pw", "k_lastgap_noscaler", "k_lastgap_scaler", "k_catgap_scaler", "k_best_block_gap",
        "best_block_gap_idx", "k_cov_last_scaler", "k_cov_nohead_scaler", "k_cov_headgroup_scaler", "k_cov_full_scaler", "k_cov_full_noscaler"]
print(D[[c for c in cols if c in D]].round(3).to_string())
print(D[["dataset", "cand_id", "variant", "head_W_rank", "h_norm_mean", "h_mean_over_spread", "h_dimstd_max_over_med", "h_frac_dead",
         "h_eff_rank_pr", "h_rankme", "h_num_rank", "h_top1_var", "h_dims90", "h_dims99", "g_dim", "g_eff_rank_pr", "g_dims90", "cov_dim", "cov_eff_rank_pr", "svm_iter_orig", "svm_iter_orig_sc"]].round(3).to_string())

# ---- F8: leituras por checkpoint (variante saved)
sv = D[D.variant == "saved"].sort_values(["dataset", "params"]).reset_index(drop=True)
AB0 = {"remote_sensing": "RS", "larvae": "LV", "larvae_raw": "LVraw", "protozoan": "PZ", "ham10000": "HAM"}
reads = [("k_orig_noscaler", "ORIG (1280, sem scaler)"), ("k_orig_scaler", "ORIG + scaler"), ("k_orig_pw", "ORIG + PCA-whiten"),
         ("k_lastgap_scaler", "GAP último bloco + scaler"), ("k_best_block_gap", "melhor GAP por bloco + scaler"),
         ("k_catgap_scaler", "concat GAP blocos + scaler"), ("k_cov_last_scaler", "COV só último bloco"), ("k_cov_full_scaler", "COV completo (fitness)")]
fig, ax = plt.subplots(figsize=(14, 4.8))
x = np.arange(len(sv))
w = 0.8 / len(reads)
for j, (c, lab) in enumerate(reads):
    if c in sv:
        ax.bar(x + (j - len(reads) / 2 + .5) * w, sv[c], width=w * .9, color=S[j % 8], label=lab, edgecolor=SURF, lw=.5)
ax.set_xticks(x, [f"{AB0[a]}\n{b[-2:]} f{int(f)}\n{p/1e3:.0f}k" for a, b, f, p in zip(sv.dataset, sv.cand_id, sv.fidelity, sv.params)], fontsize=7.5)
ax.set_ylabel("κ validação (SVM C=100, treino→val)"); ax.set_ylim(0, 1.02)
ax.set_title("Mesmo encoder, leituras diferentes: onde está a perda do ORIG (diagnóstico, nunca teste)", loc="left")
ax.legend(ncol=4, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout(); fig.savefig(f"{P}/figs/A1b_F8_readouts_per_ckpt.png", dpi=150); plt.close(fig)

# ---- F9: sonda COV por bloco (relativa ao melhor bloco) + gap x rank efetivo
AB = {"remote_sensing": "RS", "larvae": "LV", "larvae_raw": "LVraw", "protozoan": "PZ", "ham10000": "HAM"}
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
for _, r in sv.iterrows():
    g = [float(v) for v in str(r.block_cov_kappas).split(";")]
    xs = np.arange(1, len(g) + 1)
    axs[0].plot(xs, np.array(g) - r.k_orig_noscaler, marker="o", ms=3, lw=1.1, color="#b5b4ae")
    axs[0].scatter(xs[-1], g[-1] - r.k_orig_noscaler, s=18, color=S[1], zorder=3)
    axs[0].scatter(xs[int(np.argmax(g))], max(g) - r.k_orig_noscaler, s=18, color=S[0], zorder=4)
axs[0].scatter([], [], color=S[0], s=18, label="melhor bloco"); axs[0].scatter([], [], color=S[1], s=18, label="último bloco")
axs[0].axhline(0, color=INK2, lw=1); axs[0].set_xticks(range(1, 6)); axs[0].set_xlabel("bloco")
axs[0].set_ylabel("κ_val(COV do bloco) − κ_val(ORIG)"); axs[0].legend(frameon=False, fontsize=8, loc="lower right")
axs[0].set_title("COV por bloco vs ORIG (cinza = 1 checkpoint)", loc="left")
axs[1].scatter(sv.h_eff_rank_pr, sv.k_cov_full_scaler - sv.k_orig_noscaler, color=S[0], s=25)
for _, r in sv.iterrows():
    axs[1].annotate(f"{AB.get(r.dataset, r.dataset)} {r.cand_id[-2:]}" + ("" if r.fidelity == 200 else f" f{int(r.fidelity)}"),
                    (r.h_eff_rank_pr, r.k_cov_full_scaler - r.k_orig_noscaler), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords="offset points")
axs[1].set_xscale("log"); axs[1].set_xlabel("rank efetivo (participation ratio) do embedding 1280")
axs[1].set_ylabel("κ_val COV − κ_val ORIG"); axs[1].set_title("Gap × rank efetivo do 1280", loc="left")
fig.tight_layout(); fig.savefig(f"{P}/figs/A1b_F9_block_probes.png", dpi=150); plt.close(fig)

# ---- tabela markdown para o relatório
md = ["| dataset | cand fid var | params | blocos | ORIG | ORIG+sc | ORIG+PW | GAP último+sc | melhor GAP bloco+sc (idx) | concat GAP+sc | COV último | COV sem head | COV completo | COV sem scaler | rank ef. 1280 | rank ef. GAP último | top1 var 1280 |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in D.sort_values(["dataset", "cand_id", "fidelity", "variant"]).iterrows():
    f = lambda c: f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "—"
    md.append(f"| {r.dataset} | {r.cand_id} f{int(r.fidelity)} {r.variant} | {r.params/1e3:.0f}k | {r.block_channels} | {f('k_orig_noscaler')} | {f('k_orig_scaler')} | {f('k_orig_pw')} | "
              f"{f('k_lastgap_scaler')} | {f('k_best_block_gap')} ({int(r.best_block_gap_idx)}) | {f('k_catgap_scaler')} | {f('k_cov_last_scaler')} | {f('k_cov_nohead_scaler')} | "
              f"{f('k_cov_full_scaler')} | {f('k_cov_full_noscaler')} | {r.h_eff_rank_pr:.1f} | {r.g_eff_rank_pr:.1f} | {r.h_top1_var:.2f} |")
open(f"{P}/A1b_embed_diag_table.md", "w").write("\n".join(md) + "\n")
sv = D[D.variant == "saved"]
diff = lambda a, b: (sv[a] - sv[b]).describe()[["count", "mean", "50%", "min", "max"]].round(3).to_dict()
print("COV_full - ORIG", diff("k_cov_full_scaler", "k_orig_noscaler"))
print("COV_nohead - ORIG", diff("k_cov_nohead_scaler", "k_orig_noscaler"))
print("COV_full - COV_nohead", diff("k_cov_full_scaler", "k_cov_nohead_scaler"))
print("COV_full_scaler - COV_full_noscaler", diff("k_cov_full_scaler", "k_cov_full_noscaler"))
print("ORIG+sc - ORIG", diff("k_orig_scaler", "k_orig_noscaler"))
print("ORIG+pw - ORIG", diff("k_orig_pw", "k_orig_noscaler"))
print("lastGAP+sc - ORIG", diff("k_lastgap_scaler", "k_orig_noscaler"))
print("bestblockGAP - lastGAP", diff("k_best_block_gap", "k_lastgap_scaler"))
print("catGAP - lastGAP", diff("k_catgap_scaler", "k_lastgap_scaler"))
print("COV_nohead - bestblockGAP", diff("k_cov_nohead_scaler", "k_best_block_gap"))
print("COV_nohead - COV_last", diff("k_cov_nohead_scaler", "k_cov_last_scaler"))
print("best block cov - cov last", diff("k_best_block_cov", "k_cov_last_scaler"))
print("eff rank 1280 median", sv.h_eff_rank_pr.median(), "top1", sv.h_top1_var.median(), "lastgap eff", sv.g_eff_rank_pr.median(), "cov eff", sv.cov_eff_rank_pr.median())
# a0 vs saved
pv = D.pivot_table(index=["dataset", "cand_id", "fidelity"], columns="variant", values=["k_orig_noscaler", "k_cov_full_scaler", "k_lastgap_scaler", "h_eff_rank_pr"])
print(pv.round(3).to_string())
