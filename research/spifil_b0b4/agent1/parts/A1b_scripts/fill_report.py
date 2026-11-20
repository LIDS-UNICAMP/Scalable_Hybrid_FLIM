"""A1b — preenche os placeholders __X__ de D2 com os números finais de A1b_embed_diag.csv e imprime contagens."""
import numpy as np
import pandas as pd

P = "/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/agent1/parts"
D = pd.read_csv(f"{P}/A1b_embed_diag.csv")
sv = D[D.variant == "saved"].copy()


def m(a, b):
    x = (sv[a] - sv[b]).dropna()
    return f"{x.median():+.3f} [{x.min():+.3f}; {x.max():+.3f}]"


rep = {"__N__": str(len(sv)), "__COVORIG__": m("k_cov_full_scaler", "k_orig_noscaler"),
       "__NOHEAD__": m("k_cov_nohead_scaler", "k_cov_full_scaler"), "__COVSC__": m("k_cov_full_scaler", "k_cov_full_noscaler"),
       "__ORIGSC__": m("k_orig_scaler", "k_orig_noscaler"), "__ORIGPW__": m("k_orig_pw", "k_orig_noscaler"),
       "__LASTGAP__": m("k_lastgap_scaler", "k_orig_noscaler"), "__BESTBLK__": m("k_best_block_gap", "k_lastgap_scaler"),
       "__CATGAP__": m("k_catgap_scaler", "k_lastgap_scaler"), "__SECOND__": m("k_cov_nohead_scaler", "k_best_block_gap"),
       "__MULTI__": m("k_cov_nohead_scaler", "k_cov_last_scaler"),
       "__ER__": f"{sv.h_eff_rank_pr.median():.1f}", "__TOP1__": f"{sv.h_top1_var.median():.0%}",
       "__ERG__": f"{sv.g_eff_rank_pr.median():.1f}", "__ERC__": f"{sv.cov_eff_rank_pr.median():.1f}"}
p = f"{P}/A1b_aprendizagem_embedding.md"
s = open(p).read()
for k, v in rep.items():
    s = s.replace(k, v)
open(p, "w").write(s)
print(rep)
# contagens de bloco
best_not_last, deep, deep_not_last = 0, 0, 0
for _, r in sv.iterrows():
    c = [float(v) for v in str(r.block_cov_kappas).split(";")]
    nl = int(np.argmax(c)) != len(c) - 1
    best_not_last += nl
    if len(c) >= 4:
        deep += 1
        deep_not_last += c[-1] < max(c[:-1])
print("best COV block not last:", best_not_last, "/", len(sv), "| deep(>=4):", deep_not_last, "/", deep)
