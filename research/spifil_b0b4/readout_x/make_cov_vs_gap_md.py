"""Regera results/cov_vs_gap_melhores.md (cabeça atual × 2ª ordem × 2ª ordem+TTA8, SVM original no TESTE; top-5 por base pela VAL)
a partir de todos os results/*.jsonl com pct100. Rodado em loop pelo tmux md_updater."""
import json, glob, collections, datetime
from pathlib import Path
H = Path(__file__).resolve().parent / "results"
R = collections.defaultdict(dict)
for f in glob.glob(str(H / "*.jsonl")):
    if "train_pct" in f: continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("max_images"): continue
        sn = r.get("split_name") or ""
        if "/pct" in sn and "/pct100" not in sn: continue
        R[(r["dataset"], r["backbone_id"])][r["head"]] = r
DS = ['flowers', 'remote_sensing', 'ham10000', 'aid', 'coconut_trees', 'eggs', 'larvae', 'protozoan', 'eggs_raw', 'larvae_raw', 'protozoan_raw']
out = [f"# 2ª ordem (covariância) × cabeça atual — κ de TESTE, SVM original C=100\n\nAtualizado {datetime.datetime.now():%d/%m %H:%M} (automático). Top-5 por base pela melhor κ de VALIDAÇÃO; 🟢/🔴 = 2ª ordem melhora/piora > 0,01. Linhas com val = treino (validação 1,000) ficam fora do top.\n",
       "| base | modelo | κ teste GAP | κ teste 2ª ord | Δ | κ teste 2ª ord+TTA8 | Δ TTA8 − GAP |", "|---|---|---|---|---|---|---|"]
summ = {}
for ds in DS:
    rows = []
    for (d, b), h in R.items():
        if d != ds or "gap1280" not in h or "so2_4096_pca" not in h: continue
        g, s, t = h["gap1280"], h["so2_4096_pca"], h.get("so2_4096_pca_tta8")
        bv = max(x["kappa_val"] for x in (g, s, t) if x)
        rows.append((bv, b, g, s, t))
    summ[ds] = (len(rows), sum(s["kappa_test"] > g["kappa_test"] + .01 for _, _, g, s, _ in rows),
                sum(s["kappa_test"] < g["kappa_test"] - .01 for _, _, g, s, _ in rows),
                sum(s["kappa_test"] - g["kappa_test"] for _, _, g, s, _ in rows) / len(rows) if rows else 0)
    for bv, b, g, s, t in sorted([r for r in rows if r[0] < 0.999], key=lambda x: -x[0])[:5]:
        d1 = s["kappa_test"] - g["kappa_test"]
        out.append(f"| {ds} | {b.replace(ds + '_', '')} | {g['kappa_test']:.3f} | {s['kappa_test']:.3f} | {'🟢' if d1 > .01 else ('🔴' if d1 < -.01 else '⚪')} {d1:+.3f} | "
                   + (f"{t['kappa_test']:.3f} | {t['kappa_test'] - g['kappa_test']:+.3f} |" if t else "– | – |"))
out += ["", "| base | modelos | 2ª ordem melhora | piora | Δ médio |", "|---|---|---|---|---|"]
out += [f"| {ds} | {n} | {u} | {d} | {m:+.3f} |" for ds, (n, u, d, m) in summ.items()]
(H / "cov_vs_gap_melhores.md").write_text("\n".join(out) + "\n")
print("ok", sum(v[0] for v in summ.values()))
