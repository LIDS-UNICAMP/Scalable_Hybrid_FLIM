"""Gera results/curvas_pct.md (melhor × SPiFiL sem backprop por % de treino; leitura escolhida pela VAL)."""
import json, collections, datetime
from pathlib import Path
H = Path(__file__).resolve().parent / "results"
R = collections.defaultdict(dict)
for l in open(H / "curvas_pct.jsonl"):
    r = json.loads(l); _, rest = r["backbone_id"].split("curva_", 1); br, p = rest.rsplit("_p", 1)
    R[br][(int(p), r["head"])] = (r["kappa_val"], r["kappa_test"])
def pick(v, p):
    c = [v[(p, h)] for h in ("so2_4096_pca", "so2_4096_pca_tta8", "so2_4096_pca_tta2h") if (p, h) in v]
    return max(c)[1] if c else None
P = [1, 5, 10, 25, 50, 75, 100]
out = [f"# Curvas κ × % de treino — melhor modelo × mesmo modelo só SPiFiL (sem backprop)\n\nAtualizado {datetime.datetime.now():%d/%m %H:%M}. κ de TESTE, SVM original C=100; leitura (2ª ordem / +TTA8 / +TTA2h) escolhida pela validação em cada ponto. Parasitas eggs/larvae (pre): pontos <100% com amostra estratificada (ADAPTAÇÃO).\n",
       "| base | modelo | " + " | ".join(f"{p}%" for p in P) + " | ganho do backprop a 100% |", "|---|---|" + "---|" * len(P) + "---|"]
for base in sorted({k.rsplit("_", 1)[0].replace("_spifil_sem", "").replace("_best", "") for k in R}):
    b, t = R.get(base + "_best", {}), R.get(base + "_spifil_sem_bp", {})
    for name, v in (("melhor", b), ("só SPiFiL", t)):
        if not v: continue
        cells = [f"{pick(v, p):.3f}" if pick(v, p) is not None else "–" for p in P]
        g = (f"{pick(b, 100) - pick(t, 100):+.3f}" if name == "melhor" and b and t and pick(b, 100) is not None and pick(t, 100) is not None else "")
        out.append(f"| {base} | {name} | " + " | ".join(cells) + f" | {g} |")
(H / "curvas_pct.md").write_text("\n".join(out) + "\n"); print("ok")
