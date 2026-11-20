"""make_ga_best_table.py -- results/ga_best_test.jsonl -> results/ga_best_tabela.md.

base x modelo x leitura (GAP oficial | 2a ordem 4096 | 2a ordem 4096 + TTA8): kappa val, kappa teste,
F1 teste. Destaque por base: o escolhido pela VALIDACAO (sem olhar o teste); o max-teste aparece so
como referencia. FMs (regua ORIG, split 1) lidos de ga_search/modehb/reports/all_results_today.md.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORDER = ["flowers", "remote_sensing", "ham10000", "aid", "coconut_trees"]
HEADS = [("gap1280", "GAP oficial"), ("so2_4096_pca", "2a ordem 4096"),
         ("so2_4096_pca_tta8", "2a ordem 4096 + TTA8")]


def fms():
    t = (REPO / "ga_search/modehb/reports/all_results_today.md").read_text().splitlines()
    out = {}
    for i, l in enumerate(t):
        m = re.match(r"\*\*FMs congelados \(régua ORIG, pct 100\)\*\* — benchmark `(\w+)`", l)
        if m and m[1] in ORDER and m[1] not in out:
            rows = []
            for r in t[i + 4:i + 12]:
                if not r.startswith("| FM-"):
                    break
                c = [x.strip() for x in r.strip("|").split("|")]
                rows.append({"fm": c[0], "k_test": float(c[1]), "f1_test": float(c[2]),
                             "k_val": float(c[3]), "k_test_mean": c[5]})
            out[m[1]] = rows
    return out


def main():
    L = json.loads((HERE / "ga_best_list.json").read_text())
    meta = {m["id"]: m for b in L["bases"].values() for m in b["models"]}
    rows = [json.loads(l) for l in (HERE / "results/ga_best_test.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["backbone_id"], {})[r["head"]] = r
    F = fms()
    o = ["# GA (MO-DEHB hibrida / LEMONADE) -- melhores por validacao, TESTE com a SVM original", "",
         "SVM original do repo (`core.svm.fit_svm`: SVC linear, C=100, ovo, sem scaler), ajustada no "
         "TREINO, uma por leitura. Split 1, pct 100, seed 42, n=1. Escolha dos modelos sem olhar o teste "
         "(`ga_best_list.json`: melhor fitness_kappa de validacao por estrato de cada busca + melhor geral). "
         "Leituras: **GAP oficial** = `head(encoder(x))` (embedding da busca); **2a ordem 4096** = "
         "Norm(P[mu, sigma, triu sqrtm Cov]) do ultimo bloco, PCA k=88 so no treino; **+TTA8** = mesma "
         "PCA e mesma SVM (treino sem TTA), avaliacao com a media das 8 vistas rot90 x flip. "
         "Celula: kappa val / **kappa teste** (F1 teste). `val GA` = fitness_kappa da busca. "
         "CPU fp32, sem autocast. Fonte: `results/ga_best_test.jsonl`.", ""]
    summary = ["## Resumo por base (escolha pela validacao)", "",
               "| base | escolhido pela val (modelo, leitura) | kappa val | **kappa teste** | F1 teste | "
               "max-teste (referencia, nao selecionavel) | melhor FM split1 kappa / F1 | FMs que o escolhido passa | "
               "melhor geral do GA (val GA) nas 3 leituras: kappa teste GAP / 2a ordem / +TTA8 |",
               "|---|---|---|---|---|---|---|---|---|"]
    body = []
    for ds in ORDER:
        ids = [m["id"] for m in L["bases"].get(ds, {}).get("models", [])]
        got = [i for i in ids if i in by]
        if not got:
            continue
        cand = [(by[i][h]["kappa_val"], i, h) for i in got for h, _ in HEADS if h in by[i]]
        sel = max(cand)
        best_t = max((by[i][h]["kappa_test"], i, h) for i in got for h, _ in HEADS if h in by[i])
        vt = any(r.get("val_is_train") for i in got for r in by[i].values())
        body += [f"## {ds} ({len(got)}/{len(ids)} modelos prontos)" + (" -- val == treino no split: kappa val e de TREINO" if vt else ""), "",
                 "| modelo | busca/estrato | params | val GA | " + " | ".join(n for _, n in HEADS) + " |",
                 "|---|---|---|---|" + "---|" * len(HEADS)]
        for i in sorted(got, key=lambda i: -meta[i]["kappa_val_search"]):
            m = meta[i]
            cells = []
            for h, _ in HEADS:
                r = by[i].get(h)
                if r is None:
                    cells.append("—")
                    continue
                c = f"{r['kappa_val']:.3f} / **{r['kappa_test']:.3f}** ({r['f1_test']:.3f})"
                if (i, h) == (sel[1], sel[2]):
                    c = f"⭐ {c}"
                if (i, h) == (best_t[1], best_t[2]):
                    c += " ↑max-teste"
                cells.append(c)
            body.append(f"| {i} | {m['search'].replace('modehb_', '')}/{m['stratum']} | "
                        f"{m['params'] / 1e6:.2f}M | {m['kappa_val_search']:.3f} | " + " | ".join(cells) + " |")
        fm = F.get(ds, [])
        if fm:
            body += ["", "FMs congelados (regua ORIG, split 1; `all_results_today.md`): " +
                     "; ".join(f"{f['fm']} {f['k_test']:.3f} / {f['f1_test']:.3f}" for f in fm), ""]
        r = by[sel[1]][sel[2]]
        bfm = max(fm, key=lambda f: f["k_test"]) if fm else None
        passa = [f["fm"] for f in fm if r["kappa_test"] > f["k_test"]]
        summary.append(
            f"| {ds}{' †' if vt else ''} | {sel[1]} · {dict(HEADS)[sel[2]]} | {r['kappa_val']:.3f} | "
            f"**{r['kappa_test']:.3f}** | {r['f1_test']:.3f} | {best_t[0]:.3f} ({best_t[1]} · {dict(HEADS)[best_t[2]]}) | "
            + (f"{bfm['fm']} {bfm['k_test']:.3f} / {bfm['f1_test']:.3f}" if bfm else "—")
            + f" | {', '.join(passa) if passa else 'nenhum'} | "
            + (lambda g: (f"{g} : " + " / ".join(f"{by[g][h]['kappa_test']:.3f}" if h in by.get(g, {}) else "—"
                                                  for h, _ in HEADS)) if g else "—")(
                next((i for i in ids if "best_val_overall" in meta[i]["why"]), None)) + " |")
    summary += ["", "† base com `validation` == `train` no descritor do split: o kappa val e de TREINO.", "",
                "Nota: o `val GA` (fitness_kappa) da busca NAO e a regua GAP oficial: a fitness usava "
                "`readout: cov_head` (k=64) + StandardScaler, C=100 (config_resolved.yaml, eval.svm_fitness); "
                "por isso o kappa val da leitura GAP oficial fica abaixo do `val GA`. A escolha dos modelos "
                "seguiu o `val GA` (pedido); o destaque ⭐ e a melhor (modelo, leitura) pelo kappa val "
                "medido aqui, sem olhar o teste.", ""]
    (HERE / "results/ga_best_tabela.md").write_text("\n".join(o + summary + body) + "\n")
    print(HERE / "results/ga_best_tabela.md")


if __name__ == "__main__":
    main()
