"""gspan_mining.py -- Técnica 1: mineração de subgrafos frequentes (gSpan oficial: github.com/betterenvi/gSpan, pacote gspan-mining).

Etapas
 A) grafos direcionados do graph_extractor (ordem real do forward); rótulo coarse (tipo de op + kernel/stride).
 B) gSpan roda POR BASE no top-f (f = 10/20/30% por κ de VALIDAÇÃO), min_support = fração do grupo. O gSpan oficial é
    não-direcionado: a direção vai no rótulo da aresta ("origem>destino"). Cada padrão volta a ser um DiGraph e é RECONTADO
    nos grafos direcionados originais por monomorfismo (networkx DiGraphMatcher) -> correspondência exata com os originais.
 C) frequência por base (top vs demais), frequência global BALANCEADA (média das bases), enriquecimento top-demais,
    Fisher exato + Benjamini-Hochberg, e Δκ_val médio de quem contém vs quem não contém (dentro da base).
    Operadores isolados ficam numa tabela à parte (não confundir com subgrafos).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism as iso

from .common import ResourceMonitor, load_config, md_table, write_json
from .graph_extractor import from_json

IGNORE = {"input", "output"}


# ── dados ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_groups(cfg):
    from . import inventory
    gi = json.loads((Path(cfg["out_dir"]) / "graphs_index.json").read_text())
    recs = {r["architecture_id"]: r for r in inventory.load(cfg)}
    items = []
    for aid, g in gi.items():
        r = recs.get(aid)
        if not g.get("ok") or r is None or r["dataset"] not in cfg["datasets"]:
            continue
        G = from_json(json.loads(Path(g["graph"]).read_text()))
        G = G.subgraph([n for n, d in G.nodes(data=True) if d["label"] not in IGNORE]).copy()
        base_id = aid.rsplit("/f", 1)[0] if r["source"] != "growth" else aid
        items.append({"id": aid, "cand": base_id, "dataset": r["dataset"], "top_frac": r.get("top_frac"), "kappa_val": r["kappa_val"],
                      "rank": r.get("rank_val_in_dataset"), "n_ds": r.get("n_in_dataset"), "G": G, "focus": r.get("focus")})
    # uma entrada por candidato (várias fidelidades do mesmo candidato não contam várias vezes): a de maior κ_val
    best = {}
    for it in items:
        if it["cand"] not in best or it["kappa_val"] > best[it["cand"]]["kappa_val"]:
            best[it["cand"]] = it
    return list(best.values())


def closed_filter(keys, pats, H):
    """remove padrão que tem super-padrão com o MESMO vetor de ocorrências (padrões fechados)."""
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[H[i].tobytes()].append(i)
    keep = []
    for idx in groups.values():
        idx = sorted(idx, key=lambda i: -pats[keys[i]].number_of_nodes())
        kept = []
        for i in idx:
            if not any(pats[keys[j]].number_of_nodes() > pats[keys[i]].number_of_nodes()
                       and _contains((pats[keys[i]], pats[keys[j]])) for j in kept):
                kept.append(i)
        keep += kept
    return sorted(keep)


# ── gSpan oficial (captura dos padrões sem prints/pandas.append) ────────────────────────────────────────────────────
def _gspan_cls():
    from gspan_mining import gSpan

    class Miner(gSpan):
        def __init__(self, *a, **k):
            super().__init__(*a, **k); self.patterns = []

        def _report_size1(self, g, support):
            pass

        def _report(self, projected):
            if self._DFScode.get_num_vertices() < self._min_num_vertices:
                return
            code = [(e.frm, e.to, e.vevlb) for e in self._DFScode]
            gids = sorted({p.gid for p in projected})
            self.patterns.append({"code": code, "support": self._support, "gids": gids})
    return Miner


def run_gspan(graphs: list[nx.DiGraph], min_support: int, vmin: int, vmax: int, tmp: Path) -> list[dict]:
    """graphs -> padrões (DFS code). Arestas rotuladas 'lu>lv' (direção), vértices com o rótulo coarse."""
    lines = []
    for gi, G in enumerate(graphs):
        nid = {n: i for i, n in enumerate(G.nodes)}
        lines.append(f"t # {gi}")
        lines += [f"v {nid[n]} {d['label']}" for n, d in G.nodes(data=True)]
        lines += [f"e {nid[u]} {nid[v]} {G.nodes[u]['label']}>{G.nodes[v]['label']}" for u, v in G.edges]
    lines.append("t # -1")
    tmp.write_text("\n".join(lines) + "\n")
    Miner = _gspan_cls()
    m = Miner(str(tmp), min_support=min_support, min_num_vertices=vmin, max_num_vertices=vmax, is_undirected=True)
    with contextlib.redirect_stdout(io.StringIO()):
        m.run()
    return m.patterns


def code_to_digraph(code) -> nx.DiGraph:
    P = nx.DiGraph()
    for frm, to, (l1, el, l2) in code:
        # o gSpan usa rótulo vago (-1) para vértices já vistos no DFS code: mantém o rótulo original
        for n, l in ((frm, l1), (to, l2)):
            if n not in P:
                P.add_node(n, label=l)
            elif isinstance(l, str) and l != "-1":
                P.nodes[n]["label"] = l
        l1, l2 = P.nodes[frm]["label"], P.nodes[to]["label"]
        src, dst = el.split(">", 1)
        # orienta pela etiqueta da aresta; se os dois rótulos são iguais, usa a ordem do DFS code
        if (l1, l2) == (src, dst):
            P.add_edge(frm, to)
        elif (l2, l1) == (src, dst):
            P.add_edge(to, frm)
        else:
            P.add_edge(frm, to)
    return P


def pattern_key(P):
    return nx.weisfeiler_lehman_graph_hash(P, node_attr="label", iterations=3) + f"_{P.number_of_nodes()}"


def pattern_str(P) -> str:
    if nx.is_directed_acyclic_graph(P) and all(P.in_degree(n) <= 1 and P.out_degree(n) <= 1 for n in P):
        return " → ".join(P.nodes[n]["label"] for n in nx.topological_sort(P))
    return "; ".join(f"{P.nodes[u]['label']}→{P.nodes[v]['label']}" for u, v in sorted(P.edges))


def pattern_kind(P) -> str:
    labs = [P.nodes[n]["label"] for n in P]
    if any(l == "add" and P.in_degree(n) >= 2 for n, l in zip(P.nodes, labs)) or "skip1" in labs:
        return "residual"
    if any(P.out_degree(n) > 1 or P.in_degree(n) > 1 for n in P):
        return "ramificado"
    return {2: "par", }.get(P.number_of_nodes(), "sequência")


def _contains(args):
    P, G = args
    lp, lg = Counter(nx.get_node_attributes(P, "label").values()), Counter(nx.get_node_attributes(G, "label").values())
    if any(lg[k] < v for k, v in lp.items()):
        return False
    gm = iso.DiGraphMatcher(G, P, node_match=lambda a, b: a["label"] == b["label"])
    return gm.subgraph_is_monomorphic()


def _bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def mine(cfg):
    from scipy.stats import fisher_exact
    gc = cfg["gspan"]; out = Path(cfg["out_dir"]); rep = Path(cfg["report_dir"])
    items = load_groups(cfg)
    by_ds = defaultdict(list)
    for it in items:
        by_ds[it["dataset"]].append(it)
    results = {}
    with ResourceMonitor(out / "resources.jsonl", "gspan"):
        for f in cfg["inventory"]["top_fractions"]:
            pats = {}
            for ds, its in by_ds.items():
                top = [it for it in its if it["top_frac"] is not None and it["top_frac"] <= f]
                if len(top) < 3:
                    continue
                ms = max(2, int(np.ceil(gc["min_support_frac"] * len(top))))
                for p in run_gspan([it["G"] for it in top], ms, gc["min_vertices"], gc["max_vertices"], out / f"_gspan_{ds}_{f}.txt"):
                    P = code_to_digraph(p["code"])
                    if not nx.is_weakly_connected(P):
                        continue
                    pats.setdefault(pattern_key(P), P)
            # maximalidade: remove padrões contidos num maior com o mesmo conjunto de grafos (fechados aproximados) depois da recontagem
            keys = list(pats)
            G_all = [it["G"] for it in items]
            with ProcessPoolExecutor(min(32, os.cpu_count())) as ex:
                hits = list(ex.map(_contains, [(pats[k], G) for k in keys for G in G_all], chunksize=256))
            H = np.array(hits, bool).reshape(len(keys), len(G_all))
            kk = closed_filter(keys, pats, H); n_all = len(keys)
            keys = [keys[i] for i in kk]; H = H[kk]
            rows = []
            for i, k in enumerate(keys):
                P = pats[k]; row = {"pattern": pattern_str(P), "kind": pattern_kind(P), "n_nodes": P.number_of_nodes(), "key": k}
                fr_top, pvals = [], []
                for ds, its in by_ds.items():
                    idx = [j for j, it in enumerate(items) if it["dataset"] == ds]
                    top = [j for j in idx if items[j]["top_frac"] is not None and items[j]["top_frac"] <= f]
                    rest = [j for j in idx if j not in set(top)]
                    a, b = int(H[i, top].sum()), len(top) - int(H[i, top].sum())
                    c, d = int(H[i, rest].sum()), len(rest) - int(H[i, rest].sum())
                    ft = a / max(1, len(top)); fr = c / max(1, len(rest))
                    row[f"top_{ds}"] = ft; row[f"rest_{ds}"] = fr
                    _, p = fisher_exact([[a, b], [c, d]], alternative="greater") if top and rest else (None, 1.0)
                    pvals.append(p); fr_top.append(ft)
                    kv = np.array([items[j]["kappa_val"] for j in idx]); hv = H[i, idx]
                    z = (kv - kv.mean()) / (kv.std() + 1e-9)
                    row[f"dz_{ds}"] = float(z[hv].mean() - z[~hv].mean()) if hv.any() and (~hv).any() else None
                row["freq_top_balanced"] = float(np.mean(fr_top))
                row["freq_rest_balanced"] = float(np.mean([row[f"rest_{ds}"] for ds in by_ds]))
                row["enrichment"] = row["freq_top_balanced"] - row["freq_rest_balanced"]
                row["n_datasets_ge50"] = int(sum(x >= 0.5 for x in fr_top))
                row["fisher_min_p"] = float(min(pvals)); row["_pvals"] = pvals
                dzs = [row[f"dz_{ds}"] for ds in by_ds if row[f"dz_{ds}"] is not None]
                row["dz_kappa_mean"] = float(np.mean(dzs)) if dzs else None
                row["examples"] = [items[j]["id"] for j in np.where(H[i])[0] if items[j]["top_frac"] is not None
                                   and items[j]["top_frac"] <= f][:5]
                rows.append(row)
            allp = [p for r in rows for p in r["_pvals"]]
            q = _bh(allp) if allp else []
            n_ds = len(by_ds)
            for i, r in enumerate(rows):
                r["fisher_min_q"] = float(min(q[i * n_ds:(i + 1) * n_ds])) if len(q) else None
                r.pop("_pvals")
            rows.sort(key=lambda r: (-r["freq_top_balanced"], -r["n_nodes"]))
            results[str(f)] = {"patterns": rows, "pattern_graphs": {k: nx.node_link_data(pats[k], edges="edges") for k in keys},
                               "n_patterns_before_closed": n_all,
                               "n_top": {ds: sum(1 for it in its if it["top_frac"] is not None and it["top_frac"] <= f)
                                         for ds, its in by_ds.items()}}
        results["op_freq"] = _op_freq(items, by_ds)
    write_json(out / "gspan_patterns.json", results)
    report(cfg, results, list(by_ds))
    return results


def _op_freq(items, by_ds):
    """frequência de OPERADORES isolados (presença por grafo) -- tabela separada dos subgrafos."""
    rows = defaultdict(dict)
    for ds, its in by_ds.items():
        top = [it for it in its if it["top_frac"] is not None and it["top_frac"] <= 0.3]
        rest = [it for it in its if it not in top]
        for lab in {d["label"] for it in its for _, d in it["G"].nodes(data=True)}:
            rows[lab][f"top30_{ds}"] = np.mean([any(d["label"] == lab for _, d in it["G"].nodes(data=True)) for it in top]) if top else None
            rows[lab][f"rest_{ds}"] = np.mean([any(d["label"] == lab for _, d in it["G"].nodes(data=True)) for it in rest]) if rest else None
    return {k: {kk: (float(vv) if vv is not None else None) for kk, vv in v.items()} for k, v in rows.items()}


def report(cfg, res, dss):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = Path(cfg["report_dir"]); rep.mkdir(parents=True, exist_ok=True)
    md = ["# Técnica 1 — gSpan: subgrafos frequentes", "",
          "gSpan oficial (betterenvi/gSpan) por base no top-f por κ de **validação**; cada padrão é recontado nos grafos "
          "**direcionados** originais (monomorfismo). freq_top = fração das top da base que contém o padrão; "
          "*balanceada* = média das bases (cada base pesa igual). enrichment = freq_top − freq_demais (balanceadas). "
          "dz_κ = diferença de κ_val padronizado (z na base) entre quem contém e quem não contém. q = Fisher exato "
          "unilateral (top > demais), corrigido por Benjamini–Hochberg.", "",
          "**Aviso:** frequência alta ≠ bloco bom. O que sugere bloco bom é enrichment > 0 com q baixo E dz_κ > 0 em várias bases.", ""]
    for f in [k for k in res if k != "op_freq"]:
        R = res[f]["patterns"]
        md += [f"## Top {float(f):.0%} (n top por base: {res[f]['n_top']})", "", f"{len(R)} padrões fechados (de {res[f].get('n_patterns_before_closed')} minerados; 2–{cfg['gspan']['max_vertices']} nós). "
               "Uma entrada por candidato (a fidelidade de maior κ_val).", ""]
        good = distinct([r for r in R if r["enrichment"] > 0 and (r["dz_kappa_mean"] or 0) > 0], dss, 20)
        md += ["### Mais frequentes (balanceado)", "", md_table(distinct(R, dss, 20), ["pattern", "kind", "freq_top_balanced", "freq_rest_balanced",
                                                                          "enrichment", "n_datasets_ge50", "dz_kappa_mean", "fisher_min_q"]), "",
               "### Enriquecidos nas melhores (enrichment>0 e dz_κ>0), ordenados por nº de bases ≥50%", "",
               md_table(good, ["pattern", "kind", "freq_top_balanced", "enrichment", "n_datasets_ge50", "dz_kappa_mean", "fisher_min_q"]), ""]
        if good:
            md += ["Exemplos de arquiteturas (top) que contêm os 5 primeiros:", ""]
            md += [f"- `{g['pattern']}`: " + ", ".join(f"`{e}`" for e in g["examples"]) for g in good[:5]]
            md += [""]
        _plot_heat(plt, R, dss, rep / f"gspan_freq_top{int(float(f) * 100)}.png", f)
    of = res["op_freq"]
    md += ["## Operadores isolados (presença por grafo, top30 vs demais) — NÃO são subgrafos", "",
           md_table([dict(op=k, **v) for k, v in sorted(of.items())], ["op"] + [f"top30_{d}" for d in dss] + [f"rest_{d}" for d in dss], 2), ""]
    (rep / "02_gspan.md").write_text("\n".join(md) + "\n")


def distinct(R, dss, n):
    """para leitura: entre padrões com o MESMO perfil de frequência nas bases (variações do mesmo bloco), mostra só o maior."""
    seen, out = {}, []
    for r in sorted(R, key=lambda r: (-r["n_datasets_ge50"], -r["freq_top_balanced"], -r["n_nodes"])):
        sig = tuple(round(r[f"top_{d}"], 2) for d in dss) + (round(r["enrichment"], 2),)
        if sig in seen:
            continue
        seen[sig] = 1; out.append(r)
        if len(out) >= n:
            break
    return out


def pretty(p: str) -> str:
    return p.replace("maxpools2", "MaxPool↓2").replace("pw1", "conv1×1").replace("skip1", "atalho1×1").replace(
        "conv3", "conv3×3").replace("conv5", "conv5×5").replace("conv7", "conv7×7")


def _plot_heat(plt, R, dss, path, f):
    sel = distinct(R, dss, 18)
    if not sel:
        return
    M = np.array([[r[f"top_{d}"] for d in dss] for r in sel]); E = np.array([r["enrichment"] for r in sel])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 0.5 * len(sel) + 2.5), gridspec_kw={"width_ratios": [len(dss), 1.6]})
    im = a1.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    import textwrap
    a1.set_xticks(range(len(dss)), dss, rotation=30)
    a1.set_yticks(range(len(sel)), ["\n".join(textwrap.wrap(f"[{r['kind']}] " + pretty(r["pattern"]), 70)) for r in sel], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a1.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > .6 else "black")
    a1.set_title(f"frequência nas top {float(f):.0%} de cada base")
    a2.barh(range(len(sel)), E, color=["#2166ac" if e > 0 else "#b2182b" for e in E]); a2.set_ylim(len(sel) - .5, -.5); a2.set_yticks([])
    a2.axvline(0, color="k", lw=.6); a2.set_title("enrichment\n(top − demais)")
    fig.colorbar(im, ax=a1, fraction=.02); fig.tight_layout(); fig.savefig(path, dpi=140); fig.savefig(path.with_suffix(".svg")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    r = mine(load_config(a.config))
    print({k: len(v["patterns"]) for k, v in r.items() if k != "op_freq"})
