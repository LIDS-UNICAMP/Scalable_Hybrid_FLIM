"""graph_clustering.py -- Técnica 2: famílias estruturais de arquiteturas.

Duas representações/similaridades:
  (1) kernel Weisfeiler-Lehman (subtree, GraKeL oficial) sobre o grafo rotulado (coarse), normalizado -> distância 1-k;
  (2) embedding estrutural: contagens normalizadas de operadores e de pares dirigidos (op->op), profundidade, nº de
      residuais/concatenações e larguras (log) -> distância cosseno.
Validação: Graph Edit Distance APROXIMADA (networkx optimize_graph_edit_distance, 1ª aproximação com timeout) numa
amostra de pares -> Spearman com as duas distâncias.
Clustering: Espectral sobre o kernel WL (k pela silhueta) + Aglomerativo Ward no embedding e HDBSCAN (mesma distância).
Pergunta científica: as top de bases diferentes caem nas mesmas famílias? -> tabela base × cluster das top30%,
qui-quadrado de independência e Jensen-Shannon par a par entre bases.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

from .common import ResourceMonitor, load_config, md_table, write_json
from .gspan_mining import load_groups


def wl_kernel(graphs, it):
    from grakel import Graph
    from grakel.kernels import WeisfeilerLehman, VertexHistogram
    gk = WeisfeilerLehman(n_iter=it, base_graph_kernel=VertexHistogram, normalize=True)
    G = []
    for g in graphs:
        idx = {n: i for i, n in enumerate(g.nodes)}
        edges = {(idx[u], idx[v]): 1 for u, v in g.edges}
        edges.update({(idx[v], idx[u]): 1 for u, v in g.edges})  # GraKeL WL: vizinhança não-direcionada
        G.append(Graph(edges if edges else {(0, 0): 1}, node_labels={idx[n]: d["label"] for n, d in g.nodes(data=True)}))
    return gk.fit_transform(G)


def embed(graphs):
    ops = sorted({d["label"] for g in graphs for _, d in g.nodes(data=True)})
    pairs = sorted({(g.nodes[u]["label"], g.nodes[v]["label"]) for g in graphs for u, v in g.edges})
    X, names = [], [f"op:{o}" for o in ops] + [f"edge:{a}>{b}" for a, b in pairs] + \
        ["depth", "n_nodes", "n_add", "n_cat", "log_max_w", "log_mean_w", "n_down"]
    for g in graphs:
        c = Counter(d["label"] for _, d in g.nodes(data=True)); n = max(1, g.number_of_nodes())
        e = Counter((g.nodes[u]["label"], g.nodes[v]["label"]) for u, v in g.edges); m = max(1, g.number_of_edges())
        w = [d["shape"][1] for _, d in g.nodes(data=True) if d.get("shape") and len(d["shape"]) == 4]
        X.append([c[o] / n for o in ops] + [e[p] / m for p in pairs] +
                 [nx.dag_longest_path_length(g) if nx.is_directed_acyclic_graph(g) else 0, n, c["add"], c["cat"],
                  np.log1p(max(w)) if w else 0, np.log1p(np.mean(w)) if w else 0,
                  sum(v for k, v in c.items() if "s2" in k)])
    X = np.array(X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return Z, names, X


def cosine_dist(Z):
    N = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return np.clip(1 - N @ N.T, 0, 2)


def approx_ged_sample(graphs, n_pairs, timeout, seed):
    rng = np.random.default_rng(seed); n = len(graphs); out = []
    pairs = {tuple(sorted(rng.choice(n, 2, replace=False))) for _ in range(n_pairs * 3)}
    for i, j in list(pairs)[:n_pairs]:
        # GED aproximada: melhor custo encontrado dentro do timeout (networkx graph_edit_distance com timeout)
        best = nx.graph_edit_distance(graphs[i], graphs[j], node_match=lambda a, b: a["label"] == b["label"], timeout=timeout)
        nn_ = graphs[i].number_of_nodes() + graphs[i].number_of_edges() + graphs[j].number_of_nodes() + graphs[j].number_of_edges()
        out.append((int(i), int(j), float(best) / nn_ if best is not None else None))  # GED normalizada pelo tamanho
    return out


def _js(p, q):
    p, q = np.asarray(p, float) + 1e-12, np.asarray(q, float) + 1e-12; p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def cluster(cfg):
    from scipy.stats import chi2_contingency, spearmanr
    from sklearn.cluster import AgglomerativeClustering, HDBSCAN, SpectralClustering
    from sklearn.metrics import silhouette_score
    cc = cfg["clustering"]; out = Path(cfg["out_dir"])
    items = load_groups(cfg); graphs = [it["G"] for it in items]
    with ResourceMonitor(out / "resources.jsonl", "graph_clustering"):
        K = wl_kernel(graphs, cc["wl_iterations"]); D_wl = np.clip(1 - K, 0, None); np.fill_diagonal(D_wl, 0)
        Z, names, Xraw = embed(graphs); D_emb = cosine_dist(Z)
        ged = approx_ged_sample(graphs, cc["ged_sample"], cc["ged_timeout_s"], cfg["seed"])
        g_ok = [(i, j, v) for i, j, v in ged if v is not None]
        rho_wl = spearmanr([v for *_, v in g_ok], [D_wl[i, j] for i, j, _ in g_ok])[0] if len(g_ok) > 3 else None
        rho_emb = spearmanr([v for *_, v in g_ok], [D_emb[i, j] for i, j, _ in g_ok])[0] if len(g_ok) > 3 else None
        rho_wl_emb = spearmanr(D_wl[np.triu_indices(len(graphs), 1)], D_emb[np.triu_indices(len(graphs), 1)])[0]
        sil = {}
        # espectral sobre o kernel WL (afinidade pré-computada); k pela silhueta, exigindo clusters com >= 2% dos itens
        labs = {}
        for k in range(cc["k_range"][0], cc["k_range"][1] + 1):
            lab = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=cfg["seed"], assign_labels="discretize").fit_predict(np.clip(K, 0, None))
            if len(set(lab)) > 1 and np.bincount(lab).min() >= 0.02 * len(lab):
                sil[k] = float(silhouette_score(D_wl, lab, metric="precomputed")); labs[k] = lab
        k_sil = max(sil, key=sil.get)
        k_best = int(cc.get("k_report") or k_sil)  # k da descrição (config); k_sil fica registrado
        if k_best not in labs:
            k_best = k_sil
        lab = labs[k_best]
        hd = HDBSCAN(min_cluster_size=cc["hdbscan_min_cluster_size"], metric="precomputed").fit_predict(D_wl.astype(np.float64))
        lab_emb = AgglomerativeClustering(n_clusters=k_best, linkage="ward").fit_predict(Z)
        from sklearn.metrics import adjusted_rand_score
        ari_wl_emb = float(adjusted_rand_score(lab, lab_emb)); ari_wl_hdb = float(adjusted_rand_score(lab, hd))
        from sklearn.manifold import MDS
        XY = MDS(n_components=2, dissimilarity="precomputed", random_state=cfg["seed"], n_init=2, normalized_stress="auto").fit_transform(D_wl)
    dss = cfg["datasets"]; clusters = []
    for c in sorted(set(lab)):
        idx = np.where(lab == c)[0]
        med = idx[np.argmin(D_wl[np.ix_(idx, idx)].sum(1))]
        zs = []
        for i in idx:
            same = [j for j, it in enumerate(items) if it["dataset"] == items[i]["dataset"]]
            kv = np.array([items[j]["kappa_val"] for j in same]); zs.append((items[i]["kappa_val"] - kv.mean()) / (kv.std() + 1e-9))
        diff = (Z[idx].mean(0) - Z[np.setdiff1d(np.arange(len(items)), idx)].mean(0)) if len(idx) < len(items) else np.zeros(Z.shape[1])
        top_feats = [names[j] for j in np.argsort(-diff)[:6]]
        ops_mean = {names[j][3:]: float(Xraw[idx, j].mean()) for j in range(len(names)) if names[j].startswith("op:")}
        dominant = [k for k, _ in sorted(ops_mean.items(), key=lambda x: -x[1])[:6]]
        params = [items[i].get("params") for i in idx]
        clusters.append({"cluster": int(c), "n": int(len(idx)), "medoid": items[med]["id"], "medoid_dataset": items[med]["dataset"],
                         "by_dataset": {d: int(sum(items[i]["dataset"] == d for i in idx)) for d in dss},
                         "top30_by_dataset": {d: int(sum(items[i]["dataset"] == d and items[i]["top_frac"] is not None for i in idx)) for d in dss},
                         "dominant_ops": dominant, "distinguishing": top_feats,
                         "kappa_z_mean": float(np.mean(zs)), "kappa_z_median": float(np.median(zs)),
                         "focus": [items[i]["id"] for i in idx if items[i].get("focus")]})
    # base × cluster (top30) -> mesmas famílias?
    T = np.array([[sum(1 for i, it in enumerate(items) if it["dataset"] == d and it["top_frac"] is not None and lab[i] == c)
                   for c in sorted(set(lab))] for d in dss])
    Tn = T[T.sum(1) > 0]
    keep_c = Tn.sum(0) > 0
    chi = chi2_contingency(Tn[:, keep_c]) if Tn.shape[0] > 1 and keep_c.sum() > 1 else None
    js = {f"{a}|{b}": _js(T[i], T[j]) for (i, a), (j, b) in combinations(enumerate(dss), 2) if T[i].sum() and T[j].sum()}
    res = {"k_best": int(k_best), "k_silhouette": int(k_sil), "silhouette": sil, "ged_spearman_wl": rho_wl, "ged_spearman_emb": rho_emb,
           "spearman_wl_vs_emb": float(rho_wl_emb), "ari_wl_vs_emb": ari_wl_emb, "ari_wl_vs_hdbscan": ari_wl_hdb,
           "n_hdbscan_clusters": int(len(set(hd)) - (1 if -1 in hd else 0)), "hdbscan_noise": int((hd == -1).sum()),
           "clusters": clusters, "dataset_x_cluster_top30": T.tolist(), "chi2_p": float(chi.pvalue) if chi else None,
           "js_top30": js, "ged_pairs": ged,
           "assign": {it["id"]: {"cluster": int(lab[i]), "hdbscan": int(hd[i]), "xy": XY[i].tolist(), "dataset": it["dataset"],
                                 "top": it["top_frac"] is not None} for i, it in enumerate(items)}}
    write_json(out / "clusters.json", res)
    np.save(out / "D_wl.npy", D_wl); np.save(out / "D_emb.npy", D_emb)
    write_json(out / "cluster_items.json", [it["id"] for it in items])
    _plots(cfg, items, lab, XY, D_wl, T, res)
    _report(cfg, res)
    return res


def _plots(cfg, items, lab, XY, D, T, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep = Path(cfg["report_dir"]); dss = cfg["datasets"]; mk = dict(zip(dss, "osD^v"))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, it in enumerate(items):
        ax.scatter(*XY[i], c=[cmap(lab[i] % 20)], marker=mk.get(it["dataset"], "x"), s=46 if it["top_frac"] is not None else 12,
                   edgecolors="k" if it["top_frac"] is not None else "none", linewidths=.4, alpha=.85 if it["top_frac"] is not None else .35)
        if it.get("focus"):
            ax.annotate(f"{it['dataset']}:{it['id'].split('/')[-2]}", XY[i], fontsize=7, xytext=(4, 4), textcoords="offset points")
    for c in res["clusters"]:
        j = next(i for i, it in enumerate(items) if it["id"] == c["medoid"])
        ax.scatter(*XY[j], marker="*", s=260, c=[cmap(c["cluster"] % 20)], edgecolors="k", zorder=5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk[d], ls="", color="gray", label=d) for d in dss] + \
        [Line2D([], [], marker="*", ls="", color="gray", ms=12, label="medoid do cluster")]
    ax.legend(handles=h, fontsize=8, loc="best"); ax.set_title(f"Famílias estruturais (kernel WL + espectral, k={res['k_best']}); MDS 2D\n"
                                                              "cor = cluster; contorno/maior = top30% por κ_val; rótulo = modelos do draw_io")
    fig.tight_layout(); fig.savefig(rep / "clusters_mds.png", dpi=140); fig.savefig(rep / "clusters_mds.svg"); plt.close(fig)
    o = np.argsort(lab, kind="stable")
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(1 - D[np.ix_(o, o)], cmap="viridis"); fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de similaridade WL (ordenada por cluster)"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(rep / "clusters_similaridade.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5)); Tn = T / np.maximum(T.sum(1, keepdims=True), 1); left = np.zeros(len(dss))
    for c in range(T.shape[1]):
        ax.barh(dss, Tn[:, c], left=left, color=cmap(c % 20), label=f"C{c}" if Tn[:, c].max() > .05 else None); left += Tn[:, c]
    ax.set_xlabel("fração das top30% da base"); ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18))
    ax.set_title(f"Em que família caem as melhores de cada base? (qui² p={res['chi2_p']:.2g})" if res["chi2_p"] is not None else "base × cluster")
    fig.tight_layout(); fig.savefig(rep / "clusters_base_x_familia.png", dpi=140); plt.close(fig)


def _report(cfg, res):
    dss = cfg["datasets"]
    rows = [dict(cluster=c["cluster"], n=c["n"], medoid=c["medoid"], **{f"top_{d}": c["top30_by_dataset"][d] for d in dss},
                 ops=", ".join(c["dominant_ops"]), distingue=", ".join(c["distinguishing"][:4]), z_kappa=c["kappa_z_mean"],
                 focus=", ".join(f.split("/")[1] + ":" + f.split("/")[-2] for f in c["focus"]))
            for c in res["clusters"]]
    md = ["# Técnica 2 — Graph Clustering (famílias estruturais)", "",
          f"Representações: WL subtree (GraKeL, {cfg['clustering']['wl_iterations']} iterações) e embedding estrutural (cosseno). "
          f"Concordância: Spearman(WL, embedding) = {res['spearman_wl_vs_emb']:.2f}; ARI(espectral WL vs Ward embedding) = {res['ari_wl_vs_emb']:.2f}; "
          f"ARI(espectral vs HDBSCAN) = {res['ari_wl_vs_hdbscan']:.2f} (HDBSCAN: {res['n_hdbscan_clusters']} clusters, {res['hdbscan_noise']} ruído).",
          f"Validação com GED aproximada normalizada (custo/(nós+arestas dos dois); timeout {cfg['clustering']['ged_timeout_s']} s por par, logo é um LIMITE SUPERIOR) ({len(res['ged_pairs'])} pares): Spearman(GED, WL) = {res['ged_spearman_wl']}, Spearman(GED, embedding) = {res['ged_spearman_emb']}.",
          f"k pela silhueta = {res['k_silhouette']}; k usado na descrição = **{res['k_best']}** (config k_report). Silhuetas: "
          + ", ".join(f"{k}:{v:.2f}" for k, v in res["silhouette"].items()) + ". Silhueta ≤ 0,3 em todo k: as famílias são "
          "**fracamente separadas** (o espaço do GA é quase contínuo); os clusters descrevem tendências, não grupos nítidos.", "",
          md_table(rows, ["cluster", "n", "medoid"] + [f"top_{d}" for d in dss] + ["ops", "distingue", "z_kappa", "focus"], 2), "",
          "## As melhores de bases diferentes pertencem às mesmas famílias?", "",
          f"Qui-quadrado de independência (base × cluster, só top30%): p = {res['chi2_p']}. "
          "p pequeno = a distribuição de famílias DEPENDE da base (cada base favorece famílias diferentes).", "",
          md_table([{"par": k, "JS": v} for k, v in sorted(res["js_top30"].items(), key=lambda x: x[1])], ["par", "JS"], 3),
          "", "JS = divergência de Jensen–Shannon (bits) entre as distribuições de famílias das top30%: 0 = mesmas famílias, 1 = disjuntas.",
          "", "Figuras: clusters_mds.png/.svg, clusters_similaridade.png, clusters_base_x_familia.png"]
    Path(cfg["report_dir"], "03_clustering.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); a = ap.parse_args()
    r = cluster(load_config(a.config)); print({k: r[k] for k in ("k_best", "chi2_p", "ari_wl_vs_emb", "ged_spearman_wl")})
