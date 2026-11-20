"""graph_extractor.py -- CNN -> grafo direcionado com a ORDEM REAL do forward (torch.fx), shapes reais (ShapeProp).

- torch.fx.Tracer com módulos-folha: tudo de torch.nn + classes próprias não rastreáveis (DBN, ZCA, SoftHebb...), descobertas
  automaticamente (se o trace falha dentro de um módulo, ele vira folha e tenta de novo).
- Nós: input, cada op real (conv/dw/pw/proj, bn/dbn/zca, ativação, pool, add, cat, mul...), output. Identity/Dropout/getitem
  são contraídos (não mudam o fluxo).
- Atributos: label (coarse: tipo + kernel + stride, usado na mineração), fine (com canais in->out), shape de saída, params.
- Forma canônica: hash Weisfeiler-Lehman (networkx) do grafo rotulado + verificação exata de isomorfismo dentro de cada
  bucket de hash (equivalence classes exatas).
Cada modelo é carregado num subprocesso com o code-root certo (o crescimento B0-B4 usa outro pacote ga_search).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import warnings

import networkx as nx

warnings.filterwarnings("ignore", message="The hashes produced for directed graphs")

from .common import Cache, ResourceMonitor, append_jsonl, key, load_config, read_jsonl, write_json

ACTS = {"ReLU": "relu", "GELU": "gelu", "SiLU": "silu", "SELU": "selu", "ReLU6": "relu6", "LeakyReLU": "lrelu",
        "PReLU": "prelu", "Tanh": "tanh", "Sigmoid": "sigmoid", "Hardswish": "hswish", "Mish": "mish", "ELU": "elu"}
SKIP_MODULES = {"Identity", "Dropout", "Dropout2d", "Flatten"}
FUNC = {"add": "add", "iadd": "add", "cat": "cat", "mul": "mul", "imul": "mul", "sub": "sub", "matmul": "matmul",
        "adaptive_avg_pool2d": "gap", "relu": "relu", "gelu": "gelu", "silu": "silu", "selu": "selu",
        "max_pool2d": "maxpool", "avg_pool2d": "avgpool", "interpolate": "upsample", "sigmoid": "sigmoid", "mean": "mean"}


def _pair(v):
    return v[0] if isinstance(v, (tuple, list)) else v


def module_label(name: str, m) -> tuple[str | None, str | None]:
    import torch.nn as nn
    t = type(m).__name__
    if t in SKIP_MODULES:
        return None, None
    if isinstance(m, nn.Conv2d):
        k, s = _pair(m.kernel_size), _pair(m.stride)
        st = f"s{s}" if s != 1 else ""
        if m.groups > 1 and m.groups == m.in_channels:
            lab = f"dw{k}{st}"
        elif k == 1:
            lab = "skip1" if "skip" in name else f"pw1{st}"  # skip1 = projeção do atalho residual
        else:
            lab = f"conv{k}{st}"
        return lab, f"{lab}:{m.in_channels}>{m.out_channels}" + (f":g{m.groups}" if 1 < m.groups != m.in_channels else "")
    if isinstance(m, nn.Linear):
        return "linear", f"linear:{m.in_features}>{m.out_features}"
    if "Decorrelated" in t:
        return "dbn", "dbn"
    if "BatchNorm" in t:
        return "bn", "bn"
    if "ZCA" in t or "Whiten" in t:
        return "zca", "zca"
    if "SoftHebb" in t or "Hebb" in t:
        return "softhebb", "softhebb"
    if t in ACTS:
        return ACTS[t], ACTS[t]
    if "Pool" in t:
        if "Adaptive" in t:
            return "gap" if "Avg" in t else "gmp", t
        k = _pair(getattr(m, "kernel_size", 0)); s = _pair(getattr(m, "stride", None) or k)
        lab = ("maxpool" if "Max" in t else "avgpool") + (f"s{s}" if s != 1 else "")
        return lab, f"{lab}:k{k}"
    if "Norm" in t:
        return t.lower(), t.lower()
    return t.lower(), t.lower()


class _Tracer:
    def __init__(self, leaf_names):
        import torch.fx as fx
        leaf = set(leaf_names)

        class T(fx.Tracer):
            def is_leaf_module(self, m, qn):
                return type(m).__name__ in leaf or super().is_leaf_module(m, qn) or not list(m.children())
        self.T = T


def trace_module(model, input_shape, max_retry=12):
    """fx trace robusto: o módulo cujo forward não é rastreável vira folha (descoberto pelo traceback)."""
    import torch
    import torch.fx as fx
    leaf, last = set(), None
    for _ in range(max_retry):
        tr = _Tracer(leaf).T()
        try:
            g = tr.trace(model)
            gm = fx.GraphModule(model, g)
            from torch.fx.passes.shape_prop import ShapeProp
            ShapeProp(gm).propagate(torch.randn(*input_shape))
            return gm, sorted(leaf)
        except Exception as e:  # noqa: BLE001
            last = e
            culprit = _culprit(model, e)
            if culprit is None or culprit in leaf:
                break
            leaf.add(culprit)
    raise RuntimeError(f"fx trace falhou: {type(last).__name__}: {last}")


def _culprit(model, exc):
    """classe de módulo (não-torch.nn) mais interna presente no traceback."""
    import traceback
    import torch.nn as nn
    files = [f.filename for f in traceback.extract_tb(exc.__traceback__)]
    cands = {type(m).__name__: m for m in model.modules()
             if type(m).__module__.split(".")[0] != "torch" and list(m.children()) is not None}
    for fn in reversed(files):
        for n, m in cands.items():
            try:
                import inspect
                if inspect.getsourcefile(type(m)) == fn:
                    return n
            except TypeError:
                pass
    # sem pista: torna folha o primeiro módulo customizado com filhos
    for n, m in cands.items():
        if not isinstance(m, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            return n
    return None


def fx_to_digraph(gm) -> nx.DiGraph:
    G = nx.DiGraph(); mods = dict(gm.named_modules()); alias = {}
    order = 0
    for n in gm.graph.nodes:
        shape = None
        tm = n.meta.get("tensor_meta")
        if tm is not None and hasattr(tm, "shape"):
            shape = list(tm.shape)
        lab = fine = None; params = 0; nm = None
        if n.op == "placeholder":
            lab = fine = "input"
        elif n.op == "output":
            lab = fine = "output"
        elif n.op == "call_module":
            m = mods[n.target]; lab, fine = module_label(n.target, m)
            params = sum(p.numel() for p in m.parameters())
        elif n.op in ("call_function", "call_method"):
            nm = getattr(n.target, "__name__", str(n.target)).strip("_")
            lab = FUNC.get(nm)
            fine = lab
        ins = [alias.get(a.name, a.name) for a in n.all_input_nodes]
        ins = [x for x in ins if x in G]
        if lab is None:  # contraído: aponta para a(s) entrada(s)
            alias[n.name] = ins[0] if len(ins) == 1 else (ins[0] if ins else None)
            if alias[n.name] is None:
                alias.pop(n.name)
            continue
        G.add_node(n.name, label=lab, fine=fine, shape=shape, params=int(params), order=order,
                   target=str(n.target) if n.op in ("call_module", "call_function", "call_method") else n.op)
        order += 1
        for u in dict.fromkeys(ins):
            G.add_edge(u, n.name)
    return G


def graph_stats(G: nx.DiGraph) -> dict:
    labs = [d["label"] for _, d in G.nodes(data=True)]
    convs = [d for _, d in G.nodes(data=True) if d["label"].startswith(("conv", "dw", "pw", "proj"))]
    widths = [d["shape"][1] for d in convs if d.get("shape") and len(d["shape"]) == 4]
    return {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "depth": nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else None,
            "n_conv": len(convs), "n_add": labs.count("add"), "n_cat": labs.count("cat"),
            "max_width": max(widths) if widths else None, "mean_width": sum(widths) / len(widths) if widths else None,
            "out_shape": G.nodes[next(n for n, d in G.nodes(data=True) if d["label"] == "output")].get("shape")
            if any(d["label"] == "output" for _, d in G.nodes(data=True)) else None}


def wl_hash(G, attr="label", it=4):
    return nx.weisfeiler_lehman_graph_hash(G, node_attr=attr, iterations=it)


def to_json(G):
    return nx.node_link_data(G, edges="edges")


def from_json(d):
    return nx.node_link_graph(d, directed=True, edges="edges")


# ── carregamento por tipo de checkpoint (no subprocesso) ─────────────────────────────────────────────────────────────
def load_encoder(source: str, ckpt: str):
    import torch
    import torch.nn as nn
    if source == "growth":
        from ga_search.backbones.checkpoint import load_backbone  # code-root = exps/growth/code
        m, _spec = load_backbone(ckpt, "cpu")
        return nn.Sequential(*list(m.blocks))
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    return d["encoder"]


def _worker(args):
    import torch
    torch.set_num_threads(args.threads); torch.set_grad_enabled(False)
    items = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    for it in items:
        row = {"architecture_id": it["architecture_id"], "_key": it["_key"]}
        try:
            enc = load_encoder(it["source"], it["checkpoint_path"]).eval()
            gm, leafs = trace_module(enc, tuple(args.input_shape))
            G = fx_to_digraph(gm)
            gp = Path(args.graph_dir) / f"{it['_key']}.json"
            write_json(gp, to_json(G))
            row.update(ok=True, graph=str(gp), leaf_modules=leafs, wl_coarse=wl_hash(G, "label"),
                       wl_fine=wl_hash(G, "fine"), **graph_stats(G))
        except Exception as e:  # noqa: BLE001
            row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        append_jsonl(args.out, row)


def extract(cfg, recs=None, only_selected=True):
    """Extrai os grafos (cache por arquitetura+checkpoint). Retorna linhas do índice graphs.jsonl."""
    from . import inventory
    recs = recs if recs is not None else inventory.load(cfg, selected_only=only_selected)
    out_dir = Path(cfg["out_dir"]); gdir = out_dir / "graphs"; gdir.mkdir(parents=True, exist_ok=True)
    idx = out_dir / "graphs.jsonl"; cache = Cache(idx)
    todo = defaultdict(list)
    for r in recs:
        if not r.get("checkpoint_path"):
            continue
        k = key("graph-v1", r["checkpoint_path"], cfg["graph"]["input_shape"])
        if cache.get(k) and cache.get(k).get("ok"):
            continue
        todo[r["source"] == "growth"].append({"architecture_id": r["architecture_id"], "source": r["source"],
                                              "checkpoint_path": r["checkpoint_path"], "_key": k})
    W = cfg["graph"]["workers"]
    with ResourceMonitor(out_dir / "resources.jsonl", "graph_extractor"):
        procs = []
        for is_growth, items in todo.items():
            root = cfg["growth_code_root"] if is_growth else cfg["repo_root"]
            chunks = [items[i::W] for i in range(min(W, len(items)))]
            for ci, ch in enumerate(chunks):
                f = out_dir / f"_graph_items_{int(is_growth)}_{ci}.jsonl"
                f.write_text("\n".join(json.dumps(x) for x in ch) + "\n")
                env = dict(os.environ, PYTHONPATH=f"{root}:{cfg['repo_root']}", CUDA_VISIBLE_DEVICES="",
                           OMP_NUM_THREADS=str(cfg["graph"]["torch_threads"]))
                cmd = [cfg["python"], "-m", "universal_model_ga.graph_extractor", "--worker", "--items", str(f),
                       "--out", str(idx), "--graph-dir", str(gdir), "--threads", str(cfg["graph"]["torch_threads"]),
                       "--input-shape", *map(str, cfg["graph"]["input_shape"])]
                procs.append(subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.DEVNULL,
                                              stderr=open(out_dir / f"_graph_err_{int(is_growth)}_{ci}.log", "w")))
        for p in procs:
            p.wait()
    rows = {r["_key"]: r for r in read_jsonl(idx)}
    # junta com o inventário (por id) e marca classes de equivalência exatas
    by_id = {}
    for r in recs:
        if r.get("checkpoint_path"):
            k = key("graph-v1", r["checkpoint_path"], cfg["graph"]["input_shape"])
            if k in rows:
                by_id[r["architecture_id"]] = dict(rows[k], dataset=r["dataset"])
    _equivalence_classes(by_id)
    write_json(out_dir / "graphs_index.json", by_id)
    return by_id


def _equivalence_classes(by_id):
    """bucket pelo hash WL (fine) e confirma isomorfismo exato (rótulo fine) dentro do bucket."""
    buckets = defaultdict(list)
    for aid, r in by_id.items():
        if r.get("ok"):
            buckets[r["wl_fine"]].append(aid)
    cls = 0
    for _, ids in buckets.items():
        reps = []  # (id, grafo)
        for aid in ids:
            G = from_json(json.loads(Path(by_id[aid]["graph"]).read_text()))
            for rid, RG, c in reps:
                if nx.is_isomorphic(G, RG, node_match=lambda a, b: a["fine"] == b["fine"]):
                    by_id[aid]["iso_class"] = c; break
            else:
                by_id[aid]["iso_class"] = cls; reps.append((aid, G, cls)); cls += 1


def load_graph(row) -> nx.DiGraph:
    return from_json(json.loads(Path(row["graph"]).read_text()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--items"); ap.add_argument("--out")
    ap.add_argument("--graph-dir"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--input-shape", type=int, nargs=4, default=[1, 3, 224, 224]); ap.add_argument("--config")
    a = ap.parse_args()
    if a.worker:
        _worker(a)
    else:
        res = extract(load_config(a.config))
        ok = sum(1 for r in res.values() if r.get("ok"))
        print(f"grafos: {ok}/{len(res)} ok; classes de isomorfismo: {len({r.get('iso_class') for r in res.values() if r.get('ok')})}")
