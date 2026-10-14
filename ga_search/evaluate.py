# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC — Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Uma avaliacao do GA (§4): config ativa -> SPiFiL -> features -> KMeans -> CH/DBI/Silhouette (linha §5)."""

import contextlib
import hashlib
import math
import io
import json
import sqlite3
import time
import traceback
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from flim.spifil import BudgetExhausted
from ga_search import data, derived, space, spifil_ext, spifil_residual

_threads_set = False


@dataclass
class EvalContext:
    dataset: str
    pct: int
    split: int
    ds_cfg: dict
    space: dict
    ga_cfg: dict
    device: str
    seed: int


def _db(ga_cfg):
    """Conexao nova por chamada (seguro entre processos; nunca vai no pickle)."""
    con = sqlite3.connect(ga_cfg["cache"]["path"], timeout=ga_cfg["cache"]["timeout_s"])
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS evals(key TEXT PRIMARY KEY, row TEXT)")
    return contextlib.closing(con)


def _metrics(Z, ctx):
    """KMeans + as tres metricas; (status, ch, dbi, sil, n_clusters)."""
    g = ctx.ga_cfg
    if not np.isfinite(Z).all():
        return "degenerate", None, None, None, 0
    if g["clustering"]["standardize"]:
        Z = StandardScaler().fit_transform(Z)
    c = g["clustering"]
    # Features colapsadas (ex.: tanh + global_max saturando em +-1): milhares de imagens viram
    # poucos vetores iguais, o KMeans separa os repetidos e DBI~0 / Silhouette~1 saem falsos.
    Zr = Z.round(c["unique_decimals"])
    if len(np.unique(Zr, axis=0)) < c["min_unique_frac"] * len(Z):
        return "degenerate", None, None, None, 0
    lab = KMeans(n_clusters=ctx.ds_cfg["n_classes"], n_init=c["n_init"], max_iter=c["max_iter"],
                 random_state=c["random_state"]).fit_predict(Z)
    k = len(np.unique(lab))
    if k < max(2, math.ceil(c["min_clusters_frac"] * ctx.ds_cfg["n_classes"])):
        return "degenerate", None, None, None, k
    # grupo com 2+ imagens identicas e nenhum espalhamento: mesma nota falsa, so que num grupo
    if any((lab == j).sum() > 1 and np.ptp(Zr[lab == j], axis=0).max() == 0 for j in np.unique(lab)):
        return "degenerate", None, None, None, k
    s = g["silhouette"]
    m = (float(calinski_harabasz_score(Z, lab)), float(davies_bouldin_score(Z, lab)),
         float(silhouette_score(Z, lab, sample_size=min(s["sample_size"], len(Z)), random_state=s["random_state"])))
    return ("ok" if np.isfinite(m).all() else "degenerate"), *m, k


def evaluate_config(cfg: dict, ctx: EvalContext) -> dict:
    """Uma linha de results.csv (§5); nunca levanta."""
    global _threads_set
    g, t0 = ctx.ga_cfg, time.time()
    if not _threads_set:
        torch.set_num_threads(g["parallel"]["threads_per_worker"])
        _threads_set = True
    h = space.config_hash(cfg)
    row = {"dataset": ctx.dataset, "pct": ctx.pct, "split": ctx.split, "stage": None, "gen": None,
           "eval_idx": None, "cfg_hash": h, "cache_hit": False, "status": None, "error": "",
           "ch": None, "dbi": None, "silhouette": None, "objective": g["penalty"], "n_clusters": None,
           "time_s": None, "feat_dim": None, "n_eval": None, "n_train_images": None,
           "cfg_json": json.dumps(cfg, sort_keys=True)}
    key = hashlib.sha1(f"{h}|{ctx.dataset}|{ctx.pct}|{ctx.split}".encode()).hexdigest()
    try:
        with _db(g) as con:
            hit = con.execute("SELECT row FROM evals WHERE key=?", (key,)).fetchone()
        if hit:
            # O objetivo e recalculado: o cache e compartilhado e pode ter sido gravado com outra metrica.
            c = json.loads(hit[0])
            if c["status"] == "ok":
                v = c[g["objective"]]
                c["objective"] = v if g["metrics"][g["objective"]] == "min" else -v
            return {**c, "cache_hit": True}
    except Exception as e:  # cache fora do ar nao derruba a avaliacao
        row["error"] = f"cache: {e!r}"

    n_eff = data.n_classes_eff(ctx.ds_cfg, cfg["paper_classes"])
    vctx = {"n_available": data.n_available(ctx.ds_cfg, ctx.pct, ctx.split, paper_classes=cfg["paper_classes"]),
            "n_classes_eff": n_eff, "max_seeds": g["spifil"]["max_seeds"]}
    n_train, actual_out, out, learn = None, None, io.StringIO(), None
    try:
        ok, why = space.is_valid(cfg, vctx)
        if not ok:
            row.update(status="invalid", error=why)
        else:
            train = data.train_samples(ctx.ds_cfg, ctx.pct, ctx.split, one_per_class=cfg["one_per_class"],
                                       n_images=cfg["n_images"], paper_classes=cfg["paper_classes"], seed=ctx.seed)
            n_train = len(train)
            # nunca avaliar nas imagens de onde os filtros sairam (old_version: validation == treino em perc100)
            used = {x.name for x in train}
            ev = [x for x in data.eval_samples(ctx.ds_cfg, ctx.pct, ctx.split) if x.name not in used]
            build = spifil_residual.build_residual_learner if cfg.get("residual") else spifil_ext.build_learner
            with contextlib.redirect_stdout(out), torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No mask found")
                learn = build(cfg, train, device=ctx.device, leaky_slope=g["spifil"]["leaky_slope"])
                learn.fit()
                actual_out = [int(b.conv.out_channels) for b in learn.model.blocks]
                H = spifil_ext.encode(learn, ev, cfg["r"], batch=g["spifil"]["encode_batch"]).float()
            H = {"flatten": lambda t: t.reshape(len(t), -1), "global_avg": lambda t: t.mean((2, 3)),
                 "global_max": lambda t: t.amax((2, 3))}[cfg["head"]](H)
            Z = H.numpy().astype(np.float32)
            row.update(n_eval=len(Z), feat_dim=Z.shape[1])
            del H
            st, ch, dbi, sil, k = _metrics(Z, ctx)
            v = {"ch": ch, "dbi": dbi, "silhouette": sil}[g["objective"]]
            obj = None if v is None else (v if g["metrics"][g["objective"]] == "min" else -v)
            row.update(status=st, ch=ch, dbi=dbi, silhouette=sil, n_clusters=k,
                       objective=obj if st == "ok" else g["penalty"])
    except BudgetExhausted as e:
        row.update(status="budget", error=repr(e))
    except Exception as e:
        tail = "\n".join(out.getvalue().splitlines()[-5:])
        row.update(status="error", error=f"{e!r}\n{traceback.format_exc(limit=-3)}\n{tail}".strip())
    finally:
        del learn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    row["time_s"] = time.time() - t0
    n_imgs = n_train or (n_eff if cfg["one_per_class"] else cfg["n_images"])
    row["n_train_images"] = n_train
    try:
        der = derived.derived(cfg, {"n_classes": ctx.ds_cfg["n_classes"], "n_classes_eff": n_eff, "pct": ctx.pct,
                                    "n_train_images": n_imgs, "n_train_per_class": n_imgs / n_eff,
                                    "time_s": row["time_s"], "d_max": ctx.space["d_max"]}, actual_out)
        row.update({k: v for k, v in der.items() if k not in row or row[k] is None})
    except Exception as e:  # geometria degenerada de config invalida
        row["error"] = f"{row['error']}\nderived: {e!r}".strip()

    if row["status"] in ("ok", "degenerate", "budget", "invalid"):
        try:
            with _db(g) as con, con:
                con.execute("INSERT OR REPLACE INTO evals VALUES (?, ?)", (key, json.dumps(row, default=float)))
        except Exception as e:
            row["error"] = f"{row['error']}\ncache: {e!r}".strip()
    return row


def evaluate_genome(X: dict, ctx: EvalContext) -> dict:
    """Genoma -> config ativa -> evaluate_config, mais g_<gene> de todos os genes (dormentes inclusive)."""
    genes = {f"g_{n}": getattr(X.get(n), "item", lambda v=X.get(n): v)() for n in ctx.space["genes"]}
    try:
        cfg = space.active_config(X, ctx.space, ctx.ds_cfg)
    except Exception as e:
        row = {"dataset": ctx.dataset, "pct": ctx.pct, "split": ctx.split, "stage": None, "gen": None,
               "eval_idx": None, "cfg_hash": None, "cache_hit": False, "status": "error",
               "error": f"active_config: {e!r}", "objective": ctx.ga_cfg["penalty"]}
        return {**row, **genes}
    row = evaluate_config(cfg, ctx)
    # §5: g_* vem antes de cfg_json e das derivadas
    head = list(row)[:list(row).index("cfg_json")]
    return {**{k: row[k] for k in head}, **genes, **{k: row[k] for k in row if k not in head}}
