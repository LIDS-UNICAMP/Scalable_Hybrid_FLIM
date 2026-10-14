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
"""Etapa 3 — leis de escala a partir dos CSVs do GA (§5 do contrato).

Uso (da raiz do REPO):
    python -m ga_search.analyze_laws [--datasets ...] [--metrics ch dbi silhouette] [--laws ...]
                                     [--results-dir DIR] [--config ga_search/config/ga.yaml]
Sem --laws roda todas menos `transfer` (reavalia configs: só com --laws transfer).
Resíduo é gene (§7.7): a lei `residual` compara linhas residual=1 × residual=0 dos mesmos runs; a pasta
opcional <dataset>/residual/pct_*/split_* (run --residual) entra só nessa lei, se existir.

Saídas em <results_dir>/laws/: aggregated.csv (1 linha por run: best do GA + best_<métrica>),
optima.csv (run × métrica: a config ótima de cada métrica, coluna opt_metric), summary.csv
(law, metric, scope, key, formula, r2, stat, verdict, valid_in, fails_in), correlation.csv,
transfer.csv e fragments/<lei>.md. Limiares: bloco `laws:` do ga.yaml (obrigatório).

Ajustes importáveis (plot_laws.py). Pontos com x<=0 ou não finitos são descartados:
    fit_power(x, y, loglog=False) -> dict   # y = a·x^b; loglog=True: MQO em log-log (R²/AIC em log)
    fit_saturation(x, y) -> dict            # y = ymax − c·x^(−b), b>0 (c<0: desce até o piso, p.ex. DBI)
    fit_best(x, y) -> dict                  # o de menor AIC entre os dois, + "model" (power|saturation)
    aic(y, yhat, k) -> float                # n·ln(RSS/n) + 2k
    bootstrap_ci(x, y, fit=fit_power, param="b", groups=None, n_boot=1000, ci=0.95, seed=0, **kw) -> dict
        # reamostra grupos (ex.: splits) com reposição; params ganha estimate, lo, hi
Dict de ajuste: {"params": {...}, "r2", "aic", "formula", "n", "predict": callable(x) -> y}
(NaN/inf quando há pontos de menos ou o ajuste não converge).
"""
import argparse
import itertools
import json
import re
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import curve_fit

from ga_search.constants import GA_CONFIG, NAN, RUN, SIGN


# ----------------------------------------------------------------------------- ajustes
def _xy(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return x[m], y[m]


def _r2(y, yh):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - yh) ** 2).sum() / ss) if ss > 0 else NAN


def aic(y, yhat, k):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(len(y) * np.log(max(((y - yhat) ** 2).sum(), 1e-300) / len(y)) + 2 * k)


def _nofit(names, n):
    return {"params": dict.fromkeys(names, NAN), "r2": NAN, "aic": np.inf, "formula": "", "n": n,
            "predict": lambda x: np.full(np.shape(x), NAN)}


def fit_power(x, y, loglog=False):
    x, y = _xy(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or (loglog and (y <= 0).any()):
        return _nofit(("a", "b"), len(x))
    x0 = np.exp(np.log(x).mean())   # escala x (FLOPs ~ 1e9) para o ajuste ser estável
    u = x / x0
    f = lambda u, a, b: a * u ** b
    try:
        if (y > 0).all():
            b, la = np.polyfit(np.log(u), np.log(y), 1)
            a = np.exp(la)
        else:
            a, b = y.mean(), 0.0
        if not loglog:
            (a, b), _ = curve_fit(f, u, y, p0=(a, b), maxfev=20000)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return _nofit(("a", "b"), len(x))
    yh = f(u, a, b)
    yt, yht = (np.log(y), np.log(yh)) if loglog else (y, yh)
    a = float(a * x0 ** (-b))
    return {"params": {"a": a, "b": float(b)}, "r2": _r2(yt, yht), "aic": aic(yt, yht, 2), "n": len(x),
            "formula": f"y = {a:.4g}·x^{b:.3g}", "predict": lambda x, a=a, b=b: a * np.asarray(x, float) ** b}


def fit_saturation(x, y):
    x, y = _xy(x, y)
    if len(x) < 4 or np.ptp(x) == 0:
        return _nofit(("ymax", "c", "b"), len(x))
    x0 = np.exp(np.log(x).mean())
    u = x / x0
    f = lambda u, ym, c, b: ym - c * u ** (-b)
    lo, hi = y[np.argmin(x)], y[np.argmax(x)]
    try:
        (ym, c, b), _ = curve_fit(f, u, y, p0=(hi, (hi - lo) * u.min() ** 0.5 or 1e-3, 0.5),
                                  bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 10.0]), maxfev=20000)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return _nofit(("ymax", "c", "b"), len(x))
    yh = f(u, ym, c, b)
    ym, c, b = float(ym), float(c * x0 ** b), float(b)
    return {"params": {"ymax": ym, "c": c, "b": b}, "r2": _r2(y, yh), "aic": aic(y, yh, 3), "n": len(x),
            "formula": f"y = {ym:.4g} − {c:.4g}·x^(−{b:.3g})",
            "predict": lambda x, ym=ym, c=c, b=b: ym - c * np.asarray(x, float) ** (-b)}


def fit_best(x, y):
    p, s = fit_power(x, y), fit_saturation(x, y)
    best = s if s["aic"] < p["aic"] else p
    return {**best, "model": "saturation" if best is s else "power", "aic_power": p["aic"], "aic_saturation": s["aic"]}


def bootstrap_ci(x, y, fit=fit_power, param="b", groups=None, n_boot=1000, ci=0.95, seed=0, **kw):
    x, y = np.asarray(x, float), np.asarray(y, float)
    g = np.arange(len(x)) if groups is None else np.asarray(groups)
    ug = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in ug}
    rng = np.random.default_rng(seed)
    full = fit(x, y, **kw)
    est = np.array([fit(x[ii], y[ii], **kw)["params"][param] for ii in
                    (np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]) for _ in range(n_boot))]) \
        if np.isfinite(full["params"][param]) else np.array([NAN])
    lo, hi = np.nanquantile(est, [(1 - ci) / 2, (1 + ci) / 2]) if np.isfinite(est).any() else (NAN, NAN)
    return {**full, "params": {**full["params"], "estimate": full["params"][param], "lo": float(lo), "hi": float(hi)},
            "formula": f"{full['formula']}  ({param} IC{ci:.0%} [{lo:.3g}, {hi:.3g}])"}


# ----------------------------------------------------------------------------- dados
def _csv(p):
    return pd.read_csv(p) if p.exists() and p.stat().st_size else pd.DataFrame()


def _expand(df):
    """Acrescenta os escalares da config ativa (cfg.*): só genes ativos, os dormentes ficam NaN."""
    if df.empty or "cfg_json" not in df:
        return df
    c = pd.json_normalize([json.loads(s) if isinstance(s, str) else {} for s in df["cfg_json"]])
    c = c.drop(columns=[k for k in c if k == "layers"]).add_prefix("cfg.")
    c.index = df.index
    return pd.concat([df, c], axis=1)


def _optima(ev, metrics):
    """Por run e métrica, a avaliação ok de melhor valor (opt_metric = métrica)."""
    out = []
    for m in metrics:
        e = ev.dropna(subset=[m]) if m in ev else ev.iloc[:0]
        if len(e):
            idx = (e[m] * SIGN[m]).groupby([e[k] for k in RUN]).idxmax()
            out.append(e.loc[idx.values].assign(opt_metric=m))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=RUN + ["opt_metric"])


def load(rdir, datasets, metrics, ga, sub=""):
    ev, runs, gens, scal, der = [], [], [], [], []
    for ds in datasets:
        for rd in sorted((rdir / ds).glob(f"{sub}/pct_*/split_*" if sub else "pct_*/split_*")):
            key = {"dataset": ds, "pct": int(rd.parent.name.split("_")[1]), "split": int(rd.name.split("_")[1])}
            r = _csv(rd / "results.csv")
            if r.empty:
                continue
            cols = list(r.columns)
            der = der or [c for c in cols[cols.index("cfg_json") + 1:] if not re.search(r"_l\d+$", c)]
            r = r.assign(**key)
            ok = r[(r["status"] == "ok") & (r["stage"].eq("ga") if "stage" in r else True)]
            bc = yaml.safe_load(open(rd / "best_config.yaml")) if (rd / "best_config.yaml").exists() else {}
            best = ok[ok["cfg_hash"] == (bc or {}).get("cfg_hash")]
            if best.empty:
                tk = _csv(rd / "top_k.csv")
                best = tk[tk["rank"] == tk["rank"].min()].assign(**key) if len(tk) else ok.nsmallest(1, "objective")
            if best.empty:
                continue
            row = best.iloc[0].to_dict()
            row.update({f"best_{m}": (ok[m] * SIGN[m]).max() * SIGN[m] for m in metrics})
            g = _csv(rd / "generations.csv")
            if len(g):
                sr = g["stop_reason"].dropna().astype(str)
                row.update(n_gen=int(g["gen"].max()) + 1, stop_reason=sr.iloc[-1] if len(sr) else "")
                gens.append(g.assign(**key))
            row["objective_metric"] = ga["objective"]
            runs.append(row)
            ev.append(ok)
        sc = _csv(rdir / ds / "scaling" / "results.csv") if not sub else pd.DataFrame()
        if len(sc):
            scal.append(sc.assign(dataset=ds))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=RUN)
    evals = _expand(cat(ev))
    return {"evals": evals, "runs": _expand(pd.DataFrame(runs)), "opt": _optima(evals, metrics),
            "gens": cat(gens), "scal": _expand(cat(scal)), "derived": der, "metrics": metrics, "ga": ga}


# ----------------------------------------------------------------------------- veredito
def _opt(D, m):
    o = D["opt"]
    return o[o["opt_metric"] == m] if len(o) else o


def _groups(df, by):
    return [(dict(zip(by, k)), g) for k, g in df.groupby(by)] if len(df) else []


def _cv(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.std(ddof=0) / abs(s.mean())) if len(s) and s.mean() != 0 else NAN


def _isnum(s):
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _genes(o):
    """Genes ativos do ótimo: escalares da config + perfil da camada 1, só os que variam."""
    cols = [c for c in o if c.startswith("cfg.")] + [c for c in ("N_l1", "k_l1", "dilation_l1",
                                                                  "activation_l1", "pool_type_l1") if c in o]
    return [c for c in cols if o[c].astype(str).nunique() > 1]


def _verdict(t, L):
    """-> (veredito, vale_em, falha_em). Unidade (dataset, pct): vale se a fração de ok entre splits
    >= consistency. LEI: vale em todas e há >= 2 datasets; OBSERVAÇÃO: em parte; NÃO VALE: em nenhuma."""
    if t is None or t.empty:
        return "NÃO VALE", "", ""
    t = t.assign(pct=t["pct"] if "pct" in t else "*", ok=t["ok"].fillna(False).astype(bool))
    g = t.groupby(["dataset", "pct"])["ok"].mean() >= L["consistency"] - 1e-9
    where = lambda sel: "; ".join(f"{d}[{','.join(str(p) for dd, p in sel if dd == d)}]"
                                  for d in dict.fromkeys(d for d, _ in sel))
    v = "LEI" if g.all() and t["dataset"].nunique() >= 2 else "OBSERVAÇÃO" if g.any() else "NÃO VALE"
    return v, where(g[g].index), where(g[~g].index)


def _row(law, metric, key, formula, r2, stat, verdict, scope="all"):
    v, valid, fails = verdict
    return {"law": law, "metric": metric, "scope": scope, "key": key, "formula": formula, "r2": r2,
            "stat": stat, "verdict": v, "valid_in": valid, "fails_in": fails}


def _fits(groups, x, y, L, envelope=None):
    """fit_best por grupo. envelope=m: ajusta a fronteira 'melhor m com custo <= x'."""
    out = []
    for key, g in groups:
        g = g[[x, y]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(x)
        xs, ys = g[x], g[y]
        if envelope:
            ys = (ys * SIGN[envelope]).cummax() * SIGN[envelope]
            keep = ys.ne(ys.shift())
            xs, ys = xs[keep], ys[keep]
        f = fit_best(xs, ys) if len(xs) >= L["min_points"] else None
        p = f["params"] if f else {}
        d = np.sign(p.get("a", NAN) * p.get("b", NAN)) if f and f["model"] == "power" else np.sign(p.get("c", NAN))
        out.append({**key, "n": len(xs), "x_min": xs.min() if len(xs) else NAN, "model": f["model"] if f else "",
                    "r2": f["r2"] if f else NAN, "dir": d, **p,
                    "formula": f["formula"] if f else "pontos insuficientes"})
    return pd.DataFrame(out)


def _judge(t, L):
    """ok = R² >= r2_min com o modelo e a direção da maioria. -> (t, fórmula, R² mediano)."""
    good = t[t["r2"] >= L["r2_min"]]
    if good.empty:
        return t.assign(ok=False), "nenhum ajuste com R² >= r2_min", float(t["r2"].median())
    model, d = good["model"].mode()[0], good["dir"].mode()[0]
    t = t.assign(ok=(t["r2"] >= L["r2_min"]) & (t["model"] == model) & (t["dir"] == d))
    b = t.loc[t["ok"], "b"]
    return t, (f"{model} ({'cresce' if d > 0 else 'decresce'}), b mediano {b.median():.3g} "
               f"[{b.min():.3g}, {b.max():.3g}]"), float(t.loc[t["ok"], "r2"].median())


def _spearman(x, y, L):
    xy = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(xy) < L["min_points"] or xy.nunique().min() < 2:
        return NAN
    return float(stats.spearmanr(xy.iloc[:, 0], xy.iloc[:, 1]).statistic)


def _judge_rho(t, L, label):
    """ok = |ρ| >= rho_min com o sinal da maioria."""
    strong = t[t["rho"].abs() >= L["rho_min"]]
    s = np.sign(strong["rho"]).mode()[0] if len(strong) else 0
    t = t.assign(ok=(t["rho"].abs() >= L["rho_min"]) & (np.sign(t["rho"]) == s))
    return t, f"{label}: ρ mediano {t['rho'].median():+.2f}", \
        f"{int(t['ok'].sum())}/{len(t)} com |ρ|>={L['rho_min']} e sinal {'+' if s > 0 else '−' if s < 0 else '0'}"


def _constancy(o, col, law, m, L):
    """Razão estável? CV dentro de cada dataset (pcts × splits) e CV entre as médias dos datasets."""
    if col not in o or o[col].dropna().empty:
        return None, pd.DataFrame()
    t = o[["dataset", col]].dropna()
    per = t.groupby("dataset")[col].agg(mean="mean", cv=_cv, n="size").reset_index()
    between = _cv(per["mean"])
    v = _verdict(per.assign(ok=per["cv"] <= L["cv_max"]), L)
    if v[0] == "LEI" and not between <= L["cv_max"]:
        v = ("OBSERVAÇÃO", *v[1:])
    return _row(law, m, col, f"{col} ≈ {t[col].mean():.4g} (cv global {_cv(t[col]):.3g})", NAN,
                f"cv entre datasets={between:.3g}; cv dentro (mediana)={per['cv'].median():.3g}", v), per.assign(key=col)


def _dominant(o, col, law, m, L, key=None, note=""):
    """Categoria vencedora: ok = o run escolheu a categoria mais comum."""
    if col not in o or o[col].dropna().empty:
        return None, pd.DataFrame()
    t = o[RUN + [col]].dropna()
    t = t.assign(**{col: t[col].astype(str)})
    top = t[col].mode()[0]
    t = t.assign(ok=t[col] == top)
    tab = t.groupby(["dataset", "pct"])[col].agg(lambda s: s.mode()[0]).rename("vencedor").reset_index()
    return _row(law, m, key or col, f"{key or col} = {top}{note}", NAN, f"{t['ok'].mean():.0%} dos runs",
                _verdict(t, L)), tab.assign(key=key or col)


def _shape(o, col, law, m, L):
    """Perfil de largura/kernel: pirâmide (cresce com a profundidade), funil (diminui) ou constante."""
    if col not in o or o[col].dropna().empty:
        return None, pd.DataFrame()
    t = o[(o["cfg.d"] > 1) & o[col].notna()] if "cfg.d" in o else o[o[col].notna()]
    tol = L["shape_tol"]
    t = t.assign(perfil=np.select([t[col] > 1 + tol, t[col] < 1 - tol], ["pirâmide", "funil"], "constante"))
    return _dominant(t, "perfil", law, m, L, key=f"{col}:perfil", note=f" ({col} mediano {t[col].median():.3g})")


# ----------------------------------------------------------------------------- leis
def law_flops(D, L):
    secs = []
    for m in D["metrics"]:
        rows, tabs = [], []
        for x in ("flops", "activations"):
            t = _fits(_groups(D["evals"], RUN), x, m, L, envelope=m)
            if t.empty:
                continue
            t, formula, r2 = _judge(t, L)
            rows.append(_row("flops", m, x, f"{m} × {x}: {formula}", r2, f"{int(t['ok'].sum())}/{len(t)} runs",
                             _verdict(t, L)))
            tabs.append(t.assign(x=x))
        secs.append((m, rows, pd.concat(tabs) if tabs else pd.DataFrame()))
    return ("A melhor métrica alcançável com custo <= x (fronteira das avaliações ok de cada run) segue "
            "potência y=a·x^b ou saturação y=ymax−c·x^(−b); o modelo é escolhido por AIC, run a run."), secs


def law_superpixels_r2(D, L):
    secs = []
    bs = dict(n_boot=L["n_boot"], ci=L["ci"], seed=L["seed"], loglog=True)
    for m in D["metrics"]:
        o = _opt(D, m)
        o = o.dropna(subset=["cfg.r", "cfg.n_superpixels"]) if "cfg.n_superpixels" in o else o.iloc[:0]
        if o.empty:
            secs.append((m, [], pd.DataFrame()))
            continue
        pool = bootstrap_ci(o["cfg.r"] ** 2, o["cfg.n_superpixels"], groups=o["split"], **bs)
        pp, t = pool["params"], []
        for k, g in _groups(o, ["dataset"]):
            p = (f := bootstrap_ci(g["cfg.r"] ** 2, g["cfg.n_superpixels"], groups=g["split"], **bs))["params"]
            t.append({**k, "n": f["n"], "b": p["estimate"], "lo": p["lo"], "hi": p["hi"], "r2_log": f["r2"],
                      "ic_contem_1": p["lo"] <= 1 <= p["hi"], "ok": p["lo"] <= pp["estimate"] <= p["hi"]})
        t = pd.DataFrame(t)
        v = _verdict(t, L)
        if v[0] == "LEI" and not pool["r2"] >= L["r2_min"]:
            v = ("OBSERVAÇÃO", *v[1:])
        has1 = pp["lo"] <= 1 <= pp["hi"]
        rows = [_row("superpixels_r2", m, "loglog", f"n_sp* = a·(r²)^b: {pool['formula']}; b "
                     f"{'compatível com 1 (área de superpixel constante)' if has1 else 'diferente de 1'}",
                     pool["r2"], f"b={pp['estimate']:.3g} IC[{pp['lo']:.3g}, {pp['hi']:.3g}] (bootstrap sobre splits)", v)]
        secs.append((m, rows, t))
    return ("O n_superpixels ótimo cresce com a área da imagem: n_sp* ∝ (r²)^b. b=1 quer dizer superpixel de "
            "área fixa. Inclinação log-log, IC por bootstrap sobre splits; vale no dataset se o IC dele contém o "
            "b global."), secs


def _const_law(name, cols, statement):
    def law(D, L):
        secs = []
        for m in D["metrics"]:
            res = [r for r in (_constancy(_opt(D, m), c, name, m, L) for c in cols) if r[0]]
            secs.append((m, [r for r, _ in res], pd.concat([t for _, t in res]) if res else pd.DataFrame()))
        return statement, secs
    return law


law_rf_superpixel = _const_law("rf_superpixel", ["rf_sp_ratio"],
                               "rf_cum ótimo / sqrt(sp_area) é uma razão estável (CV <= cv_max) entre datasets e splits.")
law_budget_ratio = _const_law("budget_ratio", ["budget_ratio", "filters_per_class"],
                              "N/D ótimo (camada 1) e filtros por classe são constantes entre datasets (CV <= cv_max).")


def law_pct(D, L):
    secs = []
    for m in D["metrics"]:
        t = _fits(_groups(_opt(D, m), ["dataset", "split"]), "pct", m, L)
        if t.empty:
            secs.append((m, [], t))
            continue
        t["pct_sat"] = np.where(t["model"] == "saturation",
                                t["x_min"] * (1 - L["sat_frac"]) ** (-1 / t.get("b", NAN)), NAN)
        t, formula, r2 = _judge(t, L)
        sat = t.loc[t["ok"], "pct_sat"].median()
        rows = [_row("pct", m, "pct", f"{m} ótimo × pct: {formula}"
                     + (f"; satura ({L['sat_frac']:.0%}) em pct≈{sat:.3g}" if np.isfinite(sat) else ""), r2,
                     f"{int(t['ok'].sum())}/{len(t)} (dataset, split)", _verdict(t, L))]
        secs.append((m, rows, t))
    return ("A melhor métrica do run em função da pct de treino: potência ou saturação (AIC), por (dataset, split); "
            f"ponto de saturação = pct onde se alcança sat_frac da subida até ymax."), secs


def law_genes_pct(D, L):
    secs = []
    for m in D["metrics"]:
        o, rows, tabs = _opt(D, m), [], []
        for c in [c for c in _genes(o) if _isnum(o[c])]:
            t = pd.DataFrame([{**k, "rho": _spearman(g["pct"], g[c], L)} for k, g in _groups(o, ["dataset", "split"])])
            t, formula, stat = _judge_rho(t, L, f"ρ(pct, {c})")
            rows.append(_row("genes_pct", m, c, formula, NAN, stat, _verdict(t, L)))
            tabs.append(t.assign(gene=c))
        tab = pd.concat(tabs).pivot_table(index="gene", columns="dataset", values="rho", aggfunc="median").reset_index() \
            if tabs else pd.DataFrame()
        secs.append((m, rows, tab))
    return ("Cada gene ótimo (ativo) varia monotonicamente com a pct: Spearman ρ(pct, gene) por (dataset, split); "
            "tabela = ρ mediano por dataset."), secs


def law_compound_vs_single(D, L):
    s, sc = D["scal"], D["ga"]["scaling"]
    st = "A FLOPs iguais (mesmo φ, |flops_ratio/flops_target^φ − 1| <= flops_tol), o escalonamento composto vence os simples."
    if s.empty:
        return st, [("sem scaling/results.csv", [], pd.DataFrame())]
    s = s[s["status"] == "ok"]
    s = s[(s["phi"] != 0) & ((s["flops_ratio"] / sc["flops_target"] ** s["phi"] - 1).abs() <= sc["flops_tol"])]
    secs = []
    for m in D["metrics"]:
        q = s.assign(_s=s[m] * SIGN[m]).dropna(subset=["_s"]).groupby(["dataset", "phi", "scale_mode"])["_s"].max().reset_index()
        if q.empty:
            secs.append((m, [], pd.DataFrame()))
            continue
        w = q.loc[q.groupby(["dataset", "phi"])["_s"].idxmax()]
        w = w.assign(pct="φ=" + w["phi"].astype(str), ok=w["scale_mode"].str.startswith("compound"))
        tab = q.assign(**{m: q["_s"] * SIGN[m]}).pivot_table(index=["dataset", "phi"], columns="scale_mode",
                                                             values=m).reset_index()
        tab = tab.merge(w[["dataset", "phi", "scale_mode"]].rename(columns={"scale_mode": "vencedor"}))
        rows = [_row("compound_vs_single", m, "compound",
                     f"composto vence a FLOPs iguais (vencedor mais comum: {w['scale_mode'].mode()[0]})", NAN,
                     f"{w['ok'].mean():.0%} dos (dataset, φ)", _verdict(w, L))]
        secs.append((m, rows, tab))
    return st, secs


def law_layer_mode(D, L):
    secs = []
    for mode, e in D["evals"].groupby("layer_mode"):
        Dm = {**D, "evals": e, "opt": _optima(e, D["metrics"])}
        for sub in ("flops", "superpixels_r2", "rf_superpixel", "budget_ratio", "pct", "genes_pct"):
            for title, rows, tab in LAW_FN[sub](Dm, L)[1]:
                secs.append((f"{mode} · {sub} · {title}",
                             [{**r, "law": "layer_mode", "scope": f"{mode}:{sub}"} for r in rows], tab))
        cols = [c for l in range(1, 8) for c in (f"budget_ratio_l{l}", f"rf_layer_l{l}") if c in e and e[c].notna().any()]
        for m in D["metrics"]:
            res = [r for r in (_constancy(_opt(Dm, m), c, "layer_mode", m, L) for c in cols) if r[0]]
            secs.append((f"{mode} · por camada · {m}", [{**r, "scope": f"{mode}:por_camada"} for r, _ in res],
                         pd.concat([t for _, t in res]) if res else pd.DataFrame()))
    return "Cada lei acima repetida dentro de cada layer_mode (ótimos recalculados no modo) e, por camada, budget_ratio_l*/rf_layer_l*.", secs


def law_layer_profile(D, L):
    secs = []
    for m in D["metrics"]:
        res = [r for r in (_shape(_opt(D, m), c, "layer_profile", m, L) for c in ("growth_w_obs", "growth_k_obs")) if r[0]]
        secs.append((m, [r for r, _ in res], pd.concat([t for _, t in res]) if res else pd.DataFrame()))
    return ("Perfil observado da rede ótima (d>1): pirâmide = largura/kernel cresce com a profundidade (growth_*_obs > "
            "1+shape_tol), funil = diminui (< 1−shape_tol), constante = entre os dois."), secs


def law_growth(D, L):
    e = D["evals"][D["evals"]["layer_mode"] == "crescimento"] if "layer_mode" in D["evals"] else D["evals"].iloc[:0]
    opt, secs = _optima(e, D["metrics"]), []
    for m in D["metrics"]:
        o = opt[opt["opt_metric"] == m]
        res = [_constancy(o, c, "growth", m, L) for c in ("cfg.growth.width_growth", "cfg.growth.superpixel_growth")]
        res += [_shape(o, "cfg.growth.width_growth", "growth", m, L)]
        res += [_dominant(o, c, "growth", m, L) for c in
                ("cfg.growth.kernel_growth", "cfg.growth.dilation_growth", "cfg.growth.pool_schedule")]
        res = [r for r in res if r[0]]
        secs.append((m, [r for r, _ in res], pd.concat([t for _, t in res]) if res else pd.DataFrame()))
    return ("No modo crescimento, os genes de crescimento ótimos (width_growth, kernel_growth, dilation_growth, "
            "superpixel_growth, pool_schedule) são constantes/os mesmos entre datasets; width_growth classificado "
            "em pirâmide/constante/funil."), secs


def law_act_pool_head(D, L):
    secs, e = [], D["evals"]
    for m in D["metrics"]:
        for col in ("activation_l1", "pool_type_l1", "cfg.head"):
            r, tab = _dominant(_opt(D, m), col, "act_pool_head", m, L)
            if not r:
                continue
            q = e.dropna(subset=[col, m, "cfg.d"])
            if len(q):   # por d: a categoria com a melhor métrica em cada (dataset, d)
                q = q.assign(_s=q[m] * SIGN[m]).groupby(["dataset", "cfg.d", col])["_s"].max().reset_index()
                q = q.loc[q.groupby(["dataset", "cfg.d"])["_s"].idxmax()].rename(columns={col: "vencedor"})
                tab = pd.concat([tab, q[["dataset", "cfg.d", "vencedor"]].assign(key=f"{col} por d")])
            secs.append((f"{m} · {col}", [r], tab))
    return ("Qual activation / pool_type (camada 1) / head vence: a mesma categoria em todos os runs? "
            "Tabela: vencedor por (dataset, pct) entre splits e por (dataset, d) entre todas as avaliações."), secs


def law_stability(D, L):
    secs = []
    for m in D["metrics"]:
        o, rows, tabs = _opt(D, m), [], []
        if o.empty:
            continue
        for c in [m] + _genes(o):
            num = _isnum(o[c]) and o[c].nunique() > 3
            per = o.groupby(["dataset", "pct"])[c].agg(_cv if num else (lambda s: float(s.astype(str).nunique() <= 1)))
            per = per.rename("stat").reset_index()
            per["ok"] = per["stat"] <= L["cv_max"] if num else per["stat"] == 1
            rows.append(_row("stability", m, c, f"{'cv' if num else 'mesmo valor'} entre splits", NAN,
                             f"{'cv mediano' if num else 'fração estável'}={per['stat'].median():.3g}", _verdict(per, L)))
            tabs.append(per.assign(key=c))
        tab = pd.concat(tabs).pivot_table(index="key", columns="dataset", values="stat", aggfunc="median").reset_index() \
            if tabs else pd.DataFrame()
        secs.append((m, rows, tab))
    return ("O ótimo é estável entre os 3 splits de cada (dataset, pct): CV <= cv_max (numéricos com > 3 valores) "
            "ou o mesmo valor nos splits (categóricos). Tabela: estatística mediana por dataset."), secs


def law_transfer(D, L):
    from ga_search import data, space
    from ga_search.evaluate import EvalContext, evaluate_config
    path, ga, cdir, runs = D["out"] / "transfer.csv", D["ga"], D["cfg_dir"], D["runs"]
    done = _csv(path)
    seen = set(zip(done["source"], done["target"], done["pct"], done["split"])) if len(done) else set()
    gpus = (ga.get("parallel") or {}).get("gpus") or []
    device, recs = f"cuda:{gpus[0]}" if gpus else "cpu", done.to_dict("records")
    for tgt in runs["dataset"].unique():
        ds_cfg = data.load_dataset_cfg(tgt, str(cdir))
        sp = space.load_space(cdir / "search_space.yaml", ds_cfg)
        have = set(zip(runs.loc[runs["dataset"] == tgt, "pct"], runs.loc[runs["dataset"] == tgt, "split"]))
        for _, src in runs[runs["dataset"] != tgt].iterrows():
            pct, split = int(src["pct"]), int(src["split"])
            if (src["dataset"], tgt, pct, split) in seen or (pct, split) not in have:
                continue
            seed = ga["seeds"]["base"] + zlib.crc32(f"{tgt}|{pct}|{split}".encode()) % 2 ** 31
            res = evaluate_config(json.loads(src["cfg_json"]), EvalContext(
                dataset=tgt, pct=pct, split=split, ds_cfg=ds_cfg, space=sp, ga_cfg=ga, device=device, seed=seed))
            recs.append({"source": src["dataset"], "target": tgt, "pct": pct, "split": split, "cfg_hash": src["cfg_hash"],
                         **{k: res.get(k) for k in ("status", "error", "ch", "dbi", "silhouette", "objective")}})
            pd.DataFrame(recs).to_csv(path, index=False)   # incremental: retoma de onde parou
    t, secs = pd.DataFrame(recs), []
    for m in D["metrics"]:
        if t.empty:
            break
        own = runs[RUN + [f"best_{m}"]].rename(columns={"dataset": "target"})
        x = t.merge(own, on=["target", "pct", "split"])
        x["gap"] = SIGN[m] * (pd.to_numeric(x[m], errors="coerce") - x[f"best_{m}"]) / x[f"best_{m}"].abs()
        x = x.assign(dataset=x["source"] + "→" + x["target"], ok=x["gap"] >= -(1 - L["transfer_frac"]))
        rows = [_row("transfer", m, "gap", f"config transferida fica {x['gap'].median():+.1%} (mediana) do ótimo próprio",
                     NAN, f"{x['ok'].mean():.0%} a menos de {1 - L['transfer_frac']:.0%} do ótimo", _verdict(x, L))]
        secs.append((m, rows, x[["source", "target", "pct", "split", "status", m, f"best_{m}", "gap", "ok"]]))
    return ("A melhor config de um dataset, avaliada nos outros (mesma pct e split), fica a menos de "
            "1−transfer_frac do ótimo próprio do alvo."), secs


def law_correlation(D, L):
    e, ms = D["evals"], D["metrics"]
    cols = [c for c in D["derived"] if c in e and _isnum(e[c])]
    t = pd.DataFrame([{**k, "metric": m, "derived": c, "n": int(g[c].notna().sum()), "rho": _spearman(g[c], g[m] * SIGN[m], L)}
                      for k, g in _groups(e, RUN) for m in ms for c in cols])
    t.to_csv(D["out"] / "correlation.csv", index=False)
    secs = []
    for m in ms:
        tm, rows = t[t["metric"] == m] if len(t) else t, []
        for c, g in (tm.groupby("derived", sort=False) if len(tm) else []):
            g, formula, stat = _judge_rho(g, L, f"ρ({c}, qualidade {m})")
            rows.append(_row("correlation", m, c, formula, NAN, stat, _verdict(g, L)))
        tab = tm.pivot_table(index="derived", columns="dataset", values="rho", aggfunc="median").reset_index() \
            if len(tm) else pd.DataFrame()
        secs.append((m, rows, tab))
    return ("Spearman entre cada derivada e a qualidade (métrica com sinal: ρ>0 = melhora) sobre as avaliações ok "
            "de cada run; tabela = ρ mediano por dataset; correlation.csv = por run."), secs


def law_metric_agreement(D, L):
    e, ms, rows, tabs = D["evals"], D["metrics"], [], []
    for a, b in itertools.combinations(ms, 2):
        t = pd.DataFrame([{**k, "rho": _spearman(g[a] * SIGN[a], g[b] * SIGN[b], L)} for k, g in _groups(e, RUN)])
        if t.empty:
            continue
        t, formula, stat = _judge_rho(t, L, f"ρ(qualidade {a}, qualidade {b})")
        rows.append(_row("metric_agreement", f"{a}×{b}", "rho", formula, NAN, stat, _verdict(t, L)))
        tabs.append(t.assign(par=f"{a}×{b}"))
    o = D["opt"]
    if len(o):
        same = o.groupby(RUN)["cfg_hash"].nunique().eq(1).rename("ok").reset_index()
        rows.append(_row("metric_agreement", "+".join(ms), "mesmo_otimo", f"a mesma config é a ótima de {', '.join(ms)}",
                         NAN, f"{same['ok'].mean():.0%} dos runs", _verdict(same, L)))
    return ("As métricas ordenam as configs do mesmo jeito (ρ de Spearman entre qualidades, por run) e "
            "apontam o mesmo ótimo?"), [("pares de métricas", rows, pd.concat(tabs) if tabs else pd.DataFrame())]


def law_convergence(D, L):
    g, tol, recs = D["gens"], D["ga"]["stop"]["tol"], []
    for k, r in _groups(g, RUN):
        r = r.sort_values("gen")
        sr = r["stop_reason"].dropna().astype(str)
        rec = {**k, "n_gen": int(r["gen"].max()) + 1, "stop_reason": sr.iloc[-1] if len(sr) else "",
               **{c: r[c].sum() for c in ("n_evals_new", "n_cache_hits", "n_invalid") if c in r}}
        for m in D["metrics"] + ["objective"]:
            c = f"{m}_best"
            if c in r and r[c].notna().any():
                fin = r[c].dropna().iloc[-1]
                rec[f"gen_conv_{m}"] = int(r.loc[(r[c] - fin).abs() <= tol * abs(fin), "gen"].iloc[0])
        recs.append(rec)
    t, rows = pd.DataFrame(recs), []
    for m in D["metrics"]:
        c = f"gen_conv_{m}"
        if c in t:
            counts = ", ".join(f"{k}={v}" for k, v in t["stop_reason"].value_counts().items())
            rows.append(_row("convergence", m, "geracoes", f"{m}_best chega a ±tol do final na geração mediana "
                             f"{t[c].median():.3g} de {t['n_gen'].median():.3g}", NAN, f"stop_reason: {counts}",
                             ("OBSERVAÇÃO", "; ".join(t["dataset"].unique()), "")))
    return f"Resumo descritivo de generations.csv: gerações até a melhor métrica ficar a ±tol (={tol}) do final, e stop_reason.", \
        [("runs", rows, t)]


def law_residual(D, L):
    secs, w = [], L["match_log10"]
    E = pd.concat([D["evals"], D["R"]["evals"]], ignore_index=True)
    res = E["residual"].eq(1) if "residual" in E else pd.Series(False, index=E.index)
    P, R = {**D, "evals": E[~res]}, {**D, "evals": E[res], "opt": _optima(E[res], D["metrics"])}
    st = ("Resíduo (concatenação, §7): (1) a FLOPs e feat_dim iguais (mesmo bin log10 de largura match_log10) a melhor "
          "config residual vence a sem resíduo; (2) growth_c_l ótimo (≈2× ou mais? constante entre camadas, datasets e "
          "pcts?); (3) a melhor métrica satura com a profundidade d; (4) flops, budget_ratio e rf_superpixel com resíduo.")
    if R["evals"].empty:
        return st, [("sem runs residuais", [], pd.DataFrame())]
    # (1) resíduo × sem resíduo, bins de (log10 FLOPs, log10 feat_dim) presentes nos dois
    b = lambda e: e.assign(_bf=np.floor(np.log10(e["flops"]) / w), _bd=np.floor(np.log10(e["feat_dim"]) / w))
    both, recs = pd.concat([b(P["evals"]).assign(_r=0), b(R["evals"]).assign(_r=1)]), []
    for k, g in _groups(both, RUN):
        rec = {**k}
        for m in D["metrics"]:   # melhor de cada lado no run, sem casar custo
            bb = (g[m] * SIGN[m]).groupby(g["_r"]).max() * SIGN[m]
            rec.update({f"best_{m}_plain": bb.get(0, NAN), f"best_{m}_res": bb.get(1, NAN)})
        for m in D["metrics"]:
            q = g.assign(_s=g[m] * SIGN[m]).groupby(["_bf", "_bd", "_r"])["_s"].max().unstack("_r").dropna()
            rec.update({f"bins_{m}": len(q), f"win_{m}": (q[1] > q[0]).mean() if len(q) and 1 in q and 0 in q else NAN})
        recs.append(rec)
    t, rows = pd.DataFrame(recs), []
    for m in D["metrics"]:
        tb = t.assign(ok=SIGN[m] * (t[f"best_{m}_res"] - t[f"best_{m}_plain"]) > 0)
        rows.append(_row("residual", m, "best_run", "o melhor residual do run vence o melhor sem resíduo", NAN,
                         f"{int(tb['ok'].sum())}/{len(tb)} runs", _verdict(tb, L), scope="residual:vs_plain"))
        tm = t.assign(ok=t[f"win_{m}"] > 0.5)
        rows.append(_row("residual", m, "vs_plain", f"resíduo vence a FLOPs e feat_dim iguais em {t[f'win_{m}'].median():.0%} "
                         "dos bins (mediana por run)", NAN, f"{int(tm['ok'].sum())}/{len(tm)} runs com vitória > 50%",
                         _verdict(tm, L), scope="residual:vs_plain"))
    secs.append(("resíduo × sem resíduo (CH, DBI, Sil lado a lado)", rows, t))
    for m in D["metrics"]:
        o, rows, tabs = _opt(R, m), [], []
        # (2) growth_l ótimo: growth_c_l = C_l / C_(l-1) efetivo, por camada e média geométrica
        for c in [f"growth_c_l{l}" for l in range(2, 6)] + ["growth_c_geo"]:
            r, tab = _constancy(o, c, "residual", m, L)
            if r:
                rows.append({**r, "scope": "residual:growth_l"}); tabs.append(tab)
        if "growth_c_geo" in o and o["growth_c_geo"].notna().any():
            g2 = o.assign(classe=np.where(o["growth_c_geo"] <= 2 * (1 + L["shape_tol"]), "≈2×", ">2×"))
            r, tab = _dominant(g2[g2["growth_c_geo"].notna()], "classe", "residual", m, L, key="growth_c_geo:classe")
            rows.append({**r, "scope": "residual:growth_l"}); tabs.append(tab)
        # (3) saturação da profundidade: melhor métrica por d, ajuste por run
        e = R["evals"].dropna(subset=[m, "cfg.d"])
        bd = e.assign(_s=e[m] * SIGN[m]).groupby(RUN + ["cfg.d"])["_s"].max().mul(SIGN[m]).rename(m).reset_index()
        t = _fits(_groups(bd, RUN), "cfg.d", m, L)
        if len(t):
            t["d_sat"] = np.where(t["model"] == "saturation", t["x_min"] * (1 - L["sat_frac"]) ** (-1 / t.get("b", NAN)), NAN)
            t, formula, r2 = _judge(t, L)
            sat = t.loc[t["ok"], "d_sat"].median()
            rows.append(_row("residual", m, "depth", f"melhor {m} × d: {formula}"
                             + (f"; satura em d≈{sat:.3g}" if np.isfinite(sat) else ""), r2,
                             f"d ótimo mediano {o['cfg.d'].median():.3g}", _verdict(t, L), scope="residual:depth"))
            tabs.append(t.assign(key="depth"))
        secs.append((f"{m} · growth_l e profundidade", rows, pd.concat(tabs) if tabs else pd.DataFrame()))
    # (4) leis principais com resíduo
    for sub in ("flops", "budget_ratio", "rf_superpixel"):
        for title, rows, tab in LAW_FN[sub](R, L)[1]:
            secs.append((f"residual · {sub} · {title}",
                         [{**r, "law": "residual", "scope": f"residual:{sub}"} for r in rows], tab))
    return st, secs


LAW_FN = {"flops": law_flops, "superpixels_r2": law_superpixels_r2, "rf_superpixel": law_rf_superpixel,
          "budget_ratio": law_budget_ratio, "pct": law_pct, "genes_pct": law_genes_pct,
          "compound_vs_single": law_compound_vs_single, "layer_mode": law_layer_mode,
          "layer_profile": law_layer_profile, "growth": law_growth, "act_pool_head": law_act_pool_head,
          "stability": law_stability, "transfer": law_transfer, "correlation": law_correlation,
          "metric_agreement": law_metric_agreement, "convergence": law_convergence, "residual": law_residual}


# ----------------------------------------------------------------------------- saída
def _md(df):
    if df is None or df.empty:
        return "_(sem dados)_\n"
    d = df.astype(object).where(df.notna(), "").map(lambda v: f"{v:.4g}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(r) + " |" for r in d.values]) + "\n"


def _agreement(rows):
    """Por (scope, key): as métricas dão o mesmo veredito? Onde cada uma vale."""
    df, out = pd.DataFrame(rows), []
    for (scope, key), g in (df.groupby(["scope", "key"], sort=False) if len(df) else []):
        if g["metric"].nunique() < 2:
            continue
        where = "; ".join(f"{m}: {v or '—'}" for m, v in zip(g["metric"], g["valid_in"]))
        if g["verdict"].nunique() == 1:
            out.append(f"- [{scope}] `{key}`: concordam ({g['verdict'].iloc[0]}). Vale em — {where}")
        else:
            out.append(f"- [{scope}] `{key}`: divergem ({', '.join(f'{m}={v}' for m, v in zip(g['metric'], g['verdict']))})."
                       f" Vale em — {where}")
    return "\n".join(out) or "_(nada a comparar)_"


def write_fragment(out, name, statement, secs):
    md = [f"# Lei `{name}`\n", f"**Enunciado.** {statement}\n",
          "Critério: **LEI** = vale em todo (dataset, pct), consistente entre splits, com >= 2 datasets; "
          "**OBSERVAÇÃO** = vale só em parte; **NÃO VALE** = em nenhum.\n"]
    rows = []
    for title, rs, tab in secs:
        md.append(f"## {title}\n")
        for r in rs:
            r2 = f" · R²={r['r2']:.3g}" if np.isfinite(r["r2"]) else ""
            md.append(f"- **{r['verdict']}** [{r['metric']}] `{r['key']}` — {r['formula']}{r2} · {r['stat']}\n"
                      f"  - vale em: {r['valid_in'] or '—'}\n  - falha em: {r['fails_in'] or '—'}")
        md.append("\n" + _md(tab))
        rows += rs
    md += ["## Concordância entre métricas (CH, DBI, Silhouette)\n", _agreement(rows)]
    (out / "fragments" / f"{name}.md").write_text("\n".join(md) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=GA_CONFIG)
    ap.add_argument("--results-dir", help="sobrepõe ga.yaml: results_dir")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=list(SIGN), choices=list(SIGN))
    ap.add_argument("--laws", "--law", nargs="+", choices=list(LAW_FN))
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    ga = yaml.safe_load(open(a.config))
    L = ga["laws"]
    rdir = Path(a.results_dir or ga["results_dir"])
    datasets = a.datasets or sorted(p.name for p in rdir.iterdir() if p.is_dir() and any(p.glob("pct_*")))
    D = load(rdir, datasets, a.metrics, ga)
    D["R"] = load(rdir, datasets, a.metrics, ga, sub="residual")
    out = rdir / "laws"
    (out / "fragments").mkdir(parents=True, exist_ok=True)
    D.update(out=out, cfg_dir=Path(a.config).parent)
    D["runs"].to_csv(out / "aggregated.csv", index=False)
    D["opt"].to_csv(out / "optima.csv", index=False)
    laws, rows = a.laws or [k for k in LAW_FN if k != "transfer"], []
    for name in laws:
        rows += write_fragment(out, name, *LAW_FN[name](D, L))
        print(f"[ok] {name}")
    p, new = out / "summary.csv", pd.DataFrame(rows)
    old = _csv(p)
    pd.concat([old[~old["law"].isin(laws)] if len(old) else old, new], ignore_index=True).to_csv(p, index=False)
    print(f"{len(D['runs'])} runs, {len(new)} linhas de lei -> {out}")


if __name__ == "__main__":
    main()
