"""leak_stats.py -- vazamento treino→teste por base, para os modelos grandes (FMs) e para os nossos.
FMs: SVM ajustada no treino do split s e testada no teste do MESMO split s (descritores splits_incremental perc100).
  - "nome": imagem de teste listada também no treino do mesmo split;
  - "md5": imagem de teste com arquivo byte-idêntico a alguma imagem do treino do mesmo split (duplicata exata).
Nossos: o encoder foi treinado (SSL, sem rótulos) no treino da busca = split1 (split1.json ou split1_modehb*.json).
  - "visto no SSL": imagem de teste do split s que estava no treino da busca (por nome, mapeado p/ os nomes do split oficial);
  - "md5 SSL": imagem de teste do split s byte-idêntica a alguma imagem do treino da busca.
Saída: <out>/<base>/ga_vazamento.png (uma figura por base) + <out>/vazamento_resumo.md."""
import argparse, csv, hashlib, json, re, sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
from plot_splits_classes import CK  # noqa: E402

D = Path("/dados/home/moliveira/datasets/organized")
DIR = {"flowers": "flowers", "remote_sensing": "remote_sensing", "ham10000": "ham10000", "aid": "aid",
       "coconut_trees": "coconut_trees", "eggs": "Eggs", "larvae": "Larvae", "protozoan": "Protozoan",
       "eggs_raw": "Eggs", "larvae_raw": "Larvae", "protozoan_raw": "Protozoan"}
PAR = {"eggs", "larvae", "protozoan", "eggs_raw", "larvae_raw", "protozoan_raw"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hashes(paths):
    with ProcessPoolExecutor(32) as ex:
        return dict(zip(paths, ex.map(md5, paths, chunksize=64)))


def search_cfg(base):
    ck = CK[base]
    for up in ck.parents:
        for n in ("growth_state.json", "config_resolved.yaml"):
            f = up / n
            if f.is_file():
                d = (json.loads(f.read_text())["config_full"]["run"]["data"] if n.endswith(".json")
                     else yaml.safe_load(f.read_text())["data"])
                ds = d["dataset_config"]
                rt = ds.get("root", "")
                sj = ds["split_json"].replace("{root}", rt).format(split=d.get("split", 1), pct=d.get("percentage", 100))
                return json.loads(Path(sj).read_text()), Path(ds["images_dir"].replace("{root}", rt)), sj
    raise FileNotFoundError(base)


def run(base, out):
    dd = D / DIR[base]
    fm_img = dd / ("raw/orig" if base in PAR else "orig")
    fm = {s: json.loads((dd / f"splits_incremental/split{s}/data_descriptor_perc100.json").read_text()) for s in (1, 2, 3)}
    srch, s_img, sj = search_cfg(base)
    # nomes da busca -> nomes do split oficial (FMs)
    if base in PAR and base.endswith("_raw"):
        _all = {n for s_ in fm for k in fm[s_] for n in fm[s_][k]}
        tofm = lambda n: n if n in _all else re.sub(r"^\d{6}_", "", n, count=1)
    elif base in PAR:
        mp = {r["new_name_pre"]: r["old_name_raw"] for r in csv.DictReader(open(dd / "splits/split1_modehb_mapping.csv"))}
        tofm = lambda n: mp.get(n, n)
    else:
        tofm = lambda n: n
    ssl_tr = {tofm(n) for n in srch["train"]}
    # md5 de tudo (imagens do split oficial + treino da busca)
    allfm = sorted({n for s in fm for k in fm[s] for n in fm[s][k]})
    hf = hashes([str(fm_img / n) for n in allfm]); hf = {Path(k).name: v for k, v in hf.items()}
    hs = hashes([str(s_img / n) for n in srch["train"]]); ssl_md5 = set(hs.values())
    rows = []
    for s in (1, 2, 3):
        tr, te = set(fm[s]["train"]), list(fm[s]["test"])
        trm = {hf[n] for n in tr}
        n_te = len(te)
        r = dict(split=s, n_test=n_te,
                 fm_nome=sum(n in tr for n in te), fm_md5=sum(hf[n] in trm for n in te),
                 ssl_nome=sum(n in ssl_tr for n in te), ssl_md5=sum(hf[n] in ssl_md5 for n in te))
        rows.append(r)
    # duplicatas dentro do próprio teste ou entre classes (informativo)
    by = defaultdict(set)
    for n in allfm: by[hf[n]].add(n)
    dup_groups = sum(1 for v in by.values() if len(v) > 1)
    plot(base, rows, dup_groups, sj, out / base)
    return rows, dup_groups, sj


def plot(base, rows, dupg, sj, dst):
    cats = [("FMs: teste também no treino\ndo mesmo split (nome)", "fm_nome", "#2166ac"),
            ("FMs: teste byte-idêntico a\numa imagem do treino (md5)", "fm_md5", "#67a9cf"),
            ("nossos: teste já visto no SSL\n(treino da busca, split1)", "ssl_nome", "#b2182b"),
            ("nossos: teste byte-idêntico a\numa imagem vista no SSL (md5)", "ssl_md5", "#ef8a62")]
    fig, ax = plt.subplots(figsize=(11, 5.6)); x = np.arange(3); w = 0.2
    for k, (lab, key, c) in enumerate(cats):
        v = [100 * r[key] / r["n_test"] for r in rows]
        b = ax.bar(x + (k - 1.5) * w, v, w, color=c, label=lab, edgecolor="white")
        ax.bar_label(b, labels=[f"{p:.1f}%\n({r[key]})" if r[key] else "0" for p, r in zip(v, rows)], fontsize=8, padding=2)
    ax.set_xticks(x, [f"split {r['split']}\n(teste n={r['n_test']})" for r in rows])
    ax.set_ylabel("% das imagens de TESTE"); ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    ax.set_title(f"{base} — vazamento treino→teste por split", fontsize=13, fontweight="bold", loc="left")
    fig.text(0.01, 0.01, f"FMs: SVM no treino do split s, teste do split s (splits_incremental perc100). Nossos: encoder treinado (SSL) só no "
             f"treino de {Path(sj).name}.\nGrupos de arquivos byte-idênticos no dataset todo: {dupg}.", fontsize=7.5, color="#444")
    fig.tight_layout(rect=[0, 0.06, 1, 1]); dst.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst / "ga_vazamento.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(H.parent / "visualize_GA_copia")); ap.add_argument("bases", nargs="*")
    a = ap.parse_args(); out = Path(a.out)
    md = ["# Vazamento treino→teste por base", "", "% das imagens de teste. FM = mesmo split; nossos = treino SSL da busca (split1).", "",
          "| base | split | n teste | FM nome | FM md5 | nossos nome | nossos md5 | grupos md5 dup. |", "|---|---|---|---|---|---|---|---|"]
    for b in (a.bases or list(CK)):
        try:
            rows, dg, sj = run(b, out)
            for r in rows:
                f = lambda k: f"{100 * r[k] / r['n_test']:.1f}% ({r[k]})"
                md.append(f"| {b} | {r['split']} | {r['n_test']} | {f('fm_nome')} | {f('fm_md5')} | {f('ssl_nome')} | {f('ssl_md5')} | {dg} |")
            print(b, "ok", [(r["split"], r["fm_nome"], r["fm_md5"], r["ssl_nome"], r["ssl_md5"]) for r in rows], "dup", dg, flush=True)
        except Exception as e:
            print(b, "ERRO", type(e).__name__, e, flush=True)
    (out / "vazamento_resumo.md").write_text("\n".join(md) + "\n")
