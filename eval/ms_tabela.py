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
"""ms_tabela.py — refaz as tabelas de multiescala.md a partir dos CSVs `disc_md_ms_*`; so le validacao.

`python -m eval.ms_tabela` reescreve o bloco gerado (a sessao `disc_md_ms_tab` chama isso em laco).
`python -m eval.ms_tabela congela ctl` grava o hash dos CSVs do aleatorio antes de um metodo rodar;
a tabela confere esse hash a cada volta. Delta, intervalo e pareamento sao os de `eval/spf_tabela.py`:
contra o melhor aleatorio da mesma largura, t de Student pareado pelos 3 splits. Criterio em multiescala.md.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch

from core.constants import PROJECT_ROOT
from eval.spf_tabela import deltas, interval, load, num, sheet, span

MD = os.path.join(PROJECT_ROOT, "A_reports/november/multi_dataset/multiescala.md")
BEGIN, END = "<!-- gerado:inicio -->", "<!-- gerado:fim -->"
FROZEN = "/exps/moliveira/queues/ms_congelado.json"
FIRST = ["flowers", "Protozoan_subgroup", "Eggs_subgroup", "Larvae_subgroup", "aid"]
REST = ["Eggs", "Protozoan", "Larvae", "protozoan_cysts", "remote_sensing", "helminth_eggs", "helminth_larvae"]
GOOD, REGRESSION, NEEDED, TRIAGE = 0.03, -0.03, 7, 3
MAIN, OTHERS = (3, "cat"), [(3, "L"), (1, "L")]
NAMES = {"crus@geodesic": "geodésico cru", "crus@s1": "S1 Gauss + geodésico", "crus@s2": "S2 DoG + geodésico",
         "crus@s3": "S3 extremo DoG + cru", "crus@s4": "S4 extremo DoG + DoG", "crus@s5": "S5 região ∝ escala",
         "trio@s6": "S6 multiescala cru", "crus@s7": "S7 extremo LoG + cru", "quota@s8": "S8 fusão",
         "crus@centroid": "centroide", "crus@sift_nms": "SIFT filtrado", "crus@obj": "objeto sem rótulo", "fps@geodesic": "FPS"}
PAIRS = [("crus@s1", "crus@geodesic", "Gauss contra cru, mesmos pontos"), ("crus@s2", "crus@geodesic", "DoG contra cru, mesmos pontos"),
         ("crus@s4", "crus@s3", "mapa DoG contra cru, mesmos extremos"), ("crus@s5", "crus@s3", "região ∝ escala contra tamanho fixo, mesmos extremos"),
         ("trio@s6", "crus@geodesic", "várias escalas contra uma, mesmos pontos"), ("crus@s7", "crus@s3", "LoG contra DoG, patch cru"),
         ("crus@s3", "crus@geodesic", "extremo DoG contra centro geodésico, patch cru"), ("quota@s8", "crus@geodesic", "fusão contra geodésico")]


def named(arm: str) -> str:
    return NAMES.get(arm, arm)


def per_base(table: pd.DataFrame, layer: int, readout: str) -> pd.DataFrame:
    """Uma linha por (metodo, base): Delta medio dos splits, intervalo e o aleatorio usado."""
    part = table[(table.family == "metodo") & (table.layer == layer) & (table.readout == readout)]
    rows = []
    for (arm, dataset), group in part.groupby(["arm", "dataset"]):
        mean, lo, hi = interval(group.delta.to_numpy())
        rows.append({"arm": arm, "dataset": dataset, "delta": mean, "lo": lo, "hi": hi, "splits": len(group),
                     "kappa": group.val_kappa.mean(), "control": group.control.iloc[0], "control_kappa": group.control_kappa.mean()})
    return pd.DataFrame(rows)


def verdict(mine: pd.DataFrame) -> dict:
    """Contagens da linha da matriz. So base com os 3 splits conta para o veredito."""
    full = mine[mine.splits == 3]
    good = int(((full.delta >= GOOD) & (full.lo > 0)).sum())
    bad = int((full.delta < REGRESSION).sum())
    first = full[full.dataset.isin(FIRST)]
    triage = "-" if len(first) < len(FIRST) else "passa" if (first.delta > 0).sum() >= TRIAGE else "não"
    final = "aberto" if len(full) < len(FIRST + REST) else "bom" if good >= NEEDED and not bad else "não"
    return {"full": len(full), "wins": int((full.delta > 0).sum()), "good": good, "bad": bad, "triage": triage, "final": final}


def matrix(stats: pd.DataFrame) -> list:
    """Linhas: metodos. Colunas: bases. * = Delta >= +0,03 com intervalo acima de 0; ! = regressao."""
    bases = [d for d in FIRST + REST if d in set(stats.dataset)]
    head = ["| método | " + " | ".join(bases) + " | bases | Δ > 0 | boas | regressões | média | mediana | variância | pior | triagem 3/5 | 7/12 |",
            "|---|" + "---|" * (len(bases) + 10)]
    for arm in [a for a in NAMES if a in set(stats.arm)]:
        mine = stats[stats.arm == arm].set_index("dataset", drop=False)
        cells = []
        for base in bases:
            if base not in mine.index:
                cells.append("")
                continue
            r = mine.loc[base]
            mark = "*" if r.delta >= GOOD and r.lo > 0 else "!" if r.delta < REGRESSION else ""
            cells.append(num(r.delta, 3, True) + mark + ("" if r.splits == 3 else f" ({r.splits})"))
        count, d = verdict(mine), mine.delta
        head.append(f"| {named(arm)} | " + " | ".join(cells) + f" | {count['full']} | {count['wins']} | {count['good']} | {count['bad']} | "
                    f"{num(d.mean(), 3, True)} | {num(d.median(), 3, True)} | {num(d.var(), 4)} | {num(d.min(), 3, True)} | {count['triage']} | {count['final']} |")
    return head


def summary(data: pd.DataFrame) -> pd.DataFrame:
    """Por (base, braco): metricas da leitura principal, desvio entre sementes e custo da cadeia."""
    part = data[(data.layer == MAIN[0]) & (data.readout == MAIN[1])]
    keys = ["dataset", "arm"]
    out = part.groupby(keys)[["val_kappa", "val_accuracy", "val_acc", "val_f1"]].mean()
    out["seed_std"] = part.groupby(keys + ["split"]).val_kappa.std().groupby(keys).mean()
    out["draws"] = part.groupby(keys).val_kappa.size()
    chain = data[data.readout == "L"].groupby(keys + ["split", "seed"]).agg(
        init=("select_s", "sum"), infer=("extract_s", "sum"), gpu=("gpu_mib", "max"), rss=("rss_mib", "max"))
    return out.join(chain.groupby(keys).mean()).reset_index()


def metric_cells(r) -> str:
    return (f"{num(r.val_kappa)} | {num(r.val_accuracy)} | {num(r.val_acc)} | {num(r.val_f1)} | {num(r.seed_std)} | {r.draws:.0f} | "
            f"{num(r.init, 1)} | {num(r.infer, 1)} | {r.gpu:.0f} | {r.rss:.0f}")


def controls_lines(sums: pd.DataFrame) -> list:
    head = ["| base | aleatório | κ | acurácia | acurácia balanceada | macro-F1 | desvio entre sementes | leituras | init s | inferência s | GPU MiB | RAM MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    part = sums[~sums.arm.isin(NAMES)]
    for _, r in part.sort_values(["dataset", "arm"]).iterrows():
        head.append(f"| {r.dataset} | {r.arm} | {metric_cells(r)} |")
    return head


def paired_lines(stats: pd.DataFrame, sums: pd.DataFrame) -> list:
    head = ["| base | método | κ | acurácia | acurácia balanceada | macro-F1 | desvio entre sementes | leituras | init s | inferência s | GPU MiB | RAM MiB | melhor aleatório | κ dele | Δκ | IC 95% pareado | splits |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    both = stats.merge(sums, on=["dataset", "arm"])
    for base in [d for d in FIRST + REST if d in set(both.dataset)]:
        for arm in [a for a in NAMES if a in set(both.arm)]:
            for _, r in both[(both.dataset == base) & (both.arm == arm)].iterrows():
                head.append(f"| {base} | {named(arm)} | {metric_cells(r)} | {r.control} | {num(r.control_kappa)} | {num(r.delta, 3, True)} | {span(r.lo, r.hi)} | {r.splits} |")
    return head


def pair_lines(table: pd.DataFrame) -> list:
    """Diferenca direta entre dois bracos, pareada por split."""
    head = ["| comparação | base | Δκ (primeiro − segundo) | IC 95% pareado |", "|---|---|---|---|"]
    part = table[(table.layer == MAIN[0]) & (table.readout == MAIN[1])]
    wide = part.pivot_table(index=["dataset", "split"], columns="arm", values="val_kappa")
    for one, two, title in PAIRS:
        if {one, two} <= set(wide.columns):
            gap = (wide[one] - wide[two]).dropna()
            for base in [d for d in FIRST + REST if d in gap.index.get_level_values(0)]:
                mean, lo, hi = interval(gap.loc[base].to_numpy())
                head.append(f"| {title} | {base} | {num(mean, 3, True)} | {span(lo, hi)} |")
    return head


def scale_lines() -> list:
    """O registro das escalas, somado nas bases: por imagem, antes e depois do NMS."""
    parts = [sheet(p) for p in glob.glob(os.path.join(PROJECT_ROOT, "results/ms_escala/*.csv"))]
    parts = [p for p in parts if p is not None and len(p)]
    head = ["| tipo | oitava | nível | σ | resolução (em 200 px) | ms/imagem | extremos/imagem | após contraste e borda/imagem | após NMS/imagem |", "|---|---|---|---|---|---|---|---|---|"]
    if not parts:
        return head
    data = pd.concat(parts).drop_duplicates(["dataset", "split", "seed", "kind", "octave", "level"])
    for name in ("seconds", "extrema", "candidates", "kept"):
        data[name] = data[name] / data.images
    means = data.groupby(["kind", "octave", "level", "sigma", "height", "width"])[["seconds", "extrema", "candidates", "kept"]].mean().reset_index()
    for _, r in means.iterrows():
        head.append(f"| {r.kind} | {r.octave:.0f} | {r.level:.0f} | {num(r.sigma, 2)} | {r.height:.0f}×{r.width:.0f} | {num(1000 * r.seconds, 2)} | "
                    f"{num(r.extrema, 1)} | {num(r.candidates, 1)} | {num(r.kept, 1)} |")
    return head


def seeds_lines() -> list:
    """Onde as sementes caem; a mascara so entra aqui, como diagnostico."""
    parts = [sheet(p) for p in glob.glob(os.path.join(PROJECT_ROOT, "results/svm_disc_md_ms_*_diag.csv"))]
    parts = [p for p in parts if p is not None and len(p)]
    head = ["| base | ponto | sementes/imagem | baixa variância (sem máscara) | em objeto [oráculo] | área do objeto |", "|---|---|---|---|---|---|"]
    if not parts:
        return head
    data = pd.concat(parts).drop_duplicates(["dataset", "split", "seed", "point"])
    data = data[data.point.isin(["geodesic", "centroid", "sift_nms", "obj", "s3", "s7"])]
    for _, r in data.groupby(["dataset", "point"]).mean(numeric_only=True).reset_index().iterrows():
        head.append(f"| {r.dataset} | {r.point} | {num(r.per_image, 1)} | {num(r.bg_frac)} | {num(r.obj_frac)} | {num(r.area_frac)} |")
    return head


def digest(paths: list) -> dict:
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}


def control_csvs(name: str) -> list:
    return sorted(p for p in glob.glob(os.path.join(PROJECT_ROOT, f"results/svm_disc_md_ms_{name}_*.csv")) if not p.endswith("_diag.csv"))


def freeze(name: str) -> None:
    """Hash do aleatorio, versoes e commits, gravados antes de o metodo correr."""
    frozen = json.load(open(FROZEN, encoding="utf-8")) if os.path.exists(FROZEN) else {}
    head = {cwd: subprocess.run(["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            for cwd in (PROJECT_ROOT, "/dados/home/moliveira/SPiFiL")}
    frozen[name] = {"at": time.strftime("%d/%m %H:%M UTC", time.gmtime()), "csvs": digest(control_csvs(name)), "commits": head,
                    "versions": {"torch": torch.__version__, "numpy": np.__version__, "cv2": cv2.__version__, "sklearn": sklearn.__version__}}
    with open(FROZEN + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=1)
    os.replace(FROZEN + ".tmp", FROZEN)
    print(name, len(frozen[name]["csvs"]), "CSVs", frozen[name]["at"])


def frozen_lines() -> list:
    if not os.path.exists(FROZEN):
        return ["Aleatório ainda não congelado."]
    lines = []
    for name, entry in json.load(open(FROZEN, encoding="utf-8")).items():
        now = digest(control_csvs(name))
        same = now == entry["csvs"]
        lines.append(f"- `{name}`: congelado em {entry['at']}, {len(entry['csvs'])} CSVs, hash {'confere' if same else 'MUDOU'}.")
    return lines


def refresh() -> None:
    data = load(prefix="ms")
    lines = [f"Atualizada {time.strftime('%d/%m %H:%M UTC', time.gmtime())}. Só validação; o teste não é lido. {len(data)} leituras prontas.", "", *frozen_lines()]
    table = deltas(data) if not data.empty and (data.family == "controle").any() else pd.DataFrame()
    if not data.empty:
        sums = summary(data)
        lines += ["", "### Aleatório", "", *controls_lines(sums)]
    if not table.empty and (table.family == "metodo").any():
        stats = per_base(table, *MAIN)
        lines += ["", f"### Matriz método × dataset: L{MAIN[0]} `{MAIN[1]}`, leitura que decide", "",
                  "Δκ contra o melhor aleatório. `*` passa (Δ ≥ +0,03 com intervalo acima de 0), `!` regressão (Δ < −0,03), `(n)` splits medidos quando faltam. As contagens só usam base com os 3 splits.", "",
                  *matrix(stats), "", "### Tabela pareada", "", *paired_lines(stats, sums),
                  "", "### Comparações diretas", "", *pair_lines(table)]
        for layer, readout in OTHERS:
            lines += ["", f"### Matriz em L{layer} `{readout}`, só para consulta", "", *matrix(per_base(table, layer, readout))]
    lines += ["", "### Registro das escalas", "", *scale_lines(), "", "### Onde as sementes caem", "", *seeds_lines()]
    with open(MD, encoding="utf-8") as handle:
        text = handle.read()
    begin, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    temp = f"{MD}.{os.getpid()}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text[:begin] + "\n" + "\n".join(lines) + "\n" + text[end:])
    os.replace(temp, MD)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "congela":
        freeze(sys.argv[2])
    refresh()
