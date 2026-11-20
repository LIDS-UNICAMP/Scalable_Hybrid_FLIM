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
"""spf_tabela.py — refaz spifil_vs_aleatorio.md e o ranking a partir dos CSVs; so le validacao.

`python -m eval.spf_tabela` reescreve a tabela (a sessao `disc_md_spf_tab` chama isso em laco).
`python -m eval.spf_tabela congela r1` grava, das rodadas ate r1, os bracos que seguem para a
proxima: quem o job `survivors: r1` le. Rodadas posteriores nao mexem no que ja foi congelado.

Delta de um braco, por celula (base, fundo, tamanho, kernel, larguras, camada, leitura): em cada
split, kappa do braco menos kappa do melhor controle na MESMA largura real, media dos sorteios.
Melhor controle = o de maior kappa medio nos splits da celula. Intervalo: t de Student pareado
por split. Veredito de um metodo numa base: escolhe a celula em 2 splits, mede no 3o e roda os
3 (validacao cruzada entre splits), porque o maximo de muitas celulas sairia otimista.
Criterio, fixado antes: Delta >= 0,03 com intervalo acima de 0 em pelo menos 3 de 5 bases.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from core.constants import PROJECT_ROOT

MD = os.path.join(PROJECT_ROOT, "A_reports/november/multi_dataset/spifil_vs_aleatorio.md")
BEGIN, END = "<!-- gerado:inicio -->", "<!-- gerado:fim -->"
SURVIVORS = os.environ.get("SPF_SURVIVORS", "/exps/moliveira/queues/spf_sobreviventes.json")
BASES = ["flowers", "helminth_eggs", "helminth_larvae", "Protozoan_subgroup", "aid"]
FOCUS = ["flowers", "Protozoan_subgroup"]
HYPOTHESIS = ["Eggs", "Larvae", "Protozoan", "Eggs_subgroup", "Larvae_subgroup", "Protozoan_subgroup"]
GOOD, NEEDED = 0.03, 3
CELL = ["dataset", "fundo", "image_size", "kernel", "layer", "readout"]
KEEP_CONTROLS = 3
# selecao: (braco sem mapa, oraculo, prefixos dos bracos com o mapa)
SALIENCY = {"cos": ("cos", "cos@m_obj", ("cos@mapa_",)), "fps": ("fps", "fps@m_obj", ("fps@mapa_", "fpsw@mapa_")),
            "sift_nms": ("crus@sift_nms", "crus@m_nms", ("crus@nms_",))}
CHAIN = ["dataset", "fundo", "image_size", "kernel", "split", "seed"]
# rodada de profundidade: nome -> (pool por camada, lado do mapa depois de cada camada a 200 px)
DEEP_BASES = ["Protozoan_subgroup", "flowers", "Eggs_subgroup", "Larvae_subgroup", "helminth_eggs", "aid"]
DEEP = {"cedo": ("2-2-2-2-1-1-1-1", "99, 49, 24, 11, 11, 11, 11, 11"), "esp": ("2-1-2-1-2-1-2-1", "99, 99, 49, 49, 24, 24, 11, 11")}


def sheet(path: str) -> pd.DataFrame | None:
    """CSV de um job vivo: pode estar vazio, ou com a ultima linha pela metade."""
    return pd.read_csv(path, on_bad_lines="skip") if os.path.getsize(path) else None


def load(rounds: list | None = None, prefix: str = "spf") -> pd.DataFrame:
    """Linhas ok de validacao; `rounds` restringe as rodadas lidas."""
    parts = []
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, f"results/svm_disc_md_{prefix}_*.csv"))):
        name = os.path.basename(path)
        if name.endswith("_diag.csv") or (rounds and not any(f"_spf_{r}_" in name for r in rounds)):
            continue
        parts.append(sheet(path))
    parts = [p for p in parts if p is not None and len(p)]
    if not parts:
        return pd.DataFrame()
    data = pd.concat(parts, ignore_index=True)
    data = data[(data.status == "ok") & data.val_kappa.notna()]
    data["fundo"] = data.fundo.fillna("real")
    # relancar refaz celula parcial: a mesma celula conta uma vez
    return data.drop_duplicates(CHAIN + ["arm", "plan", "layer", "readout"], keep="last")


def paired(data: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (braco, celula, split): kappa do braco e de cada controle na mesma largura."""
    control = data[data.family == "controle"][CHAIN + ["layer", "readout", "widths", "arm", "val_kappa", "imp_recall", "par_recall"]]
    control = control.rename(columns={"arm": "control", "val_kappa": "control_kappa", "imp_recall": "control_imp",
                                      "par_recall": "control_par"})
    return data.merge(control, on=CHAIN + ["layer", "readout", "widths"])


def deltas(data: pd.DataFrame) -> pd.DataFrame:
    """Delta por split contra o melhor controle de cada celula, para bracos e controles."""
    pairs = paired(data)
    key = CELL + ["arm", "family", "plan"]
    by_split = pairs.groupby(key + ["control", "split"], dropna=False)[
        ["val_kappa", "val_f1", "control_kappa", "imp_recall", "par_recall", "control_imp", "control_par", "width"]].mean().reset_index()
    means = by_split.groupby(key + ["control"], dropna=False).agg(score=("control_kappa", "mean"), n=("split", "nunique")).reset_index()
    means = means[means.n == means.groupby(key, dropna=False).n.transform("max")]
    best = means.loc[means.groupby(key, dropna=False).score.idxmax()][key + ["control"]]
    out = by_split.merge(best, on=key + ["control"])
    out["delta"] = out.val_kappa - out.control_kappa
    return out


def interval(values: np.ndarray) -> tuple:
    """Media e intervalo t de 95%, pareado por split."""
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / len(values) ** 0.5
    return mean, mean - half, mean + half


def num(x: float, digits: int = 3, sign: bool = False) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:{'+' if sign else ''}.{digits}f}".replace(".", ",")


def span(lo: float, hi: float) -> str:
    return "-" if pd.isna(lo) else f"[{num(lo, 3, True)}; {num(hi, 3, True)}]"


def label(arm: str, family: str) -> str:
    return f"{arm} [oráculo]" if family == "oraculo" else arm


def crossfit(table: pd.DataFrame) -> pd.DataFrame:
    """Por braco e base: celula escolhida em 2 splits, Delta medido no 3o. So celulas com 3 splits."""
    rows = []
    key = ["dataset", "fundo", "arm", "family"]
    for names, part in table[table.family != "controle"].groupby(key, dropna=False):
        wide = part.pivot_table(index=["image_size", "kernel", "plan", "layer", "readout"], columns="split", values="delta")
        wide = wide.dropna()
        if wide.shape[1] < 3 or wide.empty:
            continue
        held = [wide[s].loc[wide.drop(columns=s).mean(axis=1).idxmax()] for s in wide.columns]
        mean, lo, hi = interval(np.array(held))
        rows.append(dict(zip(key, names)) | {"delta": mean, "lo": lo, "hi": hi, "cells": len(wide),
                                             "best": " ".join(map(str, wide.mean(axis=1).idxmax()))})
    return pd.DataFrame(rows)


def ranking(table: pd.DataFrame, bases: list = BASES) -> pd.DataFrame:
    """Metodos pelo Delta medio entre as bases medidas, com o criterio fixado."""
    fit = crossfit(table)
    if fit.empty:
        return fit
    fit = fit[(fit.fundo == "real") & (fit.family == "metodo")]
    fit = fit.assign(good=(fit.delta >= GOOD) & (fit.lo > 0))
    out = fit.groupby("arm").agg(mean_delta=("delta", "mean"), measured=("dataset", "nunique")).reset_index()
    out["good"] = out.arm.map(fit[fit.dataset.isin(bases)].groupby("arm").good.sum()).fillna(0).astype(int)
    return out.sort_values(["good", "mean_delta"], ascending=False)


def rows_table(table: pd.DataFrame) -> list:
    """A tabela crescente: uma linha por braco e celula."""
    head = ["| base | fundo | método | largura | camadas | leitura | kernel | px | κ | F1 | melhor aleatório | Δκ | IC 95% pareado | splits |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    key = CELL + ["arm", "family", "plan", "control"]
    for names, part in table[table.family != "controle"].groupby(key, dropna=False, sort=True):
        cell = dict(zip(key, names))
        mean, lo, hi = interval(part.delta.to_numpy())
        head.append(f"| {cell['dataset']} | {cell['fundo']} | {label(cell['arm'], cell['family'])} | {part.width.mean():.0f} | "
                    f"{cell['layer']} | {cell['readout']} | {cell['kernel']} | {cell['image_size']} | {num(part.val_kappa.mean())} | "
                    f"{num(part.val_f1.mean())} | {cell['control']} | {num(mean, 3, True)} | {span(lo, hi)} | {len(part)} |")
    return head


def focus_table(table: pd.DataFrame) -> list:
    """Primeira resposta: 64 e 256 filtros, flowers e parasito com fundo, os melhores por Delta."""
    head = ["| base | largura | braço | camadas | leitura | κ | aleatório | Δκ | IC 95% pareado |", "|---|---|---|---|---|---|---|---|---|"]
    part = table[table.dataset.isin(FOCUS) & (table.family == "metodo") & (table.fundo == "real")]
    for (dataset, plan), group in part.groupby(["dataset", "plan"], sort=True):
        stat = group.groupby(["arm", "layer", "readout", "control"], dropna=False).agg(
            delta=("delta", "mean"), kappa=("val_kappa", "mean"), ctrl=("control_kappa", "mean"), n=("split", "nunique")).reset_index()
        for _, r in stat[stat.n >= 3].sort_values("delta", ascending=False).head(5).iterrows():
            values = group[(group.arm == r.arm) & (group.layer == r.layer) & (group.readout == r.readout)].delta.to_numpy()
            _, lo, hi = interval(values)
            head.append(f"| {dataset} | {plan} | {r.arm} | {r.layer} | {r.readout} | {num(r.kappa)} | {num(r.ctrl)} ({r.control}) | "
                        f"{num(r.delta, 3, True)} | {span(lo, hi)} |")
    return head


def oracle_table(table: pd.DataFrame) -> list:
    """Hipotese (b): banco misto, so objeto e so fundo, com a impureza a parte."""
    head = ["| base | fundo | largura | leitura | banco | κ | Δκ vs aleatório | IC 95% pareado | recall impureza | Δ | recall demais | Δ |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    arms = ["spifil", "spifil@m_obj", "spifil@m_fundo", "crus", "crus@m_obj", "crus@m_fundo", "zca2", "zca2@m_obj", "zca2@m_fundo"]
    part = table[table.dataset.isin(HYPOTHESIS) & table.arm.isin(arms) & (table.layer == 1)]
    part = pd.concat([part[part.readout == "L"], table[table.dataset.isin(HYPOTHESIS) & table.arm.isin(arms) & (table.layer == 3) & (table.readout == "cat")]])
    for (dataset, fundo, plan, layer, readout), group in part.groupby(["dataset", "fundo", "plan", "layer", "readout"], sort=True):
        for arm in arms:
            mine = group[group.arm == arm]
            if mine.empty:
                continue
            mean, lo, hi = interval(mine.delta.to_numpy())
            name = f"{arm} [oráculo]" if "@m_" in arm else f"{arm} (misto)"
            head.append(f"| {dataset} | {fundo} | {plan} | L{layer} {readout} | {name} | {num(mine.val_kappa.mean())} | {num(mean, 3, True)} | {span(lo, hi)} | "
                        f"{num(mine.imp_recall.mean())} | {num((mine.imp_recall - mine.control_imp).mean(), 3, True)} | "
                        f"{num(mine.par_recall.mean())} | {num((mine.par_recall - mine.control_par).mean(), 3, True)} |")
    return head


def arms_table(table: pd.DataFrame, arms: list) -> list:
    """Bracos lado a lado, nas duas leituras de referencia: L1 e as 3 camadas concatenadas."""
    head = ["| base | largura | leitura | braço | κ | Δκ vs aleatório | IC 95% pareado |", "|---|---|---|---|---|---|---|"]
    part = table[table.arm.isin(arms) & (table.fundo == "real")]
    part = part[((part.layer == 1) & (part.readout == "L")) | ((part.layer == 3) & (part.readout == "cat"))]
    for (dataset, plan, layer, readout), group in part.groupby(["dataset", "plan", "layer", "readout"], sort=True):
        for arm in arms:
            mine = group[group.arm == arm]
            if len(mine):
                mean, lo, hi = interval(mine.delta.to_numpy())
                head.append(f"| {dataset} | {plan} | L{layer} {readout} | {arm} | {num(mine.val_kappa.mean())} | {num(mean, 3, True)} | {span(lo, hi)} |")
    return head


def condition_table(table: pd.DataFrame) -> list:
    """Hipotese (c): a mesma imagem com fundo real e com fundo zerado pelo label."""
    head = ["| base | largura | leitura | braço | Δκ fundo real | IC | Δκ fundo zerado [oráculo] | IC |", "|---|---|---|---|---|---|---|---|"]
    part = table[table.dataset.isin(HYPOTHESIS) & table.arm.isin(["spifil", "crus", "zca2"])]
    part = part[((part.layer == 1) & (part.readout == "L")) | ((part.layer == 3) & (part.readout == "cat"))]
    for (dataset, plan, layer, readout, arm), group in part.groupby(["dataset", "plan", "layer", "readout", "arm"], sort=True):
        cols = []
        for fundo in ("real", "zerado"):
            values = group[group.fundo == fundo].delta.to_numpy()
            mean, lo, hi = interval(values) if len(values) else (float("nan"),) * 3
            cols += [num(mean, 3, True), span(lo, hi)]
        head.append(f"| {dataset} | {plan} | L{layer} {readout} | {arm} | " + " | ".join(cols) + " |")
    return head


def recovery_table(table: pd.DataFrame) -> list:
    """Hipotese (d): quanto do ganho do oraculo volta sem mascara, no mesmo metodo `crus`."""
    head = ["| base | largura | leitura | filtro sem máscara | Δκ | ganho do oráculo | recuperado |", "|---|---|---|---|---|---|---|"]
    filters = ["crus@obj", "crus@sal", "crus@l1obj", "crus@sift_sp", "crus@sift", "crus@sift_nms", "crus@uniao"]
    part = table[table.dataset.isin(HYPOTHESIS) & (table.fundo == "real")]
    part = part[((part.layer == 1) & (part.readout == "L")) | ((part.layer == 3) & (part.readout == "cat"))]
    for (dataset, plan, layer, readout), group in part.groupby(["dataset", "plan", "layer", "readout"], sort=True):
        mean = group.groupby("arm").delta.mean()
        if not {"crus", "crus@m_obj"} <= set(mean.index):
            continue
        gain = mean["crus@m_obj"] - mean["crus"]
        for arm in filters:
            if arm in mean.index:
                share = (mean[arm] - mean["crus"]) / gain if abs(gain) >= 0.005 else float("nan")
                head.append(f"| {dataset} | {plan} | L{layer} {readout} | {arm} | {num(mean[arm], 3, True)} | {num(gain, 3, True)} | "
                            f"{'-' if pd.isna(share) else f'{share:.0%}'} |")
    return head


def saliency_table(table: pd.DataFrame) -> list:
    """Mapa sem rotulo como mascara ou peso, contra a mesma selecao sem mapa e contra o oraculo."""
    head = ["| base | largura | leitura | seleção | mapa como | κ | Δκ vs aleatório | IC 95% pareado | passa | Δκ sem mapa | Δκ oráculo | recuperado |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    part = table[(table.fundo == "real") & (((table.layer == 1) & (table.readout == "L")) | ((table.layer == 3) & (table.readout == "cat")))]
    for (dataset, plan, layer, readout), group in part.groupby(["dataset", "plan", "layer", "readout"], sort=True):
        mean = group.groupby("arm").delta.mean()
        for name, (plain, oracle, prefixes) in SALIENCY.items():
            for arm in sorted(a for a in mean.index if a.startswith(prefixes)):
                mine = group[group.arm == arm]
                delta, lo, hi = interval(mine.delta.to_numpy())
                base, top = mean.get(plain, float("nan")), mean.get(oracle, float("nan"))
                share = (delta - base) / (top - base) if top - base >= 0.005 else float("nan")
                head.append(f"| {dataset} | {plan} | L{layer} {readout} | {name} | {arm.partition('_')[2]} | {num(mine.val_kappa.mean())} | "
                            f"{num(delta, 3, True)} | {span(lo, hi)} | {'sim' if delta >= GOOD and lo > 0 else '-'} | {num(base, 3, True)} | "
                            f"{num(top, 3, True)} | {'-' if pd.isna(share) else f'{share:.0%}'} |")
    return head


def diag_table() -> list:
    """Hipotese (a): onde as sementes caem. O label so entra aqui, como oraculo de diagnostico."""
    parts = [sheet(p) for p in glob.glob(os.path.join(PROJECT_ROOT, "results/svm_disc_md_spf_*_diag.csv"))]
    parts = [p for p in parts if p is not None and len(p)]
    head = ["| base | fundo | ponto | sementes/imagem | em objeto [oráculo] | em impureza | em parasito | recall do objeto | área do objeto | baixa variância (sem máscara) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    if not parts:
        return head
    data = pd.concat(parts).drop_duplicates(["dataset", "split", "image_size", "fundo", "seed", "point"], keep="last")
    means = data.groupby(["dataset", "fundo", "point"], sort=True).mean(numeric_only=True).reset_index()
    for _, r in means.iterrows():
        head.append(f"| {r.dataset} | {r.fundo} | {r.point} | {num(r.per_image, 1)} | {num(r.obj_frac)} | {num(r.obj_frac_imp)} | {num(r.obj_frac_par)} | "
                    f"{num(r.obj_recall)} | {num(r.area_frac)} | {num(r.bg_frac)} |")
    return head


def control_table(table: pd.DataFrame) -> list:
    head = ["| controle | vezes o melhor | κ médio |", "|---|---|---|"]
    pool = table.drop_duplicates(CELL + ["plan", "control", "arm"])
    counts = pool.control.value_counts()
    means = table.groupby("control").control_kappa.mean()
    for arm, n in counts.items():
        head.append(f"| {arm} | {n} | {num(means[arm])} |")
    return head


def depth_curves(table: pd.DataFrame, value: str, per_base: bool) -> list:
    """Uma coluna por camada: a camada d e a rede de d camadas, prefixo da de 8. Sem `per_base`, media das bases."""
    part = table[table.family == "metodo"].assign(largura=lambda t: t.plan.str.split("-").str[0].astype(int))
    if value == "val_kappa":
        part = pd.concat([part, part.assign(arm="melhor aleatório", val_kappa=part.control_kappa)])
    rows = part.groupby(["dataset", "largura", "readout", "arm", "layer"])[value].mean()
    if not per_base:
        rows = rows.groupby(["largura", "readout", "arm", "layer"]).mean()
    wide = rows.unstack("layer")
    names = (["base"] if per_base else []) + ["largura", "leitura", "braço"]
    columns = [*names, *[f"L{n}" for n in wide.columns], "pico"]
    head = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for cell, line in wide.iterrows():
        values = [num(x, 3, value == "delta") for x in line]
        head.append("| " + " | ".join([*map(str, cell), *values, f"L{line.idxmax()}"]) + " |")
    return head


def rank_lines(rank: pd.DataFrame, bases: int) -> list:
    lines = [f"| método | bases boas (de {bases}) | Δκ médio | bases medidas | veredito |", "|---|---|---|---|---|"]
    for _, r in (rank.iterrows() if not rank.empty else []):
        lines.append(f"| {r.arm} | {r.good} | {num(r.mean_delta, 3, True)} | {r.measured} | {'bom' if r.good >= NEEDED else '-'} |")
    return lines


def round_lines(table: pd.DataFrame, title: str) -> list:
    """Veredito, curvas por camada e a tabela de uma rodada; cada braco contra o melhor aleatorio dela."""
    count = table.groupby("dataset").split.nunique()
    return ["", f"#### {title}", "", "Medido: " + ", ".join(f"{d} ({n} splits)" for d, n in count.items()) + ".", "",
            *rank_lines(ranking(table, DEEP_BASES), len(DEEP_BASES)), "",
            "Δκ contra o melhor aleatório, média das bases:", "", *depth_curves(table, "delta", False), "",
            "κ, média das bases:", "", *depth_curves(table, "val_kappa", False), "",
            "Δκ por base:", "", *depth_curves(table, "delta", True), "",
            "κ por base:", "", *depth_curves(table, "val_kappa", True), "",
            "Tabela:", "", *rows_table(table)]


def depth_section() -> list:
    """As rodadas `spfdeep`, uma por agenda de pool, e `spfcombo`, uma por whitening."""
    lines = ["", "### Profundidade: 1 a 8 camadas, sem backprop", "", "Critério e agendas de pool na seção Profundidade, acima."]
    for name, (pools, sides) in DEEP.items():
        data = load(prefix=f"spfdeep_{name}")
        if not data.empty and (data.family == "controle").any():
            lines += round_lines(deltas(data), f"Pools {pools} (`{name}`), lado do mapa {sides}")
    lines += ["", "### Combo: banco único por camada, 3 a 6 camadas, sem backprop", "", "Banco, ablação e whitening na seção Combo, acima."]
    for name, title, between in (("a", "(a) whitening só na leitura", False), ("w", "(b) whitening só entre camadas", True), ("w", "(c) whitening nos dois", False)):
        data = load(prefix=f"spfcombo_{name}")
        if not data.empty and (data.family == "controle").any():
            table = deltas(data)
            table = table[table.readout.str.endswith("_b") == between]
            if not table.empty:
                lines += round_lines(table, title)
    return lines


def refresh() -> None:
    data = load()
    stamp = time.strftime("%d/%m %H:%M UTC", time.gmtime())
    lines = [f"Atualizada {stamp}. Só validação; o teste não é lido. {len(data)} leituras prontas."]
    table = deltas(data) if not data.empty and (data.family == "controle").any() else pd.DataFrame()
    if not table.empty:
        rank = ranking(table)
        lines += ["", "### Ranking", "", f"Bom = Δ ≥ +0,03 com intervalo acima de 0, em ≥ {NEEDED} de 5 bases ({', '.join(BASES)}). Δ com a célula escolhida em 2 splits e medida no 3º.", ""]
        lines += rank_lines(rank, 5)
        lines += ["", "### Primeira resposta: flowers e parasito com fundo", "", *focus_table(table)]
        lines += ["", "### (a) Onde as sementes caem", "", *diag_table()]
        lines += ["", "### (b) Oráculo: banco misto, só objeto, só fundo", "", *oracle_table(table)]
        lines += ["", "### Kernels de borda e objeto contra kernels de fundo, sem máscara", "", *arms_table(table, ["crus", "crus@obj", "crus@fundo"])]
        lines += ["", "### Pontos de superpixel e SIFT", "", *arms_table(table, [
            "crus", "crus@centroid", "crus@geodesic", "crus@sift_sp", "crus@sift", "crus@sift_nms", "crus@uniao",
            "fpsd@sift", "cosd@sift", "bovw_w@sift", "bovw_sil@sift"])]
        lines += ["", "### (c) Fundo real contra fundo zerado", "", *condition_table(table)]
        lines += ["", "### (d) Quanto volta sem máscara", "", *recovery_table(table)]
        lines += ["", "### Saliência sem rótulo: máscara e peso", "", f"Passa = Δκ ≥ {num(GOOD, 2, True)} sobre o melhor aleatório com intervalo acima de 0. Recuperado = (Δκ com mapa - Δκ sem mapa) / (Δκ oráculo - Δκ sem mapa); flowers não tem label, logo não tem oráculo. Mapa escolhido sem label; ver saliencia.md.", "", *saliency_table(table)]
        lines += depth_section()
        lines += ["", "### Controles", "", *control_table(table)]
        lines += ["", "### Tabela", "", *rows_table(table)]
    with open(MD, encoding="utf-8") as handle:
        text = handle.read()
    begin, end = text.index(BEGIN) + len(BEGIN), text.index(END)
    temp = f"{MD}.{os.getpid()}.tmp"   # duas sessoes podem refazer a tabela ao mesmo tempo
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(text[:begin] + "\n" + "\n".join(lines) + "\n" + text[end:])
    os.replace(temp, MD)


def freeze(name: str) -> None:
    """Os sobreviventes da rodada `name`, calculados so com as rodadas ate ela."""
    order = ["r1", "r2", "r3", "r4", "r5"]
    data = load(order[:order.index(name) + 1])
    table = deltas(data)
    # largura livre (k interno) nao tem largura para escalar: fica na tabela, nao segue adiante
    free = set(table[table.plan.str.startswith("0-")].arm)
    methods = [m for m in ranking(table, sorted(table.dataset.unique())).arm if m not in free]
    pairs = paired(data)
    top = pairs.groupby("control").control_kappa.mean().sort_values(ascending=False).index.tolist()
    frozen = {}
    if os.path.exists(SURVIVORS):
        with open(SURVIVORS, encoding="utf-8") as handle:
            frozen = json.load(handle)
    frozen[name] = {"methods": ["spifil"] + [m for m in methods if m != "spifil"],
                    "controls": list(dict.fromkeys(top[:KEEP_CONTROLS] + ["random_z"]))}
    with open(SURVIVORS + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=1)
    os.replace(SURVIVORS + ".tmp", SURVIVORS)
    print(name, frozen[name])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "congela":
        freeze(sys.argv[2])
    refresh()
