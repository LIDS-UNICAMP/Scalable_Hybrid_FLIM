# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   FEEC — School of Electrical and        ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀           Computer Engineering             ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · FEEC · 2026                  ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""
Curriculo SPiFiL hibrido numa linha do tempo unica: cada estagio comeca onde o anterior parou.

Companheiro de ``plot_partial_train_spifil_hybrid.py``, que desenha os estagios sobrepostos
(todos partindo da epoca 0). Aquele painel compara estagio contra estagio; este mostra a
trajetoria continua, e por isso responde uma pergunta que o outro nao consegue: ONDE o kappa
cai quando o encoder ganha uma camada.

A diferenca nao e so cosmetica. O ponto herdado --- ``flim_ref_svm_kappa``, medido em
``on_fit_start`` antes do primeiro gradiente do estagio --- fica so no summary do W&B e nunca
entra na serie por epoca, entao o painel sobreposto e fisicamente incapaz de mostra-lo. Aqui
ele e lido do ``run_metadata.json`` do treino e desenhado como o primeiro ponto de cada
estagio, ligado ao fim do estagio anterior por um segmento cinza. E nesse segmento que a
queda do crescimento aparece --- e ela acontece ANTES de o estagio treinar qualquer coisa.

Reusa ``load``, ``aggregate`` e a paleta do modulo irmao: uma fonte de dados, uma paleta.

``--by stage`` e um eixo OUTRO, nao uma troca de fonte do caminho acima: le o CSV do
avaliador (``src/evaluate/eval_growth_stages.py``), que tem uma linha final por
``(familia, estagio, dataset, split)`` e nenhuma coluna ``epoch``. O eixo X passa a ser o
estagio e cada FAMILIA (grid4, g5_in_feature, g5_in_image, ...) vira uma linha no mesmo
eixo. Atencao ao que muda junto com o eixo: aquele CSV pontua no TESTE, a serie por epoca
pontua na VALIDACAO — nao sao continuacao um do outro. ``--eval-split val`` troca a fonte
deste eixo pelo CSV de validacao (sonda SVM do W&B), que o proprio avaliador escreve com
``--fetch-wandb``; o plot so o le.

Os dois modos aceitam ``--family``, mas a leem diferente. O ``epoch`` desenha UM painel por
familia (a trajetoria continua so faz sentido dentro de uma, com os pontos herdados dela);
o ``stage`` sobrepoe as familias num eixo so, para compara-las. ``--fetch`` rebaixa do W&B
os runs das familias pedidas — todas moram no mesmo projeto ``phd_thesis_grid4``, separadas
pelo segmento de familia no nome do run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_partial_train_spifil_hybrid import (  # noqa: E402
    ROOT, STAGE_COLORS, STAGE_MARKERS, STAGE_PT, STAGE_TRAINS, STAGES,
    aggregate, fetch, grid_dir, load, out_shapes,
)

OUT_DIR = ROOT / "artifacts" / "analysis_continuidade"
# No modo `stage` a serie e a FAMILIA, nao o estagio: STAGE_COLORS nao serve, esta indexada
# pelo eixo X. Mesma Okabe-Ito do modulo irmao, e o marcador como segundo canal visual.
FAMILY_COLORS = {"grid3": "#000000", "grid4": "#0072B2",
                 "g5_in_feature": "#E69F00", "g5_in_image": "#009E73",
                 # Controles de camada aleatoria e o braco com Head. Fecham as oito cores da
                 # Okabe-Ito; uma familia nova daqui em diante precisa de outra fonte.
                 "g5_random": "#CC79A7", "g5_random_in_feature": "#D55E00",
                 "g5_head": "#56B4E9",
                 # A nona familia, e a Okabe-Ito acabou na linha de cima. Roxo escuro do
                 # ColorBrewer Dark2: e o unico tom que nao encosta em nenhuma das oito
                 # (o mais proximo, o #CC79A7, e rosa claro — separa por luminancia, que e
                 # o canal que sobrevive ao daltonismo).
                 "g5_head_larvae": "#6A3D9A"}
FAMILY_MARKERS = {"grid3": "o", "grid4": "s", "g5_in_feature": "^", "g5_in_image": "D",
                  "g5_random": "v", "g5_random_in_feature": "P", "g5_head": "X",
                  "g5_head_larvae": "*"}


def inherited(metric: str = "probe/svm_kappa", family: str = "grid4") -> dict:
    """``(dataset, percentage, stage_label) -> media do valor herdado sobre os splits``.

    O valor herdado e o ``flim_ref_svm_*`` de cada estagio: a sonda rodada em
    ``on_fit_start``, sobre o encoder como ele CHEGA, antes de qualquer passo de gradiente.
    Para um estagio 3 isso e literalmente "o encoder do estagio 2 mais a camada nova crua".
    """
    key = "flim_ref_" + metric.split("/")[-1]
    acc: dict = {}
    for path in grid_dir(family).glob("*/*/run_metadata.json"):
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        value = meta.get(key)
        if value is None:
            continue
        slot = (meta["dataset"], meta["percentage"], path.parent.name)
        acc.setdefault(slot, []).append(float(value))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def plot(dataset: str, percentage: int, metric: str = "probe/svm_kappa",
         out_dir: Path = OUT_DIR, df=None, family: str = "grid4") -> Path:
    """Desenha a trajetoria continua de um braco e devolve o caminho do PNG."""
    df = load() if df is None else df
    # Uma pasta por experimento: quem diz de qual grade e o painel e o diretorio, nao o
    # nome do arquivo. Assim `eggs_pct5_..._continuo.png` volta a ser o nome em todas as
    # familias e da para comparar duas grades abrindo o mesmo nome em duas pastas.
    out_dir = Path(out_dir) / family
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = df[(df["dataset"] == dataset) & (df["percentage"] == percentage)
               & (df["family"] == family)]
    agg = aggregate(panel, metric) if not panel.empty else panel
    shapes, refs = out_shapes(family), inherited(metric, family)

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    notes = []
    if agg.empty:
        ax.text(0.5, 0.5, "sem dados", ha="center", va="center", fontsize=28,
                color="#b0b0b0", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        offset, prev_xy, boundaries = 0.0, None, []
        for stage in STAGES:
            curve = agg[agg["stage_label"] == stage]
            if curve.empty:
                continue
            color, marker = STAGE_COLORS[stage], STAGE_MARKERS[stage]
            x = curve["epoch"].to_numpy() + offset
            mean, sd, n = (curve["mean"].to_numpy(), curve["std"].to_numpy(),
                           curve["n"].to_numpy())
            # O ponto herdado abre o estagio, meia epoca antes da primeira medicao: e o
            # estado que ele RECEBEU. Sem ele a queda do crescimento fica invisivel, porque
            # a primeira epoca plotada ja e depois de uma epoca inteira de treino.
            ref = refs.get((dataset, percentage, stage))
            if ref is not None:
                ax.plot([x[0] - 0.5], [ref], marker=marker, color=color, markersize=8,
                        markerfacecolor="white", markeredgewidth=1.8, zorder=6)
                ax.plot([x[0] - 0.5, x[0]], [ref, mean[0]], "-", color=color, lw=1.2,
                        alpha=0.7, zorder=5)
            # O salto entre estagios: do fim do anterior ate o herdado deste. E aqui que o
            # crescimento cobra o preco dele.
            if prev_xy is not None:
                target = (x[0] - 0.5, ref) if ref is not None else (x[0], mean[0])
                ax.plot([prev_xy[0], target[0]], [prev_xy[1], target[1]], "-",
                        color="#909090", lw=1.1, alpha=0.9, zorder=2)
                boundaries.append((prev_xy[0] + target[0]) / 2)
            span = f"n={n.max()}" if n.min() == n.max() else f"n={n.max()}→{n.min()}"
            label = " · ".join(filter(None, [STAGE_PT[stage], STAGE_TRAINS[stage],
                                             shapes.get((dataset, stage)), span]))
            ax.fill_between(x, mean - sd, mean + sd, where=n >= 3, color=color,
                            alpha=0.16, lw=0, zorder=1)
            ax.plot(x, mean, "--", color=color, lw=1.6, zorder=3, marker=marker,
                    markevery=max(1, len(curve) // 10), markersize=4,
                    markerfacecolor="none", label=None if n.max() >= 3 else label)
            ax.plot(x, np.where(n >= 3, mean, np.nan), "-", color=color, lw=2.2,
                    zorder=4, marker=marker, markevery=max(1, len(curve) // 10),
                    markersize=5, label=label if n.max() >= 3 else None)
            prev_xy, offset = (x[-1], mean[-1]), x[-1] + 1.0
        for b in boundaries:
            ax.axvline(b, color="#c8c8c8", lw=0.9, ls=":", zorder=0)
        # Cobertura por estagio: um unico numero mentiria, porque cada estagio perde os
        # splits em epocas diferentes.
        cover = [f"{STAGE_PT[s].replace('estagio ', 'E')}:{int(g.loc[g['n'] >= 3, 'epoch'].max())}"
                 for s in STAGES
                 for g in [agg[agg["stage_label"] == s]]
                 if not g.empty and (g["n"] >= 3).any()]
        notes.append("3 splits ate a epoca — " + " · ".join(cover))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2,
                  frameon=False, fontsize=8.5, handlelength=2.6, columnspacing=1.4)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    ax.set_xlabel("epocas acumuladas — os estagios rodam em sequencia, um comeca onde o outro parou")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset} · pct{percentage} · {metric}  ({family})\n"
                 "marcador vazado no inicio de cada estagio = estado herdado, antes de treinar; "
                 "segmento cinza = o salto entre estagios",
                 fontsize=11)
    running = panel.loc[panel["run_state"] == "running", "run_name"].nunique()
    if running:
        notes.append(f"{running} run(s) ainda em treino")
    if notes:
        fig.text(0.012, 0.012, " · ".join(notes), fontsize=8.5, color="#c0392b")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.subplots_adjust(bottom=0.30)
    path = out_dir / f"{dataset}_pct{percentage}_{metric.split('/')[-1]}_continuo.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_stages(dataset: str, percentage: int, df: pd.DataFrame, families: list[str],
                metric: str = "probe/svm_kappa", out_dir: Path = OUT_DIR,
                splits: list[int] | None = None,
                stages: list[str] | None = None,
                eval_split: str = "test") -> Path:
    """Uma linha por familia sobre o eixo dos estagios, do CSV do avaliador."""
    # Este painel COMPARA familias, entao a pasta e a comparacao inteira, nao uma familia:
    # `g5_in_feature_vs_g5_in_image/`. Com uma familia so, e o nome dela e a pasta fica
    # lado a lado com a do modo `epoch`.
    out_dir = Path(out_dir) / "_vs_".join(families)
    out_dir.mkdir(parents=True, exist_ok=True)
    # O CSV grava `kappa`/`acc`/`f1` crus; o --metric vem no dialeto do W&B. Tirar o
    # prefixo faz o default de hoje (`probe/svm_kappa`) valer sem flag extra.
    column = metric.rsplit("_", 1)[-1]
    if column not in df.columns:
        raise SystemExit(f"--metric {metric}: o CSV nao tem a coluna '{column}'")
    panel = df[(df["dataset"] == dataset) & (df["percentage"] == percentage)]
    # No CSV do avaliador a familia nao e coluna: ela mora no ckpt, que e
    # artifacts/spifil_growth/<FAMILY>/<braco>/<estagio>/checkpoints/best_kappa.ckpt. O CSV
    # de validacao (--eval-split val) ja traz a coluna real — prefira ela, porque o ckpt la
    # e so compatibilidade.
    if "family" not in panel.columns:
        panel = panel.assign(family=panel["ckpt"].str.split("/").str[2])
    panel = panel[panel["family"].isin(families)]
    if splits:
        panel = panel[panel["split"].isin(splits)]
    if stages:
        panel = panel[panel["stage_label"].isin(stages)]
    order = [s for s in STAGES if s in set(panel["stage_label"])]

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    if not order:
        ax.text(0.5, 0.5, "sem dados", ha="center", va="center", fontsize=28,
                color="#b0b0b0", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for family in families:
            arm = panel[panel["family"] == family]
            if arm.empty:
                continue
            # `reindex` na ordem cronologica abre buraco (NaN) no estagio que esta familia
            # nao tem — e o caso NORMAL: larvae para no estagio 2, so um braco chega a
            # round2. O NaN quebra a linha no lugar certo em vez de emendar por cima.
            agg = arm.groupby("stage_label")[column].agg(
                mean="mean", std="std", n="count").reindex(order)
            mean, sd = agg["mean"].to_numpy(), agg["std"].fillna(0.0).to_numpy()
            n = agg["n"].dropna().to_numpy()
            span = (f"n={int(n.max())}" if n.min() == n.max()
                    else f"n={int(n.max())}→{int(n.min())}")
            color = FAMILY_COLORS.get(family, "#666666")
            ax.fill_between(np.arange(len(order)), mean - sd, mean + sd, color=color,
                            alpha=0.16, lw=0, zorder=1)
            ax.plot(np.arange(len(order)), mean, "-", color=color, lw=2.0, zorder=3,
                    marker=FAMILY_MARKERS.get(family, "o"), markersize=6,
                    label=f"{family} · {span} splits")
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([STAGE_PT[s] for s in order], fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2,
                  frameon=False, fontsize=8.5, handlelength=2.6, columnspacing=1.4)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    ax.set_xlabel("estagio do curriculo — ordem cronologica, uma medicao final por estagio")
    # A nota vermelha de hoje diz TESTE, e isso so vale para o CSV do avaliador. Com
    # --eval-split val a fonte e outra (sonda SVM do W&B, fit=train/score=validation) e a
    # nota mentiria — entao o eixo Y e o rodape dizem qual das duas esta no painel.
    ax.set_ylabel(metric if eval_split == "test" else f"{metric} — VALIDACAO")
    ax.set_title(f"{dataset} · pct{percentage} · {metric}  ({', '.join(families)})\n"
                 "media +/- desvio sobre os splits; familias no mesmo eixo, comparaveis",
                 fontsize=11)
    fig.text(0.012, 0.012,
             "CSV do avaliador: pontuado no TESTE — o modo `--by epoch` "
             "pontua na VALIDACAO; um nao e continuacao do outro" if eval_split == "test"
             else "CSV de validacao (sonda SVM do W&B: fit=train, score=validation) — "
                  "mesma medicao do `--by epoch`, NAO o TESTE do avaliador",
             fontsize=8.5, color="#c0392b")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.subplots_adjust(bottom=0.30)
    # O infixo `_val` no nome segue o do CSV: sem ele o painel de validacao sobrescreveria
    # o de teste na mesma pasta, calado, e as duas medicoes nao sao intercambiaveis.
    kind = "" if eval_split == "test" else "_val"
    path = out_dir / f"{dataset}_pct{percentage}_{metric.split('/')[-1]}_estagios{kind}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--by", choices=("epoch", "stage"), default="epoch",
                    help="epoch: trajetoria continua por epoca, do W&B, um painel por "
                         "familia. stage: eixo dos estagios, do CSV do avaliador, as "
                         "familias sobrepostas no mesmo eixo.")
    ap.add_argument("--metric", default="probe/svm_kappa")
    ap.add_argument("--family", nargs="+", default=["grid4"],
                    help="grades de artifacts/spifil_growth/ — o `_exp()` do laco de "
                         "treino, basename do --work-dir.")
    ap.add_argument("--fetch", action="store_true",
                    help="Rebaixa do W&B os runs das familias pedidas antes de plotar.")
    ap.add_argument("--pct", type=int, nargs="+", default=[5, 50])
    ap.add_argument("--dataset", nargs="+", default=["eggs", "larvae", "protozoan"])
    ap.add_argument("--split", type=int, nargs="+", default=None)
    ap.add_argument("--stage", nargs="+", choices=STAGES, default=None)
    ap.add_argument("--csv", default=None,
                    help="CSV do avaliador; default derivado de --family, --pct e "
                         "--eval-split. Um so CSV para todas as pct quando explicito.")
    ap.add_argument("--eval-split", choices=("test", "val"), default="test",
                    help="Qual medicao alimenta o eixo dos estagios. test: CSV do "
                         "avaliador (hoje). val: CSV de validacao vindo do W&B, com o "
                         "infixo `_val` no nome default.")
    ap.add_argument("--fetch-wandb", action="store_true",
                    help="(Re)escreve o CSV de validacao a partir do W&B antes de "
                         "plotar. Implica --eval-split val.")
    ap.add_argument("--wandb-entity", default="ophira-ai")
    ap.add_argument("--wandb-project", default="phd_thesis_grid4")
    ap.add_argument("--stage-agg", choices=("best", "last"), default="best",
                    help="Como colapsar a curva de um estagio num ponto: best = epoca de "
                         "maior probe/svm_kappa (o que o best_kappa.ckpt guarda); "
                         "last = ultima stage_epoch.")
    ap.add_argument("--allow-missing-runs", action="store_true",
                    help="Nao falha quando um braco em disco nao tem run no W&B.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    if args.fetch_wandb:
        args.eval_split = "val"
    if args.fetch:
        fetch(families=set(args.family))
    if args.by == "epoch":
        df = load()
        for family in args.family:
            for dataset in args.dataset:
                for percentage in args.pct:
                    print(plot(dataset, percentage, args.metric, args.out_dir,
                               df=df, family=family))
        return
    # Um CSV por porcentagem: e assim que o avaliador escreve, para pct5 e pct50 rodarem
    # em paralelo sem disputar o mesmo arquivo. So grid4 mantem o nome curto de hoje.
    tag = "" if tuple(args.family) == ("grid4",) else "_" + "_".join(args.family)
    kind = "_val" if args.eval_split == "val" else ""
    for percentage in args.pct:
        csv = Path(args.csv
                   or ROOT / "results" / f"eval_growth_stages{kind}{tag}_pct{percentage}.csv")
        if args.fetch_wandb:
            # Este plot NUNCA fala com o W&B no modo `stage`: quem sabe montar o CSV de
            # validacao e o avaliador, e ele e a UNICA implementacao da agregacao — dois
            # `--stage-agg best` em lugares diferentes divergiriam no primeiro empate.
            sys.path.insert(0, str(ROOT))
            from src.evaluate.eval_growth_stages import _wandb_val
            # Ele le `args.pct` como ESCALAR (um CSV por porcentagem); aqui `--pct` e
            # nargs="+". O Namespace trocado resolve sem tocar na assinatura de la.
            _wandb_val(argparse.Namespace(**{**vars(args), "pct": percentage}), str(csv))
        df = pd.read_csv(csv)
        for dataset in args.dataset:
            print(plot_stages(dataset, percentage, df, args.family, args.metric,
                              args.out_dir, args.split, args.stage, args.eval_split))


if __name__ == "__main__":
    main()
