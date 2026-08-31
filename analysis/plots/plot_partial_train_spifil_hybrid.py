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
Curvas contrastivas do curriculo SPiFiL hibrido, estagio contra estagio.

Le os CSVs ja baixados do W&B em ``wandb_phd_thesis_grid4/`` (um por run,
``spifil_growth_<dataset>_split<S>_pct<P>_<stage_label>.csv``), agrega a metrica
sobre os splits e desenha um grafico por ``dataset x percentage``: eixo X e a
epoca DENTRO do estagio, entao todas as curvas partem de zero e da para ler o
ganho de um estagio contra o outro.

Tres armadilhas dos dados, todas tratadas aqui:

1. ``scan_history()`` traz todo log point, inclusive os de passo de treino, e a
   maioria das celulas da metrica fica vazia. A curva por epoca sai filtrando as
   linhas nao nulas e ficando com a ultima de cada epoca.
2. O estagio verdadeiro e ``stage_label`` (``stage1``, ``stage2``,
   ``round<N>_stage3``, ``round<N>_stage4``, ``round<N>_head``). A coluna
   ``stage_config`` que o ``fetch()`` gravava valia so 1/2/3, confundia o estagio
   2 com o estagio 4, nao tinha numero para o ``head`` e nao era lida em lugar
   nenhum: saiu. Os CSVs ja baixados seguem com ela, sem uso.
3. A cobertura e desigual e ha runs ainda em treino. A contagem ``n`` de splits
   entra na legenda, o trecho com ``n < 3`` sai tracejado e sem faixa de desvio,
   o eixo X para na ultima epoca com os 3 splits (senao um unico split de 500
   epocas espreme o trecho comparavel na borda) e o rodape conta esse corte e os
   runs que ainda estao rodando. Combinacao sem dado nenhum
   gera um PNG escrito "sem dados" — nunca um numero inventado, nunca um
   arquivo faltando.

Uso::

    python -m analysis.plots.plot_partial_train_spifil_hybrid

Sem CLI: os antigos --fetch e --metric sao parametros nomeados de ``main()``, com os
mesmos defaults — ``main(fetch=True, metric="probe/svm_acc")``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# O lookup dos runs (`fetch`, `CSV_DIR`, `grid_dir`) vive na camada que acessa dado, nao
# aqui: `analysis/` e folha e ninguem importa dela. Reexportados de proposito — o modulo
# irmao (`plot_continuity_spifil_hybrid`) importa `fetch` e `grid_dir` por este nome.
from eval.growth_stages import CSV_DIR, fetch, grid_dir  # noqa: E402,F401

# analysis/plots/ esta a 2 niveis da raiz do repo (o arquivo veio de tools/, que era 1).
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "artifacts" / "analysis" / "partial_train_spifil_hybrid"

DATASETS = ("eggs", "larvae", "protozoan")
PCTS = (5, 50)
# Ordem cronologica, e e ela que da a ordem de plotagem. O `round<N>_head` substitui o par
# 3+4 da rodada, entao entra como o ULTIMO estagio da sua rodada. Um braco tem head OU 3+4,
# nunca os dois: o outro caminho fica sem curva e some do painel, e a sequencia desenhada
# sai cronologica nos dois bracos.
STAGES = ("stage1", "stage2", "round1_stage3", "round1_stage4", "round1_head",
          "round2_stage3", "round2_stage4", "round2_head")
STAGE_PT = {"stage1": "estagio 1", "stage2": "estagio 2",
            "round1_stage3": "estagio 3 (r1)", "round1_stage4": "estagio 4 (r1)",
            "round2_stage3": "estagio 3 (r2)", "round2_stage4": "estagio 4 (r2)",
            "round1_head": "head (r1)", "round2_head": "head (r2)"}
# O que recebe gradiente em cada estagio. Nao vem do `encoder_frozen` do metadata: aquele
# campo e binario e nao distingue "encoder inteiro congelado" de "so a camada nova treina",
# que e justamente o que o estagio 3 passou a fazer.
STAGE_TRAINS = {"stage1": "enc congelado", "stage2": "tudo treina",
                "round1_stage3": "so a nova", "round1_stage4": "tudo treina",
                "round2_stage3": "so a nova", "round2_stage4": "tudo treina",
                # encoder inteiro destravado + Head supervisionada; decoder fora.
                "round1_head": "enc + head sup", "round2_head": "enc + head sup"}
# Diretorio dos artefatos de treino: e de la que sai a forma da saida do encoder, que os
# CSVs do W&B nao carregam.
GRID_DIR = ROOT / "artifacts" / "spifil_growth" / "grid4"

KEYS = ["dataset", "percentage", "stage_label", "epoch"]


def load() -> pd.DataFrame:
    """Le todos os CSVs de ``wandb_phd_thesis_grid4/`` numa tabela so.

    Os 82 CSVs da grid4 foram baixados antes da coluna ``family`` existir: eles nao a
    tem, e e justamente por nao a terem que valem "grid4". Preencher aqui evita
    rebaixar 46 MB so para ganhar uma coluna constante.
    """
    df = pd.concat([pd.read_csv(f) for f in sorted(CSV_DIR.glob("*.csv"))],
                   ignore_index=True)
    if "family" not in df.columns:
        df["family"] = "grid4"
    return df.fillna({"family": "grid4"})


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Media, desvio e numero de splits por (dataset, percentage, stage_label, epoch).

    Fica so com as linhas em que a metrica existe, guarda a ultima de cada epoca
    de cada split e entao agrega sobre os splits. Desvio de amostra unica e
    indefinido: vira 0, e quem denuncia isso e o ``n``.
    """
    rows = df[df[metric].notna()].sort_values("_step")
    # `epoch` e o contador GLOBAL do Lightning. No braco com Head cada estagio retoma de
    # checkpoint, entao o contador abre num offset diferente por split e a agregacao punha
    # cada split num bucket so (n=1 em todo o estagio 2). `stage_epoch` ja e o mesmo
    # contador zerado no inicio do estagio, e nos bracos que treinam do zero os dois sao
    # identicos — por isso trocar a chave nao mexe nos paineis antigos.
    rows = rows.assign(epoch=rows["stage_epoch"].astype(int))
    rows = rows.groupby(KEYS + ["split"], as_index=False)[metric].last()
    out = rows.groupby(KEYS)[metric].agg(mean="mean", std="std", n="count").reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out.sort_values(KEYS)


# Okabe-Ito: maximamente distinta E segura para daltonismo. Indexada pelo ROTULO, nao pela
# posicao em STAGES: rotulo novo no meio da ordem cronologica nao pode empurrar a cor dos
# antigos. Sequencial (viridis) nao serve aqui — os estagios 3 e 4 sairiam no mesmo
# verde-azulado. As oito cores da paleta estao tomadas; um round3 precisa de outra fonte.
# ponytail: "#F0E442" e a mais fraca das oito sobre branco. Fica porque um braco com head
# so desenha 4 curvas (1, 2, head r1, head r2) e ela nao disputa espaco com nenhuma outra.
STAGE_COLORS = {"stage1": "#000000", "stage2": "#0072B2",
                "round1_stage3": "#E69F00", "round1_stage4": "#009E73",
                "round2_stage3": "#CC79A7", "round2_stage4": "#D55E00",
                "round1_head": "#56B4E9", "round2_head": "#F0E442"}
# Segundo canal visual: cor sozinha nao basta onde as curvas se cruzam. Estilo de linha
# nao serve de canal — solido/tracejado ja significa "3 splits" contra "menos de 3".
STAGE_MARKERS = {"stage1": "o", "stage2": "s", "round1_stage3": "^",
                 "round1_stage4": "D", "round2_stage3": "v", "round2_stage4": "P",
                 "round1_head": "X", "round2_head": "*"}


def out_shapes(family: str = "grid4") -> dict:
    """``(dataset, stage_label) -> "CxHxW"`` da saida do encoder, lido dos metadados do treino.

    Canais vem de ``channels[-1]``; o lado espacial e recomputado percorrendo o pooling da
    ``architecture.json`` (as camadas crescidas usam ``pool-stride 1``, entao a grade para de
    encolher e fica em 24). Se o metadado nao existir, devolve nada e a legenda omite a forma.

    Cuidado com o cwd: nos estagios crescidos o ``arch_json`` do metadado e RELATIVO, entao
    rodar de fora da raiz do repo faz o ``except`` engolir esses estagios em silencio e a
    legenda perde a forma deles (14 formas da raiz contra 6 de dentro de
    ``analysis/plots/``).
    """
    shapes = {}
    for meta_path in grid_dir(family).glob("*/*/run_metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
            arch = json.loads(Path(meta["arch_json"]).read_text())
        except (OSError, ValueError, KeyError):
            continue
        grid = 200
        for n in range(1, arch["nlayers"] + 1):
            pool = arch[f"layer{n}"]["pooling"]
            if pool["type"] == "max_pool":
                grid = (grid - pool["size"][0]) // pool["stride"] + 1
        shapes[(meta["dataset"], meta_path.parent.name)] = f"{meta['channels'][-1]}x{grid}x{grid}"
    return shapes


def plot(dataset: str, percentage: int, metric: str = "probe/svm_kappa",
         out_dir: Path = OUT_DIR, df: pd.DataFrame | None = None,
         family: str = "grid4") -> Path:
    """Desenha um painel dataset x percentage de UMA familia e devolve o caminho do PNG.

    O filtro por familia nao e opcional: todas as familias moram no mesmo projeto W&B e
    portanto na mesma pasta de CSVs, e sem ele `grid4`, `g5_in_feature`, `g5_in_image` e
    `g5_head` caem na mesma celula de `(dataset, percentage)` e sao mediadas juntas.

    A `main()` deste modulo nao expoe `family` de proposito: o nome do PNG nao carrega a
    familia, entao rodar outra familia aqui sobrescreveria o painel da grid4. Painel por
    familia e o do modulo irmao (`plot_continuity_spifil_hybrid`), que ja escreve numa
    pasta por familia.
    """
    df = load() if df is None else df
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = df[(df["dataset"] == dataset) & (df["percentage"] == percentage)
               & (df["family"] == family)]
    agg = aggregate(panel, metric) if not panel.empty else panel

    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    notes = []
    if agg.empty:
        ax.text(0.5, 0.5, "sem dados", ha="center", va="center", fontsize=28,
                color="#b0b0b0", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # O enquadramento pertence a regiao com os 3 splits: um unico split que
        # treinou 500 epocas nao pode espremer o trecho comparavel na borda.
        shapes = out_shapes(family)
        full = agg.loc[agg["n"] >= 3, "epoch"]
        cut = int(full.max()) if len(full) else 0
        for stage in STAGES:
            curve = agg[agg["stage_label"] == stage]
            if curve.empty:
                continue
            color, marker = STAGE_COLORS[stage], STAGE_MARKERS[stage]
            every = max(1, len(curve) // 12)
            x, mean, sd, n = (curve["epoch"].to_numpy(), curve["mean"].to_numpy(),
                              curve["std"].to_numpy(), curve["n"].to_numpy())
            span = f"n={n.max()}" if n.min() == n.max() else f"n={n.max()}→{n.min()}"
            shape = shapes.get((dataset, stage))
            label = " · ".join(filter(None, [STAGE_PT[stage], STAGE_TRAINS[stage],
                                             shape, span]))
            # Faixa so onde ha 3 splits: com 1 split o "desvio" e 0 e a faixa mentiria.
            ax.fill_between(x, mean - sd, mean + sd, where=n >= 3, color=color,
                            alpha=0.16, lw=0, zorder=1)
            # Curva inteira tracejada; por cima, so o trecho com 3 splits, solido.
            ax.plot(x, mean, "--", color=color, lw=1.7, zorder=3, marker=marker,
                    markevery=every, markersize=4, markerfacecolor="none",
                    label=None if n.max() >= 3 else label)
            ax.plot(x, np.where(n >= 3, mean, np.nan), "-", color=color, lw=2.0,
                    zorder=4, marker=marker, markevery=every, markersize=5,
                    label=label if n.max() >= 3 else None)
        if cut > 0:
            ax.set_xlim(0, cut * 1.05)
            notes.append(f"eixo X cortado na epoca {cut}, a ultima com os 3 splits")
        else:
            notes.append("nenhum estagio chega a 3 splits: eixo X na extensao total")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=2,
                  frameon=False, fontsize=8.5, handlelength=2.6, columnspacing=1.4)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_axisbelow(True)
    ax.set_xlabel("epoca dentro do estagio")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset} · pct{percentage} · {metric}\n"
                 "media +/- desvio sobre os splits; tracejado = menos de 3 splits",
                 fontsize=12)
    running = panel.loc[panel["run_state"] == "running", "run_name"].nunique()
    if running:
        notes.append(f"{running} run(s) ainda em treino")
    if notes:
        fig.text(0.012, 0.012, " · ".join(notes), fontsize=8.5, color="#c0392b")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.subplots_adjust(bottom=0.30)
    path = out_dir / f"{dataset}_pct{percentage}_{metric.split('/')[-1]}.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path



# Alias interno: o parametro `fetch` de main() sombreia a funcao homonima, importada de
# eval/growth_stages.py. O nome publico `fetch` continua existindo aqui — e por ele que
# plot_continuity_spifil_hybrid.py o importa.
_fetch = fetch

def main(fetch: bool = False, metric: str = "probe/svm_kappa") -> None:
    """Curvas contrastivas do curriculo SPiFiL: um painel por dataset x percentage.

    Um parametro por flag do argparse antigo, com o mesmo default: `fetch` rebaixa os
    CSVs do W&B antes de plotar.
    """
    # O corpo ja le tudo por `args.x`: o shim nasce so dos parametros, entao esta e a
    # primeira linha viva e locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())
    if args.fetch:
        _fetch()
    df = load()
    for dataset in DATASETS:
        for pct in PCTS:
            print(plot(dataset, pct, args.metric, df=df))


if __name__ == "__main__":
    main()
