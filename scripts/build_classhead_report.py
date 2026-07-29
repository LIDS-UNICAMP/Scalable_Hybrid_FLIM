"""build_classhead_report.py — Monta as tabelas markdown de comparação de cabeças
de classificação sobre o encoder FLIM.

Agrega, por (dataset × percentage × cabeça × encoder_mode), média ± desvio-padrão
amostral sobre os splits, e emite as tabelas do relatório em markdown.

Fontes (as que existirem em disco; as ausentes são apenas puladas):
    SVM + FLIM                  data/reports_felipe/svm/*.csv
    FLIM + MLP (ReLU, Felipe)   data/reports_felipe/flim_mlp/*.csv
    FLIM + MLP (Sigmoid, ours)  results/sigmoid2l_test_results.csv
    FLIM + MLP (ReLU, ours)     results/relu2l_test_results.csv

Uso:
    python scripts/build_classhead_report.py                       # tabelas -> stdout
    python scripts/build_classhead_report.py --pcts 5 75 --out tables.md
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_MAP = {"eggs": "eggs", "larvae": "larvae", "cistos": "protozoan", "protozoan": "protozoan"}
DATASET_ORDER = ["eggs", "larvae", "protozoan"]

METRICS = [
    ("test_accuracy", "Test Accuracy"),
    ("test_f1_weighted", "Test F1-weighted"),
    ("test_cohen_kappa", "Test Cohen κ"),
]

_KEEP = ["dataset", "split", "percentage", "encoder_mode",
         "test_accuracy", "test_f1_weighted", "test_cohen_kappa"]


def _load_felipe(subdir: str, arm: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(_ROOT, "data", "reports_felipe", subdir, "*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["dataset"] = df["dataset"].map(DATASET_MAP)
    df = df[_KEEP].copy()
    df["arm"] = arm
    return df


def _load_ours(csv_rel: str, arm: str) -> pd.DataFrame:
    path = os.path.join(_ROOT, csv_rel)
    if not os.path.exists(path):
        print(f"[build_report] AUSENTE: {csv_rel} — braço '{arm}' será omitido.", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].map(DATASET_MAP)
    df = df[_KEEP].copy()
    df["arm"] = arm
    return df


def load_all(arms: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = [
        _load_felipe("svm", "SVM + FLIM"),
        _load_felipe("flim_mlp", "FLIM + MLP (ReLU, Felipe)"),
        _load_ours("results/sigmoid2l_test_results.csv", "FLIM + MLP (Sigmoid, ours)"),
        _load_ours("results/relu2l_test_results.csv", "FLIM + MLP (ReLU, ours)"),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("[build_report] Nenhuma fonte encontrada.")
    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["dataset"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "percentage", "arm", "encoder_mode"], as_index=False)
    agg = g.agg(
        n=("split", "nunique"),
        **{f"{m}_mean": (m, "mean") for m, _ in METRICS},
        **{f"{m}_sd": (m, "std") for m, _ in METRICS},  # ddof=1
    )
    return agg


def _fmt(mean: float, sd: float, n: int) -> str:
    if pd.isna(mean):
        return "—"
    if n <= 1 or pd.isna(sd):
        return f"{mean:.3f} (n=1)"
    return f"{mean:.3f} ± {sd:.3f}"


def _arm_order(arms: list[str]) -> list[str]:
    pref = ["SVM + FLIM", "FLIM + MLP (ReLU, Felipe)", "FLIM + MLP (ReLU, ours)",
            "FLIM + MLP (Sigmoid, ours)"]
    return [a for a in pref if a in arms] + sorted(set(arms) - set(pref))


def table_for_pct(agg: pd.DataFrame, pct: int) -> str:
    sub = agg[agg["percentage"] == pct]
    if sub.empty:
        return f"_(sem dados para {pct}%)_\n"

    head = ("| Dataset | Cabeça de classificação | Modo do encoder | n (splits) | "
            + " | ".join(f"{lbl} (média ± dp)" for _, lbl in METRICS) + " |")
    sep = "|:--|:--|:--|--:|" + "--:|" * len(METRICS)
    lines = [head, sep]

    for ds in [d for d in DATASET_ORDER if d in set(sub["dataset"])]:
        rows = sub[sub["dataset"] == ds]
        best = {m: rows[f"{m}_mean"].max() for m, _ in METRICS}
        for arm in _arm_order(sorted(rows["arm"].unique())):
            for mode in ["frozen", "unfrozen"]:
                r = rows[(rows["arm"] == arm) & (rows["encoder_mode"] == mode)]
                if r.empty:
                    continue
                r = r.iloc[0]
                cells = []
                for m, _ in METRICS:
                    txt = _fmt(r[f"{m}_mean"], r[f"{m}_sd"], int(r["n"]))
                    if r[f"{m}_mean"] == best[m]:
                        txt = f"**{txt}**"
                    cells.append(txt)
                lines.append(f"| {ds} | {arm} | {mode} | {int(r['n'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def delta_table(agg: pd.DataFrame, a: str, b: str) -> str:
    """Tabela de diferença a − b (mesmo dataset/pct/encoder_mode)."""
    piv = agg.set_index(["dataset", "percentage", "encoder_mode", "arm"])
    lines = ["| Dataset | pct | Modo | Δ Accuracy | Δ F1-weighted | Δ Cohen κ |",
             "|:--|--:|:--|--:|--:|--:|"]
    keys = sorted({(d, p, e) for d, p, e, arm in piv.index if arm in (a, b)},
                  key=lambda k: (DATASET_ORDER.index(k[0]) if k[0] in DATASET_ORDER else 9, k[1], k[2]))
    any_row = False
    for ds, pct, mode in keys:
        try:
            ra, rb = piv.loc[(ds, pct, mode, a)], piv.loc[(ds, pct, mode, b)]
        except KeyError:
            continue
        cells = [f"{ra[f'{m}_mean'] - rb[f'{m}_mean']:+.3f}" for m, _ in METRICS]
        lines.append(f"| {ds} | {pct} | {mode} | " + " | ".join(cells) + " |")
        any_row = True
    if not any_row:
        return f"_(sem par comparável entre '{a}' e '{b}')_\n"
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pcts", nargs="+", type=int, default=None,
                    help="percentuais a tabelar (default: os presentes em ambos os braços 'ours')")
    ap.add_argument("--out", default=None, help="arquivo markdown de saída (default: stdout)")
    args = ap.parse_args()

    df = load_all()
    agg = aggregate(df)

    if args.pcts:
        pcts = args.pcts
    else:
        ours = agg[agg["arm"].str.contains("ours")]
        pcts = sorted(ours["percentage"].unique()) if not ours.empty else sorted(agg["percentage"].unique())

    out = []
    for pct in pcts:
        out.append(f"### Tabela — {pct}% dos dados de treino\n")
        out.append(table_for_pct(agg, pct))
        out.append("")
    out.append("### Δ ReLU (ours) − Sigmoid (ours)\n")
    out.append(delta_table(agg, "FLIM + MLP (ReLU, ours)", "FLIM + MLP (Sigmoid, ours)"))
    text = "\n".join(out)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[build_report] Escrito -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
