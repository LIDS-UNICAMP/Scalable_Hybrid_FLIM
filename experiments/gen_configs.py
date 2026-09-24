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

"""gen_configs.py — os tres geradores de artefato do repositorio, num modulo so.

Funde, como manda spec_refactor.md:2593:

* ``scripts/generate_mlp_configs.py``  -> :func:`generate_mlp_configs`
* ``scripts/download_organmnist3d.py`` -> :func:`download_organmnist3d`
* ``scripts/normalize_reports.py``     -> :func:`normalize_reports`

Sao tres tarefas independentes que nunca se chamam. O que elas tem em comum, e o
que justifica o modulo unico, e o papel: **produzir arquivo derivado a partir de
fonte externa** (cache do W&B, MedMNIST, CSVs legados). Nenhuma delas treina nada
e nenhuma delas e importada por ``core/``, ``flim/``, ``methods/`` ou ``eval/``.

Nao ha argparse: cada funcao tem defaults nomeados. Rodado da raiz do
repositorio::

    # o default do modulo — regenera configs/generated/mlp/
    python -m experiments.gen_configs

    python -c "from experiments.gen_configs import generate_mlp_configs as f; f(dry_run=True)"
    python -c "from experiments.gen_configs import download_organmnist3d as f; f(size=64)"
    python -c "from experiments.gen_configs import normalize_reports as f; f(partial=True)"

TRES ARMADILHAS DA FUSAO, todas tratadas aqui e registradas para quem vier depois:

1. ``main()`` existia nos tres arquivos. Viraram os tres nomes publicos acima; o
   ``axis_map`` que so morava dentro do ``main`` de ``download_organmnist3d.py``
   virou :data:`_AXIS_MAP`, no nivel do modulo.
2. ``_ROOT`` existia nos dois maiores com TIPOS diferentes — ``str`` em
   ``generate_mlp_configs.py:40`` e ``Path`` em ``normalize_reports.py:69``.
   Aqui e ``Path``, uma vez so.
3. ``os.environ.setdefault("MPLBACKEND", "Agg")`` era efeito colateral de import
   em ``normalize_reports.py:45``. Desceu para dentro de
   :func:`normalize_reports`, antes do import do plotter — que e onde o
   matplotlib le a variavel. O modulo voltou a ser importavel sem efeito.

E o achatamento de ``configs/generated/mlp/``: ver o comentario dentro de
:func:`generate_mlp_configs`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ORDEM DE IMPORT, NAO ESTILO — nao reordene, e nao deixe o isort reordenar.
# Neste env `import pandas` carrega o libstdc++ do SISTEMA e, depois dele,
# `from PIL import Image` morre com
#   ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29'
#   not found (required by .../site-packages/PIL/../../.././libLerc.so.4)
# PIL entra aqui por tres caminhos (secao 2 direto; secoes 1 e 3 via matplotlib,
# arrastado por src.utils e por eval_plotter), entao ele vem ANTES do pandas.
# Verificado: PIL->pandas OK, pandas->PIL quebra.
# NAO e regressao da fusao: `python scripts/normalize_reports.py` ja falha hoje,
# pela mesma razao, no import de src.evaluate.eval_plotter de dentro do main().
import numpy as np
from PIL import Image

import pandas as pd
import yaml

from core.constants import METRICS, PROJECT_ROOT
from experiments.constants import (
    ARTIFACTS_NORMALIZED_DIR,
    ARTIFACTS_PLOTS_DIR,
    CANONICAL_COLS as _CANONICAL_COLS,
    DATASET_ALIASES as DATASET_MAP,
    DATASET_LONG_TO_SHORT as _LONG_DATASET_MAP,
    MLP_CONFIGS_DIR as _CONFIGS_DIR,
    REPORTS_FELIPE_SVM_DIR,
    RESULTS_DIR,
    UNIFIED_SVM_COMPARISON_CSV,
)
from eval.svm import parse_experiment_name
from core.wandb import ENTITY, PROJECT, cached_history

# Um so, e Path. Ver armadilha 2 no topo.
_ROOT = Path(PROJECT_ROOT)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Configs de avaliacao do MLP        (era scripts/generate_mlp_configs.py)
# ═════════════════════════════════════════════════════════════════════════════

# ─── Default hyperparameters ─────────────────────────────────────────────────

_DEFAULTS = {
    "max_epochs": 1000,
    "patience": 50,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "hidden_dim": 256,
    "dropout": 0.3,
}


def build_yaml_content(run_id: str, run_name: str, freeze: bool) -> dict:
    info = parse_experiment_name(run_name)
    return {
        "run_id": run_id,
        "experiment_name": run_name,
        "dataset_name": info.dataset_name,
        "split": info.split_id,
        "percentage": info.percentage,
        "initialization_type": info.initialization_type,
        "freeze_encoder": freeze,
        **_DEFAULTS,
    }


def yaml_path(run_id: str, run_name: str, freeze: bool) -> str:
    info = parse_experiment_name(run_name)
    mode = "freeze" if freeze else "unfreeze"
    return os.path.join(
        _CONFIGS_DIR,
        mode,
        info.dataset_name,
        # Achatado: split e percentage nao sao mais diretorio. Os dois continuam
        # DENTRO do YAML (build_yaml_content, chaves "split" e "percentage"), so
        # sairam do caminho. Ver spec_refactor.md:1465-1467.
        f"{run_id}.yaml",
    )



def generate_mlp_configs(dry_run: bool = False, update_wandb: bool = False) -> None:
    """Escreve um YAML por (run_id, modo) em ``configs/generated/mlp/``.

    dry_run      imprime os caminhos sem escrever nada
    update_wandb refaz o cache do W&B antes de gerar (default: cache local)

    ARMADILHA, e a razao de esta funcao nao ser o ``generate_all`` de antes.
    ``scripts/generate_mlp_configs.py:117-122`` apagava, a cada escrita, TODO
    ``.yaml`` irmao no diretorio de destino. Isso era correto enquanto o destino
    era ``<mode>/<ds>/split_N/pct_P/``, que hospedava UM run_id. Com o caminho
    achatado para ``<mode>/<ds>/`` o diretorio passa a hospedar ~18 run_ids, e
    aquele laco apagaria 17 deles a cada arquivo escrito — os 582 YAMLs
    colapsariam para 8, imprimindo ``[DEL]`` como se fosse normal.

    A unicidade agora e por CONJUNTO VALIDO, nao por diretorio: a grade inteira e
    montada antes de qualquer escrita, e so sai do disco o ``.yaml`` que nao esta
    nela. O efeito para o run_id superado e o mesmo de antes (``deduplicate=True``
    ja o tira de ``experiments``), sem o dano colateral.
    """
    # deduplicate=True keeps only the newest run per experiment name, so
    # re-ran experiments never produce stale configs for the old run_id.
    # Reads from local cache by default; pass update_wandb=True to refresh.
    # `cached_history` devolve {run_id: metadata}; o antigo `get_runs_dict_cached`
    # devolvia {run_id: nome}. A deduplicacao e a mesma (core/wandb.py:94).
    experiments = {
        rid: meta["name"]
        for rid, meta in cached_history(
            ENTITY, PROJECT, deduplicate=True, update=update_wandb
        ).items()
    }

    planned: dict[str, dict] = {}
    skipped = 0
    for freeze in (True, False):
        for run_id, run_name in experiments.items():
            try:
                path = yaml_path(run_id, run_name, freeze)
                planned[path] = build_yaml_content(run_id, run_name, freeze)
            except ValueError as exc:
                print(f"  [SKIP] {run_name}: {exc}")
                skipped += 1

    if dry_run:
        for path in planned:
            print(f"  [DRY] {os.path.relpath(path, _ROOT)}")
        print(f"\nCreated {len(planned)} YAML(s), removed 0 stale YAML(s), skipped {skipped}.")
        return

    removed = 0
    for target_dir in sorted({os.path.dirname(p) for p in planned}):
        os.makedirs(target_dir, exist_ok=True)
        for old_yaml in sorted(os.listdir(target_dir)):
            old_path = os.path.join(target_dir, old_yaml)
            if old_yaml.endswith(".yaml") and old_path not in planned:
                os.remove(old_path)
                print(f"  [DEL] {os.path.relpath(old_path, _ROOT)}  (fora da grade atual)")
                removed += 1

    for path, content in planned.items():
        with open(path, "w") as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        print(f"  [OK] {os.path.relpath(path, _ROOT)}")

    print(f"\nCreated {len(planned)} YAML(s), removed {removed} stale YAML(s), skipped {skipped}.")


# ═════════════════════════════════════════════════════════════════════════════
# 2. OrganMNIST3D -> PNG 2D             (era scripts/download_organmnist3d.py)
# ═════════════════════════════════════════════════════════════════════════════

# Axes extracted (default: all three, tripling the dataset size):
#     axial     -> slice along depth (axis 0)
#     coronal   -> slice along height (axis 1)
#     sagittal  -> slice along width  (axis 2)
# Era o dict local `axis_map` de scripts/download_organmnist3d.py:203-208, a
# unica traducao de --axis para a lista `axes`; subiu para o modulo com o main.
_AXIS_MAP = {
    "all":      [0, 1, 2],
    "axial":    [0],
    "coronal":  [1],
    "sagittal": [2],
}
_SPLITS_ALL = ("train", "val", "test")

# ── class names (MedMNIST v2 label order for OrganMNIST3D) ──────────────────
ORGAN_NAMES = [
    "bladder",
    "femur_l",
    "femur_r",
    "heart",
    "kidney_l",
    "kidney_r",
    "liver",
    "lung_l",
    "lung_r",
    "pancreas",
    "spleen",
]


def _to_rgb(arr2d: np.ndarray) -> Image.Image:
    """Convert a 2-D uint8 grayscale array to a 3-channel PIL image."""
    rgb = np.stack([arr2d, arr2d, arr2d], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _slice_volume(
    volume: np.ndarray,
    axes: list[int],
    slices: str,
) -> list[tuple[str, np.ndarray]]:
    """
    Return a list of (tag, 2D array) for the requested axes / slice strategy.

    Args:
        volume:  Shape (D, H, W), numpy uint8.
        axes:    Which axes to slice (0=axial, 1=coronal, 2=sagittal).
        slices:  'all' → every slice; 'middle' → only the centre one.
    """
    axis_tag = {0: "ax", 1: "co", 2: "sa"}
    results: list[tuple[str, np.ndarray]] = []

    for ax in axes:
        depth = volume.shape[ax]
        if slices == "middle":
            indices = [depth // 2]
        else:          # all
            indices = list(range(depth))

        for i in indices:
            sl = np.take(volume, i, axis=ax)  # shape (H, W) or (D, W) or (D, H)
            results.append((f"{axis_tag[ax]}{i:02d}", sl))

    return results


def download_and_export(
    out_dir: str | Path,
    size: int,
    axes: list[int],
    slices: str,
    splits: list[str],
) -> None:
    try:
        import medmnist
        from medmnist import OrganMNIST3D
    except ImportError:
        raise SystemExit(
            "medmnist is not installed.  Run:  pip install medmnist"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create class subdirs
    for name in ORGAN_NAMES:
        (out_dir / name).mkdir(exist_ok=True)

    total_saved = 0

    for split in splits:
        print(f"\n=== split: {split} ===")
        ds = OrganMNIST3D(split=split, download=True, size=size)

        # medmnist datasets expose .imgs (N, D, H, W) and .labels (N, 1)
        imgs: np.ndarray = ds.imgs        # uint8, 0-255
        labels: np.ndarray = ds.labels.flatten().astype(int)

        print(f"  volumes: {len(imgs)}  shape: {imgs.shape[1:]}")

        for vol_idx, (volume, label) in enumerate(zip(imgs, labels)):
            organ = ORGAN_NAMES[label]
            tag_base = f"{split}_{vol_idx:06d}"

            for slice_tag, arr in _slice_volume(volume, axes, slices):
                fname = f"{tag_base}_{slice_tag}.png"
                fpath = out_dir / organ / fname
                if not fpath.exists():
                    _to_rgb(arr).save(fpath)
                    total_saved += 1

            if (vol_idx + 1) % 200 == 0:
                print(f"  processed {vol_idx + 1}/{len(imgs)} volumes …")

        print(f"  done — running total saved: {total_saved}")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Output dir  : {out_dir.resolve()}")
    print(f"Total images: {total_saved}")
    print("\nClass breakdown:")
    for name in ORGAN_NAMES:
        n = len(list((out_dir / name).glob("*.png")))
        print(f"  {name:<12} {n:>6} images")
    print(f"\nSet  data_dir: data/organmnist3d  in your config to use this dataset.")



def download_organmnist3d(
    out_dir: str | Path = f"{PROJECT_ROOT}/data/organmnist3d",
    size: int = 28,
    axis: str = "all",
    slices: str = "all",
    splits: tuple[str, ...] = _SPLITS_ALL,
) -> None:
    """Baixa OrganMNIST3D e exporta como fatias PNG 2D.

    out_dir  raiz de saida. ABSOLUTA de proposito: em
             ``scripts/download_organmnist3d.py:180`` o default era
             ``"data/organmnist3d"``, relativo ao cwd — o unico caminho do modulo
             que nao era ancorado em PROJECT_ROOT. Rodando da raiz o valor e o
             mesmo; de qualquer outro lugar, agora acerta.
    size     28 (default) ou 64 (MedMNIST+)
    axis     "all" | "axial" | "coronal" | "sagittal"
    slices   "all" -> toda fatia por eixo; "middle" -> so a central
    splits   quais splits do MedMNIST incluir

    As checagens abaixo sao o `choices=` do argparse que morreu; sem elas um
    valor errado viraria KeyError ou silencio.
    """
    if size not in (28, 64):
        raise SystemExit(f"download_organmnist3d: size invalido: {size!r} (esperado 28 ou 64)")
    if axis not in _AXIS_MAP:
        raise SystemExit(f"download_organmnist3d: axis invalido: {axis!r} "
                         f"(esperado um de {list(_AXIS_MAP)})")
    if slices not in ("all", "middle"):
        raise SystemExit(f"download_organmnist3d: slices invalido: {slices!r} "
                         f"(esperado 'all' ou 'middle')")
    bad = [s for s in splits if s not in _SPLITS_ALL]
    if bad:
        raise SystemExit(f"download_organmnist3d: splits invalidos: {bad} "
                         f"(esperado um subconjunto de {list(_SPLITS_ALL)})")

    download_and_export(
        out_dir=out_dir,
        size=size,
        axes=_AXIS_MAP[axis],
        slices=slices,
        splits=list(splits),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Normalizacao dos CSVs legados      (era scripts/normalize_reports.py)
# ═════════════════════════════════════════════════════════════════════════════

_RESULTS = Path(RESULTS_DIR)

# ── Constants ──────────────────────────────────────────────────────────────────

_FELIPE_SVM_RE = re.compile(
    r"^report_svm_([a-z]+)_split(\d+)_perc(\d+)\.csv$"
)


# ── Source 1: reports_felipe/svm ──────────────────────────────────────────────

def normalize_felipe_svm() -> pd.DataFrame:
    """Read all reports_felipe/svm CSVs, aggregate across splits.

    Filename pattern: report_svm_{dataset}_split{N}_perc{pct}.csv
    Metrics: test_cohen_kappa, test_accuracy, test_f1_weighted
    Returns canonical DataFrame with method=SVM_FLIM, init=flim.
    """
    svm_dir = Path(REPORTS_FELIPE_SVM_DIR)
    raw_rows: list[dict] = []

    for csv_file in sorted(svm_dir.glob("*.csv")):
        m = _FELIPE_SVM_RE.match(csv_file.name)
        if not m:
            continue
        dataset_raw, split_str, pct_str = m.group(1), m.group(2), m.group(3)
        dataset_short = DATASET_MAP.get(dataset_raw)
        if dataset_short is None:
            print(f"  [WARN] Unknown dataset '{dataset_raw}' in {csv_file.name}, skipping")
            continue

        df = pd.read_csv(csv_file)
        # SVM reports may have multiple rows (e.g. different encoder_mode); take frozen
        df = df[df["encoder_mode"] == "frozen"]
        if df.empty:
            continue

        row = df.iloc[0]
        try:
            raw_rows.append({
                "dataset_short": dataset_short,
                "split": int(split_str),
                "pretrained_pct": int(pct_str),
                "kappa": float(row["test_cohen_kappa"]),
                "acc":   float(row["test_accuracy"]),
                "f1":    float(row["test_f1_weighted"]),
            })
        except (ValueError, KeyError) as exc:
            print(f"  [WARN] Skipping {csv_file.name}: {exc}")

    if not raw_rows:
        raise RuntimeError(
            f"No valid reports_felipe/svm CSV files found under {svm_dir}"
        )

    raw_df = pd.DataFrame(raw_rows)

    # Aggregate mean ± std across splits
    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in raw_df.groupby(
        ["dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":        "SVM_FLIM",
            "init":          "flim",
            "dataset_short": dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":      n,
            "kappa":         grp["kappa"].mean(),
            "kappa_std":     grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":           grp["acc"].mean(),
            "acc_std":       grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":            grp["f1"].mean(),
            "f1_std":        grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_FLIM  : {len(result)} rows from {len(raw_rows)} split-level files")
    return result


# ── Source 2: artifacts/SVM metrics (already aggregated) ─────────────────────

def load_lejepa_svm_metrics() -> pd.DataFrame:
    """Read all metrics_SVM_*.csv from artifacts/SVM/.

    Schema: model_type, dataset_short, pretrained_pct, init,
            n_splits, kappa, kappa_std, acc, acc_std, f1, f1_std
    Returns canonical DataFrame with method=SVM_lejepa_view.

    Nota: a curva se chamava `SVM_LeJEPA` (legenda "LeJEPA (59.504)"); foi
    renomeada para `SVM_lejepa_view` / legenda `lejepa_view`. Consumidores do CSV
    unificado (plot_comparison_flim.py, plot_parameters_vs_metrics.py,
    statistics/tools/wilcoxon_*.py) já usam a chave nova.
    """
    artifacts_svm = _ROOT / "artifacts" / "SVM"
    rows: list[dict] = []

    for metrics_csv in sorted(artifacts_svm.glob("*/*/metrics_SVM_*.csv")):
        df = pd.read_csv(metrics_csv)
        if df.empty:
            continue
        row = df.iloc[0]
        try:
            rows.append({
                "method":        "SVM_lejepa_view",
                "init":          str(row["init"]),
                "dataset_short": str(row["dataset_short"]),
                "pretrained_pct": int(row["pretrained_pct"]),
                "n_splits":      int(row["n_splits"]),
                "kappa":         float(row["kappa"]),
                "kappa_std":     float(row["kappa_std"]),
                "acc":           float(row["acc"]),
                "acc_std":       float(row["acc_std"]),
                "f1":            float(row["f1"]),
                "f1_std":        float(row["f1_std"]),
            })
        except (ValueError, KeyError) as exc:
            print(f"  [WARN] Skipping {metrics_csv}: {exc}")

    if not rows:
        return None  # caller handles missing gracefully

    result = pd.DataFrame(rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_lejepa_view: {len(result)} rows from artifacts/SVM/")
    return result


# ── Source 3: ijepa_svm_aggregated.csv ────────────────────────────────────────

def normalize_ijepa_svm() -> pd.DataFrame:
    """Normalize results/ijepa_svm_aggregated.csv to canonical schema.

    Source columns: dataset, dataset_short, pretrained_pct, n_splits,
                    embedding, kappa, kappa_std, acc, acc_std, f1, f1_std
    Returns canonical DataFrame with method=SVM_IJEPA, init=ijepa.
    """
    src = _RESULTS / "ijepa_svm_aggregated.csv"
    if not src.exists():
        raise FileNotFoundError(f"ijepa_svm_aggregated.csv not found at {src}")

    df = pd.read_csv(src)

    result = pd.DataFrame({
        "method":        "SVM_IJEPA",
        "init":          "ijepa",
        "dataset_short": df["dataset_short"],
        "pretrained_pct": df["pretrained_pct"].astype(int),
        "n_splits":      df["n_splits"].astype(int),
        "kappa":         df["kappa"],
        "kappa_std":     df["kappa_std"],
        "acc":           df["acc"],
        "acc_std":       df["acc_std"],
        "f1":            df["f1"],
        "f1_std":        df["f1_std"],
    })
    print(f"[NORM] SVM_IJEPA : {len(result)} rows from results/ijepa_svm_aggregated.csv")
    return result[_CANONICAL_COLS]


# ── Source 4: svm_distillation_conv_results.csv ───────────────────────────────

def normalize_distillation_conv_svm() -> pd.DataFrame | None:
    """Normalize results/svm_distillation_conv_results.csv to canonical schema.

    Aggregates mean ± std across splits per (dataset, pretrained_pct).
    Returns canonical DataFrame with method=SVM_Distillation_Conv, init=trunc_normal.
    Returns None if the file does not exist yet.
    """
    src = _RESULTS / "svm_distillation_conv_results.csv"
    if not src.exists():
        print(f"[SKIP] svm_distillation_conv_results.csv not found — skipping distillation conv")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_distillation_conv_results.csv has no ok rows")
        return None

    df["dataset_short"] = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distillation_Conv",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distillation_Conv: {len(result)} rows from results/svm_distillation_conv_results.csv")
    return result


# ── Source 4b: svm_distillation_conv_flim_frozen_results.csv ──────────────────

def _normalize_distill_flim_frozen() -> pd.DataFrame | None:
    """Normalize results/svm_distillation_conv_flim_frozen_results.csv.

    Keeps the per-checkpoint method names from the CSV
    (SVM_Distill_1x1BN_flim_frozen_eval_knn / _eval_loss), aggregating mean ± std
    across splits per (method, dataset, pretrained_pct). Skips non-ok / NaN rows
    (partial runs are fine — whatever is available gets plotted). Returns None if
    the file is absent.
    """
    src = _RESULTS / "svm_distillation_conv_flim_frozen_results.csv"
    if not src.exists():
        print("[SKIP] svm_distillation_conv_flim_frozen_results.csv not found — "
              "rode svm_distillation_conv.py --run-filter 1x1_BN2d_1280_flim_frozen "
              "--output-csv svm_distillation_conv_flim_frozen_results primeiro")
        return None

    df = pd.read_csv(src)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = df[df["kappa"].notna()].copy()  # ignore NaN
    if df.empty:
        print("[SKIP] svm_distillation_conv_flim_frozen_results.csv has no usable rows")
        return None

    df["dataset_short"] = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (method, dataset_short, pct), grp in df.groupby(
        ["method", "dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":         method,
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] flim_frozen (eval_knn/eval_loss): {len(result)} rows "
          f"from results/svm_distillation_conv_flim_frozen_results.csv")
    return result


# ── Source 5: svm_distill_proj1280_results.csv ────────────────────────────────

def _normalize_distill_proj1280() -> pd.DataFrame | None:
    """Normalize results/svm_distill_proj1280_results.csv — embedding [B,1280].

    method=SVM_Distill_Proj1280, init=trunc_normal.
    Returns None se o arquivo ainda não existir.
    """
    src = _RESULTS / "svm_distill_proj1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_distill_proj1280_results.csv não encontrado — rode svm_distill_with_projection.py primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_distill_proj1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_Proj1280",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_Proj1280: {len(result)} rows from results/svm_distill_proj1280_results.csv")
    return result


# ── Source 6: svm_proj1280_3x3_BN2d_results.csv ──────────────────────────────

def _normalize_distill_3x3bn() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_3x3_BN2d_results.csv — proj head 3x3 BN2d.

    method=SVM_Distill_3x3BN, init=trunc_normal.
    Returns None se o arquivo ainda não existir.
    """
    src = _RESULTS / "svm_proj1280_3x3_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 3x3_BN2d_1280_one_layer primeiro")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "trunc_normal")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv sem rows ok (trunc_normal)")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_3x3BN",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_3x3BN: {len(result)} rows from results/svm_proj1280_3x3_BN2d_results.csv")
    return result


# ── Source 7: svm_proj1280_1x1_BN2d_results.csv ──────────────────────────────

def _normalize_distill_1x1bn() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_1x1_BN2d_results.csv — proj head 1x1 BN2d.

    method=SVM_Distill_1x1BN, init=trunc_normal.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 1x1_BN2d_1280_one_layer --output-csv svm_proj1280_1x1_BN2d_results primeiro")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "trunc_normal")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv sem rows ok (trunc_normal)")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN: {len(result)} rows from results/svm_proj1280_1x1_BN2d_results.csv")
    return result


def _normalize_distill_1x1bn_nonorm() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv.

    Proj head 1x1 BN2d (126k), FLIM init, treinado/avaliado SEM ImageNet norm.
    method=SVM_Distill_1x1BN_nonorm, init=flim. Linha separada para comparar
    "com norm" (SVM_Distill_1x1BN) vs "sem norm".
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm_results.csv não encontrado "
              "— rode svm_distill_with_projection.py --run-filter 1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm "
              "--no-imagenet-norm --only-ok primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] ...no_imagenet_norm... sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN_nonorm",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN_nonorm: {len(result)} rows")
    return result


# ── Source 8b: svm_2l_1x1_init_flim_256_1280_results.csv ────────────────────

def _normalize_distill_2l_400k_flim() -> pd.DataFrame | None:
    """Normalize results/svm_2l_1x1_init_flim_256_1280_results.csv.

    method=SVM_Distill_2l400K_flim, init=flim.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_2l_1x1_init_flim_256_1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_results.csv não encontrado")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K_flim: {len(result)} rows from results/svm_2l_1x1_init_flim_256_1280_results.csv")
    return result


# ── Source 8c: svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv ─────────────

def _normalize_distill_2l_400k_flim_nonorm() -> pd.DataFrame | None:
    """Normalize results/svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv.

    method=SVM_Distill_2l400K_flim_nonorm, init=flim. Backbone init-FLIM 400k
    treinado com --no-imagenet-norm, avaliado COM a projetora ativa (proj_kd →
    1280d) — mesma régua do baseline trunc SVM_Distill_2l400K, corrigindo o
    ruler mismatch (antes era o encoder 48d puro).
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv"
    if not src.exists():
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv não encontrado")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K_flim_nonorm",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K_flim_nonorm: {len(result)} rows from results/svm_2l_1x1_init_flim_256_1280_nonorm_proj1280.csv")
    return result


# ── Source 7b: svm_1x1_BN2d_1280_one_layer_flim_init_results.csv ─────────────

def _normalize_distill_1x1bn_flim() -> pd.DataFrame | None:
    """Normalize encoder_init=flim rows from results/svm_proj1280_1x1_BN2d_results.csv.

    method=SVM_Distill_1x1BN_flim, init=flim.
    Reads the same source as _normalize_distill_1x1bn() but filters encoder_init=flim,
    avoiding dependence on a separate split file that rsync can overwrite.
    Returns None se o arquivo não existir ou não tiver rows flim ok.
    """
    src = _RESULTS / "svm_proj1280_1x1_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv não encontrado (1x1BN flim)")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "flim")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_1x1_BN2d_results.csv sem rows ok com encoder_init=flim")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_1x1BN_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_1x1BN_flim: {len(result)} rows from results/svm_proj1280_1x1_BN2d_results.csv (flim only)")
    return result


# ── Source 7c: svm_3x3_BN2d_1280_one_layer_flim_init_results.csv ─────────────

def _normalize_distill_3x3bn_flim() -> pd.DataFrame | None:
    """Normalize encoder_init=flim rows from results/svm_proj1280_3x3_BN2d_results.csv.

    method=SVM_Distill_3x3BN_flim, init=flim.
    Reads the same source as _normalize_distill_3x3bn() but filters encoder_init=flim,
    avoiding dependence on a separate split file that rsync can overwrite.
    Returns None se o arquivo não existir ou não tiver rows flim ok.
    """
    src = _RESULTS / "svm_proj1280_3x3_BN2d_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv não encontrado (3x3BN flim)")
        return None

    df = pd.read_csv(src)
    df = df[(df["status"] == "ok") & (df["encoder_init"] == "flim")].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_3x3_BN2d_results.csv sem rows ok com encoder_init=flim")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_3x3BN_flim",
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_3x3BN_flim: {len(result)} rows from results/svm_proj1280_3x3_BN2d_results.csv (flim only)")
    return result


# ── Source 8: svm_proj1280_2l_1x1_BN2d_256_1280_results.csv ──────────────────

def _normalize_distill_2l_400k() -> pd.DataFrame | None:
    """Normalize results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv — proj head 2l 1x1 BN2d (48→256→1280, ~400K).

    method=SVM_Distill_2l400K, init=trunc_normal.
    Returns None se o arquivo ainda não existir ou não tiver rows ok.
    """
    src = _RESULTS / "svm_proj1280_2l_1x1_BN2d_256_1280_results.csv"
    if not src.exists():
        print("[SKIP] svm_proj1280_2l_1x1_BN2d_256_1280_results.csv não encontrado — rode svm_distill_with_projection.py --run-filter 2l_1x1_BN2d_256_1280 primeiro")
        return None

    df = pd.read_csv(src)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        print("[SKIP] svm_proj1280_2l_1x1_BN2d_256_1280_results.csv sem rows ok")
        return None

    df["dataset_short"]  = df["dataset"]
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (dataset_short, pct), grp in df.groupby(["dataset_short", "pretrained_pct"], sort=False):
        n = len(grp)
        agg_rows.append({
            "method":         "SVM_Distill_2l400K",
            "init":           "trunc_normal",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] SVM_Distill_2l400K: {len(result)} rows from results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv")
    return result


# ── Source 9: svm_flim_residual_eggs.csv ─────────────────────────────────────

def _normalize_flim_residual() -> pd.DataFrame | None:
    """Normalize results/svm_flim_residual_eggs.csv — FLIM residual encoders (eggs).

    Keeps the per-variant method names already present in the CSV
    (SVM_FLIMResidual_1_3 / SVM_FLIMResidual_2_3), aggregating mean ± std across
    splits per (method, dataset, pretrained_pct). Maps dataset_name
    (helminth-eggs) to the canonical short name (eggs). Skips non-ok / NaN rows.
    Returns None if the file is absent.
    """
    src = _RESULTS / "svm_flim_residual_eggs.csv"
    if not src.exists():
        print("[SKIP] svm_flim_residual_eggs.csv não encontrado — "
              "rode svm_flim_residual.py --flim_residual_assessment primeiro")
        return None

    df = pd.read_csv(src)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    df = df[df["kappa"].notna()].copy()  # ignore NaN
    if df.empty:
        print("[SKIP] svm_flim_residual_eggs.csv has no usable rows")
        return None

    df["dataset_short"]  = df["dataset_name"].map(
        lambda d: _LONG_DATASET_MAP.get(d, d)
    )
    df["pretrained_pct"] = df["percentage"].astype(int)

    agg_rows: list[dict] = []
    for (method, dataset_short, pct), grp in df.groupby(
        ["method", "dataset_short", "pretrained_pct"], sort=False
    ):
        n = len(grp)
        agg_rows.append({
            "method":         method,
            "init":           "flim",
            "dataset_short":  dataset_short,
            "pretrained_pct": int(pct),
            "n_splits":       n,
            "kappa":          grp["kappa"].mean(),
            "kappa_std":      grp["kappa"].std(ddof=1) if n > 1 else 0.0,
            "acc":            grp["acc"].mean(),
            "acc_std":        grp["acc"].std(ddof=1) if n > 1 else 0.0,
            "f1":             grp["f1"].mean(),
            "f1_std":         grp["f1"].std(ddof=1) if n > 1 else 0.0,
        })

    result = pd.DataFrame(agg_rows, columns=_CANONICAL_COLS)
    print(f"[NORM] FLIM residual (1_3/2_3): {len(result)} rows "
          f"from results/svm_flim_residual_eggs.csv")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def _safe(fn, partial: bool):
    """Call fn(); on failure return None if partial=True, else re-raise."""
    try:
        result = fn()
        return result
    except Exception as exc:
        if partial:
            print(f"[SKIP] {fn.__name__}: {exc}")
            return None
        raise



def normalize_reports(partial: bool = False) -> None:
    """Normaliza os CSVs legados de SVM no schema de ``artifacts/normalized/``.

    partial  pula fonte ausente em vez de falhar. Util quando ``artifacts/SVM/``
             ou outro arquivo nao esta disponivel localmente.

    Escreve ``artifacts/normalized/svm_flim_aggregated.csv``,
    ``artifacts/normalized/unified_svm_comparison.csv`` e os PNG de
    ``artifacts/plots/comparison/``. Idempotente: todo destino e sobrescrito,
    nenhum e apendado.
    """
    # headless-safe matplotlib. Era efeito colateral de import em
    # scripts/normalize_reports.py:45; aqui roda antes do import do plotter, que
    # e o unico ponto em que o matplotlib le a variavel. Ver armadilha 3.
    os.environ.setdefault("MPLBACKEND", "Agg")

    if partial:
        print("[MODE] partial=True: missing sources will be skipped.\n")

    from eval.eval_plotter import plot_method_comparison  # noqa: PLC0415

    out_dir = Path(ARTIFACTS_NORMALIZED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(ARTIFACTS_PLOTS_DIR) / "comparison"

    # ── Normalize each source (all optional in --partial mode) ────────────────
    flim_df           = _safe(normalize_felipe_svm, partial)
    lejepa_df         = _safe(load_lejepa_svm_metrics, partial)
    ijepa_df          = _safe(normalize_ijepa_svm, partial)
    distil_conv_df    = _safe(normalize_distillation_conv_svm, partial)
    distil_flim_frozen_df = _safe(_normalize_distill_flim_frozen, partial)
    distil_proj_df    = _safe(_normalize_distill_proj1280, partial)
    distil_3x3bn_df   = _safe(_normalize_distill_3x3bn, partial)
    distil_1x1bn_df      = _safe(_normalize_distill_1x1bn, partial)
    distil_1x1bn_nonorm_df = _safe(_normalize_distill_1x1bn_nonorm, partial)
    distil_1x1bn_flim_df = _safe(_normalize_distill_1x1bn_flim, partial)
    distil_3x3bn_flim_df = _safe(_normalize_distill_3x3bn_flim, partial)
    distil_2l400k_df      = _safe(_normalize_distill_2l_400k, partial)
    distil_2l400k_flim_df = _safe(_normalize_distill_2l_400k_flim, partial)
    distil_2l400k_flim_nonorm_df = _safe(_normalize_distill_2l_400k_flim_nonorm, partial)
    flim_residual_df  = _safe(_normalize_flim_residual, partial)

    # ── Save normalized FLIM SVM aggregated (if available) ────────────────────
    if flim_df is not None:
        flim_path = out_dir / "svm_flim_aggregated.csv"
        flim_df.to_csv(flim_path, index=False)
        print(f"[SAVE] {flim_path.relative_to(_ROOT)}  ({len(flim_df)} rows)")

    # ── Build and save unified comparison ─────────────────────────────────────
    dfs = [df for df in [flim_df, lejepa_df, ijepa_df] if df is not None]
    if distil_conv_df is not None:
        dfs.append(distil_conv_df)
    if distil_flim_frozen_df is not None:
        dfs.append(distil_flim_frozen_df)
    if distil_proj_df is not None:
        dfs.append(distil_proj_df)
    if distil_3x3bn_df is not None:
        dfs.append(distil_3x3bn_df)
    if distil_1x1bn_df is not None:
        dfs.append(distil_1x1bn_df)
    if distil_1x1bn_nonorm_df is not None:
        dfs.append(distil_1x1bn_nonorm_df)
    if distil_1x1bn_flim_df is not None:
        dfs.append(distil_1x1bn_flim_df)
    if distil_3x3bn_flim_df is not None:
        dfs.append(distil_3x3bn_flim_df)
    if distil_2l400k_df is not None:
        dfs.append(distil_2l400k_df)
    if distil_2l400k_flim_df is not None:
        dfs.append(distil_2l400k_flim_df)
    if distil_2l400k_flim_nonorm_df is not None:
        dfs.append(distil_2l400k_flim_nonorm_df)
    if flim_residual_df is not None:
        dfs.append(flim_residual_df)

    if not dfs:
        print("[ERROR] No data sources available. Nothing to save.")
        return

    unified = pd.concat(dfs, ignore_index=True)
    unified_path = Path(UNIFIED_SVM_COMPARISON_CSV)
    unified.to_csv(unified_path, index=False)
    print(f"[SAVE] {unified_path.relative_to(_ROOT)}  ({len(unified)} rows)")

    # ── Count validation ───────────────────────────────────────────────────────
    print("\n[VALIDATE] Row counts per method:")
    for method, grp in unified.groupby("method"):
        print(f"  {method}: {len(grp)} rows")

    # ── Generate comparison plots ──────────────────────────────────────────────
    print("\n[PLOTS] Generating comparison plots ...")
    datasets = sorted(unified["dataset_short"].unique())
    for dataset in datasets:
        ds_df = unified[unified["dataset_short"] == dataset]
        for metric in METRICS:
            plot_method_comparison(ds_df, dataset, metric, plots_dir)

    print(
        f"\n[DONE] Outputs written to:\n"
        f"  {out_dir.relative_to(_ROOT)}/\n"
        f"  {plots_dir.relative_to(_ROOT)}/"
    )


if __name__ == "__main__":
    generate_mlp_configs()
