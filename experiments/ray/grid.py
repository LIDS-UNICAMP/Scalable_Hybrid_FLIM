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

"""grid.py — produto cartesiano da secao `grid` do experiment YAML.

Uma chamada, `build(exp)`, devolve `(celulas, puladas)`: o que vai para a fila
do Ray e o que nao vai, cada pulada com o motivo escrito por extenso para o
manifesto.

Um eixo omitido no YAML (``None`` em `schema.Grid`) cai no default, que e o
conjunto de valores validos daquele eixo declarado em `schema.py:40-43`. Nao ha
tabela de default por metodo em disco; quando houver, e essa a linha que muda.

Absorve as 3 copias de `build_experiment_grid` / `validate_experiment` /
`validate_output_dir` de `scripts/{distillation,distillation_conv,
autoencoder_flim,classification_flim}_ray.py`. As copias tinham de 3 a 5 `for`
aninhados; aqui e `itertools.product`, um `for` so.

Nao ha `_log` aqui: as copias imprimiam cada pulo la dentro e cada uma tinha o
proprio `_log`. O motivo ja viaja no registro de pulada — quem imprime e o
launcher, um lugar so.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

from core.constants import NUM_CLASSES
from experiments.constants import DATASET_LONG_TO_SHORT, WRITE_TEST_FILENAME
from experiments.ray import paths
from experiments.ray.schema import DATASETS, INITS, PERCENTAGES, SPLITS, Experiment
from experiments.ray.skip import should_skip


@dataclass(frozen=True)
class Cell:
    """Uma celula da grade, com todo caminho ja resolvido e existente."""

    dataset: str
    split: int
    percentage: int
    init: str
    run_name: str
    num_classes: int
    dataset_config: str
    model_config: str
    arch_json: str
    flim_weights_path: str


def build(exp: Experiment) -> tuple[list[Cell], list[dict]]:
    """Expande a grade e valida cada celula antes de deixar entrar na fila."""
    cells: list[Cell] = []
    skipped: list[dict] = []

    for dataset, split, pct, init in itertools.product(
        exp.grid.datasets or list(DATASETS),
        exp.grid.splits or list(SPLITS),
        exp.grid.percentages or list(PERCENTAGES),
        exp.grid.init or list(INITS),
    ):
        # `method`/`runner` sao keyword-only e obrigatorios: a formula do nome
        # e a HISTORICA da familia, e familia errada = diretorio de peso errado.
        run_name = paths.run_name(exp.name, dataset, split, pct, init,
                                  method=exp.method, runner=exp.runner)
        axes = {
            "run_name": run_name,
            "dataset": dataset,
            "split": split,
            "percentage": pct,
            "init": init,
        }

        try:
            cell = _cell(exp, dataset, split, pct, init, run_name)
        except FileNotFoundError as exc:
            skipped.append({**axes, "status": "skipped", "reason": str(exc)})
            continue

        done, why = should_skip(
            run_name,
            exp.skip,
            exp.output.work_dir,
            exp.wandb.entity,
            exp.wandb.project,
        )
        if done:
            skipped.append({**axes, "status": "skipped_existing", "reason": why})
            continue

        cells.append(cell)

    return cells, skipped


def validate_output_dir(work_dir: str) -> tuple[bool, str]:
    """(ok, motivo) — cria `work_dir` e prova que da para escrever nele.

    Escreve e apaga um arquivo de verdade, em vez de perguntar ao `os.access`:
    montagem de rede so-leitura responde que da e falha na hora do checkpoint.
    """
    try:
        os.makedirs(work_dir, exist_ok=True)
        test = os.path.join(work_dir, WRITE_TEST_FILENAME)
        with open(test, "w") as fh:
            fh.write("ok")
        os.remove(test)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _cell(
    exp: Experiment, dataset: str, split: int, pct: int, init: str, run_name: str
) -> Cell:
    """Monta a celula; levanta FileNotFoundError no primeiro caminho ausente."""
    arch_json = paths.arch_json(dataset, split)
    weights = paths.flim_weights_path(dataset, split)

    split_json = paths.split_json(dataset, split, pct)
    if not os.path.isfile(split_json):
        raise FileNotFoundError(f"split JSON ausente, esperado em: {split_json}")

    # Peso FLIM so e insumo de quem inicializa com FLIM; para xavier/he/random a
    # ausencia do diretorio nao e motivo de pulo.
    if init == "flim":
        if not os.path.isfile(arch_json):
            raise FileNotFoundError(f"arch JSON ausente, esperado em: {arch_json}")
        if not os.path.isdir(weights):
            raise FileNotFoundError(f"pesos FLIM ausentes, esperados em: {weights}")

    return Cell(
        dataset=dataset,
        split=split,
        percentage=pct,
        init=init,
        run_name=run_name,
        num_classes=NUM_CLASSES[DATASET_LONG_TO_SHORT[dataset]],
        dataset_config=paths.dataset_config(dataset, split),
        model_config=paths.model_config(exp.method, init, dataset, split),
        arch_json=arch_json,
        flim_weights_path=weights,
    )


if __name__ == "__main__":
    # Check em CPU, em memoria: python -m experiments.ray.grid <yaml>
    import sys
    from collections import Counter

    import torch

    from core.constants import PARASITE_NAME
    from core.data import ParasiteDataModule
    from experiments.ray.schema import load
    from flim.spifil import train_samples

    exp = load(sys.argv[1])
    no_crop = exp.runner_args.get("no_crop", False)
    size = exp.runner_args.get("image_size", 200)
    cells, skipped = build(exp)
    print(f"{len(cells)} celulas, {len(skipped)} puladas", *{s["reason"] for s in skipped})
    for dataset in exp.grid.datasets:
        short = DATASET_LONG_TO_SHORT[dataset]
        assert any(c.dataset == dataset for c in cells), dataset
        dm = ParasiteDataModule(parasite_name=PARASITE_NAME[short], split=1, percentage=50,
                                V_train=1, imagenet_norm=False, batch_size=4, num_workers=0,
                                image_size=size, no_crop=no_crop)
        dm.setup()
        views, _ = next(iter(dm.train_dataloader()))
        assert views.shape == (4, 1, 3, size, size) and torch.isfinite(views).all(), views.shape
        picked = train_samples(short, 1, 50, 200, one_per_class=True, seed=42)
        assert not any(s.mask_path for s in picked), "mascara no grow"
        print(f"\n{dataset}: lote {tuple(views.shape)}, grow {len(picked)} imagens, sem mascara")
        for name, ds in (("train", dm.ds_train), ("val", dm.ds_val), ("test", dm.ds_test)):
            base = ds.base
            y = Counter(base.labels.get(os.path.basename(p), int(os.path.basename(p).split("_")[0]) - 1)
                        for p in base.samples)
            assert sorted(y) == list(range(max(y) + 1)) and max(y) < NUM_CLASSES[short], sorted(y)
            print(f"  {name:5} {len(base):6} {[y[k] for k in sorted(y)]}")
