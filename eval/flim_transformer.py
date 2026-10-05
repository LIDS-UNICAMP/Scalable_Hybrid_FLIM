# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
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

"""flim_transformer.py — SPiFiL (1 camada) + CrossTransformer, tudo sem backprop.

Mesmo pipeline da interface `frontend_hybrid_spifil_net` (`python -m app.transformer_app`,
`app/adapters/transformer_runner.py`), sem a tela, com os defaults dela:

    treino do pct -> 1 imagem por classe -> SPiFiL fita a camada 1 (n_kernels prototipos)
                  -> CrossTransformer.fit (forma fechada: mu, sigma, delta[l], gamma[l][h])
                     tokens = superpixels SLIC da imagem LAB; cross-attention aos prototipos
    todas as imagens de treino do pct -> mapa da camada 1 -> g do transformer
                  -> cabeca classica (logreg balanceada) -> predicao no teste

Metricas pela `core.metrics.compute_metrics` do repo (kappa, acc balanceada, f1 macro).
Nada aqui chama `.backward()`: SPiFiL e transformer sao estimados em forma fechada e a
cabeca e um sklearn, como o SVM das outras sondas.

Por celula (dataset, split, pct) grava em `artifacts/flim_transformer/<ds>_split<N>_pct<P>/`:
    spifil/        model.pt, architecture.json, labels1.txt  (Learner.export)
    transformer.pt state_dict do transformer (mu, sigma, delta, gamma, prototipos) + config
    head.joblib    cabeca classica
e uma linha em `results/flim_transformer_results.csv`, que o
`experiments/gen_configs.py` agrega no CSV unificado do `eval.eval_plotter`.

Uso:
    python -m eval.flim_transformer --dry-run
    python -m eval.flim_transformer --datasets eggs --splits 1 --percentages 5
    python -m eval.flim_transformer            # 3 datasets x 3 splits x 6 pcts
"""

import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import torch
from spifil import ArchSpec, LayerSpec, Learner, SpifilDataset
from spifil.nn.encoder import SpifilEncoder
from spifil.superpixels.slic import SLIC

from config import get_single_parasite_paths
from core.constants import NUM_CLASSES, PARASITE_NAME, PROJECT_ROOT
from core.metrics import compute_metrics
from methods.transformer import diagnostics
from methods.transformer.classifier import fit_classifier
from methods.transformer.config import CrossTransformerConfig
from methods.transformer.crossformer import CrossTransformer

_METHOD = "FLIM_Transformer"
_OUT_CSV = os.path.join(PROJECT_ROOT, "results", "flim_transformer_results.csv")
_WEIGHTS = os.path.join(PROJECT_ROOT, "artifacts", "flim_transformer")
_FIELDS = ["method", "dataset", "split", "percentage", "n_train", "n_test",
           "n_kernels", "dim", "fallback_frac", "kappa", "acc", "f1",
           "weights_dir", "fit_s", "status", "error"]
_CHUNK = 64  # ponytail: imagens por passada no encoder; suba se a memoria deixar
# ponytail: so SLIC; DISF precisa do pyift (so Linux x86_64), troque aqui se superpixel_method="disf"
_SEGMENTERS = {"slic": SLIC}


def _split_samples(dataset: str, split: int, percentage: int):
    """Amostras de treino e teste do descritor incremental (treino e validacao disjuntos)."""
    info = get_single_parasite_paths(PARASITE_NAME[dataset], split, percentage)[0]
    names = json.loads(Path(info["split_json"]).read_text())
    by_name = {s.image_path.name: s for s in
               SpifilDataset.from_folders(info["images_dir"], info["masks_dir"])}
    return ([by_name[n] for n in names["train"]], [by_name[n] for n in names["test"]])


def _one_per_class(samples, seed: int):
    """Default da interface: `random_per_class`, uma imagem por classe, seed fixa."""
    rng = random.Random(seed)
    return [rng.choice([s for s in samples if s.label == c])
            for c in sorted({s.label for s in samples})]


def _maps_and_segs(learner: Learner, encoder: SpifilEncoder, data: SpifilDataset,
                   idx, cfg: CrossTransformerConfig):
    """Mapas da camada 1 e superpixels da imagem LAB inteira, sem mascara (como o runner)."""
    maps, segs = [], []
    for i in idx:
        image = data.load_image(i)
        maps.append(encoder(image[None], upto=1)[0])
        if cfg.tokenizer != "grid":
            segs.append(_SEGMENTERS[cfg.superpixel_method]()(
                learner.color(image), None, cfg.n_superpixels))
    return maps, (segs or None)


def _encode(model: CrossTransformer, learner: Learner, encoder: SpifilEncoder,
            data: SpifilDataset, cfg: CrossTransformerConfig) -> np.ndarray:
    """g do transformer para cada imagem de ``data``, na ordem."""
    rows = []
    with torch.no_grad():
        for start in range(0, len(data), _CHUNK):
            idx = range(start, min(start + _CHUNK, len(data)))
            rows.append(model.encode(*_maps_and_segs(learner, encoder, data, idx, cfg)))
    return np.concatenate(rows)


def run_cell(dataset: str, split: int, percentage: int, cfg: CrossTransformerConfig,
             n_superpixels: int, device: str) -> dict:
    train, test = _split_samples(dataset, split, percentage)

    # SPiFiL: defaults do Learner == conf/config.yaml do SPiFiL == defaults da interface.
    # A arch e a flim3 cortada na camada 1, com out_channels = n_kernels do transformer.
    learner = Learner(
        SpifilDataset(_one_per_class(train, cfg.seed)),
        ArchSpec(layers=[LayerSpec(kernel_size=3, out_channels=cfg.n_kernels)]),
        n_superpixels=n_superpixels, device=device,
    )
    learner.prepare()
    learner.fit_layer(1)
    encoder = SpifilEncoder(learner.color, learner.model).eval()

    # Prototipos = banco da camada 1: classe de cada kernel e a imagem de onde ele veio.
    state1, data = learner.state[1], learner.data
    kernel_source = None if state1.selected is None else state1.selected.image_ids.long()
    masks = [None if m is None else m != 0 for m in (data.load_mask(i) for i in range(len(data)))]
    with torch.no_grad():
        maps, segs = _maps_and_segs(learner, encoder, data, range(len(data)), cfg)
        model = CrossTransformer(cfg).fit(maps, segs, masks, state1.bank.labels.long(), kernel_source)
        outs = [model.forward(x[None], None if segs is None else [s])
                for x, s in zip(maps, segs or [None] * len(maps))]
    fallback = float(torch.stack([diagnostics.fallback_fraction(o)[:, 0] for o in outs]).mean())

    g_train = _encode(model, learner, encoder, SpifilDataset(train), cfg)
    g_test = _encode(model, learner, encoder, SpifilDataset(test), cfg)
    # rotulo do nome do arquivo e 1-indexado; compute_metrics quer 0-indexado
    y_train = np.asarray([s.label - 1 for s in train])
    y_test = np.asarray([s.label - 1 for s in test])
    head = fit_classifier(g_train, y_train, cfg.classifier, seed=cfg.seed)
    metrics = compute_metrics(y_test, head.predict(g_test), NUM_CLASSES[dataset])

    out = Path(_WEIGHTS) / f"{dataset}_split{split}_pct{percentage}"
    learner.export(out / "spifil")
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out / "transformer.pt")
    joblib.dump(head, out / "head.joblib")

    return {"n_train": len(train), "n_test": len(test),
            # o alocador divide por classe e pode entregar menos que o pedido
            "n_kernels": len(state1.bank.labels),
            "dim": int(g_test.shape[1]), "fallback_frac": fallback,
            "weights_dir": str(out), **metrics}


def main(
    datasets: list[str] | None = None,
    splits: list[int] | None = None,
    percentages: list[int] | None = None,
    n_superpixels: int = 100,
    device: str = "cpu",
    out: str = _OUT_CSV,
    dry_run: bool = False,
) -> None:
    # `cli_kwargs` devolve escalar quando so um valor e passado
    as_list = lambda v, d: d if v is None else (v if isinstance(v, list) else [v])  # noqa: E731
    datasets = as_list(datasets, ["eggs", "larvae", "protozoan"])
    splits = as_list(splits, [1, 2, 3])
    percentages = as_list(percentages, [1, 5, 25, 50, 75, 100])
    cfg = CrossTransformerConfig()

    if dry_run:
        for ds in datasets:
            for split in splits:
                for pct in percentages:
                    train, test = _split_samples(ds, split, pct)
                    print(f"  [OK ] {ds} split{split} pct{pct:<3d} train={len(train)} "
                          f"test={len(test)} classes={len({s.label for s in train})}")
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):  # retoma: pula celula ja gravada como ok
        with open(out) as fh:
            done = {(r["dataset"], int(r["split"]), int(r["percentage"]))
                    for r in csv.DictReader(fh) if r["status"] == "ok"}
    else:
        with open(out, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_FIELDS).writeheader()

    for ds in datasets:
        for split in splits:
            for pct in percentages:
                if (ds, split, pct) in done:
                    print(f"[SKIP] {ds} split{split} pct{pct} ja no CSV")
                    continue
                row = {"method": _METHOD, "dataset": ds, "split": split, "percentage": pct}
                start = time.perf_counter()
                try:
                    row.update(run_cell(ds, split, pct, cfg, n_superpixels, device), status="ok")
                    print(f"[RESULT] {ds} split{split} pct{pct:<3d} kappa={row['kappa']:+.4f} "
                          f"acc={row['acc']:.4f} f1={row['f1']:.4f}")
                except Exception as exc:  # noqa: BLE001  uma celula ruim nao derruba a grade
                    row.update(status="error", error=repr(exc))
                    print(f"[ERROR] {ds} split{split} pct{pct}: {exc!r}")
                row["fit_s"] = round(time.perf_counter() - start, 1)
                with open(out, "a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=_FIELDS).writerow(row)
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    # Import local: eval.svm puxa pyift (so Linux x86_64); a biblioteca roda sem ele.
    from eval.svm import cli_kwargs

    main(**cli_kwargs(sys.argv[1:]))
