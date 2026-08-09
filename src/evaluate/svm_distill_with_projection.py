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
"""svm_distill_with_projection.py — SVM com Red Projection ativa (embedding [B, 1280]).

Diferença em relação a svm_distillation_conv.py:
  * svm_distillation_conv.py  → corta a proj head → embedding [B, 48]
  * este script               → mantém a proj head → embedding [B, 1280]

Fluxo do embedding:
    input [B, 3, 200, 200]
    → FLIM Encoder (conv1→conv2→conv3)        →  [B, 48, 24, 24]
    → ConvDistillationProjectionHead (1×1 convs) →  [B, 1280]   ← usado no SVM

Isso permite avaliar se a Red Projection aprendeu representações discriminativas
que o encoder sozinho não captura. Comparação justa contra:
  - SVM_Distillation_Conv  (este projeto, embedding [B, 48])
  - SVM_FLIM               (FLIM supervisionado)
  - SVM_IJEPA              (I-JEPA 1280-dim)
  - SVM_LeJEPA_trunc_normal

Resultado salvo em: results/svm_distill_proj1280_results.csv
Method name no CSV: SVM_Distill_Proj1280

Usage::

    python -m src.evaluate.svm_distill_with_projection
    python -m src.evaluate.svm_distill_with_projection --wandb-update
    python -m src.evaluate.svm_distill_with_projection --run eggs_split1_pct100
"""
from __future__ import annotations

import argparse
import os
import sys
import threading

import numpy as np
import pandas as pd
import torch
from sklearn import svm as sk_svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reutiliza infraestrutura comum ────────────────────────────────────────────
from src.evaluate.svm_distillation import (
    _OneHotDataset,
    find_distillation_runs,
    _ARTIFACTS_DIR,
    _RESULTS_DIR,
    DEVICE,
    IMAGE_SIZE,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
)
from src.data_modules.datasets.dataset import DatasetParasite
from src.data_modules.datasets.lejepa_dataset import _build_test
from src.metrics.classification import compute_metrics
from src.utils.evaluate import SVM_DIAG_MISSING, fit_svm_with_diagnostics
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.models.distillation import (
    ConvDistillationProjectionHead,
    OneLayerConvDistillationProjectionHead,
    OneLayer1x1ConvDistillationProjectionHead,
    TwoLayer1x1ConvBN2dDistillationProjectionHead,
    TEACHER_DIM,
)
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
)

_EMBED_DIM = TEACHER_DIM  # 1280


# ── Carrega student + proj_kd sem instanciar o teacher (I-JEPA) ───────────────

def _load_student_and_proj(
    ckpt_path: str,
    device: torch.device,
) -> tuple[LeJEPAFLIMModel, torch.nn.Module]:
    """Extrai student + proj_kd do checkpoint sem carregar o I-JEPA teacher.

    Auto-detecta o tipo de proj_kd pelo shape de proj_kd.proj.0.weight:
      - out_channels=1280  →  OneLayerConvDistillationProjectionHead (3×3 conv)
      - out_channels=128   →  ConvDistillationProjectionHead (4× 1×1 convs)

    Quando encoder_init='flim', os canais reais são detectados via bias files
    (ex: protozoan usa 30 canais na conv2, não 32 como no architecture.json).
    """
    # Carrega TUDO na CPU (o ckpt inclui o teacher I-JEPA ~2.5GB). Só o student +
    # proj_kd vão para a GPU depois via .to(device) — o teacher nunca toca a GPU.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams           = ckpt.get("hyper_parameters", {})
    arch_json         = hparams.get("arch_json", "")
    in_channels       = hparams.get("in_channels", 3)
    proj_dim          = hparams.get("proj_dim", 256)
    proj_hidden       = hparams.get("proj_hidden", 2048)
    encoder_init      = hparams.get("encoder_init", "trunc_normal")
    flim_weights_path = hparams.get("flim_weights_path", None)

    arch = parse_architecture(arch_json)

    # Para flim_init os canais reais podem diferir do architecture.json
    # (ex: protozoan conv2 tem 30 canais, não 32)
    if encoder_init == "flim" and flim_weights_path:
        channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
        arch = override_arch_channels(arch, channels)

    student = LeJEPAFLIMModel(
        arch=arch, in_channels=in_channels,
        proj_dim=proj_dim, proj_hidden=proj_hidden,
    )

    full_sd = ckpt["state_dict"]

    # Auto-detecta tipo e kernel_size pelo shape de proj_kd.proj.0.weight → [out, in, kH, kW]
    # out_channels == 1280, kH == 1 → 1×1 (OneLayer1x1Conv...)
    # out_channels == 1280, kH == 3 → 3×3 (OneLayerConv...)
    # out_channels != 1280, proj.3 → 1280 → 2l 1×1 BN2d 256→1280 (TwoLayer1x1ConvBN2d...)
    # out_channels != 1280          → chain 4× 1×1 (ConvDistillationProjectionHead)
    first_w  = full_sd.get("proj_kd.proj.0.weight")
    third_w  = full_sd.get("proj_kd.proj.3.weight")
    if first_w is not None and first_w.shape[0] == _EMBED_DIM:
        if first_w.shape[2] == 1:
            proj_kd = OneLayer1x1ConvDistillationProjectionHead(
                student_channels=student.embed_dim,
                teacher_dim=_EMBED_DIM,
            )
        else:
            proj_kd = OneLayerConvDistillationProjectionHead(
                student_channels=student.embed_dim,
                teacher_dim=_EMBED_DIM,
            )
    elif (
        first_w is not None and first_w.shape[2] == 1
        and third_w is not None and third_w.shape[0] == _EMBED_DIM
    ):
        # 2l_1x1_BN2d_256_1280: Conv1x1(48→mid) + BN + GELU + Conv1x1(mid→1280) + BN + GELU
        proj_kd = TwoLayer1x1ConvBN2dDistillationProjectionHead(
            student_channels=student.embed_dim,
            mid_channels=first_w.shape[0],
            teacher_dim=_EMBED_DIM,
        )
    else:
        proj_kd = ConvDistillationProjectionHead(
            student_channels=student.embed_dim,
            teacher_dim=_EMBED_DIM,
        )

    student_sd = {k[len("student."):]: v for k, v in full_sd.items() if k.startswith("student.")}
    proj_sd    = {k[len("proj_kd."):]: v for k, v in full_sd.items() if k.startswith("proj_kd.")}

    student.load_state_dict(student_sd, strict=True)
    proj_kd.load_state_dict(proj_sd,   strict=True)

    return student.to(device), proj_kd.to(device)


# ── Feature extraction — encoder + proj head [B, 1280] ───────────────────────

@torch.no_grad()
def _extract_proj(
    student:  LeJEPAFLIMModel,
    proj_kd:  ConvDistillationProjectionHead,
    loader:   DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai embeddings [B, 1280] mantendo a Red Projection ativa."""
    student.eval()
    proj_kd.eval()
    student.to(DEVICE)
    proj_kd.to(DEVICE)

    feats, labels = [], []
    for x, y in tqdm(loader, desc="  features [1280]", leave=False):
        feat_map = student.encoder(x.to(DEVICE))   # [B, 48, H', W']
        emb      = proj_kd(feat_map)                # [B, 1280]
        feats.append(emb.cpu().numpy())
        labels.extend(y.tolist() if isinstance(y, torch.Tensor) else y)

    return np.concatenate(feats), np.array(labels, dtype=np.int64)


# ── SVM ────────────────────────────────────────────────────────────────────────

def _train_svm_proj(
    student:  LeJEPAFLIMModel,
    proj_kd:  ConvDistillationProjectionHead,
    loader:   DataLoader,
    C:        float = 1e2,
    max_iter: int   = -1,
) -> Pipeline:
    """Treina o SVM da arm Proj1280 (labels 1-indexed).

    ``max_iter=-1`` (padrão) = solver sem limite; passe o cap antigo
    explicitamente só para reproduzir um CSV histórico. Diagnósticos do solver
    ficam em ``clf.fit_diagnostics_`` (leem do step ``svm`` do Pipeline).
    """
    # StandardScaler é necessário com 1280 dims para convergência do SVM linear.
    # Esta arm escala, outras não — a assimetria NÃO é unificada aqui, apenas
    # registrada na coluna svm_protocol.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    sk_svm.SVC(
            # Unbounded solver: results are deliberately NOT comparable with the
            # CSVs produced under the old max_iter cap.
            C=C, gamma="auto", kernel="linear",
            decision_function_shape="ovo", max_iter=max_iter,
        )),
    ])
    student.eval()
    proj_kd.eval()
    student.to(DEVICE)
    proj_kd.to(DEVICE)

    feats, ys = [], []
    for x, y_oh in tqdm(loader, desc="  train feats [1280]", leave=False):
        feat_map = student.encoder(x.to(DEVICE))
        emb      = proj_kd(feat_map)
        feats.append(emb.cpu().numpy())
        ys.extend((np.argmax(y_oh.numpy(), axis=1) + 1).tolist())  # 1-indexed

    X, y = np.concatenate(feats), np.array(ys, dtype=np.int64)

    stop = threading.Event()
    def _prog():
        with tqdm(desc="  SVM fit", unit="s", bar_format="{desc}: {elapsed}") as pb:
            while not stop.wait(1.0): pb.update(1)
    t = threading.Thread(target=_prog, daemon=True)
    t.start()
    fit_svm_with_diagnostics(clf, X, y, tag="SVM_Distill_Proj1280")
    stop.set(); t.join()
    return clf


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "SVM com Red Projection ativa — embedding [B, 1280] "
            "(encoder FLIM + ConvDistillationProjectionHead)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run",           default=None,
                        help="Substring filter no nome do run.")
    parser.add_argument("--run-filter",    default=None,
                        help="Sobrescreve o filtro base (padrão: next_layers_direct).")
    parser.add_argument("--output-csv",    default=None,
                        help="Nome do CSV de saída (em results/). Derivado do filtro se omitido.")
    parser.add_argument("--artifacts-dir", default=_ARTIFACTS_DIR)
    parser.add_argument("--wandb-update",  action="store_true")
    parser.add_argument("--wandb-entity",  default="ophira-ai")
    parser.add_argument("--wandb-project", default="flim-ssl")
    parser.add_argument("--no-imagenet-norm", action="store_true",
                        help="Eval transform stays LAB[0,1] (no ImageNet RGB norm). "
                             "Use for checkpoints trained with --no-imagenet-norm.")
    parser.add_argument("--only-ok", action="store_true",
                        help="Skip runs whose run_metadata.json status != 'ok' "
                             "(safe to run while other runs are still training).")
    args = parser.parse_args()

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    transform = _build_test(IMAGE_SIZE, imagenet_norm=not args.no_imagenet_norm)

    base_filter = args.run_filter if args.run_filter else "next_layers_direct"
    run_filter = base_filter
    if args.run:
        run_filter = args.run if base_filter in args.run else f"{base_filter}_{args.run}"

    runs = find_distillation_runs(args.artifacts_dir, run_filter=run_filter)
    if args.only_ok:
        runs = [r for r in runs if r.get("status") == "ok"]
    if not runs:
        print(f"[WARN] Nenhum run com filtro '{run_filter}' e checkpoint encontrado.")
        return

    # ── Resolve nomes de CSV e pasta parcial antes do loop ───────────────────
    if args.output_csv:
        csv_name = args.output_csv if args.output_csv.endswith(".csv") else f"{args.output_csv}.csv"
    else:
        safe = base_filter.replace("/", "_").replace(" ", "_")
        csv_name = f"svm_proj1280_{safe}_results.csv"
    csv_path    = os.path.join(_RESULTS_DIR, csv_name)
    csv_stem    = csv_name[:-4]  # sem ".csv"
    partial_dir = os.path.join(_RESULTS_DIR, f"partial_{csv_stem}")
    os.makedirs(partial_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SVM — distill_with_red_projection_1280  |  embedding [B, 1280]")
    print(f"Runs: {len(runs)}")
    print(f"Parciais em: {partial_dir}")
    print(f"{'='*70}")

    rows: list[dict] = []

    for meta in runs:
        run_name  = meta.get("run_name", os.path.basename(meta["_run_dir"]))
        ckpt_path = meta["_ckpt_path"]
        dataset   = meta.get("dataset", "")
        split     = int(meta.get("split", 1))
        pct       = int(meta.get("percentage", 100))

        print(f"\n── {run_name}")
        print(f"   ckpt: {os.path.relpath(ckpt_path, _ROOT)}")

        base = {
            "run_name":          run_name,
            "method":            "SVM_Distill_Proj1280",
            "dataset":           dataset,
            "split":             split,
            "percentage":        pct,
            "distillation_type": meta.get("distillation_type", "direct"),
            "encoder_init":      meta.get("encoder_init", "trunc_normal"),
            "student_embed_dim": _EMBED_DIM,
            "proj_head":         "conv_next_layers_active",
            "ckpt_path":         ckpt_path,
        }

        num_classes   = _DATASET_NUM_CLASSES.get(dataset, 9)
        parasite_name = _DATASET_PARASITE_NAME.get(dataset, dataset)

        try:
            student, proj_kd = _load_student_and_proj(ckpt_path, DEVICE)
            for p in list(student.parameters()) + list(proj_kd.parameters()):
                p.requires_grad_(False)

            print(f"   embedding: encoder[B,{student.embed_dim}] → proj_kd → [B,{_EMBED_DIM}]")

            # Train
            train_ds = DatasetParasite(
                set_name="train", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_ds, num_classes),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            clf = _train_svm_proj(student, proj_kd, train_loader)

            # Eval
            test_ds = DatasetParasite(
                set_name="test", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            feats, y_true = _extract_proj(student, proj_kd, test_loader)
            y_pred = clf.predict(feats) - 1  # volta para 0-indexed

            metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
            # Diagnósticos do solver viajam no Pipeline (fit_svm_with_diagnostics).
            diag = getattr(clf, "fit_diagnostics_", SVM_DIAG_MISSING)
            row = {**base, **metrics, **diag, "status": "ok", "error": ""}
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                os.path.join(partial_dir, f"{run_name}.csv"), index=False
            )
            print(f"   kappa={metrics['kappa']:.4f}  acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}"
                  f"  fit_status={diag['svm_fit_status']}  n_sv={diag['svm_n_sv']}")

            if args.wandb_update:
                try:
                    import wandb as _wandb  # noqa: PLC0415
                    wr = _wandb.init(
                        project=args.wandb_project, entity=args.wandb_entity,
                        name=f"svm_distil_proj1280_{run_name}",
                        config={**base, "num_classes": num_classes}, reinit=True,
                    )
                    _wandb.log({f"svm/{k}": v for k, v in metrics.items()})
                    wr.finish()
                except Exception as we:
                    print(f"   [WARN] W&B: {we}")

        except Exception as exc:
            print(f"   [ERROR] {exc}")
            row = {**base, "kappa": float("nan"), "acc": float("nan"),
                   "f1": float("nan"), **SVM_DIAG_MISSING,
                   "status": "error", "error": str(exc)}
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                os.path.join(partial_dir, f"{run_name}.csv"), index=False
            )

    # ── CSV final ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    ok = df[df["status"] == "ok"]
    print(f"\n{'='*70}")
    print(f"[DONE]  {csv_path}")
    print(f"        runs: {len(rows)}  ok: {len(ok)}  erros: {len(rows)-len(ok)}")
    if len(ok):
        print(f"        kappa médio: {ok['kappa'].mean():.4f} ± {ok['kappa'].std():.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
