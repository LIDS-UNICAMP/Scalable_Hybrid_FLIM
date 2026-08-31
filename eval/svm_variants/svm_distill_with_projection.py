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

    python -m eval.svm_variants.svm_distill_with_projection
    python -m eval.svm_variants.svm_distill_with_projection --wandb-update
    python -m eval.svm_variants.svm_distill_with_projection --run eggs_split1_pct100
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Reutiliza infraestrutura comum ────────────────────────────────────────────
from eval.svm_variants.svm_distillation import (
    _OneHotDataset,
    find_distillation_runs,
    _ARTIFACTS_DIR,
    _RESULTS_DIR,
    DEVICE,
    _DATASET_NUM_CLASSES,
    _DATASET_PARASITE_NAME,
)
from core.constants import IMAGE_SIZE, PROJECT_ROOT
from core.data.parasite_dataset import ParasiteDataset
from core.data.transforms import build_test
from core.metrics import compute_metrics
from eval.svm import (
    SVM_DIAG_MISSING,
    cli_kwargs,
    extract_proj_features,
    fit_svm,
)
from methods.lejepa.lejepa_flim_model import LeJEPAFLIMModel
from methods.distillation import (
    ConvDistillationProjectionHead,
    OneLayerConvDistillationProjectionHead,
    OneLayer1x1ConvDistillationProjectionHead,
    TwoLayer1x1ConvBN2dDistillationProjectionHead,
)
# TEACHER_DIM fica fora do __init__ curto de propósito (methods/distillation/
# __init__.py:57): vem pelo módulo, como em eval/tsne.py.
from methods.distillation.teacher_constants import TEACHER_DIM
from flim.arch import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
)


_ROOT = PROJECT_ROOT
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

    # StandardScaler é necessário com 1280 dims para convergência do SVM linear.
    # Esta arm escala, outras não — a assimetria NÃO é unificada aqui, apenas
    # registrada na coluna svm_protocol.
    return fit_svm(X, y, C=C, max_iter=max_iter, scaler=True, tag="SVM_Distill_Proj1280")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    run: str | None = None,
    run_filter: str | None = None,
    output_csv: str | None = None,
    artifacts_dir: str = _ARTIFACTS_DIR,
    wandb_update: bool = False,
    wandb_entity: str = "ophira-ai",
    wandb_project: str = "flim-ssl",
    no_imagenet_norm: bool = False,
    only_ok: bool = False,
) -> None:
    """SVM com a projecao ativa — embedding [B, 1280] (encoder FLIM + proj head).

    Args:
        run:              Substring de filtro no nome do run (`--run`).
        run_filter:       Sobrescreve o filtro base (`--run-filter`); default
                          `next_layers_direct`, resolvido no corpo.
        output_csv:       Nome do CSV em `results/` (`--output-csv`); derivado do
                          filtro quando omitido.
        artifacts_dir:    Raiz dos artefatos (`--artifacts-dir`).
        wandb_update:     Loga no W&B (`--wandb-update`).
        wandb_entity:     Entidade do W&B (`--wandb-entity`).
        wandb_project:    Projeto do W&B (`--wandb-project`).
        no_imagenet_norm: Transform de eval fica em LAB[0,1] (`--no-imagenet-norm`).
        only_ok:          Pula runs com `status != "ok"` (`--only-ok`).
    """

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    transform = build_test(IMAGE_SIZE, imagenet_norm=not no_imagenet_norm)

    base_filter = run_filter if run_filter else "next_layers_direct"
    resolved_filter = base_filter
    if run:
        resolved_filter = run if base_filter in run else f"{base_filter}_{run}"

    runs = find_distillation_runs(artifacts_dir, run_filter=resolved_filter)
    if only_ok:
        runs = [r for r in runs if r.get("status") == "ok"]
    if not runs:
        print(f"[WARN] Nenhum run com filtro '{resolved_filter}' e checkpoint encontrado.")
        return

    # ── Resolve nomes de CSV e pasta parcial antes do loop ───────────────────
    if output_csv:
        csv_name = output_csv if output_csv.endswith(".csv") else f"{output_csv}.csv"
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
            train_ds = ParasiteDataset(
                set_name="train", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            train_loader = DataLoader(
                _OneHotDataset(train_ds, num_classes),
                batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            clf = _train_svm_proj(student, proj_kd, train_loader)

            # Eval
            test_ds = ParasiteDataset(
                set_name="test", split=split, percentage=pct,
                transform=transform, loader="ift_lab", path_dataset=parasite_name,
            )
            test_loader = DataLoader(
                test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
            )
            feats, y_true = extract_proj_features(student, proj_kd, test_loader)
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

            if wandb_update:
                try:
                    import wandb as _wandb  # noqa: PLC0415
                    wr = _wandb.init(
                        project=wandb_project, entity=wandb_entity,
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
    import sys

    main(**cli_kwargs(sys.argv[1:]))
