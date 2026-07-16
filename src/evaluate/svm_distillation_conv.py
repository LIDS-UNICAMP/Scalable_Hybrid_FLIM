"""svm_distillation_conv.py — SVM evaluation para distillation next_layers_direct.

Reutiliza integralmente svm_distillation.py — só troca:
  * DistillationModule  →  DistillationConvModule
  * filtro de runs      →  apenas "next_layers_direct"
  * CSV de saída        →  results/svm_distillation_conv_results.csv

Embedding usado: student.encode() → [B, 48]   (FLIM encoder puro, SEM proj head)

Usage::

    python -m src.evaluate.svm_distillation_conv
    python -m src.evaluate.svm_distillation_conv --wandb-update
    python -m src.evaluate.svm_distillation_conv --run eggs_split1_pct100
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Reutiliza tudo do script de distillation original ─────────────────────────
from src.evaluate.svm_distillation import (
    _OneHotDataset,
    extract_features_distillation,
    find_distillation_runs,
    train_svm_distillation,
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

# ── Carrega só o student do checkpoint, sem instanciar o teacher (I-JEPA) ─────
from src.models.lejepa_flim import LeJEPAFLIMModel
from src.models.models import (
    parse_architecture,
    get_channels_from_arch,
    get_actual_channels_from_weights,
    override_arch_channels,
)
from src.models.distillation import (
    TEACHER_DIM,
    OneLayerConvDistillationProjectionHead,
    OneLayer1x1ConvDistillationProjectionHead,
)


def _load_student_from_ckpt(ckpt_path: str, device: torch.device) -> LeJEPAFLIMModel:
    """Extrai só o student (FLIM encoder) do checkpoint, sem carregar o teacher.

    O DistillationConvModule salva os pesos do student sob a chave
    'student.*' no state_dict do Lightning. Carregamos apenas esses pesos,
    evitando instanciar o FrozenTeacher (I-JEPA ViT-H/14).

    Quando encoder_init='flim', os canais reais são detectados via bias files
    (ex: protozoan usa 30 canais na conv2, não 32 como no architecture.json).
    """
    # Carrega TUDO na CPU (o ckpt inclui o teacher I-JEPA ~2.5GB). Só o student
    # vai para a GPU depois via .to(device) — o teacher nunca toca a GPU.
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Recupera hparams para reconstruir a arquitetura
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
        arch=arch,
        in_channels=in_channels,
        proj_dim=proj_dim,
        proj_hidden=proj_hidden,
    )

    # Filtra só os pesos 'student.*' do state_dict completo
    full_sd = ckpt["state_dict"]
    student_sd = {
        k[len("student."):]: v
        for k, v in full_sd.items()
        if k.startswith("student.")
    }
    student.load_state_dict(student_sd, strict=True)
    return student.to(device)


def _load_student_and_proj_from_ckpt(ckpt_path: str, device: torch.device):
    """Carrega student (FLIM encoder) E a projection head (proj_kd) do checkpoint.

    Modelado em ``_load_student_from_ckpt``, mas adicionalmente reconstrói a
    cabeça de projeção a partir dos hparams e carrega os pesos ``proj_kd.*``.
    Usado para runs FROZEN, cuja avaliação roda no embedding 1280d
    ``proj_kd(student.encoder(x))`` (o encoder 48d é constante entre épocas
    quando congelado, então o sinal útil está na projeção treinável).

    O kernel da proj head vem do hparam ``proj_kernel`` (default 3):
        kernel == 1 → OneLayer1x1ConvDistillationProjectionHead
        senão       → OneLayerConvDistillationProjectionHead

    Returns:
        (student, proj_kd) — ambos em ``device``.
    """
    student = _load_student_from_ckpt(ckpt_path, device)

    # Reabre o ckpt na CPU só para hparams + pesos proj_kd.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    proj_kernel = hparams.get("proj_kernel", 3)

    if proj_kernel == 1:
        proj_kd = OneLayer1x1ConvDistillationProjectionHead(
            student_channels=student.embed_dim, teacher_dim=TEACHER_DIM,
        )
    else:
        proj_kd = OneLayerConvDistillationProjectionHead(
            student_channels=student.embed_dim, teacher_dim=TEACHER_DIM,
        )

    full_sd = ckpt["state_dict"]
    proj_sd = {
        k[len("proj_kd."):]: v
        for k, v in full_sd.items()
        if k.startswith("proj_kd.")
    }
    proj_kd.load_state_dict(proj_sd, strict=True)
    return student, proj_kd.to(device)


@torch.no_grad()
def _extract_proj_features(student, proj_kd, dataloader: DataLoader):
    """Extrai o embedding 1280d ``proj_kd(student.encoder(x))`` sobre um loader.

    Espelha ``extract_features_distillation`` (que usa ``student.encode`` → 48d),
    mas usa a projeção treinável. Loader yields ``(inputs, labels)``.

    Returns:
        (features [N, 1280], y_true [N]) — labels 0-indexed.
    """
    import numpy as np  # noqa: PLC0415

    student.eval()
    proj_kd.eval()
    student.to(DEVICE)
    proj_kd.to(DEVICE)

    all_feats: list = []
    all_labels: list = []
    from tqdm import tqdm  # noqa: PLC0415
    for inputs, labels in tqdm(dataloader, desc="  Extracting proj features"):
        emb = proj_kd(student.encoder(inputs.to(DEVICE))).detach().cpu().numpy()
        all_feats.append(emb)
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.tolist())
        else:
            all_labels.extend(labels)
    return np.concatenate(all_feats, axis=0), np.array(all_labels, dtype=np.int64)


def _train_svm_proj(student, proj_kd, dataloader: DataLoader,
                    max_iter: int = 10_000, C: float = 1e2):
    """Treina um SVM linear sobre o embedding 1280d (labels 1-indexed).

    Espelha ``train_svm_distillation`` mas com a feature de projeção.
    Loader yields ``(inputs, one_hot_labels)``.
    """
    import threading  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from sklearn import svm as sk_svm  # noqa: PLC0415
    from tqdm import tqdm  # noqa: PLC0415

    clf = sk_svm.SVC(
        max_iter=max_iter, C=C, gamma="auto",
        decision_function_shape="ovo", kernel="linear",
    )
    student.eval()
    proj_kd.eval()
    student.to(DEVICE)
    proj_kd.to(DEVICE)

    all_feats: list = []
    all_y: list = []
    print("[INFO] Extracting train proj features for SVM...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Train proj features"):
            emb = proj_kd(student.encoder(inputs.to(DEVICE))).detach().cpu().numpy()
            all_feats.append(emb)
            y_np = np.argmax(labels.cpu().numpy(), axis=1) + 1  # 1-indexed
            all_y.extend(y_np.tolist())

    X = np.concatenate(all_feats, axis=0)
    y = np.array(all_y, dtype=np.int64)

    _stop = threading.Event()

    def _progress():
        with tqdm(desc="SVM fit", unit="s",
                  bar_format="{desc}: {elapsed} [{postfix}]") as pbar:
            while not _stop.wait(1.0):
                pbar.update(1)
            pbar.set_postfix_str("done")

    _thr = threading.Thread(target=_progress, daemon=True)
    _thr.start()
    clf.fit(X, y)
    _stop.set()
    _thr.join()
    return clf


# ── Descoberta de checkpoints: best.ckpt (knn) + best_loss.ckpt (loss) ─────────


def _is_new_style_run(meta: dict) -> bool:
    """True se o run usa as convenções NOVAS (dois checkpoints + monitor knn).

    Detecta os campos introduzidos pela nova pipeline de treino:
    ``best_loss_checkpoint``, ``freeze_encoder``, ``knn_probe`` ou um
    ``monitor`` == "val/knn_kappa"; ou ainda a presença física de
    ``best_knn_kappa.ckpt``/``best_loss.ckpt`` em checkpoints/. Runs antigos não
    têm nenhum desses sinais e caem no caminho legado (uma linha).
    """
    if any(k in meta for k in ("best_loss_checkpoint", "freeze_encoder", "knn_probe")):
        return True
    if meta.get("monitor") == "val/knn_kappa":
        return True
    ckpt_dir = os.path.join(meta["_run_dir"], "checkpoints")
    return (os.path.isfile(os.path.join(ckpt_dir, "best_knn_kappa.ckpt"))
            or os.path.isfile(os.path.join(ckpt_dir, "best_loss.ckpt")))


def _discover_ckpts(meta: dict) -> list[dict]:
    """Lista os checkpoints a avaliar de um run, um dict por checkpoint.

    Para runs NOVOS (com best_checkpoint/best_loss_checkpoint na metadata, ou
    com best_knn_kappa.ckpt/best_loss.ckpt em checkpoints/) avalia AMBOS quando
    existirem: best_knn_kappa.ckpt (monitor "val/knn_kappa") e best_loss.ckpt
    (monitor "val/loss").

    Para runs ANTIGOS comporta-se exatamente como antes: uma única linha, com o
    checkpoint já resolvido por find_distillation_runs (``_ckpt_path``, que pega
    o mais treinado e evita o best.ckpt stale de epoch 0) e monitor legado.

    Cada dict: {"ckpt_path", "ckpt_monitor"}.
    """
    # ── Run antigo: preserva o comportamento legado (uma linha) ────────────
    if not _is_new_style_run(meta):
        monitor = meta.get("monitor", "val/loss")
        return [{"ckpt_path": meta["_ckpt_path"], "ckpt_monitor": monitor}]

    ckpt_dir = os.path.join(meta["_run_dir"], "checkpoints")
    best_meta = meta.get("best_checkpoint")
    best_loss_meta = meta.get("best_loss_checkpoint")

    found: list[dict] = []

    # best_knn_kappa.ckpt → monitor val/knn_kappa
    best_path = None
    if best_meta and os.path.isfile(best_meta):
        best_path = best_meta
    else:
        cand = os.path.join(ckpt_dir, "best_knn_kappa.ckpt")
        if os.path.isfile(cand):
            best_path = cand
    if best_path:
        found.append({"ckpt_path": best_path, "ckpt_monitor": "val/knn_kappa"})

    # best_loss.ckpt → monitor val/loss
    best_loss_path = None
    if best_loss_meta and os.path.isfile(best_loss_meta):
        best_loss_path = best_loss_meta
    else:
        cand = os.path.join(ckpt_dir, "best_loss.ckpt")
        if os.path.isfile(cand):
            best_loss_path = cand
    if best_loss_path and best_loss_path != best_path:
        found.append({"ckpt_path": best_loss_path, "ckpt_monitor": "val/loss"})

    # Segurança: se nada resolvido (ex: só last.ckpt), usa o _ckpt_path legado.
    if not found:
        monitor = meta.get("monitor", "val/loss")
        found.append({"ckpt_path": meta["_ckpt_path"], "ckpt_monitor": monitor})

    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVM no FLIM encoder [B,48] dos checkpoints next_layers_direct.",
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
    # transform base (encoder 48d, runs não-frozen). Para runs frozen derivamos a
    # norma da metadata por run (no_imagenet_norm) — ver loop abaixo.
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

    print(f"\n{'='*70}")
    print(f"SVM — distillation_conv  |  embedding: FLIM encoder [B,48]")
    print(f"Runs: {len(runs)}")
    print(f"{'='*70}")

    rows: list[dict] = []

    for meta in runs:
        run_name  = meta.get("run_name", os.path.basename(meta["_run_dir"]))
        dataset   = meta.get("dataset", "")
        split     = int(meta.get("split", 1))
        pct       = int(meta.get("percentage", 100))

        # ── Modo frozen: avalia a projeção 1280d; senão o encoder 48d ───────
        freeze_encoder = bool(meta.get("freeze_encoder", False))

        # Transform por run: frozen treina com student LAB[0,1] (sem ImageNet
        # norm), então sua avaliação também deve ser LAB[0,1]. Deriva de
        # metadata no_imagenet_norm se presente, senão usa o flag CLI.
        if "no_imagenet_norm" in meta:
            run_no_norm = bool(meta["no_imagenet_norm"])
        else:
            run_no_norm = args.no_imagenet_norm
        run_transform = _build_test(IMAGE_SIZE, imagenet_norm=not run_no_norm)

        # best_knn_kappa do run (val/knn_kappa do best.ckpt); usado nas duas linhas.
        run_best_knn = meta.get("best_knn_kappa", float("nan"))
        try:
            run_best_knn = float(run_best_knn)
        except (TypeError, ValueError):
            run_best_knn = float("nan")

        num_classes   = _DATASET_NUM_CLASSES.get(dataset, 9)
        parasite_name = _DATASET_PARASITE_NAME.get(dataset, dataset)

        ckpt_entries = _discover_ckpts(meta)

        print(f"\n── {run_name}  (frozen={freeze_encoder}, ckpts={len(ckpt_entries)})")

        for entry in ckpt_entries:
            ckpt_path    = entry["ckpt_path"]
            ckpt_monitor = entry["ckpt_monitor"]

            print(f"   ckpt: {os.path.relpath(ckpt_path, _ROOT)}  [{ckpt_monitor}]")

            # method: runs não-frozen mantêm o valor existente; frozen ganha o
            # nome específico por checkpoint (knn vs best-loss).
            if freeze_encoder:
                method = ("SVM_Distill_1x1BN_flim_frozen_eval_knn"
                          if ckpt_monitor == "val/knn_kappa"
                          else "SVM_Distill_1x1BN_flim_frozen_eval_loss")
                proj_head_label = "projection_1280d"
                embed_dim = TEACHER_DIM
            else:
                method = "SVM_Distillation_Conv"
                proj_head_label = "removed_for_eval"
                embed_dim = 48

            base = {
                "run_name":          run_name,
                "method":            method,
                "dataset":           dataset,
                "split":             split,
                "percentage":        pct,
                "distillation_type": meta.get("distillation_type", "direct"),
                "encoder_init":      meta.get("encoder_init", "trunc_normal"),
                "student_embed_dim": embed_dim,
                "proj_head":         proj_head_label,
                "ckpt_path":         ckpt_path,
                "ckpt_monitor":      ckpt_monitor,
                "val_knn_kappa":     run_best_knn,
            }

            try:
                train_ds = DatasetParasite(
                    set_name="train", split=split, percentage=pct,
                    transform=run_transform, loader="ift_lab", path_dataset=parasite_name,
                )
                test_ds = DatasetParasite(
                    set_name="test", split=split, percentage=pct,
                    transform=run_transform, loader="ift_lab", path_dataset=parasite_name,
                )

                if freeze_encoder:
                    # ── Avaliação na projeção 1280d ─────────────────────────
                    student, proj_kd = _load_student_and_proj_from_ckpt(ckpt_path, DEVICE)
                    student.eval(); proj_kd.eval()
                    for p in student.parameters():
                        p.requires_grad_(False)
                    for p in proj_kd.parameters():
                        p.requires_grad_(False)
                    print(f"   proj embedding dim = {TEACHER_DIM}  (proj_kd ∘ encoder)")

                    train_loader = DataLoader(
                        _OneHotDataset(train_ds, num_classes),
                        batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                    )
                    clf = _train_svm_proj(student, proj_kd, train_loader)

                    test_loader = DataLoader(
                        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                    )
                    feats, y_true = _extract_proj_features(student, proj_kd, test_loader)
                else:
                    # ── Avaliação no encoder 48d (comportamento existente) ──
                    student = _load_student_from_ckpt(ckpt_path, DEVICE)
                    student.eval()
                    for p in student.parameters():
                        p.requires_grad_(False)
                    print(f"   student.embed_dim = {student.embed_dim}  (encoder only)")

                    train_loader = DataLoader(
                        _OneHotDataset(train_ds, num_classes),
                        batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                    )
                    clf = train_svm_distillation(student, train_loader)

                    test_loader = DataLoader(
                        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                    )
                    feats, y_true = extract_features_distillation(student, test_loader)

                y_pred = clf.predict(feats) - 1  # volta para 0-indexed

                metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
                rows.append({**base, **metrics, "status": "ok", "error": ""})
                print(f"   kappa={metrics['kappa']:.4f}  acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}")

                if args.wandb_update:
                    try:
                        import wandb as _wandb  # noqa: PLC0415
                        wr = _wandb.init(
                            project=args.wandb_project, entity=args.wandb_entity,
                            name=f"svm_distil_conv_{run_name}",
                            config={**base, "num_classes": num_classes}, reinit=True,
                        )
                        _wandb.log({f"svm/{k}": v for k, v in metrics.items()})
                        wr.finish()
                    except Exception as we:
                        print(f"   [WARN] W&B: {we}")

            except Exception as exc:
                print(f"   [ERROR] {exc}")
                rows.append({**base, "kappa": float("nan"), "acc": float("nan"),
                             "f1": float("nan"), "status": "error", "error": str(exc)})

    # ── CSV ────────────────────────────────────────────────────────────────
    if args.output_csv:
        csv_name = args.output_csv if args.output_csv.endswith(".csv") else f"{args.output_csv}.csv"
    else:
        safe = base_filter.replace("/", "_").replace(" ", "_")
        csv_name = f"svm_{safe}_results.csv"
    csv_path = os.path.join(_RESULTS_DIR, csv_name)
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
