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

"""run_metadata_callback.py — o ``run_metadata.json`` de um estagio, escrito por Callback.

Era ``_save_metadata`` (src/modules/autoencoder_flim_module.py:1226-1311), chamada a mao
pelo ``main()``/argparse que o refactor apaga. Sem escritor o arquivo some, e o laco de
crescimento morre no PRIMEIRO estagio: ``_kappa``
(experiments/ray/runners/growth.py:200-207) abre este JSON, le ``best_val_svm_kappa`` e
mata o braco quando o arquivo nao existe.

O JSON e CONTRATO, nao formato interno. Sao 504 arquivos ja gravados em ``artifacts/``
que os leitores abrem no mesmo passe que os novos, entao as chaves, os tipos e os valores
sao os de hoje:

* experiments/ray/runners/growth.py:206 — ``best_val_svm_kappa`` (float; None mata o braco)
* scripts/spifil_growth_loop.py:272 — a mesma leitura, no lancador antigo
* experiments/ray/skip.py:114 — ``status``, e so ``"ok"`` conta como feito
* src/evaluate/eval_growth_stages.py:259 — ``run_name`` e ``percentage``
* tools/plot_continuity_spifil_hybrid.py:99 — ``flim_ref_svm_*``, ``dataset``, ``percentage``
* tools/plot_partial_train_spifil_hybrid.py:237-246 — ``arch_json`` e ``channels``
* scripts/autoencoder_flim_ray.py:861-864 — ``best_val_svm_kappa`` e ``baseline_flim_svm``

Declarado no YAML, nunca por flag::

    - class_path: experiments.run_metadata_callback.RunMetadataCallback
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import lightning.pytorch as pl

from core.constants import (
    CHECKPOINTS_SUBDIR,
    FLIM_ARCH_BASE,
    NUM_CLASSES,
    PROJECT_ROOT,
    RUN_METADATA_FILENAME,
)

# {"eggs": "ch24_32_48", "larvae": "ch24_32_48", "protozoan": "ch24_30_48"} — a tag que a
# origem lia de `ARCH_TAG` (src/modules/autoencoder_flim_module.py:851). Esse dict nao
# atravessou para `core.constants`; o que atravessou foi `FLIM_ARCH_BASE`, cujo diretorio
# pai JA carrega a tag ("ch24_30_48_a0.5_f5/protozoan"). Derivar e portanto ler o valor
# unico em vez de digitar uma terceira copia dele.
ARCH_TAG: dict[str, str] = {
    dataset: os.path.basename(os.path.dirname(path)).split("_a")[0]
    for dataset, path in FLIM_ARCH_BASE.items()
}


def _git_sha() -> str:
    """SHA curto do HEAD, ou "" quando nao ha git. Origem: :1231-1236."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _parent_run_from_ckpt(init_ckpt: str) -> str:
    """``.../<run-name>/checkpoints/best_recon.ckpt`` -> ``<run-name>``. Origem: :1215-1223."""
    if not init_ckpt:
        return ""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(init_ckpt))))


def _score(checkpoint) -> float:
    """`best_model_score` como float; NaN quando nao ha. Origem: :1207-1212."""
    value = getattr(checkpoint, "best_model_score", None)
    try:
        return float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError):
        return float("nan")


class RunMetadataCallback(pl.Callback):
    """Grava ``run_metadata.json`` ao fim do fit — em `ok` e em `error`.

    Args:
        recon_loss: nome da perda de reconstrucao gravado no metadado. A flag da origem
            (:876) tinha ``choices=["bce_logits"]``, um valor so, e os 504 arquivos em disco
            confirmam: nenhum tem outro. Continua parametro, e nao literal, porque quem le o
            campo le para saber QUAL perda produziu o numero — no dia em que houver a
            segunda, o metadado tem que poder dizer.
    """

    def __init__(self, recon_loss: str = "bce_logits") -> None:
        super().__init__()
        self.recon_loss = recon_loss
        self._t0 = 0.0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # `elapsed_s` da origem era medido em volta de `trainer.fit` (:1183). Aqui o relogio
        # comeca DENTRO do fit, logo alguns segundos de setup ficam de fora.
        self._t0 = time.time()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        checkpoint = self._checkpoint(trainer, pl_module)
        self._write(
            trainer, pl_module, status="ok",
            best_ckpt=getattr(checkpoint, "best_model_path", "") or "",
            best_score=_score(checkpoint),
            elapsed=time.time() - self._t0,
        )

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                     exception: BaseException) -> None:
        # `except Exception` da origem (:1186) NAO pegava KeyboardInterrupt, e este hook pega.
        # Sem o filtro um Ctrl+C passaria a deixar `status: error` em disco onde hoje nao
        # deixa arquivo nenhum, e `skip.py:_local` mudaria de "no_metadata" para "local:error".
        if not isinstance(exception, Exception):
            return
        self._write(trainer, pl_module, status="error", error=str(exception))

    @staticmethod
    def _checkpoint(trainer: pl.Trainer, pl_module: pl.LightningModule):
        """O ModelCheckpoint que observa a metrica DESTE estagio.

        Nao e `trainer.checkpoint_callback`: esse devolve o primeiro da lista, e
        configs/default.yaml:20-46 declara tres. Quem manda e `monitor_spec`, a mesma
        propriedade de que os callbacks foram montados.
        """
        monitor = pl_module.monitor_spec[0]
        return next(
            (cb for cb in trainer.checkpoint_callbacks if cb.monitor == monitor),
            trainer.checkpoint_callback,
        )

    def _write(self, trainer: pl.Trainer, pl_module: pl.LightningModule, status: str,
               error: str = "", best_ckpt: str = "", best_score: float = float("nan"),
               elapsed: float = 0.0) -> None:
        hparams = pl_module.hparams
        dataset = hparams.dataset
        ckpt_dir = (getattr(self._checkpoint(trainer, pl_module), "dirpath", None)
                    or os.path.join(trainer.default_root_dir, CHECKPOINTS_SUBDIR))
        # `ckpt_dir = os.path.join(output_dir, "checkpoints")` na origem (:987), entao o pai
        # do dirpath E o output_dir — inclusive quando o `--trainer.default_root_dir` do
        # estagio (growth.py:230) nao foi passado.
        output_dir = os.path.dirname(ckpt_dir)

        meta = {
            "run_name": _run_name(trainer, output_dir),
            "method": "autoencoder_resnet_init_flim",
            "dataset": dataset,
            "split": int(trainer.datamodule.split),
            "percentage": int(trainer.datamodule.percentage),
            "encoder_init": "flim",
            "decoder": "resnet",
            "recon_loss": self.recon_loss,
            "architecture": ARCH_TAG[dataset],
            "channels": pl_module.channels,
            "embed_dim": pl_module.encoder_out_channels,
            "num_classes": NUM_CLASSES[dataset],
            "color_space": "lab",
            "lab_scale": "labnorm2_native_01",
            "imagenet_norm": hparams.imagenet_norm,
            "svm_multiclass": "ovo",
            "svm_max_iter": -1,
            "train_augmentation": "none",
            "arch_json": hparams.arch_json,
            "flim_weights_path": hparams.flim_weights_path,
            "ckpt_dir": ckpt_dir,
            "best_ckpt": best_ckpt,
            # Numa rodada de RECONSTRUCAO congelada o maximo monitorado e ruido de fit do SVM
            # (ver monitor_spec), entao o que vai gravado ali e a baseline de on_fit_start.
            # Com head_finetune a troca NAO vale: mesmo com o encoder congelado a Head treina
            # de verdade, o pico e sinal, e o melhor score vai para o disco como ele e.
            # O nome da chave e mantido por compatibilidade — growth.py:206 le
            # `best_val_svm_kappa` e mata o braco se faltar. Qual metrica foi de fato
            # monitorada esta em `ckpt_monitor`, logo abaixo.
            "best_val_svm_kappa": (
                best_score if (hparams.head_finetune or not hparams.freeze_encoder_flag)
                else (pl_module.baseline_metrics or {}).get("kappa", float("nan"))
            ),
            "best_monitor_score": best_score,
            # A mesma propriedade de que os callbacks foram montados — nunca redigitada aqui,
            # ou o metadado documentaria um monitor que a run nao usou.
            "ckpt_monitor": pl_module.monitor_spec[0],
            "stage_metric_prefix": pl_module.stage_prefix,
            "stage": pl_module.stage,
            # `stage` sozinho NAO identifica este braco: com head_finetune o encoder roda
            # destravado, entao a property devolve 2, igual ao estagio 2 de reconstrucao.
            # Quem separa o braco e a flag crua abaixo.
            "head_finetune": bool(hparams.head_finetune),
            # Lidos da Head CONSTRUIDA, nao re-derivados: se um dia divergirem do
            # `embed_dim`/`num_classes` acima, o metadado mostra a divergencia.
            "head_in_channels": (pl_module.head.fc.in_features
                                 if pl_module.head is not None else None),
            "head_num_classes": (pl_module.head.fc.out_features
                                 if pl_module.head is not None else None),
            # Default "avgpool2d" e o mesmo da flag que morreu (:904); growth.py:228 sempre
            # manda o valor explicito, entao na grade de crescimento o default nunca vale.
            "embed_mode": hparams.get("embed_mode", "avgpool2d"),
            "encoder_frozen": bool(hparams.freeze_encoder_flag),
            "init_ckpt": hparams.init_ckpt,
            "parent_run": _parent_run_from_ckpt(hparams.init_ckpt),
            "baseline_flim_svm": pl_module.baseline_metrics,
            # Estado do solver da ultima sonda. Sem estes tres, uma repeticao do truncamento
            # por max_iter voltaria a nao deixar rastro em disco.
            "svm_fit_status": (pl_module.last_probe_metrics or {}).get("fit_status"),
            "svm_n_iter_max": (pl_module.last_probe_metrics or {}).get("n_iter_max"),
            "svm_n_iter_sum": (pl_module.last_probe_metrics or {}).get("n_iter_sum"),
            "elapsed_s": elapsed,
            "status": status,
            "error": error,
            "git_sha": _git_sha(),
        }
        # A leitura da run — kappa de referencia, melhor kappa, delta e o veredito
        # improved/tied/degraded — em disco ao lado do checkpoint que a produziu. Vazio
        # (chaves ausentes) quando nao houve com o que comparar.
        meta.update(pl_module.verdict_summary())

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, RUN_METADATA_FILENAME), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)


def _run_name(trainer: pl.Trainer, output_dir: str) -> str:
    """O nome da run, o mesmo que ``--trainer.logger.init_args.name`` pos no W&B.

    NAO e `trainer.logger.name`: no WandbLogger essa property devolve o PROJETO
    (lightning/pytorch/loggers/wandb.py:569-577), nao o nome da run. O display name vive em
    `logger.experiment.name`. Sem logger sobra o nome da pasta, que e o que
    `run_name` vale no lancador de grade — mas nao nos estagios de crescimento, onde a
    pasta e `stage1` e a run e `<exp>_<braco>_stage1` (growth.py:229).
    """
    experiment = getattr(trainer.logger, "experiment", None)
    return getattr(experiment, "name", None) or os.path.basename(output_dir)
