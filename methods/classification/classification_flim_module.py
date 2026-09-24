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

"""classification_flim_module.py — Direct supervised classification: FLIM-init encoder +
2-layer Sigmoid head.

    encoder(x) -> [B, 48, H, W]
        -> AdaptiveAvgPool2d(1) -> flatten -> [B, 48]
        -> Linear(48, 24) -> Sigmoid
        -> Linear(24, n_classes) -> Softmax

No SSL objective, no teacher, no distillation — pure supervised training against
ground-truth labels. The encoder is always initialised with FLIM-estimated
kernels/bias before training (Experiment 3: "FLIM init + perceptron de
classificacao"). Because the head's forward() returns post-Softmax class
probabilities rather than logits, training minimises NLLLoss on log(probs)
instead of CrossEntropyLoss (which expects raw logits and applies its own
softmax internally).

Run name convention: classhead_<dataset>_split<N>_pct<P>_sigmoid2l

Movido de src/modules/classification_flim_module.py:90 (ClassificationFlimModule).
O ``if __name__ == "__main__"`` e todo o argparse ficaram na origem: aqui so vive a
LightningModule, instanciada por YAML. O encoder vem da porta unica do pacote
(``flim.build``) e a cabeca de ``core.blocks``.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from core.blocks import SigmoidClassificationModel
# core/metrics.py:46 — mesma `compute_metrics` de src/metrics/classification.py:31. Os
# dois ASTs so divergem em um ponto: `average` virou o literal "macro" em vez de um
# argumento com esse mesmo default. O unico call site deste modulo
# (on_validation_epoch_end) nunca passa `average`, entao kappa, acc e f1 saem bit-a-bit
# iguais aos do W&B de hoje. `acc` continua sendo a acuracia MACRO (balanceada).
from core.metrics import compute_metrics
from flim import build

_log = logging.getLogger(__name__)

ENCODER_INITS = ("flim",)  # Exp.3 is FLIM-init only by design; see distillation_*_module.py for random/he/xavier


class ClassificationFlimModule(pl.LightningModule):
    """FLIM-init encoder + TwoLayerSigmoidHead, trained end-to-end with NLLLoss.

    Args:
        arch_json:          Path to the FLIM architecture JSON file.
        dataset:             ``"eggs"`` | ``"larvae"`` | ``"protozoan"`` — mantido so
                              como hiperparametro registrado (nome do run, W&B, hparams
                              do checkpoint); nao seleciona mais arquitetura. Ver a nota
                              "protozoan" abaixo.
        flim_weights_path:   Directory with FLIM weight files (conv{n}-kernels.npy / conv{n}-bias.txt).
                              Always required — this module has no non-FLIM init path.
        num_classes:         9 (eggs) / 2 (larvae) / 7 (protozoan).
        hidden_dim:          Head hidden size. Defaults to encoder_out_channels // 2 (48 -> 24).
        freeze_encoder:      If True, the FLIM encoder is frozen after init (requires_grad=False)
                              and only the TwoLayerSigmoidHead is trained. Mirrors the freeze/unfreeze
                              distinction already used by the MLP probe configs (configs/evaluate/mlp/).
        output_relu:         If True, inserts a ReLU between ``Linear(hidden, C)`` and the Softmax.
        output_softplus:     If True, inserts a Softplus in the same position — the smooth
                              counterpart of ``output_relu`` (non-zero gradient everywhere).
                              Mutually exclusive with ``output_relu``.

    Nota "protozoan" (era src/modules/classification_flim_module.py:137-153): a origem
    trocava ``parse_architecture(arch_json)`` pelo dict em codigo ``PROTOZOAN_FLIM_ARCH``
    quando ``dataset == "protozoan"``. Esse ramo nao sobrevive a fronteira do pacote
    ``flim`` — ``build`` recebe caminho de json, nao dict — e tambem nao muda nada:
    ``get_actual_channels_from_weights`` le os canais dos proprios arquivos de bias e
    ``override_arch_channels`` sobrescreve ``noutput_channels``, de modo que os unicos
    campos em que os dois candidatos de json divergem (layer2 nkernels/noutput_channels,
    30 vs 32) nunca chegam ao encoder. Alem disso o caller de producao
    (scripts/classification_flim_ray.py:205, base FLIM_WEIGHTS_BASE) ja passa
    ch24_32_48_a0.5_f5/protozoan/train{N}/architecture.json, cujo conteudo e identico ao
    dict PROTOZOAN_FLIM_ARCH. Encoder resultante bit-a-bit igual nos dois caminhos.
    """

    def __init__(
        self,
        arch_json: str,
        dataset: str = "",
        flim_weights_path: Optional[str] = None,
        num_classes: int = 9,
        hidden_dim: Optional[int] = None,
        in_channels: int = 3,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        seed: int = 42,
        freeze_encoder: bool = False,
        output_relu: bool = False,
        output_softplus: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if flim_weights_path is None:
            raise ValueError(
                "flim_weights_path is required — ClassificationFlimModule always "
                "initialises the encoder with FLIM weights (Experiment 3)."
            )

        encoder = build(arch_json, init="flim", weights_path=flim_weights_path,
                        in_channels=in_channels)
        # num_features e o canal de saida real (lido dos kernels FLIM), o mesmo numero
        # que a origem tirava de channels[-1]. Ler o arch aqui violaria a fronteira flim.
        self.encoder_out_channels: int = encoder.num_features

        self.model = SigmoidClassificationModel(
            encoder=encoder, in_features=self.encoder_out_channels,
            num_classes=num_classes, hidden_dim=hidden_dim,
            output_relu=output_relu, output_softplus=output_softplus,
        )

        if freeze_encoder:
            for p in self.model.encoder.parameters():
                p.requires_grad = False

        _log.info(
            "[ClassificationFlimModule] FLIM weights loaded from %s | encoder_out=%d | "
            "hidden=%d | num_classes=%d | freeze_encoder=%s",
            flim_weights_path, self.encoder_out_channels,
            hidden_dim if hidden_dim is not None else self.encoder_out_channels // 2,
            num_classes, freeze_encoder,
        )

        self._val_preds: List[Tensor] = []
        self._val_labels: List[Tensor] = []

    # ── Helpers ────────────────────────────────────────────────────────────

    def _first_view(self, views: Union[List[Tensor], Tensor]) -> Tensor:
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0] if views.shape[1] < views.shape[0] else views[0]
        return views

    @staticmethod
    def _nll(probs: Tensor, labels: Tensor) -> Tensor:
        # forward() returns post-Softmax probabilities, not logits — NLLLoss on
        # log(probs) is the correct counterpart to CrossEntropyLoss here.
        return F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)

    # ── Forward / training ─────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        probs = self(x)
        loss = self._nll(probs, y)

        preds = probs.argmax(dim=1)
        acc = (preds == y).float().mean()

        opt = self.optimizers()
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_preds = []
        self._val_labels = []

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        probs = self(x)
        loss = self._nll(probs, y)

        self._val_preds.append(probs.argmax(dim=1).detach().cpu())
        self._val_labels.append(y.detach().cpu())
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        y_pred = torch.cat(self._val_preds)
        y_true = torch.cat(self._val_labels)
        # val/acc AQUI e acuracia MACRO (balanceada): multiclass_accuracy funcional com
        # average="macro". Convencao diferente da de ClassificationFinetuneModule, que
        # loga a MESMA chave val/acc em micro. Decisao do dono do repo: preservar as duas.
        metrics = compute_metrics(y_true, y_pred, num_classes=self.hparams.num_classes)
        self.log("val/kappa", metrics["kappa"], prog_bar=True, on_epoch=True)
        self.log("val/acc", metrics["acc"], prog_bar=False, on_epoch=True)
        self.log("val/f1", metrics["f1"], prog_bar=False, on_epoch=True)

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total = self.hparams.max_epochs or 100
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(trainable_params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler = SequentialLR(optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
