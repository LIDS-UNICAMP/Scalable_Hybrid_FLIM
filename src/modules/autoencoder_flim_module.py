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
"""
Unsupervised AutoEncoder pretraining of the FLIM encoder (Experiment Day 5).

No labels enter the loss. The encoder is trained purely by reconstructing its own
input, and the decoder is thrown away afterwards. The question the run answers is
whether the encoder comes out *better* than it went in — measured by a one-vs-one
SVM probe on the pooled embedding, fitted on the labelled train split and scored
**on validation only**. Test is never touched.

Reading the result (see the experiment brief):
  improved  — reconstruction is useful self-supervision for this tiny encoder
  tied      — the FLIM embedding is already saturated
  degraded  — reconstruction pulls the embedding toward low-level information
              (texture, a/b chromaticity, background). The known risk.

Loss: ``BCEWithLogitsLoss`` over LAB in [0, 1] (option 1 of the brief). Cross-entropy
needs a target in [0, 1], so the target is the LAB image **before** ``imagenet_norm``,
recovered by undoing the normalisation. ``ift_lab`` already returns LABNorm2 in [0, 1],
so ``lab_scale`` is the identity — no extra rescaling is invented here.

``imagenet_norm`` defaults to **False** here. The FLIM kernels were derived from markers on
LAB in [0, 1] and already carry the marker normalisation inside their kernel/bias pair, so an
ImageNet RGB ``Normalize`` on top of them moves the effective bias — ``W·(x−μ)/σ + b`` — and
every ReLU cut the markers calibrated stops falling where it should (measured: −0.33 kappa).
``src/data_modules/datasets/lejepa_dataset.py:42`` already says as much. This is a **local**
decision for the FLIM-init autoencoder, not a global inversion: the I-JEPA and distillation
arms still need ``imagenet_norm=True``, because their teacher was trained with it.

**Progressive-growth protocol.** A run is one stage of a chain, each stage a separate W&B run,
the encoder gaining one conv layer per round:

  stage 1  ``--freeze-encoder``      encoder = FLIM, frozen; decoder random.
           The decoder learns to invert a *fixed* feature space, so it stops
           injecting garbage gradient into the FLIM kernels on the epochs where
           the decoder is still noise. The probe here is the "pure frozen FLIM"
           baseline, measured in the same run, split, seed, dataloader and probe.
  stage 2  ``--init-ckpt <previous>``  encoder + decoder both trainable, decoder
           starting from the previous round rather than from noise.
  stage 3  ``--freeze-encoder --init-ckpt <previous>``  one extra conv layer grown on the
           encoder, trained frozen on top of the previous round's backbone. The decoder
           block indices shift by the layers gained; ``main()`` remaps them on load.

           Δκ = ``probe/svm_kappa`` (best) − the ``on_fit_start`` FLIM reference kappa,
           which for an unfrozen round *is* the frozen baseline: a frozen round never
           moves the encoder.

Read Δκ only after checking ``probe/flim_drift``: Δκ > 0 with the drift exploding is
not "FLIM adapted", it is "FLIM was erased and a random CNN trained in its place".

Best checkpoint is selected by ``probe/svm_kappa`` in **every** stage — the best *encoder* is
the deliverable, and one selection metric is what makes the rounds comparable. In a frozen
round that curve is only the SVM's own fit noise around a fixed embedding, so its argmax is
close to arbitrary: read it as a measurement of the round's starting point, not as training.

**Metric keys.** Every metric is prefixed ``probe/``, stage-independently, all of them derived
from the single ``self._stage_prefix`` — including the checkpoint monitor, so the monitor
string cannot drift away from the key the module logs (``EarlyStopping`` runs with
``strict=False`` and would silently train to ``max_epochs`` on a monitor that never appears).
The stage itself lives in the run name and the W&B config, not in the key, so every round of
every stage plots on ONE curve. ``stage_epoch`` is the 0-based x axis of each stage, so the
curves start at the same origin instead of a later stage resuming the earlier one's axis.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.svm import SVC
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics.classification import compute_metrics
# The probe embeds with the official evaluator's own function, not with a look-alike:
# `AutoEncoderFLIM.embed` was mathematically identical to it (conv1→conv2→conv3, global
# average pool, flatten), and two identical implementations are one drift away from
# disagreeing. Importing it also means `evaluate.EMBED_MODE` governs the probe for free.
from src.utils.evaluate import _encode_pooled
from src.models.autoencoder_resnet import AutoEncoderFLIM
from src.models.models import (
    freeze_encoder,
    get_actual_channels_from_weights,
    load_FLIM_encoder,
    override_arch_channels,
    parse_architecture,
)

_log = logging.getLogger(__name__)

# Applied by src/data_modules/datasets/lejepa_dataset.py:44 when imagenet_norm=True.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# LABNorm2 (pyift) -> CIE L*a*b*, for W&B visualisation only. Round-trip against the
# source PNG measures MAE ~= 0.011, dominated by the uint8 quantisation the transform
# pipeline already applies (lejepa_dataset.py:33) — not by this mapping.
LAB_L_SCALE = 100.0
LAB_AB_SCALE = 255.0
LAB_AB_OFFSET = 128.0

# Kappa band inside which improved/degraded is not claimed. Small on purpose: with the
# converged solver (max_iter=-1, see _svm_probe) the probe is a deterministic function of the
# feature matrix and invariant to the row order of the train loader, so the fit noise that
# made the truncated solver swing by ~0.07 kappa across permutations of the *same* features is
# gone. What remains is the sampling noise of one validation split (N~1200), well under 0.01.
KAPPA_TOLERANCE = 0.01


def _decoder_block_delta(state: dict, n_layers: int) -> int:
    """How many decoder blocks are NEW relative to ``state`` — i.e. how many layers this
    model grew past the checkpoint it resumes from.

    ``ResNetDecoder`` builds its blocks deepest-first (autoencoder_resnet.py:110-117), so
    the new blocks are indices ``0 .. delta-1`` and every old index is shifted by ``+delta``.
    ONE formula, two callers: the state_dict remap in ``main`` and the stage-3 freeze policy.
    """
    pre = "model.decoder.blocks."
    return n_layers - len({k[len(pre):].split(".")[0] for k in state if k.startswith(pre)})


def _freeze_for_growth(model: nn.Module, delta: int) -> None:
    """Stage 3 freeze policy: only the grown encoder layer and the ``delta`` new decoder
    blocks keep learning; every older encoder block, older decoder block and ``to_image``
    is frozen. ``to_image`` is always old — its width is channels[1], which growth never
    touches.
    """
    freeze_encoder(model, except_last=True)
    for param in model.decoder.blocks[delta:].parameters():
        param.requires_grad = False
    for param in model.decoder.to_image.parameters():
        param.requires_grad = False


class Head(nn.Module):
    """Cabeca de classificacao sobre o ultimo mapa do encoder.

    ``AdaptiveAvgPool2d(1)`` torna a cabeca independente da resolucao: a MESMA Head serve
    para 45x11x11 (saida de conv4) e 45x5x5 (saida de conv5). ``in_channels`` vem de
    ``channels[-1]`` do modelo ja construido e ``num_classes`` de ``NUM_CLASSES[dataset]`` —
    nenhum dos dois e chumbado aqui.
    """

    def __init__(self, in_channels: int, num_classes: int, p_drop: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.drop(self.flat(self.pool(x))))


class AutoEncoderFlimModule(pl.LightningModule):
    """FLIM-init AutoEncoder trained by reconstruction, probed by a one-vs-one SVM.

    Args:
        arch_json:          Path to the FLIM ``architecture.json``.
        dataset:            ``"eggs"`` | ``"larvae"`` | ``"protozoan"``. Only selects the
                            split folders; the architecture always comes from ``arch_json``.
                            protozoan's trained weights live in ``ch24_32_48_a0.5_f5`` even
                            though the real widths are 24/30/48, which
                            ``get_actual_channels_from_weights`` then corrects.
        flim_weights_path:  Directory with ``conv{n}-kernels.npy`` / ``conv{n}-bias.txt``.
        num_classes:        9 (eggs) / 2 (larvae) / 7 (protozoan). Used by the SVM probe only.
        imagenet_norm:      Whether the dataloader normalised the input. Drives target recovery.
                            Defaults to False — see the module docstring: the FLIM kernels
                            already embed the marker normalisation over LAB[0, 1].
        svm_probe_every:    Run the SVM probe every N validation epochs (1 = every epoch).
        freeze_encoder_flag: Freezes the FLIM encoder so only the decoder learns.
        freeze_spifil_layer: Ablation. Keeps the grafted SPiFiL filter bank (the LAST encoder
                            block) exactly as SPiFiL cut it, even in an unfrozen stage. Off by
                            default: once written into the layer those patches are ordinary
                            conv weights, like the FLIM kernels the unfrozen stages fine-tune.
        head_finetune:      Troca a reconstrucao por classificacao: uma ``Head`` sobre a
                            ultima camada do encoder, perda ``cross_entropy`` e monitor
                            ``probe/head_kappa``. O decoder e congelado (sai do otimizador).
                            Off por default — sem a flag o fluxo de reconstrucao e identico.
        init_ckpt:          Checkpoint this round resumed its backbone from, "" for round 1.
                            Recorded so ``stage`` can tell a first frozen round apart from a
                            grown-and-frozen one.
    """

    def __init__(
        self,
        arch_json: str,
        dataset: str = "",
        flim_weights_path: Optional[str] = None,
        num_classes: int = 9,
        in_channels: int = 3,
        image_size: int = 200,
        lr: float = 5e-4,
        weight_decay: float = 5e-2,
        max_epochs: int = 100,
        warmup_epochs: int = 10,
        seed: int = 42,
        imagenet_norm: bool = False,
        svm_probe_every: int = 1,
        log_recon_every: int = 10,
        freeze_encoder_flag: bool = False,
        freeze_spifil_layer: bool = False,
        head_finetune: bool = False,
        init_ckpt: str = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if flim_weights_path is None:
            raise ValueError(
                "flim_weights_path is required — this experiment always initialises "
                "the encoder with FLIM weights (encoder_init=flim)."
            )

        # O `architecture.json` manda, inclusive para protozoan. O `PROTOZOAN_FLIM_ARCH`
        # existia para nao confiar no json de ch24_30_48, mas os dois so divergem em
        # `noutput_channels`/`nkernels_*` — campos que `override_arch_channels` sobrescreve
        # logo abaixo com o que os `conv{n}-bias.txt` dizem. Ler o json e portanto
        # equivalente, e e o que deixa uma arquitetura CRESCIDA (nlayers > 3, escrita por
        # `scripts/spifil_grow.py`) chegar ate aqui em vez de ser trocada por 3 camadas.
        arch = parse_architecture(arch_json)
        channels = get_actual_channels_from_weights(flim_weights_path, arch, in_channels)
        arch = override_arch_channels(arch, channels)

        self.model = AutoEncoderFLIM(
            arch=arch, in_channels=in_channels, out_size=(image_size, image_size)
        )
        self.encoder_out_channels: int = channels[-1]
        self.channels: List[int] = channels

        load_FLIM_encoder(self.model, arch_json, flim_weights_path, channels)

        # Reference copy of the FLIM kernels, taken before any gradient touches them, so
        # stage 2 can measure how far it drifted from the prior the markers produced.
        # A plain attribute, not a buffer: it must NOT ride along in the state_dict, or
        # stage 2's load_from_checkpoint would overwrite it with stage 1's copy.
        self._flim_ref = {
            n: p.detach().cpu().clone() for n, p in self.model.encoder.named_parameters()
        }
        if freeze_encoder_flag:
            # Stage 1 (no init_ckpt): whole encoder frozen, whole decoder learning.
            # Stage 3 (init_ckpt of a SHALLOWER model): only what is new learns — the grown
            # encoder layer and the decoder blocks that came with it. Safe to decide here:
            # the state_dict is laid on top afterwards (main:914) and `load_state_dict`
            # copies data in-place, it never touches `requires_grad`.
            delta = 0
            if init_ckpt:
                try:
                    state = torch.load(init_ckpt, map_location="cpu").get("state_dict", {})
                except Exception as exc:  # unreadable / not a Lightning checkpoint
                    _log.warning("[stage 3] cannot read init_ckpt %s (%s) — falling back to "
                                 "the stage-1 freeze (whole encoder)", init_ckpt, exc)
                    state = {}
                if not state:
                    # `delta` stays 0 on purpose: an empty state would make the formula read
                    # n_layers, i.e. "everything is new", the opposite of the intended fallback.
                    _log.warning("[stage 3] init_ckpt %s yielded no state_dict — falling back to "
                                 "the stage-1 freeze (whole encoder)", init_ckpt)
                else:
                    delta = _decoder_block_delta(state, self.model.encoder.n_layers)
                    if delta == 0:
                        _log.warning("[stage 3] init_ckpt %s already has %d decoder blocks: "
                                     "nothing grew, so there is no 'new layer' to train alone "
                                     "— falling back to the stage-1 freeze (whole encoder)",
                                     init_ckpt, self.model.encoder.n_layers)
            if delta >= 1:
                _freeze_for_growth(self.model, delta)
            else:
                freeze_encoder(self.model)  # models.py:672 — resolves .encoder on its own
        # AFTER the policy on purpose: as an ablation on a stage-3 round this re-freezes the
        # grown FLIM layer, leaving only the new decoder block learning. Order is the rule.
        if freeze_spifil_layer:
            # Last block only = the grafted SPiFiL layer. `Encoder.__init__` registers each
            # block both in `.blocks` and as `.conv{n}` (models.py:163-165), and both names
            # point at the SAME Parameter objects, so freezing through one reaches both.
            last = self.model.encoder.blocks[f"conv{self.model.encoder.n_layers}"]
            for p in last.parameters():
                p.requires_grad = False
        # Depois de TODA politica de congelamento, para o log abaixo ser o que
        # configure_optimizers vai de fato ver.
        self.head = Head(channels[-1], num_classes) if head_finetune else None
        if head_finetune:
            # O decoder sai do caminho: a perda deixa de ser reconstrucao. Ele continua
            # existindo porque `AutoEncoderFLIM` sempre o constroi; congelado, fica fora do
            # otimizador — mas ainda viaja como PESO MORTO no .ckpt. Divida conhecida.
            for p in self.model.decoder.parameters():
                p.requires_grad = False
        if freeze_encoder_flag:
            # Logged only after the ablation, so the line is what configure_optimizers:706
            # will actually see. Without it a frozen round cannot be validated from the log.
            trainable = [n for n, p in self.model.named_parameters() if p.requires_grad]
            _log.info(
                "[stage %d] freeze policy | delta=%+d | trainable=%d params in %d tensors: %s",
                self.stage, delta,
                sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                len(trainable), trainable,
            )

        self.criterion = nn.BCEWithLogitsLoss()

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_norm_mean", mean, persistent=False)
        self.register_buffer("_norm_std", std, persistent=False)

        _log.info(
            "[AutoEncoderFlimModule] FLIM weights from %s | channels=%s | embed_dim=%d | "
            "imagenet_norm=%s | probe_every=%d | stage=%d (encoder %s, spifil layer %s)",
            flim_weights_path, channels, self.encoder_out_channels,
            imagenet_norm, svm_probe_every,
            self.stage, "frozen" if freeze_encoder_flag else "trainable",
            "frozen" if freeze_spifil_layer else "trainable",
        )

        # ONE source of truth for the metric namespace — and it is now the SAME string in every
        # stage and every growth round, so the kappa key is always `probe/svm_kappa` and all
        # rounds land on ONE comparable curve instead of one curve per stage. The stage identity
        # lives in the run name and in the W&B config, not in the metric key. Everything below
        # still derives from it by f-string and is never typed a second time: a monitor that does
        # not match a logged key does NOT raise (EarlyStopping is built with strict=False, which
        # is required for --svm-probe-every > 1), it just runs to max_epochs in silence.
        self._stage_prefix = "probe"

        self._val_emb: List[Tensor] = []
        self._val_labels: List[Tensor] = []
        self._val_head_logits: List[Tensor] = []
        self._sample_batch: Optional[Tensor] = None
        self._train_loader_cache = None
        self.baseline_metrics: Optional[dict] = None
        self.last_probe_metrics: Optional[dict] = None
        self._nonconv_warned = False
        self._epoch_offset: Optional[int] = None
        self._best_kappa: Optional[float] = None
        self._best_delta: Optional[float] = None

    # ── Stage identity ─────────────────────────────────────────────────────

    @property
    def stage(self) -> int:
        """1 = frozen encoder from FLIM · 2 = unfrozen end-to-end · 3 = grown by one layer and
        trained frozen, resuming the backbone from the previous round's checkpoint."""
        if not self.hparams.freeze_encoder_flag:
            return 2
        return 3 if self.hparams.init_ckpt else 1

    @property
    def stage_prefix(self) -> str:
        """``"probe"`` — the namespace of every metric this module logs, in every stage."""
        return self._stage_prefix

    @property
    def monitor_spec(self) -> Tuple[str, str, str]:
        """``(monitor, mode, checkpoint filename)`` — bottleneck kappa, in every stage.

        Read by ``main()`` instead of being re-derived there, so the callbacks cannot end up
        watching a key that is never logged.

        Kappa is the only selection signal at every stage so that all growth rounds are
        comparable. The caveat, stated honestly: with a frozen encoder the embedding is
        identical every epoch, so kappa moves only with the SVM's own fit noise (0.0675 spread
        measured here across permutations of the same feature matrix). A frozen stage's kappa
        is therefore a measurement of the round's starting point, not a training curve, and
        which epoch wins is close to arbitrary.
        """
        # Com --head-finetune quem seleciona e o kappa da Head: ali o encoder pode estar
        # congelado, mas a Head TREINA, entao o pico e sinal e nao ruido de fit. O nome do
        # arquivo continua `best_kappa` — o laco de crescimento monta o caminho por convencao.
        if self.hparams.head_finetune:
            return f"{self._stage_prefix}/head_kappa", "max", "best_kappa"
        return f"{self._stage_prefix}/svm_kappa", "max", "best_kappa"

    @property
    def epoch_metric_keys(self) -> Tuple[str, ...]:
        """Per-epoch keys this module logs, as they reach the logger.

        ``main()`` asserts its monitor is one of these. ``train_recon_loss`` is absent because
        Lightning splits it into ``_step``/``_epoch``, which makes it unusable as a monitor.
        """
        p = self._stage_prefix
        return (
            f"{p}/val_recon_loss",
            f"{p}/svm_kappa", f"{p}/svm_acc", f"{p}/svm_f1", f"{p}/svm_fit_s",
            f"{p}/svm_fit_status", f"{p}/svm_n_iter_max", f"{p}/svm_n_iter_sum",
            f"{p}/svm_n_sv",
            f"{p}/head_kappa", f"{p}/head_acc",
            f"{p}/flim_drift", "stage_epoch",
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _first_view(views: Union[List[Tensor], Tensor]) -> Tensor:
        """The datamodule yields ``[B, V, 3, H, W]``; this experiment runs with V=1."""
        if isinstance(views, (list, tuple)):
            return views[0]
        if views.ndim == 5:
            return views[:, 0]
        return views

    def _target_from_input(self, x: Tensor) -> Tensor:
        """Recover the LAB image in [0, 1] that BCE needs from the (possibly) normalised input.

        ``ift_lab`` yields LABNorm2 already in [0, 1], so undoing the ImageNet
        normalisation is the only step — there is no separate ``lab_scale``.
        """
        if not self.hparams.imagenet_norm:
            return x.clamp(0.0, 1.0)
        return (x * self._norm_std + self._norm_mean).clamp(0.0, 1.0)

    # ── Forward / training ─────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _recon_loss(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        target = self._target_from_input(x)
        logits = self(x)
        return self.criterion(logits, target), logits, target

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch  # y so entra na perda no braco --head-finetune
        x = self._first_view(views)
        if self.head is not None:
            # `y` ja e 0-based (datasets/dataset.py:173), entao entra direto. O `-1` do
            # caminho do SVM desfaz um `+1` do proprio evaluate.py:472 e nao vale aqui.
            loss = nn.functional.cross_entropy(self.head(self.model.encoder(x)), y)
        else:
            loss, _, _ = self._recon_loss(x)

        opt = self.optimizers()
        # lr is the schedule, not a result: it stays outside the stage namespace.
        self.log("train/lr", opt.param_groups[0]["lr"], on_step=True, on_epoch=False)
        self.log(f"{self._stage_prefix}/"
                 f"{'train_ce_loss' if self.head is not None else 'train_recon_loss'}",
                 loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_emb = []
        self._val_labels = []
        self._val_head_logits = []
        self._sample_batch = None

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        views, y = batch
        x = self._first_view(views)
        loss, logits, _ = self._recon_loss(x)

        # _encode_pooled already returns a detached CPU tensor — no second .detach().cpu().
        self._val_emb.append(_encode_pooled(self.model.encoder, x).float())
        self._val_labels.append(y.detach().cpu())
        # ponytail: passe extra do encoder — `_encode_pooled` acima ja devolve pooled e
        # detached, o que nao serve para a Head. Teto: fundir os dois passes se a validacao
        # ficar cara.
        if self.head is not None:
            self._val_head_logits.append(self.head(self.model.encoder(x)).float().cpu())
        if batch_idx == 0:
            self._sample_batch = (x[:4].detach().cpu(), logits[:4].detach().float().cpu())

        p = self._stage_prefix
        self.log(f"{p}/val_recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    # ponytail: the constant-prediction BCE floor (`val_recon_baseline`) is gone. It was the
    # only true constant-baseline curve in the run — one number per stage, replotted every
    # epoch as a flat line. Ceiling: the "is the loss near its floor?" ratio is no longer on
    # the panel; recompute it offline from the target's own entropy if a paper needs it.

    # ── SVM probe ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embeddings(self, loader) -> Tuple[np.ndarray, np.ndarray]:
        """One deterministic pass over *loader*, returning pooled embeddings and labels."""
        was_training = self.model.training
        self.model.eval()
        embs, labels = [], []
        for views, y in loader:
            x = self._first_view(views).to(self.device, non_blocking=True)
            embs.append(_encode_pooled(self.model.encoder, x).float())
            labels.append(y.cpu())
        if was_training:
            self.model.train()
        return torch.cat(embs).numpy(), torch.cat(labels).numpy()

    def _train_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """One deterministic pass over the labelled train split.

        The percentage does not enter the reconstruction loss; it decides how many
        labelled images the probe gets to see here.
        """
        if self._train_loader_cache is None:
            self._train_loader_cache = self.trainer.datamodule.train_dataloader()
        return self._embeddings(self._train_loader_cache)

    def on_fit_start(self) -> None:
        """Probe the untouched FLIM encoder before a single gradient step.

        This is the reference the experiment is judged against. It cannot be taken from
        the existing baseline CSVs: those score the SVM on the **test** split
        (src/evaluate/svm.py:123-137) while this probe scores on **validation**, so the
        two numbers are not comparable. Measuring the baseline here, with the identical
        protocol, splits and probe, makes improved/tied/degraded a sound verdict.
        """
        self._define_wandb_axis()
        try:
            X_tr, y_tr = self._train_embeddings()
            X_val, y_val = self._embeddings(self.trainer.datamodule.val_dataloader())
            metrics = self._svm_probe(X_tr, y_tr, X_val, y_val)
        except Exception as exc:
            _log.warning("FLIM baseline probe failed: %s", exc)
            return
        if metrics is None:
            return

        self.baseline_metrics = metrics
        _log.info(
            "[AutoEncoderFlimModule] FLIM-init reference (no training): "
            "kappa=%.4f acc=%.4f f1=%.4f | svm fit_status=%d n_iter_max=%d n_sv=%d",
            metrics["kappa"], metrics["acc"], metrics["f1"],
            metrics["fit_status"], metrics["n_iter_max"], metrics["n_sv"],
        )
        # Summary only, never a per-epoch series: replayed every epoch it was bit-identical to
        # the stage-1 kappa curve in 18/18 cells, two names for one number. It stays as the
        # reference the delta in on_validation_epoch_end is built on.
        # self.log() is not allowed this early anyway; write straight to the W&B summary.
        if isinstance(self.logger, WandbLogger):
            for key, value in metrics.items():
                self.logger.experiment.summary[
                    f"{self._stage_prefix}/flim_ref_svm_{key}"
                ] = value

    def _define_wandb_axis(self) -> None:
        """Give this stage a 0-based x axis of its own instead of the global step.

        Without it stage 2's panel starts wherever stage 1 stopped (epoch 453 in the audited
        grid) and the two curves cannot be read side by side. Guarded like the rest of the
        W&B-specific code, so a run with no logger — or a non-W&B one — is unaffected.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        try:
            import wandb

            wandb.define_metric("stage_epoch")
            wandb.define_metric(f"{self._stage_prefix}/*", step_metric="stage_epoch")
            # An exact name beats a glob in W&B: the per-step train loss is logged between
            # validations, where stage_epoch has no value, so it keeps the global step.
            wandb.define_metric(f"{self._stage_prefix}/train_recon_loss_step")
        except Exception as exc:
            _log.debug("W&B step-metric definition skipped: %s", exc)

    def _svm_probe(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> Optional[dict]:
        """One-vs-one linear SVM, fitted on train, scored on validation.

        Hyperparameters mirror ``src/utils/evaluate.py:270-278`` — including the absence of a
        scaler — with **one deliberate departure**: ``max_iter``. The historical cap of 10 000
        truncated the solver in 18 of 18 cells of this grid (``fit_status_=1``), so the kappa
        it returned measured where the iteration counter stopped, not the encoder; permuting
        the rows of the same feature matrix moved it by up to 0.87. ``max_iter=-1`` lets libsvm
        run to its own stopping criterion, which also makes the probe order-invariant. The
        price is explicit: this curve is **not** numerically comparable with the CSVs written
        by the capped evaluators.

        The solver's own state comes back with the metrics. Discarding ``fit_status_`` and
        ``n_iter_`` is exactly why the truncation survived 36 runs unnoticed.
        """
        if len(np.unique(y_tr)) < 2:
            _log.warning("SVM probe skipped: train split has a single class.")
            return None
        # ponytail: única SVC fora de ``src/utils/evaluate.py:fit_svm`` — config idêntica, mas fit_svm imprime a linha de diagnóstico e sobe uma thread tqdm por segundo a cada validação; teto: unificar quando fit_svm tiver modo silencioso (tools/check_refactor_equivalence.py trava o drift).
        clf = SVC(
            max_iter=-1,
            C=1e2,
            degree=3,
            gamma="auto",
            coef0=0,
            decision_function_shape="ovo",
            kernel="linear",
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)

        fit_status = int(getattr(clf, "fit_status_", -1))
        n_iter = np.asarray(getattr(clf, "n_iter_", 0))  # one entry per one-vs-one pair
        if fit_status != 0 and not self._nonconv_warned:
            self._nonconv_warned = True
            _log.warning(
                "SVM probe did NOT converge (fit_status_=%d, n_iter max=%d) even with "
                "max_iter=-1. Every kappa from here on measures the solver, not the "
                "embedding — do not read the verdict.",
                fit_status, int(n_iter.max()) if n_iter.size else -1,
            )

        metrics = compute_metrics(y_val, y_pred, num_classes=self.hparams.num_classes)
        metrics.update(
            fit_status=fit_status,
            n_iter_max=int(n_iter.max()) if n_iter.size else 0,
            n_iter_sum=int(n_iter.sum()),
            n_sv=int(np.sum(clf.n_support_)),
        )
        self.last_probe_metrics = metrics
        return metrics

    # ── FLIM drift ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _log_flim_drift(self) -> None:
        """Relative distance from the FLIM kernels the markers produced.

        Guards the reading of Δκ. Reconstruction loss falling while the drift explodes does
        not mean the FLIM encoder *adapted* — it means it was overwritten, and what remains
        is a randomly-seeded CNN that got extra training steps. Only logged in stage 2: in
        stage 1 the encoder is frozen, so the drift is 0 by construction.
        """
        if self.hparams.freeze_encoder_flag:
            return
        drift = torch.stack([
            (p.detach().cpu() - self._flim_ref[n]).norm()
            / self._flim_ref[n].norm().clamp_min(1e-12)
            for n, p in self.model.encoder.named_parameters()
            if n in self._flim_ref
        ]).mean()
        self.log(f"{self._stage_prefix}/flim_drift", drift, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if not self._val_emb or self.trainer.sanity_checking:
            return

        p = self._stage_prefix
        # 0-based x for THIS stage, logged every validation epoch. On a crash resume from
        # last.ckpt the offset is the epoch training restarts at, so the curve still starts at
        # 0: the resumed process is a new W&B run, and an axis shifted by the crash epoch would
        # not line up with stage 1's.
        if self._epoch_offset is None:
            self._epoch_offset = self.current_epoch
        self.log("stage_epoch", float(self.current_epoch - self._epoch_offset), on_epoch=True)

        self._log_flim_drift()

        # Antes do gate do svm_probe_every de proposito: com --head-finetune este e O monitor,
        # e ele nao pode faltar nas epocas em que a sonda SVM e pulada.
        if self._val_head_logits:
            # compute_metrics faz o argmax sozinho quando y_pred e (N, C) — logits crus entram.
            head_metrics = compute_metrics(
                torch.cat(self._val_labels), torch.cat(self._val_head_logits),
                num_classes=self.hparams.num_classes,
            )
            self.log(f"{p}/head_kappa", head_metrics["kappa"], prog_bar=True, on_epoch=True)
            self.log(f"{p}/head_acc", head_metrics["acc"], on_epoch=True)

        every = max(1, int(self.hparams.svm_probe_every))
        if (self.current_epoch % every) != 0 and self.current_epoch != self.trainer.max_epochs - 1:
            return

        X_val = torch.cat(self._val_emb).numpy()
        y_val = torch.cat(self._val_labels).numpy()

        try:
            X_tr, y_tr = self._train_embeddings()
            t0 = time.time()
            metrics = self._svm_probe(X_tr, y_tr, X_val, y_val)
            fit_s = time.time() - t0
        except Exception as exc:  # a probe failure must not kill the pretraining run
            _log.warning("SVM probe failed at epoch %d: %s", self.current_epoch, exc)
            return

        if metrics is None:
            return

        self.log(f"{p}/svm_kappa", metrics["kappa"], prog_bar=True, on_epoch=True)
        self.log(f"{p}/svm_acc", metrics["acc"], prog_bar=False, on_epoch=True)
        self.log(f"{p}/svm_f1", metrics["f1"], prog_bar=False, on_epoch=True)
        self.log(f"{p}/svm_fit_s", fit_s, prog_bar=False, on_epoch=True)
        # Solver state next to the number it produced: a kappa with fit_status != 0 is a
        # reading of the iteration counter, and the curve has to say so on its own.
        self.log(f"{p}/svm_fit_status", float(metrics["fit_status"]), on_epoch=True)
        self.log(f"{p}/svm_n_iter_max", float(metrics["n_iter_max"]), on_epoch=True)
        self.log(f"{p}/svm_n_iter_sum", float(metrics["n_iter_sum"]), on_epoch=True)
        self.log(f"{p}/svm_n_sv", float(metrics["n_sv"]), on_epoch=True)

        self._best_kappa = (
            metrics["kappa"] if self._best_kappa is None
            else max(self._best_kappa, metrics["kappa"])
        )

        # ponytail: the delta against the on_fit_start FLIM reference is bookkeeping only, no
        # longer a pair of curves. In stage 1 the encoder never moves, so both series were a
        # constant line by construction, and in stage 2 they are `svm_kappa` shifted by a
        # constant. Ceiling: the delta is now readable only at the end, from
        # verdict_summary() / run_metadata.json, not epoch by epoch on the W&B panel.
        if self.baseline_metrics is not None:
            delta = metrics["kappa"] - self.baseline_metrics["kappa"]
            self._best_delta = delta if self._best_delta is None else max(self._best_delta, delta)

        self._log_reconstruction()

    # ── Verdict ────────────────────────────────────────────────────────────

    def verdict_summary(self) -> dict:
        """improved / tied / degraded, decided against ``KAPPA_TOLERANCE``.

        ``svm_kappa_delta_best`` is the best probe kappa of this stage minus the
        ``on_fit_start`` FLIM reference. For stage 2 that reference is the frozen-FLIM number,
        since stage 1 never moves the encoder, so the delta answers the actual question:
        did reconstruction improve the FLIM embedding?

        Read it next to ``{stage}/flim_drift``. A positive delta with an exploding drift is
        **not** "FLIM adapted" — it is "FLIM was erased and a random CNN trained in its place",
        and the deliverable of this experiment is the FLIM encoder, not that CNN.

        Empty when there is nothing to compare (reference probe or every epoch probe failed).
        """
        ref = (self.baseline_metrics or {}).get("kappa")
        if ref is None or self._best_kappa is None:
            return {}
        delta = self._best_kappa - ref
        return {
            "flim_ref_svm_kappa": ref,
            "best_svm_kappa": self._best_kappa,
            "svm_kappa_delta_best": delta,
            "kappa_tolerance": KAPPA_TOLERANCE,
            "verdict": (
                "improved" if delta > KAPPA_TOLERANCE
                else "degraded" if delta < -KAPPA_TOLERANCE
                else "tied"
            ),
        }

    def on_fit_end(self) -> None:
        summary = self.verdict_summary()
        if not summary:
            return
        _log.info(
            "[%s] verdict=%s | best kappa=%.4f vs FLIM reference %.4f (delta %+.4f, tol %.3f)",
            self._stage_prefix, summary["verdict"], summary["best_svm_kappa"],
            summary["flim_ref_svm_kappa"], summary["svm_kappa_delta_best"], KAPPA_TOLERANCE,
        )
        if not isinstance(self.logger, WandbLogger):
            return
        try:
            for key, value in summary.items():
                self.logger.experiment.summary[f"{self._stage_prefix}/{key}"] = value
        except Exception as exc:
            _log.debug("W&B verdict summary skipped: %s", exc)

    # ── W&B reconstruction preview ─────────────────────────────────────────

    def _lab01_to_rgb(self, lab01: np.ndarray) -> np.ndarray:
        """``[3,H,W]`` LABNorm2 in [0,1] -> ``[H,W,3]`` sRGB in [0,1].

        Without this the panel would render L/a/b as if they were R/G/B and invite the
        wrong conclusion about reconstruction quality.
        """
        from skimage.color import lab2rgb

        c = np.transpose(lab01, (1, 2, 0)).astype(np.float64)
        lab = np.stack(
            [
                c[..., 0] * LAB_L_SCALE,
                c[..., 1] * LAB_AB_SCALE - LAB_AB_OFFSET,
                c[..., 2] * LAB_AB_SCALE - LAB_AB_OFFSET,
            ],
            axis=-1,
        )
        return np.clip(lab2rgb(lab), 0.0, 1.0)

    def _log_reconstruction(self) -> None:
        if self._sample_batch is None or self.logger is None:
            return
        if not isinstance(self.logger, WandbLogger):
            return
        every = max(1, int(self.hparams.log_recon_every))
        if (self.current_epoch % every) != 0:
            return

        try:
            import wandb

            x, logits = self._sample_batch
            target = self._target_from_input(x.to(self._norm_mean.device)).cpu().numpy()
            recon = torch.sigmoid(logits).numpy()

            images = []
            for i in range(target.shape[0]):
                pair = np.concatenate(
                    [self._lab01_to_rgb(target[i]), self._lab01_to_rgb(recon[i])], axis=1
                )
                images.append(wandb.Image(pair, caption=f"epoch {self.current_epoch} — alvo | recon"))
            self.logger.experiment.log(
                {f"{self._stage_prefix}/reconstruction": images}, commit=False
            )
        except Exception as exc:
            _log.debug("Reconstruction preview skipped: %s", exc)

    # ── Optimizer ──────────────────────────────────────────────────────────

    def configure_optimizers(self):
        warmup = self.hparams.warmup_epochs or 10
        total = self.hparams.max_epochs or 100
        # Same filter as classification_flim_module.py:232 — stage 1's frozen encoder is
        # then respected for free, with no optimizer state allocated for its kernels.
        optimizer = AdamW(
            # self.parameters(), nao self.model.parameters(): a Head vive fora de `model`.
            # Sem Head os dois conjuntos sao identicos, entao nada muda no fluxo de recon.
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        sched_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup)
        sched_cosine = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-5)
        scheduler = SequentialLR(
            optimizer, schedulers=[sched_warmup, sched_cosine], milestones=[warmup]
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ── Standalone entry-point ─────────────────────────────────────────────────────

NUM_CLASSES = {"eggs": 9, "larvae": 2, "protozoan": 7}
ARCH_TAG = {"eggs": "ch24_32_48", "larvae": "ch24_32_48", "protozoan": "ch24_30_48"}


def _dataset_short_to_parasite_name(dataset: str) -> str:
    return {
        "eggs": "helminth-eggs_split_2",
        "larvae": "helminth-larvae_split_2",
        "protozoan": "protozoan-cysts_split_2",
    }[dataset]


def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Train one unsupervised AutoEncoder (FLIM encoder + ResNet decoder) run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=["eggs", "larvae", "protozoan"])
    p.add_argument("--split", required=True, type=int)
    p.add_argument("--percentage", required=True, type=int)
    p.add_argument("--arch-json", required=True)
    p.add_argument("--flim-weights-path", required=True,
                   help="Directory with conv{n}-kernels.npy / conv{n}-bias.txt.")
    p.add_argument("--recon-loss", default="bce_logits", choices=["bce_logits"],
                   help="Reconstruction loss. Cross-entropy over LAB in [0,1] as logits.")
    p.add_argument("--freeze-encoder", action="store_true", default=False,
                   help="Freeze the encoder so only the decoder learns. The probe then "
                        "measures the encoder as this round received it. Combinable with "
                        "--init-ckpt: that pair is a grown round trained frozen.")
    p.add_argument("--init-ckpt", default="",
                   help="Resume the backbone from this checkpoint of the previous growth "
                        "round instead of from a random decoder. Loaded with the decoder "
                        "block indices shifted by the layers gained, and weights only — the "
                        "optimizer, scheduler and epoch counter are NOT restored.")
    p.add_argument("--freeze-spifil-layer", action="store_true", default=False,
                   help="Keep the grafted SPiFiL filter bank (the last encoder block) exactly "
                        "as SPiFiL cut it. Off by default: those patches are cut, not trained, "
                        "but once written into the layer they are ordinary conv weights just "
                        "like the FLIM kernels the unfrozen stages already fine-tune. Pass it "
                        "for the ablation that keeps the bank fixed.")
    p.add_argument("--head-finetune", action="store_true", default=False,
                   help="Troca reconstrucao por classificacao: Head (GAP -> dropout -> linear) "
                        "sobre a ultima camada do encoder, perda cross_entropy, selecao por "
                        "probe/head_kappa. O decoder e congelado e sai do otimizador. Sem a "
                        "flag o fluxo de reconstrucao segue identico.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=50,
                   help="EarlyStopping patience, in epochs, on probe/svm_kappa — the one "
                        "metric every stage selects by. Matches the patience "
                        "configs/default.yaml uses for the SSL runs.")
    p.add_argument("--embed-mode", default="avgpool2d", choices=["avgpool2d", "flatten"],
                   help="How src.utils.evaluate reduces the LAST encoder layer's map for "
                        "the probe (conv3 on the ungrown model, conv{3+rounds} after growth): "
                        "global average pool to [B, C] (the heads' convention) or the "
                        "spatially flattened map. Governs the official evaluator too.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=200)
    p.add_argument("--svm-probe-every", type=int, default=1)
    p.add_argument("--log-recon-every", type=int, default=10)
    p.add_argument("--log-every-n-steps", type=int, default=1,
                   help="Com que frequencia o Lightning descarrega as metricas de step "
                        "(train/lr, probe/train_recon_loss) para o logger. Default 1 = "
                        "toda step. O default 10 do Lightning esconde a curva inteira "
                        "nos splits pequenos: pct5 tem 124 imagens de treino, ou 4 steps "
                        "por epoca, entao nada seria logado antes da terceira epoca.")
    # Default OFF, and only here: the FLIM kernels already embed the marker normalisation over
    # LAB[0, 1], so normalising again moves their effective bias and decalibrates every ReLU
    # cut (-0.33 kappa, measured). The I-JEPA/distillation arms keep it ON — their teacher was
    # trained with it. BooleanOptionalAction so --no-imagenet-norm still parses.
    p.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=False,
                   help="ImageNet RGB Normalize on top of the ift_lab LAB[0,1] input. Wrong "
                        "for FLIM-init (lejepa_dataset.py:42), hence off by default here.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="journal_02_2026_hybrid_FLIM")
    p.add_argument("--wandb-entity", default="ophira-ai")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index this run trains on. Omitted keeps Lightning's default "
                        "(devices=1, the first visible GPU), which is what a single run wants; "
                        "a launcher firing several experiments at once passes a different index "
                        "per run so they do not collide. Replaces CUDA_VISIBLE_DEVICES on this "
                        "path — everything a run needs travels as an explicit flag. Ignored when "
                        "no CUDA device is present.")
    return p


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.init_ckpt and not os.path.isfile(args.init_ckpt):
        parser.error(f"--init-ckpt does not exist: {args.init_ckpt}")

    # Checked here, before the datamodule spends minutes loading: a wrong index has to fail now.
    if args.gpu is not None:
        if not torch.cuda.is_available():
            _log.info("no CUDA device visible, ignoring --gpu %d", args.gpu)
            args.gpu = None
        elif args.gpu not in range(torch.cuda.device_count()):
            n = torch.cuda.device_count()
            parser.error(f"--gpu {args.gpu} out of range: this machine has "
                         f"{n} CUDA device(s), valid indices 0..{n - 1}")

    # The two flags are orthogonal, not exclusive: frozen without a checkpoint is the first
    # round (1), unfrozen is the end-to-end round (2), and frozen *with* a checkpoint is a
    # grown round trained frozen on top of the previous one (3).
    stage = 2 if not args.freeze_encoder else (3 if args.init_ckpt else 1)
    if stage == 2 and not args.init_ckpt:
        _log.warning(
            "Stage 2 without --init-ckpt: the encoder is unfrozen against a RANDOM decoder, "
            "so the first epochs push noise into the FLIM kernels. That is the ablation, "
            "not the protocol."
        )

    # One module-level switch, read at call time inside _encode_pooled, so the probe and the
    # official evaluator cannot end up in different embedding modes. Set before the Trainer.
    import src.utils.evaluate as _ev
    _ev.EMBED_MODE = args.embed_mode

    pl.seed_everything(args.seed, workers=True)

    output_dir = args.output_dir or os.path.join(
        _ROOT, "artifacts", "autoencoder_resnet_init_flim", args.run_name
    )
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    from src.data_modules.parasite_data_module_lejepa_splited import (
        ParasiteLejepaDataModuleSplited,
    )

    # V_train=V_eval=1 -> the dataset uses the deterministic test transform
    # (parasite_lejepa.py:109). Reconstruction target == input, and the LeJEPA colour
    # augmentations (ColorJitter/Grayscale/Solarize) never touch the LAB channels.
    datamodule = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(args.dataset),
        split=args.split, percentage=args.percentage,
        image_size=args.image_size, V_train=1, V_eval=1,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
        loader="ift_lab",
        imagenet_norm=args.imagenet_norm,
    )

    module_kwargs = dict(
        arch_json=args.arch_json,
        dataset=args.dataset,
        flim_weights_path=args.flim_weights_path,
        num_classes=NUM_CLASSES[args.dataset],
        image_size=args.image_size,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        imagenet_norm=args.imagenet_norm,
        svm_probe_every=args.svm_probe_every,
        log_recon_every=args.log_recon_every,
        freeze_encoder_flag=args.freeze_encoder,
        freeze_spifil_layer=args.freeze_spifil_layer,
        head_finetune=args.head_finetune,
        init_ckpt=args.init_ckpt,
    )

    # Weights only, deliberately. `trainer.fit(ckpt_path=...)` would also restore the epoch
    # counter, the optimizer AND the scheduler — the new round would resume on the tail of the
    # cosine at lr ~= eta_min=1e-5 and "converge" without having moved. It would also make the
    # W&B logger resume the previous run, fusing two runs that must stay separate. And it could
    # not work at all across a growth round: the model is a layer wider than the checkpoint.
    # Building the module first is what rebuilds `_flim_ref` from the original FLIM kernels
    # (the reference probe/flim_drift needs) and what leaves the newly grown layer holding the
    # SPiFiL bank `load_FLIM_encoder` just wrote — the old state_dict is only laid on top.
    module = AutoEncoderFlimModule(**module_kwargs)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu").get("state_dict", {})
        # ResNetDecoder builds its blocks DEEPEST FIRST (autoencoder_resnet.py:108-116), so
        # block 0 is always the bottleneck's up-block: gaining a layer pushes every old index
        # up by `delta`. A name-based strict=False load would hit a shape mismatch on each of
        # them and silently throw the whole trained decoder away. `to_image` keeps its name —
        # its width is channels[1], which growth does not touch.
        pre = "model.decoder.blocks."
        delta = _decoder_block_delta(state, module.model.encoder.n_layers)
        # As chaves `head.*` sao DESCARTADAS: a Head e da camada, nao do braco. Duas razoes.
        # (a) O espaco de features que ela classificava ganhou uma camada inteira desde este
        # checkpoint, entao aqueles pesos decidem sobre outra coisa — sem descartar, o remap
        # nao toca em `head.*`, elas atravessam e sobrescrevem a Head recem-construida, e a
        # Head da rodada 2 sai byte-identica a da rodada 1. (b) A heranca silenciosa vira
        # crash assim que a largura mudar entre rodadas: `load_state_dict` levanta size
        # mismatch MESMO com strict=False, e hoje so nao levanta por coincidencia aritmetica
        # (o allocator do SPiFiL faz out_channels // n_classes * n_classes, entao eggs vai
        # 48 -> 45 -> 45 e as formas casam). Descartadas, a Head nasce nova sobre o espaco de
        # features atual — que e o que "plugar a Head na saida da camada nova" significa.
        remapped, dropped_head = {}, 0
        for k, v in state.items():
            if k.startswith("head."):
                dropped_head += 1
                continue
            if k.startswith(pre):
                idx, rest = k[len(pre):].split(".", 1)
                k = f"{pre}{int(idx) + delta}.{rest}"
            remapped[k] = v
        missing, unexpected = module.load_state_dict(remapped, strict=False)
        _log.info(
            "[stage %d] backbone from %s | decoder block shift delta=%+d | "
            "missing=%d (left at their FLIM/SPiFiL init) unexpected=%d | "
            "head keys dropped=%d (a Head sempre nasce nova sobre a camada atual)",
            stage, args.init_ckpt, delta, len(missing), len(unexpected), dropped_head,
        )

    # Both callbacks follow the bottleneck kappa, in every stage: the best *encoder* is the
    # deliverable of every growth round, and one metric for all of them is what makes the
    # rounds comparable. In a frozen round that curve is the SVM's fit noise around a fixed
    # embedding (0.0675 spread, measured on protozoan/split1/pct5 across permutations of the
    # same feature matrix), so its argmax is close to arbitrary — read that round's kappa as
    # the starting point it measured, not as a training curve. See monitor_spec.
    #
    # strict=False: with --svm-probe-every > 1 the kappa is missing on the epochs the probe
    # skips, and a strict callback aborts the run instead of waiting for the next one. That
    # tolerance is also the trap — a monitor naming a key the module never logs would not
    # raise either, it would just train max_epochs and select nothing. So the monitor is read
    # off the module (one stage variable, see monitor_spec) and checked against the keys the
    # module actually logs, which is the check strict=False gives up.
    monitor, mode, ckpt_name = module.monitor_spec
    assert monitor in module.epoch_metric_keys, (
        f"monitor {monitor!r} is not logged by the module: {module.epoch_metric_keys}"
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename=ckpt_name,
        monitor=monitor, mode=mode,
        save_last=False, save_top_k=1,
    )
    early_stop = EarlyStopping(
        monitor=monitor, mode=mode, patience=args.patience, strict=False,
    )
    _log.info("[stage %d] selection + early stop on %s (%s) -> %s.ckpt | metric prefix %s/ | "
              "device %s",
              module.stage, monitor, mode, ckpt_name, module.stage_prefix,
              "default" if args.gpu is None else f"cuda:{args.gpu}")

    logger_list: list = []
    if args.wandb:
        os.environ["WANDB_CONSOLE"] = "off"
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.run_name, save_dir=output_dir,
            )
            wandb_logger.experiment.tags = tuple(dict.fromkeys(
                list(wandb_logger.experiment.tags or ())
                # O braco com Head roda SEM --freeze-encoder, entao `stage` vale 2 aqui —
                # a mesma identidade do estagio 2 de reconstrucao. A tag e o que separa os
                # dois no board, em vez de um quarto valor de `stage` (ver _save_metadata).
                # "unsupervised" tambem sai: com a Head a perda e cross_entropy sobre rotulo.
                + ["journal_02_2026_hybrid_FLIM", "autoencoder", "flim_init",
                   "supervised" if args.head_finetune else "unsupervised",
                   "head_finetune" if args.head_finetune else
                   {1: "stage1_frozen", 2: "stage2_fine_tune",
                    3: "stage3_grown_frozen"}[stage]]
            ))
            wandb_logger.log_hyperparams({
                "dataset": args.dataset,
                "split": args.split,
                "percentage": args.percentage,
                "encoder_init": "flim",
                "decoder": "resnet",
                "recon_loss": args.recon_loss,
                "architecture": ARCH_TAG[args.dataset],
                "channels": module.channels,
                "color_space": "lab",
                "loader": "ift_lab",
                "lab_scale": "labnorm2_native_01",
                "imagenet_norm": args.imagenet_norm,
                "svm_multiclass": "ovo",
                "svm_kernel": "linear",
                "svm_C": 1e2,
                # -1, not the evaluators' 10 000: see _svm_probe. Logged so a board mixing
                # these runs with the older ones shows which curve came from which solver.
                "svm_max_iter": -1,
                "svm_probe_every": args.svm_probe_every,
                "svm_probe_split": "fit=train, score=validation",
                "train_augmentation": "none",
                "num_classes": NUM_CLASSES[args.dataset],
                "embed_dim": module.encoder_out_channels,
                "arch_json": args.arch_json,
                "flim_weights_path": args.flim_weights_path,
                # Stage lineage — the metric keys no longer carry it, so the board reads it
                # from here. `stage` groups the board; `parent_run` pairs each round with the
                # round whose kappa is its baseline, so Δκ is computable straight from W&B
                # without opening a checkpoint.
                "stage": stage,
                "embed_mode": args.embed_mode,
                "encoder_frozen": args.freeze_encoder,
                "encoder_trainable": not args.freeze_encoder,
                "freeze_spifil_layer": args.freeze_spifil_layer,
                "init_ckpt": args.init_ckpt,
                "parent_run": _parent_run_from_ckpt(args.init_ckpt),
                "ckpt_monitor": monitor,
                "stage_metric_prefix": module.stage_prefix,
                "gpu": args.gpu,
            })
            logger_list.append(wandb_logger)
        except Exception as _e:
            _log.warning("W&B logger init failed: %s", _e)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # Sem --gpu, devices=1 = a primeira GPU visível; com --gpu, essa GPU e só ela.
        devices=1 if args.gpu is None else [args.gpu],
        callbacks=[checkpoint_cb, early_stop],
        logger=logger_list or False,
        log_every_n_steps=args.log_every_n_steps, enable_progress_bar=True, deterministic=False,
    )

    # Only finds last.ckpt from older runs: the callback no longer emits save_last, so a
    # crashed stage restarts from scratch. Never the stage-1 -> stage-2 handoff either: each
    # stage writes to its own artifacts directory, so stage 2 starts from --init-ckpt weights
    # with a fresh optimizer and scheduler.
    resume_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if not (os.path.isfile(resume_ckpt) and os.path.getsize(resume_ckpt) > 0):
        resume_ckpt = None

    t0 = time.time()
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_ckpt)
    except Exception as exc:
        _log.error("Training failed: %s", exc, exc_info=True)
        _save_metadata(args, module, output_dir, ckpt_dir, status="error", error=str(exc))
        return 1

    _save_metadata(
        args, module, output_dir, ckpt_dir, status="ok",
        best_ckpt=checkpoint_cb.best_model_path or "",
        best_score=_score(checkpoint_cb),
        elapsed=time.time() - t0,
    )

    if args.wandb:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass
    return 0


def _score(checkpoint_cb) -> float:
    s = getattr(checkpoint_cb, "best_model_score", None)
    try:
        return float(s.item() if hasattr(s, "item") else s)
    except (TypeError, ValueError):
        return float("nan")


def _parent_run_from_ckpt(init_ckpt: str) -> str:
    """``artifacts/<method>/<run-name>/checkpoints/best_recon.ckpt`` -> ``<run-name>``.

    Derived rather than taken as a flag: the brief fixes the CLI surface, and the stage-1
    run name is already encoded in the path stage 2 is pointed at.
    """
    if not init_ckpt:
        return ""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(init_ckpt))))


def _save_metadata(args, module, output_dir, ckpt_dir, status="ok", error="",
                   best_ckpt="", best_score=float("nan"), elapsed=0.0) -> None:
    import json
    import subprocess as _sp

    try:
        git_sha = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        git_sha = ""

    meta = {
        "run_name": args.run_name,
        "method": "autoencoder_resnet_init_flim",
        "dataset": args.dataset,
        "split": args.split,
        "percentage": args.percentage,
        "encoder_init": "flim",
        "decoder": "resnet",
        "recon_loss": args.recon_loss,
        "architecture": ARCH_TAG[args.dataset],
        "channels": module.channels,
        "embed_dim": module.encoder_out_channels,
        "num_classes": NUM_CLASSES[args.dataset],
        "color_space": "lab",
        "lab_scale": "labnorm2_native_01",
        "imagenet_norm": args.imagenet_norm,
        "svm_multiclass": "ovo",
        "svm_max_iter": -1,
        "train_augmentation": "none",
        "arch_json": args.arch_json,
        "flim_weights_path": args.flim_weights_path,
        "ckpt_dir": ckpt_dir,
        "best_ckpt": best_ckpt,
        # For a frozen RECONSTRUCTION round the monitored maximum is SVM fit noise (see
        # monitor_spec), so what is recorded there is the on_fit_start baseline instead.
        # Com --head-finetune a troca NAO vale: mesmo com o encoder congelado a Head treina de
        # verdade, o pico e sinal, e o melhor score vai para o disco como ele e.
        # O nome da chave e mantido por compatibilidade — scripts/spifil_growth_loop.py:110 le
        # `best_val_svm_kappa` e mata o braco se faltar. Qual metrica foi de fato monitorada
        # esta em `ckpt_monitor`, logo abaixo (probe/head_kappa no braco com Head).
        "best_val_svm_kappa": (
            best_score if (args.head_finetune or not args.freeze_encoder)
            else (module.baseline_metrics or {}).get("kappa", float("nan"))
        ),
        "best_monitor_score": best_score,
        # Same stage variable the callbacks were built from — never re-typed here, or the
        # metadata would document a monitor the run did not use.
        "ckpt_monitor": module.monitor_spec[0],
        "stage_metric_prefix": module.stage_prefix,
        "stage": module.stage,
        # `stage` sozinho NAO identifica este braco: com --head-finetune o encoder roda
        # destravado, entao a property devolve 2, igual ao estagio 2 de reconstrucao. A
        # property fica como esta de proposito — `stage` e documentado como 1/2/3 (o main()
        # re-deriva o mesmo numero em :969 para a config do W&B, e a tag de la e montada por
        # um dict {1,2,3}), e nenhum leitor no repo le este campo: eval_growth_stages.py:33
        # rotula pelo NOME do diretorio e ignora `stage` explicitamente. Um quarto valor
        # espalharia risco sem ganhar leitor. Quem separa o braco e a flag crua abaixo.
        "head_finetune": bool(args.head_finetune),
        # Lidos da Head CONSTRUIDA, nao re-derivados de args: se um dia divergirem do
        # `embed_dim`/`num_classes` acima, o metadado mostra a divergencia em vez de esconder.
        "head_in_channels": module.head.fc.in_features if module.head is not None else None,
        "head_num_classes": module.head.fc.out_features if module.head is not None else None,
        "embed_mode": args.embed_mode,
        "encoder_frozen": bool(args.freeze_encoder),
        "init_ckpt": args.init_ckpt,
        "parent_run": _parent_run_from_ckpt(args.init_ckpt),
        "baseline_flim_svm": module.baseline_metrics,
        # Solver state of the last probe. Without these three a repeat of the max_iter
        # truncation would again leave no trace on disk.
        "svm_fit_status": (module.last_probe_metrics or {}).get("fit_status"),
        "svm_n_iter_max": (module.last_probe_metrics or {}).get("n_iter_max"),
        "svm_n_iter_sum": (module.last_probe_metrics or {}).get("n_iter_sum"),
        "elapsed_s": elapsed,
        "status": status,
        "error": error,
        "git_sha": git_sha,
    }
    # The reading of the run — reference kappa, best kappa, delta and the improved/tied/
    # degraded call — on disk next to the checkpoint that produced it, so the verdict does not
    # live only in a W&B summary. Empty (keys absent) when there was nothing to compare.
    meta.update(module.verdict_summary())
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
