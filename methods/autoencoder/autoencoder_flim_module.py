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

Movido de ``src/modules/autoencoder_flim_module.py:185`` (AutoEncoderFlimModule) sem
nenhuma alteracao de corpo nas steps, na perda ou nos defaults. O bloco
``if __name__ == "__main__"`` e todo o argparse ficaram na origem: cada flag daquele
parser vira ``init_args`` no YAML. O relatorio do MA-06 lista o corte linha a linha.

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

  stage 1  ``freeze_encoder_flag=True``  encoder = FLIM, frozen; decoder random.
           The decoder learns to invert a *fixed* feature space, so it stops
           injecting garbage gradient into the FLIM kernels on the epochs where
           the decoder is still noise. The probe here is the "pure frozen FLIM"
           baseline, measured in the same run, split, seed, dataloader and probe.
  stage 2  ``init_ckpt=<previous>``  encoder + decoder both trainable, decoder
           starting from the previous round rather than from noise.
  stage 3  ``freeze_encoder_flag=True`` + ``init_ckpt=<previous>``  one extra conv layer
           grown on the encoder, trained frozen on top of the previous round's backbone.
           The decoder block indices shift by the layers gained; ``__init__`` remaps them
           on load.

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
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from sklearn.svm import SVC
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# core/blocks/init.py:44 — copia byte-a-byte de src/models/models.py:672, conferida
# pelo MA-06. As funcoes de init nao sao re-exportadas por core.blocks de proposito,
# entao o caminho e explicito.
from core.blocks.init import freeze_encoder
# core/metrics.py:46 — mesma `compute_metrics` de src/metrics/classification.py:31, com
# a unica diferenca de `average` ter virado literal "macro" em vez de argumento com esse
# mesmo default. Os dois call sites deste modulo nunca passam `average`, entao kappa, acc
# e f1 saem bit-a-bit iguais aos do W&B de hoje. `acc` continua sendo a acuracia
# BALANCEADA do torchmetrics (media por classe), a convencao que este modulo sempre usou.
from core.metrics import compute_metrics
from methods.autoencoder.autoencoder_flim import AutoEncoderFLIM
# The probe embeds with the official evaluator's own function, not with a look-alike:
# `AutoEncoderFLIM.embed` was mathematically identical to it (conv1→conv2→conv3, global
# average pool, flatten), and two identical implementations are one drift away from
# disagreeing. Importing it also means `evaluate.EMBED_MODE` governs the probe for free.
# ponytail: _encode_pooled sem casa na arvore; vai para eval/ quando E-EVAL rodar
from eval.svm import EMBED_MODES, _encode_pooled

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
#
# MA-06: valor duplicado em scripts/constants.py:258 e ja copiado para
# core/constants.py:181. Mantido local aqui de proposito — o agente de constantes e
# quem religa os consumidores; mexer nisso agora seria pisar no corte dele.
KAPPA_TOLERANCE = 0.01


def _decoder_block_delta(state: dict, n_layers: int) -> int:
    """How many decoder blocks are NEW relative to ``state`` — i.e. how many layers this
    model grew past the checkpoint it resumes from.

    ``ResNetDecoder`` builds its blocks deepest-first (resnet_decoder.py), so the new
    blocks are indices ``0 .. delta-1`` and every old index is shifted by ``+delta``.
    ONE formula, two callers: the state_dict remap and the stage-3 freeze policy, both
    in ``AutoEncoderFlimModule.__init__``.
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

    MA-06: esta classe NAO foi apagada em favor de ``core.blocks.MLPHead``. As duas nao sao
    a mesma cabeca. Aqui e GAP -> Flatten -> Dropout(0.2) -> UMA Linear, com chaves de
    state_dict ``fc.weight``/``fc.bias``; a MLPHead (core/blocks/mlp_head.py:36) e
    GAP -> view -> Sequential de TRES Linear com ReLU e Dropout(0.3), chaves
    ``classifier.{0,3,6}.*``. Trocar uma pela outra mudaria a arquitetura da cabeca, o
    numero de parametros e as chaves gravadas — todo .ckpt com ``head.*`` deixaria de
    carregar. Divergencia registrada no relatorio.
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
                            though the real widths are 24/30/48, which the ``flim`` package
                            then corrects.
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
                            grown-and-frozen one, and loaded (weights only) at the end of
                            ``__init__``.
        embed_mode:         Como o mapa da ULTIMA camada do encoder e reduzido para a sonda
                            SVM: ``"avgpool2d"`` (GAP, ``channels[-1]``-d) ou ``"flatten"``
                            (mapa achatado, ``C*H*W``-d). ``None`` (default) herda
                            ``src.utils.evaluate.EMBED_MODE`` — que e EXATAMENTE o que este
                            modulo faz hoje: base ``"avgpool2d"`` (evaluate.py:243), ou o
                            valor que o processo tiver rebindado. O laco de crescimento manda
                            ``"flatten"`` SEMPRE explicito (growth.py:228, spifil_growth_loop.py:426);
                            e escolha DAQUELE experimento, nao default de biblioteca. So
                            avaliacao: nao entra na loss nem no gradiente.
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
        embed_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # ponytail: divida — `embed_mode=None` cai no global `src.utils.evaluate.EMBED_MODE`,
        # que src/evaluate/eval_growth_stages.py:391 e o argparse da origem
        # (src/modules/autoencoder_flim_module.py:980) ainda rebindam. O parametro CONVIVE com
        # o rebind de proposito: sem a chave no YAML o comportamento e byte a byte o de hoje.
        # O global morre quando src/evaluate/ for migrado; so ai o default vira literal.
        if embed_mode is not None and embed_mode not in EMBED_MODES:
            raise ValueError(
                f"embed_mode invalido: {embed_mode!r}; use um de {list(EMBED_MODES)}, "
                f"ou None para herdar o default do avaliador (src.utils.evaluate.EMBED_MODE)."
            )

        if flim_weights_path is None:
            raise ValueError(
                "flim_weights_path is required — this experiment always initialises "
                "the encoder with FLIM weights (encoder_init=flim)."
            )

        # O `architecture.json` manda, inclusive para protozoan. O `PROTOZOAN_FLIM_ARCH`
        # existia para nao confiar no json de ch24_30_48, mas os dois so divergem em
        # `noutput_channels`/`nkernels_*` — campos que o proprio pacote `flim` sobrescreve
        # com o que os `conv{n}-bias.txt` dizem. Ler o json e portanto equivalente, e e o
        # que deixa uma arquitetura CRESCIDA (nlayers > 3, escrita por
        # `scripts/spifil_grow.py`) chegar ate aqui em vez de ser trocada por 3 camadas.
        #
        # MA-06: o parse do arch, o override de canais e o load dos pesos FLIM saem daqui e
        # entram em `AutoEncoderFLIM`, que os pega por `flim.build` — a unica fronteira do
        # pacote. A ORDEM e a mesma de antes (pesos FLIM dentro do encoder ANTES de
        # `_flim_ref` e ANTES de qualquer politica de congelamento), entao nada muda.
        self.model = AutoEncoderFLIM(
            arch_json=arch_json,
            init="flim",  # literal: este modulo ja levanta ValueError sem os pesos FLIM
            weights_path=flim_weights_path,
            in_channels=in_channels,
            out_size=(image_size, image_size),
        )
        # [3, 24, 32, 48] — os canais REAIS, lidos de volta do arch que `flim.build` ja
        # sobrescreveu com a contagem de kernels dos `conv{n}-bias.txt` (flim/arch.py:71-76).
        # Mesmo valor que `get_actual_channels_from_weights` devolvia aqui, sem furar a
        # fronteira do pacote `flim`; e a mesma conta que resnet_decoder.py:69-72 faz sobre
        # este mesmo arch. `channels` continua publico: o `run_metadata.json` e o
        # `log_hyperparams` do lancador leem `module.channels`.
        _arch = self.model.encoder.arch
        self.channels: List[int] = [in_channels] + [
            _arch[f"layer{n}"]["conv"]["noutput_channels"]
            for n in range(1, _arch["nlayers"] + 1)
        ]
        self.encoder_out_channels: int = self.channels[-1]
        channels = self.channels

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
            # the state_dict is laid on top afterwards (end of __init__) and
            # `load_state_dict` copies data in-place, it never touches `requires_grad`.
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
                freeze_encoder(self.model)  # core/blocks/init.py:44 — resolves .encoder on its own
        # AFTER the policy on purpose: as an ablation on a stage-3 round this re-freezes the
        # grown FLIM layer, leaving only the new decoder block learning. Order is the rule.
        if freeze_spifil_layer:
            # Last block only = the grafted SPiFiL layer. `Encoder.__init__` registers each
            # block both in `.blocks` and as `.conv{n}` (flim/encoder.py:90-91), and both
            # names point at the SAME Parameter objects, so freezing through one reaches both.
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
            # Logged only after the ablation, so the line is what configure_optimizers
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
        # is required for svm_probe_every > 1), it just runs to max_epochs in silence.
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

        self._load_init_ckpt(init_ckpt)

    # ── Backbone da rodada anterior ────────────────────────────────────────

    def _load_init_ckpt(self, init_ckpt: str) -> None:
        """Weights only, deliberately, e como ULTIMO passo do ``__init__``.

        MA-06: bloco movido do antigo ``main()`` (src/modules/autoencoder_flim_module.py:1033
        -1069), que rodava exatamente aqui — depois do ``__init__`` inteiro, sobre o modulo ja
        construido. Ele tinha de viajar junto com a flag: ``init_ckpt`` vira ``init_args`` no
        YAML, e sem este bloco o caminho no YAML so mudaria a politica de congelamento e
        deixaria de carregar peso nenhum — as rodadas 2 e 3 mudariam de numero em silencio.

        `trainer.fit(ckpt_path=...)` would also restore the epoch counter, the optimizer AND
        the scheduler — the new round would resume on the tail of the cosine at
        lr ~= eta_min=1e-5 and "converge" without having moved. It would also make the W&B
        logger resume the previous run, fusing two runs that must stay separate. And it could
        not work at all across a growth round: the model is a layer wider than the checkpoint.
        Building the module first is what rebuilds `_flim_ref` from the original FLIM kernels
        (the reference probe/flim_drift needs) and what leaves the newly grown layer holding
        the SPiFiL bank the `flim` loader just wrote — the old state_dict is only laid on top.
        """
        if not init_ckpt:
            return
        state = torch.load(init_ckpt, map_location="cpu").get("state_dict", {})
        # ResNetDecoder builds its blocks DEEPEST FIRST (resnet_decoder.py), so block 0 is
        # always the bottleneck's up-block: gaining a layer pushes every old index up by
        # `delta`. A name-based strict=False load would hit a shape mismatch on each of them
        # and silently throw the whole trained decoder away. `to_image` keeps its name — its
        # width is channels[1], which growth does not touch.
        pre = "model.decoder.blocks."
        delta = _decoder_block_delta(state, self.model.encoder.n_layers)
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
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        _log.info(
            "[stage %d] backbone from %s | decoder block shift delta=%+d | "
            "missing=%d (left at their FLIM/SPiFiL init) unexpected=%d | "
            "head keys dropped=%d (a Head sempre nasce nova sobre a camada atual)",
            self.stage, init_ckpt, delta, len(missing), len(unexpected), dropped_head,
        )

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

        Read by the launcher instead of being re-derived there, so the callbacks cannot end
        up watching a key that is never logged.

        Kappa is the only selection signal at every stage so that all growth rounds are
        comparable. The caveat, stated honestly: with a frozen encoder the embedding is
        identical every epoch, so kappa moves only with the SVM's own fit noise (0.0675 spread
        measured here across permutations of the same feature matrix). A frozen stage's kappa
        is therefore a measurement of the round's starting point, not a training curve, and
        which epoch wins is close to arbitrary.
        """
        # Com head_finetune quem seleciona e o kappa da Head: ali o encoder pode estar
        # congelado, mas a Head TREINA, entao o pico e sinal e nao ruido de fit. O nome do
        # arquivo continua `best_kappa` — o laco de crescimento monta o caminho por convencao.
        if self.hparams.head_finetune:
            return f"{self._stage_prefix}/head_kappa", "max", "best_kappa"
        return f"{self._stage_prefix}/svm_kappa", "max", "best_kappa"

    @property
    def epoch_metric_keys(self) -> Tuple[str, ...]:
        """Per-epoch keys this module logs, as they reach the logger.

        The launcher asserts its monitor is one of these. ``train_recon_loss`` is absent
        because Lightning splits it into ``_step``/``_epoch``, which makes it unusable as a
        monitor.
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
        views, y = batch  # y so entra na perda no braco head_finetune
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
        self._val_emb.append(
            _encode_pooled(self.model.encoder, x, self.hparams.embed_mode).float())
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
            embs.append(
                _encode_pooled(self.model.encoder, x, self.hparams.embed_mode).float())
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
        # ponytail: unica SVC fora de ``src/utils/evaluate.py:fit_svm`` — config identica, mas
        # fit_svm imprime a linha de diagnostico e sobe uma thread tqdm por segundo a cada
        # validacao; teto: unificar quando fit_svm tiver modo silencioso
        # (tools/check_refactor_equivalence.py trava o drift).
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

        # Antes do gate do svm_probe_every de proposito: com head_finetune este e O monitor,
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
