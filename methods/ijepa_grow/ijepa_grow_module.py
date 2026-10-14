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
"""Aluno SPiFiL/FLIM que prediz os tokens de um I-JEPA ViT-H/14 congelado.

Composicao, nao subclasse de ``AutoEncoderFlimModule``: o AE sempre constroi o decoder
(methods/autoencoder/autoencoder_flim.py), e aqui ele nao pode existir no state_dict. Do AE
fica o mesmo encoder (``flim.build``), o mesmo prefixo ``probe/`` e os atributos que
``RunMetadataCallback`` le (channels, monitor_spec, stage, ...).
"""

from __future__ import annotations

import math
from typing import Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from flim import build
from methods.ijepa_grow.context import ContextTokenizer, gather_tokens
from methods.ijepa_grow.predictor import ConvPredictor, Predictor

KEEP_PREFIXES = ("model.encoder.", "ctx_proj.", "predictor.")


class IJepaGrowModule(pl.LightningModule):
    """Encoder FLIM -> ctx_proj -> predictor, alvo = LayerNorm dos tokens do I-JEPA.

    ``old_layers``: camadas conv1..conv{old_layers} treinam com ``lr * old_layer_lr_scale``.
    Os init_args de decoder/head/sonda (fresh_decoder, head_finetune, svm_probe,
    freeze_encoder_flag, ...) sao aceitos so porque growth.py os passa; nao fazem nada.
    """

    def __init__(
        self,
        arch_json: str,
        flim_weights_path: str,
        dataset: str = "",
        num_classes: int = 9,
        in_channels: int = 3,
        old_layers: int = 0,
        old_layer_lr_scale: float = 0.1,
        ctx_dim: int = 384,
        predictor: str = "transformer",
        pred_depth: int = 6,
        pred_emb_dim: int = 384,
        pred_heads: int = 6,
        pred_mlp_ratio: float = 4.0,
        loss: str = "smooth_l1",
        lr: float = 1e-3 * 32 / 2048,
        final_lr: float = 1e-6,
        warmup_epochs: int = 5,
        weight_decay: float = 0.05,
        final_weight_decay: Optional[float] = None,
        max_epochs: int = 100,
        init_ckpt: str = "",
        embed_mode: Optional[str] = None,
        imagenet_norm: bool = False,
        freeze_encoder_flag: bool = False,
        fresh_decoder: bool = False,
        head_finetune: bool = False,
        svm_probe: bool = False,
    ) -> None:
        super().__init__()
        if predictor not in ("transformer", "conv"):
            raise ValueError(f"predictor invalido: {predictor!r}; use transformer ou conv")
        if loss not in ("smooth_l1", "mse"):
            raise ValueError(f"loss invalida: {loss!r}; use smooth_l1 ou mse")
        self.save_hyperparameters()
        self.model = nn.Module()
        self.model.encoder = build(arch_json, "flim", flim_weights_path, in_channels)
        arch = self.model.encoder.arch
        self.channels = [in_channels] + [arch[f"layer{n}"]["conv"]["noutput_channels"]
                                         for n in range(1, arch["nlayers"] + 1)]
        self.encoder_out_channels = self.channels[-1]
        self.ctx_proj = ContextTokenizer(self.encoder_out_channels, ctx_dim)
        self.predictor = (ConvPredictor(ctx_dim) if predictor == "conv"  # CNN-JEPA: 3 blocos fixos, ignora pred_*
                          else Predictor(ctx_dim, pred_emb_dim, pred_depth, pred_heads, pred_mlp_ratio))
        # Atributos que RunMetadataCallback le do AE; aqui nao ha sonda nem Head.
        self.head = None
        self.baseline_metrics = None
        self.last_probe_metrics = None
        self.stage_prefix = "probe"
        self.stage = 2
        object.__setattr__(self, "teacher", None)  # fora da arvore de modulos
        self._load_init_ckpt(init_ckpt)

    def _load_init_ckpt(self, init_ckpt: str) -> None:
        """So pesos: encoder (camadas que ja existiam), predictor, ctx_proj se o shape bater."""
        if not init_ckpt:
            return
        own = self.state_dict()
        state = torch.load(init_ckpt, map_location="cpu", weights_only=False).get("state_dict", {})
        state = {k: v for k, v in state.items()
                 if k.startswith(KEEP_PREFIXES) and k in own and own[k].shape == v.shape}
        self.load_state_dict(state, strict=False)

    @property
    def monitor_spec(self):
        return f"{self.stage_prefix}/val_loss", "min", "best_kappa"

    def verdict_summary(self) -> dict:
        return {}

    # ── Professor ──────────────────────────────────────────────────────────

    def setup(self, stage: str) -> None:
        if stage != "fit" or self.teacher is not None:
            return
        from methods.lejepa.ijepa_encoder import IJEPAEncoder  # noqa: PLC0415 — ViT so no fit
        teacher = IJEPAEncoder(device=self.trainer.strategy.root_device).to(torch.bfloat16).eval()
        teacher.requires_grad_(False)
        object.__setattr__(self, "teacher", teacher)
        self._teacher_sum = self._teacher_checksum()

    def _teacher_checksum(self) -> float:
        return sum(p.double().sum().item() for p in self.teacher.parameters())

    def on_fit_end(self) -> None:
        assert self._teacher_checksum() == self._teacher_sum, "pesos do teacher mudaram no fit"

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        bad = [k for k in checkpoint["state_dict"] if not k.startswith(KEEP_PREFIXES)]
        assert not bad, f"chaves fora de encoder/ctx_proj/predictor no ckpt: {bad[:5]}"

    # ── Passo ──────────────────────────────────────────────────────────────

    def _loss(self, batch: dict, split: str):
        ctx_idx, tgt_idx = batch["ctx_idx"], batch["tgt_idx"]
        tokens = self.ctx_proj(self.model.encoder(batch["student"]))
        preds = self.predictor(gather_tokens(tokens, ctx_idx), ctx_idx, tgt_idx)
        with torch.no_grad():
            h = self.teacher.patch_tokens(batch["teacher"]).float()
            h = F.layer_norm(h, (h.size(-1),))
            target = torch.stack([gather_tokens(h, t) for t in tgt_idx])
        fn = F.smooth_l1_loss if self.hparams.loss == "smooth_l1" else F.mse_loss
        loss = fn(preds.float(), target)
        bs = ctx_idx.size(0)
        self.log(f"{self.stage_prefix}/{split}_loss", loss, prog_bar=True, on_step=split == "train",
                 on_epoch=True, batch_size=bs)
        self.log(f"{self.stage_prefix}/{split}_fallback", batch["fallback"].float(),
                 on_step=False, on_epoch=True, reduce_fx="sum", batch_size=bs)
        return loss

    def training_step(self, batch: dict, batch_idx: int):
        self.log("train/lr", self.optimizers().param_groups[-1]["lr"], on_step=True, on_epoch=False)
        return self._loss(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int):
        return self._loss(batch, "val")

    # ── Otimizador ─────────────────────────────────────────────────────────

    def _progress(self) -> float:
        return min(1.0, self.global_step / max(1, self.trainer.estimated_stepping_batches))

    def on_train_batch_start(self, batch, batch_idx) -> None:
        hp = self.hparams
        if hp.final_weight_decay is None:
            return
        wd = hp.final_weight_decay + 0.5 * (hp.weight_decay - hp.final_weight_decay) * (
            1 + math.cos(math.pi * self._progress()))
        for group in self.optimizers().param_groups:
            group["weight_decay"] = wd

    def configure_optimizers(self):
        hp = self.hparams
        enc = self.model.encoder
        old = [p for n in range(1, hp.old_layers + 1) for p in getattr(enc, f"conv{n}").parameters()]
        old_ids = {id(p) for p in old}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in old_ids]
        groups = [{"params": rest, "lr": hp.lr}]
        if old:
            groups.insert(0, {"params": old, "lr": hp.lr * hp.old_layer_lr_scale})
        optimizer = AdamW(groups, lr=hp.lr, weight_decay=hp.weight_decay)
        total = max(1, self.trainer.estimated_stepping_batches)
        warmup = total * hp.warmup_epochs / max(1, self.trainer.max_epochs)
        floor = hp.final_lr / hp.lr

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            t = (step - warmup) / max(1, total - warmup)
            return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * min(1.0, t)))

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": LambdaLR(optimizer, factor), "interval": "step"}}
