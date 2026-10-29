"""Progressive extension of the FLIM autoencoder, with spatial 1280-channel heads.

Encoder blocks are supplied by ``flim.progressive.fit_new_block``. This module
never initializes their weights. Checkpoints contain Python modules and must
only be loaded from trusted experiment files (``weights_only=False``).
"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.blocks.init import INIT_FNS
from methods.autoencoder.resnet_decoder import ResNetDecoder


def auxiliary_init(module: nn.Module, policy: str = "he") -> None:
    """Use the existing central policy, exclusively on a newly created auxiliary."""
    if policy not in {"he", "xavier"}:
        raise ValueError("convolutional auxiliary_init must be he or xavier")
    INIT_FNS[policy](module)
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            if child.weight is not None:
                nn.init.ones_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


class EvolutionaryCNN(nn.Module):
    embedding_dim = 1280

    def __init__(self, config: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        if int(config.get("embedding_dim", 1280)) != 1280:
            raise ValueError("progressive embeddings must have exactly 1280 dimensions")
        self.encoder = nn.Sequential()
        self.projections = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.effective_specs: list[dict] = []
        self.provenance: dict = {}
        self.phase = "A"
        self._frozen_blocks: set[int] = set()

    @property
    def depth(self):
        return len(self.effective_specs)

    def add_block(self, block: nn.Module, effective_spec: dict):
        if self.depth >= min(10, int(self.config.get("max_depth", 10))):
            raise ValueError("maximum evolutionary depth reached")
        spec = copy.deepcopy(effective_spec)
        if spec.get("connection", "sequential") != "sequential":
            raise ValueError("only validated sequential SPiFiL blocks are enabled")
        channels = int(spec.get("actual_channels", spec["channels"]))
        if channels <= 0:
            raise ValueError("actual channels must be positive")
        if hasattr(block, "out_channels") and block.out_channels != channels:
            raise ValueError("effective channels disagree with fitted block")
        spec["channels"] = channels
        self.encoder.add_module(str(self.depth), block)
        self.effective_specs.append(spec)
        projection = nn.Conv2d(channels, 1280, 1)
        arch = {"nlayers": self.depth}
        for i, item in enumerate(self.effective_specs, 1):
            arch[f"layer{i}"] = {"conv": {"noutput_channels": item["channels"]},
                                    "pooling": {"stride": item.get("pool_stride", 1)}}
        size = int(self.config.get("image_size", 200))
        decoder = nn.Sequential(nn.Conv2d(1280, channels, 1),
                                ResNetDecoder(arch, int(self.config.get("in_channels", 3)), (size, size)))
        policy = self.config.get("auxiliary_init", "he")
        auxiliary_init(projection, policy)
        auxiliary_init(decoder, policy)
        self.projections.append(projection)
        self.decoders.append(decoder)
        self.provenance[f"projection{self.depth}"] = {"source": "auxiliary", "policy": policy}
        self.provenance[f"decoder{self.depth}"] = {"source": "auxiliary", "policy": policy}
        return self

    def _depth(self, depth):
        depth = self.depth if depth is None else int(depth)
        if not 1 <= depth <= self.depth:
            raise ValueError("requested depth is not present")
        return depth

    def project(self, x, depth=None):
        depth = self._depth(depth)
        for block in list(self.encoder.children())[:depth]:
            x = block(x)
        return self.projections[depth - 1](x)

    def embed(self, x, depth=None):
        return F.adaptive_avg_pool2d(self.project(x, depth), 1).flatten(1)

    def forward(self, x, depth=None):
        depth = self._depth(depth)
        return self.decoders[depth - 1](self.project(x, depth))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for index in self._frozen_blocks:
                self.encoder[index].eval()
            for head in list(self.projections)[:-1] + list(self.decoders)[:-1]:
                head.eval()
        return self

    def inference_model(self, depth=None):
        """Independent CPU copy containing only the prefix, current P_i and GAP."""
        depth = self._depth(depth)
        return nn.Sequential(copy.deepcopy(self.encoder[:depth]).cpu(),
                             copy.deepcopy(self.projections[depth - 1]).cpu(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(1)).eval()


def _image(batch):
    if isinstance(batch, dict):
        batch = batch.get("image", batch.get("images"))
    elif isinstance(batch, (tuple, list)):
        batch = batch[0]
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        raise TypeError("loader must provide image tensors")
    return batch


class ProgressiveTrainer:
    """One persistent AdamW per individual; old states survive both growth phases.

    ``load_checkpoint`` is an instance method and replaces ``self.model``. Access
    the restored model through ``trainer.model`` after loading.
    """
    def __init__(self, model: EvolutionaryCNN, config: dict):
        self.config = copy.deepcopy(config)
        requested = config.get("device", "auto")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
        self.model = model.to(self.device)
        self.optimizer = None
        self.phase = "A"
        self.epoch = 0
        self.sync_optimizer()
        if model.depth:
            self.set_phase("A")

    def sync_optimizer(self):
        self.model.to(self.device)
        known = set() if self.optimizer is None else {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        new = [p for p in self.model.parameters() if id(p) not in known]
        if new:
            options = {"lr": float(self.config.get("lr", 1e-3)), "weight_decay": float(self.config.get("weight_decay", 0.0))}
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(new, **options)
            else:
                self.optimizer.add_param_group({"params": new, **options})

    def set_phase(self, phase):
        if phase not in {"A", "B"}:
            raise ValueError("phase must be A or B")
        self.sync_optimizer()
        self.phase = self.model.phase = phase
        frozen = set(range(max(0, self.model.depth - 1))) if phase == "A" else set()
        self.model._frozen_blocks = frozen
        for index, block in enumerate(self.model.encoder):
            block.requires_grad_(index not in frozen)
        for index, (projection, decoder) in enumerate(zip(self.model.projections, self.model.decoders)):
            active = index == self.model.depth - 1
            projection.requires_grad_(active)
            decoder.requires_grad_(active)
        lr = float(self.config.get("refine_lr", self.config.get("lr", 1e-3) * .1)) if phase == "B" else float(self.config.get("lr", 1e-3))
        if self.optimizer:
            self.optimizer.zero_grad(set_to_none=True)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        self.model.train()

    def train_epoch(self, loader):
        if not self.model.depth:
            raise ValueError("fit a SPiFiL block before training")
        self.model.train()
        total, count = 0.0, 0
        for batch in loader:
            x = _image(batch).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            reconstruction = self.model(x)
            loss = F.l1_loss(reconstruction, x)
            structural_weight = float(self.config.get("ssim_weight", 0.0))
            if structural_weight:
                # Local SSIM with documented 3x3 uniform windows, data range 1.
                mx, my = F.avg_pool2d(x, 3, 1, 1), F.avg_pool2d(reconstruction, 3, 1, 1)
                vx = F.avg_pool2d(x * x, 3, 1, 1) - mx.square()
                vy = F.avg_pool2d(reconstruction.square(), 3, 1, 1) - my.square()
                cov = F.avg_pool2d(x * reconstruction, 3, 1, 1) - mx * my
                ssim = ((2 * mx * my + .01**2) * (2 * cov + .03**2) /
                        ((mx.square() + my.square() + .01**2) * (vx + vy + .03**2)))
                loss = loss + structural_weight * (1 - ssim.mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite reconstruction loss")
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach()) * len(x)
            count += len(x)
        if not count:
            raise ValueError("empty training loader")
        self.epoch += 1
        return {"reconstruction_loss": total / count, "epoch": self.epoch, "phase": self.phase}

    def save_checkpoint(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = {id(p): name for name, p in self.model.named_parameters()}
        groups = [[names[id(p)] for p in group["params"]] for group in self.optimizer.param_groups] if self.optimizer else []
        state = {"format_version": 1, "model": copy.deepcopy(self.model).cpu(),
                 "config": self.config, "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                 "optimizer_groups": groups, "phase": self.phase, "epoch": self.epoch,
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                 "metadata": metadata}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("format_version") != 1:
            raise ValueError("unsupported progressive checkpoint format")
        self.model = state["model"].to(self.device)
        self.config = state["config"]
        named = dict(self.model.named_parameters())
        groups = [{"params": [named[name] for name in names]} for names in state["optimizer_groups"]]
        self.optimizer = torch.optim.AdamW(groups, lr=float(self.config.get("lr", 1e-3))) if groups else None
        if self.optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
        self.phase, self.epoch = state["phase"], state["epoch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self.model.train()
        return state["metadata"]
