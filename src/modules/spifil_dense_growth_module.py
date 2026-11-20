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
"""
Grow the dense network of ``configs/model/spifil_dense_growth.yaml`` one block at a time.

Per cell (dataset x split x percentage), for every block S, DB1, ..., H in order:

  1. **SPiFiL**: run the frozen network grown so far over the SPiFiL fit set, take one
     patch per candidate pixel with the new block's kernel, rank by Fisher score
     (Mahalanobis), allocate the filter budget per class, pick with the diversity-aware
     greedy, and fold the picks into conv weights (``spifil.nn.builder.build_filters``).
     A dense block sees its whole input — the concatenation so far — so the new filters
     are patches of it; the skip carries that input to the output untouched.
  2. **Freeze** everything that existed before (``freeze_encoder``); only the new conv is
     trainable. The concat has no weight, so the residual path is frozen by construction.
  3. **Fit** (steps in ``train_steps``): a fresh ``ResNetDecoder`` starting at the block's
     resolution and channels reconstructs the 200x200 LAB input (BCE with logits, no
     labels). Best epoch on ``val_recon_loss`` wins; the decoder is then dropped.
  4. **Check**: the frozen weights are bit-identical to before the fit, the dense block's
     first ``in_channels`` output channels are exactly its input, and the block's
     (channels, resolution) match ``expected_table``. Any failure raises.
  5. **Save** the encoder only: ``<run_dir>/stage{NN}_{NAME}/{encoder.pt,architecture.json,
     stage_metadata.json}`` — ``src.models.dense_growth.load_stage`` rebuilds it.

**Candidates.** k x k blocks use SPiFiL's own seeds: superpixel medoids inside the object
mask (layer 0), projected onto the block's input grid. 1 x 1 blocks (TL1, TL2, H) take
*every* grid cell the object mask covers — a 1 x 1 patch is a single pixel, and only pixels
of the object qualify, never background or padding. The fit set is ``images_per_class``
images per class drawn (seeded) from the cell's own training split, impurities excluded
as SPiFiL does; validation and test never enter it.

**Exact widths.** SPiFiL's ``UniformAllocator`` drops the remainder of
``out_channels // n_classes`` (protozoan: 32 -> 30). The table must be reproduced exactly,
so :class:`ExactAllocator` hands the remainder to the first classes, caps each class at what
it can supply, and moves any shortfall to the classes that have spare candidates.

A finished stage is never refitted: re-running the same command resumes after the last
saved stage (crash resume), and a cell with ``run_metadata.json`` status ``ok`` is skipped.

    python -m src.modules.spifil_dense_growth_module --config configs/model/spifil_dense_growth.yaml

runs the whole ``grid`` of the YAML (``run`` sets log dir, concurrency and CUDA allocator).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from skimage.io import imread
from torch import Tensor

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from spifil import DiversitySelector, FisherScorer, LayerSpec, Mahalanobis, Medoids, Seeds
from spifil.nn.builder import build_filters
from spifil.patches import extract_patches
from spifil.scoring import per_class_ranks
from spifil.selection import candidate_order
from spifil.types import PatchSet

from src.models.autoencoder_resnet import ResNetDecoder
from src.models.dense_growth import DenseGrowthNet, check_table, save_stage
from src.models.encoders import freeze_encoder
from src.transforms.lab_augment import LabAugment

_log = logging.getLogger(__name__)


# ─── SPiFiL: one block ────────────────────────────────────────────────────────


class ExactAllocator:
    """Exactly ``total`` filters over the classes, each capped by what it can supply.

    The cap mirrors ``DiversitySelector``'s pool cap (``len(patches) // n_classes``), so the
    selector never returns fewer picks than allocated.
    """

    def __call__(self, labels: Tensor, total: int) -> Dict[int, int]:
        classes, counts = torch.unique(labels, return_counts=True)
        classes, counts = classes.tolist(), counts.tolist()
        cap = {c: min(n, len(labels) // len(classes)) for c, n in zip(classes, counts)}
        if sum(cap.values()) < total:
            raise RuntimeError(
                f"{total} filters asked, but the candidates only supply {sum(cap.values())} "
                f"(per class {cap}); raise spifil.images_per_class for this dataset")
        base, rem = divmod(total, len(classes))
        alloc = {c: min(cap[c], base + (i < rem)) for i, c in enumerate(classes)}
        short = total - sum(alloc.values())
        while short:
            for c in classes:
                if short and alloc[c] < cap[c]:
                    alloc[c] += 1
                    short -= 1
        return alloc


class FitImage:
    """One SPiFiL fit image: the exact tensor the network sees, its mask, class, seeds."""

    def __init__(self, image: Tensor, mask: np.ndarray, label: int, seeds: np.ndarray) -> None:
        self.image, self.mask, self.label, self.seeds = image, mask, label, seeds


def build_fit_set(dm, dataset: str, masks_dir: str, sp: dict, seed: int) -> List[FitImage]:
    """``images_per_class`` training images per class (impurities out), with their seeds."""
    from spifil import DISF, SLIC

    superpixels = DISF() if sp["superpixels"] == "disf" else SLIC()
    paths = dm.ds_train.base.samples
    by_class: Dict[int, List[int]] = {}
    for i, p in enumerate(paths):
        by_class.setdefault(int(os.path.basename(p).split("_")[0]), []).append(i)

    rng = np.random.default_rng(seed)
    fit: List[FitImage] = []
    for label in sorted(by_class):
        if label in sp["exclude_classes"][dataset]:
            continue
        idx = rng.permutation(by_class[label])[: sp["images_per_class"][dataset]]
        for i in sorted(idx.tolist()):
            image = dm.ds_train[i][0][0]  # [3, H, W] LAB in [0, 1], as in training
            mask = imread(os.path.join(masks_dir, os.path.basename(paths[i])))
            mask = (mask[..., 0] if mask.ndim == 3 else mask).astype(np.int32)
            feats = image.numpy().astype(np.float32)
            regions = superpixels(feats, mask, sp["n_superpixels"])
            fit.append(FitImage(image, mask > 0, label, Medoids()(regions, feats)))
    return fit


@torch.no_grad()
def spifil_block(net: DenseGrowthNet, spec: dict, n_new: int, fit: List[FitImage],
                 sp: dict, device: torch.device, seed: int) -> tuple[Tensor, Tensor, dict]:
    """SPiFiL filters for ``net``'s last block, fitted on the output of the blocks before it."""
    k, s, upto = spec["kernel_size"], net.stride // spec["stride"], len(net.blocks) - 1
    layer = LayerSpec(kernel_size=k, out_channels=n_new)
    sets = []
    for image_id, im in enumerate(fit):
        feats = net(im.image.unsqueeze(0).to(device), upto=upto).squeeze(0)
        grid = tuple(feats.shape[1:])
        if k == 1:
            # Every grid cell the object covers; a cell counts if any object pixel is in it.
            m = torch.from_numpy(im.mask).float()[None, None]
            m = F.max_pool2d(m, s, s, ceil_mode=True)[0, 0] if s > 1 else m[0, 0]
            coords = m[: grid[0], : grid[1]].nonzero()
        else:
            coords = torch.unique(torch.from_numpy(im.seeds) // s, dim=0)
        if len(coords):
            seeds = Seeds.from_coords(coords, label=im.label, grid=grid)
            sets.append(extract_patches(feats, seeds, layer, image_id=image_id).to("cpu"))
    patches = PatchSet.cat(sets)
    n_all = len(patches)
    if n_all > sp["max_candidates"]:
        g = torch.Generator().manual_seed(seed)
        patches = patches.select(torch.randperm(n_all, generator=g)[: sp["max_candidates"]].sort().values)
    patches = patches.to(device)

    ranks = per_class_ranks(FisherScorer()(patches, Mahalanobis()), patches.labels)
    budget = ExactAllocator()(patches.labels, n_new)
    pick = DiversitySelector(alpha=sp["alpha"], pool_factor=sp["pool_factor"])(
        patches, ranks, budget).nonzero(as_tuple=True)[0]
    selected = patches.select(pick[candidate_order(patches.select(pick), ranks[pick])])
    if len(selected) != n_new:
        raise RuntimeError(f"{spec['name']}: SPiFiL picked {len(selected)} filters, needs {n_new}")
    bank = build_filters(selected, layer, stdev_factor=sp["stdev_factor"])
    info = {"candidates": n_all, "scored": len(patches), "filters_per_class": budget}
    return bank.weight, bank.bias, info


# ─── Training one block with a throwaway decoder ──────────────────────────────


class DenseStepModule(pl.LightningModule):
    """Frozen prefix + one trainable block + a fresh decoder, trained by reconstruction."""

    def __init__(self, net: DenseGrowthNet, decoder: nn.Module, lr: float,
                 weight_decay: float, max_epochs: int, stage: str = "",
                 augment=None) -> None:
        super().__init__()
        self.stage = stage
        self.augment = augment  # LabAugment or None; training_step only
        self.net, self.decoder = net, decoder
        self.lr, self.weight_decay, self.max_epochs = lr, weight_decay, max_epochs
        self.block = net.blocks[-1]
        self.loss = nn.BCEWithLogitsLoss()
        self.best_loss, self.best_epoch, self.history = float("inf"), -1, []
        self.best_block = copy.deepcopy(self.block.state_dict())
        self._val: List[Tensor] = []
        self._train: List[Tensor] = []

    @staticmethod
    def _x(batch) -> Tensor:
        views, _ = batch  # labels discarded: training is unsupervised
        return views[:, 0] if views.dim() == 5 else views

    def _recon(self, x: Tensor) -> Tensor:
        # ift_lab gives LAB in [0, 1] and imagenet_norm is off: the input is the target.
        return self.loss(self.decoder(self.net(x)), x.clamp(0.0, 1.0))

    def training_step(self, batch, batch_idx):
        # Augmentation only where there is backprop: SPiFiL built this block's filters from
        # the original images, and validation/SVM see originals too. The augmented image is
        # input and target at once, so geometry and L changes stay consistent.
        x = self._x(batch)
        if self.augment is not None:
            x = self.augment(x)
        loss = self._recon(x)
        self._train.append(loss.detach())
        self.log("train_recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._recon(self._x(batch))
        self._val.append(loss.detach())
        self.log("val_recon_loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            self._val.clear()
            return
        loss = float(torch.stack(self._val).mean())
        train = float(torch.stack(self._train).mean()) if self._train else float("nan")
        self._val.clear()
        self._train.clear()
        self.history.append(loss)
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, self.current_epoch
            self.best_block = copy.deepcopy(self.block.state_dict())
        if wandb.run is not None:
            # Own x axis per stage: every stage's curve starts at epoch 0.
            wandb.log({f"{self.stage}/epoch": self.current_epoch,
                       f"{self.stage}/val_recon_loss": loss,
                       f"{self.stage}/train_recon_loss": train,
                       f"{self.stage}/best_val_recon_loss": self.best_loss,
                       f"{self.stage}/lr": self.trainer.optimizers[0].param_groups[0]["lr"]})

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs, eta_min=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# ─── The growth loop ──────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=_ROOT, text=True).strip()
    except Exception:
        return ""


def _stage_done(stage_dir: str) -> bool:
    return os.path.isfile(os.path.join(stage_dir, "encoder.pt"))


def grow(cfg: dict, dataset: str, split: int, run_dir: str, device: torch.device) -> List[dict]:
    from config import get_single_parasite_paths
    from src.data_modules.parasite_data_module_lejepa_splited import (
        ParasiteLejepaDataModuleSplited,
    )
    from src.modules.autoencoder_flim_module import _dataset_short_to_parasite_name

    tr, sp, pct = cfg["train"], cfg["spifil"], cfg["grid"]["percentage"]
    parasite = _dataset_short_to_parasite_name(dataset)
    dm = ParasiteLejepaDataModuleSplited(
        parasite_name=parasite, split=split, percentage=pct,
        image_size=tr["image_size"], V_train=1, V_eval=1,
        batch_size=tr["batch_size"], num_workers=tr["num_workers"],
        pin_memory=True, persistent_workers=tr["num_workers"] > 0,
        loader="ift_lab", imagenet_norm=False,
    )
    dm.setup()
    masks_dir = get_single_parasite_paths(parasite, split, pct)[0]["masks_dir"]
    fit = None  # built lazily: a fully resumed run never needs it
    aug_cfg = cfg.get("augment") or {}
    augment = LabAugment(aug_cfg) if aug_cfg.get("enabled") else None

    net = DenseGrowthNet(in_channels=3).to(device)
    summary: List[dict] = []
    for i, spec in enumerate(cfg["architecture"]):
        name = spec["name"]
        stage_dir = os.path.join(run_dir, f"stage{i:02d}_{name}")
        freeze_encoder(net)                     # everything grown so far
        block = net.add_block(spec).to(device)  # the only trainable parameters
        n_new = block.conv.out_channels

        if _stage_done(stage_dir):
            net.load_state_dict(torch.load(os.path.join(stage_dir, "encoder.pt"), map_location=device))
            _log.info("[%s] resumed from %s", name, stage_dir)
            summary.append({"stage": name, "resumed": True})
            continue

        t0 = time.time()
        if fit is None:
            fit = build_fit_set(dm, dataset, masks_dir, sp, tr["seed"])
            _log.info("SPiFiL fit set: %d images", len(fit))
        weight, bias, info = spifil_block(net, spec, n_new, fit, sp, device, tr["seed"])
        with torch.no_grad():
            block.conv.weight.copy_(weight)
            block.conv.bias.copy_(bias)
        _log.info("[%s] SPiFiL: %d filters from %d candidates %s", name, n_new,
                  info["candidates"], info["filters_per_class"])

        frozen = {k: v.clone() for k, v in net.state_dict().items()
                  if not k.startswith(f"blocks.{i}.")}
        train_info: dict = {"trained": False}
        if name in cfg["train_steps"]:
            size = (tr["image_size"] - 1) // net.stride + 1
            decoder = ResNetDecoder.from_resolution(
                block.out_channels, size, out_size=(tr["image_size"], tr["image_size"]))
            module = DenseStepModule(net, decoder, tr["lr"], tr["weight_decay"], tr["max_epochs"],
                                     stage=name, augment=augment)
            if wandb.run is not None:
                wandb.define_metric(f"{name}/epoch")
                wandb.define_metric(f"{name}/*", step_metric=f"{name}/epoch")
            trainer = pl.Trainer(
                max_epochs=tr["max_epochs"], accelerator="gpu" if device.type == "cuda" else "cpu",
                devices=1, precision=tr.get("precision", "32-true"),
                callbacks=[EarlyStopping("val_recon_loss", patience=tr["patience"], mode="min")],
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(module, datamodule=dm)
            block.load_state_dict(module.best_block)
            net.to(device)
            train_info = {"trained": True, "best_val_recon_loss": module.best_loss,
                          "best_epoch": module.best_epoch, "epochs_run": len(module.history),
                          "val_recon_loss_history": module.history,
                          "decoder_params": sum(p.numel() for p in decoder.parameters())}
            del module, decoder, trainer  # the decoder is never saved
            torch.cuda.empty_cache()

        # ── checks: frozen untouched, residual intact, table respected ──
        for k, v in net.state_dict().items():
            if k in frozen and not torch.equal(v, frozen[k]):
                raise AssertionError(f"[{name}] frozen weight {k} changed during training")
        x = torch.stack([im.image for im in fit[:4]]).to(device)
        with torch.no_grad():
            before = net(x, upto=i)
            out = net(x)
        if block.concat and not torch.equal(out[:, : block.in_channels], before):
            raise AssertionError(f"[{name}] dense block did not carry its input through the skip")
        check_table(net, {s["name"]: cfg["expected_table"][s["name"]] for s in cfg["architecture"][: i + 1]},
                    tr["image_size"])

        meta = {"stage": name, "stage_index": i, "dataset": dataset, "split": split,
                "percentage": pct, "spec": spec, "spifil": info, **train_info,
                "elapsed_s": time.time() - t0, "git_sha": _git_sha()}
        save_stage(net, stage_dir, tr["image_size"], meta)
        if wandb.run is not None:
            wandb.run.summary.update({
                f"{name}/best_val_recon_loss": train_info.get("best_val_recon_loss"),
                f"{name}/best_epoch": train_info.get("best_epoch"),
                f"{name}/epochs_run": train_info.get("epochs_run"),
                f"{name}/out_channels": block.out_channels,
                f"{name}/spifil_candidates": info["candidates"],
                f"{name}/filters_per_class": {str(c): n for c, n in info["filters_per_class"].items()},
                f"{name}/elapsed_s": meta["elapsed_s"],
                "stages_saved": i + 1})
        summary.append({k: meta[k] for k in ("stage", "elapsed_s")} |
                       {"best_val_recon_loss": train_info.get("best_val_recon_loss")})
        _log.info("[%s] saved -> %s (%.0fs)", name, stage_dir, meta["elapsed_s"])
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default=os.path.join(_ROOT, "configs/model/spifil_dense_growth.yaml"))
    # Set by the grid runner for each child; everything else comes from the YAML.
    p.add_argument("--dataset", choices=["eggs", "larvae", "protozoan"], help=argparse.SUPPRESS)
    p.add_argument("--split", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dataset is None:
        return run_grid(args.config, cfg)
    return run_cell(cfg, args.dataset, args.split)


def run_grid(config_path: str, cfg: dict) -> int:
    """Every ``grid`` cell as a child process, ``run.concurrency`` at a time, one log per cell."""
    from concurrent.futures import ThreadPoolExecutor

    run = cfg["run"]
    log_dir = os.path.join(_ROOT, run["log_dir"])
    os.makedirs(log_dir, exist_ok=True)
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": run["cuda_alloc_conf"]}
    cells = [(d, s) for d in cfg["grid"]["datasets"] for s in cfg["grid"]["splits"]]
    if cfg["wandb"]["enabled"]:
        if not wandb.api.api_key:
            _log.error("wandb.enabled is true but this machine is not logged in: run `wandb login`")
            return 1

    def child(cell) -> int:
        d, s = cell
        log = os.path.join(log_dir, f"{d}_split{s}.log")
        _log.info("start %s split%d -> %s", d, s, log)
        with open(log, "a") as fh:
            rc = subprocess.run([sys.executable, "-m", "src.modules.spifil_dense_growth_module",
                                 "--config", config_path, "--dataset", d, "--split", str(s)],
                                cwd=_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        _log.info("done  %s split%d rc=%d", d, s, rc)
        return rc

    with ThreadPoolExecutor(max_workers=run["concurrency"]) as pool:
        codes = list(pool.map(child, cells))
    failed = [c for c, rc in zip(cells, codes) if rc]
    _log.info("grid finished: %d/%d ok%s", len(cells) - len(failed), len(cells),
              f", failed {failed}" if failed else "")
    return 1 if failed else 0


def cell_name(cfg: dict, dataset: str, split: int, suffix: str | None = None) -> str:
    """``spifil_dense<name_suffix>_<ds>_split<s>_pct<p>`` — shared with the evaluator."""
    suffix = cfg.get("name_suffix", "") if suffix is None else suffix
    return f"spifil_dense{suffix}_{dataset}_split{split}_pct{cfg['grid']['percentage']}"


def seed_from(cfg: dict, dataset: str, split: int, run_dir: str) -> None:
    """``init_from``: copy a finished run's stages up to ``upto`` into this run.

    The growth loop then resumes them (frozen, never refitted) and grows only what comes
    after. Each copied stage must describe exactly the blocks this config asks for there.
    """
    src_cfg = cfg["init_from"]
    src = os.path.join(_ROOT, src_cfg["output_root"],
                       cell_name(cfg, dataset, split, src_cfg.get("name_suffix", "")))
    names = [s["name"] for s in cfg["architecture"]]
    for i, name in enumerate(names[: names.index(src_cfg["upto"]) + 1]):
        stage = f"stage{i:02d}_{name}"
        dst = os.path.join(run_dir, stage)
        if os.path.isdir(dst):
            continue
        with open(os.path.join(src, stage, "architecture.json")) as f:
            got = json.load(f)["layers"][i]
        want = cfg["architecture"][i]
        if any(got.get(k) != v for k, v in want.items()):
            raise ValueError(f"init_from {src}/{stage} is {got}, config asks for {want}")
        shutil.copytree(os.path.join(src, stage), dst)
        _log.info("[%s] copied from %s", name, src)


def run_cell(cfg: dict, dataset: str, split: int) -> int:
    pct = cfg["grid"]["percentage"]
    run_name = cell_name(cfg, dataset, split)
    run_dir = os.path.join(_ROOT, cfg["output_root"], run_name)
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if os.path.isfile(meta_path) and json.load(open(meta_path)).get("status") == "ok":
        _log.info("%s already finished — skipped", run_name)
        return 0
    os.makedirs(run_dir, exist_ok=True)
    if cfg.get("init_from"):
        seed_from(cfg, dataset, split, run_dir)

    pl.seed_everything(cfg["train"]["seed"], workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = {"run_name": run_name, "method": "spifil_dense_growth", "dataset": dataset,
            "split": split, "percentage": pct, "config": cfg, "git_sha": _git_sha()}
    wb = cfg["wandb"]
    if wb["enabled"]:
        # id = run name + config hash: relaunching the same config resumes its run, while a
        # changed config (epochs, patience, SPiFiL, ...) opens a new one instead of appending.
        cfg_hash = hashlib.sha1(json.dumps(
            {k: cfg[k] for k in ("architecture", "spifil", "train", "grid", "augment") if k in cfg},
            sort_keys=True).encode()).hexdigest()[:8]
        wandb.init(project=wb["project"], entity=wb["entity"], name=run_name,
                   id=f"{run_name}-{cfg_hash}",
                   resume="allow", dir=run_dir, tags=wb["tags"],
                   config={"dataset": dataset, "split": split, "percentage": pct, **cfg})
    t0 = time.time()
    try:
        meta["stages"] = grow(cfg, dataset, split, run_dir, device)
        meta["status"] = "ok"
    except Exception as exc:
        _log.error("growth failed: %s", exc, exc_info=True)
        meta.update(status="error", error=str(exc))
    meta["elapsed_s"] = time.time() - t0
    if torch.cuda.is_available():
        meta["peak_gpu_mem_mib"] = torch.cuda.max_memory_allocated() / 2**20
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if wb["enabled"]:
        wandb.finish(exit_code=0 if meta["status"] == "ok" else 1)
    return 0 if meta["status"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
