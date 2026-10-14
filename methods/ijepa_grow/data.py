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

"""Dados do ijepa_grow: crop aleatorio do bbox do objeto + mascaras multibloco do I-JEPA.

O mesmo crop vai para o aluno (LAB [0,1], loader ift_lab) e para o professor (RGB
ImageNet-norm). Uma celula patch_size x patch_size do grid so entra em contexto ou
alvo se tiver no maximo max_background_leak de fundo (mascara == 0).
Sorteio dos blocos portado de facebookresearch/ijepa src/masks/multiblock.py.
"""

import math
import os

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resized_crop

from config import get_single_parasite_paths
from core.data.loaders import ift_lab_loader, pil_loader
from core.data.parasite_dataset import ParasiteDataset

# LABNorm2 do pixel preto (= fundo das imagens), medido com ift_lab_loader
LAB_BLACK = torch.tensor([0.0, 0.46726218, 0.5330766]).view(3, 1, 1)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _block_size(g, scale, aspect_ratio, grid):
    """Tamanho (h, w) do bloco, sorteado uma vez por batch como no I-JEPA."""
    r = torch.rand(1, generator=g).item()
    keep = int(grid * grid * (scale[0] + r * (scale[1] - scale[0])))
    ar = aspect_ratio[0] + r * (aspect_ratio[1] - aspect_ratio[0])
    h = min(max(int(round(math.sqrt(keep * ar))), 1), grid - 1)
    w = min(max(int(round(math.sqrt(keep / ar))), 1), grid - 1)
    return h, w


def _block(g, size, allowed, min_keep, tries=20):
    """Bloco h x w em posicao aleatoria, intersectado com allowed (grid, grid).

    Devolve (indices, 1 se caiu no fallback senao 0). Fallback = melhor tentativa;
    se nenhuma tocou celula permitida, todas as permitidas.
    """
    h, w = size
    grid = allowed.shape[0]
    best = None
    for _ in range(tries):
        top = torch.randint(0, grid - h, (1,), generator=g).item()
        left = torch.randint(0, grid - w, (1,), generator=g).item()
        m = torch.zeros_like(allowed)
        m[top:top + h, left:left + w] = True
        idx = torch.nonzero((m & allowed).flatten()).flatten()
        if len(idx) >= min_keep:
            return idx, 0
        if best is None or len(idx) > len(best):
            best = idx
    if len(best) == 0:
        best = torch.nonzero(allowed.flatten()).flatten()
    return best, 1


def sample_masks(eligible, generator, p_size=None, e_size=None, num_pred_masks=4, num_enc_masks=1,
                 pred_mask_scale=(0.15, 0.2), enc_mask_scale=(0.85, 1.0), aspect_ratio=(0.75, 1.5),
                 allow_overlap=False, min_keep=10):
    """eligible: BoolTensor(grid*grid) -> (ctx_idx, [tgt_idx] * num_pred_masks, n_fallback).

    Alvos primeiro; o contexto exclui as celulas-alvo quando allow_overlap e False.
    p_size/e_size vem do collate (um por batch); None sorteia aqui.
    """
    grid = math.isqrt(eligible.numel())
    elig = eligible.view(grid, grid)
    if p_size is None:
        p_size = _block_size(generator, pred_mask_scale, aspect_ratio, grid)
    if e_size is None:
        e_size = _block_size(generator, enc_mask_scale, (1.0, 1.0), grid)
    n_fb = 0
    tgts = []
    taken = torch.zeros_like(elig)
    for _ in range(num_pred_masks):
        idx, fb = _block(generator, p_size, elig, min_keep)
        n_fb += fb
        tgts.append(idx)
        taken.view(-1)[idx] = True
    allowed = elig if allow_overlap else elig & ~taken
    if not allowed.any():  # todo elegivel virou alvo: a unica saida e sobrepor
        allowed = elig
    ctx = []
    for _ in range(num_enc_masks):
        idx, fb = _block(generator, e_size, allowed, min_keep)
        n_fb += fb
        ctx.append(idx)
    return torch.cat(ctx).unique(), tgts, n_fb


def _pad(idx_list, n=None):
    """Completa cada lista repetindo os proprios indices ate n (default: maior tamanho do batch)."""
    n = n or max(len(x) for x in idx_list)
    return torch.stack([x[torch.arange(n) % len(x)] for x in idx_list])


class _CropDataset(Dataset):
    """Le imagem LAB + RGB + mascara e aplica o mesmo RandomResizedCrop do bbox do objeto."""

    def __init__(self, base, masks_dir, crop_size, crop_scale, patch_size, max_background_leak, seed, train):
        self.base, self.masks_dir = base, masks_dir
        self.crop_size, self.crop_scale, self.patch_size = crop_size, crop_scale, patch_size
        self.leak, self.seed, self.train = max_background_leak, seed, train

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _box(g, y0, x0, bh, bw, scale, ratio=(3 / 4, 4 / 3)):
        """RandomResizedCrop.get_params restrito ao bbox, com gerador proprio."""
        for _ in range(10):
            a = bh * bw * (scale[0] + torch.rand(1, generator=g).item() * (scale[1] - scale[0]))
            ar = math.exp(math.log(ratio[0]) + torch.rand(1, generator=g).item() * math.log(ratio[1] / ratio[0]))
            w, h = round(math.sqrt(a * ar)), round(math.sqrt(a / ar))
            if 0 < w <= bw and 0 < h <= bh:
                t = y0 + torch.randint(0, bh - h + 1, (1,), generator=g).item()
                l = x0 + torch.randint(0, bw - w + 1, (1,), generator=g).item()
                return t, l, h, w
        return y0, x0, bh, bw

    def crop(self, i):
        """(lab, rgb, mask, eligible) do crop i; deterministico na validacao (seed + i)."""
        path = self.base.samples[i]
        seed = torch.randint(2 ** 62, (1,)).item() if self.train else self.seed + i
        g = torch.Generator().manual_seed(seed)
        mask = pil_to_tensor(Image.open(os.path.join(self.masks_dir, os.path.basename(path))).convert("L")) > 0
        ys, xs = torch.nonzero(mask[0], as_tuple=True)
        y0, x0 = ys.min().item(), xs.min().item()
        bh, bw = ys.max().item() + 1 - y0, xs.max().item() + 1 - x0
        size = [self.crop_size, self.crop_size]
        for scale in (self.crop_scale, (1.0, 1.0)):  # sem celula elegivel: re-crop com o bbox inteiro
            box = self._box(g, y0, x0, bh, bw, scale)
            m = resized_crop(mask.float(), *box, size, InterpolationMode.NEAREST)
            frac = F.avg_pool2d((m == 0).float()[None], self.patch_size).flatten()
            eligible = frac <= self.leak
            if eligible.any():
                break
        else:  # nem o bbox inteiro tem celula limpa: fica com as menos sujas
            eligible = frac <= frac.min()
        lab = torch.from_numpy(np.ascontiguousarray(ift_lab_loader(path))).permute(2, 0, 1).float()
        rgb = pil_to_tensor(pil_loader(path)).float() / 255
        lab = resized_crop(lab, *box, size, antialias=True)
        rgb = resized_crop(rgb, *box, size, antialias=True)
        return lab, rgb, m, eligible

    def __getitem__(self, i):
        lab, rgb, _, eligible = self.crop(i)
        label = int(os.path.basename(self.base.samples[i]).split("_")[0]) - 1
        return lab, rgb, eligible, label, i


class _MaskCollator:
    """Sorteia os blocos (tamanho unico por batch) e monta o dict do contrato."""

    def __init__(self, dm, train):
        self.dm, self.train = dm, train

    def __call__(self, items):
        dm = self.dm
        lab, rgb, elig, y, idx = zip(*items)
        seed = torch.randint(2 ** 62, (1,)).item() if self.train else dm.seed + idx[0]
        g = torch.Generator().manual_seed(seed)
        grid = dm.crop_size // dm.patch_size
        p_size = _block_size(g, dm.pred_mask_scale, dm.aspect_ratio, grid)
        e_size = _block_size(g, dm.enc_mask_scale, (1.0, 1.0), grid)
        ctx, tgts, n_fb = [], [], 0
        for e in elig:
            c, t, fb = sample_masks(e, g, p_size, e_size, dm.num_pred_masks, dm.num_enc_masks,
                                    allow_overlap=dm.allow_overlap, min_keep=dm.min_keep)
            ctx.append(c)
            tgts.append(t)
            n_fb += fb
        student = torch.stack(lab)
        for b, c in enumerate(ctx):
            keep = torch.zeros(grid * grid, dtype=torch.bool)
            keep[c] = True
            keep = keep.view(grid, grid).repeat_interleave(dm.patch_size, 0).repeat_interleave(dm.patch_size, 1)
            student[b] = torch.where(keep, student[b], LAB_BLACK)
        return {
            "student": student,
            "teacher": normalize(torch.stack(rgb), IMAGENET_MEAN, IMAGENET_STD),
            "ctx_idx": _pad(ctx),
            "tgt_idx": torch.stack([_pad([t[k] for t in tgts], max(len(x) for t in tgts for x in t))
                                    for k in range(dm.num_pred_masks)]),
            "y": torch.tensor(y),
            "fallback": torch.tensor(n_fb),
        }


class IJepaCropDataModule(L.LightningDataModule):
    """Crop do bbox + mascaras I-JEPA. Aceita todas as chaves de configs/dataset/<ds>/split_<s>.yaml;
    loader, image_size, V_train, V_eval e use_mask sao ignoradas (o crop e sempre crop_size, LAB + RGB).
    """

    def __init__(self, *, parasite_name, split, percentage, batch_size=32, num_workers=8,
                 pin_memory=True, persistent_workers=True, loader="ift_lab", image_size=200,
                 V_train=12, V_eval=1, use_mask=False, crop_size=224, crop_scale=(0.3, 1.0),
                 patch_size=14, num_pred_masks=4, pred_mask_scale=(0.15, 0.2), aspect_ratio=(0.75, 1.5),
                 num_enc_masks=1, enc_mask_scale=(0.85, 1.0), allow_overlap=False, min_keep=10,
                 max_background_leak=0.1, seed=0):
        super().__init__()
        self.parasite_name, self.split, self.percentage = parasite_name, split, percentage
        self.batch_size, self.num_workers, self.pin_memory = batch_size, num_workers, pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.crop_size, self.crop_scale, self.patch_size = crop_size, tuple(crop_scale), patch_size
        self.num_pred_masks, self.pred_mask_scale = num_pred_masks, tuple(pred_mask_scale)
        self.aspect_ratio, self.num_enc_masks = tuple(aspect_ratio), num_enc_masks
        self.enc_mask_scale, self.allow_overlap = tuple(enc_mask_scale), allow_overlap
        self.min_keep, self.max_background_leak, self.seed = min_keep, max_background_leak, seed

    def _dataset(self, set_name):
        base = ParasiteDataset(set_name, self.split, self.percentage, path_dataset=self.parasite_name)
        masks_dir = get_single_parasite_paths(self.parasite_name, self.split, self.percentage)[0]["masks_dir"]
        return _CropDataset(base, masks_dir, self.crop_size, self.crop_scale, self.patch_size,
                            self.max_background_leak, self.seed, set_name == "train")

    def setup(self, stage=None):
        self.ds_train = self._dataset("train")
        self.ds_val = self._dataset("validation")

    def _loader(self, ds, train):
        return DataLoader(ds, batch_size=self.batch_size, shuffle=train, drop_last=train,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, collate_fn=_MaskCollator(self, train))

    def train_dataloader(self):
        return self._loader(self.ds_train, True)

    def val_dataloader(self):
        return self._loader(self.ds_val, False)
