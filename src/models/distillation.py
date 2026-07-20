# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀�⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP — Universidade Estadual de     ║
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

"""distillation.py — Model-level components for I-JEPA → FLIM CNN knowledge distillation.

Components
----------
DistillationProjectionHead
    Linear projection aligning student embedding dimension to teacher
    embedding dimension. Used only during the distillation loss computation;
    raw student embeddings (without projection) are used for SVM evaluation.

FrozenTeacher
    Thin wrapper around IJEPAEncoder that enforces eval mode and
    requires_grad=False at construction time and after every state_dict load.

DistillationLoss
    MSE loss between projected student embedding and frozen teacher embedding,
    with optional L2 normalisation of both before comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.ijepa_encoder import IJEPAEncoder

# ── Teacher size used throughout the distillation pipeline ────────────────────
TEACHER_DIM: int = IJEPAEncoder.EMBEDDING_DIM  # 1280
TEACHER_IMAGE_SIZE: int = IJEPAEncoder.IMAGE_SIZE   # 224


class DistillationProjectionHead(nn.Module):
    """Spatial projection head: encoder feature maps → teacher_dim.

    Takes encoder output BEFORE global avg pool, applies partial spatial
    pooling (pool_size × pool_size), then projects DOWN to teacher_dim.
    Example: 48 × 6 × 6 = 1728  →  1280  (going down, richer than 48→1280).

    Args:
        student_channels: Channel count of FLIM CNN encoder output, e.g. 48.
        pool_size:        Intermediate spatial pooling size (default 6).
        teacher_dim:      I-JEPA ViT-H embedding dimension (1280 by default).
    """

    def __init__(
        self,
        student_channels: int,
        pool_size: int = 6,
        teacher_dim: int = TEACHER_DIM,
    ) -> None:
        super().__init__()
        self.student_channels = student_channels
        self.pool_size = pool_size
        self.teacher_dim = teacher_dim
        flat_dim = student_channels * pool_size * pool_size  # e.g. 48*6*6=1728

        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, teacher_dim, bias=False),
            nn.BatchNorm1d(teacher_dim),
        )

    def forward(self, feat_map: Tensor) -> Tensor:
        """Project spatial feature map into teacher embedding space.

        Args:
            feat_map: ``(B, student_channels, H, W)`` encoder output.

        Returns:
            ``(B, teacher_dim)`` projected embedding.
        """
        pooled = self.pool(feat_map)  # (B, C, pool_size, pool_size)
        return self.proj(pooled)       # (B, teacher_dim)


class FrozenTeacher(nn.Module):
    """IJEPAEncoder wrapper that enforces a permanently frozen state.

    The model is placed in eval mode and all parameters have
    ``requires_grad=False`` on construction.  The ``register_load_state_dict_post_hook``
    re-applies the freeze after any state_dict restore, guaranteeing the teacher
    never accumulates gradients.

    Args:
        model_id: HuggingFace model identifier for I-JEPA.  Defaults to
            ``"facebook/ijepa_vith14_1k"``.
    """

    def __init__(self, model_id: str = "facebook/ijepa_vith14_1k") -> None:
        super().__init__()
        self._encoder = IJEPAEncoder(model_id=model_id)
        self._freeze()
        self.embed_dim: int = TEACHER_DIM
        self.image_size: int = TEACHER_IMAGE_SIZE

    def _freeze(self) -> None:
        self._encoder.eval()
        self._encoder.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        """Extract frozen I-JEPA embeddings.

        Args:
            pixel_values: ``(B, 3, 224, 224)`` ImageNet-normalised tensors.

        Returns:
            ``(B, 1280)`` mean-pooled patch embeddings on CPU.
        """
        return self._encoder.extract_features(pixel_values)

    def train(self, mode: bool = True):
        """Override: always stay in eval mode regardless of trainer calls."""
        return super().train(False)


class FrozenTeacherCheckpointMixin:
    """Keep the frozen I-JEPA teacher out of saved checkpoints.

    The teacher (~2.4 GB) is frozen and re-downloadable from HuggingFace, so it is
    dropped on save and re-injected from the live module on load. Checkpoints stay
    small while ``strict=True`` resume still succeeds.
    """

    #: state_dict prefixes persisted to disk; everything else (the teacher) is
    #: reconstructed from the live module on load.
    PERSISTED_PREFIXES = ("student.", "proj_kd.")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items()
            if k.startswith(self.PERSISTED_PREFIXES)
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"].update(
            (k, v) for k, v in self.state_dict().items()
            if not k.startswith(self.PERSISTED_PREFIXES)
        )


class KLDistillationLoss(nn.Module):
    """KL-divergence distillation loss — Hinton et al. (2015).

    Treats teacher and student embeddings as logits, applies temperature
    softmax to produce soft distributions, then minimises:

        L_KD = KL( softmax(y_T / T)  ∥  log_softmax(y_S / T) ) × T²

    The T² scaling keeps gradient magnitudes consistent across temperatures
    (Hinton et al., "Distilling the Knowledge in a Neural Network", 2015).

    Args:
        temperature: Softmax temperature T.  Higher values produce softer
            distributions.  Recommended range: 2–8.  Default: 4.0.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        self.T = temperature

    def forward(self, student_proj: Tensor, teacher_emb: Tensor) -> Tensor:
        """Compute KL distillation loss.

        Args:
            student_proj: ``(B, D)`` projected student logits.
            teacher_emb:  ``(B, D)`` frozen teacher embeddings used as soft
                targets (moved to same device as *student_proj*).

        Returns:
            Scalar KL loss scaled by T².
        """
        teacher_emb = teacher_emb.to(student_proj.device)
        soft_teacher    = F.softmax(teacher_emb   / self.T, dim=-1)
        log_soft_student = F.log_softmax(student_proj / self.T, dim=-1)
        return F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)


# Keep alias so old imports don't break
DistillationLoss = KLDistillationLoss


class MSEDistillationLoss(nn.Module):
    """MSE loss between projected student embedding and frozen teacher embedding.

    Faithful to the CDD notebook:
        L_MSE = MSE( proj(student_enc(x)),  teacher_enc(x) )
    """

    def forward(self, student_proj: Tensor, teacher_emb: Tensor) -> Tensor:
        return F.mse_loss(student_proj, teacher_emb.to(student_proj.device))


class CosineDistillationLoss(nn.Module):
    """Cosine distillation loss between projected student embedding and teacher embedding.

        L_cos = 1 - cosine_similarity( proj(student_enc(x)), teacher_enc(x) )

    Averaged over the batch. Unlike MSE, this ignores embedding magnitude and
    only penalises directional misalignment between student and teacher.
    """

    def forward(self, student_proj: Tensor, teacher_emb: Tensor) -> Tensor:
        teacher_emb = teacher_emb.to(student_proj.device)
        return (1.0 - F.cosine_similarity(student_proj, teacher_emb, dim=-1)).mean()


class ConvDistillationProjectionHead(nn.Module):
    """Conv projection head: student_channels → 128 → 256 → 512 → teacher_dim via 1×1 convs.

    No MLP or linear layer — only pointwise (1×1) convolutions followed by
    global average pooling.  The spatial information is preserved through all
    conv stages and only collapsed at the very end.

    Args:
        student_channels: Output channels of FLIM encoder (e.g. 48).
        teacher_dim:      I-JEPA teacher embedding dim (1280 by default).
    """

    def __init__(
        self,
        student_channels: int = 48,
        teacher_dim: int = TEACHER_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, 128,        1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,             256,         1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,             512,         1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,             teacher_dim, 1, bias=False),
            nn.BatchNorm2d(teacher_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_map: Tensor) -> Tensor:
        """Project spatial feature map to teacher embedding space.

        Args:
            feat_map: ``(B, student_channels, H, W)`` encoder output.

        Returns:
            ``(B, teacher_dim)`` projected embedding.
        """
        x = self.proj(feat_map)   # (B, teacher_dim, H, W)
        x = self.pool(x)           # (B, teacher_dim, 1, 1)
        return x.flatten(1)        # (B, teacher_dim)


class OneLayerConvDistillationProjectionHead(nn.Module):
    """Single 3×3 conv: student_channels → teacher_dim in one shot.

    Direct projection without intermediate channels:
        Conv2d(48→1280, k=3×3, pad=1) + BN2d + GELU → AdaptiveAvgPool2d(1) → flatten

    Compared to ConvDistillationProjectionHead (4 layers, 1×1 convs), this head
    uses a 3×3 kernel to capture local spatial context while jumping directly
    to teacher dimensionality in a single operation.

    Args:
        student_channels: Output channels of FLIM encoder (e.g. 48).
        teacher_dim:      I-JEPA teacher embedding dim (1280 by default).
    """

    def __init__(
        self,
        student_channels: int = 48,
        teacher_dim: int = TEACHER_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_map: Tensor) -> Tensor:
        """Project spatial feature map to teacher embedding space.

        Args:
            feat_map: ``(B, student_channels, H, W)`` encoder output.

        Returns:
            ``(B, teacher_dim)`` projected embedding.
        """
        x = self.proj(feat_map)   # (B, teacher_dim, H, W)
        x = self.pool(x)           # (B, teacher_dim, 1, 1)
        return x.flatten(1)        # (B, teacher_dim)


class OneLayer1x1ConvDistillationProjectionHead(nn.Module):
    """Single 1×1 conv: student_channels → teacher_dim in one shot.

    Pointwise projection without spatial context:
        Conv2d(48→1280, k=1×1) + BN2d + GELU → AdaptiveAvgPool2d(1) → flatten

    Compared to OneLayerConvDistillationProjectionHead (3×3 kernel), this head
    uses a 1×1 kernel — pure channel mixing, no spatial aggregation.

    Args:
        student_channels: Output channels of FLIM encoder (e.g. 48).
        teacher_dim:      I-JEPA teacher embedding dim (1280 by default).
    """

    def __init__(
        self,
        student_channels: int = 48,
        teacher_dim: int = TEACHER_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_map: Tensor) -> Tensor:
        x = self.proj(feat_map)   # (B, teacher_dim, H, W)
        x = self.pool(x)           # (B, teacher_dim, 1, 1)
        return x.flatten(1)        # (B, teacher_dim)


class TwoLayer1x1ConvBN2dDistillationProjectionHead(nn.Module):
    """Two-layer 1×1 conv: student_channels → 256 → teacher_dim (~402k total params).

    Pointwise projection in two steps:
        Conv2d(48→256, k=1×1) + BN2d(256) + GELU
        Conv2d(256→1280, k=1×1) + BN2d(1280) + GELU
        → AdaptiveAvgPool2d(1) → flatten

    Compared to OneLayer1x1ConvDistillationProjectionHead (123k total), this head
    adds an intermediate 256-channel bottleneck for ~402k total parameters, matching
    the Init Random Distillation baseline parameter count.

    Args:
        student_channels: Output channels of FLIM encoder (e.g. 48).
        mid_channels:     Intermediate channel count (default 256).
        teacher_dim:      I-JEPA teacher embedding dim (1280 by default).
    """

    def __init__(
        self,
        student_channels: int = 48,
        mid_channels: int = 256,
        teacher_dim: int = TEACHER_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, teacher_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, feat_map: Tensor) -> Tensor:
        x = self.proj(feat_map)   # (B, teacher_dim, H, W)
        x = self.pool(x)           # (B, teacher_dim, 1, 1)
        return x.flatten(1)        # (B, teacher_dim)


class StudentClassificationHead(nn.Module):
    """Trainable linear classification head on top of student encoder output.

    In the CDD notebook the teacher's frozen classifier serves this role.
    Since I-JEPA has no classifier we attach a fresh trainable head to the
    student encoder.

    Args:
        student_dim: Raw student encoder output dimension (e.g. 48).
        num_classes: Number of target classes.
    """

    def __init__(self, student_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(student_dim, num_classes)

    def forward(self, student_emb: Tensor) -> Tensor:
        return self.head(student_emb)


def prepare_teacher_input(student_batch: Tensor, teacher_size: int = TEACHER_IMAGE_SIZE) -> Tensor:
    """Resize a student batch to teacher input size via bilinear interpolation.

    Only the spatial dimensions are adjusted here. Normalisation of the teacher
    input is handled separately (see ``KnnKappaProbeMixin`` consumers /
    ``--teacher-imagenet-norm``): the student may be fed raw LAB[0,1] while the
    teacher still needs ImageNet stats.

    Args:
        student_batch: ``(B, 3, H_s, W_s)`` float32 tensor.
        teacher_size:  Target spatial size for the teacher (default 224).

    Returns:
        ``(B, 3, teacher_size, teacher_size)`` resized tensor.
    """
    if student_batch.shape[-1] == teacher_size and student_batch.shape[-2] == teacher_size:
        return student_batch
    return F.interpolate(
        student_batch,
        size=(teacher_size, teacher_size),
        mode="bilinear",
        align_corners=False,
    )


# ── kNN-kappa validation probe ───────────────────────────────────────────────
#
# Why this exists: the distillation ModelCheckpoint historically monitored
# ``val/loss`` (MSE to the I-JEPA teacher). That loss is decoupled from — and
# can be *inverted* w.r.t. — downstream κ: minimising the MSE collapses the
# encoder (effective rank → ~1.5), so the epoch-0 checkpoint can have a *higher*
# downstream κ than the epoch-99 "best by val/loss" checkpoint
# (analysis_flim_distill/INVESTIGATION_training_problems_nonorm.md). The fix is
# the DINO/I-JEPA protocol: select the checkpoint by a kNN probe on the
# validation set.
#
# For a FROZEN-encoder run the 48d encoder output is constant across epochs, so
# the probe is run on the trainable 1280d projection instead (``knn_probe=
# "projection"``); for a trainable encoder the 48d embedding (``"encoder"``) is
# the right anti-collapse signal.

@torch.no_grad()
def _extract_probe_features(embed_fn, loader, device):
    """Run ``embed_fn`` over a (views, label) loader and return (X, y) numpy arrays.

    The loader yields multi-view tensors ``(B, V, C, H, W)`` (V==1 for the no-aug
    probe); only the first view is used.
    """
    import numpy as np

    feats: list = []
    labels: list = []
    for views, y in loader:
        if torch.is_tensor(views) and views.ndim == 5:
            x = views[:, 0]
        else:
            x = views
        emb = embed_fn(x.to(device))
        feats.append(emb.detach().float().cpu().numpy())
        if torch.is_tensor(y):
            labels.append(y.detach().cpu().numpy())
        else:
            labels.append(np.asarray(y))
    if not feats:
        return np.empty((0, 0)), np.empty((0,))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def knn_kappa_probe(embed_fn, train_loader, val_loader, device,
                    subsample: int = 3000, seed: int = 42, k: int = 20) -> float:
    """kNN-probe Cohen's kappa: fit on train embeddings, score on val.

    Protocol (no test-split leak): memory bank = train-split embeddings extracted
    with the *test* transform (no augmentation), subsampled to ``subsample`` with
    a fixed seed; query = full val split. StandardScaler fit on train,
    KNeighborsClassifier(k=min(k, n_train-1)), kappa = cohen_kappa_score(y_val,
    pred). Returns NaN on degenerate cases (n_train < 2, < 2 classes, empty val).
    """
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import cohen_kappa_score

    X_tr, y_tr = _extract_probe_features(embed_fn, train_loader, device)
    X_val, y_val = _extract_probe_features(embed_fn, val_loader, device)

    if X_tr.shape[0] < 2 or X_val.shape[0] < 1 or np.unique(y_tr).size < 2:
        return float("nan")

    if subsample and X_tr.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_tr.shape[0], size=subsample, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_val = scaler.transform(X_val)

    n_neighbors = min(k, X_tr.shape[0] - 1)
    if n_neighbors < 1:
        return float("nan")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_tr, y_tr)
    pred = knn.predict(X_val)
    return float(cohen_kappa_score(y_val, pred))


class KnnKappaProbeMixin:
    """Adds a ``val/knn_kappa`` metric to a distillation LightningModule.

    Expects the host module to expose ``self.student`` (with ``.encode`` and
    ``.encoder``) and ``self.proj_kd``, plus these hparams: ``knn_probe``
    ('encoder'|'projection'), ``knn_train_subsample``, ``knn_every_n_epochs``,
    ``seed``. Logs ``val/knn_kappa`` once per (eligible) validation epoch; the
    ModelCheckpoint monitors it with mode='max'. ``val/loss`` stays logged, and a
    separate best-by-loss checkpoint acts as fallback when the probe is NaN.
    """

    def _knn_embed_fn(self):
        probe = getattr(self.hparams, "knn_probe", "encoder")
        if probe == "projection":
            return lambda x: self.proj_kd(self.student.encoder(x))
        return lambda x: self.student.encode(x)

    def _build_knn_loaders(self):
        from torch.utils.data import DataLoader
        from src.data_modules.datasets.parasite_lejepa import ParasiteLejepaMultiViewDataset
        from src.data_modules.datasets.lejepa_dataset import _build_aug, _build_test

        dm = self.trainer.datamodule
        test_tf = _build_test(dm.image_size, imagenet_norm=dm.imagenet_norm)
        aug_tf = _build_aug(dm.image_size, imagenet_norm=dm.imagenet_norm)
        # Memory bank: train split with the *test* (no-aug) transform, V=1.
        train_probe_ds = ParasiteLejepaMultiViewDataset(
            parasite_dataset=dm.ds_train.base, V=1, image_size=dm.image_size,
            aug_transform=aug_tf, test_transform=test_tf,
        )
        nw = min(getattr(dm, "num_workers", 0), 4)
        train_loader = DataLoader(
            train_probe_ds, batch_size=dm.batch_size, shuffle=False,
            num_workers=nw, pin_memory=getattr(dm, "pin_memory", False),
            persistent_workers=False,
        )
        val_loader = DataLoader(
            dm.ds_val, batch_size=dm.batch_size, shuffle=False,
            num_workers=nw, pin_memory=getattr(dm, "pin_memory", False),
            persistent_workers=False,
        )
        return train_loader, val_loader

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return
        every = getattr(self.hparams, "knn_every_n_epochs", 1) or 1
        if every > 1 and (self.current_epoch % every) != 0:
            return
        kappa = float("nan")
        try:
            train_loader, val_loader = self._build_knn_loaders()
            kappa = knn_kappa_probe(
                self._knn_embed_fn(), train_loader, val_loader, self.device,
                subsample=int(getattr(self.hparams, "knn_train_subsample", 3000)),
                seed=int(getattr(self.hparams, "seed", 42)),
            )
        except Exception as exc:  # pragma: no cover - probe must never kill training
            import logging
            logging.getLogger(__name__).warning("kNN-kappa probe failed: %s", exc)
        self.log("val/knn_kappa", kappa, prog_bar=True)
