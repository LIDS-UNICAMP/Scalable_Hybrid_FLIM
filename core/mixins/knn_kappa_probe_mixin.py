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

"""Sonda de validacao kNN, movida de src/models/distillation.py:842-984.

A funcao publica ``knn_kappa_probe`` e a privada ``_extract_probe_features``
ficam aqui, junto do mixin que as usa — nao viram utilitario solto. Tamanho de
batch e ordem das linhas sao os da origem: o ruido conhecido de kappa vem do
batching e mexer nisso mudaria numero.

HISTORICO: existia aqui uma segunda sonda, ``svm_kappa_probe``, que fitava o
``fit_svm`` do projeto a cada epoca de validacao para escolher o checkpoint pela
mesma regua do fim. Ela foi REMOVIDA: rodava com ``max_iter=-1`` dentro do
processo do treino, entao enquanto o libsvm nao voltava a epoca nao avancava e a
GPU ficava a 0% — ha registro de 2176s numa unica epoca e de um fit de 16h que
nunca voltou. A selecao de checkpoint e por ``val/knn_kappa``; o ``fit_svm``
continua sendo a regua do fim, em ``eval/``, que e onde ele custa uma vez por run
em vez de uma vez por epoca.
"""

import torch

from core.metrics import compute_metrics


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
    # Mesma metrica do resto do projeto: kappa sai de core/metrics.py, nunca de
    # um cohen_kappa_score local. num_classes cobre o maior rotulo visto — classe
    # ausente entra como linha/coluna zerada e nao muda o kappa.
    num_classes = int(max(np.max(y_val), np.max(pred))) + 1
    return compute_metrics(y_val, pred, num_classes)["kappa"]


class KnnKappaProbeMixin:
    """Adds ``val/knn_kappa`` a um modulo de destilacao.

    Expects the host module to expose ``self.student`` (with ``.encode`` and
    ``.encoder``) and ``self.proj_kd``, plus these hparams: ``knn_probe``
    ('encoder'|'projection'), ``knn_train_subsample``, ``knn_every_n_epochs``,
    ``seed``.

    Chave logada por epoca de validacao elegivel:

    * ``val/knn_kappa`` — e esta que o ModelCheckpoint monitora com mode='max'.

    ``knn_every_n_epochs`` pula a sonda nas epocas que nao forem multiplas dele.
    Ao pular, ``val/knn_kappa`` NAO e logada, e o ModelCheckpoint nao aceita
    monitor ausente: com ``knn_every_n_epochs > 1`` o checkpoint PRECISA de
    ``every_n_epochs`` casado (ou ``strict: false``), senao o fit morre com
    MisconfigurationException — foi exatamente esse o bug que derrubou 41 runs.
    ``val/loss`` continua logado, e o checkpoint best-by-loss segue como rede de
    seguranca quando a sonda devolve NaN.
    """

    def _knn_embed_fn(self):
        probe = getattr(self.hparams, "knn_probe", "encoder")
        if probe == "projection":
            return lambda x: self.proj_kd(self.student.encoder(x))
        return lambda x: self.student.encode(x)

    def _build_knn_loaders(self):
        from torch.utils.data import DataLoader
        # Mesmos objetos da origem, so renomeados na mudanca para core/: a classe
        # ParasiteLejepaMultiViewDataset virou MultiViewDataset e _build_aug/_build_test
        # perderam o underscore. AST identico nas tres (a unica diferenca em
        # MultiViewDataset e o acento na mensagem de um TypeError inalcancavel daqui).
        # Batch, ordem das linhas, shuffle, seed e subsample: intocados.
        from core.data.multi_view_dataset import MultiViewDataset
        from core.data.transforms import build_aug, build_test

        dm = self.trainer.datamodule
        test_tf = build_test(dm.image_size, imagenet_norm=dm.imagenet_norm)
        aug_tf = build_aug(dm.image_size, imagenet_norm=dm.imagenet_norm)
        # Memory bank: train split with the *test* (no-aug) transform, V=1.
        train_probe_ds = MultiViewDataset(
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
        import logging
        _log = logging.getLogger(__name__)

        if getattr(self.trainer, "sanity_checking", False):
            return
        knn_every = getattr(self.hparams, "knn_every_n_epochs", 1) or 1
        # A sonda SVM foi REMOVIDA daqui. Ela fitava um `fit_svm` com
        # `max_iter=-1` dentro do processo do treino, toda epoca: enquanto o
        # libsvm nao voltava a epoca nao andava e a GPU ficava a 0% (ha registro
        # de 2176s numa epoca e de um fit de 16h que nunca voltou). A selecao de
        # checkpoint agora e por `val/knn_kappa`. O `fit_svm` continua sendo a
        # regua do FIM, em `eval/` -- e la que ele pertence.
        if not (knn_every <= 1 or (self.current_epoch % knn_every) == 0):
            return

        subsample = int(getattr(self.hparams, "knn_train_subsample", 3000))
        seed = int(getattr(self.hparams, "seed", 42))
        # Batch, ordem das linhas, shuffle e seed sao os de _build_knn_loaders:
        # o ruido conhecido de kappa vem dai e mexer nisso mudaria numero.
        try:
            train_loader, val_loader = self._build_knn_loaders()
        except Exception as exc:  # pragma: no cover - probe must never kill training
            _log.warning("probe loaders failed: %s", exc)
            train_loader = val_loader = None

        kappa = float("nan")
        if train_loader is not None:
            try:
                kappa = knn_kappa_probe(
                    self._knn_embed_fn(), train_loader, val_loader, self.device,
                    subsample=subsample, seed=seed,
                )
            except Exception as exc:  # pragma: no cover
                _log.warning("kNN-kappa probe failed: %s", exc)
        self.log("val/knn_kappa", kappa, prog_bar=True)
