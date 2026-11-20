"""linear_probe.py — probe linear (regressao logistica) sobre features congeladas.

Criterio de crescimento da Especificacao 1 (§2.7/§7): o ganho de acuracia do probe
decide se um estagio continua. Minimo de proposito: encoder congelado, features
padronizadas (media/desvio do treino), LogisticRegression com random_state fixo.

Os loaders seguem o formato de src/transforms/object_crop.py:
(views[B,1,3,S,S], labels[B], masks[B,1,S,S]) — usa views[:, 0] e os rotulos.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def _features(embed_fn, loader, device) -> tuple:
    feats, labels = [], []
    for batch in loader:
        x, y = batch[0], torch.as_tensor(batch[1])
        if x.ndim == 5:
            x = x[:, 0]
        feats.append(embed_fn(x.to(device)).float().cpu())
        labels.append(y.argmax(1) if y.ndim == 2 else y)   # one-hot -> indice
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def linear_probe(embed_fn, train_loader, val_loader, device) -> float:
    """Acuracia (%) no val de uma regressao logistica ajustada no treino.

    ``embed_fn(x[B,3,S,S]) -> [B,D]``; se for metodo de um nn.Module, o modulo e
    posto em eval() durante o probe e o modo anterior e restaurado ao final.
    """
    module = getattr(embed_fn, "__self__", None)
    was_training = isinstance(module, torch.nn.Module) and module.training
    if isinstance(module, torch.nn.Module):
        module.eval()
    try:
        train_x, train_y = _features(embed_fn, train_loader, device)
        val_x, val_y = _features(embed_fn, val_loader, device)
    finally:
        if was_training:
            module.train()
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=5000, random_state=0)
    clf.fit(scaler.transform(train_x), train_y)
    return float(100.0 * np.mean(clf.predict(scaler.transform(val_x)) == val_y))
