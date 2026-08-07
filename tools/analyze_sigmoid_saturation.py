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
# ╚══════════════════════════════════════════════════════════════════════════════════════╝
"""Análise do achatamento de magnitude ReLU -> Sigmoid na cabeça de classificação.

Esta NÃO é uma análise sintética: para cada checkpoint `sigmoid2l_classhead_*`
já treinado (pesos congelados, sem gradiente), roda o *test set* pelo forward do
próprio modelo e extrai DOIS vetores reais por amostra:

  P = avgpool(encoder(x))   -> 48-d, "após ReLU"  (a última não-linearidade do
                               encoder é a ReLU da conv3, portanto P >= 0)
  S = Sigmoid(layer1(P))    -> 24-d, "após Sigmoid" (valores comprimidos em [0,1])

O objetivo é medir QUANTO a camada Sigmoid achata / colapsa a magnitude presente
em P. Como P e S vivem em escalas diferentes (P é ilimitado >= 0; S está em [0,1]),
todas as métricas usadas são INVARIANTES A ESCALA — comparam a *forma* da
distribuição, não os valores absolutos:

  cv_P, cv_S              coeficiente de variação (std/mean) das normas L2 por
                          amostra; adimensional, logo comparável entre P e S.
  cv_ratio_S_over_P       colapso relativo do spread de magnitude (cv_S / cv_P).
  dynrange_P, dynrange_S  faixa dinâmica p95/p5 das normas (razão, sem unidade).
  spearman_normP_normS    correlação de postos entre ||P|| e ||S|| — invariante a
                          qualquer transformação monotônica; ~0 => o "quão longe"
                          codificado na magnitude de P não sobrevive à Sigmoid.
  sigmoid_sat_frac        fração das unidades sigmoides saturadas (<0.01 ou >0.99).
  dead_chan_P             fração de canais mortos (variância ~0) em P.
  effdim_P, effdim_S      dimensionalidade efetiva (participation ratio da
                          covariância); invariante a escala isotrópica.
  entropy_P, entropy_S    entropia (nats) do histograma dos valores renormalizados
                          para [0,1] via min-max — logo invariante a escala.
  probe_acc_P, probe_acc_S acurácia de um linear probe (logistic regression sobre
                          features padronizadas); mede a informação de classe
                          RETIDA em cada representação, independente da escala.

Saída: um CSV com UMA linha por checkpoint válido (default
`results/relu_vs_sigmoid_flatten.csv`).

Uso:
    python tools/analyze_sigmoid_saturation.py
    python tools/analyze_sigmoid_saturation.py --pattern 'sigmoid2l_classhead_*' \
        --out results/relu_vs_sigmoid_flatten.csv --image-size 200
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = "/dados/home/moliveira/Scalable_Hybrid_FLIM"
sys.path.insert(0, _ROOT)
from src.data_modules.parasite_data_module_lejepa_splited import ParasiteLejepaDataModuleSplited
from src.modules.classification_flim_module import ClassificationFlimModule, _dataset_short_to_parasite_name

_ART = os.path.join(_ROOT, "artifacts", "classification_flim")


def cv_norms(V):
    """Coeficiente de variação (std/mean) das normas L2 por amostra.

    Retorna (cv, normas). Adimensional -> comparável entre P e S apesar da
    diferença de escala.
    """
    n = np.linalg.norm(V, axis=1)
    m = n.mean()
    return float(n.std() / m) if m > 0 else 0.0, n


def dyn_range(norms):
    """Faixa dinâmica p95/p5 das normas (razão de percentis; sem unidade)."""
    p5, p95 = np.percentile(norms, [5, 95])
    return float(p95 / p5) if p5 > 0 else float("inf")


def part_ratio(V):
    """Dimensionalidade efetiva via participation ratio dos autovalores da covariância.

    PR = (sum lambda)^2 / sum(lambda^2). Invariante a escala isotrópica: mede
    quantas direções carregam variância de fato.
    """
    Vc = V - V.mean(0, keepdims=True)
    cov = (Vc.T @ Vc) / max(len(Vc) - 1, 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def entropy_vals(V, bins=50):
    """Entropia (nats) do histograma dos valores renormalizados para [0,1].

    O min-max antes de histogramar torna a métrica invariante a escala: mede o
    espalhamento da distribuição de valores, não a sua amplitude absoluta.
    """
    x = V.ravel()
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return 0.0
    h, _ = np.histogram((x - lo) / (hi - lo), bins=bins, range=(0, 1), density=False)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def linear_probe(V, y, seed=0):
    """Acurácia (cross-val) de uma logistic regression -> info de classe retida.

    As features são padronizadas (StandardScaler), removendo escala, de modo que
    o probe compara o CONTEÚDO linearmente separável de P e S em pé de igualdade.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    Vs = StandardScaler().fit_transform(V)
    try:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        k = min(5, np.bincount(y).min())  # k-fold limitado pela classe minoritária
        if k < 2:
            return float("nan")
        return float(cross_val_score(clf, Vs, y, cv=k).mean())
    except Exception:
        return float("nan")


@torch.no_grad()
def analyze_run(run_dir, device, image_size=200):
    """Roda o test set de um checkpoint e devolve o dicionário de métricas (ou None).

    Retorna None se o checkpoint não tiver metadados/best_kappa.ckpt válidos.
    """
    meta_p = os.path.join(run_dir, "run_metadata.json")
    ck = os.path.join(run_dir, "checkpoints", "best_kappa.ckpt")
    name = os.path.basename(run_dir)
    if not os.path.exists(meta_p) or not os.path.exists(ck) or os.path.getsize(ck) == 0:
        return None
    meta = json.load(open(meta_p))
    ds, split, pct = meta["dataset"], int(meta["split"]), int(meta["percentage"])
    nc = int(meta.get("num_classes", 9))
    frozen = bool(meta.get("freeze_encoder", False))
    imagenet_norm = not bool(meta.get("no_imagenet_norm", False))

    # Carrega o modelo treinado e separa encoder / cabeça (pesos congelados).
    mod = ClassificationFlimModule.load_from_checkpoint(ck, map_location=device).eval().to(device)
    enc, head = mod.model.encoder, mod.model.head

    dm = ParasiteLejepaDataModuleSplited(
        parasite_name=_dataset_short_to_parasite_name(ds), split=split, percentage=pct,
        image_size=image_size, V_train=1, V_eval=1, batch_size=32, num_workers=4,
        pin_memory=True, persistent_workers=False, loader="ift_lab", imagenet_norm=imagenet_norm)
    dm.setup("test")

    Ps, Ss, ys = [], [], []
    for batch in dm.test_dataloader():
        views, y = batch
        # As views podem vir como lista/tupla de tensores, ou como um único tensor
        # 5D [B, V, C, H, W] (múltiplas views). Pegamos sempre a primeira view.
        if isinstance(views, (list, tuple)):
            v = views[0]
        elif views.ndim == 5:                     # [B, V, C, H, W] -> [B, C, H, W]
            v = views[:, 0]
        else:
            v = views
        x = v.to(device)
        feat = enc(x)
        P = head.pool(feat).flatten(1)          # 48-d, após ReLU (>= 0)
        S = head.sigmoid(head.layer1(P))        # 24-d, após Sigmoid (em [0,1])
        Ps.append(P.cpu().numpy())
        Ss.append(S.cpu().numpy())
        ys.append(y.numpy())
    P = np.concatenate(Ps)
    S = np.concatenate(Ss)
    y = np.concatenate(ys).astype(int)

    cvP, nP = cv_norms(P)
    cvS, nS = cv_norms(S)
    sat = float(((S < 0.01) | (S > 0.99)).mean())            # fração saturada da sigmoid
    dead_P = float((P.std(0) < 1e-6).mean())                 # canais mortos (ReLU) em P
    rho = float(pd.Series(nP).corr(pd.Series(nS), method="spearman"))  # o "quão longe" sobrevive?
    return {
        "run": name, "dataset": ds, "split": split, "pct": pct,
        "mode": "frozen" if frozen else "unfrozen", "imagenet_norm": imagenet_norm, "N": len(y),
        # magnitude / achatamento
        "cv_P": round(cvP, 4), "cv_S": round(cvS, 4),
        "cv_ratio_S_over_P": round(cvS / cvP, 4) if cvP else np.nan,
        "dynrange_P": round(dyn_range(nP), 3), "dynrange_S": round(dyn_range(nS), 3),
        "spearman_normP_normS": round(rho, 4),
        "sigmoid_sat_frac": round(sat, 4), "dead_chan_P": round(dead_P, 4),
        # estrutura
        "effdim_P": round(part_ratio(P), 3), "effdim_S": round(part_ratio(S), 3),
        "entropy_P": round(entropy_vals(P), 4), "entropy_S": round(entropy_vals(S), 4),
        # info de classe retida
        "probe_acc_P": round(linear_probe(P, y), 4), "probe_acc_S": round(linear_probe(S, y), 4),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Extrai métricas de achatamento ReLU->Sigmoid dos checkpoints sigmoid2l.")
    p.add_argument("--pattern", default="sigmoid2l_classhead_*",
                   help="glob dos diretórios de checkpoint dentro de artifacts/classification_flim "
                        "(default: sigmoid2l_classhead_*).")
    p.add_argument("--out", default=os.path.join(_ROOT, "results", "relu_vs_sigmoid_flatten.csv"),
                   help="caminho do CSV de saída (default: results/relu_vs_sigmoid_flatten.csv).")
    p.add_argument("--image-size", type=int, default=200,
                   help="tamanho da imagem passado ao data module (default: 200).")
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = sorted(d for d in glob.glob(os.path.join(_ART, args.pattern)) if os.path.isdir(d))
    print(f"device={dev}  runs={len(runs)}  pattern={args.pattern!r}")
    rows = []
    for rd in runs:
        try:
            r = analyze_run(rd, dev, image_size=args.image_size)
            if r:
                rows.append(r)
                print(f"  {r['run']:52s} satS={r['sigmoid_sat_frac']:.2f} "
                      f"cvP={r['cv_P']:.2f} cvS={r['cv_S']:.2f} rho={r['spearman_normP_normS']:.2f} "
                      f"probeP={r['probe_acc_P']:.2f} probeS={r['probe_acc_S']:.2f}")
        except Exception as e:
            print(f"  [ERR] {os.path.basename(rd)}: {e}")
    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
