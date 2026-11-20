"""eval_backbone_head.py -- backbone CONGELADO x cabeca SEM TREINO, SVM original do repo.

Para cada backbone congelado mede kappa e F1 de VALIDACAO e de TESTE com a SVM original
(`core.svm.fit_svm`: SVC linear, C=100, ovo, max_iter=-1, SEM scaler -- nunca mudar), uma SVM
reajustada por representacao, sempre no TREINO. O teste e so relatorio (never_feeds_fitness).

Cabecas (nenhuma treinada; nada ajustado em val/teste; embedding fp32, sem autocast, sem TF32,
sem TTA):

  gap1280      embedding oficial do checkpoint: head(encoder(x)) (growth / modehb; dim = a da
               propria cabeca, 1280 ou 2560) ou, nos hibridos de outubro (Lightning), o GAP
               (avgpool2d) do mapa L4 = 1280-d, como em results/svm_disc_bp01w64c1280_test_avg.csv
               (eval/svm.py run_job, source=growth_stages, embed_mode=avgpool2d).
  so2_1280_pca z = Norm(P[mu(X), sigma(X), sqrtm(Cov(X))_triu]) sobre X = mapa do ULTIMO bloco
               (posicoes x canais), P = PCA dos canais ajustada SO no treino (pixels amostrados com
               semente por imagem). k=48: 48+48+48*49/2 = 1272, + 8 zeros = 1280.
  so2_4096_pca idem com k=88: 88+88+88*89/2 = 4092, + 4 zeros = 4096.
  so2_1280_rnd / so2_4096_rnd  CONTROLE: mesma leitura com P = projecao ortonormal fixa semeada
               (QR de gaussiana, seed fixa), sem nada ajustado nos dados.
  xb_1280_pca / xb_4096_pca  COVARIANCIA ENTRE BLOCOS: para cada par (i<j) de blocos, o mapa maior
               e reamostrado (adaptive_avg_pool2d) para a grade do menor, cada bloco e projetado pela
               sua PCA (SO treino), e a cross-cov kp x kp (centrada por imagem sobre as posicoes) e
               achatada; todos os pares concatenados; kp = floor(sqrt(D / n_pares)) (limitado pelos
               canais do bloco); zeros ate D.
  Norm = signed-sqrt + L2 sobre o vetor inteiro (zeros de preenchimento continuam zero).
  sqrtm: eigh em fp64 (autovalores < 0 grampeados em 0). Momentos/cov em fp64 na CPU.

Entradas (--item id=...,kind=...,ckpt=...; ou --preset):
  growth  backbone.pt do growth (ga_search.backbones.load_backbone, sha256 conferido contra o
          result.json do no); dados = growth_state.json:config_full.run.data acima do no.
  modehb  encoder.pt das buscas modehb ({"encoder": Sequential, "head": EmbeddingHead});
          dados = config_resolved.yaml:data acima do candidato.
  hybrid  best_kappa.ckpt Lightning dos hibridos de outubro (disc_bp01w64c1280); dados =
          ParasiteDataset (ift_lab, build_test, imagenet_norm=False) nos splits da discovery
          (new_split_parasito/<ds>/splits_incremental/split<s>/data_descriptor_perc<p>.json; md5
          conferido contra o repo discovery quando existe).

Saida: jsonl, uma linha por (backbone, cabeca). Retomada idempotente: chave
(backbone_id, dataset, split_name, head); backbone com todas as cabecas feitas e pulado.

Uso (ver README.md desta pasta):
  python research/spifil_b0b4/readout_x/eval_backbone_head.py --preset table --device cuda:0 \
      --out research/spifil_b0b4/readout_x/results_athena.jsonl
  python .../eval_backbone_head.py --table --out <jsonl>     # (re)escreve tabela.md
Compativel com torch 2.2 (nenhuma API so da 2.7).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GROWTH_CODE = Path("/dados/home/moliveira/exps/growth/code")
DISCOVERY = Path("/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery")

HEADS = ("gap1280", "so2_1280_pca", "so2_4096_pca", "so2_1280_rnd", "so2_4096_rnd",
         "xb_1280_pca", "xb_4096_pca")
TTA_HEAD = "so2_4096_pca_tta8"            # so com --tta8 (coluna extra; as outras nao mudam)
TTA_BASE = {TTA_HEAD: "so2_4096_pca"}     # cabeca TTA -> cabeca cuja PCA/SVM (treino) ela usa
SO2_K = {1280: 48, 4096: 88}          # k + k + k(k+1)/2 <= D (maior k): 1272 / 4092
RND_SEED = 12345
PCA_SEED = 0
PARASITE = {"eggs": "helminth-eggs", "larvae": "helminth-larvae", "protozoan": "protozoan-cysts"}


def _setup_path(code_root: Path):
    for p in (str(REPO), str(code_root)):        # code_root fica em sys.path[0]
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


# ───────────────────────────────────────────────────────────── leituras (puras, testaveis)
def so2_dim(k: int) -> int:
    return 2 * k + k * (k + 1) // 2


def pad_to(z, D: int):
    import torch
    if z.shape[1] > D:
        raise ValueError(f"vetor {z.shape[1]} > {D}")
    return torch.cat([z, z.new_zeros(z.shape[0], D - z.shape[1])], 1)


def norm_ssqrt_l2(z):
    z = z.sign() * z.abs().sqrt()
    return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _sqrtm_psd(C):
    import torch
    C = (C + C.transpose(1, 2)) / 2
    w, U = torch.linalg.eigh(C)
    return (U * w.clamp(min=0).sqrt()[:, None, :]) @ U.transpose(1, 2)


_SO2_DEV = os.environ.get("READOUT_SO2_DEVICE", "cpu")  # "cuda": momentos/cov/eigh em fp64 na GPU (mesma conta, mesma precisão; patch da Jaci 19/11)


def _flat(x):
    """[B,C,H,W] -> [B,HW,C] fp64 (CPU por padrão; GPU se READOUT_SO2_DEVICE=cuda)."""
    return x.detach().double().to(_SO2_DEV).permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])


def so2_vector(fmap, mu, V, D: int):
    """Leitura de 2a ordem de UM mapa: Norm(pad([mu, sigma, triu(sqrtm(Cov))])) -> [B, D] fp32."""
    import torch
    X = _flat(fmap)
    Y = (X - mu.to(X.device)) @ V.to(X.device)              # [B, HW, k]
    m = Y.mean(1)
    Yc = Y - m[:, None, :]
    cov = Yc.transpose(1, 2) @ Yc / Y.shape[1]
    sd = cov.diagonal(dim1=1, dim2=2).clamp(min=0).sqrt()
    S = _sqrtm_psd(cov)
    k = V.shape[1]
    iu = torch.triu_indices(k, k, device=S.device)
    z = torch.cat([m, sd, S[:, iu[0], iu[1]]], 1).cpu()
    return norm_ssqrt_l2(pad_to(z, D)).float()


def xb_kp(chans, D: int) -> int:
    n_pairs = len(chans) * (len(chans) - 1) // 2
    if n_pairs == 0:
        raise ValueError("cross-block exige >= 2 blocos")
    return max(1, int(math.isqrt(D // n_pairs)))


def xb_vector(fmaps, pcas, D: int):
    """Cross-cov entre todos os pares de blocos, achatada, concatenada, Norm -> [B, D] fp32."""
    import torch
    import torch.nn.functional as F
    kp = xb_kp([f.shape[1] for f in fmaps], D)
    parts = []
    for i in range(len(fmaps)):
        for j in range(i + 1, len(fmaps)):
            a, b = fmaps[i], fmaps[j]
            hw = min(a.shape[2] * a.shape[3], b.shape[2] * b.shape[3])
            grid = a.shape[2:] if a.shape[2] * a.shape[3] == hw else b.shape[2:]
            a = F.adaptive_avg_pool2d(a.float(), tuple(grid))
            b = F.adaptive_avg_pool2d(b.float(), tuple(grid))
            ya = (_flat(a) - pcas[i][0]) @ pcas[i][1][:, :kp]
            yb = (_flat(b) - pcas[j][0]) @ pcas[j][1][:, :kp]
            ya, yb = ya - ya.mean(1, keepdim=True), yb - yb.mean(1, keepdim=True)
            parts.append((ya.transpose(1, 2) @ yb / ya.shape[1]).flatten(1))
    return norm_ssqrt_l2(pad_to(torch.cat(parts, 1), D)).float()


class PixelMoments:
    """Momentos de canal acumulados em fp64 sobre pixels amostrados (semente por IMAGEM ->
    independente do batch). So recebe imagens de TREINO (quem chama garante; testado)."""

    def __init__(self, n_px: int = 64, seed: int = PCA_SEED):
        self.n_px, self.seed = int(n_px), int(seed)
        self.s1 = self.s2 = None
        self.n = 0

    def add(self, fmap, img_ids):
        import torch
        # mesmos valores de _flat(fmap)[b, idx]: amostra antes de converter (fp64 e exato)
        f = fmap.detach().permute(0, 2, 3, 1).reshape(fmap.shape[0], -1, fmap.shape[1])
        rows = []
        for b, gid in enumerate(img_ids):
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + int(gid))
            hw = f.shape[1]
            idx = torch.randperm(hw, generator=g)[: min(self.n_px, hw)]
            rows.append(f[b, idx.to(f.device)].double().cpu())
        P = torch.cat(rows)
        if self.s1 is None:
            self.s1 = torch.zeros(P.shape[1], dtype=torch.float64)
            self.s2 = torch.zeros(P.shape[1], P.shape[1], dtype=torch.float64)
        self.s1 += P.sum(0)
        self.s2 += P.T @ P
        self.n += P.shape[0]

    def pca(self, k: int):
        """-> (mu [C], V [C, min(k,C)]) componentes em ordem decrescente de variancia."""
        import torch
        mu = self.s1 / self.n
        cov = (self.s2 - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        cov = (cov + cov.T) / 2
        _, V = torch.linalg.eigh(cov)
        V = V.flip(1)[:, : min(k, V.shape[1])]
        # sinal canonico (maior |entrada| positiva): o eigh nao fixa o sinal do autovetor
        sgn = V.gather(0, V.abs().argmax(0, keepdim=True)).sign()
        return mu, (V * torch.where(sgn == 0, torch.ones_like(sgn), sgn)).contiguous()


def random_proj(C: int, k: int, seed: int = RND_SEED):
    import torch
    g = torch.Generator().manual_seed(seed + 7919 * C)
    k = min(k, C)
    Q, _ = torch.linalg.qr(torch.randn(C, k, generator=g, dtype=torch.float64))
    return torch.zeros(C, dtype=torch.float64), Q.contiguous()


class Readouts:
    """Ajuste (SO treino) + leitura das 6 cabecas de 2a ordem / cross-block."""

    def __init__(self, n_px: int = 64):
        self.n_px = n_px
        self.moments = None
        self.fitted = False

    def add_train(self, blocks, img_ids):
        if self.moments is None:
            self.moments = [PixelMoments(self.n_px, PCA_SEED + i) for i in range(len(blocks))]
        for m, f in zip(self.moments, blocks):
            m.add(f, img_ids)

    def fit(self, chans):
        kmax = max(SO2_K.values())
        self.pca = [m.pca(max(kmax, xb_kp(chans, 4096) if len(chans) > 1 else 1))
                    for m in self.moments]
        C = chans[-1]
        self.so2 = {}
        for D, k in SO2_K.items():
            mu, V = self.pca[-1]
            self.so2[(D, "pca")] = (mu, V[:, :k])
            self.so2[(D, "rnd")] = random_proj(C, k)
        self.fitted = True
        return self

    def __call__(self, blocks, heads=None):
        assert self.fitted, "Readouts.fit (treino) antes de ler"
        out = {}
        for D in SO2_K:
            for kind in ("pca", "rnd"):
                if heads is None or f"so2_{D}_{kind}" in heads:
                    mu, V = self.so2[(D, kind)]
                    out[f"so2_{D}_{kind}"] = so2_vector(blocks[-1], mu, V, D)
            if len(blocks) > 1 and (heads is None or f"xb_{D}_pca" in heads):
                out[f"xb_{D}_pca"] = xb_vector(blocks, self.pca, D)
        return out

    def tta8(self, blocks_fn, x, D: int = 4096, blocks_id=None):
        """so2_{D}_pca com TTA8: media dos vetores das 8 vistas (rot90 k=0..3 x flip horizontal),
        com a MESMA PCA (treino, sem TTA). A SVM (linear) e a da cabeca so2_{D}_pca; para
        kernel linear, media dos vetores == media das decision_function das 8 vistas."""
        import torch
        mu, V = self.so2[(D, "pca")]
        acc = None
        for flip in (False, True):
            xf = torch.flip(x, dims=(3,)) if flip else x
            for k in range(4):
                if k == 0 and not flip and blocks_id is not None:   # vista identidade ja calculada
                    bl = blocks_id
                else:
                    bl = blocks_fn(torch.rot90(xf, k, dims=(2, 3)))
                z = so2_vector(bl[-1], mu, V, D).double()
                acc = z if acc is None else acc + z
        return (acc / 8).float()


# ───────────────────────────────────────────────────────────── modelos
class Wrapped:
    """blocks(x) -> lista de mapas; official(blocks) -> embedding oficial."""

    def __init__(self, kind, blocks_fn, official_fn, modules, meta):
        self.kind, self.blocks, self.official, self.modules, self.meta = (
            kind, blocks_fn, official_fn, modules, meta)


def _free_gpu(model):
    """19/11: extracao na GPU, SVM na CPU -- solta a memoria da GPU antes das SVMs (as features ja estao em numpy)."""
    import torch
    for m in getattr(model, "modules", None) or []:
        if hasattr(m, "cpu"): m.cpu()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def _seq_blocks(mods):
    def run(x):
        outs = []
        for m in mods:
            x = m(x)
            outs.append(x)
        return outs
    return run


def _find_up(path: Path, names):
    for d in [path.parent, *path.parents]:
        for n in names:
            if (d / n).is_file():
                return d / n
    return None


def file_digest(path, algo="md5"):
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(kind, ckpt: Path, device, init=None):
    import torch
    import torch.nn.functional as F
    if init == "spifil":           # variante so-SPiFiL: mesma arquitetura, sem treino (spifil_init.py)
        import spifil_init as SI
        if kind == "modehb":
            enc, head, (n0, n1) = SI.spifil_init_modehb(ckpt, device)
            mods, official = list(enc), (lambda bl: head(bl[-1]))
        elif kind == "growth":
            model, (n0, n1) = SI.spifil_init_growth(ckpt, device)
            mods, official = list(model.blocks), (lambda bl: model.head(bl[-1]))
        else:
            enc, (n0, n1) = SI.spifil_init_hybrid(ckpt, device)
            mods = [getattr(enc, f"conv{n}") for n in range(1, getattr(enc, "n_layers", 3) + 1)]
            official = lambda bl: F.adaptive_avg_pool2d(bl[-1], 1).flatten(1)
        if n0 != n1:
            raise ValueError(f"{ckpt}: arquitetura so-SPiFiL difere do treinado ({n0} != {n1} params)")
        return Wrapped(kind, _seq_blocks(mods), official, mods, {"variant": "spifil_init", "params": n0,
                       "exact_width": getattr(mods[0] if kind != "modehb" else enc, "spifil_exact_width", None)})
    if kind == "growth":
        from ga_search.backbones.checkpoint import load_backbone
        res = ckpt.parent / "result.json"
        sha = json.loads(res.read_text()).get("checkpoint_sha256") if res.is_file() else None
        if sha and file_digest(ckpt, "sha256") != sha:
            raise ValueError(f"{ckpt}: sha256 != result.json checkpoint_sha256")
        model, spec = load_backbone(ckpt, "cpu", expected_sha256=sha)
        model = model.to(device).eval()
        mods = list(model.blocks)
        return Wrapped(kind, _seq_blocks(mods), lambda bl: model.head(bl[-1]), [model],
                       {"sha256_checked": bool(sha), "arch": spec.arch_hash()})
    if kind == "modehb":
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        enc, head = ck["encoder"].to(device).eval(), ck["head"].to(device).eval()
        return Wrapped(kind, _seq_blocks(list(enc)), lambda bl: head(bl[-1]), [enc, head],
                       {"cand_id": ck.get("cand_id"), "fidelity": ck.get("fidelity")})
    if kind == "hybrid":
        from eval.growth_stages import _encoder
        enc, _head, epoch = _encoder({"ckpt": str(ckpt)}, device)
        mods = [getattr(enc, f"conv{n}") for n in range(1, getattr(enc, "n_layers", 3) + 1)]
        return Wrapped(kind, _seq_blocks(mods),
                       lambda bl: F.adaptive_avg_pool2d(bl[-1], 1).flatten(1), [enc],
                       {"ckpt_epoch": epoch})
    raise ValueError(kind)


# ───────────────────────────────────────────────────────────── dados
class _Indexed(object):
    """Dataset -> (img, label, idx) com subconjunto opcional (smoke)."""

    def __init__(self, base, keep=None):
        self.base, self.keep = base, keep if keep is not None else list(range(len(base)))

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        item = self.base[self.keep[i]]
        return item[0], int(item[1]), self.keep[i]


class _Cached(object):
    """ReconstructionDataset materializado uma vez em memmap fp32 (valores identicos), compartilhado
    entre processos pelo page cache. Chave = sha1(nomes+caminhos, size, inorm). Lock por arquivo."""

    def __init__(self, base, cache_dir: Path, log=print):
        import fcntl
        import torch
        self.labels = [int(s.label) for s in base.samples]
        key = hashlib.sha1(json.dumps([[str(s.image_path) for s in base.samples],
                                       repr(base.transform)]).encode()).hexdigest()[:16]
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{key}_{len(base)}.f32"
        x0 = base[0][0]
        self.shape = (len(base), *x0.shape)
        with open(cache_dir / f"{key}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not path.is_file():
                t0 = time.time()
                tmp = path.with_suffix(".tmp")
                mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=self.shape)
                from torch.utils.data import DataLoader
                dl = DataLoader(base, batch_size=32, shuffle=False, num_workers=8)
                i = 0
                for x, _y, _n in dl:
                    mm[i:i + len(x)] = x.numpy()
                    i += len(x)
                mm.flush()
                del mm
                os.replace(tmp, path)
                log(f"  [CACHE] gravado {path.name} {self.shape} em {time.time() - t0:.0f}s")
            fcntl.flock(lk, fcntl.LOCK_UN)
        self.mm = np.load(path, mmap_mode="r")
        assert self.mm.shape == self.shape, (self.mm.shape, self.shape)
        self.samples = base.samples

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, i):
        import torch
        return torch.from_numpy(np.array(self.mm[i])), self.labels[i], self.samples[i].name


def _subset(n, max_images, seed=0):
    if not max_images or n <= max_images:
        return None
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n, max_images, replace=False).tolist())


def load_splits(kind, ckpt: Path, args):
    """-> dict split -> Dataset (img, label, idx), dataset, split_name, n_classes."""
    if kind in ("growth", "modehb"):
        import yaml
        from ga_search import data as existing_data
        from ga_search.progressive_data import ReconstructionDataset
        cfg_path = _find_up(ckpt, ["growth_state.json", "config_resolved.yaml"])
        if cfg_path is None:
            raise FileNotFoundError(f"{ckpt}: sem growth_state.json/config_resolved.yaml acima")
        if cfg_path.name == "growth_state.json":
            d = json.loads(cfg_path.read_text())["config_full"]["run"]["data"]
        else:
            d = yaml.safe_load(cfg_path.read_text())["data"]
        ds = dict(d["dataset_config"])
        for key in ("images_dir", "masks_dir", "split_json", "labels_json"):
            if ds.get(key):
                ds[key] = ds[key].replace("{root}", str(ds.get("root", "")))
        split, pct = int(d.get("split", 1)), int(d.get("percentage", 100))
        keys = {"train": ds.get("train_key", "train"), "validation": "validation",
                "test": (ds.get("eval") or {}).get("final_test_key", "test")}
        size, inorm = int(d.get("image_size", 224)), bool(d.get("imagenet_norm", False))
        out = {}
        for s, key in keys.items():
            samples = existing_data._samples(ds, pct, split, key)
            base = ReconstructionDataset(samples, size, inorm)
            if getattr(args, "data_cache", None):
                base = _Cached(base, Path(args.data_cache))
            out[s] = _Indexed(base, _subset(len(samples), args.max_images))
        sj = ds["split_json"].format(split=split, pct=pct)
        args.inc_dir = Path(ds.get("root", "")) / "splits_incremental" / f"split{split}"   # --train-pcts
        split_name = (f"{ds.get('name', d.get('dataset'))}/split{split}/pct{pct} "
                      f"[{os.path.relpath(sj, ds.get('root', '/'))} md5={file_digest(sj)[:12]}; "
                      f"cfg={cfg_path.name}]")
        n_classes = int(ds.get("n_classes") or 0) or None
        vt = ({s.name for s in existing_data._samples(ds, pct, split, keys["train"])}
              == {s.name for s in existing_data._samples(ds, pct, split, "validation")})
        return out, str(d.get("dataset")), split_name, n_classes, vt
    # hybrid
    from config import get_single_parasite_paths
    from core.constants import IMAGE_SIZE, NUM_CLASSES
    from core.data.parasite_dataset import ParasiteDataset
    from core.data.transforms import build_test
    m = re.search(r"(eggs|larvae|protozoan|flowers|remote_sensing|ham10000|aid|coconut_trees)"
                  r"_split(\d+)_pct(\d+)", str(ckpt))
    if not m:
        raise ValueError(f"{ckpt}: nao achei <ds>_split<s>_pct<p> no caminho")
    short, split, pct = m[1], int(m[2]), int(m[3])
    if short not in PARASITE:
        # bloco copiado de exps/oct_hybrid/code/.../eval_backbone_head.py (precisa --code-root
        # oct_hybrid/code): mesmas imagens/split 1 das buscas, sha256 conferido; ParasiteDataset/
        # build_test/ift_lab do treino do hibrido.
        from oct_hybrid import bases as _oct
        tf = build_test(IMAGE_SIZE, imagenet_norm=False)
        out = {}
        for s in ("train", "validation", "test"):
            base = ParasiteDataset(set_name=s, split=split, percentage=pct, transform=tf,
                                   loader="ift_lab", path_dataset=short)
            out[s] = _Indexed(base, _subset(len(base), args.max_images))
        sj = get_single_parasite_paths(short, split, pct)[0]["split_json"]
        sha = file_digest(sj, "sha256")
        if sha != _oct.BASES[short][2]:
            raise ValueError(f"{sj}: sha256 != split_sha256 da busca ({short})")
        args.inc_dir = Path(_oct.search_split_json(short)).parents[1] / "splits_incremental" / f"split{split}"
        split_name = (f"{short}/split{split}/pct{pct} [busca:{_oct.search_split_json(short)} "
                      f"sha256={sha[:12]} (=busca)]")
        with open(sj) as fh:
            js = json.load(fh)
        vt = set(js["train"]) == set(js["validation"])
        return out, short, split_name, NUM_CLASSES[short], vt
    name = PARASITE[short]
    tf = build_test(IMAGE_SIZE, imagenet_norm=False)
    out = {}
    for s in ("train", "validation", "test"):
        base = ParasiteDataset(set_name=s, split=split, percentage=pct, transform=tf,
                               loader="ift_lab", path_dataset=name)
        out[s] = _Indexed(base, _subset(len(base), args.max_images))
    sj = get_single_parasite_paths(name, split, pct)[0]["split_json"]
    args.inc_dir = Path(sj).parent                     # --train-pcts: o proprio splits_incremental/split<s> do split
    md5 = file_digest(sj)
    rel = sj.split("data/to_modules/")[-1]
    disc = DISCOVERY / "data" / "to_modules" / rel
    if disc.is_file() and file_digest(disc) != md5:
        raise ValueError(f"split {rel}: md5 difere do repo discovery")
    split_name = (f"discovery:{rel} md5={md5[:12]}"
                  f"{' (=discovery)' if disc.is_file() else ''}")
    # ARMADILHA: nos descritores perc100 da discovery `validation` == `train` (mesma lista).
    # O kappa_val desses hibridos e entao kappa NO TREINO; a linha leva val_is_train=true.
    with open(sj) as fh:
        js = json.load(fh)
    vt = set(js["train"]) == set(js["validation"])
    return out, short, split_name, NUM_CLASSES[short], vt


# ───────────────────────────────────────────────────────────── extracao + SVM
def _loader(ds, args):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=False)


def extract(model: Wrapped, splits, device, args, log=print):
    """Passe 1: momentos de canal SO no treino. Passe 2: todas as cabecas em train/val/test."""
    import torch
    ro = Readouts(n_px=args.n_px)
    chans = None
    with torch.no_grad():
        for x, _y, idx in _loader(splits["train"], args):
            bl = model.blocks(x.to(device).float())
            chans = [b.shape[1] for b in bl]
            ro.add_train(bl, idx.tolist())
        ro.fit(chans)
        log(f"  [PCA] so treino: n_train={len(splits['train'])} blocos={chans} "
            f"pixels/bloco={ro.moments[-1].n}")
        feats, labels = {}, {}
        heads = getattr(args, "heads_active", None)
        for s in ("train", "validation", "test"):
            acc, ys = {}, []
            for x, y, _idx in _loader(splits[s], args):
                x = x.to(device).float()
                bl = model.blocks(x)
                out = {"gap1280": model.official(bl).float().cpu()}
                out.update(ro(bl, heads))
                if getattr(args, "tta8", False) and s != "train":   # treino nunca ve TTA
                    out[TTA_HEAD] = ro.tta8(model.blocks, x, blocks_id=bl)
                for h, v in out.items():
                    acc.setdefault(h, []).append(v)
                ys.append(torch.as_tensor(y))
            feats[s] = {h: torch.cat(v).numpy() for h, v in acc.items()}
            labels[s] = torch.cat(ys).numpy()
    return feats, labels, chans


def score(feats, labels, n_classes=None):
    from core.metrics import compute_metrics
    from core.svm import fit_svm, svm_protocol
    allys = np.concatenate([labels[s] for s in labels])
    classes = np.unique(allys)
    remap = {int(c): i for i, c in enumerate(classes)}
    y = {s: np.array([remap[int(v)] for v in labels[s]]) for s in labels}
    nc = len(classes)          # rotulos remapeados 0..nc-1 (uniao train/val/test; nada ajustado)
    rows = {}
    for h in feats["train"]:
        Xtr = feats["train"][h]
        t0 = time.perf_counter()
        clf = fit_svm(Xtr, y["train"], tag=f"SVM_{h}")        # C=100, linear, ovo, sem scaler
        r = {"dim": int(Xtr.shape[1]), "svm_protocol": svm_protocol(clf),
             "svm_fit_status": clf.fit_diagnostics_["svm_fit_status"]}
        for s, tag in (("validation", "val"), ("test", "test")):
            m = compute_metrics(y[s], clf.predict(feats[s][h]), num_classes=nc)
            r[f"kappa_{tag}"], r[f"f1_{tag}"] = float(m["kappa"]), float(m["f1"])
            r[f"acc_{tag}"] = float(m["acc"])
        r["svm_s"] = round(time.perf_counter() - t0, 2)
        rows[h] = r
        for ht, hb in TTA_BASE.items():
            if hb == h and ht in feats["validation"]:
                rt = {"dim": int(feats["validation"][ht].shape[1]),
                      "svm_protocol": r["svm_protocol"], "svm_fit_status": r["svm_fit_status"],
                      "svm_from_head": h}
                for s, tag in (("validation", "val"), ("test", "test")):
                    m = compute_metrics(y[s], clf.predict(feats[s][ht]), num_classes=nc)
                    rt[f"kappa_{tag}"], rt[f"f1_{tag}"] = float(m["kappa"]), float(m["f1"])
                    rt[f"acc_{tag}"] = float(m["acc"])
                rt["svm_s"] = 0.0
                rows[ht] = rt
    return rows


def extract_pct(model, splits, device, args):
    """% de treino, protocolo da Jaci (--train-pct dela), com UM forward para todos os p: por p, treino =
    data_descriptor_perc<p>.json (splits_incremental) ∩ treino da busca, em ordem ordenada (= _samples);
    PCA da so2 SO nesse p% (ultimo bloco, semente de pixel = posicao no subconjunto, como no Readouts);
    SVM so nesse p%; val/teste = SEMPRE os da busca. -> ({f"{h}@p{p}": linha}, canais)."""
    import torch
    base = splits["train"].base
    names = [os.path.basename(str(getattr(s, "image_path", s))) for s in base.samples]
    assert splits["train"].keep == list(range(len(base))) and names == sorted(names) or args.kind_hybrid
    S, sub = set(names), {}
    for p in args.train_pcts:
        f = Path(args.inc_dir or "") / f"data_descriptor_perc{p}.json"
        if not f.is_file():
            print(f"  [PCT] sem {f}")
            continue
        d = set(json.loads(f.read_text())["train"])
        # treino_p (regra alinhada com a Jaci, 19/11): 1) nomes exatos, se >= 95% do esperado; 2) sem o prefixo
        # <classe:06d>_ dos *_raw de Eggs/Protozoan (splits/split1_modehb_raw_build.py), se >= 95%; 3) fallback
        exp = 0.95 * p / 100 * len(names)
        m, src = np.array([n in d for n in names]), "descritor"
        if m.sum() < exp:
            m2 = np.array([re.sub(r"^\d{6}_(?=\d{6}_)", "", n) in d for n in names])
            if m2.sum() >= exp: m, src = m2, "descritor_mapeado_raw"
        if m.sum() < exp and hasattr(base.samples[0], "label"):
            # ADAPTACAO (= Jaci): descritor nao e o split da busca -> amostra estratificada semeada Random(42+split)
            # de p% do TREINO da busca, por classe, min. 1 por classe; val/teste da busca
            import random as _rnd
            by = {}
            for j, x in enumerate(base.samples): by.setdefault(x.label, []).append((x.name, j))
            g = _rnd.Random(42 + int(Path(args.inc_dir).name.removeprefix("split"))); m = np.zeros(len(names), bool)
            for lab in sorted(by):
                xs = sorted(by[lab]); g.shuffle(xs)
                m[[j for _n, j in xs[:max(1, round(len(xs) * p / 100))]]] = True
            src = "fallback_estratificado"
            print(f"  [PCT] ADAPTACAO p={p}: amostra estratificada {int(m.sum())}/{len(names)} do treino da busca")
        sub[p] = (m, np.cumsum(m) - 1, f, len(d), src)
    mom, chans, D = {}, None, 4096
    with torch.no_grad():
        for x, _y, idx in _loader(splits["train"], args):            # passe 1: momentos por p
            bl = model.blocks(x.to(device).float())
            chans = [b.shape[1] for b in bl]
            for p, (m, pos, _f, _n, _s) in sub.items():
                k = [j for j, i in enumerate(idx.tolist()) if m[i]]
                if k:
                    mom.setdefault(p, PixelMoments(args.n_px, PCA_SEED + len(bl) - 1)).add(
                        bl[-1][k], [int(pos[int(idx[j])]) for j in k])
        pca = {p: mo.pca(max(SO2_K.values())) for p, mo in mom.items()}    # = Readouts.fit (k=88)
        feats, labels = {}, {}
        for s in ("train", "validation", "test"):                          # passe 2: cabecas por p
            acc, ys = {}, []
            for x, y, _idx in _loader(splits[s], args):
                x = x.to(device).float()
                bl = model.blocks(x)
                acc.setdefault("gap1280", []).append(model.official(bl).float().cpu())
                for p, (mu, V) in pca.items():
                    acc.setdefault(f"so2_4096_pca@p{p}", []).append(so2_vector(bl[-1], mu, V[:, :SO2_K[D]], D))
                if getattr(args, "tta8", False) and s != "train":   # = Readouts.tta8, uma vista por vez p/ todos os p
                    tacc = {}
                    for flip in (False, True):
                        xf = torch.flip(x, dims=(3,)) if flip else x
                        for k in range(4):
                            bv = bl if (k == 0 and not flip) else model.blocks(torch.rot90(xf, k, dims=(2, 3)))
                            for p, (mu, V) in pca.items():
                                z = so2_vector(bv[-1], mu, V[:, :SO2_K[D]], D).double()
                                tacc[p] = z if p not in tacc else tacc[p] + z
                    for p, z in tacc.items():
                        acc.setdefault(f"{TTA_HEAD}@p{p}", []).append((z / 8).float())
                ys.append(torch.as_tensor(y))
            feats[s] = {h: torch.cat(v).numpy() for h, v in acc.items()}
            labels[s] = torch.cat(ys).numpy()
    del pca, mom
    _free_gpu(model)
    out = {}
    for p, (m, _pos, f, n_desc, src) in sub.items():
        ix = np.flatnonzero(m)
        hk = {"gap1280": "gap1280", "so2_4096_pca": f"so2_4096_pca@p{p}", TTA_HEAD: f"{TTA_HEAD}@p{p}"}
        sf = {s: {h: feats[s][k][ix] if s == "train" else feats[s][k] for h, k in hk.items() if k in feats[s]}
              for s in feats}
        sl = {s: labels[s][ix] if s == "train" else labels[s] for s in labels}
        try:
            res = score(sf, sl)
        except Exception as e:                 # ex.: 1 classe so no p=1
            print(f"  [PCT] p={p} falhou: {e!r}")
            continue
        for h, r in res.items():
            out[f"{h}@p{p}"] = {**r, "head": h, "pct": p, "n_train": len(ix), "n_desc": n_desc,
                                "desc": str(f), "desc_md5": file_digest(f), "pct_source": src}
    return out, chans


FM_SPLIT = {"ham10000": "split1_modehb.json"}             # split da busca (o resto: split1.json)
ORGANIZED = Path("/dados/home/moliveira/datasets/organized")


def fm_pct(npz, args):
    """Foundation model (DINOv2/I-JEPA/AIMv2/ConvNeXt/ResNet-50): embedding congelado JA extraido pelo
    benchmark da discovery (/exps/moliveira/benchmark/emb/<modelo>/<ds>.npz: X, y, names). So a leitura
    nativa do FM ("fm_emb": CLS/GAP do proprio modelo); sem mapas -> sem so2. Mesmo protocolo de %:
    treino = descritor ∩ treino da busca (ordenado), val/teste = os da busca, SVM original."""
    z = np.load(npz)
    ds = npz.stem
    sj = ORGANIZED / ds / "splits" / FM_SPLIT.get(ds, "split1.json")
    busca = json.loads(sj.read_text())
    row = {str(n): i for i, n in enumerate(z["names"])}
    pick = lambda ns: np.array([row[n] for n in sorted(ns)])
    feats = {s: {"fm_emb": z["X"][pick(busca[s])]} for s in ("validation", "test")}
    labels = {s: z["y"][pick(busca[s])] for s in ("validation", "test")}
    tr = set(busca["train"])
    out = {}
    for p in args.train_pcts:
        f = ORGANIZED / ds / "splits_incremental" / "split1" / f"data_descriptor_perc{p}.json"
        if not f.is_file():
            continue
        d = set(json.loads(f.read_text())["train"])
        ix = pick(d & tr)
        res = score({"train": {"fm_emb": z["X"][ix]}, **feats}, {"train": z["y"][ix], **labels})
        for h, r in res.items():
            out[f"{h}@p{p}"] = {**r, "head": h, "pct": p, "n_train": len(ix), "n_desc": len(d),
                                "desc": str(f), "desc_md5": file_digest(f)}
    split_name = f"{ds}/split1 [{sj.name} md5={file_digest(sj)[:12]}; emb={npz}]"
    return out, ds, split_name, {s: [0] * n for s, n in (("train", len(tr)), ("validation", len(labels["validation"])),
                                                          ("test", len(labels["test"])))}


# ───────────────────────────────────────────────────────────── presets
_GR = "artifacts/hmodehb_b0b4/growth"


def _glob_node(root: Path, ds_dir: str, node: str, fid=None):
    pats = [f"{_GR}/*/{ds_dir}/**/nodes/{node}/**/backbone.pt",
            f"{_GR}/*/{ds_dir}/**/nodes/{node}/backbone.pt"]
    hits = sorted({p for pat in pats for p in glob.glob(str(root / pat), recursive=True)})
    if fid is not None:
        hits = [h for h in hits if re.fullmatch(rf"f?{fid}", Path(h).parent.name)]
    return hits


def preset_table(args):
    """Lista da tabela. Growth: glob em <growth_root>/artifacts/hmodehb_b0b4/growth/*/<ds_dir>/...;
    exige 1 match (0 -> pulado com aviso; >1 -> erro com a lista, desambigue com --item)."""
    gr = Path(args.growth_root)
    items = []

    def growth(bid, ds_dir, node, fid=None, optional=False):
        hits = _glob_node(gr, ds_dir, node, fid)
        if len(hits) == 1:
            items.append({"id": bid, "kind": "growth", "ckpt": hits[0]})
        elif not hits:
            print(f"[PRESET] {bid}: 0 backbone.pt para {ds_dir}/nodes/{node}"
                  f"{'/f' + str(fid) if fid else ''} -- PULADO{' (opcional)' if optional else ''}")
        else:
            raise SystemExit(f"[PRESET] {bid}: ambiguo ({len(hits)}):\n  " + "\n  ".join(hits))

    c306 = REPO / "artifacts/modehb_spifil/remote_sensing/state/seed42/candidates/c00306/f100/encoder.pt"
    if c306.is_file():
        items.append({"id": "RS_B0_seed01_c00306", "kind": "modehb", "ckpt": str(c306)})
    else:   # seed01 do growth = conversao bitwise do c00306
        growth("RS_B0_seed01_c00306", "remote_sensing_fix4", "seed01")
    growth("RS_B1_fix5_B1g0c00_f45", "remote_sensing_fix5", "B1g0c00", 45)
    growth("RS_B2_fix4_B2g1c02_f15", "remote_sensing_fix4", "B2g1c02", 15)
    growth("RS_B2_fix4_B2g1c02_f45", "remote_sensing_fix4", "B2g1c02", 45)
    if args.nossl_b1:
        items.append({"id": "RS_B1_fix5_nossl", "kind": "growth", "ckpt": args.nossl_b1})
    else:
        growth("RS_B1_fix5_nossl", "remote_sensing_fix5_nossl", "B1*", None, optional=True)
    growth("FL_B0_seed01", "flowers*", "seed01")
    for ds in ("eggs", "larvae", "protozoan"):
        for s in (1, 2, 3):
            p = REPO / f"artifacts/spifil_growth/disc_bp01w64c1280/{ds}_split{s}_pct100/L4/checkpoints/best_kappa.ckpt"
            items.append({"id": f"HYB_oct_{ds}_s{s}_pct100", "kind": "hybrid", "ckpt": str(p)})
    return items


def parse_item(s):
    d = dict(kv.split("=", 1) for kv in s.split(","))
    for k in ("id", "kind", "ckpt"):
        if k not in d:
            raise SystemExit(f"--item precisa de id=,kind=,ckpt= ({s})")
    return d


# ───────────────────────────────────────────────────────────── tabela
TABLE_COLS = [("gap1280", "GAP 1280"), ("so2_1280_pca", "2a ordem 1280"),
              ("so2_4096_pca", "2a ordem 4096"), ("xb_1280_pca", "cross-block 1280"),
              ("xb_4096_pca", "cross-block 4096")]


TABLE_IDS = (["RS_B0_seed01_c00306", "RS_B1_fix5_B1g0c00_f45", "RS_B2_fix4_B2g1c02_f15",
              "RS_B2_fix4_B2g1c02_f45", "RS_B1_fix5_nossl", "FL_B0_seed01"]
             + [f"HYB_oct_{d}_s{s}_pct100" for d in ("eggs", "larvae", "protozoan") for s in (1, 2, 3)])


def write_table(jsonl: Path, md: Path, title=None):
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault((r["backbone_id"], r["dataset"], r["split_name"]), {})[r["head"]] = r
    L = [f"# Backbone x cabeca (sem treino) -- SVM original C=100, sem scaler",
         "", f"Fonte: `{jsonl}` ({len(rows)} linhas). Cada celula: kappa_val / **kappa_test** "
         "(F1_test). Teste = so relatorio (never_feeds_fitness).", ""]
    if title:
        L += [title, ""]
    L += ["| backbone | dataset | " + " | ".join(c for _, c in TABLE_COLS)
          + " | controle rnd 1280 / 4096 (k_test) |",
          "|---|---|" + "---|" * len(TABLE_COLS) + "---|"]
    for (bid, ds, _sn), hs in sorted(by.items()):
        cells = []
        if any(r.get("val_is_train") for r in hs.values()):
            bid = bid + " †"
        for h, _ in TABLE_COLS:
            r = hs.get(h)
            cells.append("—" if r is None else
                         f"{r['kappa_val']:.3f} / **{r['kappa_test']:.3f}** ({r['f1_test']:.3f})"
                         + (f" [{r['dim']}d]" if h == "gap1280" and r["dim"] != 1280 else ""))
        rnd = "/".join("—" if hs.get(h) is None else f"{hs[h]['kappa_test']:.3f}"
                       for h in ("so2_1280_rnd", "so2_4096_rnd"))
        L.append(f"| {bid} | {ds} | " + " | ".join(cells) + f" | {rnd} |")
    L += ["", "Dimensoes: 2a ordem 1280 = k48 (48 mu + 48 sigma + 1176 triu = 1272 + 8 zeros); "
          "4096 = k88 (88+88+3916 = 4092 + 4 zeros). cross-block: kp=floor(sqrt(D/n_pares)) por par, "
          "zeros ate D. GAP do growth/modehb = embedding da propria cabeca (dim marcada se != 1280).",
          "", "† validation == train no descritor do split (perc100 da discovery): o kappa_val dessa "
          "linha e medido no TREINO; so o kappa_test e informativo.",
          ""]
    seen = {b for b, _d, _s in by}
    falta = [b for b in TABLE_IDS if b not in seen]
    if falta:
        L += ["Pendentes (rodar no athena com `--preset table`, ver README.md): "
              + ", ".join(falta) + ".", ""]
    if any(r.get("max_images") for r in rows):
        L.insert(2, "**SMOKE (--max-images): numeros sem valor, so prova que o pipeline roda.**\n")
    md.write_text("\n".join(L))
    print(f"[TABELA] {md}")


# ───────────────────────────────────────────────────────────── main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--item", action="append", default=[], help="id=..,kind=growth|modehb|hybrid,ckpt=..")
    ap.add_argument("--preset", choices=["table"], default=None)
    ap.add_argument("--only", nargs="*", default=None, help="filtra ids (substring)")
    ap.add_argument("--growth-root", default=str(GROWTH_CODE))
    ap.add_argument("--code-root", default=str(GROWTH_CODE if GROWTH_CODE.is_dir() else REPO),
                    help="raiz importada primeiro (precisa de ga_search.backbones)")
    ap.add_argument("--nossl-b1", default=None, help="backbone.pt do B1 sem LeJEPA (fix5_nossl)")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.jsonl")))
    ap.add_argument("--table", action="store_true", help="so (re)escreve tabela.md a partir do --out")
    ap.add_argument("--table-md", default=str(Path(__file__).with_name("tabela.md")))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-px", type=int, default=64, help="pixels amostrados por imagem de treino (PCA)")
    ap.add_argument("--max-images", type=int, default=None, help="SO smoke: limita cada split")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--heads", nargs="*", default=None,
                    help="subconjunto de HEADS a calcular (default: todas, como antes)")
    ap.add_argument("--tta8", action="store_true",
                    help=f"acrescenta {TTA_HEAD}: so2_4096_pca com TTA8 (rot90 x flip) so em val/teste; "
                         "PCA e SVM do treino sem TTA (as outras cabecas nao mudam)")
    ap.add_argument("--data-cache", default=None, help="dir de cache memmap fp32 dos splits (modehb/growth)")
    ap.add_argument("--train-pcts", nargs="*", type=int, default=None,
                    help="SVM em p%% do treino (splits_incremental); grava so as linhas por pct")
    args = ap.parse_args(argv)
    if args.heads:
        bad = set(args.heads) - set(HEADS)
        if bad:
            raise SystemExit(f"--heads desconhecidas: {sorted(bad)}")
    args.heads_active = tuple(args.heads) if args.heads else None
    want = list(args.heads_active or HEADS) + ([TTA_HEAD] if args.tta8 else [])
    if args.train_pcts:
        want = [f"{h}@p{p}" for p in args.train_pcts for h in want]
    if args.tta8 and "so2_4096_pca" not in (args.heads_active or HEADS):
        raise SystemExit("--tta8 exige a cabeca so2_4096_pca")

    if args.table:
        write_table(Path(args.out), Path(args.table_md))
        return
    _setup_path(Path(args.code_root))
    # SIGSEGV no jemalloc_bg_thd do libarrow (vem com o sklearn->pandas) nas variantes so-SPiFiL:
    # desliga a thread de fundo do jemalloc do arrow (antes do import do sklearn)
    os.environ.setdefault("JE_ARROW_MALLOC_CONF", "background_thread:false")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if (socket.gethostname() == "athena" and os.environ.get("CUDA_VISIBLE_DEVICES") in ("2", "3")
            and args.device.startswith("cuda")):   # athena 2/3 dividem placa com treino: teto duro
        # 12 GiB: pico medido ate ~9 GB (fit SPiFiL da variante 8,7; ham10000 growth B1g0c00 > 8)
        torch.cuda.set_per_process_memory_fraction(12 * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    items = [parse_item(s) for s in args.item] + (preset_table(args) if args.preset else [])
    if args.only:
        items = [it for it in items if any(o in it["id"] for o in args.only)]
    if not items:
        raise SystemExit("nada a avaliar (--item / --preset)")
    for it in items:
        print(f"[ITEM] {it['id']:<28} {it['kind']:<7} "
              f"{'OK ' if Path(it['ckpt']).is_file() else 'FALTA'} {it['ckpt']}")
    if args.dry_run:
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for l in out.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["backbone_id"], r["ckpt"], r.get("max_images"),
                          r["head"] + (f"@p{r['pct']}" if "pct" in r else "")))
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else ""
    for it in items:
        ckpt = Path(it["ckpt"]).resolve()
        if all((it["id"], str(ckpt), args.max_images, h) in done for h in want):
            print(f"[SKIP] {it['id']} (todas as cabecas no {out.name})")
            continue
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if it["kind"] == "fm":                                 # embedding de FM ja extraido (so CPU)
            res, dataset, split_name, labels = fm_pct(ckpt, args)
            want = [f"fm_emb@p{p}" for p in args.train_pcts]
            model, feats, chans, val_is_train = type("M", (), {"meta": {"fm": ckpt.parent.name}})(), None, None, False
        else:
            model = load_model(it["kind"], ckpt, device, it.get("init"))
        if it["kind"] != "fm":
            splits, dataset, split_name, n_classes, val_is_train = load_splits(it["kind"], ckpt, args)
            if val_is_train:
                print(f"  [AVISO] {it['id']}: validation == train no descritor -> kappa_val e de TREINO")
            print(f"[RUN] {it['id']} | {dataset} | {split_name} | n="
                  f"{[len(splits[s]) for s in ('train', 'validation', 'test')]}")
            if args.train_pcts:
                args.kind_hybrid = it["kind"] == "hybrid"
                res, chans = extract_pct(model, splits, device, args)
                feats, labels = None, {s: [0] * len(splits[s]) for s in splits}
            else:
                feats, labels, chans = extract(model, splits, device, args)
                _free_gpu(model)
                res = score(feats, labels, n_classes)
        peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
        md5 = file_digest(ckpt)
        with out.open("a") as fh:
            for h in want:
                if h not in res or (it["id"], str(ckpt), args.max_images, h) in done:
                    continue
                r = res[h]
                row = {"backbone_id": it["id"].removesuffix("__pct") + (f"_p{r['pct']}" if "pct" in r else ""),
                       "kind": it["kind"], "dataset": dataset,
                       "split_name": split_name, "head": r.get("head", h), "dim": r["dim"],
                       "kappa_val": r["kappa_val"], "f1_val": r["f1_val"],
                       "kappa_test": r["kappa_test"], "f1_test": r["f1_test"],
                       "acc_val": r["acc_val"], "acc_test": r["acc_test"],
                       "n_train": r.get("n_train", len(labels["train"])), "n_val": len(labels["validation"]),
                       "n_test": len(labels["test"]), "ckpt": str(ckpt), "md5": md5,
                       "torch": torch.__version__, "host": socket.gethostname(), "gpu": gpu,
                       "tempo": round(time.time() - t0, 1), "pico_gpu_gb": round(peak, 3),
                       "svm_protocol": r["svm_protocol"], "svm_fit_status": r["svm_fit_status"],
                       "blocks_channels": chans, "n_px_pca": args.n_px, "tta": h in TTA_BASE,
                       "svm_from_head": r.get("svm_from_head"),
                       "precision": "fp32_noautocast_notf32", "max_images": args.max_images,
                       "val_is_train": bool(val_is_train),
                       "never_feeds_fitness": True, "meta": model.meta,
                       "item": {k: v for k, v in it.items() if k not in ("id", "kind", "ckpt")}}
                if "pct" in r:
                    row.update(pct=r["pct"], n_train_descritor=r["n_desc"], descritor=r["desc"],
                               descritor_md5=r["desc_md5"], split=1 if "split1" in r["desc"] else None,
                               pct_source=r.get("pct_source", "descritor"))
                fh.write(json.dumps(row, default=str) + "\n")
                print(f"  [RESULT] {it['id']} {h:<13} dim={r['dim']:<5} "
                      f"k_val={r['kappa_val']:+.4f} k_test={r['kappa_test']:+.4f} "
                      f"f1_test={r['f1_test']:.4f}")
        del model, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_table(out, Path(args.table_md))


if __name__ == "__main__":
    main()
