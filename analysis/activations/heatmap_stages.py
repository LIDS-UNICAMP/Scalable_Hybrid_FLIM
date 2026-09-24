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
"""Heatmap de atenção do encoder, uma imagem de validação por estágio do currículo.

Para cada célula ``(dataset, split, percentage)`` carrega o ``best_kappa.ckpt`` de todos
os estágios (``stage1``, ``stage2``, ``round1_stage3`` …), calcula a saliência do mapa de
features do encoder (RMS por posição), e escreve um PNG por imagem e por estágio. A escala
de cor é fixa dentro da célula, então os estágios são comparáveis lado a lado.

O PNG é composto direto em numpy e salvo com PIL — sem figura do matplotlib, que custava
~0,1 s por arquivo e transformava o grid inteiro (171.912 PNGs) numa noite de trabalho.
A paleta padrão é ``RdYlGn_r``, o verde→amarelo→vermelho de eye-tracking; ``--sigma`` e
``--alpha`` são o ajuste fino do aspecto de nuvem conforme a resolução da imagem.

Escrever é idempotente por sobrescrita: rodar de novo reescreve os PNGs por cima.

Saída::

    artifacts/heatmap/stages/<dataset>/split_<N>/pct_<P>/<estágio>/heatmap_<id>.png

Uso::

    # tudo: 171.912 PNGs — veja antes a contagem do dry_run=True
    python -m analysis.activations.heatmap_stages

Os antigos flags viraram parametros nomeados de ``heatmap_stages()``, com os
mesmos defaults; as listas (``dataset``, ``split``, ``percentage``, ``stage``,
``image_id``) tomam o lugar do ``action="append"``::

    heatmap_stages(dry_run=True)                     # dimensiona, nao escreve nada
    heatmap_stages(dataset=["eggs"], split=[1], percentage=[5], limit=3)
    heatmap_stages(stage=["stage1", "round2_stage4"])   # so os estagios extremos
    heatmap_stages(sigma=3.0, alpha=2.0, cmap="turbo", limit=5)
    heatmap_stages(skip_existing=True, device="cuda:1")  # retomar de onde parou

O mesmo conjunto de chaves pode vir de um YAML, que sobrescreve os parametros::

    heatmap_stages(config="<caminho>.yaml")
"""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from tqdm import tqdm  # noqa: E402
import yaml  # noqa: E402

# analysis/activations/ esta a 2 niveis da raiz do repo (o arquivo veio de tools/, que era 1).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.data import ParasiteDataModule  # noqa: E402
from methods.autoencoder import AutoEncoderFlimModule  # noqa: E402
from core.constants import PARASITE_NAME  # noqa: E402

GRID = ROOT / "artifacts" / "spifil_growth" / "grid4"
OUT = ROOT / "artifacts" / "heatmap" / "stages"
DATASETS, SPLITS, PERCENTAGES = ["eggs", "larvae", "protozoan"], [1, 2, 3], [5, 50]
STAGE_RE = re.compile(r"(?:round(\d+)_)?stage(\d)$")


BATCH = 32   # o mesmo do datamodule do treino

TITLE_SIZE = 9   # legível sobre a imagem nativa de 200 px; cresce junto com --scale
CBAR_W = 14      # largura da barra de intensidade, em px de imagem (também multiplicada por --scale)
CBAR_TICKS = 5   # 0, ¼, ½, ¾ e vmax

# a DejaVuSans vem junto com o matplotlib, então não há dependência nova
FONT_PATH = matplotlib.font_manager.findfont("DejaVu Sans")


@lru_cache(maxsize=8)
def _font(size: int):
    """TrueType e não a bitmap do ``load_default()``: é o que faz o "·" e o "é" do título saírem.

    Cacheada porque são 171.912 PNGs e o ``--scale`` fixa o tamanho: uma carga, não uma por arquivo.
    """
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:   # noqa: BLE001 - fonte é acessório; sem ela o PNG ainda vale
        return ImageFont.load_default()


def _stage_order(stage: str) -> tuple[int, int]:
    """Ordem cronológica do currículo: stage1 < stage2 < round1_stage3 < … < round2_stage4."""
    m = STAGE_RE.match(stage)
    return (int(m[1] or 0), int(m[2]))


def _find_stages(ds: str, split: int, pct: int, wanted: list[str] | None):
    """Estágios da célula que têm checkpoint, em ordem cronológica: [(rótulo, ckpt, treinando)]."""
    found = []
    for ckpt in sorted((GRID / f"{ds}_split{split}_pct{pct}").glob("*/checkpoints/best_kappa.ckpt")):
        stage = ckpt.parents[1].name
        if not STAGE_RE.match(stage) or (wanted and stage not in wanted):
            continue
        # sem run_metadata.json = estágio ainda treinando; entra mesmo assim, marcado no título
        found.append((stage, ckpt, not (ckpt.parents[1] / "run_metadata.json").exists()))
    return sorted(found, key=lambda t: _stage_order(t[0]))


def _load_encoder(ckpt: Path, device: str):
    """Encoder em eval + época do checkpoint + nº de canais de saída."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    grow = blob["hyper_parameters"].get("flim_weights_path") or ""
    override = {}
    if grow and not os.path.isabs(grow):
        # estágios 3/4 gravam o caminho do round*_grow relativo à raiz do repo; sem
        # absolutizar, a carga só funciona com cwd == raiz
        grow = str(ROOT / grow)
        override = dict(arch_json=f"{grow}/architecture.json", flim_weights_path=grow)
    module = AutoEncoderFlimModule.load_from_checkpoint(ckpt, map_location=device, **override)
    encoder = module.eval().model.encoder
    with torch.no_grad():
        n_ch = encoder(torch.zeros(1, 3, 200, 200, device=device)).shape[1]
    return encoder, int(blob["epoch"]), n_ch


@torch.no_grad()
def _heat(encoder, x: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,H,W]: RMS por posição do mapa de features, reamostrado ao tamanho da imagem."""
    # .mean(1) e não .sum(1): dividir pelo nº de canais é o que mantém o valor comparável
    # quando o estágio cresce de 48 para 45/42 canais
    h = encoder(x).pow(2).mean(1).sqrt()
    return F.interpolate(h[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


def _focus(h: torch.Tensor) -> list[float]:
    """Fração da massa nos 10% de pixels mais quentes — adimensional, comparável entre estágios."""
    flat = h.flatten(1)
    k = max(1, int(0.1 * flat.shape[1]))
    return (flat.topk(k, dim=1).values.sum(1) / flat.sum(1).clamp_min(1e-12)).tolist()


def _cell_vmax(encoders, loader, device: str, n_imgs: int = 64) -> float:
    """vmin é sempre 0 (o ReLU torna o zero um valor físico); vmax é o p99,5 da célula.

    Percentil e não máximo: um outlier achataria todos os painéis. Um único vmax
    compartilhado por todos os estágios da célula — normalizar cada painel sozinho faria
    todo estágio sair com o mesmo contraste e a comparação viraria mentira.
    """
    vals, seen = [], 0
    for views, _ in loader:
        x = AutoEncoderFlimModule._first_view(views).to(device, non_blocking=True)
        for enc in encoders:
            vals.append(_heat(enc, x).flatten().cpu().numpy())
        seen += x.shape[0]
        if seen >= n_imgs:
            break
    return max(float(np.percentile(np.concatenate(vals), 99.5)), 1e-6)


def _draw(png: Path, src: str, h: np.ndarray, vmax: float, title: str,
          sigma: float, alpha: float, cmap: str, scale: int) -> None:
    """Compõe o RGBA em numpy e grava com PIL — sem figura do matplotlib.

    Uma figura com ``savefig(bbox_inches="tight")`` custava ~0,1 s por PNG, o que dava ~5 h
    só de desenho para o grid inteiro. Aqui não há eixo, canvas nem colorbar: a saliência
    vira cor, entra por cima do fundo com alpha proporcional, e a única coisa desenhada é
    a faixa de título e a barra de intensidade.
    """
    # fundo em cinza replicado nos 3 canais para o colormap ser a única cor da figura
    bg = np.asarray(Image.open(src).convert("L"), dtype=np.float32)[..., None] / 255.0

    sal = gaussian_filter(h.astype(np.float32), sigma=sigma)   # dá o aspecto de "nuvem"
    # NÃO é min-max por imagem: o divisor é o vmax da célula (_cell_vmax). Normalizar cada
    # PNG sozinho faria todo estágio sair com o mesmo contraste e a comparação viraria mentira
    sal = np.clip(sal / vmax, 0.0, 1.0)
    lut = plt.get_cmap(cmap)
    # teto de 0,75 na opacidade: mesmo no pixel mais quente a imagem tem que aparecer por baixo
    a = np.clip(sal * alpha, 0.0, 0.75)[..., None]
    heat = Image.fromarray(((bg * (1.0 - a) + lut(sal)[..., :3] * a) * 255.0).astype(np.uint8))
    if scale != 1:
        # LANCZOS e não NEAREST: a saliência já é suave, o upscale não tem borda dura para inventar
        heat = heat.resize((heat.width * scale, heat.height * scale), Image.LANCZOS)

    font = _font(TITLE_SIZE + 4 * (scale - 1))
    pad, lh = 4 * scale, font.size + 3 * scale
    lines = title.split("\n")
    band = 2 * pad + lh * len(lines)

    # Barra de intensidade: a MESMA rampa do overlay, quente em cima, 0 embaixo. Ela é o que
    # torna a comparação entre estágios legível — o vmax é o da célula inteira, então o mesmo
    # vermelho vale o mesmo valor em todos os PNGs da célula, e a barra diz qual valor é esse.
    ramp = np.repeat(lut(np.linspace(1.0, 0.0, heat.height))[:, None, :3], CBAR_W * scale, axis=1)
    cbar = Image.fromarray((ramp * 255.0).astype(np.uint8))
    ticks = [(i / (CBAR_TICKS - 1), f"{vmax * i / (CBAR_TICKS - 1):.3g}") for i in range(CBAR_TICKS)]

    # a faixa de título é mais larga que a imagem; alargar o canvas em vez de encolher a fonte
    # mantém o título legível e a imagem no mesmo tamanho em todos os estágios
    w = max(int(max(font.getlength(ln) for ln in lines)) + 2 * pad,
            2 * pad + heat.width + cbar.width + int(max(font.getlength(t) for _, t in ticks)) + 3 * pad)
    out = Image.new("RGB", (w, band + heat.height + pad), "white")
    out.paste(heat, (pad, band))
    out.paste(cbar, (2 * pad + heat.width, band))
    draw = ImageDraw.Draw(out)
    for i, ln in enumerate(lines):
        draw.text((pad, pad + i * lh), ln, font=font, fill="black")
    x = 2 * pad + heat.width + cbar.width
    for frac, lab in ticks:
        y = band + int((1.0 - frac) * (heat.height - 1))
        draw.line([(x, y), (x + pad // 2, y)], fill="black", width=max(1, scale // 2))
        draw.text((x + pad, y - font.size // 2), lab, font=font, fill="black")
    # compress_level=1: o PNG fica ~30% maior e a gravação, várias vezes mais rápida
    out.save(png, compress_level=1)


def _plan(args):
    """[(ds, split, pct, estágios, datamodule, nº de imagens)] para as células pedidas."""
    plan = []
    for ds in args.dataset:
        for split in args.split:
            for pct in args.percentage:
                stages = _find_stages(ds, split, pct, args.stage)
                if not stages:
                    continue
                dm = ParasiteDataModule(
                    parasite_name=PARASITE_NAME[ds],
                    split=split, percentage=pct, image_size=200,
                    V_train=1, V_eval=1, batch_size=BATCH, num_workers=args.num_workers,
                    pin_memory=True, persistent_workers=args.num_workers > 0,
                    loader="ift_lab", imagenet_norm=False,
                )
                dm.setup()
                n = len(dm.ds_val)
                if args.image_id:   # o plano conta so as imagens pedidas
                    have = {Path(q).stem for q in dm.ds_val.base.samples}
                    n = sum(1 for i in args.image_id if i in have)
                plan.append((ds, split, pct, stages, dm, n if args.limit is None else min(args.limit, n)))
    return plan


def _subset_loader(dm, wanted: list[str] | None, batch_size: int = BATCH):
    """Loader restrito aos ``--image-id`` pedidos, na ordem em que foram passados.

    Sem isso, achar uma imagem entre as 4.547 da validacao do protozoan custaria varrer o
    conjunto inteiro. ``dm.ds_val`` e indexavel e ``.base.samples`` esta na mesma ordem, entao
    um ``Subset`` resolve. Devolve ``(loader, caminhos)``; ``None`` se nenhum id casar.
    """
    srcs = dm.ds_val.base.samples
    if not wanted:
        return dm.val_dataloader(), srcs
    by_id = {Path(p).stem: i for i, p in enumerate(srcs)}
    idx = [by_id[i] for i in wanted if i in by_id]
    if not idx:
        return None, []
    sub = torch.utils.data.Subset(dm.ds_val, idx)
    loader = torch.utils.data.DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, [srcs[i] for i in idx]


def _run_cell(ds, split, pct, stages, dm, n_imgs, args, pbar):
    loaded = [(*_load_encoder(ckpt, args.device), stage, training) for stage, ckpt, training in stages]
    # A escala sai das MESMAS imagens que serao desenhadas: com --image-id, tirar o p99,5
    # da validacao inteira daria um vmax que a imagem pedida talvez nunca alcance.
    vmax_loader, _ = _subset_loader(dm, args.image_id, BATCH)
    vmax = _cell_vmax([e[0] for e in loaded], vmax_loader or dm.val_dataloader(), args.device)

    # pct entra na árvore porque a validação MUDA com a porcentagem (eggs split1: 2431
    # imagens no pct5, 1279 no pct50); sem ele os dois escreveriam por cima um do outro
    dirs = {}
    for *_, stage, _ in loaded:
        dirs[stage] = OUT / ds / f"split_{split}" / f"pct_{pct}" / stage
        dirs[stage].mkdir(parents=True, exist_ok=True)

    loader, srcs = _subset_loader(dm, args.image_id, BATCH)
    if loader is None:
        tqdm.write(f"[aviso] nenhum --image-id encontrado em {ds} split{split} pct{pct}; celula pulada")
        return
    done = 0
    for views, _ in loader:
        if done >= n_imgs:
            break
        x = AutoEncoderFlimModule._first_view(views).to(args.device, non_blocking=True)
        batch = min(x.shape[0], n_imgs - done)
        for enc, epoch, n_ch, stage, training in loaded:
            h = _heat(enc, x)
            focus, heat = _focus(h), h.cpu().numpy()
            for b in range(batch):
                src = srcs[done + b]
                png = dirs[stage] / f"heatmap_{Path(src).stem}.png"
                if not (args.skip_existing and png.exists()):
                    title = (
                        f"{ds} split{split} pct{pct} · {stage} · ép. {epoch}"
                        f"{' · EM TREINO' if training else ''}\n"
                        f"{n_ch} canais · foco10% = {focus[b]:.3f} · {Path(src).stem}"
                    )
                    _draw(png, src, heat[b], vmax, title, args.sigma, args.alpha, args.cmap,
                          args.scale)
                pbar.update(1)
        done += batch
        pbar.set_postfix_str(f"{ds} s{split} pct{pct} {done}/{n_imgs}")


def heatmap_stages(config=None, dataset=None, split=None, percentage=None, stage=None,
                   image_id=None, limit=None, sigma=6.0, alpha=1.4, scale=2, cmap="RdYlGn_r",
                   skip_existing=False, device=None, num_workers=4, dry_run=False) -> None:
    """Escreve os PNGs de heatmap das celulas pedidas (ou so conta, com ``dry_run``).

    Os antigos ``action="append"`` viraram listas: ``dataset``, ``split``,
    ``percentage``, ``stage`` e ``image_id`` recebem a lista inteira. Vazias
    (``None``) valem, como antes, "todas" — menos ``stage`` e ``image_id``, que
    valem "sem filtro". ``limit`` corta nas N primeiras imagens de cada celula,
    ``sigma``/``alpha`` sao o ajuste fino da nuvem, ``scale`` amplia o PNG de
    200x200 e ``skip_existing`` retoma sem reescrever o que ja existe.

    O que os ``choices=`` do argparse validavam agora e responsabilidade de quem
    chama: ``dataset`` em DATASETS, ``split`` em SPLITS, ``percentage`` em
    PERCENTAGES. ``config`` e um YAML opcional cujas chaves sobrescrevem os
    parametros recebidos.
    """
    # o corpo ja falava `args.x`: os parametros viram o shim, com o YAML por cima.
    params = dict(locals()); params.pop("config")
    if config:
        params.update(yaml.safe_load(Path(config).read_text()))
    args = SimpleNamespace(**params)
    # o default de --device era dinamico; resolvido aqui para nao rodar no import
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.dataset = args.dataset or DATASETS
    args.split = args.split or SPLITS
    args.percentage = args.percentage or PERCENTAGES

    plan = _plan(args)
    total = sum(n * len(stages) for _, _, _, stages, _, n in plan)

    if args.dry_run:
        by_ds: dict[str, int] = {}
        for ds, split, pct, stages, _, n in plan:
            print(f"  {ds:10s} split{split} pct{pct:2d}  {n:5d} imgs × {len(stages)} estágios "
                  f"= {n * len(stages):6d} PNGs   [{', '.join(s for s, _, _ in stages)}]")
            by_ds[ds] = by_ds.get(ds, 0) + n * len(stages)
        for ds, k in by_ds.items():
            print(f"{ds:10s} {k:7d} PNGs")
        print(f"TOTAL      {total:7d} PNGs em {len(plan)} células")
        return

    with tqdm(total=total, desc="heatmap por estágio", unit="png") as pbar:
        for ds, split, pct, stages, dm, n in plan:
            _run_cell(ds, split, pct, stages, dm, n, args, pbar)


if __name__ == "__main__":
    heatmap_stages()
