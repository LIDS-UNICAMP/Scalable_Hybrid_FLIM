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
"""Normas dos tokens de patch do DINOv2-giant (sem registradores), saida do ultimo bloco antes da LN,
nas 500 imagens de treino que `spf_grade` sorteia (draw 42). Grava `<OUT>/<base>/<imagem>.npz` com a grade
de normas e a caixa do conteudo no letterbox de 200 px. A imagem entra sem letterbox (lado maior 518,
o menor arredondado a 14), entao nenhum token cai no preenchimento. CPU.
`python A_reports/november/analise_textura/dino_pontos.py <base> <split> [<split> ...]`
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

from benchmark.lista import MODELS, dinov2
from eval.l1_sem_rotulo import train_pool

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT, SIZE, LONG, PATCH = Path("/exps/moliveira/fontes_filtros/dino_pontos"), 200, 518, 14
MEAN, STD = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def drawn(base: str, split: int) -> list:
    job = {"dataset_name": base, "split": split, "percentage": 50}
    pool = train_pool(job)
    return random.Random(42).sample(pool, min(500, len(pool)))  # = `bank_pool` sem `por_classe`


def main(base: str, splits: list) -> None:
    torch.set_num_threads(int(os.environ.get("THREADS", 16)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = dinov2(*MODELS["dinov2_giant"][:2])()[0].eval().to(dev)
    (OUT / base).mkdir(parents=True, exist_ok=True)
    for s in dict.fromkeys(x for sp in splits for x in drawn(base, sp)):
        out = OUT / base / f"{s.image_path.name}.npz"
        if out.exists():
            continue
        img = Image.open(s.image_path).convert("RGB")
        w, h = img.size
        k = LONG / max(h, w)
        gh, gw = max(1, round(h * k / PATCH)), max(1, round(w * k / PATCH))
        x = torch.from_numpy(np.asarray(img.resize((gw * PATCH, gh * PATCH), Image.BICUBIC))).permute(2, 0, 1) / 255.0
        with torch.inference_mode():
            hid = net(((x - MEAN) / STD)[None].to(dev), output_hidden_states=True).hidden_states[-1][0, 1:].cpu()
        a = SIZE / max(h, w)  # a mesma conta de `core.data.transforms.letterbox`
        ch, cw = round(h * a), round(w * a)
        np.savez(out, norm=hid.norm(dim=-1).view(gh, gw).numpy(),
                 box=np.array([(SIZE - ch) // 2, (SIZE - cw) // 2, ch, cw]))
        print(base, s.image_path.name, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
