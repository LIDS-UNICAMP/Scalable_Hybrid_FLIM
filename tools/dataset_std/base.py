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

"""base.py — Contrato comum dos padronizadores de dataset.

Cada dataset implementa apenas discover(); a base cuida de numerar as classes,
renomear para {class_id:06d}_{image_id:08d}, mover os arquivos e escrever o
manifest que o tools/make_splits.py consome.
"""

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

REGISTRY = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class RawItem:
    image: Path
    label: str
    group: str | None = None
    mask: Path | None = None
    covariates: dict = field(default_factory=dict)


class DatasetStandardizer:
    name = "override me"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["out_root"]) / cfg["name"]

    def discover(self):
        raise NotImplementedError

    # -- a base a partir daqui ------------------------------------------------

    def _resize_to(self):
        return (self.cfg.get("preprocessing") or {}).get("resize_to")

    def _place(self, src: Path, dst: Path, size):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if size is None:
            shutil.move(str(src), str(dst))          # move, nunca copia
            return
        from PIL import Image
        with Image.open(src) as im:
            im.resize(tuple(size)).save(dst)
        src.unlink()

    def run(self):
        if (self.out / "manifest.json").exists() and not self.cfg.get("force"):
            print(f"[skip] {self.cfg['name']}: ja organizado")
            return self.out

        items = list(tqdm(self.discover(), desc=f"discover {self.cfg['name']}", unit="img"))
        if not items:
            raise RuntimeError(f"{self.cfg['name']}: discover() nao retornou nada")

        labels = sorted({it.label for it in items})
        class_id = {lab: i + 1 for i, lab in enumerate(labels)}

        # ordem estavel por classe -> image_id reprodutivel entre execucoes
        items.sort(key=lambda it: (class_id[it.label], str(it.image)))

        size = self._resize_to()
        seq, records = {}, []
        for it in tqdm(items, desc=f"move {self.cfg['name']}", unit="img"):
            cid = class_id[it.label]
            seq[cid] = seq.get(cid, 0) + 1
            stem = f"{cid:06d}_{seq[cid]:08d}"
            fname = stem + it.image.suffix.lower()
            self._place(it.image, self.out / "orig" / fname, size)
            if it.mask is not None:
                self._place(it.mask, self.out / "label" / fname, size)
            records.append({
                "file": fname,
                "class_id": cid,
                "label": it.label,
                "group": it.group or stem,   # sem grupo real -> cada imagem e o seu grupo
                "covariates": it.covariates,
            })

        manifest = {
            "dataset": self.cfg["name"],
            "classes": {str(class_id[lab]): lab for lab in labels},
            "has_mask": any(it.mask is not None for it in items),
            "items": records,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        (self.out / "classes.json").write_text(json.dumps(manifest["classes"], indent=1))
        return self.out
