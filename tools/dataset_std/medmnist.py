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

"""medmnist.py — discover() dos subsets 2D do MedMNIST para o padronizador.

Layout de origem: <subset>/<train|val|test>/<class_index>/*.png, com class_index
sendo uma pasta de inteiro puro. Uma instancia trata UM subset, escolhido por
cfg["subset"].

So os subsets ja extraidos e 2D entram. nodulemnist3d/organmnist3d ficam de fora:
em disco sao volumes (28,28,28) em .npy, nao imagem. chestmnist so existe como
.npz e nada aqui extrai npz.

Sem mascara e sem grupo de leakage: os patient ids do MedMNIST nao vem em disco.
O split original vira covariavel para auditoria depois do re-split.
"""

from pathlib import Path

from .base import DatasetStandardizer, RawItem, register

SPLITS = ("train", "val", "test")

# nomes oficiais do label dict do MedMNIST v2, em snake_case; subset ausente
# daqui nao e 2D/extraido e o KeyError em discover() denuncia o YAML errado.
# retinamnist_128 vem com tupla vazia: o MedMNIST publica so o grau "0".."4" da
# retinopatia, sem nome legivel, entao cai no class_<idx>.
LABELS = {
    "bloodmnist": (
        "basophil",
        "eosinophil",
        "erythroblast",
        "immature_granulocytes",
        "lymphocyte",
        "monocyte",
        "neutrophil",
        "platelet",
    ),
    "breastmnist": ("malignant", "normal_benign"),
    "dermamnist": (
        "actinic_keratoses_and_intraepithelial_carcinoma",
        "basal_cell_carcinoma",
        "benign_keratosis_like_lesions",
        "dermatofibroma",
        "melanoma",
        "melanocytic_nevi",
        "vascular_lesions",
    ),
    "octmnist": (
        "choroidal_neovascularization",
        "diabetic_macular_edema",
        "drusen",
        "normal",
    ),
    "organsmnist": (
        "bladder",
        "femur_left",
        "femur_right",
        "heart",
        "kidney_left",
        "kidney_right",
        "liver",
        "lung_left",
        "lung_right",
        "pancreas",
        "spleen",
    ),
    "retinamnist_128": (),
}


@register
class MedMNIST(DatasetStandardizer):
    name = "medmnist"

    def discover(self):
        root = Path(self.cfg["root"]).expanduser().resolve()
        subset = self.cfg["subset"]
        names = LABELS[subset]
        for split in SPLITS:
            for class_dir in sorted((root / subset / split).iterdir()):
                if not class_dir.is_dir():
                    continue
                idx = int(class_dir.name)
                label = names[idx] if idx < len(names) else f"class_{idx}"
                for img in sorted(class_dir.glob("*.png")):
                    yield RawItem(
                        image=img,  # absoluto: nome de arquivo repete entre splits/classes
                        label=label,
                        covariates={"orig_split": split},
                    )
