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

"""Re-exports da destilacao: encurta o class_path do YAML, de src.modules.distillation_twolayer_module.DistillationTwoLayerModule para methods.distillation.DistillationTwoLayerModule."""

# teacher
from .frozen_teacher import FrozenTeacher

# heads
from .distillation_projection_head import DistillationProjectionHead
from .conv_distillation_projection_head import ConvDistillationProjectionHead
from .one_layer_conv_distillation_projection_head import OneLayerConvDistillationProjectionHead
from .one_layer_1x1_conv_distillation_projection_head import OneLayer1x1ConvDistillationProjectionHead
from .two_layer_1x1_conv_bn2d_distillation_projection_head import TwoLayer1x1ConvBN2dDistillationProjectionHead
from .student_classification_head import StudentClassificationHead

# losses
from .kl_distillation_loss import KLDistillationLoss
from .mse_distillation_loss import MSEDistillationLoss
from .cosine_distillation_loss import CosineDistillationLoss
from .kd_loss import kd_loss

# helpers
from .teacher_input import prepare_teacher_input

# modules lightning
from .distillation_module import DistillationModule
from .distillation_conv_module import DistillationConvModule
from .distillation_one_layer_module import DistillationOneLayerModule
from .distillation_two_layer_module import DistillationTwoLayerModule

# cli.py fica de fora de proposito: e entrypoint, nao superficie de biblioteca —
# importa-lo aqui faria todo `import methods.distillation` puxar o parser.
# DistillationLoss (alias de KLDistillationLoss em kl_distillation_loss.py) tambem
# fica fora: nome generico demais ao lado das tres losses concretas e colide com a
# leitura de "classe base"; quem depende do nome antigo importa do modulo.

__all__ = [
    "FrozenTeacher",
    "DistillationProjectionHead",
    "ConvDistillationProjectionHead",
    "OneLayerConvDistillationProjectionHead",
    "OneLayer1x1ConvDistillationProjectionHead",
    "TwoLayer1x1ConvBN2dDistillationProjectionHead",
    "StudentClassificationHead",
    "KLDistillationLoss",
    "MSEDistillationLoss",
    "CosineDistillationLoss",
    "kd_loss",
    "prepare_teacher_input",
    "DistillationModule",
    "DistillationConvModule",
    "DistillationOneLayerModule",
    "DistillationTwoLayerModule",
]
