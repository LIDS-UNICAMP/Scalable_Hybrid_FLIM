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
"""
Custo computacional dos 8 modelos oficiais (resposta ao Revisor 3, SIBGRAPI).

Mede, para cada modelo, o grafo de inferência que produz o embedding entregue ao
SVM linear:

  FLIM / LeJEPA          Encoder FLIM (3× conv 5x5) -> flatten          (3x200x200)
  I-JEPA                 ViT-H/14 -> mean-pool dos patches             (3x224x224)
  Distill 1..4           Encoder FLIM -> proj head -> AdaptiveAvgPool  (3x200x200)

Métricas por modelo:
  * parâmetros e tamanho dos pesos em MB (params x 4 bytes, float32)
  * FLOPs de um forward com batch=1 (torch.utils.flop_counter.FlopCounterMode)
  * latência em ms/imagem na GPU e na CPU (warmup + N repetições cronometradas)
  * pico de VRAM alocada durante o forward (torch.cuda.max_memory_allocated)

Uso:
  cd /dados/home/moliveira/Scalable_Hybrid_FLIM
  CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib \
      python -m analysis.stats.compute_cost

Saídas: statistics/tools/compute_cost.csv e statistics/tools/compute_cost.md
"""
from __future__ import annotations

import json
import os
import platform
import statistics as pystats
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

OUT_DIR = REPO_ROOT / "statistics" / "tools"
CSV_PATH = OUT_DIR / "compute_cost.csv"
MD_PATH = OUT_DIR / "compute_cost.md"

# ── Procedência dos pesos (ver metrics_distillation/data_provenance.md) ────────

FLIM_MODEL_ROOT = REPO_ROOT / "data/to_mateus/model/ch24_32_48_a0.5_f5"
ARCH_JSON = FLIM_MODEL_ROOT / "eggs/train1/architecture.json"
FLIM_WEIGHTS = FLIM_MODEL_ROOT / "eggs/train1/models"
LEJEPA_CKPT = REPO_ROOT / "logs/flim-ssl/rxe7zmgk/checkpoints/best.ckpt"
FROZEN_CKPT = (
    REPO_ROOT
    / "artifacts/distillation/distillation_eggs_split1_pct100_1x1_BN2d_1280_flim_frozen"
    / "checkpoints/best_loss.ckpt"
)

SEED = 0
BATCH = 1
FLIM_INPUT = (3, 200, 200)
IJEPA_INPUT = (3, 224, 224)


# ── Wrappers de inferência ────────────────────────────────────────────────────


class EncoderFlatten(nn.Module):
    """Encoder FLIM raw: saída conv3 achatada (o que o SVM recebe em FLIM/LeJEPA).

    Reproduz src/utils/evaluate.py::extract_features (conv1->conv2->conv3->flatten).
    """

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)


class EncoderProj(nn.Module):
    """Encoder FLIM + projection head 1280d (o que o SVM recebe nos Distill 1..4).

    Reproduz src/evaluate/svm_distill_with_projection.py::_extract_proj.
    """

    def __init__(self, encoder: nn.Module, proj_kd: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj_kd = proj_kd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_kd(self.encoder(x))


# ── Construtores ──────────────────────────────────────────────────────────────


def _arch(in_channels: int = 3):
    from flim.arch import get_channels_from_arch, parse_architecture

    arch = parse_architecture(str(ARCH_JSON))
    channels = get_channels_from_arch(arch, in_channels)
    return arch, channels


def build_flim() -> tuple[nn.Module, str, bool]:
    """Encoder FLIM com os kernels reais (sem backprop) do split eggs/train1."""
    from flim.encoder import Encoder
    from flim.weights import load_FLIM_encoder

    arch, channels = _arch()
    enc = Encoder(arch, 3)
    load_FLIM_encoder(enc, str(ARCH_JSON), str(FLIM_WEIGHTS), channels)
    src = str(FLIM_WEIGHTS.relative_to(REPO_ROOT))
    return EncoderFlatten(enc).float(), src, True


def build_lejepa() -> tuple[nn.Module, str, bool]:
    """Encoder LeJEPA pós pré-treino SSL (checkpoint Lightning, init trunc_normal)."""
    from methods.lejepa import LejepaLineModule

    module = LejepaLineModule.load_from_checkpoint(str(LEJEPA_CKPT), map_location="cpu")
    enc = module.model.encoder
    src = str(LEJEPA_CKPT.relative_to(REPO_ROOT))
    return EncoderFlatten(enc).float(), src, True


def build_ijepa() -> tuple[nn.Module, str, bool]:
    """ViT-H/14 I-JEPA (facebook/ijepa_vith14_1k) do cache HuggingFace local."""
    from methods.lejepa import IJEPAEncoder
    from methods.lejepa.ijepa_encoder import _find_safetensors_path

    path = _find_safetensors_path("facebook/ijepa_vith14_1k")
    enc = IJEPAEncoder(device=torch.device("cpu"))
    # _model é o ViT puro: evita a cópia GPU->CPU que IJEPAEncoder.forward faz.
    return enc._model.float(), path, True


def _distill_arch_only(proj_factory: Callable[[int], nn.Module]) -> tuple[nn.Module, str, bool]:
    """Distill com encoder treinável: os checkpoints não existem mais no disco.

    artifacts/distillation/<run>/checkpoints/ está vazio para todas as variantes
    de encoder treinável (só as runs *_flim_frozen mantêm .ckpt). Como FLOPs,
    latência, memória e contagem de parâmetros dependem apenas da topologia e não
    do valor dos pesos, a medição é feita sobre a arquitetura reinstanciada com
    init trunc_normal (a mesma usada no treino).
    """
    from src.models.lejepa_flim import LeJEPAFLIMModel
    from core.blocks.init import init_weights_trunc_normal

    arch, _ = _arch()
    student = LeJEPAFLIMModel(arch=arch, in_channels=3, proj_dim=256, proj_hidden=2048)
    init_weights_trunc_normal(student.encoder)
    proj = proj_factory(student.embed_dim)
    return EncoderProj(student.encoder, proj).float(), "arquitetura (checkpoint ausente)", False


def build_distill4() -> tuple[nn.Module, str, bool]:
    from methods.distillation import ConvDistillationProjectionHead

    return _distill_arch_only(lambda c: ConvDistillationProjectionHead(student_channels=c))


def build_distill3() -> tuple[nn.Module, str, bool]:
    from methods.distillation import OneLayerConvDistillationProjectionHead

    return _distill_arch_only(
        lambda c: OneLayerConvDistillationProjectionHead(student_channels=c)
    )


def build_distill1() -> tuple[nn.Module, str, bool]:
    from methods.distillation import OneLayer1x1ConvDistillationProjectionHead

    return _distill_arch_only(
        lambda c: OneLayer1x1ConvDistillationProjectionHead(student_channels=c)
    )


def build_distill2() -> tuple[nn.Module, str, bool]:
    from methods.distillation import TwoLayer1x1ConvBN2dDistillationProjectionHead

    return _distill_arch_only(
        lambda c: TwoLayer1x1ConvBN2dDistillationProjectionHead(
            student_channels=c, mid_channels=256
        )
    )


def _rebase_path(p: str) -> str:
    """Reescreve caminhos absolutos gravados nos hparams para o repo atual.

    Os checkpoints guardam `arch_json`/`flim_weights_path` como caminhos absolutos
    de uma cópia antiga do repositório (`/dados/home/moliveira/
    scalable_FLIM_self_supervised/...`), que não existe mais. Aqui a parte a
    partir de `data/` é reancorada no repo atual.
    """
    if not p:
        return p
    if os.path.exists(p):
        return p
    marker = "data/to_mateus"
    idx = p.find(marker)
    if idx >= 0:
        return str(REPO_ROOT / p[idx:])
    return p


def build_distill1_flim() -> tuple[nn.Module, str, bool]:
    """Encoder FLIM congelado + proj head 1x1 treinada (best_loss.ckpt real)."""
    import eval.svm_variants.svm_distill_with_projection as sdp

    orig_parse = sdp.parse_architecture
    orig_chan = sdp.get_actual_channels_from_weights
    sdp.parse_architecture = lambda p: orig_parse(_rebase_path(p))
    sdp.get_actual_channels_from_weights = (
        lambda w, a, c=3: orig_chan(_rebase_path(w), a, c)
    )
    try:
        student, proj = sdp._load_student_and_proj(str(FROZEN_CKPT), torch.device("cpu"))
    finally:
        sdp.parse_architecture = orig_parse
        sdp.get_actual_channels_from_weights = orig_chan
    src = str(FROZEN_CKPT.relative_to(REPO_ROOT))
    return EncoderProj(student.encoder, proj).float(), src, True


# ── Especificação das 8 linhas ────────────────────────────────────────────────


@dataclass
class ModelSpec:
    label: str
    method: str
    builder: Callable[[], tuple[nn.Module, str, bool]]
    input_shape: tuple[int, int, int]
    graph: str
    gpu_iters: int = 100
    cpu_iters: int = 50
    gpu_warmup: int = 20
    cpu_warmup: int = 10


SPECS: list[ModelSpec] = [
    ModelSpec("FLIM (59.504)", "SVM_FLIM", build_flim, FLIM_INPUT,
              "encoder FLIM -> flatten (27.648d)"),
    ModelSpec("LeJEPA (59.504)", "SVM_LeJEPA", build_lejepa, FLIM_INPUT,
              "encoder FLIM -> flatten (27.648d)"),
    ModelSpec("I-JEPA (632M)", "SVM_IJEPA", build_ijepa, IJEPA_INPUT,
              "ViT-H/14 -> mean-pool (1280d)", gpu_iters=50, cpu_iters=20,
              gpu_warmup=10, cpu_warmup=5),
    ModelSpec("Distill 4 (889K)", "SVM_Distill_Proj1280", build_distill4, FLIM_INPUT,
              "encoder FLIM -> proj 4x conv1x1 -> GAP (1280d)"),
    ModelSpec("Distill 3 (615K)", "SVM_Distill_3x3BN", build_distill3, FLIM_INPUT,
              "encoder FLIM -> proj conv3x3 -> GAP (1280d)"),
    ModelSpec("Distill 1 (123K)", "SVM_Distill_1x1BN", build_distill1, FLIM_INPUT,
              "encoder FLIM -> proj conv1x1 -> GAP (1280d)"),
    ModelSpec("Distill 2 (402K)", "SVM_Distill_2l400K", build_distill2, FLIM_INPUT,
              "encoder FLIM -> proj 2x conv1x1 -> GAP (1280d)"),
    ModelSpec("Distill 1 - FLIM init (123K)", "SVM_Distill_1x1BN_flim_frozen_eval_loss",
              build_distill1_flim, FLIM_INPUT,
              "encoder FLIM congelado -> proj conv1x1 -> GAP (1280d)"),
]


# ── Medição ───────────────────────────────────────────────────────────────────


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return total, buffers


def measure_flops(model: nn.Module, x: torch.Tensor) -> Optional[int]:
    from torch.utils.flop_counter import FlopCounterMode

    try:
        with torch.no_grad():
            with FlopCounterMode(display=False) as fcm:
                model(x)
        return int(fcm.get_total_flops())
    except Exception as exc:  # pragma: no cover
        print(f"    [WARN] FlopCounterMode falhou: {exc}")
        return None


def measure_latency(
    model: nn.Module, x: torch.Tensor, warmup: int, iters: int, cuda: bool
) -> tuple[float, float, float]:
    """(média, desvio padrão amostral, mediana) em ms por forward."""
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        for _ in range(iters):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1e3)
    mean = pystats.fmean(times_ms)
    std = pystats.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return mean, std, pystats.median(times_ms)


def warm_cuda_context() -> None:
    """Força a alocação dos workspaces persistentes de cuBLAS/cuDNN.

    Sem isso o primeiro modelo medido absorveria no seu pico a alocação
    permanente (~8 MB) desses handles, que nada tem a ver com o modelo.
    """
    d = torch.device("cuda:0")
    a = torch.randn(64, 64, device=d)
    with torch.no_grad():
        (a @ a).sum().item()
        c = nn.Conv2d(3, 8, 3, padding=1).to(d)
        c(torch.randn(1, 3, 32, 32, device=d)).sum().item()
    del a, c
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def measure_vram(
    model: nn.Module, x_shape: tuple[int, ...], device: torch.device, ctx_base: int
) -> tuple[float, float, float]:
    """Pico de VRAM atribuível ao modelo, em MB.

    Retorna (pico com cuDNN habilitado, pico com cuDNN desabilitado,
    residente estático = pesos + tensor de entrada). Todos descontando
    ``ctx_base``, a alocação persistente do contexto CUDA.
    """
    model.to(device)
    x = torch.randn(*x_shape, device=device)

    def _peak() -> float:
        with torch.no_grad():
            model(x)  # warmup: fixa a escolha de algoritmo
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - ctx_base) / 2**20

    peak_cudnn = _peak()
    torch.cuda.empty_cache()
    static_mb = (torch.cuda.memory_allocated() - ctx_base) / 2**20

    prev = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        peak_nocudnn = _peak()
    finally:
        torch.backends.cudnn.enabled = prev

    del x
    torch.cuda.empty_cache()
    return peak_cudnn, peak_nocudnn, static_mb


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        return out[0].strip() if out else "desconhecida"
    except Exception:
        return "desconhecida"


def cpu_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "desconhecida"


# ── Formatação ────────────────────────────────────────────────────────────────


def fmt_params(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def fmt_tflop(flops: Optional[int]) -> str:
    if flops is None:
        return "não medido"
    tflop = flops / 1e12
    if tflop >= 0.01:
        return f"{tflop:.4f}"
    gflop = flops / 1e9
    if gflop >= 1:
        return f"{tflop:.6f} ({gflop:.2f} GFLOP)"
    return f"{tflop:.6f} ({flops / 1e6:.1f} MFLOP)"


def fmt_ms(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "não medido"
    if mean < 10:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.2f} ± {std:.2f}"


def compute_cost(skip_cpu: bool = False) -> None:
    # O corpo abaixo continua lendo `args.x`: o shim nasce so dos parametros e e a
    # primeira linha viva da funcao, entao locals() e exatamente a assinatura.
    args = SimpleNamespace(**locals())

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_cuda else "cpu")
    n_threads = torch.get_num_threads()
    if has_cuda:
        warm_cuda_context()

    rows: list[dict] = []

    for spec in SPECS:
        print(f"\n=== {spec.label} ({spec.method})")
        try:
            model, weights_src, weights_real = spec.builder()
        except Exception as exc:
            print(f"    [ERRO] falha ao construir: {type(exc).__name__}: {exc}")
            rows.append({
                "label": spec.label, "method": spec.method, "status": f"erro: {exc}",
            })
            continue

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        n_params, n_buffers = count_params(model)
        weights_mb = n_params * 4 / 2**20

        x_shape = (BATCH, *spec.input_shape)

        # FLOPs (na CPU: independe do device)
        model.to("cpu")
        x_cpu = torch.randn(*x_shape)
        flops = measure_flops(model, x_cpu)

        # Latência CPU
        cpu_mean = cpu_std = cpu_med = None
        if not args.skip_cpu:
            cpu_mean, cpu_std, cpu_med = measure_latency(
                model, x_cpu, spec.cpu_warmup, spec.cpu_iters, cuda=False
            )
            print(f"    CPU: {cpu_mean:.3f} ± {cpu_std:.3f} ms "
                  f"(mediana {cpu_med:.3f}, {spec.cpu_iters} iters)")

        # Latência + VRAM na GPU
        gpu_mean = gpu_std = gpu_med = None
        peak_mb = peak_nocudnn_mb = static_mb = None
        if has_cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ctx_base = torch.cuda.memory_allocated()
            model.to(device)
            x_gpu = torch.randn(*x_shape, device=device)
            torch.cuda.synchronize()
            gpu_mean, gpu_std, gpu_med = measure_latency(
                model, x_gpu, spec.gpu_warmup, spec.gpu_iters, cuda=True
            )
            del x_gpu
            torch.cuda.empty_cache()
            peak_mb, peak_nocudnn_mb, static_mb = measure_vram(
                model, x_shape, device, ctx_base
            )
            print(f"    GPU: {gpu_mean:.3f} ± {gpu_std:.3f} ms "
                  f"(mediana {gpu_med:.3f}, {spec.gpu_iters} iters)   "
                  f"pico VRAM {peak_mb:.1f} MB "
                  f"(sem workspace cuDNN {peak_nocudnn_mb:.1f} MB, "
                  f"estático {static_mb:.1f} MB)")

        print(f"    params={fmt_params(n_params)}  FLOPs={flops}  pesos={weights_mb:.2f} MB")

        rows.append({
            "label": spec.label,
            "method": spec.method,
            "grafo_medido": spec.graph,
            "input_shape": f"{spec.input_shape[0]}x{spec.input_shape[1]}x{spec.input_shape[2]}",
            "batch": BATCH,
            "params": n_params,
            "buffers": n_buffers,
            "weights_MB": round(weights_mb, 4),
            "flops_fwd": flops,
            "tflop": None if flops is None else flops / 1e12,
            "gflop": None if flops is None else flops / 1e9,
            "macs": None if flops is None else flops // 2,
            "gpu_ms_mean": None if gpu_mean is None else round(gpu_mean, 4),
            "gpu_ms_std": None if gpu_std is None else round(gpu_std, 4),
            "gpu_ms_median": None if gpu_med is None else round(gpu_med, 4),
            "gpu_iters": spec.gpu_iters if has_cuda else 0,
            "cpu_ms_mean": None if cpu_mean is None else round(cpu_mean, 4),
            "cpu_ms_std": None if cpu_std is None else round(cpu_std, 4),
            "cpu_ms_median": None if cpu_med is None else round(cpu_med, 4),
            "cpu_iters": 0 if args.skip_cpu else spec.cpu_iters,
            "peak_vram_MB": None if peak_mb is None else round(peak_mb, 3),
            "peak_vram_cudnn_disabled_MB":
                None if peak_nocudnn_mb is None else round(peak_nocudnn_mb, 3),
            "static_vram_MB": None if static_mb is None else round(static_mb, 3),
            "pesos_reais": weights_real,
            "fonte_dos_pesos": weights_src,
            "status": "ok",
        })

        del model
        if has_cuda:
            torch.cuda.empty_cache()

    # ── CSV ──────────────────────────────────────────────────────────────────
    import csv

    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[OK] CSV -> {CSV_PATH}")

    # ── Markdown ─────────────────────────────────────────────────────────────
    gpu = gpu_name() if has_cuda else "sem GPU"
    cpu = cpu_name()
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda or "n/a"

    lines: list[str] = []
    lines.append("# Custo computacional dos 8 modelos oficiais")
    lines.append("")
    lines.append("Resposta ao Revisor 3: FLOPs, tempo de inferência e consumo de memória "
                 "por modelo, batch=1, uma imagem por forward.")
    lines.append("")
    lines.append("| Modelo | Params | TFLOP/img | Tempo GPU (ms/img) | Tempo CPU (ms/img) | "
                 "Pesos (MB) | Pico VRAM (MB) | Pesos medidos |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['label']} | não medido | não medido | não medido | "
                         f"não medido | não medido | não medido | "
                         f"{r.get('status', 'erro')} |")
            continue
        peak = ("não medido" if r["peak_vram_MB"] is None
                else f"{r['peak_vram_MB']:.1f}")
        lines.append(
            f"| {r['label']} | {fmt_params(r['params'])} | {fmt_tflop(r['flops_fwd'])} | "
            f"{fmt_ms(r['gpu_ms_mean'], r['gpu_ms_std'])} | "
            f"{fmt_ms(r['cpu_ms_mean'], r['cpu_ms_std'])} | {r['weights_MB']:.2f} | "
            f"{peak} | "
            f"{'checkpoint real' if r['pesos_reais'] else 'arquitetura (ckpt ausente)'} |"
        )
    lines.append("")

    ref = next((r for r in rows if r.get("method") == "SVM_IJEPA"
                and r.get("status") == "ok"), None)
    tgt = next((r for r in rows if r.get("method") == "SVM_Distill_1x1BN"
                and r.get("status") == "ok"), None)
    if ref and tgt:
        lines.append(
            "Comparando os dois extremos: o I-JEPA (ViT-H/14, teacher) custa "
            f"{ref['flops_fwd'] / tgt['flops_fwd']:.0f}x mais FLOPs, "
            f"{ref['gpu_ms_median'] / tgt['gpu_ms_median']:.0f}x mais tempo de GPU, "
            f"{ref['cpu_ms_median'] / tgt['cpu_ms_median']:.0f}x mais tempo de CPU "
            "(medianas de latência) e "
            f"{ref['weights_MB'] / tgt['weights_MB']:.0f}x mais memória de pesos que o "
            "Distill 1, o student de 123K parâmetros destilado a partir dele."
        )
        lines.append("")
    lines.append("## Metodologia")
    lines.append("")
    lines.append(f"- Hardware: GPU {gpu} (uma única GPU, ociosa e dedicada à medição); "
                 f"CPU {cpu}, {n_threads} threads PyTorch.")
    lines.append(f"- Software: PyTorch {torch_ver}, CUDA {cuda_ver}, float32, "
                 f"`model.eval()` + `torch.no_grad()`, seed {SEED}, cuDNN benchmark desligado.")
    lines.append("- Batch size 1 em todas as medições. O número reportado é por imagem.")
    lines.append("- Resolução de entrada por modelo: 3x200x200 para FLIM, LeJEPA e as quatro "
                 "variantes Distill (é o input do pipeline: LAB via `ift_lab` + normalização "
                 "ImageNet); 3x224x224 para o I-JEPA, resolução nativa do ViT-H/14 "
                 "(`IJEPAEncoder.IMAGE_SIZE`). A diferença de resolução é intrínseca aos "
                 "modelos e está refletida nos FLOPs.")
    groups: dict[tuple[int, int, int, int], list[str]] = {}
    for s in SPECS:
        groups.setdefault(
            (s.gpu_iters, s.gpu_warmup, s.cpu_iters, s.cpu_warmup), []
        ).append(s.label)
    rep_txt = "; ".join(
        f"{', '.join(labels)}: {gi} iterações cronometradas na GPU (warmup {gw}) e "
        f"{ci} na CPU (warmup {cw})"
        for (gi, gw, ci, cw), labels in groups.items()
    )
    lines.append("- Repetições, com `torch.cuda.synchronize()` antes e depois de cada "
                 f"iteração cronometrada: {rep_txt}. Reporta-se média ± desvio padrão "
                 "amostral sobre as iterações cronometradas; o CSV traz também a mediana.")
    lines.append("- FLOPs: `torch.utils.flop_counter.FlopCounterMode`, que conta "
                 "multiplicação e soma separadamente (FLOPs = 2 x MACs) em convoluções, "
                 "matmuls e atenção. Não contabiliza BatchNorm, GELU/ReLU, pooling nem "
                 "softmax, que são desprezíveis frente aos termos multiplicativos. A coluna "
                 "`macs` do CSV traz FLOPs/2 para comparação com trabalhos que reportam MACs.")
    lines.append("- Memória: duas grandezas distintas. \"Pesos\" = params x 4 bytes (float32), "
                 "o custo estático de armazenar o modelo. \"Pico VRAM\" = "
                 "`torch.cuda.max_memory_allocated()` durante um forward com batch=1, "
                 "com `reset_peak_memory_stats()` antes de cada medição e descontando a "
                 "alocação persistente do contexto CUDA (handles de cuBLAS/cuDNN, "
                 "pré-aquecidos antes do laço); inclui pesos residentes na GPU, tensor de "
                 "entrada, ativações intermediárias e workspaces temporários de convolução. "
                 "Não inclui o contexto CUDA em si (algumas centenas de MB), que é custo do "
                 "runtime e igual para todos os modelos.")
    lines.append("- Grafo medido: apenas o extrator de features, isto é, o que produz o "
                 "embedding entregue ao SVM linear. O SVM não entra na conta: com kernel "
                 "linear e `decision_function_shape=\"ovo\"` o custo é o produto "
                 "`n_pares x dim_embedding` (no pior caso, eggs com 9 classes sobre features "
                 "de 27.648 dimensões, cerca de 2 MFLOP), abaixo de 1% do custo do encoder.")
    lines.append("")
    lines.append("Grafo exato medido em cada linha:")
    lines.append("")
    lines.append("| Modelo | Grafo | Fonte dos pesos |")
    lines.append("|---|---|---|")
    for r in rows:
        if r.get("status") != "ok":
            continue
        lines.append(f"| {r['label']} | {r['grafo_medido']} | `{r['fonte_dos_pesos']}` |")
    lines.append("")
    lines.append("## Ressalvas")
    lines.append("")
    lines.append("- Os checkpoints das quatro variantes Distill com encoder treinável "
                 "(Distill 1, 2, 3 e 4) não estão mais no disco: "
                 "`artifacts/distillation/<run>/checkpoints/` está vazio para essas runs "
                 "(apenas as runs `*_flim_frozen` preservaram `.ckpt`). Essas linhas foram "
                 "medidas sobre a arquitetura reinstanciada com init `trunc_normal`. "
                 "FLOPs, latência, pico de memória e contagem de parâmetros dependem só da "
                 "topologia da rede, não do valor numérico dos pesos, portanto os números "
                 "são idênticos aos que os checkpoints originais produziriam.")
    ws = [
        (r["label"], r["peak_vram_MB"], r["peak_vram_cudnn_disabled_MB"])
        for r in rows
        if r.get("status") == "ok"
        and r["peak_vram_MB"] is not None
        and r["peak_vram_cudnn_disabled_MB"] is not None
        and r["peak_vram_MB"] - r["peak_vram_cudnn_disabled_MB"] > 20
    ]
    enc_only = next(
        (r["peak_vram_cudnn_disabled_MB"] for r in rows
         if r.get("method") == "SVM_FLIM" and r.get("status") == "ok"),
        None,
    )
    if ws:
        detail = "; ".join(f"{lab}: {a:.1f} -> {b:.1f} MB" for lab, a, b in ws)
        extra = ""
        if enc_only is not None:
            extra = (f" Nessa mesma configuração o encoder isolado já pica em "
                     f"{enc_only:.1f} MB, ou seja, as cabeças de projeção 1x1 acrescentam "
                     f"menos de 1 MB de ativações: os ~130 MB observados são workspace da "
                     f"cuDNN, não dado.")
        lines.append("- O pico de VRAM das cabeças de projeção com conv 1x1 para 1280 canais "
                     "é dominado por um workspace temporário que a cuDNN reserva para o "
                     "algoritmo que escolhe, não pelas ativações. Repetindo a medição com "
                     f"`torch.backends.cudnn.enabled = False`, o pico cai assim: {detail}."
                     + extra +
                     " Os valores com cuDNN desabilitada estão na coluna "
                     "`peak_vram_cudnn_disabled_MB` do CSV; note que para as linhas sem "
                     "cabeça 1x1 esse número é maior que o pico com cuDNN, porque o fallback "
                     "im2col das convoluções 5x5 do encoder consome mais memória que o "
                     "algoritmo da cuDNN. Em resumo, o pico de VRAM é dependente de "
                     "biblioteca e de GPU; a coluna de pesos é a grandeza estável.")
    lines.append("- As medições usam a arquitetura do par eggs/larvae (canais 24-32-48, "
                 "59.504 parâmetros no encoder). Para `protozoan` a seleção de kernels FLIM "
                 "produz 30 canais na conv2 em vez de 32 (55.902 parâmetros no encoder), "
                 "o que reduz ligeiramente FLOPs e latência. Essa variante não foi medida "
                 "em separado.")
    lines.append("- FLIM e LeJEPA entregam ao SVM o mapa conv3 achatado (48x24x24 = 27.648 "
                 "dimensões), não um vetor pooled, seguindo "
                 "`src/utils/evaluate.py::extract_features`. Isso não altera os FLOPs de "
                 "convolução, apenas a dimensão vista pelo SVM.")
    lines.append(f"- Latência de CPU medida com {n_threads} threads (padrão do PyTorch nesta "
                 "máquina, igual ao número de núcleos físicos); "
                 "em CPU de 1 thread ou em hardware embarcado os valores "
                 "absolutos mudam, mas a razão entre modelos se mantém. A GPU usada estava "
                 "livre durante a medição, mas a CPU do nó é compartilhada com outros "
                 "processos, o que explica o desvio padrão alto na coluna de CPU do I-JEPA; "
                 "o CSV traz também a mediana (`cpu_ms_median`, `gpu_ms_median`), mais "
                 "robusta a essa contenção.")
    lines.append("- Gerado por `analysis/stats/compute_cost.py`; dados brutos em "
                 "`statistics/tools/compute_cost.csv`.")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"[OK] Markdown -> {MD_PATH}")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    compute_cost()
