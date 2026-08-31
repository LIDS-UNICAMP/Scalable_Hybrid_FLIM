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

"""pyift_strategy.py — carregador de imagens LABNorm2 do pyift.

So a interface e o comando: o ``import pyift`` e SEMPRE tardio (dentro da
funcao). Assim ``import flim`` funciona numa maquina sem o container, e a coisa
so quebra — com mensagem clara — na hora em que a estrategia e executada.

Decisao de backend, tomada uma unica vez no construtor:
  1. ``import pyift.pyift`` + sonda funcional              -> mode = "local"
  2. daemon do Docker + imagem resolvida na ordem
     (a) ja presente, (b) ``docker pull``, (c) ``docker build``
                                                           -> mode = "docker"
  3. os dois falharam: PIL RGB, com aviso                  -> mode = "pil"

O contrato de retorno e o mesmo nos tres modos: um float32 (H, W, 3) por
imagem, em [0, 1] — LABNorm2 quando o pyift roda, RGB quando cai no PIL.

Este modulo nao e importado por ``methods/*`` nem re-exportado em
``flim/__init__.py``: a unica fronteira e ``flim.build``.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

_HERE             = Path(__file__).resolve().parent
_PROJECT_ROOT     = _HERE.parent                 # flim/ -> raiz do repositorio
DOCKER_IMAGE_NAME = "pyift-lab-loader:latest"
# O Dockerfile.pyift e o _docker_pyift_worker.py que ele copia continuam em
# src/analysis/; esse diretorio e, ao mesmo tempo, o contexto do build.
DOCKER_CONTEXT    = _PROJECT_ROOT / "src" / "analysis"
DOCKER_DOCKERFILE = DOCKER_CONTEXT / "Dockerfile.pyift"
# Imagem remota tentada antes do build local. None pula o pull.
DOCKER_REGISTRY_IMAGE: str | None = "felipesalvagnini/pyift-lab-loader:latest"
# Maximo de imagens por invocacao do container — limita o pico de memoria do .npz.
_CHUNK_SIZE       = 500


# ── Runner local ──────────────────────────────────────────────────────────────

def _probe_local() -> bool:
    """True se pyift.pyift importa E funciona de verdade."""
    log.info("[pyift] Probing local pyift.pyift import…")
    try:
        import pyift.pyift as ift  # noqa: F401  ← IMPORT TARDIO
        _ = ift.LABNorm2_CSPACE    # sonda de atributo: confirma a extensao C carregada
        log.info("[pyift] Local pyift.pyift available.")
        return True
    except Exception as exc:
        log.info("[pyift] Local pyift.pyift unavailable: %s", exc)
        return False


def _load_local(paths: list[str]) -> list[np.ndarray]:
    """Le as imagens como LABNorm2 float32 (H, W, 3) pelo pyift nativo."""
    import pyift.pyift as ift  # ← IMPORT TARDIO

    results: list[np.ndarray] = []
    for path in paths:
        image  = ift.ReadImageByExt(str(path))
        mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
        arr    = mimage.AsNumPy().squeeze().astype(np.float32)
        results.append(arr)
    return results


# ── Runner Docker ─────────────────────────────────────────────────────────────

def _docker_daemon_available() -> bool:
    """True se o daemon do Docker responde a ``docker info``."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_image_exists() -> bool:
    """True se a imagem pyift ja esta presente localmente."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _pull_docker_image() -> bool:
    """Tenta puxar a imagem do registry. True se ela ficou disponivel local."""
    if DOCKER_REGISTRY_IMAGE is None:
        return False
    log.info("[pyift] Pulling image from registry: %s …", DOCKER_REGISTRY_IMAGE)
    try:
        result = subprocess.run(
            ["docker", "pull", "--platform", "linux/amd64", DOCKER_REGISTRY_IMAGE],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            log.info("[pyift] Pull failed (image may not exist in registry).")
            return False
        # Retag com o nome local para o comando de run ficar estavel.
        subprocess.run(
            ["docker", "tag", DOCKER_REGISTRY_IMAGE, DOCKER_IMAGE_NAME],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("[pyift] Image pulled and tagged as %s.", DOCKER_IMAGE_NAME)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        log.info("[pyift] Pull attempt failed: %s", exc)
        return False


def _build_docker_image() -> None:
    """Constroi a imagem localmente a partir do Dockerfile.pyift."""
    log.info("[pyift] Building Docker image %s locally…", DOCKER_IMAGE_NAME)
    subprocess.run(
        [
            "docker", "build",
            "--platform", "linux/amd64",
            "-t", DOCKER_IMAGE_NAME,
            str(DOCKER_CONTEXT),
            "-f", str(DOCKER_DOCKERFILE),
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    log.info("[pyift] Docker image built successfully.")


def _probe_docker() -> bool:
    """True se o Docker esta disponivel e a imagem esta pronta.

    Ordem: (1) imagem ja local, (2) pull do registry, (3) build do
    Dockerfile.pyift. Os tres falharam -> False.
    """
    log.info("[pyift] Probing Docker fallback…")

    if not _docker_daemon_available():
        log.info("[pyift] Docker daemon not running — fallback unavailable.")
        return False

    # Caminho rapido: a imagem ja esta local.
    if _docker_image_exists():
        log.info("[pyift] Docker image already present: %s.", DOCKER_IMAGE_NAME)
        return True

    # Pull do registry antes do build (mais rapido que compilar tudo).
    if _pull_docker_image():
        return True

    if not DOCKER_DOCKERFILE.is_file():
        log.warning("[pyift] Dockerfile.pyift not found at %s — cannot build.", DOCKER_DOCKERFILE)
        return False

    try:
        _build_docker_image()
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("[pyift] Docker image build failed: %s", exc)
        return False


def _load_docker_chunk(
    container_paths: list[str],
    tmp_dir: Path,
) -> list[np.ndarray]:
    """Dispara UM container para um chunk de caminhos.

    ``container_paths`` sao os caminhos vistos DENTRO do container
    (prefixados com /workspace/...); ``tmp_dir`` e o diretorio do host montado
    em /job. Devolve um float32 (H, W, 3) em [0, 1] por caminho.
    """
    paths_file   = tmp_dir / "paths.json"
    results_file = tmp_dir / "results.npz"

    with open(paths_file, "w") as f:
        json.dump(container_paths, f)

    # Descarta resultado velho, se sobrou.
    if results_file.exists():
        results_file.unlink()

    log.info(
        "[pyift] Docker: running container for %d images…", len(container_paths)
    )
    subprocess.run(
        [
            "docker", "run",
            "--rm",
            "--platform", "linux/amd64",
            "-v", f"{_PROJECT_ROOT}:/workspace:ro",
            "-v", f"{tmp_dir}:/job",
            DOCKER_IMAGE_NAME,
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if not results_file.exists():
        raise RuntimeError(
            f"Docker container finished but results.npz not found at {results_file}"
        )

    archive = np.load(str(results_file), allow_pickle=False)
    n       = len(container_paths)
    results = [archive[f"arr_{i}"] for i in range(n)]
    log.info("[pyift] Docker: received %d arrays from container.", n)
    return results


def _load_docker(host_paths: list[str]) -> list[np.ndarray]:
    """Le o lote via Docker, em chunks, para segurar a memoria."""
    # Caminho absoluto do host -> caminho relativo ao /workspace do container.
    def _to_container(p: str) -> str:
        try:
            rel = Path(p).relative_to(_PROJECT_ROOT)
        except ValueError:
            raise ValueError(
                f"Image path {p!r} is outside project root {_PROJECT_ROOT}. "
                "Docker mount only covers the project root."
            )
        return f"/workspace/{rel}"

    container_paths = [_to_container(p) for p in host_paths]
    results: list[np.ndarray] = []

    with tempfile.TemporaryDirectory(prefix="pyift_job_") as tmp:
        tmp_dir = Path(tmp)
        for chunk_start in range(0, len(container_paths), _CHUNK_SIZE):
            chunk_c = container_paths[chunk_start : chunk_start + _CHUNK_SIZE]
            log.info(
                "[pyift] Docker chunk %d–%d / %d",
                chunk_start + 1,
                chunk_start + len(chunk_c),
                len(container_paths),
            )
            results.extend(_load_docker_chunk(chunk_c, tmp_dir))

    return results


# ── Fallback PIL ──────────────────────────────────────────────────────────────

def _load_pil(paths: list[str]) -> list[np.ndarray]:
    """Le as imagens como RGB float32 (H, W, 3) pelo PIL — fallback de cor."""
    from PIL import Image as PILImage

    results: list[np.ndarray] = []
    for path in paths:
        img = PILImage.open(path).convert("RGB")
        results.append(np.array(img, dtype=np.float32) / 255.0)
    return results


# ── Estrategia ────────────────────────────────────────────────────────────────

class PyIFTStrategy:
    """Carregador local-primeiro, Docker-depois.

    Detecta o runtime uma unica vez no construtor; cada ``load_batch`` seguinte
    vai direto para o backend escolhido.

    allow_pil_fallback:
        True (default) degrada em silencio para PIL-RGB quando nem o pyift local
        nem o Docker estao disponiveis. False levanta RuntimeError no lugar.
    """

    def __init__(self, allow_pil_fallback: bool = True) -> None:
        self._mode = self._detect_mode(allow_pil_fallback)

    @staticmethod
    def _detect_mode(allow_pil_fallback: bool) -> str:
        if _probe_local():
            return "local"

        if _probe_docker():
            return "docker"

        if allow_pil_fallback:
            log.warning(
                "[pyift] Neither local pyift nor Docker is available. "
                "Falling back to PIL-RGB loading — colour space will differ "
                "from the FLIM training pipeline (LABNorm2 expected)."
            )
            return "pil"

        raise RuntimeError(
            "pyift is not available locally and Docker fallback failed. "
            "Install the LIDS-UNICAMP pyift wheel or start Docker daemon."
        )

    @property
    def mode(self) -> str:
        """Backend ativo: ``'local'`` | ``'docker'`` | ``'pil'``."""
        return self._mode

    def load_batch(self, paths: Sequence[str]) -> list[np.ndarray]:
        """Le ``paths`` e devolve um float32 (H, W, 3) por imagem.

        local  -> pyift.pyift nativo, import tardio a cada chamada.
        docker -> uma invocacao (em chunks) do container para o lote inteiro.
        pil    -> PIL RGB como float32.
        """
        path_list = list(paths)

        if self._mode == "local":
            log.info("[pyift] Loading %d images via local pyift.", len(path_list))
            return _load_local(path_list)

        if self._mode == "docker":
            log.info(
                "[pyift] Loading %d images via Docker fallback.", len(path_list)
            )
            return _load_docker(path_list)

        log.info("[pyift] Loading %d images via PIL (RGB fallback).", len(path_list))
        return _load_pil(path_list)
