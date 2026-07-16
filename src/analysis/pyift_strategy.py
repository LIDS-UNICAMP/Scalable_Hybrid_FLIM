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

"""pyift_strategy.py — Local-first, Docker-fallback image loader for pyift LABNorm2.

Public API
----------
PyiftLoader
    .mode         → "local" | "docker" | "pil"
    .load_batch(paths) → list[np.ndarray]
        Each array is float32 (H, W, 3) LABNorm2 in [0, 1].

Execution decision (made once at construction):
  1. Try ``import pyift.pyift`` + a quick functional probe.
     → mode = "local" on success.
  2. If local fails, check Docker daemon and resolve the image via:
       a. already present locally
       b. ``docker pull`` from DOCKER_REGISTRY_IMAGE
       c. ``docker build`` from Dockerfile.pyift
     → mode = "docker" on success.
  3. If both fail, fall back to PIL-RGB loading (with a warning).
     → mode = "pil".

The caller receives the same list[np.ndarray] contract regardless of mode.
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

# ── Constants ─────────────────────────────────────────────────────────────────

_HERE             = Path(__file__).resolve().parent
_PROJECT_ROOT     = _HERE.parent.parent          # src/analysis → project root
DOCKER_IMAGE_NAME = "pyift-lab-loader:latest"
DOCKER_DOCKERFILE = _HERE / "Dockerfile.pyift"
# Remote registry image to attempt pulling before building locally.
# Set to None to skip pull and go straight to local build.
DOCKER_REGISTRY_IMAGE: str | None = "felipesalvagnini/pyift-lab-loader:latest"
# Max images per Docker invocation — limits peak .npz memory usage.
_CHUNK_SIZE       = 500


# ── Local runner ──────────────────────────────────────────────────────────────

def _probe_local() -> bool:
    """Return True if pyift.pyift is importable AND functionally usable."""
    log.info("[pyift] Probing local pyift.pyift import…")
    try:
        import pyift.pyift as ift  # noqa: F401
        _ = ift.LABNorm2_CSPACE    # attribute probe — verifies C extension loaded
        log.info("[pyift] Local pyift.pyift available.")
        return True
    except Exception as exc:
        log.info("[pyift] Local pyift.pyift unavailable: %s", exc)
        return False


def _load_local(paths: list[str]) -> list[np.ndarray]:
    """Load images as LABNorm2 float32 (H, W, 3) using native pyift."""
    import pyift.pyift as ift

    results: list[np.ndarray] = []
    for path in paths:
        image  = ift.ReadImageByExt(str(path))
        mimage = ift.ImageToMImage(image, color_space=ift.LABNorm2_CSPACE)
        arr    = mimage.AsNumPy().squeeze().astype(np.float32)
        results.append(arr)
    return results


# ── Docker runner ─────────────────────────────────────────────────────────────

def _docker_daemon_available() -> bool:
    """Return True if the Docker daemon responds to ``docker info``."""
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
    """Return True if the pyift Docker image is already present locally."""
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
    """Try to pull the image from the configured registry.

    Returns True if the pull succeeded and the image is now available locally.
    """
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
        # Tag the pulled image with our local name so the run command is stable.
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
    """Build the pyift Docker image locally from Dockerfile.pyift."""
    log.info("[pyift] Building Docker image %s locally…", DOCKER_IMAGE_NAME)
    subprocess.run(
        [
            "docker", "build",
            "--platform", "linux/amd64",
            "-t", DOCKER_IMAGE_NAME,
            str(_HERE),
            "-f", str(DOCKER_DOCKERFILE),
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    log.info("[pyift] Docker image built successfully.")


def _probe_docker() -> bool:
    """Return True if Docker is available and the image is ready to use.

    Resolution order:
    1. Image already present locally    → use it immediately.
    2. Pull from registry               → tag locally, use it.
    3. Build from Dockerfile.pyift      → use it.
    4. All three fail                   → return False.
    """
    log.info("[pyift] Probing Docker fallback…")

    if not _docker_daemon_available():
        log.info("[pyift] Docker daemon not running — fallback unavailable.")
        return False

    # Fast path: image is already local.
    if _docker_image_exists():
        log.info("[pyift] Docker image already present: %s.", DOCKER_IMAGE_NAME)
        return True

    # Try registry pull first (faster than a full build).
    if _pull_docker_image():
        return True

    # Fall back to local build.
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
    """Run one Docker container invocation for a chunk of image paths.

    Args:
        container_paths: Absolute image paths as seen INSIDE the container
                         (i.e. prefixed with /workspace/...).
        tmp_dir:         Host-side temp directory mounted at /job in container.

    Returns:
        List of float32 (H, W, 3) arrays in [0, 1].
    """
    paths_file   = tmp_dir / "paths.json"
    results_file = tmp_dir / "results.npz"

    # Write the paths manifest.
    with open(paths_file, "w") as f:
        json.dump(container_paths, f)

    # Remove stale results file if present.
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
    """Batch-load images via Docker, processing in chunks to limit memory."""
    # Convert host absolute paths → container-relative paths under /workspace.
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
            chunk_results = _load_docker_chunk(chunk_c, tmp_dir)
            results.extend(chunk_results)

    return results


# ── PIL fallback ──────────────────────────────────────────────────────────────

def _load_pil(paths: list[str]) -> list[np.ndarray]:
    """Load images as RGB float32 (H, W, 3) using PIL — colour-space fallback."""
    from PIL import Image as PILImage

    results: list[np.ndarray] = []
    for path in paths:
        img = PILImage.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        results.append(arr)
    return results


# ── Strategy ──────────────────────────────────────────────────────────────────

class PyiftLoader:
    """Local-first, Docker-fallback image loader.

    Detects the available runtime once at construction; subsequent
    ``load_batch`` calls go directly to the selected backend.

    Parameters
    ----------
    allow_pil_fallback:
        When True (default), silently degrade to PIL-RGB loading if both
        local pyift and Docker are unavailable.  When False, raise
        RuntimeError instead.
    """

    def __init__(self, allow_pil_fallback: bool = True) -> None:
        self._mode = self._detect_mode(allow_pil_fallback)

    # ── Mode detection ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_mode(allow_pil_fallback: bool) -> str:
        # 1. Local
        if _probe_local():
            return "local"

        # 2. Docker
        if _probe_docker():
            return "docker"

        # 3. PIL fallback
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

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Active backend: ``'local'`` | ``'docker'`` | ``'pil'``."""
        return self._mode

    def load_batch(self, paths: Sequence[str]) -> list[np.ndarray]:
        """Load *paths* and return one float32 (H, W, 3) array per image.

        - local  → native pyift.pyift, lazy per-call.
        - docker → single (chunked) Docker invocation for the whole batch.
        - pil    → PIL RGB as float32, lazy per-call.

        All modes return LABNorm2 if pyift is used, or RGB if PIL is used.
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

        # pil
        log.info("[pyift] Loading %d images via PIL (RGB fallback).", len(path_list))
        return _load_pil(path_list)
