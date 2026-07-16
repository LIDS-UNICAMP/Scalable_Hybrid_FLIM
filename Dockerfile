# ─── Base image ───────────────────────────────────────────────────────────────
# mambaorg/micromamba gives a minimal conda-compatible environment manager.
# The default user inside this image is "mambauser" (UID 1000).
# OS: Ubuntu 22.04 (jammy), x86_64.
# CUDA 12.1 — compatible with NVIDIA L40S (Ada Lovelace) and any driver ≥ 525.
FROM mambaorg/micromamba:1.5.8-jammy-cuda-12.1.0

# Switch to root to install system dependencies, then drop back to mambauser.
USER root

# ─── System packages ──────────────────────────────────────────────────────────
# libgl1 / libglib2.0 are needed by OpenCV headless at runtime.
# libift (pyift C backend) may need libgomp for OpenMP threading.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        git \
        wget \
        unzip \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─── Conda environment ────────────────────────────────────────────────────────
# Python 3.11 — matches host env (Python 3.11.14) and the bundled pyift wheel:
#   pyift_whl/3_11/pyift-0.1-cp311-cp311-linux_x86_64.whl
COPY requirements.txt /tmp/requirements.txt

RUN micromamba create -n scalable_FLIM -c pytorch -c nvidia -c conda-forge \
        python=3.11 \
        pytorch=2.2 \
        torchvision \
        pytorch-cuda=12.1 \
        --yes \
    && micromamba clean --all --yes

# Activate environment for subsequent RUN steps
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_DEFAULT_ENV=scalable_FLIM

# Install pip dependencies (requirements.txt pins numpy<2 for torch 2.2 compatibility)
RUN micromamba run -n scalable_FLIM pip install --no-cache-dir -r /tmp/requirements.txt

# ─── Jupyter ──────────────────────────────────────────────────────────────────
RUN micromamba run -n scalable_FLIM pip install --no-cache-dir \
        notebook \
        ipywidgets

# ─── Project code ─────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

# ─── pyift ────────────────────────────────────────────────────────────────────
# pyift is a private C-extension wheel (NOT on PyPI, gitignored in repo).
# Installation criteria from: data/to_mateus/model/utils/2_flim_classification.ipynb
#
# Download the wheels zip from Google Drive, install the cp311 wheel, then
# copy the compiled .so into site-packages/pyift/ (mirrors the notebook step).
RUN wget -q -O /tmp/pyift_whl.zip \
        "https://drive.usercontent.google.com/download?id=1ddlcSqwli4UIFlJvqhVLUc8kduFlsXDj&export=download&authuser=2&confirm=yes" \
    && unzip -q /tmp/pyift_whl.zip -d /tmp/pyift_whl \
    && rm /tmp/pyift_whl.zip \
    && micromamba run -n scalable_FLIM pip install --no-cache-dir \
         /tmp/pyift_whl/pyift_whl/3_11/pyift-0.1-cp311-cp311-linux_x86_64.whl \
    && SITE_PKG=$(micromamba run -n scalable_FLIM python -c "import site; print(site.getsitepackages()[0])") \
    && cp /tmp/pyift_whl/pyift_whl/3_11/pyift/_pyift.cpython-311-x86_64-linux-gnu.so \
          "${SITE_PKG}/pyift/_pyift.cpython-311-x86_64-linux-gnu.so" \
    && rm -rf /tmp/pyift_whl

# Allow the non-root mambauser to write results / logs inside the project.
RUN chown -R mambauser:mambauser /workspace

USER mambauser

# ─── Environment activation shim ─────────────────────────────────────────────
ENV PATH="/opt/conda/envs/scalable_FLIM/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=scalable_FLIM

# ─── Validation (build-time smoke tests) ──────────────────────────────────────
RUN python -c "import torch; print('torch', torch.__version__)"
RUN python -c "import ray; print('ray', ray.__version__)"
RUN python -c "import notebook; print('notebook ok')"
RUN python -c "import pyift.pyift as ift; print('pyift ok')"
RUN python -c "import lightning; print('lightning ok')"

# ─── Default command ──────────────────────────────────────────────────────────
# The container is a prepared execution environment — it does NOT auto-start
# any service.  Specify the command at runtime, for example:
#
#   docker run --gpus all -v $(pwd):/workspace <image> \
#       python -m src.evaluate.ray_mlp --num-gpus 8
#
#   docker run --gpus all -v $(pwd):/workspace -p 8888:8888 <image> \
#       jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
#
CMD ["bash"]
