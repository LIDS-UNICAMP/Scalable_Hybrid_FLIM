#!/usr/bin/env bash
# setup_env.sh — Install the scalable_FLIM conda environment on a fresh Ubuntu machine.
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# What this script does:
#   1. Downloads and installs Mambaforge (mamba + conda, Python 3.11)
#   2. Creates the scalable_FLIM environment from environment.yml
#   3. Installs uv and the Python deps from requirements.txt with it
#   4. Downloads the pyift wheels from Google Drive (pyift_whl/ is gitignored)
#   5. Installs pyift from the downloaded wheel
#   6. Validates all imports
#
# Requirements:
#   - Ubuntu x86_64
#   - NVIDIA GPU with CUDA 12.1 drivers installed on the host
#   - The repository cloned locally (this script lives at the repo root)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBAFORGE_INSTALLER="/tmp/Miniforge3-Linux-x86_64.sh"
MAMBAFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MAMBA_INSTALL_DIR="$HOME/miniforge3"
ENV_NAME="scalable_FLIM"
PYTHON_MINOR="11"

# Google Drive URL for the pyift wheels zip (source: 2_flim_classification.ipynb)
PYIFT_GDRIVE_URL="https://drive.usercontent.google.com/download?id=1ddlcSqwli4UIFlJvqhVLUc8kduFlsXDj&export=download&authuser=2&confirm=yes"
PYIFT_ZIP="/tmp/pyift_whl.zip"

echo "============================================================"
echo " scalable_FLIM environment setup"
echo " Repo : $REPO_ROOT"
echo " Env  : $ENV_NAME (Python 3.$PYTHON_MINOR)"
echo "============================================================"

# ── Step 1: Install Mambaforge if not present ─────────────────────────────────
if command -v mamba &>/dev/null; then
    echo "[1/6] mamba already installed — skipping download."
else
    echo "[1/6] Downloading Mambaforge (~100 MB) ..."
    wget --progress=dot:mega -O "$MAMBAFORGE_INSTALLER" "$MAMBAFORGE_URL"
    echo "[1/6] Installing Mambaforge to $MAMBA_INSTALL_DIR ..."
    bash "$MAMBAFORGE_INSTALLER" -b -p "$MAMBA_INSTALL_DIR"
    rm -f "$MAMBAFORGE_INSTALLER"

    export PATH="$MAMBA_INSTALL_DIR/bin:$PATH"
    "$MAMBA_INSTALL_DIR/bin/conda" init bash
    echo "[1/6] Mambaforge installed. Shell initialised."
    echo "      NOTE: open a new terminal (or run: source ~/.bashrc) after setup."
fi

if [[ -d "$MAMBA_INSTALL_DIR/bin" ]]; then
    export PATH="$MAMBA_INSTALL_DIR/bin:$PATH"
fi

# ── Step 2: Create the conda environment ──────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment '$ENV_NAME' from environment.yml ..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[2/6] Environment '$ENV_NAME' already exists — skipping creation."
    echo "      To recreate it, run: conda env remove -n $ENV_NAME"
else
    mamba env create -f "$REPO_ROOT/environment.yml"
    echo "[2/6] Environment '$ENV_NAME' created."
fi

ENV_PY=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.executable)")
SITE_PKG=$(conda run -n "$ENV_NAME" python -c "import site; print(site.getsitepackages()[0])")
echo "[2/6] python:       $ENV_PY"
echo "[2/6] site-packages: $SITE_PKG"

# ── Step 3: Install uv and the Python deps from requirements.txt ──────────────
echo ""
echo "[3/6] Setting up uv ..."
if command -v uv &>/dev/null; then
    echo "[3/6] uv already installed ($(uv --version)) — skipping download."
else
    echo "[3/6] Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin (or $CARGO_HOME/bin); make it available now.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[3/6] Installing requirements.txt into '$ENV_NAME' with uv ..."
# Install into the conda env's interpreter. PyTorch/torchvision are already
# provided by conda (CUDA build); uv will keep them since requirements.txt only
# pins lower bounds that the conda versions already satisfy.
uv pip install --python "$ENV_PY" -r "$REPO_ROOT/requirements.txt"
echo "[3/6] Python dependencies installed."

# ── Step 4: Download pyift wheels from Google Drive ───────────────────────────
echo ""
echo "[4/6] Checking pyift wheels ..."

WHL="$REPO_ROOT/pyift_whl/3_${PYTHON_MINOR}/pyift-0.1-cp3${PYTHON_MINOR}-cp3${PYTHON_MINOR}-linux_x86_64.whl"

if [[ -f "$WHL" ]]; then
    echo "[4/6] pyift wheel already present — skipping download."
else
    echo "[4/6] pyift_whl/ not found (gitignored). Downloading from Google Drive..."
    wget --progress=dot:mega -O "$PYIFT_ZIP" "$PYIFT_GDRIVE_URL"
    echo "[4/6] Extracting to $REPO_ROOT ..."
    unzip -q "$PYIFT_ZIP" -d "$REPO_ROOT"
    rm -f "$PYIFT_ZIP"
    if [[ ! -f "$WHL" ]]; then
        echo "ERROR: wheel still not found at $WHL after extraction."
        echo "       The zip layout may have changed. Check the contents of pyift_whl/."
        exit 1
    fi
    echo "[4/6] pyift wheels downloaded."
fi

# ── Step 5: Install pyift ─────────────────────────────────────────────────────
echo ""
echo "[5/6] Installing pyift (Python 3.$PYTHON_MINOR wheel) ..."

SO_SRC="$REPO_ROOT/pyift_whl/3_${PYTHON_MINOR}/pyift/_pyift.cpython-3${PYTHON_MINOR}-x86_64-linux-gnu.so"
SO_DST="$SITE_PKG/pyift/_pyift.cpython-3${PYTHON_MINOR}-x86_64-linux-gnu.so"

uv pip install --python "$ENV_PY" --no-cache "$WHL"

if [[ -f "$SO_SRC" ]]; then
    cp -f "$SO_SRC" "$SO_DST"
    echo "[5/6] pyift .so copied to $SO_DST"
else
    echo "WARNING: .so not found at $SO_SRC — import pyift.pyift may fail."
fi

# ── Step 6: Validate ──────────────────────────────────────────────────────────
echo ""
echo "[6/6] Validating installation ..."
conda run -n "$ENV_NAME" python --version
conda run -n "$ENV_NAME" python -c "import torch;           print('torch    ', torch.__version__)"
conda run -n "$ENV_NAME" python -c "import numpy;           print('numpy    ', numpy.__version__)"
conda run -n "$ENV_NAME" python -c "import ray;             print('ray      ', ray.__version__)"
conda run -n "$ENV_NAME" python -c "import lightning;       print('lightning', lightning.__version__)"
conda run -n "$ENV_NAME" python -c "import pyift.pyift as ift; print('pyift     ok')"
conda run -n "$ENV_NAME" python -c "import notebook;        print('notebook  ok')"

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Activate with:"
echo "   conda activate $ENV_NAME"
echo ""
echo " Then run:"
echo "   wandb login"
echo "   python scripts/generate_mlp_configs.py"
echo "   python -m src.evaluate.ray_mlp --num-gpus 8"
echo "============================================================"
