# Installation Guide

> **Running experiments**: see [README.md](README.md)

---

## Requirements

- Ubuntu x86_64 (22.04 or 24.04)
- NVIDIA GPU with driver ≥ 525 (`nvidia-smi` must show CUDA 12.x)
- Python 3.11
- Repository cloned locally

---

## Option A — conda / mamba (automated script)

Run the provided setup script once from the repo root:

```bash
chmod +x setup_env.sh
./setup_env.sh
```

The script performs these steps automatically:

| Step | What happens |
|---|---|
| 1 | Downloads and installs **Miniforge3** (mamba + conda) to `~/miniforge3` |
| 2 | Creates the `scalable_FLIM` environment from `environment.yml` (Python 3.11, PyTorch 2.2 + CUDA 12.1) |
| 3 | Downloads the **pyift wheel** from Google Drive and installs the `cp311` wheel, then copies the compiled `.so` into site-packages |
| 4 | Validates all key imports: `torch`, `ray`, `lightning`, `pyift`, `notebook` |

After the script finishes, open a new terminal and activate:

```bash
conda activate scalable_FLIM
wandb login
```

### Manual steps (conda, without the script)

```bash
# 1. Install Mambaforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
export PATH="$HOME/miniforge3/bin:$PATH"
~/miniforge3/bin/conda init bash

# 2. Create the environment
mamba env create -f environment.yml

# 3. Download and install pyift
conda activate scalable_FLIM
wget -O /tmp/pyift_whl.zip "https://drive.usercontent.google.com/download?id=1ddlcSqwli4UIFlJvqhVLUc8kduFlsXDj&export=download&authuser=2&confirm=yes"
unzip /tmp/pyift_whl.zip -d /tmp/pyift_whl
pip install /tmp/pyift_whl/pyift_whl/3_11/pyift-0.1-cp311-cp311-linux_x86_64.whl
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp /tmp/pyift_whl/pyift_whl/3_11/pyift/_pyift.cpython-311-x86_64-linux-gnu.so $SITE/pyift/

# 4. Validate
python -c "import torch, ray, lightning, pyift.pyift; print('all ok')"

# 5. Login
wandb login
```

### Activate in future sessions

```bash
conda activate scalable_FLIM
```

---

## Option B — uv (fast, no conda)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that replaces pip + venv.

### Step 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 2 — Create virtual environment

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### Step 3 — Install PyTorch 2.2 + CUDA 12.1

PyTorch must be installed **before** the rest of the requirements so that uv resolves dependencies against the CUDA build.

```bash
uv pip install torch==2.2.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install project dependencies

```bash
uv pip install -r requirements.txt
```

### Step 5 — Install pyift

`pyift` is a private C-extension wheel (not on PyPI).

```bash
# Download
wget -q --show-progress -O pyift_whl.zip \
    "https://drive.usercontent.google.com/download?id=1ddlcSqwli4UIFlJvqhVLUc8kduFlsXDj&export=download&authuser=2&confirm=yes"

# Extract
unzip -q pyift_whl.zip -d pyift_whl

# Install wheel
uv pip install pyift_whl/3_11/pyift-0.1-cp311-cp311-linux_x86_64.whl

# Copy compiled .so into site-packages
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
cp pyift_whl/3_11/pyift/_pyift.cpython-311-x86_64-linux-gnu.so \
   "$SITE/pyift/_pyift.cpython-311-x86_64-linux-gnu.so"

# Cleanup
rm -rf pyift_whl pyift_whl.zip
```

### Step 6 — Validate

```bash
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda)"
python -c "import ray; print('ray', ray.__version__)"
python -c "import lightning; print('lightning ok')"
python -c "import pyift.pyift as ift; print('pyift ok')"
python -c "import notebook; print('notebook ok')"
```

Expected output:

```
torch 2.2.0+cu121 | CUDA 12.1
ray 2.x.x
lightning ok
pyift ok
notebook ok
```

### Step 7 — Login to W&B

```bash
wandb login
```

### Activate in future sessions

```bash
source .venv/bin/activate
```

---

## Docker (2× NVIDIA L40S)

Build the image (CUDA 12.1, Python 3.11, PyTorch 2.2, Ray, pyift):

```bash
docker build -t scalable-flim .
```

Run the dual-GPU queue inside the container:

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/data:/workspace/data \
  -e WANDB_API_KEY \
  scalable-flim \
  python -m src.evaluate.ray_mlp_queue
```

> Export `WANDB_API_KEY` in your shell first; the `-e WANDB_API_KEY` flag forwards it automatically.

Launch Jupyter Notebook:

```bash
docker run --gpus all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  scalable-flim \
  jupyter notebook --ip=0.0.0.0 --no-browser
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11` not found | `sudo apt install python3.11 python3.11-venv` |
| `nvidia-smi` shows CUDA < 12.1 | Update driver: `sudo apt install nvidia-driver-525` |
| `import pyift` fails after install | Re-run the `.so` copy step |
| `torch.cuda.is_available()` returns `False` | Confirm CUDA build was installed (not CPU wheel) |
| `wandb` install fails | `requirements.txt` pins `wandb<0.18` — use that version |
| `_pyift` ImportError at runtime | The `.so` was not copied; re-run the copy command |
