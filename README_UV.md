# Environment Setup with uv

Fast, reproducible environment setup using [uv](https://github.com/astral-sh/uv)
(no conda required).

---

## Requirements

- Ubuntu x86_64 (22.04 or 24.04)
- NVIDIA GPU with driver ≥ 525 (`nvidia-smi` must show CUDA 12.x)
- Python 3.11 available on the system (`python3.11 --version`)
- Repository cloned locally

---

## Step 1 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Verify:

```bash
uv --version
```

---

## Step 2 — Create virtual environment

From the repository root:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

---

## Step 3 — Install PyTorch 2.2 + CUDA 12.1

PyTorch must be installed **before** the rest of the requirements so that uv
resolves all dependencies against the CUDA build (not the CPU wheel from PyPI).

```bash
uv pip install torch==2.2.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.__version__, '| CUDA', torch.version.cuda)"
# expected: 2.2.0+cu121 | CUDA 12.1
```

---

## Step 4 — Install project dependencies

```bash
uv pip install -r requirements.txt
```

> `numpy<2` and `wandb>=0.16,<0.18` are already pinned in `requirements.txt`
> for compatibility with PyTorch 2.2.

---

## Step 5 — Install pyift

`pyift` is a private C-extension wheel (not on PyPI). It must be downloaded
from Google Drive, installed via the bundled `.whl`, and then the compiled
`.so` must be copied into site-packages.

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
rm -rf /tmp/pyift_whl /tmp/pyift_whl.zip


!ln -s $abs_path/_pyift.*.so $site_packages_path/pyift/
```

---

## Step 6 — Validate

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

---

## Step 7 — Login to W&B

```bash
wandb login
```

---

## Activate in future sessions

```bash
source .venv/bin/activate
```

---

## Running experiments

### SSL Pre-training

```bash
python src/main.py fit \
  --config configs/default.yaml \
  --config configs/data/percentage/helminth-eggs_split_1/100/lejepa_line.yaml \
  --config configs/model/lejepa_line_xavier.yaml \
  --trainer.accelerator=gpu
```

### Dual-GPU queued evaluation (2× NVIDIA L40S)

```bash
# Preview queue (no execution)
python -m src.evaluate.ray_mlp_queue --dry-run

# Run all experiments (8 per GPU, both freeze and unfreeze modes)
python -m src.evaluate.ray_mlp_queue

# Resume an interrupted run
python -m src.evaluate.ray_mlp_queue --resume

# Single experiment test
python -m src.evaluate.ray_mlp_queue \
    --max-experiments-per-gpu 1 \
    --experiment-filter <run_id_substring>
```

### SVM evaluation

```bash
python -m src.evaluate.svm
# → results/svm_results.csv
```

### MLP fine-tuning (sequential)

```bash
python -m src.evaluate.mlp --mode freeze
# → results/mlp_results.csv
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3.11` not found | `sudo apt install python3.11 python3.11-venv` |
| `nvidia-smi` shows CUDA < 12.1 | Update NVIDIA driver to ≥ 525: `sudo apt install nvidia-driver-525` |
| `import pyift` fails after install | Re-run the `.so` copy step in Step 5 |
| `torch.cuda.is_available()` returns `False` | Verify driver with `nvidia-smi`; confirm CUDA build was installed (step 3) |
| `wandb` install fails with Go error | Already handled — `requirements.txt` pins `wandb<0.18` |
