# Software Design Description (SDD)
## Project: scalable_FLIM_self_supervised

---

## 1. Overview

This project implements a self-supervised learning (SSL) pipeline for visual feature learning using the **FLIM (Feature Learning from Image Markers)** methodology. The core goal is to train lightweight convolutional encoders — whose architecture is derived from FLIM's marker-based kernel estimation — with minimal annotation. The primary application domain is **parasite classification** (helminth eggs, helminth larvae, protozoan cysts), with infrastructure generalizable to other image datasets.

The SSL pretraining strategy is based on **LeJEPA** (Lean Joint-Embedding Predictive Architecture), which uses a multi-crop invariance objective combined with **SIGReg** (Sketched Isotropic Gaussian Regularization) to learn compact, semantically rich representations without labels.

Downstream evaluation is performed via linear SVM probes and fine-tuned MLP heads on frozen or unfrozen encoder representations.

---

## 2. Architecture

```
scalable_FLIM_self_supervised/
├── src/
│   ├── main.py                         # LightningCLI entry point
│   ├── models/
│   │   ├── models.py                   # FLIM Encoder, ClassificationModel, AutoEncoder, weight loading
│   │   ├── encoders.py                 # TIMM-based generic encoder builder
│   │   ├── lejepa.py                   # LeJEPAModel (TIMM backbone + ProjectionHead)
│   │   ├── lejepa_flim.py              # LeJEPAFLIMModel (FLIM backbone + ProjectionHead)
│   │   ├── lejepa_line_module.py       # (likely linear probe or line variant)
│   │   ├── ijepa_encoder.py            # I-JEPA encoder variant
│   │   └── custom_cnn.py               # Custom CNN architectures
│   ├── modules/
│   │   ├── lejepa_module.py            # Lightning module: LeJEPA SSL with TIMM encoder
│   │   ├── lejepa_flim_module.py       # Lightning module: LeJEPA SSL with FLIM encoder
│   │   ├── lejepa_line_module.py       # Lightning module: linear probe evaluation
│   │   └── classifier_module.py        # Lightning module: supervised fine-tuning
│   ├── losses/
│   │   ├── lejepa_loss.py              # SIGReg (SimpleSIGReg, RealSIGReg), invariance_loss
│   │   ├── epps_pulley.py              # Epps-Pulley characteristic function test statistic
│   │   └── base.py                     # Base loss utilities
│   ├── data_modules/
│   │   ├── datasets/
│   │   │   ├── dataset.py              # DatasetParasite: base parasite image dataset
│   │   │   ├── lejepa_dataset.py       # Multi-crop dataset for SSL pretraining
│   │   │   ├── multicrop_dataset.py    # Multi-crop augmentation dataset
│   │   │   └── parasite_lejepa.py      # Parasite-specific LeJEPA dataset wrapper
│   │   ├── lejepa.py                   # LightningDataModule for LeJEPA SSL
│   │   ├── parasite.py                 # LightningDataModule for parasite classification
│   │   └── parasite_data_module_lejepa_splited.py  # Split-aware parasite data module
│   ├── transforms/
│   │   └── multicrop.py               # Multi-crop / multi-view augmentation transforms
│   ├── evaluate/
│   │   ├── unified_eval.py             # Unified evaluation: SVM, MLP-freeze, MLP-unfreeze
│   │   ├── eval_plotter.py             # Composite result plots and metric-vs-pct curves
│   │   ├── svm.py                      # SVM evaluation utilities
│   │   ├── mlp.py                      # MLP evaluation utilities
│   │   ├── ray_mlp.py                  # Distributed MLP evaluation with Ray
│   │   └── wandb_resolver.py           # WandB run resolver for artifact management
│   ├── metrics/
│   │   └── classification.py          # Classification metrics (accuracy, F1, etc.)
│   ├── analysis/
│   │   ├── tsne_flim.py               # t-SNE visualization of FLIM encoder embeddings
│   │   ├── pyift_strategy.py          # PyIFT data loading strategy
│   │   └── _docker_pyift_worker.py    # Docker-based PyIFT worker process
│   └── utils/                         # Utilities: CLI wrappers, evaluation helpers
├── configs/
│   ├── default.yaml                   # Global training defaults (seed, trainer, W&B)
│   ├── data/                          # Dataset configs (percentage splits)
│   ├── model/                         # Model configs (per-parasite FLIM YAMLs)
│   ├── evaluate/                      # Evaluation configs
│   └── fine_tune/                     # Fine-tuning configs
├── scripts/                           # Helper scripts (config generation, dataset download)
├── run_experiments.py                 # Grid experiment runner
├── Dockerfile                         # Container definition for GPU training
└── data/                              # FLIM pretrained weights and architecture JSONs
```

---

## 3. Core Components

### 3.1 FLIM Encoder (`src/models/models.py`)

The **Encoder** class builds a convolutional backbone dynamically from an `architecture.json` file produced by the FLIM tool. Each layer consists of:
- `Conv2d` (kernel size and dilation from JSON)
- Optional `ReLU`
- Optional `MaxPool2d`

Key functions:
- `parse_architecture(arch_json)`: Loads FLIM architecture JSON.
- `get_channels_from_arch(arch, in_channels)`: Extracts channel progression.
- `get_actual_channels_from_weights(weights_path, arch, in_channels)`: Reads actual kernel counts from saved FLIM bias files (FLIM filter selection may prune kernels below the spec).
- `load_FLIM_encoder(model, arch_json, weights_path, channels)`: Loads FLIM-estimated kernels and biases from `.npy` and `.txt` files into the PyTorch encoder.
- `freeze_encoder / unfreeze_encoder`: Gradient control utilities.

**Parameter regime**: FLIM architectures (e.g., ch24_32_48) produce encoders with very few parameters — consistent with the "flyweight" (<100K parameters) design philosophy of the FLIM/IFT research group.

### 3.2 LeJEPA Models

Two model variants:

**`LeJEPAModel`** (`src/models/lejepa.py`): Uses a TIMM backbone (e.g., `vit_small_patch16_224`, `resnet50`) with a 3-layer MLP projection head (`ProjectionHead`). Intended for comparison against larger architectures.

**`LeJEPAFLIMModel`** (`src/models/lejepa_flim.py`): Uses the FLIM `Encoder` backbone. Spatial features are collapsed via `AdaptiveAvgPool2d(1)` to produce `[B, embed_dim]` embeddings, followed by the same `ProjectionHead`. This is the primary research model.

**`ProjectionHead`**: 3-layer MLP with BatchNorm1d and ReLU, following the LeJEPA reference design. Output dimension is configurable (`proj_dim`, default 256).

### 3.3 SSL Loss (`src/losses/lejepa_loss.py`)

The training objective combines two terms:

```
loss = lam * SIGReg(proj) + (1 - lam) * invariance_loss(proj)
```

- **`SimpleSIGReg`**: Moment-matching regularizer. Penalizes deviation of random projections from zero mean and unit variance (approximates Gaussianity, computationally cheap).
- **`RealSIGReg`** (alias: `SIGReg`): Full Epps-Pulley characteristic function test statistic for Gaussianity testing (more powerful, higher cost). Uses `EppsPulley` from `src/losses/epps_pulley.py`.
- **`invariance_loss`**: MSE between each view's embedding and the mean over all views — alignment objective.

`lam` default: 0.05 (paper-recommended range: 1e-3 to 1e-1).

### 3.4 Lightning Modules

**`LeJEPAModule`** (`src/modules/lejepa_module.py`): SSL pretraining with TIMM backbone. Accepts multi-crop batches `(List[Tensor], labels)`, ignores labels.

**`LeJEPAFLIMModule`** (`src/modules/lejepa_flim_module.py`): SSL pretraining with FLIM backbone. Supports four encoder initialization strategies:
- `"random"`: default PyTorch init
- `"he"`: Kaiming initialization
- `"xavier"`: Xavier initialization
- `"flim"`: Load FLIM-estimated weights from disk

**`ClassificationFinetuneModule`** (`src/modules/classifier_module.py`): Supervised fine-tuning with a linear head on top of a pretrained SSL encoder (frozen or unfrozen).

All modules use **AdamW** optimizer with **linear warmup + cosine annealing** LR schedule.

### 3.5 Data Modules and Datasets

- **`DatasetParasite`**: Base dataset for three parasite categories (helminth eggs, helminth larvae, protozoan cysts).
- **`lejepa_dataset.py`**: Multi-crop dataset for SSL — generates multiple augmented views per image for the LeJEPA invariance objective.
- **`multicrop.py`**: Multi-view augmentation transforms (global and local crops).
- **`parasite_data_module_lejepa_splited.py`**: Split-aware data module supporting percentage-based label splits for semi-supervised evaluation.

### 3.6 Evaluation Pipeline (`src/evaluate/unified_eval.py`)

Three evaluation protocols on frozen encoder representations:
1. **SVM** — Linear SVM on extracted embeddings (`sklearn`).
2. **MLP-freeze** — MLP head with frozen encoder, fine-tuned.
3. **MLP-unfreeze** — MLP head with unfrozen encoder, end-to-end fine-tuned.

Results are stored under `artifacts/{SVM,MLP}/{dataset}/lejepa_pct_{pct}/` with CSV metrics and prediction files.

### 3.7 Analysis Module (`src/analysis/`)

- **`tsne_flim.py`**: Inference-only t-SNE visualization of FLIM encoder embeddings across test splits and training percentages. Outputs PNG figures.
- **`pyift_strategy.py`**: PyIFT-based data loading strategy (uses IFT-processed image data).
- **`_docker_pyift_worker.py`**: Docker worker for PyIFT processing.

---

## 4. Execution Flow

### 4.1 SSL Pretraining (FLIM backbone)

```
run_experiments.py
  → discovers configs/data/percentage/*.yaml
  → for each (dataset, split, percentage, model_variant):
      → python -m src.main fit
          --config configs/default.yaml
          --config configs/data/...yaml
          --config configs/model/lejepa_line_flim_{dataset}_train{N}.yaml
          [--trainer.max_epochs N]
```

The `LeJEPAFLIMModule` orchestrates:
1. Parse FLIM architecture JSON → build `LeJEPAFLIMModel`
2. Optionally load FLIM-estimated weights (`encoder_init="flim"`)
3. Multi-crop batch → `model.forward(views)` → `(emb, proj)`
4. Compute `SIGReg(emb)` + `invariance_loss(emb)`
5. Backprop, AdamW update, cosine LR schedule
6. Log to Weights & Biases

### 4.2 Downstream Evaluation

```
python -m src.evaluate.unified_eval --model all --dataset all
  → For each (dataset, pct):
      → Extract features from best SSL checkpoint
      → Train SVM on frozen features
      → Fine-tune MLP (freeze / unfreeze)
      → Save metrics CSV + predictions CSV
      → Generate composite plots (eval_plotter.py)
```

### 4.3 t-SNE Analysis

```
python -m src.analysis.tsne_flim
  → For each FLIM model × split × percentage:
      → Load test-split images
      → Extract encoder embeddings (inference only)
      → Apply t-SNE (sklearn)
      → Save PNG to tsne_analisys/{problem}/split{N}/perc{pct}/{train_id}.png
```

---

## 5. Implementation Characteristics

- **Framework**: PyTorch Lightning (LightningCLI, LightningModule, LightningDataModule)
- **Config system**: OmegaConf/YAML (hierarchical, CLI-overridable)
- **Experiment tracking**: Weights & Biases (WandbLogger)
- **Architecture loading**: Dynamic from FLIM-generated `architecture.json`
- **Weight initialization**: FLIM-estimated kernels or standard random init (He/Xavier)
- **Multi-crop SSL**: V views per image (configurable), multi-crop augmentation pipeline
- **Evaluation**: Sklearn SVM + PyTorch MLP (frozen/unfrozen), Ray for parallelism
- **Encoder size**: Flyweight regime — FLIM architectures (e.g., ch24_32_48) yield encoders with very few parameters (typically <100K)
- **Loss regime**: SIGReg (Gaussianity) + invariance (alignment), lam=0.05 default
- **Datasets**: Parasite microscopy images (eggs, larvae, protozoan cysts)
- **LR schedule**: Linear warmup (default 10 epochs) + cosine annealing
- **Optimizer**: AdamW (lr=5e-4, weight_decay=5e-2)

---

## 6. Operational Details

- **Entry point**: `src/main.py` via `python -m src.main fit/predict/test`
- **Grid runner**: `run_experiments.py` (discovers YAML configs, spawns subprocesses)
- **Experiment runner (SSL + Ray)**: `scripts/run_ssl_ray.py`
- **Containerization**: `Dockerfile` (GPU training), `src/analysis/Dockerfile.pyift` (PyIFT worker)
- **Dependencies**: PyTorch, Lightning, TIMM, scikit-learn, wandb, pyrootutils, Ray (optional)
- **Environment**: `environment.yml` (conda), `requirements.txt` (pip)

---

## 7. Technical Notes

- FLIM kernel files: `conv{n}-kernels.npy` (weights) and `conv{n}-bias.txt` (bias + count) stored per dataset/split under `data/to_mateus/model/`.
- Architecture JSON specifies `nlayers`, per-layer `kernel_size`, `dilation_rate`, `noutput_channels`, `pooling` (type, size, stride).
- FLIM filter selection may produce fewer kernels than specified (`nkernels = nclasses * nkernels_per_marker`, capped by `nkernels_per_image`) — `get_actual_channels_from_weights` handles this discrepancy.
- The `LeJEPAFLIMModule` supports both pure SSL (random/he/xavier init) and warm-started SSL (flim init), enabling investigation of the benefit of FLIM-estimated initialization over random initialization.
- Downstream evaluation uses percentage-based label splits (e.g., 5%, 10%, 25%, 50%, 100%) to simulate the low-label regime.
- The `unified_eval.py` resolves best checkpoints from W&B run artifacts via `wandb_resolver.py`.

---

## Knowledge Distillation (I-JEPA → FLIM CNN)

Ver detalhes completos em:
- [`discussion_about_distill.md`](discussion_about_distill.md) — implementação, resultados, diagnóstico projection head dominance
- [`distillation_direct_loss.md`](distillation_direct_loss.md) — fluxo do modo direct, componentes

**Resumo:**
- Student: FLIM CNN encoder [B,48] + ConvProjectionHead (1×1 convs, 48→1280)
- Teacher: I-JEPA ViT-H/14 (frozen, 1280-dim)
- Loss: MSE(student_proj, teacher_emb) no modo direct
- Módulo ativo: `distillation_conv_module.py` (next_layers_direct)
- 54 runs concluídos (3 datasets × 3 splits × 6 pcts)

**Problema identificado:** Projection Head Dominance — a head aprende sozinha, encoder [B,48] não fica discriminativo para eggs/protozoan. Larvae (2 classes) funciona bem. Com a proj head ativa no SVM [B,1280], desempenho de eggs volta a ~0.88.

**SVM scripts:**
- `src/evaluate/svm_distillation_conv.py` → embedding [B,48] sem proj head
- `src/evaluate/svm_distill_with_projection.py` → embedding [B,1280] com proj head (Red Projection)
