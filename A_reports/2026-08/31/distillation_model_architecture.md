# Distillation Model Architecture

Comparação completa layer by layer entre o student (FLIM CNN + ConvDistillationProjectionHead)
e o teacher (I-JEPA ViT-H/14).

---

## O que é FLIM e o que não é

**FLIM** é um método que gera filtros convolucionais a partir de anotações de usuário (marcadores).
Ele produz um `architecture.json` com a estrutura das camadas e arquivos de pesos pré-calculados
(`conv1-kernels.npy`, `conv1-bias.txt`, etc.).

| Componente | Arquivo | É FLIM? |
|---|---|---|
| `Encoder` (CNN) | `src/models/models.py` | **SIM** — arquitetura do JSON |
| `parse_architecture()` | `src/models/models.py` | **SIM** |
| `load_FLIM_encoder()` | `src/models/models.py` | **SIM** — carrega pesos `.npy`/`.txt` |
| `LeJEPAFLIMModel` | `src/models/lejepa_flim.py` | não — wrapper SSL |
| `ConvDistillationProjectionHead` | `src/models/distillation.py` | não — ponte student→teacher |
| `DistillationProjectionHead` | `src/models/distillation.py` | não — variante linear |
| `FrozenTeacher` / `IJEPAEncoder` | `src/models/distillation.py` / `ijepa_encoder.py` | não — professor ViT-H/14 |
| `MSEDistillationLoss` / `KLDistillationLoss` | `src/models/distillation.py` | não — losses |

> **Nota sobre os pesos:** na destilação, o `Encoder` FLIM é inicializado com `trunc_normal`
> (pesos aleatórios). O `load_FLIM_encoder()` existe mas **não é chamado** no pipeline de
> destilação. O que vem do FLIM é apenas a **arquitetura** (estrutura das camadas), não os pesos.

---

## Dois módulos de destilação

| | `DistillationModule` | `DistillationConvModule` |
|---|---|---|
| Arquivo | `src/modules/distillation_module.py` | `src/modules/distillation_conv_module.py` |
| Cabeça de projeção | `DistillationProjectionHead` | `ConvDistillationProjectionHead` |
| Fluxo | pool(6×6) → flatten → Linear | 1×1 convs progressivos → GAP |
| Dimensão intermediária | 1728 (48×6×6) | mantém 24×24 até o final |
| Direção | 1728 → 1280 (desce) | 48 → 128 → 256 → 512 → 1280 (sobe) |
| Nome do run | `...modeldirect` | `...next_layers_direct` |

---

## STUDENT — FLIM CNN + ConvDistillationProjectionHead

### Arquitetura (ch24_32_48, input 200×200)

```
Input                          [B,  3, 200, 200]
│
├─ FLIM ENCODER  (arquitetura JSON — pesos: trunc_normal init)
│
│  conv1  Conv2d(3→24, 5×5, pad=2, dil=1)   →  [B, 24, 200, 200]   params:   1,800
│         ReLU
│         MaxPool2d(3×3, stride=2)           →  [B, 24,  99,  99]   params:      24
│                                                          subtotal:           1,824
│
│  conv2  Conv2d(24→32, 5×5, pad=2, dil=1)  →  [B, 32,  99,  99]   params:  19,200
│         ReLU
│         MaxPool2d(3×3, stride=2)           →  [B, 32,  49,  49]   params:      32
│                                                          subtotal:          19,232
│
│  conv3  Conv2d(32→48, 5×5, pad=2, dil=1)  →  [B, 48,  49,  49]   params:  38,400
│         ReLU
│         MaxPool2d(3×3, stride=2)           →  [B, 48,  24,  24]   params:      48
│                                                          subtotal:          38,448
│
│  feat_map = [B, 48, 24, 24]              FLIM encoder total:       59,504 params
│
│  → GlobalAvgPool2d(1) + flatten → [B, 48]   student_emb  (só logging, sem gradiente direto)
│
├─ CONV PROJECTION HEAD  (não é FLIM — ponte student→teacher dim)
│  Spatial 24×24 preservado durante toda a projeção
│
│  Conv2d(48→128,   1×1, no bias)   →  [B, 128,  24, 24]   params:   6,144
│  BatchNorm2d(128)                 →  [B, 128,  24, 24]   params:     256
│  ReLU
│
│  Conv2d(128→256,  1×1, no bias)   →  [B, 256,  24, 24]   params:  32,768
│  BatchNorm2d(256)                 →  [B, 256,  24, 24]   params:     512
│  ReLU
│
│  Conv2d(256→512,  1×1, no bias)   →  [B, 512,  24, 24]   params: 131,072
│  BatchNorm2d(512)                 →  [B, 512,  24, 24]   params:   1,024
│  ReLU
│
│  Conv2d(512→1280, 1×1, no bias)   →  [B,1280,  24, 24]   params: 655,360
│  BatchNorm2d(1280)                →  [B,1280,  24, 24]   params:   2,560
│
│  AdaptiveAvgPool2d(1)             →  [B,1280,   1,   1]
│  Flatten                          →  [B,1280]
│
│                                     ConvProjHead total:   829,696 params
│
└─ student_proj = [B, 1280]          STUDENT TOTAL:        889,200 params
```

### Campo receptivo do FLIM Encoder

```
conv1 (k=5):  RF =  5 px
pool1 (k=3):  RF =  7 px  | jump = 2
conv2 (k=5):  RF = 15 px  | jump = 2
pool2 (k=3):  RF = 19 px  | jump = 4
conv3 (k=5):  RF = 35 px  | jump = 4
pool3 (k=3):  RF = 43 px  | jump = 8
```

Cada posição em `feat_map [48, 24, 24]` representa uma janela de **43×43 px** na imagem original.
A conv 1×1 da proj head não altera o campo receptivo — apenas projeta canais.

---

## TEACHER — I-JEPA ViT-H/14 (frozen, sem gradiente)

```
Input                          [B,  3, 224, 224]
│
├─ PATCH EMBEDDINGS
│  Conv2d(3→1280, k=14, stride=14)  →  [B,1280,  16,  16]   params: 752,640
│  flatten + transpose              →  [B, 256, 1280]
│  + PositionEmbeddings (learnable) →  [B, 256, 1280]        params: 327,680
│
├─ ×32 TRANSFORMER BLOCKS  (cada bloco = 19,677,440 params)
│
│  ┌─ Block N ──────────────────────────────────────────────────────────────┐
│  │                                                                        │
│  │  LayerNorm(1280, eps=1e-6)         →  [B, 256, 1280]   params:  2,560 │
│  │                                                                        │
│  │  MULTI-HEAD ATTENTION (16 heads, head_dim=80, scale=1/√80)            │
│  │  Q = Linear(1280→1280, bias)       →  [B, 256, 1280]   params:1,639,680│
│  │  K = Linear(1280→1280, bias)       →  [B, 256, 1280]   params:1,639,680│
│  │  V = Linear(1280→1280, bias)       →  [B, 256, 1280]   params:1,639,680│
│  │  reshape → [B, 16, 256, 80]                                            │
│  │  attn = softmax(Q·Kᵀ / √80)       →  [B,  16, 256, 256]               │
│  │  out  = attn · V                   →  [B, 256, 1280]                   │
│  │  Linear(1280→1280, bias)           →  [B, 256, 1280]   params:1,639,680│
│  │  + residual                                                             │
│  │                    attn subtotal:                        6,558,720      │
│  │                                                                        │
│  │  LayerNorm(1280, eps=1e-6)         →  [B, 256, 1280]   params:  2,560 │
│  │                                                                        │
│  │  MLP                                                                   │
│  │  Linear(1280→5120, bias) + GELU   →  [B, 256, 5120]   params:6,558,720│
│  │  Linear(5120→1280, bias)           →  [B, 256, 1280]   params:6,554,880│
│  │  + residual                                                             │
│  │                    MLP subtotal:                        13,113,600      │
│  │                                                                        │
│  │  Total por bloco:                                       19,677,440      │
│  └────────────────────────────────────────────────────────────────────────┘
│
│  32 × 19,677,440 = 629,678,080 params
│
├─ LayerNorm(1280) final             →  [B, 256, 1280]   params:   2,560
│  Mean pooling sobre 256 patches    →  [B, 1280]
│
└─ teacher_emb = [B, 1280]          I-JEPA TOTAL:   630,762,240 params (frozen)
```

### Distribuição de parâmetros do I-JEPA

| Componente | Params | % total |
|---|---|---|
| PatchEmbeddings | 752,640 | 0.1% |
| PositionEmbeddings | 327,680 | 0.1% |
| 32× Attention | 209,879,040 | 33.3% |
| 32× MLP | 419,635,200 | 66.5% |
| LayerNorms | 166,400 | ~0% |
| **Total** | **630,762,240** | 100% |

---

## Comparação Direta

| | Student (FLIM + ConvProj) | Teacher (I-JEPA) |
|---|---|---|
| **Params totais** | **889,200** | **630,762,240** |
| **Ratio** | 1× | **709× maior** |
| **Input** | [B, 3, 200, 200] | [B, 3, 224, 224] |
| **Output** | [B, 1280] | [B, 1280] |
| **Tipo de operação** | Conv 5×5 + MaxPool | Self-Attention global |
| **Campo receptivo** | ~43×43 px (local) | 224×224 px (global — todo o patch grid) |
| **Indução espacial** | sim — translação equivariante | não — posição via embedding aprendido |
| **N° de camadas** | 3 conv + 4 conv 1×1 proj | 32 Transformer blocks |
| **Dim máxima interna** | 1280 (só na proj head) | 5120 (MLP interno de cada bloco) |
| **Normalização** | BN2d / BN1d | LayerNorm por patch |
| **Não-linearidade** | ReLU | GELU |
| **Gradiente** | treina | frozen |

---

## Fluxo completo de um training step (modo `direct`)

```
Imagem [B, 3, 200, 200]
    │
    ├─── STUDENT PATH (gradiente flui aqui) ──────────────────────────────────┐
    │                                                                          │
    │   FLIM Encoder (conv1→conv2→conv3)                                       │
    │   [B,3,200,200] → [B,48,24,24]                                           │
    │         │                                                                │
    │         ├──► GlobalAvgPool+flat → [B,48]   student_emb (só logging)     │
    │         │                                                                │
    │         └──► ConvDistillationProjectionHead                              │
    │              1×1 convs: 48→128→256→512→1280                              │
    │              GAP(1) + flatten → [B,1280]   student_proj                  │
    │                                                    │                    │
    ├─── TEACHER PATH (@no_grad, frozen) ────────────────┼────────────────────┘
    │                                                    │
    │   resize bilinear → [B,3,224,224]                  │
    │   I-JEPA ViT-H/14 (32 blocos Transformer)          │
    │   Mean pool patches → [B,1280]  teacher_emb        │
    │                                    │               │
    │                          MSE(student_proj, teacher_emb)
    │                          = loss
    │                          ↓
    │                          backward()
    │                          atualiza: FLIM Encoder + ConvProjHead
    │
    Loss flui por: MSELoss → proj[9] → proj[6] → proj[3] → proj[0] → conv3 → conv2 → conv1
```

### Params que recebem gradiente vs. frozen

| Componente | Params | Gradiente |
|---|---|---|
| FLIM conv1 | 1,824 | **treina** |
| FLIM conv2 | 19,232 | **treina** |
| FLIM conv3 | 38,448 | **treina** |
| ConvProjHead | 829,696 | **treina** |
| I-JEPA ViT-H/14 | 630,762,240 | frozen |

---

## Contexto dos módulos no código

```
src/
├── models/
│   ├── models.py              Encoder (FLIM), parse_architecture, load_FLIM_encoder
│   ├── lejepa_flim.py         LeJEPAFLIMModel (wrapper SSL em volta do Encoder)
│   ├── distillation.py        DistillationProjectionHead, ConvDistillationProjectionHead,
│   │                          FrozenTeacher, MSEDistillationLoss, KLDistillationLoss
│   └── ijepa_encoder.py       IJEPAEncoder (ViT-H/14 carregado do cache HuggingFace)
│
└── modules/
    ├── distillation_module.py       DistillationModule      (proj head: Linear 1728→1280)
    └── distillation_conv_module.py  DistillationConvModule  (proj head: 1×1 convs 48→1280)
```
