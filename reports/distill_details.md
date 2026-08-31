# Detalhes dos experimentos de destilação

Modelo treinado e contagem de parâmetros por agrupamento de runs (wandb `ophira-ai/flim-ssl`).

---

## Convenções

- **O que conta:** `encoder FLIM + cabeça de destilação (proj_kd)`.
- **O que não conta:** teacher I-JEPA ViT-H/14 (630.762.240 params, congelado) e o
  `multi_layer_perceptron` vestigial (4.825.600 params) — ver [Armadilhas](#armadilhas-de-contagem).
- **Dataset de referência:** protozoan (é o dos 6 runs de exemplo). Eggs/larvae entre parênteses.
- **Por que diferem:** protozoan tem `conv2` com **30** canais; eggs/larvae com **32**. Δ = 3.602 params.

### Backbone comum a todos os grupos

Encoder FLIM de 3 blocos, entrada 200×200. Idêntico nos seis grupos — só muda a inicialização.

```
[B,3,200,200]
     │
     ├─ conv1  Conv2d(3→24, 5×5, pad=2, bias=True) ──> [B,24,200,200]      1.824
     │  ReLU ──> MaxPool2d(3×3, s=2) ──────────────> [B,24, 99, 99]
     │
     ├─ conv2  Conv2d(24→30, 5×5, pad=2, bias=True) ─> [B,30, 99, 99]     18.030
     │  ReLU ──> MaxPool2d(3×3, s=2) ──────────────> [B,30, 49, 49]
     │
     ├─ conv3  Conv2d(30→48, 5×5, pad=2, bias=True) ─> [B,48, 49, 49]     36.048
     │  ReLU ──> MaxPool2d(3×3, s=2) ──────────────> [B,48, 24, 24]
     ▼
  feat_map [B,48,24,24]                        encoder FLIM = 55.902 params
```

> Eggs/larvae: `conv2` 24→**32** (19.232) e `conv3` 32→48 (38.448) → encoder **59.504**.

---

## 1. `2l_1x1_init_flim_256_1280_no_imagenet_norm`

- **Módulo:** `DistillationTwoLayerModule` — [src/modules/distillation_twolayer_module.py](src/modules/distillation_twolayer_module.py)
- **Cabeça:** `TwoLayer1x1ConvBN2dDistillationProjectionHead`
- **Init do encoder:** pesos FLIM reais (`encoder_init="flim"`), encoder **treinável**
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado (1280-d)
- **Normalização:** LAB[0,1] **sem** `Normalize` ImageNet — afeta só o pré-processamento
- **Parâmetros:** **398.942** = 55.902 + 343.040 · *(eggs/larvae: 402.544)*

```
feat_map [B,48,24,24]
     │
     ├─ Conv2d(48→256, 1×1, bias=False) ──> [B, 256,24,24]     12.288
     │  BatchNorm2d(256) ─> GELU                                   512
     │
     ├─ Conv2d(256→1280, 1×1, bias=False) ─> [B,1280,24,24]   327.680
     │  BatchNorm2d(1280) ─> GELU                                2.560
     │
     └─ AdaptiveAvgPool2d(1) ──────────────> [B,1280]
     ▼
  student_proj [B,1280]                        cabeça = 343.040 params
```

---

## 2. `1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm`

- **Módulo:** `DistillationOneLayerModule` (`proj_kernel=1`) — [src/modules/distillation_onelayer_module.py](src/modules/distillation_onelayer_module.py)
- **Cabeça:** `OneLayer1x1ConvDistillationProjectionHead`
- **Init do encoder:** pesos FLIM reais
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado
- **Normalização:** `--no-imagenet-norm` ativo
- **Nota:** variante **mais enxuta** da família — uma única camada de projeção
- **Parâmetros:** **119.902** = 55.902 + 64.000 · *(eggs/larvae: 123.504)*

```
feat_map [B,48,24,24]
     │
     ├─ Conv2d(48→1280, 1×1, bias=False) ──> [B,1280,24,24]    61.440
     │  BatchNorm2d(1280) ─> GELU                                2.560
     │
     └─ AdaptiveAvgPool2d(1) ──────────────> [B,1280]
     ▼
  student_proj [B,1280]                         cabeça = 64.000 params
```

---

## 3. `3x3_BN2d_1280_one_layer_init_flim`

- **Módulo:** `DistillationOneLayerModule` (`proj_kernel=3`, default) — [src/modules/distillation_onelayer_module.py](src/modules/distillation_onelayer_module.py)
- **Cabeça:** `OneLayerConvDistillationProjectionHead`
- **Init do encoder:** pesos FLIM reais (`proj_type=3x3_init_flim_1280`, init forçado pelo variante)
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado
- **Normalização:** ImageNet **ligada**
- **Nota:** mesma topologia do grupo 2, mas kernel 3×3 → **9× o custo** da projeção
- **Parâmetros:** **611.422** = 55.902 + 555.520 · *(eggs/larvae: 615.024)*

```
feat_map [B,48,24,24]
     │
     ├─ Conv2d(48→1280, 3×3, pad=1, bias=False) ──> [B,1280,24,24]   552.960
     │  BatchNorm2d(1280) ─> GELU                                      2.560
     │
     └─ AdaptiveAvgPool2d(1) ─────────────────────> [B,1280]
     ▼
  student_proj [B,1280]                            cabeça = 555.520 params
```

---

## 4. `2l_1x1_init_flim_256_1280`

- **Módulo:** `DistillationTwoLayerModule` — [src/modules/distillation_twolayer_module.py](src/modules/distillation_twolayer_module.py)
- **Cabeça:** `TwoLayer1x1ConvBN2dDistillationProjectionHead`
- **Init do encoder:** pesos FLIM reais
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado
- **Normalização:** ImageNet **ligada** — é a **única** diferença para o grupo 1
- **Nota:** arquitetura idêntica ao grupo 1. A flag `--no-imagenet-norm` só chega ao
  `ParasiteLejepaDataModuleSplited` (`imagenet_norm = not args.no_imagenet_norm`) e nunca ao
  construtor do modelo — mesma `arch_json`, mesmos `proj_dim=256` / `teacher_dim=1280`
- **Parâmetros:** **398.942** = 55.902 + 343.040 · *(eggs/larvae: 402.544)*

```
feat_map [B,48,24,24]
     │
     ├─ Conv2d(48→256, 1×1, bias=False) ──> [B, 256,24,24]     12.288
     │  BatchNorm2d(256) ─> GELU                                   512
     │
     ├─ Conv2d(256→1280, 1×1, bias=False) ─> [B,1280,24,24]   327.680
     │  BatchNorm2d(1280) ─> GELU                                2.560
     │
     └─ AdaptiveAvgPool2d(1) ──────────────> [B,1280]
     ▼
  student_proj [B,1280]                        cabeça = 343.040 params
```

---

## 5. `next_layers_direct`

- **Módulo:** `DistillationConvModule` — [src/modules/distillation_conv_module.py](src/modules/distillation_conv_module.py)
- **Cabeça:** `ConvDistillationProjectionHead`
- **Init do encoder:** `trunc_normal` — só a **arquitetura** vem do FLIM, os pesos **não** são carregados
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado
- **Nota:** variante **mais profunda** (4 camadas de projeção); resolução 24×24 preservada até o fim
- **Parâmetros:** **885.598** = 55.902 + 829.696 · *(eggs/larvae: 889.200)*

```
feat_map [B,48,24,24]
     │
     ├─ Conv2d(48→128, 1×1, bias=False) ──> [B, 128,24,24]      6.144
     │  BatchNorm2d(128) ─> ReLU                                   256
     │
     ├─ Conv2d(128→256, 1×1, bias=False) ─> [B, 256,24,24]     32.768
     │  BatchNorm2d(256) ─> ReLU                                   512
     │
     ├─ Conv2d(256→512, 1×1, bias=False) ─> [B, 512,24,24]    131.072
     │  BatchNorm2d(512) ─> ReLU                                 1.024
     │
     ├─ Conv2d(512→1280, 1×1, bias=False) > [B,1280,24,24]    655.360
     │  BatchNorm2d(1280)          (sem ReLU final)              2.560
     │
     └─ AdaptiveAvgPool2d(1) ─> Flatten ──> [B,1280]
     ▼
  student_proj [B,1280]                        cabeça = 829.696 params
```

---

## 6. `pct75_modeldirect`

- **Módulo:** `DistillationModule` — [src/modules/distillation_module.py](src/modules/distillation_module.py)
- **Cabeça:** `DistillationProjectionHead` — **variante linear**, a única sem cabeça convolucional
- **Init do encoder:** `trunc_normal` (pesos FLIM não carregados)
- **Destilação:** `direct` (MSE) contra I-JEPA ViT-H/14 congelado
- **Nota 1:** `pct75` = fração de dados de treino (75%), **não** uma variação de arquitetura —
  pct1…pct100 compartilham o mesmo modelo
- **Nota 2:** variante **mais pesada**, por causa da matriz densa 1728×1280
- **Parâmetros:** **2.270.302** = 55.902 + 2.214.400 · *(eggs/larvae: 2.273.904)*

```
feat_map [B,48,24,24]
     │
     ├─ AdaptiveAvgPool2d(6) ──────────────> [B,48,6,6]
     ├─ Flatten ───────────────────────────> [B,1728]        (48·6·6)
     │
     ├─ Linear(1728→1280, bias=False) ─────> [B,1280]     2.211.840
     │  BatchNorm1d(1280)   (sem ReLU, sem camada intermediária)  2.560
     ▼
  student_proj [B,1280]                      cabeça = 2.214.400 params
```

---

## Resumo comparativo (protozoan)

| # | Agrupamento | Módulo | Cabeça | Init | Params |
|---|---|---|---|---|---|
| 2 | `1x1_BN2d_1280_one_layer_flim_init_no_imagenet_norm` | `OneLayer` | 1×1: 48→1280 | FLIM | **119.902** |
| 1 | `2l_1x1_init_flim_256_1280_no_imagenet_norm` | `TwoLayer` | 1×1: 48→256→1280 | FLIM | **398.942** |
| 4 | `2l_1x1_init_flim_256_1280` | `TwoLayer` | 1×1: 48→256→1280 | FLIM | **398.942** |
| 3 | `3x3_BN2d_1280_one_layer_init_flim` | `OneLayer` | 3×3: 48→1280 | FLIM | **611.422** |
| 5 | `next_layers_direct` | `Conv` | 1×1: 48→128→256→512→1280 | `trunc_normal` | **885.598** |
| 6 | `pct75_modeldirect` | `Module` | linear: pool6 → 1728→1280 | `trunc_normal` | **2.270.302** |

Ordenado por nº de parâmetros. Para eggs/larvae, some 3.602 a cada valor.

---

## Armadilhas de contagem

`sum(p.numel())` sobre o `state_dict` do checkpoint dá o número **errado**. Três motivos:

1. **MLP vestigial.** `LeJEPAFLIMModel` sempre instancia
   `multi_layer_perceptron = ProjectionHead(48→2048→2048→256)` = **4.825.600 params**.
   No modo `direct` ele nunca é chamado no forward e não recebe gradiente — mas está no
   `state_dict`. Contar cru daria ~5,2M em vez de 398.942 no grupo 1.
2. **Encoder duplicado.** `Encoder` registra os blocos duas vezes (`blocks.conv1...` e
   `conv1...`, apontando para os mesmos objetos). `parameters()` deduplica; o `state_dict` não.
3. **Protozoan ≠ eggs/larvae.** `conv2` tem 30 canais no protozoan, lido por
   `get_actual_channels_from_weights` da 1ª linha de `conv{n}-bias.txt`.

---

## Notas de verificação

- **Nenhum `.ckpt` sobreviveu em disco** para estes seis grupos: os diretórios `checkpoints/`
  em `artifacts/distillation/` estão vazios e `best_checkpoint` aponta para
  `/dados/home/moliveira/scalable_FLIM_self_supervised/...`, caminho que não existe mais.
  Todas as contagens vieram de **instanciar os modelos pelo próprio código** (torch 2.2.2,
  env `scalable_FLIM`), reproduzindo o caminho real
  `parse_architecture` → `get_actual_channels_from_weights` → `override_arch_channels` →
  `LeJEPAFLIMModel` + cabeça. Um checkpoint de run gêmeo (`cosine_flim_*_one_layer_flim_init`)
  confirmou o valor do grupo 2.
- **Correção a [distillation_model_architecture.md](distillation_model_architecture.md):**
  as linhas `MaxPool2d(3×3, stride=2) params: 24 / 32 / 48` estão mal rotuladas — `MaxPool2d`
  não tem parâmetros. Esses valores são os **bias das `Conv2d`** (bias=True por default em
  `build_encoder_from_arch`). Subtotais e total (59.504) estão corretos.
- **Números desatualizados no repo:** [metrics_distillation/summary_distillation.md](metrics_distillation/summary_distillation.md)
  e [scripts/plot_parameters_vs_metrics.py](scripts/plot_parameters_vs_metrics.py) assumem
  `ch24_32_48` genericamente e superestimam o protozoan em 3.602 params.
- **`encoder_init=trunc_normal` é quase um no-op:** `init_weights_trunc_normal` aplica
  `timm.init_weights_vit_timm`, que só faz trunc-normal em `nn.Linear`. Como o encoder só tem
  `Conv2d`, cai em `reset_parameters()` → **kaiming_uniform padrão do PyTorch**.
- **`DistillationConvModule` não herda `FrozenTeacherCheckpointMixin`**, então nos runs
  `next_layers_direct` o teacher I-JEPA (~630M params) foi salvo junto nos checkpoints.
  Não afeta a contagem do student, mas explica o tamanho desses arquivos.
- **Grupo homônimo:** existe `..._3x3_BN2d_1280_one_layer_flim_init` (sufixo `_flim_init`,
  gerado por `proj_type=3x3_bn2d_1280` + `--flim-init`), distinto do grupo 3. Arquitetura e
  contagem idênticas; só o nome e a rota de lançamento diferem.
