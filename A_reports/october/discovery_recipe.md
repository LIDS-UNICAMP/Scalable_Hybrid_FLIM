# Receita fim a fim do melhor modelo da discovery (`disc_bp01w64c1280`)

Documento para **reproduzir** o modelo e **rodar em outro dataset**, inclusive sem fundo
constante e sem máscara. Todo valor abaixo foi lido do `config.yaml` resolvido do run, dos
pesos em disco ou do código, com `arquivo:linha`. Caminhos de código são relativos à raiz do
worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery` (branch
`discovery_hybrid_spifil`). `SPiFiL/...` quer dizer `/dados/home/moliveira/SPiFiL/src/spifil/...`.
Artefatos: `/dados/home/moliveira/artifacts/spifil_growth/disc_bp01w64c1280/<ds>_split<s>_pct50/`.

---

## 1. Resumo em 5 linhas

1. **O que é:** uma CNN pequena de 4 camadas (32/48/64/1280 filtros, ~200 mil parâmetros) que
   aprende sozinha a redesenhar a imagem (autoencoder). Cada camada nasce de filtros recortados
   pelo SPiFiL e depois é ajustada por backprop.
2. **Como se lê:** a última camada vira um vetor de 1280 números (GeM p=3), e um SVM linear
   decide a classe.
3. **Resultado (κ de teste, 50%, média de 3 splits):** eggs **0.891**, larvae **0.939**,
   protozoan **0.886**. O I-JEPA dá 0.940 / 0.943 / 0.864. Protozoan passa o I-JEPA, larvae fica
   a 0.004 e eggs a 0.049.
4. **Em aberto:** o orçamento de 60 épocas (`disc_g20_base_e60`, só eggs s1 tem as 4 camadas) e
   as duas versões sem fundo (`disc_g20_nomask_e60`, `disc_g20_bg1_e60`), que ainda estão **na
   fila** e não têm nenhum ponto em disco.
5. **O que ainda falta entender:** eggs. Os eixos de treino de 1 knob dão empate; a hipótese
   atual é que falta campo receptivo.

---

## 2. Receita fim a fim, passo a passo

Pense numa casa construída **andar por andar**. Para cada andar:
(a) o SPiFiL escolhe os tijolos olhando poucas imagens rotuladas;
(b) o autoencoder "lixa" os tijolos tentando redesenhar a imagem, sem olhar rótulo;
(c) só então se constrói o próximo andar, em cima do andar já lixado.
No fim, um juiz simples (o SVM) olha o último andar e diz a classe.

### 2.1 Dados

| campo | valor | fonte |
|---|---|---|
| loader | `ift_lab`: pyift `LABNorm2`, 3 canais em [0,1] | `core/data/loaders.py:32-38`; `config.yaml` `data.init_args.loader` |
| image_size | 200 | `config.yaml` `model/data.init_args.image_size` |
| transform (treino, val, teste) | `Resize(200)` (lado menor) → `CenterCrop(200)`; passa por uint8 e volta a float (o LAB é quantizado em 256 níveis) | `core/data/transforms.py:50-64` |
| imagenet_norm | false | `config.yaml` |
| augmentation | **nenhuma**: `V_train=1` usa o transform de teste; `geo_aug=false`; o `build_aug` não entra | `core/data/multi_view_dataset.py:115`; `core/data/parasite_data_module.py:228-236` |
| batch | 32, `shuffle=True`, `drop_last=False`, 8 workers | `config.yaml`; `parasite_data_module.py:259-260` |
| rótulo | prefixo do nome do arquivo: `int("<classe>_...".split("_")[0]) - 1` | `core/data/parasite_dataset.py:106` |
| seed | `seed_everything: 42` | `config.yaml` |

**Splits.** Para cada split s ∈ {1,2,3}, a semente é `42 + s` (`core/data/splits.py:246`).
Cada classe é dividida meio a meio entre "pool de treino" e **teste**
(`splits.py:85-99`). Do pool de treino, `percentage` (aqui 50) vai para **train**, e o resto do
pool vira **validation** (`splits.py:108-160`). O JSON lido é
`splits_incremental/split{s}/data_descriptor_perc50.json` (`config.py:196-201`).

| dataset | classes | train | val | test |
|---|---|---|---|---|
| eggs | 9 | 1276 | 1279 | 2557 |
| larvae | 2 | 878 | 879 | 1757 |
| protozoan | 7 | 2390 | 2392 | 4786 |

(Contagem lida dos JSONs de split 1–3; é igual nos 3 splits.)

### 2.2 SPiFiL em cada camada

O SPiFiL é como **recortar carimbos**. Ele pega poucas imagens, divide cada uma em pedacinhos
(superpixels), recorta um quadradinho no centro de cada pedaço e guarda como filtro os
quadradinhos que mais separam as classes.

**Onde roda:** em processo, dentro do worker Ray, por `flim.spifil.grow`
(`experiments/ray/runners/growth.py:277-301`). A biblioteca `SPiFiL` em si **não foi alterada**
(seção 3.2).

| passo | o que acontece aqui | fonte |
|---|---|---|
| **1. Escolha das imagens** | `one_per_class=true` (default): **1 imagem por classe**, só do split de **treino** do pct50. `paper_classes=true`: só as classes que o artigo marcou (eggs 8 de 9, protozoan 6 de 7: a classe impureza fica fora; larvae 2 de 2). A imagem é o id do artigo (`scripts/spifil_paper_seeds.json`) **se** ele caiu neste train; senão, sorteio com `random.Random(42)`. | `flim/spifil.py:428-478`, `:258` |
| **2. Imagem e máscara** | RGB uint8, `Resize`+`CenterCrop` 200. Máscara do dataset com `NEAREST`, binarizada `>0`. | `flim/spifil.py:318-365` |
| **3. Superpixels** | caminho "na imagem" (`spifil_in_image` e `spifil_in_feature` ficam false, então cai no `else`). **DISF, n=50**, rodado no `LabNorm` do próprio SPiFiL sobre o RGB, **só dentro da máscara**. Cada superpixel vira 1 semente (medoide), com o rótulo da imagem. | `flim/spifil.py:572-600`; `SPiFiL/preparation.py:72-88`; `SPiFiL/superpixels/disf.py:88-103` |
| **4. Sementes viajam** | as coordenadas são reescaladas para a grade da camada: 200 (L1), 99 (L2), 49 (L3), 24 (L4). | `flim/spifil.py:410-425` |
| **5. De onde sai o recorte** | L1: da própria imagem `ift_lab` (encoder vazio = `Identity`). Lk: das ativações das camadas L1..L(k-1) **já treinadas** (best ckpt do estágio anterior), com o mesmo transform de teste. | `flim/spifil.py:788-792`, `:481-490` |
| **6. Ranking** | Mahalanobis com covariância **Ledoit-Wolf** (shrinkage) sobre todos os patches, score de **Fisher** = distância média às outras classes / distância média à própria classe, rank por classe. | `SPiFiL/learner.py:220-229`; `SPiFiL/metrics.py:65-110`; `SPiFiL/scoring.py:40-46` |
| **7. Seleção** | `DiversitySelector(alpha=0.5, pool_factor=3)`, mistura rank e diversidade (Euclidiana). `UniformAllocator`: `out_channels // n_classes` filtros por classe; o resto é descartado. | `SPiFiL/selection.py:76-107`; `SPiFiL/allocation.py:33-52` |
| **8. Filtro e bias** | z-score por feature (desvio + `stdev_factor` 0.01), norma 1, `K = z / desvio`, **bias = −(K · média)**. A conv calcula `unit_norm(zscore(patch)) · zscore(x)`. | `SPiFiL/nn/builder.py:3-11`, `:122-129`; `SPiFiL/config.py:46` |
| **9. Orçamento N ≤ D** | `budget_check: false`: **não** recusa quando há menos sementes que dimensões (o Ledoit-Wolf mantém a inversa definida). É o comportamento do SPiFiL original. | `flim/spifil.py:300-315` |
| **10. Grava** | pesos em formato FLIM (`conv{k}-kernels.npy`, `conv{k}-bias.txt`, `architecture.json`) em `L{k}_grow/`, mais `spifil_bundle/labels1.txt` (de que classe veio cada filtro). | `flim/spifil.py:603-621`, `:881` |

**Imagens escolhidas no pct50 (artigo + sorteio):** eggs s1 3+5, s2 4+4, s3 1+7; larvae s1 2+0,
s2 1+1, s3 0+2; protozoan s1 4+2, s2 5+1, s3 3+3. (Contado cruzando os ids do JSON do artigo com o
train de cada split.)

**Filtros por camada (lidos dos `conv{k}-bias.txt` e de `labels1.txt`):**

| camada | kernel | pool | eggs (8 classes) | larvae (2) | protozoan (6) |
|---|---|---|---|---|---|
| L1 | 5×5 | max 3×3 s2 | 32 (4/classe) | 32 (16/classe) | **30** (5/classe; o allocator joga 2 fora) |
| L2 | 5×5 | max 3×3 s2 | 48 | 48 | 48 |
| L3 | 5×5 | max 3×3 s2 | 64 | 64 | **60** |
| L4 | 1×1 | nenhum | 400 SPiFiL + **880 aleatórios** | 100 + **1180** | 300 + **980** |

**`fill_random_last` (só na L4).** Com 50 sementes por imagem e 1 imagem por classe, o SPiFiL só
tem 50 candidatos por classe, então entrega 50 × n_classes filtros. O que falta até 1280 é
**ruído gaussiano** (`torch.Generator` com seed 42), cada filtro reescalado para a **norma média
dos filtros SPiFiL** da camada, com **bias 0** (`flim/spifil.py:863-877`;
`growth.py:394-395` liga só em `k == layerwise`). Conferido: `conv4-bias.txt` tem exatamente
880 / 1180 / 980 zeros.

### 2.3 Crescimento camada a camada

Laço de `experiments/ray/runners/growth.py:353-412`, com `layerwise: 4`. Para k = 1..4:
`grow` cria `L{k}_grow/` e depois o estágio de treino `L{k}` treina a partir dele.

| estágio | o que treina | lr | decoder | parte de |
|---|---|---|---|---|
| L1 | **conv1 treina** (não fica congelada, ver nota) | 5e-4 | novo | SPiFiL de `L1_grow` |
| L2 | conv2 (nova) + conv1 (antiga) | nova 5e-4, antiga **5e-5** (×0.1) | novo | conv1 do best de L1, conv2 do SPiFiL |
| L3 | conv3 + conv1..2 | nova 5e-4, antigas 5e-5 | novo | best de L2 + SPiFiL |
| L4 | conv4 (1×1, 1280) + conv1..3 | nova 5e-4, antigas 5e-5 | novo | best de L3 + SPiFiL + aleatórios |

- **Nota sobre a L1.** O runner manda `freeze_encoder_flag=true` na L1 (`growth.py:399-401`,
  `config.yaml` da L1), mas com `fresh_decoder=true` o módulo segue outro ramo:
  `freeze_encoder(except_last=True)` congela tudo **menos a última camada**, que na L1 é a
  própria conv1 (`methods/autoencoder/autoencoder_flim_module.py:384-388`;
  `core/blocks/init.py:51-63`). Medido: a conv1 muda **11%** (norma relativa) entre `L1_grow` e
  o best da L1 em eggs s1.
- **L2..L4.** `layerwise_backprop: 0.1` vira `freeze_encoder_flag=false`, `old_layers=k-1`,
  `old_layer_lr_scale=0.1` (`growth.py:259-263`). O otimizador cria um grupo por escala
  (`autoencoder_flim_module.py:1084-1100`). Por isso não há `_assert_frozen`
  (`growth.py:404`).
- **`fresh_decoder`.** Do checkpoint anterior só viaja o encoder; o decoder nasce do zero a cada
  camada (`autoencoder_flim_module.py:521-523`).
- **Os `conv1..k-1` de `L{k}_grow/`** são cópia dos pesos SPiFiL crus (`flim/spifil.py:603-621`).
  Quem traz os treinados é o `init_ckpt`, carregado no fim do `__init__`
  (`autoencoder_flim_module.py:495-554`).
- **Sem regra de parada entre camadas:** são sempre 4 camadas. Com `evaluate.SVM: false` não
  existe κ por estágio (`growth.py:406-407`).

### 2.4 Loss

- **BCE com logits**, por pixel, contra a própria entrada LAB em [0,1] (3 canais)
  (`autoencoder_flim_module.py:642-652`, `:657-678`).
- **`recon_bg_weight: 0.1`.** Cada pixel ganha o peso `m = 0.1 + 0.9 · fg`. A loss é
  `Σ(bce · m) / (Σm · 3)` (`:670-678`).
- **Como sai o `fg`, sem rótulo e sem máscara:** pega-se a moldura de **4 px** nas 4 bordas, a
  **mediana** por canal dessa moldura é "a cor do fundo", e um pixel é objeto quando
  `‖LAB − mediana‖₂ > 0.04` (`:670-674`). Funciona porque estas imagens têm fundo constante.
- **Atenção:** a **validação usa a mesma loss ponderada** (`:718`). Então a escolha do checkpoint
  também depende do `fg`.

### 2.5 Otimizador, LR e seleção

| campo | valor | fonte |
|---|---|---|
| otimizador | AdamW, lr 5e-4, weight decay 0.05 | `config.yaml`; `autoencoder_flim_module.py:1101-1104` |
| warmup | `LinearLR`, 10 épocas, fator inicial 0.01 | `:1105` |
| cosine | `CosineAnnealingLR`, `T_max = max_epochs − 10`, `eta_min = 1e-5` (absoluto: o grupo das antigas também termina em 1e-5) | `:1106-1109` |
| passo do scheduler | por época | `:1110` |
| max_epochs | **200** (`disc_bp01w64c1280`); **60** no novo padrão em validação (`disc_g20_base_e60`) | YAMLs, `overrides` |
| early stopping | patience 50, `min_delta 0`, em **`probe/val_recon_loss` (min)** | `config.yaml`; `autoencoder_flim_module.py:589-612` |
| checkpoint | `save_top_k=1`, min `probe/val_recon_loss`; o arquivo chama `best_kappa.ckpt`, mas **não** é escolhido por κ | `:589-612` |
| `svm_probe` | **false**: o runner manda `--model.init_args.svm_probe=false` porque `evaluate.SVM: false` | `experiments/ray/runners/train.py:97` |

O `config.yaml` resolvido ainda mostra `monitor: probe/svm_kappa`, porque é o callback do YAML de
modelo. O `configure_callbacks` troca isso em tempo de execução, e o `run_metadata.json` confirma
`"ckpt_monitor": "probe/val_recon_loss"`.

Épocas do best na L4, split 1: eggs 126, larvae 66, protozoan 118. Nas L1–L3 o best cai na época
193–198 de 200 (`A_reports/october/discovery_early_stop.md:31`). Com 60 épocas, a patience de 50
praticamente nunca dispara.

### 2.6 Arquitetura

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| canais (eggs/larvae) | 32 | 48 | 64 | 1280 |
| canais (protozoan) | 30 | 48 | 60 | 1280 |
| kernel | 5 | 5 | 5 | 1 |
| padding | 2 ("same") | 2 | 2 | 0 |
| ativação | ReLU | ReLU | ReLU | ReLU |
| pool | max 3×3 s2, sem padding | idem | idem | nenhum |
| mapa de saída | 99×99 | 49×49 | 24×24 | 24×24 |

- Bloco = `Conv2d → ReLU → MaxPool2d` (`flim/arch.py:99-111`; `flim/spifil.py:624-651`).
- **Sem normalização** (`batch_norm=false`), **sem SE, GRN ou âncora** (`config.yaml`).
- **Parâmetros do encoder:** **200.944** (eggs e larvae) e **188.468** (protozoan). O
  `state_dict` mostra o dobro (401.888 / 376.936) porque cada bloco está registrado duas vezes,
  como `blocks.conv{n}` e `conv{n}`.
- **Decoder (só para treinar, não vai para o SVM):** um `ResidualUpBlock` por camada, em ordem
  inversa, que sobe a resolução pelo stride do pool. No fim, interpolação bilinear para 200×200
  e `Conv 3×3 → 3` logits (`methods/autoencoder/resnet_decoder.py:59-88`). São ~0,95 M valores no
  `state_dict` (eggs/larvae) e ~0,89 M (protozoan), contando os buffers do BN.

### 2.7 Leitura e SVM

- **Leitura:** o encoder roda conv1→conv4 e aplica **GeM p=3** no mapa 1280×24×24:
  `(média(max(x, 1e-6)³))^(1/3)` por canal → **1280-d** (`eval/svm.py:310-330`, `:353-355`). Em
  palavras simples: é uma média que dá mais peso aos picos, como perguntar "este carimbo apareceu
  forte em algum lugar?".
- **SVM** (`core/svm.py:143-185`): `sklearn.svm.SVC(kernel="linear", C=100, decision_function_shape="ovo", max_iter=-1, gamma="auto")`,
  **sem StandardScaler**, **sem grid search**. É ajustado no **train** do pct50 (as mesmas imagens
  do autoencoder, agora com rótulo) e prevê na **validation** (para escolher) ou no **test** (só
  para reportar) (`eval/svm.py:714-731`).
- **Métrica:** Cohen κ multiclasse do torchmetrics, mais acurácia balanceada e F1 macro
  (`core/metrics.py:46-76`).
- **Checkpoint avaliado:** `L4/checkpoints/best_kappa.ckpt` (`ckpt_selection: best`, `stages: [L4]`).

### 2.8 Comandos exatos

Rode um de cada vez. O SVM vem só depois do treino terminar e roda um por vez. Dá para ver o
plano antes com `--dry-run`.

```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/growth_disc_bp01w64c1280.yaml
```

```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/eval_svm_disc_bp01w64c1280_val_gem.yaml && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/eval_svm_disc_bp01w64c1280_test_gem.yaml
```

Para o novo padrão de 60 épocas, troque os YAMLs por `growth_disc_g20_base_e60.yaml` e
`eval_svm_disc_g20_base_e60_{val,test}_gem.yaml`.

- **`LD_LIBRARY_PATH`:** sem ele o import do PIL morre em `GLIBCXX_3.4.29` (`README.md:248-249`).
- **`RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`:** o `grow` roda dentro do worker Ray, que não tem
  `num_gpus`. Sem essa variável o Ray zera o `CUDA_VISIBLE_DEVICES` e o grow não acha GPU.
- **`skip: state`:** pula os pontos prontos (`execution_state.json`). Quando um run é morto, o
  relançamento pode refazer pontos; confira os "pulado(s)" no log.

---

## 3. O que mudamos no SPiFiL e no pipeline

### 3.1 Tabela

"Original" = o SPiFiL do artigo (`SPiFiL/conf/config.yaml` e `SPiFiL/conf/arch/paper.yaml`:
3 camadas k5 16/32/48, DISF 50, máscara, sem backprop) ou, quando indicado, o R1 desta branch
(`experiments/autoencoder/growth_disc_r1.yaml`). Os efeitos são **κ de validação, média de 3
splits**, do log de decisões (`A_reports/october/discovery_hybrid_spifil_decisions.md`).

| item | original | nosso | por quê | efeito medido |
|---|---|---|---|---|
| Como a rede é montada | o SPiFiL ajusta as 3 camadas em sequência, **sem backprop** | **híbrido**: cada camada sai das features do encoder **já treinado** pelo autoencoder (`growth.py:353-401`) | a camada nova tem que vir depois do que o modelo aprendeu | SPiFiL × init aleatória na mesma forma (L3, 3 seeds × 3 splits): **+0.017** no geral, protozoan **+0.046**, larvae +0.013, eggs −0.008 (empate) (log:214-218) |
| Larguras | 16/32/48 (paper), 24/32/48 (R1) | **32/48/64** + L4 1280 | o w64 deu +0.011 em protozoan (log:159-162) | bp01w64 vira base: L3 eggs 0.906, larvae 0.845, protozoan 0.865 (log:229-231) |
| Kernel/pool por camada | kernel único, max-pool 3/2 em todas | `layer_kernels: [5,5,5,1]`, `layer_pools: [2,2,2,1]` (`growth.py:361-369`, commit `e926563`) | permitir a L4 1×1 sem pool | — (é o que viabiliza a linha de baixo) |
| L4 1×1×1280 (nova) | não existe | 1×1, 1280 canais, 24×24 | o 1×1 SPiFiL = um "protótipo" de superpixel; 1280 protótipos = um "saco de palavras visuais" | dentro da leitura avg: +0.05 / +0.01 / +0.08 sobre a L3 (log:238-246) |
| `fill_random_last` (novo) | o SPiFiL entrega o que conseguir | completa até 1280 com ruído na norma média SPiFiL, bias 0 (`flim/spifil.py:863-877`, commit `3962811`) | o SPiFiL só tem 50 sementes por imagem | sem ablação isolada; mais filtros SPiFiL e menos aleatórios (sp100) **empata** em larvae (log:367-369) |
| Backprop nas antigas | — (sem backprop) | `layerwise_backprop: 0.1` (antigas lr×0.1, a nova lr×1) (commit `f262a45`) | o SPiFiL dá a partida e o backprop ajusta o conjunto | bp01 × fg01 em eggs L3: **+0.010** nos 3 splits (log:176-179). Lei 0.1→0.3→1.0: 0.899→0.905→0.908, sobe pouco (log:404-407) |
| Peso do fundo na recon | — (sem recon) / R1: BCE uniforme | `recon_bg_weight: 0.1` (commit `2af8c81`) | o fundo é ~97% dos pixels e "puxava" os filtros | **empate** 1.0 / 0.1 / 0 em L3 e L4 (log:422) |
| Decoder | — | novo a cada camada (`fresh_decoder`) | o decoder velho tem forma de outra rede | — |
| Seleção de checkpoint | — | min `val_recon_loss`, patience 50 (`autoencoder_flim_module.py:589-612`) | regra do usuário: decidir sem classificação | — |
| Épocas | — | 200 por camada → **60** em validação (`disc_g20_base_e60`) | o treino era lento; o best caía em 193–198 só porque o cosine ainda descia (log:413-417) | **pendente** (aceite: gem L4 a ≤0.018 de 0.899/0.910/0.875) |
| Leitura | SVM no mapa (flatten) ou na média | **GeM p=3 na L4, 1280-d** (commit `f263732`); nada de flatten, leitura < 2000-d | regra do usuário + literatura (A5) | gem × avg na L4: eggs 0.899 × 0.880, larvae 0.910 × 0.917, protozoan **0.875** × 0.837 (log:319-321) |
| Leituras `gem_fg` / `avgpool2d_fg` | — | só no objeto (fg sem rótulo), **só em eval** (`eval/svm.py:333-345`, commit `a14ced9`) | saber se o SVM precisa do fundo | gem_fg: 0.899 / 0.900 / 0.871 contra gem 0.899 / 0.910 / 0.875 → **empate**, o fundo é indiferente (`results/svm_disc_bp01w64c1280_val_gemfg.csv`; ainda não está no log) |
| SE / GRN / âncora L2-SP / BN / máscara MAE / geo_aug | — | opções `se_reduction`, `grn`, `spifil_anchor`, `batch_norm`, `mask_ratio`, `geo_aug`; **todas desligadas** no modelo (`config.yaml`) | candidatos da g20 | BN neutro em eggs (log:409-411); geo_aug +0.010 em larvae, dentro do ruído (log:371-374), neutro em eggs (log:399-402); SE/GRN/âncora **pendentes** |
| Escolha das imagens | 1 por classe, ids do artigo | ids do artigo **se** estiverem no train, senão sorteio seed 42 (`flim/spifil.py:428-478`) | a 5% só 3 dos 48 ids do artigo caem no train; forçar quebraria a escassez | — |
| Trava N ≤ D | não existe no SPiFiL | o wrapper tem a trava, mas aqui fica **desligada** (`budget_check: false`) | igual ao original | — |

### 3.2 O que **não** mudou no SPiFiL

- **O pacote `SPiFiL` está intocado:** 3 commits (`git log` em `/dados/home/moliveira/SPiFiL`) e
  árvore limpa. Tudo o que é novo mora no nosso wrapper `flim/spifil.py` ou no runner.
- Continuam os mesmos: superpixel **DISF com 50** por imagem, **máscara do dataset** (sementes só
  no parasita), semente = **medoide**, cor `LabNorm` para o superpixel, métrica **Mahalanobis +
  Ledoit-Wolf**, score **Fisher**, `DiversitySelector(0.5, 3)`, `UniformAllocator`, filtro
  z-score com `stdev_factor` 0.01 e bias `−K·média`, ReLU, max-pool 3×3 stride 2 e **1 imagem por
  classe** (`SPiFiL/learner.py:220-229`; `SPiFiL/conf/config.yaml`).
- Não usamos `grow_impl: a3` (z-score populacional): perdeu para o padrão (log:141-146).

---

## 4. Dependências de fundo e de rótulo (para rodar em dataset sem fundo)

### 4.1 Todos os lugares que assumem fundo constante, máscara ou rótulo

| # | dependência | onde | como desligar | custo / o que foi medido |
|---|---|---|---|---|
| 1 | **SPiFiL com a máscara do dataset**: o DISF só segmenta dentro da máscara, então toda semente cai no parasita | `flim/spifil.py:345-365` (`Cropped.load_mask`), `:812`; `SPiFiL/preparation.py:72-88` | `runner_args.impurities: true` → `load_mask` devolve `None` e as sementes vão para o quadro inteiro | sem máscara, ~97% dos candidatos são fundo. O Fisher **tende** a punir esses candidatos, mas não garante (`flim/spifil.py:346-357`). **Não medido**: `disc_g20_nomask_e60` está na fila |
| 2 | **SPiFiL com rótulos de 1 imagem por classe**: o Fisher e o allocator precisam saber a classe de cada semente | `SPiFiL/scoring.py:40-46`, `SPiFiL/allocation.py:47-52`; `flim/spifil.py:469-475` | **não há chave no YAML** que deixe o SPiFiL sem rótulo. O pacote tem `DistanceSumScorer` (`SPiFiL/scoring.py:49-55`), mas o `grow` não o expõe. A única saída sem rótulo é `random_layer_classic: true` (kaiming, sem SPiFiL) | o SPiFiL é **supervisionado por design**, com poucas imagens (8 em eggs) |
| 3 | **JSON de ids do artigo** | `flim/spifil.py:461`: `json[...][dataset]` sem `.get` | dataset novo: acrescente `"<ds>": {}` em `scripts/spifil_paper_seeds.json` **e** use `paper_classes: false`. Com `true` e entrada vazia, nenhuma classe sobra (`:467-468`) | sem a entrada: `KeyError` **mesmo com `paper_classes: false`** |
| 4 | **Peso do fundo na recon pela mediana da borda** (fundo = cor da moldura de 4 px) | `methods/autoencoder/autoencoder_flim_module.py:670-674`; vale também na validação, que escolhe o checkpoint (`:718`) | `recon_bg_weight: 1.0` → BCE plana; o atalho está em `:661-663` | **empate 1.0 / 0.1 / 0** (log:422). Em dataset sem fundo constante a mediana da borda é lixo, e o `fg` vira sorteio |
| 5 | **Leituras `gem_fg` / `avgpool2d_fg`** (mesma mediana da borda) | `eval/svm.py:333-345` | `embed_mode: gem` (ou `avgpool2d`) no YAML de eval | gem_fg **empata** com gem (seção 3.1). O fundo não é necessário na leitura |
| 6 | **Rótulo pelo nome do arquivo** `<classe>_<id>.png`, classes a partir de 1 | `core/data/parasite_dataset.py:106`; `SPiFiL/data.py:30` | renomear os arquivos | — |
| 7 | **Registro do dataset**: `config.DATASETS["parasito"]` (`images`, `masks`, `splits`, `splits_incremental`), `PARASITE_NAME`, `NUM_CLASSES`, `DATASET_NUM_CLASSES`, `DATASET_LONG_TO_SHORT`, `FLIM_ARCH_BASE`/`FLIM_WEIGHTS_BASE`, `configs/dataset/<ds>/split_<s>.yaml`, `configs/model/autoencoder/flim/<ds>/split_<s>.yaml` | `config.py:70-83`; `core/constants.py:67-72`, `:151-162`, `:192-196`; `experiments/ray/paths.py:87`, `:92` | criar as entradas | o `layerwise` "do zero" **ainda lê um `architecture.json` FLIM de molde** (`growth.py:203-213`; `flim/spifil.py:787-792`): sem ele o L1 não nasce |
| 8 | **Pasta de máscaras** | `flim/spifil.py:449` | com `impurities: true` ela pode faltar: o SPiFiL só avisa "No mask found" por imagem (`SPiFiL/data.py:164-177`) | — |
| 9 | **Recorte central** `Resize(lado menor)` + `CenterCrop(200)` | `core/data/transforms.py:50-56`; `flim/spifil.py:331-334` | — | em imagem não quadrada as bordas somem; supõe objeto no centro |
| 10 | **Classe impureza sem filtro SPiFiL** (`paper_classes`) | `flim/spifil.py:465-468` | `paper_classes: false` | em eggs o SVM decide 9 classes, mas só 8 têm filtros SPiFiL |
| 11 | **Análise de drift** (`act_drift/`) usa a máscara do dataset | log:354-360 | só análise, não entra no modelo | — |

### 4.2 O que está rodando agora

- `disc_g20_nomask_e60` = `base_e60` + `impurities: true`
  (`experiments/autoencoder/growth_disc_g20_nomask_e60.yaml`).
- `disc_g20_bg1_e60` = `base_e60` + `recon_bg_weight: 1.0`
  (`experiments/autoencoder/growth_disc_g20_bg1_e60.yaml`).
- Os dois rodam nos 3 datasets × 3 splits, e o eval é `eval_svm_disc_g20_{nomask,bg1}_e60_{val,test}_gem.yaml`.
- **Estado em 02/11:** os dois estão **na fila de GPU**, em 7º e 8º lugar, e ainda não existe
  nenhuma pasta de saída. A própria `base_e60` só tem eggs s1 com L4 pronta.

### 4.3 Receita sem fundo recomendada (até sair o resultado)

Parta de `growth_disc_g20_base_e60.yaml` e mude só `runner_args` e `overrides`:

```yaml
runner_args:
  # ... igual à base_e60 ...
  impurities:    true    # SPiFiL sem máscara (não medido; nomask_e60 na fila)
  paper_classes: false   # dataset novo: sem ids do artigo; 1 imagem por classe por sorteio (seed 42)
overrides:
  model.init_args.recon_bg_weight: 1.0   # BCE plana; 1.0 ≈ 0.1 ≈ 0 medido
```

- Acrescente `"<ds>": {}` em `scripts/spifil_paper_seeds.json` e registre o dataset (linha 7 da
  tabela 4.1).
- No eval use `embed_mode: gem` (nunca `gem_fg`), na L4.
- **Risco:** o `recon_bg_weight: 1.0` e o `gem` sem fg já têm medida de empate. O `impurities: true`
  **não tem**. A combinação dos dois também não foi medida.
- Se o `nomask` perder muito κ, o plano B é fornecer uma máscara grosseira do objeto. Não existe
  hoje outra forma de manter as sementes fora do fundo.

---

## 5. Leis e achados até agora

Ruído de referência: o κ de um re-run varia ≈ 0.01 por split (log:131), a leitura avg ≈ 0.03
(log:248-250), e o pipeline inteiro não repete bit a bit (log:381-384). **Só decida pela média
dos 3 splits.**

| lei / achado | números | fonte |
|---|---|---|
| **SPiFiL contribui como inicializador** | contra aleatório (L3): +0.046 protozoan, +0.013 larvae, −0.008 eggs; o aleatório varia 0.012–0.022 entre seeds, o SPiFiL ≤0.008 | log:214-218 |
| SPiFiL **não** acelera a otimização | o aleatório chega a uma loss menor em 16 de 21 estágios | log:122-125 |
| **Backprop nas antigas:** sobe pouco e de forma monotônica | eggs 0.899 → 0.905 → 0.908 (bp 0.1/0.3/1.0); larvae 0.910 → 0.914 | log:404-407 |
| **Profundidade (mapa pequeno) custa κ** | R2, eggs: L3 0.898 → L8 ~0.81 | log:127-132 |
| **L4 1×1 só ajuda quando a leitura joga fora a posição** | avg: +0.04; 1×1×96 lida achatada: −0.007 | log:238-258 |
| **A leitura pesa mais que quase todo knob de treino** | ~0.09 entre avg e flatten na mesma L3 | log:233-237 |
| **GeM na L4 é a melhor leitura < 2000-d** | 0.899 / 0.910 / 0.875 (val) | log:319-321 |
| **Kernel 3 perde** | e_k3 −0.024 em eggs (passa do ruído) | log:391-393 |
| **Peso do fundo não importa** | 1.0 ≈ 0.1 ≈ 0 | log:422 |
| **O fundo não importa no SVM** | gem_fg ≈ gem | CSV (seção 3.1) |
| **O treino apaga o contraste objeto/fundo do SPiFiL**, sem custar κ | energia dentro/fora: L2 2.0 → 1.2, L3 2.9 → 1.0 | log:354-360 |
| **Sementes 50 → 100: plano** | larvae 0.903 contra 0.910 | log:367-369 |
| **geo_aug** | larvae +0.010 (ruído); eggs neutro | log:371-374, :399-402 |
| **Platô de 1 knob** | eggs ~0.90, larvae ~0.91 (val) | log:386-390, :399-402 |
| **Teste final** | 0.891 / 0.939 / 0.886 contra o I-JEPA 0.940 / 0.943 / 0.864 | log:345-349; `discovery_ranking_test.md:5-7` |
| **Melhor de cada dataset pela val** (ainda sem teste) | eggs `g20_e_bp10` 0.908; larvae `g20_l_aug` 0.919; protozoan bp01w64c1280 0.875 | `discovery_ranking_test.md:5-7` |
| **Hipótese aberta para eggs** | campo receptivo das L1–L3 ~40 px ≤ tamanho do ovo; leis d5/d6/dil2/sp25/sp100 na fila | log:431-441 |

---

## 6. Checklist de reprodução

- [ ] Env `scalable_FLIM`, `spifil` editable de `/dados/home/moliveira/SPiFiL` (commit `4845021`, árvore limpa).
- [ ] `data/` presente (imagens, máscaras, splits e o FLIM de molde em `data/to_mateus/model/...`): vem de jaci, não do git.
- [ ] `export LD_LIBRARY_PATH=.../scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` no shell do lançamento.
- [ ] `--dry-run` mostra 4 estágios `L1..L4` por braço e um `grow` por camada.
- [ ] `L4_grow/conv{1..4}-bias.txt`: a primeira linha dá 32/48/64/1280 (protozoan 30/48/60/1280).
- [ ] `L4_grow/conv4-bias.txt` tem 880 / 1180 / 980 zeros (eggs/larvae/protozoan).
- [ ] `L{k}_grow/spifil_bundle/labels1.txt`: L4 com 50 filtros por classe (eggs classes 1..8, sem a 9).
- [ ] O log do grow mostra `classe X: <id> (artigo|sorteio)`, batendo com 3+5 / 4+4 / 1+7 em eggs.
- [ ] `L4/config.yaml`: `lr 0.0005`, `weight_decay 0.05`, `warmup_epochs 10`, `recon_bg_weight 0.1`, `old_layers 3`, `old_layer_lr_scale 0.1`, `fresh_decoder true`, `svm_probe false`, `batch_size 32`, `loader ift_lab`, `geo_aug false`.
- [ ] `L1/config.yaml`: `freeze_encoder_flag true`, `old_layers 0`. Mesmo assim, a conv1 muda (~11% em eggs s1).
- [ ] `L4/run_metadata.json`: `"ckpt_monitor": "probe/val_recon_loss"`. O campo `best_val_svm_kappa` guarda a **loss** (ex.: 0.5126), não um κ.
- [ ] Parâmetros do encoder: 200.944 (eggs/larvae) e 188.468 (protozoan); o state_dict mostra o dobro.
- [ ] Eval: `embed_mode: gem`, `stages: [L4]`, `ckpt_selection: best`; a linha do CSV tem `dim = 1280`.
- [ ] κ de teste dentro de ~±0.02 de 0.891 / 0.939 / 0.886 (o pipeline não é bit-reprodutível).
- [ ] O SVM só roda depois do treino, um por vez.

---

### Armadilhas encontradas ao escrever este documento

- **A L1 não fica congelada**, apesar de `freeze_encoder_flag: true` (seção 2.3).
- No `run_metadata.json`, **`best_val_svm_kappa` é a val recon loss**, e `"architecture": "ch24_32_48"` é um rótulo fixo
  (`experiments/run_metadata_callback.py:172`, `:191-197`). O `best_kappa.ckpt` é escolhido pela loss.
- **Dataset novo quebra em `flim/spifil.py:461`** sem entrada no JSON do artigo, com qualquer
  `paper_classes`. Ele também precisa de um `architecture.json` FLIM de molde.
- **O SPiFiL usa rótulo** (Fisher + allocator) e não tem chave para rodar sem rótulo.
- A loss de **validação** também é ponderada pelo fg da borda, então o fundo entra na escolha do checkpoint.
