# VICReg (e VICRegL): regras de implementação fiéis ao artigo e ao código oficial

> Escopo: como implementar VICReg dentro de `ga_search/modehb/ssl_trainers.py::train_global_ssl`, o que é
> **fiel** ao artigo/código e o que é **adaptação** ao nosso caso (CNN rasa com SPiFiL, 224 px,
> datasets de 1k a 10k imagens, GPU de 48 GB, batch 64 fixo).
> Fontes lidas: PDF arXiv:2105.04906v3 (ICLR 2022), PDF arXiv:2210.01571 (VICRegL, NeurIPS 2022),
> `facebookresearch/vicreg@main` (`main_vicreg.py`, `augmentations.py`, `evaluate.py`, `README.md`),
> `facebookresearch/VICRegL@main` (`main_vicregl.py`, `utils.py`). Lidos em 17/11/2026.
> Convenção: **[A]** = artigo, **[C]** = código oficial, **[N]** = nosso código, **[H]** = hipótese/derivação nossa (não está no artigo).

---

## 1. Referência

- Bardes, A., Ponce, J., LeCun, Y. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022. arXiv:2105.04906.
- Bardes, A., Ponce, J., LeCun, Y. *VICRegL: Self-Supervised Learning of Local Visual Features*. NeurIPS 2022. arXiv:2210.01571.
- Código oficial: https://github.com/facebookresearch/vicreg (`main_vicreg.py`, `augmentations.py`, `evaluate.py`) e https://github.com/facebookresearch/VICRegL (`main_vicregl.py`, `utils.py`).

## 2. Resumo

- Arquitetura de *joint embedding* com dois ramos (siameses e com pesos compartilhados no caso padrão): encoder `f_θ` → **representação** `y` (usada a jusante) → **expander** `h_φ` → **embedding** `z` (onde a perda é calculada). O expander é descartado após o pré-treino ([A] §4, Fig. 1).
- A perda tem três termos ([A] §4.1, Eq. 1–6):
  - **invariância** `s`: MSE entre `z` e `z'`, sem normalização;
  - **variância** `v`: hinge no **desvio-padrão** (não na variância) de cada dimensão ao longo do batch, alvo γ=1;
  - **covariância** `c`: soma dos quadrados dos elementos fora da diagonal da covariância de cada ramo, dividida por `d`.
- `v` e `c` são aplicados a **cada ramo separadamente**. Isso dispensa stop-gradient, preditor, EMA, normalização do embedding e batch grande ([A] §2). Sem `v`, colapsa ([A] Tab. 7).
- Conclusões que importam para nós: (i) depende muito da **dimensão do expander** ([A] Tab. 12: 256→55,9%, 8192→68,6%); (ii) é **pouco sensível a batch** entre 128 e 4096 ([A] Tab. 13), desde que o lr base seja ajustado; (iii) os mesmos coeficientes funcionaram em MNIST e CIFAR ([A] App. D.4); (iv) há um precedente oficial de dataset pequeno (ESC-50, 1600 amostras) treinado com **AdamW lr 5e-4, 100 épocas** ([A] App. D.2).
- VICRegL estende o critério para **mapas locais** (vetores de feature por posição), casando posições pela localização na imagem e pelo vizinho mais próximo em L2. É relevante para o SPiFiL-2 (invariância local entre vistas).

## 3. Receita oficial (ResNet-50, ImageNet)

| Item | Artigo [A] | Código oficial [C] | Observação |
|---|---|---|---|
| Perda | `λ·s + μ·[v(Z)+v(Z')] + ν·[c(Z)+c(Z')]` (Eq. 6) | `sim·mse + std·(mean(relu(1-std_x))/2 + mean(relu(1-std_y))/2) + cov·(off(cov_x)²/D + off(cov_y)²/D)` (`main_vicreg.py:194-220`) | **O código divide o termo de variância por 2**; a Eq. 6 e o Alg. 1 ([A] App. A) não. O μ efetivo do código é 12,5 por ramo. Siga o código. |
| λ (inv), μ (var), ν (cov) | 25, 25, 1 ([A] §4.2; Tab. 7) | `--sim-coeff 25 --std-coeff 25 --cov-coeff 1` (`:58-63`) | Regra de estabilidade: λ=μ, ν<μ; ν>μ ou λ≠μ muito diferentes → instável ([A] D.4). |
| γ (alvo do std) | 1 ([A] Eq. 1) | `1 - std` (`:207`) | |
| ε | 1e-4 ([A] §4.2) | `x.var(dim=0) + 0.0001` (`:205`) | `var` não-viesada (N−1), padrão do PyTorch. |
| Covariância | `1/(n−1) Σ (z−z̄)(z−z̄)ᵀ` (Eq. 3) | `x.T @ x / (batch_size − 1)` após centrar (`:202-210`) | O código usa o batch **global** (`args.batch_size`) após `FullGatherLayer`. |
| Normalização por d | `1/d` no termo de covariância (Eq. 4) | `.div(num_features)` = 8192 (`:211-213`) | |
| Invariância | `1/n Σ‖z−z'‖²` (Eq. 5) | `F.mse_loss(x, y)` (`:198`) = média sobre n **e d** | Diferem por um fator d. O código é a referência (λ=25 foi escolhido com o código). |
| Encoder | ResNet-50, saída 2048 | `resnet50(zero_init_residual=True)` (`:189-191`) | |
| Expander | 3 FC de 8192; BN+ReLU nas duas primeiras ([A] §4.2) | `Linear(+bias)→BN→ReLU` ×2 + `Linear(bias=False)` (`Projector`, `:223-232`) | Sem BN na saída: o embedding **não** é normalizado ([A] Tab. 8, D.5). |
| Avaliação | Representação do encoder; expander descartado ([A] §4) | salva `backbone.state_dict()` (`:164`) | Avaliação **antes** do expander. |
| Augmentations | BYOL "simetrizada": crop 0.08–1 → 224, flip 0.5, ColorJitter(0.4,0.4,0.2,0.1) p=0.8, grayscale 0.2, blur p=0.5 (kernel 23), solarização p=0.1, normalização ImageNet ([A] App. C.1) | **Assimétrica, igual ao BYOL**: blur p=1.0/0.1, solarização p=0.0/0.2; resto igual; `RandomResizedCrop(224, BICUBIC)` com escala padrão (0.08, 1.0) (`augmentations.py:37-89`) | O texto do artigo traz `scale=(0.08, 0.1)`, que é erro de digitação. **O código é a referência.** |
| Blur | p, kernel 23 | `PIL GaussianBlur(sigma)`, σ~U(0.1, 2.0) (`augmentations.py:14-23`) | |
| Batch | 2048 ([A] §4.2) | `--batch-size 2048` (`:50`); README: 512 para 100 épocas | |
| Otimizador | LARS ([A] §4.2) | LARS, momentum 0.9, η=0.001; bias e BN (`p.ndim==1`) **fora** do weight decay e da adaptação LARS (`:109-115, 235-299`) | Um único lr para todos os parâmetros (diferente do Barlow Twins). |
| lr | `lr = batch/256 × base_lr`, base 0.2 (1000 ép.); 0.3 (100 ép.) ([A] §4.2, C.4) | `base_lr * batch_size / 256` (`:170`) | Na ablação de batch: base 0.8 (128), 0.5 (256), 0.4 (512), 0.3 (≥1024) ([A] C.4). |
| Warmup / decaimento | 10 épocas a partir de 0; cosseno até "0.002" ([A] §4.2) | warmup linear de 10 épocas por passo; cosseno até `base_lr*0.001` (`:167-181`) | 0.001×1.6 = 0.0016 ≈ 0.002. |
| Épocas | 1000 (principal); 100 nas ablações ([A] §5, §6) | padrão `--epochs 100` (`:48`) | 100 épocas dão conclusões consistentes com 1000 ([A] §6). |
| Weight decay | 1e-6 | `--wd 1e-6` (`:54`) | |
| Precisão | — | AMP fp16 + `GradScaler`, perda dentro do autocast (`:128-142`) | |
| BN multi-GPU | — | `SyncBatchNorm` (`:107`) | Estatísticas de BN e de var/cov sobre o batch global. |
| Avaliação linear | SGD lr 0.02, wd 1e-6, batch 256, 100 ép., cosseno; RRC 0.2–1.0 + flip; val: center crop 224 ([A] C.2) | `evaluate.py`: SGD momentum 0.9, `CosineAnnealingLR`, `RandomResizedCrop(224)` (escala padrão 0.08–1), `Resize(256)+CenterCrop(224)`; backbone em `eval()` no modo freeze; README `--lr-head 0.02` | Transferência VOC07: **SVM linear (LIBLINEAR), C por validação cruzada** ([A] C.3). |

### 3.1 O que o artigo diz sobre batch pequeno, dimensão do expander e datasets pequenos

- **Batch** ([A] Tab. 13, App. D.7; 100 épocas, lr base com grid por batch): 128 → 67,3; 256 → 67,9; 512 → 68,2; 1024 → 68,3; **2048 → 68,6**; 4096 → 67,8. A queda com batch pequeno é pequena (−1,2 p.p. em 128), **desde que o lr base suba** (0.8 em 128, C.4). O artigo não testa batch abaixo de 128.
- **Dimensão do expander** ([A] Tab. 12): 256 → 55,9; 512 → 59,2; 1024 → 62,4; 2048 → 65,1; 4096 → 67,3; **8192 → 68,6**; 16384 → 68,8. A dependência vem do mecanismo de decorrelação herdado do Barlow Twins ([A] D.6). A dimensão da **representação** também conta (Narrow ResNet, [A] D.1/Tab. 9).
- **BN no expander** ([A] Tab. 4 e Tab. 8): BN nas camadas escondidas dá +1,0 a 1,2 p.p.; padronizar o **embedding** piora (−0,2); projetar o embedding na esfera L2 piora 3,5 p.p.
- **Coeficientes** ([A] Tab. 7): Inv+Var sem Cov = 57,5; (1,1,1), (1,10,1) e (10,1,1) colapsam; (5,5,1) 68,1; (10,10,1) 68,2; **(25,25,1) 68,6**; (50,50,1) 68,3.
- **Datasets pequenos**: "excelentes resultados em MNIST, CIFAR-10 e CIFAR-100 com exatamente os mesmos valores" ([A] D.4). No ESC-50 (1600 amostras de treino, ResNet-18, ramos assimétricos) usaram **AdamW lr 0.0005 por 100 épocas** ([A] D.2). É o único precedente oficial de dataset pequeno e não usa LARS.
- **Estabilidade**: a diferença entre seeds é < 0,1 p.p. na avaliação linear ([A] §5.1).
- **VICRegL** ([A-L] §3.2): os autores sugerem que **o batch ótimo fica perto da dimensão da representação** (ResNet-50 2048 ↔ batch 2048; ConvNeXt 768/1024 ↔ batch 512, e o ConvNeXt-S foi treinado com batch 384). Isso é uma hipótese dos autores, não um teorema.

### 3.2 VICRegL (mapas locais): o que é relevante para nós

- Perda ([A-L] Eq. 4): `L = α·ℓ(z⊕, z'⊕) + (1−α)·{Ls(z,z') + Ls(z',z) + Ld(z,z') + Ld(z',z)}`.
  - `ℓ` é o VICReg global sobre a representação **pooled** passada pelo expander global (8192-8192-8192).
  - `Ls` casa cada posição do mapa com a posição mais próxima **na imagem original** (coordenadas absolutas do crop) e mantém os top-γ pares, com γ=20.
  - `Ld` casa cada posição com o vizinho mais próximo **em L2 no espaço de embedding**.
  - O critério local usa um **projetor local** 512-512-512, aplicado por posição, antes do casamento ([A-L] §3.1, §3.3).
- α: 0.9 (classificação quase intacta: 71,2 contra 71,5) e 0.75 (melhor segmentação linear: VOC 55,9 mIoU) ([A-L] Tab. 1).
- Implementação ([A-L] §3.3): ResNet-50 com LARS, batch 2048, lr 0.1, wd 1e-6, warmup 10, cosseno até 0.002. ConvNeXt com **AdamW**, batch 384, lr 1e-3, wd 1e-6, cosseno até 1e-5.
- Código ([C-L] `main_vicregl.py:40-82`): `--alpha 0.75`, `--num_matches 20 4`, `--l2_all_matches 1` (no casamento por L2 o código **mantém todos os pares**, não os top-γ), `--maps-mlp 512-512-512`, `--optimizer adamw`, `--base-lr 0.0005` (escalado por batch/256, `utils.py:44-57`), `--weight-decay 0.05` (o artigo diz 1e-6), `--maps-lr-ratio 0.1`, `--end-lr-ratio 0.001`. O VICReg local (`_vicreg_loss`, `:284-314`) calcula variância e covariância **sobre o batch, para cada par casado**, e divide var e cov por 2 entre ramos.
- Multi-crop **não funcionou com ResNet** ([A-L] §3.2). Para uma CNN rasa, use só 2 vistas.

## 4. Regras de implementação (checklist)

**Perda (siga `main_vicreg.py:194-220`)**
- [ ] `repr_loss = F.mse_loss(z_a, z_b)` (média sobre N·D), sem normalizar `z`.
- [ ] Centrar cada ramo pela média do batch antes de var/cov.
- [ ] `std = sqrt(var(dim=0, unbiased) + 1e-4)`; `std_loss = mean(relu(1-std_a))/2 + mean(relu(1-std_b))/2` (com o **/2** do código).
- [ ] `cov = Xᶜᵀ Xᶜ / (N−1)`; `cov_loss = off(cov_a)²·sum/D + off(cov_b)²·sum/D`.
- [ ] `loss = 25·repr + 25·std + 1·cov`; `ε=1e-4`; `γ=1`.
- [ ] `off_diagonal` idêntico ao oficial (`x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()`).
- [ ] Var/cov/MSE em **fp32**, mesmo com o forward em bf16.
- [ ] Use `N` = tamanho real do batch, com `drop_last=True` (já é o padrão do nosso loader, `run_search.py:455`).

**Arquitetura**
- [ ] Representação = saída que vai para o SVM (no nosso caso `head(encoder(x))`, `EmbeddingHead`, dim ≥ 1280).
- [ ] Expander `d→8192→8192→8192`: `Linear(bias=True)+BN1d+ReLU` ×2, depois `Linear(bias=False)`. **Sem** BN, L2 ou ReLU na saída.
- [ ] Dois ramos com **pesos compartilhados**, sem stop-gradient, sem preditor e sem EMA.
- [ ] Descartar o expander no fim (`extra.clear()`) e avaliar `h` (antes do expander).

**Dados**
- [ ] Exatamente 2 vistas, geradas por duas distribuições **diferentes** (`t` e `t'` do BYOL): blur 1.0/0.1 e solarização 0.0/0.2.
- [ ] RRC escala (0.08, 1.0), razão (3/4, 4/3), interpolação bicúbica, saída 224.
- [ ] Flip horizontal 0.5; ColorJitter(0.4, 0.4, 0.2, 0.1) com p=0.8; grayscale p=0.2; blur σ~U(0.1, 2.0).

**Otimização**
- [ ] LARS (momentum 0.9, η=0.001), excluindo `p.ndim==1` do wd e da adaptação; um único lr para todos.
- [ ] `lr = base_lr·B/256`; warmup linear de 0 por passo durante 10 épocas; cosseno até `0.001·lr`.
- [ ] wd 1e-6; sem gradient clipping (o oficial não usa).
- [ ] Ajustar o lr por passo (`adjust_learning_rate` antes de cada `step`).

**Avaliação**
- [ ] Encoder congelado em `eval()`; sonda linear sobre a representação pré-expander.

## 5. Divergências e o que falta no nosso código

Arquivo: `ga_search/modehb/ssl_trainers.py` (494 linhas, lido em 17/11/2026).

| # | Onde [N] | Hoje | VICReg oficial | Gravidade |
|---|---|---|---|---|
| 1 | `_SSL` (`:144`), `method_name` (`:148-160`) | só `simclr`, `lejepa`, `ae` | precisa de `vicreg` (sem `reg`; o VICReg já é o regularizador) | **bloqueante** |
| 2 | `V = n_views or {...}` (`:293-295`) | 2/4/1 | 2 vistas | baixa |
| 3 | `_make_projector` (`:222-224`) e `ssl_proj` (`:331-332`) | 2 camadas (128→64 ou 512→128) | 3 camadas 8192; última sem bias | **alta** ([A] Tab. 12: 256 → −12,7 p.p.) |
| 4 | `build_opt` (`:322-336`) | `AdamW` com **lr fixo** em 2 grupos (`lr_old=1e-4`, `lr_new=1e-3`, `search_v3.yaml:157-158`), sem warmup e sem cosseno | LARS, `0.2·B/256`, warmup de 10 épocas, cosseno até 0.001× | **alta** |
| 5 | `weight_decay=0.0` (default, `:266`); `run_search.py:712` **não repassa** `training.weight_decay: 1e-4` | wd efetivo = 0 | 1e-6, excluindo bias/BN | média (confirme se o wd=0 é intencional) |
| 6 | `AugConfig` (`:76-85`), `make_views` (`:88-138`) | crop (0.3, 1), vflip 0.5, brilho/contraste **aditivos** ±0.2 em LAB, jitter de canal 0.05, bilinear, padding por reflexão; **sem** jitter de cor p=0.8, grayscale, blur ou solarização; vistas **simétricas** | BYOL assimétrico (§3) | **alta** (Barlow Twins Fig. 3: tirar augmentations derruba muito, e o VICReg herda isso) |
| 7 | `batch_size: 64` FIXO (`search_v3.yaml:154`) | 64 | 2048 (ablação mínima 128) | **alta**, ver §6.2 (piso da covariância) |
| 8 | Escala de entrada | LAB[0,1] (`ift_lab_loader`) sem normalização | normalização ImageNet em RGB | adaptação obrigatória (os kernels SPiFiL foram aprendidos em LAB[0,1]) |
| 9 | Precisão (`amp_dtype`, `:281-283`) | bf16 opt-in, perdas em fp32 | fp16 + GradScaler, perda no autocast | aceitável (o nosso é mais conservador) |
| 10 | `_scale_conv_groups` (`:185-203`), `ElementProtect`, WiSE (`run_search.py:708-723`) | proteção dos kernels SPiFiL e mistura WiSE | não existem | adaptação (princípio Hybrid SPiFiL) |
| 11 | Logs | `pos_sim`/`neg_sim` sobre `hs` | o oficial só registra a perda | adicione `std` médio de `z` e de `h`, `cov_loss`, rank efetivo |
| 12 | Avaliação | SVM linear C=100 sobre `h` congelado | SGD linear 100 ép. (ImageNet); **SVM linear no VOC07** | compatível em espírito (sonda linear congelada, pré-expander). **Nunca mudar o SVM.** |

## 6. Adaptação ao nosso caso

### 6.1 Onde fica cada peça

- **Encoder** = backbone SPiFiL (2 a 4 blocos conv, `c_out` de 16 a 1024 canais).
- **Representação** = `h = EmbeddingHead(encoder(x))`, com dim `embed_dim ≥ 1280` (obrigatório pela MISSION). **[H]** Com `pooling=gap/gem` e cabeça linear (`act=none`), o rank estrutural de `h` é ≤ `n_blocks·c_out` (ver `structural_rank` em `run_search.py:732`). A dimensão *efetiva* da representação costuma ser bem menor que 1280.
- **Expander** = MLP 8192-8192-8192 (fiel), entrada `embed_dim`. Com `embed_dim=1280` são ≈144,8 M parâmetros (≈0,58 GB em fp32; com grad e momento do LARS, ≈1,7 GB; com AdamW, ≈2,3 GB). Cabe em 48 GB e **não conta** no orçamento Bk (MISSION: projetores são registrados à parte).
- O gene `projector` **não se aplica**: o expander é obrigatório no VICReg. Não empilhe `ssl_proj` + expander.

### 6.2 Batch 64: piso matemático do termo de covariância (ponto crítico)

**[H] Derivação nossa, verificada numericamente.** A covariância empírica de N amostras tem rank ≤ N−1. Se cada dimensão tem std ≈ 1 (o que o hinge exige), então `‖C‖²_F ≥ tr(C)²/(N−1)`. Daí, para cada ramo:

`c(Z) ≥ D/(N−1) − 1`

| N | D | piso de `c(Z)` por ramo | medido com `z` gaussiano aleatório |
|---|---|---|---|
| 2048 (oficial) | 8192 | 3,0 | 4,0 |
| 128 (menor batch da Tab. 13) | 8192 | 63,5 | 64,5 |
| **64 (nosso)** | **8192** | **129,0** | **130,0** |
| 64 | 1280 | 19,3 | 20,3 |
| 64 | 256 | 3,1 | 4,1 |

Consequência: com N=64 e D=8192, o termo de covariância tem um piso ~40× maior que no regime oficial. Ele **não** pode ir a zero e domina o equilíbrio λ=μ=25, ν=1. O artigo funcionou com piso de 63,5 (batch 128, 67,3%), mas não testou 129.

Opções, a decidir pelo usuário (lr e batch "nunca ajustados" são regras do repositório):
- (a) **Fiel**: N=64, D=8192, coeficientes oficiais. Registre `cov_loss` e o piso `D/(N−1)−1` no W&B.
- (b) **Adaptação de batch**: `ssl_batch_size` separado (256 a 1024; cabe em 48 GB para larguras ≤ 256 a 224 px, mas meça). O VICRegL sugere batch ≈ dimensão da representação, e a nossa dimensão efetiva é ≈ `n_blocks·c_out`.
- (c) **Adaptação de D**: manter `D/(N−1)` ≈ 4 (como o oficial) leva a D≈256 com N=64. Mas a Tab. 12 mostra perda forte com D pequeno no ImageNet. Trate como hipótese.
- (d) **Adaptação de ν**: escalar ν por `(N−1)/(N_oficial−1)` ≈ 0,031. **[H]** Não é oficial; a App. C.4 usou ν=0,01 só quando o VICReg foi acoplado a BYOL/SimSiam.

Cada opção é uma **variante registrada com nome distinto** (`vicreg`, `vicreg_bs512`, `vicreg_d2048`...), nunca uma troca silenciosa.

### 6.3 Otimizador e agenda

- **Fiel (`vicreg`)**: LARS com `lr_peak = 0.2·64/256 = 0.05`. A C.4 mostra que batch menor pede base maior (0.8 em 128), mas não há valor para 64. Não extrapole sem registrar.
- **Precedente oficial de dataset pequeno (`vicreg_adamw`)**: AdamW lr 5e-4, 100 épocas ([A] D.2; o VICRegL ConvNeXt também usa AdamW). Combina com o nosso AdamW, mas exige warmup e cosseno, que hoje não existem.
- **Unidade de época**: a nossa "época" tem ≤ 2048 imagens = 32 passos (`max_images_per_epoch`, `search_v3.yaml:155`). Com fidelidade AE:SSL = 1:2, as épocas de SSL vão de ~3 a ~133, ou seja, ≤ ~4,3 k passos. O oficial faz 625 passos/época × 100–1000 épocas. **[H]** Use `warmup_steps = min(10 épocas, 10% dos passos totais)`: 10/100 é a razão do protocolo de 100 épocas.
- **Retomada DEHB** (`step_offset`): defina a agenda sobre os passos totais da **maior fidelidade** e indexe pelo passo global. Assim a promoção continua a mesma curva, sem reiniciar o warmup.
- **Grupos old/new e proteção SPiFiL (adaptação)**: `lr_g(t) = lr(t)·s_g`, com `s_new=1`, `s_old=lr_old/lr_new=0.1` e `s_spifil=conv_lr_scale` (1 / 0.01 / 0 = freeze). **[H]** No LARS, o passo relativo por camada é ~`lr·η`, independente da escala do gradiente. Portanto `s_g` controla a velocidade relativa de mudança dos kernels SPiFiL, que é o que o princípio "backprop soma ao SPiFiL, nunca desfaz" pede.
- wd 1e-6 só em `p.ndim>1`.

### 6.4 Augmentations em LAB[0,1] (o BYOL é definido em RGB)

| BYOL [C] | Implementação fiel | Aproximação em LAB (adaptação) |
|---|---|---|
| RRC(224, scale 0.08–1, bicúbico) | `scale=(0.08,1)` no `AugConfig` | A fonte já tem 224 px: crop de 8% ≈ 63 px ampliado 3,5×. Parasitas são pequenos, então meça a cobertura do objeto (`scripts/measure_object_coverage.py`, `src/transforms/object_crop.py`). Variante: (0.3, 1) atual. |
| HFlip 0.5 | igual | vflip 0.5 adicional (microscopia sem orientação), registrado como adaptação |
| ColorJitter(0.4,0.4,0.2,0.1) p=0.8, ordem aleatória | LAB→RGB (`skimage.color.lab2rgb`, ou a inversa exata do `LABNorm2` do pyift), `torchvision.transforms.v2.ColorJitter`, RGB→LAB | brilho: L·U(0.6, 1.4); contraste: em torno da média de L; saturação: (a,b) escalados em torno do neutro; matiz: rotação de (a,b) por ±0,1·2π |
| Grayscale p=0.2 | via RGB | (a,b) ← neutro. **Meça** o valor neutro do `LABNorm2` (não assuma 0.5). |
| GaussianBlur p=1.0 / 0.1, σ~U(0.1, 2.0) | `torchvision.transforms.v2.functional.gaussian_blur` por amostra, kernel 23 | igual (o blur independe do espaço de cor) |
| Solarization p=0.0 / 0.2 (limiar 128) | via RGB | L ← 1−L onde L ≥ 0,5 (aproximação) |
| Normalize ImageNet | **não** aplicar | manter LAB[0,1], porque o SPiFiL e o SVM foram calibrados assim (gotcha `--no-imagenet-norm`). A transformação de avaliação deve ser idêntica à do treino. |

`kornia` não está instalado no env `scalable_FLIM`; `skimage 0.26` e `torchvision 0.22.1` estão.

### 6.5 Pseudo-código no `train_global_ssl` (conforme o oficial)

```python
# ---- perdas (fp32; portado de main_vicreg.py:194-220, single-GPU => FullGatherLayer = identidade)
def off_diagonal(x):
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss(z_a, z_b, sim=25.0, std=25.0, cov=1.0, eps=1e-4):
    z_a, z_b = z_a.float(), z_b.float()
    N, D = z_a.shape
    repr_loss = F.mse_loss(z_a, z_b)
    x, y = z_a - z_a.mean(0), z_b - z_b.mean(0)
    std_x, std_y = torch.sqrt(x.var(0) + eps), torch.sqrt(y.var(0) + eps)
    std_loss = F.relu(1 - std_x).mean() / 2 + F.relu(1 - std_y).mean() / 2      # /2 = código oficial
    cov_x, cov_y = (x.T @ x) / (N - 1), (y.T @ y) / (N - 1)
    cov_loss = off_diagonal(cov_x).pow(2).sum() / D + off_diagonal(cov_y).pow(2).sum() / D
    total = sim * repr_loss + std * std_loss + cov * cov_loss
    return total, {"ssl/loss_vicreg_inv": repr_loss, "ssl/loss_vicreg_var": std_loss,
                   "ssl/loss_vicreg_cov": cov_loss, "ssl/vicreg_cov_floor": D / (N - 1) - 1}

def make_expander(d, spec="8192-8192-8192", last_bias=False):            # Projector oficial
    f = [d] + [int(s) for s in spec.split("-")]
    layers = []
    for i in range(len(f) - 2):
        layers += [nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)]
    layers.append(nn.Linear(f[-2], f[-1], bias=last_bias))
    return nn.Sequential(*layers)

# ---- integração
_SSL = ("simclr", "lejepa", "ae", "vicreg", "barlow")
# method_name: "vicreg" exige reg == "none" (o VICReg já é var+cov); outro reg => variante "vicreg+<reg>"
V = 2 if ssl in ("vicreg", "barlow") else ...
# build_opt:
if ssl == "vicreg":
    extra["expander"] = make_expander(d, cfg.get("expander", "8192-8192-8192")).to(device)
    # gene `projector` ignorado (expander obrigatório)
groups = split_param_groups(...); groups = _scale_conv_groups(groups, encoder, conv_lr_scale)
if optim == "lars":                                       # fiel
    groups = [{**g, "lr_scale": g["lr"] / lr_new} for g in groups]    # s_old=0.1, s_spifil=conv_lr_scale
    opt = LARS(groups, lr=0, weight_decay=1e-6, momentum=0.9, eta=1e-3,
               weight_decay_filter=lambda p: p.ndim == 1, lars_adaptation_filter=lambda p: p.ndim == 1)
# laço:
views = make_views_byol(x, generator=gen)                  # (t, t') assimétricas, ver §6.4
lr = cosine_warmup(global_step, total_steps, warmup_steps, base_lr * B / 256, end_ratio=1e-3)
for g in opt.param_groups: g["lr"] = lr * g.get("lr_scale", 1.0)
with amp():
    h = head(encoder(torch.cat(views, 0)))                 # [2B, embed_dim]  <- representação (SVM)
    z = extra["expander"](h)                               # [2B, 8192]       <- embedding (perda)
total, parts = vicreg_loss(*z.float().split(B, 0))
# ... backward / skip NaN / opt.step() iguais ao atual; logar std(h), std(z), rank efetivo de h
# fim: extra.clear()  -> avaliar h (pré-expander) com o SVM C=100 original
```

O LARS deve ser copiado literalmente de `main_vicreg.py:245-299`, sem reescrever.

### 6.6 O que é fiel e o que é adaptação

| Fiel (igual a [A]/[C]) | Adaptação (registrar como variante) |
|---|---|
| Fórmula da perda do código (com /2 na var), λ=μ=25, ν=1, ε=1e-4, γ=1 | batch 64 (oficial 2048) e o piso de covariância decorrente |
| Expander 3×8192, BN+ReLU nas escondidas, última sem bias | entrada do expander = `EmbeddingHead` (≥1280), não o GAP do ResNet |
| 2 vistas, pesos compartilhados, sem SG/preditor/EMA | LAB[0,1] sem normalização ImageNet; jitter de cor via LAB↔RGB ou aproximado |
| Avaliação pré-expander, encoder congelado, sonda linear | sonda = SVM C=100 do repositório (não SGD 100 ép.) |
| LARS (η=1e-3, m=0.9, exclusão ndim==1), lr=base·B/256, warmup e cosseno até 0.001× | proteção SPiFiL (`conv_lr_scale`, `ElementProtect`), escalas old/new, WiSE |
| wd 1e-6 | épocas = fidelidades DEHB (≤ ~4,3 k passos); warmup em fração dos passos |
| Augmentations BYOL assimétricas (com o código como referência) | vflip; escala de crop sobre uma fonte de 224 px; solarização aproximada em L |
| | bf16 no forward, perdas em fp32 (o oficial usa fp16 + GradScaler) |
| | variante `vicreg_adamw` (AdamW 5e-4, D.2) como precedente oficial de dataset pequeno |

## 7. Testes de conformidade

Sugestão de local: `tests/modehb/test_ssl_vicreg.py`. Rodam em CPU e sem GPU.

1. **Paridade com o oficial**: copie `VICReg.forward` de `main_vicreg.py:194-220` com `FullGatherLayer` = identidade e `args.batch_size = N`. Para `z` aleatório (N=64, D=512, seed fixa), `vicreg_loss` deve bater com a perda oficial com `atol=1e-6` em fp32.
2. **Diferença documentada em relação ao Alg. 1**: sem o /2 da variância, `std_loss` deve dobrar. O teste fixa que seguimos o código.
3. **`off_diagonal`**: deve ser igual a `C[~torch.eye(D, dtype=bool)]`.
4. **Casos-limite**:
   - `z` constante → `std_loss = 1 − sqrt(1e-4) = 0.99` (com /2 nos dois ramos) e `cov_loss = 0`;
   - `z_a = z_b` → `repr_loss = 0`;
   - colunas gaussianas com std 1 → `std_loss ≈ 0`.
5. **Piso da covariância**: para `z` gaussiano com N=64 e D=8192, `cov_loss/2 ≥ D/(N−1) − 1` (≈129), conforme §6.2.
6. **Expander**: as camadas devem ser exatamente `[Linear(d,8192,bias), BN, ReLU, Linear(8192,8192,bias), BN, ReLU, Linear(8192,8192,bias=False)]`.
7. **LARS**: um passo em um modelo de brinquedo deve ser igual ao do `LARS` oficial copiado. Parâmetros 1-D não recebem wd nem adaptação.
8. **Agenda**: `lr(0)=0`; `lr(warmup)=base·B/256`; `lr(fim)=0.001·base·B/256`; monotônica após o warmup. Com `step_offset` deve continuar sem reiniciar.
9. **Avaliação pré-expander**: após `train_global_ssl`, `extra` deve estar vazio. A feature do SVM tem dim `embed_dim` e é idêntica a `head(encoder(x))` em `eval()`.
10. **Augmentations**: com seed fixa, contar frequências em 10 k amostras (jitter ≈0.8, grayscale ≈0.2, blur 1.0/0.1, solarização 0.0/0.2); a escala do crop deve cair em [0.08, 1].
11. **Smoke de ponta a ponta** (R9 da FILA): 2 épocas num dataset sintético pequeno.
    - sem NaN;
    - `std(z)` por dimensão cresce em direção a 1;
    - perda bf16 contra fp32 com diferença relativa < 1e-2;
    - `tests/modehb/test_ssl_trainers.py` continua passando.
12. **Não regressão do SVM**: o κ é calculado com `core.svm.fit_svm` C=100, sem nenhuma alteração.
