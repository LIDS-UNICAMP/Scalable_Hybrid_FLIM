# Barlow Twins: regras de implementação fiéis ao artigo e ao código oficial

> Escopo: como implementar Barlow Twins dentro de `ga_search/modehb/ssl_trainers.py::train_global_ssl`,
> o que é **fiel** ao artigo/código e o que é **adaptação** ao nosso caso (CNN rasa com SPiFiL, 224 px,
> datasets de 1k a 10k imagens, GPU de 48 GB, batch 64 fixo).
> Fontes lidas: PDF arXiv:2103.03230v3 (ICML 2021) e `facebookresearch/barlowtwins@main`
> (`main.py`, `evaluate.py`, `README.md`), lidos em 17/11/2026.
> Convenção: **[A]** = artigo, **[C]** = código oficial, **[N]** = nosso código, **[H]** = hipótese/derivação nossa.
> Ver também `vicreg.md` (mesma pasta): o VICReg herda o termo de decorrelação do Barlow Twins.

---

## 1. Referência

- Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S. *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. ICML 2021 (PMLR 139). arXiv:2103.03230.
- Código oficial: https://github.com/facebookresearch/barlowtwins (`main.py`, `evaluate.py`).

## 2. Resumo

- Duas vistas distorcidas da mesma imagem passam pela **mesma rede**: encoder → **representação** (usada a jusante) → projetor → **embedding** ([A] §2.1, §2.2).
- A perda leva a **matriz de correlação cruzada** C (D×D) entre os embeddings dos dois ramos, normalizados **ao longo do batch**, para a **identidade** ([A] Eq. 1–2):
  - a diagonal → 1 dá **invariância**;
  - fora da diagonal → 0 dá **redução de redundância**, ponderada por λ.
- Não precisa de batch grande, preditor, stop-gradient nem EMA ([A] §1, §5.1). Ganha com **embeddings de dimensão muito alta** ([A] Fig. 4).
- Pontos fracos que importam para nós:
  - é **sensível a tirar augmentations**, como o SimCLR e ao contrário do BYOL ([A] Fig. 3);
  - a normalização por batch é essencial: sem BN no MLP e sem normalização, cai para 53,4% ([A] Tab. 5);
  - os embeddings podem encolher antes da padronização. O código evita isso com o ε do BN (o VICReg discute em [A-VICReg] App. B).

## 3. Receita oficial (ResNet-50, ImageNet)

| Item | Artigo [A] | Código oficial [C] | Observação |
|---|---|---|---|
| Perda | `L = Σᵢ(1−Cᵢᵢ)² + λ Σᵢ Σ_{j≠i} Cᵢⱼ²` (Eq. 1) | `on_diag = (diag(c)−1)².sum(); off_diag = off(c)².sum(); loss = on_diag + lambd·off_diag` (`main.py:218-220`) | **Soma**, não média: a escala da perda é ~D. |
| C | `Cᵢⱼ = Σ_b zᴬ_{b,i} zᴮ_{b,j} / (√Σ_b(zᴬ_{b,i})² √Σ_b(zᴮ_{b,j})²)`, com z centrado (Eq. 2) | `c = bn(z1).T @ bn(z2)`, depois `c.div_(batch_size)` e `all_reduce` (`main.py:212-216`) | `bn = nn.BatchNorm1d(D, affine=False)` (`:205`): estatística **viesada** (÷N) e `eps=1e-5`. |
| Pseudocódigo | `z_norm = (z − mean)/z.std(0)` (std **não-viesada**), `c = z_aᵀ z_b / N` ([A] Alg. 1) | BN (std viesada) | Diferença de fator `(N−1)/N` em C. Com N=64 isso dá 1,6%. **Siga o código.** |
| λ | 5·10⁻³ (melhor de uma busca, [A] §2.2) | `--lambd 0.0051` (`:39-40`) | Pouco sensível entre 0.002 e 0.02 ([A] Fig. 5: 70,8 a 71,4). λ **fixo** em todas as dimensões do projetor ([A] legenda da Fig. 4). |
| Normalização por batch | BN de cada ramo separadamente, sem parâmetros afins | `self.bn(z1)` e `self.bn(z2)` separados, em `train()` (estatística do batch) | Com várias GPUs: C somado (`all_reduce`) e dividido pelo batch global. |
| Encoder | ResNet-50 sem a FC, saída 2048 | `torchvision.models.resnet50(zero_init_residual=True)`, `fc = Identity` (`:191-192`) | |
| Projetor | 3 lineares de 8192; BN+ReLU depois das duas primeiras ([A] §2.2) | `Linear(bias=False)→BN→ReLU` ×2 + `Linear(bias=False)` (`:195-202`) | **Todas** as lineares sem bias (no VICReg as escondidas têm bias). |
| Avaliação | Representações (saída do ResNet, 2048); o projetor só alimenta a perda ([A] §2.2) | salva `backbone.state_dict()` (`:151-152`) | Avaliação **antes** do projetor. As ablações usam features res5 de 2048 ([A] §4). |
| Augmentations | Iguais às do BYOL: crop, resize 224, flip, jitter de cor, grayscale, blur e solarização. Crop e resize sempre; as outras 5 com probabilidade; blur e solarização com probabilidades **diferentes** por vista ([A] §2.2) | `Transform` (`:290-326`): `RandomResizedCrop(224, BICUBIC)` (escala 0.08–1), HFlip 0.5, `ColorJitter(0.4,0.4,0.2,0.1)` p=0.8, Grayscale 0.2, **blur p=1.0 / 0.1** (σ~U(0.1, 2.0), PIL), **solarização p=0.0 / 0.2**, normalização ImageNet | Idênticas às do código do VICReg. |
| Batch | 2048 ([A] §2.2) | `--batch-size 2048` (`:31`) | "Funciona bem com batches de até 256" ([A] §2.2, Fig. 2). |
| Otimizador | LARS ([A] §2.2) | LARS, momentum 0.9, η=0.001, `weight_decay_filter=True` e `lars_adaptation_filter=True` (`:99-101, 224-263`) | Bias e BN (`p.ndim==1`) fora do wd e da adaptação LARS. |
| lr | 0.2 (pesos) e **0.0048 (bias e BN)**, ambos ×batch/256 ([A] §2.2) | `--learning-rate-weights 0.2 --learning-rate-biases 0.0048` (`:33-36`); dois grupos por `param.ndim` (`:91-97`) | Bias e BN têm lr ~40× menor porque escapam da normalização do LARS e a perda tem escala ~D. |
| Warmup / decaimento | 10 épocas; depois cosseno que reduz o lr por **1000×** ([A] §2.2) | `adjust_learning_rate` (`:155-168`): warmup linear por passo, cosseno até `0.001·base` | |
| Épocas | 1000 (principal); 300 nas ablações ([A] §2.2, §4) | `--epochs 1000` (`:29`) | |
| Weight decay | **1.5·10⁻⁶** ([A] §2.2) | `--weight-decay 1e-6` (`:37`) | **Divergência artigo × código.** Siga o código, mas registre. |
| Precisão | — | AMP fp16 + `GradScaler`, perda no autocast (`:122-134`) | |
| BN multi-GPU | — | `SyncBatchNorm` (`:89`) | |
| Avaliação linear | 100 ép., lr 0.3, cosseno, SGD com momentum, wd 1e-6, batch 256; treino com RRC 224 + flip; teste com resize 256 + center crop 224 ([A] App. B.1) | `evaluate.py`: `--lr-classifier 0.3`, `--weight-decay 1e-6`, `SGD(momentum=0.9)`, `CosineAnnealingLR`, `RandomResizedCrop(224)`; README: `--lr-classifier 0.3` | Transferência VOC07: **SVM linear com C por validação cruzada** ([A] App. C.1). |
| Hardware | 32 V100, ~124 h para 1000 épocas ([A] §2.2) | | |

### 3.1 O que o artigo diz sobre batch pequeno, dimensão do projetor e datasets pequenos

- **Batch** ([A] §4, Fig. 2; lr do LARS com grid por batch, 300 épocas): faixa testada de 128 a 4096. Melhor **71,7% com batch 1024**; "desempenho quase inalterado para batch de até 256". O SimCLR perde ~4 p.p. em 256. O valor exato em 128 só está no gráfico. O artigo não testa batch < 128.
- **Dimensão do projetor** ([A] Fig. 4, de 16 a 16384): o desempenho **continua subindo** em toda a faixa testada, apesar do gargalo de 2048 da representação. BYOL e SimCLR saturam cedo. Mais camadas também ajudam, com saturação em 3 ([A] §4). Os autores sugerem ir além de 16 k como trabalho futuro (limitado por memória, [A] §5.2).
- **Ablação da perda** ([A] Tab. 5, 300 ép.):
  - baseline 71,4;
  - só a diagonal 57,3;
  - só fora da diagonal 0,1 (colapso);
  - normalização na dimensão de feature 69,8;
  - sem BN no MLP 71,2;
  - **sem BN no MLP e sem normalização 53,4**;
  - cross-entropy com temperatura 63,3.
- **Augmentations** ([A] Fig. 3): sem grayscale + jitter, "crop + blur" e "só crop" causam quedas grandes (até ~−25 p.p. no gráfico).
- **Assimetria** ([A] Tab. 6): stop-gradient → 70,5; preditor → 70,2; os dois → 61,3. **Não** adicione.
- **Datasets pequenos**: o artigo **não** traz experimentos de pré-treino em datasets pequenos. Só transfere para Places-205, VOC07 e iNat18 com sonda linear ([A] §3.2, Tab. 3). Não há indicação oficial de hiperparâmetros para 1k a 10k imagens.
- **Variante citada pelos autores** ([A] §5.2, App. A): calcular o termo de redundância sobre a **autocorrelação de um único ramo** (como faz o VICReg) "leva a desempenho semelhante (não mostrado)". Não é a receita oficial.
- **Interpretação** ([A] App. A): a perda é um proxy de Information Bottleneck sob parametrização gaussiana. A entropia pode ser estimada "com muito menos amostras" que a do InfoNCE, que é o argumento a favor de batch pequeno.

## 4. Regras de implementação (checklist)

**Perda (siga `main.py:187-221`)**
- [ ] `bn = nn.BatchNorm1d(D, affine=False)` em `train()`, aplicado **separadamente** a `z1` e `z2` (estatística do próprio batch, viesada, eps=1e-5).
- [ ] `c = bn(z1).T @ bn(z2) / N` (D×D), com N = batch real; `drop_last=True`.
- [ ] `on = (diag(c) − 1)².sum()`; `off = off_diagonal(c)².sum()`; `loss = on + 0.0051·off`.
- [ ] **Soma** (não média) sobre D: a escala da perda é ~D, e por isso os lrs do LARS são os que são.
- [ ] `off_diagonal` idêntico ao oficial.
- [ ] C e a perda em **fp32** (a matmul D×D com D=8192 em bf16 perde precisão nos Cᵢⱼ pequenos).
- [ ] O `bn` da perda não tem parâmetros e **não** pode ficar em `eval()` durante o SSL (usaria `running_mean`/`var`).

**Arquitetura**
- [ ] Representação = `h = head(encoder(x))` (feature do SVM).
- [ ] Projetor `d→8192→8192→8192`: `Linear(bias=False)+BN1d+ReLU` ×2, depois `Linear(bias=False)`.
- [ ] Siamês com pesos compartilhados; **sem** preditor, stop-gradient ou EMA ([A] Tab. 6).
- [ ] Descartar o projetor e o `bn` no fim; avaliar `h` (antes do projetor).

**Dados**
- [ ] 2 vistas: `t` (blur 1.0, solarização 0.0) e `t'` (blur 0.1, solarização 0.2); o resto igual (§3).

**Otimização**
- [ ] LARS (m=0.9, η=0.001), dois grupos: pesos (`ndim>1`) com `0.2·lr(t)` e bias/BN (`ndim==1`) com `0.0048·lr(t)`. Exclusão `ndim==1` de wd e da adaptação.
- [ ] `lr(t)` = `B/256` × (warmup linear de 10 épocas → cosseno até 0.001×).
- [ ] wd 1e-6 (código; o artigo diz 1.5e-6). Registre qual foi usado.

**Avaliação**
- [ ] Sonda linear com encoder congelado (`eval()`) sobre `h`.

## 5. Divergências e o que falta no nosso código

Arquivo: `ga_search/modehb/ssl_trainers.py` (lido em 17/11/2026).

| # | Onde [N] | Hoje | Barlow Twins oficial | Gravidade |
|---|---|---|---|---|
| 1 | `_SSL` (`:144`), `method_name` (`:148-160`) | só `simclr`, `lejepa`, `ae` | precisa de `barlow` (sem `reg`) | **bloqueante** |
| 2 | `V` (`:293-295`) | 2/4/1 | 2 vistas | baixa |
| 3 | `_make_projector` (`:222-224`) e `ssl_proj` (`:331-332`) | 2 camadas (128→64 ou 512→128), com bias | 3×8192, sem bias | **alta** (Fig. 4: o BT depende da dimensão do projetor mais que os outros métodos) |
| 4 | Normalização da perda | não existe BN de embedding | `BatchNorm1d(D, affine=False)` por ramo | **bloqueante** para a fidelidade (Tab. 5) |
| 5 | `build_opt` (`:322-336`) | `AdamW` com **lr fixo** em 2 grupos (old 1e-4 / new 1e-3), sem warmup e sem cosseno | LARS, pesos 0.2 e bias/BN 0.0048 (×B/256), warmup de 10 ép., cosseno até 0.001× | **alta** |
| 6 | `weight_decay=0.0` (`:266`); `run_search.py:712` **não repassa** `training.weight_decay: 1e-4` | wd efetivo = 0 | 1e-6 (código) / 1.5e-6 (artigo), sem bias/BN | média (confirme se é intencional) |
| 7 | `AugConfig`/`make_views` (`:76-138`) | crop (0.3, 1), vflip, brilho/contraste aditivos em LAB, jitter de canal 0.05; **sem** jitter de cor p=0.8, grayscale, blur ou solarização; vistas simétricas | BYOL assimétrico | **alta** ([A] Fig. 3: o BT **não é robusto** a tirar augmentations) |
| 8 | `batch_size: 64` FIXO (`search_v3.yaml:154`) | 64 | 2048 (testado até 128) | **alta**, ver §6.2 |
| 9 | Entrada | LAB[0,1], sem normalização ImageNet | RGB normalizado ImageNet | adaptação obrigatória |
| 10 | Precisão | bf16 opt-in, perdas em fp32 | fp16 + GradScaler, perda no autocast | aceitável |
| 11 | Proteção SPiFiL / WiSE | `_scale_conv_groups`, `ElementProtect`, mistura WiSE | não existem | adaptação (Hybrid SPiFiL) |
| 12 | Avaliação | SVM C=100 sobre `h` congelado | SGD linear (ImageNet); **SVM linear no VOC07** | compatível em espírito; **nunca mudar o SVM** |

## 6. Adaptação ao nosso caso

### 6.1 Onde fica cada peça

- **Encoder** = backbone SPiFiL (2 a 4 blocos, `c_out` de 16 a 1024).
- **Representação** = `EmbeddingHead` com dim ≥ 1280 (obrigatório). **[H]** Com cabeça linear e GAP/GeM, o rank estrutural é ≤ `n_blocks·c_out`. A dimensão efetiva fica abaixo de 1280, como o "gargalo de 2048" do ResNet no artigo.
- **Projetor** = 8192-8192-8192, sem bias, com entrada `embed_dim`. Com 1280 são ≈144,7 M parâmetros (≈0,58 GB fp32; ≈1,7 GB com grad e momento do LARS). Cabe em 48 GB. Fica fora do orçamento Bk (MISSION).
- O gene `projector` **não se aplica** (o projetor é obrigatório). Não empilhe `ssl_proj`.

### 6.2 Batch 64: piso do termo de redundância (ponto crítico)

**[H] Derivação nossa, verificada numericamente.** Depois de centrar, `C = (1/N)·Âᵀ B̂` tem rank ≤ N−1. Então `‖C‖²_F ≥ tr(C)²/(N−1)`. Com a diagonal ≈ 1 (invariância satisfeita):

`Σ_{i≠j} Cᵢⱼ² ≥ D²/(N−1) − D`

| N | D | piso de `λ·off` (λ=0.0051) | medido (vistas idênticas, gaussianas) | escala máxima do termo da diagonal (=D) |
|---|---|---|---|---|
| 2048 (oficial) | 8192 | ≈125 | 167 | 8192 |
| 128 (menor batch da Fig. 2) | 8192 | ≈2653 | 2694 | 8192 |
| **64 (nosso)** | **8192** | **≈5390** | **5430** | **8192** |
| 64 | 1280 | ≈126 | 133 | 1280 |

Consequência: com N=64 e D=8192, o piso do termo de redundância fica em ~66% da escala do termo de invariância, contra ~1,5% no oficial. Na prática, o trade-off λ fica ~43× maior que o oficial, muito fora da faixa testada (0.002 a 0.02, Fig. 5). O otimizador pode "pagar" redundância sacrificando a diagonal, ou seja, a invariância.

Opções, a decidir pelo usuário (batch e lr não são ajustados por regra do repositório). Cada uma é uma variante com nome próprio:
- (a) **Fiel** (`barlow`): N=64, D=8192, λ=0.0051. Registre `on`, `off`, `λ·off` e o piso no W&B.
- (b) **Batch maior** (`barlow_bs512` etc.): o BT é "quase inalterado até 256" (Fig. 2); 256 a 1024 cabe em 48 GB para larguras moderadas a 224 px (meça).
- (c) **λ escalado** (`barlow_lam_scaled`): manter `λ·D/(N−1)` igual ao oficial dá `λ ≈ 0.0051·63/2047 ≈ 1.6e-4`. **[H]** Não é oficial; o artigo mantém λ fixo em D (Fig. 4), mas só testou N ≥ 128.
- (d) **D menor**: com D=1280 e N=64 o piso relativo (~10%) fica mais perto do oficial. A Fig. 4 mostra perda com D pequeno no ImageNet. Hipótese a testar, não uma recomendação.

### 6.3 Otimizador e agenda

- **Fiel (`barlow`)**: LARS com `lr(t)·0.2` para pesos e `lr(t)·0.0048` para bias/BN, onde `lr_pico = 64/256 = 0.25`. Pico efetivo dos pesos = 0.05; bias/BN = 0.0012.
- O artigo não tem precedente de AdamW nem de dataset pequeno. Uma variante AdamW (como a nossa atual) é **adaptação**. Com AdamW a escala ~D da perda deixa de importar, mas a separação de lr de bias/BN, pensada para o LARS, também perde sentido. Registre se for usar.
- Unidade de época: ≤ 32 passos (`max_images_per_epoch: 2048`). Épocas de SSL ~3 a ~133 por fidelidade. **[H]** Warmup = `min(10 épocas, 10% dos passos)`. Cosseno até 0.001× sobre os passos da **maior fidelidade**, indexado pelo passo global (`step_offset`), para que a promoção DEHB continue a curva.
- **Proteção SPiFiL**: `lr_g(t) = lr(t)·{0.2 | 0.0048}·s_g`, com `s_new=1`, `s_old=0.1` e `s_spifil=conv_lr_scale`. **[H]** No LARS, `s_g` controla a taxa relativa de mudança dos kernels, compatível com "backprop soma ao SPiFiL".
- wd 1e-6 só em `ndim>1`.

### 6.4 Augmentations em LAB[0,1]

Mesmo mapeamento de `vicreg.md` §6.4 (as transformações do código são idênticas):
- RRC 0.08–1 sobre uma fonte de 224 px (meça a cobertura do parasita);
- flip horizontal 0.5, mais vflip como adaptação;
- ColorJitter(0.4, 0.4, 0.2, 0.1) p=0.8 via LAB↔RGB ou aproximado em L/(a,b);
- grayscale 0.2 → (a,b) neutros (medir o neutro do `LABNorm2`);
- blur 1.0/0.1 com σ~U(0.1, 2.0);
- solarização 0.0/0.2;
- **sem** normalização ImageNet (o SPiFiL e o SVM estão em LAB[0,1]).

O BT é o método que mais sofre quando as augmentations ficam fracas ([A] Fig. 3). Usar o `AugConfig` atual seria uma divergência grave, não um detalhe.

### 6.5 Pseudo-código no `train_global_ssl` (conforme o oficial)

```python
def off_diagonal(x):                                    # main.py:180-184
    n, m = x.shape; assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowLoss(nn.Module):                            # main.py:204-221 (single-GPU)
    def __init__(self, D, lambd=0.0051):
        super().__init__()
        self.bn = nn.BatchNorm1d(D, affine=False)       # eps=1e-5, var viesada, sem parâmetros
        self.lambd = lambd
    def forward(self, z1, z2):
        z1, z2 = z1.float(), z2.float()
        N = z1.shape[0]
        c = self.bn(z1).T @ self.bn(z2) / N             # BN separada por ramo; D x D em fp32
        on = (torch.diagonal(c) - 1).pow(2).sum()
        off = off_diagonal(c).pow(2).sum()
        D = c.shape[0]
        return on + self.lambd * off, {"ssl/loss_bt_on": on, "ssl/loss_bt_off": off,
                                       "ssl/bt_off_floor": D * D / (N - 1) - D}

def make_bt_projector(d, spec="8192-8192-8192"):        # main.py:195-202 (TODAS sem bias)
    f = [d] + [int(s) for s in spec.split("-")]
    layers = []
    for i in range(len(f) - 2):
        layers += [nn.Linear(f[i], f[i + 1], bias=False), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)]
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

# build_opt
if ssl == "barlow":
    extra["projector"] = make_bt_projector(d).to(device)
    extra["bt_loss"] = BarlowLoss(8192).to(device)      # sem parâmetros; precisa estar em train()
    # grupos LARS: para cada grupo atual (old/new/_spifil) separar ndim>1 (x0.2) e ndim==1 (x0.0048)
    groups = [{**g, "params": ps, "lr_scale": g["lr"] / lr_new * (0.2 if w else 0.0048)}
              for g in groups for w, ps in ((True,  [p for p in g["params"] if p.ndim > 1]),
                                            (False, [p for p in g["params"] if p.ndim == 1])) if ps]
    opt = LARS(groups, lr=0, weight_decay=1e-6, weight_decay_filter=True,
               lars_adaptation_filter=True)             # cópia literal de main.py:224-263
# laço
lr = cosine_warmup(global_step, total_steps, warmup_steps, peak=B / 256, end_ratio=1e-3)
for g in opt.param_groups: g["lr"] = lr * g["lr_scale"]
views = make_views_byol(x, generator=gen)               # t / t' assimétricas
with amp():
    h = head(encoder(torch.cat(views, 0)))              # representação (SVM)
    z = extra["projector"](h)                           # embedding (perda)
total, parts = extra["bt_loss"](*z.split(B, 0))         # fp32
# ... backward / skip NaN / step iguais ao atual; logar mean(diag c), mean(|off c|), std(h), rank efetivo
# fim: extra.clear() -> SVM C=100 original sobre h
```

### 6.6 O que é fiel e o que é adaptação

| Fiel (igual a [A]/[C]) | Adaptação (registrar como variante) |
|---|---|
| Perda do código: BN sem afinidade por ramo, `c/N`, soma, λ=0.0051 | batch 64 (oficial 2048, testado ≥128) e o piso decorrente; λ escalado ou D menor como variantes |
| Projetor 3×8192, BN+ReLU nas escondidas, **sem bias** | entrada = `EmbeddingHead` (≥1280) |
| Siamês, sem preditor/SG/EMA | LAB[0,1], jitter de cor via LAB↔RGB ou aproximado; vflip; crop sobre fonte de 224 px |
| Avaliação pré-projetor, congelada, sonda linear | sonda = SVM C=100 do repositório |
| LARS (η=1e-3, m=0.9, exclusão ndim==1), pesos 0.2 / bias 0.0048 × B/256, warmup de 10 ép., cosseno até 0.001× | proteção SPiFiL (`conv_lr_scale`, `ElementProtect`), escalas old/new, WiSE |
| wd 1e-6 (código) | épocas = fidelidades DEHB; warmup em fração dos passos; bf16 no forward com perda em fp32 |
| Augmentations BYOL assimétricas | variante AdamW: **sem precedente** no artigo do BT |

## 7. Testes de conformidade

Sugestão de local: `tests/modehb/test_ssl_barlow.py` (CPU).

1. **Paridade com o oficial**: copie `BarlowTwins.forward` (`main.py:207-221`) com `all_reduce` = identidade e o backbone/projetor substituídos por identidade. Para `z1, z2` aleatórios (N=64, D=512), `BarlowLoss` deve bater com `atol=1e-5` em fp32.
2. **Pseudocódigo × código**: a versão do Alg. 1 (std não-viesada) difere por um fator `(N−1)/N` em C. O teste fixa que seguimos o código (BN viesada).
3. **Propriedades**:
   - `z1 = z2` com colunas independentes e N ≫ D → `on ≈ 0`;
   - reescalar uma coluna de `z1` por qualquer constante > 0 não muda a perda (invariância da correlação, [A] App. A);
   - `z` constante → BN produz ~0 e `on ≈ D`.
4. **Piso**: N=64, D=8192, vistas idênticas → `off ≥ D²/(N−1) − D` (≈1,06·10⁶), conforme §6.2.
5. **Modo do BN**: com `BarlowLoss.eval()` o resultado muda. O teste garante que o trainer o mantém em `train()` durante o SSL.
6. **Projetor**: nenhuma `Linear` com bias; BN só nas escondidas; dims `[d, 8192, 8192, 8192]`.
7. **LARS e grupos**: todo parâmetro `ndim==1` (bias de conv do encoder, BN, `p_raw` do GeM) cai no grupo 0.0048 e fica fora do wd e da adaptação. Um passo é igual ao do LARS oficial copiado.
8. **Agenda**: `lr(0)=0`, pico `B/256` × {0.2, 0.0048}, fim 0.001× pico, continuidade com `step_offset`.
9. **Pré-projetor**: após o treino, `extra` está vazio e a feature do SVM é `head(encoder(x))` em `eval()`, com dim `embed_dim`.
10. **Augmentations**: frequências em 10 k amostras (jitter 0.8, grayscale 0.2, blur 1.0/0.1, solarização 0.0/0.2; escala do crop em [0.08, 1]); determinismo com seed.
11. **Smoke de ponta a ponta** (R9):
    - 2 épocas em dataset sintético, sem NaN;
    - `mean(diag c)` sobe;
    - bf16 × fp32 na perda com diferença relativa < 1e-2;
    - `tests/modehb/test_ssl_trainers.py` continua passando.
12. **SVM intocado**: κ via `core.svm.fit_svm` C=100 original.
