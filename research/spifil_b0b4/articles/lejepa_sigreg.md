# LeJEPA + SIGReg — regras de implementação fiéis ao artigo e comparação com o nosso código

## Referência

- **Artigo:** R. Balestriero, Y. LeCun. *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv:2511.08544 (v1 11/11/2025, v2 12/11/2025, **v3 14/11/2025**). Lido na versão HTML do arXiv (`arxiv.org/html/2511.08544`, v3) e no PDF.
- **Código oficial:** `github.com/rbalestr-lab/lejepa`, commit `c293d29` (25/01/2026). Arquivos lidos:
  - `lejepa/univariate/epps_pulley.py` (classe `EppsPulley`, L16–99)
  - `lejepa/multivariate/slicing.py` (classe `SlicingUnivariateTest`, L15–153)
  - `MINIMAL.md` (exemplo completo oficial, ViT-S/8 + Imagenette, "130 linhas")
  - `README.md` (seção "GOTO hyperparameters", L50–86)
  - `scripts/launch_inet10.py`, `scripts/launch_{epps,proj,views,rtokens}_ablation.md` (linhas de comando que geraram a Fig. 9 e a Tab. 1)
  - Obs.: o script de treino (`scripts/je.py`, baseado em `stable_pretraining`) **não está no repositório**; só as linhas de comando. Onde só ele definiria um valor, este documento diz "não especificado".
- Convenção de citação: "Sec. X" / "Tab. X" / "Eq. X" / "Fig. X" = artigo; `arquivo:Lnn` = código oficial; `nosso:arquivo:Lnn` = este repo.

---

## Resumo do método

1. **Tese (Sec. 3):** a distribuição ótima dos embeddings de um JEPA para minimizar o risco de *probes* lineares (Sec. 3.1: anisotropia aumenta a variância e, com ridge, o viés do estimador) e não-lineares kNN/kernel (Sec. 3.2) é a **Gaussiana isotrópica** N(0, I).
2. **SIGReg (Sec. 4, Def. 2, Eq. 5):** para forçar isso em alta dimensão, projeta os embeddings em |𝔸| direções aleatórias unitárias (Cramér-Wold "sketch") e aplica em cada projeção 1-D um teste univariado de normalidade contra N(0,1); a perda é a **média** sobre as direções (média e não máximo, para não ter gradiente esparso — Sec. 4.2, logo após a Eq. 5). O teste escolhido é **Epps–Pulley** (Sec. 4.2.3): distância L2 ponderada entre a função característica empírica e exp(−t²/2). Testes de momentos (Jarque–Bera) são instáveis/não-identificáveis (Sec. 4.2.1, Teo. 3) e testes de CDF exigem ordenação (Sec. 4.2.2).
3. **Predição/invariância (Sec. 5.1, Eqs. 6–8):** todas as V vistas predizem a **média das vistas globais** (distância L2²). Sem preditor, sem stop-gradient, sem teacher–student, sem EMA obrigatória.
4. **LeJEPA (Eq. 9):** `L = (λ/V)·Σ_v SIGReg(z_{·,v}) + ((1−λ)/B)·Σ_n L_pred(z_{n,·})` — **um** hiperparâmetro (λ).
5. **Afirmações empíricas relevantes:** estável em λ, nº de vistas, batch, slices e arquitetura (Sec. 6.1); a perda de treino correlaciona com a acurácia de *probe* (Spearman ≈ 85%, ≈ 99% com `loss/λ^0.4`, Sec. 6.2, Eq. 10); pré-treino *in-domain* em datasets pequenos (1k–11k imagens) supera modelos de fundação (Sec. 6.3, Tab. 3, Tab. 5).

---

## Receita oficial (tabela)

| Item | Valor no artigo | Valor no código oficial | Fonte |
|---|---|---|---|
| Perda total | `λ·mean_v SIGReg + (1−λ)·L_pred` | `sigreg*lamb + inv*(1-lamb)` | Eq. 9; listagem LeJEPA (Sec. 5.1); `MINIMAL.md:L174` |
| λ (default) | **0,05** ("robust default"), com V=8 | 0,05 (ablações ViT-L); 0,02 com V=4 (MINIMAL); grade {0,01; 0,02; 0,05; 0,1} (Fig. 9) | Sec. 6.1 §"Stability across standard hyperparameters"; `MINIMAL.md:L218`; `launch_inet10.py:L4` |
| λ × V | λ ótimo cresce com V: V=2→0,01; **V=4→0,02** (84,68); V=8→0,05 (84,32) | — | Tab. 7 / Fig. 8 (ResNet-50, IN-100, 400 ép.) |
| Teste univariado | Epps–Pulley, w(t)=exp(−t²/σ²), σ=1 (listagem: janela exp(−t²/2)) | `EppsPulley` | Sec. 4.2.3; `epps_pulley.py:L16–99` |
| Integração | trapézio, **17 nós**, domínio **[−5, 5]** (listagem e recomendação) | trapézio em **[0, 3]**, 17 nós, pesos dobrados por simetria (= [−3,3]) | Sec. 5.1 §"The SIGReg Loss"; Sec. 6.1 §"Stability across Epps-Pulley"; `epps_pulley.py:L62–78`; `MINIMAL.md:L60` |
| Escala da estatística | `T = N·∫…` (N = batch total, ×world_size em DDP) | `(err @ weights) * N * world_size` | Eq. Epps–Pulley; listagem SIGReg; `epps_pulley.py:L99` |
| Nº de slices \|𝔸\| | recomendado **1024**; 512 já competitivo; 256 na listagem | 1024 (README), 1000 (Fig. 9/views), 256 (MINIMAL) | Sec. 6.1; listagem SIGReg (`num_slices=256`); `MINIMAL.md:L70`; `launch_inet10.py:L4` |
| Direções | `randn(D, M)` normalizadas por coluna, **re-amostradas a cada passo**, mesmas para todas as vistas e GPUs (seed = global_step) | idem, `torch.Generator` semeado por `global_step` (MAX all_reduce) | Sec. 4.3 §"SGD Beats the Curse…", Fig. 7; `slicing.py:L131–143` |
| SIGReg por vista | sim: média sobre as V vistas, cada vista com N=B amostras | `SIGReg(proj)` com proj `[V,N,K]` → EP por vista, média | Eq. 9; listagem LeJEPA (`mean(SIGReg(emb) for emb in a_emb)`); `MINIMAL.md:L66–73` |
| Centro da predição | média das V_g vistas globais; **para não-ViT (ResNet): global_views = all_views** | `proj.mean(0)` (todas as vistas) | Eqs. 7–8; legenda da listagem LeJEPA (Sec. 5.1); `MINIMAL.md:L172` |
| Normalização de L_pred | ‖·‖²₂ (soma em D) | `.square().mean()` (média em V, B **e D**) | Eq. 6–8 vs `MINIMAL.md:L172` |
| Vistas | **V=8: 2 globais 224×224 + 6 locais 96×96** | 2 globais 224 (escala 0,3–1,0) + 6 locais 98 (escala 0,05–0,3); ablações ViT-L: 238/98 | Sec. 6.1 "Experiment Details 1"; `README.md:L55–58`; `launch_views_ablation.md` |
| Vistas p/ CNN (Fig. 9, timm < 20M) | — | `n_views=8`, `multi_crop=false`, `resolution=224` | `launch_inet10.py:L4` |
| Augmentations (todas as vistas) | não detalhadas no artigo | RandomResizedCrop; HFlip p=0,5; ColorJitter(0,4; 0,4; 0,2; 0,1) p=0,8; Grayscale p=0,2; GaussianBlur p=0,5; Solarize p=0,2 (thr 128); Normalize(ImageNet) | `README.md:L50–64` |
| Augmentations (MINIMAL) | — | RRC(128, scale 0,08–1,0); ColorJitter(0,8;0,8;0,8;0,2) p=0,8; Gray 0,2; Blur k=7 σ∈[0,1;2] p=0,5; Solarize p=0,2; HFlip | `MINIMAL.md:L102–110` |
| Batch | recomendado **≥ 128**; 128→72,20 / 256→74,15 / 512→74,72 / 1024→74,07 | 512 (ablações/Fig. 9), 256 (MINIMAL) | Sec. 6.1; Tab. 1c; `launch_*`; `MINIMAL.md:L218` |
| Otimizador | AdamW, lr ∈ {5e−3, 5e−4}, wd ∈ {1e−1, 1e−2, 1e−5}, sem schedule de wd | lr 5e−4 (README); wd **5e−2 ViT / 5e−4 ResNet**; Fig. 9: lr {3e−3, 1e−4}, wd {3e−2, 1e−5}; MINIMAL lr 2e−3, wd 5e−2 | Sec. 6.1 "Experiment Details 1"; `README.md:L68–76`; `launch_inet10.py:L4`; `MINIMAL.md:L154,L218` |
| Schedule de lr | "standard linear warm-up cosine-annealing" | lr final = lr/1000; MINIMAL: warmup de 1 época (`LinearLR start_factor=0.01`) + cosseno | Sec. 6.1; `README.md:L75–76`; `MINIMAL.md:L157–161` |
| Épocas | IN-1k 100; IN-100 400; Galaxy10 400; MINIMAL (Imagenette) 800 | 400 (Fig. 9), 100 (ablações) | Tab. 1; Fig. 8; Sec. 6.3; `MINIMAL.md:L218` |
| EMA / teacher–student | **não necessário**; SWA opcional só p/ ViT (pequeno ganho) | `teacher_student=true` nas ablações ViT-L; `false` na Fig. 9 (CNNs incl.) | Sec. 6.1 §"Removal of popular heuristics"; Tab. 4 |
| Preditor | não usar | `predictor=false` | Sec. 6.1; Tab. 4 |
| Projetor | MLP; 3 camadas > 2 > 1 (ResNet-50: 83,93 / 82,44 / 79,71); **proj. dim menor é melhor** (64 > … > 1024, Tab. 1d) | `MLP(512, [2048, 2048, proj_dim], BatchNorm1d)`; emb 512, proj 512 (ablações), 16 (MINIMAL) | Tab. 4; Tab. 1d; `MINIMAL.md:L88`; `launch_proj_ablation.md` |
| Onde a perda é aplicada | no **projetor** (eixo "proj. dim." da Tab. 1d) | `inv_loss` e `sigreg` sobre `proj`; probe sobre `emb` (backbone) com `detach` | Tab. 1d; `MINIMAL.md:L170–175` |
| Precisão | — | bf16 autocast, sem grad clipping, sem GradScaler necessário | `README.md:L74`; `MINIMAL.md:L168`, texto após `MINIMAL.md:L218` |
| Avaliação | linear probe com backbone congelado (top-1) | LN + Linear sobre emb (online, wd 1e−7) ; offline: CLS das 2 últimas camadas + LayerNorm, AdamW wd 1e−6, mesmo schedule | Sec. 6; `MINIMAL.md:L151–155`; `README.md:L77–86` |

---

## Regras de implementação (checklist numerado)

### A. Perda (SIGReg + predição)

1. **SIGReg = média, sobre |𝔸| direções unitárias, da estatística Epps–Pulley** de `{aᵀz_n}_{n=1..N}` contra N(0,1). Usar média, não máximo. *(Def. 2, Eq. 5 e parágrafo seguinte)*
2. **Epps–Pulley:** `T = N·∫|φ̂(t) − e^{−t²/2}|²·w(t) dt`, com `φ̂(t) = mean_n exp(i t x_n)` e `w(t)=e^{−t²/2}`. Implementar com cos/sin (parte real/imag), sem ordenação. *(Eq. Epps–Pulley, Sec. 4.2.3; `epps_pulley.py:L80–99`)*
3. **Quadratura:** trapézio com **17 nós**. Domínio [−5,5] (artigo) ou [0,3] com pesos dobrados por simetria (código oficial). Ambos equivalentes na Tab. 1a (74,21 vs 74,16 com 512 slices); **não usar [−1,1]** (~−2 pp). *(Sec. 5.1; Tab. 1a; `epps_pulley.py:L72–78`)*
4. **Multiplicar por N** (batch efetivo, × world_size em DDP). O viés de minibatch é O(1/N) e "mínimo, sem preocupação mesmo para minibatches de 16". *(Teo. 6; Sec. 5.1)*
5. **Direções:** `A = randn(D, M); A /= A.norm(dim=0)`; **re-amostrar a cada passo** (re-amostrar com M=16 supera um conjunto fixo de milhares de direções). As mesmas A para todas as vistas no passo. *(Sec. 4.3; Fig. 7; `slicing.py:L131–145`)*
6. **Nº de slices:** começar com **1024** (512 já competitivo). Teoria: |𝔸| = O(K) basta para embeddings suaves. Custo desprezível (Tab. 6: 0,47 ms para N=512, M=512; 0,67 ms para M=8192). *(Sec. 6.1; Teo. 5; Tab. 6)*
7. **SIGReg é calculado por vista** (N = B amostras de uma vista) e **promediado sobre as V vistas**; não juntar V·B amostras num único teste. *(Eq. 9; listagem LeJEPA)*
8. **Predição:** `L_pred = (1/V)·Σ_v ‖μ_n − z_{n,v}‖²`, com `μ_n` = média das vistas **globais**. **Em CNN/ResNet: todas as vistas são "globais"** → `μ_n` = média de todas as V vistas. *(Eqs. 6–8; legenda da listagem LeJEPA, Sec. 5.1)*
9. **Total:** `L = λ·SIGReg + (1−λ)·L_pred`. Sem stop-gradient, sem preditor, sem teacher–student, sem *register tokens*. *(Eq. 9; Sec. 6.1 §"Removal of popular heuristics")*
10. **λ:** 0,05 como default **com V=8**; ajustar λ proporcionalmente a V (V=4 → 0,02; V=2 → 0,01). *(Sec. 6.1; Fig. 8; Tab. 7)*
11. A normalização de `L_pred` do código oficial é `.square().mean()` (média também sobre D), o que difere da Eq. 9 (soma em D) por um fator D. Os valores de λ do código (0,02/0,05) estão calibrados **com a média**. Manter `.mean()`. *(`MINIMAL.md:L172` vs Eq. 8)*

### B. Arquitetura e onde aplicar a perda

12. Backbone qualquer (timm/torchvision), sem inicialização especial. *(listagem LeJEPA, legenda; `MINIMAL.md` §"The Code")*
13. **Projetor MLP com BatchNorm** após o embedding do backbone: 3 camadas (`[2048, 2048, proj_dim]`) é o melhor na Tab. 4 para ResNet-50. **Aplicar SIGReg + predição na saída do projetor**; dimensão de projeção pequena (64–128) é melhor que grande (Tab. 1d). *(Tab. 4; Tab. 1d; `MINIMAL.md:L88, L170–173`)*
14. O probe usa o **embedding do backbone** (antes do projetor). *(`MINIMAL.md:L175`; `README.md:L77–86`)*
15. EMA/SWA: não necessários; SWA opcional apenas em ViT. *(Sec. 6.1; Tab. 4)*

### C. Vistas e augmentations

16. **V = 8** (2 globais + 6 locais) como default; para CNN no sweep da Fig. 9: **8 vistas, todas 224 px, sem multi-crop**. *(Sec. 6.1 "Experiment Details 1"; `launch_inet10.py:L4`)*
17. Globais: RandomResizedCrop **224, escala (0,3; 1,0)**; locais: **96–98 px, escala (0,05; 0,3)**. *(Sec. 6.1; `README.md:L58`)*
18. Mesmas transformações fotométricas/geométricas em todas as vistas: HFlip 0,5; ColorJitter (b 0,4; c 0,4; s 0,2; h 0,1) p=0,8; Grayscale 0,2; GaussianBlur 0,5; Solarize 0,2; normalização média/desvio. *(`README.md:L55–64`)*

### D. Otimização

19. **AdamW**; lr 5e−4 como ponto de partida (artigo varre {5e−3, 5e−4}); **wd 5e−4 para ResNets**, 5e−2 para ViT; sem schedule de wd. *(Sec. 6.1; `README.md:L68–73`)*
20. **Warmup linear + cosseno** até lr/1000 (MINIMAL: warmup de 1 época, fator inicial 0,01). *(Sec. 6.1; `README.md:L75–76`; `MINIMAL.md:L157–161`)*
21. Batch **≥ 128** (Tab. 1c). bf16; sem grad clipping. *(Sec. 6.1; `MINIMAL.md`, texto após L218)*
22. Treinar longo em dataset pequeno: 400 épocas em Galaxy10 (11k) e IN-100; 800 no MINIMAL (Imagenette ≈ 9,5k). *(Sec. 6.3; Fig. 8; `MINIMAL.md:L218`)*

### E. Seleção de modelo

23. A perda de treino é informativa: usar `train_loss / λ^0.4` para seleção sem rótulo (Spearman ≈ 99%). *(Sec. 6.2, Eq. 10, Fig. 11)*

---

## O que o artigo diz sobre batch pequeno, poucos dados e CNNs

- **Batch pequeno.** Teo. 6: o viés do gradiente do EP é O(1/N); "achamos esse viés mínimo e sem preocupação mesmo para minibatches tão pequenos quanto 16" (Sec. 5.1). Empiricamente (Tab. 1c, ViT-L, IN-1k, 100 ép.): 128 → 72,20; 256 → 74,15; 512 → 74,72; 1024 → 74,07. "LeJEPA atinge desempenho competitivo com batch de apenas 128" e a recomendação é **≥128** (Sec. 6.1). O README/MINIMAL afirma que LeJEPA "funciona com mini-batches e datasets muito pequenos" (`MINIMAL.md:L35`), mas **nenhum experimento do artigo usa batch < 128**.
- **Poucos dados.** Sec. 6.3: Galaxy10 (11.008 imagens de treino), **hiperparâmetros default, 400 épocas**, supera DINOv2/v3 e I-JEPA em linear probe e fine-tuning (Fig. 12, Tab. 3). Tab. 5: pré-treino do zero *in-domain* em **flowers102 com 1.020 imagens** (ResNet-18: 74,57%; resnext26ts: 82,19%), inet10 (13k), galaxy10. A Tab. 3 cobre 1–1000 rótulos/classe no probe.
- **CNNs.** A listagem LeJEPA (Sec. 5.1) diz: "para arquiteturas não-ViT (p.ex. ResNet), set `global_views = all_views`". Fig. 9: ~50 modelos timm de 8 famílias (< 20M parâmetros) em INet10 atingem 91,5–95% de linear probe "out of the box", com lr e wd validados. Recomenda arquiteturas padrão (ResNet, ViT) em vez de especializadas como EfficientNet (Sec. 6.1). CNNs testadas: ResNet-18/34/50, ConvNeXtV2 (nano e H), resnext26ts, LeViT (Tab. 3–5; Fig. 9). **O artigo não testa CNNs rasas (3–5 camadas) nem extratores com kernels fixos ou pré-inicializados como o SPiFiL**: para o nosso caso isso é extrapolação.

---

## Protocolo de avaliação

- **Linear probe com backbone congelado**, top-1 (todas as tabelas, Sec. 6). Fig. 12/Tab. 3 também usam fine-tuning completo.
- **Features:** embedding do **backbone**, não do projetor (`MINIMAL.md:L175`: `probe(emb.detach())`). Offline: "concatenação do CLS das duas últimas camadas" (ViT); sem CLS, média dos patch tokens; **LayerNorm** antes do linear; AdamW wd 1e−6; mesmo schedule do pré-treino (`README.md:L77–86`). No MINIMAL o embedding é a saída `num_classes=512` do backbone timm (`MINIMAL.md:L80–85`, ViT). Para CNN o repositório não especifica; por inferência (timm), seria pooling global seguido dessa linear de 512 dimensões.
- Probe online (monitoramento): LN + Linear treinados junto, com `detach`, lr 1e−3, wd 1e−7 (`MINIMAL.md:L151–155`).
- Teoria: Sec. 3.1 (linear/OLS/ridge), Sec. 3.2 e Apêndice A (kNN por raio, kernel).

---

## Hiperparâmetros por tabela

| Tabela/Figura | Setup | Hiperparâmetros fixos | Resultado-chave |
|---|---|---|---|
| **Tab. 1a** (Epps–Pulley) | ViT-L/14, IN-1k, 100 ép. | λ=0,05, bs 512, emb 512, proj 512, lr 5e−4, wd 5e−2, V=8, 238/98 px, teacher–student on (`launch_epps_ablation.md`) | [−1,1]: 71,8–72,9; [−3,3]: 74,0–75,0; [−5,5]: 73,7–74,8; nós 5/17/41 ≈ iguais; 2048 slices > 512 (+~0,5 pp) |
| **Tab. 1b** (vistas) | idem | bs escala com V (256/432/512/640) | Vg=1 ruim (53–69); Vg=2: V=4 72,26, V=6 73,07, V=8 74,24, V=10 74,06; Vg=4, V=10: 75,08 |
| **Tab. 1c** (batch) | idem | V=8 | 128: 72,20; 256: 74,15; 512: 74,72; 1024: 74,07 |
| **Tab. 1d** (emb/proj) | idem, slices 1024/4096 | — | proj 64 melhor (75,29–75,65), 1024 pior (73,65–74,79); emb 2048 ≥ 512 marginalmente |
| **Tab. 1e** (registers) | idem | — | 0–8 registers: diferença ≤ 0,3 pp |
| **Tab. 4** | IN-100, 400 ép.; ResNet-50, ViT-S/8, ViT-T/8 | — | projetor 3 > 2 > 1 camadas (ResNet-50: 83,93 / 82,44 / 79,71); preditor não ajuda; SWA ajuda ViT |
| **Tab. 5** | do zero, *in-domain*, linear probe | default (Sec. 6.1) | flowers102 (1.020 imgs): ResNet-18 74,57; galaxy10: ResNet-34 77,29 vs I-JEPA ViT-H 62,93 |
| **Tab. 3 / Fig. 12** | Galaxy10, 400 ép., 3 seeds | default | ResNet-34 congelado, todos os rótulos: 78,17 vs DINOv3 ViT-S 71,38 |
| **Tab. 6** | custo do SIGReg (V100) | — | N=512, M=512, 16 nós: 0,47 ms |
| **Tab. 7 / Fig. 8** | ResNet-50, IN-100, 400 ép. | — | V=2: λ*=0,01 (83,49); **V=4: λ*=0,02 (84,68)**, λ=0,05 → 83,00; V=8: λ*=0,05 (84,32) |
| **Fig. 9** | ~50 timm < 20M, INet10, 400 ép. | bs 512, V=8 (224, sem multi-crop), 1000 slices, emb 512, proj 512, λ∈{0,01..0,1}, lr∈{3e−3,1e−4}, wd∈{3e−2,1e−5} (`launch_inet10.py:L4`) | 91,5–95% |
| **MINIMAL** | ViT-S/8, Imagenette 128 px | λ=0,02, V=4, proj 16, lr 2e−3, bs 256, 800 ép., 256 slices, t∈[0,3] | 90,7% em ~200 min |

Inconsistências internas da fonte (registradas, não resolvidas): a Sec. 6.1 recomenda "Vg=2, **Vl=8**", mas "Experiment Details 1" e o README usam **V=8 no total** (2+6); o artigo fala em locais de 96 px e o código em 98 px (patch 14); a listagem do artigo integra em [−5,5] e o código oficial em [0,3].

---

## Divergências do nosso código

Código analisado: `ga_search/modehb/ssl_trainers.py` (`train_global_ssl`), `ga_search/modehb/regularizers.py`, `src/losses/lejepa_loss.py` (`RealSIGReg`, `invariance_loss`), `src/losses/epps_pulley.py`, chamada em `ga_search/modehb/run_search.py:L707–717` e configs `ga_search/config/modehb/search_v3*.yaml`.

Legenda: **F** = já fiel; **D** = divergência a corrigir para ficar fiel; **A** = divergência que é adaptação necessária ou deliberada (manter, documentar).

| # | Item | Artigo / código oficial | Nosso | Status | Ação |
|---|---|---|---|---|---|
| 1 | Estatística Epps–Pulley | trapézio [0,3], 17 nós, pesos dobrados, `×N×world_size` | `src/losses/epps_pulley.py:L50–87`, linha a linha igual a `epps_pulley.py:L62–99` | **F** | — |
| 2 | Slicing | `randn(D,M)/‖·‖`, re-amostrado a cada passo, mesmo A p/ todas as vistas | `RealSIGReg` (`lejepa_loss.py:L109–116`): idem, com RNG global (não `Generator(global_step)`) | **F** (single-GPU) | Opcional: `Generator` semeado por step, para reprodutibilidade independente do resto do RNG |
| 3 | SIGReg por vista, média em V | Eq. 9 | `RealSIGReg(H)` com `H=[V,B,D]` → EP por vista, `.mean()` sobre V e slices (`ssl_trainers.py:L389–392`) | **F** | — |
| 4 | Centro da predição | CNN: média de **todas** as vistas | `invariance_loss = (H.mean(0) − H).square().mean()` (`lejepa_loss.py:L136`) = `MINIMAL.md:L172` | **F** | — |
| 5 | Combinação | `λ·SIGReg + (1−λ)·inv` | `ssl_trainers.py:L395` idem | **F** | — |
| 6 | Nº de slices | 1024 recomendado (256 só no MINIMAL) | `n_proj=256` (`lejepa_loss.py:L95`; `reg_kwargs` não passado em `run_search.py:L712`) | **D** (leve) | `reg_kwargs={"n_proj": 1024}`. Para D=1280 sem projetor, 256 < D (Teo. 5 pede \|𝔸\|=O(K)) |
| 7 | **Projetor / onde aplicar a perda** | MLP 3 camadas com BN (`[2048,2048,proj_dim]`), proj_dim 64–128; perda no projetor; probe no backbone | default `projector=False` → perda **direto no embedding do SVM** (D = `embed_dim`, até 1280; sementes `projector: false`). Gene `projector=True` usa MLP **2 camadas** 512→128 (`ssl_trainers.py:L222–224, L331–332`) | **D** | Projetor ligado sempre em `lejepa`; `Linear(d,2048)-BN-ReLU-Linear(2048,2048)-BN-ReLU-Linear(2048,128)`; descartar no fim (já é feito: `extra.clear()`) |
| 8 | **λ vs V** | λ=0,05 ⇔ V=8; V=4 ⇒ λ≈0,02 (Tab. 7: 84,68 vs 83,00 com 0,05) | V=4 com λ=0,05 (`ssl_trainers.py:L293`; `run_search.py:L65`) | **D** | Ou V=8 + λ=0,05, ou V=4 + λ=0,02 |
| 9 | **Augmentations** | ColorJitter forte (0,4/0,4/0,2/0,1, p=0,8), Gray 0,2, Blur 0,5, Solarize 0,2, HFlip, crop (0,3–1,0) (+ locais 0,05–0,3) | `AugConfig` (`ssl_trainers.py:L76–85`): crop (0,3–1,0) ✔, flips H/V, brilho ±0,2, contraste ±0,2, jitter por canal ±0,05; **sem blur, gray, solarize, sem p=0,8** | **D** + **A** | Fortalecer, com versão LAB (ver Adaptação §3). O vflip é adaptação válida (microscopia) |
| 10 | Multi-crop | 2 globais 224 + 6 locais 96 (ViT); CNN no sweep: 8 globais 224 | 4 vistas, todas `out_size=H` (224), mesma escala | **D** (nº de vistas) / **F** (sem locais em CNN é o que o sweep CNN faz) | V=8 globais; locais 96 px opcionais (CNN+pooling aceita) como ablação |
| 11 | **Otimizador: wd** | AdamW wd 5e−4 (ResNet) / grade {1e−1,1e−2,1e−5} | `weight_decay=0.0` por default; a config tem `weight_decay: 1.0e-4`, mas **ela não é repassada** a `train_global_ssl` (`run_search.py:L712–717`), então o valor efetivo é 0 | **D** (bug de plumbing) | Passar `weight_decay=tr["weight_decay"]`; valor 5e−4 |
| 12 | **Schedule de lr** | warmup linear + cosseno até lr/1000 | lr **fixo** ("never tuned", `ssl_trainers.py:L272`) | **D** | `LinearLR(0.01→1, ~1 época ou 5% dos passos)` + `CosineAnnealingLR(eta_min=lr/1000)` |
| 13 | lr | 5e−4 (README); grade {5e−3, 5e−4} | `lr_new=1e−3`, `lr_old=1e−4` (configs `search_v3*.yaml`). O pedido cita 1e−3/1e−2; 1e−2 fica **acima** de todo valor usado no artigo | **A** (grupos) / **D** (1e−2) | lr de pico ≤ 5e−3; o par old/new é adaptação SPiFiL (manter) |
| 14 | **Batch** | ≥ 128 (Tab. 1c) | 64, "FIXO; nunca autoajustado" (`search_v3*.yaml`) | **D** | 128–256 (ver Adaptação §4); decisão do usuário, porque o batch está travado por regra |
| 15 | Duração | 400–800 épocas completas em 1k–11k imagens | fidelidade ≤ 200 épocas totais × 2/3 SSL, com "época" ≤ 2048 imagens (32 passos): no máximo ≈ 4,3k passos | **A** / **D** | Medir em passos; mirar ≥ 8k passos na fidelidade máxima (Galaxy10: 400 × 11k / 512 ≈ 8,6k) |
| 16 | Normalização de entrada | Normalize(ImageNet mean/std) | LAB em [0,1], sem padronização | **A** | Manter LAB (pipeline SPiFiL); opcional: padronizar por canal com estatística do treino |
| 17 | Precisão | bf16 | fp32 default; bf16 opt-in com perdas em fp32 (`ssl_trainers.py:L310–314`) | **F/A** | bf16 ok; manter SIGReg em fp32 (cos/sin) |
| 18 | Grad clip | nenhum | `grad_clip=None` default | **F** | — |
| 19 | EMA / teacher / preditor | nenhum | nenhum | **F** | — |
| 20 | Grupos de lr + proteção de convs SPiFiL (`conv_lr_scale`, `ElementProtect`) + WiSE-FT | não existe | `ssl_trainers.py:L185–203`; `run_search.py:L708–720` | **A** | Manter (princípio Hybrid SPiFiL); documentar como fora do artigo |
| 21 | Avaliação | linear probe (LN + Linear, AdamW) no embedding do backbone | SVM C=100 (readout `cov_head`, StandardScaler) no embedding da head | **A** | **Não mudar** (regra do repo: SVM original C=100). Opcional: reportar também um probe LN+Linear no mesmo embedding como régua do artigo |
| 22 | Normalização de `L_pred` | código: `.mean()` em D; Eq. 9: soma em D | `.mean()` | **F** (ao código) | — |
| 23 | NaN-skip | não existe | pula passos não-finitos (`ssl_trainers.py:L399–418`) | **A** | Manter (só proteção) |

---

## Adaptação para o nosso caso

Contexto: CNN rasa com kernels SPiFiL (Hybrid SPiFiL: backprop soma, não desfaz), entradas de 224 px em LAB [0,1], datasets de 1k a 10k imagens e GPU de 48 GB. Cada regra diz se é **[fiel]** ao artigo ou uma **[adaptação]** necessária.

1. **Perda inalterada [fiel].** Manter `EppsPulley` + `RealSIGReg` + `invariance_loss` exatamente como estão (itens 1–5 e 22 já conformes). Subir `n_proj` para **1024** [fiel]; custo desprezível (Tab. 6).
2. **Projetor sempre ligado em `lejepa` [fiel].** Usar MLP de 3 camadas com BN, `d → 2048 → 2048 → 128` (ou `d → 1024 → 1024 → 128`, se a memória apertar; isso é **[adaptação]**, pois o artigo não testa larguras menores que 2048 no projetor). A perda vai no projetor, o SVM lê o embedding da head e o projetor é descartado. Isso separa o objetivo SSL do espaço do SVM, como no artigo (`MINIMAL.md:L175`). Com proj_dim=128, 1024 slices é ≫ K. **Pergunta aberta:** a teoria (Sec. 3) diz que isotropia é ótima *para o espaço do probe*. Aplicar SIGReg direto no embedding do SVM (o nosso default atual) é coerente com a teoria, mas não é o que o artigo faz na prática (Tab. 1d/Tab. 4). Por isso vale manter `projector` como gene de ablação, com o default fiel = True.
3. **Augmentations LAB [adaptação do README, L55–64].** Equivalentes em LAB [0,1]:
   - crop global (0,3; 1,0) [fiel]; H+V flip (e rotações de 90°) [adaptação: microscopia não tem orientação canônica];
   - "brilho 0,4" → deslocamento em L de ±0,4·(faixa) ou multiplicativo em L, com p=0,8 [adaptação];
   - "contraste 0,4" → escala em torno da média de L, ±0,4, p=0,8 [adaptação];
   - "saturação 0,2" → escala de (a−0,5, b−0,5) por 1±0,2 [adaptação];
   - "hue 0,1" → rotação do vetor (a, b) em ±0,1·π [adaptação];
   - grayscale p=0,2 → a=b=0,5 [adaptação];
   - GaussianBlur p=0,5, σ∈[0,1; 2] [fiel];
   - solarize p=0,2 em L (L → 1−L acima de 0,5) [adaptação; testar, pode atrapalhar coloração diagnóstica].
   - Regra: validar visualmente as vistas (`preview_crops/`) e fazer ablação "fotometria forte vs atual" antes de fixar.
4. **Vistas, λ e batch [fiel na regra λ×V; adaptação no tamanho].** Preferir **V=8 globais em 224 + λ=0,05** (sweep CNN da Fig. 9 + Tab. 7). Se a memória não permitir, **V=4 + λ=0,02** (Tab. 7, MINIMAL). Batch **B=128** como alvo (≥ 128, Tab. 1c); com B·V = 1024 imagens de 224² por passo, uma CNN rasa cabe em 48 GB com bf16 nos genomas estreitos. Regra de queda, em ordem: (i) bf16; (ii) V=8→4 com λ=0,02; (iii) B=128→64 (aceito pelo Teo. 6, mas espere uma perda de ~2 pp, extrapolando a tendência 128→512 da Tab. 1c); nunca B<16. Locais de 96 px são uma ablação opcional (CNN com pooling global aceita), não o default para CNN.
5. **Otimizador [fiel].** AdamW, **wd 5e−4** (corrigir o plumbing: hoje o valor efetivo é 0), lr de pico 5e−4 a 1e−3 para as camadas novas. [adaptação] Camadas SPiFiL herdadas com lr×0,1 (o par `lr_old/lr_new` atual), ou a proteção do gene `protect`. Nunca 1e−2.
6. **Schedule [fiel] + DEHB [adaptação].** Warmup linear de 1 época (ou 5% dos passos, o que for maior), começando em 0,01·lr, seguido de cosseno até lr/1000, **calculado sobre o nº de passos SSL desta chamada**. Na promoção de fidelidade com *resume* (`step_offset`), reiniciar warmup+cosseno sobre os passos adicionais (estilo SGDR). Isso é adaptação: o artigo não tem treino multi-fidelidade.
7. **Duração [adaptação guiada pelo artigo].** Contar em **passos**, não em "épocas" truncadas de 2048 imagens. Na fidelidade máxima, mirar ≥ 8k passos de B=128 (ordem de grandeza de Galaxy10: 400 ép. × 11k imgs / 512). Nas fidelidades baixas, o ranking do DEHB pode usar a perda de treino LeJEPA (Sec. 6.2) como sinal barato.
8. **Seleção sem rótulo [fiel, opcional].** Logar `loss_total / λ^0.4` (Eq. 10) e checar a correlação de Spearman com o κ do SVM nos nossos runs. Se for alta, ela serve como critério de early-stop ou fitness auxiliar.
9. **Avaliação [adaptação obrigatória].** Manter o SVM C=100 do repo como régua principal. O linear probe do artigo (LN+Linear, AdamW, wd 1e−6) pode ser reportado à parte, mas nunca misturado com o κ do SVM.
10. **Hybrid SPiFiL [adaptação, fora do artigo].** `conv_lr_scale`/`ElementProtect`/WiSE-FT não interagem com a definição da perda. Documentar que o artigo assume todos os parâmetros treináveis com o mesmo lr.

---

## Testes de conformidade sugeridos

1. **Paridade numérica com o pacote oficial.** Com `x = randn(B=64, D=128)` e as mesmas direções A (gerador fixo), comparar `RealSIGReg(x[None])` com `SlicingUnivariateTest(EppsPulley(t_max=3, n_points=17), num_slices=M)(x)` do repositório oficial (instalado num venv isolado ou copiado só para o teste). Tolerância: 1e−6 em fp32.
2. **Piso e sensibilidade da estatística.** Para `x ~ N(0, I)`, o SIGReg deve ficar perto do piso de viés `∫w(t)(1−e^{−t²})dt` (termo O(1/N) do Teo. 6 vezes o fator N da estatística, ou seja, constante) e não crescer com B. Para `x` colapsado (constante), anisotrópico (`diag(10, 0,1, …)`) ou deslocado (média 2), deve ser ≫ piso.
3. **Re-amostragem de direções.** Duas chamadas consecutivas usam A diferentes, e a mesma chamada usa a mesma A para todas as V vistas (verificar com hook em `proj @ A`).
4. **Experimento sintético "X" (Fig. 6/16).** Com N=100 pontos em D=512, as 2 primeiras coordenadas em forma de "X" e o resto N(0,1), otimizar os pontos diretamente com SIGReg (M=16, re-amostrado) por ~1000 passos. As 2 coordenadas devem voltar a ficar gaussianas (teste de Shapiro/EP nos marginais) e a covariância → I.
5. **Forma da perda (Eq. 9).** Teste unitário: `total == λ·mean_v SIGReg(H[v]) + (1−λ)·mean((H.mean(0) − H)²)`, e o SIGReg calculado **por vista** (não sobre `H.reshape(V·B, D)`).
6. **Regra λ×V (Tab. 7) em miniatura.** Num dataset (ex.: larvae split 1), grade V∈{4, 8} × λ∈{0,01; 0,02; 0,05; 0,1}, avaliada com o SVM C=100. Esperado: o ótimo de λ cresce com V. Se não crescer, registrar como desvio do nosso regime.
7. **Isotropia ao fim do treino.** No espaço onde o SIGReg é aplicado (projetor), checar média ≈ 0, autovalores da covariância ≈ 1 (razão máx/mín < 10) e posto efetivo ≈ D. No embedding do SVM, logar o mesmo como diagnóstico, sem exigir.
8. **Plumbing de hiperparâmetros.** Teste que falha se o `weight_decay` dos param groups de `train_global_ssl` diferir de `training.weight_decay` da config, se `n_proj` ≠ 1024, ou se o lr não seguir warmup+cosseno (lr no passo 0 ≈ 0,01·lr, no pico ≈ lr, no fim ≈ lr/1000).
9. **Correlação perda × κ (Sec. 6.2).** Em ≥ 10 runs variando lr/wd/λ/épocas, calcular o Spearman entre `loss/λ^0.4` e o κ do SVM. O artigo reporta ρ ≈ 0,85–0,99. Valor baixo indica que o nosso regime (CNN rasa + SPiFiL + SVM) quebra uma premissa do artigo.
10. **Sanidade das augmentations.** Salvar uma grade de 8 vistas por imagem (antes/depois da conversão LAB→RGB) e checar que a faixa fica em [0,1], que a fotometria forte não destrói a estrutura do parasita e que vistas diferentes da mesma imagem não são quase idênticas (cosseno médio entre vistas no embedding de uma rede aleatória < 0,99).
