# Relatório Halloween (31/10–02/11): discovery híbrida SPiFiL, log completo (Athena + Jaci)

Todos os experimentos foram parados em **02/11, por volta das 23h UTC**, nas duas máquinas. Este arquivo junta tudo o que foi feito e medido.

**Fontes e siglas.** DEC = `A_reports/october/discovery_hybrid_spifil_decisions.md` (citado pelo título da entrada); REC = `discovery_recipe.md`; RANK = `discovery_ranking_test.md`; DIAG = `discovery_eggs_diag.md`; ES = `discovery_early_stop.md`; LIT2 = `discovery_literature_round2.md`; CONV = `discovery_conv_techniques.md`; BOARD = `discovery_insights_board.md`; SLP = `discovery_scaling_law_prompt.md` (todos em `A_reports/october/`); ATH = `A_reports/november/report_discovery_athena.md`; LDD = `november/low_data_diag.md`; LDL = `november/low_data_literature.md`; SL = `november/scaling_law.md`; CSV = `results/svm_disc_*.csv` (agregados no Apêndice A); IJ = `/dados/home/moliveira/Scalable_Hybrid_FLIM/results/ijepa_svm_aggregated.csv`. Números da Jaci que ainda não estão na Athena aparecem como **"Jaci, relatado"** ou **"Jaci, relatório final 02/11"**.

κ = κ de Cohen do SVM linear oficial (`core/svm.py` + `core/metrics.py`). Quando não se diz outra coisa, é o κ de **validação**, média de 3 splits, com 50% dos dados.

---

## 0. Resumo para criança

1. Fizemos uma rede pequena (uns 200 mil números) que aprende a olhar fotos de parasitas sem ninguém dizer o nome de cada um.
2. Cada andar da rede começa com "carimbos" recortados das próprias fotos (é o SPiFiL). Depois a rede lixa esses carimbos tentando redesenhar a foto.
3. No fim, um juiz simples (o SVM) olha o resumo que a rede fez e diz qual é o parasita.
4. Quase todo o ganho veio de **como fazemos o resumo**, e não de mudar a rede.
5. O resumo bom tem 3 truques: olhar mais os pontos fortes (GeM), olhar a foto girada de 8 jeitos (TTA) e deixar todas as pistas com a mesma força (whitening).
6. Com metade dos dados, empatamos com o I-JEPA (um modelo gigante treinado no ImageNet) nos ovos e passamos dele nas larvas e nos protozoários.
7. Com muito pouco dado (1–5%) ainda perdemos. O I-JEPA já viu 1,2 milhão de fotos; nós vimos de 17 a 46.
8. Ficaram pela metade: a lei de escala (encolher a rede), o treino com poucos dados e a repetição final na Jaci.

**Tabela principal: κ de TESTE com 50% dos dados, média de 3 splits**

| dataset | nosso modelo | I-JEPA (1 vista) | Δ | fonte |
|---|---|---|---|---|
| eggs | **0.941** (e_bp10 + gem_tta_pw; 0.943/0.940/0.940) | 0.940 | +0.001 (empate) | Jaci, relatado; DEC "eggs empata com o I-JEPA no teste" |
| larvae | **0.968** (bp01w64c1280 + gem_tta_pw) | 0.943 | +0.025 | CSV `bp01w64c1280_test_gemttapwr` |
| protozoan | **0.928** (bp01w64c1280 + gem_tta_pw) | 0.864 | +0.064 | idem |

A base `bp01w64c1280` sozinha dá 0.933 em eggs (0.949/0.935/0.916). O I-JEPA vem de IJ: 0.9397 / 0.9433 / 0.8636. **Ressalva:** o I-JEPA foi medido com uma vista só, sem TTA nem whitening (DEC "TTA confirmado no teste").

---

## 1. Objetivo e regras do trabalho

- **Objetivo:** κ de teste ≥ I-JEPA nos 3 datasets com 50% dos dados (DEC "Alvo"). **Atingido** (tabela acima).
- **Arquitetura:** só CNN, com entrada **LAB** (`ift_lab`).
- **SPiFiL:** toda camada nasce de kernels SPiFiL, depois vem o backprop. O SPiFiL pode ser só o inicializador (DEC "o SPiFiL pode ser só o inicializador"). A contribuição dele é provada contra um controle `random_layer_classic` com várias seeds.
- **Sem fine-tune com rótulo:** o encoder treina só com autoencoder (decoder novo a cada camada). O rótulo entra apenas no SVM e na escolha dos filtros pelo Fisher dentro do SPiFiL (REC §4).
- **Tamanho:** encoder + projetora ≤ **5M** parâmetros. O teto era 1M e subiu para 5M; o decoder fica fora da conta (DEC 30/10). O modelo final tem 200.944 parâmetros.
- **Validação escolhe, teste só reporta.** A regra vale para camada, leitura e eixo (DEC "TESTE dos melhores modelos").
- **Sem flatten.** A leitura e o κ de treino têm sempre **< 2000-d** (DEC "nada de flatten").
- **GPU:** na Athena, **20 GB por GPU** somando todos os nossos processos. Na Jaci, **70%** da GPU (68.5 GB). O SVM roda um por vez, sob flock, só depois do treino.
- **Decisão:** pela média de 3 splits. Δ abaixo de ~0.02 é ruído. Cada família muda uma coisa só contra a referência **no mesmo orçamento de épocas**.

## 2. O modelo final

| parte | o que é | fonte |
|---|---|---|
| entrada | LAB 200×200, sem aug | REC §2.1 |
| camadas | 4 camadas: 5×5 / 5×5 / 5×5 / 1×1, com 32/48/64/1280 filtros (protozoan 30/48/60/1280); ReLU; maxpool 3×3 stride 2 nas 3 primeiras; mapa final 24×24; sem BN, SE ou GRN | REC §2.6 |
| parâmetros | 200.944 (eggs/larvae), 188.468 (protozoan) | REC §2.6 |
| SPiFiL por camada | DISF com 50 superpixels dentro da máscara, 1 imagem por classe (`paper_classes: true`, impureza fora), Fisher + Ledoit-Wolf, DiversitySelector, alocação uniforme. Cada camada é estimada sobre as features do encoder **já treinado** | REC §2.2, §3.1 |
| L4 1×1 | 400/100/300 filtros SPiFiL (eggs/larvae/protozoan); o resto até 1280 é aleatório na norma média (`fill_random_last`) | REC §2.2 |
| treino | camada nova com lr 5e-4; antigas ×0.1 (`layerwise_backprop: 0.1`); AdamW com wd 0.05; warmup 10 + cosine; **200 épocas**; ckpt pela menor `probe/val_recon_loss` (patience 50) | REC §2.3–2.5; ES |
| loss | BCE por pixel, com o fundo pesando 0.1 (`recon_bg_weight`) | REC §2.4 |
| leitura | **`gem_tta_pw`**: GeM p=3 na L4 → média das 8 rotações/flips → PCA-whitening com shrinkage (1/√(var+0.1·média(var))), fitado no treino → L2 → `SVC` linear com C=100 | REC §2.7 |

**Receita por dataset (escolhida na val):**

| dataset | receita | val | teste |
|---|---|---|---|
| eggs | **bp 1.0** (`growth_disc_g20_e_bp10`) + gem_tta_pw | 0.949 | 0.941 (Jaci) |
| larvae | base bp01w64c1280 + gem_tta_pw | 0.961 | 0.968 |
| protozoan | base bp01w64c1280 + gem_tta_pw | 0.926 | 0.928 |

Fontes: DEC "com whitening, o aug para de somar" e "a leitura final absorve"; SL. Com a leitura final, aug e bp03 empatam.

---

## 3. Cronologia completa

Lista por fase. Para cada família: data, o que mudou, a hipótese do SPiFiL, o κ e o veredito.

### 3.1 Fase a1–a3 / R1 / R2 / rand (29–31/10, flatten na L3)

| data | família | mudança | hipótese SPiFiL | κ val (e/l/p) | veredito |
|---|---|---|---|---|---|
| 29/10 | r1 | 3 camadas 5×5, 24/32/48, flatten | baseline reproduzível | L3 0.897/0.830/0.848 (CSV) | referência |
| 30/10 | a1_mask50 | máscara 50% na recon | MAE ajuda | interrompido em 1/9 (DEC 30/10) | — |
| 30/10 | a1_fg01 | fundo 0.1 na BCE | o fundo (97% dos pixels) apaga o foco do SPiFiL no objeto | 0.892/0.835/0.855 | empata; o mecanismo caiu (Spearman 0.20) |
| 30/10 | a2_w64 | largura 32/48/64 | mais protótipos | protozoan 0.860 (+0.011, 3/3) | empata em eggs/larvae |
| 30/10 | a3_popz | z-score populacional | — | 0.886/0.820/0.844 | perde |
| 30/10 | a3_sp200 | 200 superpixels | mais sementes | larvae oscila 0.890 / 0.776 | ruído |
| 31/10 | r2 | 8 camadas com dilation 2/4 | dilation aumenta o RF sem encolher o mapa | eggs L3 0.894 → L8 0.817 (CSV) | perde; interrompido 05:50 |
| 31/10 | rand, rand_s1, rand_s2 | filtros aleatórios na mesma norma | o SPiFiL contribui? | L3 rand: protozoan 0.809 contra SPiFiL 0.855 | **SPiFiL +0.046 em protozoan, +0.013 larvae, −0.008 eggs** |
| 31/10 | a2_bp01 | antigas treinam ×0.1 | o backprop junto com o SPiFiL ajusta o conjunto | 0.902/0.829/0.866 | ganha (+0.010 eggs, 3/3) |
| 31/10 | w64fg | fg01 + w64 | largura + fundo | 0.897/0.833/— | não soma |

### 3.2 Fase bp01w64 / c1280 (31/10–01/11)

| data | família | mudança | hipótese | κ val | veredito |
|---|---|---|---|---|---|
| 01/11 | bp01w64 | bp01 + 32/48/64 | — | L3 flat 0.906/0.845/0.865 | **nova base** |
| 31/10 | bp01c1280 | L4 1×1×1280, leitura avg | 1×1 SPiFiL = protótipo; 1280 = "saco de palavras visuais" | avg L3→L4 +0.04 a +0.06 | ganha no avg |
| 01/11 | bp01w64c1280 | as duas juntas | — | L4 avg 0.880/0.917/0.837; **gem 0.899/0.910/0.875** | **backbone final** |
| 01/11 | bp01w64c96 | L4 1×1×96 flat | 1×1 recombina sem perder a posição | −0.007 | refutada |

### 3.3 Fase g20, 200 épocas (01–02/11; base bp01w64c1280, gem L4)

Valores de κ val gem L4, média de 3 splits. Referências: eggs ~0.900 (base 0.899, e_base 0.902); larvae 0.910 (média do grupo 0.913); protozoan 0.875.

| família | mudança | hipótese SPiFiL | κ | veredito |
|---|---|---|---|---|
| e_bp03 / e_bp10 | backprop 0.3 / 1.0 | SPiFiL só como init | 0.905 / 0.908 | sobe fraco |
| l_bp03 / l_bp10 | idem em larvae | idem | **0.933** / 0.914 | pico em 0.3 |
| e_w96 / e_w128 | largura ×1.5 / ×2 | mais protótipos por classe | 0.901 / 0.900 | plana |
| e_k3 / e_k7 | kernel 3 / 7 | o patch do superpixel carrega a forma | 0.875 / 0.902 | k3 perde; k7 satura |
| e_aug / l_aug | geo_aug | invariância à rotação | 0.903 / 0.919 | neutro com gem |
| e_bn / l_bn | BatchNorm | estabiliza a escala | 0.897 / 0.887 | neutro / perde −0.023 |
| e_bg0 / l_bg0 | fundo 0 | foco total no objeto | 0.892 / 0.917 | empata |
| l_sp100 | 100 superpixels | mais SPiFiL, menos aleatório na 1×1 | 0.903 | empata |
| e_grn / l_grn | GRN na L4 | canais raros competem | 0.898 / 0.911 | empata |
| e_anc03 / e_anc3 / l_anc3 | âncora L2-SP λ 0.3/3 | preservar o contraste objeto/fundo | 0.813 / 0.829 / 0.865 | **perde −0.05 a −0.09** |
| e_base / p_base | base repetida | ruído do pipeline | e 0.902 | ruído ±0.02 por split, ~0.003 na média |
| l_w96, l_c2560, p_* (200) | — | — | l_w96 só s1; l_c2560 abandonada (>2000-d); p_* pausados no teto de 20 GB | — |

### 3.4 Fase e60 (02/11; 60 épocas, comparar com base_e60 0.875/0.903/0.861)

| família | mudança | κ gem | veredito |
|---|---|---|---|
| base_e60 | 60 épocas | 0.875/0.903/0.861 | gate: eggs **rejeita** (−0.024), larvae aceita (−0.007), protozoan aceita na borda (−0.014) |
| e_base_e100 | 100 épocas | 0.873 | lei do orçamento: 60 ≈ 100 < 200 |
| e_base_e400 | 400 épocas | interrompida (2 splits até L4, sem SVM) | — |
| e_se / l_se | SE | 0.871 / 0.909 | neutro |
| e_dil2 | dilation 2 na L3 | 0.868 | empata |
| e_d5 / e_d6 | +1 / +2 camadas | 0.857 / 0.851 | cai |
| e_res49 | pools 2/2/1/1 (L4 49×49) | 0.847 | perde |
| e_sp25 / e_sp100 | 25 / 100 superpixels | 0.881 / 0.868 | plana |
| e_imp / p_imp | `paper_classes: false` (a impureza ganha semente) | 0.847 / 0.859 | perde em eggs (−0.028); empata em protozoan |
| p_aug, p_bp03, p_bn (e60) | — | ver Apêndice A | p_bp03 + pw 0.914 contra 0.907 (empate) |
| mask50_e60 | MAE 50% | 0.875/0.903/0.842 | empata; protozoan −0.019 |
| bg1_e60 | fundo com peso 1 | 0.873/0.900/0.856 | empata |
| nomask_e60 | SPiFiL sem máscara, impurities true | 0.882/0.903/0.852 | empata; teste gem 0.883/0.905/0.855 (Jaci, relatório final 02/11) |

### 3.5 Fase eggs-focus (02/11)

- **Diagnóstico (DIAG):** 218 dos 220 erros do gem L4 são ovo ↔ impureza. 137 são impureza → ovo e 81 são ovo → impureza. Só 2 erros são ovo ↔ ovo.
- **A impureza não tem semente SPiFiL** (`paper_classes`). Dar uma semente a ela piora (e_imp).
- O que **não** resolveu: resolução, RF (dil2, k7, d5, d6), largura, sementes, âncora, SE, GRN, BN. São ~25 eixos sem ganho.
- O único botão de treino que mexeu: **orçamento** (60 → 200: +0.024) e **bp 1.0** quando combinado com a leitura final.

### 3.6 Fase das leituras (01–02/11, só eval, mesmo encoder bp01w64c1280)

| leitura | eggs | larvae | protozoan | nota |
|---|---|---|---|---|
| flatten L3 (histórico) | 0.906 | 0.845 | 0.865 | proibido depois (>2000-d) |
| avg L4 | 0.880 | 0.917 | 0.837 | |
| spp L3 | 0.878 | 0.872 | 0.829 | |
| gem L4 | 0.899 | 0.910 | 0.875 | |
| gem_dense (L3+L4) | 0.905 | 0.904 | 0.871 | empata |
| gem_fg (só no objeto) | 0.899 | 0.900 | 0.871 | empata: o SVM não precisa do fundo |
| gem_tta | 0.912 | 0.946 | 0.917 | ganha |
| gem_tta_pw **sem** shrinkage | 0.24–0.44 | ~0 | — | **colapsa** (inválido) |
| gem_pwr | 0.928 | — | — | |
| **gem_tta_pw(r)** | **0.942** | **0.961** | **0.926** | final |

Teste do mesmo encoder: avg 0.880/0.928/0.845 → gem 0.891/0.939/0.886 → tta 0.908/0.952/0.921 → **tta_pw 0.933/0.968/0.928** (REC §2.7).

### 3.7 Escolha de filtros no SPiFiL (02/11; e_aug + uma opção; referência e_aug + gem_tta 0.920)

| família | opção | hipótese | estado |
|---|---|---|---|
| e_aug_wh | ZCA nos patches antes do Fisher | o Fisher escolhe forma, não brilho | **interrompida**: s1/s2 até L4, s3 até L2; sem SVM |
| e_aug_pn | patch_norm | idem | interrompida: s1/s2 até L4; sem SVM |
| e_aug_div | NMS com \|cos\|>0.9 (71 de 400 candidatos são quase duplicatas) | filtros duplicados | interrompida: s1/s2 até L4; sem SVM |
| e_aug_knn | média de 5 vizinhos | centróide menos ruidoso | interrompida: s1/s2 até L4; sem SVM |
| e_aug_ipc3 | 3 imagens por classe | N > D | interrompida: s1/s2 até L4, s3 L1; sem SVM |

Estado dos checkpoints conferido em `/dados/home/moliveira/artifacts/spifil_growth/` (Athena). **Nenhuma dessas opções tem κ.**

### 3.8 ND (`feat_whiten`, Network Deconvolution de canal no treino)

- `disc_g20_nd_e200`: lançada às 21:49 e relançada às 21:50 com `gpu_ids [1]` (BOARD).
- **Interrompida sem nenhum checkpoint.** Nada a reportar.

### 3.9 Receitas finais (aug/bp10 + leitura final)

| família | κ val gem_tta_pw | teste | fonte |
|---|---|---|---|
| base bp01w64c1280 | 0.942/0.961/0.926 | 0.933/0.968/0.928 | CSV |
| e_bp10 | **0.949** | **0.941** | CSV val; Jaci, relatado (teste) |
| e_aug | 0.937 | ver Ap. A | CSV |
| l_aug | 0.957 | 0.964 (Jaci, relatado) | BOARD 19:14 |
| l_bp03 | 0.951 | 0.960 (Jaci, relatado) | DEC |
| p_bp03_e60 | 0.914 | 0.918 (Jaci, relatado) | BOARD 21:45 |
| e_bp10_aug | gem_tta 0.926; gem 0.915; avg L4 0.891; avg L3 0.837 | 0.918 / 0.907 / 0.884 / 0.843 | Jaci, relatado |
| l_bp10_aug | gem_tta 0.940; gem 0.915; avg L4 0.914 | 0.958 / 0.947 / 0.944 | Jaci, relatado |
| p_bp10_aug, p_aug200 | — | — | interrompidas (Athena): p_bp10_aug com s1 até L4; p_aug200 com s1/s2 até L4 |

O bp10 + aug não tem gem_tta_pw. Com gem_tta, ele fica abaixo de e_bp10 + tta (0.925) em val. **A receita de eggs continua bp10 sem aug.**

### 3.10 Lei de escala `disc_sl_*` (02/11, a partir das 17h)

- **Único ponto com κ:** `sl_w0125_e60` (4/6/8/160 filtros, N ≈ 3.558), larvae gem L4 **0.853 ± 0.009**, contra 0.903 da base_e60 (−0.050) (CSV; SL).
- O resto foi interrompido (lista na seção 6).

### 3.11 Poucos dados `disc_ld_*`

- As 7 famílias foram planejadas na fila do A19 (DEC "baixo dado"): ipc, c400, sp200, saug, bp003, e1000, noback.
- **Nenhuma foi lançada** (Jaci, relatório final 02/11). O que chegou a rodar foram as evals do protocolo justo `ssl100` (§4.13).

### 3.12 final_rep / final_nobg (Jaci)

- `disc_final_rep` repete o backbone. `disc_final_nobg` testa a receita sem fundo completa (impurities true + paper_classes false + bg 1.0).
- **As duas foram interrompidas** com eggs s1/s2 até L3 e **sem SVM** (Jaci, relatório final 02/11).

---

## 4. Todas as leis

### 4.1 Backprop (gem L4; com pw entre parênteses)

| bp | eggs | larvae | protozoan |
|---|---|---|---|
| 0 (fg01, L3 flat) | 0.893 | 0.835 | 0.855 |
| 0.1 | 0.899 (0.942) | 0.910 (0.961) | 0.875 (0.926) |
| 0.3 | 0.905 | **0.933** (0.951) | e60: (0.914 contra 0.907) |
| 1.0 | 0.908 (**0.949**) | 0.914 | — |

Em eggs a lei é monotônica e fraca. Em larvae tem pico em 0.3, que some com o pw. Só o bp10 em eggs sobrevive à leitura final (DEC).

### 4.2 Kernel (eggs, 200 épocas): k3 0.875 → k5 ~0.900 → k7 0.902. Ganha de 3 para 5 e satura.

### 4.3 Largura

| eixo | pontos | κ |
|---|---|---|
| eggs, largura das convs | 32/48/64 → 48/64/96 → 64/96/128 | 0.899 → 0.901 → 0.900 (plana) |
| protozoan, w64 (flat, R1) | 24/32/48 → 32/48/64 | 0.848 → 0.860 |
| larvae, escala m (e60) | m=1/8 → m=1 | 0.853 → 0.903 |
| 1×1 1280 → 2560 | — | 2560 abandonada (>2000-d) |

### 4.4 Profundidade

| | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|
| eggs e60 (gem, última camada) | — | 0.875 | 0.857 | 0.851 | — |
| R2 eggs (flat, por camada) | L3 0.894 | L4 0.873 | L5 0.857 | L6 0.854 | L8 0.817 |

Cai a cada camada além da L4. Exceção: a **1×1 sobre a L3** sobe +0.04 no avg (c1280).

### 4.5 Campo receptivo, resolução e dilation (eggs, e60)

| variação | κ | Δ |
|---|---|---|
| dil2 | 0.868 | −0.007 |
| res49 | 0.847 | −0.028 |
| k3 (200 épocas) | 0.875 | −0.026 |

O RF de ~40 px do k5 basta. Diminuir perde; aumentar não ganha.

### 4.6 Âncora L2-SP

| λ | eggs | larvae |
|---|---|---|
| 0 | 0.899 | 0.910 |
| 0.3 | 0.813 | — |
| 3 | 0.829 | 0.865 |

Qualquer âncora derruba o κ. O SPiFiL é ponto de partida, não restrição.

### 4.7 BN / SE / GRN

| | eggs | larvae |
|---|---|---|
| BN | 0.897 (neutro) | 0.887 (−0.023) |
| SE (e60) | 0.871 vs 0.875 | 0.909 vs 0.903 |
| GRN | 0.898 | 0.911 |

Ficam todos fora da receita.

### 4.8 Seeds e SPiFiL contra aleatório (L3, 3 seeds × 3 splits)

| | eggs | larvae | protozoan |
|---|---|---|---|
| SPiFiL (fg01) | 0.892 | 0.835 | 0.855 |
| aleatório (média de seeds) | ~0.900 | ~0.823 | 0.809 |
| Δ | −0.008 | +0.013 | **+0.046** |

O aleatório varia 0.012–0.022 entre seeds; o SPiFiL repetido varia ≤ 0.008. Na curva de loss, o aleatório desce mais em 16 de 21 estágios: **a loss não vê a vantagem do SPiFiL**.

### 4.9 Impureza (`paper_classes: false`): eggs 0.847 contra 0.875 (−0.028); protozoan 0.859 contra 0.861. Dar semente à impureza não ajuda.

### 4.10 Fundo, máscara, nomask, mask50

| variação | eggs | larvae | protozoan |
|---|---|---|---|
| peso do fundo 1.0 / 0.1 / 0 (L3) | empate | empate | empate |
| bg1_e60 (gem) | 0.873 | 0.900 | 0.856 |
| nomask_e60 (gem) | 0.882 | 0.903 | 0.852 |
| mask50_e60 (gem) | 0.875 | 0.903 | 0.842 |
| base_e60 | 0.875 | 0.903 | 0.861 |
| gem_fg (só objeto na leitura) | 0.899 | 0.900 | 0.871 |

O treino apaga o contraste objeto/fundo do SPiFiL (L3 de 2.9 para 1.0, act_drift A8), e mesmo assim o κ não cai. O fundo é indiferente no treino e na leitura.

### 4.11 Orçamento de épocas (gem; pw entre parênteses)

| épocas | eggs | larvae | protozoan |
|---|---|---|---|
| 60 | 0.875 | 0.903 (0.956) | 0.861 (0.907) |
| 100 | 0.873 | — | — |
| 200 | ~0.900 (0.942) | 0.910 (0.961) | 0.875 (0.926) |
| 400 | interrompida | — | — |

Nas L1–L3, o best cai na época 193–198 de 200 (ES): o cosine ainda está baixando a loss. Com patience 20 se economiza só 4%; com 60 épocas, ~69%. **60 épocas só servem para larvae.**

### 4.12 Leitura (avg → gem → tta → pw) e decomposição

- Teste, mesmo encoder: eggs 0.880 → 0.891 → 0.908 → **0.933**; larvae 0.928 → 0.939 → 0.952 → **0.968**; protozoan 0.845 → 0.886 → 0.921 → **0.928**.
- Ganho sobre o gem na val (só pw / só TTA / os dois): eggs +0.029/+0.013/+0.043; larvae +0.010/+0.036/+0.051; protozoan +0.010/+0.043/+0.052. Os dois ganhos se somam.
- O whitening domina em eggs (detalhe fino). O TTA domina em larvae e protozoan (objetos sem orientação certa).
- O TTA corta impureza → ovo de 137 para 111 (DEC "o gem_tta corta").

### 4.13 Curva de dados 1–100% (teste, gem_tta_pw; Jaci, relatado)

| pct | eggs | eggs sem pw | IJ eggs | larvae | sem pw | IJ | protozoan | sem pw | IJ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.268 | 0.186 | 0.661 | 0.606 | 0.564 | 0.822 | 0.486 | 0.428 | 0.586 |
| 5 | 0.676 | 0.621 | 0.864 | 0.821 | 0.766 | 0.881 | 0.728 | 0.687 | 0.725 |
| 25 | 0.889 | 0.841 | 0.917 | 0.930 | 0.926 | 0.919 | 0.898 | 0.875 | 0.831 |
| 50 | 0.933 | 0.908 | 0.940 | 0.968 | 0.952 | 0.943 | 0.928 | 0.921 | 0.864 |
| 75 | 0.960 | 0.941 | 0.945 | 0.979 | 0.974 | 0.941 | 0.942 | 0.933 | 0.882 |
| 100 | 0.967 | 0.946 | 0.957 | 0.981 | 0.971 | 0.950 | 0.954* | 0.947 | 0.892 |

\* protozoan 100% tem 2 splits.

- **O whitening ganha nos 18 pontos.** Logo, a hipótese "o whitening colapsa com poucos dados" está refutada.
- Passamos o I-JEPA com ≥25% em larvae/protozoan e com ≥75% em eggs. Perdemos com 1–5%.
- `pcts_val_gemttapwaug8` é **parcial** (13 pontos; eggs 0.468, larvae 0.699 no pct baixo) (Jaci, relatório final 02/11).

**Poucos dados e protocolo justo.**
- O I-JEPA é `ijepa_vith14_1k` **congelado**, pré-treinado no ImageNet-1k. O `pretrained_pct` só escolhe os rótulos do SVM (`eval/svm_variants/svm_ijepa.py:275-286`; LDD §3).
- Com 1%, o nosso autoencoder vê 24/17/46 imagens e faz ~200 passos, 30–40× menos que com 50% (LDD).
- **Protocolo ssl100:** o encoder treina em 100% das imagens sem rótulo, e o SVM recebe só os rótulos do pct. Jaci, relatado, gem_tta_pw, val | teste:

| | eggs | larvae | protozoan |
|---|---|---|---|
| svm1 | 0.392 \| 0.393 | 0.733 \| 0.775 | 0.561 \| 0.542 |
| svm5 | 0.823 \| 0.812 | 0.880 \| 0.908 | 0.823 \| 0.818 |
| I-JEPA 1% / 5% (teste) | 0.661 / 0.864 | 0.822 / 0.881 | 0.586 / 0.725 |

- Contra o pct1 normal, o ssl100 dá +0.125 / +0.169 / +0.056 com 1%.
- Com 5%, o ssl100 passa o I-JEPA em larvae (0.908 contra 0.881) e em protozoan (0.818 contra 0.725), mas não em eggs.
- O whitening fica preso em n−1 dimensões (24/17/46 com 1%). O Ledoit-Wolf (`pwlw`) **não rodou**.

### 4.14 A leitura absorve os ganhos de treino

| eixo | Δ com gem | Δ com gem_tta_pw |
|---|---|---|
| l_bp03 | +0.023 | −0.010 |
| l_aug | +0.009 | −0.004 |
| e_aug | +0.004 | −0.005 |
| e_bp10 | +0.009 | **+0.007** (o único que sobra) |
| p_bp03 (e60) | — | +0.007 |

### 4.15 Ruído do pipeline

- Mesmo config retreinado: ±0.02 por split e ~0.003 na média de 3 splits (e_base contra base).
- O avg é ~3× mais ruidoso que o flatten (\|Δ\| mediano 0.032 contra 0.011).
- Larvae chega a ±0.05 num split (223 casos na classe minoritária).
- **O pipeline não é bit-reprodutível:** a época do best muda (154 contra 184), e o SPiFiL da camada seguinte sai diferente.

---

## 5. Insights

1. **A leitura vale mais que a rede.** Com o mesmo encoder, eggs vai de 0.880 para 0.933 no teste.
2. **O SPiFiL é ponto de partida, não restrição.** Ele ganha onde o aleatório vai mal (protozoan +0.046). Prender os pesos nele custa até −0.09.
3. **O SPiFiL deixa o treino reprodutível.** A repetição varia ≤ 0.008, contra 0.012–0.022 do aleatório.
4. **A loss de reconstrução não prevê o κ** entre inicializadores diferentes. Só prevê dentro do mesmo encoder com mais backprop.
5. **O treino apaga o contraste objeto/fundo** do SPiFiL, e isso não custa κ.
6. **99% dos erros de eggs são ovo ↔ impureza.** A impureza não tem semente, e dar semente a ela piora o κ.
7. **O TTA corta as impurezas que pareciam ovo** porque a semelhança vale só numa pose. A média das 8 vistas desfaz isso.
8. **O whitening só funciona com shrinkage.** Sem ele, o SVM colapsa (larvae κ ~0).
9. **Whitening e TTA se somam** e cobrem erros diferentes: o whitening pega o detalhe fino, o TTA pega a orientação.
10. **A leitura forte absorve aug e bp03.** Só o bp10 em eggs sobrevive.
11. **Platô de construção:** ~25 eixos sem ganho. Só k3, âncora e BN se mexem, e para baixo.
12. **A profundidade cai depois da L4**, e o RF de ~40 px basta. A 1×1 larga é a exceção que ajuda.
13. **Eggs precisa de treino longo** (o detalhe fino vem na cauda do cosine). Larvae se contenta com 60 épocas.
14. **O fundo e a máscara são indiferentes** no treino, nas sementes e na leitura. A receita deve rodar em dataset sem fundo (falta confirmar com final_nobg).
15. **Decidir só pela média de 3 splits.** O pipeline não repete bit a bit.
16. **Com poucos dados, o limite é de dados, não de leitura.** O I-JEPA tem o prior do ImageNet; pré-treinar sem rótulo em 100% fecha parte do buraco.
17. **Encolher 8× a largura custa 0.05 em larvae** (único ponto da lei de escala).
18. **A comparação justa ainda não foi feita:** falta o I-JEPA com TTA + whitening.

---

## 6. O que foi interrompido e o que falta

Tudo parou em 02/11, ~23h UTC. Nada foi apagado. Fonte: Jaci, relatório final 02/11, mais o `ls` dos ckpts na Athena.

**Interrompido na Athena, sem SVM:**
- e_aug_wh, e_aug_pn, e_aug_div, e_aug_knn, e_aug_ipc3 (2 splits completos cada);
- e_base_e400 (2 splits);
- nd_e200 (nenhum ckpt);
- p_aug200 (2 splits);
- p_bp10_aug (1 split);
- sl_c160_e200, sl_c320_e200 (eggs, 1 split cada);
- sl_c320_e60 (larvae 3/3, protozoan parcial);
- sl_d3_e200 (eggs s1);
- sl_w025_e200 (eggs, 2 splits);
- sl_w050_e200 (eggs, 2 splits);
- o último ponto da curva (protozoan s3 pct100).

Treinadas e sem SVM: sl_w025_e60 e sl_w050_e60 (larvae 3/3, protozoan parcial), e_bp10_aug e l_bp10_aug (os SVMs delas rodaram na Jaci). **Ficaram ~70 SVMs pendentes na Athena.**

**Interrompido na Jaci, sem SVM:**
- final_rep e final_nobg: eggs s1/s2 até L3;
- sl_w200_e200: eggs s1 L1–L2;
- sl_w200_e60: larvae s1 e s2 completos;
- sl_c1920_e200: eggs s1 L1–L3;
- sl_c1920_e60, sl_c640_e60, sl_c160_e60, sl_d3_e60: larvae s1–s3 completos;
- sl_c640_e200: eggs s1 L1–L3;
- sl_d5_e200: eggs s1 L1–L2;
- p_w96_e60: s1 L1–L3.

**Não rodou:**
- pwlw (pcts e ssl100);
- o resto do aug8;
- bg1 no teste;
- os extras do l_bp03;
- l_w96;
- o gem_tta_pw do sl_w0125_e60 (só existe o gem da val);
- bp01w64c96 test_avg;
- as 7 `disc_ld_*`.

Na Jaci há 28 CSVs em `A_reports/november/jaci/`.

**Lei de escala:** só tem 1 ponto (w0125_e60 larvae 0.853 contra 0.903). Faltam todos os outros.

**Próximos experimentos, por prioridade:**
1. Rodar os SVMs pendentes do que já está treinado: as sl_* com larvae 3/3, e_aug_{wh,pn,div,knn,ipc3}, p_aug200 e e400. Usar val gem, val gem_tta_pw e test gem_tta_pw, um por vez.
2. Rodar o I-JEPA com TTA + whitening (régua justa).
3. Terminar final_rep e final_nobg (reprodutibilidade e dataset sem fundo).
4. Terminar a lei de escala (w, c, d) e ajustar e(N) = E + A·N^−α com ≥ 3 pontos, controlando pela fração SPiFiL.
5. Poucos dados: pwlw no ssl100, depois ld_e1000, ld_noback, ld_saug, ld_c400 e o resto.
6. Eggs: cauda do cosine (100 épocas + eta_min). Precisa de uma opção nova no scheduler.
7. Ortogonalidade, LCN e ND: só se as opções de seleção de filtros ganharem.

---

## 7. Infraestrutura

- **Athena:** 4 GPUs de 49 GB, teto de 20 GB por GPU, 96 núcleos. O gargalo era a CPU do SVM (load 245 em 30/10).
- **Jaci:** 1 RTX PRO 6000 de 96 GB, 128 núcleos. Teto de 70% = 68.5 GB, com 12 runs concorrentes a ~62–66 GB. O sm da GPU ficou em 100% (saturado). O dataloader ficou ocioso (2 workers bastam) e o load em ~20/128 (Jaci, relatado).
- **Repositório:** worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, branch `discovery` (antes `discovery_hybrid_spifil`). A Jaci recebe os YAMLs via sync; os CSVs da Jaci ainda não foram trazidos.
- **Agentes:** A0 coordena; A1–A3 fazem as frentes iniciais; A4 cuida de curvas, κ e diagnósticos; A5/A5b fazem a literatura; A6 é o guardião do teto de GPU e da fila (`gpu_queue.txt`); A7 implementa (SE, GRN, âncora); A8 mede act_drift; A12 auditou a receita; A15 desenhou a lei de escala; A17–A19 cuidam de poucos dados; há ainda um agente na Jaci.

**Armadilhas encontradas:**
- **Contexto fantasma em `cuda:0`:** processos criavam contexto CUDA na GPU 0 mesmo quando iam rodar em outra. Corrigido com `torch.cuda.set_device`.
- **flock sem FIFO:** a trava do SVM não garante a ordem de chegada, então a fila de SVM não sai na ordem enfileirada.
- **gpu_ids:** um YAML com `gpu_ids` inline (`nd_e200`) caiu na GPU 0 da Athena. Na Jaci os YAMLs precisam de `gpu_ids: [0]`.
- **`artifacts/` é symlink na Jaci:** sem `-k`, o rsync ignora a árvore inteira.
- **PCA sem shrinkage colapsa:** os CSVs `_gemttapw` antigos são inválidos.
- **A estimativa de memória das c\* estava errada.** O valor corrigido é ~3.6 GB, e o sl_d3 mediu 4.85 GB contra 2.5 GB estimados.
- **Duas vezes o teto foi passado:**
  - Athena: 20.03 GB às 18:46. O A6 pausou o sl_c160_e60.
  - Jaci: 69.3 GB às 20:11. Corrigido com um gate a 64 GB.
- **`skip: state` não sobrevive a kill:** o relançamento das 18:24 de 30/10 refez pontos.
- **`best_val_svm_kappa` guarda uma loss**, e o `freeze_encoder_flag` não congela a L1 (REC).
- **Dataset novo dá KeyError** em `flim/spifil.py:461`.

## 8. Como reproduzir

```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/growth_disc_bp01w64c1280.yaml
```

```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/eval_svm_disc_bp01w64c1280_val_gemttapwr.yaml && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/eval_svm_disc_bp01w64c1280_test_gemttapwr.yaml
```

- **Eggs:** use `growth_disc_g20_e_bp10.yaml` + `eval_svm_disc_g20_e_bp10_{val,test}_gemttapwr.yaml`.
- **Triagem com 60 épocas** (só larvae): `growth_disc_g20_base_e60.yaml`.
- **Checklist completo:** REC §6.

## 9. Mapa de arquivos

- **Relatórios:**
  - `A_reports/october/`: decisions, recipe, ranking_test, eggs_diag, early_stop, literature_ideas, literature_round2, conv_techniques, act_drift, insights_board, scaling_law_prompt;
  - `A_reports/november/`: report_discovery_athena, low_data_diag, low_data_literature, scaling_law(.md/.png);
  - na Jaci: `A_reports/november/jaci/` (28 CSVs).
- **YAMLs:** `experiments/autoencoder/growth_disc_*.yaml` e `eval_svm_disc_*.yaml` (578 arquivos `*disc_*`). Os do protocolo ssl100 são `eval_svm_disc_bp01w64c1280_ssl100_svm{1,5}_*`.
- **Código:**
  - `eval/svm.py`: leituras gem, spp, gem_dense, *_fg, gem_tta, sufixos `_pw`/`_pwr`;
  - `flim/spifil.py`: fill_random_last, whiten, patch_norm, select_diverse, knn_smooth, images_per_class;
  - runner growth: layerwise_backprop, layer_kernels, layer_pools;
  - `methods/autoencoder/autoencoder_flim_module.py`: recon_bg_weight, se_reduction, grn, spifil_anchor, feat_whiten.
- **Scripts de análise:** `scratchpad/a4/` (eggs_diag, es_sim, scaling_law.py) e `ranking.py`.
- **Artefatos:**
  - Athena: `/dados/home/moliveira/artifacts/spifil_growth/<família>/<ds>_split<s>_pct<p>/L<k>/checkpoints/best_kappa.ckpt`, mais `act_drift/`;
  - Jaci: a mesma árvore, sob o symlink `artifacts/spifil_growth`;
  - resultados: `results/svm_disc_*.csv` (220 arquivos na Athena).

---

## Apêndice A: agregados de todos os `results/svm_disc_*.csv` (Athena)

- Só linhas com `status == ok`. Média ± desvio padrão entre splits; n = número de linhas.
- O nome do arquivo já traz a família, o eval_set (val/test) e a leitura. Sem sufixo, a leitura é a padrão do YAML (flatten nas famílias antigas, avg nas c1280).
- Os arquivos `_gemttapw` (sem o r) são o colapso sem shrinkage e ficam como histórico. Nos novos, `_pwr` é igual a `_pw`.

| arquivo (results/svm_*.csv) | camada | ds | pct | κ | acc | f1 | n |
|---|---|---|---|---|---|---|---|
| disc_a1_fg01_val | L1 | eggs | 50 | 0.887 ± 0.012 | 0.926 ± 0.012 | 0.904 ± 0.015 | 3 |
| disc_a1_fg01_val | L2 | eggs | 50 | 0.895 ± 0.007 | 0.931 ± 0.010 | 0.909 ± 0.006 | 3 |
| disc_a1_fg01_val | L3 | eggs | 50 | 0.897 ± 0.008 | 0.930 ± 0.014 | 0.909 ± 0.011 | 5 |
| disc_a1_fg01_val | L3 | larvae | 50 | 0.836 ± 0.005 | 0.917 ± 0.008 | 0.918 ± 0.002 | 3 |
| disc_a1_fg01_val | L3 | protozoan | 50 | 0.855 ± 0.021 | 0.856 ± 0.027 | 0.846 ± 0.026 | 3 |
| disc_a2_bp01_val | L3 | eggs | 50 | 0.902 ± 0.008 | 0.935 ± 0.017 | 0.915 ± 0.010 | 3 |
| disc_a2_bp01_val | L3 | larvae | 50 | 0.829 ± 0.012 | 0.917 ± 0.006 | 0.914 ± 0.006 | 3 |
| disc_a2_bp01_val | L3 | protozoan | 50 | 0.866 ± 0.015 | 0.863 ± 0.017 | 0.851 ± 0.009 | 3 |
| disc_a2_w64_val | L3 | eggs | 50 | 0.896 ± 0.015 | 0.934 ± 0.019 | 0.911 ± 0.020 | 3 |
| disc_a2_w64_val | L3 | larvae | 50 | 0.823 ± 0.031 | 0.914 ± 0.020 | 0.912 ± 0.015 | 3 |
| disc_a2_w64_val | L3 | protozoan | 50 | 0.860 ± 0.008 | 0.859 ± 0.018 | 0.850 ± 0.015 | 3 |
| disc_a3_popz_val | L1 | eggs | 50 | 0.895 ± 0.007 | 0.933 ± 0.009 | 0.912 ± 0.011 | 3 |
| disc_a3_popz_val | L1 | larvae | 50 | 0.763 ± 0.032 | 0.877 ± 0.024 | 0.882 ± 0.016 | 3 |
| disc_a3_popz_val | L1 | protozoan | 50 | 0.769 ± 0.014 | 0.798 ± 0.021 | 0.782 ± 0.017 | 3 |
| disc_a3_popz_val | L2 | eggs | 50 | 0.894 ± 0.021 | 0.931 ± 0.023 | 0.907 ± 0.022 | 3 |
| disc_a3_popz_val | L2 | larvae | 50 | 0.796 ± 0.022 | 0.900 ± 0.021 | 0.898 ± 0.011 | 3 |
| disc_a3_popz_val | L2 | protozoan | 50 | 0.829 ± 0.027 | 0.836 ± 0.029 | 0.829 ± 0.026 | 3 |
| disc_a3_popz_val | L3 | eggs | 50 | 0.887 ± 0.016 | 0.925 ± 0.013 | 0.905 ± 0.021 | 3 |
| disc_a3_popz_val | L3 | larvae | 50 | 0.821 ± 0.018 | 0.915 ± 0.017 | 0.910 ± 0.009 | 3 |
| disc_a3_popz_val | L3 | protozoan | 50 | 0.844 ± 0.016 | 0.853 ± 0.019 | 0.838 ± 0.014 | 3 |
| disc_a3_sp200_val | L3 | eggs | 50 | 0.888 ± 0.004 | 0.922 ± 0.011 | 0.904 ± 0.008 | 3 |
| disc_a3_sp200_val | L3 | larvae | 50 | 0.837 ± 0.058 | 0.920 ± 0.037 | 0.919 ± 0.029 | 3 |
| disc_a3_sp200_val | L3 | protozoan | 50 | 0.852 ± 0.003 | 0.855 ± 0.020 | 0.845 ± 0.008 | 3 |
| disc_bp01c1280_test | L3 | eggs | 50 | 0.832 ± 0.027 | 0.831 ± 0.015 | 0.838 ± 0.019 | 3 |
| disc_bp01c1280_test | L3 | larvae | 50 | 0.915 ± 0.012 | 0.960 ± 0.013 | 0.957 ± 0.006 | 3 |
| disc_bp01c1280_test | L3 | protozoan | 50 | 0.752 ± 0.031 | 0.701 ± 0.041 | 0.748 ± 0.047 | 3 |
| disc_bp01c1280_test | L4 | eggs | 50 | 0.875 ± 0.007 | 0.890 ± 0.008 | 0.882 ± 0.004 | 3 |
| disc_bp01c1280_test | L4 | larvae | 50 | 0.928 ± 0.004 | 0.966 ± 0.012 | 0.964 ± 0.002 | 3 |
| disc_bp01c1280_test | L4 | protozoan | 50 | 0.836 ± 0.021 | 0.798 ± 0.022 | 0.832 ± 0.020 | 3 |
| disc_bp01c1280_val | L3 | eggs | 50 | 0.819 ± 0.013 | 0.816 ± 0.015 | 0.832 ± 0.013 | 3 |
| disc_bp01c1280_val | L3 | larvae | 50 | 0.899 ± 0.022 | 0.956 ± 0.013 | 0.950 ± 0.011 | 3 |
| disc_bp01c1280_val | L3 | protozoan | 50 | 0.750 ± 0.053 | 0.690 ± 0.040 | 0.737 ± 0.045 | 3 |
| disc_bp01c1280_val | L4 | eggs | 50 | 0.868 ± 0.003 | 0.877 ± 0.004 | 0.884 ± 0.010 | 3 |
| disc_bp01c1280_val | L4 | larvae | 50 | 0.912 ± 0.021 | 0.958 ± 0.014 | 0.956 ± 0.011 | 3 |
| disc_bp01c1280_val | L4 | protozoan | 50 | 0.828 ± 0.033 | 0.781 ± 0.024 | 0.813 ± 0.020 | 3 |
| disc_bp01w64_test_avg | L3 | eggs | 50 | 0.848 ± 0.032 | 0.839 ± 0.031 | 0.847 ± 0.028 | 3 |
| disc_bp01w64_test_avg | L3 | larvae | 50 | 0.914 ± 0.015 | 0.958 ± 0.011 | 0.957 ± 0.007 | 3 |
| disc_bp01w64_test_avg | L3 | protozoan | 50 | 0.797 ± 0.008 | 0.766 ± 0.013 | 0.801 ± 0.013 | 3 |
| disc_bp01w64_test_gem | L3 | eggs | 50 | 0.824 ± 0.027 | 0.867 ± 0.041 | 0.843 ± 0.026 | 3 |
| disc_bp01w64_test_gem | L3 | larvae | 50 | 0.891 ± 0.006 | 0.947 ± 0.010 | 0.945 ± 0.003 | 3 |
| disc_bp01w64_test_gem | L3 | protozoan | 50 | 0.809 ± 0.013 | 0.827 ± 0.015 | 0.819 ± 0.004 | 3 |
| disc_bp01w64_test_spp | L3 | eggs | 50 | 0.873 ± 0.034 | 0.900 ± 0.037 | 0.878 ± 0.036 | 3 |
| disc_bp01w64_test_spp | L3 | larvae | 50 | 0.886 ± 0.018 | 0.949 ± 0.009 | 0.943 ± 0.009 | 3 |
| disc_bp01w64_test_spp | L3 | protozoan | 50 | 0.832 ± 0.007 | 0.854 ± 0.014 | 0.834 ± 0.006 | 3 |
| disc_bp01w64_val | L3 | eggs | 50 | 0.906 ± 0.020 | 0.935 ± 0.026 | 0.918 ± 0.023 | 3 |
| disc_bp01w64_val | L3 | larvae | 50 | 0.845 ± 0.022 | 0.923 ± 0.015 | 0.922 ± 0.011 | 3 |
| disc_bp01w64_val | L3 | protozoan | 50 | 0.865 ± 0.020 | 0.871 ± 0.015 | 0.860 ± 0.017 | 3 |
| disc_bp01w64_val_avg | L3 | eggs | 50 | 0.845 ± 0.041 | 0.840 ± 0.054 | 0.848 ± 0.049 | 3 |
| disc_bp01w64_val_avg | L3 | larvae | 50 | 0.904 ± 0.032 | 0.958 ± 0.016 | 0.952 ± 0.016 | 3 |
| disc_bp01w64_val_avg | L3 | protozoan | 50 | 0.797 ± 0.015 | 0.759 ± 0.033 | 0.794 ± 0.019 | 3 |
| disc_bp01w64_val_gem | L3 | eggs | 50 | 0.811 ± 0.028 | 0.848 ± 0.043 | 0.827 ± 0.035 | 3 |
| disc_bp01w64_val_gem | L3 | larvae | 50 | 0.831 ± 0.031 | 0.927 ± 0.012 | 0.916 ± 0.015 | 3 |
| disc_bp01w64_val_gem | L3 | protozoan | 50 | 0.798 ± 0.021 | 0.812 ± 0.015 | 0.802 ± 0.021 | 3 |
| disc_bp01w64_val_spp | L3 | eggs | 50 | 0.878 ± 0.020 | 0.903 ± 0.027 | 0.883 ± 0.027 | 3 |
| disc_bp01w64_val_spp | L3 | larvae | 50 | 0.872 ± 0.010 | 0.948 ± 0.013 | 0.936 ± 0.005 | 3 |
| disc_bp01w64_val_spp | L3 | protozoan | 50 | 0.829 ± 0.016 | 0.844 ± 0.013 | 0.825 ± 0.007 | 3 |
| disc_bp01w64c1280_test | L3 | eggs | 50 | 0.843 ± 0.019 | 0.841 ± 0.018 | 0.844 ± 0.022 | 3 |
| disc_bp01w64c1280_test | L3 | larvae | 50 | 0.914 ± 0.018 | 0.958 ± 0.013 | 0.957 ± 0.009 | 3 |
| disc_bp01w64c1280_test | L3 | protozoan | 50 | 0.788 ± 0.023 | 0.761 ± 0.013 | 0.795 ± 0.014 | 3 |
| disc_bp01w64c1280_test | L4 | eggs | 50 | 0.880 ± 0.013 | 0.894 ± 0.016 | 0.886 ± 0.014 | 3 |
| disc_bp01w64c1280_test | L4 | larvae | 50 | 0.928 ± 0.023 | 0.966 ± 0.005 | 0.964 ± 0.011 | 3 |
| disc_bp01w64c1280_test | L4 | protozoan | 50 | 0.845 ± 0.014 | 0.820 ± 0.018 | 0.847 ± 0.011 | 3 |
| disc_bp01w64c1280_test_avg | L4 | eggs | 50 | 0.880 ± 0.013 | 0.894 ± 0.016 | 0.886 ± 0.014 | 3 |
| disc_bp01w64c1280_test_avg | L4 | larvae | 50 | 0.928 ± 0.023 | 0.966 ± 0.005 | 0.964 ± 0.011 | 3 |
| disc_bp01w64c1280_test_avg | L4 | protozoan | 50 | 0.845 ± 0.014 | 0.820 ± 0.018 | 0.847 ± 0.011 | 3 |
| disc_bp01w64c1280_test_avgfg | L3 | eggs | 50 | 0.833 ± 0.020 | 0.882 ± 0.016 | 0.854 ± 0.019 | 3 |
| disc_bp01w64c1280_test_avgfg | L3 | larvae | 50 | 0.887 ± 0.027 | 0.945 ± 0.024 | 0.944 ± 0.013 | 3 |
| disc_bp01w64c1280_test_avgfg | L3 | protozoan | 50 | 0.776 ± 0.028 | 0.790 ± 0.031 | 0.784 ± 0.029 | 3 |
| disc_bp01w64c1280_test_avgfg | L4 | eggs | 50 | 0.874 ± 0.014 | 0.906 ± 0.024 | 0.892 ± 0.015 | 3 |
| disc_bp01w64c1280_test_avgfg | L4 | larvae | 50 | 0.922 ± 0.024 | 0.965 ± 0.010 | 0.961 ± 0.012 | 3 |
| disc_bp01w64c1280_test_avgfg | L4 | protozoan | 50 | 0.856 ± 0.016 | 0.867 ± 0.019 | 0.849 ± 0.021 | 3 |
| disc_bp01w64c1280_test_gem | L4 | eggs | 50 | 0.891 ± 0.023 | 0.915 ± 0.030 | 0.900 ± 0.023 | 3 |
| disc_bp01w64c1280_test_gem | L4 | larvae | 50 | 0.939 ± 0.011 | 0.973 ± 0.005 | 0.969 ± 0.006 | 3 |
| disc_bp01w64c1280_test_gem | L4 | protozoan | 50 | 0.886 ± 0.011 | 0.886 ± 0.013 | 0.874 ± 0.013 | 3 |
| disc_bp01w64c1280_test_gemdense | L4 | eggs | 50 | 0.891 ± 0.024 | 0.915 ± 0.029 | 0.900 ± 0.024 | 3 |
| disc_bp01w64c1280_test_gemdense | L4 | larvae | 50 | 0.932 ± 0.019 | 0.972 ± 0.008 | 0.966 ± 0.009 | 3 |
| disc_bp01w64c1280_test_gemdense | L4 | protozoan | 50 | 0.882 ± 0.012 | 0.877 ± 0.013 | 0.864 ± 0.019 | 3 |
| disc_bp01w64c1280_test_gemfg | L4 | eggs | 50 | 0.893 ± 0.018 | 0.911 ± 0.033 | 0.906 ± 0.020 | 3 |
| disc_bp01w64c1280_test_gemfg | L4 | larvae | 50 | 0.936 ± 0.016 | 0.971 ± 0.003 | 0.968 ± 0.008 | 3 |
| disc_bp01w64c1280_test_gemfg | L4 | protozoan | 50 | 0.891 ± 0.013 | 0.885 ± 0.021 | 0.874 ± 0.020 | 3 |
| disc_bp01w64c1280_test_gemtta | L4 | eggs | 50 | 0.908 ± 0.018 | 0.927 ± 0.020 | 0.912 ± 0.017 | 3 |
| disc_bp01w64c1280_test_gemtta | L4 | larvae | 50 | 0.952 ± 0.007 | 0.978 ± 0.004 | 0.976 ± 0.004 | 3 |
| disc_bp01w64c1280_test_gemtta | L4 | protozoan | 50 | 0.921 ± 0.010 | 0.902 ± 0.014 | 0.902 ± 0.010 | 3 |
| disc_bp01w64c1280_test_gemttapwr | L4 | eggs | 50 | 0.933 ± 0.016 | 0.929 ± 0.023 | 0.940 ± 0.016 | 3 |
| disc_bp01w64c1280_test_gemttapwr | L4 | larvae | 50 | 0.968 ± 0.008 | 0.981 ± 0.006 | 0.984 ± 0.004 | 3 |
| disc_bp01w64c1280_test_gemttapwr | L4 | protozoan | 50 | 0.928 ± 0.006 | 0.897 ± 0.013 | 0.912 ± 0.007 | 3 |
| disc_bp01w64c1280_test_l12 | L1 | eggs | 50 | 0.165 ± 0.086 | 0.273 ± 0.076 | 0.274 ± 0.092 | 3 |
| disc_bp01w64c1280_test_l12 | L1 | larvae | 50 | 0.856 ± 0.025 | 0.929 ± 0.019 | 0.928 ± 0.013 | 3 |
| disc_bp01w64c1280_test_l12 | L1 | protozoan | 50 | 0.235 ± 0.023 | 0.316 ± 0.029 | 0.302 ± 0.033 | 3 |
| disc_bp01w64c1280_test_l12 | L2 | eggs | 50 | 0.741 ± 0.034 | 0.707 ± 0.044 | 0.719 ± 0.034 | 3 |
| disc_bp01w64c1280_test_l12 | L2 | larvae | 50 | 0.905 ± 0.004 | 0.954 ± 0.007 | 0.952 ± 0.002 | 3 |
| disc_bp01w64c1280_test_l12 | L2 | protozoan | 50 | 0.583 ± 0.068 | 0.487 ± 0.051 | 0.522 ± 0.066 | 3 |
| disc_bp01w64c1280_val | L3 | eggs | 50 | 0.842 ± 0.017 | 0.835 ± 0.011 | 0.850 ± 0.013 | 3 |
| disc_bp01w64c1280_val | L3 | larvae | 50 | 0.900 ± 0.015 | 0.952 ± 0.009 | 0.950 ± 0.007 | 3 |
| disc_bp01w64c1280_val | L3 | protozoan | 50 | 0.786 ± 0.025 | 0.749 ± 0.006 | 0.787 ± 0.011 | 3 |
| disc_bp01w64c1280_val | L4 | eggs | 50 | 0.880 ± 0.016 | 0.894 ± 0.015 | 0.891 ± 0.014 | 3 |
| disc_bp01w64c1280_val | L4 | larvae | 50 | 0.917 ± 0.016 | 0.964 ± 0.011 | 0.959 ± 0.008 | 3 |
| disc_bp01w64c1280_val | L4 | protozoan | 50 | 0.837 ± 0.015 | 0.806 ± 0.013 | 0.834 ± 0.007 | 3 |
| disc_bp01w64c1280_val_avgfg | L3 | eggs | 50 | 0.831 ± 0.021 | 0.890 ± 0.006 | 0.855 ± 0.022 | 3 |
| disc_bp01w64c1280_val_avgfg | L3 | larvae | 50 | 0.850 ± 0.026 | 0.921 ± 0.028 | 0.925 ± 0.013 | 3 |
| disc_bp01w64c1280_val_avgfg | L3 | protozoan | 50 | 0.768 ± 0.026 | 0.786 ± 0.035 | 0.775 ± 0.024 | 3 |
| disc_bp01w64c1280_val_avgfg | L4 | eggs | 50 | 0.884 ± 0.014 | 0.912 ± 0.005 | 0.899 ± 0.016 | 3 |
| disc_bp01w64c1280_val_avgfg | L4 | larvae | 50 | 0.904 ± 0.012 | 0.948 ± 0.011 | 0.952 ± 0.006 | 3 |
| disc_bp01w64c1280_val_avgfg | L4 | protozoan | 50 | 0.846 ± 0.011 | 0.853 ± 0.014 | 0.836 ± 0.009 | 3 |
| disc_bp01w64c1280_val_gem | L4 | eggs | 50 | 0.899 ± 0.009 | 0.930 ± 0.003 | 0.914 ± 0.009 | 3 |
| disc_bp01w64c1280_val_gem | L4 | larvae | 50 | 0.910 ± 0.012 | 0.963 ± 0.007 | 0.955 ± 0.006 | 3 |
| disc_bp01w64c1280_val_gem | L4 | protozoan | 50 | 0.875 ± 0.002 | 0.881 ± 0.009 | 0.862 ± 0.005 | 3 |
| disc_bp01w64c1280_val_gemdense | L4 | eggs | 50 | 0.905 ± 0.017 | 0.931 ± 0.008 | 0.918 ± 0.017 | 3 |
| disc_bp01w64c1280_val_gemdense | L4 | larvae | 50 | 0.904 ± 0.016 | 0.957 ± 0.018 | 0.952 ± 0.008 | 3 |
| disc_bp01w64c1280_val_gemdense | L4 | protozoan | 50 | 0.871 ± 0.005 | 0.871 ± 0.006 | 0.853 ± 0.006 | 3 |
| disc_bp01w64c1280_val_gemfg | L4 | eggs | 50 | 0.899 ± 0.007 | 0.923 ± 0.008 | 0.912 ± 0.013 | 3 |
| disc_bp01w64c1280_val_gemfg | L4 | larvae | 50 | 0.899 ± 0.010 | 0.958 ± 0.007 | 0.950 ± 0.005 | 3 |
| disc_bp01w64c1280_val_gemfg | L4 | protozoan | 50 | 0.871 ± 0.003 | 0.865 ± 0.013 | 0.849 ± 0.001 | 3 |
| disc_bp01w64c1280_val_gempwr | L4 | eggs | 50 | 0.928 ± 0.011 | 0.926 ± 0.011 | 0.941 ± 0.007 | 3 |
| disc_bp01w64c1280_val_gempwr | L4 | larvae | 50 | 0.920 ± 0.018 | 0.954 ± 0.014 | 0.960 ± 0.009 | 3 |
| disc_bp01w64c1280_val_gempwr | L4 | protozoan | 50 | 0.885 ± 0.003 | 0.850 ± 0.028 | 0.873 ± 0.020 | 3 |
| disc_bp01w64c1280_val_gemtta | L4 | eggs | 50 | 0.912 ± 0.028 | 0.938 ± 0.023 | 0.924 ± 0.027 | 3 |
| disc_bp01w64c1280_val_gemtta | L4 | larvae | 50 | 0.946 ± 0.020 | 0.975 ± 0.014 | 0.973 ± 0.010 | 3 |
| disc_bp01w64c1280_val_gemtta | L4 | protozoan | 50 | 0.917 ± 0.007 | 0.905 ± 0.024 | 0.898 ± 0.008 | 3 |
| disc_bp01w64c1280_val_gemttapw | L4 | eggs | 50 | 0.342 ± 0.102 | 0.270 ± 0.053 | 0.322 ± 0.063 | 3 |
| disc_bp01w64c1280_val_gemttapw | L4 | larvae | 50 | 0.005 ± 0.009 | 0.501 ± 0.003 | 0.469 ± 0.005 | 3 |
| disc_bp01w64c1280_val_gemttapwr | L4 | eggs | 50 | 0.942 ± 0.004 | 0.947 ± 0.013 | 0.950 ± 0.007 | 3 |
| disc_bp01w64c1280_val_gemttapwr | L4 | larvae | 50 | 0.961 ± 0.018 | 0.981 ± 0.011 | 0.980 ± 0.009 | 3 |
| disc_bp01w64c1280_val_gemttapwr | L4 | protozoan | 50 | 0.926 ± 0.008 | 0.906 ± 0.031 | 0.913 ± 0.016 | 3 |
| disc_bp01w64c1280_val_l12 | L1 | eggs | 50 | 0.161 ± 0.097 | 0.268 ± 0.085 | 0.274 ± 0.101 | 3 |
| disc_bp01w64c1280_val_l12 | L1 | larvae | 50 | 0.842 ± 0.036 | 0.931 ± 0.015 | 0.921 ± 0.018 | 3 |
| disc_bp01w64c1280_val_l12 | L1 | protozoan | 50 | 0.233 ± 0.021 | 0.313 ± 0.026 | 0.302 ± 0.028 | 3 |
| disc_bp01w64c1280_val_l12 | L2 | eggs | 50 | 0.730 ± 0.025 | 0.699 ± 0.036 | 0.720 ± 0.029 | 3 |
| disc_bp01w64c1280_val_l12 | L2 | larvae | 50 | 0.902 ± 0.022 | 0.957 ± 0.016 | 0.951 ± 0.011 | 3 |
| disc_bp01w64c1280_val_l12 | L2 | protozoan | 50 | 0.569 ± 0.071 | 0.475 ± 0.054 | 0.510 ± 0.072 | 3 |
| disc_bp01w64c96_val | L4 | eggs | 50 | 0.895 ± 0.004 | 0.927 ± 0.014 | 0.909 ± 0.011 | 3 |
| disc_bp01w64c96_val | L4 | larvae | 50 | 0.844 ± 0.032 | 0.928 ± 0.015 | 0.922 ± 0.016 | 3 |
| disc_bp01w64c96_val | L4 | protozoan | 50 | 0.858 ± 0.012 | 0.864 ± 0.014 | 0.848 ± 0.014 | 3 |
| disc_g20_base_e60_test | L3 | eggs | 50 | 0.829 ± 0.012 | 0.827 ± 0.021 | 0.832 ± 0.015 | 3 |
| disc_g20_base_e60_test | L3 | larvae | 50 | 0.894 ± 0.022 | 0.953 ± 0.009 | 0.947 ± 0.011 | 3 |
| disc_g20_base_e60_test | L3 | protozoan | 50 | 0.782 ± 0.029 | 0.760 ± 0.029 | 0.790 ± 0.019 | 3 |
| disc_g20_base_e60_test | L4 | eggs | 50 | 0.852 ± 0.011 | 0.849 ± 0.020 | 0.861 ± 0.012 | 3 |
| disc_g20_base_e60_test | L4 | larvae | 50 | 0.919 ± 0.013 | 0.961 ± 0.009 | 0.959 ± 0.006 | 3 |
| disc_g20_base_e60_test | L4 | protozoan | 50 | 0.818 ± 0.022 | 0.783 ± 0.024 | 0.812 ± 0.019 | 3 |
| disc_g20_base_e60_test_gem | L4 | eggs | 50 | 0.859 ± 0.015 | 0.894 ± 0.030 | 0.876 ± 0.013 | 3 |
| disc_g20_base_e60_test_gem | L4 | larvae | 50 | 0.918 ± 0.011 | 0.965 ± 0.011 | 0.959 ± 0.006 | 3 |
| disc_g20_base_e60_test_gem | L4 | protozoan | 50 | 0.874 ± 0.016 | 0.867 ± 0.011 | 0.863 ± 0.011 | 3 |
| disc_g20_base_e60_test_gemttapwr | L4 | larvae | 50 | 0.948 ± 0.017 | 0.976 ± 0.010 | 0.974 ± 0.008 | 3 |
| disc_g20_base_e60_test_gemttapwr | L4 | protozoan | 50 | 0.908 ± 0.003 | 0.885 ± 0.013 | 0.893 ± 0.008 | 3 |
| disc_g20_base_e60_val | L3 | eggs | 50 | 0.826 ± 0.010 | 0.822 ± 0.027 | 0.832 ± 0.019 | 3 |
| disc_g20_base_e60_val | L3 | larvae | 50 | 0.881 ± 0.015 | 0.948 ± 0.009 | 0.941 ± 0.008 | 3 |
| disc_g20_base_e60_val | L3 | protozoan | 50 | 0.780 ± 0.040 | 0.765 ± 0.014 | 0.790 ± 0.012 | 3 |
| disc_g20_base_e60_val | L4 | eggs | 50 | 0.830 ± 0.004 | 0.839 ± 0.032 | 0.843 ± 0.020 | 3 |
| disc_g20_base_e60_val | L4 | larvae | 50 | 0.899 ± 0.018 | 0.956 ± 0.015 | 0.950 ± 0.009 | 3 |
| disc_g20_base_e60_val | L4 | protozoan | 50 | 0.815 ± 0.026 | 0.782 ± 0.020 | 0.809 ± 0.016 | 3 |
| disc_g20_base_e60_val_gem | L4 | eggs | 50 | 0.875 ± 0.017 | 0.902 ± 0.035 | 0.884 ± 0.020 | 3 |
| disc_g20_base_e60_val_gem | L4 | larvae | 50 | 0.903 ± 0.001 | 0.955 ± 0.004 | 0.952 ± 0.000 | 3 |
| disc_g20_base_e60_val_gem | L4 | protozoan | 50 | 0.861 ± 0.011 | 0.872 ± 0.004 | 0.853 ± 0.007 | 3 |
| disc_g20_base_e60_val_gem_el | L4 | eggs | 50 | 0.875 ± 0.017 | 0.902 ± 0.035 | 0.884 ± 0.020 | 3 |
| disc_g20_base_e60_val_gem_el | L4 | larvae | 50 | 0.903 ± 0.001 | 0.955 ± 0.004 | 0.952 ± 0.000 | 3 |
| disc_g20_base_e60_val_gemttapwr | L4 | larvae | 50 | 0.956 ± 0.012 | 0.980 ± 0.004 | 0.978 ± 0.006 | 3 |
| disc_g20_base_e60_val_gemttapwr | L4 | protozoan | 50 | 0.907 ± 0.006 | 0.881 ± 0.021 | 0.893 ± 0.012 | 3 |
| disc_g20_bg1_e60_val | L3 | eggs | 50 | 0.808 ± 0.004 | 0.816 ± 0.020 | 0.820 ± 0.014 | 3 |
| disc_g20_bg1_e60_val | L3 | larvae | 50 | 0.899 ± 0.024 | 0.956 ± 0.017 | 0.950 ± 0.012 | 3 |
| disc_g20_bg1_e60_val | L3 | protozoan | 50 | 0.771 ± 0.032 | 0.732 ± 0.030 | 0.768 ± 0.029 | 3 |
| disc_g20_bg1_e60_val | L4 | eggs | 50 | 0.832 ± 0.016 | 0.857 ± 0.048 | 0.848 ± 0.029 | 3 |
| disc_g20_bg1_e60_val | L4 | larvae | 50 | 0.912 ± 0.019 | 0.957 ± 0.014 | 0.956 ± 0.009 | 3 |
| disc_g20_bg1_e60_val | L4 | protozoan | 50 | 0.804 ± 0.026 | 0.776 ± 0.019 | 0.802 ± 0.020 | 3 |
| disc_g20_bg1_e60_val_gem | L4 | eggs | 50 | 0.873 ± 0.016 | 0.910 ± 0.017 | 0.889 ± 0.017 | 3 |
| disc_g20_bg1_e60_val_gem | L4 | larvae | 50 | 0.900 ± 0.023 | 0.959 ± 0.011 | 0.950 ± 0.011 | 3 |
| disc_g20_bg1_e60_val_gem | L4 | protozoan | 50 | 0.856 ± 0.006 | 0.850 ± 0.015 | 0.843 ± 0.007 | 3 |
| disc_g20_e_anc03_test | L3 | eggs | 50 | 0.752 ± 0.030 | 0.766 ± 0.026 | 0.770 ± 0.018 | 3 |
| disc_g20_e_anc03_test | L4 | eggs | 50 | 0.718 ± 0.055 | 0.695 ± 0.058 | 0.713 ± 0.050 | 3 |
| disc_g20_e_anc03_test_gem | L4 | eggs | 50 | 0.814 ± 0.026 | 0.861 ± 0.035 | 0.834 ± 0.025 | 3 |
| disc_g20_e_anc03_val | L3 | eggs | 50 | 0.735 ± 0.009 | 0.753 ± 0.010 | 0.763 ± 0.005 | 3 |
| disc_g20_e_anc03_val | L4 | eggs | 50 | 0.687 ± 0.048 | 0.670 ± 0.050 | 0.691 ± 0.047 | 3 |
| disc_g20_e_anc03_val_gem | L4 | eggs | 50 | 0.813 ± 0.029 | 0.844 ± 0.052 | 0.825 ± 0.037 | 3 |
| disc_g20_e_anc3_test | L3 | eggs | 50 | 0.761 ± 0.032 | 0.771 ± 0.043 | 0.776 ± 0.027 | 3 |
| disc_g20_e_anc3_test | L4 | eggs | 50 | 0.716 ± 0.043 | 0.691 ± 0.040 | 0.707 ± 0.022 | 3 |
| disc_g20_e_anc3_test_gem | L4 | eggs | 50 | 0.824 ± 0.026 | 0.865 ± 0.029 | 0.838 ± 0.024 | 3 |
| disc_g20_e_anc3_val | L3 | eggs | 50 | 0.732 ± 0.021 | 0.746 ± 0.023 | 0.751 ± 0.015 | 3 |
| disc_g20_e_anc3_val | L4 | eggs | 50 | 0.701 ± 0.025 | 0.675 ± 0.024 | 0.696 ± 0.012 | 3 |
| disc_g20_e_anc3_val_gem | L4 | eggs | 50 | 0.829 ± 0.020 | 0.855 ± 0.043 | 0.840 ± 0.028 | 3 |
| disc_g20_e_aug_test | L3 | eggs | 50 | 0.859 ± 0.018 | 0.855 ± 0.025 | 0.863 ± 0.018 | 3 |
| disc_g20_e_aug_test | L4 | eggs | 50 | 0.888 ± 0.007 | 0.895 ± 0.001 | 0.894 ± 0.007 | 3 |
| disc_g20_e_aug_test_gem | L4 | eggs | 50 | 0.902 ± 0.020 | 0.917 ± 0.023 | 0.911 ± 0.021 | 3 |
| disc_g20_e_aug_test_gemtta | L4 | eggs | 50 | 0.917 ± 0.012 | 0.928 ± 0.020 | 0.921 ± 0.013 | 3 |
| disc_g20_e_aug_test_gemttapwr | L4 | eggs | 50 | 0.940 ± 0.006 | 0.935 ± 0.016 | 0.945 ± 0.008 | 3 |
| disc_g20_e_aug_test_spp | L3 | eggs | 50 | 0.875 ± 0.024 | 0.908 ± 0.027 | 0.882 ± 0.026 | 3 |
| disc_g20_e_aug_val | L3 | eggs | 50 | 0.854 ± 0.010 | 0.850 ± 0.005 | 0.864 ± 0.009 | 3 |
| disc_g20_e_aug_val | L4 | eggs | 50 | 0.881 ± 0.011 | 0.895 ± 0.008 | 0.892 ± 0.015 | 3 |
| disc_g20_e_aug_val_gem | L4 | eggs | 50 | 0.903 ± 0.019 | 0.928 ± 0.021 | 0.918 ± 0.023 | 3 |
| disc_g20_e_aug_val_gemtta | L4 | eggs | 50 | 0.920 ± 0.017 | 0.936 ± 0.019 | 0.930 ± 0.017 | 3 |
| disc_g20_e_aug_val_gemttapwr | L4 | eggs | 50 | 0.937 ± 0.001 | 0.933 ± 0.011 | 0.944 ± 0.004 | 3 |
| disc_g20_e_base_e100_test | L3 | eggs | 50 | 0.846 ± 0.019 | 0.847 ± 0.037 | 0.854 ± 0.016 | 3 |
| disc_g20_e_base_e100_test | L4 | eggs | 50 | 0.862 ± 0.008 | 0.874 ± 0.024 | 0.869 ± 0.010 | 3 |
| disc_g20_e_base_e100_test_gem | L4 | eggs | 50 | 0.873 ± 0.020 | 0.900 ± 0.029 | 0.885 ± 0.022 | 3 |
| disc_g20_e_base_e100_val | L3 | eggs | 50 | 0.845 ± 0.010 | 0.854 ± 0.033 | 0.857 ± 0.018 | 3 |
| disc_g20_e_base_e100_val | L4 | eggs | 50 | 0.854 ± 0.006 | 0.868 ± 0.017 | 0.868 ± 0.008 | 3 |
| disc_g20_e_base_e100_val_gem | L4 | eggs | 50 | 0.873 ± 0.016 | 0.907 ± 0.027 | 0.886 ± 0.022 | 3 |
| disc_g20_e_base_test | L3 | eggs | 50 | 0.842 ± 0.020 | 0.833 ± 0.010 | 0.845 ± 0.015 | 3 |
| disc_g20_e_base_test | L4 | eggs | 50 | 0.879 ± 0.011 | 0.894 ± 0.012 | 0.885 ± 0.011 | 3 |
| disc_g20_e_base_test_gem | L4 | eggs | 50 | 0.892 ± 0.031 | 0.914 ± 0.025 | 0.902 ± 0.027 | 3 |
| disc_g20_e_base_val | L3 | eggs | 50 | 0.842 ± 0.008 | 0.833 ± 0.014 | 0.851 ± 0.013 | 3 |
| disc_g20_e_base_val | L4 | eggs | 50 | 0.878 ± 0.012 | 0.895 ± 0.013 | 0.889 ± 0.005 | 3 |
| disc_g20_e_base_val_gem | L4 | eggs | 50 | 0.902 ± 0.016 | 0.928 ± 0.021 | 0.912 ± 0.020 | 3 |
| disc_g20_e_base_val_gemtta | L4 | eggs | 50 | 0.908 ± 0.024 | 0.923 ± 0.031 | 0.912 ± 0.026 | 3 |
| disc_g20_e_bg0_test | L3 | eggs | 50 | 0.847 ± 0.027 | 0.846 ± 0.026 | 0.850 ± 0.024 | 3 |
| disc_g20_e_bg0_test | L4 | eggs | 50 | 0.881 ± 0.014 | 0.891 ± 0.019 | 0.883 ± 0.010 | 3 |
| disc_g20_e_bg0_test_gem | L4 | eggs | 50 | 0.889 ± 0.022 | 0.907 ± 0.028 | 0.895 ± 0.020 | 3 |
| disc_g20_e_bg0_test_spp | L3 | eggs | 50 | 0.862 ± 0.027 | 0.900 ± 0.032 | 0.870 ± 0.027 | 3 |
| disc_g20_e_bg0_val | L3 | eggs | 50 | 0.840 ± 0.017 | 0.838 ± 0.029 | 0.840 ± 0.024 | 3 |
| disc_g20_e_bg0_val | L4 | eggs | 50 | 0.883 ± 0.007 | 0.894 ± 0.019 | 0.892 ± 0.006 | 3 |
| disc_g20_e_bg0_val_gem | L4 | eggs | 50 | 0.892 ± 0.019 | 0.914 ± 0.018 | 0.902 ± 0.019 | 3 |
| disc_g20_e_bn_test | L3 | eggs | 50 | 0.593 ± 0.032 | 0.542 ± 0.034 | 0.551 ± 0.016 | 3 |
| disc_g20_e_bn_test | L4 | eggs | 50 | 0.873 ± 0.015 | 0.875 ± 0.015 | 0.881 ± 0.022 | 3 |
| disc_g20_e_bn_test_gem | L4 | eggs | 50 | 0.884 ± 0.024 | 0.910 ± 0.021 | 0.892 ± 0.024 | 3 |
| disc_g20_e_bn_test_spp | L3 | eggs | 50 | 0.883 ± 0.015 | 0.912 ± 0.018 | 0.886 ± 0.018 | 3 |
| disc_g20_e_bn_val | L3 | eggs | 50 | 0.591 ± 0.023 | 0.532 ± 0.042 | 0.547 ± 0.029 | 3 |
| disc_g20_e_bn_val | L4 | eggs | 50 | 0.865 ± 0.007 | 0.863 ± 0.004 | 0.877 ± 0.003 | 3 |
| disc_g20_e_bn_val_gem | L4 | eggs | 50 | 0.897 ± 0.013 | 0.927 ± 0.005 | 0.911 ± 0.017 | 3 |
| disc_g20_e_bp03_test | L3 | eggs | 50 | 0.826 ± 0.034 | 0.813 ± 0.034 | 0.829 ± 0.034 | 3 |
| disc_g20_e_bp03_test | L4 | eggs | 50 | 0.882 ± 0.015 | 0.891 ± 0.005 | 0.886 ± 0.015 | 3 |
| disc_g20_e_bp03_test_gem | L4 | eggs | 50 | 0.901 ± 0.016 | 0.919 ± 0.022 | 0.912 ± 0.014 | 3 |
| disc_g20_e_bp03_test_spp | L3 | eggs | 50 | 0.873 ± 0.028 | 0.908 ± 0.023 | 0.882 ± 0.025 | 3 |
| disc_g20_e_bp03_val | L3 | eggs | 50 | 0.818 ± 0.029 | 0.813 ± 0.027 | 0.818 ± 0.035 | 3 |
| disc_g20_e_bp03_val | L4 | eggs | 50 | 0.882 ± 0.013 | 0.897 ± 0.007 | 0.892 ± 0.014 | 3 |
| disc_g20_e_bp03_val_gem | L4 | eggs | 50 | 0.905 ± 0.021 | 0.922 ± 0.023 | 0.914 ± 0.015 | 3 |
| disc_g20_e_bp10_test | L3 | eggs | 50 | 0.837 ± 0.012 | 0.811 ± 0.038 | 0.824 ± 0.038 | 3 |
| disc_g20_e_bp10_test | L4 | eggs | 50 | 0.889 ± 0.001 | 0.899 ± 0.007 | 0.891 ± 0.004 | 3 |
| disc_g20_e_bp10_test_gem | L4 | eggs | 50 | 0.902 ± 0.017 | 0.917 ± 0.023 | 0.908 ± 0.017 | 3 |
| disc_g20_e_bp10_test_gemttapwr | L4 | eggs | 50 | 0.941 ± 0.002 | 0.935 ± 0.010 | 0.944 ± 0.003 | 3 |
| disc_g20_e_bp10_test_spp | L3 | eggs | 50 | 0.878 ± 0.024 | 0.909 ± 0.029 | 0.888 ± 0.024 | 3 |
| disc_g20_e_bp10_val | L3 | eggs | 50 | 0.837 ± 0.015 | 0.807 ± 0.036 | 0.826 ± 0.039 | 3 |
| disc_g20_e_bp10_val | L4 | eggs | 50 | 0.883 ± 0.016 | 0.899 ± 0.003 | 0.890 ± 0.018 | 3 |
| disc_g20_e_bp10_val_gem | L4 | eggs | 50 | 0.908 ± 0.012 | 0.924 ± 0.008 | 0.917 ± 0.012 | 3 |
| disc_g20_e_bp10_val_gemtta | L4 | eggs | 50 | 0.924 ± 0.010 | 0.928 ± 0.015 | 0.925 ± 0.013 | 3 |
| disc_g20_e_bp10_val_gemttapwr | L4 | eggs | 50 | 0.948 ± 0.002 | 0.941 ± 0.011 | 0.950 ± 0.009 | 3 |
| disc_g20_e_d5_e60_test | L4 | eggs | 50 | 0.815 ± 0.017 | 0.829 ± 0.019 | 0.830 ± 0.017 | 3 |
| disc_g20_e_d5_e60_test | L5 | eggs | 50 | 0.828 ± 0.029 | 0.842 ± 0.036 | 0.840 ± 0.026 | 3 |
| disc_g20_e_d5_e60_test_gem | L5 | eggs | 50 | 0.850 ± 0.026 | 0.894 ± 0.041 | 0.868 ± 0.028 | 3 |
| disc_g20_e_d5_e60_val | L4 | eggs | 50 | 0.807 ± 0.017 | 0.824 ± 0.023 | 0.823 ± 0.019 | 3 |
| disc_g20_e_d5_e60_val | L5 | eggs | 50 | 0.825 ± 0.023 | 0.844 ± 0.015 | 0.842 ± 0.020 | 3 |
| disc_g20_e_d5_e60_val_gem | L5 | eggs | 50 | 0.857 ± 0.021 | 0.899 ± 0.026 | 0.875 ± 0.021 | 3 |
| disc_g20_e_d6_e60_test | L5 | eggs | 50 | 0.809 ± 0.018 | 0.825 ± 0.014 | 0.822 ± 0.016 | 3 |
| disc_g20_e_d6_e60_test | L6 | eggs | 50 | 0.825 ± 0.009 | 0.843 ± 0.009 | 0.843 ± 0.012 | 3 |
| disc_g20_e_d6_e60_test_gem | L6 | eggs | 50 | 0.851 ± 0.012 | 0.900 ± 0.019 | 0.869 ± 0.012 | 3 |
| disc_g20_e_d6_e60_val | L5 | eggs | 50 | 0.806 ± 0.018 | 0.822 ± 0.013 | 0.818 ± 0.026 | 3 |
| disc_g20_e_d6_e60_val | L6 | eggs | 50 | 0.820 ± 0.009 | 0.839 ± 0.010 | 0.840 ± 0.007 | 3 |
| disc_g20_e_d6_e60_val_gem | L6 | eggs | 50 | 0.851 ± 0.012 | 0.918 ± 0.016 | 0.874 ± 0.002 | 3 |
| disc_g20_e_dil2_e60_test | L3 | eggs | 50 | 0.819 ± 0.015 | 0.830 ± 0.026 | 0.838 ± 0.016 | 3 |
| disc_g20_e_dil2_e60_test | L4 | eggs | 50 | 0.843 ± 0.018 | 0.840 ± 0.037 | 0.854 ± 0.016 | 3 |
| disc_g20_e_dil2_e60_test_gem | L4 | eggs | 50 | 0.867 ± 0.015 | 0.898 ± 0.025 | 0.882 ± 0.017 | 3 |
| disc_g20_e_dil2_e60_val | L3 | eggs | 50 | 0.805 ± 0.003 | 0.817 ± 0.028 | 0.824 ± 0.020 | 3 |
| disc_g20_e_dil2_e60_val | L4 | eggs | 50 | 0.832 ± 0.011 | 0.845 ± 0.034 | 0.850 ± 0.018 | 3 |
| disc_g20_e_dil2_e60_val_gem | L4 | eggs | 50 | 0.868 ± 0.012 | 0.905 ± 0.011 | 0.884 ± 0.005 | 3 |
| disc_g20_e_grn_test | L3 | eggs | 50 | 0.842 ± 0.015 | 0.842 ± 0.016 | 0.848 ± 0.013 | 3 |
| disc_g20_e_grn_test | L4 | eggs | 50 | 0.880 ± 0.006 | 0.893 ± 0.015 | 0.884 ± 0.006 | 3 |
| disc_g20_e_grn_test_gem | L4 | eggs | 50 | 0.889 ± 0.021 | 0.911 ± 0.026 | 0.899 ± 0.018 | 3 |
| disc_g20_e_grn_val | L3 | eggs | 50 | 0.838 ± 0.008 | 0.840 ± 0.024 | 0.852 ± 0.015 | 3 |
| disc_g20_e_grn_val | L4 | eggs | 50 | 0.871 ± 0.009 | 0.885 ± 0.003 | 0.883 ± 0.015 | 3 |
| disc_g20_e_grn_val_gem | L4 | eggs | 50 | 0.898 ± 0.010 | 0.915 ± 0.018 | 0.908 ± 0.010 | 3 |
| disc_g20_e_imp_e60_test | L3 | eggs | 50 | 0.822 ± 0.024 | 0.822 ± 0.033 | 0.828 ± 0.019 | 3 |
| disc_g20_e_imp_e60_test | L4 | eggs | 50 | 0.843 ± 0.023 | 0.847 ± 0.033 | 0.850 ± 0.021 | 3 |
| disc_g20_e_imp_e60_test_gem | L4 | eggs | 50 | 0.859 ± 0.009 | 0.887 ± 0.032 | 0.866 ± 0.013 | 3 |
| disc_g20_e_imp_e60_val | L3 | eggs | 50 | 0.820 ± 0.017 | 0.829 ± 0.013 | 0.829 ± 0.007 | 3 |
| disc_g20_e_imp_e60_val | L4 | eggs | 50 | 0.825 ± 0.006 | 0.832 ± 0.016 | 0.834 ± 0.008 | 3 |
| disc_g20_e_imp_e60_val_gem | L4 | eggs | 50 | 0.847 ± 0.014 | 0.878 ± 0.043 | 0.860 ± 0.026 | 3 |
| disc_g20_e_k3_test | L3 | eggs | 50 | 0.841 ± 0.013 | 0.835 ± 0.022 | 0.840 ± 0.009 | 3 |
| disc_g20_e_k3_test | L4 | eggs | 50 | 0.861 ± 0.014 | 0.867 ± 0.018 | 0.863 ± 0.009 | 3 |
| disc_g20_e_k3_test_gem | L4 | eggs | 50 | 0.881 ± 0.017 | 0.910 ± 0.025 | 0.890 ± 0.018 | 3 |
| disc_g20_e_k3_test_spp | L3 | eggs | 50 | 0.855 ± 0.034 | 0.901 ± 0.031 | 0.861 ± 0.032 | 3 |
| disc_g20_e_k3_val | L3 | eggs | 50 | 0.827 ± 0.005 | 0.822 ± 0.009 | 0.835 ± 0.004 | 3 |
| disc_g20_e_k3_val | L4 | eggs | 50 | 0.861 ± 0.008 | 0.872 ± 0.012 | 0.869 ± 0.008 | 3 |
| disc_g20_e_k3_val_gem | L4 | eggs | 50 | 0.875 ± 0.022 | 0.900 ± 0.021 | 0.884 ± 0.024 | 3 |
| disc_g20_e_k7_test | L3 | eggs | 50 | 0.859 ± 0.019 | 0.867 ± 0.021 | 0.870 ± 0.024 | 3 |
| disc_g20_e_k7_test | L4 | eggs | 50 | 0.886 ± 0.024 | 0.899 ± 0.013 | 0.894 ± 0.022 | 3 |
| disc_g20_e_k7_test_gem | L4 | eggs | 50 | 0.897 ± 0.021 | 0.923 ± 0.026 | 0.906 ± 0.022 | 3 |
| disc_g20_e_k7_val | L3 | eggs | 50 | 0.859 ± 0.007 | 0.869 ± 0.013 | 0.872 ± 0.016 | 3 |
| disc_g20_e_k7_val | L4 | eggs | 50 | 0.879 ± 0.021 | 0.896 ± 0.006 | 0.890 ± 0.020 | 3 |
| disc_g20_e_k7_val_gem | L4 | eggs | 50 | 0.901 ± 0.015 | 0.924 ± 0.019 | 0.913 ± 0.020 | 3 |
| disc_g20_e_res49_e60_test | L3 | eggs | 50 | 0.736 ± 0.011 | 0.711 ± 0.011 | 0.728 ± 0.012 | 3 |
| disc_g20_e_res49_e60_test | L4 | eggs | 50 | 0.752 ± 0.026 | 0.726 ± 0.031 | 0.743 ± 0.028 | 3 |
| disc_g20_e_res49_e60_test_gem | L4 | eggs | 50 | 0.849 ± 0.019 | 0.890 ± 0.029 | 0.863 ± 0.021 | 3 |
| disc_g20_e_res49_e60_val | L3 | eggs | 50 | 0.728 ± 0.016 | 0.715 ± 0.004 | 0.734 ± 0.013 | 3 |
| disc_g20_e_res49_e60_val | L4 | eggs | 50 | 0.747 ± 0.014 | 0.735 ± 0.014 | 0.746 ± 0.021 | 3 |
| disc_g20_e_res49_e60_val_gem | L4 | eggs | 50 | 0.847 ± 0.018 | 0.886 ± 0.015 | 0.861 ± 0.015 | 3 |
| disc_g20_e_se_e60_test | L3 | eggs | 50 | 0.806 ± 0.004 | 0.809 ± 0.019 | 0.815 ± 0.010 | 3 |
| disc_g20_e_se_e60_test | L4 | eggs | 50 | 0.843 ± 0.017 | 0.864 ± 0.027 | 0.857 ± 0.019 | 3 |
| disc_g20_e_se_e60_test_gem | L4 | eggs | 50 | 0.860 ± 0.020 | 0.899 ± 0.024 | 0.876 ± 0.021 | 3 |
| disc_g20_e_se_e60_val | L3 | eggs | 50 | 0.806 ± 0.032 | 0.802 ± 0.014 | 0.817 ± 0.018 | 3 |
| disc_g20_e_se_e60_val | L4 | eggs | 50 | 0.838 ± 0.011 | 0.858 ± 0.016 | 0.850 ± 0.005 | 3 |
| disc_g20_e_se_e60_val_gem | L4 | eggs | 50 | 0.871 ± 0.008 | 0.903 ± 0.022 | 0.887 ± 0.017 | 3 |
| disc_g20_e_sp100_e60_test | L3 | eggs | 50 | 0.825 ± 0.016 | 0.824 ± 0.032 | 0.822 ± 0.028 | 3 |
| disc_g20_e_sp100_e60_test | L4 | eggs | 50 | 0.835 ± 0.019 | 0.829 ± 0.021 | 0.836 ± 0.010 | 3 |
| disc_g20_e_sp100_e60_test_gem | L4 | eggs | 50 | 0.857 ± 0.019 | 0.894 ± 0.032 | 0.875 ± 0.018 | 3 |
| disc_g20_e_sp100_e60_val | L3 | eggs | 50 | 0.818 ± 0.005 | 0.820 ± 0.036 | 0.821 ± 0.025 | 3 |
| disc_g20_e_sp100_e60_val | L4 | eggs | 50 | 0.818 ± 0.012 | 0.809 ± 0.010 | 0.822 ± 0.009 | 3 |
| disc_g20_e_sp100_e60_val_gem | L4 | eggs | 50 | 0.868 ± 0.021 | 0.913 ± 0.036 | 0.888 ± 0.031 | 3 |
| disc_g20_e_sp25_e60_test | L3 | eggs | 50 | 0.837 ± 0.014 | 0.850 ± 0.026 | 0.844 ± 0.012 | 3 |
| disc_g20_e_sp25_e60_test | L4 | eggs | 50 | 0.859 ± 0.016 | 0.863 ± 0.025 | 0.865 ± 0.017 | 3 |
| disc_g20_e_sp25_e60_test_gem | L4 | eggs | 50 | 0.878 ± 0.019 | 0.905 ± 0.025 | 0.890 ± 0.015 | 3 |
| disc_g20_e_sp25_e60_val | L3 | eggs | 50 | 0.822 ± 0.012 | 0.836 ± 0.013 | 0.836 ± 0.012 | 3 |
| disc_g20_e_sp25_e60_val | L4 | eggs | 50 | 0.849 ± 0.006 | 0.855 ± 0.024 | 0.862 ± 0.012 | 3 |
| disc_g20_e_sp25_e60_val_gem | L4 | eggs | 50 | 0.881 ± 0.018 | 0.909 ± 0.015 | 0.895 ± 0.020 | 3 |
| disc_g20_e_w128_test | L3 | eggs | 50 | 0.884 ± 0.009 | 0.914 ± 0.017 | 0.891 ± 0.010 | 3 |
| disc_g20_e_w128_test | L4 | eggs | 50 | 0.891 ± 0.012 | 0.903 ± 0.020 | 0.892 ± 0.010 | 3 |
| disc_g20_e_w128_test_gem | L4 | eggs | 50 | 0.891 ± 0.023 | 0.914 ± 0.023 | 0.901 ± 0.024 | 3 |
| disc_g20_e_w128_val | L3 | eggs | 50 | 0.883 ± 0.023 | 0.910 ± 0.025 | 0.893 ± 0.030 | 3 |
| disc_g20_e_w128_val | L4 | eggs | 50 | 0.899 ± 0.005 | 0.919 ± 0.023 | 0.906 ± 0.010 | 3 |
| disc_g20_e_w128_val_gem | L4 | eggs | 50 | 0.900 ± 0.015 | 0.933 ± 0.019 | 0.911 ± 0.015 | 3 |
| disc_g20_e_w96_test | L3 | eggs | 50 | 0.875 ± 0.011 | 0.886 ± 0.014 | 0.879 ± 0.011 | 3 |
| disc_g20_e_w96_test | L4 | eggs | 50 | 0.887 ± 0.005 | 0.894 ± 0.011 | 0.887 ± 0.007 | 3 |
| disc_g20_e_w96_test_gem | L4 | eggs | 50 | 0.900 ± 0.018 | 0.916 ± 0.019 | 0.905 ± 0.019 | 3 |
| disc_g20_e_w96_val | L3 | eggs | 50 | 0.873 ± 0.014 | 0.884 ± 0.023 | 0.879 ± 0.018 | 3 |
| disc_g20_e_w96_val | L4 | eggs | 50 | 0.885 ± 0.005 | 0.895 ± 0.008 | 0.894 ± 0.008 | 3 |
| disc_g20_e_w96_val_gem | L4 | eggs | 50 | 0.901 ± 0.013 | 0.929 ± 0.017 | 0.914 ± 0.015 | 3 |
| disc_g20_l_anc3_test | L3 | larvae | 50 | 0.840 ± 0.038 | 0.926 ± 0.012 | 0.920 ± 0.019 | 3 |
| disc_g20_l_anc3_test | L4 | larvae | 50 | 0.866 ± 0.016 | 0.931 ± 0.014 | 0.933 ± 0.008 | 3 |
| disc_g20_l_anc3_test_gem | L4 | larvae | 50 | 0.873 ± 0.027 | 0.933 ± 0.012 | 0.937 ± 0.013 | 3 |
| disc_g20_l_anc3_val | L3 | larvae | 50 | 0.819 ± 0.044 | 0.908 ± 0.021 | 0.909 ± 0.022 | 3 |
| disc_g20_l_anc3_val | L4 | larvae | 50 | 0.833 ± 0.026 | 0.913 ± 0.014 | 0.917 ± 0.013 | 3 |
| disc_g20_l_anc3_val_gem | L4 | larvae | 50 | 0.865 ± 0.028 | 0.937 ± 0.016 | 0.932 ± 0.014 | 3 |
| disc_g20_l_aug_test | L3 | larvae | 50 | 0.921 ± 0.016 | 0.963 ± 0.016 | 0.961 ± 0.008 | 3 |
| disc_g20_l_aug_test | L4 | larvae | 50 | 0.925 ± 0.002 | 0.965 ± 0.006 | 0.963 ± 0.001 | 3 |
| disc_g20_l_aug_test_gem | L4 | larvae | 50 | 0.946 ± 0.011 | 0.976 ± 0.003 | 0.973 ± 0.006 | 3 |
| disc_g20_l_aug_test_gemtta | L4 | larvae | 50 | 0.960 ± 0.016 | 0.984 ± 0.004 | 0.980 ± 0.008 | 3 |
| disc_g20_l_aug_test_gemttapwr | L4 | larvae | 50 | 0.964 ± 0.016 | 0.981 ± 0.010 | 0.982 ± 0.008 | 3 |
| disc_g20_l_aug_val | L3 | larvae | 50 | 0.893 ± 0.022 | 0.949 ± 0.020 | 0.947 ± 0.011 | 3 |
| disc_g20_l_aug_val | L4 | larvae | 50 | 0.901 ± 0.003 | 0.958 ± 0.005 | 0.951 ± 0.002 | 3 |
| disc_g20_l_aug_val_gem | L4 | larvae | 50 | 0.919 ± 0.004 | 0.967 ± 0.004 | 0.960 ± 0.002 | 3 |
| disc_g20_l_aug_val_gemtta | L4 | larvae | 50 | 0.953 ± 0.003 | 0.980 ± 0.006 | 0.976 ± 0.001 | 3 |
| disc_g20_l_aug_val_gemttapwr | L4 | larvae | 50 | 0.957 ± 0.008 | 0.976 ± 0.001 | 0.979 ± 0.004 | 3 |
| disc_g20_l_bg0_test | L3 | larvae | 50 | 0.914 ± 0.015 | 0.962 ± 0.013 | 0.957 ± 0.008 | 3 |
| disc_g20_l_bg0_test | L4 | larvae | 50 | 0.932 ± 0.012 | 0.967 ± 0.012 | 0.966 ± 0.006 | 3 |
| disc_g20_l_bg0_test_gem | L4 | larvae | 50 | 0.934 ± 0.008 | 0.970 ± 0.009 | 0.967 ± 0.004 | 3 |
| disc_g20_l_bg0_val | L3 | larvae | 50 | 0.872 ± 0.004 | 0.948 ± 0.014 | 0.936 ± 0.002 | 3 |
| disc_g20_l_bg0_val | L4 | larvae | 50 | 0.895 ± 0.023 | 0.953 ± 0.020 | 0.948 ± 0.011 | 3 |
| disc_g20_l_bg0_val_gem | L4 | larvae | 50 | 0.917 ± 0.008 | 0.964 ± 0.010 | 0.959 ± 0.004 | 3 |
| disc_g20_l_bn_test | L3 | larvae | 50 | 0.899 ± 0.006 | 0.955 ± 0.010 | 0.949 ± 0.003 | 3 |
| disc_g20_l_bn_test | L4 | larvae | 50 | 0.929 ± 0.010 | 0.962 ± 0.009 | 0.964 ± 0.005 | 3 |
| disc_g20_l_bn_test_gem | L4 | larvae | 50 | 0.926 ± 0.008 | 0.966 ± 0.004 | 0.963 ± 0.004 | 3 |
| disc_g20_l_bn_val | L3 | larvae | 50 | 0.878 ± 0.033 | 0.949 ± 0.013 | 0.939 ± 0.016 | 3 |
| disc_g20_l_bn_val | L4 | larvae | 50 | 0.915 ± 0.025 | 0.960 ± 0.011 | 0.958 ± 0.013 | 3 |
| disc_g20_l_bn_val_gem | L4 | larvae | 50 | 0.887 ± 0.012 | 0.941 ± 0.010 | 0.943 ± 0.006 | 3 |
| disc_g20_l_bp03_test_gem | L4 | larvae | 50 | 0.934 ± 0.004 | 0.972 ± 0.003 | 0.967 ± 0.002 | 3 |
| disc_g20_l_bp03_val | L3 | larvae | 50 | 0.895 ± 0.012 | 0.950 ± 0.013 | 0.947 ± 0.006 | 3 |
| disc_g20_l_bp03_val | L4 | larvae | 50 | 0.898 ± 0.019 | 0.956 ± 0.003 | 0.949 ± 0.009 | 3 |
| disc_g20_l_bp03_val_gem | L4 | larvae | 50 | 0.933 ± 0.005 | 0.966 ± 0.005 | 0.967 ± 0.003 | 3 |
| disc_g20_l_bp03_val_gemttapwr | L4 | larvae | 50 | 0.951 ± 0.023 | 0.973 ± 0.007 | 0.975 ± 0.012 | 3 |
| disc_g20_l_bp10_test | L3 | larvae | 50 | 0.921 ± 0.013 | 0.966 ± 0.005 | 0.960 ± 0.006 | 3 |
| disc_g20_l_bp10_test | L4 | larvae | 50 | 0.945 ± 0.017 | 0.975 ± 0.009 | 0.972 ± 0.009 | 3 |
| disc_g20_l_bp10_test_gem | L4 | larvae | 50 | 0.943 ± 0.006 | 0.976 ± 0.007 | 0.972 ± 0.003 | 3 |
| disc_g20_l_bp10_val | L3 | larvae | 50 | 0.900 ± 0.020 | 0.954 ± 0.008 | 0.950 ± 0.010 | 3 |
| disc_g20_l_bp10_val | L4 | larvae | 50 | 0.900 ± 0.015 | 0.950 ± 0.003 | 0.950 ± 0.007 | 3 |
| disc_g20_l_bp10_val_gem | L4 | larvae | 50 | 0.914 ± 0.013 | 0.960 ± 0.006 | 0.957 ± 0.007 | 3 |
| disc_g20_l_bp10_val_gemtta | L4 | larvae | 50 | 0.927 ± 0.018 | 0.963 ± 0.010 | 0.963 ± 0.009 | 3 |
| disc_g20_l_grn_test | L3 | larvae | 50 | 0.918 ± 0.018 | 0.961 ± 0.005 | 0.959 ± 0.009 | 3 |
| disc_g20_l_grn_test | L4 | larvae | 50 | 0.932 ± 0.012 | 0.968 ± 0.002 | 0.966 ± 0.006 | 3 |
| disc_g20_l_grn_test_gem | L4 | larvae | 50 | 0.939 ± 0.010 | 0.972 ± 0.005 | 0.969 ± 0.005 | 3 |
| disc_g20_l_grn_val | L3 | larvae | 50 | 0.882 ± 0.019 | 0.944 ± 0.010 | 0.941 ± 0.010 | 3 |
| disc_g20_l_grn_val | L4 | larvae | 50 | 0.909 ± 0.008 | 0.960 ± 0.007 | 0.955 ± 0.004 | 3 |
| disc_g20_l_grn_val_gem | L4 | larvae | 50 | 0.911 ± 0.025 | 0.961 ± 0.003 | 0.956 ± 0.013 | 3 |
| disc_g20_l_se_e60_test | L3 | larvae | 50 | 0.879 ± 0.011 | 0.941 ± 0.013 | 0.939 ± 0.005 | 3 |
| disc_g20_l_se_e60_test | L4 | larvae | 50 | 0.905 ± 0.014 | 0.953 ± 0.010 | 0.953 ± 0.007 | 3 |
| disc_g20_l_se_e60_test_gem | L4 | larvae | 50 | 0.915 ± 0.010 | 0.965 ± 0.005 | 0.958 ± 0.005 | 3 |
| disc_g20_l_se_e60_val | L3 | larvae | 50 | 0.867 ± 0.029 | 0.936 ± 0.024 | 0.934 ± 0.015 | 3 |
| disc_g20_l_se_e60_val | L4 | larvae | 50 | 0.882 ± 0.011 | 0.941 ± 0.009 | 0.941 ± 0.005 | 3 |
| disc_g20_l_se_e60_val_gem | L4 | larvae | 50 | 0.909 ± 0.017 | 0.960 ± 0.008 | 0.955 ± 0.009 | 3 |
| disc_g20_l_sp100_test | L3 | larvae | 50 | 0.923 ± 0.007 | 0.964 ± 0.004 | 0.961 ± 0.003 | 3 |
| disc_g20_l_sp100_test | L4 | larvae | 50 | 0.930 ± 0.003 | 0.967 ± 0.008 | 0.965 ± 0.002 | 3 |
| disc_g20_l_sp100_test_gem | L4 | larvae | 50 | 0.936 ± 0.013 | 0.972 ± 0.005 | 0.968 ± 0.006 | 3 |
| disc_g20_l_sp100_val | L3 | larvae | 50 | 0.897 ± 0.017 | 0.952 ± 0.016 | 0.948 ± 0.009 | 3 |
| disc_g20_l_sp100_val | L4 | larvae | 50 | 0.902 ± 0.007 | 0.953 ± 0.008 | 0.951 ± 0.003 | 3 |
| disc_g20_l_sp100_val_gem | L4 | larvae | 50 | 0.903 ± 0.014 | 0.959 ± 0.005 | 0.951 ± 0.007 | 3 |
| disc_g20_l_w96_val_gem | L4 | larvae | 50 | 0.919 ± 0.020 | 0.965 ± 0.011 | 0.960 ± 0.010 | 3 |
| disc_g20_mask50_e60_val | L3 | eggs | 50 | 0.822 ± 0.015 | 0.838 ± 0.027 | 0.835 ± 0.018 | 3 |
| disc_g20_mask50_e60_val | L3 | larvae | 50 | 0.880 ± 0.040 | 0.934 ± 0.029 | 0.940 ± 0.020 | 3 |
| disc_g20_mask50_e60_val | L3 | protozoan | 50 | 0.785 ± 0.036 | 0.771 ± 0.043 | 0.793 ± 0.037 | 3 |
| disc_g20_mask50_e60_val | L4 | eggs | 50 | 0.861 ± 0.008 | 0.879 ± 0.021 | 0.868 ± 0.018 | 3 |
| disc_g20_mask50_e60_val | L4 | larvae | 50 | 0.894 ± 0.023 | 0.952 ± 0.013 | 0.947 ± 0.012 | 3 |
| disc_g20_mask50_e60_val | L4 | protozoan | 50 | 0.812 ± 0.030 | 0.780 ± 0.041 | 0.805 ± 0.035 | 3 |
| disc_g20_mask50_e60_val_gem | L4 | eggs | 50 | 0.875 ± 0.011 | 0.900 ± 0.015 | 0.887 ± 0.012 | 3 |
| disc_g20_mask50_e60_val_gem | L4 | larvae | 50 | 0.903 ± 0.006 | 0.958 ± 0.012 | 0.951 ± 0.003 | 3 |
| disc_g20_mask50_e60_val_gem | L4 | protozoan | 50 | 0.841 ± 0.016 | 0.842 ± 0.022 | 0.819 ± 0.017 | 3 |
| disc_g20_nomask_e60_val | L3 | eggs | 50 | 0.830 ± 0.011 | 0.849 ± 0.031 | 0.851 ± 0.022 | 3 |
| disc_g20_nomask_e60_val | L3 | larvae | 50 | 0.853 ± 0.009 | 0.930 ± 0.007 | 0.926 ± 0.005 | 3 |
| disc_g20_nomask_e60_val | L3 | protozoan | 50 | 0.681 ± 0.070 | 0.684 ± 0.045 | 0.718 ± 0.046 | 3 |
| disc_g20_nomask_e60_val | L4 | eggs | 50 | 0.813 ± 0.019 | 0.820 ± 0.049 | 0.831 ± 0.029 | 3 |
| disc_g20_nomask_e60_val | L4 | larvae | 50 | 0.878 ± 0.027 | 0.941 ± 0.012 | 0.939 ± 0.013 | 3 |
| disc_g20_nomask_e60_val | L4 | protozoan | 50 | 0.735 ± 0.049 | 0.702 ± 0.040 | 0.742 ± 0.035 | 3 |
| disc_g20_nomask_e60_val_gem | L4 | eggs | 50 | 0.882 ± 0.009 | 0.914 ± 0.011 | 0.896 ± 0.004 | 3 |
| disc_g20_nomask_e60_val_gem | L4 | larvae | 50 | 0.903 ± 0.049 | 0.956 ± 0.018 | 0.951 ± 0.025 | 3 |
| disc_g20_nomask_e60_val_gem | L4 | protozoan | 50 | 0.852 ± 0.018 | 0.853 ± 0.017 | 0.842 ± 0.015 | 3 |
| disc_g20_p_aug_e60_test | L3 | protozoan | 50 | 0.775 ± 0.047 | 0.754 ± 0.034 | 0.785 ± 0.031 | 3 |
| disc_g20_p_aug_e60_test | L4 | protozoan | 50 | 0.813 ± 0.014 | 0.789 ± 0.010 | 0.816 ± 0.013 | 3 |
| disc_g20_p_aug_e60_test_gem | L4 | protozoan | 50 | 0.862 ± 0.023 | 0.865 ± 0.022 | 0.855 ± 0.016 | 3 |
| disc_g20_p_aug_e60_val | L3 | protozoan | 50 | 0.765 ± 0.054 | 0.729 ± 0.023 | 0.761 ± 0.021 | 3 |
| disc_g20_p_aug_e60_val | L4 | protozoan | 50 | 0.807 ± 0.029 | 0.789 ± 0.052 | 0.812 ± 0.031 | 3 |
| disc_g20_p_aug_e60_val_gem | L4 | protozoan | 50 | 0.851 ± 0.024 | 0.864 ± 0.011 | 0.845 ± 0.003 | 3 |
| disc_g20_p_base_test | L3 | protozoan | 50 | 0.791 ± 0.006 | 0.767 ± 0.008 | 0.800 ± 0.009 | 3 |
| disc_g20_p_base_test | L4 | protozoan | 50 | 0.843 ± 0.002 | 0.813 ± 0.010 | 0.843 ± 0.003 | 3 |
| disc_g20_p_base_test_gem | L4 | protozoan | 50 | 0.888 ± 0.011 | 0.881 ± 0.021 | 0.876 ± 0.016 | 3 |
| disc_g20_p_base_val | L3 | protozoan | 50 | 0.790 ± 0.022 | 0.759 ± 0.029 | 0.794 ± 0.024 | 3 |
| disc_g20_p_base_val | L4 | protozoan | 50 | 0.840 ± 0.003 | 0.807 ± 0.021 | 0.835 ± 0.014 | 3 |
| disc_g20_p_base_val_gem | L4 | protozoan | 50 | 0.881 ± 0.010 | 0.883 ± 0.011 | 0.872 ± 0.014 | 3 |
| disc_g20_p_bn_e60_test | L3 | protozoan | 50 | 0.280 ± 0.055 | 0.313 ± 0.043 | 0.310 ± 0.066 | 3 |
| disc_g20_p_bn_e60_test | L4 | protozoan | 50 | 0.775 ± 0.005 | 0.748 ± 0.014 | 0.789 ± 0.010 | 3 |
| disc_g20_p_bn_e60_test_gem | L4 | protozoan | 50 | 0.849 ± 0.004 | 0.852 ± 0.009 | 0.851 ± 0.009 | 3 |
| disc_g20_p_bn_e60_val | L3 | protozoan | 50 | 0.270 ± 0.043 | 0.308 ± 0.034 | 0.305 ± 0.056 | 3 |
| disc_g20_p_bn_e60_val | L4 | protozoan | 50 | 0.768 ± 0.006 | 0.725 ± 0.015 | 0.771 ± 0.010 | 3 |
| disc_g20_p_bn_e60_val_gem | L4 | protozoan | 50 | 0.843 ± 0.010 | 0.841 ± 0.013 | 0.830 ± 0.006 | 3 |
| disc_g20_p_bp03_e60_test | L3 | protozoan | 50 | 0.787 ± 0.040 | 0.766 ± 0.019 | 0.796 ± 0.018 | 3 |
| disc_g20_p_bp03_e60_test | L4 | protozoan | 50 | 0.832 ± 0.018 | 0.804 ± 0.016 | 0.828 ± 0.014 | 3 |
| disc_g20_p_bp03_e60_test_gem | L4 | protozoan | 50 | 0.874 ± 0.008 | 0.879 ± 0.011 | 0.862 ± 0.006 | 3 |
| disc_g20_p_bp03_e60_val | L3 | protozoan | 50 | 0.787 ± 0.045 | 0.758 ± 0.040 | 0.790 ± 0.034 | 3 |
| disc_g20_p_bp03_e60_val | L4 | protozoan | 50 | 0.823 ± 0.018 | 0.795 ± 0.023 | 0.818 ± 0.016 | 3 |
| disc_g20_p_bp03_e60_val_gem | L4 | protozoan | 50 | 0.872 ± 0.012 | 0.878 ± 0.020 | 0.852 ± 0.009 | 3 |
| disc_g20_p_bp03_e60_val_gemttapwr | L4 | protozoan | 50 | 0.914 ± 0.004 | 0.886 ± 0.016 | 0.892 ± 0.014 | 3 |
| disc_g20_p_imp_e60_test | L3 | protozoan | 50 | 0.800 ± 0.025 | 0.774 ± 0.020 | 0.800 ± 0.019 | 3 |
| disc_g20_p_imp_e60_test | L4 | protozoan | 50 | 0.827 ± 0.019 | 0.805 ± 0.031 | 0.823 ± 0.024 | 3 |
| disc_g20_p_imp_e60_test_gem | L4 | protozoan | 50 | 0.870 ± 0.017 | 0.861 ± 0.009 | 0.857 ± 0.011 | 3 |
| disc_g20_p_imp_e60_val | L3 | protozoan | 50 | 0.791 ± 0.021 | 0.763 ± 0.006 | 0.782 ± 0.002 | 3 |
| disc_g20_p_imp_e60_val | L4 | protozoan | 50 | 0.819 ± 0.024 | 0.790 ± 0.008 | 0.811 ± 0.018 | 3 |
| disc_g20_p_imp_e60_val_gem | L4 | protozoan | 50 | 0.859 ± 0.013 | 0.856 ± 0.026 | 0.838 ± 0.015 | 3 |
| disc_r1_rand_s1_val | L3 | eggs | 50 | 0.909 ± 0.003 | 0.925 ± 0.014 | 0.923 ± 0.012 | 3 |
| disc_r1_rand_s1_val | L3 | larvae | 50 | 0.831 ± 0.033 | 0.919 ± 0.016 | 0.915 ± 0.017 | 3 |
| disc_r1_rand_s1_val | L3 | protozoan | 50 | 0.806 ± 0.012 | 0.819 ± 0.021 | 0.811 ± 0.014 | 3 |
| disc_r1_rand_s2_val | L3 | eggs | 50 | 0.886 ± 0.004 | 0.913 ± 0.002 | 0.903 ± 0.006 | 3 |
| disc_r1_rand_s2_val | L3 | larvae | 50 | 0.830 ± 0.035 | 0.915 ± 0.014 | 0.915 ± 0.018 | 3 |
| disc_r1_rand_s2_val | L3 | protozoan | 50 | 0.817 ± 0.018 | 0.836 ± 0.030 | 0.826 ± 0.022 | 3 |
| disc_r1_rand_val | L3 | eggs | 50 | 0.909 ± 0.011 | 0.924 ± 0.008 | 0.920 ± 0.008 | 3 |
| disc_r1_rand_val | L3 | larvae | 50 | 0.808 ± 0.034 | 0.903 ± 0.021 | 0.904 ± 0.017 | 3 |
| disc_r1_rand_val | L3 | protozoan | 50 | 0.805 ± 0.009 | 0.820 ± 0.014 | 0.809 ± 0.018 | 3 |
| disc_r1_val | L1 | eggs | 50 | 0.890 ± 0.010 | 0.924 ± 0.006 | 0.904 ± 0.015 | 3 |
| disc_r1_val | L1 | larvae | 50 | 0.762 ± 0.030 | 0.880 ± 0.026 | 0.881 ± 0.015 | 3 |
| disc_r1_val | L1 | protozoan | 50 | 0.765 ± 0.004 | 0.798 ± 0.016 | 0.791 ± 0.010 | 3 |
| disc_r1_val | L2 | eggs | 50 | 0.892 ± 0.013 | 0.930 ± 0.015 | 0.910 ± 0.018 | 3 |
| disc_r1_val | L2 | larvae | 50 | 0.783 ± 0.038 | 0.890 ± 0.028 | 0.892 ± 0.019 | 3 |
| disc_r1_val | L2 | protozoan | 50 | 0.825 ± 0.008 | 0.834 ± 0.017 | 0.829 ± 0.013 | 3 |
| disc_r1_val | L3 | eggs | 50 | 0.897 ± 0.010 | 0.938 ± 0.012 | 0.913 ± 0.014 | 3 |
| disc_r1_val | L3 | larvae | 50 | 0.830 ± 0.016 | 0.920 ± 0.016 | 0.915 ± 0.008 | 3 |
| disc_r1_val | L3 | protozoan | 50 | 0.848 ± 0.006 | 0.848 ± 0.016 | 0.842 ± 0.013 | 3 |
| disc_r2_val | L1 | eggs | 50 | 0.887 ± 0.012 | 0.926 ± 0.012 | 0.904 ± 0.015 | 3 |
| disc_r2_val | L2 | eggs | 50 | 0.897 ± 0.005 | 0.932 ± 0.011 | 0.909 ± 0.009 | 3 |
| disc_r2_val | L3 | eggs | 50 | 0.894 ± 0.007 | 0.923 ± 0.016 | 0.906 ± 0.010 | 3 |
| disc_r2_val | L4 | eggs | 50 | 0.873 ± 0.018 | 0.909 ± 0.023 | 0.889 ± 0.013 | 3 |
| disc_r2_val | L5 | eggs | 50 | 0.857 ± 0.011 | 0.901 ± 0.020 | 0.875 ± 0.014 | 3 |
| disc_r2_val | L6 | eggs | 50 | 0.854 ± 0.021 | 0.898 ± 0.024 | 0.872 ± 0.026 | 3 |
| disc_r2_val | L7 | eggs | 50 | 0.838 ± 0.008 | 0.881 ± 0.024 | 0.857 ± 0.013 | 3 |
| disc_r2_val | L8 | eggs | 50 | 0.817 ± 0.008 | 0.865 ± 0.014 | 0.841 ± 0.009 | 3 |
| disc_sl_w0125_e60_val_gem | L4 | larvae | 50 | 0.853 ± 0.009 | 0.936 ± 0.013 | 0.927 ± 0.004 | 3 |
| disc_w64fg_test_avg | L3 | eggs | 50 | 0.829 ± 0.030 | 0.832 ± 0.035 | 0.835 ± 0.024 | 3 |
| disc_w64fg_test_avg | L3 | larvae | 50 | 0.914 ± 0.003 | 0.953 ± 0.011 | 0.957 ± 0.001 | 3 |
| disc_w64fg_test_avg | L3 | protozoan | 50 | 0.798 ± 0.026 | 0.764 ± 0.016 | 0.796 ± 0.015 | 3 |
| disc_w64fg_val | L3 | eggs | 50 | 0.897 ± 0.015 | 0.933 ± 0.021 | 0.914 ± 0.022 | 3 |
| disc_w64fg_val | L3 | larvae | 50 | 0.833 ± 0.029 | 0.916 ± 0.017 | 0.916 ± 0.014 | 3 |
| disc_w64fg_val | L3 | protozoan | 50 | 0.866 ± 0.012 | 0.856 ± 0.013 | 0.849 ± 0.006 | 3 |
