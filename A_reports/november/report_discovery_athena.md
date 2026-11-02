# Discovery híbrido SPiFiL na Athena (31/10–02/11): relatório consolidado

Para o usuário e para o agente de Jaci que vai continuar. Toda cifra vem dos arquivos citados.
Abreviações das fontes: **DEC** = `A_reports/october/discovery_hybrid_spifil_decisions.md` (seção = título da entrada),
**REC** = `discovery_recipe.md`, **RANK** = `discovery_ranking_test.md`, **DIAG** = `discovery_eggs_diag.md`,
**ES** = `discovery_early_stop.md`, **LIT2** = `discovery_literature_round2.md`, **CONV** = `discovery_conv_techniques.md`,
**SL** = `discovery_scaling_law_prompt.md`, **BOARD** = `discovery_insights_board.md` (todos em `A_reports/october/`).

**Em uma frase:** uma CNN pequena (~200 mil parâmetros) que nasce dos "carimbos" do SPiFiL e aprende sozinha a
redesenhar a imagem já empata com o I-JEPA em eggs e passa dele em larvae e protozoan — e quase todo o ganho veio
de **como lemos** o que a rede vê, não de mudar a rede.

---

## 1. Resumo em 5 linhas

1. **Modelo:** `disc_bp01w64c1280`, CNN de 4 camadas 32/48/64/1280, cada camada nasce do SPiFiL e é ajustada por autoencoder sem rótulo (REC:1).
2. **Leitura final** `gem_tta_pw`: GeM na L4 → média das 8 rotações/flips → PCA-whitening com shrinkage → L2 → SVM linear (REC:1, DEC "whitening do vetor").
3. **κ teste, 50%, média de 3 splits:** eggs **0.933**, larvae **0.968**, protozoan **0.928** (conferido em `results/svm_disc_bp01w64c1280_test_gemttapwr.csv`: eggs 0.949/0.935/0.916, larvae 0.971/0.974/0.959, protozoan 0.926/0.934/0.923).
4. **I-JEPA (teste, 50%, 1 vista só):** 0.940 / 0.943 / 0.864 (conferido em `/dados/home/moliveira/Scalable_Hybrid_FLIM/results/ijepa_svm_aggregated.csv`: 0.9397 / 0.9433 / 0.8636). Δ: eggs −0.007 (empate), larvae +0.025, protozoan +0.064 (REC:1).
5. **Ressalva:** o I-JEPA não teve TTA; para comparar de igual para igual ele precisa da mesma leitura (DEC "TTA confirmado no teste").

## 2. O modelo

Resumo de `A_reports/october/discovery_recipe.md` (leia lá os `arquivo:linha`). Imagine uma casa construída andar por
andar: o SPiFiL escolhe os tijolos olhando 1 imagem por classe, o autoencoder lixa os tijolos tentando redesenhar a
imagem, e só então vem o próximo andar (REC §2).

| parte | o que é | fonte |
|---|---|---|
| entrada | LAB (`ift_lab`), 200×200, sem aug na base | REC §2.1 |
| arquitetura | 4 camadas: 5×5/5×5/5×5/1×1, 32/48/64/1280 (protozoan 30/48/60/1280), ReLU, maxpool 3×3 s2 nas 3 primeiras; mapa final 24×24; sem BN/SE/GRN | REC §2.6 |
| parâmetros | encoder 200.944 (eggs/larvae), 188.468 (protozoan); decoder fora da conta | REC §2.6 |
| SPiFiL por camada | DISF com 50 superpixels dentro da máscara, 1 imagem por classe (`paper_classes: true`, impureza fora), ranking Fisher com Ledoit-Wolf, `DiversitySelector`, alocação uniforme por classe | REC §2.2 |
| L4 1×1 | 400/100/300 filtros SPiFiL (eggs/larvae/protozoan); o resto até 1280 é aleatório na norma média (`fill_random_last`) | REC §2.2 |
| treino | camada nova lr 5e-4, antigas ×0.1 (`layerwise_backprop: 0.1`), decoder novo por camada, AdamW wd 0.05, warmup 10 + cosine até 1e-5, 200 épocas, ckpt por `probe/val_recon_loss` | REC §2.3, §2.5 |
| loss | BCE por pixel, fundo pesa 0.1 (`recon_bg_weight`); fg sem rótulo pela mediana da borda | REC §2.4 |
| leitura | `gem_tta_pw` (1280-d) → `fit_svm` + `compute_metrics` oficiais | REC §2.7 |

## 3. O que foi feito (cronológico)

κ = κ de validação, média de 3 splits, salvo indicação. "Ganha / empata / perde" contra a referência da época.

| data | família / eixo | mudança | κ | veredito | fonte |
|---|---|---|---|---|---|
| 29/10 | R1 | 3 camadas 5×5 24/32/48, flatten | L3 0.897/0.829/0.848 | referência | DEC "κ val L3 completo do fg01" |
| 30/10 | fg01 | fundo 0.1 na loss | 0.892/0.835/0.855 | empata | idem |
| 30/10 | popz | z-score populacional | 0.886/0.820/0.844 | perde | idem |
| 31/10 | R2 (8 camadas, dilation) | profundidade | eggs L3 0.898 → L8 ~0.81 | perde | DEC "R2 interrompido" |
| 31/10 | rand (3 seeds) | filtros aleatórios em vez de SPiFiL | protozoan 0.809 vs 0.855; larvae +0.013; eggs −0.008 | SPiFiL ganha | DEC "controle aleatório fechado" |
| 31/10 | bp01 | antigas treinam ×0.1 | 0.902/0.829/0.866 | ganha | DEC "bp01 é o melhor nos 3" |
| 31/10 | w64fg | largura 32/48/64 | 0.897/0.833/— | empata | DEC "w64fg não soma" |
| 01/11 | bp01w64 | bp01 + largura | 0.906/0.845/0.865 | ganha (nova base) | DEC "bp01w64 é a nova base" |
| 01/11 | c1280 | + L4 1×1×1280, leitura avg | L3→L4 avg +0.04 a +0.06 | ganha no avg | DEC "L4 1x1x1280 sobe o κ" |
| 01/11 | bp01w64c96 | L4 1×1×96 flatten | −0.007 média | empata/perde | DEC "bp01w64c96 não ajuda" |
| 01/11 | regra < 2000-d | fim do flatten | — | regra | DEC "nada de flatten" |
| 01/11 | spp, gem | leituras novas | gem L4 0.899/0.910/0.875 | gem ganha | DEC "GeM na L4 1x1×1280" |
| 01/11 | g20 (20 eixos) | 1 knob por família | ver §4 | platô | DEC "grade g20" |
| 02/11 | gem_dense | GeM L3+L4 | 0.905/0.904/0.871 | empata | DEC "gem_dense empata" |
| 02/11 | gem_fg | pooling só no objeto | 0.899/0.900/0.871 | empata | DEC "o SVM não precisa do fundo" |
| 02/11 | base_e60 | 60 épocas | eggs −0.024, larvae −0.007, protozoan −0.014 | só larvae/protozoan aceitam (com gem) | DEC "gate das 60 épocas fechado" |
| 02/11 | e_imp / p_imp | semente de impureza | eggs −0.028, protozoan −0.002 | perde/empata | DEC "semente de impureza também não ajuda" |
| 02/11 | e_dil2, e_d5, e_d6, e_res49 | RF/profundidade/resolução (60 ép.) | −0.007 / −0.018 / −0.024 / −0.028 | empata/perde | DEC "eggs num platô"; "lei da profundidade" |
| 02/11 | gem_tta | TTA 8 vistas | 0.912/0.946/0.917 | **ganha** | DEC "TTA confirmado no teste" |
| 02/11 | aug + TTA | geo_aug no treino | e 0.920, l 0.953 | ganha com TTA | DEC "aug no treino + TTA"; "somam também em larvae" |
| 02/11 | e_bp10 + TTA | backprop 1.0 | 0.925 | ganha | DEC "e_bp10 + TTA é o maior eggs" |
| 02/11 | gem_tta_pw (sem shrinkage) | PCA-whitening puro | eggs ~0.24–0.44, larvae ~0 | colapsa (inválido) | DEC "o _pw sem shrinkage colapsou" |
| 02/11 | gem_tta_pwr | whitening com shrinkage | 0.942/0.961/— | **ganha** | DEC "whitening do vetor" |
| 02/11 | e_bp10 + pwr | bp 1.0 + leitura final | 0.949 | ganha (mais estável) | DEC "com whitening, o aug para de somar" |
| 02/11 | l_bp03 | bp 0.3 em larvae | gem 0.933 | ganha | DEC "lei do backprop em larvae tem pico em 0.3" |
| 02/11 | base_e60 + pw | gate com leitura final | larvae −0.005 aceita; protozoan −0.019 rejeita | só larvae | DEC "gate das 60 épocas com a leitura final" |

## 4. Leis encontradas

- **Backprop:** eggs sobe até 1.0 (gem 0.899 → 0.905 → 0.908; com pwr 0.949); larvae tem pico em 0.3 (0.910 → 0.933 → 0.914) (DEC "lei do backprop fechada"; "pico em 0.3").
- **Kernel (eggs):** k3 0.875 → k5 ~0.900 → k7 0.902: ganha de 3 para 5 e satura (DEC "lei do kernel fechada").
- **Largura (eggs):** 32/48/64 0.899; 48/64/96 0.901; 64/96/128 0.900: plana (DEC "lei da largura em eggs é plana").
- **Profundidade (eggs, 60 ép.):** 4 camadas 0.875 → 5 0.857 → 6 0.851: cai; R2 de 8 camadas também caía (DEC "lei da profundidade"; "R2 interrompido").
- **Âncora L2-SP:** λ=0 0.899, λ=0.3 0.813, λ=3 0.829 em eggs; larvae λ=3 −0.048: qualquer âncora derruba (DEC "lei da âncora fechada").
- **Orçamento (eggs):** 60 ép. 0.875, 100 ép. 0.873, 200 ép. ~0.900; e400 rodando. Larvae aceita 60, protozoan só com gem (DEC "lei do orçamento"; "gate ... com a leitura final"). Nas L1–L3 o best cai na época 193–198 de 200 (ES; REC §2.5).
- **Seeds:** com 3 seeds do aleatório, o SPiFiL ganha protozoan (+0.046) e é mais reprodutível (repetição ≤0.008 vs 0.012–0.022 do aleatório). Ruído do pipeline por split até ±0.02, média de 3 splits ~0.003 (DEC "controle aleatório fechado"; "lei do kernel fechada").
- **Sementes SPiFiL:** sp25 0.881, sp50 0.875, sp100 0.868 (eggs, 60 ép.); larvae 50→100 plana (DEC "aug no treino + TTA"; "l_sp100").
- **Aug (geo_aug):** sozinho neutro com gem; soma com TTA (+0.008 eggs, +0.007 larvae); com whitening, para de somar em eggs (0.937 vs 0.942) (DEC "com whitening, o aug para de somar").
- **BN / SE / GRN:** BN neutro em eggs (0.897), perde em larvae (−0.023); SE e GRN empatam nos dois (DEC "BN perde em larvae"; "SE neutro"; "l_grn"; "e_grn").
- **Fundo:** peso 1.0 / 0.1 / 0 empata; pooling só no objeto empata (DEC "gem_dense empata"; "o SVM não precisa do fundo").
- **Leitura (o eixo que mais rende):** eggs avg 0.880 → gem 0.891 → tta 0.912 → tta+whitening 0.942 (val; DEC "whitening do vetor"). Decomposição do ganho sobre o gem (pwr / TTA / os dois): eggs +0.029/+0.013/+0.043; larvae +0.010/+0.036/+0.051; protozoan ~+0.008/+0.043/+0.052. Whitening domina em eggs (detalhe fino), TTA em larvae/protozoan (objetos sem orientação) (DEC "whitening e TTA são aditivos").

## 5. Insights principais

1. **A leitura vale mais que a rede.** Mesmo encoder, eggs teste 0.880 → 0.933 só trocando a leitura (REC:1).
2. **SPiFiL é ponto de partida, não restrição.** Ganha do aleatório onde ele vai mal (protozoan +0.046) e deixa o treino reprodutível; prender os pesos nele custa até −0.09 (DEC "controle aleatório fechado"; "lei da âncora").
3. **O treino apaga o contraste objeto/fundo do SPiFiL** (L3 2.9 → 1.0), mas isso não custa κ (DEC "quanto o backprop muda os mapas").
4. **99% dos erros de eggs são ovo ↔ impureza** (218 de 220); a impureza não tem semente SPiFiL, mas dar semente a ela piora (DIAG; DEC "eggs erra ovo × impureza"; "gate das 60 épocas: rejeitado em eggs").
5. **TTA corta as impurezas tomadas por ovo** (137 → 111) (DEC "o gem_tta corta justamente").
6. **Whitening precisa de shrinkage**: sem ele, o SVM colapsa (DEC "o _pw sem shrinkage colapsou").
7. **Platô de construção:** ~25 eixos de arquitetura/treino sem ganho acima do ruído; só k3 e âncora/BN mexem, e para baixo (DEC "eggs num platô"; "lei da profundidade").
8. **A loss não prevê o κ** entre inicializadores diferentes; só dentro do mesmo encoder com mais backprop (DEC "bp01 vence o fg01"; "bp01w64c96 não ajuda").
9. **Eggs precisa de treino longo**; larvae se contenta com 60 épocas (DEC "gate das 60 épocas fechado").
10. **Decidir só pela média de 3 splits:** o pipeline não é bit-reprodutível e larvae oscila ±0.05 num split (DEC "base de larvae foi um sorteio alto"; "correção. A estabilidade do fg01").
11. **Receitas por dataset hoje:** eggs bp 1.0 + leitura final (0.949 val); larvae aug + TTA (0.953) ou bp 0.3; protozoan em aberto (p_aug200) (DEC "com whitening, o aug para de somar"; "o que soma com o TTA").

## 6. Em andamento (02/11, noite)

Fonte: `tmux ls` e `/tmp/claude-1033/.../scratchpad/gpu_queue.txt` na Athena.

| família | o que testa | estado |
|---|---|---|
| e_aug_pn, e_aug_div, e_aug_ipc3, e_aug_knn | escolha de filtros do SPiFiL: patch_norm, NMS de quase duplicatas (71 de 400 no smoke), média de 5 vizinhos, 3 imagens por classe (DEC "4 opções parecidas com o whitening") | treinando (`g20_e_aug_*`) |
| e_aug_wh | ZCA nos patches antes do Fisher, em todas as camadas (DEC "e_aug_wh") | treinando |
| e_bp10_aug, l_bp10_aug, p_bp10_aug | bp 1.0 + aug nos 3 (receita única) | treinando |
| p_aug200 | aug a 200 ép. em protozoan | treinando |
| nomask_e60, bg1_e60 | receita sem fundo/máscara (REC §1) | na fila de SVM (`svmq_g20_*`) |
| e_base_e400 | 4º ponto da lei do orçamento | treinando |
| curvas 1–100% | `disc_bp01w64c1280_pcts`, leitura gem_tta_pw no teste | SVM (`pwrq_pcts`, `testq_pcts`) |
| lei de escala `disc_sl_*` | largura m ∈ {1/8,1/4,1/2,2}, 1×1 ∈ {160,320,640,1920}, d3, d5 (DEC "lei de escala lançada") | w0125/w025/w050/c160/c320/d3 treinando ou em SVM; na fila: w200_e200, w200_e60, c640_e60, c1920_e200, c1920_e60, d3_e60, d5_e200, c160_e60, c640_e200 (+ p_w96_e60) |

Outros SVMs pendentes na fila: l_bp03, p_bn_e60, mask50_e60 (κ com ressalva de máscara sorteada, DEC "TTA na leitura é o primeiro ganho"), base_e60 pwr.

## 7. Próximos experimentos (ordem de prioridade, para Jaci)

| # | o que é | YAML | hipótese SPiFiL | custo |
|---|---|---|---|---|
| 1 | **Confirmar as receitas finais no teste**: e_bp10 (eggs), l_aug / l_bp03 (larvae), p_aug200 / p_bp10_aug (protozoan), todos com gem_tta_pw | `experiments/autoencoder/eval_svm_disc_g20_<fam>_test_gemttapwr.yaml` (existem para e_aug; criar os demais copiando) | SPiFiL de partida + rede livre dá o resumo mais invariante | só SVM, CPU, 1 por vez |
| 2 | **I-JEPA com TTA (e whitening)** para comparação justa | não existe; criar YAML de eval do I-JEPA com `embed_mode` igual | — (régua) | só eval |
| 3 | **Curvas 1–100% com gem_tta_pw** (terminar e plotar) | `eval_svm_disc_bp01w64c1280_pcts_test_gemttapwr.yaml` | o SPiFiL ajuda mais com pouco dado | só SVM |
| 4 | **Cauda do cosine em eggs**: cosine de 100 ép. + 50–100 ép. em eta_min | não existe; precisa de opção nova no scheduler (DEC "lei do orçamento") | o detalhe ovo × impureza é aprendido na cauda de lr baixa; SPiFiL já deu bons filtros | 1 família eggs × 3 splits, ~200 ép. |
| 5 | **Lei de escala** `disc_sl_*` (terminar a fila e ajustar κ × N) | `experiments/autoencoder/growth_disc_sl_*.yaml` | modelos pequenos são "mais SPiFiL" (até 100% da 1×1); controlar pela fração SPiFiL | ~11 famílias; e200 em eggs |
| 6 | **Ortogonalidade dos filtros** no fine-tune (‖WᵀW−I‖) | não existe (chave nova no módulo + `RUNNER_ARGS`) | mantém a diversidade dos carimbos SPiFiL que o backprop tende a juntar (LIT2 item 7) | 1 família × 3 splits |
| 7 | **LCN** (normalização local de contraste) na entrada/patches | não existe | tira brilho local para o Fisher escolher forma; parente do patch_norm — rodar só se e_aug_pn ganhar | 1 família |
| 8 | **Whitening estilo Network Deconvolution** em toda camada no treino | não existe | o `whiten` atual já branqueia os patches de TODAS as camadas na seleção (DEC "4 opções"), mas não durante o treino; só vale se e_aug_wh ganhar | código novo + 1 família |
| 9 | Protozoan: decidir a receita (p_aug200 vs p_bp10_aug) e o orçamento (60 ép. rejeitado com pwr) | `growth_disc_g20_p_aug200.yaml`, `growth_disc_g20_p_bp10_aug.yaml` | — | já treinando |

## 8. Tabela de workers para Jaci

**Jaci (R3, 02/11):** 1 GPU NVIDIA RTX PRO 6000 Blackwell, 97.887 MiB, 0 MiB em uso por outros; `nproc` 128.

- Teto 70%: 0,70 × 97.887 ≈ **68.500 MiB**. Livre para nós ≈ 68,5 GB (ninguém mais usa a GPU).
- CPU: 0,70 × 128 = **89 núcleos**. Com 10 runs × `cpus_per_experiment: 8` = 80 núcleos (62%). `num_workers` do dataloader fica em 2 por run, como no YAML.
- Picos por run, medidos em Athena (A6): base/c* ~3,6 GB, c1920 ~4,0, d3 ~2,5, d5 ~4,7, w96 ~5,6, w200 ~7 (estimado, não medido).

| GPU | run | família (YAML em `experiments/autoencoder/`) | pico estimado |
|---|---|---|---|
| 0 | 1 | growth_disc_sl_w200_e200 | ~7,0 GB |
| 0 | 2 | growth_disc_sl_w200_e60 | ~7,0 GB |
| 0 | 3 | growth_disc_sl_c1920_e200 | ~4,0 GB |
| 0 | 4 | growth_disc_sl_c1920_e60 | ~4,0 GB |
| 0 | 5 | growth_disc_sl_c640_e200 | ~3,6 GB |
| 0 | 6 | growth_disc_sl_c640_e60 | ~3,6 GB |
| 0 | 7 | growth_disc_sl_c160_e60 | ~3,6 GB |
| 0 | 8 | growth_disc_sl_d3_e60 | ~2,5 GB |
| 0 | 9 | growth_disc_sl_d5_e200 | ~4,7 GB |
| 0 | 10 | growth_disc_g20_p_w96_e60 | ~5,6 GB |
| | | **soma** | **~45,6 GB** (≤ 68,5) |

- Os 10 YAMLs já estão com `gpu_ids: [0]`, `max_per_gpu: 1` e `cpus_per_experiment: 8`. Saíram da fila de Athena.
- Folga de ~23 GB fica de reserva: o w200 não foi medido. Meça nos primeiros 5 min de cada run; se a soma medida passar de 68,5 GB, pause o run mais novo.
- SVM depois de cada treino, **um por vez** (flock local em Jaci) com `nice -n 10`. Evals por família: `eval_svm_<fam>_val_gem.yaml`, `_val_gemttapwr.yaml`, `_test_gemttapwr.yaml`.


## 9. Regras de operação

- **GPU:** Athena, teto de **20 GB por GPU** somando todos os processos do moliveira; o que não cabe vai para `gpu_queue.txt` (DEC "teto de 20 GB"). **Jaci: até 70%** da memória de cada GPU.
- **SVM um por vez**, sob `flock`, com `nice`, sempre **depois** do treino; a CPU é o gargalo (DEC "SVM desativado"; "tudo sai da GPU 3").
- **Sem flatten:** leitura e κ de treino só com avg/gem/spp/tta/pw abaixo de **2000-d** (DEC "nada de flatten").
- **Teste só para reportar.** A escolha (camada, leitura, eixo) sai da validação (DEC "Alvo"; "TESTE dos melhores modelos").
- **Decidir pela média de 3 splits**; Δ < ~0.02 é ruído; em larvae usar ≥2 seeds para Δ pequeno (DEC "base de larvae foi um sorteio alto").
- **Nada de âncora L2-SP** (DEC "lei da âncora fechada").
- **SPiFiL sempre inicializa** (nunca random, salvo o controle); uma mudança por família contra a referência no **mesmo orçamento** (200 ép. vs `e_base`; 60 ép. vs `base_e60`) (DEC "regra que prevalece"; "lei do kernel fechada").
- **Experimento novo = YAML** em `experiments/autoencoder/`, lançado com `python -m experiments.ray.launch <yaml>`; chave nova vai em `RUNNER_ARGS`.
- **Whitening só com shrinkage** (`_pwr`/`_pw` atual), PCA fitado no treino (DEC "o _pw sem shrinkage colapsou").
- **Ambiente:** `LD_LIBRARY_PATH` na `scalable_FLIM`; relançar growth morto pode refazer pontos (`skip: state` não sobrevive a kill — confira "pulado(s)") (DEC "o relançamento das 18:24 NÃO pulou").
- **Dataset novo:** precisa de entrada em `scripts/spifil_paper_seeds.json` e `paper_classes: false`, senão KeyError (DEC "auditoria da receita").
- **Ckpt:** guardar o best, apagar o last.
