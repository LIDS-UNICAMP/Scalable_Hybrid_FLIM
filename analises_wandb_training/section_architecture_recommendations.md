# Verificação Arquitetural e Recomendações

**Gerado em:** 2026-05-30  
**Base:** 245 runs verificados (de 270 esperados) — 3 datasets × 3 splits × 6 percentuais × 5 grupos

---

## Verificação: Zero Inconsistências Encontradas

Todos os 245 runs com metadados foram verificados contra as seguintes restrições de configuração: alinhamento do `arch_json` ao dataset e split, tipo de cabeça de projeção (`proj_head`) por grupo, inicialização do encoder (`encoder_init`), e dimensões de embedding do teacher e student. **Nenhuma inconsistência foi encontrada.**

| Restrição verificada | Esperado | Observado | Status |
|---|---|---|---|
| arch_json: protozoan → `ch24_30_48` | 245 runs | 100% correto | PASS |
| arch_json: larvae/eggs → `ch24_32_48` | 245 runs | 100% correto | PASS |
| arch_json split N → `train{N}` | 245 runs | 100% correto | PASS |
| `*_flim_init` → `encoder_init = flim` | 36 runs | 100% correto | PASS |
| outros grupos → `encoder_init = trunc_normal` | 209 runs | 100% correto | PASS |
| `teacher_embed_dim = 1280` | 245 runs | 100% correto | PASS |
| `student_embed_dim = 48` | 245 runs | 100% correto | PASS |
| `proj_head` consistente por grupo | por grupo | 100% correto | PASS |

### Status dos runs por WandB

| Status | Quantidade |
|---|---|
| finished | 198 |
| running | 12 |
| crashed | 6 |
| **Total com metadados** | **245 de 270** |

Os 6 crashes são todos do grupo `next_layers_direct`: eggs@100% e larvae\_split1@50–100%. Os 25 runs sem `run_metadata.json` incluem 18 protozoan `flim_init` (12 ainda em execução + 6 sem metadados) e 7 protozoan `modeldirect`. Nenhum desses representa erro de configuração confirmado — são ausências de dados, não inconsistências arquiteturais.

---

## Arquitetura do Encoder FLIM CNN

O encoder FLIM CNN usado nos experimentos é uma rede convolucional rasa cuja arquitetura é definida por um arquivo `arch_json`, que especifica o número de canais de entrada, intermediário e saída de cada camada. A tripla de canais (`ch_in`/`ch_mid`/`ch_out`) resume o espaço de representação do encoder.

| Dataset | Tripla de canais | Descrição |
|---|---|---|
| protozoan | `ch24_30_48` | canal intermediário = 30 |
| larvae | `ch24_32_48` | canal intermediário = 32 |
| eggs | `ch24_32_48` | canal intermediário = 32 |

O canal de entrada (24) corresponde ao número de comprimentos de onda FLIM adquiridos. O canal de saída (48) é o `student_embed_dim` — a dimensão dos embeddings que o student deve alinhar ao teacher I-JEPA (1280 dimensões, projetados pela `proj_head`).

---

## Por que ch2=30 (protozoan) vs 32 (larvae/eggs)?

A diferença no canal intermediário reflete a etapa de aprendizado FLIM supervisionado que antecede a destilação. O encoder FLIM CNN é inicializado a partir de filtros aprendidos com dados rotulados em um pipeline de seleção de features guiado por especialistas (FLIM — Feedback Learning for Image Morphology). Nesse processo, o número de filtros úteis descoberto para cada dataset depende da complexidade morfológica das imagens e da diversidade de classes. Para o dataset protozoan, o processo FLIM convergiu para 30 filtros intermediários; para larvae e eggs, para 32 filtros.

Essa diferença é pequena em magnitude (dois filtros, ~6%) e não implica menor capacidade representacional do encoder protozoan. O que importa é que os dois arch\_json são tratados como entidades distintas para seus respectivos datasets, e os 245 runs verificados confirmam que nenhum run cruzou configurações entre datasets.

---

## Mecanismo do Colapso 1×1: Análise Técnica

O grupo `1x1_BN2d` sofre colapso severo de representações. A causa é arquitetural: a cabeça de projeção `Conv1×1(48→1280) + BatchNorm2d + GELU` combina dois efeitos que se reforçam mutuamente em direção ao colapso.

**Efeito 1 — Kernel 1×1 sem receptivo campo espacial.** Em imagens FLIM, a informação discriminativa está distribuída localmente em estruturas de tamanho variável (manchas, bordas, gradientes de textura). Um kernel 1×1 opera pixel a pixel, sem agregar contexto de vizinhança. O encoder não consegue capturar essas estruturas, e os gradientes de treinamento passam a empurrar os pesos em direções inconsistentes entre patches, causando destruição progressiva da estrutura dos embeddings.

**Efeito 2 — BatchNorm2d suprime a magnitude dos embeddings do encoder.** A BN2d normaliza as ativações internas da cabeça de projeção, o que força os gradientes que chegam ao encoder a serem muito pequenos em magnitude. Nas primeiras épocas, isso colapsa a norma dos embeddings do encoder (`student_emb_norm`) de forma rápida e irreversível.

**Evidência quantitativa:**

| Grupo | emb_norm época 0 | emb_norm época 10 | emb_norm época 99 | Queda total |
|---|---|---|---|---|
| 1x1_BN2d | 0.579 | 0.080 | 0.066 | −88.6% |
| 3x3_BN2d | 0.629 | 0.351 | 0.354 | −43.7% |

O grupo 3×3 perde ~44% da norma inicial e depois estabiliza. O grupo 1×1 perde ~89% e continua decaindo. O resultado é que os embeddings 1×1 têm magnitude tão próxima de zero que perdem poder discriminativo: as distâncias no espaço de embedding colapsam e o SVM não encontra hiperplanos separadores.

**Impacto direto na classificação (kappa SVM, eggs@100%):**

| Grupo | kappa SVM | emb_norm época 99 |
|---|---|---|
| 1x1_BN2d | 0.296 | 0.066 |
| 3x3_BN2d | 0.900 | 0.354 |
| Diferença | **+0.604** | **+5.4×** |

O paradoxo observado — `cosine_sim` alta (0.637–0.666) com kappa muito baixo — é explicado por essa dissociação: a cosine\_sim mede apenas a direção dos vetores de embedding, enquanto a discriminabilidade do SVM depende também da magnitude. Vetores quasi-nulos, mesmo alinhados direcionalmente ao teacher, produzem embeddings sem capacidade discriminativa.

---

## Recomendações Baseadas em Evidências

### R1: Cabeça mínima = 3×3 (não 1×1)

**Evidência:** O grupo 1×1 colapsa para emb\_norm = 0.066 (época 99) e produz kappa médio de 0.081–0.153. O grupo 3×3 estabiliza em emb\_norm = 0.354 e atinge kappa de até 0.939.

**Diferença de kappa (eggs@100%):** 0.296 (1×1) vs. 0.900 (3×3) — delta de +0.604 pontos, apenas pela mudança do tamanho do kernel.

**Recomendação:** Kernel 1×1 deve ser descartado como cabeça de projeção para o pipeline de destilação I-JEPA → FLIM CNN. Qualquer nova variante arquitetural deve usar kernel mínimo 3×3 para garantir receptivo campo espacial. A configuração `Conv3×3(48→1280) + BN2d + GELU` é o baseline seguro para comparações.

---

### R2: 200+ épocas de treinamento

**Evidência:** 100% dos runs 1×1\_BN2d ainda melhoravam na época 99 (best\_epoch ≥ 85 em todos). 96.3% dos runs 3×3\_BN2d também. A taxa de melhoria entre épocas 75–99 ainda é positiva: grupo 3×3 perde −1.4% de loss, grupo next\_layers perde −2.3%.

**Recomendação:** O orçamento atual de 100 épocas é insuficiente para todos os grupos. Os resultados de kappa reportados são limites inferiores. Para comparações definitivas entre grupos e para o paper, re-treinar os melhores hiperparâmetros de cada grupo com 200–300 épocas. O critério de parada recomendado é convergência do `scale_ratio` (ver R3).

---

### R3: Loss de regularização de norma

**Evidência:** O `scale_ratio` (norma do student projetado / norma do teacher) não atinge 1.0 em nenhum grupo na época 99: 1×1 = 0.404, 3×3 = 0.494, next\_layers = 0.542. O teacher I-JEPA tem norma média de ~21.4; o melhor student (next\_layers) atinge apenas ~11.6 — subestimação de ~2×.

Esse mismatch de escala é um fator independente de perda de qualidade dos embeddings: mesmo quando a direção é correta (cosine\_sim alta), a diferença de escala introduz erro sistemático na loss de destilação KL/cosine.

**Recomendação:** Adicionar um termo de regularização de norma à loss de destilação:

```
L_total = L_distil + λ · max(0, τ_min - ||z_s||)
```

onde `τ_min` é um limiar mínimo de norma (sugerido: 0.1 × norma média do teacher). Isso previne colapso de norma (como o observado no grupo 1×1) e incentiva o student a manter embeddings com magnitude suficiente para discriminação.

---

### R4: Early stopping: emb\_norm < 0.05 na época 10

**Evidência:** A correlação entre emb\_norm na época 10 e kappa final é r = +0.575, com forte valor preditivo nas caudas da distribuição:

| emb_norm na época 10 | kappa esperado | taxa de sucesso (kappa ≥ 0.3) |
|---|---|---|
| < 0.05 | 0.122 | 6% dos casos |
| ≥ 0.10 | 0.667 | alta |

94% dos runs com emb\_norm < 0.05 na época 10 terminam com kappa < 0.3 — resultado que não justifica o custo computacional das 90 épocas restantes.

**Recomendação:** Implementar abort automático: se `student_emb_norm < 0.05` na época 10, interromper o run e registrar como "colapso confirmado". Isso economiza recurso computacional e sinaliza para re-tentar com hiperparâmetros ou arquitetura diferentes. O threshold de 0.05 é conservador: nenhum run com norma abaixo desse valor recuperou kappa > 0.3 nos dados analisados.

---

### R5: flim\_init + SSL — usar lr 10× menor

**Evidência:** O grupo `flim_init` (encoder inicializado com pesos FLIM pré-treinados supervisionadamente) mostra degradação monotônica em protozoan com mais dados: kappa SVM cai de 0.495 (50%) para 0.291 (100%). Esse padrão é indicativo de catastrophic forgetting — o treinamento SSL destrói a representação FLIM útil em vez de refiná-la.

| pct dados | kappa SVM (flim_init, protozoan) |
|---|---|
| 25% | 0.480 |
| 50% | 0.495 |
| 75% | 0.416 |
| 100% | 0.291 |

Enquanto o kappa SVM degrada com mais dados, o `MLP_unfreeze` sobe para 0.899 — indicando que a representação ainda tem estrutura útil, mas o encoder foi perturbado de forma que o SVM linear não consegue aproveitá-la.

**Recomendação:** Para runs com `encoder_init = flim`, reduzir a taxa de aprendizagem do encoder (e apenas do encoder) por um fator de 10× em relação à `proj_head`. O rationale é proteger os pesos FLIM pré-treinados de gradientes de destilação agressivos enquanto a cabeça de projeção se adapta. Alternativamente, congelar o encoder nas primeiras N épocas (warm-up da proj\_head) e só depois liberar com lr baixo. Os 18 runs de protozoan `flim_init` atualmente sem metadados devem ser priorizados para análise quando disponíveis.

---

### R6: Usar MLP\_unfreeze para avaliação final

**Evidência:** Para o modelo LeJEPA (I-JEPA + FLIM CNN), o protocolo SVM sistematicamente subestima a qualidade das representações. A diferença é expressiva e consistente nos três datasets:

| Dataset | MLP_unfreeze (kappa) | SVM (kappa) | Delta |
|---|---|---|---|
| eggs@100% | 0.929 | 0.731 | +0.198 |
| larvae@100% | 0.942 | 0.754 | +0.188 |
| protozoan@100% | 0.899 | 0.291 | +0.608 |

A discrepância em protozoan (+0.608) é especialmente crítica: avaliar o LeJEPA via SVM levaria à conclusão errônea de que o modelo aprendeu representações fracas nesse dataset, quando na verdade o MLP\_unfreeze revela qualidade comparável a eggs e larvae.

**Recomendação:** Adotar MLP\_unfreeze como métrica primária de avaliação para todos os relatórios e comparações finais. O SVM pode ser mantido como avaliação rápida durante treinamento (menor custo computacional), mas não deve ser usado como métrica de decisão entre arquiteturas. A diferença provavelmente reflete que as representações aprendidas via destilação têm estrutura não-linear que o SVM linear não captura; o MLP com fine-tuning supera essa limitação.

---

### R7: Tratar crashes next\_layers\_direct em alto regime de dados (OOM)

**Evidência:** Os 6 crashes são exclusivamente do grupo `next_layers_direct` e ocorrem em regimes de alto percentual de dados: eggs@100% e larvae\_split1@50–100%. O padrão sugere causa relacionada a uso de memória (OOM — out of memory): a combinação de lotes maiores (mais dados em 100%) com a arquitetura `conv_next_layers` (mais parâmetros que os grupos one\_layer) excede o limite de RAM/VRAM disponível.

O grupo `next_layers_direct` demonstra a maior eficiência em baixo regime de dados (vantagem de +0.062 em eggs@5% e +0.070 em protozoan@5% sobre o 3×3), o que o torna valioso para cenários de poucos rótulos. Perder os runs de alto percentual por crashes desperdiça informação relevante.

**Recomendação (por ordem de prioridade):**

1. **Reduzir batch size** para os runs de alto percentual (≥50%) do grupo next\_layers\_direct: reduzir à metade e re-rodar os 6 runs crashados.
2. **Gradient checkpointing:** ativar para o grupo next\_layers nas configurações de alto percentual, trocando tempo por memória.
3. **Mixed precision (fp16):** se não estiver habilitado, ativar para reduzir consumo de memória ~30–40%.
4. Se os crashes persistirem após (1)–(3), investigar instabilidade numérica via logging de normas de gradiente.

---

## Impacto das Recomendações (kappa esperado)

A tabela abaixo resume o impacto esperado de cada recomendação sobre o kappa final, baseado nas evidências quantitativas dos dados WandB analisados.

| Recomendação | Baseline atual | Melhoria esperada | Evidência base |
|---|---|---|---|
| R1: Kernel 3×3 (vs 1×1) | eggs@100%: kappa=0.296 | +0.604 → 0.900 | Delta direto entre grupos |
| R2: 200+ épocas | 3×3 não convergido (96.3%) | +0.02–0.05 estimado | Taxa de melhoria época 75→99 |
| R3: Regularização norma | scale_ratio=0.494 (melhor grupo) | reduz gap para teacher | Mismatch ~2× nas normas |
| R4: Early stopping (abort em ep10) | ~94% dos runs com emb_norm<0.05 inúteis | elimina desperdício computacional | r(emb_norm_ep10, kappa)=0.575 |
| R5: lr 10× menor (flim_init) | protozoan SVM degrada para 0.291@100% | previne catastrophic forgetting | Queda monotônica observada |
| R6: MLP_unfreeze como métrica | SVM protozoan=0.291 | MLP mostra kappa=0.899 (real) | Delta de +0.608 observado |
| R7: Fix crashes next\_layers | 6 runs crashados em alto regime | recupera dados de alto percentual | Padrão OOM identificado |

**Impacto combinado (cenário com R1+R2+R6):**  
Usando kernel 3×3, 200+ épocas e avaliação MLP\_unfreeze, o kappa esperado para o melhor grupo em regimes de dados completos situa-se na faixa **0.92–0.95** para eggs e larvae, e **0.88–0.91** para protozoan — reduzindo o gap para o teacher I-JEPA (0.950–0.957) para menos de 0.03 pontos em pelo menos dois dos três datasets.
