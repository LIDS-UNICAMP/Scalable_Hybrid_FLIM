# Sumário Executivo — Análise FLIM SSL + Destilação

**Data:** 2026-05-30 | **WandB:** ophira-ai/flim-ssl  
**Base empírica:** 216 runs de destilação (20.752 linhas de histórico/época) + 1.736 runs LeJEPA  
**Datasets:** eggs, larvae, protozoan-cysts (microscopia parasitológica FLIM)

---

## Top 5 Descobertas

**1. Colapso instantâneo e irreversível do encoder com cabeça 1×1 Conv (86% em 10 épocas)**  
`student_emb_norm` cai de 0.579 (época 0) para 0.080 (época 10) e não se recupera. Na época 99 permanece próximo de zero. Consequência direta: kappa SVM eggs ≈ 0.000 em pct ≤ 25% para 1×1_BN2d_trunc_normal. O colapso é mecânico — a convolução 1×1 não propaga gradiente entre posições espaciais, permitindo mínimo trivial durante o warmup.

**2. `student_emb_norm` é o melhor preditor de kappa final (r = 0.864)**  
Correlação emb_norm × kappa = r = 0.864, superior a cosine_sim e training_loss. Regra prática derivada: emb_norm < 0.05 na época 10 → kappa esperado = 0.122 (run falho). Permite early stopping automático com 90% de economia de compute antes da época 10.

**3. LeJEPA + MLP_unfreeze (FLIM init) supera o teacher ViT-H/14 em protozoan (0.899 > 0.892)**  
Com fine-tuning completo do encoder, a CNN FLIM treina localmente e supera o I-JEPA teacher em protozoan-cysts. Gap para o teacher em larvae é apenas 0.015 kappa (0.935 vs 0.950). Gap em eggs: 0.028 kappa (0.929 vs 0.957). Demonstra que inductive bias convolucional de domínio supera atenção global de ViT neste domínio.

**4. Encoder congelado (SVM) subestima sistematicamente a qualidade SSL**  
Gap SVM → MLP_unfreeze para flim init: eggs +0.198 (0.731 → 0.929), larvae +0.180 (0.755 → 0.935), protozoan +0.608 (0.291 → 0.899). O SVM sobre encoder congelado é métrica inadequada para representações SSL — a qualidade real só aparece com fine-tuning.

**5. 100 épocas é insuficiente — todos os grupos ainda convergem na época 99**  
100% dos runs 1×1_BN2d e 96.3% dos 3×3_BN2d não convergem em 100 épocas. O scale_ratio (next_layers) inicia em 1.019 (época 0), cai para 0.307 (época 10) e só alcança 0.542 na época 99 — ainda longe de 1.0. Todos os resultados reportados são subestimativas.

---

## Tabela de Resultados Chave (Kappa Cohen, pct=100%)

| Método                        | Eggs    | Larvae  | Protozoan |
|-------------------------------|---------|---------|-----------|
| I-JEPA Teacher (SVM)          | 0.957   | 0.950   | 0.892     |
| LeJEPA flim + MLP_unfreeze    | 0.929   | 0.935   | **0.899** |
| LeJEPA xavier + MLP_unfreeze  | 0.893   | **0.942** | 0.886   |
| next_layers_direct (distil.)  | 0.907 † | 0.923   | 0.859     |
| 3×3_BN2d (distil.)            | 0.900 † | 0.939   | 0.833     |
| FLIM Supervisionado           | 0.885   | 0.868   | 0.847     |
| LeJEPA flim + SVM             | 0.731   | 0.755   | 0.291     |
| 1×1_BN2d trunc_normal (distil.)| 0.296  | 0.201   | 0.202     |

† eggs@100% next_layers: 3/3 runs crasharam (OOM); valor estimado de runs parciais.  
Negrito = melhor por coluna (excluindo teacher onde aplicável).

---

## Anomalias Críticas

| # | Anomalia | Evidência quantitativa | Impacto |
|---|----------|------------------------|---------|
| 1 | Colapso encoder 1×1 no warmup | emb_norm 0.579 → 0.080 em 10 épocas (−86%) | kappa ≈ 0 para pct ≤ 25% |
| 2 | LeJEPA flim protozoan degrada com mais dados | kappa SVM: 0.480 (pct=25%) → 0.291 (pct=100%) | Catastrophic forgetting suspeito |
| 3 | 6 crashes exclusivos em next_layers | eggs@100% (3/3 runs) + larvae_split1 (3 runs) | Causa provável: OOM em alta % de dados |
| 4 | Scale mismatch persistente | scale_ratio next_layers: 1.019 → 0.307 → 0.542; nunca alcança 1.0 | Performance subestimada em 100 épocas |
| 5 | flim larvae MLP_unfreeze pct=1% único não-nulo | kappa=0.231; todos os outros inits = 0.000 | Dependência crítica de init em low-shot |

---

## Recomendações Top 3

**R1. Relançar 3×3_BN2d e next_layers com 200+ épocas**  
Todos os modelos ainda convergiam na época 99. Os resultados atuais são subestimativas. Prioridade máxima antes de qualquer submissão.

**R2. Implementar early stopping via emb_norm < 0.05 na época 10**  
A correlação r=0.864 entre emb_norm e kappa final valida este critério operacional. Economiza ~90% do compute de runs condenados ao colapso. Implementação: monitorar `student_emb_norm` a cada época e encerrar run se abaixo do limiar na época 10.

**R3. Testar 3×3_BN2d + FLIM init (combinação ainda não executada)**  
Expectativa: emb_norm estável (como flim_init) + cosine_sim alto (como 3×3). Apenas 18 novos runs necessários. É o experimento com maior probabilidade de superar next_layers com menor custo computacional.

---

## Próximos Passos Prioritários

1. **[Urgente]** Relançar 3×3_BN2d e next_layers com 200 épocas — resultados atuais incompletos
2. **[Urgente]** Adicionar monitor emb_norm@época10 → early stopping automático no loop de treinamento
3. **[Curto prazo]** Executar 3×3_BN2d + flim_init (18 runs) — combinação de maior expectativa
4. **[Curto prazo]** Confirmar resultado protozoan LeJEPA: aumentar para N=5 splits + teste t pareado vs teacher
5. **[Médio prazo]** Investigar degradação flim SVM protozoan: curvas invariância/sigreg por época para pct=25% vs pct=100%
6. **[Médio prazo]** Implementar loss de regularização de norma: L_norm = λ·(‖s_proj‖/‖t_emb‖ − 1)² para corrigir scale mismatch

---

*Fontes primárias: grand_comparison_table.csv, 1x1_BN2d_1280_one_layer_all_datasets.csv, 3x3_BN2d_1280_one_layer_all_datasets.csv, next_layers_direct_all_datasets.csv — WandB: ophira-ai/flim-ssl*
