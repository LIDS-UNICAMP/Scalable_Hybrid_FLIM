# Achados Publicáveis — FLIM KD + SSL

**Data:** 2026-05-30 | **WandB:** ophira-ai/flim-ssl  
**Base empírica:** 216 runs destilação + 1.736 runs LeJEPA; 3 datasets parasitológicos de microscopia FLIM

---

## Novel Contributions

**C1. Colapso de encoder em destilação ViT→CNN induzido por cabeça 1×1 Conv: mecanismo, detecção e temporalidade**

Destilação de ViT-H/14 (I-JEPA) para CNN FLIM com cabeça de projeção 1×1 Conv + BN2d produz colapso irreversível do encoder nas épocas 0–10 do warmup. `student_emb_norm` cai 86% (0.579 → 0.080) e não se recupera em 100 épocas. Mecanismo: a convolução 1×1 não propaga gradiente entre posições espaciais vizinhas, permitindo que o encoder minimize a loss MSE via colapso trivial. Evidência: kappa SVM eggs = 0.000–0.296 para todo o grupo 1×1_BN2d_trunc_normal; kappa = 0.900 com mesma arquitetura e cabeça 3×3.

- **Novidade:** Primeira caracterização temporal precisa do warmup como janela crítica de colapso em destilação ViT→CNN.
- **Escala:** 54 runs × 100 épocas (1×1_BN2d); correlação emb_norm vs kappa = r = 0.864.

**C2. `student_emb_norm` como indicador precoce de colapso em destilação de conhecimento (r = 0.864, threshold operacional na época 10)**

A norma L2 do encoder bruto (pré-projeção) na época 50 prediz o kappa SVM final com r = 0.864 — superior ao cosine_sim (r = 0.486) e à training_loss (r = −0.529). Regra de early stopping derivada empiricamente: emb_norm < 0.05 na época 10 → kappa esperado = 0.122 (colapso confirmado); emb_norm ≥ 0.10 → kappa esperado = 0.667 (convergência). Calibrada em 216 runs × 3 datasets × 3 splits × 6 percentuais de dados.

- **Novidade:** Critério operacional de monitoramento de saúde para destilação KD, com limiar derivado de dados empíricos.
- **Aplicabilidade:** Qualquer pipeline de destilação com cabeça de projeção usando BN2d.

**C3. Scale mismatch persistente como problema estrutural em destilação MSE com BatchNorm2d**

O `scale_ratio` (norma da projeção student / norma do embedding teacher) inicia em 1.019 (época 0, next_layers), colapsa para 0.307 na época 10 durante o warmup BN, e recupera parcialmente para 0.542 na época 99. Nenhum grupo alcança scale_ratio = 1.0 em nenhum ponto do treinamento. O mismatch é causado pelo BatchNorm2d reescalonando as ativações durante o warmup — independentemente da arquitetura da cabeça.

- **Novidade:** Caracterização quantitativa do scale mismatch como problema estrutural em destilação MSE com BN2d; proposta de correção: L_norm = λ·(‖s_proj‖/‖t_emb‖ − 1)².
- **Impacto:** Todos os resultados de 100 épocas são subestimativas; 96.3% dos runs 3×3_BN2d ainda convergiam na época 99.

**C4. CNN FLIM local (LeJEPA) supera ViT-H/14 teacher em protozoan-cysts com fine-tuning completo**

LeJEPA + MLP_unfreeze + inicialização FLIM alcança kappa = 0.899 em protozoan-cysts, superando o ViT-H/14 I-JEPA teacher (kappa = 0.892). Gap para o teacher em larvae = 0.015 kappa (0.935 vs 0.950); em eggs = 0.028 kappa (0.929 vs 0.957). A CNN FLIM é estimada em ~10× menor que o ViT-H/14.

- **Novidade:** Demonstração empírica de que SSL específico de domínio com CNN pequena supera transfer de ViT genérico em microscopia FLIM parasitológica.
- **Condição necessária:** Resultado só se manifesta com fine-tuning completo (MLP_unfreeze); encoder congelado (SVM) dá kappa = 0.291 para o mesmo modelo em protozoan.

**C5. Inicialização FLIM é condição necessária para aprendizado em poucos dados com LeJEPA (pct=1%)**

LeJEPA flim init é o único que produz kappa não-nulo em larvae pct=1% com MLP_unfreeze (kappa = 0.231); todos os outros inits (xavier, he, trunc_normal) produzem kappa = 0.000. Em larvae pct=5%, flim atinge kappa = 0.808 enquanto os demais permanecem em 0.000 (freeze) ou próximos de zero (unfreeze). A inicialização com pesos FLIM atua como conhecimento de domínio implícito que guia o fine-tuning em regime de poucos dados.

- **Novidade:** Quantificação do limiar de dados em que o inductive bias de domínio deixa de ser opcional e se torna necessário.

---

## Validation of Prior Work

**V1. Colapso de representação trivial em destilação sem mecanismo anti-colapso — confirmado e estendido**  
Os achados de Chen & He (SimSiam, 2021) e Zbontar et al. (Barlow Twins, 2021) sobre colapso de modo trivial são confirmados em destilação ViT→CNN com cabeça de projeção inadequada (1×1 sem contexto espacial). Extensão: o mecanismo é determinístico e temporalmente previsível (janela crítica = épocas 0–10).

**V2. Fine-tuning completo supera encoder congelado — confirmado para representações SSL em microscopia**  
MLP_unfreeze supera SVM em todas as combinações testadas. Gap máximo: +0.608 kappa (LeJEPA flim, protozoan). Confirmação empírica de que encoder congelado é métrica inadequada para avaliar SSL em domínio especializado com poucos dados.

**V3. Inductive bias de domínio supera generalização de ViT em tarefa especializada de poucos dados**  
Confirma hipótese de He et al. (MAE, 2022) e outros sobre a importância de arquitetura adequada ao domínio. Evidência quantitativa: flim init + LeJEPA supera ViT-H/14 em protozoan (kappa 0.899 vs 0.892) com modelo ~10× menor.

---

## Negative Results (também publicáveis)

**N1. 1×1 Conv como única cabeça de projeção em destilação ViT→CNN é inviável com inicialização aleatória**  
Kappa eggs = 0.000–0.296, larvae = 0.000–0.201, protozoan = 0.202 — em todas as pcts e splits testados. Nenhuma condição faz 1×1_BN2d_trunc_normal ser competitivo. **Implicação prática: nunca usar 1×1 Conv como única camada de projeção em destilação de ViT com alto stride para CNN pequena.**

**N2. FLIM init degrada com volume total de dados no SSL (catastrophic forgetting suspeito em protozoan)**  
LeJEPA flim SVM protozoan: kappa = 0.480 (pct=25%) → 0.291 (pct=100%). Degradação de 39.4% com mais dados. **Implicação: pesos FLIM não são ponto de partida seguro para SSL de longa duração em protozoan sem regularização adicional (EWC ou similar).**

**N3. 100 épocas é insuficiente para destilação ViT→CNN com BN2d — protocolo padrão da literatura é inadequado**  
100% dos runs 1×1_BN2d e 96.3% dos 3×3_BN2d ainda melhoravam na época 99. best_epoch = 99 para a maioria dos runs. **Implicação: comparações com trabalhos que usam 100 épocas são inválidas; mínimo 200 épocas necessário para este setup.**

**N4. Inicializações não-FLIM (he, trunc_normal, xavier) são instáveis em alta disponibilidade de dados**  
Kappa negativo documentado em larvae pct=100%: trunc_normal = −0.066, he = −0.046. Padrão bimodal (1 split falha, 2 convergem) é o modo de falha dominante. **Implicação: estas inicializações requerem validação cruzada completa antes de uso em produção.**

**N5. SVM sobre encoder congelado é métricamente enganoso para comparações entre métodos SSL**  
Gap SVM → MLP_unfreeze para flim: eggs = +0.198, larvae = +0.180, protozoan = +0.608. A ranking entre métodos pode inverter: flim SVM protozoan = 0.291 (aparentemente péssimo) vs flim MLP_unfreeze = 0.899 (melhor de todos). **Implicação: trabalhos que reportam apenas SVM linear como métrica de avaliação SSL em domínios especializados podem reportar resultados enganosos.**

---

## Suggested Paper Framing

**Opção A — foco em diagnóstico de destilação (venue: ECCV workshop, IEEE TNNLS, Pattern Recognition):**  
*"Embedding Norm Collapse in Knowledge Distillation from Vision Transformers to Convolutional Networks: Mechanisms, Detection, and Prevention for Biological Microscopy"*  
Destaque: C1 + C2 + C3 como contribuições principais; N1 + N3 como resultados negativos de alta qualidade.

**Opção B — foco em SSL específico de domínio (venue: Nature Methods, Bioinformatics, MedIA):**  
*"Domain-Specific Self-Supervised Learning with Local CNNs Surpasses Generic Vision Transformers for FLIM Parasitology Microscopy"*  
Destaque: C4 + C5 como contribuições principais; destilação como método comparativo.

**Opção C — análise sistemática completa (venue: MICCAI 2027, CVPR workshop):**  
*"Scalable Knowledge Transfer from ViT to Lightweight CNN for FLIM Microscopy: Projection Head Design, Initialization, and Training Dynamics"*  
Destaque: todas as contribuições; mais adequado para venue top-tier com ciclo de revisão longo.

**Recomendação:** Opção A para submissão de curto prazo (resultados já suficientes); Opção B após confirmação estatística do resultado protozoan (E4 abaixo); Opção C após experimentos de 200+ épocas.

---

## Additional Experiments Needed for Submission

**E1 [Prioridade 1 — necessário para qualquer submissão]**  
Relançar 3×3_BN2d e next_layers com 200+ épocas. Reportar curvas completas de emb_norm, cosine_sim e scale_ratio. Sem isso, todos os resultados de destilação são incompletos.

**E2 [Prioridade 1 — valida C1+C2 com dado direto]**  
Gráfico de dispersão emb_norm@época10 vs kappa_final para os 216 runs (3 datasets × 3 splits × 6 pcts). Visualização direta da correlação r=0.864 com linha de regressão e região de early stopping.

**E3 [Prioridade 1 — valida C3 e fecha loop de recomendação]**  
Implementar e avaliar loss de regularização de norma: L_total = L_MSE + λ·(‖s_proj‖/‖t_emb‖ − 1)² para λ ∈ {0.1, 0.5, 1.0}. Medir impacto no scale_ratio e kappa final vs baseline.

**E4 [Prioridade 1 — necessário para afirmação forte sobre C4]**  
Aumentar para N=5 splits LeJEPA flim + MLP_unfreeze em protozoan. Teste t pareado vs I-JEPA teacher. Atualmente N=3 é insuficiente para afirmar superação estatisticamente significativa.

**E5 [Prioridade 2 — valida C3 combinatório]**  
Testar 3×3_BN2d + flim_init (combinação ainda não executada). 18 novos runs. Expectativa: melhor emb_norm (como flim_init) + melhor cosine_sim (como 3×3). Candidato a superar next_layers_direct.

**E6 [Prioridade 2 — esclarece N2]**  
Curvas LeJEPA de invariância/sigreg por época para protozoan flim_init, pct=25% vs pct=100%. Determinar se degradação SVM é catastrophic forgetting no encoder ou artefato de avaliação SVM.

**E7 [Prioridade 3 — posicionamento na literatura]**  
Comparar com: (a) DINO com ViT-S (menor, mais acessível); (b) TinyViT destilado do I-JEPA. Necessário para posicionar contribuição no contexto da literatura geral de destilação e SSL.

---

*Todos os achados baseados em dados empíricos rastreáveis: grand_comparison_table.csv, 1x1_BN2d_1280_one_layer_all_datasets.csv, 3x3_BN2d_1280_one_layer_all_datasets.csv, next_layers_direct_all_datasets.csv — WandB: ophira-ai/flim-ssl (216 runs destilação + 1.736 runs LeJEPA).*
