# Missão SPiFiL Progressivo B0–B4 — especificação do usuário (17/11/2026)

> Texto-guia para TODOS os agentes. Regras de ouro herdadas: ga_search/modehb/FILA_E_INSTRUCOES.md (R1 SVM = core.svm.fit_svm C=100; R2 SVM original no teste; teste NUNCA na seleção; R9 validar de ponta a ponta antes de produção; GPU nunca parada; não apagar nada; pt-BR).
> Inventário inicial: ga_search/modehb/reports/all_results_today.md (snapshot; confira os pontos críticos nos artefatos originais). Dados do athena na Jaci: /dados/home/moliveira/exps/queues/relatorios_athena/all_results_bundle/athena/.

## Decisão do usuário (17/11 ~01:30) — cabeça de alta dimensão OBRIGATÓRIA
"A 1280 ou mais deve existir porque ela separa bem quando for usar no futuro. Vetores de alta dimensão."
- Todo backbone tem cabeça com embedding de dimensão ≥ 1280. Nunca remover nem reduzir abaixo de 1280 para caber no orçamento.
- O custo da cabeça continua contado no orçamento Bk. Para caber, otimize a forma da cabeça (pooling `moments`/meanstd, projeção de baixo rank, bilinear compacto, etc.) mantendo saída ≥ 1280, nunca a dimensão.

## Adendo (17/11): Attention SEM pesos treináveis — ver ATTENTION_ADDENDUM.md
Agente 6 (novo): avalia a viabilidade com os outros agentes, implementa os operadores isolados (zero parâmetros), testa e planeja E0–E8 na athena GPU 3. Os papéis dos Agentes 2/3/4/5 estão no adendo.

## Ordem e dependências
1. Agente 5 (infra) começa primeiro e fica ativo como orquestrador de filas.
2. Agente 1 (diagnóstico) roda em paralelo (CPU, sem GPU).
3. Agente 2 (inicialização abstrata) — só implementação/experimentos após gate do Agente 1 (pode fazer revisão de literatura e desenho antes).
4. Agente 3 (crescimento B0–B4) — após gate do Agente 2.
5. Agente 4 (MO-DEHB hierárquico) — após gate do Agente 3.
6. Agente CHECK: audita cada gate (independente; não confia em resumos, abre os artefatos).
Paralelizar o máximo possível sem violar gates.

## Regras absolutas de máquinas (Agente 5; prioridade sobre os demais)
- ATHENA GPU 2 (49 GB): NÃO MEXER (não encerrar, suspender, mover, alterar CUDA_VISIBLE_DEVICES, configs, checkpoints, fila; não ocupar com testes novos).
- ATHENA GPU 3 (49 GB): DEDICAR À NOVA PESQUISA. Inventariar processos (nossos × terceiros); interromper com segurança os nossos (checkpoint/entre jobs; nunca kill -9 indiscriminado); migrar para a fila da Jaci; nunca encerrar processos de terceiros.
- JACI (96 GB): NÃO MEXER nos experimentos existentes (não encerrar, pausar, reiniciar, substituir fila, reservar recursos já alocados, mudar configs ativas). Só ADICIONAR à fila os jobs migrados da GPU 3, sem preempção, retomando (nunca do zero).
- Migração: rsync sobre SSH, staging, sem --delete, hash dos críticos, validação de leitura e de resume (sem treino produtivo), publicação atômica, original preservado no athena.
- Travas contra dois controladores sobre o mesmo search_state. Fila persistente/idempotente (queued/running/checkpointed/failed/completed).
- Entregáveis A5: infrastructure_inventory.md, gpu3_migration_manifest.csv, jaci_resume_queue.json, athena_gpu3_research_queue.json, checkpoint_integrity_report.md, resume_validation_report.md, execution_status.md.
- Fila GPU 3 por fases: A testes de implementação; B ablações (RS, Flowers, HAM, parasitas raw/pre); C crescimento B0→B4; D busca evolutiva. Fase seguinte só com gate aprovado.
- HPC: limites de threads CPU/BLAS (OPENBLAS/OMP/MKL), profiling do pipeline, BF16 validado com FP32 nas ops sensíveis, OOM → diagnóstico (sem repetição infinita); mudança científica por memória = variante distinta registrada.

## Agente 1 — Auditoria científica e planejamento (sem modificar produção, sem GPU, sem treino)
Fontes: all_results_today.md (inventário), ga_search/modehb/ + configs, MO-DEHB/LEMONADE/PNAS-CS, pooling_head.py (EmbeddingHead, CovReadout), init SPiFiL (FisherScorer, Mahalanobis, FPS, SoftHebb), AE local, LeJEPA+SIGReg, relatórios/rankings de validação, checkpoints, genótipos, receitas, estados.
Diagnóstico: A Arquitetura (prof., largura, params, FLOPs × κ; redução espacial; blocos distintos?; conv × dwsep × inception; pequenos dominam grandes?). B Inicialização SPiFiL (como por camada; diversidade dos patches profundos; nº efetivo de candidatos, rank, redundância; SIFT/FPS, superpixels, Fisher, Mahalanobis; supervisionado × label-free; filtros mortos/colapso). C Aprendizagem (AE local por bloco; joint × decoder_first × noae; LeJEPA+SIGReg sobre os pesos SPiFiL; WiSE e frequência de α=0; convergência × fidelidade 5/15/45/100/200; vantagem indevida dos pequenos nas promoções). D Embedding (ORIG, COV, GTP separados; por que mapas intermediários com covariância vão bem e o 1280 nem sempre; normalização, escala, pooling, espectro, redundância, dimensão; encoder × cabeça). E Generalização (splits; HAM10000 por lesion_id; duplicatas/vazamento; parasitas fundo preto × raw; não comparar protocolos incompatíveis; seleção por validação × melhor retrospectivo no teste).
Quantitativas: 1 κ val × params/FLOPs/profundidade; 2 ORIG × COV; 3 benefício marginal por bloco; 4 curvas de treino e WiSE por arquitetura; 5 eficiência das mutações widen/deepen/residual/pooling/grow; 6 melhores checkpoints por orçamento; 7 selecionadas por validação × melhores só no teste; 8 diferenças textura/objeto/cena aérea/dermatoscopia. Separar medido × correlação × hipótese × causalidade.
Sementes por dataset (só treino/validação): B0 ≤300k, B1 ≤800k, B2 ≤1,5M, B3 ≤3M, B4 ≤5,5M (lacunas registradas). Prioridade RS, HAM10000, Flowers; depois Eggs, Larvae, Protozoan (raw/pre separados); Coconut e AID como controle.
Hipóteses testáveis (controle, tratamento, VI, métrica, nº seeds, orçamento, critério): profundidade ótima por domínio; insuficiência de patches simples em camadas profundas; protótipos abstratos/invariantes; preservação por aprendizagem local; fusão intermediária no embedding final; herança × reinicialização; saturação antes de 5,5M.
Papers: Coates 2011 (https://proceedings.mlr.press/v15/coates11a.html), PCANet (https://arxiv.org/abs/1404.3606), Belilovsky 2019 (https://proceedings.mlr.press/v97/belilovsky19a.html), EfficientNet (https://proceedings.mlr.press/v97/tan19a.html), MO-DEHB (https://arxiv.org/abs/2305.04502); literatura adicional registrada.
Entregáveis: diagnosis.md, seed_registry.csv, experiment_matrix.csv, research_hypotheses.md, evaluation_contract.md — tudo rastreável a dataset, split, arquitetura, checkpoint, params, fidelidade e leitura.
Gate → Agente 2: diagnóstico concluído, protocolos padronizados, sementes identificadas, experimentos priorizados (não só impressões qualitativas).

## Agente 2 — SPiFiL abstrato e aprendizagem local progressiva
Interface KernelInitializer.fit(prefix, train_data, block_spec, context) (shapes corretos, origem registrada, testável). Novos métodos em módulos opcionais; NÃO modificar SPiFiL histórico.
SPiFiL-0 (baseline atual: patches, superpixels, FPS, SIFT, Fisher, Mahalanobis, normalização; registrar nº candidatos, cobertura, normas, rank, redundância). SPiFiL-1 protótipos hierárquicos (k-means/++, PCA/SVD, clustering com whitening, FPS em protótipos, diversidade/representatividade, SoftHebb). SPiFiL-2 invariância entre vistas (correspondências locais, consistência de atribuições à la SwAV, VICRegL, estabilidade, diversidade, repetibilidade; invariâncias adequadas ao domínio). SPiFiL-3 contextual/preditiva (masked feature reconstruction, context→target à la I-JEPA, AE de features, protótipos latentes, anti-trivial; prefixo congelado; transformação latente→kernel explícita com dimensão/suporte/canais). SPiFiL-4 composição e herança (preservar filtros, novos por SPiFiL-1/2/3, residual ~identidade, herança, aprendizagem local, preservação de representação) — demonstrar contra crescimento padrão.
Aprendizagem local: AE local principal + noae controle; joint, decoder_first, pixel, features, masked, preditiva; descarte do decoder; protect freeze/lowlr/full. Global: LeJEPA+SIGReg real do repo como baseline (outras losses só como braços identificados).
Experimentos: começar RS e Flowers; fixar arquitetura e variar init; depois fixar init e variar profundidade (3,4,6; 8 após validação); SPiFiL-0..3; seleção baseline × estabilidade/diversidade; AE × noAE; orçamento pequeno e promoção; sem cartesiano completo sem priorização do A1; controlar params, FLOPs, resolução, dados, steps, seeds auxiliares.
Diagnósticos por camada: shapes, nº e rank de candidatos, ativação, variância, % canais mortos, redundância, estabilidade entre vistas, similaridade antes/depois de AE e SSL, energia espectral, rank efetivo, qualidade do embedding final, sondas intermediárias só diagnósticas; uso de rótulos explícito + versão label-free.
Papers: De Souza 2020 FLIM (doi 10.1109/SIBGRAPI51738.2020.00024), PCANet, Coates, SoftHebb (https://arxiv.org/abs/2107.05747), SwAV (https://arxiv.org/abs/2006.09882), VICRegL (NeurIPS 2022), I-JEPA (https://arxiv.org/abs/2301.08243), MAE (CVPR 2022), Belilovsky 2019 — princípio, adaptação CNN/SPiFiL, custo, limitações, hipótese verificável. Não apresentar SPiFiL-1..4 como comprovados.
Entregáveis: interface modular, implementação isolada, testes unitários/integração, reprodutibilidade, ablações por dataset/profundidade, ranking em validação, abstract_spifil_study.md, configs versionadas, checkpoints aprovados.
Gate → Agente 3: ≥1 mecanismo novo com benefício reproduzível OU evidência clara de que não supera o SPiFiL tradicional (registrar negativos).

## Agente 3 — Backbones autocontidos B0–B4
Orçamentos máximos B0 300k, B1 800k, B2 1,5M, B3 3M, B4 5,5M (encoder + cabeça 1280; decoders/projetores registrados à parte; proibido > 5,5M). Estágios: stem SPiFiL, local, intermediário, abstrato, head 1280 (nem todos obrigatórios; profundidade evolutiva). Operadores existentes preservados (conv, dwsep, inception, residual, ativações, max/avg/GeM/moments, BN/ZCA/DBN, SoftHebb). SPiFiL-0..4 conforme profundidade/comportamento.
Crescimento: width, depth (morfismo), residual (~0), compound (resolução fixa), structured (inception/dwsep), stage-aware, abstract init. Herança/morfismo: registrar ancestral, mecanismo, pesos herdados/novos, shapes, params, diferença pai×filho antes do treino (eval(), reprodutível); não alegar preservação exata quando não houver; herança × reinicialização na mesma arquitetura.
Embedding: GAP, GeM, moments, multiescala, projeções leves intermediárias, concat→1280, compact bilinear/2ª ordem com dimensão fixa, regularização; cabeça SEMPRE 1280; custo real; não usar CovReadout externo como substituto silencioso.
Famílias/linhagens: por porte, melhor por validação, eficiente em FLOPs, diverso, exploratório; DAG de ancestralidade; Bk autocontido.
Experimentos A–H: largura; profundidade; width+depth; por estágio; residual; init abstrata; herança × sem; cabeça multiescala × original — κ val, F1, rank, FLOPs, latência, memória, tamanho, custo; curvas de saturação.
Domínios: parasitas 2–4 blocos (maiores como hipótese); Flowers 3–6 e multiescala; RS contexto/multiescala/profundidade moderada × largura; HAM textura/cor/contexto; AID e Coconut como controles.
Papers: Net2Net, LEMONADE, EfficientNet, Once-for-All, FPN, Bilinear CNN, Compact Bilinear Pooling.
Entregáveis: BackboneSpec/GrowthSpec versionados, checkpoints autocontidos, registro de linhagens, testes de herança, gráficos κ×params×FLOPs, comparação de mecanismos, saturação, backbone_growth_study.md.
Gate → Agente 4: expansões testadas, representação final padronizada, ≥1 experimento controlado por domínio prioritário.

## Agente 4 — MO-DEHB hierárquico B0–B4
Genoma hierárquico: ArchitectureSpec, InitializationSpec, LocalLearningSpec, GrowthSpec, GlobalTrainingSpec (reparos condicionais; não vetor contínuo indiferenciado). Ordem de 17 passos por candidato (dataset/Bk → ancestral por validação → mutação → init dos blocos novos → build/params/FLOPs → herança → preservação → SPiFiL → AE local → SSL global → WiSE → avaliação ORIG na validação → estabilidade/colapso/latência/FLOPs → filho × ancestral mesmo protocolo → população/Pareto → promoção multi-fidelidade → checkpoint/linhagem/config).
Fitness nova versão: max κ val (ORIG: embedding 1280, core.svm.fit_svm, C=100, sem scaler) e min FLOPs; COV/head+scaler/GTP só auxiliares; teste isolado. Cinco subpopulações B0–B4 com proteção, recombinação entre compatíveis, transferência controlada, reserva exploratória, Pareto global só entre comparáveis; curvas de aprendizagem como sinal auxiliar.
Experimento principal por dataset: A MO-DEHB independente por porte sem herança; B LEMONADE; C PNAS-CS; D crescimento progressivo com SPiFiL abstrato; E D + agregação multiescala. Δκ = κ_val(filho) − κ_val(ancestral) e custo; saturação; resultados não monotônicos permitidos.
Integridade: seeds fixas/repetidas, splits imutáveis, configs com hash, candidatos = genoma+pesos+fidelidade, checkpoints completos, origem dos dados, teste fora do controller, resume sem duplicar, fitness sem misturar C=0,1/C=100, limites de memória/BLAS, testes de OOM/NaN/interrupção. Jobs vão para a fila do Agente 5.
Papers: MO-DEHB, LEMONADE, Net2Net, EfficientNet, OFA, Barlow Twins, VICReg, LeJEPA (https://arxiv.org/abs/2511.08544).
W&B: HV × geração e GPU-h, κ×params, κ×FLOPs, curvas B0–B4, melhoria filho/ancestral, efeito por tipo SPiFiL, sobrevivência por operador, contribuição por linhagem, distribuição prof./largura, falhas, curvas.
Entregáveis: evolução integrada e testada, config independente, testes unitários/e2e, simulação de interrupção/resume, genealogia, validação e Pareto, progressive_nas_results.md, exportação B0–B4.
Gate final: comparação controlada busca independente × crescimento convencional × crescimento com init abstrata, sem teste na seleção.
