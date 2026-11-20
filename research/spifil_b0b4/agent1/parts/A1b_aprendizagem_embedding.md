# A1b — Aprendizagem (C) e Embedding (D) · quantitativas 2, 4 e 7

_Agente 1b (parte do Agente 1), 16–17/11/2026. Só leitura, CPU (`OMP_NUM_THREADS=4 nice -n 15 CUDA_VISIBLE_DEVICES=""`). Não treinei, não mexi em produção. O teste aparece só como leitura de relatórios já existentes (`test_svm_C100.jsonl`); o diagnóstico novo do embedding usa só treino→validação._

Rótulos usados em todo o texto:
- **[MEDIDO]**: número lido de um artefato ou calculado por mim, com a fonte citada.
- **[CORRELAÇÃO]**: associação observacional; pode haver confundidores.
- **[HIPÓTESE]**: explicação plausível que ainda não foi testada.
- **[CAUSAL]**: só quando há intervenção controlada (mesmo candidato, uma variável mudando). Neste material isso só acontece em: WiSE α (mesmos pesos, blend), fidelidade (mesma linhagem), Fase 0 do PNAS-CS (mesmo genoma, AE on/off) e as leituras do diagnóstico (mesmo encoder, readout diferente).

## Artefatos

| arquivo | conteúdo |
|---|---|
| `A1b_data.csv` | 1000 jobs (1 linha = candidato × fidelidade) de 19 `search_state.json`: Jaci `artifacts/modehb_{hybrid,lemonade}/*`, `*_C01_final/remote_sensing`, `modehb_hybrid_C01_2049/*` e o bundle do athena `pnas_cs/search_state.json`. Colunas: `source_state`, `job`, `cand_id`, `fidelity`, `checkpoint`, genoma (ae_schedule, ae_target, protect, pooling…), `wise_alpha`, `wise_k_{0..1}`, `ssl_final_loss`, `ssl_epochs`, `ae_epochs_per_layer`, estatísticas do AE por camada e as métricas label-free (`lf_*`). Junta o teste ORIG/COV por (run, dataset, cand, fid) quando existe. |
| `A1b_tests_all.csv` | 265 linhas de teste únicas (Jaci + bundle), deduplicadas pela mais recente. |
| `A1b_orig_cov_tests.csv` | 263 linhas de teste com ORIG e COV, com params e estrato. |
| `A1b_lineages.csv` | 151 linhagens (mesmo candidato em ≥2 fidelidades). |
| `A1b_seed_curves_summary.csv`, `A1b_seed_curves_long.csv` | 31 curvas de sementes SPiFiL-pequeno (L0 × L6 LeJEPA), de `exps/spifil_growth/ssl_pequeno_lejepa/*/L*/epoch_metrics.csv`. |
| `A1b_embed_diag.jsonl` / `A1b_embed_diag.csv` | diagnóstico CPU de 17 checkpoints (variantes saved/α=0/α=1) com geometria e sondas SVM treino→val. |
| `A1b_scripts/` | `extract_jobs.py`, `seeds.py`, `figs_learning.py`, `embed_diag.py`, `summarize_embed.py`, `ols.py`: reproduzem tudo. |
| `figs/A1b_F1…F9` | figuras citadas abaixo. |

**Ressalva que vale para todo o documento:** o fitness (κ de validação) **não** é a régua ORIG. `search_v3.yaml:142-143` usa `svm_fitness.readout: cov_head` (`run_search.py:550-557`, `_embed_cov`), com StandardScaler e C=100. Portanto: todo κ_val de `search_state` (fitness, `wise_kappas`, calibração) está na régua **COV-val**, e o número reportado no teste é **ORIG** (head 1280, sem scaler). Essa diferença é a raiz das seções D e 7.

## 10 achados (resumo)

1. **[MEDIDO] A busca seleciona numa régua e reporta noutra.**
   - O fitness, o WiSE e a calibração usam COV-val (`cov_head` + StandardScaler, `search_v3.yaml:142-143`); o número reportado é ORIG (head 1280, sem scaler).
   - val(COV) ordena bem o teste COV (ρ mediano 0,80 em 19 grupos) e quase nada do teste ORIG (ρ 0,09; negativo em 9/19).
   - Arrependimento ORIG do sel-val: mediana 0,047, máximo 0,307.
2. **[CAUSAL] COV ≫ ORIG vem da 2ª ordem dos blocos, não do head nem do scaler.**
   - Em 17 checkpoints (treino→val), a covariância dos blocos sem o grupo head reproduz o COV (Δ +0,000).
   - O scaler explica +0,014; o head não acrescenta nada à média do último bloco (+0,021).
   - A 2ª ordem contra a melhor 1ª ordem por bloco dá **+0,107** na mediana (até +0,52).
3. **[MEDIDO] O embedding 1280 é degenerado em geometria.**
   - Rank efetivo mediano 3,4 de 1280, com 51% da variância num único eixo; GAP do último bloco com rank efetivo 1,8.
   - Com `moments`, o 1×1 projeta para 320 canais (gargalo aleatório na âncora).
   - Caso limite: larvae c00001 tem ORIG κ = 0,000 (classe majoritária) e PCA-whiten 0,785.
4. **[MEDIDO/CORRELAÇÃO] O ORIG não escala com o porte e o COV sim.**
   - Por década de params (efeito fixo de dataset): ORIG +0,013 (nulo), COV +0,072 (p<10⁻³⁰).
   - O gap cresce de tiny (+0,07) para huge (+0,15); a profundidade piora o ORIG (ρ −0,27).
   - Em modelos com ≥4 blocos, o último bloco é pior que um intermediário em 10/11 (sonda COV).
5. **Conclusão encoder × cabeça.** A informação está no encoder (blocos intermediários, 2ª ordem); a perda está na cabeça/leitura (1×1 + pooling de 1ª ordem só do último bloco, anisotrópica). Há uma componente secundária no encoder: o último bloco degrada nos modelos profundos e com SSL nas bases raw.
6. **[CAUSAL] O LeJEPA+SIGReg muda pouco os pesos SPiFiL na régua da busca.**
   - κ(α=1) − κ(α=0): mediana +0,003 em 656 jobs; ganho +0,015 em f200 nas mesmas linhagens, enquanto a perda SSL cai pela metade.
   - Nas sementes, o SSL ajuda aid/ham/eggs (+0,01 a +0,10) e **degrada** larvae (−0,12 a −0,34 no fim) e protozoan (nunca supera o L0).
7. **[CAUSAL] O SSL pode destruir o ORIG sem que o fitness veja.** De α=0 para o α salvo, o COV nunca piora (11/11) e o ORIG piora em 5/11 (larvae_raw c00003: 0,569 → 0,322 com COV 0,780 → 0,795). O WiSE escolhe α por uma régua cega ao head.
8. **[MEDIDO] O WiSE é, em boa parte, seleção de ruído.**
   - α=0 em 23% dos jobs (0% flowers – 37% protozoan_raw), sem relação com estrato, params ou depth.
   - O ganho do α* (+0,011 a +0,022) tem o mesmo tamanho do otimismo de escolher o máximo de 5 (+0,016) e do ruído entre α vizinhos (DP 0,01–0,08).
9. **[CAUSAL, mesma linhagem] Fidelidade não reordena nada.**
   - ρ(κ f5, κ f200) = 0,985 em 64 linhagens; ganho mediano f5→f200 de +0,006.
   - A calibração satura em 5 épocas em 48/59 entradas. O AE só roda em f5 (2 épocas/camada).
   - **AE local:** joint × decoder_first sem efeito no κ (−0,001 ± 0,011; joint só reduz o rank efetivo do bloco, −0,08). O alvo pixel/masked supera o feature (+0,03–0,04). Na Fase 0 do PNAS, AE × noAE dá +0,010 (n=4 pares, dentro do ruído).
10. **[MEDIDO, por desenho] A promoção protegida dá vaga aos tiny independentemente do κ.**
    - remote_sensing f5→f15: 6–7 dos 8 tiny promovidos ficam abaixo do corte global top-k (κ mediano 0,41 contra 0,57 dos huge), enquanto 3–6 huge acima do corte ficam de fora.
    - Como a fidelidade não reordena, o orçamento das promoções compra ~0. A vantagem dos tiny no teste ORIG tem outra causa: o ORIG não escala (achado 4).

---

## C. Aprendizagem

### C1. AE local: joint × decoder_first × noAE

- **[MEDIDO]** Na busca, o AE local roda só na primeira fidelidade: `ae_epochs_per_layer = 2` em todos os 354 jobs iniciais f5 (C=100). Nas promoções o valor é 0, porque a retomada só continua o SSL (`run_search.py:666`, `split_epochs(5,(1,2)) = (2,3)`). Logo, "fidelidade" só acrescenta épocas de LeJEPA, nunca de AE.
- **[CORRELAÇÃO, ajustada]** Usei OLS sobre os 354 jobs iniciais f5 (C=100, sem PNAS), controlando dataset, kind, log10(params), protect, ae_target, pooling e projector (`A1b_scripts/ols.py`):
  - `ae_schedule = joint` (referência decoder_first):
    - κ da âncora (α=0, SPiFiL+AE sem SSL): **−0,001 ± 0,011** (p=0,90);
    - κ do fitness: −0,004 ± 0,009 (p=0,70).
    - Ou seja, nenhum efeito detectável no κ.
  - O que muda é a estrutura: `joint` reduz o rank efetivo relativo da saída do bloco, medido pelo próprio log do AE, em **−0,084 ± 0,017** (p<10⁻⁵).
  - `ae_target`: pixel +0,033 ± 0,013 (p=0,014) e masked +0,028 (p=0,13) na âncora, contra feature. No fitness: pixel +0,040 (p=0,0005), masked +0,038 (p=0,02). O alvo `feature` é o pior.
  - `protect`: full −0,020 (p=0,11) e lowlr −0,009 (p=0,45) contra freeze. Não é significativo, mas a direção é coerente com "mexer nos filtros SPiFiL não ajuda".
- **[CAUSAL, n mínimo]** Fase 0 do PNAS-CS (`pnas_cs/phase0_ablacao_ae.json` + `search_state`): remote_sensing, P0, f45, 2 genomas × {noae, decoder_first, joint}, seed 42.

  | genoma | κ α=0 noAE (SPiFiL puro) | κ α=0 decoder_first | κ α=0 joint | κ fitness noAE / dec_first / joint |
  |---|---|---|---|---|
  | 77-155 (k3,k5), lowlr | 0,479 | 0,485 | 0,508 | 0,479 / 0,485 / 0,508 |
  | 37-75-149, full | 0,474 | 0,487 | 0,465 | 0,474 / 0,502 / 0,493 |

  - Δ médio (AE − noAE) na âncora: **+0,010**. Os quatro pares dão +0,006, +0,029, +0,013 e −0,009.
  - Esse Δ é bem menor que o erro-padrão do κ com n_val=159 (~0,04).
  - Há um confundidor: noAE tem 45 épocas de SSL e AE tem 15/camada + 30 de SSL.
  - **Conclusão: AE local não tem efeito demonstrável.** Com 2 épocas por camada (o valor da busca), o AE é praticamente inócuo. joint e decoder_first não se distinguem no κ.
- **[HIPÓTESE]** O AE com alvo pixel preserva informação de baixo nível que o SPiFiL já captura, por isso é quase neutro. O alvo `feature` empurra o bloco para a identidade do prefixo e reduz diversidade, por isso é pior. Teste proposto: mesmo genoma × {noae, pixel, feature, masked} × 3 seeds, f45, RS + Flowers.

### C2. LeJEPA+SIGReg sobre os pesos SPiFiL

O WiSE dá um experimento controlado gratuito: para cada job, κ(α=0) é o SPiFiL(+AE) com o head inicial e κ(α=1) é o mesmo modelo depois do SSL, na mesma régua COV-val.

- **[CAUSAL, régua COV-val]** κ(α=1) − κ(α=0) em 656 jobs C=100 (`A1b_data.csv`, F2):
  - mediana **+0,003**, média +0,005, IQR [−0,014; +0,021];
  - positivo em 55% dos jobs.
  - Por dataset (mediana): ham10000 +0,012; protozoan +0,010; eggs +0,005; aid +0,003; RS 0,000; larvae 0,000; protozoan_raw −0,004; larvae_raw −0,009.
- **[CAUSAL, mesma linhagem]** Nas 53 linhagens de calibração (mesmo candidato f5→f200, C=100; F3):
  - o ganho do SSL puro sobe de **0,000 (f5) para +0,015 (f200)** na mediana;
  - o ganho do α escolhido vai de +0,011 para +0,022;
  - no mesmo intervalo, `ssl_final_loss` cai de 0,68 para 0,30 (mediana).
  - **Ou seja, a perda SSL cai pela metade e o κ quase não se move:** perda desacoplada do κ, como já se via na destilação (CLAUDE.md).
- **[MEDIDO, sementes SPiFiL-pequeno, LeJEPA completo L6 × L0 sem SSL; 27 runs com L0; F4]**
  - aid: +0,07 a +0,10 (pico perto do fim) → **o SSL ajuda**.
  - eggs: 0 a +0,07.
  - ham10000: +0,01 a +0,05.
  - larvae: pico cedo (épocas 0–55) e depois **degrada** −0,12 a −0,34 até a última época. Aqui ρ(train_loss, κ_val) é positivo (0,22–0,74): menos perda significa menos κ.
  - protozoan: **nunca supera o L0** (argmax na época 0 em 6/6; última época −0,02 a −0,12).
  - Mediana do ganho no máximo: +0,022, que é otimista porque é o máximo sobre dezenas de épocas. Mediana do ganho na última época: −0,020.
- **[CORRELAÇÃO → HIPÓTESE]** O SSL global LeJEPA+SIGReg ajuda em cenas e texturas naturais (aid, ham) e prejudica parasitas com fundo preto (larvae, protozoan). Duas hipóteses:
  - (a) as vistas aumentadas de objetos centrados em fundo preto deixam a invariância trivial;
  - (b) o SIGReg isotropiza as dimensões dominadas pelo fundo.

  Teste proposto: o mesmo L0 com máscara de objeto nas vistas, contra sem máscara.

### C3. WiSE: frequência de α=0 e κ por α

- **[MEDIDO]** Distribuição do α escolhido (C=100; F1):
  - **α=0 em 23%** dos jobs (hybrid 24%, lemonade 21%, PNAS 27%); α=1 em 27%; frações intermediárias no resto.
  - α=0 por dataset: flowers 0%, protozoan 8%, eggs 14%, ham 14%, RS 24%, aid 31%, larvae 33%, larvae_raw 33%, protozoan_raw 37%.
  - α=0 por estrato: tiny 0,20, small 0,30, medium 0,26, huge 0,15; PNAS P0 0,46, P1 0,06, P2 0,12, P3 0,36, P4 0,18. Não há tendência monotônica.
  - Por fidelidade, α=1 sobe de 25% (f5) para 44% (f200).
  - Por protect (todas as réguas, C=100 e C=0,1), a frequência de α=0 é quase igual (freeze 0,27, full 0,22, lowlr 0,22).
- **[MEDIDO, sanidade]** κ(α=0) é **idêntico em todas as fidelidades da mesma linhagem** (64 linhagens conferidas). A âncora é fixa e a leitura é determinística, então a régua não tem ruído de execução.
- **[MEDIDO]** O ruído de seleção do α tem a mesma ordem de grandeza do efeito:
  - o desvio-padrão de κ(0,5) − κ(0,25), que são pesos quase iguais, é 0,009 (aid), 0,008 (eggs), 0,021–0,023 (ham, protozoan, RS, flowers), 0,040 (larvae) e 0,078 (larvae_raw);
  - o otimismo da escolha, κ(α*) − média_α κ(α), é +0,016 na mediana;
  - o "ganho do WiSE" (+0,011 a +0,022) é desse mesmo tamanho.
- **Leitura:**
  - **[CAUSAL]** O LeJEPA+SIGReg muda pouco a representação na régua da busca.
  - **[HIPÓTESE forte]** O α escolhido é, em grande parte, seleção de ruído do conjunto de validação, principalmente em RS (n_val=159) e larvae (n_val=240).

### C4. Convergência × fidelidade (5/15/45/100/200)

- **[MEDIDO]** A calibração (1 representante por estrato em todas as fidelidades) declara `saturation_epochs = 5` em **48 das 59** entradas (todas as buscas). As exceções são, quase todas, tiny ou huge com 15–45 épocas.
- **[CAUSAL, mesma linhagem]** Nas 64 linhagens que chegam a f200 (`A1b_lineages.csv`):
  - **ρ de Spearman(κ f5, κ f_final) = 0,985**;
  - ganho mediano f5→f200 de +0,006 (C=100); 55% das linhagens têm |ganho| ≤ 0,01;
  - por estrato: tiny +0,010, small +0,006, medium 0,000, huge +0,007. Os grandes não convergem mais devagar.
  - Nas cadeias de promoção C=0,1: f15 −0,001, f45 −0,006, f100 −0,001, f200 +0,010.
- **Conclusão [MEDIDO]:** na régua da busca, a escada de fidelidades não reordena os candidatos. O ranking em f5 já é o final, e o orçamento de f15–f200 (incluindo o GPU-h das promoções) compra ≈ +0,01 de κ.
- **[HIPÓTESE]** Isso acontece porque o fitness é dominado pela âncora SPiFiL (que não muda com a fidelidade) e pela leitura COV de 2ª ordem dos blocos intermediários, que o SSL quase não altera (ver D).

### C5. Vantagem indevida dos pequenos nas promoções

- **[MEDIDO]** remote_sensing, as três buscas com promoções f5→f15 (F5). A taxa de promoção por estrato é quase uniforme (≈ 1/3 em cada), por desenho: `min_share = 0,25` + ranqueamento não dominado (κ × FLOPs) dentro do estrato (`strata.py:159`). O efeito disso:

  | busca | pool f5 | promovidos | corte top-k global (κ) | promovidos abaixo do corte (tiny) | tiny promovidos / tiny no top-k | huge acima do corte e **não** promovidos |
  |---|---|---|---|---|---|---|
  | H-LEM C=100 | 80 | 27 | 0,535 | 14 (7) | 8 / 4 | 3 |
  | H-MODEHB C=0,1 | 74 | 27 | 0,541 | 11 (6) | 8 / 3 | 6 |
  | H-LEM C=0,1 | 75 | 25 | 0,543 | 11 (7) | 8 / 2 | 5 |

  - Mediana do κ dos tiny promovidos no H-LEM C=100: **0,414**, contra 0,565 dos huge promovidos.
  - Há tiny promovidos com κ 0,33–0,40, nas posições 70–81 de 81.
- **[CAUSAL por desenho, não por dados]** A promoção protegida, mais o objetivo de FLOPs, garante vaga aos tiny independentemente do κ. Ao mesmo tempo, 3–6 huge acima do corte global ficam de fora.
  - Como a fidelidade não reordena (C4), esses tiny não "se recuperam" depois.
  - A vantagem dos pequenos **nos rankings finais de teste ORIG** tem outra causa: o ORIG não escala com params (D/Q2). Ela não vem das promoções.
- **[HIPÓTESE / proposta]** A promoção deve usar o ganho marginal esperado por fidelidade (que é ~0) e um limiar de κ por estrato relativo ao global (por exemplo, não promover abaixo do quantil 50% global). Mais simples: dado C4, reduzir a escada para f5 → f45 e gastar o orçamento em mais candidatos/seeds.

---

## D. Embedding — por que COV ≫ ORIG nos grandes; encoder × cabeça

### D1. ORIG × COV no teste (quantitativa 2) — dados existentes

- **[MEDIDO]** Em 263 linhas de teste com as duas leituras no mesmo checkpoint (`A1b_orig_cov_tests.csv`, F6):
  - κ COV − κ ORIG: **mediana +0,123** (IQR +0,069 a +0,176), COV > ORIG em 90%;
  - por estrato: tiny +0,071, small +0,111, medium +0,152, huge +0,151;
  - por dataset: protozoan_raw +0,42, larvae_raw +0,32, eggs_raw +0,25, flowers +0,17, eggs +0,17, aid +0,16, larvae +0,15, ham +0,13, RS +0,11, protozoan +0,01, coconut −0,02.
- **[CORRELAÇÃO, com efeito fixo de dataset]** Por década de parâmetros:
  - **COV +0,072 ± 0,005 (p<10⁻³⁰)**;
  - **ORIG +0,013 ± 0,010 (p=0,19, nulo)**;
  - gap +0,058 ± 0,009.
  - Profundidade: ρ(gap, depth) = +0,30 e ρ(ORIG, depth) = −0,27 (os dois p<10⁻⁴).
  - **O ORIG não melhora (e com profundidade piora) quando o encoder cresce. O COV melhora.**
- **[MEDIDO]** Métricas label-free do próprio search_state sobre o embedding 1280 de validação:
  - o rank efetivo (participation ratio de λ, `embedding_eval.py:93`) tem **mediana 2,7–4,2 de 1280 dimensões** (por pooling: gap 2,7, gem 3,5, moments 4,2);
  - número de condição de 10⁸ a 10¹¹.
  - ρ(cond, κ ORIG) = −0,26 (p=5·10⁻⁵).

### D2. Diagnóstico CPU (≤20 checkpoints) — treino→val, SVM C=100

### D2. Diagnóstico CPU (17 checkpoints) — treino→val, SVM C=100

**Protocolo** (`A1b_scripts/embed_diag.py`; dados em `A1b_embed_diag.jsonl` e `A1b_embed_diag.csv`; tabela completa em `A1b_embed_diag_table.md`; F8, F9):
- dados carregados por `load_config(search_v3.yaml)` + `progressive_data.load_data`. O `data_sha256` é igual ao do `search_state` nos 5 datasets;
- treino estratificado ≤4000 (protozoan e ham ≤2000), validação inteira (protozoan: 1500 estratificadas);
- `core.svm.fit_svm`, linear, C=100, ovo, libsvm;
- **o teste não foi tocado.**

**Amostra:**
- remote_sensing: hybrid c00001–c00004 f200, c00004 f5 e lemonade_C01 c00098 f200;
- larvae: c00001–c00004 f200;
- larvae_raw: c00003 e c00004;
- protozoan: c00002, c00003 e c00004;
- ham10000: c00002 e c00004.

São 17 checkpoints de 20k a 2M de parâmetros. Variantes:
- **saved**: o α escolhido pelo WiSE;
- **a0**: âncora SPiFiL(+AE) com o head inicial;
- **a1**: estado SSL puro (só RS).

**Validação do pipeline [MEDIDO]:** nos 10 checkpoints com protocolo idêntico ao da busca (RS, larvae, larvae_raw: treino e validação inteiros), o κ_val COV recalculado na CPU bate exatamente com o `fitness_kappa` do search_state em 8 de 10. Exemplos: RS c00001 0,494 = 0,494; c00004 0,570 = 0,570; larvae c00003 0,950 = 0,950; larvae_raw c00004 0,879 = 0,879. As duas diferenças estão em blocos de posto deficiente: larvae c00001 (0,655 contra 0,683) e larvae_raw c00003 (0,795 contra 0,810). O eigh na CPU não é bitwise igual ao cusolver, como documenta `pooling_head._eigh_host`. Em protozoan e ham usei subamostras (treino 2000; val 1500 em protozoan), então os κ ficam 0,02–0,06 abaixo do fitness e só valem para comparações internas.

**Decomposição do gap COV − ORIG (variante saved, n=17 checkpoints; mediana [mín; máx]):**

| comparação (mesmo encoder) | Δκ_val | o que isola |
|---|---|---|
| COV completo − ORIG | **+0.165 [+0.000; +0.655]** | gap total na validação (o do teste é +0,12) |
| COV sem o grupo head − COV completo | +0.000 [-0.134; +0.019] | **o head 1280 não contribui** para o COV |
| COV com scaler − COV sem scaler | +0.014 [-0.009; +0.147] | o StandardScaler explica pouco |
| ORIG + scaler − ORIG | +0.034 [-0.056; +0.641] | escala por dimensão: pouco no geral, mas resgata casos degenerados |
| ORIG + PCA-whiten − ORIG | +0.009 [-0.124; +0.785] | idem |
| GAP do último bloco + scaler − ORIG | +0.021 [-0.064; +0.241] | **o head não acrescenta nada** à média do último bloco |
| melhor GAP por bloco − GAP do último | +0.014 [+0.000; +0.177] | blocos intermediários são um pouco melhores |
| concat GAP de todos os blocos − GAP do último | +0.017 [-0.058; +0.202] | multiescala de 1ª ordem ajuda pouco |
| **COV sem head − melhor GAP por bloco** | **+0.107 [+0.006; +0.521]** | **2ª ordem (covariância) contra 1ª ordem: a maior parcela** |
| COV de todos os blocos − COV do último bloco | +0.049 [-0.042; +0.296] | multiescala de 2ª ordem |

**Geometria do embedding 1280 [MEDIDO, treino+val]:**
- rank efetivo (participation ratio) mediano **3.4 de 1280**;
- a 1ª componente carrega **51%** da variância;
- GAP do último bloco: rank efetivo 1.8;
- COV: rank efetivo 13.0.
- Conforme o pooling:
  - com `gap`, o head é exatamente linear sobre a média do último bloco, e o posto de W é igual a C_último;
  - com `moments`, o 1×1 projeta para **320 canais** (1280/4) antes das estatísticas, o que vira um gargalo aleatório (α=0) quando C_último = 818–1024.
- O libsvm sem scaler precisa de até 3,9·10⁷ iterações (larvae_raw c00003, ham c00002): fronteiras quase degeneradas.
- Caso extremo: **larvae c00001** (2-16 canais):
  - κ_val ORIG = **0,000** (prevê só a classe majoritária; no teste C=0,1 também deu 0,000);
  - rank efetivo 1,15, top1 = 93% da variância, 26% de dimensões mortas;
  - com scaler 0,641, com PCA-whiten 0,785, COV 0,655.

**Perfil por bloco [MEDIDO]:**
- O melhor bloco na sonda COV **não é o último em 12/17 checkpoints**. Nos modelos com ≥4 blocos, o último é pior que um intermediário em **10/11**.
- Exemplos (COV por bloco):
  - larvae_raw c00004: 0,000 / 0,573 / **0,810** / 0,744 / 0,602;
  - RS c00004 f200: 0,466 / 0,537 / **0,597** / 0,479;
  - ham c00004: 0,376 / 0,412 / **0,504** / 0,477 / 0,408.
- Na 1ª ordem (GAP), o padrão é o mesmo, com amplitude menor.

**SSL e ORIG [CAUSAL, mesmo checkpoint, α=0 → α salvo, 11 checkpoints com α*>0]:**
- O COV nunca piora (Δ de 0,000 a +0,227, porque o α foi escolhido por ele). Já o ORIG **piora em 5/11** e a correlação entre os dois Δ é ρ = −0,26.
- Pioras do ORIG:
  - larvae_raw c00003 (α=1): ORIG 0,569 → **0,322**, COV 0,780 → 0,795;
  - larvae_raw c00004 (α=0,5): 0,397 → 0,288 (COV +0,109);
  - protozoan c00003 (α=0,25): 0,607 → 0,527;
  - larvae c00001: 0,061 → 0,000;
  - RS c00004: 0,413 → 0,377.
- Melhoras do ORIG: RS c00001 +0,113, protozoan c00002 +0,100, c00004 +0,091.
- **O WiSE escolhe α por uma régua (COV) cega ao que o SSL faz com o head e o último bloco, então o α escolhido pode ser o pior para o ORIG.**

### D3. Conclusão: encoder × cabeça

- **[CAUSAL, leituras do mesmo encoder]** A informação discriminativa **está no encoder**, sobretudo nos **blocos intermediários** e nas **estatísticas de 2ª ordem**. Dois pontos mostram isso:
  - a covariância dos blocos, sem o head, recupera todo o ganho do COV;
  - o head 1280 não acrescenta nada nem ao COV nem à média do último bloco.
- **O problema está na cabeça/leitura**, que é 1×1 + pooling de 1ª ordem só do último bloco. Ela:
  - (i) descarta a 2ª ordem (a maior parcela, ≈ +0,10 de κ na mediana);
  - (ii) lê só o último bloco, que nos modelos profundos é pior que um intermediário;
  - (iii) entrega um espaço muito anisotrópico (rank efetivo ≈ 3, um eixo com ~50% da variância), em que o SVM linear sem scaler fica mal condicionado. Em casos degenerados (larvae c00001) isso sozinho zera o κ.
- **Há uma componente secundária no encoder:** o último bloco dos modelos grandes/profundos degrada em relação aos intermediários. Nas bases raw de parasitas, o SSL piora justamente o último bloco e o head (ORIG cai até −0,25 com COV estável).
  - "COV ≫ ORIG nos grandes" vem disso: modelos maiores têm mais blocos e mais canais. O COV aproveita os intermediários e a 2ª ordem, e o ORIG fica preso à média do último bloco projetada.
  - **[HIPÓTESE]** O teto do ORIG não cresce com params (+0,013/década, nulo) porque a média espacial do último bloco satura em poucos fatores (rank efetivo do GAP do último bloco ≈ 1,8), independentemente da largura.
- **Consequências para o desenho (A3/A4):**
  - (a) a cabeça 1280 autocontida precisa embutir multiescala + 2ª ordem de dimensão fixa (compact bilinear / sqrtm-cov PCA-k por bloco, depois projeção para 1280) **e treinar sobre isso**, em vez de usar o COV externo;
  - (b) normalizar a saída (signed-sqrt + L2, ou whitening aprendido) antes do SVM sem scaler;
  - (c) o fitness deve medir ORIG-val, para que WiSE, promoções e escolha do α vejam o head;
  - (d) a profundidade útil deve ser escolhida pela sonda por bloco, e não pela profundidade nominal.
- **Teste causal proposto:**
  - fixar 3 encoders (tiny, medium, huge) por dataset e variar só a cabeça: atual / cov-PCA64 → linear 1280 / multiescala GeM → 1280 / + signed-sqrt-L2;
  - métrica: κ_val ORIG; 3 seeds de cabeça;
  - critério: Δ ≥ 0,05 sobre a cabeça atual em ≥2 de 3 datasets prioritários.

---

## Quantitativa 4 — curvas de treino e WiSE por arquitetura

- **[CORRELAÇÃO, OLS n=656 jobs C=100]** Ganho do SSL puro, κ(α=1) − κ(α=0), com efeito fixo de dataset:

  | variável | coef | p | leitura |
  |---|---|---|---|
  | log10(fidelidade) | **+0,008 ± 0,003** | 0,009 | o único efeito claro: +0,008 por década de épocas |
  | protect=full (vs freeze) | +0,009 ± 0,005 | 0,048 | deixar os filtros SPiFiL se moverem dá um pouco mais de ganho de SSL |
  | log10(params) | −0,003 ± 0,003 | 0,36 | nulo |
  | depth | −0,001 ± 0,001 | 0,32 | nulo |
  | pooling gem / moments | +0,001 / −0,004 | >0,38 | nulo |
  | ae_schedule joint | −0,003 ± 0,005 | 0,57 | nulo |

- **[MEDIDO]** Mediana do ganho do SSL por profundidade:

  | depth | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---|---|---|---|---|---|---|
  | ganho SSL | 0,000 | +0,006 | +0,007 | 0,000 | +0,004 | −0,001 | −0,002 |
  | freq. α=0 | 0,28 | 0,19 | 0,23 | 0,26 | 0,22 | 0,15 | 0,33 |

  A frequência de α=0 não depende de params, depth nem fidelidade (OLS, todos p>0,16).
- **[MEDIDO]** `ssl_final_loss` não ordena a qualidade:
  - a mediana por depth oscila entre 0,42 e 1,00 sem relação com o κ;
  - nas linhagens, a perda cai 2× e o κ sobe +0,015 (C2).
- **[MEDIDO, sementes]** Curvas longas (até 815 épocas):
  - o ponto de máximo κ_val varia de 0 (protozoan, larvae split3) a 141 épocas (ham);
  - em larvae, o κ cai enquanto a perda cai (ρ>0);
  - em aid, eggs e ham o κ sobe junto com a queda da perda (ρ = −0,6 a −0,96).
  - **Selecionar o checkpoint pela perda SSL seria errado em parasitas.**

## Quantitativa 7 — selecionados por validação × melhores só no teste

_O teste aparece aqui só como auditoria retrospectiva. Nada aqui serve para seleção._

- **[MEDIDO]** 19 grupos (busca × dataset, n ≥ 3 checkpoints testados; Jaci + bundle; F7):

  | métrica | mediana | IQR / faixa |
  |---|---|---|
  | ρ Spearman val(COV) × **teste COV** | **0,80** | IQR 0,64–0,92 |
  | ρ val(COV) × **teste ORIG** | **0,09** | IQR −0,41 a +0,57 |
  | ρ teste ORIG × teste COV | 0,36 | |

  - Nos grupos com ρ(val, ORIG) negativo (hybrid coconut −0,75, protozoan −0,54, ham −0,49, aid −0,18, larvae_raw −0,42), a validação escolhe sistematicamente o pior ORIG.
- **[MEDIDO]** Arrependimento no teste ORIG (melhor ORIG − ORIG do selecionado pela validação):
  - mediana **0,047**; IQR 0,017–0,108; máximo **0,307** (hybrid larvae_raw: sel-val c00004 f45 = 0,188 contra c00001 f200 = 0,495);
  - outros: pnas RS 0,224, lemonade eggs 0,177, lemonade larvae 0,165;
  - o sel-val coincide com o melhor ORIG em 4/19 grupos;
  - na régua COV, o sel-val coincide com o melhor COV em 10/19 grupos, e nos demais a perda é ≤0,09.
- **[CAUSAL quanto ao mecanismo, por construção]**
  - A busca otimiza κ_val na leitura COV (StandardScaler + covariância de 2ª ordem de todos os blocos + head), e o número reportado é ORIG (só o head, sem scaler).
  - A seleção funciona bem dentro da própria régua (ρ 0,80). A escolha do "melhor por validação" é coerente com COV e quase aleatória para ORIG.
  - Isso explica por que os "melhores no teste ORIG" são frequentemente tiny (c00001 com 20–22k params em coconut, ham, larvae_raw e flowers): o ORIG não cresce com params (D1), e o fitness não o vê.
- **Recomendação para o contrato de avaliação (A1):** se a régua oficial é ORIG, o fitness tem de ser ORIG-val (head 1280, sem scaler, C=100), e COV vira só auxiliar. É o que a MISSION exige para o Agente 4 ("Fitness nova versão: max κ val (ORIG…)"). Hoje a busca **não** cumpre isso.

---

## Limitações

- Todo resultado da busca é **n=1** (split 1, seed 42). O "ruído" estimado aqui é o do conjunto de validação (entre α vizinhos e de seleção), não o de seeds de treino. Faltam seeds repetidas para separar a variância de treino.
- A régua COV-val do search_state e o meu COV na CPU diferem em blocos de posto deficiente (eigh CPU × cusolver). Em protozoan e ham usei subamostras (as comparações valem dentro do diagnóstico).
- O diagnóstico usa 17 checkpoints, quase todos de calibração (4 por dataset), em 5 datasets. Eggs, flowers, AID e coconut (athena) não foram reavaliados na CPU.
- O teste só aparece como auditoria retrospectiva (Q2 e Q7) de arquivos já existentes; nada aqui seleciona por teste.
- As OLS são observacionais: o genoma é amostrado pelo DE e pela busca (não é aleatório puro), então há confusão residual entre genes.

## Figuras

| fig | conteúdo |
|---|---|
| F1 `figs/A1b_F1_wise_alpha_freq.png` | frequência de α (0 / intermediário / 1) por dataset |
| F2 `figs/A1b_F2_kappa_by_alpha.png` | κ(α) − κ(0) por dataset (mediana e IQR) |
| F3 `figs/A1b_F3_fidelity_lineages.png` | ganho do SSL puro e do WiSE × fidelidade nas 53 linhagens |
| F4 `figs/A1b_F4_seed_curves.png` | curvas L6 − L0 das sementes SPiFiL-pequeno |
| F5 `figs/A1b_F5_promotions_strata.png` | promovidos × estrato × corte global (RS) |
| F6 `figs/A1b_F6_orig_vs_cov.png` | ORIG × COV no teste; gap × params |
| F7 `figs/A1b_F7_val_vs_test_rank.png` | ρ val(COV) × teste COV / ORIG por grupo |
| F8 `figs/A1b_F8_readouts_per_ckpt.png` | 8 leituras por checkpoint (diagnóstico treino→val) |
| F9 `figs/A1b_F9_block_probes.png` | sonda COV por bloco e gap × rank efetivo (o rank efetivo sozinho não explica o gap) |
