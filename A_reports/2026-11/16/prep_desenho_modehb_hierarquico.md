# Agente 4 — Preparação: desenho do MO-DEHB hierárquico B0–B4

> Fase de PREPARAÇÃO (16/11/2026). Só literatura e desenho: nenhum código de produção, treino ou uso de GPU.
> Implementação e experimentos ficam bloqueados até o gate do Agente 3 (MISSION.md §Ordem).
> Regras herdadas: FILA_E_INSTRUCOES.md R1 (SVM `core.svm.fit_svm`, C=100), R2 (régua ORIG no teste), R3 (fitness = κ × FLOPs), R9 (e2e antes de produção). Teste nunca entra na seleção.
> Notação: **[M]** = medido/lido no código ou no paper; **[H]** = hipótese ou decisão de desenho a validar.

---

## 0. Resumo executivo

1. O framework atual (`ga_search/modehb/`) já tem quase tudo o que o desenho precisa: NDS, HV exato, crowding/EpsNet, DE rand/1 com crossover binomial, Hyperband/SH, estratos por parâmetros com promoções protegidas, replay determinístico, OOM retry, `_fatal_cuda`, herança LEMONADE e Net2Net (`external_seed.grow_blocks`), PNAS-CS com compound scaling e `fit_scaling_law`.
2. **Falta** [M]:
   - genoma hierárquico (hoje é um vetor plano de 8 + 8×11 genes);
   - fitness ORIG (hoje a fitness é `cov_head` + `StandardScaler`, a régua COV);
   - Δκ filho × ancestral;
   - identidade de candidato que inclua a herança;
   - hash de config no estado;
   - trava contra dois controladores;
   - estratos B0–B4 (hoje são tiny/small/medium/huge com teto de 3,5M, e o PNAS usa 0,5/1/2/3,5/5,5M com teto ×1,05).
3. Proposta:
   - pacote novo **`ga_search/hmodehb/`**, que IMPORTA as primitivas de `modehb.py`, `strata.py` e `external_seed.py` sem modificá-las;
   - config nova `ga_search/config/hmodehb_b0b4/`;
   - artefatos em `artifacts/hmodehb_b0b4/`;
   - prefixo de id próprio.
   Só duas mudanças mínimas e retrocompatíveis em código compartilhado: (a) `monitor_report._kind` e `ROOT_GLOBS`; (b) opcionalmente, o `fit_sig` do `_controller`. As duas vêm com testes de regressão.

---

## 1. Literatura: o que importa para seleção, herança, regularização e multi-fidelidade

| Paper | Seleção | Herança | Regularização | Multi-fidelidade | Consequência para o desenho |
|---|---|---|---|---|---|
| **MO-DEHB** (2305.04502) [M] | NDS por budget. Desempate na última frente por crowding (NSGA-II) ou EpsNet. Na DE, o filho compete com o alvo na população global ordenada por NDS: se não há dominância, sai o membro da última frente com **menor contribuição de HV**. Métrica de avaliação: log HV-diff. | Não há; evoluem-se configurações. | — | Hyperband (η=3, b_min/b_max). Uma subpopulação por budget. O parent pool são os top-1/η da fidelidade inferior. rand1 com 3 pais e fallback para a população global se houver menos de 3. | Já implementado (`select_promotions`, `mo_select`, `_parent_pool`, `_global_pop`). Mantido como núcleo. O paper reporta início fraco em alguns benchmarks, o que justifica as sementes (B0 calibrado). |
| **LEMONADE** (1804.09081) [M] | População = conjunto não-dominado em (f_exp, f_cheap). Pai amostrado com p ∝ 1/p_KDE(f_cheap), e filhos aceitos com a mesma regra antes de treinar (favorece regiões pouco povoadas de params/FLOPs). | Lamarckiana: morfismos exatos (inserir camada identidade, Net2Wider ou zero-pad, skip convexo com λ=0) e aproximados (podar, remover, trocar por separável), estes com destilação. | Não usa Shake-Shake nem DropPath. | Não há; os dois estágios separam objetivos baratos de caros. | `lemonade.py` **não tem a amostragem por KDE** (usa sorteio na frente e `morph_prob`). Para o braço B ser "LEMONADE" de fato: implementar o KDE como opção, ou declarar o desvio. Usar o KDE sobre (log params, log FLOPs) por B_k na reserva exploratória. |
| **Net2Net** (1511.05641) [M] | — | Wider: cópia via g(j) e divisão dos pesos de saída por \|{x: g(x)=g(j)}\|. Deeper: identidade, vale para ReLU e não para sigmoide; BN exige ajuste. Ruído para quebrar simetria, que torna a preservação só aproximada. lr do aluno ≈ 1/10 do professor. | Não visa regularizar. | — | `external_seed.grow_blocks` já usa colunas de consumo zero (variante "zero-init" do Net2Wider) e identidade ou residual-zero. **Registrar a preservação medida (eval(), batch fixo), nunca alegar exatidão** quando há BN, ZCA, skip ou inception (MISSION A3). Aplicar `lr_old` ≈ 0,1×`lr_new` nos blocos herdados (o `protect=lowlr` já faz 0,01). |
| **EfficientNet** (ICML'19) [M] | Busca do B0 com ACC×(FLOPS/T)^w, w=−0,07. | — | dropout 0,2→0,5; stochastic depth 0,8. | Compound: d=α^φ, w=β^φ, r=γ^φ, com α·β²·γ²≈2, logo FLOPs ≈ 2^φ (α=1,2, β=1,1, γ=1,15 no ImageNet). | Resolução fixa aqui (γ=1), então FLOPs ∝ α·β². O grid (α,β) de `pnas_cs.py` é rebuscado no nosso regime (300k–5,5M). Os α,β do ImageNet **não** se transferem. O objetivo escalar do MnasNet é só uma referência: aqui o objetivo é Pareto (κ, FLOPs). |
| **Once-for-All** (1908.09791) [M] | Busca evolutiva sobre preditores de acurácia e latência (16K sub-redes amostradas). | Progressive shrinking kernel → profundidade → largura; a resolução é elástica o tempo todo. Primeiras D camadas mantidas; canais ordenados por norma L1. KD da rede maior. | — | A rede maior treina primeiro e as sub-redes herdam. | Direção oposta ao crescimento (shrink). Úteis: (i) a ordem de canais por importância como critério de quais filtros SPiFiL herdar ao **podar** (transferência B_k→B_{k−1}, desligada por padrão); (ii) preditor κ(genoma, curva) como sinal auxiliar opcional, **nunca** como fitness. |
| **Barlow Twins** (PMLR v139) [M] | — | — | L = Σ(1−C_ii)² + λΣ_{i≠j}C_ij², λ=5e-3, projetor 3×8192 (o desempenho cresce com a dimensão). | — | Projetor de 8192 é caro para modelos de 300k. Usar só como **métrica auxiliar de redundância** (off-diagonal da correlação cruzada entre vistas) no passo 13, sem entrar na fitness. |
| **VICReg** (openreview) [M] | — | — | λ=μ=25, ν=1, γ=1, ε=1e-4. Termo de variância com **desvio padrão** (o gradiente da variância some). | — | Métricas de colapso no passo 13: std média por dimensão (colapso → 0) e c(Z) off-diagonal. Já há equivalentes em `embedding_eval` (rank efetivo, std). Alinhar os limiares. |
| **LeJEPA** (2511.08544) [M parcial] | — | — | Predição + SIGReg (Epps-Pulley em projeções 1D aleatórias): um hiperparâmetro de trade-off, sem stop-grad, teacher ou scheduler, custo linear. [resumo, não verificado] λ≈0,05, 512–1024 slices, batch ≥128. A isotropia minimiza o viés de probes lineares/k-NN **sob restrição escalar de covariância**. | A loss de treino correlaciona com o linear probe no ImageNet (Fig. 1). | O SSL global é **LeJEPA+SIGReg real do repo** (`regularizers.SIGReg`, padrão 1024 slices). `sigreg_lambda`=0,05 já é o default. [H] A correlação loss↔probe não foi validada nos nossos domínios (o CLAUDE.md registra o descolamento MSE↔κ na destilação), então **não usar a loss de SSL para seleção**; ela é só um diagnóstico. |

Lacunas declaradas: o LEMONADE (seções > 5.1), o LeJEPA (seção ≥ 4) e os apêndices do OFA foram lidos só por resumo. Os números marcados [resumo] precisam ser checados no PDF antes de virar hiperparâmetro.

---

## 2. Auditoria do framework real

### 2.1 `modehb.py` (2085 linhas)

| Componente | Onde | Estado [M] | Uso no H-MODEHB |
|---|---|---|---|
| Genoma | `GLOBAL_GENES` (11) + `LAYER_GENES` (8) × 8 camadas, vetor contínuo [0,1]^DIM, decode por bins | Plano: `init`, `ssl`, `reg`, `protect`, `ae_schedule` e `pooling` são globais e misturados com a arquitetura. Profundidade ≤ 8. | **Substituir** pelo genoma de 5 specs (§3). `decode`/`encode` continuam disponíveis para o braço A ("legacy view") e para os testes de equivalência. |
| `repair` | l.296 | Idempotente, nunca muda `init`. Aplica `repair_for_spifil` por camada. | Reusar como reparo **da ArchitectureSpec**. As outras specs ganham reparos próprios. |
| `arch_hash`/`canonical` | l.396–410 | Hash do genoma ativo **sem a origem dos pesos**. | **Ponto crítico**: `_evaluate` faz cache por `(arch_hash, seed, fid, role)`. Um filho herdado com a mesma arquitetura de um candidato já avaliado **recebe o resultado em cache** e a herança é ignorada (`ctx["inherit"]` só entra quando `not cached`). No H-MODEHB, a chave do candidato tem de incluir a origem dos pesos (§7.2). |
| NDS / HV / crowding / EpsNet | l.525–620 | NDS do pymoo. HV exato em 2D, ref fixo 1,1. | Reusar tal como está. |
| `mo_select` | l.623 | Regra do paper (dominância, senão menor contribuição de HV na última frente). | Reusar dentro de cada B_k. |
| Brackets | `_hb_schedule` via `dehb.DEHBBase` | `fidelities=[5,15,45]` (padrão), η=3. Bracket 0 aumentado para caber as sementes. | Reusar com fidelidades em épocas da missão (5/15/45/100/200) [H: 5/15/45 na busca, 100/200 como extensão]. |
| Estratos | `strata.py`, `_insert_stratified`, `_promote` | Quotas `min_share`, ranking dentro do estrato e exploração por inclinação (`learning_slope`). `protected_brackets`. | Reusar com `strata = B0..B4` (§5). |
| Replay/resume | `run()`, `_call` | Replay por **índice de job**: confere `arch_hash`, fidelidade e papel (`ReplayMismatch`). `_save` atômico, que nunca encolhe o estado. | Reusar. **Lacunas**: (i) `state["config"]` guarda só o bloco `search`, sem `eval`/`train`, então uma mudança da fitness não é detectada no resume; (ii) um crash entre `result.json` escrito e `_save` refaz o job (custo duplicado e run W&B duplicado). |
| OOM retry | `OOM_RETRIES=10`, `OOM_RETRY_SLEEP_S=300` | Limitado (sem repetição infinita). **Diverge de R5** (FILA: "4×180 s"). | Parametrizar por config (`oom.retries`, `oom.sleep_s`) e registrar `oom_retries` no job. Após esgotar, `status=failed`, `infeasible=oom` e diagnóstico de memória (estimativa × pico). |
| `_fatal_cuda` | l.671 | Contexto CUDA envenenado: relança **sem registrar**, e o resume refaz o job. | Reusar. Testar que não sobra registro parcial. |
| PNAS-CS / LEMONADE hooks | `_pnas_*`, `_morph_child`, `_inherit_for` | Opt-in por bloco YAML. Ausente = comportamento original. | Os braços B e C rodam **o código existente** com config nova. |

### 2.2 `strata.py`
- `DEFAULT_STRATA` = tiny ≤300k < small ≤800k < medium ≤1,5M < huge ≤3,5M. **Não** é B0–B4. A estratificação aceita `strata:` por YAML (contíguos) e `param_cap`.
- `param_estimate` é estrutural (meta device). Após a avaliação, o `params` medido reclassifica (`_evaluate`). `_count` corrige pela mediana estimado/medido, porque o SPiFiL pode construir menos canais.
- `stratified_promotions`: quota `max(1, floor(min_share·n))` por estrato presente, depois o global, depois `explore_extra` por inclinação.
- Modelo de memória por fase (A SPiFiL / B por camada / C SSL), `mem_repair`, `spifil_images_repair` (N > D). Reusar para não gerar candidatos que estouram a GPU 3 (49 GB).

### 2.3 `run_search.py`
- **Fitness atual** (`fitness_mode: svm_kappa`): `svm_fitness(C=100, scaler=standard, readout=cov_head, cov_k=64, kappa_weight=1.0, tta=1, max_train=4000)` (search_v3*.yaml l.142). Isso é a régua **COV**. A fitness v2 exige ORIG: `readout=head`, `scaler=none`.
- Colapso: com `svm_kappa`, o colapso dimensional não penaliza (opt-in). O colapso total multiplica por `collapse_total_multiplier` (0,0).
- **WiSE** (`_wise_select`): o α ∈ {0, .25, .5, .75, 1} é escolhido pelo **mesmo κ de validação** que vira fitness. A fitness é então um máximo sobre 5 leituras, com viés otimista pequeno mas sistemático (§4.3).
- Checkpoint `encoder.pt` completo: encoder e head como módulos, genome, `anchor` (SPiFiL+AE), `ssl_state`, `wise_alpha`, `protect_masks`. O resume da promoção continua `ssl_state` (a trajetória SSL), não o blend.
- `_controller`: o id do W&B é `ctrl-sha1(output_dir_resolvido + fit_sig)[:16]`, com `fit_sig = C, kappa_weight`. **Não inclui `readout`, `scaler`, `tta` nem `whiten`**: ORIG e COV no mesmo diretório colidiriam no mesmo controller.
- `main`: escreve `config_resolved.yaml` só se ainda não existe (uma mudança posterior não é detectada), `_cap_cpu_threads` (OMP/OPENBLAS/MKL) e `resume` automático se o estado existe. **Não há trava** contra dois processos no mesmo `search_state.json` (exigência do A5).
- Dados: `load_data` devolve train/val e `identity["sha256"]` (split_hash). O teste não é carregado (bom), mas nenhum teste automatizado garante isso.

### 2.4 `pnas_cs.py`, `lemonade.py`, `monitor_report.py`
- **PNAS-CS**: famílias (base, α, β) com níveis P0–P4 = 0,5/1/2/3,5/5,5M, faixas por ponto médio geométrico e `cap_factor=1,05` (5,775M, **viola** "proibido > 5,5M"). Crescimento P_k→P_{k+1} por `_grow_fp`. Fase 0 com ablação AE. `fit_scaling_law` (κ = a + b·log10 params, mais saturação por ganho < 0,01) é reutilizável para o §6.
- **LEMONADE**: morfismos deepen/widen/residual/kernel, `reusable_prefix`, `function_preserving` opt-in e `pick_parent_checkpoint(le_child|max)`. Sem KDE.
- **monitor_report** (régua de TESTE, fora do controller): ORIG = `fit_svm` C=100 sem scaler em `head(encoder(x))`, mais a COV de referência. Chave `kind|ds|sig|cand|fid`. Pontos de colisão:
  - `_kind()` decide pelo **nome do diretório** (substrings `pnas_cs`, `lemonade`, `seeded`, senão `hybrid`). Um diretório novo `artifacts/hmodehb_b0b4/...` cairia em `hybrid`, e o braço B (`.../lemonade/...`) em `lemonade`. Como os `cand_id` recomeçam em `c00001`, **as chaves colidiriam com as buscas H-MODEHB e H-LEM existentes** e resultados seriam pulados como "já feitos".
  - `_fit_sig` cai na convenção de nome de diretório porque o estado não guarda o bloco `eval`.

---

## 3. Genoma hierárquico (5 specs)

O genoma deixa de ser um vetor indiferenciado. Ele vira um **dataclass versionado** com cinco specs. Cada spec tem genes próprios, máscara de atividade condicional e reparo próprio. A DE opera **por segmento** (um vetor por spec), e o crossover troca **specs inteiras ou blocos inteiros**, nunca genes soltos de specs diferentes.

```
HGenome(v=1) = {
  dataset, bk, seed,
  arch:   ArchitectureSpec,
  init:   InitializationSpec,
  local:  LocalLearningSpec,
  growth: GrowthSpec,
  glob:   GlobalTrainingSpec,
  lineage: {ancestor_cand_id, ancestor_ckpt_sha256, ancestor_fidelity}   # não-gene, faz parte da identidade
}
```

### 3.1 ArchitectureSpec
- `stages`: lista de até 4 estágios: `stem` | `local` | `intermediate` | `abstract`. Nem todos são obrigatórios, e a profundidade é evolutiva.
- Por bloco: `op` ∈ {conv, dwsep, inception, pointwise, depthwise}, `kernel` ∈ {1,3,5,7}, `channels` (int, escala log), `residual`, `activation`, `norm` (∈ {none, batch, zca, dbn, batch_zca, batch_dbn}), `pool` (∈ {none, max, avg}), `softhebb`.
- `head`: `pooling` ∈ {gap, gem, moments, meanstd}, `multiscale` (lista de taps de estágios, só no braço E), `embed_dim` = **1280 fixo** (não é gene).
- **Reparos**:
  - herdados de `modehb.repair`: pointwise ⇒ k=1; dwsep/depthwise fora do stem; residual só se cin==cout e sem pool (ou projeção 1×1); nº de pools com mapa final ≥ `min_spatial`;
  - `channel_space` monotônico opcional;
  - **orçamento**: params(encoder + head 1280) ∈ faixa de B_k, ou falha `infeasible=param_cap` (5,5M é teto duro, sem fator 1,05);
  - `mem_repair` contra o orçamento da GPU 3;
  - taps multiescala só em estágios existentes; a projeção leve de cada tap entra no orçamento.

### 3.2 InitializationSpec (por bloco NOVO; blocos herdados ficam `inherit`)
- `method` ∈ {spifil0, spifil1_proto, spifil2_inv, spifil3_pred, spifil4_compose, random}. A interface é `KernelInitializer.fit(prefix, train_data, block_spec, context)` do Agente 2. Os métodos 1–4 só entram se aprovados no gate do A2.
- `images_per_class` ∈ {1,2,4,8,16}, `selector` ∈ {fisher, mahalanobis, fps}, `labels` ∈ {supervised, label_free}.
- **Reparos condicionais**:
  - op não suportado pelo SPiFiL ⇒ `repair_for_spifil` ajusta o **op**, nunca troca spifil→random em silêncio;
  - `spifil2/3` exigem prefixo ≥1 bloco (o stem usa spifil0/1);
  - N>D via `spifil_images_repair`;
  - `random` só nos braços de controle (gêmeos), registrado.

### 3.3 LocalLearningSpec (só blocos com elementos novos)
- `ae` ∈ {on, off}. Se `on`: `target` ∈ {pixel, feature, masked, predictive}, `schedule` ∈ {joint, decoder_first}, `protect` ∈ {freeze, lowlr, full}.
- `ae=off` ⇒ os demais genes ficam **inativos** (mascarados do hash e da DE, como as camadas inativas hoje). O decoder é descartado e registrado à parte (não conta nos params de B_k).

### 3.4 GrowthSpec (só com ancestral)
- `op` ∈ {none (busca independente), width, depth, residual, compound, structured (inception/dwsep), stage_aware, abstract_init}, `factor` (widen ∈ {1.25, 1.5, 2}), `position` (estágio/bloco), `inherit_mode` ∈ {function_preserving, prefix, reinit}, `noise_eps` (quebra de simetria Net2Net).
- **Reparos**:
  - `op=none` ⇔ sem ancestral (braço A);
  - deepen-identidade só com ReLU e sem BN/skip; senão `residual_zero` (`deepen: auto`, como no PNAS);
  - o resultado tem de cair em B_{k} alvo; se não cair, o fator é reajustado (busca 1D, como `fit_level`);
  - `reinit` é o controle "herança × reinicialização" da mesma arquitetura (A3 G).

### 3.5 GlobalTrainingSpec
- `ssl` ∈ {lejepa, none} (outras losses só como braços identificados), `reg` ∈ {sigreg} (weak_sigreg só como braço), `sigreg_lambda` ∈ {0.02, 0.05, 0.1}, `projector` ∈ {F, T}, `lr_ratio_inherited` ∈ {0.01, 0.1}.
- **Não são genes**: épocas (vêm da fidelidade), grade WiSE (fixa no config), batch (fixo 64).

### 3.6 Operadores
- **DE rand/1 + binomial** por spec, com `mutation_factor` e `crossover_prob` por spec (padrão 0,5/0,5). A máscara é o AND de três máscaras: genes ativos, condicionais e estágio.
- **Recombinação só entre compatíveis**: mesmo B_k e mesma topologia de estágios (nº de estágios e posições de pool). Fora disso, só mutação. O crossover entre specs troca a spec inteira do mutante (ex.: InitializationSpec de A com ArchitectureSpec de B).
- **Crescimento** = GrowthSpec aplicado a um ancestral; reusa `pnas_cs.scale_genome/insert_layer/mutate` e `lemonade.morph`.

---

## 4. Fluxo de 17 passos por candidato, com pontos de extensão

| # | Passo | Implementação / ponto de extensão |
|---|---|---|
| 1 | Dataset e B_k alvo | `_next_stratum` (round-robin B0..B4) sobre `strata=B0..B4`. Dataset fixo por processo. |
| 2 | Ancestral por **validação** | Novo `hmodehb.ancestry.pick_ancestor(bk)`: frente ORIG da subpop B_{k−1} (crescimento) ou B_k (mutação) na maior fidelidade disponível. Desempate por menor FLOPs e depois por crowding. Só `κ_val`, nunca teste. Sem ancestral no braço A. |
| 3 | Mutação | DE por spec (§3.6) ou GrowthSpec (morfismo). Registra `op`. |
| 4 | Init dos blocos novos | `KernelInitializer.fit` (A2). Hoje é `init.init_block` em `_grow`. |
| 5 | Build, params, FLOPs | `strata.param_estimate` antes de treinar e `profile()` após. Rejeitar > teto de B_k antes de gastar GPU. |
| 6 | Herança | `external_seed.grow_blocks` (function_preserving), `lemonade.load_inherited_blocks` (prefix) ou nada (reinit). |
| 7 | Preservação | `‖f_filho − f_pai‖/‖f_pai‖` no batch fixo de validação, em `eval()`, **antes** do treino. Já existe `_fp_info` no PNAS; generalizar. Registrar, não exigir zero. |
| 8 | SPiFiL nos filtros novos | `grow_blocks` (Net2Wider com colunas de consumo zero) mais o alocador SPiFiL. |
| 9 | AE local | `local_ae.train_local_ae` só nos blocos com elementos novos, conforme LocalLearningSpec. |
| 10 | SSL global | `ssl_trainers.train_global_ssl` (LeJEPA+SIGReg) com `ElementProtect`. |
| 11 | WiSE | `_wise_select` com a mesma grade para filho e ancestral (§4.3). |
| 12 | Avaliação ORIG (val) | **Novo** `fitness_v2` (§5): head 1280, `fit_svm` C=100, sem scaler. |
| 13 | Estabilidade, colapso, latência, FLOPs | `EmbeddingEvaluator` (rank efetivo, std), mais std/cov VICReg e redundância Barlow como **auxiliares**, `profile` (latência, MACs) e pico de memória. |
| 14 | Filho × ancestral | Δκ, ΔFLOPs, Δparams no mesmo protocolo (§6). |
| 15 | População / Pareto | `_insert_stratified` / `mo_select` dentro de B_k, mais Pareto global só entre comparáveis. |
| 16 | Promoção multi-fidelidade | `_promote` → `stratified_promotions` (quotas B0–B4 + exploração por inclinação). |
| 17 | Checkpoint, linhagem, config | `encoder.pt` + `sha256`, `lineage.jsonl` (aresta pai→filho com op, mecanismo, pesos herdados/novos, shapes, preservação), `config_sha`, `fitness_sig`. |

### 4.1 Onde fica cada passo
- **Controller** (`hmodehb/hsearch.py`, subclasse de `modehb.MODEHB`): passos 1–3, 5 (estimado), 15–17. Sobrescreve `_acquire`, `_decode`, `_encode`, `_evaluate` (chave) e `_insert`.
- **Evaluator** (`hmodehb/hevaluate.py`, subclasse de `run_search.Evaluator`): passos 4–14. Sobrescreve `_evaluate` e `_svm_quality`; reusa `_grow`, `_grow_fp`, `_wise_select` e `profile`.

### 4.2 Multi-fidelidade
Fidelidades da busca 5/15/45 épocas, η=3; extensões 100/200 via `_extend` (já fora do HV). O "orçamento calibrado por porte" (calibração por estrato, já existente) evita que B3/B4 sejam descartados por convergirem devagar. A promoção exploratória por `learning_slope` vale em B2–B4 (`explore_strata`).

### 4.3 WiSE e viés de seleção [H]
O α é escolhido pelo mesmo κ_val que vira fitness. O efeito é um máximo de 5 leituras, levemente otimista.
- **Mitigação**: a mesma grade e a mesma regra para filho e ancestral (o Δκ fica justo). Registrar sempre `kappa_val_alpha0`, que é a leitura sem seleção.
- **Ablação** (Fase B do A5): dividir o val estratificado em `val_sel` (escolhe α) e `val_fit` (fitness) e medir a diferença. Se for > 0,005 κ, adotar a divisão na v2.1.

---

## 5. Fitness v2

**Objetivos (minimizados, em [0,1])**: `f0 = 1 − κ_val_ORIG`; `f1 = log10(MACs/1e6)/log10(1e11/1e6)`, recortado em [0,1] (limites fixos, então o HV é comparável entre runs). Latência não entra (`use_latency=False`).

**κ_val_ORIG**:
- embedding `head(encoder(x))` de 1280 dimensões, `tta=1`, `whiten=none`;
- `core.svm.fit_svm` linear, **C=100**, ovo, `max_iter=-1`, **scaler=none**;
- `kappa_weight=1.0`; `max_train` estratificado e semeado (registrado);
- val completo, nunca subamostrado; teste nunca carregado.

`gram_reduce` é permitido (decisões idênticas até ~1e-14) e registrado.

**Falhas**:
- embedding não finito → `status=failed`, `reason=nan`;
- colapso total (std < 1e-4) → `status=ok`, `κ` medido e `collapse.total=True` (o κ já é ~0, então sem multiplicador oculto);
- colapso dimensional → só flag.

**Auxiliares (nunca em F)**, em `result["aux"]`: `kappa_cov` (cov_head, C=100, standard), `kappa_gtp`, `macro_f1`, `balanced_acc`, `kappa_val_alpha0`, métricas label-free, std/cov VICReg, redundância Barlow, loss final de SSL, inclinação da curva.

**Assinatura**:
```
fitness_sig = sha256(json({"v":"fitness_v2","readout":"head","embed_dim":1280,"svm":"core.svm.fit_svm",
  "C":100.0,"scaler":"none","kappa_weight":1.0,"tta":1,"whiten":"none","max_train":N,
  "wise_alphas":[...],"val_sha256":split_hash,"flops_bounds":[1e6,1e11]}))[:12]
```
Gravada em cada job, no `state` e no id do controller W&B. O resume e o Pareto **recusam** misturar assinaturas, o que elimina o problema "C=0.1 × C=100" por construção.

**Implementação**: `hevaluate._svm_quality` chama `supervised_fitness.svm_fitness(..., C=100, scaler="none", kappa_weight=1.0, whiten="none")` com `readout="head"`. Não alterar `svm_fitness` nem `core.svm`.

---

## 6. Subpopulações B0–B4, Δκ e saturação

### 6.1 Faixas
`strata: {B0:[0,300000], B1:[300000,800000], B2:[800000,1500000], B3:[1500000,3000000], B4:[3000000,5500000]}`, `param_cap: 5500000`. Os params medidos incluem encoder + head 1280. Decoders e projetores ficam registrados à parte.

### 6.2 Mecânica
- **Proteção**: `min_share=0.2` (1 quota por B_k em cada promoção) e `_insert_stratified` (um estrato sub-representado expulsa o pior do estrato mais povoado).
- **Recombinação entre compatíveis** (§3.6).
- **Transferência controlada**:
  - para cima: crescimento a partir do ancestral escolhido no passo 2, com no máximo `transfer_per_bracket` (padrão 1) por par B_k→B_{k+1};
  - para baixo (poda à la OFA/LEMONADE-ANM): desligada por padrão.
- **Reserva exploratória**: 15% das aquisições do rung 0 são aleatórias novas (sem ancestral), amostradas por 1/KDE em (log params, log FLOPs) dentro de B_k. Mais `explore_extra` por inclinação.
- **Pareto global** só entre jobs com o mesmo `(dataset, split_hash, fitness_sig, fidelidade)`. Extensões ficam fora.

### 6.3 Δκ e custo
- `Δκ = κ_val(filho, f) − κ_val(ancestral, f_anc)`, como na missão. Também `Δκ_matched`, com o ancestral **na mesma fidelidade**: se ele não existir, enfileira-se a promoção do ancestral (custo contabilizado, papel `control_ancestor`). Mais `ΔMACs`, `Δparams`, `Δgpu_h` e a eficiência `Δκ / Δlog10 MACs`.
- Δκ pode ser negativo, e resultados não monotônicos são permitidos e reportados.
- A sobrevivência por operador é a fração de filhos com Δκ_matched > 0 e que entram na frente de B_k.

### 6.4 Saturação
- Por (dataset, braço): o melhor κ_val ORIG por B_k na fidelidade máxima, mais `pnas_cs.fit_scaling_law` (κ = a + b·log10 params).
- B_sat é o primeiro B_k cujo ganho sobre B_{k−1} fica abaixo de 0,01, com IC bootstrap (1000 reamostras do val) contendo 0.
- Registrar curvas por fidelidade (5/15/45/100/200) para separar "saturou o porte" de "faltou treino".

---

## 7. Integridade

### 7.1 Config e dados
- `config_sha` = sha256 do YAML resolvido canônico (chaves ordenadas; excluídos só `wandb.mode` e `device`). Fica em `state["config_sha"]` e em cada job.
- No resume: se diferir → `ConfigDrift` (erro), salvo `--allow-drift chave1,chave2`, que registra a deriva em `state["drift_log"]`.
- `state["config_full"]` guarda o config inteiro, incluindo `eval` e `train` (fecha a lacuna de §2.1).
- `split_hash` (`identity.sha256`) e `spifil_pool` ids são conferidos no resume. Splits imutáveis. HAM10000 por `lesion_id` (A1/E).

### 7.2 Ids e chave de candidato
- `cand_id = "h4" + "-" + arm + "-" + "c%05d"` (ex.: `h4-D-c00012`). Não colide com `c00012` das buscas existentes.
- `cand_key = sha256(canonical(HGenome sem seed) + lineage.ancestor_ckpt_sha256 + init_provenance + seed)`.
- O cache e o resume da promoção usam `cand_key`, não `arch_hash`. Corrige a colisão herança × cache de §2.1. Promoção é a mesma `cand_key` com fidelidade maior.

### 7.3 Resume sem duplicar
1. Antes de `_call`, grava `pending/<job>.json` com `cand_key`, fidelidade e `config_sha` (escrita atômica).
2. Se o processo cai depois do `result.json` e antes do `_save`, o resume encontra `result.json` com a mesma `cand_key` e o `sha256` do checkpoint conferido, e **adota** o resultado sem re-treinar e sem novo run W&B.
3. O replay por índice continua (determinismo do RNG). `ReplayMismatch` agora também compara `cand_key` e `fitness_sig`.
4. Trava: `fcntl.flock` em `<out>/controller.lock`, com pid, host e início. O segundo controlador sai com erro (exigência do A5).

### 7.4 Execução segura
- Sementes: 3 seeds de busca por (dataset, braço). As seeds auxiliares (SVM subsample, SPiFiL pool) são derivadas e registradas.
- BLAS e threads: `_cap_cpu_threads` (reusado) mais `svm_blas_threads=1`.
- OOM: retries parametrizados (R5: 4×180 s). Depois, `failed/oom` com estimativa × pico. Uma redução científica por memória (ex.: menos `images_per_class`) é **variante distinta** registrada (`mem_variant`), nunca uma troca silenciosa.
- `_fatal_cuda`: relança sem registro, e a fila do A5 reinicia o processo.

### 7.5 Teste fora do controller
O controller e o evaluator nunca carregam o split de teste. O `monitor_report` (processo separado) mede ORIG no teste para todos os jobs (R2). Nada volta à busca.

---

## 8. Experimento principal A–E (por dataset)

Ordem dos datasets: RS, HAM10000, Flowers → Eggs, Larvae, Protozoan (raw e pre separados) → AID e Coconut (controle). Mesmo orçamento de GPU-h por braço, mesmas fidelidades, mesma fitness v2, 3 seeds.

| Braço | Definição | Código |
|---|---|---|
| **A** MO-DEHB independente por porte | 5 subpops B0–B4, sem herança e sem transferência; init SPiFiL-0; DE por spec. | H-MODEHB, `growth.op=none` |
| **B** LEMONADE | Morfismos + Lamarck sobre B0–B4. KDE opcional (`lemonade.kde: true`, novo); sem ele, o desvio é declarado. | `lemonade.py` existente + estratos B0–B4 |
| **C** PNAS-CS | Compound scaling (α,β) com `budgets=[3e5,8e5,1.5e6,3e6,5.5e6]` e `cap_factor=1.0`. | `pnas_cs.py` existente, só config |
| **D0** crescimento convencional (controle do gate) | Crescimento progressivo B0→B4, init SPiFiL-0 nos blocos novos. | H-MODEHB, `growth.op≠none`, `init.method=spifil0` |
| **D** crescimento + SPiFiL abstrato | D0 com o melhor SPiFiL-k aprovado pelo A2. | idem, `init.method=spifil_k` |
| **E** D + agregação multiescala | `arch.head.multiscale` (taps → projeção leve → concat → 1280). | idem |

**Gate final**: A × D0 × D, sem teste na seleção.
- HV(GPU-h), com IC entre seeds.
- Melhor κ_val ORIG por B_k.
- Teste Wilcoxon pareado por (dataset, B_k).
- O teste ORIG só é reportado para a frente selecionada em validação (mais "melhor retrospectivo" separado, A1/E7).

---

## 9. Painéis W&B (projeto do controller, um run por `(dataset, arm, seed, fitness_sig)`)

| Painel | Métricas |
|---|---|
| HV × geração e × GPU-h | `hv/global`, `hv/B{k}`, `cost/gpu_hours_cum` (reusa `ControllerRun.log_generation`) |
| κ × params, κ × FLOPs | scatter `kappa_val_orig` × `params`/`macs`, cor = B_k, forma = braço |
| Curvas B0–B4 | melhor κ_val por B_k × fidelidade; `fit_scaling_law` sobreposto; marcador de B_sat |
| Melhoria filho/ancestral | histograma de `delta_kappa` e `delta_kappa_matched` por operador; scatter Δκ × Δlog MACs |
| Efeito por tipo de SPiFiL | boxplot κ_val por `init.method` (blocos novos), mesmo B_k e fidelidade |
| Sobrevivência por operador | fração que entra na frente de B_k, por `growth.op`/DE/aleatório |
| Contribuição por linhagem | DAG (tabela `lineage`: pai, filho, op, Δκ); HV contribuído por raiz |
| Distribuição prof./largura | histogramas de profundidade e largura média por B_k e por geração |
| Falhas | contagem por `reason` (oom, nan, param_cap, mem, fatal); `oom_retries` |
| Curvas de treino | `ssl/loss_total`, `ae/loss` por bloco, `wise/kappa_a*`, preservação (passo 7) |

---

## 10. Namespace e isolamento (não quebrar H-MODEHB, H-LEM, H-SEED, PNAS-CS)

| Item | Novo | Existente (intocado) |
|---|---|---|
| Código | `ga_search/hmodehb/{genome.py, specs.py, hsearch.py, hevaluate.py, fitness_v2.py, ancestry.py, lineage.py, integrity.py, run_hsearch.py}` | `ga_search/modehb/*` (só importado) |
| Configs | `ga_search/config/hmodehb_b0b4/{base.yaml, arms/{A,B,C,D0,D,E}.yaml, datasets/→ link lógico aos datasets existentes}` | `ga_search/config/modehb/*` |
| Artefatos | `artifacts/hmodehb_b0b4/<dataset>/<arm>/seed<k>/` (nomes de diretório **sem** as substrings `lemonade`, `seeded`, `pnas_cs`) | `artifacts/modehb_*`, `artifacts/pnas_cs` |
| Ids | `h4-<arm>-c#####`, controller `hctrl-sha1(out+fitness_sig+config_sha)` | `c#####`, `ctrl-*` |
| W&B | grupo `hmodehb_b0b4/<dataset>`, tag `fitness_v2` | inalterado |

**Mudanças mínimas em código compartilhado** (retrocompatíveis, com teste de regressão):
1. `monitor_report.py`: `ROOT_GLOBS += ["artifacts/hmodehb_b0b4/*/*/*"]`; `_kind()` passa a reconhecer `hmodehb_b0b4` **antes** das outras substrings (`kind="h4_<arm>"`); `_fit_sig` lê `state["fitness_sig"]` quando existe.
2. (Opcional) `run_search._controller`: incluir `readout`/`scaler` no `fit_sig` só quando diferem do legado. Sem isso, o H-MODEHB usa o próprio `_controller`.

Nenhuma outra linha em `ga_search/modehb/`. Os braços B e C rodam o código existente através de `run_hsearch.py`, que só injeta estratos B0–B4, fitness v2 e namespace. Precisam do hook de readout `head`/scaler `none`, que **já é suportado** por config (`svm_fitness.readout: head, scaler: none`).

---

## 11. Testes necessários

### 11.1 Unitários (`tests/hmodehb/`)
1. **Specs**: encode→decode→encode idempotente por spec. Os reparos são idempotentes e respeitam as condicionais: `ae=off` mascara os genes; GrowthSpec vale `none` ⇔ sem ancestral; deepen-identidade nunca em BN/skip. Propriedade com 1000 genomas aleatórios.
2. **Orçamento**: nenhum genoma aceito com params > teto de B_k nem > 5,5M; classificação nas fronteiras (300000 → B0, 300001 → B1).
3. **Compatibilidade**: o crossover só ocorre entre mesma B_k e mesma topologia; os filhos permanecem válidos.
4. **cand_key**: mesma arquitetura com ancestrais diferentes dá chaves diferentes (regressão da colisão de cache). A promoção mantém a chave.
5. **fitness_v2**: `fit_svm` mockado recebe `C=100`; a fitness não usa scaler nem whiten; o readout é `head`, com 1280 dimensões; `kappa_weight=1`. A assinatura muda se qualquer campo muda. NaN dá `failed/nan`.
6. **Δκ / saturação**: casos à mão (não monotônico, ancestral ausente, IC).
7. **Promoções protegidas** B0–B4: quotas com n < 5 estratos, com `counts` e com exploração por inclinação.
8. **Integridade**: `ConfigDrift` em mudança de `eval.svm_fitness.C`; `--allow-drift` registra; a trava `flock` impede o segundo controlador.
9. **monitor_report**: `_kind` e `_key` para `hmodehb_b0b4/.../lemonade/...` não colidem com `modehb_lemonade`. Os testes existentes de `tests/modehb/` seguem verdes.

### 11.2 End-to-end (R9; CPU/GPU pequena, dataset sintético de `tests/modehb/data`)
- Mini-busca real por braço (A, B, C, D0, D, E) com fidelidades [1,2,4], η=2, 2 brackets, `max_steps_per_epoch=2`, pelo menos um candidato em cada B_k (larguras sintéticas) e um B4 de ~5M.
- Exercitar sementes, DE, crescimento com herança function_preserving e prefix, reinit, WiSE, promoção, extensão e gêmeo.
- **Zero failed** (exceto as falhas injetadas).
- Conferir: DAG de linhagem acíclico com toda aresta presente no `lineage.jsonl`; preservação registrada; Pareto por B_k; exportação B0–B4 (`best_by_bk.json` + checkpoints autocontidos que recarregam e reproduzem o embedding bit a bit em `eval()`).
- **Regressão**: replay de um `search_state.json` real (cópia de `artifacts/modehb_hybrid/<base>`) com o código `modehb` após as mudanças em `monitor_report` → nenhum `ReplayMismatch`.

### 11.3 Interrupção, resume e falhas
1. `kill -9` em pontos sorteados (≥ 10 rodadas): durante o SPiFiL, o AE, o SSL, o WiSE, a escrita do checkpoint, entre `result.json` e `_save`, e entre brackets.
   - O resume produz uma lista de jobs **idêntica** à de uma execução sem interrupção (mesmos `cand_key`, F e ordem).
   - Nenhum `cand_id` duplicado, nenhum run W&B de candidato duplicado e o mesmo id de controller.
   - Checkpoint parcial (`.tmp`) nunca é adotado.
2. **OOM injetado** (mock que levanta `OutOfMemoryError` k vezes): k ≤ retries → `ok` com `oom_retries=k`; k > retries → `failed/oom`, sem loop infinito.
3. **Fatal CUDA injetado**: nenhum registro parcial; o resume refaz o job.
4. **NaN injetado** no SSL: `failed/nan` e a busca segue.
5. **Arquivo ausente** (checkpoint do ancestral apagado ou movido): o filho falha com `reason=missing_ancestor`, ou cai para `reinit` como **variante distinta** se configurado. Nunca crash do controller.
6. **Deriva de config** no resume: erro explícito.
7. **Dois controladores** no mesmo diretório: o segundo sai.

---

## 12. Riscos e pendências para o gate

- Depende do gate do A3 (BackboneSpec/GrowthSpec versionados, cabeça 1280 padronizada) e do A2 (interface `KernelInitializer` e o SPiFiL-k aprovado). Até lá, D = D0.
- A mudança COV → ORIG na fitness altera o ranking em relação às buscas atuais. Os resultados v2 **não** se comparam com as frentes das buscas existentes (réguas diferentes), só o teste ORIG do monitor.
- O custo de `Δκ_matched` (promoção do ancestral) precisa de teto (≤ 20% do GPU-h do braço) [H].
- Os números do LeJEPA e do OFA marcados [resumo] devem ser verificados antes de virarem hiperparâmetros.
- Reconciliar o OOM retry: código 10×300 s × R5 4×180 s (decisão do usuário/A5).
- Sem trava de controlador nas buscas existentes: fora do escopo do A4 (A5), mas o H-MODEHB já nasce com ela.
