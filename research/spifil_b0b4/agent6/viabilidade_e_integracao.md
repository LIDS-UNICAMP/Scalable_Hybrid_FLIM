# Agente 6: viabilidade da atenção fixa (sem pesos) em cada plano

> 17/11/2026, host jaci. Fontes:
> - MISSION.md (inclusive R11: embedding ≥ 1280);
> - ATTENTION_ADDENDUM.md;
> - FILA_E_INSTRUCOES.md (R1–R11);
> - documentos dos Agentes 1–5 (citados por arquivo e seção).
>
> Implementação: `ga_search/modehb/fixed_attention.py`, só na cópia `scratchpad/attn6`. Patch em `agent6/fixed_attention.patch`. Nada de produção foi tocado.
> Status de cada afirmação: **[medido]** = artefato ou teste; **[doc]** = documento de outro agente; **[hipótese]** = proposta a testar.

## 0. O que o operador é, e o que ele garante

- **Zero parâmetros.** Nenhum `nn.Parameter`. As estatísticas de alinhamento (média, PCA, CCA) ficam em *buffers* ajustados só no treino. **[medido]** 148 testes, entre eles `sum(p.numel() for p in attn.parameters()) == 0` para todos os tipos e alinhamentos, e o embedder também com 0.
- **Dois tipos, sempre identificados.**
  - *Sem parâmetros*: A self, B pool, E covariância, e C/D com `alignment=none`.
  - *Com estatística de treino*: C/D com `shared_channels`, `pca` ou `svd`. A flag `fitted_statistics` vai para cada linha de resultado.
- **Preservação de função.** Y = (1−α)X + α·Op(X). Com **α=0, a saída é bitwise igual a `head(encoder(x))`**, com curto-circuito e também calculando o operador (`force_compute`). **[medido]** Testado para os 5 tipos × 6 poolings.
  - Inserir um operador com α=0 num crescimento é um morfismo **exato** (veredito `exact` na GrowthSpec do A3).
  - Subir α depois é uma mudança explícita e registrada.
- **R11 respeitado.** Nenhum operador muda o número de canais. A cabeça original continua com a mesma dimensão (≥ 1280). **[medido]** `test_genes_never_change_embedding_dim`.
- **Diferenciável nas ativações.** O gradiente atravessa o operador em `train()`. **[medido]** `test_gradient_reaches_activations` e o caminho α=1. Não há o que atualizar.
- **Pós-treino.** `embed()` roda em `eval()` + `no_grad()`. `probe/fit` não mexem nas running stats de BN. **[medido]** `test_fit_and_probe_do_not_touch_bn_running_stats`.
- **Memória.** As queries ficam em resolução cheia (N = H·W). As keys/values são `tokens` ∈ {16, 49, 196}, por pooling adaptativo. A conta é feita em pedaços exatos de queries, O(B·chunk·M), e não O(N²).
  - Exemplo: mapa 56² com 196 tokens e lote de 64 dá 64·3136·196·4 B ≈ 157 MB de logits. A atenção cheia 3136² daria 2,5 GB.
  - O reparo reduz os tokens se mesmo assim estourar.
- **Custo** (smoke em CPU no RS, `agent6/smoke_cpu_rs.md`). O FLOP de atenção **não aparece** no `profile()` atual, que só conta Conv2d/Linear (A3 §2.5). `attention_cost()` precisa ser somado ao f1.
  - **Unidade (CHECK C5): FLOPs = 2·MAC em todo o texto.** O campo `flops` da busca é MAC (conferido: o `encoder_macs()` do módulo bate com ele nos 74 checkpoints).
  - Na c00009 (280 canais, 56²): self ≈ 0,18 GFLOP, cross ≈ 0,14, covariância ≈ 0,99. O encoder com a cabeça tem 1,75 GFLOP (0,875 GMAC). A covariância equivale a **57% do encoder**, não ao dobro.

---

## 1. Agente 1: diagnóstico (A1a/A1b/A1d)

**Onde entra:** como **leitura diagnóstica** sobre os mesmos checkpoints, sem treino. Os E0–E8 usam exatamente os checkpoints "melhor por validação por (dataset, kind, estrato)", na lista `agent6/checkpoints_E0_E8.csv`.

**O que muda:**
- O **E0 já é a régua ORIG na validação**, que segundo A1a §D l.345 "não existe em nenhum job". Em 74 checkpoints, o E0 preenche essa lacuna sem custo de treino.
- Os demais E1–E5 respondem à hipótese do adendo: uma CNN rasa ganha contexto global com operações fixas? Isso se cruza com A1a §A2 (profundidade ótima de 4–6, pools 3–4) e §A4 (+1 bloco: Δκ mediano −0,011).

**Riscos ligados aos achados do A1:**

| Achado do A1 | Consequência para a atenção | Mitigação no código |
|---|---|---|
| Mapas ótimos de 28²–14² (784–196 posições); 7² piora (A1a §A2) | Com 7², tokens=49 já é o mapa inteiro (Q=K=V=X exato); com 56² (c00009 RS), keys são *pooled* | reparo `tokens ≤ posições`; pedaços de queries |
| Rank efetivo das ativações 0,11–0,16·C nos blocos profundos; \|corr\| 0,42–0,50; 40–59% dos filtros com par cos>0,9 (A1a §B3) | similaridade cosseno alta e pouco discriminativa, softmax quase uniforme, operador vira "média global" | τ pequeno (0,03–0,1), `normalization=zscore`, controle `aggregation=mean` para medir se a similaridade importa |
| Canais mortos raros, mas até 32% em huge raw (larvae_raw c00004 bloco 4) | z-score e cosseno com variância/norma zero | `EPS` no z-score e no cosseno, encolhimento no branqueamento; **[medido]** sem NaN com metade dos canais mortos, mapas nulos e constantes, bf16/fp16 |
| COV − ORIG mediana +0,123 (A1b §D1) | E5 (2ª ordem dentro do mapa) tem de ser comparado ao COV sem duplicá-lo | E5 age antes da cabeça e lê ORIG; o COV entra como coluna de referência |
| n=1 seed, regret ORIG de 0,047 (A1b Q7) | as diferenças de κ entre E0 e Ek podem ser ruído | relatar Δκ com IC bootstrap por imagem (pareado: mesmo SVM e mesmas imagens) |

**Perguntas ao Agente 1 (com a minha resposta padrão):**
1. Posso usar `A1a_best_by_budget.csv` como lista oficial, ou a lista nova `agent6/checkpoints_E0_E8.csv`? A nova é o melhor por (dataset, **kind** hybrid/lemonade, estrato), regerada no momento do lançamento e congelada com md5. → *Padrão: a nova; ela é um superconjunto por kind.*
2. Você aceita o E0 (ORIG na validação, C=100, sem scaler) como preenchimento da sua tabela de ORIG na validação? → *Padrão: sim, marcado `source=agent6_E0`.*
3. Para o τ: você consegue medir, no lote fixo de treino, a entropia da softmax de similaridade cosseno entre posições por bloco, para τ ∈ {0,03; 0,1; 0,5}? É label-free e ajuda a cortar a grade. → *Padrão: eu calculo no smoke da GPU 3 e te mando.*
4. Os 4 casos de canais mortos > 10% (larvae_raw c00004, protozoan_raw c00004, RS c00054, flowers c00002) devem entrar como teste de estresse no E0–E8? → *Padrão: sim, como "estresse", fora do ranking.*
5. Para "rasos × profundos" no relatório: o corte é por profundidade (2–3 × ≥5 blocos) ou por resolução final (≥28² × ≤14²)? → *Padrão: os dois, com profundidade como eixo principal.*

---

## 2. Agente 2: SPiFiL abstrato (prep_literatura_e_desenho.md)

**Onde entra:**
- (a) **Validação dos operadores e alinhamentos sem treino**, conforme os papéis definidos no adendo. A fase A0 do A2 já é "init-only, cabeça aleatória fixa", então os operadores podem ser lidos como readouts extras nessa fase, a custo zero de treino.
- (b) **SPiFiL-3 (contexto/preditivo):**
  - a atenção fixa pode gerar o **alvo de contexto** (Y = Op(X) do prefixo congelado) para o I-JEPA local;
  - é um braço opcional, S3-attn, com S3-rand como controle.
- (c) **Alinhamento compartilhado:**
  - o S1 usa ZCA/PCA (A2 §5.1) e o skip project usa PCA (l.384);
  - o `Alignment` (PCA+Procrustes / CCA, fp64, sinal fixado) pode ser a mesma utilidade, em vez de cada um ter a sua.

**O que muda:** o contrato `KernelInitializer.fit(...)` não muda. A interface só cobre pesos e bias de conv, e a atenção não tem pesos.

**Riscos:**
- **SPiFiL-2 com softmax(r/τ)+Sinkhorn** (τ=0,1). Se a atenção estiver no prefixo durante o S2, as respostas r mudam. Ordem obrigatória: init do bloco → atenção (fixa) → AE/SSL. A atenção nunca entra antes do fit do bloco que ela alimenta.
- **AE local `decoder_first` e protect.**
  - Com `protect=freeze`, a atenção dentro do encoder não altera nada.
  - Com `lowlr` ou `full`, o gradiente que chega aos filtros SPiFiL passa pela softmax e muda a dinâmica.
  - É a 2ª ablação do adendo e só roda depois do E0–E8.
- **Leitura init-only ruidosa (A2 §10.2).** As diferenças entre operadores podem ficar abaixo do ruído. Use o mesmo critério do A2 (Δκ ≥ +0,02 em 3/3 seeds).

**Perguntas ao Agente 2:**
1. Posso registrar os operadores como readouts extras na sua fase A0 (cabeça aleatória fixa)? → *Padrão: sim, só E0/E1/E2/E5 (sem alinhamento), marcados `probe_diag`.*
2. O S3 pode ter o braço S3-attn, com alvo = atenção fixa do prefixo? → *Padrão: sim, depois do seu gate e só em RS/Flowers.*
3. Unificamos o PCA/ZCA (S1, skip project) com o `Alignment` fp64 de sinal fixo? → *Padrão: sim. Eu exponho `_whitener`/`_sym_eigh` e você chama.*
4. Qual τ você usa no S2, e a temperatura é por canal ou global? → *Padrão: global, τ=0,1, para alinhar com o E1–E5.*
5. Com `protect=lowlr` (0,01), aceita a 2ª ablação ("operador dentro do AE local") só com α ∈ {0, 0,25}? → *Padrão: sim. Com α=0 o operador é inerte, o que serve de controle exato.*
6. Os diagnósticos por camada (A2 §7) devem incluir a entropia da atenção e a massa por escala do D? → *Padrão: sim. `last_scale_mass` já sai do operador.*

---

## 3. Agente 3: backbones B0–B4 (prep_literatura_e_specs.md)

**Onde entra:**
- (a) **BackboneSpec.head**: novo campo `attention` (os genes do adendo), aplicado antes do 1×1 (`location=final`) ou sobre o mapa projetado (`location=head`). `type`, `taps` e `proj` continuam iguais.
- (b) **GrowthSpec**: nova operação `attention_insert(α=0)`, com veredito `exact`. A preservação já é testada bitwise.
- (c) **Experimento H** (cabeça multiescala × original): o operador D é a versão **sem parâmetros** de `multiscale_gem`/`fpn_lite`.
  - Libera de 150k a 500k parâmetros para o encoder no mesmo Bk (A3 §7).
  - Exige os 2 controles do próprio A3: mesmo encoder e mesmo `params_total`.

**O que muda:**
- `cost.flops` precisa somar `attention_cost()["flops"]`, porque o hook atual só conta Conv/Linear (A3 §2.5).
- `latency_ms` muda **[medido no smoke]**.
- `params_total` não muda: 0 treináveis.
- Os buffers do alinhamento (C/D) entram no checkpoint como `fitted_values` (≈ C·k por fonte), registrados à parte e **fora** do orçamento de parâmetros treináveis.

**Riscos:**
- **Cabeça recriada aleatória em todo filho** (`run_search.py:700-702`, A3 §2.4). A preservação no nível do embedding só vale se a cabeça for herdada (`head_widen`/`inherit_zero_cols`). Sem isso, o α=0 preserva o mapa, mas não o 1280.
- **WiSE e protect.**
  - O WiSE interpola **pesos** e a atenção não tem pesos, então o WiSE não a vê.
  - O α da atenção é um gene **fixo**, nunca escolhido pelo WiSE. Escolher α na validação junto com o WiSE dobraria o viés otimista (A4 §4.3).
  - O `protect=freeze` só preserva no passo 0 (A3 §2.1). Depois, a atenção muda os gradientes.
- **Memória O(N·M).**
  - As famílias B0→B4 do A3 terminam em 14×14, e 196 tokens já é Q=K=V=X exato e barato.
  - O risco real é a **covariância (E) com `location=head`**: C=1280 e Gram 1280² por imagem. O reparo devolve para `final`.
- **FLOPs da covariância.**
  - O(C²·N). Com 642 canais (mediana do bloco 6, A1a) e 14², dá ≈ 0,32 GFLOP (2·MAC).
  - Com 1024 canais e 28², dá ≈ 3,3 GFLOP, o que pode dominar o f1.
  - Medido na lista congelada v2 (`custo_E1_E5_2mac.csv`): o E5 passa de 25% do encoder em 10 de 74 checkpoints e passa de 100% em 2 (flowers hybrid c00002: 1,98×; ham lemonade c00002: 1,08×).

**Perguntas ao Agente 3:**
1. O campo fica em `head.attention`, ou num `stage[i].attention` (location = bloco)? → *Padrão: `head.attention` com `location ∈ {final, head, i}`. O `stage` fica só para a 2ª ablação.*
2. `attention_insert(α=0)` entra na GrowthSpec com veredito `exact` e preservação medida pelo seu `preservation`? → *Padrão: sim.*
3. A cabeça vai ser herdada (`inherit_zero_cols`) nos filhos B_k → B_{k+1}? Sem isso, a preservação vale só no mapa. → *Padrão: precisa ser herdada; senão eu registro `verdict=approx`.*
4. Os `fitted_values` (buffers PCA/CCA) ficam fora do orçamento Bk? → *Padrão: sim, registrados em `extras_not_counted.fitted_stats`, porque não são treináveis.*
5. No experimento H, você inclui "D sem parâmetros + mesmo params_total" como 3º braço? → *Padrão: sim.*
6. Teto de FLOPs da atenção: proponho FLOPs(operador) ≤ 25% de 2·MAC(encoder + cabeça), na mesma unidade (`fixed_attention.cost_ratio`), com reparo para tokens menores, `final` ou `none`. → *Padrão: 25%.*

---

## 4. Agente 4: MO-DEHB hierárquico (prep_desenho_modehb_hierarquico.md)

**Onde entra:** os genes do adendo viram um sub-segmento **opcional** de `ArchitectureSpec.head`, chamado `attention`:
- `type` ∈ {none, fixed_self, fixed_pool, fixed_cross, fixed_multiscale, fixed_covariance};
- `location`, `source_layers`, `similarity`, `temperature`, `tokens`, `alpha`, `exclude_self`, `normalization`, `aggregation` e `alignment`;
- a codificação na DE é por índice categórico;
- é um segmento próprio da DE, como as outras specs (A4 §3.6).

**O que muda:**
- **Máscara condicional:** `type=none` mascara os outros 10 genes do hash e da DE, como em `ae=off` (A4 §3.3). Assim os genomas antigos mantêm o `cand_key`.
- **Reparo:** `repair_attention_genes(genes, block_channels, block_hw, head_width, batch, mem_budget)`, chamado no passo 5 (build/params/FLOPs). As notas vão para `result["attention_repair"]`.
- **Fitness v2 ORIG:**
  - f0 = 1 − κ_val ORIG, sem mudança: o embedding é a mesma cabeça ≥1280;
  - **f1 precisa somar `attention_cost()["flops"]`**;
  - `fitness_sig` **não** muda, porque os genes são do genoma e não do protocolo;
  - o TTA continua fixo em 1, como no adendo.
- **Recombinação:** os genes de atenção podem cruzar entre pais do mesmo Bk quando `location`/`source_layers` existem nos dois. Senão, o reparo resolve.

**Riscos:**
- **Explosão combinatória:** 6 × 3 × 2 × 6 × 3 × 5 × 2 × 3 × 3 × 4 ≈ 1,9·10⁵ combinações. O adendo pede "nada de busca completa antes das ablações". Proposta:
  - os genes só ficam ativos **depois** do E0–E8;
  - a grade é cortada para os operadores aprovados.
- **Viés de seleção:** τ e α escolhidos pela mesma validação do WiSE. A mitigação é o `val_sel`/`val_fit` que o A4 já prevê (§4.3).
- **Custo:** o cross/multiscale com `pca`/`svd` exige `fit()` em imagens de treino. Com 160 imagens × 256 posições, leva segundos **[medido no smoke]**.

**Perguntas ao Agente 4:**
1. Segmento `arch.head.attention` (proposto) ou uma 6ª spec, `AttentionSpec`? → *Padrão: segmento dentro de `arch.head`, para não mudar o número de specs da DE.*
2. Com `type=none`, os demais genes ficam mascarados do `cand_key`? → *Padrão: sim, para os genomas atuais continuarem válidos.*
3. O f1 soma o FLOP da atenção (`attention_cost`) ao MAC do `profile()`, na mesma unidade (2·MAC)? → *Padrão: sim. Converter MAC → FLOP de forma consistente, porque hoje o campo `flops` grava MACs (A3 §2.5).*
4. Quais operadores ficam ativos no GA? → *Padrão: só os aprovados no E0–E8, com Δκ_val ≥ +0,01 (IC bootstrap > 0) em ≥ 2 datasets prioritários. Com o mesmo critério, o α fica em {0, 0,25, 0,5} e o τ em {0,05, 0,1, 0,25}.*
5. Mutação "liga a atenção com α=0" como mutação neutra (morfismo exato) para o LEMONADE? → *Padrão: sim. O α só sobe por mutação posterior.*
6. Os resultados com atenção vão para um braço separado (A–E + "F: E + atenção"), ou dentro dos braços D/E? → *Padrão: braço F separado, para não contaminar a comparação principal.*

---

## 5. Agente 5: infraestrutura e fila da athena GPU 3 (agent5/*.md)

**Onde entra:** os E0–E8 são **só avaliação** (forward + fit de SVM), sem treino. Propostas para a fila:
- **fase A** (testes de implementação): `A-attn-01`, com a suíte de testes no athena (torch 2.2.2) e o smoke em 1 checkpoint na GPU 3;
- **fase B** (ablações): `B-attn-E0E8-<dataset>`, um job por dataset (lista exata em `agent6/fila_gpu3_attention.json`).

**O que muda:**
- **Antes de tudo:** rsync **somente leitura** dos 74 `encoder.pt` + `config_resolved.yaml` da Jaci para `athena:/dados/home/moliveira/artifacts/attn6_ckpts/` (staging), com md5 conferido. Os originais nunca são tocados.
- O código roda de uma **cópia congelada** (`attn6`) no athena, e não da árvore viva, até o patch ser aprovado.

**Riscos:**
- **Memória:** pico estimado < 6 GB por job, com bs 32 (mapas de até 112² × 1024 canais no larvae_raw c00024). Na GPU 3 (49 GB livres) cabem 4 jobs em paralelo.
- **SVM em CPU:** precisa de `OPENBLAS_NUM_THREADS=2` (§4g). O fit no RS leva ~0,5 s (`fitness_svm_fit_s`); no HAM, com ~7k imagens de treino × 1280, alguns minutos.
- **torch 2.2.2 no athena:** o módulo usa `torch.linalg.pinv(rtol=)`, que existe desde a 1.11, e `antialias=` em `F.interpolate`, que existe desde a 1.11. A fase A confirma.
- **Datasets no athena:** é preciso conferir se `datasets/organized/{remote_sensing,ham10000,aid,...}` existem lá com o mesmo `split_sha256`.

**Perguntas ao Agente 5:**
1. Os E0–E8 entram na fase B (ablações) ou numa fase "B0-avaliação", liberada só com o gate A? → *Padrão: fase B, com `gate_requerido: A`.*
2. O staging dos checkpoints vai para `athena:/dados/home/moliveira/artifacts/attn6_ckpts/<kind>/<dataset>/<cand>/<fid>/` (fora de qualquer `search_state`)? → *Padrão: sim.*
3. Quantos jobs em paralelo na GPU 3 para este bloco? → *Padrão: 4 (≈ 24 GB), deixando espaço para os jobs do A2.*
4. Os resultados vão para o W&B (`external_eval`, `never_feeds_fitness`) e para a tabela? → *Padrão: W&B sim; a tabela de apresentação só depois do gate, com `protocol=attn6_E*`.*
5. Os datasets do athena têm o mesmo `split_sha256` da Jaci? → *Padrão: o job A-attn-01 confere; se divergir, o dataset fica fora.*

---

## 6. Resumo de viabilidade

| Plano | Integração | Viável agora? | Bloqueio |
|---|---|---|---|
| A1 | leitura diagnóstica E0–E8, que preenche ORIG na validação | **sim** | nenhum |
| A2 | readouts extras na A0; braço S3-attn; alinhamento comum | sim (A0); S3 depois do gate | gate A1→A2 |
| A3 | `head.attention`, `attention_insert(α=0)` exato, braço sem parâmetros no H | sim, depois do E0–E8 | herança da cabeça; FLOPs no custo |
| A4 | segmento opcional `arch.head.attention`, mascarado em `none`, f1 com FLOPs | depois do E0–E8 e do gate A3 | corte da grade |
| A5 | fase A (testes + smoke) → fase B (E0–E8) na GPU 3 | **sim** | rsync dos checkpoints + `split_sha256` no athena |

**Recomendação.** Rodar **E0–E8 já** (fase A → B), porque é só avaliação. Os genes do GA e a 2ª ablação ficam para depois. Só os operadores com ganho reprodutível na **validação** seguem adiante.
