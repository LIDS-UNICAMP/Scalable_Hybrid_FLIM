# Agente 1 — Diagnóstico consolidado (SPiFiL B0–B4)

_Síntese do Agente 1, 17/11/2026. Fontes: as quatro partes em `parts/` (A1a arquitetura/init, A1b aprendizagem/embedding, A1c generalização, A1d literatura) e seus CSVs, `../check/gate_B_attn_audit.md`, `../../../ga_search/modehb/FILA_E_INSTRUCOES.md` (R1–R11) e `../MISSION.md`. Só leitura e CPU. Nenhum treino, nenhum processo, nenhum arquivo de produção alterado. A única medição nova é o md5 de checkpoints sem md5 registrado (leitura de arquivo, para o `seed_registry.csv`)._

## Convenções

**Classes de evidência** (toda afirmação traz uma):

| classe | significado |
|---|---|
| **MEDIDO** | número lido de um artefato ou recalculado sobre ele |
| **CORRELAÇÃO** | associação observacional (genomas amostrados pela busca, sem randomização) |
| **CAUSAL** | intervenção controlada: o mesmo candidato com uma única variável mudando. Neste material isso só existe em 5 situações: WiSE α (mesmos pesos, blend), linhagem de fidelidade, Fase 0 do PNAS-CS (AE on/off), leituras diferentes do mesmo encoder e efeitos "por construção" do código |
| **HIPÓTESE** | explicação plausível, não testada. Vira teste em `research_hypotheses.md` |

**Evidência:** `arquivo:linha`, relativo a `research/spifil_b0b4/`. `A1a` = `agent1/parts/A1a_arquitetura_init.md`, `A1b` = `agent1/parts/A1b_aprendizagem_embedding.md`, `A1c` = `agent1/parts/A1c_generalizacao.md`, `A1d` = `agent1/parts/A1d_literatura.md`, `CHECK` = `check/gate_B_attn_audit.md`, `FILA` = `../../ga_search/modehb/FILA_E_INSTRUCOES.md`.

**Réguas** (nunca comparar números entre elas):

| nome | definição |
|---|---|
| **ORIG** | `head(encoder(x))` com dimensão ≥ 1280, `core.svm.fit_svm` C=100, sem scaler. **ORIG_full** usa o train inteiro. **ORIG_sub4000** usa um subconjunto estratificado de 4000 |
| **COV** | `cov_head`: 2ª ordem por bloco (PCA k=64) + head, signed-sqrt + L2 e StandardScaler, C=100, max_train 4000, α do WiSE |

**Limite que vale para tudo.** Todo κ da busca é n=1 (split 1, seed 42, pct 100) (A1a:5; A1b:334). O ruído estimado aqui é o do conjunto de validação, não o de seeds de treino. O erro-padrão do κ com n_val=159 (RS) é ≈ 0,04 (A1b:83; A1c:157).

---

## 1. A busca seleciona pela régua COV e reporta pela ORIG

| afirmação | classe | evidência |
|---|---|---|
| A fitness, o WiSE e a calibração usam COV-val (`search_v3.yaml:142-143`, `run_search.py:550-557`). | MEDIDO | A1a:27; A1b:24; A1c:24 |
| O número reportado é ORIG no teste (R2). | MEDIDO | FILA:11 |
| ORIG-val não existe em nenhum job das buscas atuais. Só existe nas buscas legadas `C100_kw0.75_head_none` (RS e flowers, pipeline antiga), nos 17 checkpoints do diagnóstico CPU do A1b e no E0 do Agente 6. | MEDIDO | A1a:21-27, 345; A1b:202; `agent6/results/E0_val_jaci_cpu.jsonl:1` |
| Em 19 grupos (busca × dataset), ρ(val COV, teste COV) tem mediana **0,80** e ρ(val COV, teste ORIG) tem mediana **0,09** (IQR −0,41 a +0,57), negativa em 9 de 19. | CORRELAÇÃO (auditoria retrospectiva do teste, não seleção) | A1b:28-31, 310-318 |
| Pelo agrupamento do A1a: mediana **0,07**, com RS 0,07, aid −0,14, protozoan −0,35 e larvae_raw −0,42. | CORRELAÇÃO | A1a:85 |
| Arrependimento ORIG do selecionado por validação: mediana **0,047**, máximo **0,307** (larvae_raw: sel-val c00004 f45 = 0,188 contra c00001 f200 = 0,495). O sel-val coincide com o melhor ORIG em 4 de 19 grupos. | MEDIDO | A1b:319-323 |
| A seleção é coerente dentro da própria régua (ρ 0,80) e quase aleatória para ORIG. | CAUSAL, por construção do código | A1b:324-327 |
| Caso novo: aid c00025 tem COV-val 0,819 e ORIG_sub4000-val **0,636** no mesmo checkpoint (embed_dim 2560). | MEDIDO, n=1 | `agent6/results/E0_val_jaci_cpu.jsonl:1`; `agent6/checkpoints_E0_E8_congelado_20261117_0159.csv:2` |
| Os 74 checkpoints do B-attn foram escolhidos por COV e lidos por ORIG, então não são os melhores em ORIG. | MEDIDO | CHECK:130 |

**Consequência.** O ranking por validação que existe hoje não serve para escolher sementes na régua oficial. Antes de congelar as sementes ORIG é preciso re-pontuar os checkpoints em ORIG-val (X00 da matriz).

## 2. O embedding 1280 é degenerado; o ganho está na 2ª ordem e nos blocos intermediários

| afirmação | classe | evidência |
|---|---|---|
| Rank efetivo (participation ratio) do embedding 1280: mediana **3,4**. A 1ª componente carrega **51%** da variância. O GAP do último bloco tem rank efetivo 1,8; o COV, 13,0. | MEDIDO (17 checkpoints) | A1b:219-223 |
| As métricas label-free do próprio `search_state` concordam: rank efetivo de 2,7 a 4,2 de 1280 (gap 2,7, gem 3,5, moments 4,2), número de condição de 10⁸ a 10¹¹, ρ(cond, κ ORIG) = −0,26. | MEDIDO / CORRELAÇÃO | A1b:175-178 |
| Com `gap`, o head é linear sobre a média do último bloco (posto = C_último). Com `moments`, o 1×1 projeta para **320 canais** antes das estatísticas, o que é um gargalo aleatório na âncora. | MEDIDO, leitura do código | A1b:224-226 |
| Caso-limite: larvae c00001 tem ORIG-val **0,000** (prevê só a classe majoritária), rank efetivo 1,15 e 93% da variância num eixo. Com scaler sobe para 0,641; com PCA-whiten, 0,785. | MEDIDO | A1b:228-231 |
| Decomposição do gap COV − ORIG no mesmo encoder (mediana em 17 checkpoints): gap total +0,165; **o head 1280 não acrescenta nada** ao COV (+0,000) nem à média do último bloco (+0,021); scaler +0,014; **2ª ordem × melhor 1ª ordem por bloco: +0,107** (até +0,52); multiescala de 2ª ordem +0,049; multiescala de 1ª ordem +0,017. | CAUSAL (mesmo encoder, leitura diferente) | A1b:204-217 |
| O melhor bloco não é o último em 12 de 17 checkpoints. Nos modelos com ≥ 4 blocos, o último é pior que um intermediário em **10 de 11**. | MEDIDO | A1b:233-239 |
| Por década de params, com efeito fixo de dataset: ORIG +0,013 ± 0,010 (p = 0,19, nulo) e COV +0,072 ± 0,005 (p < 10⁻³⁰). ρ(ORIG, profundidade) = −0,27. | CORRELAÇÃO | A1b:169-174 |
| No teste, COV − ORIG tem mediana +0,123, cresce de tiny (+0,07) para huge (+0,15) e chega a +0,42 em protozoan_raw. | MEDIDO (auditoria) | A1b:165-168 |
| O teto do ORIG não sobe com params porque a média espacial do último bloco satura em poucos fatores. | HIPÓTESE | A1b:263 |
| A literatura dá um mecanismo plausível: covariância com normalização por raiz de matriz (MPN-COV, iSQRT-COV), redução compacta (Compact Bilinear) e risco de covariância mal condicionada com poucas posições espaciais. | literatura (não é resultado nosso) | A1d:217-228 |

**Consequência para R11.** A cabeça atual cumpre "≥ 1280" só no nome: a dimensão efetiva é ~3. R11 exige vetor de alta dimensão "que separa bem" (FILA:19; MISSION:7-9). Proposta para o contrato: **reportar o rank efetivo do embedding ao lado de `embed_dim`**. A cabeça precisa incorporar 2ª ordem e multiescala de dimensão fixa e saída normalizada (signed-sqrt + L2 ou whitening ajustado no treino), medida em ORIG (A1b:264-272; A1d:225), e não o CovReadout externo como substituto silencioso (MISSION:56).

## 3. O SSL muda pouco e pode estragar o ORIG

| afirmação | classe | evidência |
|---|---|---|
| κ(α=1) − κ(α=0) em 656 jobs (COV-val): mediana **+0,003**, IQR [−0,014; +0,021], positivo em 55%. Por dataset vai de ham +0,012 a larvae_raw −0,009. | CAUSAL (mesmos pesos, blend) | A1b:92-95 |
| Na mesma linhagem, de f5 a f200, o ganho do SSL vai de 0,000 a +0,015 enquanto a perda SSL cai de 0,68 para 0,30: perda e κ estão desacoplados. | CAUSAL (linhagem) | A1b:96-100 |
| Com `freeze`/`lowlr` (72% dos jobs) a SSL praticamente não move as convs: ‖ΔW‖/‖W‖ mediano de 0 a 0,03%; em `full`, 3,6%. | MEDIDO (38 checkpoints) | A1a:278-288 |
| De α=0 ao α salvo, o COV nunca piora (11 de 11), mas o **ORIG piora em 5 de 11**. Exemplo: larvae_raw c00003 vai de 0,569 para **0,322** enquanto o COV vai de 0,780 para 0,795. ρ entre os dois Δ = −0,26. | CAUSAL (mesmo checkpoint), n=11 | A1b:241-250 |
| Sementes SPiFiL-pequeno com LeJEPA longo: o SSL ajuda aid (+0,07 a +0,10), eggs (0 a +0,07) e ham (+0,01 a +0,05); **degrada** larvae (−0,12 a −0,34 no fim); em protozoan **nunca supera o L0**. Mediana do ganho na última época: −0,020. | MEDIDO | A1b:101-107; FILA:245-247 |
| Selecionar o checkpoint pela perda SSL é errado em parasitas: em larvae ρ(perda, κ) > 0. | MEDIDO | A1b:300-304; FILA:180-184 |
| O SSL global prejudica objetos centrados em fundo preto, por (a) invariância trivial ou (b) o SIGReg isotropizando dimensões de fundo. | HIPÓTESE | A1b:108-112 |
| AE local: joint × decoder_first sem efeito no κ (−0,001 ± 0,011); alvo pixel/masked > feature (+0,03 a +0,04); AE só roda em f5 (2 épocas por camada); Fase 0 do PNAS dá AE − noAE = +0,010 (n=4, dentro do ruído, com confundidor de épocas). | CORRELAÇÃO (OLS) e CAUSAL com n mínimo | A1b:64-85 |

## 4. A escolha do α do WiSE é em boa parte ruído

| afirmação | classe | evidência |
|---|---|---|
| α=0 (descarta a SSL) em **23%** dos jobs; α=1 em 27%. Por dataset vai de flowers 0% a protozoan_raw 37%, sem tendência com estrato, params, profundidade ou fidelidade (OLS, todos p > 0,16). | MEDIDO | A1b:116-121, 296; A1a:286 |
| κ(α=0) é idêntico em todas as fidelidades da mesma linhagem (64 linhagens), então a régua é determinística. | MEDIDO | A1b:122 |
| O DP de κ(0,5) − κ(0,25) vai de 0,008 a 0,078 conforme o dataset. O otimismo de escolher o máximo de 5 é +0,016, do mesmo tamanho do "ganho do WiSE" (+0,011 a +0,022). | MEDIDO | A1b:123-126 |
| O α escolhido é seleção de ruído do conjunto de validação, sobretudo em RS (n_val = 159) e larvae (n_val = 240). | HIPÓTESE forte | A1b:129 |
| Além disso, o α é escolhido por uma régua (COV) cega ao head (ver §3). | CAUSAL (por construção) | A1b:250 |

## 5. A escada de fidelidade não reordena os candidatos

| afirmação | classe | evidência |
|---|---|---|
| ρ Spearman(κ f5, κ f_final) = **0,985** em 64 linhagens. Ganho mediano f5→f200 de +0,006; 55% das linhagens com \|ganho\| ≤ 0,01. Os grandes não convergem mais devagar (huge +0,007). | CAUSAL (mesma linhagem), régua COV | A1b:131-139 |
| A calibração declara `saturation_epochs = 5` em 48 de 59 entradas. | MEDIDO | A1b:133 |
| Promoções (mesmo candidato, mais épocas): f5→f15 mediana 0,000 (n=202); f15→f45 +0,001; f45→f100 0,000; f100→f200 0,000. | MEDIDO | A1a:167 |
| A fidelidade só acrescenta épocas de LeJEPA, nunca de AE (`run_search.py:666`). | MEDIDO (código) | A1b:66 |
| O fitness é dominado pela âncora SPiFiL e pela 2ª ordem dos blocos intermediários, que o SSL quase não move. | HIPÓTESE | A1b:140 |
| **Lacuna:** tudo isso é na régua COV. Não se sabe se a ordem também se mantém em ORIG-val, que é sensível ao head e ao último bloco, justamente o que o SSL altera (§3). | — | X11 da matriz |

## 6. A promoção protegida favorece os tiny

| afirmação | classe | evidência |
|---|---|---|
| Em remote_sensing, f5→f15, 6 a 7 dos 8 tiny promovidos ficam abaixo do corte global top-k. Mediana do κ dos tiny promovidos: 0,414, contra 0,565 dos huge promovidos. Há tiny promovidos nas posições 70–81 de 81, enquanto 3–6 huge acima do corte ficam de fora. | MEDIDO | A1b:142-153 |
| O mecanismo é `min_share = 0,25` mais ranqueamento não dominado κ × FLOPs dentro do estrato (`strata.py:159`). | CAUSAL (por desenho) | A1b:144, 154 |
| Como a fidelidade não reordena (§5), o orçamento das promoções compra ≈ 0. A vantagem dos tiny no **teste ORIG** tem outra causa: o ORIG não escala com params (§2). | MEDIDO + inferência | A1b:155-156; A1a:85 |
| O MO-DEHB original não trata a correlação entre fidelidades. Promoção por SH favorece quem aprende rápido. | literatura | A1d:190, 196 |
| 8 de 9 buscas MO-DEHB atuais não passaram de sementes, calibração e aleatórios: nenhum job `de`, nenhuma promoção. Pares pai→filho só existem em RS. | MEDIDO | A1a:46 |

## 7. Vazamentos e comparações incompatíveis

| afirmação | classe | evidência |
|---|---|---|
| **Recapturas do mesmo objeto** entre splits nos parasitas pre (confirmadas visualmente no raw). eggs: 7 test↔train, 5 val↔train (1 byte a byte), 1 val↔test. larvae: 2 test↔train, 4 val↔train. São limites inferiores, com recall desconhecido. Impacto direto no κ de teste: ≥ 0,37% (eggs) e ≥ 0,83% (larvae). | MEDIDO | A1c:7-11, 90-91; `parts/A1c_dup_pairs.csv` (`veredito_visual=mesmo`) |
| No protozoan pre a questão é indeterminada (hash sem poder discriminante). | MEDIDO | A1c:11, 92 |
| **HAM10000 por lesão.** No `split1_modehb`, train∩val = 0 lesões, então a validação é limpa. Mas **1312 imagens do teste (26,2%)** têm a lesão em train∪val: MEL 41%, DF 42%, BCC 32%, NV 16%. Há ainda 1 duplicata com `lesion_id` diferente (ISIC_0025226 = ISIC_0030074). | MEDIDO | A1c:12-17, 98-113 |
| Os FMs no HAM treinam o SVM em 5007 imagens (val ≡ train, κ val = 1,000 inválido), contra as nossas 4129. | MEDIDO | A1c:18-21, 130 |
| A semente externa antiga do HAM viu o nosso val; foi retreinada só no treino. | MEDIDO | FILA:45 |
| **flowers:** 1 cópia byte a byte train↔test com **rótulo diferente** (rose × tulip) + 1 par T1 + 2 outros entre splits. **aid:** 1 par idêntico train↔test. RS e coconut: nenhum. | MEDIDO | A1c:26-28, 83-88 |
| **pre × raw:** FM-pre > FM-raw em 9 de 9 casos (+0,010 a +0,085). A comparação "nosso pre × FM raw" é enviesada a favor do nosso. | MEDIDO | A1c:23, 121; FILA:249 |
| O inventário `all_results_today.md` mistura 12 protocolos incompatíveis: régua, split, pct, C=0,1, máximo retrospectivo do teste usado como "nosso", resolução e dimensão. | MEDIDO | A1c:117-134 |
| O plano do B-attn não declarava esses riscos (C8). | MEDIDO | CHECK:44, 146-154 |
| Como o Δκ entre braços é pareado (mesmas imagens), o vazamento afeta o κ **absoluto** e a comparação com FMs, não o Δ entre braços. | inferência | CHECK:154 |

## 8. Profundidade e inicialização (A1a)

### 8a. Arquitetura

| afirmação | classe | evidência |
|---|---|---|
| Na fitness atual, κ val sobe com params/largura: ρ agregado 0,41/0,49, de 0,48 a 0,72 por grupo. Na fitness legada (head, sem scaler) ρ(params) = −0,05 e ρ(prof.) = −0,36. **"Maior = melhor" depende da régua.** | CORRELAÇÃO | A1a:76-84 |
| A profundidade com params controlado tem parcial negativa em 14 de 17 grupos. OLS: **−0,20 z por bloco** [−0,29; −0,11]; log10(params) +0,88. | CORRELAÇÃO | A1a:83, 125-133 |
| Ótimo em **3–4 pools** (mapa final de 28 a 14). Com 5 pools (7×7) e profundidade 7–8 o κ piora (z ≈ −0,6 a −0,7). | CORRELAÇÃO | A1a:92-109 |
| Inception: +0,42 z [0,11; 0,70] com params/profundidade/pools controlados; dwsep indistinto; residual sem efeito. | CORRELAÇÃO | A1a:125-137 |
| +1 bloco: Δκ mediano **−0,011**, 29% melhoram (n=38). Os 8 operadores de mutação têm mediana negativa; só 29 de 97 filhos superam o pai (p = 9×10⁻⁵). | MEDIDO (pares; só RS) | A1a:143-166 |
| Parte disso é regressão à média: os pais vêm da frente (percentil 0,71). | HIPÓTESE / viés | A1a:154 |
| **Saturação precoce:** em 6 de 11 datasets o melhor κ val (COV) já está em B0/B1. **B4 está praticamente vazio** (`stratified.param_cap` 3,5M; só o PNAS-CS RS passa disso). | MEDIDO | A1a:176-191 |
| As famílias PNAS-CS não sobem com o porte (P1 0,577 → P2 0,583 → P3 0,548). | MEDIDO | A1a:153 |
| A literatura prevê saturação e não monotonicidade abaixo do EfficientNet-B0. Nos portes pequenos, a profundidade deve ser protegida antes da largura (TinyNet). | literatura | A1d:153-160, 210-211 |

### 8b. Inicialização SPiFiL

| afirmação | classe | evidência |
|---|---|---|
| As sementes são superpixels SLIC na imagem LAB 224, **mesmo nas camadas profundas**, reescaladas para a grade do prefixo. Nos parasitas são restritas à máscara. | MEDIDO (código) | A1a:205 |
| A init é **supervisionada**: Fisher com Mahalanobis + Ledoit-Wolf, alocação e seleção por classe. Existe um `DistanceSumScorer` label-free, mas não é usado. Não há SIFT no caminho usado. | MEDIDO (código) | A1a:207-214 |
| O orçamento N > D conta **sementes duplicadas** (`spifil_ops.py:302-309`; `strata.py:436-476`). | MEDIDO (código) | A1a:211, 243 |
| Sementes distintas por imagem: 100 na grade 224 → 48 na grade 7 (naturais); nos parasitas, → **10–14 na grade 7** e 25–38 na grade 14. | MEDIDO (simulação exata) | A1a:219-228 |
| N efetivo / D < 1 em **57%** das camadas de grade 7 nos parasitas e em 21% das de grade 14. | MEDIDO (estimativa) | A1a:230-243 |
| Redundância: ~40% dos filtros têm um par com cos > 0,95, **inclusive no SPiFiL puro** (âncora freeze). Ramos 1×1: 71% com par cos > 0,9. | MEDIDO (38 checkpoints) | A1a:254-269 |
| Rank efetivo das ativações / C cai de 0,49 (bloco 0) para 0,11–0,16 (blocos 3–6). Não há filtros mortos; canais mortos ≈ 0%, exceto em huge raw. | MEDIDO | A1a:270-272 |
| O colapso é redundância e baixo rank, não morte. | HIPÓTESE (coerente com as medidas) | A1a:272 |
| Como a SSL quase não move as convs (§3), a init de cada camada **define** a representação. Uma init pobre nas camadas profundas não é corrigida depois, e isso explicaria por que blocos extras não ajudam. | HIPÓTESE (mecanismo candidato) | A1a:289 |
| **SPiFiL × aleatório:** o aleatório vence em 6 de 8 gêmeos (Δ mediano −0,023, Wilcoxon p = 0,20), mas em pipeline antiga, f5, n=1 e com o SPiFiL 12% mais estreito. **Não há gêmeo aleatório nas buscas atuais** (`random_baseline: false`), então hoje não há evidência de que a init SPiFiL supere a aleatória. | MEDIDO, n pequeno e confundido | A1a:291-306 |
| ipc (doadores por classe) tem ρ positivo com κ, mas está confundido com a largura (o reparo aumenta o ipc dos largos). | CORRELAÇÃO confundida | A1a:245 |

## 9. O que se segue (resumo para os Agentes 2–4)

1. **Trocar a régua de seleção para ORIG-val** em fitness, WiSE, promoção e escolha de checkpoint (R2, R3; MISSION:66). Antes disso, re-pontuar os checkpoints existentes em ORIG-val (X00). Sem isso o `seed_registry.csv` ORIG fica com lacunas (34 de 55 células).
2. **A cabeça ≥ 1280 é o gargalo principal** (§2). O primeiro experimento barato é sem treino: os mesmos encoders com cabeças diferentes (X02/X03).
3. **Init:** falta o controle aleatório na pipeline atual (X04). As camadas profundas precisam de candidatos na grade da feature, com N distinto ≥ 5·D (X05). Protótipos (SPiFiL-1) atacam a redundância de ~40% (X06).
4. **SSL/AE:** medir em ORIG. Testar a máscara de objeto nas vistas dos parasitas. Escolher o checkpoint SSL por sonda de validação, nunca pela perda (X07/X08).
5. **WiSE e escada de fidelidade:** fixar o α ou escolhê-lo com validação dividida (X09). Reduzir a escada para f5→f45 se a ordem também se mantiver em ORIG (X11). O orçamento liberado vai para seeds repetidas (X01).
6. **Promoção:** limiar de κ global por estrato (X13).
7. **Vazamento:** usar a validação do `split1_modehb` (limpa). Relatar o teste com sensibilidade (HAM por lesão; parasitas sem recapturas; flowers/aid sem duplicatas). Comparar com FM só no mesmo domínio e no mesmo train (`evaluation_contract.md` §7–8).

## 10. Rastreabilidade do `seed_registry.csv`

- **COV:** extraído de `parts/A1a_data.csv`, que é um snapshot dos `search_state` de 16/11 ~22:05 (A1a:11). Filtro: assinatura `C100_kw1_cov_head_standard_mt4000`, status ok (busca) ou `ok(athena_test_only)` marcado. Faixas **na faixa** (B0 ≤ 300k; B1 300k–800k; B2 800k–1,5M; B3 1,5–3M; B4 3–5,5M). Sem candidato na faixa = `lacuna=sim`, com o melhor de faixa menor como fallback.
- **Desempate do COV:** maior κ, depois menor fidelidade, depois menor FLOPs. Nos empates exatos o CHECK registrou desempate lexicográfico da fidelidade (CHECK:127).
- **ORIG:** só onde existe ORIG-val medido, identificado pela sub-régua: `ORIG_full` (A1b, RS/larvae/larvae_raw), `ORIG_subamostra` (A1b, protozoan/ham), `ORIG_sub4000` (E0 do Agente 6) e `ORIG_sub4000_legado` (buscas `modehb_spifil`/`strat`/`v3`, pipeline antiga). As sub-réguas não são comparáveis entre si.
- **md5:** o do Agente 6 (`agent6/checkpoints_E0_E8_congelado_20261117_0159.csv`) quando existe. Senão foi calculado agora na Jaci. Para jobs migrados do athena, conferir com o original.
- **FLOPs:** a coluna `flops` da busca é **MAC** (`run_search.py:322-360`, `inference_macs`, gravado em `:784`; CHECK:41). O registro traz `flops_MAC` e `flops_2MAC`.
- **Teste:** os valores de teste **não** entram no registro; só a flag `teste_historico_existe`.
- **Atualizações conhecidas depois do snapshot** (CHECK:39) estão na coluna `obs`: ham B2 c00043 0,612; larvae_raw B1 c00026 0,915; flowers hybrid B3 novo; coconut B4 c00012 `failed` (CHECK:40).
- **Sementes externas** (SPiFiL-pequeno + LeJEPA) não entram como "melhores": a régua delas é a sonda GeM `lin_select_kappa`, nem ORIG nem COV (FILA:183). A de RS (RegNet 0,98M, faixa B2) tem COV-val 0,564 no smoke (FILA:126), abaixo da B2 do registro (0,591). Re-pontuar em ORIG-val junto com o X00.
