# Agente 1 — Diagnóstico consolidado (SPiFiL B0–B4)

_Síntese do Agente 1, 17/11/2026. Fontes: as quatro partes em `parts/` (A1a arquitetura/init, A1b aprendizagem/embedding, A1c generalização, A1d literatura) e seus CSVs, `../check/gate_B_attn_audit.md`, `../../ga_search/modehb/FILA_E_INSTRUCOES.md` (R1–R11) e `../MISSION.md`. Só leitura e CPU. Nenhum treino, nenhum processo, nenhum arquivo de produção alterado. A única medição nova é o md5 de checkpoints sem md5 registrado (leitura de arquivo, para o `seed_registry.csv`)._

_**v1.1 (17/11):** correções C1–C3, C5 e C12 do `check/gate_A1_audit.md` (rótulos, número casado da 2ª ordem, fração de imagens, registro refeito e riscos em §11). A v0 está em `agent1/synth_scripts/archive/diagnosis_v0.md`._

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
| Em 19 grupos (busca × dataset), ρ(val COV, teste COV) tem mediana **0,80** e ρ(val COV, teste ORIG) tem mediana **0,09** (IQR −0,41 a +0,57), negativa em 9 de 19. Reproduzido pelo CHECK (0,801 / 0,091). Só com C=100 (15 grupos): 0,829 / 0,091. Deduplicando por candidato (pseudo-replicação entre fidelidades), ρ ORIG cai para −0,05. Grupos com n = 3–4 dão ρ = ±1 isoladamente: vale a mediana, não o grupo. | CORRELAÇÃO (auditoria retrospectiva do teste, não seleção) | A1b:28-31, 310-318; `check/gate_A1_audit.md:57-71` |
| Pelo agrupamento do A1a: mediana **0,07**, com RS 0,07, aid −0,14, protozoan −0,35 e larvae_raw −0,42. | CORRELAÇÃO | A1a:85 |
| Arrependimento ORIG do selecionado por validação: mediana **0,047**, máximo **0,307** (larvae_raw: sel-val c00004 f45 = 0,188 contra c00001 f200 = 0,495). O sel-val coincide com o melhor ORIG em 4 de 19 grupos. | MEDIDO | A1b:319-323 |
| A seleção é coerente dentro da própria régua (ρ 0,80) e quase aleatória para ORIG. | MEDIDO (código: a fitness lê COV) + CORRELAÇÃO (ρ ≈ 0,09 é observacional) | A1b:324-327 |
| Caso novo: aid c00025 tem COV-val 0,819 e ORIG_sub4000-val **0,636** no mesmo checkpoint (embed_dim 2560). | MEDIDO, n=1 | `agent6/results/E0_val_jaci_cpu.jsonl:1`; `agent6/checkpoints_E0_E8_congelado_20261117_0159.csv:2` |
| Os 74 checkpoints do B-attn foram escolhidos por COV e lidos por ORIG, então não são os melhores em ORIG. | MEDIDO | CHECK:130 |

**Consequência.** O ranking por validação que existe hoje não serve para escolher sementes na régua oficial. Antes de congelar as sementes ORIG é preciso re-pontuar os checkpoints em ORIG-val (X00 da matriz).

## 2. O embedding 1280 é degenerado em geometria (amostra de 17 checkpoints); o ganho está na 2ª ordem e nos blocos intermediários

| afirmação | classe | evidência |
|---|---|---|
| Rank efetivo (participation ratio) do embedding 1280: mediana **3,4**. A 1ª componente carrega **51%** da variância. O GAP do último bloco tem rank efetivo 1,8; o COV, 13,0. **Amostra:** 17 checkpoints, 13 de calibração (c00001–c00004, f200), 6 de RS, 1 de busca C=0,1 e nenhum de eggs/flowers/aid/coconut. Vale para essa amostra, não para "o embedding 1280" em geral. | MEDIDO (17 checkpoints) | A1b:219-223; `check/gate_A1_audit.md:168` |
| As métricas label-free do próprio `search_state` concordam: rank efetivo de 2,7 a 4,2 de 1280 (gap 2,7, gem 3,5, moments 4,2), número de condição de 10⁸ a 10¹¹, ρ(cond, κ ORIG) = −0,26. | MEDIDO / CORRELAÇÃO | A1b:175-178 |
| Com `gap`, o head é linear sobre a média do último bloco (posto = C_último). Com `moments`, o 1×1 projeta para **320 canais** antes das estatísticas, o que é um gargalo aleatório na âncora. | MEDIDO, leitura do código | A1b:224-226 |
| Caso-limite: larvae c00001 tem ORIG-val **0,000** (prevê só a classe majoritária), rank efetivo 1,15 e 93% da variância num eixo. Com scaler sobe para 0,641; com PCA-whiten, 0,785. | MEDIDO | A1b:228-231 |
| Decomposição do gap COV − ORIG no mesmo encoder (mediana em 17 checkpoints): gap total +0,165; **o head 1280 não acrescenta nada** ao COV (+0,000) nem à média do último bloco (+0,021); scaler +0,014; multiescala de 2ª ordem +0,049; multiescala de 1ª ordem +0,017. **Efeito da 2ª ordem em comparação casada:** 2ª − 1ª ordem só no último bloco = **+0,072** [−0,08; +0,56]; multiescala 2ª − multiescala 1ª = +0,133. O +0,107 do A1b ("COV sem head", com todos os blocos, contra o melhor GAP escolhido na própria validação) não é casado e não deve ser citado como efeito da 2ª ordem. Confundidor declarado: a dimensão difere (COV com PCA-64 por bloco ≫ C canais), então parte do efeito pode ser capacidade. | CAUSAL na leitura (mesmo encoder), com confundidor de dimensão | A1b:204-217; `check/gate_A1_audit.md:102` (recalculado do `A1b_embed_diag.csv`) |
| O melhor bloco não é o último em 12 de 17 checkpoints. Nos modelos com ≥ 4 blocos, o último é pior que um intermediário em **10 de 11**. | MEDIDO | A1b:233-239 |
| Por década de params, com efeito fixo de dataset: ORIG +0,013 ± 0,010 (p = 0,19, nulo) e COV +0,072 ± 0,005 (p < 10⁻³⁰). ρ(ORIG, profundidade) = −0,27. | CORRELAÇÃO | A1b:169-174 |
| No teste, COV − ORIG tem mediana +0,123, cresce de tiny (+0,07) para huge (+0,15) e chega a +0,42 em protozoan_raw. | MEDIDO (auditoria) | A1b:165-168 |
| O teto do ORIG não sobe com params porque a média espacial do último bloco satura em poucos fatores. | HIPÓTESE | A1b:263 |
| A literatura dá um mecanismo plausível: covariância com normalização por raiz de matriz (MPN-COV, iSQRT-COV), redução compacta (Compact Bilinear) e risco de covariância mal condicionada com poucas posições espaciais. | literatura (não é resultado nosso) | A1d:217-228 |

**Consequência para R11 (HIPÓTESE).** Pode ser que a cabeça atual cumpra "≥ 1280" só no nome. O rank efetivo de ~3 sugere isso, mas o participation ratio depende de escala e não mede separabilidade linear. Os próprios dados pesam contra a anisotropia como causa: na mediana, ORIG+PCA-whiten − ORIG = **+0,009** e ORIG+scaler − ORIG = +0,034 (A1b:211-212). Isotropizar quase não muda o κ fora dos casos degenerados. O gargalo provável é a informação que a cabeça descarta (2ª ordem, blocos intermediários), não a forma do espaço. R11 exige vetor de alta dimensão "que separa bem" (FILA:19; MISSION:7-9). O contrato **reporta** o rank efetivo ao lado de `embed_dim` só como diagnóstico, **nunca como gate**; o critério é sempre o κ ORIG-val (C6). A cabeça precisa incorporar 2ª ordem e multiescala de dimensão fixa e saída normalizada (signed-sqrt + L2 ou whitening ajustado no treino), medida em ORIG (A1b:264-272; A1d:225), e não o CovReadout externo como substituto silencioso (MISSION:56).

## 3. O SSL muda pouco e pode estragar o ORIG

| afirmação | classe | evidência |
|---|---|---|
| κ(α=1) − κ(α=0) em 656 jobs (COV-val): mediana **+0,003**, IQR [−0,014; +0,021], positivo em 55%. Por dataset vai de ham +0,012 a larvae_raw −0,009. | CAUSAL (mesmos pesos, blend) | A1b:92-95 |
| Na mesma linhagem, de f5 a f200, o ganho do SSL vai de 0,000 a +0,015 enquanto a perda SSL cai de 0,68 para 0,30: perda e κ estão desacoplados. | CAUSAL (linhagem) | A1b:96-100 |
| Com `freeze`/`lowlr` (72% dos jobs) a SSL praticamente não move as convs: ‖ΔW‖/‖W‖ mediano de 0 a 0,03%; em `full`, 3,6%. | MEDIDO (38 checkpoints) | A1a:278-288 |
| De α=0 ao α salvo, o COV nunca piora (11 de 11), mas o **ORIG piora em 5 de 11**. Exemplo: larvae_raw c00003 vai de 0,569 para **0,322** enquanto o COV vai de 0,780 para 0,795. Parte das pioras está dentro do ruído (RS c00004: −0,036 < EP ≈ 0,04 com n_val = 159). O ρ = −0,26 com n=11 não sustenta conclusão. | CAUSAL (mesmo checkpoint), n=11, sem faixa de ruído | A1b:241-250; `check/gate_A1_audit.md:107` |
| Sementes SPiFiL-pequeno com LeJEPA longo: o SSL ajuda aid (+0,07 a +0,10), eggs (0 a +0,07) e ham (+0,01 a +0,05); **degrada** larvae (−0,12 a −0,34 no fim); em protozoan **nunca supera o L0**. Mediana do ganho na última época: −0,020. | MEDIDO | A1b:101-107; FILA:245-247 |
| Selecionar o checkpoint pela perda SSL é errado em parasitas: em larvae ρ(perda, κ) > 0. | MEDIDO | A1b:300-304; FILA:180-184 |
| O SSL global prejudica objetos centrados em fundo preto, por (a) invariância trivial ou (b) o SIGReg isotropizando dimensões de fundo. | HIPÓTESE | A1b:108-112 |
| AE local: joint × decoder_first sem efeito no κ (−0,001 ± 0,011); alvo pixel/masked > feature (+0,03 a +0,04); AE só roda em f5 (2 épocas por camada); Fase 0 do PNAS dá AE − noAE = +0,010 (n=4, dentro do ruído, com confundidor de épocas). | CORRELAÇÃO (OLS) e CAUSAL com n mínimo | A1b:64-85 |

## 4. A escolha do α do WiSE é em boa parte ruído

| afirmação | classe | evidência |
|---|---|---|
| α=0 (descarta a SSL) em **23%** dos jobs; α=1 em 27%. Por dataset vai de flowers 0% a protozoan_raw 37%. | MEDIDO | A1b:116-121; A1a:286 |
| Não há tendência do α=0 com estrato, params, profundidade ou fidelidade (OLS, todos p > 0,16). | CORRELAÇÃO (ausência de associação observacional) | A1b:296 |
| κ(α=0) é idêntico em todas as fidelidades da mesma linhagem (64 linhagens), então a régua é determinística. | MEDIDO | A1b:122 |
| O DP de κ(0,5) − κ(0,25) vai de 0,008 a 0,078 conforme o dataset. O otimismo de escolher o máximo de 5 é +0,016, do mesmo tamanho do "ganho do WiSE" (+0,011 a +0,022). | MEDIDO | A1b:123-126 |
| O α escolhido é seleção de ruído do conjunto de validação, sobretudo em RS (n_val = 159) e larvae (n_val = 240). | HIPÓTESE forte | A1b:129 |
| Além disso, o α é escolhido por uma régua (COV) cega ao head (ver §3). | CAUSAL (por construção) | A1b:250 |

## 5. A escada de fidelidade não reordena os candidatos

| afirmação | classe | evidência |
|---|---|---|
| ρ Spearman(κ f5, κ f_final) = **0,985** em 64 linhagens (0,992 só com C=100; 0,993 no `search_state` vivo, n=72). Ganho mediano f5→f200 de +0,006; 55% das linhagens com \|ganho\| ≤ 0,01. Os grandes não convergem mais devagar (huge +0,007). **Ressalva:** o ρ agregado é inflado pela variância entre datasets e estratos (cada grupo tem ~4 linhagens de calibração com κ muito diferentes; centrando por grupo, 0,954). Vale para candidatos distantes. **Não foi testado entre quase-empatados perto do corte de promoção**, que é onde a reordenação importaria (X11). | CAUSAL (mesma linhagem), régua COV, mistura 11 linhagens C=0,1 | A1b:131-139; `check/gate_A1_audit.md:62, 72-73` |
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
| Como a fidelidade não reordena (§5), o orçamento das promoções compra ≈ 0. | MEDIDO (ganho f5→f15 ≈ 0) | A1b:155; A1a:167 |
| A vantagem dos tiny no **teste ORIG** viria de o ORIG não escalar com params (§2), e não das promoções. | HIPÓTESE (sem intervenção) | A1b:156; A1a:85 |
| O MO-DEHB original não trata a correlação entre fidelidades. Promoção por SH favorece quem aprende rápido. | literatura | A1d:190, 196 |
| 8 de 9 buscas MO-DEHB atuais não passaram de sementes, calibração e aleatórios: nenhum job `de`, nenhuma promoção. Pares pai→filho só existem em RS. | MEDIDO | A1a:46 |

## 7. Vazamentos e comparações incompatíveis

| afirmação | classe | evidência |
|---|---|---|
| **Recapturas do mesmo objeto** entre splits nos parasitas pre (confirmadas visualmente no raw). eggs: 7 test↔train, 5 val↔train (1 byte a byte), 1 val↔test. larvae: 2 test↔train, 4 val↔train. São limites inferiores, com recall desconhecido. **Fração das imagens de teste afetadas** (não impacto no κ): ≥ 0,37% em eggs (7/1903) e ≥ 0,83% em larvae (2/240). O impacto no κ não foi medido (X17). | MEDIDO | A1c:7-11, 90-91; `parts/A1c_dup_pairs.csv` (`veredito_visual=mesmo`) |
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

1. **Trocar a régua de seleção para ORIG-val** em fitness, WiSE, promoção e escolha de checkpoint (R2, R3; MISSION:66). Antes disso, re-pontuar os checkpoints existentes em ORIG-val (X00), depois de cumprir C7 (CPU × GPU) e C8 (sub4000 × full). Sem isso a régua ORIG do `seed_registry.csv` fica com lacunas: 33 de 55 células dataset × faixa sem nenhuma sub-régua medida, e as medidas que existem têm 1–3 checkpoints por faixa.
2. **A cabeça ≥ 1280 é o gargalo mais provável** (§2; a parte "anisotropia" é HIPÓTESE e os dados pesam contra ela). O primeiro experimento barato é sem treino: os mesmos encoders com cabeças diferentes (X02/X03).
3. **Init:** falta o controle aleatório na pipeline atual (X04). As camadas profundas precisam de candidatos na grade da feature, com N distinto ≥ 5·D (X05). Protótipos (SPiFiL-1) atacam a redundância de ~40% (X06).
4. **SSL/AE:** medir em ORIG. Testar a máscara de objeto nas vistas dos parasitas. Escolher o checkpoint SSL por sonda de validação, nunca pela perda (X07/X08).
5. **WiSE e escada de fidelidade:** fixar o α ou escolhê-lo com validação dividida (X09). Reduzir a escada para f5→f45 se a ordem também se mantiver em ORIG (X11). O orçamento liberado vai para seeds repetidas (X01).
6. **Promoção:** limiar de κ global por estrato (X13).
7. **Vazamento:** usar a validação do `split1_modehb` (limpa). O número oficial do teste já está pré-registrado: coluna limpa onde há vazamento conhecido, com a completa ao lado (`evaluation_contract.md` §7.1). Comparar com FM só no mesmo domínio e no mesmo train (`evaluation_contract.md` §7–8).

## 10. Rastreabilidade do `seed_registry.csv` (v1.1, refeito pelas condições C1–C3 do CHECK)

- **Gerador único e versionado:** `agent1/synth_scripts/seed_registry.py` (v1.1). Ele roda em dois passos:
  - `--make-snapshot` copia as entradas para `synth_scripts/snapshot_20261117_A1v1/` (.gz + `manifest.json` com md5, bytes e mtime de cada original): 57 arquivos, sendo os `search_state.json` e `config_resolved.yaml` de `modehb_hybrid`, `modehb_lemonade` e das buscas legadas, o `search_state` do PNAS-CS do bundle athena, `pnas_cs.yaml`, o `E0_val_jaci_cpu.jsonl` do Agente 6 (congelado com as 5 linhas E0 de agora), o CSV de checkpoints do Agente 6 e o `A1b_embed_diag.csv`;
  - o modo normal gera o registro **só a partir do snapshot**, depois de conferir o md5 de cada entrada.
- **Log:** `synth_scripts/seed_registry_20261117_A1v1.log.json` (md5 do script, do manifesto e da saída). Duas execuções deram o mesmo md5 de saída (`9f4106c6…`).
- **Versões anteriores** arquivadas em `synth_scripts/archive/` (registro v0, rascunho, script v0, documentos v0).
- **População completa (C1):** todos os jobs dos `search_state` atuais (1589 jobs lidos), não só os que têm teste. Exemplos do efeito: eggs_raw B0 passou de c00001 (0,527) para c00025 (0,886); protozoan B0, de 0,7725 para 0,867.
  - ham B2 = c00043 (0,612), larvae_raw B1 = c00026 (0,915), flowers B3 preenchido (lemonade c00004) e larvae_raw B4 preenchido (c00024).
  - coconut c00012 está fora (status `failed` no `search_state`).
  - A assinatura de fitness vem do `config_resolved.yaml` de cada busca (PNAS-CS: `pnas_cs.yaml:145-146`). Buscas C=0,1 ficam fora.
- **Desempate (C3):** é o do contrato §2.3. Entre os candidatos com κ > κmax − MDE (empate prático), fica o de menos FLOPs, depois menos params, depois menor fidelidade numérica, depois menor `cand_id`.
  - MDE provisório: 0,02, ou 0,03 com n_val ≤ 240 (RS, larvae, larvae_raw), até o X01 medir.
  - As colunas `argmax_estrito`, `n_cand_na_faixa` e `n_empate_pratico` mostram o que a regra mudou. Exemplo: RS B3 escolhe lemonade c00019 (0,582) entre 12 empatados, em vez do argmax c00028 (0,611).
  - ham B2 e larvae_raw B1 continuam c00043 e c00026, porque não têm empate.
- **ORIG (C2):** uma linha por sub-régua, nunca misturadas no mesmo argmax:
  - `ORIG_full_cpu32`: A1b em RS/larvae/larvae_raw e E0 com train inteiro;
  - `ORIG_sub4000_cpu32`: E0 com train > 4000;
  - `ORIG_sub2000_cpu32`: A1b em protozoan/ham, só para comparação interna;
  - `ORIG_legado_gpuamp`: buscas `modehb_spifil`/`strat`/`v3`. É outra receita (sem AE/protect atuais), portanto **outra população**.
  - κ ORIG ≤ 0,05 é **inutilizável**. Exemplo: larvae c00001 (κ 0,000) foi excluído e listado em `obs`. A faixa larvae B0 ficou com c00021 (0,8095, E0).
  - Faixas com < 3 medidos levam o aviso "re-pontuar a faixa inteira no X00 antes de congelar".
- **Caminhos (C2):** quando o diretório da busca foi renomeado depois do job, o caminho é reescrito para o diretório real e a troca fica registrada em `obs`. Exemplo: o `search_state` de `modehb_strat_prefix_2130` guarda `artifacts/modehb_strat/...`. Todos os checkpoints escolhidos existem na Jaci.
- **md5 e FLOPs:**
  - md5 vem do CSV do Agente 6 ou é calculado na geração.
  - `flops_MAC` é o `flops` da busca (MAC de Conv2d/Linear apenas, C9); `flops_2MAC` = 2×.
  - As linhas legadas vêm de outra pipeline. Exemplo: RS B4 c00099 com 90,5 GMAC. Não pôr no mesmo Pareto.
- **Teste:** nenhum número de teste é lido pelo gerador (a coluna `teste_historico_existe` da v0 foi retirada).
- **Sementes externas** (SPiFiL-pequeno + LeJEPA) continuam fora: a régua delas é a sonda GeM (FILA:183). A de RS tem COV-val 0,564 no smoke (FILA:126). Entram no X00.

## 11. Riscos declarados (C12)

1. **Snapshot vivo.** As buscas continuam rodando. O registro vale para o snapshot `20261117_A1v1` (md5 no manifesto). Qualquer congelamento de sementes gera um snapshot novo com o mesmo script, nunca uma edição à mão.
2. **Otimismo do argmax na mesma validação.** Escolher o máximo entre dezenas de candidatos com n_val = 159 (RS) ou 240 (larvae) e EP ≈ 0,04 dá um viés positivo de ~1–2 EP no κ do vencedor. O desempate por MDE reduz o efeito, mas não o elimina. Mitigação: confirmar os finalistas por faixa em metades A/B fixas da validação (mesma divisão do X09) antes de congelar, e reportar o κ do escolhido medido na outra metade.
3. **O teste das sementes já foi visto.** Por R2/R7, todo job do GA já tem ORIG-teste calculado (`test_svm_C100.jsonl`), e o diagnóstico leu rankings de teste (auditoria A1b Q7, A1c §6). Para esses checkpoints o teste **não** é um holdout virgem. O relatório final tem de declarar isso, e a comparação nosso × FM usa o manifesto congelado (contrato §4).
4. **Amostra do diagnóstico do embedding.** 17 checkpoints, quase todos de calibração, sem eggs/flowers/aid/coconut (§2).
5. **CPU FP32 × GPU-AMP no ORIG (C7).** Primeira evidência: RS c00306 dá 0,5714 em CPU e 0,5719 na busca (GPU-AMP). Com embeddings mal condicionados (libsvm até 3,9·10⁷ iterações), o κ pode mudar mais.
6. **ORIG_sub4000 × ORIG_full (C8)** não foi validado onde o train passa de 4000 (HAM, eggs, aid, coconut, protozoan).
7. **FLOPs cegos a operadores sem parâmetros (C9).** Pooling moments/GeM, covariância, sqrtm, normalizações e atenção fixa contam zero. O GA tenderia a preferir operadores caros e "grátis".
8. **Cobertura do WiSE re-pontuável.** O X09 supõe que todo checkpoint guarda `anchor` e `ssl_state`. O A1b só obteve α=0/α=1 em parte dos 17. Conferir a cobertura antes de orçar.
9. **Grupos pequenos e pseudo-replicação** nos ρ de auditoria (§1): só a mediana é interpretável.
