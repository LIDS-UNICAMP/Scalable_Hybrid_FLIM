# Plano final: lei de montagem do SPiFiL híbrido

07/11/2026. Fecha P1 (literatura), P2 (datasets), P3 (desenho) e P4 (arquitetura). Detalhes e contas ficam neles; aqui fica só o que vale.

**Em uma frase:** vamos medir como o erro cai com mais imagens (lei de aprendizado) e qual forma de rede erra menos pelo mesmo custo (lei arquitetural), sempre com rede nascida do SPiFiL e treinada sem rótulo no gradiente. Primeiro achamos a regra em 7 datasets, depois a congelamos e a testamos em 5 que ela não viu.

**Produto:** uma regra de montagem (B0 a B3) com previsão de κ escrita antes, não uma config única. Chegar a até 0,01 do melhor modelo do benchmark já satisfaz, mas isso é meta, não critério da lei.

## 1. Hipóteses

`*` marca as 7 primárias, corrigidas por Holm. O resto é exploratório: relata, não decide.

| # | Pergunta | Passa se |
|---|---|---|
| HA1\* | e(D)=E+B·D^−b descreve cada curva? | Resíduo ≤ 1,5σ em ≥ 5 de 7 datasets; b entre 0,3 e 1,0 |
| HA2\* | b é parecido entre famílias? | Compartilhado empata (ΔAICc < 2) e o erro ao deixar família de fora é ≤ 1,3× o do individual |
| HA3 | A curva prevê o que não viu? | Ajusta até 50%, prevê 75 e 100%: \|Δκ\| mediano ≤ 0,015 |
| HA4 | Tamanho muda E e B, não b? | b de B1 e de B3 dentro do intervalo |
| HA5 | Imagens de SSL e rótulos são eixos diferentes? | b_ssl e b_rótulo com intervalos separados |
| HA6 | A curva mede dados, não passos? | 5% com os passos de 100%: diferença ≤ 0,02 |
| HF1\* | A forma importa a custo igual? | Melhor menos pior ≥ 0,02 em ≥ 3 dos 5 decisores |
| HF2\* | Crescer junto vence crescer 1 eixo? | ≥ +0,01 agrupado, ou mesmo κ com metade do custo |
| HF3\* | A divisão do orçamento muda por dataset? | Trocar a divisão perde ≥ 0,005 agrupado |
| HF4 | Resolução satura no tamanho nativo? | r\* ≤ nativo em 3 de 3 |
| HF5 | Profundidade segue log da resolução? | d\* cresce com r, ou d=3 fica a 0,01 do melhor |
| HX1 | A melhor forma vale a 5% e a 50%? | Mesma ordem, ou diferença ≤ 0,01 |
| HP1\* | Alguma perda com λ ≤ médio vence só reconstrução? | ≥ +0,01 agrupado, intervalo sem 0 |
| HP2 | O λ de B0 serve em B3? | Perde ≤ 0,005 |
| HP3 | Dá para escolher perda sem rótulo? | Dimensão efetiva de z contra κ: Spearman ≥ 0,7 |
| HS1\* | SPiFiL vence aleatório na mesma forma? | ≥ +0,01, intervalo sem 0, mesmo sinal em ≥ 2 de 3 famílias |
| HS2 | Cabeça só SPiFiL não perde? | Perde ≤ 0,01 |

## 2. Variáveis

| Tipo | Variável | Níveis |
|---|---|---|
| Dados | Imagens de treino D | Contagem real em 1, 5, 10, 25, 50, 75, 100% |
| | Rótulos no SVM | 1, 5, 25% com encoder a 100% (`svm_train_pct`, só CPU) |
| Forma | d, convoluções espaciais | 3, 4, 5, 6 |
| | w, multiplicador de largura | 0,5 a 4; ¼, ½ e 6 só no bloco de `a` |
| | r, lado da imagem | r livre: 128 a min(640, nativo). r preso: 200 |
| Perda | R(z) | Só reconstrução, Weak-SIGReg, Strong-SIGReg, VICReg, VICRegL, Barlow, CorInfoMax, UR-JEPA |
| | λ | {0,01; 0,001}; VICRegL e CorInfoMax {0,001; 0,0001}. São 15 braços |
| Início | Cabeça | H0 (SPiFiL + resto sorteado), H1 (c = candidatos), H2 (`seed_point: union`) |
| | Controle | SPiFiL sempre; `random_layer: true` só na amostra |
| Bloco | Dataset, família, split, semente | 3 splits em todo ponto que decide; 3 sementes só em F0 |
| Fixa | | 50 superpixels DISF, sem máscara, `raw/`, kernels 5,5,5,3,1, pool nas 3 primeiras, 60 épocas por estágio, lote 32, lr 5e-4, backprop 0,1, leitura `gem_tta_pw` abaixo de 2000-d, cabeça ≤ 1280 no múltiplo de C mais perto, val e teste fixos |
| Resposta | e=1−κ de validação do SVM; F1 e acurácia ao lado | CSV do eval |
| Custo | FLOPs e parâmetros sobre a largura **real** (`architecture.json`), tempo, memória | Offline e `run_metadata.json` |
| Registrada | Fração SPiFiL da cabeça, dimensão efetiva de z, canais mortos, `reg_ratio`, `val_recon_loss` | |

O teste fica fora de todo ajuste. Abre uma vez, no fim, com a lei congelada.

## 3. Equações

- **Aprendizado:** e(D) = E + B·D^(−b), ajustada em e (nunca em log), D em imagens reais, peso σ(p) = σ₅₀·(50/p)^0,35. Split é efeito aleatório: 18 pontos aninhados valem 3 curvas.
- **Versões, da mais livre à mais presa:** individual, por família, compartilhada, e hierárquica com encolhimento (padrão para reportar).
- **Conjunta:** e(N,D) = E + A·N^(−a) + B·D^(−b). Se a soma falhar, e = E + A·x^(−a) + B·x^c·D^(−b), que dá x\* ∝ D^s.
- **Superfície por dataset:** e(d,w,r) = c₀ + c_d·ln d + c_w·ln w + c_r·ln r + q_d·(ln d)² + q_w·(ln w)² + q_r·(ln r)². Com r preso, 5 coeficientes.
- **Custo:** custo ≈ w^1,85 · r² · d^0,12 (seção 8). Por degrau de orçamento φ = log₂(F/F₀): β^1,85·γ² ≈ 2, com d fora da conta.
- **Regra com uma constante:** π é a parcela do orçamento que vai para a largura, β = 2^(π/1,85), γ = 2^((1−π)/2). Prior π=0,5: β=1,21, γ=1,19. Com r preso, π=1: β=1,45.
- **Escolha entre versões:** AICc (mais constantes só com ΔAICc ≥ 2); deixar 1 família de fora é o juiz e vence se discordar; previsão fora da amostra; intervalo por perfil e bootstrap paramétrico.
- **Aviso:** "compartilhada venceu" pode ser falta de poder. Reportar sempre τ com intervalo.

## 4. Papéis dos datasets (opção A: só os 12)

Família inteira no mesmo papel, contada como 1 unidade. São 5 famílias, não 6: 453 das 1.000 imagens de `remote_sensing` vêm do `aid`.

| Papel | Datasets |
|---|---|
| Descoberta, r livre | Eggs, Larvae, flowers |
| Descoberta, r preso em 200 | helminth_eggs, Eggs_subgroup, helminth_larvae, Larvae_subgroup |
| Confirmação | Protozoan, protozoan_cysts, Protozoan_subgroup |
| Externa | aid, remote_sensing |

- **Decisores** (enxergam diferença): helminth_eggs, Eggs, flowers, Eggs_subgroup, helminth_larvae. Larvae e Larvae_subgroup só dão direção.
- **Representantes**, um por família, para triagem: helminth_eggs, helminth_larvae, flowers.
- **Ressalva:** na opção A a "externa" não é inédita. Nenhum dos 12 tem teste virgem, e isso vai declarado.
- `Protozoan_subgroup`, `aid` e `remote_sensing` da grade `sc` ficam lacrados: não entram na descoberta.

## 5. Fases, runs e custo (orçamento médio)

Uma cadeia é um modelo treinado do começo ao fim, estágio por estágio.

| Fase | Pergunta | O que roda | Cadeias | GPU-h |
|---|---|---|---|---|
| **F0** | Quanto é o ruído? | B0, representantes, 3 sementes × 3 splits, Weak-SIGReg λ=0,01. Mais 6 SVMs nas réplicas da `sc` | 27 | 7 |
| **F1** | Qual perda e λ? | Degrau 1: 15 braços × 3 representantes, 15 épocas. Degrau 2: top 5, cheio, + Eggs. Degrau 3: finalistas + reconstrução, 3 splits, 5 decisores | 95 | 24 |
| **F2** | Qual forma por orçamento? | LHS por dataset (12 pontos r livre, 8 r preso), 50%, 1 split. Depois top 2 por orçamento, 3 splits | 170 | 160 |
| | Cabeça | H1 e H2 em B0 e B3 | 12 | 8 |
| **F3** | Lei de aprendizado | Fronteira × frações, nesta ordem: 100, 5, 25, 1, 10, 75 | 174 | 111 |
| | Expoente `a` | w ¼, ½, 4 em 2 datasets | 54 | 22 |
| | Passos e ponte | HA6 e HX1 | 24 | 8 |
| | Controle aleatório | B0 e B3, representantes, 3 splits | 18 | 12 |
| | Regras | Regra comum e regra EfficientNet | 18 | 12 |
| | 200 épocas | B1, representantes | 3 | 5 |
| **Congela** | | Lei escrita, com previsão de cada dataset | | |
| **F4** | Confirmação | Protozoan, previsão antes de treinar | 27 | 62 |
| **F5** | Externa | Só depois de F4. O teste abre aqui, uma vez | 33 | 38 |
| **Total** | | | **655** | **468** |

- **Tempo:** 4,9 dias em 4 GPUs livres; 7,3 com margem ×1,5. SVM um por vez: 16 h, em paralelo com a fase seguinte.
- **Baselines à parte:** EfficientNet treinado do zero, ~250 h de slot (P4 §13).
- **Se apertar, cortar nesta ordem:** 75%, depois B3 em F3, depois top k de 2 para 1 em Eggs.
- **Regra de forma:** ajusta, pega os melhores por orçamento e **mede de novo**. O valor que vai para a tabela vem do remedido, nunca do ponto escolhido, para não pagar o viés de "melhor de N".

## 6. Critérios escritos antes

Vale o que está aqui. Mudar depois de ver dado exige dizer no relatório.

**Parada por fase:**

| Fase | Se | Então |
|---|---|---|
| F0 | Desvio entre sementes ≤ 0,01 | 1 semente por split no resto |
| | Entre 0,01 e 0,02 | 2 sementes só no top k e em B1 de F3 (+~35%) |
| | > 0,02 | **Pausa.** O sorteio de imagens do SPiFiL vira fator antes de seguir |
| F1 | Braço colapsa (NaN, `val_recon_loss` sobe > 5%, ou dimensão efetiva de z abaixo da de reconstrução) | Sai na hora |
| | Melhor braço − reconstrução < 0,005, intervalo com 0 | Fica a mais simples |
| | ρ(15 épocas, 60 épocas) < 0,5 | Repete o degrau 1 com 30 épocas |
| F2 | Faixa entre formas ≤ 2 desvios | Superfície plana: pula B2 e B3 de F3, "menor modelo suficiente" |
| | Top 1 mudou em > 2 orçamentos ao remedir | 1 rodada extra de top 3, só nos decisores. No máximo 1 |
| F3 | Largura de b compartilhado ≤ 0,2 depois de 100, 5, 25, 1 | Pula 10 e 75, exceto em 2 datasets |
| Geral | GPU-h gasto > 1,3× do orçamento | Para e reporta o que tem |

**Sucesso e falha:**

| Lei | Sucesso | Rebaixa, sem matar | Falha |
|---|---|---|---|
| Aprendizado | HA1 e HA2 passam; erro mediano ≤ 0,015 e máximo ≤ 0,03 em F4 | Vale por família | Erro mediano > 0,03 em F4, ou b fora do intervalo em 2 de 3 famílias |
| Arquitetural | Em F4, regra a ≤ 0,01 do melhor medido em ≥ 80% de dataset × orçamento, **e** HF2 passa | HF1 falha: "menor modelo suficiente". HF3 falha: regra compartilhada | Regra perde da melhor config comum **e** não gasta 2× menos |
| Perda | HP1 passa | Vence em 1 família | Nenhuma vence: reconstrução sozinha |
| SPiFiL | HS1 passa | Só em modelos pequenos | Intervalo com 0 em todas as famílias: cai a parte SPiFiL da lei |
| Externa | Previsão escrita antes acerta ≥ 3 de 4 células, intervalo de 95% | 2 de 4 | ≤ 1 de 4 |

**"A 0,01" é agrupado e pareado por split.** 3 splits não enxergam 0,01 em nenhum dataset sozinho: por dataset só se reporta a diferença. Nos 5 decisores são 15 pares, faixa ±0,011.

## 7. O que muda em relação aos YAMLs de hoje

- `best_by_probe: false` e sem `epoch_probe`: o checkpoint sai por `val_recon_loss`, sem rótulo. Sonda desligada também economiza ~42% da época. Essa perda é quase plana, então na prática o efeito é perto de "última época".
- `c2560` fora: a leitura passaria de 2000-d e a L4 morre. Cabeça sempre ≤ 1280.
- Largura sempre múltiplo de C: o SPiFiL joga fora o resto sem avisar. Fundo preto (helminth_*) leva `redundancy_cos: 0.9999` e `redundancy_replace: true`. Em helminth_larvae, com 2 classes, não havia o que repor (no teste, 32/48/64 viraram 29/34/45), então leva também `images_per_class: 3`, que voltou a 32/48/64.
- Grade `sc` é só piloto: entra como nível λ=0 com indicador de regime, nunca misturada sem ele.
- F1 degrau 1 usa `warmup_epochs: 3`, para manter a proporção de 10 em 60 épocas.
- VICRegL leva `max_grid: 24`: compara cada posição com todas, e no mapa 99×99 da L1 só essa tabela pede 12 GB (morreu no teste real). O termo local passa a ver no máximo a grade final, 24×24; o artigo usa 7×7. O termo global não muda.
- F0 roda em Weak-SIGReg λ=0,01, a primeira da ordem de R8, de 1 vista. Mede o ruído no regime novo, não no de só reconstrução, que é um dos 15 braços de F1. O par aleatório da amostra de controle usa o mesmo regime.

## 8. Divergências resolvidas

| Tema | Eles disseram | Fica | Por quê |
|---|---|---|---|
| Quais frações entram no ajuste | P2: só células L. P1: só ≥ 25 imagens por classe | **L e R entram, com peso pelo ruído. Reportar b com e sem R.** X fica fora | Só L dá largura 1,72 contra 0,23, e 9 de 28 curvas reais batem na borda. Se os dois b discordarem, a lei vale só acima de ~5 imagens por classe |
| Fração 10% | Não listada em P2 | **Incluída** | Já existe no disco, é L em Eggs e helminth_eggs |
| Custo | α·β²·γ²≈2 do EfficientNet | **custo ≈ w^1,85 · r² · d^0,12** | Aqui uma camada a mais custa ~3%, porque roda na grade pequena; a L1 e a cabeça crescem linear em w. P3 mediu w^1,81 a 1,92 e d^0,11 a 0,13. A aproximação só propõe; quem aprova é a fórmula exata de MACs (P4 §3) |
| Rótulo no início | "Sem rótulo no treino" | **Declarado às claras: a inicialização SPiFiL usa classes** | `one_per_class` sorteia imagens por classe e o alocador divide filtros por classe (`largura // C`). São C × ipc imagens rotuladas por camada. "Sem rótulo" vale para o gradiente e para a escolha do checkpoint, não para o início. A tabela final traz quantas imagens rotuladas tocaram o modelo |
| `best_by_probe` | P4: exemplo com `true` | `false` | Regra nova, sonda varia 0,02 a 0,04 entre réplicas exatas |
| Quais perdas | P4: VICRegL e UR-JEPA só se ajudarem; CorInfoMax fora | As 7 no degrau 1 | Custa 15 de 60 épocas |
| Alcance em N | B0 a B3, 1 década | + w ¼, ½, 4 | `a` não sai com 1 década |
| Cabeça "nunca aleatória" | P3 perguntou | H0 segue como referência, fração SPiFiL registrada; H1 e H2 em F2 decidem por HS2 | Com 2 classes só 100 de 1280 filtros nascem do SPiFiL |
| Grade `sc` | Fila cheia | Só piloto, parada para dar lugar à descoberta. Checkpoints ficam | Outro regime: sem SSL, com sonda, só 200 px, 1 split |

## 9. Limites

- Simulações de P3 têm premissas próprias: comparam desenhos, não preveem κ. `b` piloto (0,47 a 0,70) vem de encoder congelado, não do nosso regime.
- `a` provavelmente sai só como cota ("a < 0,4"). α, β, γ saem grosseiros: separam "a largura leva a maior parte", não 1,21 de 1,35.
- 3 famílias derrubam uma regra, não cravam um expoente.
- Memória acima de w1 e custo de 448 px ainda não medidos. r=640 foi subestimado em até 1,6×: daí a margem.
- O ganho da regra por dataset sobre a melhor config comum (~0,004 de κ) fica abaixo do que 3 splits enxergam. A regra deve ganhar em custo, não em κ.

## 10. Mudanças depois de ver dado

Registradas aqui porque a seção 6 manda dizer. Resultados em `results_log.md`.

- **F1 degrau 2 maior que o plano:** 3 splits em vez de 1, com reconstrução como referência. São 72 cadeias no lugar de 20. Em troca, o degrau 3 encolhe: 4 dos 5 decisores já saem com 3 splits, falta só Eggs_subgroup.
- **Controle aleatório antecipado:** em B0 o aleatório empatou ou ganhou do SPiFiL. Sementes 43 e 44 em helminth_larvae e flowers entram já, pareadas com as sementes 43 e 44 de F0. A semente 42 não foi repetida, porque sortearia os mesmos filtros.
- **Regra de F0 sem resposta única:** o desvio agrupado (0,018) cai na faixa do meio, mas flowers sozinho (0,026) passa de 0,02, onde o plano manda pausar. A regra não dizia se o desvio é agrupado ou por dataset. Decisão pendente; o degrau 2 foi lançado por ordem do coordenador, sem resolver isso.
