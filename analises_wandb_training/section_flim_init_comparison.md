# 1x1_flim_init vs trunc_normal: Comparação Detalhada

## Sumário

Este documento analisa o comportamento de duas estratégias de inicialização para o encoder CNN 1×1 BN2d no pipeline de destilação I-JEPA → FLIM: **flim_init** (pesos pré-treinados do FLIM para filtros de segmentação microscópica) e **trunc_normal** (inicialização aleatória truncada, baseline). A comparação revela um paradoxo fundamental: a inicialização FLIM produz normas de embedding 170× maiores na época 99, mas converge para um alinhamento cosine *inferior* (0.571 vs 0.637). Essa aparente contradição tem explicação estrutural clara e tem implicações diretas para a estratégia de destilação.

---

## Tabela Comparativa: Métricas por Época

| Época | loss (flim) | loss (trunc) | cosine (flim) | cosine (trunc) | emb_norm (flim) | emb_norm (trunc) | scale_ratio (flim) | scale_ratio (trunc) |
|------:|------------:|-------------:|--------------:|---------------:|----------------:|-----------------:|-------------------:|--------------------:|
|     0 |       0.717 |        0.621 |         0.018 |          0.019 |         270.169 |            0.579 |              0.761 |               0.807 |
|    10 |       0.377 |        0.330 |         0.315 |          0.407 |          48.549 |            0.080 |              0.328 |               0.222 |
|    25 |       0.279 |        0.272 |         0.478 |          0.536 |          16.379 |            0.066 |              0.342 |               0.300 |
|    50 |       0.250 |         N/A  |         0.544 |           N/A  |          11.034 |             N/A  |              0.390 |                N/A  |
|    75 |       0.245 |         N/A  |         0.559 |           N/A  |          11.300 |             N/A  |              0.409 |                N/A  |
|    99 |       0.241 |        0.229 |         0.571 |          0.637 |          11.248 |            0.066 |              0.422 |               0.404 |

**Interpretação geral:** trunc_normal supera flim_init em todas as métricas de qualidade na época 99 (loss menor, cosine maior), apesar de partir de um estado inicial melhor em termos de loss (0.621 vs 0.717). A inicialização FLIM impõe um custo de adaptação ao espaço representacional do teacher I-JEPA que não é totalmente amortizado em 100 épocas.

---

## O Paradoxo Central: Norma Alta × Cosine Sim Baixo

### O fenômeno

Na época 99, flim_init apresenta `emb_norm ≈ 11.248` enquanto trunc_normal apresenta `emb_norm ≈ 0.066` — uma diferença de **170×**. Intuitivamente, embeddings com norma maior parecem "mais informativos". No entanto, a similaridade cosine do flim_init (0.571) é *inferior* à do trunc_normal (0.637).

### Por que isso acontece?

A chave está em entender o que a similaridade cosine mede: **alinhamento direcional** no espaço de embedding, independente da magnitude. A fórmula é:

```
cos(θ) = (z_student · z_teacher) / (||z_student|| × ||z_teacher||)
```

A norma do embedding aparece tanto no numerador (via produto interno) quanto no denominador (via normalização), cancelando-se. O que determina o cosine é puramente a **orientação** dos vetores no espaço de alta dimensão.

### A origem estrutural do paradoxo

Os pesos FLIM foram otimizados para **filtros locais de segmentação microscópica**: detecção de bordas, texturas de larvas, morfologia de ovos, gradientes de intensidade em imagens de fluorescência (FLIM). Esses filtros geram ativações fortes — daí a norma alta — mas em **direções do espaço de features que são estruturalmente diferentes** das representações aprendidas pelo teacher I-JEPA.

O teacher I-JEPA é um ViT-H/14 treinado em ImageNet com **self-attention global**: suas representações capturam semântica de cena, relações espaciais de longa distância, e abstrações de alto nível. As direções que o ViT-H/14 considera "informativas" são ortogonais (ou quase) às direções que os filtros FLIM consideram informativas.

Resultado: o encoder FLIM gera embeddings com **sinal forte** (norma alta) mas apontando em **direção errada** para o teacher. O trunc_normal, partindo de zero, não tem esse viés estrutural — seus pesos se orientam livremente para as direções do teacher desde o início do treinamento.

### A dinâmica de adaptação

```
flim_init:  ep0=0.018  →  ep10=0.315  →  ep25=0.478  →  ep99=0.571
trunc_norm: ep0=0.019  →  ep10=0.407  →  ep25=0.536  →  ep99=0.637
```

O flim_init **nunca alcança** o trunc_normal em alinhamento cosine. O gap em época 10 (0.315 vs 0.407, diferença de 0.092) reduz para 0.066 em época 99 — mas não fecha. Isso sugere que os pesos FLIM estão aprisionados em um **mínimo local diferente**: representações locais bem estruturadas mas mal alinhadas com o teacher global.

---

## Evolução da emb_norm por Dataset

### flim_init: colapso controlado da norma

| Dataset   | ep0     | ep10   | ep25   | ep50   | ep99    |
|-----------|--------:|-------:|-------:|-------:|--------:|
| eggs      | 343.337 | 68.474 | 19.131 | 11.594 | 11.364  |
| larvae    | 211.529 | 38.848 | 16.345 | 11.684 | 11.133  |
| protozoan | 248.375 | 33.213 | 12.303 |  8.858 | N/A     |

**Padrão:** a emb_norm começa extremamente alta (210–343) e colapsa ~95% até a época 99, estabilizando em torno de 11.0–11.4. O treinamento é dominado pelo processo de **normalização implícita** — o otimizador aprende a reduzir a magnitude dos pesos FLIM antes de aprender a orientá-los corretamente.

**Por dataset:** eggs apresenta a maior norma inicial (343.337), consistente com imagens de fluorescência com alta variação de intensidade. larvae tem norma inicial menor (211.529), sugerindo que os filtros FLIM são mais uniformes para esse tipo de imagem. A convergência final (ep99) é similar para eggs e larvae (~11.2–11.4), indicando que o treinamento eventualmente normaliza as representações para um nível comparável independente do dataset.

**Protozoan:** os dados de ep50 e ep99 estão ausentes (18 runs registrados em 2026-05-30 possivelmente ainda em execução no servidor). Os dados disponíveis (ep0=248.375, ep10=33.213, ep25=12.303) indicam colapso igualmente abrupto, compatível com os outros datasets.

---

## Interpretação dos Pesos FLIM no Contexto de Destilação

### O que os pesos FLIM trazem de positivo

Os pesos pré-treinados FLIM carregam **conhecimento de domínio real**: estrutura de bordas em imagens de fluorescência, filtros otimizados para microscopia, sensibilidade a texturas biológicas. Isso se reflete na loss inicial mais *baixa que o esperado* para uma inicialização com norma tão alta — o modelo já entende o "tipo de imagem" mesmo sem conhecer o espaço do teacher.

Além disso, a convergência em critério de loss estagna mais cedo: a curva flim_init mostra **75% dos runs não convergindo em 100 épocas** (comparado a 100% para trunc_normal — veja seção de Convergência). Isso indica que o ponto de partida FLIM já está em uma região do espaço de loss que o otimizador considera "bom o suficiente" mais cedo.

### O problema estrutural

O pipeline de destilação usa **KL divergence** entre as distribuições de patch embeddings do student e do teacher. Para minimizar essa divergência, o student precisa mapear os patches FLIM para o **espaço de representação do ViT-H/14**. Esse é fundamentalmente um problema de alinhamento de espaço vetorial entre dois modelos com topologias representacionais distintas:

- **Teacher (ViT-H/14):** atenção global, patches 14×14, representações contextuais
- **Student (CNN 1×1):** receptivo local, convolução pontual, sem contexto espacial
- **Inicialização FLIM:** filtros de segmentação otimizados para reconhecimento de estruturas locais em FLIM

A convolução 1×1 com pesos FLIM é, por construção, um mapeamento de features locais para features locais. O ViT espera features que codificam **contexto global**. Esse mismatch estrutural é a raiz do paradoxo: norma alta (sinal forte, bem estruturado localmente) + cosine baixo (mal alinhado globalmente).

---

## Velocidade de Aprendizado

### Cosine similarity como proxy de alinhamento com o teacher

```
Variação de cosine por intervalo de épocas:

                    flim_init    trunc_normal
ep0  → ep10:   +0.297          +0.388         ← trunc aprende 31% mais rápido
ep10 → ep25:   +0.163          +0.129         ← flim recupera velocidade
ep25 → ep99:   +0.093          +0.101         ← similar na fase final
```

**Época 0–10 (fase crítica):** trunc_normal avança 0.388 pontos de cosine enquanto flim_init avança apenas 0.297. A diferença é máxima aqui porque o encoder FLIM está realizando dois processos simultâneos: (1) reduzir a norma (de 270 para 48, queda de 82%) e (2) re-orientar os pesos na direção do teacher. Esses processos competem pelos gradientes, diluindo a velocidade de aprendizado efetivo.

**Época 10–25 (fase de aceleração):** flim_init surpreendentemente *supera* trunc_normal em velocidade (+0.163 vs +0.129). Isso ocorre porque, após o colapso inicial da norma, os pesos FLIM estabilizaram em uma geometria interna mais estruturada — o encoder começa a "traduzir" seus filtros locais para o espaço do teacher com mais eficiência.

**Época 25–99 (fase de plateau):** as velocidades convergem (~0.093–0.101), mas o gap absoluto de 0.066 pontos de cosine persiste. Não há evidência de que gap fecharia com mais épocas — ambos parecem ter atingido seus respectivos platôs.

---

## Os 18 Runs Faltantes de Protozoan

Para flim_init, apenas **eggs (18 runs) + larvae (18 runs) = 36 runs** estão completos. Os 18 runs de protozoan foram registrados em 2026-05-30 mas provavelmente ainda estão em execução no servidor.

**Impacto nas métricas:**
- A `emb_norm` média global reportada (11.248 na época 99) é calculada sobre eggs + larvae apenas, sem protozoan
- Com base nos dados parciais de protozoan (ep25=12.303 vs eggs=19.131, larvae=16.345), a norma de protozoan converge para valores menores — a inclusão dos dados completos de protozoan deve *reduzir* ligeiramente a emb_norm média final
- As métricas de cosine_sim e loss não são afetadas se forem médias por run e não por dataset

**Recomendação:** reanalisar as métricas agregadas após a conclusão dos runs de protozoan. A comparação flim_init vs trunc_normal permanece válida para eggs e larvae, mas a generalização para todos os datasets só será conclusiva com os dados completos de protozoan.

---

## Convergência: flim_init é mais rápido?

### Critério de convergência de loss

- **flim_init:** 75% dos runs não convergem em 100 épocas
- **trunc_normal:** 100% dos runs não convergem em 100 épocas

À primeira vista, isso indica que flim_init **converge mais cedo** — apenas 25% dos runs ainda estão descendo ativamente a loss no fim do treinamento, enquanto trunc_normal está ainda em descida em 100% dos runs.

### Interpretação cuidadosa

Esse resultado pode ser lido de duas formas:

1. **Favorável ao flim_init:** os pesos pré-treinados fornecem um ponto de partida próximo a um mínimo razoável. O encoder atinge uma solução "boa o suficiente" mais rapidamente porque o espaço de features FLIM já é estruturado — o otimizador precisa de menos iterações para encontrar uma representação estável.

2. **Desfavorável ao flim_init:** os runs flim_init convergem cedo porque ficam **presos em mínimos locais rasos**. O ponto de partida FLIM está em uma região do espaço de loss que é bem condicionada localmente (filtros estruturados) mas globalmente subótima (mal alinhada com o teacher). O treinamento para antes de atingir o mínimo global, enquanto trunc_normal continua descendo e eventualmente alcança loss 0.229 vs 0.241.

**A evidência favorece a interpretação 2:** trunc_normal tem loss final menor (0.229 vs 0.241) e cosine final maior (0.637 vs 0.571) apesar de ainda estar "em movimento" em época 99. Isso é consistente com trunc_normal ainda descendo em direção a um mínimo mais profundo, enquanto flim_init estancou em um platô subótimo.

---

## Conclusões

### Rankings finais (época 99)

| Métrica         | Melhor          | Diferença relativa |
|-----------------|-----------------|--------------------|
| loss            | trunc_normal    | -5.0% (0.229 vs 0.241) |
| cosine_sim      | trunc_normal    | +11.6% (0.637 vs 0.571) |
| emb_norm        | trunc_normal*   | 170× menor (0.066 vs 11.248) |
| scale_ratio     | similar         | +4.3% (0.404 vs 0.422) |
| convergência    | flim_init       | 75% vs 100% ainda em descida |

*Norma menor não é necessariamente melhor — depende do regime de normalização downstream.

### Implicações para o pipeline de destilação

1. **trunc_normal é superior em qualidade final** para este pipeline específico (CNN 1×1 → ViT-H/14), medido por loss e cosine_sim em 100 épocas.

2. **O paradoxo norma-cosine** é explicado pelo mismatch estrutural entre o espaço de features FLIM (local, filtros de segmentação) e o espaço do teacher I-JEPA (global, contextual). Norma alta indica sinal forte, não alinhamento correto.

3. **flim_init tem vantagem de convergência antecipada** mas paga com um platô subótimo. Para domínios onde a qualidade final é crítica, trunc_normal é preferível.

4. **A fase crítica de adaptação é época 0–10:** é nessa janela que flim_init sofre o maior custo — simultânea normalização da magnitude e reorientação direcional. Estratégias de mitigação incluem learning rate warmup mais lento ou regularização L2 mais forte nas primeiras épocas para flim_init.

5. **Dados incompletos de protozoan** impedem conclusão definitiva sobre generalização cross-dataset. Revisão recomendada após conclusão dos 18 runs pendentes.
