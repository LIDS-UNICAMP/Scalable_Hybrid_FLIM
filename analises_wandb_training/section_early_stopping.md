# Early Stopping e Correlações Training→Avaliação

## Por que student_emb_norm é o Melhor Preditor (r=0.864)

A correlação de Pearson entre métricas de treino e kappa SVM downstream revela um resultado
contraintuitivo: a norma dos embeddings do student (`student_emb_norm`) prediz o desempenho
final com r=0.864 na época 50, superando tanto `cosine_sim` (r=0.486) quanto `train_loss`
(r=-0.529).

A explicação é conceitual: as duas métricas capturam fenômenos distintos.

- **cosine_sim** mede alinhamento direcional entre projeções do student e do teacher. É possível
  obter cosine_sim alto mesmo quando os embeddings têm norma próxima de zero — o student "aponta
  na direção certa" mas com vetores degenerados. Esse alinhamento superficial não gera
  representações discriminativas para a tarefa downstream.

- **emb_norm** mede a qualidade intrínseca do encoder: se ele aprendeu representações
  não-triviais. Norma baixa indica colapso representacional, independentemente de quanto o
  student imita a direção do teacher.

O caso do bloco `1x1_BN2d` ilustra o problema com clareza: nessas runs, `cosine_sim=0.637`
(valor razoável, sugerindo alinhamento), mas `emb_norm=0.066` (colapso). O kappa resultante
é próximo de zero. Em outras palavras, o student consegue imitar a *direção* do teacher sem
jamais aprender representações com energia suficiente para separar classes.

A qualidade do encoder (norma) é mais fundamental que o alinhamento da projeção. Norma alta
é condição necessária — embora não suficiente — para kappa alto.

---

## Tabela de Correlações por Época

Correlações de Pearson entre métricas de treino medidas em cada época e o kappa SVM final,
calculadas sobre todas as runs com dados disponíveis naquela época.

| Época | r(cosine_sim) | r(emb_norm) | r(scale_ratio) | r(train_loss) | n runs |
|------:|:-------------:|:-----------:|:--------------:|:-------------:|-------:|
|    10 |    +0.343     |   +0.575    |    +0.337      |    −0.423     |    158 |
|    25 |    +0.452     |   +0.798    |    +0.552      |    −0.513     |    162 |
|    50 |    **+0.486** | **+0.864**  |  **+0.547**    |  **−0.529**   |    162 |
|    75 |    +0.503     |   +0.859    |    +0.538      |    −0.533     |    162 |
| final |    +0.438     |   +0.845    |    +0.492      |    −0.481     |    168 |

Sinal das correlações é consistente com a intuição: emb_norm e cosine_sim mais altos → kappa
mais alto; train_loss mais alto → kappa mais baixo.

---

## Threshold Crítico: emb_norm < 0.05 na Época 10

O threshold emb_norm = 0.05 na época 10 separa de forma binária runs viáveis de runs em colapso:

| Condição               | Kappa médio | n runs | Interpretação        |
|:-----------------------|:-----------:|-------:|:---------------------|
| emb_norm ≥ 0.05        |    0.647    |    121 | Encoder ativo        |
| emb_norm < 0.05        |    0.122    |     37 | Colapso representacional |
| **Diferença**          | **+0.525**  |        |                      |

A separação de +0.525 kappa é expressiva. Refinando o threshold:

| Threshold              | Kappa (acima) | n   | Kappa (abaixo) | n   | Δ kappa |
|:-----------------------|:-------------:|----:|:--------------:|----:|:-------:|
| emb_norm ≥ 0.05        |     0.647     | 121 |     0.122      |  37 |  +0.525 |
| emb_norm ≥ 0.10        |     0.667     | 117 |     0.119      |  41 |  **+0.548** |
| emb_norm ≥ 0.20        |     0.687     | 107 |     0.184      |  51 |  +0.503 |
| emb_norm ≥ 0.30        |     0.683     |  55 |     0.440      | 103 |  +0.243 |

O threshold ótimo de separação binária é **emb_norm = 0.10 na época 10** (maior diferença:
Δ=0.548). Acima desse valor, o kappa esperado é 0.667; abaixo, 0.119 — praticamente chance.

Threshold de cosine_sim na época 25 também discrimina, mas de forma menos extrema:

| Condição               | Kappa médio | n   |
|:-----------------------|:-----------:|----:|
| cosine_sim ≥ 0.65      |    0.637    |  89 |
| cosine_sim < 0.65      |    0.402    |  73 |
| **Diferença**          | **+0.235** |     |

A separação por cosine_sim (Δ=0.235) é substancialmente inferior à separação por emb_norm
(Δ=0.548), confirmando que emb_norm é o sinal mais discriminativo já nas épocas iniciais.

---

## Análise por Grupo

Os três grupos de arquitetura apresentam comportamentos distintos que afetam a força da
correlação intra-grupo:

**1x1_BN2d**
O bloco de convolução 1×1 com BatchNorm 2D exibe colapso quase determinístico na época 10:
`emb_norm` converge para ~0.07 na maioria das runs, independentemente dos hiperparâmetros.
Como todas as runs problemáticas concentram-se num patamar muito baixo de norma, a
correlação inter-grupo é alta mas a variância intra-grupo é estreita. O kappa downstream é
próximo de zero para essas configurações, confirmando que o bottleneck arquitetural é o
responsável pelo colapso, não os hiperparâmetros de treinamento.

**3x3_BN2d**
O bloco 3×3 com BatchNorm 2D apresenta `emb_norm` estável no intervalo 0.35–0.39 ao longo
das épocas avaliadas. Com menor variância na norma, a correlação intra-grupo é mais fraca —
outros fatores (learning rate, dataset, pct_training) dominam a variabilidade do kappa. Ainda
assim, a tendência geral persiste: runs com emb_norm mais alto no extremo superior tendem a
kappa mais alto.

**next_layers**
Configurações que substituem camadas mais profundas da rede apresentam comportamento
heterogêneo. Crashes observados em configurações de alto pct_training podem distorcer a
correlação calculada nesse subgrupo, pois os crashes introduzem outliers com norma
artificialmente baixa por razões diferentes do colapso representacional típico. A análise
agrupada deve tratar esses outliers com cautela.

---

## Quando a Correlação se Estabiliza?

A evolução da correlação `r(emb_norm, kappa)` ao longo das épocas revela três fases:

| Intervalo       | r       | Δr     | Interpretação                                    |
|:----------------|:-------:|:------:|:-------------------------------------------------|
| Épocas 1→10     | 0.575   | —      | Sinal inicial: colapso vs. não-colapso separável |
| Épocas 10→25    | +0.223  | grande | Maior ganho informacional por época              |
| Épocas 25→50    | +0.066  | médio  | Refinamento do sinal; ainda vale monitorar       |
| Épocas 50→75    | −0.005  | ~zero  | Correlação estabiliza (~r=0.86)                  |
| Épocas 75→final | −0.014  | ~zero  | Sem ganho adicional; final ligeiramente inferior |

A queda marginal de r=0.864 (época 50) para r=0.845 (final) pode refletir pequenas instâncias
de overfitting de trajetória: runs que atingem norma alta em época 50 mas regridem ligeiramente
até o final tendem a ser penalizadas.

**Conclusão:** a época 25 é o **ponto ótimo de avaliação**. Com apenas 25% do orçamento total
de treinamento (assumindo 100 épocas), já se obtém r=0.798 — 92% do valor máximo atingível
(r=0.864 em época 50). O custo marginal de esperar até a época 50 é pequeno, mas o custo de
decidir antes da época 25 (r=0.575 na época 10) é substancial.

---

## Protocolo de Early Stopping (3 passos)

Baseado nos thresholds e correlações acima, propõe-se um protocolo sequencial de três verificações:

```
Passo 1 — Época 10: diagnóstico de colapso precoce
  SE student_emb_norm < 0.05
  ENTÃO encerrar run
  (kappa esperado: 0.12; probabilidade de kappa > 0.5 ≈ 0)

Passo 2 — Época 25: diagnóstico de alinhamento + qualidade
  SE cosine_sim < 0.50 E emb_norm < 0.15
  ENTÃO encerrar run
  (combinação indica ausência de alinhamento E norma insuficiente)

Passo 3 — Época 50: avaliação de crescimento
  SE emb_norm ainda crescendo entre épocas 25 e 50
  ENTÃO continuar até 200 épocas
  SENÃO avaliar se platô é aceitável antes de continuar
```

Cada passo é condição necessária — uma run que passa nos três checks tem alta probabilidade
de atingir kappa > 0.5. O passo 1 é o mais crítico: as 37 runs com emb_norm < 0.05 na época
10 consumiriam recursos sem nenhuma perspectiva de resultado útil.

Estimativa de economia: se o passo 1 for aplicado prospectivamente, ~23% das runs (37/158)
seriam encerradas na época 10, poupando os recursos das épocas 11–200 para essas
configurações.

---

## Implementação Prática

Para usar o protocolo em novos experimentos, as seguintes métricas devem ser logadas durante
o treinamento e verificadas nos passos correspondentes:

1. **`student_emb_norm`** — norma L2 média dos embeddings do student no batch de validação.
   Deve ser logada a cada época. Threshold crítico: 0.05 (época 10), 0.15 (época 25).

2. **`cosine_sim`** — similaridade cosseno média entre embeddings student e teacher no espaço
   de projeção. Threshold secundário: 0.50 (época 25), usado apenas em conjunto com emb_norm.

3. **`train_loss`** — já disponível em qualquer pipeline de distilação. Correlação negativa
   com kappa (r=−0.529) mas menos discriminativa que emb_norm isoladamente.

Ordem de prioridade para monitoramento: emb_norm > train_loss > cosine_sim > scale_ratio.

Sugestão de implementação no callback de treinamento:

```python
# Época 10
if epoch == 10 and student_emb_norm < 0.05:
    logger.warning(f"Run {run_id}: emb_norm={student_emb_norm:.3f} < 0.05. Encerrando.")
    raise EarlyStopException("colapso_ep10")

# Época 25
if epoch == 25 and cosine_sim < 0.50 and student_emb_norm < 0.15:
    logger.warning(f"Run {run_id}: cosine_sim={cosine_sim:.3f}, emb_norm={student_emb_norm:.3f}. Encerrando.")
    raise EarlyStopException("baixo_alinhamento_ep25")
```

---

## Limitações da Análise

**1. Causalidade vs. correlação**
As correlações são observacionais. emb_norm alto co-ocorre com kappa alto, mas não está
provado que intervir para forçar emb_norm alto (e.g., via regularização de norma) produziria
o mesmo resultado. A norma pode ser sintoma de uma arquitetura funcional, não a causa do
desempenho.

**2. Threshold não generaliza automaticamente**
Os thresholds (0.05, 0.10, 0.15) foram derivados do conjunto de runs atual. Para novas
arquiteturas ou datasets, os valores absolutos de emb_norm podem diferir. O threshold deve
ser re-calibrado ao introduzir novas famílias de arquitetura.

**3. Grupos não balanceados**
Os 37 runs com emb_norm < 0.05 são predominantemente do grupo `1x1_BN2d`. A correlação
global r=0.864 reflete em parte a separação entre grupos arquiteturais, não apenas variação
intra-grupo. Análises intra-grupo mostrarão correlações mais fracas.

**4. Épocas avaliadas são pontos discretos**
A análise usa épocas 10, 25, 50, 75 e final. A dinâmica entre esses pontos não foi observada.
É possível que emb_norm exiba oscilações entre épocas que o protocolo não capturaria.

**5. Correlação final ligeiramente inferior à época 50**
r=0.845 no final vs. r=0.864 na época 50 sugere que algumas runs com emb_norm alto em época
50 regridem até o final. Isso pode indicar instabilidade tardia em certas configurações que
não é capturada pelo protocolo de 3 passos proposto acima.

**6. n=158 na época 10 vs. n=168 no final**
A diferença de 10 runs indica que algumas runs não chegaram a logar métricas na época 10
(crashes muito precoces). Essas runs não estão incluídas na análise do passo 1, o que pode
subestimar ligeiramente a proporção de colapsas totais.
