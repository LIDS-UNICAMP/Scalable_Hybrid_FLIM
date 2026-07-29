# Sanity check — experimentos `_relu2l`

**Data:** 2026-07-28 · **git:** `503fb5a`
**O que foi auditado:** os 18 runs `artifacts/classification_flim/classhead_*_relu2l`
(eggs, larvae, protozoan × splits 1-3 × 5% e 75% dos dados, todos com o encoder destravado),
avaliados em `results/relu2l_test_results.csv`.
**Como:** 4 investigações independentes e paralelas — pipeline de treino, dados e splits,
checkpoints e avaliação, e cruzamento com as análises que já existiam no repositório.

**O que disparou a auditoria:** 8 dos 18 runs prevendo sempre a mesma classe, com κ = 0.

Só estão listados aqui os problemas que precisam ser conhecidos.

---

## CRÍTICO

### C1 — O ReLU antes do Softmax trava o treino. Em 4 runs o modelo nunca aprendeu nada

**O que acontece.** A cabeça termina em `Linear(24,C) → ReLU → Softmax`. O ReLU zera todo número
negativo. Quando as saídas do `Linear` ficam todas negativas, viram todas zero — e aí o modelo
não recebe mais nenhum sinal de aprendizado (o gradiente é exatamente zero). Não existe volta:
o ReLU não deixa esses valores voltarem a ser positivos.

**Prova de que é isso.** Testando o modelo **na inicialização**, antes de qualquer treino:

| run | % das imagens com todas as saídas ≤ 0 | classes vivas |
|:--|--:|--:|
| larvae split1 | **100%** | 0 de 2 |
| larvae split2 | 85% | 0,15 de 2 |
| larvae split3 | 0% | 1,76 de 2 |

Ou seja: `larvae` split1 **já nasce morto**. E o split3, que nasce vivo, é justamente o que funciona.

**Confirmação no treino** (histórico do W&B lido direto dos arquivos locais): 100 épocas com
`kappa = 0.0000` do início ao fim, e a loss parada em `0.693147` — que é exatamente ln(2), o valor
de um modelo que responde "50% para cada classe" sempre. O modelo passou 100 épocas sem se mexer.

**Comparação que fecha o caso:** mesmo dataset, mesmo split, mesma seed, mesmos pesos FLIM, só
tirando o ReLU (grupo `sigmoid2l_`): larvae s1 κ = **0.92**, larvae s2 κ = **0.94**, protozoan s1 κ = **0.83**.
Contra 0.00 nos três com ReLU.

**O que fazer.** Tirar o ReLU da saída. Ele é redundante (o Softmax já normaliza) e impede o modelo
de expressar evidência negativa. Se o ReLU for exigência do experimento, usar `Softplus` ou `LeakyReLU`.

---

### C2 — Os outros 4 colapsos NÃO são o ReLU. É desbalanceamento

Isso corrige o que está escrito na §5 do report de 28/07, que atribuiu os 8 colapsos à mesma causa.

Em `protozoan` (splits 1 e 3) e `eggs` (split 3, 5%), **nenhuma imagem** tem todas as saídas zeradas.
O modelo simplesmente aprendeu a chutar sempre a **classe mais frequente** — protozoan tem 59,8% de
uma classe só, eggs tem 65,4%. Esse é o colapso clássico por desbalanceamento, que acontece com
qualquer cabeça, com ou sem ReLU.

| run | saída uniforme? | classe prevista | quanto ela representa |
|:--|:--|:--|--:|
| larvae s1, s2 | sim (100% / 84%) | classe 0 — a **minoritária** | 12,7% |
| protozoan s1, s3 | não | classe 6 — a **majoritária** | 59,7% |
| eggs s3 pct5 | não | classe 8 — a **majoritária** | 65,4% |

São dois problemas diferentes com o mesmo sintoma. Vale notar que em `protozoan s1` o ReLU **também**
mata permanentemente as classes 0 e 3 (elas nunca podem ser previstas), então ele piora o quadro —
mas não é ele que causa o colapso ali.

---

### C3 — As métricas dos nossos CSVs são "macro", com nome de coluna errado

`compute_metrics` usa `average="macro"` por padrão, e o `multiclass_accuracy` do torchmetrics também.
Resultado, nos arquivos `relu2l_test_results.csv` **e** `sigmoid2l_test_results.csv`:

- `test_accuracy` **não** é a acurácia normal — é a acurácia **balanceada** (média do acerto por classe);
- `test_f1_weighted` **não** é ponderado — é **F1 macro**.

Isso faz os nossos números parecerem muito piores do que são:

| run | o que está no CSV | acurácia real (global) |
|:--|--:|--:|
| eggs_s3_pct75 | 0.486 | **0.833** |
| eggs_s1_pct75 | 0.411 | **0.779** |
| protozoan_s3_pct75 | 0.561 | **0.846** |
| protozoan_s2_pct75 | 0.264 | **0.725** |

Os runs de 75% estão em **73–85% de acurácia global**, não em 26–49%.

Os CSVs do Felipe (`data/reports_felipe/`) usam a métrica global. Então **as comparações de acurácia e
F1 entre "nossos" resultados e os dele estão erradas** — e sempre contra os nossos. Isso vale também
para o report de 22/07.

**Exceção importante:** o **kappa não tem essa ambiguidade** (não existe versão macro dele). Todas as
conclusões baseadas em κ continuam válidas e já são comparáveis com o Felipe hoje, sem precisar
recalcular nada.

---

### C4 — O `best_kappa.ckpt` de 8 runs é a época 0. Avaliamos modelos não treinados

O checkpoint é escolhido por `monitor="val/kappa", mode="max"`. Quando o kappa fica preso em 0.0, a
primeira época nunca é superada — e o arquivo salvo é o do começo do treino.

```
larvae s1/s2 pct5 e pct75 → época 0     protozoan s1 pct75 → época 0
protozoan s3 pct5         → época 0     eggs s1/s2/s3 pct5 → épocas 1 a 3
```

Nos runs saudáveis: épocas 60 a 97. E nos colapsados o encoder mal se moveu em relação aos pesos FLIM
originais (diferença ≤ 0,0019, contra 1,5–6,8 nos saudáveis) — ou seja, ele nem chegou a treinar.

Isso não muda a conclusão (na época 99 esses runs ainda tinham κ = 0), mas significa que os números
reportados descrevem a **inicialização**, não o modelo final.

---

## ALTO

### A1 — `eggs_split2_pct5` está pior que o acaso. Investigar antes de publicar

Este é o único caso que parece um erro de verdade, e não uma consequência da arquitetura:

```
acurácia global = 3,6%   (com 9 classes, o acaso é 11%)
previsões: 2265 das 2557 imagens na classe 3
mas a classe 3 é só 2,4% do conjunto de teste
```

O modelo despeja quase tudo numa classe rara. Não é colapso — é comportamento anti-correlacionado.
A métrica macro (0,1315) esconde isso completamente. **Vale conferir os rótulos e o split desse run.**

### A2 — `last.ckpt` não é a última época. Os pesos finais foram perdidos

Nos 36 checkpoints inspecionados (18 `relu2l` + 18 `sigmoid2l`), o `last.ckpt` tem exatamente a mesma
época do `best_kappa.ckpt`. Causa: nesta versão do Lightning, com `save_last=True`, o `last` só é
gravado quando um checkpoint "melhor" também foi gravado.

Duas consequências: **(1)** não dá para analisar o modelo da época 99 sem retreinar; **(2)** um rerun
com o mesmo nome retomaria do checkpoint *melhor*, não do último, sem avisar.
**Correção:** `save_last="link"` ou um `ModelCheckpoint` extra sem `monitor`.

### A3 — Comparar média por dataset inverteu uma conclusão

A média indicava que o ReLU era **melhor** em `eggs`. Comparando split a split, o quadro muda:

| dataset | split | κ ReLU | κ Sigmoid |
|:--|--:|--:|--:|
| eggs | 1 | 0.526 | **0.798** |
| eggs | 2 | **0.498** | 0.028 |
| eggs | 3 | **0.656** | 0.530 |
| larvae | 1 | 0.000 | **0.918** |
| larvae | 2 | 0.000 | **0.944** |
| protozoan | 1 | 0.000 | **0.828** |

O "ganho" em `eggs` vinha só do split 2 — onde quem colapsou foi o run **Sigmoid**. Comparando em pares,
**o ReLU perde em 6 dos 9 splits**. Quando há runs colapsados na amostra, média ± desvio engana:
usar comparação pareada por split.

### A4 — O problema não é o encoder, é a otimização da cabeça

Já existia no repositório o controle que prova isso: `results/relu_vs_sigmoid_flatten.csv` mostra que um
classificador linear simples, sobre o encoder FLIM **congelado**, acerta **96,8%** em larvae split1.
Os nossos runs `relu2l` de larvae, com o encoder **livre** (mais capacidade, portanto), dão κ = 0.

Ou seja: a informação está lá nas features. Quem falha é o treino da cabeça.

### A5 — A Sigmoid da camada oculta está saturada, e isso estraga até os runs bons

A saída da camada oculta quase não varia entre imagens diferentes (desvio de 0,004 a 0,04 nos runs
colapsados, contra 0,29 a 0,32 nos bons). Como o sinal da saída passa a ser decidido pelo viés (bias) e
não pela imagem, o destino de cada classe já fica definido na inicialização — é por isso que o colapso
depende do split e **não** da quantidade de dados.

Efeito num run que "funcionou": `eggs_split1_pct75` tem **4 das 9 classes que nunca podem ser previstas**,
e κ 0.526 contra 0.811 do mesmo run sem ReLU.

### A6 — Comparar com o MLP do Felipe é comparar 6 diferenças ao mesmo tempo

As tabelas do report põem lado a lado modelos que diferem em muito mais que o ReLU:

| | nosso `relu2l` | `flim_mlp` do Felipe |
|:--|:--|:--|
| cabeça | 1 401 params | 46 601 params (**33× maior**) |
| ativação oculta | Sigmoid | ReLU + Dropout 0.3 |
| ReLU na saída | **sim** | não |
| weight decay | **5e-2** | 1e-4 (**500× menor**) |
| lr do encoder | igual ao resto | 10× menor que o resto |
| épocas | 100 fixas | 300 com early stopping |

O `weight_decay = 5e-2` numa cabeça de 1,4 mil parâmetros é, sozinho, um forte candidato a causar
colapso — e não tem nada a ver com o ReLU. Só a comparação **ReLU-nosso vs Sigmoid-nosso** isola o ReLU
de verdade; as demais tabelas não.

---

## MÉDIO

### M1 — O treino roda sem nenhuma augmentation
`V_train=1` faz o dataset usar o transform determinístico de teste
(`src/data_modules/datasets/parasite_lejepa.py:109`). Nenhuma augmentation no fine-tuning supervisionado.
Afeta os dois grupos igual, mas tira toda a aleatoriedade que ajudaria a escapar do colapso.

### M2 — Em `pct100`, treino e validação são o mesmo conjunto
Os JSONs `data_descriptor_perc100.json` têm listas idênticas (mesmo hash) em `train` e `validation`, nos
3 datasets e 3 splits. Não afeta este grupo (usa 5% e 75%), mas **invalida a escolha de checkpoint em
qualquer experimento a 100%**.

### M3 — A flag `no_imagenet_norm` não é salva no `run_metadata.json`
A avaliação (`eval_sigmoid2l_test.py:75`) sempre assume que a normalização foi usada. Está certo desta
vez, mas qualquer grade futura com `--no-imagenet-norm` seria avaliada com a normalização errada, em silêncio.

### M4 — A normalização ImageNet é aplicada sobre imagem LAB
Médias e desvios de RGB aplicados na saída LAB do loader `ift_lab`. Conceitualmente errado. É igual nos
dois grupos, então não explica a diferença — mas contribui para a saturação do A5.

### M5 — A loss não tem peso por classe
`F.nll_loss` sem `weight`, com larvae 87% numa classe e protozoan 60% numa classe. Prever sempre a
majoritária vira um mínimo local barato. Não é a causa, mas reduz a margem de segurança.

---

## Verificado e SEM problema (para ninguém reinvestigar)

- **Dados íntegros.** Sem sobreposição entre treino, validação e teste; a soma bate com o total de imagens
  em disco (eggs 5112, larvae 3514, protozoan 9568); nenhum arquivo faltando ou duplicado.
- **Splits equivalentes entre si.** larvae tem treino `{classe 0: 167, classe 1: 1150}` e teste
  `{0: 223, 1: 1534}` — **os mesmos números** nos splits 1, 2 e 3. A ideia de que "o split 1 tem uma
  classe só" está descartada, e o padrão "s1/s2 falha, s3 funciona" não vem dos dados.
- **Rótulos e número de classes corretos** (9 / 2 / 7), consistentes entre treino e teste.
- **O ReLU não some no carregamento.** `output_relu=True` está salvo nos hiperparâmetros dos 18
  checkpoints e é reconstruído no load. Hipótese descartada.
- **O encoder FLIM foi carregado certo**, conferido contra `data/to_mateus/model/.../train{N}/models`.
- **Grid e manifest limpos.** 18 linhas `status=ok`, nada pulado ou reaproveitado.
- **Sem troca de split.** O comando real de cada run confirma `--split N` e os pesos de `train{N}` batendo.
- **Hiperparâmetros idênticos nos 18 runs**; sem early stopping; todos completaram a época 99.
- **A avaliação bate com o treino.** As métricas de validação do W&B batem com o CSV de teste, e os 8 runs
  com κ = 0 no teste têm `best_val_kappa = 0.0` gravado no treino. **O problema é de treino, não da avaliação.**

---

## O que fazer, em ordem

| # | Ação | Por quê |
|:--|:--|:--|
| 1 | Tirar o ReLU antes do Softmax (ou trocar por `Softplus`) | C1 |
| 2 | Investigar `eggs_split2_pct5` (3,6% de acurácia global) | A1 — único candidato a erro real |
| 3 | Publicar as métricas na convenção global (ou pôr κ em primeiro lugar nas tabelas) | C3 |
| 4 | Trocar o `monitor="val/kappa"` por um critério que não fique preso em zero | C4 |
| 5 | Corrigir a §5 do report de 28/07: são dois mecanismos, não um | C2 |
| 6 | Usar comparação pareada por split nos relatórios | A3 |
| 7 | `save_last="link"` ou um `ModelCheckpoint` sem `monitor` | A2 |
| 8 | Baixar o `weight_decay` (5e-2 é alto demais para uma cabeça de 1,4K params) | A6 |
| 9 | Ligar augmentation; corrigir `perc100`; salvar `no_imagenet_norm` | M1, M2, M3 |
