# Diagnóstico de eggs — onde o κ se perde (A4, 2026-11-02, só validação)

**Resposta curta:** quase nenhum erro de eggs é ovo confundido com outro ovo. **99% dos erros são ovo ↔ impureza.** São impurezas classificadas como ovo (sobretudo Ascaris, H. nana e Taenia) e ovos finos ou de casca fina classificados como impureza (Enterobius, Ancylostoma, Schistosoma). A L4 1x1 + GeM já corta pela metade os ovos perdidos para impureza na L3. O que sobra é discriminar ovo de impureza parecida, não separar uma espécie de ovo da outra.

## Como medi

- **Modelo:** bp01w64c1280, eggs, 50%, splits 1–3. Leituras: **gem L4** (a principal) e **avg L3** (para ver se o erro já existe na L3).
- **Avaliação:** o mesmo caminho do `eval/svm._run_svm_job`: `eval.growth_stages._encoder`, `eval.svm.extract_features`, `core.svm.fit_svm` (fit no treino) e `core.metrics.compute_metrics` (na validação). Sobre as predições roda só a `sklearn.metrics.confusion_matrix`. Nenhum SVM foi duplicado.
  - Rodou na CPU, sob `flock scratchpad/svm.lock` e com `nice -n 10`.
  - O κ bate com `results/svm_disc_bp01w64c1280_val_gem.csv`: 0,909 / 0,891 / 0,894.
- **Morfologia:** máscaras de `data/to_modules/new_split_parasito/helminth-eggs/masks`, nas 3.837 imagens de validação dos 3 splits.
  - Medidas em 200×200: área, diâmetro equivalente, eixos (PCA da máscara), contraste ΔE em LAB (média dentro contra fora) e textura (desvio de L dentro do objeto, Lstd).
- **Classes:** a ordem de `eval/constants.py`.
- **Scripts:** `scratchpad/a4/eggs_diag.py` e `scratchpad/a4/eggs_cross.py`. Saídas em `scratchpad/a4/eggs_diag/`.

## 1. Confusão e recall por classe (soma dos 3 splits)

| classe | n val | recall avg L3 | **recall gem L4** | ovo → impureza (L4) | impureza → esta classe (L4) |
|---|---|---|---|---|---|
| H. nana | 261 | 0,904 | 0,958 | 4,2% | 25 |
| H. diminuta | 60 | 0,950 | 0,983 | 1,7% | 10 |
| Ancylostoma | 111 | 0,694 | **0,883** | **11,7%** | 8 |
| Enterobius | 93 | **0,452** | **0,849** | **15,1%** | 17 |
| Ascaris | 252 | 0,817 | 0,937 | 6,0% | **33** |
| Trichuris | 282 | 0,926 | 0,961 | 3,5% | 17 |
| Schistosoma | 93 | 0,860 | 0,903 | **9,7%** | 9 |
| Taenia | 177 | 0,927 | 0,955 | 4,5% | 18 |
| Impurities | 2.508 | 0,950 | 0,945 | — | — |

| erros (3 splits) | total | **ovo ↔ impureza** | ovo → impureza | impureza → ovo | ovo ↔ ovo |
|---|---|---|---|---|---|
| avg L3 (κ 0,840) | 332 | **327 (98%)** | 201 | 126 | 5 |
| **gem L4 (κ 0,898)** | 220 | **218 (99%)** | 81 | **137** | **2** |

- **O erro já existe na L3**, e com mais força. Da L3 para a L4, a quantidade de ovos perdidos para impureza cai de 201 para 81: Enterobius vai de 0,45 para 0,85 de recall e Ancylostoma de 0,69 para 0,88.
- **As impurezas classificadas como ovo não caem** (126 → 137). Elas viram o maior bloco de erro na L4.

## 2. Morfologia por classe (val; mediana [IQR])

| classe | área (px) | diâm. equiv. | eixo maior / menor | ΔE LAB | Lstd (textura) |
|---|---|---|---|---|---|
| Taenia | 499 [481–519] | 25 | 26 / 25 (redondo) | 0,49 | 0,077 |
| Trichuris | 578 [560–596] | 27 | 37 / 20 (alongado) | 0,53 | 0,079 |
| Enterobius | 651 [624–702] | 29 | 40 / 21 (alongado) | 0,61 | 0,078 |
| H. nana | 726 [673–774] | 30 | 33 / 29 | 0,66 | 0,082 |
| **Impurities** | **1.084 [575–1.862]** | 37 | 43 / 32 | 0,55 | 0,087 |
| Ancylostoma | 1.149 [1.106–1.219] | 38 | 45 / 33 | 0,54 | **0,143** |
| Ascaris | 1.530 [1.434–1.655] | 44 | 48 / 41 | 0,50 | 0,063 |
| H. diminuta | 2.463 [2.258–2.622] | 56 | 58 / 55 | 0,47 | 0,071 |
| Schistosoma | 3.353 [3.153–3.703] | 65 | **95** / 46 | 0,43 | 0,135 |

- **Cada classe de ovo tem tamanho muito estreito** (o IQR é ±5–10%), e as classes têm tamanhos bem diferentes entre si. Por isso o ovo ↔ ovo quase não erra: o tamanho sozinho já separa as espécies.
- **As impurezas cobrem toda a faixa de tamanho dos ovos** (IQR de 575 a 1.862 px). Uma impureza do tamanho de um Enterobius, Ancylostoma, H. nana ou Ascaris não se distingue pelo tamanho; a decisão depende de detalhe interno (casca, embrião, forma fina).
- O contraste ΔE e a textura média das impurezas (0,55 e 0,087) ficam no meio dos valores dos ovos. Pela média, as impurezas não se separam nem por contraste nem por textura.

## 3. Cruzamento: tamanho ou textura?

- **Entre ovos (os poucos erros ovo ↔ ovo), o tamanho manda:**
  - Spearman(confusão, distância de |log área|) = −0,43 (p = 0,024, 28 pares) na L3 e −0,29 (p = 0,14) na L4. Classes de tamanho parecido se confundem mais.
  - A textura (−0,25 e −0,20) e o ΔE (−0,20 e −0,16) não são significativos.
- **Ovo ↔ impureza, que é onde está o κ:**
  - Os ovos que mais vão para impureza são os **finos ou alongados** (Enterobius, 15%), os de **textura interna alta** (Ancylostoma, Lstd 0,143; Schistosoma, 0,135) e os **grandes e com espinho** (Schistosoma, eixo maior de 95 px, maior que o campo receptivo).
  - As impurezas que viram ovo caem sobretudo em Ascaris (33), H. nana (25) e Taenia (18): ovos **redondos e de textura lisa** (Lstd 0,063–0,082), onde a impureza redonda se parece com o ovo.
  - **Conclusão:** não é tamanho, porque as impurezas cobrem toda a faixa, e não é textura média. É **forma fina e detalhe interno** (casca, embrião, espinho) que a rede ainda não separa da impureza.

## 4. Hipóteses de construção (por ordem de prioridade)

1. **Resolução da L4 / detalhe fino** (a principal).
   - O erro restante está em casca fina, forma alongada e estrutura interna, que são detalhes de 2–4 px.
   - Na L4 (24×24), cada célula cobre ~8 px. A casca do Enterobius e o embrião do Ancylostoma ficam abaixo de uma célula.
   - **Teste:** tirar um pool (L4 a 49×49, pool só em L1–L2) e usar **dilation** em L3–L4 para manter o campo receptivo. É o eixo "resolução da L4 + dilation" da lista.
   - **Previsão:** o recall do Enterobius e do Ancylostoma sobe, e o erro impureza → ovo (Ascaris, H. nana) cai.
2. **Campo receptivo / contexto** (o e_d5/e_d6 e o k7 já estão na fila).
   - O Schistosoma (95 px, com espinho lateral) e as impurezas grandes passam do campo receptivo de L1–L3 (~30–45 px).
   - **O e_k3 perdeu −0,024**, porque o campo receptivo diminuiu. Isso é coerente com a hipótese.
   - **Previsão:** o k7 e o d5/d6 ajudam sobretudo o Schistosoma (recall 0,90) e o H. diminuta, e pouco o Enterobius.
3. **Sementes SPiFiL para a impureza** (sp25/sp100 estão na fila, mas variam de forma uniforme).
   - A impureza é **65% da validação** e muito heterogênea, mas recebe as mesmas sementes de qualquer classe (`one_per_class`).
   - Com poucos protótipos de impureza, a impureza parecida com ovo cai no protótipo do ovo: 137 falsos positivos, o maior bloco de erro na L4.
   - **Teste:** mais sementes ou imagens de **impureza** no SPiFiL (sementes por classe proporcionais à variabilidade, ou várias imagens de impureza). O sp100 uniforme provavelmente não basta; em larvae ele empatou.

**Consequência para a meta:** com 99% do erro em ovo ↔ impureza, eliminar ~metade dos 137 falsos positivos de impureza → ovo levaria o κ de eggs de ~0,90 para perto de 0,93. É um teto aproximado, só pela proporção de erros.
