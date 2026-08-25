# Como os estágios evoluem — currículo FLIM + crescimento SPiFiL

Retrato de 2026-08-22 22:57. Grade `grid4`, em `artifacts/spifil_growth/grid4/`.

> ## ⚠️ Tudo neste documento é VALIDAÇÃO, não teste
>
> Todo kappa aqui vem da **sonda SVM interna**, que roda a cada época durante o treino e
> pontua no conjunto de **validação**. Não é o avaliador oficial e não é o conjunto de teste.
>
> O próprio código avisa que os dois não são comparáveis (`src/evaluate/eval_autoencoder.py:59-63`):
> a sonda usa um `SVC` próprio (`src/modules/autoencoder_flim_module.py:485`), enquanto o
> avaliador oficial usa o `fit_svm` compartilhado (`src/utils/evaluate.py:372`) sobre o teste.
>
> **Não misture estes números com os de `docs/tabela_resultados.md`**, que são de teste. E não
> cite nenhum número daqui como resultado final de tese sem rodar a avaliação de teste antes.

Todo o grid rodou com `--embed-mode flatten`. Não há contraparte no braço de 48 dimensões.

---

## 1. O que é cada estágio

| Estágio | Encoder | Decoder | O que treina |
|---|---|---|---|
| **1** | FLIM congelado | aleatório | só o decoder |
| **2** | destravado | vem do est.1 | tudo |
| **3** (r1, r2, …) | ganhou uma camada SPiFiL nova; o resto congelado | ganhou um bloco novo aleatório; o resto congelado | **só a camada nova + o bloco novo** |
| **4** (r1, r2, …) | destravado | destravado | tudo |

Os estágios 1 e 2 rodam uma vez. Os estágios 3 e 4 se repetem a cada rodada de crescimento.

Duas armadilhas de leitura que vão te morder se você for aos dados crus:

**O campo `stage` mente.** Ele vale só 1/2/3 e registra o estágio 4 como 2, porque o estágio 4
é destravado e o código deriva o estágio das flags (`src/modules/autoencoder_flim_module.py:833`).
O rótulo verdadeiro está no **nome da run**: `spifil_growth_<dataset>_split<N>_pct<P>_<rótulo>`.

**O campo `best_val_svm_kappa` mente no estágio 3.** Ele é substituído pela linha de base
sempre que `--freeze-encoder` está ligado (`src/modules/autoencoder_flim_module.py:1172-1175`) —
e o estágio 3 usa essa flag mesmo treinando a camada nova. **Use `best_svm_kappa`.** Todos os
números deste documento usam `best_svm_kappa`.

## 2. O que o modelo é

Entrada `[B, 3, 200, 200]`, LAB em [0,1]. O encoder encolhe 200 → 99 → 49 → 24. A saída dele é
o gargalo, e é o que o SVM lê:

| Dataset | Classes | Canais antes de crescer | Depois de crescer | Embedding |
|---|---|---|---|---|
| eggs | 9 | `[3, 24, 32, 48]` | `[3, 24, 32, 48, 45]` | 48 → **45** |
| larvae | 2 | `[3, 24, 32, 48]` | `[3, 24, 32, 48, 48]` | 48 → **48** |
| protozoan | 7 | `[3, 24, **30**, 48]` | `[3, 24, 30, 48, 42]` | 48 → **42** |

A camada nova usa `--pool-stride 1`, então ela **não encolhe** o mapa: a grade fica em 24×24
para sempre. O que muda é só o número de canais.

O decoder reconstrói a própria imagem de entrada, `[B, 3, 200, 200]`, com `BCEWithLogitsLoss`.
**O decoder nunca vê o embedding** — ele recebe o mapa `[B, C, 24, 24]` inteiro. O embedding é
um caminho paralelo, só para a sonda e o avaliador.

## 3. O resultado — qual estágio é o melhor

| Braço | FLIM cru | est.1 | est.2 | est.3 r1 | est.4 r1 | est.3 r2 | est.4 r2 | Melhor | Foi até |
|---|---|---|---|---|---|---|---|---|---|
| `eggs_split1_pct5` | 0.5780 | 0.5780 | 0.6896 | 0.6100 | 0.6083 | – | – | **est.2** | est.4 r1 |
| `eggs_split1_pct50` | 0.8685 | 0.8685 | 0.9082 | 0.8853 | 0.8724 | – | – | **est.2** | est.4 r1 |
| `eggs_split2_pct5` | 0.6826 | 0.6826 | 0.6838 | 0.6129 | 0.6382 | – | – | **est.2** | est.4 r1 |
| `eggs_split2_pct50` | 0.8610 | 0.8610 | 0.9083 | 0.8765 | 0.8996 | – | – | **est.2** | est.4 r1 |
| `eggs_split3_pct5` | 0.5697 | 0.5697 | 0.6602 | 0.6083 | 0.6095 | – | – | **est.2** | est.4 r1 |
| `eggs_split3_pct50` | 0.8608 | 0.8608 | 0.9255 | 0.9046 | 0.9155 | – | – | **est.2** | est.4 r1 |
| `larvae_split1_pct5` | 0.4443 | 0.4491 | 0.4491 | 0.4943 | 0.4943 | 0.4996 | 0.4951 | **est.3 r2** | est.4 r2 |
| `larvae_split1_pct50` | 0.7982 | 0.7982 | 0.8567 | 0.8578 | 0.8781 | 0.7889 | 0.8664 | **est.4 r1** | est.4 r2 |
| `larvae_split2_pct5` | 0.4852 | 0.4852 | 0.5793 | 0.6022 | 0.6597 | 0.7129 | 0.7423 | **est.4 r2** | est.4 r2 |
| `larvae_split2_pct50` | 0.7835 | 0.7835 | 0.7969 | 0.7573 | 0.8455 | 0.8287 | 0.8420 | **est.4 r1** | est.4 r2 |
| `larvae_split3_pct5` | 0.6839 | 0.6839 | 0.6839 | 0.7187 | 0.7482 | 0.7595 | 0.7572 | **est.3 r2** | est.4 r2 |
| `larvae_split3_pct50` | 0.7966 | 0.8026 | 0.8763 | 0.8592 | 0.8592 | – | – | **est.2** | est.4 r1 |
| `protozoan_split1_pct5` | 0.5732 | 0.5737 | 0.6661 | 0.6245 | 0.6386 | – | – | **est.2** | est.4 r1 |
| `protozoan_split1_pct50` | 0.8412 | 0.8412 | 0.8719 | – | – | – | – | **est.2** | est.2 (est.3 r1 rodando) |
| `protozoan_split2_pct5` | 0.5927 | 0.5930 | 0.6526 | 0.6182 | 0.6207 | – | – | **est.2** | est.4 r1 |
| `protozoan_split2_pct50` | 0.7752 | 0.7765 | 0.8543 | – | – | – | – | **est.2** | est.2 (est.3 r1 rodando) |
| `protozoan_split3_pct5` | 0.6604 | 0.6604 | 0.7033 | 0.6651 | 0.6708 | – | – | **est.2** | est.4 r1 |
| `protozoan_split3_pct50` | 0.8184 | 0.8184 | 0.8732 | – | – | – | – | **est.2** | est.2 (est.3 r1 rodando) |

### Resumo

| Dataset | Melhor estágio | Foi até |
|---|---|---|
| **eggs** | **estágio 2**, nos 6 braços | estágio 4 rodada 1 |
| **protozoan** | **estágio 2**, nos 6 braços | estágio 4 rodada 1 (pct5); pct50 ainda no estágio 3 |
| **larvae** | pós-crescimento em 5 de 6 braços | estágio 4 rodada 2 |

Contando só os braços com estágio 4 fechado, e chamando de "ajudou" quando o estágio 4 termina
acima do estágio 2 por mais que a tolerância de 0,01:

| Dataset | Braços completos | Ajudou | Empatou | Atrapalhou |
|---|---|---|---|---|
| eggs | 6 | **0** | 2 | 4 |
| protozoan | 3 | **0** | 0 | 3 |
| larvae | 6 | **4** | 1 | 1 |
| **total** | **15** | **4** | 3 | 8 |

Larvae não chegou à rodada 3 porque o grid rodou com `--max-rounds 2`, não porque o laço decidiu
parar. Eggs e protozoan pararam sozinhos na rodada 1: o `should_stop`
(`scripts/spifil_growth_loop.py:160-165`) só continua se o estágio 4 bater o estágio 2 por mais
que 0,01, e não bateu em nenhum braço.

## 4. Onde o kappa cai, e por quê

A queda acontece **no instante do crescimento, antes de qualquer época de treino**. O número
que prova isso é o `flim_ref_svm_kappa`, medido em `on_fit_start`: para um estágio 3 ele é
literalmente "encoder do estágio 2 mais a camada nova crua".

| Braço | melhor est.2 | herdado no est.3 | custo |
|---|---|---|---|
| `eggs_split1_pct5` | 0.6896 | 0.6056 | **−0.0839** |
| `eggs_split1_pct50` | 0.9082 | 0.8586 | **−0.0496** |
| `eggs_split2_pct5` | 0.6838 | 0.5937 | **−0.0901** |
| `eggs_split2_pct50` | 0.9083 | 0.8556 | **−0.0527** |
| `eggs_split3_pct5` | 0.6602 | 0.6089 | **−0.0513** |
| `eggs_split3_pct50` | 0.9255 | 0.8434 | **−0.0821** |
| `larvae_split1_pct5` | 0.4491 | 0.4443 | **−0.0048** |
| `larvae_split1_pct50` | 0.8567 | 0.7910 | **−0.0657** |
| `larvae_split2_pct5` | 0.5793 | 0.5668 | **−0.0125** |
| `larvae_split2_pct50` | 0.7969 | 0.7361 | **−0.0608** |
| `larvae_split3_pct5` | 0.6839 | 0.6969 | **−-0.0130** |
| `larvae_split3_pct50` | 0.8763 | 0.7950 | **−0.0813** |
| `protozoan_split1_pct5` | 0.6661 | 0.6220 | **−0.0440** |
| `protozoan_split2_pct5` | 0.6526 | 0.6181 | **−0.0345** |
| `protozoan_split3_pct5` | 0.7033 | 0.6414 | **−0.0619** |

Duas coisas que **não** causam essa queda, ambas descartadas por prova direta:

- **Não é peso treinado se perdendo.** `conv1`, `conv2` e `conv3` do checkpoint do estágio 3 são
  bit-idênticos aos do estágio 2 (diferença 0,000e+00). O decoder treinado também sobrevive, com
  os índices deslocados por `delta` (`src/modules/autoencoder_flim_module.py:905-913`).
- **Não é o warmup do learning rate.** O estágio 3 tem exatamente o mesmo `warmup_epochs=10` e
  **sobe** na mesma janela em que o estágio 4 desce. Se fosse o warmup, os dois cairiam.

O que causa: o SVM deixa de ler o mapa da `conv3` e passa a ler o da `conv4`. **É outro espaço
de características.** Os filtros novos são recortes SPiFiL escolhidos por um critério não
supervisionado, que não sabe nada das classes, e eles recomprimem os canais sem saber quais
direções o SVM estava usando.

A queda no começo do **estágio 4** tem outra causa: é o encoder soltando. A correlação entre o
learning rate e a taxa de deriva dos kernels FLIM nas primeiras épocas é **+0,987**, e o fundo
do kappa cai dentro de duas épocas do pico da deriva.

## 5. A hipótese que explica os três datasets

O alocador do SPiFiL reparte os filtros igualmente entre as classes e descarta o resto:
`(out_channels // n_classes) × n_classes`. Com `out_channels = 48`:

| Dataset | Classes | Embedding depois | Encolhimento | Crescer ajudou? |
|---|---|---|---|---|
| **larvae** | 2 | 48 → 48 | **nenhum** | sim, 4 de 6 |
| **eggs** | 9 | 48 → 45 | −6,25% | 0 de 6 |
| **protozoan** | 7 | 48 → 42 | **−12,5%** | 0 de 3 |

Quanto mais a camada nova estrangula o embedding, pior o resultado. Larvae é o único dataset em
que a camada nova não perde canal nenhum — e é o único em que crescer funciona.

**Isto é hipótese, não prova.** São três pontos. O teste direto seria escolher `out_channels`
divisível pelo número de classes (63 para protozoan, 54 para eggs) e ver se a queda some.

## 6. As fragilidades — leia antes de citar qualquer número

**O melhor kappa costuma ser um pico que o treino não sustenta.** Nos 3 braços de larvae pct5 o
kappa pica nas primeiras 5 épocas e depois **desaba 0,10 a 0,13**. Em 12 das 34 runs de larvae o
melhor kappa está na época ≤ 2 — o treino daquele estágio não contribuiu nada. Se você reportar
o kappa da **última época** em vez do pico, o veredito de larvae inverte para 2 ajudou, 1
empatou, 3 atrapalhou, e o placar geral cai de 4 para **2 de 15**.

**A rodada 2 quase não rende.** Ganho médio da rodada 1: +0,052. Da rodada 2: +0,015, com
mediana +0,0009. O crescimento satura depois da primeira camada.

**O protocolo não é reprodutível o bastante para o efeito medido.** O `Trainer` roda com
`deterministic=False`, então o estágio 2 termina com um encoder ligeiramente diferente a cada
execução. O SPiFiL recorta os filtros das ativações desse encoder com um seletor guloso — uma
perturbação minúscula muda **quais** patches são escolhidos. Medido entre `grid3` e `grid4`, o
mesmo braço com o mesmo protocolo deu kappa diferindo em até **0,12**. Isso é maior que qualquer
efeito de crescimento que estamos tentando medir.

Para registro: o `--seed` do `spifil_grow` **existe e é aplicado**
(`scripts/spifil_grow.py:366-368`), e o pacote SPiFiL não contém nenhuma chamada de RNG. A
irreprodutibilidade não vem de falta de seed; vem do `deterministic=False` no treino anterior.

**Kappa em larvae é hipersensível.** São 2 classes com 87% de desbalanceamento. No pct5, uma
imagem a mais acertada vale **+0,0048** de kappa — a tolerância de 0,01 são duas imagens. A
variação entre splits sem treino nenhum é 0,24.

**As estatísticas de BatchNorm do decoder velho continuam se movendo no estágio 3.**
`requires_grad=False` desliga o gradiente, mas `running_mean` e `running_var` não são pesos
treináveis. Medido: 0 de 20 pesos se moveram, **18 de 18 estatísticas se moveram**. O decoder
velho não está 100% parado.

## 7. Conclusão

**O estágio 2 é o que entrega.** Descongelar o encoder FLIM e treinar tudo junto ganha do FLIM
cru em praticamente todos os braços dos três datasets.

**O crescimento, como está hoje, não paga o próprio custo.** Ele funciona num dataset de 2
classes onde a camada nova não estrangula o embedding, e mesmo lá o ganho vive dentro do ruído
de reprodução.

Se a ideia merecer mais uma tentativa antes de ser abandonada, a ordem que faz sentido é:

1. **`out_channels` divisível pelo número de classes** — testa a hipótese do estrangulamento
   diretamente, e é só um argumento de linha de comando.
2. **`deterministic=True` no Trainer** — sem isso não dá para separar efeito de ruído, e hoje o
   ruído é maior que o efeito.
3. **Reportar média de janela em vez do pico** — o pico está escolhendo estados que o treino
   não sustenta.

## 8. Onde estão os dados

| O quê | Onde |
|---|---|
| Checkpoints e metadados | `artifacts/spifil_growth/grid4/<braço>/<estágio>/` |
| Histórico por época, baixado do W&B | `wandb_phd_thesis_grid4/` (78 CSVs, um por run) |
| Projeto no W&B | `ophira-ai/phd_thesis_grid4` |
| Gráficos por estágio, sobrepostos | `artifacts/analysis/partial_train_spifil_hybrid/` |
| Gráficos em linha do tempo contínua | `artifacts/analysis_continuidade/` |
| Script dos gráficos sobrepostos, e o `fetch()` do W&B | `tools/plot_partial_train_spifil_hybrid.py` |
| Script da linha do tempo contínua | `tools/plot_continuity_spifil_hybrid.py` |
| Auditoria do currículo contra o código | `docs/auditoria_curriculo_spifil.md` |
| Resultados de **teste** (não confundir com este documento) | `docs/tabela_resultados.md` |

Para atualizar tudo com dados novos do W&B:

```bash
cd tools
python plot_partial_train_spifil_hybrid.py --fetch   # rebaixa e regenera os sobrepostos
python plot_continuity_spifil_hybrid.py              # regenera a linha do tempo contínua
```
