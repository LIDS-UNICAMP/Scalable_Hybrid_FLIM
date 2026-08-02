# `statistics/tools/`: medições para a resposta ao Revisor 3 (SIBGRAPI camera-ready)

Seis scripts independentes, cada um com um par de saídas (`.csv` com os números brutos,
`.md` com o relatório legível) no próprio diretório. Consolidação das quatro primeiras saídas em
[`reports_sibgrapi_camera_ready.md`](../../reports_sibgrapi_camera_ready.md) na raiz do repo.

Os quatro primeiros scripts responderam ao Revisor 3 (um-contra-todos, baseline FLIM). Os dois
últimos são posteriores e têm desenhos próprios:

- `wilcoxon_flim_init.py` — **comparação controlada de um fator** que sustenta a tese *flyweight*:
  init FLIM congelada vs. init aleatória com a **mesma** cabeça de 123K.
- `wilcoxon_equivalence.py` — **par-a-par + equivalência (TOST)**. Cobre os 28 pares (os quatro
  primeiros scripts cobrem só 7) e, sobretudo, distingue *"não detectei diferença"* de
  *"demonstrei que a diferença é pequena"*. Necessário porque o texto do artigo faz afirmações de
  empate que a ausência de significância não sustenta.

> **Resultado que afeta a redação do artigo:** sobre as 18 células completas não há empate
> demonstrado em nenhum dos 28 pares. Mas descartando a fração de 1% (regime em que o FLIM
> colapsa), **FLIM e I-JEPA tornam-se equivalentes em F1 e acurácia dentro de ±0.024** — empate
> demonstrado, não apenas não-refutado. Toda afirmação de equivalência precisa declarar o regime
> de supervisão a que se aplica. Ver §3 de
> [`wilcoxon_equivalence.md`](wilcoxon_equivalence.md).

Os 8 modelos oficiais são os mesmos das figuras do artigo: FLIM (59.504), LeJEPA (59.504),
I-JEPA (632M), Distill 4 (889K), Distill 3 (615K), Distill 1 (123K), Distill 2 (402K) e
Distill 1 - FLIM init (123K). Mapeamento rótulo → chave `method` → CSV de origem na seção 8 de
[`scripts/README_plot_comparison_flim.md`](../../scripts/README_plot_comparison_flim.md).

---

## 1. Ambiente

O `matplotlib`/`PIL` do env conda `scalable_FLIM` quebra com `GLIBCXX_3.4.29 not found` se o
`LD_LIBRARY_PATH` não apontar para as libs do env. **Sempre** prefixe:

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib
PY=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python
```

O `statsmodels` (0.14.6) foi instalado no env `scalable_FLIM` para estas análises: é ele que
fornece `multipletests` (correção de Holm e Bonferroni). `scikit-posthocs` **não** é dependência
e não deve ser usado aqui: seu `posthoc_wilcoxon` é all-vs-all (28 comparações com 8 modelos),
enquanto o desenho pedido pelo revisor é um-contra-todos contra o FLIM (7 comparações).

Versões usadas: scipy 1.17.1, statsmodels 0.14.6, numpy 1.26.4, pandas 3.0.3, PyTorch 2.2.2,
CUDA 12.1.

---

## 2. Scripts

| Script | O que faz | Como rodar | O que produz |
|:--|:--|:--|:--|
| `measure_compute_cost.py` | Mede, para cada um dos 8 modelos, o grafo de inferência que produz o embedding entregue ao SVM linear: parâmetros, pesos em MB, FLOPs de um forward (batch=1, `FlopCounterMode`, FLOPs = 2×MACs), latência em ms/img em GPU e CPU (warmup + N repetições cronometradas) e pico de VRAM alocada, com e sem cuDNN. | `CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$LD_LIBRARY_PATH $PY statistics/tools/measure_compute_cost.py` | `compute_cost.csv`, `compute_cost.md` |
| `wilcoxon_f1.py` | Wilcoxon pareado de sinais na coluna `f1`, um-contra-todos, baseline `SVM_FLIM`, n = 18 células `(dataset, fração)`, p exato, Holm + Bonferroni, rank-biserial e IC95% bootstrap. Inclui tabela secundária exploratória por dataset (n = 6). | `$PY statistics/tools/wilcoxon_f1.py` | `wilcoxon_f1.csv`, `wilcoxon_f1.md` |
| `wilcoxon_acc.py` | Idem na coluna `acc`. | `$PY statistics/tools/wilcoxon_acc.py` | `wilcoxon_acc.csv`, `wilcoxon_acc.md` |
| `wilcoxon_kappa.py` | Idem na coluna `kappa` (kappa de Cohen). | `$PY statistics/tools/wilcoxon_kappa.py` | `wilcoxon_kappa.csv`, `wilcoxon_kappa.md` |
| `wilcoxon_equivalence.py` | **Desenho diferente:** todos os 28 pares entre os 8 modelos oficiais, nas 3 métricas. Cada par recebe **dois** testes — superioridade (Wilcoxon bilateral) e **equivalência (TOST)** com margem pré-especificada δ = 0.05, mais `delta_min` (menor margem em que a equivalência se sustenta). Duas famílias de correção: focal (5 pares que sustentam afirmações de empate no texto) e completa (28 pares). | `$PY statistics/tools/wilcoxon_equivalence.py` | `wilcoxon_equivalence.csv`, `wilcoxon_equivalence.md` |
| `wilcoxon_flim_init.py` | **Desenho diferente:** comparação controlada de um fator entre dois modelos de params idênticos (123.504 = backbone FLIM 59.504 + cabeça 1×1 BN2d 64.000), variando só a origem dos pesos do encoder — `SVM_Distill_1x1BN` (`trunc_normal`, treinável) vs. `SVM_Distill_1x1BN_flim_frozen_eval_loss` (FLIM real, congelado). Wilcoxon pareado nas 3 métricas (f1/kappa/acc), n = 18 células, Holm + Bonferroni sobre a família de 3. Secundárias: por dataset (n = 6), robustez do braço FLIM (ckpt knn vs. best-loss; encoder congelado vs. liberado) e descritiva por fração de rótulos. | `$PY statistics/tools/wilcoxon_flim_init.py` | `wilcoxon_flim_init.csv`, `wilcoxon_flim_init.md` |

Os três scripts de Wilcoxon aceitam `--csv` (entrada, padrão
`artifacts/normalized/unified_svm_comparison.csv`), `--baseline` (padrão `SVM_FLIM`) e `--alpha`
(padrão 0.05). O `measure_compute_cost.py` aceita `--skip-cpu` para omitir a latência de CPU, que
é a parte lenta da medição.

Cada script sobrescreve o próprio par de saídas. Rodar os três de Wilcoxon em sequência:

```bash
for M in f1 acc kappa; do $PY statistics/tools/wilcoxon_${M}.py; done
```

---

## 3. Arquivos de resultado

| Arquivo | Conteúdo |
|:--|:--|
| `compute_cost.csv` | Uma linha por modelo. Colunas além das da tabela do `.md`: `macs` (FLOPs/2, para comparar com trabalhos que reportam MACs), `gpu_ms_median` e `cpu_ms_median` (mais robustas à contenção de CPU do nó), `peak_vram_cudnn_disabled_MB` (pico com `torch.backends.cudnn.enabled = False`), `static_vram_MB`, `input_shape`, `pesos_reais` e `fonte_dos_pesos`. |
| `compute_cost.md` | Tabela principal, metodologia (hardware, resolução por modelo, repetições, convenção de FLOPs, definição das duas grandezas de memória), grafo exato medido por linha e ressalvas. |
| `wilcoxon_f1.csv` | Uma linha por comparação (7). Medianas e médias das diferenças com IC95%, contagem de vitórias, W/W+/W−, p bruto, p Holm, p Bonferroni, rank-biserial, flags de significância e conclusão. |
| `wilcoxon_f1.md` | Tabela principal, tabela de sinais e médias, tabela secundária por dataset, leitura dos resultados e comando de reprodução. Contém a nota que explica as quatro linhas com W = 51 e p bruto idêntico. |
| `wilcoxon_equivalence.csv` | Uma linha por (família, métrica, par): 15 focais + 15 focais sem a fração de 1% (`familia = focal_sem_1pct`, coluna `pcts`) + 84 da matriz completa. Traz `p_sup_*` e `p_tost_*` (bruto/Holm/Bonferroni), `delta`, `delta_min`, `veredito` (4 categorias), `afirmacao_sustentada` nos pares focais e `sign_convention` explícita. |
| `wilcoxon_equivalence.md` | Desenho e tabela 2×2 dos vereditos, justificativa da margem δ (com o piso de ruído entre splits), tabela da família focal por métrica, **decomposição por fração de rótulos (§1b)**, **família focal sem a fração de 1% (§1c)**, **consistência por dataset (§1d)**, matriz completa dos 28 pares, **§3 "o que isso muda no texto do artigo"** com redação sugerida por afirmação, e ressalvas. |
| `wilcoxon_flim_init.csv` | Uma linha por (análise, métrica, escopo): 3 da tabela principal + 3 por dataset + 2 de robustez + 1 colapsada (n = 3). Colunas `analise` (`principal`/`por_dataset`/`robustez`/`colapsado_n3`), `escopo`, `baseline_params`/`model_params` e `sign_convention` explícita. `p_holm`/`p_bonferroni` ficam vazios fora da família principal, de propósito. |
| `wilcoxon_flim_init.md` | Tabela principal nas 3 métricas, medianas por braço, F1 médio por fração de rótulos, tabela por dataset, robustez do braço FLIM, leitura dos resultados, seção "o que este teste NÃO diz", derivação do `p` (2/2¹⁸, piso do teste exato) e ressalvas — incluindo a **não-independência** das 18 células. |
| `wilcoxon_acc.csv` / `wilcoxon_acc.md` | Idem para acurácia. |
| `wilcoxon_kappa.csv` / `wilcoxon_kappa.md` | Idem para kappa. O CSV traz também `baseline_mean`/`model_mean` e `sign_convention` explícita. |

Os nomes de coluna dos três CSVs de Wilcoxon **não** são idênticos entre si (os scripts foram
escritos em paralelo): por exemplo o de F1 usa `method`/`label`, o de acurácia usa
`model`/`model_label` e o de kappa usa `model_method`/`model_label`. Os valores e as convenções de
sinal são os mesmos nos três.

---

## 4. Entrada e procedência

Os quatro testes de Wilcoxon leem `artifacts/normalized/unified_svm_comparison.csv`, montado por
`scripts/normalize_reports.py`. Cada célula pareada é a média sobre os 3 splits, e o pareamento é
feito pela chave `(dataset_short, pretrained_pct)`, com aborto se as chaves de dois modelos não
coincidirem.

Filtro obrigatório aplicado nos três scripts um-contra-todos: `SVM_lejepa_view` (ex-`SVM_LeJEPA`) aparece no CSV com
cinco inicializações (`flim`, `he`, `random`, `trunc_normal`, `xavier`), e somente `trunc_normal`
entra no artigo. Sem esse filtro o teste fica errado. O `wilcoxon_flim_init.py` não usa o LeJEPA,
mas aplica o mesmo tipo de filtro em `SVM_Distill_1x1BN` (`init == "trunc_normal"`).

**Nomenclatura de `pretrained_pct`:** essa coluna é o **% de dados rotulados usados para treinar o
SVM linear**, não uma fração de pré-treino (o teacher I-JEPA é frozen) — seção 3 de
[`data_provenance.md`](../../metrics_distillation/data_provenance.md). Os relatórios
`wilcoxon_f1.md`, `wilcoxon_acc.md` e `wilcoxon_kappa.md` chamam esse eixo de "pré-treino", o que
é impreciso; `wilcoxon_flim_init.md` usa o termo correto. Consequência prática para redação: os
valores de F1 0,42 → 0,83 dessa comparação são de **100% dos rótulos**, não de 5% (a 5% são
0,25 → 0,55).

Como cada CSV de origem foi gerado (treino → checkpoint → SVM eval → CSV):
[`metrics_distillation/data_provenance.md`](../../metrics_distillation/data_provenance.md).

---

## 5. Ressalvas que precisam acompanhar qualquer citação destes números

- Os `.ckpt` das runs Distill 1, 2, 3 e 4 não estão mais no disco. Essas quatro linhas de
  `compute_cost.csv` foram medidas sobre a arquitetura reinstanciada (coluna `pesos_reais` =
  `False`). FLOPs, latência, memória e contagem de parâmetros dependem só da topologia, portanto
  são idênticos aos que os checkpoints dariam, mas isso precisa estar declarado.
- O pico de VRAM das cabeças de projeção 1×1 → 1280 está dominado por workspace da cuDNN
  (~132 MB), não por ativação; com cuDNN desabilitada cai para 25 a 28 MB. A coluna de **pesos** é
  a grandeza estável.
- Resoluções de entrada diferem: 3×200×200 para FLIM/LeJEPA/Distill, 3×224×224 para o I-JEPA.
- Ausência de significância com n = 18 e família de 7 hipóteses é inconclusividade, não
  equivalência.
- Distill 4 (889K) inverte o sinal da mediana entre F1/acurácia e kappa. A causa não está
  resolvida. Ver seção 3.3 de
  [`reports_sibgrapi_camera_ready.md`](../../reports_sibgrapi_camera_ready.md).
