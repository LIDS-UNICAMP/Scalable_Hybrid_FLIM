# `statistics/tools/`: medições para a resposta ao Revisor 3 (SIBGRAPI camera-ready)

Quatro scripts independentes, cada um com um par de saídas (`.csv` com os números brutos,
`.md` com o relatório legível) no próprio diretório. Consolidação das quatro saídas em
[`reports_sibgrapi_camera_ready.md`](../../reports_sibgrapi_camera_ready.md) na raiz do repo.

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
| `wilcoxon_acc.csv` / `wilcoxon_acc.md` | Idem para acurácia. |
| `wilcoxon_kappa.csv` / `wilcoxon_kappa.md` | Idem para kappa. O CSV traz também `baseline_mean`/`model_mean` e `sign_convention` explícita. |

Os nomes de coluna dos três CSVs de Wilcoxon **não** são idênticos entre si (os scripts foram
escritos em paralelo): por exemplo o de F1 usa `method`/`label`, o de acurácia usa
`model`/`model_label` e o de kappa usa `model_method`/`model_label`. Os valores e as convenções de
sinal são os mesmos nos três.

---

## 4. Entrada e procedência

Os três testes de Wilcoxon leem `artifacts/normalized/unified_svm_comparison.csv`, montado por
`scripts/normalize_reports.py`. Cada célula pareada é a média sobre os 3 splits, e o pareamento é
feito pela chave `(dataset_short, pretrained_pct)`, com aborto se as chaves de dois modelos não
coincidirem.

Filtro obrigatório aplicado nos três scripts: `SVM_LeJEPA` aparece no CSV com cinco
inicializações (`flim`, `he`, `random`, `trunc_normal`, `xavier`), e somente `trunc_normal` entra
no artigo. Sem esse filtro o teste fica errado.

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
