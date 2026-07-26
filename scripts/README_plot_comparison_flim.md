# `plot_comparison_flim.py` — gráficos de comparação FLIM vs LeJEPA vs I-JEPA vs Distillation

Gera os gráficos **F1 / kappa / acurácia × fração de dados de pré-treino** (1%, 5%, 25%,
50%, 75%, 100%) para os três datasets (**Helminth Eggs**, **Helminth Larvae**,
**Protozoan Cysts**), comparando os **8 modelos oficiais** do experimento.

- Script: [`scripts/plot_comparison_flim.py`](plot_comparison_flim.py)
- Lógica de plotagem: [`src/evaluate/eval_plotter.py`](../src/evaluate/eval_plotter.py) → `plot_merge_comparison`, `plot_method_comparison`
- Saída: `artifacts/plots/plots_compare_to_flim/`

---

## 1. Ambiente / como rodar

O ambiente conda `scalable_FLIM` tem as dependências, mas o `matplotlib`/`PIL` quebra com
`GLIBCXX_3.4.29 not found` se o `LD_LIBRARY_PATH` não apontar para as libs do env. **Sempre**
prefixe o comando:

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
  scripts/plot_comparison_flim.py --mode row --metrics f1
```

---

## 2. Modos de plotagem (`--mode`)

| `--mode`     | O que gera | Saída |
|--------------|------------|-------|
| `merge`      | Painéis **lado a lado** (horizontal, 1 linha × 3 colunas). Eixo X repetido em cada painel; eixo Y só no painel da esquerda. **Este é o layout da figura do artigo.** | `merge_plots/{metric}.png` |
| `row`        | Painéis **empilhados na vertical** (3 linhas × 1 coluna). Eixo **Y repetido em cada linha**; eixo **X compartilhado**, mostrado só na base. Legenda contida abaixo dos painéis (2 colunas), nunca vaza. | `merge_plots/row/{metric}.png` |
| `individual` | Um PNG por (dataset × métrica). | `individual/…` |
| `all`        | `individual` + `merge` + `row` de uma vez. | todas as pastas acima |

Padrão: `--mode all`.

---

## 3. Métricas e datasets

```bash
--metrics f1 kappa acc      # padrão: as três; gera um arquivo por métrica
--datasets eggs larvae protozoan   # padrão: todos os presentes no CSV
```

---

## 4. Controle de fontes (aumentar / diminuir o texto)

Todos valem para `--mode merge` **e** `--mode row`. Aumente o número para fonte maior,
diminua para menor.

| Parâmetro              | Controla                                       | Padrão |
|------------------------|------------------------------------------------|:------:|
| `--subtitle-fontsize`  | título de cada dataset (acima do painel)       | 30 |
| `--ylabel-fontsize`    | label do eixo Y (`F1-score`); e o xlabel se usar `--show-xlabel` | 28 |
| `--tick-fontsize`      | porcentagens do eixo X (`1% … 100%`)           | 28 |
| `--ytick-fontsize`     | números do eixo Y                              | 24 |
| `--legend-fontsize`    | texto dos itens da legenda                     | 24 |

Exemplo (tudo menor):

```bash
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
  scripts/plot_comparison_flim.py --mode row --metrics f1 \
  --subtitle-fontsize 20 --ylabel-fontsize 18 \
  --tick-fontsize 16 --ytick-fontsize 16 --legend-fontsize 16
```

---

## 5. Legenda

| Parâmetro        | Efeito | Padrão |
|------------------|--------|:------:|
| `--legend-ncol`  | Nº de colunas da legenda. `-1` = tudo numa linha (horizontal). No modo `row` o padrão `-1` cai automaticamente para **2 colunas** (rótulos longos cabem, sem sobrepor). | -1 |
| `--legend-y`     | **Distância da legenda ao gráfico** (posição vertical em coordenadas da figura). **Negativo → afasta** (empurra pra baixo); **positivo → aproxima** (puxa pra cima). | 0.0 |

Notas sobre `--legend-y`:
- No modo **`merge`** (horizontal) é a posição absoluta logo abaixo da borda — costuma-se usar
  valores levemente negativos, ex.: `--legend-y -0.05`.
- No modo **`row`** (vertical) é um *offset* sobre a faixa que o script já reserva
  automaticamente embaixo dos painéis — ou seja, a legenda já vem separada por padrão e o
  `--legend-y` só faz o ajuste fino.

```bash
# afastar a legenda no modo row
... scripts/plot_comparison_flim.py --mode row --metrics f1 --legend-y -0.03
```

---

## 6. Tamanho da figura, linhas e marcadores

| Parâmetro                 | Efeito | Padrão |
|---------------------------|--------|:------:|
| `--fig-width-per-dataset` | largura (polegadas) por painel. No `row`, é a largura da figura toda. | 13 |
| `--fig-height`            | altura (polegadas). No `row` a altura total escala com o nº de datasets. | 10 |
| `--linewidth`             | espessura das linhas | 3.5 |
| `--markersize`            | tamanho dos marcadores | 14 |
| `--show-xlabel`           | mostra o label `Pre-training Data` no eixo X (padrão: oculto) | off |

---

## 7. Entrada / saída de arquivos

| Parâmetro | Padrão |
|-----------|--------|
| `--csv`   | `artifacts/normalized/unified_svm_comparison.csv` |
| `--out`   | `artifacts/plots/plots_compare_to_flim` |

O `--csv` é o **CSV unificado** montado por [`scripts/normalize_reports.py`](normalize_reports.py).
Se ele não existir, rode `python scripts/normalize_reports.py` primeiro.

---

## 8. Os 8 modelos oficiais e a procedência de cada curva

Apenas estas 8 chaves aparecem nas curvas/legenda (o filtro `_filter()` no script). Cada uma
vem de um CSV de origem diferente:

| Rótulo na legenda            | `method` (chave)                            | CSV de origem |
|------------------------------|---------------------------------------------|---------------|
| FLIM (59.504)                | `SVM_FLIM`                                   | `data/reports_felipe/svm/report_svm_*.csv` |
| LeJEPA (59.504)              | `SVM_LeJEPA` (init `trunc_normal`)           | `artifacts/SVM/*/*/metrics_SVM_*.csv` |
| I-JEPA (632M)                | `SVM_IJEPA`                                   | `results/ijepa_svm_aggregated.csv` |
| Distill 4 (889K)             | `SVM_Distill_Proj1280`                        | `results/svm_distill_proj1280_results.csv` |
| Distill 3 (615K)             | `SVM_Distill_3x3BN`                            | `results/svm_proj1280_3x3_BN2d_results.csv` |
| Distill 1 (123K)             | `SVM_Distill_1x1BN`                            | `results/svm_proj1280_1x1_BN2d_results.csv` |
| Distill 2 (402K)             | `SVM_Distill_2l400K`                           | `results/svm_proj1280_2l_1x1_BN2d_256_1280_results.csv` |
| Distill 1 — FLIM init (123K) | `SVM_Distill_1x1BN_flim_frozen_eval_loss`      | `results/svm_distillation_conv_flim_frozen_results.csv` |

> Como cada CSV de origem é gerado (treino → checkpoint → SVM eval → CSV):
> ver [`metrics_distillation/data_provenance.md`](../metrics_distillation/data_provenance.md).

Os pesos/checkpoints (em `logs/flim-ssl/*/checkpoints/`) **não** são necessários para
redesenhar os gráficos — só para regerar os CSVs de origem do zero. O plot lê apenas o CSV
unificado.

---

## 9. Receitas rápidas

```bash
# figura do artigo (horizontal, F1)
... scripts/plot_comparison_flim.py --mode merge --metrics f1

# versão vertical para coluna estreita (row), F1
... scripts/plot_comparison_flim.py --mode row --metrics f1

# vertical, fontes maiores e legenda mais afastada
... scripts/plot_comparison_flim.py --mode row --metrics f1 \
      --subtitle-fontsize 34 --ylabel-fontsize 30 --legend-fontsize 26 --legend-y -0.03

# tudo (individual + horizontal + vertical), as 3 métricas
... scripts/plot_comparison_flim.py --mode all
```

(`...` = o prefixo `LD_LIBRARY_PATH=... python` da seção 1.)
