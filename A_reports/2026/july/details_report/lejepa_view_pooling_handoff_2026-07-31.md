# Handoff — LeJEPA com pooling + curva `lejepa_view`

**Data:** 2026-07-31 · **git base:** `06b7b1f` · **Estado:** trabalho **parado no meio**, nada commitado
**Relatório relacionado:** [`why_svm_beats_mlp_2026-07-29.md`](why_svm_beats_mlp_2026-07-29.md)

Documento de retomada. Registra o que foi feito, o que **não** foi feito, o bloqueio que
impede terminar, e as duas decisões pendentes.

---

## 0. TL;DR para quem retoma

1. **A1 e A2 estão prontos** (código alterado, verificado, não commitado): a extração agora é
   `AdaptiveAvgPool2d(1)` + flatten (**48-d**, não 27.648-d), e a curva foi renomeada para
   `lejepa_view`.
2. **B1 e B2 não rodaram.** Motivo: **os checkpoints do baseline LeJEPA não existem mais no
   disco.** Nenhuma das 54 células canônicas tem peso local.
3. **Há uma armadilha ativa:** se você rodar `scripts/normalize_reports.py` agora, vai gerar
   uma curva **rotulada `lejepa_view`** (que sugere pooling) mas **preenchida com números
   antigos do flatten 27.648-d**. Ver §5 — é o item mais perigoso deste documento.
4. Um subagente reverteu o refresh do cache do W&B. Precisa refazer (§6).

---

## 1. A pergunta que originou isto

O LeJEPA foi avaliado no SVM com **flatten puro, sem pooling**. A representação correta é
**pooling seguido de flatten**. O pedido: trocar a extração, apagar o caminho sem pooling,
reavaliar os 3 experimentos e renomear a curva para `lejepa_view`.

O contexto de por que isso importa está em [`why_svm_beats_mlp_2026-07-29.md`](why_svm_beats_mlp_2026-07-29.md):
o repo tinha **duas convenções de feature convivendo**, e a figura de 8 modelos do artigo
mistura as duas.

---

## 2. Mapa das convenções (medido, entrada 200×200)

Cadeia do encoder FLIM/LeJEPA, verificada instanciando `build_encoder_from_arch` com o
`architecture.json` real de `eggs/train1`:

```
entrada  (1, 3, 200, 200)
conv1    (1, 24,  99,  99)     Conv2d(5x5) + ReLU + MaxPool(3x3, s2)
conv2    (1, 32,  49,  49)
conv3    (1, 48,  24,  24)  <- feat_map final, ESPACIAL
```

A partir daí, quem consome o quê:

| Consumidor | Convenção | Dimensão (antes) |
|:--|:--|--:|
| `train_svm` / `extract_features` (`src/utils/evaluate.py`) | **flatten direto** | **27.648** |
| `MLPHead` / `TwoLayerSigmoidHead` (`models.py:324,375`) | pool + flatten | 48 |
| `svm_classification_flim.py` (`_EncoderProbe`) | pool + flatten | 48 |
| Todas as proj heads de destilação | pool interno + flatten | 1.280 |
| I-JEPA (`ijepa_encoder.py:194`, `x.mean(dim=1)`) | pool | 1.280 |

**O pooling mora nas cabeças, não no encoder.** Por isso `train_svm`, que entra pelo encoder
por baixo (`model.conv1/conv2/conv3` direto, sem chamar `forward`), escapava do pooling.

### Quem usava o caminho flatten — 5 call-sites, não 1

| Call-site | Família | Dimensão antes → depois |
|:--|:--|:--|
| `src/evaluate/unified_eval.py:207,222` | LeJEPA (alvo) | 27.648 → 48 |
| `src/evaluate/svm.py:121,134` | LeJEPA | 27.648 → 48 |
| `src/utils/evaluate.py:404,427` (`main()`) | LeJEPA | 27.648 → 48 |
| `src/evaluate/svm_flim_residual.py:109,122` | FLIM residual, **não LeJEPA** | 41.472 → 72 / 46.080 → 80 |
| `src/evaluate/classical_classifiers.py:190,206` | kNN/QDA/RF/LightGBM/GP, **nem SVM nem LeJEPA** | 27.648 → 48 |

Consequência: **kNN, QDA, Random Forest, LightGBM e Gaussian Process rodaram todos sobre
27.648 dimensões.** Para kNN e QDA em particular isso é um regime completamente diferente
de 48-d (distância euclidiana e estimativa de covariância em alta dimensão com ~2,5k amostras).

---

## 3. O que FOI feito (working tree, não commitado)

`git diff --stat` sobre `06b7b1f` — 11 arquivos, +68/−34:

```
metrics_distillation/data_provenance.md |  9 +++++---
scripts/README_plot_comparison_flim.md  |  2 +-
scripts/normalize_reports.py            | 11 ++++++---
scripts/plot_comparison_flim.py         | 10 ++++-----
scripts/plot_parameters_vs_metrics.py   |  2 +-
src/evaluate/eval_plotter.py            | 14 ++++++++----
src/utils/evaluate.py                   | 40 ++++++++++++++++++++++++---------
statistics/tools/README.md              |  2 +-
statistics/tools/wilcoxon_acc.py        |  4 ++--
statistics/tools/wilcoxon_f1.py         |  4 ++--
statistics/tools/wilcoxon_kappa.py      |  4 ++--
```

### A1 — extração passa a ser pooling + flatten

Arquivo único: [`src/utils/evaluate.py`](../../../../src/utils/evaluate.py)

- **linha 45** — `import torch.nn as nn`.
- **linhas 233–248** — `_POOL = nn.AdaptiveAvgPool2d(1)` e helper `_encode_pooled(model, inputs)`:
  conv1→conv2→conv3, aplica `_POOL`, devolve `.flatten(start_dim=1).detach().cpu()`.
  Sem hard-code de 48 — a dimensão sai do `channels[-1]` do encoder.
- **linha 285** — `train_svm` usa `_encode_pooled`.
- **linha 300** — `clf.fit(all_feats, all_y)`; o `.flatten(start_dim=1)` residual foi removido.
- **linha 356** — `extract_features` usa `_encode_pooled`.
- docstrings atualizados (`:255-259`, `:333-345`).

**O caminho sem pooling foi apagado, não desativado** — não há flag, parâmetro opcional nem
branch morto.

Verificação executada (saída real):

```
channels from weights : [3, 24, 32, 48]
conv1 (2, 24, 99, 99) conv2 (2, 32, 49, 49) conv3 (2, 48, 24, 24)
naive flatten dim     : 27648
extract_features -> (6, 48) labels (6,)
train_svm n_features_in_ : 48
OK: feature dim = 48 (expected 48, not 27648)
```

### A2 — curva renomeada para `lejepa_view`

Decisão de nomenclatura: `method` interno = **`SVM_lejepa_view`** (preserva o prefixo `SVM_`
das outras 7 curvas); rótulo visível = **`lejepa_view`**; chave de estilo =
`lejepa_view_trunc_normal`.

Os 3 pontos de núcleo:

1. **Produtor da chave** — `scripts/normalize_reports.py:175` (+ docstring `:158-163`, print `:194`).
2. **Rótulo de legenda** — `scripts/plot_comparison_flim.py:99` e `src/evaluate/eval_plotter.py:405`;
   estilo em `eval_plotter.py:388`.
3. **Filtro das 8 chaves oficiais** — `scripts/plot_comparison_flim.py:224`; roteamento em
   `eval_plotter.py:472-474` e `:656-658` (`_line_key`, duas cópias).

Consumidores atualizados para não quebrar: `scripts/plot_parameters_vs_metrics.py:92`,
`statistics/tools/wilcoxon_{f1,acc,kappa}.py`, e as docs
(`README_plot_comparison_flim.md:128`, `data_provenance.md:42,63,80`,
`statistics/tools/README.md:90`).

**Não** foram alterados, de propósito: `scripts/plot_svm_vs_mlp_pct.py` e o fallback
`f"SVM_LeJEPA_{init}"` em `eval_plotter.py:475,659` — aquele script é outro gráfico
(FLIM+SVM vs FLIM+MLP) e reusa esse fallback com `init=frozen/unfrozen` para curvas **MLP**.

> **Sobre "os 3 experimentos":** não existem 3 edições. O nome vive num **único lugar
> compartilhado** pelos 3 painéis — uma edição cobre eggs, larvae e protozoan. Comprovado:
> o `_filter()` retorna 18 linhas = 3 datasets × 6 pcts.

---

## 4. O BLOQUEIO — os checkpoints do LeJEPA sumiram

Este é o motivo de B1 e B2 não terem rodado.

- As **54 células canônicas** `lejepa_line_<ds>_split_N_pct_P_model_trunc_normal` existem no
  W&B. **Zero têm checkpoint local.**
- Restaram **29 diretórios** em `logs/flim-ssl/` (80 `.ckpt`), dos quais 21 identificáveis —
  e são **variantes**, não o baseline:

| Grupo de runs | Datasets | Cobertura real |
|:--|:--|:--|
| `trunc_normal_bs256_mc2g8l` | **eggs, larvae, protozoan** | 3 splits, **só pct=100** |
| `trunc_normal_bs150_mc8g8l_3lproj` | só eggs | split 1, **todos os 6 pcts** (+ split2 pct75) |
| `trunc_normal_bs256` | só eggs | pct=100 |
| 8 runs | — | não resolvem nem com o cache atualizado |

- Os checkpoints **não carregam identidade**: `hyper_parameters` só tem
  `{'encoder_init': 'trunc_normal'}`, sem dataset/split/pct. Não há `.yaml`/`.json`/`wandb`
  nos run dirs. A identificação **depende inteiramente do cache do W&B**.
- Os 90 CSVs em `artifacts/SVM/` foram gerados quando existiam ~297 run dirs (ver `report.md`
  §13.3). Esses pesos se foram.

**Conclusão:** a curva LeJEPA do artigo **não é reproduzível com pooling** sem re-treinar o
SSL. O que dá para fazer com o que existe:

- **Recorte A** — 3 datasets × 3 splits, **pct=100 apenas** (9 ckpts, `bs256_mc2g8l`).
  Atende "3 experimentos", mas o gráfico vira um ponto por dataset, não uma curva.
- **Recorte B** — **eggs apenas**, curva completa 1/5/25/50/75/100 (split 1, `bs150_3lproj`).
  Único recorte com eixo x de verdade.
- **Recorte C** — re-treinar as 54 células (`scripts/run_ssl_ray.py`). Dias de GPU.

Em qualquer caso, são **variantes** (`bs256_mc2g8l`, `bs150_3lproj`), não o baseline do
artigo — qualquer número daí precisa vir rotulado como tal.

---

## 5. ARMADILHA — não rode `normalize_reports.py` sem ler isto

`artifacts/normalized/unified_svm_comparison.csv` (de 22/jul) ainda tem `SVM_LeJEPA` gravado.
Com o rename do A2, o `_filter()` retorna **0 linhas** de LeJEPA nesse CSV — ou seja, **a
curva sumiria do gráfico** até o CSV ser regenerado.

**Mas regenerar agora produz um resultado enganoso.** O `normalize_reports.py` lê
`artifacts/SVM/*/*/metrics_SVM_*.csv`, que foram gerados **pelo caminho flatten 27.648-d**.
O resultado seria uma curva **rotulada `lejepa_view`** — nome que promete pooling — contendo
**números do flatten**. Pior que a situação atual, porque o erro fica escondido atrás de um
nome correto.

**Ordem correta:** primeiro re-rodar a avaliação SVM (gerando `metrics_SVM_*.csv` novos com
48-d), **depois** `normalize_reports.py`, **depois** o plot. Enquanto os `metrics_SVM_*.csv`
forem os antigos, o rename é só cosmético.

Pelo mesmo motivo, `statistics/tools/wilcoxon_*.{csv,md}` já gerados continuam com o rótulo
e os números antigos até serem re-executados.

### CSVs que ficaram inconsistentes com o código novo

Gerados pelo caminho flatten, **não reproduzíveis** rodando o código atual:

- `results/svm_results.csv`
- `results/classical_classifiers_results.csv`
- `artifacts/SVM/*/metrics_SVM_*.csv`
- o CSV do FLIM residual escrito por `run_flim_residual_assessment()`

Qualquer comparação que misture números velhos com novos está comparando **embeddings
diferentes**.

---

## 6. Incidente — cache do W&B revertido

Durante a execução, um subagente rodou `git checkout -- configs/wandb_update/ids_wandb.json`
sem instrução para isso, **descartando o refresh do cache** que eu tinha acabado de fazer
(2160 → 2587 runs). Estado atual confirmado: 2160 runs, `updated_at = 2026-06-01`, e o run
`1rfba2ff` (que tem checkpoint local) **não está no cache**.

Sem esse refresh **não dá para identificar nenhum dos 29 checkpoints locais**. Refazer com:

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.utils.wandb_cache --update
```

Leva ~10 s e o W&B está acessível (credenciais em `~/.netrc`).

---

## 7. Decisões pendentes (é aqui que você retoma)

**Decisão 1 — escopo de B1/B2.** Recorte A (3 datasets, pct=100), recorte B (curva de eggs),
os dois, ou re-treinar. Ver §4. Minha sugestão: **os dois recortes**, cada um rotulado com
sua variante — extrai o máximo do que existe sem fingir que é o baseline.

**Decisão 2 — alcance do A1.** Hoje a mudança atinge os **5** call-sites (§2), incluindo
`classical_classifiers.py` e `svm_flim_residual.py`, que não são LeJEPA. Isso é coerente com
"a representação correta é pooling", mas invalida `results/classical_classifiers_results.csv`
e o CSV do residual. A alternativa (isolar só o LeJEPA) exigiria **manter os dois caminhos
vivos**, o que contradiz "não deixe branch morto". Como está: atinge todos.

---

## 8. O que NÃO foi feito

- **B1** — avaliação SVM nos 3 experimentos com `test_accuracy` / `test_cohen_kappa`. Bloqueado por §4.
- **B2** — curvas em `new_hope_jepa/` (accuracy, f1, kappa). Depende de B1. A pasta **não foi criada**.
- Nenhum commit. Nenhuma avaliação executada. Nenhum plot gerado.
- Resíduo de documentação não corrigido: `src/evaluate/svm_distillation.py:32,179` ainda
  descrevem o pipeline FLIM/SVM como acessando "`.conv1/.conv2/.conv3` spatial" para
  contrastar com o caminho destilado — comentário que agora descreve comportamento inexistente.

---

## 9. Sequência para retomar

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM

# 1. refazer o cache do W&B (senao os ckpts locais nao sao identificaveis)
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m src.utils.wandb_cache --update

# 2. decidir o escopo (secao 7) e rodar a avaliacao SVM -> gera metrics_SVM_*.csv NOVOS (48-d)
#    (o resolver so enxerga runs que estejam no cache E com ckpt local)

# 3. so DEPOIS de (2): regenerar o CSV unificado com o nome novo
python scripts/normalize_reports.py

# 4. plotar as 3 metricas para new_hope_jepa/
LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib \
/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python \
  scripts/plot_comparison_flim.py --mode merge --metrics acc f1 kappa \
  --out new_hope_jepa
```

O passo 3 antes do passo 2 é exatamente a armadilha da §5.

---

## 10. Fontes

| Item | Onde |
|:--|:--|
| Formas dos tensores | medido via `build_encoder_from_arch` + `architecture.json` de `eggs/train1` |
| Cobertura de checkpoints | `logs/flim-ssl/` (29 dirs, 80 ckpt) × `configs/wandb_update/ids_wandb.json` |
| Convenções de feature | `src/utils/evaluate.py`, `src/models/models.py:324,375`, `src/evaluate/svm_distillation.py:179-181` |
| Contexto do problema | [`why_svm_beats_mlp_2026-07-29.md`](why_svm_beats_mlp_2026-07-29.md) |
| Diff atual | `git diff` sobre `06b7b1f` (11 arquivos, não commitado) |
