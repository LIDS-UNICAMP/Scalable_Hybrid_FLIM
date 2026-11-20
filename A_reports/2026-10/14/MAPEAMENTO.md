# MAPEAMENTO — genes do GA × SPiFiL × datasets

Passo 0 (seção 12). Gerado em 2026-10-12 a partir da inspeção read-only (onda W0).

- SPiFiL = `/dados/home/moliveira/SPiFiL/src/spifil` (spifil 1.0.0, editable no env `scalable_FLIM`)
- SHF = `/dados/home/moliveira/Scalable_Hybrid_FLIM` (o adapter `flim/spifil.py` já estende o SPiFiL por herança/composição)

Legenda de status: `existe` (chave/arg do SPiFiL), `existe_fixo` (existe, mas sem chave para variar),
`via_heranca` (subclasse/composição, sem reescrever o núcleo), `unsupported`.

---

## 1. Genes

### 1.1 Genes de 4.1

| gene | status | onde | como expor | observação |
|---|---|---|---|---|
| `dilation` | existe | `config.py:22` `LayerSpec.dilation` (int ≥1, checado em `patches.py:72`); usado em `nn/blocks.py:166` e `patches.py:81` | `arch.layers[i].dilation` | — |
| `pool_type` | existe | `LayerSpec.pool_type` ∈ {max,avg,none}, validado em `nn/blocks.py:100` | `arch.layers[i].pool_type` | — |
| `pool_size` | existe | `LayerSpec.pool_size` (int=3); janela real `2*(size//2)+1` (`blocks.py:181`) | `arch.layers[i].pool_size` | **2 e 3 dão a mesma janela (3)**. Domínio {2,3,5} vira na prática {3,5} |
| `pool_stride` | existe | `LayerSpec.pool_stride` (int=2); ignorado se `pool_type=none` (`learner.py:587`) | `arch.layers[i].pool_stride` | sementes projetadas pelo stride |
| `relu` | existe | `LayerSpec.relu: bool` (`blocks.py:157`) | `arch.layers[i].relu` | substituído por `activation` (4.2) |
| `stdev_factor` | existe | `ArchSpec.stdev_factor=0.01` (`config.py:46`), global | `arch.stdev_factor` | gene em log10 → `10**x` |
| `superpixels` | existe | grupo Hydra `superpixels`: `SLIC` (`superpixels/slic.py:19`), `DISF` (`superpixels/disf.py:50`, via pyift) | `superpixels=slic\|disf` | — |
| `compactness` | existe | `SLIC(compactness=10.0)` | `superpixels.compactness` | sem validação de faixa |
| `sigma` | existe | `SLIC(sigma=0.0)` | `superpixels.sigma` | — |
| `adjacency_radius` | existe | `DISF(adjacency_radius=1.0)` | `superpixels.adjacency_radius` | docstring: >1 vaza pelas bordas |
| `n_init_seeds` | existe | `DISF(n_init_seeds=None)`; `None` → `10*n_superpixels` (`disf.py:100`) | `superpixels.n_init_seeds` | `auto` ↦ `null` |
| `n_superpixels` | existe | `SpifilConfig.n_superpixels=100`, `Learner(n_superpixels=)` | raiz | só na camada 0 (ver 1.2) |
| `seed_extractor` | existe | `Medoids` / `Centroids` / `GeodesicCenters` (`superpixels/centers.py`) | `seed_extractor=...` | `geodesic` precisa do pyift |
| `metric` | existe | `metrics.py:50,65` | `metric=mahalanobis\|euclidean` | — |
| `scorer` | existe | `scoring.py:40,49` | `scorer=fisher\|distance_sum` | — |
| `selector` | existe | `DiversitySelector`, `TopN` (`selection.py`) | `selector=diversity\|topn` | — |
| `alpha` | existe | `DiversitySelector(alpha=0.5)` | `selector.alpha` | — |
| `pool_factor` | existe | `DiversitySelector(pool_factor=3)` | `selector.pool_factor` | — |
| `one_per_class` | via_heranca | o SPiFiL usa todo o `SpifilDataset` recebido; SHF `flim/spifil.py:428` `train_samples(one_per_class=)` | montar o `SpifilDataset` com o subconjunto (composição) | — |
| `n_images` | via_heranca | mesmo mecanismo, passo uniforme (`flim/spifil.py:454`) | idem | limitado pelo nº de imagens da pct |
| `paper_classes` | via_heranca | SHF `flim/spifil.py:465-467` lê `scripts/spifil_paper_seeds.json` | composição com `train_samples` | **os ids do JSON estão na numeração ANTIGA** — ver §2.3 |
| `impurities` | via_heranca | SHF `Cropped(SpifilDataset)` (`flim/spifil.py:315`) sobrescreve `load_mask()`; sem máscara, o SLIC/DISF segmenta o quadro todo | `use_mask` no dataset | só em datasets com máscara |
| `spifil_in` | via_heranca | `image` é nativo (superpixel na camada 0, sementes projetadas, `learner.py:480`). `features`: SHF `seeds_in_features` / `build_layer0` (`flim/spifil.py:493,572`) injeta `learn.state[0]` | `image`: nada. `features`: crescer uma camada por vez, recalculando o `state[0]` sobre a saída do encoder já treinado | ver nota A |

### 1.2 Variáveis de 4.2

A arquitetura do SPiFiL já é uma lista `ArchSpec.layers: list[LayerSpec]` (`config.py:21-46`, `nn/model.py:67`).
O aprendizado é sequencial, camada a camada (`strategies.py:38-41`, `learner.py:426`).
**Os modos `comum`, `crescimento` e `atomico` se reduzem a gerar essa lista**, sem herança.

| variável | status | onde | como expor |
|---|---|---|---|
| `r` (resize) | via_heranca | o SPiFiL não redimensiona. O SHF tem `Cropped(samples, image_size)` / `Resized` (`flim/spifil.py:318,870`) | dataset que redimensiona antes do `Learner` (composição) |
| `k` | existe | `LayerSpec.kernel_size`; k par vira `2*(k//2)+1` (`builder.py:193`) | `arch.layers[i].kernel_size` |
| `w` (N) | existe | `LayerSpec.out_channels`; `UniformAllocator` dá `N // n_classes` por classe e descarta o resto (`allocation.py:52,94`) | `arch.layers[i].out_channels` |
| `d` | existe | `len(arch.layers)`, sem máximo (≥1, `learner.py:214`) | tamanho da lista ([1,7] ok) |
| `activation` | via_heranca | só `relu: bool` (`blocks.py:160`) | subclasse de `SpifilConvBlock` sobrescrevendo `forward`, injetada por `Learner(block_factory=)` (`learner.py:208`). Recarregar exige o mesmo `block_factory` |
| `pool_type` (entre camadas) | existe | por camada | idem 4.1 |
| `head` | via_heranca | o SPiFiL devolve o mapa (B,C,H,W) (`nn/model.py:94`) | redução fora do SPiFiL em `evaluate.py`: flatten / mean / max sobre (H,W) |
| `layer_mode` | n/a | não existe; é só o gerador da lista | `space.py` |
| `width_growth`, `kernel_growth`, `dilation_growth`, `pool_schedule` | existe | geram a lista de `LayerSpec` | `space.py` |
| `superpixel_growth`, `n_superpixels_l` | via_heranca | o SPiFiL só calcula superpixels na camada 0 | só com `spifil_in=features`: o laço "uma camada por vez" da nota A recebe um `n_superpixels` diferente em cada camada |
| atômicos `k_l`, `N_l`, `dilation_l`, `pool_*_l` | existe | um `LayerSpec` por camada | lista |
| `activation_l` | via_heranca | mesmo `block_factory` | campo por camada lido pela subclasse |

**Nota A — `spifil_in=features` em rede multicamada (decidido).** O SPiFiL faz `fit` de todas as camadas com as
sementes da camada 0 projetadas. O gancho de composição é o evento `after_encode`, que o `Learner._fit_layer`
dispara logo depois de criar `state[l]` e ANTES de `score_layer(l)` (`learner.py:~480`). Um `Callback`
(`spifil_ext.FeatureSuperpixels`) troca `state[l].seeds` pelas de `flim.spifil.seeds_in_features` sobre as
features da camada l, com o `n_superpixels_l` da camada seguinte. Fica um `fit` só, sem tocar no núcleo.
Limitação herdada de `seeds_in_features`: nas features usa sempre SLIC (compactness padrão) + Medoids.

### 1.3 Fixos (4.3)

| fixo | status | onde | chave |
|---|---|---|---|
| `estimator=shrinkage` | existe | `Mahalanobis(estimator=...)` (`metrics.py:89`), Ledoit-Wolf em torch | `metric.estimator` |
| `allocator=uniform` | existe | `UniformAllocator` (`allocation.py:33`) | `allocator=uniform` |
| `color=labnorm` | existe | `LabNorm` (`color.py:72`) | `color=labnorm` |
| `budget_check` | via_heranca | **não existe no SPiFiL**; é a `FitStrategy` `GrowOneLayer(budget_check=True)` do SHF (`flim/spifil.py:285-316`), que recusa quando nº de patches ≤ D (`BudgetExhausted`) | `Learner(strategy=)`; no GA vira restrição prévia, não exceção |

### 1.4 Resumo

- `unsupported`: **nenhum**.
- Extensões em `spifil_ext.py`:
  1. `ActivationBlock(SpifilConvBlock)` + `block_factory`: activation / activation_l. Pequena.
  2. dataset com resize, máscara opcional e subconjunto de imagens (composição com `Cropped` / `train_samples` do SHF). Pequena.
  3. `FeatureSuperpixels(Callback)` para `spifil_in=features` (nota A), reaproveitando `seeds_in_features` do SHF.
  4. `head` (flatten / global_avg / global_max): 3 linhas em `evaluate.py`.
- Entrada e saída do SPiFiL:
  - treino: `Learner(data, arch, *, superpixels, seed_extractor, metric, scorer, selector, allocator, color, block_factory, strategy, n_superpixels, device).fit()` (`learner.py:196`);
  - features: `learn.model(x, upto)`;
  - export: `.export(dir)`.
- Formato aceito direto (`best_config.yaml`): Hydra `SpifilConfig` (`config.py:57`), carregado por `spifil.cli.build_learner(cfg)`.
  O schema **rejeita** chaves desconhecidas, então `one_per_class`, `n_images`, `paper_classes`, `impurities`, `spifil_in`, `r`, `head`, `activation` e `budget_check` vão num bloco à parte (`ga_extra:`), lido pelo nosso runner.
  O bloco `spifil:` entra intacto no `build_learner`.

---

## 2. Datasets

### 2.1 new_version (`/dados/home/moliveira/datasets/organized/<nome>`): mesmo layout nos 5

```
{root}/orig/{cls:06d}_{idx:08d}.{png|jpg}
{root}/label/{mesmo nome}                        # só onde has_mask=true
{root}/classes.json   {root}/manifest.json   {root}/split_report.json
{root}/splits/split{s}.json                       # {train, validation, test}: listas de nomes
{root}/splits_incremental/split{s}/data_descriptor_perc{p}.json   # p ∈ {1,5,10,25,50,75,100}
```

- 70/15/15, `stratified_group`, seed 42.
- Só o `train` diminui com a pct (aninhado, perc100 = `splits/splitN.json`).
- **`validation` e `test` são fixos em todas as pcts** → conjunto de avaliação = `validation` ✔.
- Rótulo = prefixo `CCCCCC` = `class_id` (1-based), também no `manifest.json`.
- Existe **perc10** além das 6 pedidas.

| dataset | imgs | classes | tamanho | máscara | train perc 1/5/25/50/75/100 | val | test |
|---|---|---|---|---|---|---|---|
| `helminth_eggs` | 5112 png | 9 (6=Impurities, 3344) | 200×200 RGB | sim (L, {0,255}) | 40/184/897/1791/2688/3580 | 766 | 766 |
| `helminth_larvae` | 3514 png | 2 (1=Impurities 3068, 2=Strongyloides 446) | 200×200 RGB | sim | 26/124/615/1230/1845/2460 | 527 | 527 |
| `protozoan_cysts` | 9568 png | 7 (6=Impurities, 5716) | 200×200 RGB | sim | 72/339/1678/3351/5027/6698 | 1435 | 1435 |
| `coconut_trees` | 13587 png | 2 (coco 10268, non_coco 3319) | 90×90 RGB | não | 96/477/…/9511 | 2038 | 2038 |
| `aid` | 10000 jpg | 30 | 600×600 RGB | não | 84/364/1761/3498/5259/6996 | 1502 | 1502 |

### 2.2 old_version (`SHF/data/to_modules/new_split_parasito/<P>`), separada em 3 configs

`P` ∈ {`helminth-eggs`, `helminth-larvae`, `protozoan-cysts`}.
`data/to_mateus/` é cópia idêntica (verificado com `cmp`). `data/flim` são só diagramas.

```
{root}/images/{cls:06d}_{idx:08d}.png    {root}/masks/{mesmo nome}
{root}/splits/split{s}.json                                       # train / validation / test
{root}/splits_incremental/split{s}/data_descriptor_perc{p}.json   # p ∈ {1,5,25,50,75,100}
```

- 50/50 estratificado (`data/create_splits.py`).
- `validation` = pool de treino − selecionados. **Muda com a pct; em perc100, validation == train.**
- `test` é fixo. Numeração de classes ANTIGA (a mesma do `spifil_paper_seeds.json`).

| config | classes | train perc 1/5/25/50/75/100 | test |
|---|---|---|---|
| `dataset_old_helminth_eggs` | 9 (9=impureza) | 24/124/636/1276/1913/2555 | 2557 |
| `dataset_old_helminth_larvae` | 2 | 17/87/438/878/1317/1757 | 1757 |
| `dataset_old_protozoan_cysts` | 7 (7=impureza) | 46/235/1192/2390/3584/4782 | 4786 |

**A old_version não tem avaliação fixa:** usa a `validation` da pct, sem as imagens dos filtros (§4). O teste nunca é usado.

### 2.3 Armadilha: a new_version renumerou classes e arquivos

A new_version renumerou as classes em ordem alfabética e reindexou os arquivos.
O prefixo antigo está em `manifest.items[].covariates.orig_class_prefix`.

- eggs: Impurities 9→6.
- larvae: 1↔2 trocados, e o índice dentro da classe recomeçou.
- protozoan: Impurities 7→6, e o resto foi reordenado.

Consequência: os ids de `scripts/spifil_paper_seeds.json` **não existem ou apontam para outra classe** na new_version.
O `data.py` precisa traduzir (antigo → novo) pelo `manifest.json`.
O `paper_classes` de cada `dataset_*.yaml` deve ser escrito pelo **nome** da classe, não pelo id.

---

## 3. Restrições e validações

O SPiFiL quase não valida configuração: dataclasses Hydra, sem pydantic, sem `__post_init__` no config.

**Falham cedo** (construtor):

- `layers` vazio (`learner.py:214`);
- `pool_type` inválido (`blocks.py:100`);
- `in_channels < 1` (`blocks.py:98`);
- estimator inválido (`metrics.py:93`);
- `images_dir` inexistente (`data.py:98`).

**Falham tarde** (no `fit`):

- `k` / `dilation` < 1 (`patches.py:72`);
- seleção vazia, por exemplo `N < n_classes` → `N // n_classes = 0` (`nn/builder.py:105`);
- nenhuma semente com rank > 0 (`learner.py:438`);
- shapes de cor e máscara (`color.py:88`, `disf.py:91`).

**Só aviso**: n ≤ D na covariância (`metrics.py:121`); sem máscara (`data.py:173`).

**Não checadas em lugar nenhum** → o `is_valid()` implementa:

| restrição | regra no `is_valid` |
|---|---|
| `pool_stride ≤ pool_size` | literal |
| N vs D | `N ≥ n_classes` (allocator) e nº de patches por camada (`n_imgs · n_superpixels`) > `D = C_in · span²`, com `span = 2*(k//2)+1` (o `budget_check`) |
| RF < lado da imagem | `(span−1)·dilation + 1 < H_l`, com `H_l` após os strides anteriores, e grid final ≥ 1 |
| `n_images ≤` imagens da pct | literal |
| memória | o scoring monta matriz densa N×N (N = n_imgs · n_superpixels): ~12–14·N² bytes. **N=50k ≈ 35 GB**, o teto de uma A6000 → limite configurável em `ga.yaml` |

`C_in` da camada l é o nº de filtros **obtidos** em l−1 (pode ser < `out_channels` por causa da divisão inteira), não o pedido.

---

## 4. Decisões tomadas no portão (2026-10-12)

1. Código em `ga_search/` com CLI, como pede a spec. Execução como módulo: `python -m ga_search.run_ga` (o adapter `flim.spifil` exige a raiz do repo no path).
2. old_version: avaliação = `validation` da própria pct (só treino + validação; o teste NUNCA entra na busca — revisto em 2026-10-12). Em perc100 `validation` == treino, então o `evaluate.py` exclui da avaliação as imagens de onde os filtros foram recortados.
3. Porcentagens: só as 6 pedidas (perc10 ignorada).
4. `pool_size` ∈ {3, 5, 7}.
5. `paper_classes` por NOME da classe; na new_version o `one_per_class` sorteia (os ids do artigo não sobrevivem à renumeração).
6. `relu` (4.1) substituído por `activation` (4.2).

## 5. Ambiente

- torch 2.2.2 (CUDA 12.1, 4× A6000 48 GB). O SPiFiL só usa GPU se `device=cuda:i`.
  - Em CPU: SLIC / DISF / leitura.
  - Em GPU: patches, covariância, scoring, conv.
- O SPiFiL não tem paralelismo próprio (só intra-op do torch: 48 threads por padrão). Workers com CUDA → `spawn`.
- 96 núcleos, ~1 TB RAM.
- **pymoo NÃO instalado** → `/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m pip install pymoo`.
- cupy e numba ausentes (não fazem falta).
