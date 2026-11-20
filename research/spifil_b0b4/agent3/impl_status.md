# Agente 3: implementação inicial de BackboneSpec, GrowthSpec, morfismos e B0–B4

_16/11/2026. A implementação foi autorizada pelo usuário para acelerar o trabalho. Tudo rodou só em CPU (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 15`). Não houve treino, GPU nem dado real._

## Entrega

- **Patch:** `research/spifil_b0b4/agent3/a3impl.patch`, md5 `07c72deae9f480e9e5018721faa652c7` (também em `a3impl.patch.md5`).
  - Só **cria arquivos novos**: `ga_search/backbones/` (8 arquivos) e `tests/backbones/` (7 arquivos), com 2266 linhas.
  - Nenhum arquivo de `ga_search/modehb/` foi tocado. O pacote só **importa** `ConvBlock`/`InceptionConv`/ZCA/DBN de `ops_norm_act.py` e `EmbeddingHead` de `pooling_head.py`.
  - `git apply --check` passa na árvore viva. O patch **não foi aplicado** nela.
- **Cópia de trabalho:** `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/a3impl/`.
  - É um git próprio, com dois commits: a baseline (cópia de `modehb/*.py`) e a entrega.
- **Suíte:** `python -m pytest -q tests/backbones` dá **85 passed em ~9 s** (CPU, 4 threads).

Para aplicar e rodar:

```bash
git apply research/spifil_b0b4/agent3/a3impl.patch
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python -m pytest -q tests/backbones
```

## O que foi implementado

| Item pedido | Onde | Estado |
|---|---|---|
| BackboneSpec v1 (stem, estágios local/intermediate/abstract, cabeça ≥ 1280, custo, proveniência, extras não contados) | `spec.py` | Feito. Dataclasses, JSON canônico, `arch_sha256` e `id = dataset/Bk/hash12`. O `from_json` detecta adulteração. |
| GrowthSpec v1 (operações, herança por camada, preservação, params/MACs/FLOPs, protect, twin) | `spec.py` + `morphisms.finalize` | Feito. `id = pai12->filho12:ops`, `content_sha256` e ida e volta em JSON. |
| Construtor a partir do spec | `build.py` | Feito. Cada camada é um `ConvBlock` do repo e a cabeça é a `EmbeddingHead` do repo. Há uma variante `lowrank`: 1×1 C→r sem bias seguido da `EmbeddingHead`. |
| Params encoder + cabeça, com decoder/projetor à parte | `build.count_params`, `CostSpec.params_extras` | Feito. Os extras são registrados e nunca somados. |
| FLOPs = 2×MACs, explícito | `build.count_macs` | Feito. Mesma metodologia de `run_search.inference_macs` (termos de kernel de Conv2d/Linear). Encoder e cabeça são contados à parte, e a definição fica gravada em `cost.flops_definition`. |
| Teto duro de 5,5M + faixas B0–B4 | `spec.BUDGETS/HARD_CAP/band_of`, `build.check_budget` | Feito. Acima de 5,5M dá `BudgetError` sempre, inclusive no `finalize` de um morfismo. A classe declarada é verificada sem tolerância (`cap_factor = 1,0`). |
| Reparo da forma da cabeça sem reduzir a dimensão | `budget.repair_head`, `repair_family_head` | Feito. A ordem de tentativa é forma atual → meanstd → moments → lowrank (r = 512…32). Cada reparo gera um spec novo com hash novo e log. Se nada couber, dá `BudgetError`. O `embed_dim` nunca muda. |
| Morfismos com teste numérico em `eval()` | `morphisms.py`, `preservation.py` | Feito. Detalhes na tabela abaixo. |
| Família plana de referência B0→B4 por morfismo exato | `family.py` | Feito. Cadeia exata e verificada a 224 px. |
| Checkpoint autocontido | `checkpoint.py` | Feito. Um único arquivo com `spec_json`, `state_dict`, `state_sha256` e TorchScript, que carrega com `torch.load(weights_only=True)`. |

## Morfismos: exatidão e condições

O `GrowthSpec` registra, para cada operação:
- `exact_claimed` e a `condition` que garante a exatidão;
- por camada: origem, shapes de pai e filho, `in_map`/`out_map` (canal do filho → canal do pai, −1 = novo), elementos herdados × novos e o init dos novos;
- `preservation`: rel-L2, max|Δ| e cosseno mínimo, por camada herdada, no mapa final e no embedding 1280. Inclui o sha256 das imagens usadas.

O veredito segue estas regras:
- `exact` quando a operação foi declarada exata e a medida ficou ≤ 1e-5;
- `violated` quando foi declarada exata e passou de 1e-5;
- quando não foi declarada exata: `approx` (≤ 1e-2) ou `not_preserving`.

| Operação | Exata? | Condição / observação |
|---|---|---|
| `widen` conv densa / pointwise (+ `kernel_grow` k→k+2m) | Sim | Colunas de entrada novas = 0 |
| `widen` com BN | Sim | γ, β, running_mean, running_var e num_batches_tracked copiados por canal. Testado em eval(). Em treino também deveria ser exato (estatística por canal), mas isso não foi testado. |
| `widen` com InstanceNorm | Sim | Afim por canal |
| `widen` com residual identity / gated | Sim | O mapa de saída é igual ao de entrada (o skip leva canal novo para canal novo). Exige cout = cin. |
| `widen` com residual project | Sim | 1×1 com colunas novas = 0 |
| `widen` dwsep / depthwise (cin = cout) | Sim | Depthwise novo para canal novo; pointwise com colunas novas = 0. Depthwise com multiplicador > 1 é recusado. |
| `widen` inception | Sim (rel ~1e-7) | Widen por ramo. A partição `branches` fica explícita no spec, e os canais novos ficam intercalados por ramo, com o mapa acompanhando. |
| `widen` ZCA | **Só com o branqueamento congelado** | W = [[W_pai, 0], [0, I]] e μ = [μ_pai, 0]. Re-`fit` no filho quebra a exatidão. Isso fica registrado na `condition`. |
| `widen` DBN | **Só em eval()** (running_W congelado) | running_W por grupo, em bloco-diagonal com I. O filho é recusado (`NotExactError`) se um grupo do pai for partido entre grupos do filho, por exemplo com inception mais DBN de grupos pequenos. |
| `widen` GroupNorm / LayerNorm | **Não** | Recusado, a menos que se passe `allow_approx=True`. Nesse caso fica declarado `exact_claimed=False` e o veredito é medido. |
| `head_widen` | Sim | Colunas zero no 1×1 da cabeça (ou no `reduce` da lowrank). Testado com gap, gem, gem_learn, meanstd, moments, combo, cabeça com BN e lowrank. **A cabeça passa a ser herdada (`inherit_zero_cols`), e não recriada aleatória.** |
| `deepen_identity` | Sim, se a entrada for ≥ 0 | Exige ReLU/ReLU6 na camada anterior e nenhum skip. Com BN (rm = 0, rv = 1, γ = √(1+eps)), só em eval(). Fora dessa condição é recusado, ou declarado aproximado com `allow_approx`. |
| `deepen_residual_zero` | Sim, para qualquer entrada | y = act(conv=0) + x, com act ∈ {silu, gelu, relu} e conv ∈ {conv, dwsep, inception}, norm none ou batch. Testado após uma GELU, ou seja, com entrada negativa. Com ReLU a operação é exata mas não treinável no zero, e isso fica registrado. |
| `add_skip_zero` (residual de contribuição zero numa camada existente) | Sim | `gated` (y + λx, λ = 0) quando cin = cout; senão `project` com 1×1 = 0 |

Há um **controle negativo** (`test_negative_control_detects_violation`): somar 0,5 às colunas "novas" faz o veredito virar `violated`. Ele mostra que a medida detecta quebra de preservação.

## Família de referência B0→B4 (224 px, cabeça meanstd 1280)

A tabela §4 usava GeM. Com GeM, o B1 dá 881.664 > 800k. O `repair_family_head` procura uma única forma de cabeça que sirva para os cinco (o `head_widen` exato precisa da mesma forma ao longo da cadeia). A primeira que coube foi **meanstd 1280**: 1×1 → 640, μ‖σ.

| Bk | hash12 | params encoder | params cabeça | total | GMACs | GFLOPs |
|---|---|---|---|---|---|---|
| B0 | c7f23392f942 | 127.632 | 72.320 | 199.952 | 0,406 | 0,812 |
| B1 | b181fb45979c | 655.104 | 113.280 | 768.384 | 1,288 | 2,575 |
| B2 | c7b5777d9661 | 1.164.544 | 164.480 | 1.329.024 | 2,233 | 4,467 |
| B3 | 7d585257b0ad | 2.574.656 | 184.960 | 2.759.616 | 4,572 | 9,145 |
| B4 | 546a2a596f7a | 4.717.344 | 256.640 | 4.973.984 | 7,642 | 15,284 |

Cada membro cai na própria faixa: B0 ≤ 300k, B1 em (300k, 800k], B2 em (800k, 1,5M], B3 em (1,5M, 3M] e B4 em (3M, 5,5M].

Arestas, medidas com 2 imagens sintéticas U[0,1] a 224 px em eval() e fp32:

| Aresta | Veredito | rel-L2 mapa | rel-L2 embedding | cos mín. | Herdados | Novos |
|---|---|---|---|---|---|---|
| B0→B1 (deepen_identity ×2 + widen + head_widen) | exact | 0 | 0 | 1,0 | 199.952 | 568.432 |
| B1→B2 (deepen_identity + widen + head_widen) | exact | 0 | 0 | 1,0 | 768.384 | 560.640 |
| B2→B3 (deepen_identity ×2 + widen + head_widen) | exact | 0 | 0 | 1,0 | 1.329.024 | 1.430.592 |
| B3→B4 (widen + head_widen) | exact | 0 | 0 | 1,0 | 2.759.616 | 2.214.368 |

- O teste de ponta a ponta confirma que o embedding do B4 recém-crescido é igual ao do B0 (rel ≤ 1e-5). Nos canais herdados, o mapa final 14×14 também é igual.
- O arch_hash de cada filho é igual ao do spec-alvo da família.
- Com GeM, os totais e os GMACs batem com a tabela §4: 0,272/0,882/1,494/2,945/5,231M e 0,42/1,31/2,27/4,61/7,69 G.

## Testes (`tests/backbones/`, 85)

| Arquivo | O que verifica |
|---|---|
| `test_morphisms.py` (49) | Preservação ≤ 1e-5 em eval()/CPU para cada morfismo exato: 14 casos de widen, 9 de head_widen, 6 de deepen_identity, 12 de residual_zero, 3 de add_skip_zero e 1 aresta composta com máscaras e ida e volta do GrowthSpec. Também cobre: LN/GN declarados não exatos, recusa de identity com entrada negativa, recusa de DBN com grupo partido e o controle negativo. |
| `test_cost_budget.py` (22) | Params e MACs contra fórmula analítica independente: família × gem/meanstd/moments, dwsep, inception, project, gated, lowrank e extras não contados. FLOPs = 2×MACs. Números da tabela §4. Contagem meta igual à real. Teto duro e de classe. Reparo de cabeça (meanstd, lowrank e erro quando nada cabe). Família nas faixas. Morfismo acima de 5,5M recusado. |
| `test_spec_hash.py` (6) | Hash dourado. Estabilidade entre processos (PYTHONHASHSEED 0/1/2). Invariância a custo, proveniência, dataset e `init`. Sensibilidade à arquitetura. JSON ida e volta com detecção de adulteração. Validações (embed_dim < 1280 proibido, identity com cin ≠ cout, ordem dos estágios). Faixas. |
| `test_family_chain.py` (3) | Cadeia B0→B4 a 224 px: alvos e faixas, toda aresta exata, B0 ≡ B4 no embedding. |
| `test_checkpoint.py` (5) | Salvar, carregar com strict e inferir com resultado idêntico. Inferir num subprocesso `python -I` em diretório vazio, sem `PYTHONPATH` e com `ga_search` não importável (verificado), carregando só pelo arquivo via TorchScript; rel < 1e-6 com ZCA, DBN, inception, dwsep, skip e cabeça lowrank. Checkpoint adulterado é recusado. Filho crescido carrega sem o pai. |

## Decisões e desvios registrados

1. **Pool do stem:** MaxPool(2,2) do `ConvBlock` em vez de 3×3 s2 pad 1. As resoluções são as mesmas (224→112→56→28→14) e params e MACs não mudam (pooling não entra nos MACs).
2. **Inserção de profundidade:** a camada nova entra antes da camada que faz pool do estágio (convenção do PNAS-CS).
3. **O hash exclui** custo, proveniência, dataset, split, budget_class e o `init` de cada camada (proveniência de pesos). Entram input, stem, stages e head. Por isso, um gêmeo reinicializado tem o mesmo arch_hash do filho e se distingue pelo checkpoint.
4. **Pesos novos:** por padrão usam a init do PyTorch com semente registrada (`new_init` no GrowthSpec). O gancho `widen(..., new_filters=fn)` recebe os filtros novos do SPiFiL (contrato do Agente 2). A exatidão independe desse init.
5. **`taps`:** só `["last"]` na v1. As cabeças multiescala, `fpn_lite`, `bilinear_lowrank` e `tensor_sketch` ficam pendentes; o spec recusa outros valores em vez de ignorá-los.
6. **Preservação nos testes:** usa dados sintéticos U[0,1]. Em produção, o `finalize(res, x, data=...)` deve receber imagens de **validação** (nunca teste), e a origem e o sha ficam gravados.

## Pendente (fora deste incremento)

- `widen_replicate` (Net2Net com divisão pelo fator), `prune_l1`/`to_dwsep` (ANM, linhagem shrink) e `reinit_twin` automático.
- Cabeças multiescala, FPN-lite, bilinear de baixo rank e tensor sketch (experimento H).
- Integração com `run_search`/MO-DEHB (Agente 4): a herança da cabeça mudaria o significado de WiSE α=0 e precisa entrar como variante registrada.
- Latência, pico de memória e tamanho do checkpoint no `CostSpec`, medidos na GPU 3 pela fila do Agente 5.
- Registro JSONL do DAG (`lineage.jsonl`). Os ids e hashes já estão prontos para isso.
