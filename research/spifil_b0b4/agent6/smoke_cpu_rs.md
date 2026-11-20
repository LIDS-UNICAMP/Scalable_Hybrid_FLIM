# Agente 6: smoke da atenção fixa em CPU (remote_sensing, VALIDAÇÃO), v2

> **Isto NÃO é resultado científico.** O smoke só confere o encanamento. São 159 imagens de validação e 1 seed, e nenhum IC exclui o zero. O teste nunca foi lido.
> A **v2** refaz o smoke depois das correções do CHECK: C1 (sorteio uniforme no ajuste), C2 (preds por imagem), C5 (unidades), C6 (ambiente por linha) e TF32 desligado. O v1 continua em `smoke/` como histórico.

## Condições

- **Execução:** Jaci, só CPU, com `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 nice -n 15`. Nenhuma GPU foi usada. Ambiente gravado em cada linha: torch 2.7.1+cu128, sklearn 1.8.0, numpy 1.26.4, Python 3.11.15.
- **Código:**
  - `fixed_attention.py` (md5 `f4bc107c…`);
  - `eval_fixed_attention.py` (md5 `f4fc9a2d…`).
- **Checkpoints** (somente leitura):
  - RS lemonade c00030 f5: 3 blocos, 14@112² / 56@56² / 210@28²; encoder + cabeça = 2,407 GFLOP (2·MAC).
  - RS hybrid c00009 f5: 2 blocos, 70@112² / 280@56²; cabeça moments; 1,751 GFLOP (2·MAC).
- **Protocolo:**
  - SVM `core.svm.fit_svm`, linear, C=100, sem scaler. A régua é `ORIG_full`, porque o RS tem 682 imagens de treino, menos que max_train.
  - Validação: 159 imagens.
  - fp32, sem TF32.
  - Ajuste do alinhamento: 160 imagens de treino **sorteadas** (seed 0).
- **Saídas:** `agent6/smoke_v2/`:
  - `smoke_rs_val.jsonl`, 20 linhas, cada uma com `preds/*.npz`;
  - `dk_bootstrap_smoke.csv`;
  - `plumbing_c00030.jsonl`;
  - `compat_ref_jaci.jsonl`;
  - `full_suite.log`.

## κ de validação (encanamento)

| ckpt | E0 | E1 pool | E2 self | E3 cross | E4 multiescala | E5 cov | E6 tta4 | E7 tta8 | E8 multiescala | selecionado (val) |
|---|---|---|---|---|---|---|---|---|---|---|
| c00030 | 0,460 | 0,467 | **0,475** | 0,461 | 0,446 | 0,475 | 0,468 | 0,482 | **0,488** | E8 (E2+multiescala) |
| c00009 | 0,458 | 0,425 | **0,480** | 0,458 | 0,465 | 0,458 | **0,508** | 0,452 | 0,459 | E6 (E2+tta4) |

**Bootstrap pareado contra o E0** (`paired_bootstrap.py`, 1000 reamostras): todos os IC95 incluem 0. Por exemplo, c00009 E6 dá +0,050 [−0,006; 0,112] e c00030 E8 dá +0,028 [−0,031; 0,084]. Como esperado com 159 imagens, **nada aqui é evidência**.

**O efeito do C1 aparece no E3/E4.** Com o sorteio uniforme, o ajuste vê **14/14 classes**, contra 4/14 antes. Os valores mudaram:

| ckpt | operador | v1 | v2 |
|---|---|---|---|
| c00030 | E3 | 0,4530 | 0,4605 |
| c00030 | E4 | 0,4461 | 0,4464 |
| c00009 | E3 | 0,4651 | 0,4584 |
| c00009 | E4 | 0,4581 | 0,4651 |

Os operadores sem ajuste (E0, E1, E2, E5) e o TTA do c00030 reproduziram o v1 exatamente.

## Encanamento conferido

| Verificação | Resultado |
|---|---|
| Linhas finitas, `embed_dim`, parâmetros | todas finitas; `embed_dim` 1280; `attn_trainable_params=0` em todas |
| Preds por imagem (C2) | `sample_id`, `y` e `pred` (159 cada), com md5; o Δκ do bootstrap bate com os κ das linhas |
| `--exclude-ids` | funciona (demonstração com 2 ids em `dk_bootstrap_smoke_excl_demo.csv`) |
| Retomada | relançar pula o checkpoint completo |
| `--split test` sem `--frozen-from` | recusado |
| Caminho de relato (`--split test --plumbing-check`, loader de **validação**) | E0 / E2_noTTA / E8_frozen = 0,4603 / 0,4753 / 0,4881, idênticos à validação; linhas com `used_for_selection=false` e `reporting_only=true` |
| Compatibilidade (C7) | referência da Jaci com os 74 checkpoints da lista v2: 74/74 ok; a autocomparação também dá 74/74. A comparação no athena é o job A-attn-03 |

## Custo medido (por imagem; FLOPs = 2·MAC, igual ao encoder)

| ckpt | operador | GFLOP | fração do encoder | ajuste | ms/img (CPU, carga ~35) |
|---|---|---|---|---|---|
| c00030 | E1 / E2 / E3 / E4 / E5 | 0,003 / 0,034 / 0,027 / 0,018 / 0,140 | 0,001 / 0,014 / 0,011 / 0,007 / 0,058 | E3/E4: 5,2 s | 30–33 (E0 33) |
| c00009 | E1 / E2 / E3 / E4 / E5 | 0,011 / 0,180 / 0,137 / 0,159 / 0,989 | 0,006 / 0,103 / 0,078 / 0,091 / **0,565** | E3/E4: 17–18 s | 99–132 (E0 99) |

**Leitura:**
- O E5 na c00009 custa **57% do encoder**, e não o dobro, como dizia o v1, que misturava MAC com 2·MAC.
- Na lista v2, o E5 passa de 25% do encoder em 10/74 checkpoints e de 100% em 2/74 (`custo_E1_E5_2mac.csv`).
- TTA: tta4 ≈ ×3,8; tta8 ≈ ×7,6; multiescala ≈ ×2,8 a 2,9 na latência de parede.
- O SVM leva 0,06–0,08 s com 682 × 1280. A calibração da v1 no eggs, com 3997 × 1280, deu 13,9 s.

## Suíte

- `tests/modehb/test_fixed_attention.py`: **152 passed**.
- `tests/` inteira da cópia: **2460 passed, 4 skipped, 0 failed** (88 s, CPU).
