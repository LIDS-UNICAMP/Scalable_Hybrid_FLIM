# Contrato de avaliação — SPiFiL B0–B4 (vale para os Agentes 2, 3, 4, 5, 6 e CHECK)

_Agente 1, 17/11/2026. Precedência: as regras do usuário R1–R11 (`ga_search/modehb/FILA_E_INSTRUCOES.md:10-20`) e a `MISSION.md` valem acima deste texto. Este contrato só as torna operacionais. Mudanças são feitas por versão nova do arquivo, registrada, e nunca em silêncio. Versão: **v1**._

## 1. Régua oficial: ORIG

| item | valor obrigatório |
|---|---|
| Embedding | `z = head(encoder(x))`, com `embed_dim ≥ 1280` (R11; `modehb.py:320` força o mínimo). A cabeça faz parte do modelo e do orçamento Bk (MISSION:8-9) |
| Modo | `encoder.eval()`, `head.eval()`, `torch.no_grad()`, FP32 nas operações sensíveis (cov/eigh/sqrtm) |
| Pré-processamento | o mesmo do teste (Resize + CenterCrop 224 do `progressive_data`), sem TTA na régua oficial |
| SVM | `core.svm.fit_svm(X, y, C=100, scaler=False)`: SVC linear, ovo, `max_iter=-1` (`core/svm.py:143`; R1). Nada de C, scaler, solver ou class_weight diferentes |
| Treino do SVM | **ORIG_full**: o split de treino inteiro. **ORIG_sub4000** (subconjunto estratificado *semeado*, ≤ 4000) só é aceito dentro do laço da busca, por custo, declarado na linha e com os índices (ou o md5 deles) gravados |
| Avaliação | split de **validação** para toda decisão; teste só no relatório final (§4) |
| Diagnóstico obrigatório | `svm_protocol = "scaler=none;kernel=linear;C=100;ovo;max_iter=-1"`, `fit_status` (tem de ser 0), `n_iter`, `embed_dim` e **rank efetivo** (participation ratio) do embedding de validação |

- **Armadilha de ferramenta.** `ga_search/modehb/svm_on_demand.py` tem `--scaler standard` como padrão (`svm_on_demand.py:212`). Para ORIG é obrigatório passar `--readout head --scaler none --C 100`.
- **Rank efetivo.** É exigido porque o embedding atual tem rank efetivo ≈ 3 de 1280 (`agent1/parts/A1b_aprendizagem_embedding.md:219-223`). Dimensão nominal ≥ 1280 não basta para dizer que "separa bem".

## 2. Seleção: só pela validação ORIG

1. **Fitness = (κ ORIG-val, FLOPs em 2·MAC). Mais nada** (R3).
   - É proibido usar F1, quality com kappa_weight < 1, COV, scaler ou TTA na fitness.
   - Pareto e hipervolume só entre candidatos da mesma faixa Bk, do mesmo dataset e da mesma sub-régua.
2. **Toda decisão intermediária usa a mesma régua:**
   - α do WiSE;
   - promoção multi-fidelidade;
   - escolha de checkpoint/época, incluindo o SSL das sementes (nunca pela perda: `A1b:300-304`; FILA:180-184);
   - early stopping;
   - escolha de operador ou cabeça;
   - política de TTA (esta só depois, e só na validação; `ATTENTION_ADDENDUM.md:65-71`).
3. **Empates:** \|Δκ\| < MDE do dataset (§5) conta como empate prático. Desempate determinístico e declarado: menos FLOPs (2·MAC), depois menos params, depois menor fidelidade numérica (5 < 15 < 45 < 100 < 200; nunca a ordem lexicográfica das strings, ver CHECK:127), depois menor `cand_id`.
4. **Fonte de verdade do status:** o `search_state.json` (job `ok`). O `result.json` sozinho não basta (CHECK:40, C4).
5. **Snapshot congelado.** Uma seleção é um *manifesto*:
   - CSV com dataset, faixa, cand, fid, caminho, md5 do checkpoint, κ ORIG-val, sub-régua, regra de desempate e timestamp;
   - o manifesto e as listas derivadas são gerados por **um** script, com o md5 de cada artefato no log (CHECK:39, C3).

## 3. Réguas auxiliares: permitidas, identificadas, nunca decisivas

Toda linha de resultado tem o campo `regua`, e um número só é comparado com outro da mesma régua.

| `regua` | definição | uso permitido |
|---|---|---|
| `ORIG_full` | §1, train inteiro | **oficial** (seleção na validação, reporte no teste) |
| `ORIG_sub4000` | §1, subconjunto semeado ≤ 4000 | fitness dentro da busca, declarada. A escolha final é re-medida em ORIG_full |
| `COV` | `cov_head` (2ª ordem por bloco, PCA k=64, + head), signed-sqrt + L2, StandardScaler, C=100, mt4000 (`search_v3.yaml:142-143`) | coluna auxiliar e diagnóstico. **Proibida** em fitness, WiSE e promoção daqui em diante |
| `COV_noscaler`, `ORIG+scaler`, `ORIG+PW` | variantes de leitura do diagnóstico A1b | diagnóstico |
| `GTP` / `gem_tta_pw` / `cat_tta_pw` | régua do discovery (TTA + whitening) | histórico; não comparar com ORIG |
| `SONDA_GeM` (`probe/lin_select_kappa`) | sonda das sementes LeJEPA | só dentro do treino das sementes; re-medir em ORIG antes de usar |
| sondas por bloco | SVM C=100 sobre GAP/GeM/COV de um bloco | só diagnóstico, nunca seleção |
| `ORIG_TTA4/8/ms` | ORIG com TTA (média normalizada) | complementar nos finalistas, com o custo multiplicativo registrado |

Números de buscas C=0,1 / kw 0,75 (`*_C01_*`) nunca aparecem como "nosso" nem entram em comparação com C=100 (`agent1/parts/A1c_generalizacao.md:127-128`).

## 4. Teste: só no fim, com a escolha congelada

1. O teste só é lido por um runner que **recusa** `--split test` sem `--frozen-from <manifesto>`. É o padrão do Agente 6 (CHECK:84).
2. Antes da leitura, o manifesto (§2.5) fica gravado com md5. Ele contém o md5 do código (git commit + md5 do diff, runner e seletor), as versões de torch/sklearn, o host e o UUID da GPU (CHECK:42-43, C6/C7).
3. Uma leitura de teste por manifesto. Depois de ler o teste, **nenhuma** re-seleção, re-ajuste de α, cabeça, TTA ou checkpoint. Se algo mudar, gera-se um manifesto novo, e o teste anterior fica registrado como "visto".
4. Proibido apresentar o "melhor retrospectivo no teste" (máx-teste) como "nosso" (A1c:126). Ele só pode aparecer como coluna de auditoria, rotulado `oracle_test`.
5. PCA, whitening, alinhamento, normalização e SVM são ajustados **só no treino** e reutilizados em validação e teste (ATTENTION_ADDENDUM.md:39-40).

## 5. Splits, seeds, estatística e n mínimo

### 5.1 Splits (imutáveis; sha256 conferido em toda execução)

Splits usados (A1c:57-67):

| dataset | split | train / val / test |
|---|---|---|
| RS | `split1.json`, agrupado por cena | 682 / 159 / 159 |
| HAM10000 | `split1_modehb.json` | 4129 / 878 / 5008 |
| eggs, eggs_raw | `split1_modehb(_raw).json` | 8885 / 1903 / 1903 |
| larvae, larvae_raw | `split1_modehb(_raw).json` | 1118 / 240 / 240 |
| protozoan, protozoan_raw | `split1_modehb(_raw).json` | 26159 / 5607 / 5606 |
| flowers | `split1.json` | 1922 / 412 / 412 |
| aid | `split1.json` | 6996 / 1502 / 1502 |
| coconut_trees | `split1.json` | 9511 / 2038 / 2038 |

- Os YAMLs ficam em `ga_search/config/modehb/datasets/*.yaml`.
- Toda linha grava `split_json`, o sha256 dele e o `data_sha256` do `progressive_data`.
- Os pares pre e raw são a mesma partição (A1c:69-78), mas são **datasets distintos**: nunca se misturam numa mesma média nem num mesmo Pareto.

### 5.2 Seeds

- Seeds de treino `{42, 43, 44}` no mínimo para qualquer afirmação comparativa; `{42..46}` quando o Δ esperado for < 0,02 ou o MDE do dataset passar de 0,03.
- Seeds auxiliares (subconjunto do SVM, metades A/B da validação, alinhamento, projeções aleatórias) são fixas e gravadas.
- Resultado com n=1 seed é rotulado `n_seeds=1` e não sustenta afirmação causal.

### 5.3 Estatística

- **Predições por imagem são obrigatórias** (`sample_id`, `y`, `pred`, split, com md5) em validação e teste (CHECK:38, C2).
- **Bootstrap pareado** por imagem: B = 10 000 reamostragens com as mesmas imagens nos dois braços. IC 95% percentil do Δκ. Com várias seeds, Δ médio entre seeds e IC hierárquico (reamostrar seeds e depois imagens).
- **MDE por dataset** é medido em X01 (`experiment_matrix.csv`). Até lá, o piso é 0,02, ou 0,03 em RS (n_val 159, EP ≈ 0,04) e larvae (240) (A1b:83; A1c:157).
- **Regra de decisão padrão:** um braço "vence" se Δ médio ≥ MDE **e** IC 95% > 0 em ≥ 2 de 3 datasets prioritários (RS, Flowers, HAM). "Sem efeito" se o IC estiver contido em ±MDE. O resto é inconclusivo.
- Correção de Holm entre os braços de um mesmo experimento.
- Resultados negativos são registrados (MISSION:51).

### 5.4 n mínimo e classes raras

- κ por dataset só com n_eval ≥ 150.
- Toda tabela traz n_eval e o tamanho da menor classe. Exemplos: RS tem classe com 1 imagem de teste; eggs, 12 (A1c:140-157).
- Em diferença entre braços abaixo de 2 imagens equivalentes no κ, declarar "dentro da granularidade".

## 6. FLOPs, params, memória

- **Unidade oficial: FLOPs = 2·MAC**, campo `flops_2mac`.
  - O campo `flops` da busca atual é **MAC** (`run_search.py:322-360` `inference_macs`, gravado em `:784`). Converter com ×2 e nunca misturar (CHECK:41, C5).
  - O `seed_registry.csv` traz as duas colunas.
- Os FLOPs incluem encoder + cabeça + qualquer operador sem parâmetros (atenção fixa, cov/sqrtm, normalizações) da inferência de uma imagem 224. TTA é um multiplicador à parte (`tta_cost`).
- Params do orçamento Bk = encoder + cabeça (≥ 1280). Decoders do AE e projetores do SSL são registrados à parte e não contam (MISSION:53). Teto duro: 5,5M.
- Registrar também o pico de memória e a latência (ms/imagem, bs=1 e bs do eval), porque FLOPs escondem o custo de ativações (A1d:214-215).

## 7. Vazamentos: tratamento obrigatório

| dataset | problema (evidência) | tratamento |
|---|---|---|
| HAM10000 | val limpa por lesão; **1312 imagens de teste (26,2%)** com lesão em train∪val; 1 duplicata com `lesion_id` diferente (ISIC_0025226 = ISIC_0030074) (A1c:12-17, 98-113) | seleção na validação do `split1_modehb` (válida). Teste relatado em duas colunas: `test_full` e `test_lesion_clean` (só imagens cuja lesão não está em train∪val, excluindo também o par ISIC). F1 por classe MEL/DF/BCC/NV. Pendência: criar `split1_lesion.json` (70/15/15 por lesão) e migrar para ele em versão nova do contrato |
| eggs, larvae (pre) | recapturas do mesmo objeto entre splits: eggs 7 test↔train, 5 val↔train, 1 val↔test; larvae 2 test↔train, 4 val↔train; são limites inferiores (A1c:7-11) | sensibilidade `test_no_recapture` excluindo os pares `veredito_visual=mesmo` de `agent1/parts/A1c_dup_pairs.csv` (e as imagens de val envolvidas numa re-medição da validação). Declarar que não existe id de lâmina/campo |
| protozoan (pre) | recaptura **indeterminada** (A1c:11, 92) | declarar no relatório |
| flowers | 1 cópia byte a byte train↔test com rótulo diferente + 3 pares entre splits (A1c:26-28, 88) | `test_no_dup` excluindo os 4 pares |
| aid | 1 par idêntico train↔test (A1c:85) | `test_no_dup` |
| sementes externas | a semente HAM antiga viu o nosso val (FILA:45) | proibido usar `*_VIU_NOSSO_VAL_nao_usar`; só `ham10000_trainonly` |

- O Δ entre braços é pareado, então o vazamento afeta o κ **absoluto** e a comparação com FMs (CHECK:154). Os dois têm de vir com a sensibilidade ao lado.
- Regra do X17: se a versão limpa inverter uma ordem com Δ > MDE ou trocar o sinal de (nosso − FM), o número oficial passa a ser o limpo.

## 8. Foundation models: só no mesmo domínio e no mesmo protocolo

- **pre × FM-pre e raw × FM-raw. Nunca cruzado.** Isso vale porque FM-pre > FM-raw em 9 de 9 casos (+0,010 a +0,085) (A1c:23, 121; FILA:249).
  - FM-raw está em `benchmark/modehb_raw`; FM-pre em `benchmark/modehb_pre`.
  - Enquanto AIMv2 e DINOv2 em pre não existirem, a comparação pre traz "FM parcial (3 de 5)".
- Mesmo split, mesmas imagens, **mesmo train do SVM** e mesma régua ORIG (C=100, sem scaler).
  - No HAM, os FMs precisam ser re-ajustados no train de 4129 (`split1_modehb`), e não em 5007 (A1c:18-21, 130).
- n=1 (split1) contra o valor split1 do FM. Média ± DP só quando os dois lados tiverem n=3 (A1c:123, 177).
- A resolução nativa de cada FM (448/518 px) é registrada, não corrigida (A1c:131).
- COV, GTP e TTA do nosso lado nunca são comparados com ORIG do FM (A1c:122).

## 9. Proveniência obrigatória (por linha de resultado)

| grupo | campos |
|---|---|
| dados | `dataset`, `domain` (pre/raw/natural), `split_json`, `split_sha256`, `data_sha256`, `n_train`, `n_eval`, `split_eval` (validation/test), `subset_indices_md5` |
| modelo | `checkpoint` (caminho), `ckpt_md5`, `cand_id`, `fid`, `seed`, `parent_id`/ancestral, `genome`/spec com hash, `init` (SPiFiL-k/aleatório, com versão), `ae`, `ssl`, `wise_alpha`, `protect` |
| custo | `params` (encoder+cabeça), `params_aux` (decoder/projetor), `flops_2mac`, `peak_mem`, `ms_per_image`, `tta_cost`, `gpu_hours` |
| avaliação | `regua`, `svm_protocol`, `fit_status`, `n_iter`, `embed_dim`, `erank`, `kappa`, `macro_f1`, `pred_file` + md5, `used_for_selection`, `never_feeds_fitness` |
| ambiente | `host`, `gpu_uuid`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `torch`, `sklearn`, `git_commit`, `code_md5` (runner, seletor, diff), `config_resolved_md5`, `timestamp` |
| decisão | `manifest_id` (para o teste), `selection_rule`, `tie_break` |

Sem esses campos o resultado não entra em ranking, gate nem relatório.

## 10. Proibições (resumo)

1. Selecionar ou ajustar qualquer coisa com imagens ou rótulos de teste.
2. Mudar C, scaler ou solver do SVM oficial (R1).
3. Usar COV, GTP, scaler ou TTA em fitness, WiSE ou promoção.
4. Comparar números de réguas, domínios (pre/raw), splits, pct ou C diferentes na mesma tabela sem rótulo explícito.
5. Reduzir a cabeça abaixo de 1280 para caber no orçamento (R11), ou usar o CovReadout externo como substituto silencioso da cabeça (MISSION:56).
6. Escolher o checkpoint de SSL pela perda.
7. Misturar MAC com 2·MAC.
8. Apresentar o máx-teste, ou checkpoints C=0,1, como "nosso".

## 11. Checklist do gate (o CHECK confere)

- [ ] Manifesto de seleção congelado, com md5, gerado por script único.
- [ ] Todas as linhas com `regua`; decisões só com `ORIG_full`/`ORIG_sub4000` declaradas.
- [ ] Predições por imagem salvas (validação e teste).
- [ ] ≥ 3 seeds e IC bootstrap pareado para cada afirmação comparativa.
- [ ] FLOPs em 2·MAC.
- [ ] Sensibilidade de vazamento (§7) no relatório de teste.
- [ ] FMs só do mesmo domínio e com o mesmo train.
- [ ] `embed_dim ≥ 1280` e rank efetivo reportado.
