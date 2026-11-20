# CHECK: auditoria do gate "B-attn" (E0–E8, atenção fixa, athena GPU 3)

> Agente CHECK, 17/11/2026. A auditoria rodou na Jaci, só em CPU (`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 15`). Não toquei em nenhum processo, nem na árvore viva, nem na fila. Também não criei `gates/gate_B.approved`: essa decisão fica com o orquestrador.
>
> **Objeto da auditoria:**
> - `research/spifil_b0b4/agent6/`: plano, fila, CSV, listas, smoke, tools e patch;
> - cópia congelada `scratchpad/attn6`.
>
> **Artefatos que eu gerei:** `scratchpad/check_battn/` (o caminho completo está no fim). São eles:
> - `full_suite.log`;
> - `test_extra_battn.py`;
> - `real_ckpt_check.py` e `real_ckpt_check.log`;
> - `indep_select.py`;
> - `order_check.py`.

## Veredito: **APROVADO COM CONDIÇÕES**

O núcleo está correto e verificado:
- os operadores têm zero parâmetros treináveis;
- α=0 dá identidade exata;
- o teste só é lido depois do congelamento;
- a SVM é a ORIG com C=100;
- a fila usa só a GPU 3.

Há porém **4 defeitos que impedem rodar o B como está**, e eles valem antes do primeiro job `B-attn-E0E8-*`:
- C1: o ajuste do alinhamento usa amostras enviesadas por classe;
- C2: o bootstrap pareado não pode ser calculado, porque as predições não são salvas;
- C3: a lista de checkpoints já divergiu e o congelamento não está implementado;
- C4: um dos checkpoints está como `failed` na busca.

A fase A (A-attn-00/01/02) pode seguir, com as condições C6 e C7 incluídas no próprio A. As condições C5 e C8 são de documentação e precisam estar cumpridas antes do relatório de teste e de qualquer gate de operador.

### Condições

| # | Bloqueia | Condição |
|---|---|---|
| **C1** | B | **O alinhamento E3/E4 é ajustado nas 160 primeiras imagens do treino, e o treino está ordenado por classe.** A causa está em `fixed_attention.py:836` (`xs = x_train[:n_img]`) e `eval_fixed_attention.py:92`, combinados com `_stratified_indices`, que devolve índices **ordenados** (`run_search.py:1118`). O ajuste vê só **1 classe** em HAM, flowers, larvae e coconut, 2 em eggs, aid e protozoan_raw, e 4 de 14 em RS (`order_check.py`). Não há vazamento nisso, mas o PCA/CCA fica enviesado e o E3/E4 entra em desvantagem diante de E0/E1/E2/E5. **Correção:** amostra semeada uniforme (permutação com seed fixa sobre o subconjunto de treino, sem usar rótulo), com os índices ou o md5 gravados na linha. Depois, refazer o smoke. |
| **C2** | B | **O critério de aprovação (§5: IC bootstrap pareado > 0) não pode ser calculado.** O runner grava só κ e F1 (`eval_fixed_attention.py:80-83`) e não guarda predição nem id por imagem. Sem isso, o bootstrap exige rodar a GPU de novo. **Correção:** salvar `sample_id`, `y` e `pred` de cada linha (val e teste), em jsonl ou npz, com md5. |
| **C3** | B | **A lista móvel não tem congelamento implementado.** O `select_checkpoints.py` só escreve o CSV: não regenera `lists/*.txt` nem os `checkpoints` da fila, que estão fixos no JSON. A seleção que refiz hoje a partir do `search_state.json` já diverge do CSV: ham10000 hybrid B2 → c00043 (0,612 contra 0,577), larvae_raw hybrid B1 → c00026 (0,915 contra 0,863) e um estrato novo, flowers hybrid B3. **Correção:** escolher uma das duas saídas e registrar a escolha. (a) Congelar o CSV atual como está. (b) Gerar o CSV, as listas, a fila e o `--files-from` do rsync com um script só, gravando o md5 de cada artefato no log do A-attn-00. |
| **C4** | B | **coconut_trees hybrid B4 c00012 f5 está como `failed` no `search_state.json`** (`error: params 3541364 > cap 3500000`), embora o `result.json` diga `ok`. O seletor só lê o `result.json` (`select_checkpoints.py:54`). **Correção:** usar o status do job no `search_state` como fonte de verdade (ou cruzar as duas fontes) e excluir ou substituir essa linha, registrando o motivo. |
| C5 | relatório | **As unidades de FLOPs estão misturadas.** O `flops` da busca é **MAC**: `run_search.py:322-362`, `inference_macs`. O `attention_cost` usa **2·MAC**. As frases afetadas são `smoke_cpu_rs.md:54,65` ("O E5 dobra os FLOPs da c00009"), `plano_execucao_E0_E8.md:106` ("28 contra 119 GFLOP") e `viabilidade_e_integracao.md:28,124`. Na mesma unidade, os encoders somam 238,2 GFLOP, contra 28,3 GFLOP dos E1–E5. O E5 da c00009 equivale a 57% do encoder, e não a 2×. Corrigir o critério §5.3 ("≤ 25% do encoder") para a mesma unidade. |
| C6 | A-attn-00 | **Congelamento do código.** O `codigo_md5` da fila não inclui o runner nem o seletor. Acrescentar `tools/eval_fixed_attention.py` = `9a4fcd3ecfc0b5a1deb8432338d4e0e6` e `tools/select_checkpoints.py` = `cc06c220985a89ffdc364f31c7c60277`. Gravar também `torch`, `sklearn`, host e UUID da GPU em cada linha. |
| C7 | A-attn-02 | **Compatibilidade de versão.** Os `encoder.pt` são pickles de módulo inteiro (`weights_only=False`) salvos com torch **2.7.1** na Jaci, e o athena roda torch **2.2.2**. Só a direção 2.2 → 2.7 foi testada (`agent5/infrastructure_inventory_jaci.md:16`), e o smoke cobre apenas 2 checkpoints. **Correção:** carregar os **74** no athena, fazer 1 forward E0 e comparar com a Jaci. Pôr `CUDA_DEVICE_ORDER=PCI_BUS_ID` em todos os comandos de GPU e conferir no `nvidia-smi` que o processo está na GPU-3c72f1c7… (PCI AF:00.0). |
| C8 | relatório | **Riscos do A1c não estão declarados no plano** (ver item 8). Declarar no relatório de teste, com sensibilidade, e identificar a régua do E0 como `ORIG_sub4000`, que difere do ORIG histórico (train inteiro). |

## Evidência por item

### 1. Zero parâmetros treináveis em todos os operadores e genes: **OK**

- **Código.**
  - `_FixedOp` e as subclasses não criam `nn.Parameter` (`fixed_attention.py:249-268`).
  - As estatísticas são `register_buffer` (`:421`).
  - O encoder e a cabeça ficam fora dos submódulos, em `self._net` (`:752-770`).
  - Uma busca por `optim|Parameter|Linear|backward|step` no módulo e no runner dá zero ocorrências de código. A única linha é a docstring `:3`.
- **Suíte do agente.**
  - `test_fixed_attention.py`: **148 passed** (1,84 s). Inclui `test_zero_trainable_parameters` (tipo × alinhamento), `test_cost_reports_zero_params...` e `test_genes_never_change_embedding_dim`.
  - Essa suíte usa uma configuração de genes por tipo. Para cobrir o espaço inteiro, escrevi testes extras.
- **Testes extras do CHECK** (`check_battn/test_extra_battn.py`): **407 passed** (5,0 s). Cobrem:
  - varredura aleatória de 60 sementes × 6 poolings da cabeça sobre **todo o `GENE_SPACE`**: os 6 tipos, `location` ∈ {final, head, 0, 1, 2, −1, 5}, fontes, similaridade, normalização, agregação, `exclude_self`, tokens, α, τ e `align_k`;
  - em cada caso: `parameters()==[]`, nenhum `nn.Linear` nem `Parameter`, buffers com `requires_grad=False`, `cost.trainable_params==0`, dimensão ≥ 1280 inalterada e saída finita;
  - uma análise AST do módulo e do runner, que não encontrou `optim`/`SGD`/`Adam`/`Parameter`/`Linear`/`backward`/`step`/`zero_grad`;
  - `embed()` em `eval()` + `no_grad()`: sem `requires_grad`, e com o encoder e a cabeça em `eval` mesmo depois de terem sido postos em `train()`.
- **Checkpoints reais** (`real_ckpt_check.log`): 7 checkpoints com md5 conferido, entre eles flowers c00002 (1020 canais), larvae B4 (2560-d) e aid B1 (5 blocos).
  - E1–E5: `p=0` e saída finita em todos.
  - Pesos e buffers do SPiFiL, incluindo as running stats do BN, ficam **bit a bit** intocados depois do fit e do forward.
- **Smoke:** todas as 27 linhas têm `attn_trainable_params=0` (`agent6/smoke/*.jsonl`).

### 2. α=0 ⇒ identidade exata; o gradiente chega às ativações: **OK**

- **Código.** `residual_mix` devolve `x` quando α=0 (`:238-245`), e `_skip()` faz o curto-circuito (`:266`).
- **Suíte do agente.**
  - `test_alpha_zero_is_exact_identity_end_to_end` (5 tipos × poolings, com `torch.equal`, inclusive com `force_compute=True`);
  - `test_gradient_reaches_activations`, `test_operator_gradient_includes_attention_path` e `test_pooling_gradient`.
- **Extras:**
  - 40 configurações aleatórias com α=0: `emb(x)` é igual bit a bit a `head(enc(x))`, também com `force_compute`, e `cost.flops == 0`;
  - **o gradiente na entrada é igual bit a bit ao do modelo sem operador**;
  - nas 360 configurações com α qualquer, `x.grad` é finito e diferente de zero.
- **Checkpoints reais.** `E0 == head(encoder(x))` bit a bit nos 7 (tipo `Sequential`; iterar sobre os blocos é o mesmo que chamar o forward).

### 3. Estatísticas ajustadas só no treino; teste só depois do congelamento: **OK no vazamento, FALHA na amostra de ajuste (C1)**

**O que está certo:**
- **Origem dos dados.** O `load_data` só monta train e val, e o hash de identidade também só cobre train e val (`progressive_data.py:119-168`). O `_real_samples` proíbe teste (`:104`).
- **Teste.** Só é lido por `_test_loader` quando `--split test` (`eval_fixed_attention.py:45`), e o runner recusa `--split test` sem `--frozen-from` (`:141-143`). A fila só roda o teste no `B-attn-test-report`, que depende dos 11 jobs B. O caminho `--plumbing-check` usa o loader de validação e reproduziu os κ de validação (0,4603 / 0,4753 / 0,4881; `smoke/plumbing_c00030.jsonl`).
- **Alinhamento.** É ajustado só em `Xtr` (`:92`), sem rótulo (`Alignment.fit`, `:425-458`). O `embed` não reajusta nada: `test_alignment_deterministic_and_train_only` mostra buffers iguais depois de `embed`/`tta8` em val, e que sem fit o forward dá erro.
- **Rótulos de teste.** São usados só para o `np.unique` do remapeamento de rótulos no relato (`:75`), o que é inócuo.
- **Resultados da busca.** Nenhum dos 74 `result.json` e nenhum `search_state` tem campo de teste.
- **Relato de teste.** Ele reajusta PCA/CCA/SVM **no treino** com a mesma seed. É determinístico (fp64 em CPU), e é por isso que o plumbing reproduz exatamente os κ.

**Falha:** as "160 primeiras" imagens do treino são viesadas por classe. Composição de classes dessas imagens:

| base | composição |
|---|---|
| HAM | {1: 160} |
| flowers | {1: 160} |
| larvae | {1: 160} |
| coconut | {1: 160} |
| eggs | {1: 158, 2: 2} |
| aid | {1: 144, 2: 16} |
| protozoan_raw | {1: 93, 2: 67} |
| RS | 4 de 14 classes |

O treino está ordenado por classe nas 8 bases verificadas. Correção em C1.

### 4. SVM `core.svm.fit_svm` com C=100 e protocolo declarado: **OK (com C8)**

- **Chamada.** `fit_svm(Xtr, ytr0, C=100.0, scaler=False, tag="SVM_ATTN6")` (`eval_fixed_attention.py:79`), e `core/svm.py:143`.
- **Protocolo.** Cada linha grava `svm_protocol` = `scaler=none;kernel=linear;C=100;ovo;max_iter=-1` (confirmado nas 27 linhas do smoke e na calibração do eggs: `fit_status=0`).
- **Régua da busca.** Fica identificada à parte em `search_ruler` = `cov_head|C=100|scaler=standard|tta=1` (CSV) e em `kappa_val_search`.
- **Ressalva.** O treino da SVM é o subconjunto estratificado de 4000 imagens. É o mesmo da fitness, e não o train inteiro do ORIG histórico. Rotular como `ORIG_sub4000` (C8).

### 5. FLOPs, memória e latência, inclusive o E5: **OK (unidades: C5)**

- **Custo por linha.** `cost.flops` (2·MAC + softmax e normas), `peak_bytes`, `fitted_values`, `ms_per_image`, `attn_fit_s`, `svm_fit_s`, `tta_cost` e `peak_cuda_gb` (`eval_fixed_attention.py:101-108`; `attention_cost`, `fixed_attention.py:972-1020`).
- **E5.** A fórmula é `3nc + 2c²n + 5c² + 2c²n + 3nc`. Recalculada em flowers c00002, dá 3,273 GFLOP, igual ao `real_ckpt_check`. Em c00009, dá 0,989.
- **E5 contra o encoder, em 2·MAC:**
  - é **maior que o encoder** em 2 de 74 checkpoints: flowers hybrid c00002 (1,98×) e ham lemonade c00002 (1,08×);
  - passa de 25% do encoder em 9 de 74;
  - comparando com o MAC cru (a unidade misturada), seria maior em 5 de 74.
- **Lacuna.** O `profile()` da busca não enxerga o operador; isso já está registrado em `viabilidade_e_integracao.md:100,156`. O custo do E5 vale para o f1 do GA, não para o B.

### 6. Seleção dos checkpoints só pela validação: **OK com C3/C4**

- **Seletor.** O `select_checkpoints.py` lê só `result.json` (com `status=ok`) e `config_resolved.yaml`. O critério é `fitness_kappa`, gravado em `run_search.py:741` a partir de `_svm_quality`/WiSE na validação.
- **Integridade do CSV.** Os 74 md5 do CSV batem com o disco (0 divergências). `fitness_kappa` e `params` são idênticos aos do `result.json`. Os 74 `config_resolved` existem. O CSV, a fila e as listas formam o mesmo conjunto (74 = 74 = 74, md5 iguais).
- **Seleção refeita a partir do `search_state.json`** (`indep_select.py`, 811 jobs ok): **69 de 74 iguais**. Os 5 restantes:
  - **2 empates exatos de κ entre fidelidades:** larvae lemonade B0 c00001 (f100 contra f5, 0,98384) e RS lemonade B1 c00054 (f15 contra f5, 0,56933). O desempate é lexicográfico (`"f100" < "f15" < "f5"`). Isso é aceitável, mas precisa ser declarado.
  - **2 checkpoints novos depois do snapshot** (ham B2 c00043; larvae_raw B1 c00026) e 1 estrato novo (flowers hybrid B3): C3.
  - **1 job `failed` no `search_state`:** coconut B4 c00012 (C4).
- **Ressalva.** A seleção foi feita com a régua COV, e o E0 lê com a ORIG. Está declarado em `plano_execucao_E0_E8.md:14`, mas os checkpoints não são os melhores pela régua ORIG (A1c §0.5).

### 7. Comparação justa: mesmos splits, mesmos checkpoints, TTA fora do fitness: **OK com C1/C2**

- **Mesmos dados.** Uma decodificação por dataset (`dkey = (sha256, split, max_train)`, `:178-182`). As mesmas `Xtr`/`Xev` e o mesmo subconjunto da SVM servem a E0..E8 de cada checkpoint, e a lista é a mesma para a validação e para o teste.
- **Hiperparâmetros fixados a priori** (`E_PRESETS`, `fixed_attention.py:1046-1066`), sem grade.
- **Seleções:**
  - E6–E8 só para o melhor E0–E5 pela validação (`:229-250`);
  - o desempate favorece o mais simples;
  - a linha `selected_by_validation` alimenta o teste.
- **TTA fora do fitness.** As linhas levam `never_feeds_fitness=True` e são gravadas em `exps/attn6/out`. Não há escrita em `artifacts/` nem em `search_state`.
- **Ressalvas:**
  - C1 põe E3/E4 em desvantagem;
  - o reparo `align_k` cai para o menor nº de canais das fontes (E4: 32 → 5 em eggs c00015, → 7 em ham c00002, → 14 no smoke), e isso degenera o E4 em parte dos checkpoints. Está registrado em `repair_notes`; separar esses casos no relatório;
  - o E0 é `head(encoder(x))` exato, conferido bit a bit em checkpoints reais.

### 8. Riscos de vazamento e de régua do A1c (HAM por lesão, recapturas, pre×raw): **FALHA (não declarados, C8)**

O plano e a fila não citam nada disto (grep sem ocorrências de `lesion`, `recaptura`, `A1c` ou `dup`):

- **HAM.** O val é limpo por lesão: train∩val = 0 (A1c §4), então a seleção pela validação vale. O teste do HAM, porém, tem **26,2%** das imagens com a lesão em train∪val. O relato de teste precisa marcar isso.
- **Parasitas.** Há recapturas do mesmo objeto: eggs val↔train 5 e test↔train 7; larvae val↔train 4 e test↔train 2 (limites inferiores). Fazer a sensibilidade excluindo os pares `veredito_visual=mesmo` de `A1c_dup_pairs.csv`.
- **flowers.** Tem uma cópia byte a byte com **rótulo diferente** train↔test, mais 1 par T1. O aid tem 1 par idêntico train↔test.
- **pre × raw.** O plano já trata pre e raw como datasets distintos, o que está certo. Falta declarar que eles nunca se comparam com FM de outro domínio.
- **Impacto no Δκ.** Como o Δκ entre os E é pareado (mesmo checkpoint, mesmas imagens), o vazamento afeta todos os braços. O risco está no κ **absoluto** do relato e em qualquer comparação com FMs.

### 9. A fila não toca na GPU 2 nem na Jaci: **OK (blindagem em C7)**

- **Comandos.** Os 12 comandos de GPU usam `CUDA_VISIBLE_DEVICES=3`. O A-attn-01 usa `CUDA_VISIBLE_DEVICES=''`. O A-attn-00 só faz rsync de leitura a partir da Jaci, sem `--delete` e sem GPU.
- **Regras da fila.** O JSON tem `"gpu2": "NAO MEXER"` e `"jaci": "NAO usar GPU"`, e cada job B usa `flock` por diretório.
- **Ordem das GPUs.** No athena as 4 GPUs são iguais (49140 MiB), então o risco de ordem CUDA ≠ ordem PCI é baixo. Mesmo assim, exigir `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Watchdog.** Ele observa a GPU 3, mas só emite alertas e só religa o monitor da GPU 2 (`watchdog.py:1-10`).
- **Coordenação.** Os ~6 GB × 4 jobs precisam ser coordenados com a fila de A dos Agentes 2/3 na mesma GPU 3.

### 10. Reprodutibilidade: **PARCIAL (C6/C7)**

**OK:**
- **Patch.** O md5 de `fixed_attention.patch` (`3e01844f…`) confere. Aplicado num repositório vazio, ele reproduz exatamente o `fixed_attention.py` (`f26b5c30…`) e o teste (`2974f5d7…`) da cópia.
- **Cópia.** É idêntica à árvore viva em `ga_search/`, `core/` e `flim/`, com duas exceções: `fixed_attention.py`, que é novo, e `watchdog.py`, que é irrelevante.
- **Seeds:**
  - alinhamento com seed=0 e sorteio por imagem, independente do bs (`test_alignment_deterministic_and_train_only`);
  - subconjunto da SVM com a seed da busca;
  - SVM determinística.
- **Smoke.** Duas execuções idênticas em CPU.
- **Retomada.** É idempotente pela linha `selected_by_validation`.

**Faltas:**
- os md5 do runner e do seletor (C6);
- as versões de torch e sklearn por linha (C6);
- a validação do carregamento 2.7 → 2.2 dos 74 checkpoints (C7);
- o cudnn não está em modo determinístico. A tolerância |Δκ| ≤ 0,01 está declarada no A-attn-02.

## Suíte de testes da cópia

| Execução | Resultado |
|---|---|
| `tests/` inteira (é só `tests/modehb`), CPU | **2456 passed, 4 skipped, 0 failed**, 81 s (`check_battn/full_suite.log`). Bate com o relato do Agente 6. |
| `tests/modehb/test_fixed_attention.py` | **148 passed**, 1,84 s |
| Extras do CHECK (`check_battn/test_extra_battn.py`) | **407 passed**, 5,0 s |
| Checkpoints reais (7, CPU, md5 conferido) | E0 bit a bit, E1–E5 com `p=0` e finitos, pesos intocados (`check_battn/real_ckpt_check.log`) |

Diretório dos artefatos do CHECK: `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/check_battn/`
