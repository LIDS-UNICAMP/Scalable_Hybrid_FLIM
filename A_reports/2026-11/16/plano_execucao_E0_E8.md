# Agente 6: plano de execução E0–E8 (athena GPU 3, fila do Agente 5)

> 17/11/2026, **v2**: incorpora as condições C1–C8 da auditoria do CHECK (`check/gate_B_attn_audit.md`), detalhadas no §9.
> - **Fila:** `agent6/fila_gpu3_attention_congelado_20261117_0159_v2.json`, no formato do `agent5/athena_gpu3_research_queue.json`, gerada junto com a lista pelo `tools/freeze_E0_E8.py`.
> - **Proibido:** GPU da Jaci, GPU 2 do athena, processos/tmux em execução e código vivo (`ga_search/modehb/*.py`).
> - **Código:** cópia congelada (scratchpad/attn6 + `agent6/fixed_attention.patch` v2) em `athena:/dados/home/moliveira/exps/attn6/code`. Os md5 de módulo, testes e tools estão em `codigo_md5` da fila e em `MANIFEST_congelado_20261117_0159_v2.md5`.
> - **Unidades:** **FLOPs = 2·MAC** em todo o texto e no código (`fixed_attention.FLOPS_UNIT`). O campo `flops` da busca é **MAC** e aparece na coluna `flops_mac`.

## 1. Checkpoints: lista congelada v2 (os MESMOS em todos os E0–E8)

**Lista de origem.** `agent6/checkpoints_E0_E8_congelado_20261117_0159.csv`, com 75 linhas e md5 `33fbec50…`. O Agente 5 a gerou na fase A e já fez o staging no athena.

**Congelamento v2.** O `tools/freeze_E0_E8.py --from-csv <origem> --tag congelado_20261117_0159_v2` gera de uma vez, com md5 de cada um (log em `freeze_congelado_20261117_0159_v2.log`):
- `checkpoints_E0_E8_congelado_20261117_0159_v2.csv`, com **74** checkpoints;
- `lists_congelado_20261117_0159_v2/<dataset>.txt`;
- `rsync_files_from_congelado_20261117_0159_v2.txt`;
- a fila;
- `MANIFEST_…v2.md5`.

**Fonte de verdade = `search_state.json` (C4).** Cada linha é validada contra o status do job no search_state:
- **coconut_trees hybrid B4 c00012 f5 foi EXCLUÍDO.** O job está `failed` (`params 3541364 > cap 3500000`), embora o `result.json` diga `ok`. Não existe outro candidato `ok` em B4 com `job_time ≤ cutoff`.
- O cutoff é o mtime do CSV de origem: 16/11 22:59:58 na Jaci, que corresponde a 17/11 01:59 no athena.
- Todas as outras 74 linhas têm job `ok` e md5 igual ao do disco.

**Critério de seleção** (`tools/select_checkpoints.py`):
- maior `fitness_kappa`, que é o κ de **validação** da própria busca, na régua **COV** (`cov_head | C=100 | scaler=standard | tta=1`);
- só entram jobs `ok` no search_state **e** no `result.json`, em `seed42`;
- estratos da missão por `params`: B0 ≤300k, B1 ≤800k, B2 ≤1,5M, B3 ≤3M, B4 ≤5,5M;
- o teste nunca é lido.

**Desempate declarado** (não-bloqueante do CHECK): maior κ, depois **maior fidelidade**, depois menor `cand_id`, depois menor job.
- A lista de origem foi feita com o seletor antigo, que desempatava de forma lexicográfica. Isso afeta 2 empates exatos: larvae lemonade B0, onde ficou c00001 f100 (f200 tem o mesmo κ 0,9838), e RS lemonade B1, onde ficou c00054 f15 (f45 tem o mesmo κ 0,5693).
- Para não mexer no congelamento, **mantive as escolhas da lista congelada**, registradas no log. A regra nova vale para os próximos congelamentos.

| dataset | kind | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
| remote_sensing | hybrid | c00009 f5 (133k; 0,591) | c00006 f5 (323k; 0,528) | c00015 f5 (1,48M; 0,570) | c00020 f5 (2,38M; 0,599) | — |
| remote_sensing | lemonade | c00012 f45 (30k; 0,556) | c00054 f15 (736k; 0,569) | c00066 f5 (1,41M; 0,591) | c00028 f5 (1,82M; 0,611) | c00042 f15 (3,15M; 0,562) |
| ham10000 | hybrid | c00025 f5 (170k; 0,586) | c00034 f5 (467k; 0,601) | c00043 f5 (1,35M; 0,612) | c00020 f5 (2,53M; 0,586) | — |
| ham10000 | lemonade | c00012 f5 (17k; 0,514) | c00002 f45 (409k; 0,587) | c00003 f15 (1,31M; 0,547) | c00007 f5 (1,82M; 0,572) | — |
| flowers | hybrid | c00001 f100 (21k; 0,653) | c00002 f200 (684k; 0,748) | c00003 f45 (895k; 0,757) | c00004 f15 (1,89M; 0,742) | — |
| flowers | lemonade | c00001 f45 (182k; 0,702) | c00013 f5 (482k; 0,705) | c00003 f15 (1,31M; 0,757) | c00004 f15 (2,18M; 0,726) | — |
| aid | hybrid | c00025 f5 (213k; 0,819) | c00038 f5 (502k; 0,830) | c00023 f5 (805k; 0,815) | c00020 f5 (2,14M; 0,845) | c00012 f5 (3,35M; 0,768) |
| aid | lemonade | c00001 f15 (148k; 0,706) | — | — | — | — |
| coconut_trees | hybrid | c00009 f5 (81k; 0,714) | c00014 f5 (336k; 0,778) | c00007 f5 (1,48M; 0,790) | c00004 f100 (2,30M; 0,790) | **excluído (C4)** |
| coconut_trees | lemonade | c00001 f45 (201k; 0,736) | c00002 f15 (594k; 0,663) | c00003 f200 (1,34M; 0,795) | c00004 f100 (2,22M; 0,787) | — |
| eggs | hybrid | c00025 f5 (133k; 0,904) | c00006 f5 (753k; 0,941) | c00019 f5 (863k; 0,934) | c00024 f5 (1,85M; 0,945) | — |
| eggs | lemonade | c00001 f45 (185k; 0,899) | c00009 f5 (753k; 0,937) | c00014 f5 (931k; 0,915) | c00015 f5 (2,02M; 0,918) | — |
| eggs_raw | hybrid | c00025 f5 (133k; 0,886) | c00030 f5 (431k; 0,880) | c00023 f5 (881k; 0,893) | c00004 f200 (1,70M; 0,896) | — |
| larvae | hybrid | c00021 f5 (120k; 0,935) | c00026 f5 (349k; 0,950) | c00027 f5 (1,39M; 0,967) | c00004 f15 (1,97M; 0,950) | c00024 f5 (3,49M; 0,967) |
| larvae | lemonade | c00001 f100 (201k; 0,984) | c00009 f5 (797k; 0,952) | c00006 f5 (1,35M; 0,950) | c00007 f5 (1,97M; 0,902) | — |
| larvae_raw | hybrid | c00021 f5 (120k; 0,785) | c00026 f5 (349k; 0,915) | c00023 f5 (885k; 0,884) | c00004 f45 (1,97M; 0,915) | c00024 f5 (3,49M; 0,863) |
| protozoan | hybrid | c00021 f5 (137k; 0,773) | c00006 f5 (333k; 0,897) | c00003 f100 (875k; 0,890) | c00020 f5 (2,64M; 0,836) | c00012 f5 (3,04M; 0,835) |
| protozoan_raw | hybrid | c00005 f5 (55k; 0,681) | c00006 f5 (775k; 0,776) | c00003 f100 (1,31M; 0,797) | c00004 f200 (1,81M; 0,785) | — |

Formato das células: candidato, fidelidade (params; κ val da busca, régua COV).

**Fora da lista:** seed43, buscas `*_C01_*` (outra régua), semeadas e PNAS-CS.

## 2. Experimentos

| Exp | Operador (`fixed_attention.E_PRESETS`) | TTA |
|---|---|---|
| E0 | nenhum: `head(encoder(x))`, a cabeça original do checkpoint | não |
| E1 | `fixed_pool`, query GAP, cosseno, τ=0,1, α=0,5 | não |
| E2 | `fixed_self`, 49 tokens, cosseno, τ=0,1, α=0,25 | não |
| E3 | `fixed_cross`, L_final ← L_final−1, `svd` (CCA, k=32), 49 tokens, α=0,25 | não |
| E4 | `fixed_multiscale`, até 3 camadas anteriores + a própria, `svd` k=32, α=0,25 | não |
| E5 | `fixed_covariance`, correlação entre canais (`zscore`), τ=0,1, α=0,25 | não |
| E6 | melhor de E0–E5 **pela validação** | TTA4 |
| E7 | idem | TTA8 |
| E8 | idem | multiscale (0,75 / 1 / 1,25) |

- Os hiperparâmetros são fixados *a priori*, sem grade.
- **Geometria do TTA:** `rot_flip` em RS, AID, coconut, HAM e parasitas; `flip_scale` em flowers.
- **Agregação do TTA:** média uniforme.
- **align_k degenerado** (não-bloqueante do CHECK): quando o reparo reduz `align_k` a ≤ 5 dimensões, a linha sai com `align_degenerate=true` e a nota `DEGENERADO`. Esses casos vão para uma tabela **à parte** e não entram na conta de aprovação do E3/E4.
  - Na lista v2, o E3 não tem nenhum caso.
  - O E4 tem 5 casos, todos por um bloco raso de 1–5 canais: eggs lemonade B3 (k=5), flowers lemonade B1 (k=5), larvae hybrid B0 (k=2), larvae_raw hybrid B0 (k=2) e protozoan hybrid B0 (k=1). Fonte: `custo_E1_E5_2mac.csv`.

## 3. Protocolo

1. **Dados.** Uso o `config_resolved.yaml` da busca do checkpoint, com o mesmo split, `split_sha256` (conferido athena = Jaci no A-attn-02), pct e resolução 224. Precisão **fp32, TF32 desligado** (`allow_tf32=False`); a fitness usa AMP, e isso fica registrado.
2. **Régua do E0 = `ORIG_sub4000`** (C8). O SVM é treinado no subconjunto **estratificado** de `max_train`=4000 que a fitness usa (`run_search._stratified_indices`, com a seed da busca).
   - Isso **difere do ORIG histórico** (monitor, R2), que usa o train inteiro. Cada linha grava `ruler`, `n_train` e `n_train_total`.
   - Em RS, flowers e larvae o train tem menos de 4000 imagens; nesses casos `ruler=ORIG_full`.
   - A validação é sempre completa.
3. **SVM.** `core.svm.fit_svm`, linear, C=100, ovo, sem scaler. Métricas: κ (`core.metrics.compute_metrics`) e F1 macro.
4. **Ajuste do alinhamento (C1).** PCA, CCA e z-score do E3/E4 são ajustados em **160 imagens sorteadas de modo uniforme** no subconjunto de treino.
   - O sorteio é `fixed_attention.fit_sample_indices`: uma permutação com seed fixa (`--fit-seed 0`), **sem rótulo**. Antes eram as 160 primeiras de um treino ordenado por classe.
   - Usa 256 posições por imagem.
   - Cada linha grava `align_fit_indices`, `align_fit_ids_md5`, `align_fit_seed` e, só como diagnóstico, `align_fit_n_classes`.
   - O ajuste é reutilizado em validação e teste.
5. **Predições por imagem (C2).** Cada linha, de validação ou de teste, grava `preds/<hash>_<exp>_<split>.npz` com `sample_id`, `y` e `pred`, mais `preds_md5`.
   - O `tools/paired_bootstrap.py` calcula o Δκ pareado contra o E0, com IC95 por 1000 reamostras das imagens, **sem refazer a GPU**.
   - `--exclude-ids` roda a sensibilidade sem as imagens suspeitas (§8).
6. **Seleção.**
   - O E6–E8 parte do melhor E0–E5 **pela validação**, com desempate a favor do operador mais simples.
   - O vencedor final também sai da validação, gravado como `selected_by_validation=true`.
   - Os checkpoints foram escolhidos pela **régua COV**, e o E0–E8 lê na régua ORIG (C8). Eles **não** são os melhores pela ORIG (A1c §0.5). O relatório declara isso, e qualquer comparação de "melhor modelo" usa a régua de seleção explícita.
7. **Teste.**
   - Só roda no `B-attn-test-report`, **depois** de todos os B e com a seleção congelada.
   - Por checkpoint, relata E0, o vencedor sem TTA e o vencedor congelado, cada um com `used_for_selection=false` e `reporting_only=true`.
   - Nada é reajustado com dados de teste. O runner recusa `--split test` sem `--frozen-from`.
8. **Ambiente por linha (C6).** Cada linha grava:
   - `env.torch`, `env.sklearn`, `env.numpy`, `env.python` e `env.host`;
   - `env.gpu_uuid`, `env.gpu_pci_bus_id` e `env.gpu_physical_index` (via nvidia-smi, na ordem PCI);
   - `env.cuda_device_order` e `env.cuda_visible_devices`;
   - `env.code_md5` do módulo e do runner, e o `checkpoint_md5`.
9. **Custo, em 2·MAC (C5).** Cada linha grava:
   - `cost.flops` do operador e `cost.flops_unit="2*MAC"`;
   - `cost.encoder_macs`, com a mesma metodologia do `profile()` da busca (bate com o `flops` da busca nos 74);
   - `cost.encoder_flops_2mac`, `cost.attn_over_encoder` e `cost.attn_within_budget` (≤ 25%);
   - `peak_bytes`, `ms_per_image`, `attn_fit_s`, `svm_fit_s` e `tta_cost`;
   - parâmetros treináveis do operador, sempre 0.

## 4. Fila (resumo de `fila_gpu3_attention_congelado_20261117_0159_v2.json`)

**Todo comando de GPU** usa `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3` (C7). Cada job B traz um `verificacao_gpu`: `nvidia-smi --query-compute-apps=pid,gpu_uuid | grep GPU-3c72f1c7` (PCI AF:00.0).

| job | fase | depende de | o que faz | estimativa |
|---|---|---|---|---|
| A-attn-00 / 01 / 02 | A | — | **completed** (agent5 `attn_fase_A`): staging dos 75, suíte no torch 2.2.2, `split_sha256`, smoke na GPU 3 (\|Δκ\| GPU×CPU ≤ 0,0072) | — |
| A-attn-00b | A | — | delta: código do patch v2 → `exps/attn6/code`, conferindo `codigo_md5`. Nenhum checkpoint novo: os 74 já estão no staging | 0,1 h |
| A-attn-01b | A | 00b | suíte no athena com o patch v2, em CPU | 0,05 h |
| **A-attn-03** | A | 01b | **C7**: carrega os **74** no torch 2.2.2, faz forward E0 numa entrada sintética fixa e compara com a referência da Jaci (torch 2.7.1, `compat_ref_congelado_20261117_0159_v2.npz`). Critério: cos ≥ 0,9999 e erro relativo L2 ≤ 1e-3, TF32 off; confere o UUID da GPU 3 | 0,1 h |
| B-attn-E0E8-\<dataset\> ×11 | B (gate A) | A-attn-03 | E0–E8 na validação, com preds por imagem | 0,1–0,8 GPU-h cada |
| B-attn-test-report | B | os 11 | relato do teste da seleção congelada, com as ressalvas do §8 | ~2 h |

**Custo do B:**
- ≈ 4,8 GPU-h, a ~2,5 ms por imagem·passe (21,1 passes por checkpoint). O A-attn-02 mediu 57 s de parede para 2 checkpoints × 9 experimentos no RS.
- SVM ≈ 2,6 h de CPU (medido: 13,9 s para 3997 × 1280).
- Parede ≈ **2,5 h com 4 jobs em paralelo**, cerca de 6 GB cada (pico medido no A-attn-02: 5,1 GB).

**FLOPs do operador** (2·MAC, `custo_E1_E5_2mac.csv`):
- Nos 74 checkpoints, E1–E5 somam 30,7 GFLOP por imagem, contra **235,4 GFLOP** dos encoders com a cabeça.
- As medianas da razão operador/encoder são: E1 0,001; E2 0,008; E3 0,007; E4 0,007; E5 0,053.

## 5. Critério de aprovação de operador (gate B → genes / 2ª ablação)

Um operador fica **aprovado para B0–B4** se:
1. **Ganho:** Δκ_val ≥ +0,01 com **IC95 do bootstrap pareado > 0** (`paired_bootstrap.py`), em **≥ 2 dos 3 datasets prioritários** (RS, HAM, flowers).
2. **Sem regressão:** nenhuma regressão < −0,01 na mediana dos demais.
3. **Custo:** **FLOPs(operador) ≤ 25% de 2·MAC(encoder + cabeça)**, na mesma unidade (`attn_within_budget`), avaliado por checkpoint.
   - **E5 marcado (C5):** passa de 25% em **10/74** e passa de 100% em **2/74**:

     | checkpoint | E5 / encoder |
     |---|---|
     | flowers hybrid B1 c00002 f200 | **1,98** |
     | ham10000 lemonade B1 c00002 f45 | **1,08** |
     | coconut hybrid B0 c00009 f5 | 0,74 |
     | RS hybrid B0 c00009 f5 | 0,56 |
     | larvae hybrid B1 c00026 f5 | 0,54 |
     | larvae_raw hybrid B1 c00026 f5 | 0,54 |
     | flowers lemonade B1 c00013 f5 | 0,38 |
     | RS lemonade B0 c00012 f45 | 0,35 |
     | eggs lemonade B3 c00015 f5 | 0,26 |
     | larvae hybrid B2 c00027 f5 | 0,26 |

   - Nesses checkpoints o E5 conta como "fora do orçamento", mesmo que ganhe κ.
   - E1–E4 ficam abaixo de 25% em todos os 74.

O relatório separa backbones rasos (2–3 blocos) de profundos (≥5), e separa os E3/E4 degenerados (§2). O TTA é avaliado à parte e nunca entra no fitness arquitetural.

## 6. Depois do B: 2ª ablação (operador dentro do AE local e do LeJEPA+SIGReg)

Só com operadores aprovados (no máximo 2) e com gate B.

**Desenho:** arquitetura-semente B1 do A2, profundidade 4, em RS e flowers, com `protect=lowlr`.

**Braços:**
- (i) sem operador;
- (ii) operador só pós-treino;
- (iii) operador durante o AE/SSL, com α=0,25;
- (iv) operador presente com α=0, controle exato.

**Escala:** 3 seeds, fidelidade f45, 24 runs, ≈ 12 GPU-h. Mesmo protocolo: ORIG, C=100, seleção na validação, preds por imagem.

## 7. Depois: genes no GA (Agente 4)

- Segmento opcional `arch.head.attention`, cortado aos operadores aprovados, com α ∈ {0; 0,25; 0,5} e τ ∈ {0,05; 0,1; 0,25}.
- `type=none` mascara o resto no `cand_key`.
- `repair_attention_genes` faz o reparo.
- O f1 soma `attention_cost().flops` (2·MAC) a 2·MAC do encoder, **na mesma unidade**. Hoje o `flops` da busca é MAC e precisa ser convertido.
- Mutação neutra `attention_insert(α=0)`.
- Validação R9 antes da produção.

## 8. Riscos de régua e de vazamento (A1c, C8). Declarados no relatório de teste

O Δκ entre os E é **pareado** (mesmo checkpoint, mesmas imagens, mesmo SVM). Por isso os riscos abaixo atingem todos os braços da mesma forma. Eles comprometem o **κ absoluto** e qualquer comparação com FMs ou com outras réguas.

- **HAM10000.**
  - O val está limpo por lesão (train∩val = 0), então a seleção é válida.
  - O **teste** (`split1_modehb`) tem **26,2%** das imagens (1312 de 4274/5008) com a lesão também em train ou val. Há ainda 1 duplicata com `lesion_id` diferente: ISIC_0025226 = ISIC_0030074 (A1c §4).
  - O κ de teste do HAM sai marcado como **otimista**. A recomendação do A1c é criar o `split1_lesion.json`, fora deste plano.
- **Recapturas em eggs e larvae** (pre). São pares do mesmo objeto entre splits, com `veredito_visual=mesmo` em `agent1/parts/A1c_dup_pairs.csv`, e os números são limites inferiores:
  - eggs: val↔train ≥5, test↔train ≥7;
  - larvae: val↔train ≥4, test↔train ≥2.
  - **Sensibilidade obrigatória:** κ e Δκ recalculados com `paired_bootstrap.py --exclude-ids`, sem as imagens desses pares, na validação e no teste. Isso dispensa GPU, graças às preds por imagem.
  - No protozoan pre a questão é indeterminada (A1c).
- **Duplicatas em flowers e aid.**
  - flowers: 1 cópia byte a byte train↔test **com rótulo diferente**, mais 1 par T1.
  - aid: 1 par idêntico train↔test.
  - Entram na mesma sensibilidade.
- **pre × raw.** Os datasets pre (fundo preto) e raw são tratados como distintos. Um κ de pre ou raw **só se compara com FM avaliado no mesmo domínio**: os κ de FM disponíveis para parasitas são em raw.
- **Régua.** O E0 é `ORIG_sub4000`, não o ORIG histórico (train inteiro). A seleção dos checkpoints foi feita pela régua COV (§3.6). Os κ do E0 não substituem a tabela ORIG da apresentação sem essa identificação.
- **n=1 seed por checkpoint.** Daí o IC bootstrap obrigatório.

## 9. Status das condições do CHECK

| # | Status | Onde |
|---|---|---|
| C1 | **feito** | `fixed_attention.fit_sample_indices` e `FixedAttentionEmbedder.fit(indices=…)`; runner grava índices, md5 dos ids e seed; testes `test_fit_sample_is_uniform_seeded_not_first_n` e `test_embedder_fit_uses_uniform_sample_and_records_indices`; smoke v2 refeito |
| C2 | **feito** | runner: `preds/*.npz` (sample_id, y, pred) com md5 em toda linha de val e teste; `tools/paired_bootstrap.py` |
| C3 | **feito** | `tools/freeze_E0_E8.py`: CSV, listas, `--files-from`, fila e MANIFEST num passo só, com md5 (`freeze_…v2.log`) |
| C4 | **feito** | `select_checkpoints.py` usa o search_state como verdade; coconut hybrid B4 c00012 excluído (sem substituto ok até o cutoff) |
| C5 | **feito** | `FLOPS_UNIT="2*MAC"`, `encoder_macs`, `cost_ratio`; documentos corrigidos; E5 > 25% marcado (§5) |
| C6 | **feito** | `codigo_md5` da fila com runner, seletor, freeze, compat e bootstrap; torch, sklearn, host e UUID/PCI por linha |
| C7 | **feito no código; falta rodar no athena** | `CUDA_DEVICE_ORDER=PCI_BUS_ID` em todos os comandos; `tools/ckpt_compat_check.py` + referência da Jaci; job A-attn-03 |
| C8 | **feito** (este §8 e §3.2/§3.6) | — |
| não-bloqueantes | **feitos** | align_k ≤ 5 separado (§2); desempate entre fidelidades declarado (§1) |
