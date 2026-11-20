# Agente 2 — status da implementação (driver de baseline + interface + SPiFiL-0 + integração S1/S2/S3)

> 17/11/2026. A implementação está numa cópia: `scratchpad/a2impl` (rsync sem artifacts/wandb/.git; `artifacts/` = symlinks para o vivo, só leitura).
> A árvore viva não foi editada. Não usei GPU: todos os testes foram em CPU (`CUDA_VISIBLE_DEVICES=""`).
> Patch único: `research/spifil_b0b4/agent2/a2impl.patch` (md5 em `a2impl.patch.md5`). Fila: `research/spifil_b0b4/agent2/fila_gpu3_a2.json`.
> **Nenhum número deste documento é resultado científico.** São testes de ponta a ponta em f1 com 62 imagens de treino e 34 de validação, que só provam que o código roda.

## 1. O que o patch contém (só arquivos NOVOS; nada de produção é alterado)

| arquivo | autor | conteúdo |
|---|---|---|
| `ga_search/abstract_init/base.py` | A2 | Interface `KernelInitializer` com `fit` e `init_block`, além de `TrainData`, `InitContext`, `InitBudget` e `ParentInfo`. Funções `conv_param_keys`, `validate_init` e `assemble_block` (que não consome o RNG global), `make_provenance`, `check_train_only`, `check_prefix_unchanged` e `supports`/`supported_convs`. Registro de métodos. `InfeasibleCandidate` = o de `flim.progressive`. |
| `ga_search/abstract_init/spifil0.py` | A2 | **S0** `spifil0`: chama `init.init_block` e devolve o mesmo bloco, bit a bit. **S0+** `s0plus`: as opções `whiten`/`patch_norm`/`knn_smooth`/`select_diverse` do `GrowOneLayer` existente, mais a dobra ZCA/escala de `flim.spifil.grow`, por troca temporária de classe restaurada no `finally`. **S0-data** `s0data`: ipc = ceil(256/n_classes), com os doadores aninhados do pool. |
| `ga_search/abstract_init/genomes.py` + `genomes_b1.json` | A2 | Genomas fixos, exatos do `search_state`: `rs_b1` (c00009), `rs_b1_band` (c00054, opcional), `flowers_b1` (c00002) e `seed_small` (topologia L0 da semente SPiFiL pequeno, sem importar pesos). |
| `ga_search/abstract_init/run_ablation.py` | A2 | Driver de baseline e ablação (§2). |
| `ga_search/abstract_init/spifil1.py` | A2b | SPiFiL-1, aplicado sem mudança do `spifil1.patch` (md5 7753ae8d…). |
| `ga_search/abstract_init/spifil2.py` | A2 (S2) | SPiFiL-2, aplicado sem mudança do `spifil2.patch` (md5 cae300a8…). |
| `ga_search/abstract_init/spifil3.py` | A2 (S3) | SPiFiL-3 / SPiFiL-3-rand, aplicado sem mudança do `spifil3.patch` (md5 f00e929b…). |
| `tests/abstract_init/test_base_spifil0.py`, `test_run_ablation.py` | A2 | Testes da interface, do S0/S0+/S0-data e do runner. |
| `tests/abstract_init/test_spifil{1,2,3}.py` | A2b/S2/S3 | Testes dos respectivos autores. |

A interface publicada às 23:10 (`agent2/interface/base.py`, md5 ead81954…) ganhou três adições compatíveis. Os patches S1/S2/S3 rodam sem mudança sobre a versão final.
- `TrainData.donor_pool`;
- `supports()`/`supported_convs`;
- `fork_rng` em `conv_param_keys` e na montagem.

## 2. Driver: `python -m ga_search.abstract_init.run_ablation`

- **Pipeline real.** Usa o objeto `run_search.Evaluator` de produção, sem modificação: init → AE local (Fase B) → LeJEPA+SIGReg (Fase C) → WiSE α ∈ {0, .25, .5, .75, 1} → SVM. O config é `ga_search/config/modehb/search_v3.yaml`, que reproduz o `config_resolved.yaml` das buscas hybrid C100 a menos de output_dir, device e gpu_cap.
- **Braços.**
  - `--inits`: `spifil` (S0 de produção), `random` (gêmeo He com a topologia EFETIVA do SPiFiL do mesmo genoma/seed/AE, `--random-mode twin`), mais qualquer método registrado, com opções `nome:k=v,k=v`. Exemplos: `spifil1`, `s0plus`, `s0data`, `spifil2:variant=s2_noinv`, `spifil2:variant=s2_aug_domain,aug_policy=rs`, `spifil3`, `spifil3_rand`.
  - `--ae`: `noae` = `ae_mode off`, com as mesmas épocas totais no SSL; `joint`; `decoder_first`.
  - `--seeds` (padrão 42,1042,2042): seed do genoma (SPiFiL, AE, cabeça, SSL) e do embaralhamento do treino. Split, pool de doadores e subconjunto do SVM ficam fixos: são a régua.
  - `--protect {full,lowlr,freeze}` sobrescreve o protect do genoma e entra no nome do run.
- **Réguas, separadas.**
  - `kappa_val_ORIG`: `head(encoder(x))` (dim ≥ 1280; a dimensão medida vai em `embed_dim_orig_measured`), `core.svm.fit_svm` linear, C=100, sem scaler.
  - `kappa_val_COV`: cov_head + StandardScaler, C=100.
  - O α do WiSE é escolhido pela ORIG (`--wise-ruler orig`, a fitness nova da missão). A COV é medida no α escolhido; em todos os α só com `--other-ruler-all-alphas`.
  - `kappa_val_ORIG_alpha0` = SPiFiL(+AE) sem SSL.
  - O teste não é carregado: `progressive_data.load_data` só devolve treino e validação, e o registro traz `test_read: false`.
- **Inicializadores abstratos.** Durante o run, `ga_search.modehb.init.init_block` é trocado por um despachante, restaurado no `finally`. O despachante monta `TrainData` (doadores, pool, `loader_factory` do subconjunto estratificado de TREINO do SVM; y = −1 para os métodos label-free) e `InitContext`, com `InitBudget(max_seconds=--init-max-seconds, padrão 1800)`.
- **Aplicabilidade.** Se o método não suporta algum bloco do genoma (S3: só conv/pointwise), o run vira `not_applicable` sem treinar. Um `InfeasibleCandidate("...unsupported...")` também vira `not_applicable`.
- **Saída e retomada.**
  - `artifacts/a2_ablation/<ds>/results.jsonl` (append com flock + fsync) e `candidates/<run>/f45/{encoder.pt, result.json, val_embedding.pt, ablation.json}`.
  - Um run com `ablation.json` está feito: status ok, infeasible ou not_applicable.
  - Um diretório sem `ablation.json` (run morto) é renomeado para `*.interrompido_<ts>`, sem apagar nada, e refeito com as mesmas seeds.
  - `failed` não é definitivo: relançar tenta de novo.
  - OOM: até 10 retries com 300 s de espera (como a R5).
- **Ambiente.** OPENBLAS=2, OMP=8 e MKL=8 são forçados antes de importar numpy/torch (`--keep-thread-env` desliga). `--cov-eigh-workers 1` serve para o torch 2.2 do athena.

## 3. Testes

- **Unitários** (`tests/abstract_init`, 166 passed em CPU, ~13 s; inclui 63 do S1, 35 do S2 e 20 do S3):
  - shapes de 9 tipos de bloco (conv k3/5/7, pointwise, dwsep, inception, depthwise, residual identity/project);
  - `validate_init` rejeita chaves, shapes, dtype e NaN errados;
  - a montagem não consome o RNG global;
  - prefixo intocado, e a mutação do prefixo vira `InfeasibleCandidate`;
  - o gêmeo aleatório tem a mesma estrutura;
  - **S0 bit a bit igual a `init.init_block`** (4 tipos × camada 0/1, inclusive o estado do RNG global depois);
  - determinismo;
  - S0+ restaura o `GrowOneLayer`, dobra o ZCA exatamente (1e-10) e, com tudo desligado, fica bit a bit igual ao S0;
  - S0-data: ipc e doadores aninhados;
  - **label-free**: o despachante entrega y = −1 e as imagens são invariantes a permutação dos rótulos; o S1 dá pesos idênticos com rótulos permutados;
  - runner: plano (spifil antes do gêmeo), parsing, réguas R1, fórmula do WiSE, quarentena, jsonl e protect.
- **Suíte inteira:** `tests/` = **2474 passed, 4 skipped**, 0 falhas (91 s), numa cópia LIMPA da árvore viva com o patch aplicado por `git apply` (`scratchpad/a2check`). Os arquivos aplicados são idênticos aos testados (cmp).
- **Ponta a ponta** (CPU, f1, `--debug-train 70 --debug-val 42`; o RS usa 62 imagens de treino após a estratificação; o Flowers usa debug 40/30): um run por braço. Tabela na §4.
  - **kill -9 e retomada.** O grid foi morto no meio do 2º run e relançado. O run 1 foi pulado, o run morto foi movido para `rs_b1__spifil0__joint__s42/f1.interrompido_20261116_232429` e os 11 seguintes rodaram. O κ do run 1 é idêntico entre dois processos diferentes (t1 e t2): ORIG 0,286940, COV 0,335288, α 0,5.
  - **S0 bit a bit no pipeline completo.** `spifil` × `spifil0` (joint, s42): encoder, cabeça e âncora WiSE idênticos tensor a tensor, com os mesmos κ.
  - **Reprodução da produção.** `rs_b1` tem params 132 620 e larguras efetivas 70-280, iguais ao c00009 (`result.json` de produção).
  - **S3** com o padrão de 300 passos: `InfeasibleCandidate("budget: max_seconds")` no passo 69, em CPU com carga. Por isso existe `--init-max-seconds` (padrão 1800 na fila). Com `spifil3:steps=20`, S3 e S3-rand rodam ok.
- **torch 2.2.** Não há env com torch 2.2 na Jaci (os disponíveis são 2.7.1 e 2.4.1), então não deu para testar o import. A varredura estática do módulo não encontrou APIs > 2.2 (`torch.amp.*`, `add_safe_globals`, `rms_norm`, `torch.compile`, `nn.attention`). Ainda assim, o job `a2_B0_smoke_gpu` da fila (3 runs f5) existe para validar no athena antes do resto.

## 4. Runs de ponta a ponta (f1, poucas imagens — só validação de código, NÃO comparar)

| teste | run | status | κ val ORIG | κ val COV | α WiSE (ORIG) | κ ORIG α=0 | larguras efetivas | params | s/run |
|---|---|---|---|---|---|---|---|---|---|
| t1 | `rs_b1__spifil__joint__s42` | ok | 0.2869 | 0.3353 | 0.5000 | 0.2517 | 70-280 | 132620 | 135.0341 |
| t2 | `rs_b1__random__joint__s42` | ok | 0.3894 | 0.4661 | 0.2500 | 0.3200 | 70-280 | 132620 | 98.9801 |
| t2 | `rs_b1__s0data-images_total112__joint__s42` | ok | 0.2517 | 0.3034 | 0.0000 | 0.2517 | 70-280 | 132620 | 267.6606 |
| t2 | `rs_b1__s0plus__joint__s42` | ok | 0.1912 | 0.3710 | 1.0000 | 0.1850 | 70-280 | 132620 | 181.6717 |
| t2 | `rs_b1__spifil0__joint__s42` | ok | 0.2869 | 0.3353 | 0.5000 | 0.2517 | 70-280 | 132620 | 179.7601 |
| t2 | `rs_b1__spifil1__joint__s42` | ok | 0.3372 | 0.3034 | 0.2500 | 0.2437 | 73-287 | 137702 | 121.6228 |
| t2 | `rs_b1__spifil2-aug_policyrs-variants2_aug_domain__joint__s42` | ok | 0.3698 | 0.3007 | 0.2500 | 0.2724 | 73-287 | 137702 | 228.7049 |
| t2 | `rs_b1__spifil2-variants2__joint__s42` | ok | 0.2717 | 0.2288 | 1.0000 | 0.2422 | 73-287 | 137702 | 255.6350 |
| t2 | `rs_b1__spifil2-variants2_dcpen__joint__s42` | ok | 0.3487 | 0.2972 | 0.0000 | 0.3487 | 73-287 | 137702 | 224.0642 |
| t2 | `rs_b1__spifil2-variants2_noinv__joint__s42` | ok | 0.3654 | 0.3034 | 0.0000 | 0.3654 | 73-287 | 137702 | 251.1462 |
| t2 | `rs_b1__spifil3__joint__s42` | not_applicable | — | — | — | — | — | — | — |
| t2 | `rs_b1__spifil3_rand__joint__s42` | not_applicable | — | — | — | — | — | — | — |
| t2 | `rs_b1__spifil__joint__s42` | ok | 0.2869 | 0.3353 | 0.5000 | 0.2517 | 70-280 | 132620 | 133.7219 |
| t3 | `rs_b1__random__decoder_first__s42` | ok | 0.3894 | 0.4661 | 0.2500 | 0.3200 | 70-280 | 132620 | 150.1398 |
| t3 | `rs_b1__random__noae__s42` | ok | 0.3871 | 0.4366 | 1.0000 | 0.3788 | 70-280 | 132620 | 108.5665 |
| t3 | `rs_b1__spifil__decoder_first__s42` | ok | 0.2869 | 0.3353 | 0.5000 | 0.2517 | 70-280 | 132620 | 186.3271 |
| t3 | `rs_b1__spifil__noae__s42` | ok | 0.2876 | 0.3061 | 1.0000 | 0.2214 | 70-280 | 132620 | 148.4298 |
| t4 | `seed_small__spifil1__joint__s42` | ok | 0.1699 | 0.2945 | 0.0000 | 0.1699 | 40-80-152-152-152-312 | 1383536 | 68.7012 |
| t4 | `seed_small__spifil3__joint__s42` | infeasible | — | — | — | — | — | — | — |
| t4 | `seed_small__spifil3_rand__joint__s42` | ok | 0.3320 | 0.3579 | 0.0000 | 0.3320 | 40-80-152-152-152-312 | 1383536 | 63.0840 |
| t5 | `flowers_b1__random__joint__s42` | ok | 0.4667 | 0.6439 | 1.0000 | 0.2739 | 15-50-135-1020 | 683680 | 39.1256 |
| t5 | `flowers_b1__spifil2-aug_policyflowers-variants2_aug_domain__joint__s42` | ok | 0.4615 | 0.6433 | 0.0000 | 0.4615 | 15-54-136-1024 | 694610 | 92.7742 |
| t5 | `flowers_b1__spifil__joint__s42` | ok | 0.3718 | 0.6006 | 0.2500 | 0.3237 | 15-50-135-1020 | 683680 | 496.7067 |
| t6 | `seed_small__spifil3-steps20__joint__s42` | ok | 0.2938 | 0.3553 | 0.0000 | 0.2938 | 40-80-152-152-152-312 | 1383536 | 267.3769 |
| t6 | `seed_small__spifil3_rand-steps20__joint__s42` | ok | 0.3320 | 0.3579 | 0.0000 | 0.3320 | 40-80-152-152-152-312 | 1383536 | 59.6659 |
| t7 | `rs_b1__random__joint__s42__plowlr` | ok | 0.3894 | 0.4661 | 0.2500 | 0.3200 | 70-280 | 132620 | 136.2696 |
| t7 | `rs_b1__spifil__joint__s42__plowlr` | ok | 0.2588 | 0.3041 | 0.2500 | 0.2517 | 70-280 | 132620 | 158.1548 |

Observações (código, não ciência):
- `spifil`, `spifil0` e `spifil__decoder_first` dão κ idênticos (S0 bit a bit; em f1 o decoder_first não tem época de aquecimento).
- O gêmeo `random` herda as larguras EFETIVAS do SPiFiL (70-280; Flowers 15-50-135-1020).
- **S1/S2 constroem a largura PEDIDA** (73-287, 137 702 params), enquanto o S0 encolhe pelo alocador (70-280, 132 620 params). O mesmo vale para o `random` gêmeo do S0. A comparação S1/S2 × S0 tem essa diferença de ~4 % de params: registrar na análise.
- `seed_small__spifil3` com 300 passos estourou os 600 s por camada em CPU (infeasible). Com `steps=20`, ok (t6). Na fila vale `--init-max-seconds 1800`.
- t7 = código FINAL (runner md5 cb484fd1…): `--protect lowlr` ok, `embed_dim_orig_measured` = 1280.


## 5. Achados de implementação que afetam a interpretação

1. **`protect=freeze` nos dois genomas B1** (c00009 RS e c00002 Flowers).
   - No braço SPiFiL, o lr das convs é 0 no AE e no SSL. Só as normas (afim de BN/DBN), a cabeça e o decoder treinam.
   - No braço aleatório, o AE usa lr 1 nas convs, mas o SSL também as congela (o `p_scale` do SSL não olha o init). É a semântica de produção.
   - Por isso a fila tem o braço `--protect lowlr` (só init=spifil, pedido do coordenador).
2. **joint × decoder_first.** Em f1 os dois são bit a bit iguais, porque `n_warm = round(0,5 × 1) = 0`. **Em f45 são diferentes**: n_warm = 8 de 15, e as normas treinam com lr cheio. A fila mantém os dois. (A minha 1ª mensagem ao coordenador dizia o contrário; já foi corrigida.)
3. **O B1 de RS é um modelo B0.** O "melhor ≤ 800k" do A1a (c00009, 133k) é o mesmo do B0. O melhor dentro da faixa 300–800k na Jaci é o LEMONADE c00054 (736k), disponível como `rs_b1_band`; ficou fora da fila.
4. **Os genomas B1 foram escolhidos pela fitness antiga** (cov_head + scaler). A régua ORIG desses genomas só passa a existir com este baseline.
5. **S3 não se aplica a `rs_b1` nem a `flowers_b1`** (dwsep). Na fila, o S3 só roda na topologia `seed_small` (só conv).
6. **S0-data em RS** com 256 imagens dá ipc 19, N ≈ 26 600 e matriz de Fisher N×N ≈ 2,8 GB (fp32). Em Flowers, o genoma já usa ipc 32 (160 imagens), então ipc = ceil(256/5) = 52. O pool foi ampliado para 52 de forma aninhada: os 32 primeiros doadores não mudam.
7. **S2 `s2_aug_domain`** usa `aug_policy=rs` no RS e `aug_policy=flowers` no Flowers: só hflip, sem vflip. É variante registrada, não verdade.

## 6. Estimativas (fila `fila_gpu3_a2.json`)

Bases medidas em produção (`result.json`, Jaci, torch 2.7):
- RS c00009 f5: 0,094 h e pico de **34,3 GB alocados / 35,9 GB reservados**. Esse pico era com a fitness cov_head nos 5 α; aqui o cov roda uma vez só.
- RS f45 da busca: mediana 0,34 h, máximo 0,51 h.
- Flowers c00002: f5 0,116 h com pico 9,8/11,5 GB; promoção f15→f45 (30 épocas SSL) 0,354 h.

Por run em f45:

| dataset / genoma | h por run | pico reservado |
|---|---|---|
| rs_b1 | 0,40 | 36 GB |
| flowers_b1 | 0,70 | 12 GB |
| seed_small RS | 0,50 | 24 GB |
| seed_small Flowers | 0,80 | 14 GB |

- Acréscimo de fit por método: S1 +0,02 h, S2 +0,08 h, S3 +0,25 h (300 passos por camada).
- No torch 2.2 do athena pode ser mais lento. O smoke B0 mede isso.

Total:
- itens 1–2 (baseline RS e Flowers, incluindo o lowlr): 54 runs, ~30 GPU-h;
- item 3: 228 runs novos, ~147 GPU-h;
- **total ~177 GPU-h, ~7,4 dias numa lane**.

Concorrência sugerida:
- RS (36 GB) sozinho;
- dois jobs de Flowers juntos cabem (2 × 12–14 GB);
- RS + Flowers juntos (≥ 48–50 GB) não cabem com folga nos 49 GB.

## 7. Pendências / riscos

- O smoke na GPU do athena com torch 2.2 (job B0) é obrigatório antes do resto (R9).
- O pico de 36 GB do RS vem da produção, com outra fitness. Se o smoke medir bem menos, dá para paralelizar.
- O S3 em f45 depende do tempo por camada (`max_seconds` 1800). Um infeasible por tempo não é determinístico; está registrado em `details`.
- `rs_b1_band` (736k) e o S0-lf (label-free do S0) não foram para a fila.

## 8. v2 (17/11, após o retorno do athena: torch 2.2.2, MKL)

Patch `a2impl_v2.patch` (md5 c3dfc7a19e19bb5ad250611553f27934); diff v1→v2 em `a2impl_v1_to_v2.diff` (md5 68a19ada397e5744afd2339a8085c78c). Driver, interface e S0 não mudaram.

1. **`test_invariance_terms_drive_the_choice`** (S2 > noinv), que falhou no athena: 0,3507 < 0,3786.
   - O teste passou a ser um efeito PAREADO em 5 réplicas fixas de dados/seeds. Exige, ao mesmo tempo, vantagem média ≥ 0,05 E vitória em ≥ 4 de 5 réplicas, contra o noinv e contra a média dos candidatos.
   - Na Jaci, as médias de estabilidade são S2 0,59 × noinv 0,42 × candidatos 0,36. O S2 vence nas 5 réplicas com n = 24. Com n = 48 empata ou perde em 2 de 5.
   - **Se falhar no athena, isso é evidência de que o S2 não vence o noinv de forma robusta** (o resultado depende da plataforma). Fica registrado como negativo, sem afrouxar o teste.
2. **MKL DLASCL.** Novo `_safe_linalg.py` com `safe_eigh`/`safe_eigvalsh`/`check_finite_rows`, usado em TODOS os eigh do S1, S2 e S3 e na entrada dos k-means.
   - NaN/Inf → `InfeasibleCandidate`.
   - Matriz zero (canais mortos) → autovalores 0 e identidade, sem chamar o LAPACK.
   - Jitter padrão 0: o resultado é bit a bit igual ao eigh direto (testado).
3. **OOM do smoke RS** (DBN no SSL, > 22 GiB). Não mudei a ciência; bf16 ou batch menor seriam uma variante nova.
   - A fila ganhou `gate_memoria_livre_gb` por job: rs_b1 40, seed_small RS 28, flowers_b1 16, seed_small Flowers 18.
   - Regra: não usar `--gpu-cap-gb` < 38 no RS.
   - O runner continua refazendo um OOM até 10 vezes, com 300 s de espera.

Suíte numa cópia limpa com o v2: **2475 passed, 7 skipped, 0 falhas**.
