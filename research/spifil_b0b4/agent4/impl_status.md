# Agente 4 — status da implementação inicial (infraestrutura de integridade do H-MODEHB)

> 17/11/2026. Implementação feita numa cópia própria (sem tocar a árvore viva nem processos), só CPU.
> Patch: `research/spifil_b0b4/agent4/a4impl.patch` (md5 em `a4impl.patch.md5`: `f9cd1cd207e6541d9b78c2798fb69788`).
> `git apply --check` contra a árvore viva: OK (exit 0). **O patch ainda não foi aplicado.**
> Cópia: `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/a4impl/` (git local; baseline = commit `f9acfdd`).

## O que entra (17 arquivos, +2407/−5)

| Arquivo | Conteúdo |
|---|---|
| `ga_search/hmodehb/bk.py` | Faixas B0–B4 fixas `(lo, hi]`, teto duro de 5.500.000 (sem o ×1,05), `MIN_EMBED_DIM=1280`. `stratified_block` impõe as faixas: qualquer outra faixa ou teto → `BandError`. |
| `ga_search/hmodehb/fitness_v2.py` | Validação da régua ORIG (`C=100`, `scaler=none`, `readout=head`, `kappa_weight=1`, `tta=1`, `whiten=none`, `libsvm`). Também: `spec`/`signature` (fitness_sig), objetivos com limites fixos [1e6, 1e11], `finalize` (quality = κ ORIG; colapso vira só flag; diagnósticos em `aux`; NaN → `NonFiniteFitness`) e o **carimbo de régua por decisão** (`ruler_sig`, `check_decisions`, `RulerViolation`). |
| `ga_search/hmodehb/integrity.py` | `config_sha` do config completo; `check_drift`/`ConfigDrift` com `allow_drift` e registro; `ControllerLock` (flock); `cand_id` `h4-<braço>-c#####`; `lineage_key`/`cand_key`; `adoptable` (mesma `cand_key` e `fitness_sig`, checkpoint com o mesmo sha256); escrita atômica. |
| `ga_search/hmodehb/hsearch.py` | `HMODEHB(modehb.MODEHB)`. Sobrescreve `_fresh_state`, `run` (trava, régua, deriva), `_new_id`, `_evaluate` (cache e resume por chave), `_call` (replay mais adoção, `pending_job.json`, `hjobs/<job>.json`), retries de OOM parametrizados com `reason`, `_evaluate_twin` (por cand_id) e `best_by_bk`. |
| `ga_search/hmodehb/hevaluate.py` | `HEvaluator(run_search.Evaluator)`. `_svm_features` usa sempre `head(encoder(x))` com tta 1 e nunca cov_head. `_svm_score` aplica a régua ORIG fixa e recusa embedding < 1280. `result.json` grava `cand_key`, `fitness_sig`, `config_sha`, `checkpoint_sha256` e `decision_rulers`. Exige validação inteira (`val_max_samples: null`). |
| `ga_search/hmodehb/run_hsearch.py` | `load_hconfig` (base + braço + `--set`), `build`, `controller_id`/`hcontroller` (W&B `hctrl-sha1(dir\|braço\|fitness_sig)`, nome `h4-controller-<ds>-<braço>-v2_<sig>`, grupo `hmodehb_b0b4/<ds>`), `main` (a trava é tomada antes de carregar os dados) e `summary.json`/`best_by_bk.json`. |
| `ga_search/config/hmodehb_b0b4/{base.yaml, arms/A.yaml, arms/B.yaml}` | Config autocontido: fitness v2, B0–B4, `paths.state_dir = artifacts/hmodehb_b0b4/arm_{arm}/{dataset}/state`, `hmodehb.oom = 4×180 s` (R5). Fidelidades 5/15/45 com extensão 100/200. |
| `ga_search/modehb/monitor_report.py` (única mudança em código compartilhado, retrocompatível) | `ROOT_GLOBS += artifacts/hmodehb_b0b4/arm_*`; `_kind(root, state)` devolve `h4_<braço>` (lido do estado) **antes** das substrings; `_fit_sig` devolve `v2_<fitness_sig>` quando o estado o tem; rótulo "H-MODEHB B0-B4 braço X" no CSV. Estados legados (sem `kind`/`fitness_sig`) dão exatamente o que davam antes. |
| `tests/hmodehb/` | 93 testes (`test_h4_bk_fitness.py`, `test_h4_integrity.py`, `test_h4_search.py`, `test_h4_e2e.py`) mais `_helpers.py` (avaliador sintético) e `_runner.py` (processo controlador para kill -9 e dois controladores). |

## Os 8 itens pedidos

1. **Subclasses**: `HMODEHB(MODEHB)` e `HEvaluator(Evaluator)`. Nenhuma linha de `modehb.py`, `run_search.py`, `strata.py` ou `lemonade.py` mudou.
2. **Fitness v2**: κ de validação ORIG (head ≥ 1280, `fit_svm` com C=100, sem scaler) e FLOPs com limites fixos.
   - Auxiliares só em `aux`. `fitness_sig` vai no estado e em cada job; retomar com outra régua → `FitnessSigMismatch`.
   - **Pedido do coordenador (achado do A1b)**: toda decisão usa ORIG-val.
     - O α do WiSE e o κ final passam por `_svm_features`/`_svm_score`, que são forçados a ORIG e carimbados.
     - Seleção, promoção, calibração e extensão leem `quality`/F, que o controller confere como = clip(κ ORIG).
     - O controller confere ainda que o α do WiSE é o argmax do κ ORIG-val.
     - Um carimbo diferente da régua gravada → `RulerViolation`: a busca para e nada é gravado.
     - Testes que falham se o COV voltar a decidir: `test_real_cov_in_any_decision_fails` (pipeline real com o WiSE lendo cov_head), `test_decision_with_other_ruler_stops_search_unrecorded` e `test_check_decisions_rejects_other_ruler` (8 casos).
3. **cand_key** = genoma efetivo + seed + origem dos pesos (`scratch`/`inherit` com sha256 do checkpoint do pai/`external`) + fidelidade + `fitness_sig`. O cache usa `cand_key`; o resume de promoção usa `lineage_key` (sem a fidelidade).
   - Teste de regressão: o mesmo genoma herdado não recebe mais o resultado em cache.
   - O teste também documenta que o MODEHB histórico descarta essa herança.
4. **Trava flock** em `<out>/controller.lock` (pid, host e início). O lock é do kernel e some com kill -9. O segundo controlador sai com `ControllerLocked` (exit 3 no runner).
5. **`config_sha` do config completo** (search + eval + train + data + hmodehb; excluídos só `wandb.mode`, `device` e `hmodehb.allow_drift`) e `config_full` no estado.
   - Deriva no resume → `ConfigDrift`, que lista as chaves.
   - `--allow-drift chave` registra a mudança em `drift_log`.
6. **Adoção no resume**: `hjobs/<job>.json` (controller) ou `candidates/<id>/f<F>/result.json` (avaliador).
   - Exige a mesma `cand_key` e `fitness_sig` e checkpoint com o mesmo sha256.
   - Não roda o job de novo e não abre run W&B. Um JSON parcial ou um checkpoint trocado nunca é adotado.
7. **B0–B4** com faixas fixas e teto duro de 5,5M.
   - Pré-checagem estrutural antes de gastar GPU.
   - Params medidos acima do teto → `failed/param_cap`.
   - As promoções protegidas do `strata.py` são reusadas com 1 quota por B_k.
8. **Ids** `h4-<braço>-c#####`, `kind=h4_<braço>` no estado e no monitor_report (sem colisão com hybrid/lemonade), e controller W&B com a régua no id.

Falhas com motivo explícito em `result.reason`: `oom` (após N retries, sem loop infinito), `nan`, `missing_ancestor`, `param_cap`, `embed_dim`, `error`. CUDA fatal → nada é gravado e o resume refaz o job.

## Testes (rodados na cópia, CPU, `CUDA_VISIBLE_DEVICES=""`)

- **Suíte inteira**: `pytest tests/` → **2398 passed, 7 skipped, 0 failed** (6 min 7 s).
  - Baseline antes do patch: 2305 passed, 7 skipped. Os 93 novos são todos do `tests/hmodehb/`.
- **Golden existentes verdes**:
  - `test_seed_off_genome_sequence_identical_to_before_patch[search_v3.yaml|search_v3_lemonade.yaml]`
  - `test_existing_configs_identical_with_features_off`
  - `test_resume_identical_no_replay_mismatch`
  - `test_strata/test_lemonade ... resume_replays_identically`
  - `test_oom_retry` (`_fit_sig`/`_kind` legados)
- **Unitários**:
  - fronteiras B_k (300000→B0, 300001→B1, 5500001→None);
  - faixas e teto recusados;
  - réguas recusadas (C=0.1, scaler standard, cov_head, pw, kw 0.75, tta 8, liblinear, penalidade dimensional);
  - assinatura sensível a cada campo;
  - `finalize`/NaN;
  - chamada do SVM com C=100/none/none/1.0;
  - embedding < 1280 recusado;
  - `config_sha` canônico e deriva;
  - flock no mesmo processo e entre processos (incluindo kill -9 do dono);
  - `cand_key`, adoção, `monitor_report` (árvore falsa: `lemonade|...|c00001` × `h4_B|...|h4-B-c00001`);
  - id do controller W&B.
- **Controller (avaliador sintético)**:
  - determinismo;
  - promoção mantém `lineage_key`;
  - herança no braço B com sha do pai;
  - resume + `--allow-drift` reproduz a busca contínua;
  - `ReplayMismatch` com chave adulterada;
  - OOM 2× → ok com `oom_retries=2`; OOM esgotado → `failed/oom` com exatamente 1+2 chamadas;
  - NaN → `failed/nan` e a busca segue;
  - CUDA fatal → sem registro e o resume fica idêntico;
  - ancestral ausente → `missing_ancestor`;
  - adoção sem re-treino (das duas fontes) e recusa de checkpoint corrompido.
- **E2E**:
  - **kill -9 real em pontos exatos**: start, before_result, after_result, after_hjob, after_save, mais after_result tardio.
    - O resume dá a mesma sequência (ids, `cand_key`, F, op) da execução contínua.
    - Após result.json/hjobs o job é adotado; antes, roda uma vez a mais.
    - Nunca sobra `.tmp`.
  - **12 kill -9 em instantes aleatórios** (import, replay, avaliação, save) + 1 execução final:
    - sequência idêntica;
    - re-execuções ≤ número de kills.
  - **Dois controladores** (processos): o segundo sai na hora e o primeiro termina idêntico.
  - **Deriva via linha de comando**: exit 4; com `--allow-drift`, segue e registra.
  - **Mini-busca REAL** (`HEvaluator`: SPiFiL → AE local → LeJEPA+SIGReg → WiSE → SVM ORIG; sintético 32 px, fidelidades [1, 2], ~20 s):
    - zero falhas, todos os jobs v2 com embed ≥ 1280;
    - 5 decisões do WiSE por job, todas ORIG, e α = argmax κ ORIG;
    - checkpoint com sha256 conferido, que recarrega bit a bit em `eval()`;
    - adoção no resume sem chamar o avaliador.
  - **Pipeline real com falhas injetadas**:
    - NaN injetado no SSL → `failed/nan` e a busca segue;
    - kill -9 real depois do result.json → adotado;
    - COV injetado no WiSE → `RulerViolation`.

## Desvios declarados e pendências

- **Braço C (PNAS-CS) não roda no H-MODEHB**: o `pnas_cs.py` define os níveis por pontos médios geométricos com cap ×1,05, que não são B0–B4. O controller recusa com `NotImplementedError`. Para habilitar, é preciso um override de `level_strata` (sem mexer no `pnas_cs.py`).
- **Braços D0/D/E**: sem config. Dependem dos gates A2 (`KernelInitializer`) e A3 (`GrowthSpec`). O **genoma hierárquico de 5 specs** (desenho §3) **não** foi implementado: o controller ainda usa o genoma plano do MODEHB, com toda a integridade por cima.
- **Braço B**: LEMONADE sem a amostragem por KDE (desvio do paper, declarado no YAML).
- **NaN estrito**: loss SSL final NaN → `failed/nan`, mesmo que o WiSE α=0 tivesse um modelo válido (segue o desenho §11.3). Revisar se o usuário preferir "α=0 salva o candidato".
- **`kappa_cov` em `aux`** ainda não é calculado na busca. O `monitor_report` continua medindo a COV de referência no teste, fora do controller.
- **Δκ filho × ancestral**, Pareto só entre comparáveis, `lineage.jsonl`/DAG e painéis W&B novos ficam para a próxima etapa.
- **OOM**: o padrão do H-MODEHB é 4×180 s (R5). O MODEHB histórico segue 10×300 s, intocado.
- **Valores [H]** do `base.yaml` (`memory_budget_gb: 38`, `gpu_process_cap_gb: 44`, `num_workers: 16`) precisam da Fase A do A5 na GPU 3.
- **Aplicar o patch** (`git apply research/spifil_b0b4/agent4/a4impl.patch`) muda o `monitor_report` vivo. O efeito prático é nulo até existir `artifacts/hmodehb_b0b4/`, mas o processo do monitor só pega a mudança quando for reiniciado (decisão do A5 ou do usuário).
- O `graphify update .` não foi rodado: a árvore viva não foi alterada.
