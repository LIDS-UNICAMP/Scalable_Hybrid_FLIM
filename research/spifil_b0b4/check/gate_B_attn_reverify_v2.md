# CHECK: reverificação de C1–C4 (gate B-attn) no patch v2 do Agente 6

> 17/11/2026, Jaci. Só CPU com `nice -n 19`. Nenhum processo foi tocado e a árvore viva não foi modificada: usei só `git apply --check`. Scripts e saídas estão em `scratchpad/rv2/`.

## Veredito: **LIBERA FASE B**

As quatro condições bloqueantes foram cumpridas. As pendências abaixo não bloqueiam. A C7 continua sendo pré-requisito do A-attn-02/03 no athena, como já estava na auditoria, mas não faz parte deste gate.

## Integridade do patch e testes

| Verificação | Resultado |
|---|---|
| md5 de `fixed_attention_v2.patch` | `21774ff15b715668d68399c3b79ce456`: **confere** |
| `git apply --check` na árvore viva | **OK** (só 2 arquivos novos: `ga_search/modehb/fixed_attention.py` e `tests/modehb/test_fixed_attention.py`) |
| Patch aplicado num repositório vazio | reproduz `f4bc107c…` e `2fa8c5a1…`, iguais à cópia e ao MANIFEST |
| Cópia `attn6` contra a árvore viva (`ga_search/`, `core/`, `flim/`) | só diferem logs e o `FILA_E_INSTRUCOES.md`, mais o `fixed_attention.py` novo. Nenhum código diverge |
| `pytest tests/modehb/test_fixed_attention.py` (cópia, CPU) | **152 passed** em 1,8 s. Inclui `test_fit_sample_is_uniform_seeded_not_first_n` e `test_embedder_fit_uses_uniform_sample_and_records_indices` |

## C1: alinhamento E3/E4 com sorteio uniforme semeado. **CUMPRIDA**

- **Sorteio.** `fit_sample_indices` (`fixed_attention.py:775`) usa `torch.randperm(n)` com `Generator(seed+7919)` e não recebe rótulo.
- **Uso do sorteio.** O `fit()` (`:888-891`) usa esse sorteio no lugar de `x_train[:n_img]`, a menos que receba `indices` explícitos.
- **Rótulo.** O `ytr` aparece no runner só no diagnóstico `align_fit_n_classes` (`eval_fixed_attention.py:174`).
- **Índices gravados.** Cada linha E3/E4 do `smoke_v2/smoke_rs_val.jsonl` traz `align_fit_indices` (160), `align_fit_seed=0`, `align_fit_ids_md5=8879a26e…` e `align_fit_n_classes=14`.
- **Cobertura num caso ordenado por classe** (meu script `rv2/c1_cover.py`, que copia a função literalmente):
  - **Sintéticos ordenados:** k=14, 2, 7 e 3 dão cobertura total. Com as 160 primeiras, cobriam 1 classe (2 no caso k=3).
  - **Subconjuntos reais** (`_stratified_indices`, já ordenados), mostrados como v1 (160 primeiras) contra v2 (sorteio):

    | Base | v1 | v2 |
    |---|---|---|
    | RS | 4 | 14/14 |
    | eggs | 2 | 9/9 |
    | ham10000 | 1 | 7/7 |
    | flowers | 1 | 5/5 |
    | larvae | 1 | 2/2 |
    | protozoan_raw | 2 | 7/7 |
    | aid | 2 | 30/30 |
    | coconut | 1 | 2/2 |

## C2: predições por imagem e bootstrap pareado. **CUMPRIDA**

- **Arquivos de predição.** As 18 linhas E0–E8 (2 checkpoints) têm `preds/<hash>_<exp>_<split>.npz` com `sample_id`, `y` e `pred`.
  - O md5 do `.npz` bate com o `preds_md5` nas 18.
  - `preds_n` = 159, e os `sample_id` são únicos.
  - O κ recalculado do `.npz` bate com o `kappa` da linha (diferença < 1e-7, por float32).
- **Caminho de teste.** O `emit()` (`:292-294`) grava as predições no mesmo ponto para validação e teste. O caminho de teste foi exercitado pelo plumbing (`plumbing_c00030.jsonl`: 3 linhas `validation_plumbing` com `preds_file`).
- **`paired_bootstrap.py`.** Rodei no smoke_v2 (B=500): 16 comparações. Os Δκ são **idênticos** aos de `dk_bootstrap_smoke.csv` (os IC mudam só pelo B). O `--exclude-ids` funciona: excluí 2 ids, e o resultado deu n=157 e `n_excluidas=2`.

## C3: congelamento. **CUMPRIDA**

- **MANIFEST.** Todas as entradas conferem com `md5sum -c`:
  - os artefatos, relativos a `agent6/`;
  - as entradas `code/…`, relativas à cópia `attn6`;
  - o `fixed_attention_v2.patch.md5`.
- **Mesmo conjunto de checkpoints.** O CSV v2 tem **74** linhas, todas únicas. As listas, a fila e o rsync batem com ele:
  - as listas `lists_congelado_…_v2/*.txt` têm 74 entradas, com a mesma contagem por dataset;
  - a fila cita 74 checkpoints, com `md5` igual ao do CSV em todos os 74. O `artefatos_md5` e o `codigo_md5` da fila batem com o MANIFEST;
  - no rsync, os encoders do CSV estão todos presentes, e os 18 `config_resolved` batem.
- **md5 em disco.** O md5 dos 74 `encoder.pt` no disco bate com o CSV (0 divergências).
- **Rastreabilidade.** O `freeze_…v2.log` registra a origem (`…0159.csv`, md5 `33fbec50…`), o cutoff, a exclusão C4 e os 2 empates mantidos.

## C4: checkpoints `failed`. **CUMPRIDA**

- **Status no `search_state.json`.** Cruzei os 74 do CSV com os jobs (`cand_id` + fidelidade): **todos `ok`**, e nenhum aparece como `failed`.
- **coconut hybrid B4 c00012 f5.** Está `failed` no `search_state`, e está **ausente** do CSV v2, das listas, da fila e do rsync. O CSV v1 o tinha.
- **Substituto.** Não existe. Nenhum job `ok` do coconut hybrid passa de 3M params (o maior é c00004, com 2,33M, que já é o B3). O estrato B4 fica vazio, como o freeze declara.

## Pendências (não bloqueiam)

1. **`preds_file` relativo.** Ele é gravado relativo ao cwd do runner (`smoke_v2/preds/…`). Ou o bootstrap roda no mesmo cwd, ou o `preds_file` passa a ser absoluto ou relativo ao `--out`. Precisei reescrever os caminhos para rodar.
2. **Encoder a mais no rsync.** O `rsync_files_from…v2.txt` tem **75** encoders: o extra é o checkpoint do smoke `modehb_lemonade/remote_sensing/…/c00030/f5`, que está em `SMOKE` no `freeze_E0_E8.py:46-47`. Isso é intencional, mas convém declarar no log que 93 = 74 + 1 de smoke + 18 configs.
3. **Classes raras no sorteio.** O sorteio uniforme pode deixar de fora uma classe rara com outra seed (no eggs, a menor classe tem 0,65%). Na seed 0 a cobertura foi total em todas as bases. O `align_fit_n_classes` fica gravado, e convém relatá-lo no relatório B.
4. **C7 ainda não rodou.** O A-attn-03 (compat torch 2.7 → 2.2 dos 74 no athena, UUID `GPU-3c72f1c7`) continua obrigatório antes de usar a GPU, como a auditoria já previa.
5. **Empates mantidos.** A lista congelada manteve larvae lemonade B0 c00001 f100 e RS lemonade B1 c00054 f15. Os dois estão declarados no log. OK.
