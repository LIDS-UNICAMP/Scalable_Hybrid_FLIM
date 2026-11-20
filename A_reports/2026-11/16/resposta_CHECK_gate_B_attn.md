# Agente 6: resposta à auditoria do CHECK (gate B-attn, APROVADO COM CONDIÇÕES)

> 17/11/2026, Jaci, só CPU. As correções foram feitas na cópia `scratchpad/attn6`; a árvore viva não foi tocada (`git apply --check` OK).

## Entregáveis v2

| Arquivo (`research/spifil_b0b4/agent6/`) | md5 |
|---|---|
| `fixed_attention_v2.patch` (módulo + testes, arquivos novos) | `21774ff15b715668d68399c3b79ce456` |
| `ga_search/modehb/fixed_attention.py` (v2) | `f4bc107c0830791e555ca5ae4d8fce54` |
| `tests/modehb/test_fixed_attention.py` (v2) | `2fa8c5a1c3ea0a4f480b26836d34810f` |
| `tools/eval_fixed_attention.py` (runner) | `f4fc9a2d5ad09242d13771d352f9ddd4` |
| `tools/select_checkpoints.py` | `60d1163197922d2a4c6fef39dab3a0ab` |
| `tools/freeze_E0_E8.py` | `388e0036d6eb2eeafaa106a4dfd793fe` |
| `tools/ckpt_compat_check.py` | `c8d084273f6d4fb244b20f768d615d73` |
| `tools/paired_bootstrap.py` | `b8f52e3e87b42480abde189399c5f06c` |
| `checkpoints_E0_E8_congelado_20261117_0159_v2.csv` (74) | `c25d6fa53ed70476f0b3458ed3ce11b3` |
| `fila_gpu3_attention_congelado_20261117_0159_v2.json` | `8f1c1472529051ecb7b93f436ddb6292` |
| `rsync_files_from_congelado_20261117_0159_v2.txt` | `dca8e8da3351bd7804c367c01cdbfda3` |
| `compat_ref_congelado_20261117_0159_v2.npz` (referência da Jaci, torch 2.7.1) | `460a2b8b84e2458709ee5024f758ab05` |
| `custo_E1_E5_2mac.csv` | `d6f8d2e6ec1bd55fcf2a40e2c043758a` |

Os md5 de tudo isso também estão em `fixed_attention_v2.patch.md5`, `MANIFEST_congelado_20261117_0159_v2.md5` e `freeze_congelado_20261117_0159_v2.log`. As entradas `code/…` do MANIFEST são relativas à raiz do código. O diff de revisão está em `fixed_attention_v1_para_v2.diff`.

Os arquivos v1 (`fixed_attention.patch`, `checkpoints_E0_E8.csv`, `lists/` e `fila_gpu3_attention.json`) foram **mantidos** como histórico, porque não apagamos nada. **Estão substituídos pelos v2.**

## Status por condição

| # | Status | O que foi feito | Evidência |
|---|---|---|---|
| **C1** | **feito** | `fit_sample_indices(n, n_img, seed)` faz uma permutação semeada uniforme, sem rótulo. `FixedAttentionEmbedder.fit(..., indices=None)` passa a sortear em vez de pegar `x_train[:n_img]`. O runner grava `align_fit_indices`, `align_fit_ids_md5`, `align_fit_seed` e `align_fit_n_classes` (só diagnóstico). | 2 testes novos. No smoke v2, o ajuste vê **14/14 classes** no RS, contra 4/14 antes. O κ do E3 mudou: c00030 0,4530 → 0,4605; c00009 0,4651 → 0,4584. |
| **C2** | **feito** | Toda linha de val ou teste grava `preds/<hash>_<exp>_<split>.npz` (sample_id, y, pred) com `preds_md5`. O `paired_bootstrap.py` calcula Δκ pareado com IC95 e `--exclude-ids` sem rodar a GPU de novo. | `smoke_v2/preds/*.npz`; `smoke_v2/dk_bootstrap_smoke.csv` (o Δκ bate com o κ das linhas) |
| **C3** | **feito** | `freeze_E0_E8.py` gera, num passo só, CSV, listas, `--files-from`, fila e MANIFEST com md5, partindo da lista congelada `…0159.csv` (75). | `freeze_…v2.log` |
| **C4** | **feito** | O seletor usa o **search_state** como fonte de verdade (job `ok` **e** `result.json` ok). **coconut hybrid B4 c00012 f5 foi excluído**: está `failed`, com `params 3541364 > cap 3500000`, e não há substituto ok com job ≤ cutoff. Os outros 74 estão ok e com md5 igual ao do disco. | log do freeze |
| **C5** | **feito** | `FLOPS_UNIT="2*MAC"`; `encoder_macs()` usa a mesma metodologia do `profile()` (**bate com o `flops` da busca nos 74**); `cost_ratio()` dá `attn_over_encoder` e `attn_within_budget` (≤ 25%), gravados em toda linha. Documentos corrigidos: plano, smoke, viabilidade e literatura. | `custo_E1_E5_2mac.csv`: encoders 235,4 GFLOP (2·MAC), E1–E5 30,7. **E5 > 25% em 10/74, > 100% em 2/74** (flowers hybrid c00002 1,98×; ham lemonade c00002 1,08×). E1–E4 ficam < 25% em todos. A c00009 dá 0,565, não 2×. |
| **C6** | **feito** | `codigo_md5` da fila com módulo, testes, runner, seletor, freeze, compat e bootstrap. Cada linha grava `env`: torch, sklearn, numpy, python, host, `gpu_uuid`, `gpu_pci_bus_id`, `gpu_physical_index`, `cuda_device_order`, `cuda_visible_devices` e `code_md5`, além do `checkpoint_md5`. | linhas do smoke v2 |
| **C7** | **feito no código; falta rodar no athena** | `CUDA_DEVICE_ORDER=PCI_BUS_ID` nos 13 comandos de GPU da fila. `ckpt_compat_check.py` carrega os 74 e faz E0 numa entrada sintética fixa, com TF32 off. A referência da Jaci (torch 2.7.1) foi gravada (74/74 ok; a autocomparação na Jaci também dá 74/74). O job **A-attn-03** compara no athena (torch 2.2.2) com cos ≥ 0,9999 e erro relativo ≤ 1e-3, e confere o UUID `GPU-3c72f1c7` do processo. O runner também desliga o TF32. | `smoke_v2/compat_ref_jaci.jsonl` |
| **C8** | **feito** | O plano §8 declara: HAM com 26,2% do teste com a lesão em train/val (κ de teste marcado como otimista); recapturas eggs/larvae com sensibilidade obrigatória (`--exclude-ids` dos pares `veredito_visual=mesmo`); duplicatas flowers/aid; pre/raw só contra FM do mesmo domínio; régua `ORIG_sub4000` (gravada em `ruler`); seleção pela régua COV. | `plano_execucao_E0_E8.md` §3, §8 |
| não-bloq. | **feito** | **align_k ≤ 5:** `align_degenerate=true` e nota `DEGENERADO`, com tabela à parte. São 5 casos, todos do E4: eggs lem B3 (k=5), flowers lem B1 (k=5), larvae hyb B0 (k=2), larvae_raw hyb B0 (k=2) e protozoan hyb B0 (k=1). **Desempate declarado:** κ, depois maior fidelidade, depois menor cand, depois menor job. Os 2 empates da lista congelada ficaram como estavam e estão registrados. | plano §1, §2 |

## Verificação

- **Testes:** `tests/modehb/test_fixed_attention.py` passou com **152 passed** (eram 148, mais 4 novos para C1, C5 e degenerado). A suíte `tests/` inteira da cópia, em CPU, deu **2460 passed, 4 skipped, 0 failed** (88 s).
- **Smoke v2:** CPU, validação, RS c00030 e c00009, E0–E8, em `agent6/smoke_v2/`; números em `smoke_cpu_rs.md`. Os 9 κ de E0–E8 do c00030 batem com o v1, exceto E3/E4, que mudaram por causa do C1.
- **Próximo passo do Agente 5:** rodar A-attn-00b (código), A-attn-01b (suíte) e A-attn-03 (compat C7) no athena. A fase B continua dependendo do gate.
