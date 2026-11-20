# Fase A v2 (atencao fixa, Agente 6) — athena, GPU 3 — 2026-11-17

Congelamento: congelado_20261117_0159_v2. Fila: fila_gpu3_attention_congelado_20261117_0159_v2.json (md5 8f1c1472...).
Desvio consciente da fila: o codigo v2 foi montado em /dados/home/moliveira/exps/attn6/code_v2 (a fila manda recopiar em code/,
mas code/ v1 esta em uso pelo B-E0 rodando; v1 ficou intacto, fixed_attention.py f26b5c30...). Saidas em out/A-v2/ (nao out/A-attn-03).

## A-attn-00b — staging + codigo v2: OK
- rsync --ignore-existing --files-from (dry-run): 0 arquivos a copiar. Lista v2 (93 linhas) = lista v1 menos coconut hybrid c00012.
- 74/74 checkpoints do CSV v2 presentes, md5 74 ok / 0 diferente / 0 faltando; config_resolved todos presentes.
- code_v2 = copia de code/ sem os 2 arquivos v1 + `patch -p1 < fixed_attention_v2.patch` (md5 21774ff1... ok) + tools/ e lists v2 da jaci.
- codigo_md5 da fila: 7/7 conferem (fixed_attention.py f4bc107c, test_fixed_attention.py 2fa8c5a1, 5 tools). artefatos_md5: 14/14 conferem.

## A-attn-01b — suite CPU (torch 2.2.2): OK (criterio 0 failed)
- tests/ completo: 2457 passed, 7 skipped, 0 failed, 39 warnings (158 s). rc=0.
- tests/modehb/test_fixed_attention.py sozinho: 152 passed.

## A-attn-03 — compat torch 2.7.1 (jaci) -> 2.2.2 (athena), cuda, TF32 off: OK (criterio 0 divergencias; uuid confere)
- n=74, falhas=0, todos finitos e com dimensao igual.
- cos minimo = 0.99999976 (flowers|modehb_hybrid|B1|c00002|f200) >= 0.9999.
- erro relativo L2 maximo = 7.36e-05 (coconut_trees|modehb_lemonade|B1|c00002|f15) <= 1e-3.
- GPU do processo (nvidia-smi): GPU-3c72f1c7-f2c9-d905-fdee-9b0b0ce802be, CUDA_DEVICE_ORDER=PCI_BUS_ID, gpu_ok=true.

Teste nao foi lido. E1-E8 e avaliacao de teste nao rodados (gate_B_attn.approved inexistente).
