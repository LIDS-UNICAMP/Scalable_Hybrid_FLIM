# Fase A — atenção fixa (Agente 6), execução no athena GPU 3 — 2026-11-17 02:05:03

## A-attn-00 (staging somente leitura) ✅
- Código congelado: jaci scratchpad attn6 → athena /dados/home/moliveira/exps/attn6/code (rsync, sem artifacts). md5 ga_search/modehb/fixed_attention.py = f26b5c302f2b55e4a2b86e519146d4b0 e tests/modehb/test_fixed_attention.py = 2974f5d72c909dbca92ec7f59685cb8f, iguais ao fixed_attention.patch.md5.
- Lista regerada na Jaci (tools/select_checkpoints.py artifacts) e CONGELADA: checkpoints_E0_E8_congelado_20261117_0159.csv (md5 33fbec50d9fe42992470091b8b8d7b97; 75 linhas). Difere do CSV antigo (74): +flowers hybrid B3 c00004 f15; ham10000 B2 c00023→c00043; larvae_raw B1 c00022→c00026 (as buscas andaram).
- 75 encoder.pt + config_resolved.yaml + os 2 do smoke (RS hybrid c00009 f5; RS lemonade c00030 f5, md5 44994c28… conferido) → /dados/home/moliveira/exps/attn6/ckpts (rsync --files-from, 94 arquivos). md5 dos 75 = coluna md5 do CSV: 0 diferenças. Lista: /dados/home/moliveira/exps/attn6/md5_attn6.txt. Árvore em modo somente leitura (chmod a-w). Todos estavam na Jaci.

## A-attn-01 (testes, torch 2.2.2) ✅
- tests/modehb/test_fixed_attention.py: GPU 3 148 passed; CPU 148 passed.
- tests/ inteira em CPU: 2453 passed, 7 skipped, 0 failed (141 s). Log: out/A-attn-01/full_suite_cpu.log.

## A-attn-02 (split + smoke GPU 3, só validação) ✅
- split_sha256: athena = Jaci = YAML nos 12 datasets (aid, coconut_trees, eggs, eggs_raw, flowers, food101, ham10000, larvae, larvae_raw, protozoan, protozoan_raw, remote_sensing).
- Smoke E0–E8 em RS hybrid c00009 f5 e RS lemonade c00030 f5 (validação; SVM core.svm.fit_svm linear C=100 sem scaler): 20 linhas, 0 NaN; |κ_GPU − κ_CPU| ≤ 0,0072 nas 16 comparáveis (máximo em c00030 E2; o resto 0,0000). Pico de memória da GPU 3 inteira: 5.132 MiB (< 8 GB); RSS 2,3 GB; 57 s de parede para 2 checkpoints × 9 experimentos.
- Saídas: out/A-attn-02/{val.jsonl, smoke.log, split_sha_athena.txt, split_sha_jaci.txt}.

## Fase B: BLOQUEADA até existir research/spifil_b0b4/gates/gate_B_attn.approved.
