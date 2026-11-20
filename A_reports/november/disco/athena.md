# Disco do athena (moliveira) — 2026-11-11

Estudo só de leitura. Nada foi apagado. Nenhum job de treino nosso rodando (só a sessão do Claude).

## Filesystems (df -h)
| FS | Tamanho | Usado | Livre | Uso |
|---|---|---|---|---|
| / | 98G | 79G | 15G | 85% |
| /exps | 3.7T | 3.1T | 558G | 86% |
| /dados | 37T | 35T | 1.4T | 97% |

## Totais por pasta
| Pasta | Tamanho |
|---|---|
| /exps/moliveira/image_cache | 264G |
| /exps/moliveira/fontes_filtros | 147G |
| /exps/moliveira/results_sol | 33G (29G alternative_encoders) |
| /exps/moliveira/hf_cache | 32G |
| /exps/moliveira/benchmark | 26G (hf 19G, uv_cache 5.4G, emb 2.4G) |
| /exps/moliveira/spifil_growth | 15G |
| /exps/moliveira/ray_antigo | 12G |
| /exps/moliveira/bench_ref | 5.4G |
| /exps/moliveira/recebido_jaci | 2.9G |
| /exps/moliveira/temporarios_sistema + sessao_scratchpad + claude_tmp_antigo | 3.6G |
| /exps/moliveira/eval, bench_batch, pilotos, results, queues, analises, tmp | 369M, 292M, 220M, 122M, 4.6M, 88K, 0 |
| /dados/home/moliveira/artifacts (= Scalable_Hybrid_FLIM/artifacts, symlink) | 190G (distillation 155G, spifil_growth 35G) |
| /dados/home/moliveira/Scalable_Hybrid_FLIM | 68G (logs/flim-ssl 65G, logs/wandb 1.6G) |
| .claude/worktrees (3) | ~61M |
| /dados/home/moliveira/datasets | 38G |
| /dados/home/moliveira/.cache | 28G (uv 14G, wandb 12G, huggingface 2.4G) |
| /dados/home/moliveira/miniforge3 | 11G |
| /dados/home/moliveira/Hybrid_SPiFiL_net | 5.5G |
| Scalable_Hybrid_FLIM_discovery | 162M |
| presentation_* | ~91M |
| SPiFiL | 12M |
| /tmp/claude-1033 | 8.4G |
| /tmp/ray | 135M |

## Top 15 subpastas
1. artifacts/distillation 155G (cada distill4_all_kd_unfrozen_protozoan_* ~7.1G)
2. image_cache/Protozoan 90G
3. logs/flim-ssl 65G
4. fontes_filtros/ms 46G
5. fontes_filtros/cl 38G
6. artifacts/spifil_growth 35G
7. image_cache/Eggs 34G
8. results_sol/alternative_encoders 29G
9. image_cache/aid 26G
10. image_cache/protozoan_cysts 23G
11. image_cache/Protozoan_subgroup 20G
12. benchmark/hf 19G
13. image_cache/Larvae 17G
14. .cache/uv 14G
15. image_cache/helminth_eggs 13G

image_cache por resolução (todas as bases): 640=112G, 448=56G, 320=29G, 200=26G, 256=18G, 1280=7.4G, 128=5.5G, demais <4G.

## Candidatos
| Classe | Caminho | Tamanho | Seguro? | Motivo |
|---|---|---|---|---|
| a | /exps/moliveira/ray_antigo | 12G | sim | sessões Ray velhas, nenhum ray rodando |
| a | /tmp/ray | 135M | sim | idem |
| a | /tmp/claude-1033 (scratchpads antigos) | 8.4G | sim, menos a sessão atual | temporário |
| a | temporarios_sistema, sessao_scratchpad, claude_tmp_antigo, pilotos | 3.8G | sim | temporário/piloto |
| a | __pycache__ (repos) | ~10M | sim | regenera |
| a | /dados/home/moliveira/.cache/wandb | 12G | sim | cache de artefatos W&B |
| a | Scalable_Hybrid_FLIM/logs/wandb | 1.6G | sim se sincronizado (`wandb sync --view` para conferir) | runs locais |
| a | eval/espelho_growth* | ~0 | sim | só symlinks |
| a | recebido_jaci/exps | 2.9G | perguntar | cópia de jaci; conferir duplicata antes |
| a | .cache/uv + benchmark/uv_cache | 19.4G | sim | cache de pacotes, rebaixa sozinho |
| a | hf_cache: dinov2-giant / with-registers-giant / timm dinov2 giant (4.3G cada), ijepa_vitg16 3.8G, ijepa_vith* 3×2.4G, convnext-xlarge 1.5G, aimv2-large ×3 1.2G, dinov2-large 1.2G, convnext large/base 1.3G | 32G | **perguntar** | baixar de novo leva minutos; benchmark/hf (19G) parece duplicar o mesmo |
| b | last.ckpt em artifacts/distillation | 16.7G (162 arquivos) | perguntar | família de destilação antiga |
| b | last.ckpt com best ao lado (fora de hyb_receita_inteira*) | 0.43G (44) | sim | regra best>last |
| b | artifacts/distillation inteiro | 155G | **perguntar** | experimentos abandonados |
| b | spifil_growth/hyb_curva_receita* (/exps) | 1.8G (ckpts 1.4G) | **perguntar** | layerwise antigo abandonado |
| b | logs/flim-ssl | 65G | perguntar | best mora em subpasta; pesos referenciados por eval |
| c | image_cache 640/448 (só se não usados mais) | até 168G | perguntar | recomputar = reler e redimensionar dataset, ~minutos a 1h por base |
| c | fontes_filtros/ms, cl e grades terminadas | até 147G | perguntar | recomputar = rodar extração de filtros de novo (horas de GPU) |
| d | datasets, results*, A_reports, CSVs, miniforge3, SPiFiL | — | manter | dados e ambiente |

## Total recuperável
- (a) seguro: ~58G (sem hf_cache); +32G de hf_cache se aprovado.
- (b) checkpoints: 0.4G seguro; até ~220G (distillation + flim-ssl + hyb_curva_receita) se aprovado.
- (c) caches: até ~315G (image_cache grandes + fontes_filtros), a custo de recomputar.
