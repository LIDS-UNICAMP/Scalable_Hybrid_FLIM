# Disco em jaci (moliveira) — 2026-11-11

Estudo só de leitura (nada apagado). `/` 94% (29G livres); `/dados` 74% (731G livres).

## Totais por pasta (/dados/home/moliveira)
| Pasta | Tamanho |
|---|---|
| exps | 210G |
| artifacts | 84G |
| scalable_FLIM_self_supervised | 47G |
| datasets | 37G |
| mateus_oliveira | 23G |
| .cache (uv 17G) | 17G |
| Scalable_Hybrid_FLIM | 9.1G |
| .vscode-server | 5.4G |
| miniforge3 | 3.9G |
| Scalable_Hybrid_FLIM_discovery | 89M |
| /tmp/ray (em `/`) | 2.6G |
| /var/tmp | 28K |

## 15 maiores subpastas
exps/tmp 76G · artifacts/distillation 71G · exps/fontes_filtros 56G · datasets/organized 37G · exps/image_cache 36G · exps/tmp/ms_fontes 32G · scalable_FLIM_self_supervised/artifacts/distillation 25G · exps/hf/hub 23G · scalable_FLIM_self_supervised/logs 21G (flim-ssl 15G, wandb 5.5G) · .cache/uv 17G · exps/spifil_growth 14G · datasets/organized/Eggs 14G · artifacts/spifil_growth 13G · exps/image_cache/food101 12G (200_nocrop) · exps/tmp/wsal_fontes 9.5G

## Candidatos
| Classe | Caminho | Tamanho | Seguro? | Motivo |
|---|---|---|---|---|
| a | /tmp/ray/session_* antigas (todas menos as 8 referenciadas por processos vivos: ..._2077626, 2232579, 2370156, 2502428, 2594366, 2599484, 2949311, 3032673) | ~2.1G | sim | nenhum processo usa; libera `/`, que está a 94% |
| a | `__pycache__` (todos) | 11M | sim | recriado sozinho |
| a | scalable_FLIM_self_supervised/logs/wandb + /wandb | 6.9G | só listar | status de sync não verificado run a run |
| a | exps/spifil_growth/hyb_receita_inteira_lejepa_oom_cap28_1210 | 355M | perguntar | marcado como descartado |
| a | exps/spifil_growth/hyb_receita_inteira_lejepa_receita_antiga | 182M | perguntar | marcado como descartado |
| a | exps/spifil_growth/hyb_receita_inteira_lejepa_falso_inicio_1130 | 196K | perguntar | marcado como descartado |
| a | exps/queues/logs/*.pausa09nov | ~40K | perguntar | logs pausados |
| b | 206 × last.ckpt com best*.ckpt ao lado (fora de hyb_receita_inteira*) | 4.4G | sim (regra best>last) | o best fica |
| b | hyb_receita_inteira* best_kappa.ckpt | 2.3G (todos os ckpt da família) | NÃO tocar | best_kappa = última época |
| b | exps/spifil_growth/hyb_curva_receita (layerwise abandonado) | 3.3G (2.5G ckpt) | perguntar | experimento abandonado |
| c | exps/tmp/*_fontes (ms 32G, wsal 9.5G, gauss 9.4G, salw 9.3G, sol 8.6G, largura 3.6G, ham 2.3G) | ~75G | provável; perguntar | parados desde 08/11, nenhum processo os referencia; recomputar = refazer as fontes das grades |
| c | exps/fontes_filtros (receita_tta, gelu, restr5m_*, pf_res*, encdeep…) | 56G | perguntar por subpasta | fontes de grades; checar quais ainda são usadas |
| c | exps/image_cache (food101 12G, Protozoan 4.3G, aid 2.9G…) | 36G | perguntar | só apagar tamanhos fora de uso; recomputar = reprocessar imagens |
| c | .cache/uv | 17G | sim | cache de pacotes, rebaixado sob demanda |
| c | miniforge3/pkgs/cache, .nv/ComputeCache | 2.4G | sim | caches reconstruíveis |
| c | Ray object spill | ~0 | — | sem spill relevante em /tmp/ray |
| d | datasets/ | 37G | manter | dados |
| d | exps/encoders, work_dirs em execução, CSVs de resultado | — | manter | em uso / resultados |

## Recuperável por classe
- (a) seguro: ~2.1G (Ray) + 11M; com pergunta: +0.54G; wandb só listado 6.9G
- (b) seguro: 4.4G; com pergunta: +3.3G
- (c) seguro: ~19.4G (uv, conda, nv); com pergunta: até ~167G (fontes tmp 75G + fontes_filtros 56G + image_cache 36G)
- Total seguro ≈ 26G; potencial com aval ≈ 197G.
