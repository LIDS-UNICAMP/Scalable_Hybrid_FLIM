# Fila de treinos (07/11/2026, 07:50 UTC)

Teto: 35 GB do moliveira por GPU, carga de CPU até ~110, sem afetar o crispim.

Ver a fila (rodando, feitos e erros por faixa):

```
S=/exps/moliveira/queues/logs/law.status; for f in /exps/moliveira/queues/law_lanes/*.txt; do n=$(basename $f .txt); e=$(grep -Ff $f $S | grep '^END' | wc -l); b=$(grep -Ff $f $S | grep '^END' | grep -vc 'rc=0'); r=$(grep -Ff $f $S | tail -n 1 | awk '$1=="START"{print $3}'); echo "$n: feitos $e de $(wc -l < $f), erros $b, rodando ${r:-nada}"; done
```

## Metas
- Objetivo A: lei de aprendizado (kappa contra dados e parâmetros).
- Objetivo B: lei arquitetural: qual combinação de profundidade, largura e resolução rende mais para um orçamento computacional.
- Plano fechado em `plano/PLANO_FINAL.md`: orçamento médio, 655 cadeias, 468 GPU-h.

## Rodando agora: F1 degrau 2 e controle (`disc_md_law_f1d2_*`, `disc_md_law_ctl_rand_s43/s44_*`)
Entrou às 07:44. 40 YAMLs, 84 cadeias, B0, 200 px, 50 superpixels, 50%, 60 épocas por estágio.

| Bloco | YAMLs | Cadeias | O que mede |
|---|---|---|---|
| Degrau 2 | 36 | 72 | UR-JEPA 0,01, Strong-SIGReg 0,001, Barlow 0,001, VICRegL 0,0001, VICReg 0,01 e só reconstrução; flowers, helminth_eggs, helminth_larvae, Eggs; 3 splits. Eggs tem 1 YAML por split, para não prender uma faixa por 3 cadeias longas |
| Controle | 4 | 12 | Aleatório nas sementes 43 e 44, helminth_larvae e flowers, 3 splits. Par do F0 das mesmas sementes: fecha 9 pares por dataset |

Faixas `d2_*` em `/exps/moliveira/queues/law_lanes/`, sessões `disc_md_law_d2_<faixa>`, que fecham sozinhas. Logs em `/exps/moliveira/queues/logs/law.*`.

| GPU | Faixas | Teto por faixa | Outros do moliveira | Soma dos tetos |
|---|---|---|---|---|
| 0 | d2_g0_a a d (4) | 4.500 MiB | `disc_md_base0` (teto 12.000) | 34,9 GB |
| 1 | d2_g1_a a d (4) | 4.500 MiB | `disc_md_base1` (teto 9.000) | 31,9 GB |
| 2 | d2_g2_a (1), sem VICRegL | 3.600 MiB | `disc_md_base640` (teto 30.000, medido ~25 GB) | ~35,4 GB se base640 bater no teto; medido 30,5 GB |
| 3 | d2_g3_a a e (5) | 4.500 MiB | nenhum | 27,4 GB |

**Memória medida** nas 81 cadeias do lote anterior: pico reservado 3.808 MiB (VICRegL), sem OOM. Teste real em Eggs, 3 regimes: Barlow 3.130 MiB, VICRegL 3.808 MiB, mesma largura de helminth_eggs. Medido às 07:53: 16, 14, 30 e 9 GB nas GPUs 0 a 3. Carga: pico de 114 por cerca de 1 min na entrada das 14 faixas, depois 70 a 85.

**Previsão:** ~60 milhões de imagens vistas a ~1.900 img/s: treino até por volta das 17:30, SVMs até 19:00. Margem de 1 a 2 h.

**Depois do treino:** a sessão `disc_md_law_svm2` espera as 14 faixas fecharem e roda `disc_md_law_eval_ctl2` e `_f1d2`, um por vez, sob `svm.lock`, `nice -n 10`, só validação, leitura `gem_tta_pw` em L4.

## Pronto: primeira fase da lei
57 YAMLs, 81 cadeias, 0 erros, de 02:12 a 06:56; SVMs até 07:30. Resultados em `results_log.md`.

## Parada
- Grade `sc` (`disc_md_sc0..9`): parada em 07/11 para dar lugar à descoberta. Era só piloto; checkpoints ficam em `/exps/moliveira/spifil_growth/disc_md_sc*`. Não entra na lei sem indicador de regime.
- Os 6 SVMs das réplicas w1 (L3) rodaram: κ de validação 0,870, desvio 0,016, faixa 0,041 (`results_log.md`). Misturam GPU e época escolhida pela sonda.

## Linha de base (36 runs), segue rodando
4 camadas 32/48/64/1280, 200 épocas, 50 superpixels, 3 splits.
200 px: helminth_eggs, protozoan_cysts (sem cópias), depois Protozoan, 3 subgrupos, helminth_larvae. 320 px: Eggs, flowers. 640 px: Larvae, aid, remote_sensing.

## Duas máquinas (07/11, 15:30 UTC)

- **Jaci** (`spifil_explore`, 1 GPU de 96 GB, sem `/exps`: usa `/dados/home/moliveira/exps`): `disc_md_spfdeep_esp_jaci.yaml`, `disc_md_spfcombo_w_jaci.yaml`. Saem da fila do Athena: o YAML só existe com o sufixo `_jaci`, então as filas `disc_md_spfdeep_q` e `disc_md_spfdeep_combo` terminam com erro de arquivo ao chegar neles. Esse erro é esperado.
- **Athena**: `r1` a `r5`, `h`, `h_raw`, `hz`, `sal_par`, `deep_cedo`, `combo_a`, multiescala.
- Resultados do Jaci voltam para `/exps/moliveira/results` por rsync que o usuário roda. Cada YAML grava em nomes próprios: não há colisão.

## Divisão atualizada (07/11, 20:40 UTC)

- **Jaci:** `spfdeep_esp`, `spfcombo_w`, depois `ms_a2`, `ms_b2` (7 bases). `ms_a2` e `ms_b2` só existem com sufixo `_jaci`: a fila `disc_md_ms_q` para com erro de arquivo depois de `ms_ctl2`, e esse erro é esperado. `ms_a2` só começa depois que o `ctl2` for congelado aqui.
- **Athena:** `r2` a `r5`, `h_raw`, `ms_ctl2`, `deep_cedo`, `combo_a`, e a rodada de agrupamento C3 a C9 quando o código ficar pronto (usa os mapas de saliência, que só existem aqui).
- 21:08: `spf_hz` passou para o Jaci (`disc_md_spf_hz_jaci.yaml`); a fila `disc_md_spf_hl` termina com erro de arquivo depois de `h_raw`, esperado.
- 21:20: `spf_r3`, `r4` e `r5` passaram para o Jaci (`*_jaci.yaml`). A sessão `disc_md_spf_r_corte` encerra a fila `disc_md_spf_r` assim que o `r2` termina e `congela r2` grava os sobreviventes; assim o Athena não grava `r3` a `r5` falsos no JSON. No Jaci, o caminho do JSON vem de `SPF_SURVIVORS`, e cada rodada termina com `python -m eval.spf_tabela congela rN` lá mesmo. Para isso, os CSVs de `r1` e `r2` vão junto no rsync.
- 21:25: agrupamento `disc_md_cl_ctl` e depois `disc_md_cl_a` no Athena (sessão `disc_md_cl`, logs `cl.*`), por precisar de 8 a 10 GB de RAM por job. `disc_md_salw_a_jaci` vai para o Jaci depois do rsync. `cl_*_jaci` existem, mas não rodam: o agrupamento roda só no Athena.
- 23:00: `disc_md_sol_p6` (Scattering, SoftHebb, PCANet, 12 bases) no Athena, sessão `disc_md_sol_p6`, logs `sol.*`; o Jaci está com a GPU cheia.
- 23:45: SOL Rodada 1 dividida. Athena (sessão `disc_md_sol_r1`, `max_per_gpu: 1`): `sol_p5` (pooling) e depois `sol_p2` (polaridade). Jaci, depois do rsync: `sol_p3` (diversidade funcional), `sol_p1_n6`, `sol_p1` (normalização), um YAML novo quando outro sair.

## Banco por classe (08/11, 01:58 UTC)

- Sessão `disc_md_porclasse` (Athena, fecha sozinha, logs `/exps/moliveira/tmp/disc_md_porclasse{_piloto,}.log`): piloto `disc_md_porclasse_piloto` (flowers, split 1, 64 filtros), depois `disc_md_porclasse` (flowers, Larvae_subgroup, remote_sensing, Eggs_subgroup, Protozoan_subgroup, aid, coconut_trees, food101; 3 splits; 64 e 256). Cada uma espera a carga do minuto ficar abaixo de 110.
- `amostra.por_classe: [1, 5, 10, 15, 30]` (código novo em `eval/spf_grade.py`, `bank_pool`): para cada k, `_pc{k}B.csv` (k imagens de treino por classe, regime B) e `_pc{k}A.csv` (mesmo total sorteado sem classe, regime A, controle). Coluna `regime`.
- `disc_md_porclasse_jaci.yaml` pronto para quando houver rsync; não lançado.

## Bases novas coconut_trees e food101 (08/11, 01:43–01:59 UTC, Athena)

Registradas em `core/constants.py:82-83` (`ORGANIZED`; o resto deriva). Cache 200 `no_crop` em `/exps/moliveira/image_cache/{coconut_trees,food101}/200_nocrop` (1,6 GB em 1:41; 12 GB em 27:57). Piloto coconut_trees (split 1, sorteio 42, 64 filtros, fps e random_z): 8:25, 0 erros, κ val L3 fps 0,740 / random_z 0,675 (`/exps/moliveira/results_sol/ext/`).
YAMLs `experiments/autoencoder/disc_md_*_ext.yaml` (18), só as 2 bases, `max_per_gpu: 1`. Agora roda só coconut_trees (`--set grid.datasets=[coconut_trees]`); food101 entra depois, numa fila atrás, após o piloto de food101 medir o custo. Logs `/exps/moliveira/queues/logs/ext.<yaml>.coconut.log`. Sessões fecham sozinhas; as `h*` esperam carga < 100 antes de cada YAML.

| Sessão | GPU | Sequência (coconut_trees) |
|---|---|---|
| disc_md_ext_g0 | 0 | cl_ctl_ext → cl_a_ext |
| disc_md_ext_g1 | 1 | spfcombo_a_ext → spfdeep_cedo_ext |
| disc_md_ext_g2 | 2 | spf_r1_ext → sol_p2_ext |
| disc_md_ext_g3 | 3 | piloto food101 → sol_p5_ext → sol_p6_ext |
| disc_md_ext_h0 | 0 | sol_wsal_ext → ms_ctl_ext → sha256 dos CSVs do ctl (`ext.ms_ctl_ext.coconut.sha256`) → ms_a_ext → ms_b_ext |
| disc_md_ext_h1 | 1 | spfcombo_w_ext → spfdeep_esp_ext |
| disc_md_ext_h2 | 2 | salw_a_ext → sol_p3_ext |
| disc_md_ext_h3 | 3 | sol_p1_n6_ext → sol_p1_ext |
- 01:5x: `disc_md_sol_hebb1` (SoftHebb Fila 1, 12 bases) no Athena, `max_per_gpu: 1`; sem rsync possível para o Jaci.
- 02:06: carga chegou a 133 (crispim ~58 núcleos). Sessão `disc_md_ext_gate` (log `/exps/moliveira/queues/logs/ext.gate.log`): no máximo 4 jobs `_ext` rodando; os outros ficam em SIGSTOP (nada é morto) e são soltos, o mais antigo primeiro, só com carga < 110 e RAM livre > 40 GB. Fecha sozinha quando não sobrar sessão nem job `_ext`.
- food101: ainda NÃO está na fila. A sessão que esperaria as g*/h* foi barrada pela permissão; falta lançar.
- 02:26: nossos por GPU passaram de 15 GB (GPU 2 com ~23 GB). O vigia foi trocado por `gate2.py` (mesma sessão `disc_md_ext_gate`, mesmo log): não solta nem deixa nascer job `_ext` enquanto alguma GPU tiver nossos ≥ 13 GB (nvidia-smi por usuário); job com menos de 3 min é congelado antes de abrir CUDA; solta um por vez, com 2 min entre eles, conferindo a soma antes. Nada morto.
- 11:40: `disc_md_porclasse` parado no Athena (sem teto de GPU; 8 de 24 jobs). Vai para o Jaci como `disc_md_porclasse_jaci`, com o braço da receita (`fpsw@mapa_w0.5_1`) e `fps` sem saliência.
- 12:10: food101 dos experimentos que só existiam no athena vira `*_extf_jaci.yaml` (8); SoftHebb Fila 2 vai para o Jaci (`disc_md_sol_hebb2_jaci`); a sessão armada `disc_md_sol_hebb2` do athena foi desarmada.

## Estado em 08/11, 12:40 UTC

- **Jaci:** `r4` (58/60) → `r5`; imagens por classe (`porclasse`, 7 bases, e `porclasse_food`); `sol_wsal_extf`, `spfcombo_w_extf`, `sol_hebb2`. Fila automática (`spf_jaci_fila2`): `sol_p6_extf`, `salw_a_extf`, `ms_*_extf`, `cl_*_extf`, `spfdeep_esp_extf`, `sol_p3_extf`, `sol_p1_n6_extf`, `sol_p1_extf`, `sol_p2_extf`, `sol_p5_extf`, `spfcombo_a_extf`, `spfdeep_cedo_extf`, `spf_r1_extf`.
- **Athena:** `sol_p2` (polaridade), `sol_hebb1`, coconut_trees em todos os `_ext` (vigia `disc_md_ext_gate`). Concluídos aqui: `cl_ctl`, `cl_a`, `sol_p5`, `sol_p6`, `spfcombo_a`, `spfdeep_cedo`, `spf_h_raw`, `ms_ctl2`, testes de fundo.
- Pedido em andamento: relação entre tamanho da imagem e arquitetura (Jaci).
- 08/11 13:40 UTC: novo `disc_md_gauss_jaci.yaml` (saliência gaussiana: peso por gaussiana com centro e covariância do mapa gmr, k = 1, 2, 3; `fpsw@mapa_gauss*` e `fps@foco_gauss*` contra o corte 0,5 da receita; 12 bases, 64 e 256). Código: `gauss()` em `eval/spf_grade.py`, chave `pesos.gauss`. Próxima rodada no jaci, depois do teste de largura.
- 08/11 16:53 UTC: `disc_md_gelu_jaci.yaml` (GELU contra ReLU na receita; `ACTS` ganhou "gelu" em `eval/spf_grade.py`). Fila3 do jaci, depois do sorteio 43 das leis e antes do benchmark.
- 08/11 18:45 UTC, pool de foco (aid, flowers, remote_sensing, coconut; `/exps/moliveira/reports/rodada2/pool_foco.md`):
  - **athena:** VLAD (`pf`/`vlad_off`, `vlad_on`) rodando desde 18:28; resolução (`pf_res320_off`, `pf_res128_on`, `pf_res448_off`) na fila automática `fila_athena` (lança quando o nosso máximo por GPU ≤ 10,8 GB e o load < 86).
  - **jaci:** fila3 depois do `leis_p1b_s43`: `pf_leis_off`, `pf_leis_on`, `pf_hebb_d3_*`, `pf_hebb_rand_d3`, `pf_hebb_d4_*`, `pf_hebb_rand_d4`.
  - Benchmark: por último nas duas máquinas.
