# Continuidade: pausa geral de 09/11/2026 (~02:00 UTC)

A pedido do usuário, todos os treinos e execuções foram pausados nas duas máquinas, e as máquinas foram liberadas. O rsync está bloqueado para o Claude (`deny` em `~/.claude/settings.json`; cópia anterior em `settings.json.bak_antes_rsync_block`). A sessão `spifil_explore` no jaci foi encerrada.
Resultados, catálogo e decisões: `multi_dataset/ablacao_completa.md` (índice), `ablacao/`, `multi_dataset/decisoes_spifil.md`, `decisao_E0_E7/decisao.md`, `encdeep/G3_pilotos.md`, `analise_textura/`, `relatorio_07_08nov.md`, e `/exps/moliveira/reports/rodada2/{plano,auditoria,pool_foco}.md`.

## Como retomar
Todo YAML retoma pelas linhas `ok` do CSV (o `spf_grade` pula células completas). O `skip: state` do lançador NÃO sobrevive a kill: os jobs voltam a ser enviados, mas as células prontas são puladas.
- athena: `cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib:$LD_LIBRARY_PATH RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 GPU_CAP_MIB=4200 PYTHONPATH=/exps/moliveira/queues/gpucap OMP_NUM_THREADS=2 RAY_TMPDIR=/exps/moliveira/tmp/ray TMPDIR=/exps/moliveira/tmp; nice -n 10 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/<yaml> --set 'resources.gpu_ids=[...]'`. Teto: 15 GB nossos por GPU (picos curtos permitidos). Os YAMLs `hyb_*` precisam de `GPU_CAP_MIB=6000` no athena (oficial) e de `14000` no jaci (6×256).
- jaci: o mesmo, com os caminhos `/dados/home/moliveira/exps/...` e os YAMLs `_jaci`. A GPU é compartilhada com o root (deixar ≥ 4 GB livres), RAM disponível ≥ 15 GB.

## O que foi interrompido

### athena (estado às ~02:00 UTC)
| YAML (experiments/autoencoder/) | progresso | observação |
|---|---|---|
| `disc_md_gelu.yaml` (na sessão era `gelu`) | 16/36 jobs | GELU contra ReLU, 6 bases |
| `disc_md_leis_p1b.yaml` (protozoários, sorteio 42) | 4/6 | leis de construção |
| `disc_md_leis_p1c_s43.yaml` (protozoários, sorteio 43) | 4/6 | leis de construção |
| `disc_md_plano_helminth.yaml` | 8/12 | regra do fundo preto (`plano_tol`) antes e depois |
| `disc_md_e07_off.yaml` | 0/4 | E1/E3/E7 em aid e remote_sensing (prioridade do usuário) |
| `disc_md_pf_res448_off.yaml` | 1/4 | 448 px em aid e remote_sensing |
| `hyb_ham10000_oficial.yaml` | 0/3 pontos | modelo oficial `disc_bp01w64c1280_bg1` (backprop) no HAM10000 a 50% |
| `disc_md_*_ext.yaml` (coconut_trees nos experimentos antigos, 8 sessões + gate) | parcial | baixa prioridade |
| `svm_layerwise5` (laço de SVM de outra sessão, YAMLs `eval_svm_layerwise5_{aug,bn}_avg.yaml`) | parcial | interrompido junto, por "todas as execuções" |
| `bench_athena` (12 modelos) | não começou | estava esperando o fim do SPiFiL |
A fila automática `fila_athena` estava vazia quando parou.

### jaci (do diário `/dados/home/moliveira/exps/queues/logs/diario_offline.md`)
| YAML `_jaci` | progresso | faltam |
|---|---|---|
| `disc_md_porclasse_jaci.yaml` | 11/21 | 10 |
| `disc_md_gauss_jaci.yaml` | 44/72 | 28 |
| `disc_md_largura_jaci.yaml` | 8/12 | 4 |
| `disc_md_leis_p1_s43_jaci.yaml` | 2/4 | 2 |
| `disc_md_encdeep_nw_on_jaci.yaml` | 10/12 | 2 |
| `disc_md_e07_on_jaci.yaml` | 2/4 | 2 |
| `hyb_ham10000_jaci.yaml` (piloto) | 0/1 | refazer; depois `hyb_ham10000_jaci_conf.yaml` |
| `hyb_ham10000_jaci_rand.yaml` | 0/3 | 3 |
Nunca começaram (fila3): `leis_p1b_s43`, `pf_leis_off`, `pf_leis_on`, `pf_res384_on_b`, `pf_res512_on_b`, `hyb_ham10000_jaci_conf`; o benchmark de 6 modelos.
Pausados de propósito antes: food101 (16 experimentos, lista no diário), `disc_md_spf_r5_jaci.yaml.pausado` (protocolo antigo; a recomendação é cancelar), `sol_hebb2` (19/36).

## Ordem recomendada para retomar
1. `e07_off` / `e07_on` (respondem E1, E3 e E7 do relatório de decisão).
2. `encdeep_nw_on` (fecha a confirmação de "sem whitening em L4/L5"; em aid já está confirmado +0,020).
3. `hyb_ham10000_oficial` (athena) e `hyb_ham10000_jaci` → `_conf` + `_rand` (jaci): backprop híbrido no HAM10000.
4. `plano_helminth`, `pf_res448_off`, `pf_res384/512_on_b` (fecham a resolução).
5. `gelu`, `porclasse`, `gauss`, `largura`, as leis restantes.
6. Benchmark por último.

## Decisões em aberto para o usuário
- Cancelar o `r5` de vez.
- `aimv2_large_native` (224 px, nativo ou pular) e o login do DINOv3 no benchmark.
- Commit dos ~630 arquivos fora do git (Fase 0 da refatoração; plano em `auditoria_padrao_repo.md`).
- O teste de limitar quantos filtros cada imagem fornece (hipótese da análise `kernels_classe`).
- Ler só a última camada no coconut_trees (confirmar com 3 splits).
- W&B ligado nos YAMLs `hyb_*` (um run falho `nm67ebyj` foi registrado no jaci).

Estado bruto do athena no momento da pausa: `continuidade_09nov_athena_estado.txt` (nesta pasta). Últimas linhas do diário do jaci: `/exps/moliveira/queues/jaci_diario_ultimas.txt`.

## Atualização (09/11, ~16:00 UTC): curvas 1–100% retomadas a pedido do usuário
- Rodando no athena (tmux `curva_receita`): `hyb_curva_receita.yaml`, a receita SPiFiL (6 × 256, k5, pools 2-2-2-2-1-1, 200 px) com o backprop layerwise do oficial (200 épocas por camada, paciência 30, best por val/loss). Protozoan, Eggs e Larvae impuros; 1, 5, 25, 50, 75 e 100%; 3 splits; 54 treinos; 1 job por GPU com GPU_CAP_MIB=20000 (2 por GPU com 12 GB deu OOM). Depois, na mesma cadeia: eval val → eval test → `hyb_curva_receita_rand` (controle aleatório) → evals do rand.
- As curvas do oficial `hyb_curva_parasitos` foram paradas: o usuário já tinha esses resultados.
- **Cuidado com a avaliação:** o avaliador (`experiments/ray/runners/eval.py` `_plan_growth_stages`) procura `<artifacts_dir>/<family>/<arm>/<estagio>/checkpoints/best_kappa.ckpt`, mas o growth grava em `<arm>/<Lk>/flim-ssl/<id>/checkpoints/`. Os evals `eval_svm_hyb_curva_receita*` apontam para a árvore-espelho `/exps/moliveira/eval/espelho_growth` (a do rand é `/exps/moliveira/eval/espelho_growth_rand/hyb_curva_receita_rand/...`). **Os atalhos são refeitos sozinhos a cada 2 min pela sessão tmux `espelho_sync` enquanto `curva_receita` roda; se ela não estiver rodando, refazer à mão antes de cada eval** com: `src=/exps/moliveira/spifil_growth/<family>; M=<espelho>; for c in $(find $src -path '*/L*/flim-ssl/*/checkpoints/best_kappa.ckpt'); do arm=$(echo $c | sed "s#$src/##" | cut -d/ -f1); st=$(echo $c | sed "s#$src/##" | cut -d/ -f2); d=$M/<family>/$arm/$st; mkdir -p $d/checkpoints; ln -sfn $c $d/checkpoints/best_kappa.ckpt; done`. Se a cadeia rodar o eval antes disso, ele não acha nada (0 ckpt).
- Avaliação parcial (só L1, Protozoan, split 1, val): 1% κ 0,287; 5% κ 0,511 (`results/svm_hyb_curva_receita_parcial_L1.csv`). DINOv2-giant e I-JEPA ViT-g: curvas completas em `results/benchmark/{dinov2-giant,ijepa_vitg16_22k}.csv`.
