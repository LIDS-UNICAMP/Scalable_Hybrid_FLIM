# ga_search — o que dá para rodar aqui e como rodar

> Escrito em 20/11/2026 (branch `genetic_algorithm`). Todos os smokes abaixo foram rodados **de verdade, só em CPU**,
> com saída em `/dados/home/moliveira/artifacts/_smoke_doc_ga/<tipo>`.

## O que dá para rodar aqui

| Tipo / ferramenta | Entrada | O que faz |
|---|---|---|
| MO-DEHB v3 (busca híbrida) | `modehb/run_search.py` + `config/modehb/search_v3.yaml` | GA+Hyperband multiobjetivo (κ val × FLOPs), população SPiFiL, leitura cov_head + scaler |
| MO-DEHB v3 com semente | `search_v3_seeded.yaml` | igual ao v3, com sementes externas (SPiFiL pequeno) |
| LEMONADE | `search_v3_lemonade.yaml` | evolução por morfismos de rede (filhos herdam pesos do pai) |
| LEMONADE com semente | `search_v3_lemonade_seeded.yaml` | LEMONADE partindo do SPiFiL pequeno (variantes > 5,5M são puladas) |
| GA v4 | `search_v4.yaml` | MO-DEHB com faixas B0–B4 até 5,5M, leitura so2_4096 **sem scaler**, OOM sem punição |
| PNAS-CS | `pnas_cs.yaml` | Progressive NAS com compound scaling a partir do SPiFiL pequeno (régua v3) |
| PNAS-CS v4 | `pnas_cs_v4.yaml` | PNAS-CS com a régua do GA v4 (so2_4096 sem scaler) |
| H-MODEHB (braços A…) | `hmodehb/run_hsearch.py --arm A` + `config/hmodehb_b0b4/base.yaml` | busca hierárquica B0–B4 (fitness v2 ORIG) |
| Cadeia de crescimento B0→B4 | `hmodehb/growth_chain.py` + `config/hmodehb_b0b4/growth*.yaml` | cresce a rede faixa a faixa por morfismos sem estragar o SPiFiL |
| GA antigo (pymoo) | `run_ga.py` + `config/ga.yaml` | Etapa 1 antiga: MixedVariableGA por dataset × split × pct |
| Crescimento pelo launcher Ray | `python -m experiments.ray.launch experiments/autoencoder/growth_*.yaml` | crescimento ANTIGO (`flim.spifil.grow` + autoencoder FLIM); **não** é a cadeia B0→B4 |
| Rede universal | `universal_model_ga/cross_dataset_evaluator.py` | não é GA: reconstrói cada arquitetura achada pelo GA em cada base e mede quem generaliza |
| Leis de escala / plots | `run_scaling.py`, `analyze_laws.py`, `plot_laws.py` | ajustes e gráficos sobre resultados já existentes |
| Testar o melhor | `research/spifil_b0b4/readout_x/{select_ga_best,run_ga_best,eval_backbone_head,run_all_untested}.py` | teste com SVM original + TTA8, depois da busca |

## A régua (vale para todos)

- **Fitness = κ de VALIDAÇÃO da SVM original** (`core.svm.fit_svm`: linear, C=100, ovo, `max_iter=-1`) **+ FLOPs**.
  O **teste nunca seleciona**; só é medido depois.
- **Leitura:** régua antiga (v3, LEMONADE, PNAS-CS, H-MODEHB) = `cov_head` + StandardScaler (no log aparece
  `scaler=standard`). Régua v4 (GA v4, PNAS-CS v4) = `so2_4096` **sem scaler** (`scaler=none`). Confirmado nos logs dos smokes.
- **Teto 5,5M parâmetros:** `param_cap: 5500000` (v4) / `hard_cap: 5500000` (growth). Acima disso = infeasible, antes do treino.

## Pré-requisitos (todos os tipos)

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM
export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib   # OBRIGATÓRIO (senão GLIBCXX_3.4.29 no PIL)
PY=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python
```

- Dados: `/dados/home/moliveira/datasets/organized/<ds>` (flowers, remote_sensing, ham10000, aid, coconut_trees).
- `artifacts/` do repo é symlink para `/dados/home/moliveira/artifacts` (rsync precisa de `-k`).
- Pesos FLIM: só o crescimento antigo do launcher (`init: flim`). Os PNAS-CS / *_seeded leem o SPiFiL pequeno
  (`artifacts/external_ssl_pequeno/...`), que já existe.
- Bases permitidas: só as 5 naturais acima. **Nenhum parasita** (nem pre, nem raw).

## Smoke em CPU (o mesmo molde para todo `run_search`)

Molde (troque `<CFG>`):

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 timeout 900 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/<CFG>.yaml --dataset remote_sensing --seed 42 --set device=cpu --set wandb.enabled=false --set output_dir=/dados/home/moliveira/artifacts/_smoke_doc_ga/<CFG> --set 'search.fidelities=[1,3]' --set search.max_brackets=1 --set search.initial_size=1 --set search.pnas_cs.phase0.enabled=false --set search.stratified.calibration=false --set data.num_workers=8 --set train.max_images_per_epoch=128
```

- `<CFG>` ∈ `search_v3`, `search_v3_seeded`, `search_v3_lemonade`, `search_v3_lemonade_seeded`, `search_v4`, `pnas_cs`, `pnas_cs_v4`.
- `search.pnas_cs.phase0.enabled=false` só importa nos PNAS-CS; nos outros é inofensivo.
- As chaves do `--set` são as do config **já traduzido** (`search.*`, `train.*`, `data.*`, `output_dir`, `device`, `wandb.*`).

H-MODEHB (braço A):

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 timeout 900 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.hmodehb.run_hsearch --arm A --dataset remote_sensing --seed 42 --set device=cpu --set wandb.enabled=false --set output_dir=/dados/home/moliveira/artifacts/_smoke_doc_ga/hsearch_A --set 'search.fidelities=[1,3]' --set search.max_brackets=1 --set search.initial_size=1 --set search.stratified.calibration=false --set data.num_workers=8 --set train.max_images_per_epoch=128
```

Cadeia de crescimento B0→B4 (`--set` = treino/dados, `--gset` = config de crescimento):

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 timeout 900 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.hmodehb.growth_chain --dataset remote_sensing --seed 42 --device cpu --out /dados/home/moliveira/artifacts/_smoke_doc_ga/growth_chain --growth-config ga_search/config/hmodehb_b0b4/growth_v3.yaml --set wandb.enabled=false --set data.num_workers=8 --set train.max_images_per_epoch=128 --gset fid_low=1 --gset fid_high=3 --gset generations=1 --gset n_children=1 --gset n_parents=1 --gset end_band=B1
```

GA antigo (só estima o custo; o `--smoke` dele usa `helminth_larvae` e grava em `ga_search/results_smoke`):

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_VISIBLE_DEVICES= /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.run_ga --smoke --estimate
```

Crescimento antigo pelo launcher (só o plano, nada executa):

```bash
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/growth_grid4.yaml --dry-run --set skip=state
```

Rede universal: só `--help` foi testado (o `build`/`eval` real leva horas e `report` reescreve `universal_model_ga/runs/v1`).

## Execução real (GPU 2 ou 3 do athena)

**Quem tem YAML no padrão `experiments/<metodo>/<nome>.yaml` + `experiments.ray.launch`:** só o crescimento ANTIGO
(`experiments/autoencoder/growth_*.yaml`, runner `growth`). O schema (`experiments/ray/schema.py`) só conhece os
runners `train`, `growth`, `eval`. **Nenhum** dos GAs novos (MO-DEHB, LEMONADE, GA v4, PNAS-CS, H-MODEHB, cadeia
B0→B4) está no launcher: eles rodam direto com `--config <yaml>` (o YAML é o experimento; `--set` só para override).
Para seguir a regra "experimento novo = YAML", copie o YAML irmão em `ga_search/config/modehb/` em vez de empilhar `--set`.

```bash
# MO-DEHB / LEMONADE / GA v4 / PNAS-CS (troque o yaml e o dataset); sessão tmux fecha sozinha, log via tee
tmux new -d -s ga_v4_rs 'cd /dados/home/moliveira/Scalable_Hybrid_FLIM && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.modehb.run_search --config ga_search/config/modehb/search_v4.yaml --dataset remote_sensing --seed 42 2>&1 | tee logs/ga_v4_rs_$(date +%Y%m%d_%H%M).log'

# H-MODEHB
tmux new -d -s hs_A_rs 'cd /dados/home/moliveira/Scalable_Hybrid_FLIM && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.hmodehb.run_hsearch --arm A --dataset remote_sensing --seed 42 2>&1 | tee logs/hs_A_rs_$(date +%Y%m%d_%H%M).log'

# Cadeia B0→B4 (out NOVO para cada config nova: fitness_sig muda)
tmux new -d -s growth_rs 'cd /dados/home/moliveira/Scalable_Hybrid_FLIM && export LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 && /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m ga_search.hmodehb.growth_chain --dataset remote_sensing --seed 42 --device cuda --growth-config ga_search/config/hmodehb_b0b4/growth_v3.yaml --out artifacts/hmodehb_b0b4/growth_v3/remote_sensing 2>&1 | tee logs/growth_rs_$(date +%Y%m%d_%H%M).log'

# Crescimento antigo via launcher (YAML no padrão; GPUs via resources.gpu_ids do YAML — use [2,3])
cd /dados/home/moliveira/Scalable_Hybrid_FLIM && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/<nome>.yaml

# Rede universal (depois que os GAs terminarem)
$PY -m universal_model_ga.cross_dataset_evaluator --stage build --workers 4 --threads 8
CUDA_DEVICE_ORDER=PCI_BUS_ID $PY -m universal_model_ga.cross_dataset_evaluator --stage eval --device cuda:2 --shard 0/2
$PY -m universal_model_ga.cross_dataset_evaluator --stage report
```

## Saídas, retomada e teste do melhor

- **Onde fica:** `paths.state_dir` do YAML + `seed<N>`, ex. `artifacts/modehb_v4/<ds>/state/seed42/` com
  `search_state.json` (tudo que já foi avaliado) e `candidates/` (um diretório por candidato: encoder, métricas).
  Pareto/logs/checkpoints em `artifacts/modehb_v4/<ds>/{pareto,logs,checkpoints}`. Cadeia B0→B4: tudo dentro do `--out`.
- **Retomar:** é o **mesmo comando** (no `run_search`, `--resume` explícito faz o replay do `search_state.json`).
  O replay compara o config: se mudou algo que altera a busca → `ConfigDrift` e para. Na cadeia B0→B4 o drift é
  aceito por chave com `--allow-drift <chave>`.
- **Testar o melhor** (sempre depois, nunca na seleção): `research/spifil_b0b4/readout_x/select_ga_best.py` →
  `run_ga_best.py` (roda `eval_backbone_head.py` com SVM original + `--tta8`, cabeças gap1280 + so2_4096_pca + TTA8);
  `run_all_untested.py` varre todos os modelos sem teste (ga/pnas/growth) e escolhe GPU pela memória real.

## Armadilhas conhecidas

- **`memory_budget_gb` muda o replay:** a fila sobrescreve conforme a GPU; retomar com outro valor muda o plano → ConfigDrift / pontos refeitos. Retome com o mesmo valor.
- **OOM de contenção:** dois processos na mesma GPU estouram; v4 trata OOM sem punir o candidato (retry), mas não lance dois GAs na mesma GPU sem teto (`gpu_process_cap_gb`).
- **Só GPUs 2 e 3** do athena para treino; 0/1 só testes ≤ 1 h.
- **Launcher Ray (growth antigo):** precisa `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`; matar o processo perde o `skip:state` (refaz pontos — confira "pulado(s)").
- **Não mexer no git da pasta principal com runner vivo** (checkout/troca de branch troca o código no meio da busca). Use um worktree/clone para experimentar.
- `LD_LIBRARY_PATH` esquecido derruba tudo no import do PIL.
- `run_ga.py` tem que rodar como módulo (`-m ga_search.run_ga`); como script dá `ModuleNotFoundError: ga_search`.
- O `--smoke` do GA antigo usa `helminth_larvae` (parasita): serve só para ver se o código ainda importa e estima.
- `pgrep` pode devolver o PID do servidor tmux: confira com `ps -o cmd` antes de matar.

## Resumo dos smokes (CPU, remote_sensing, 20/11/2026)

Os 10 smokes rodaram **ao mesmo tempo** na CPU (OMP 12 cada), com timeout de 15 min. Todos chegaram à fitness
de verdade (SVM C=100 no κ de val) e gravaram `search_state.json` + `candidates/`, mas nenhum fechou o bracket
no tempo. Com `output_dir` sobrescrito o estado fica direto em `<output_dir>/` (sem `seed42/`). Retomar = mesmo
comando + `--resume`.

| tipo | config | comando smoke | status | tempo |
|---|---|---|---|---|
| MO-DEHB v3 | `search_v3.yaml` | molde `run_search` | não terminou (3 candidatos, 9 fits SVM; scaler=standard) | >15 min |
| MO-DEHB v3 semente | `search_v3_seeded.yaml` | molde `run_search` | não terminou (4 cand., 18 fits) | >15 min |
| LEMONADE | `search_v3_lemonade.yaml` | molde `run_search` | não terminou (2 cand., 5 fits) | >15 min |
| LEMONADE semente | `search_v3_lemonade_seeded.yaml` | molde `run_search` | não terminou (4 cand., 14 fits; variantes >5,5M puladas) | >15 min |
| GA v4 | `search_v4.yaml` | molde `run_search` | não terminou (3 cand., 9 fits; scaler=none) | >15 min |
| PNAS-CS | `pnas_cs.yaml` | molde `run_search` | não terminou (3 cand., 9 fits) | >15 min |
| PNAS-CS v4 | `pnas_cs_v4.yaml` | molde `run_search` | não terminou (2 cand., 8 fits) | >15 min |
| H-MODEHB A | `hmodehb_b0b4/base.yaml` | `run_hsearch --arm A` | não terminou (3 cand., 9 fits) | >15 min |
| Cadeia B0→B4 | `hmodehb_b0b4/growth_v3.yaml` | `growth_chain` | não terminou (seeds B0 pontuadas, 7 fits; sem filho aceito ainda) | >15 min |
| GA antigo | `config/ga.yaml` | `run_ga --smoke --estimate` | ok (estimativa: 2,4 h na Etapa 1) | 6 min |
| Crescimento launcher | `experiments/autoencoder/growth_grid4.yaml` | `launch --dry-run` | ok (18 pontos, 144 comandos, nada executado) | 15 s |
| Rede universal | — | `cross_dataset_evaluator --help` | ok só import/CLI | 0 s |
