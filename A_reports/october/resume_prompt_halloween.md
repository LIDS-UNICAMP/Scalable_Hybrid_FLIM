# Prompt para retomar a descoberta (parada em 02/11/2026, ~23h UTC)

Colar como está numa sessão nova do Claude Code em Athena, a partir de
`/dados/home/moliveira/Scalable_Hybrid_FLIM`.

---

> Retome a descoberta "hybrid SPiFiL" que parou em 02/11/2026. Não refaça nada que já existe.
>
> **1. Leia antes de agir, nesta ordem:**
>
> - `Scalable_Hybrid_FLIM_discovery/A_reports/october/report_halloween_october.md`: o log completo, com resultados, leis, insights, o que foi interrompido e o que falta (seção 6).
> - `A_reports/october/discovery_recipe.md`: a receita do modelo.
> - `A_reports/october/discovery_hybrid_spifil_decisions.md`: as decisões em ordem, das últimas 40 entradas para trás.
> - `A_reports/november/report_discovery_athena.md`: o plano Athena/Jaci e a tabela de workers.
> - Também `low_data_diag.md`, `low_data_literature.md` e `scaling_law.md`.
>
> **2. Onde está o trabalho:**
>
> - **Código e YAMLs:** worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, branch **`discovery`**. Nunca na `main`.
> - **Pesos:** `/dados/home/moliveira/artifacts/spifil_growth/disc_*`.
> - **SVMs:** os CSVs ficam em `results/svm_disc_*.csv` e os artefatos em `artifacts/eval/svm_disc_*`.
> - **Jaci (só executa):** `ssh jaci`, repositório em `/dados/home/moliveira/Scalable_Hybrid_FLIM`, na mesma branch.
>   - O agente de lá é a sessão Claude `explore_experiments_jaci`: ache com `ListAgents` e fale com `SendMessage`.
>   - Os CSVs de Jaci ficam em `A_reports/november/jaci/` no repositório de lá.
>   - Antes de começar, confira se os `rsync` de volta (Jaci → Athena) já foram feitos. Se não foram, me dê os comandos.
>
> **3. Estado em que paramos:**
>
> - **Modelo final:**
>   - CNN de 4 camadas (32/48/64 filtros 5×5 com pool 2, depois uma 1×1 de 1280 filtros), ~200 mil parâmetros, entrada LAB.
>   - O SPiFiL inicializa cada camada; a 1×1 é completada com filtros aleatórios.
>   - O treino é um autoencoder camada a camada, com backprop de 0,1 nas camadas antigas (1,0 em eggs) e 200 épocas.
>   - A leitura é `gem_tta_pw`: GeM, depois TTA de 8 vistas, depois PCA-whitening com shrinkage, depois o SVM linear.
> - **Teste a 50%:** eggs 0,941, larvae 0,968, protozoan 0,928. O I-JEPA dá 0,940 / 0,943 / 0,864.
> - **Curva de 1% a 100%:** ganhamos do I-JEPA com ≥25% dos dados (≥75% em eggs) e perdemos com 1–5%.
> - **Protocolo justo** (pré-treino com 100% sem rótulo, SVM com 5% dos rótulos): ganhamos em larvae e protozoan e perdemos em eggs (0,812 contra 0,864). Com 1% dos rótulos ainda perdemos nos 3.
> - **Interrompido (seção 6 do report):**
>   - **Escolha de filtros do SPiFiL:** `e_aug_{wh,pn,div,ipc3,knn}`.
>   - **Whitening durante o treino:** `disc_nd_e200`.
>   - **Receitas:** `e_base_e400`, `p_aug200`, `p_bp10_aug`.
>   - **Lei de escala:** `disc_sl_*` em Athena e em Jaci, com 1 ponto só medido.
>   - **Validação final:** `disc_final_rep` e `disc_final_nobg`.
>   - **Baixo dado:** `disc_ld_*`, nenhum lançado.
>   - **SVMs pendentes:** pwlw, aug8, bg1, l_w96, entre outros.
>   - **Último ponto da curva:** protozoan split 3 com 100%.
>
> **4. Prioridades para retomar, nesta ordem:**
>
> 1. **SVMs pendentes das famílias que já têm checkpoint completo**, sem treino novo:
>    - ssl100 com `gem_tta_pwlw`;
>    - pcts com `aug8` e `pwlw`;
>    - bg1, l_w96 e as `disc_sl_*` com splits completos.
> 2. **Fechar a lei de escala** (`disc_sl_*`): κ, F1 e acurácia contra o número de parâmetros, com o ajuste e(N) = E + A·N^−α. O script é `scratchpad/a4/scaling_law.py`; se o scratchpad não existir mais, recrie a partir de `A_reports/november/scaling_law.md`.
> 3. **Validação do backbone:** `disc_final_rep` (o backbone repetido nos 3 datasets) e `disc_final_nobg` (a receita sem fundo).
> 4. **Baixo dado, 1% e 5%, sempre com backprop** (decisão do usuário):
>    - Ordem: `disc_ld_ipc` → `c400` → `sp200` → `saug` → `bp003` → `e1000`.
>    - `noback` entra só como controle.
>    - Usar o protocolo justo `svm_train_pct`.
> 5. **As 5 ideias de escolha de filtros do SPiFiL** e o `disc_nd_e200`.
>
> **5. Regras (todas valem):**
>
> - **Modelo:** só CNN, sem transformer. O SPiFiL inicializa e o backprop vem sempre junto. Sem fine-tuning com rótulo. Até 5 milhões de parâmetros.
> - **Avaliação:**
>   - A escolha é sempre pela validação; o teste só serve para reportar.
>   - Sem flatten, e a leitura sempre abaixo de 2000-d.
>   - O SVM só roda depois do treino, sob flock e com `nice -n 10`. Na Jaci podem rodar várias filas de SVM se a CPU estiver livre.
> - **GPU:**
>   - Athena: no máximo 20 GB por GPU, somando todos os processos do moliveira. O alvo é ~19,5 GB de pico com estimativa medida, ou 19,0 GB sem medida.
>   - Jaci: no máximo 70% da GPU (68,5 GB). Lá a GPU satura com ~12 treinos; `num_workers` 2 já basta.
> - **Experimentos:**
>   - Só via YAML em `experiments/autoencoder/`, lançado com `python -m experiments.ray.launch`.
>   - Exportar sempre `LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib:$LD_LIBRARY_PATH RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`.
>   - Em Jaci, o `gpu_ids` é `[0]`.
>   - Uma mudança por família, registrada no log de decisões com a hipótese SPiFiL.
> - **Commits:** sem trailer e sem assinatura. Só Athena commita. O push e o rsync quem roda sou eu: me dê o comando completo, em linha.
> - **Estilo:** nunca `.sh`; ponytail; vários agentes em paralelo; explicação em português, nível criança e breve.
> - **Armadilhas conhecidas:**
>   - `skip: state` não pula pontos prontos do growth.
>   - O flock não segue ordem de chegada.
>   - `pgrep` devolve o PID do servidor tmux.
>   - O `_pw` sem shrinkage colapsa.
>   - Em Jaci, `artifacts/spifil_growth` precisa de symlink.
>
> Comece me dando, em uma tabela, o que vai rodar primeiro, onde (Athena ou Jaci) e quanto de GPU cada coisa usa. Só depois lance.
