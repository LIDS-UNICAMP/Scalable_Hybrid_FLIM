# Uso de recursos e ajuste de CPU

Branch `gpt_sol_G_A`, sessão `gpt_sol_G_A_ga`, GPU física 2. O experimento
continua com os mesmos dados, genomas, perda, batch size, sementes, splits,
limites e checkpoints. Um overlay de execução aplica recursos adicionais sem
invalidar a identidade científica e registra seus próprios hashes.

## Medições

Antes: média de 100% de CPU do processo (um núcleo), 1,89 GiB RSS, 1.036 MiB
de memória da GPU e 11% de utilização GPU em uma janela externa de 20 s.
Há 96 CPUs lógicas/48 núcleos físicos e quatro RTX A6000 de 48 GB; o job usa
somente a GPU 2.

Depois: 16 threads numéricas, 32 workers de pré-processamento e até 32 tarefas
DBCV. O carregamento chegou a aproximadamente 1.600% de CPU agregado (16
núcleos equivalentes). A inferência apresentou amostras de utilização GPU
até 71%, com cerca de 800–844 MiB de VRAM. O agrupamento HDBSCAN conserva um
trecho serial: nessa etapa, uma janela externa de 20 s mediu 100,57% CPU e
0% GPU. Portanto o pico não deve ser confundido com utilização contínua.

As imagens pré-processadas ocupam 2.295.360.000 bytes em RAM, mantendo os valores
originais float32. O RSS do processo principal ficou em aproximadamente
4,63 GiB. A soma dos RSS dos processos de cache ficou perto de 23 GiB, mas inclui
páginas de bibliotecas compartilhadas e não representa memória física exclusiva.

Benchmark matricial no tamanho da validação (4.547×1.280, float64):

| Threads | Tempo, s |
| ---: | ---: |
| 1 | 0,6596 |
| 8 | 0,2126 |
| 16 | 0,1643 |
| 24 | 0,1715 |
| 32 | 0,1968 |
| 48 | 0,1854 |

Foram escolhidas 16 threads pelo tempo medido; usar 48 não foi mais rápido.

Também foi testada a divisão da consulta KDTree em threads: em 1.000×1.280,
serial levou 1,402 s, 16 threads 2,348 s e 32 threads 2,508 s. Embora os
resultados fossem exatamente iguais, essa alteração foi descartada por ser
mais lenta. A consulta serial com 4.547 amostras levou 28,904 s no benchmark;
esse trecho explica o núcleo ocupado enquanto a GPU aguarda.

## Alterações aplicadas

- `ga_search/progressive_cpu.py`: leitura paralela uma vez, em processos loky
  sem CUDA e com uma thread interna por worker; cache em RAM preserva tensores,
  IDs, labels, batches, ordem e consumo do RNG Torch ao iterar o DataLoader.
- `ga_search/progressive_dbcv.py`: clusters e pares em paralelo e reconstrução
  da MST sem interseções repetidas; testes verificam igualdade exata dos
  resultados, inclusive empates, ruído e embeddings de 1280 dimensões.
- `ga_search/progressive_runtime.py`: limites reais das bibliotecas já
  carregadas, overlay dos helpers, timers por etapa e monitor de recursos.
  A configuração legada do launcher sobrescrevia OMP com 1 no import;
  somente editar o shell não resolveria isso.

Seis testes dos helpers passaram, e uma demonstração sintética completa de duas
profundidades terminou com o overlay. O job real foi interrompido de forma
controlada e retomado pelos checkpoints, preservando as épocas concluídas.
O backup de segurança está em `results/nas_search/before_cpu_tuning`.

## Tempo estimado

Nos primeiros cinco ciclos instrumentados, os tempos completos por época foram
36,93; 34,53; 34,26; 35,93 e 35,77 s. A mediana foi **35,77 s**. Treino:
aproximadamente 0,6–0,9 s; extração: 3 s; reconstrução: 4 s; agrupamento:
26–29 s, dos quais DBCV levou cerca de 2–4 s. O restante é majoritariamente
HDBSCAN em alta dimensão, que tem uma implementação serial no algoritmo usado.

Estimativa inicial da busca restante: **4–10 horas**. É uma projeção, não um
horário garantido: considera até vinte descendentes adicionais após a primeira
geração e arquiteturas viáveis com cerca de três a cinco profundidades.
Rejeições e cache encurtam a execução; arquiteturas mais profundas podem
ultrapassar essa faixa. O indivíduo atual de quatro profundidades tem ordem de
15–25 minutos restantes nesse ritmo. A estimativa deve ser atualizada à medida
que as gerações forem concluídas.

Dados medidos e premissas: `reports/progressive_nas_cpu_tuning.json`.
Telemetria contínua: `results/nas_search/runtime_timings.jsonl`,
`runtime_resources.jsonl`, `runtime_profile.json` e `runtime_cache.json`.

```bash
tmux attach -t gpt_sol_G_A_ga
tail -f results/nas_search/console.log
```
