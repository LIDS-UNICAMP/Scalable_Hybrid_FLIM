# Buscas com 50% de treino

Lançadas na branch `gpt_sol_G_A`, split 1 e seed 42. A execução protozoan
de 5% em `results/nas_search` foi interrompida de forma controlada; seus
checkpoints e histórico foram preservados. As duas buscas de 50% começam
em diretórios novos, com filtros obtidos dos respectivos conjuntos de treino.

| Dataset | Treino | Validação | GPU física | Sessão tmux | Saída |
| --- | ---: | ---: | ---: | --- | --- |
| protozoan-cysts 50% | 2.390 | 2.392 | 2 | `gpt_sol_G_A_proto50` | `results/nas_protozoan_50` |
| helminth-eggs 50% | 1.276 | 1.279 | 3 | `gpt_sol_G_A_eggs50` | `results/nas_eggs_50` |

Foram verificados IDs/paths de treino e validação disjuntos, sete doadores
SPiFiL no protozoan e nove no eggs, todos do treino. São usados os descritores
de split existentes do percentual de 50%; esses conjuntos de validação diferem
dos descritores de 5%, portanto comparações entre percentuais exigem essa
distinção. O teste não é carregado.

Cada busca usa população seis, até cinco gerações e dez blocos, cinco épocas
por fase, lambda 0,004, cobertura mínima 0,8 e os limites computacionais
configurados. Recursos por busca: 16 threads numéricas, 32 workers de cache
e até 32 tarefas DBCV. As GPUs são distintas e os artefatos são separados.

Configurações versionáveis:

- `ga_search/config/progressive_nas_protozoan_50.yaml`
- `ga_search/config/progressive_nas_eggs_50.yaml`

Cada diretório de saída contém `run.yaml`, `launch.sh`, `launch.json`,
`preflight.json`, `console.log` e a telemetria de execução.

```bash
tmux attach -t gpt_sol_G_A_proto50
tmux attach -t gpt_sol_G_A_eggs50

tail -f results/nas_protozoan_50/console.log
tail -f results/nas_eggs_50/console.log
```

`Ctrl-b d` desanexa sem interromper. Os resultados finais ainda estão pendentes.
