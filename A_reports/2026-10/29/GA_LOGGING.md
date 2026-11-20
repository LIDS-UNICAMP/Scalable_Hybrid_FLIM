# Logs de GA, W&B e monitoramento

As buscas NAS têm logs locais estruturados e uma integração W&B opcional.
Os experimentos de protozoan 50% e eggs 50% habilitam envio **online** ao
projeto `ophira-ai/progressive_nas_ga`. Outros experimentos mantêm logging
remoto desabilitado por padrão.

## Eventos e métricas

`events.jsonl` registra sequência monotônica, horário, dataset/configuração,
gerações, início/fim de avaliações, genomas solicitados/efetivos, canais reais,
cache/elitismo, pais, crossover, mutação, prefixos invalidados, motivos de
inviabilidade, estatísticas da população, melhor indivíduo e fim/interrupção.

Também registra cada época/profundidade: fases A/B, L1 de treino/validação,
J, DBI e DBI normalizado, Hoyer e dispersão, cobertura, DBCV/status, número de
clusters, custos, checkpoint e obtenção dos filtros SPiFiL. O overlay de CPU
envia tempos por etapa e uso de recursos ao mesmo run W&B.

Curvas principais no W&B:

- `ga/J_min`, `ga/J_mean`, `ga/J_max`, `ga/best_valid_J`;
- `ga/valid_count`, `ga/invalid_count`, `ga/generation`;
- `validation/J`, `validation/J_valid`, `validation/valid`, `validation/L1`;
- `clustering/DBI`, `DBI_norm`, `Hoyer`, `Hoyer_std`, `coverage`, `DBCV`;
- `fitness/clustering_term`, `fitness/sparsity_term`;
- `model/*`, `spifil/*`, `timing/*`, `resources/*`, `svm/*`.

`validation/J` mantém a penalidade determinística dos inválidos.
`validation/J_valid` e os resumos de fitness só consideram avaliações válidas,
evitando curvas dominadas por 1e12. O W&B mostra gerações/índices a partir de 1;
os eventos e checkpoints internos mantêm os índices originais a partir de 0.

Tabelas `ga/individual` incluem arquitetura, status e motivo. Snapshots dos
logs/configurações/identidade/estado do GA são enviados como artefatos
`nas-logs` em cada fim de geração e no encerramento. Os snapshots têm diretórios
exclusivos por sequência para permitir upload assíncrono. Imagens, máscaras e
pesos não são enviados por esse mecanismo. Credenciais são removidas antes de
configuração, eventos ou snapshots serem serializados para o SDK.

## Configuração

```yaml
wandb:
  enabled: true
  entity: ophira-ai
  project: progressive_nas_ga
  name: ga_protozoan-cysts_split1_pct50_seed42
  group: gpt_sol_G_A_pct50
  tags: [gpt_sol_G_A, spifil, progressive, genetic-search, pct50, split1]
  mode: online
  upload_artifacts: true
  init_timeout: 90
```

O run ID, a URL e a sequência são persistidos em `logger_state.json`.
Retomar o mesmo diretório usa o mesmo run ID. Modo online não cai
silenciosamente para offline. Falhas do logger principal interrompem o
experimento como falhas de infraestrutura, sem penalizar uma arquitetura.
Falhas isoladas da telemetria opcional de runtime são registradas em
`runtime_logging_errors.jsonl` e ficam disponíveis para inspeção.

## SVM complementar

As novas configurações de 50% usam `svm_scaler: true` e
`svm_max_iter: 10000`, mantendo o SVM central do repositório e seus diagnósticos.
O ajuste anterior de protozoan ficou por horas com `max_iter=-1`; o limite
evita iterações ilimitadas. `svm/fit_status=1` indica que o solver atingiu o
limite sem convergir; esses valores devem ser interpretados com esse aviso.
Normalização/limite alteram o protocolo complementar, não o fitness J usado
para selecionar arquiteturas/checkpoints. As durações e o protocolo são
registrados por profundidade.

## Relançamentos

Saídas novas, preservando os diretórios anteriores:

| Dataset | GPU | Sessão | Saída |
| --- | ---: | --- | --- |
| Protozoan 50%, split 1 | 2 | `gpt_sol_G_A_proto50` | `results/nas_protozoan_50_wandb` |
| Eggs 50%, split 1 | 3 | `gpt_sol_G_A_eggs50` | `results/nas_eggs_50_wandb` |

Runs online verificados:
[protozoan](https://wandb.ai/ophira-ai/progressive_nas_ga/runs/bb6b6236d6cc) e
[eggs](https://wandb.ai/ophira-ai/progressive_nas_ga/runs/76e29b73043e).

Continuam 16 threads numéricas, 32 workers de cache e até 32 tarefas DBCV por
busca. `launch.sh` verifica a branch `gpt_sol_G_A` e usa `--resume` dentro do
diretório novo; a primeira execução começa do zero com a nova identidade.

## Monitor

```bash
PY=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python
"$PY" -m ga_search.progressive_monitor \
  results/nas_protozoan_50_wandb results/nas_eggs_50_wandb --interval 60
```

O daemon escreve `reports/ga_monitor_latest.md` e
`results/ga_monitor/status.json`, sem modificar o experimento. Consulta eventos,
resultados, estado do GA e a API W&B. Diferencia URL local de upload verificado,
tolera consistência eventual e alerta após atraso ou falhas repetidas. Não
duplica arquiteturas por retenção de elites/cache. O agente revisor registra
interpretações em `reports/ga_monitor_insights.md`.

O monitor verifica cobertura/ruído, melhora de J, causas de inviabilidade,
diversidade, custos e gargalos. Queda da L1 sem melhora de J não é tratada como
melhora dos clusters; resultados entre datasets/percentuais diferentes não
comprovam superioridade relativa.

## Validação

Uma busca sintética real com SPiFiL, duas arquiteturas e uma geração confirmou
via API W&B **34 registros de histórico**, eventos por época/indivíduo/geração
e **dois artefatos nas-logs**. Essa verificação está separada dos dados reais,
com nome/tag `upload-verification-synthetic`.

Os testes incluem SDK W&B offline real, snapshots sem dados/pesos, remoção de
credenciais, resumo válido, identidade/sequência em retomada, preservação do
estado/RNG do GA e falha de logging não convertida em inviabilidade.
A suíte final passou os 47 testes em 46,816 s; registros em
`reports/ga_logging_validation/`.

```bash
"$PY" -m analysis.checks.check_progressive_nas
```

W&B 0.27 rejeita o booleano usado pelo CLI `python -m unittest` no teardown
protobuf. O runner acima usa unittest normalmente e fecha o SDK com código
inteiro, sem modificar a instalação W&B. O serviço local `wandb-core` precisa
de permissões de socket; no ambiente desta sessão, os testes SDK foram rodados
fora do sandbox. Isso não se confunde com envio online: os testes offline não
comprovam upload, que foi verificado separadamente pela API.
