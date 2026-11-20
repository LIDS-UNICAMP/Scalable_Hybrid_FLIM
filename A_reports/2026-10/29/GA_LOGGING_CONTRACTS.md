# Logging GA e W&B — contratos e propriedade

Branch: gpt_sol_G_A. Extensão solicitada: logs de algoritmo genético, upload
online no W&B e relançamento dos dois datasets em 50%, split 1.

## Interfaces

- `ExperimentLogger(config, directory)` em progressive_logging.py. `config` é
  a configuração completa; seção `wandb` tem enabled=false por padrão, project,
  entity opcional, name/group/tags opcionais, mode online/offline (online nas
  execuções reais), upload_artifacts=true, init_timeout=60.
- `logger.emit(event: str, **payload)` escreve JSONL local com sequência/time
  e, quando habilitado, envia métricas/tabelas W&B. Dados incluem generation
  (índice interno zero-based), individual_index, genome_id/individual_id,
  requested_genome/effective_genome, status/reason, phase/epoch/depth, result
  (dict de métricas/custos), parents, first_changed_prefix e cache.
- `logger.finish(status='finished')` finaliza o run e envia os logs estruturados
  e resumos como artefatos. Não enviar imagens, máscaras ou credenciais.
- Atributos `logger.url`, `logger.run_id`, `logger.enabled`. W&B online não pode
  cair silenciosamente em offline; falha de init propaga. ID e sequência local
  persistem para retomada. SDK importado somente com enabled=true.
- `GeneticSearch(..., logger=None)` aceita logger opcional sem alterar fitness,
  operadores, cache ou defaults. Eventos: search_start/resume, generation_start,
  individual_start/result, best_update, generation_end, offspring, search_end.
  Preservar RNG e conteúdo do checkpoint evolutivo existente.
- Coordinator integra logger ao ProgressiveExperiment: experiment_start,
  individual_cache_hit, initialization, epoch_end, depth_end, individual_end,
  experiment_end/interrupted/failed. Métodos de logging explícitos no pipeline,
  sem depender do overlay de desempenho. Runtime envia recursos/timers ao logger
  ativo sem iniciar outro run W&B.

## Propriedade exclusiva

- A1: ga_search/progressive_logging.py e
  analysis/checks/test_progressive_logging.py (novo). Implementa logger/W&B,
  testes locais e SDK offline reais. Não editar outros arquivos.
- A2: ga_search/progressive_genetics.py e
  analysis/checks/test_progressive_genetics.py. Implementa eventos GA e testes
  do callback; não editar logger/pipeline/config.
- A3: revisão somente leitura de integração e plano de verificação online;
  pode receber teste exclusivo integrado após implementação.
- Coordenador: progressive.py, progressive_runtime.py, configurações YAML,
  launcher tmux, documentação e validação online/relançamento.

As execuções atuais serão interrompidas de forma controlada, seus diretórios
preservados e as buscas relançadas em novos diretórios com W&B online. Não
reutilizar silenciosamente a identidade de uma execução anterior sem logging.

## Integração concluída

Após a primeira versão de A1, o coordenador assumiu exclusivamente o logger e
seu teste para finalizar métricas nomeadas, snapshots exclusivos e validação do
SDK. A1 fez a revisão final somente leitura. A2 entregou eventos GA e testes
integrados; A3 implementou o monitor e permaneceu como agente dedicado à leitura
das execuções reais. Não houve edições concorrentes dos arquivos.

A suíte final executou 47 testes com sucesso. Uma busca sintética real enviou
34 eventos e dois artefatos `nas-logs`, confirmados pela API W&B. Os dois runs
reais de 50% foram relançados com envio online, também confirmado pela API.
O daemon de monitoramento está na sessão `gpt_sol_G_A_monitor`, e o agente A3
registra interpretações em `reports/ga_monitor_insights.md`.
