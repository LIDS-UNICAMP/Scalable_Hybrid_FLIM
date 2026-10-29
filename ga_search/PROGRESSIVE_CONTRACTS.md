# Contratos NAS progressivo (branch gpt_sol_G_A)

## Reconhecimento e barreira A → B

Python/PyTorch; SPiFiL editable em `/dados/home/moliveira/SPiFiL` somente leitura.
Nenhum AGENTS.md encontrado no repositório, ancestrais inspecionados ou biblioteca.
A1: crescimento; A2: modelo/treinamento; A3: busca/avaliação; A4: execução/testes.
Reutilizar `flim.spifil` (Trunk, Cropped, build_layer0, GrowOneLayer,
Learner, ArchSpec e LayerSpec), `core.blocks.init.INIT_FNS`, dados/splits
de `ga_search.data` e dispatch opt-in no runner `growth`. Experimentos legados
mantêm critérios e avaliação existentes.

## Interfaces

- Genoma: dict `blocks`, lista de dicts `op=conv`, `kernel=3|5`, `channels>0`,
  `pool_stride=1|2`, `bn=bool`, `dropout` em [0,1),
  `activation=relu|relu6|silu`, `connection=sequential`. Profundidade 1..10.
  Inicialmente sem separáveis, 1×1 no encoder, residual ou concat: suporte não
  comprovado. Validação rejeita, nunca troca filtros por aleatórios.
- `flim.progressive.fit_new_block(prefix, samples, block_spec, config,
  prefix_id=...) -> (nn.Module, provenance)`; `config` contém image_size,
  device, seed, n_superpixels, imagenet_norm, impurities, superpixels,
  spifil_in_image, spifil_in_feature, budget_check. Prefixo eval, congelado,
  no_grad; somente uma LayerSpec. Resultado inclui actual_channels,
  requested_channels, sample_ids, prefix_id, seed, orçamento, tempo e origem.
  Canais reais reparam explicitamente o genoma. Falhas levantam
  `InfeasibleCandidate`, com motivo serializável pelo coordenador.
- `methods.autoencoder.progressive.EvolutionaryCNN(config)` começa vazio;
  `add_block(block, effective_spec)` mantém prefixo e cria P_i/D_i próprios.
  `encoder` retorna mapa; `project(x, depth=None)` retorna Z [B,1280,H,W];
  `embed` retorna [B,1280]; `forward` reconstrói [B,C,200,200]. Auxiliares
  inicializados pela política central; não visitar encoder com initializer.
- `ProgressiveTrainer(model, config)` mantém um Adam por indivíduo;
  `sync_optimizer` adiciona somente parâmetros novos;
  `set_phase('A'|'B')`, `train_epoch(loader)`, `save_checkpoint(path, **metadata)`
  e `load_checkpoint(path)` preservam estado e RNG. A congela prefixo e BN;
  B libera encoder e reduz LR. `config` inclui lr, refine_lr, device.
- `ga_search.progressive_metrics.evaluate_embeddings(embeddings, config)`
  retorna dict status/reason, dbi/dbi_norm, hoyer_mean/hoyer_std,
  coverage/n_clusters/dbcv/J. HDBSCAN, DBI sem ruído; zero/não finitos explícitos;
  menor J melhor; lambda .004/.04; coverage .8; penalidade fixa configurável.
  `extract_embeddings(model, loader, ...)` preserva IDs/checkpoint em NPZ.
- `ga_search.progressive_genetics`: validação de genoma, geração, crossover,
  mutação, cache SHA256 e busca escalar. Avaliador injetado `(genome)->dict`
  contendo J. Busca salva população/histórico/RNG a cada avaliação/geração.
  Sem herança entre indivíduos; alteração de prefixo invalida dependentes.
  Chave exige genoma efetivo, dados/splits/conteúdo, configuração, seeds,
  sample_ids SPiFiL, versões e prefix_id. Não usar cache legado.
- Checkpoint progressivo guarda modelo inteiro, optimizer, RNGs, configuração,
  specs efetivos, profundidade/fase/época e metadados. IDs de prefixo/checkpoint
  SHA256 do arquivo. Seleção de checkpoint por J apenas neste modo.

## Propriedade exclusiva no grupo B

- B1/A1: `flim/progressive.py`, `analysis/checks/test_progressive_spifil.py`.
- B2/A2: `methods/autoencoder/progressive.py`,
  `analysis/checks/test_progressive_model.py`.
- B3/A3: `ga_search/progressive_metrics.py`,
  `analysis/checks/test_progressive_metrics.py`.
- B4/coordenador (sequencial com integração):
  `ga_search/progressive_genetics.py`, seu teste; integração/configuração,
  schema, dispatch growth/run_ga, documentação, smoke e revisão final.

Há quatro slots totais; A4 é sequencial após A3. B4 fica com coordenador
enquanto B1/B2/B3 trabalham em paralelo. Nenhum arquivo recebe edição concorrente.

## Integração e revisão concluídas

- A2 assumiu exclusivamente `test_progressive_genetics.py` após B2 e implementou
  `progressive_cost.py`; A1 implementou `progressive_data.py` após B1.
- Coordenador integrou configuração, genética, runner, CLI, logging e retomada.
- D2/A2 revisou somente leitura. Foram corrigidos: cache com checkpoint ausente
  ou hash alterado; histórico e proveniência de cache; política trunc_normal sem
  suporte de inicialização Conv2d. Auxiliares aceitam He/Xavier.
- D1/A3 implementou testes integrados com SPiFiL real, interrupção/retomada,
  duas profundidades, geração completa e cache. A1 corrigiu somente o check
  legado para consumir APIs atuais, sem mudar comportamento de produção.
- Dados, filtros e configurações são identificados por conteúdo. O namespace de
  cache inclui também fontes da biblioteca SPiFiL editable e componentes locais.
- `complementary_probe` usa `core.svm.fit_svm` e `core.metrics.compute_metrics`.
  Sua seleção continua independente de J; o protocolo legado não foi alterado.
