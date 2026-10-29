# SPiFiL: redução de memória por blocos

Larvae200/split1/75% produziu44911patches com100doadores. ScorerFisher original mantém matrizes NxN de distâncias e máscaras; picoSPiFiL observado~28.3GiB e processoCUDAreservado38.14GiB. Não é memória dos pesos finais.

Implementação local em ga_search/progressive_spifil_memory.py, usada pelo overlay progressive_runtime. Biblioteca externa intacta. Mesma covariância, doadores, patches, whitening completo e fórmula Fisher; distâncias e máscaras processadas em blocos com chunk_size do metric4096. Liberação dos buffers em cada bloco; gc e CUDAempty_cache somente após ajuste, sem remover pesos vivos. Telemetria passa a distinguir cuda_allocated_bytes e cuda_reserved_bytes.

Verificação executada emCPUeCUDA:24casos, scores allclose atol=rtol1e-6 e rankings iguais emtodososcasos. CPUbitwiseigual; GPU mostroudiferençamáxima1.788e-7; nãohágarantia geralbitwise nem garantia sobreempatesquasenúmericos. Não afirmar que equivalência numérica prova superioridade ou igualdade de todos os filtros em todos os datasets.

Benchmark sintético CUDA comN44911,D27: picoalocado2.418GiB, reserva3.834GiB,1.164s. Usa mesmaquantidadedepatches do casoreal, mas features aleatórias; não é ainda medição do novo pico do pipeline completo. Arquivos checks.jsonl ebenchmark.json guardam evidências.

Reprodução: PYTHONPATH=/tmp/flim_nas_deps:$PWD python -m analysis.checks.check_spifil_memory (ambiente scalable_FLIM). CUDA requer GPU visível.

Retomada operacional Larvae: driverRay pausado enquanto subprocessooriginal recebeSIGINT egravaestadointerrompido. Processo de retomada usa mesmo arm_config e --resume; driver é liberado depois do término dessa busca, evitando duasbuscasLarvae simultâneas. Mantém somente treino/validação.

Medição real após correção: Larvae200/split1/100% com44953patches registrou picoSPiFiL2.4705GiB (antes~28.3GiB com44911patches). ProcessoCUDA6976MiB(~6.81GiB) na amostrahost; valores instantâneos variam comtreino. Redução não estimada apenas do benchmark.

A primeira tentativa de retomada75% encontrou bugpreexistente: mapa classes YAML usa chavesint, enquanto experiment.json guarda strings; comparaçãoestrita rejeitava mesma configuração. Overlay agora normaliza viaJSONantesda comparação. Config,versions eidentityhash com dadosanteriores foram verificadosiguais após normalização. Checkpoints75%preservados. Driver retomou100% automaticamente após saídaSIGINT; foi então pausado e script emtmux aguarda100%concluir, retoma75% e libera driver. Assim não háduasbuscasLarvae simultâneas. Os outros datasets seguem.
