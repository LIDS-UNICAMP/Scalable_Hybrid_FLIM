# GA: backprop até1000épocas, paciência50

Protocolo ativo nos três datasets: uma imagem de treino porclasse, selecionada pelo fluxo de amostragem existente. Nunca ampliar automaticamente os doadores se faltar orçamento; candidatoinviável é registrado. Extração de múltiplos patches da única imagem segue SPiFiLoriginal.

Cada profundidade tem fasesA(novobloco+projeção+decoder,prefixocongelado) eB(refinamentoconjunto), cadauma comteto1000épocas. Earlystopping independente porfase após50épocas consecutivas semmelhorar Jválido da validação. min_delta0; épocas comclusteringinválido contamcomoausênciademelhoria. Contadoresphase_best_J/phase_stale/phase_early_stopped salvos no checkpoint e restaurados na retomada. Checkpointselecionado continua usando menorJ; epochstop não muda oGApatience3gerações.

12configs ×18células=216buscas:3datasets,4resoluções128/160/200/256,3splits,6porcentagens1/5/25/50/75/100. Mesmo holdouttrain-onlyfixo do protocolo multires. Nenhumtestsplit. Configuraçãocientíficamudou; resultadosanteriores ficam emresults/nas_multires e novos emresults/nas_multires_1000. Não misturar caches/checkpoints entreprotocolos.

Execução/retomada: bash results/nas_multires_1000/launch_eggs.sh (oularvae/protozoan). UmGAporfila; Eggs+Protozoan compartilhamGPU2, LarvaeGPU3,16CPUthreads porbusca. W&Bophira-ai/progressive_nas_ga comgrupos1000_pat50. Monitor reports/nas_multires_1000_live.md.

Validação: dois testes de integração comSPiFiL ebackprop reais, métrica controlada paraisolar a parada/resume. Limites reduzidos para testar rapidamente. Ambos passaram em1.297s; evidência reports/nas_1000/checks.log.12configs/216células foram revisados semskips. Essa validação de componente não é resultadoexperimentaldarede.
