# Relatório de integridade dos checkpoints — migração athena GPU 3 → Jaci (staging)

Gerado pelo Agente 5 (athena), 17/11/2026. Método: rsync -a sobre SSH, sem --delete, para artifacts/_staging_gpu3/; depois do rsync, conferência nos dois lados de: (1) número de jobs no search_state.json; (2) md5 do search_state.json; (3) md5 de TODOS os candidates/*/f*/encoder.pt. Original intacto no athena.

| kind | base | seed | jobs athena=jaci | md5 search_state | encoder.pt conferidos | todos iguais | marcador |
|---|---|---|---|---|---|---|---|
| modehb_hybrid_seed43 | flowers | seed43 | 2=2 | 0efd16054567… | 2 | ✅ | modehb_hybrid_seed43_flowers.ok |
| modehb_hybrid_seed43 | larvae | seed43 | 0=0 | sem_search_s… | 0 | ✅ | modehb_hybrid_seed43_larvae.ok |
| modehb_hybrid | coconut_trees | seed42 | 39=39 | def7dccada05… | 35 | ✅ | modehb_hybrid_coconut_trees.ok |
| modehb_lemonade | flowers | seed42 | 36=36 | 68942de115fc… | 32 | ✅ | modehb_lemonade_flowers.ok |
| modehb_lemonade | coconut_trees | seed42 | 24=24 | 0bddf9f736b4… | 21 | ✅ | modehb_lemonade_coconut_trees.ok |
| modehb_hybrid | eggs | seed42 | 55=55 | 00be53667baa… | 51 | ✅ | modehb_hybrid_eggs.ok |
| modehb_hybrid | eggs_raw | seed42 | 51=51 | 3ee5651f036f… | 47 | ✅ | modehb_hybrid_eggs_raw.ok |

Notas:
- Interrupção sempre ENTRE jobs (restart_between_jobs: espera o job em andamento ser salvo no search_state e só então manda SIGTERM). Nenhum job ficou pela metade; o resume na Jaci continua do próximo job (ReplayMismatch protege genoma/fidelidade/papel).
- Nº de encoder.pt < nº de jobs é esperado: jobs que falharam (ex.: hybrid coconut job 31, params > cap) ou sem checkpoint promovível não têm encoder.pt.
- seed43 larvae: 0 jobs e sem search_state.json (foi pausado ~3 min após iniciar, antes do 1º job); só config_resolved.yaml e o log foram copiados (2 arquivos, contagem igual). Resume = começar do zero, sem perda.
- seeded (5 buscas: modehb_seeded coconut/eggs/eggs_raw e modehb_lemonade_seeded coconut/flowers): nunca iniciaram no athena; não há estado a migrar. Entram na fila da Jaci como jobs novos (comandos originais em /exps/moliveira/queues/gpu3_migracao/waiters/).
- Listas md5 completas: athena /exps/moliveira/queues/gpu3_migracao/md5/<kind>_<base>.{athena,jaci}.
- Trava contra dois controladores: no athena, os laços e waiters foram encerrados (sessões tmux removidas, pids conferidos mortos); o monitor de teste do athena só LÊ os states originais (não escreve no search_state). Na Jaci, publicar o staging no destino final só depois de validar o resume.
