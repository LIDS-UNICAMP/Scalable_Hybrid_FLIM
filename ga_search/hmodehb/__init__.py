"""H-MODEHB B0-B4 (Agente 4): MO-DEHB hierarquico com integridade de busca.

Namespace novo e isolado: importa as primitivas de ``ga_search.modehb`` (MODEHB, Evaluator, strata,
lemonade) sem modifica-las. Buscas existentes (H-MODEHB, H-LEM, H-SEED, PNAS-CS) nao mudam.

Entrada: ``python -m ga_search.hmodehb.run_hsearch --arm A --dataset remote_sensing``.
Desenho: research/spifil_b0b4/agent4/prep_desenho_modehb_hierarquico.md (secoes 5, 6.1, 7, 10).
"""
