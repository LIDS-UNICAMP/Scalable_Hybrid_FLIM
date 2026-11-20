"""lejepa_diag -- biblioteca comum do diagnostico do impacto do LeJEPA (P0 = antes, P1 = depois).

Modulos: extract (embeddings + mapas por bloco), readouts (GAP / COV oficiais + por camada),
svm_eval (core.svm.fit_svm C=100 sem scaler, metricas na VALIDACAO), analysis (rank efetivo,
espectro, CKA, controles), textura (blur / controles / cosseno), run (CLI sobre o manifest).
Nada aqui redefine o embedding nem o readout oficiais: tudo reusa ga_search.modehb.pooling_head.
"""
