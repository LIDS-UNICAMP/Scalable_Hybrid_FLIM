# Diagnóstico do LeJEPA nas representações SPiFiL

Pergunta: o LeJEPA destrói a informação discriminativa do SPiFiL, ou só a torna menos acessível à cabeça GAP de 1280?

**Resposta curta:** só a torna menos acessível. Em larvae, o GAP cai em média −0,089 (até −0,206) e a covariância não cai (+0,017). Em RS, as duas leituras sobem pouco. O LeJEPA aumenta o uso de textura e colapsa a cabeça GAP. Detalhes em [conclusions.md](conclusions.md).

| Arquivo | Conteúdo |
|---|---|
| [methodology.md](methodology.md) | checkpoints (P0 = anchor, P1 = ssl_state do mesmo encoder.pt), protocolo e leituras |
| [results_remote_sensing.md](results_remote_sensing.md) | matriz 2×2, por camada e textura (RS) |
| [results_larvae.md](results_larvae.md) | idem para larvae (base de parasitas no lugar de protozoan) |
| [conclusions.md](conclusions.md) | as 7 perguntas, cada resposta classificada |
| `figures/<dataset>/01…08` | gráficos (PNG + SVG + CSV) |
| `data/` | metrics.csv, delta_kappa.csv, layer_metrics.csv, texture_metrics.csv, checkpoint_manifest.json |

Código em `research/lejepa_diag/`:
- biblioteca: extract, readouts, svm_eval, analysis, textura (19 testes);
- drivers: matrix_2x2.py, layer_diag.py, texture_min.py.

Tudo foi medido sem retreinar, em fp32, com a SVM original C=100 e só na validação.
