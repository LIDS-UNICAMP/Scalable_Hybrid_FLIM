# Curvas parciais do GA

Geradas exclusivamente a partir das medições de validação em `results/nas_multires_1000`. Nenhuma extrapolação, simulação de valores, uso de teste ou média de splits incompletos.

- `kappa_validation_r128.png` / `.svg`: kappa dos vencedores registrados pelo GA (menor J), por split e porcentagem. Diamante aberto indica busca em andamento; lacunas indicam ausência de resultado.
- `kappa_validation_r128_candidate_comparison.png` / `.svg`: acrescenta o maior kappa entre os candidatos viáveis já avaliados, com linha pontilhada. É diagnóstico exploratório na própria validação; não foi usado para alterar a seleção do GA e não é estimativa independente de generalização.
- `validation_curves.csv`: dados das curvas, estado, métricas e IDs.
- `snapshot.json`: resultados e candidatos completos no instante registrado.
- `winners.json`: menores J observados e maiores kappas de candidatos, com genoma efetivo e caminho do modelo. Comparações de menor J entre células não substituem a seleção dentro de cada célula.

Regerar: `MPLCONFIGDIR=/tmp/mpl-progressive python -m analysis.plots.plot_ga_partial_validation` no ambiente scalable_FLIM. O script não altera treinamentos, checkpoints ou critérios.

Achado: Larvae split2/5% tem candidato viável com kappa0,8281 (uma camada,17576parâmetros), enquanto o vencedor pelo menorJ nessa mesma célula tem kappa0,0640. A SVM do candidato atingiu10000iterações. Eggs split1/25%: candidato0,3559 contra vencedor0,1775; também atingiu o limite. Protozoan split1/1%: candidato0,4109 contra vencedor0,2681; SVM do candidato convergiu. Todos são valores de validação, não de teste.
