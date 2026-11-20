# Auditoria: kernels por agrupamento (07/11, 20:25 UTC)

Só validação. Δκ em L3 `cat` contra o melhor aleatório. O pedido está em `/exps/moliveira/reports/prompt_clustering_resumo.md`.

## Estado de cada comparação

| ID | Estado | Braço | Falta |
|---|---|---|---|
| C0 aleatório | já avaliado | `random_z`, `xavier` | nada |
| C1 centro geodésico | já avaliado | `crus@geodesic` | nada |
| C2 SIFT filtrado | já avaliado | `crus@sift_nms` | nada |
| C3 superpixel + grafo + centroide | não implementado | | código |
| C4 superpixel + grafo + medoide | não implementado | | código |
| C5 SIFT 128-d + centroide | parcial | | 2 linhas |
| C6 SIFT 128-d + medoide | avaliado em 2 bases | `bovw_w@sift` | outras bases |
| C7 patches SIFT + cluster | implementado, não avaliado | `km_w@sift` | YAML |
| C8 híbrido | não implementado | | código |
| C9 fusão + cluster | parcial, não avaliado | `km_w_m@uniao` | YAML; grafo depende de C3 |
| K-means | avaliado em 2 bases | `km_w`, `km_w_m` | outras bases |

Spectral, Leiden, Louvain, grafo kNN, janela orientada e Z-score do próprio patch não existem. Leiden pediria instalação; Louvain sai de `networkx`.

## As 10 perguntas

1. Cluster melhor que patch individual: parcial, 2 bases. `km_w` empata com patch sorteado (+0,004 [−0,012; +0,021]) e perde de `fps` (−0,028 [−0,049; −0,007] em 64; −0,044 [−0,072; −0,016] em 256).
2. Grafo melhor que K-means: sem dado.
3. Centroide ou medoide: parcial, empate (+0,004 [−0,014; +0,023]). O medoide atual é aproximado.
4. Descritor melhor que patch: parcial, não. `fpsd@sift` − `fps@sift` = −0,031 [−0,049; −0,013] em 256.
5. SIFT + patch: sem dado para o híbrido real. União de pontos não ganha.
6. k automático e diversidade: parcial, não. `km_sil` fica no piso da grade, k = 8.
7. Diversidade e redundância funcional: parcial. `fps` 0,876, K-means 0,930, uniforme 0,945 (correlação com o vizinho, L1).
8. Consistência contra aleatório: agrupamento não. `km_w` ganha em Protozoan_subgroup e perde em flowers.
9. Fundo: parcial. Só fundo piora −0,162 [−0,237; −0,086]; `fpsw@mapa_peso` − `fps` = +0,043 [+0,019; +0,067] em 64, +0,015 [−0,013; +0,043] em 256.
10. Profundidade: sem dado para agrupamento.

## Ressalvas

- O agrupamento só rodou em flowers e Protozoan_subgroup, com 3 splits e 2 sorteios.
- Em flowers, com 64 filtros, `km_w` tem 53% de patches planos, contra 2% em `fps`: o cluster puxa para o que é comum.
- A comparação depende dos controles: `fps@geodesic` em Larvae_subgroup cai de +0,250 para +0,127 com todos os aleatórios. O "passa 3 de 5" de `multiescala.md` depende disso.
- `spifil` escolhe por classe e usa menos imagens: não entra como método sem rótulo.
- Nenhuma fila atual mede C3, C4, C8 ou cluster em profundidade.
