# Prompt do usuário, 07/11: kernels por agrupamento de superpixels e SIFT

Regra central: primeiro auditar, depois complementar a implementação, por último pôr na fila só o que ainda não foi testado. O que já foi respondido não se refaz. Sem backprop, sem pré-treino, sem rótulo na seleção de patches, sem máscara na seleção. Preservar resultados anteriores. Não interromper nem duplicar jobs ativos.

## Auditoria (classificar: já implementada e avaliada | parcial | implementada e não avaliada | não implementada)
1. Repositório, experimentos, relatórios, fila.
2. Clustering de superpixels, patches SIFT, K-means, K-medoids, Spectral, grafos.
3. Centroides, medoides ou representantes de cluster como kernels.
4. Descritor SIFT 128-d junto com patches de pixels.
5. Z-score correto.
6. O que foi testado, em quais bases, com qual resultado.

## Método A: superpixels
SLIC ou equivalente, ~400 candidatos por imagem, patch redimensionado ao kernel, Z-score (P-mu)/(sigma+eps), eps configurável, tratar e registrar variância quase nula. Agrupar sem rótulo, sem fixar k: grafo, Leiden/Louvain, Spectral com k automático, Affinity Propagation, hierárquico com corte automático. Parâmetros escolhidos só com treino. Similaridade: euclidiana, cosseno, correlação, textura, estrutura, combinações. Kernel = centroide, re-Z-score, checar variância, registrar tamanho e dispersão do cluster, guardar patches. Comparar com medoide. Adaptar k à arquitetura: um por cluster; top clusters até o orçamento; juntar ou dividir clusters; número variável em experimento separado. Comparação principal com filtros iguais ao aleatório.

## Método B: SIFT
Guardar coordenadas, escala, orientação, resposta, descritor, patch. Janela fixa, proporcional à escala, orientada. Z-score. B1 agrupa descritores 128-d e faz kernel com os patches. B2 agrupa patches. B3 híbrido. Nunca usar os 128 valores como kernel.

## Comparações
C0 aleatório; C1 centro geodésico atual; C2 SIFT filtrado atual; C3 superpixel + grafo + centroide; C4 superpixel + grafo + medoide; C5 SIFT 128-d + cluster + centroide dos patches; C6 idem medoide; C7 patches SIFT + cluster; C8 híbrido; C9 fusão superpixel + SIFT + cluster. K-means como baseline adicional.

## Medir
Candidatos, número de clusters, tamanho, dispersão, redundância dos centroides, filtros finais, desempenho. Separar qualidade de quantidade de filtros: orçamento igual na comparação principal. Todas as bases, mesmos splits, sementes, arquitetura, pooling e classificador. Δκ contra aleatório pareado, consistente entre bases.

## Figuras (Matplotlib)
Superpixels, keypoints, candidatos, grafos, clusters, patches do mesmo cluster, centroides e medoides, kernels, similaridade, ativações, redundância funcional, desempenho.

## Fundo
Sem máscara na seleção. Saliência, contraste, centralidade geodésica, estabilidade multiescala e coerência espacial podem ser características auxiliares. Anotação só para análise.

## Fila
Rodada barata primeiro (C3 a C9). Promissores vão a 4, 5, 6 e 8 camadas, 64 e 256 filtros. Resultados em /exps/moliveira/results, temporários em /exps.

## Relatório final: 10 perguntas
1 cluster melhor que patch individual; 2 grafo melhor que K-means; 3 centroide ou medoide; 4 descritor SIFT melhor que patch cru; 5 SIFT + patch melhora; 6 k automático e diversidade; 7 diversidade reduz redundância funcional; 8 supera aleatório de forma consistente; 9 sensível ao fundo ou objeto; 10 ganho persiste com profundidade.
