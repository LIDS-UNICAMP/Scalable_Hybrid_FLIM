# Adendo (usuário, 17/11): Attention SEM pesos treináveis no GA SPiFiL B0–B4

Objetivo: todos os mecanismos de atenção são operadores matemáticos, com ZERO parâmetros treináveis, sem backprop próprio e sem W_Q/W_K/W_V aprendidas.
Hipótese: os mapas SPiFiL já permitem relações globais por similaridade. O GA (MO-DEHB + LEMONADE + PNAS-CS) descobre quando a atenção ajuda backbones rasos ou profundos.
NÃO substituir nem modificar SPiFiL-0..4: a atenção é complementar, como GeM, whitening e cov pooling.

## Operadores
- **A. Fixed Self-Attention:** Q=K=V=X; Y = softmax(X̂X̂ᵀ/τ)·X.
  - Toda posição interage com todas.
  - Similaridade cosseno e produto interno normalizado.
  - Opção de excluir a diagonal.
  - Fusão residual com α fixo.
  - Nenhuma projeção treinável.
- **B. Fixed Attention Pooling:** query global = GAP ou GeM das ativações; similaridade query×posições → softmax → média ponderada. Comparar com GAP, GeM e moments.
- **C. Fixed Cross-Attention:** uma camada profunda faz as queries (ex. L3); camadas rasas dão keys e values (L1/L2).
  - Alinhamento NÃO treinável: canais compartilhados consistentes, ou PCA/SVD/estatísticas ajustadas só no treino, sem rótulos.
  - Não tratar projeções arbitrárias como espaços semânticos equivalentes.
  - Sempre comparar com a versão sem cross-attention.
- **D. Fixed Multi-Scale Attention:** combinar blocos por similaridade entre features alinhadas e agregação ponderada; sem convoluções novas.
- **E. Fixed Covariance / Second-Order Attention:** pesos de atenção ou agregação a partir de estatísticas de 2ª ordem. Comparar com o CovReadout existente, sem duplicar funcionalidade.

## Genes
- `attention_type`: none, fixed_self, fixed_pool, fixed_cross, fixed_multiscale, fixed_covariance
- `attention_location`: estágio ou bloco
- `attention_source_layers`
- `attention_similarity`: cosine, normalized_dot
- `attention_temperature`: valores > 0
- `attention_tokens`: 16, 49, 196
- `attention_residual_alpha`: 0, 0.25, 0.5, 0.75, 1
- `attention_exclude_self`: true, false
- `attention_normalization`: none, l2, zscore
- `attention_aggregation`: mean, weighted_mean, gem
- `attention_alignment`: none, shared_channels, pca, svd

Os genes só controlam escolhas matemáticas. Precisa de reparos para shapes, memória, dimensões, tokens e limites computacionais.

## Regras
- Zero parâmetros treináveis: sem nn.Linear treinável, sem QKV, sem otimizador próprio.
- PCA/SVD ajustados SÓ no treino e reutilizados em val e teste.
- Distinguir operador sem parâmetros de operador com estatística ajustada no treino.
- Dentro do encoder durante AE/SSL: o operador continua diferenciável em relação às ativações, mas a atenção não tem parâmetros a atualizar.
- Atenção só pós-treino: roda em eval() e no_grad().
- Sem alteração silenciosa dos pesos SPiFiL nem dos checkpoints históricos.
- Embedding ≥ 1280 (R11), teto de 5,5M parâmetros.
- Contabilizar FLOPs, memória e latência mesmo com zero parâmetros.

## Experimentos (primeiro nos MESMOS checkpoints, sem treinar)

| Exp | Operador | TTA |
|---|---|---|
| E0 | GAP/GeM original | não |
| E1 | Fixed Attention Pooling | não |
| E2 | Fixed Self-Attention | não |
| E3 | Fixed Cross-Attention L3↔L2 | não |
| E4 | Fixed Multi-Scale Attention | não |
| E5 | Fixed Covariance Attention | não |
| E6 | melhor de E0–E5 | TTA4 |
| E7 | melhor de E0–E5 | TTA8 |
| E8 | melhor de E0–E5 | multi-scale TTA |

- SVM linear C=100, mesmos splits, protocolos identificados separadamente.
- Nunca ajustar nada com imagens ou rótulos de teste.
- Depois: 2ª ablação com o operador presente durante o AE local e o LeJEPA+SIGReg.

## TTA
- Obrigatório como avaliação complementar dos finalistas: sem TTA, TTA4, TTA8 e multi-scale.
- Transformações válidas por dataset.
- Agregação definida explicitamente (média normalizada).
- PCA, normalização e SVM ajustados só no treino.
- Registrar o custo multiplicativo de inferência.
- TTA fora do fitness arquitetural inicial; a política de TTA só pode ser otimizada depois, e só na validação.

## Papers
Vaswani 2017 (1706.03762), Non-local 2018 (1711.07971), CaiT (2103.17239), CrossViT (2103.14899), SE-Net (1709.01507), Bilinear CNN (ICCV 2015), Shanmugam 2021 TTA (2011.11156). A versão sem pesos é hipótese própria, não reprodução desses métodos.

## Papéis
- Agente 2: valida operadores e alinhamentos sem treino.
- Agente 3: integra aos backbones B0–B4.
- Agente 4: adiciona os genes ao GA (opcionais).
- Agente 5: testes na athena GPU 3, sem tocar na GPU 2 nem na Jaci.
- Nada de busca completa antes das ablações controladas.

## Entregáveis
- Módulo de atenção fixa.
- Testes: zero parâmetros treináveis; equivalência, shapes e estabilidade.
- Comparação com GAP/GeM/CovReadout.
- κ com e sem atenção, e com e sem TTA.
- FLOPs, memória e latência.
- Genes integrados ao GA.
- Relatório de quais mecanismos ajudam backbones rasos × profundos.
- Lista de operadores aprovados para B0–B4.

**Objetivo final:** descobrir se uma CNN SPiFiL rasa alcança representações mais globais com operações fixas, reduzindo a necessidade de profundidade convolucional.
