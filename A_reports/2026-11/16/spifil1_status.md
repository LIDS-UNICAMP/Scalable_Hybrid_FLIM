# SPiFiL-1 (protótipos hierárquicos) — status do Agente 2b (16/11/2026)

O SPiFiL-1 foi implementado como arquivo novo, `ga_search/abstract_init/spifil1.py` (classe `Spifil1Prototypes`, registrada como `"spifil1"`, versão 0.1.0), contra a interface `ga_search/abstract_init/base.py` do Agente 2a (md5 `ead8195416ede86726f5bcece852e155`, copiada sem alteração). O que ele faz:
- lê só o `loader_factory` de treino; no padrão label-free nunca lê `y` nem máscara;
- extrai patches densos do prefixo congelado, com N ≤ `max_patches` e N > D por construção;
- remove a média, se `patch_norm=mean_only`, e aplica ZCA com eps = 0,01·média(λ);
- roda k-means esférico ou euclidiano, com semeadura k-means++ ou aleatória;
- seleciona por `mass_fps`, `fps`, `pca` (PCANet, sem ZCA) ou `all`;
- refina com SoftHebb opcional, chamando `ga_search.modehb.softhebb.fit_softhebb` sem modificá-lo; o bias fica intocado, e há a opção `recenter_bias`;
- dobra o whitening de forma exata, w = c·Z·P e b = −w·μ;
- normaliza a escala da resposta (`unit_response_std`) num holdout de treino, opcional e registrado.

Tipos de bloco cobertos, com as chaves e shapes de `SpifilOpsBlock`:
- conv k1/3/5/7 e pointwise;
- depthwise, com um banco por canal;
- dwsep: depthwise e, em seguida, pointwise sobre a saída linear do depthwise;
- inception, por ramo;
- residual identity;
- residual project, com o skip vindo das top componentes principais (PCA) e preenchimento por protótipos quando C_out > C_in.

A proveniência registra N, D, N/D, nº de protótipos, rank efetivo dos candidatos e do banco, redundância (|cos| > 0,9), diversidade, cobertura (cos > 0,8 no espaço branqueado), massa, eps, bancos degenerados, `labels_used`, `label_source`, `variant` (`spifil1-lf`/`spifil1-sup`) e o hash das imagens, e é serializável em JSON. A versão com rótulos (`label_policy="supervised"` no config **e** no TrainData) usa amostragem estratificada e k-means por classe, e fica marcada. Canais mortos geram filtros zero, nunca NaN.

Testes: `tests/abstract_init/test_spifil1.py`, **63 passed** em cerca de 6 s, em CPU. Cobrem:
- shapes em 13 specs;
- determinismo, com a mesma seed bit a bit e seeds diferentes dando saídas diferentes;
- só treino: split val/test recusado, só o loader de treino é lido, doadores e diag_batch envenenados nunca são tocados, e o orçamento de tempo gera `InfeasibleCandidate`;
- invariância a permutação e a rótulos aleatórios, além de rótulos envenenados nunca lidos;
- equivalência numérica da dobra: na fórmula pura, até 1e-9 em float64; contra a conv em todas as posições, até 1e-4;
- ausência de NaN com canais mortos em todos os tipos de bloco;
- variantes e SoftHebb.

Suíte inteira da cópia: **2368 passed, 7 skipped, 0 falhas** (85 s).

Benchmark em CPU com 4 threads: 256 imagens, D = 576, N ≈ 9·10⁴, C_out = 152, K = 456, em **17,7 s**.

Entrega: `spifil1.patch` (md5 em `spifil1.patch.md5`), só com os 2 arquivos novos. O patch depende de `base.py` e `__init__.py` do Agente 2a, que não estão incluídos. O método **não está comprovado**: falta a ablação em dados reais (RS/Flowers), que depende do gate e da fila.
