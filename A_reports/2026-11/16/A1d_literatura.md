# A1d — Literatura do Agente 1 (diagnóstico SPiFiL B0–B4)

> Autor: Agente 1d. Data: 16/11/2026. Fontes lidas no texto integral (PDF extraído) para os 5 papers obrigatórios; literatura adicional lida só pelo abstract (indicado em cada item).
> Convenção: **[paper]** = afirmação do paper, com número; **[nosso]** = fato do inventário `ga_search/modehb/reports/all_results_today.md` (linha citada); **[predição]** = hipótese derivada, ainda NÃO testada. Nenhuma predição aqui é resultado nosso.
> Lembrete de régua: toda predição sobre κ refere-se à régua **ORIG** (SVM `core.svm.fit_svm`, C=100, sem scaler, embedding 1280) salvo quando escrito COV.

## Mapa rápido: paper → hipótese do Agente 1

| Hipótese (MISSION §Agente 1) | Coates 2011 | PCANet 2015 | Belilovsky 2019 | EfficientNet 2019 | MO-DEHB 2023 | Adicional |
|---|---|---|---|---|---|---|
| Profundidade ótima por domínio | 1 camada basta em CIFAR/NORB | 2 estágios bastam; 3 não ajuda | separabilidade linear sobe com a profundidade, mas satura mais cedo com objetivo local raso | profundidade sozinha satura (~80% top-1) | — | Bello 2021: profundidade só quando há overfitting |
| Patches simples insuficientes nas camadas profundas | campo receptivo grande → espaço grande demais para cobrir | estágio 2 = PCA sobre os mapas, funciona só com imagens alinhadas | k=1 (objetivo raso) satura; k=2/3 vai mais longe | — | — | — |
| Protótipos (k-means/whitening) | k-means + whitening + triangle = melhor método | PCA (protótipo linear) ≈ estado da arte em faces/texturas | — | — | — | — |
| Aprendizagem camada a camada | — | camada a camada sem backprop | camada a camada escala até ImageNet | — | — | — |
| Compound scaling | — | — | — | α=1,2 β=1,1 γ=1,15 | — | TinyNet: ao encolher, resolução/profundidade > largura |
| Saturação antes de 5,5M | nº de features ajuda até 1600–4000 | dim. explode com estágios | — | B0 inteiro = 5,3M | — | — |
| Embedding / 2ª ordem | pooling por quadrantes (4K dims) | histogramas por bloco + SPP → PCA 1280 | — | — | — | B-CNN, MPN-COV, iSQRT-COV, Compact Bilinear |
| Busca κ×FLOPs | — | — | — | busca ACC×(FLOPs/T)^w | NDS + crowding/EpsNet em DEHB | — |

---

## 1. Coates, Ng & Lee (AISTATS 2011) — An Analysis of Single-Layer Networks in Unsupervised Feature Learning

Fonte: https://proceedings.mlr.press/v15/coates11a.html (PDF lido).

### Princípio
Uma única camada de features não supervisionadas: extrai patches w×w densos, normaliza (brilho/contraste), aplica opcionalmente whitening (ZCA), aprende K bases/centróides com um entre 4 algoritmos (sparse AE, sparse RBM, k-means, GMM), mapeia cada patch para K features, faz sum-pooling em 4 quadrantes (4K dims) e classifica com SVM linear. A tese é que **os hiperparâmetros da rede (whitening, K, stride, campo receptivo) pesam tanto quanto o algoritmo de aprendizagem**.

### Resultados-chave (números)
- CIFAR-10 (teste): k-means (triangle) 1600 features **77,9%**; com 4000 features **79,6%** (estado da arte na época, acima de Conv-DBN de 2 camadas 78,9%). Sparse AE 73,4%; sparse RBM 72,4%; k-means hard 68,6%; pixels crus 37,3%.
- NORB: k-means triangle 4000 features **97,21%**; sparse AE 96,9%; k-means hard 96,9%.
- STL-10 (96→32 px): k-means triangle 1600 → **51,5% ±1,73** vs pixels 31,8%.
- Ativação "triangle" `f_k(x)=max(0, μ(z)−z_k)`, z_k = ‖x−c_k‖: zera ~metade das features (competição simples). Ganho triangle × hard: 77,9 vs 68,6 (+9,3 pp).
- Whitening: ambíguo para AE/RBM; **crucial para k-means e GMM** (filtros nítidos e localizados só com whitening).
- K (100→1600): todos os algoritmos melhoram com mais features.
- Stride: s=2 já custa ≥3 pp; s=4 custa ≥5 pp (extração densa importa).
- Campo receptivo: melhor 6 px (testados 6, 8, 12); campos grandes caem porque o espaço de patches fica grande demais para ser coberto por poucos centróides e aumenta a redundância (cópias deslocadas de bordas).

### O que se aplica ao nosso caso
- O SPiFiL é a mesma família: kernels = patches/protótipos escolhidos dos dados (FPS/superpixels/Fisher), sem gradiente. Coates diz que **a seleção/codificação do patch é menos importante que whitening, K e densidade de extração**. Isso aponta para auditar (A1-B) whitening/normalização do patch antes do FPS e o nº efetivo de candidatos (rank), e não só o critério de seleção.
- "Triangle" é uma ativação competitiva barata; equivale a uma distância ao protótipo com limiar adaptativo — candidata natural para a camada stem SPiFiL (não substitui ReLU sem teste).
- K grande ajuda porque o SVM linear se beneficia de muitas projeções não lineares → coerente com a cabeça 1280 + SVM C=100 (ORIG).
- Stride 1 / extração densa: em 224 px e orçamento de FLOPs, downsampling agressivo no stem custa κ; trade-off direto com o objetivo FLOPs do MO-DEHB.

### Predições para as nossas hipóteses
- **Profundidade ótima:** 1 camada bem configurada já é forte em dados alinhados/pequenos → predição: em parasitas (fundo preto, objeto centrado) o ganho marginal do 3º/4º bloco é pequeno; o benefício marginal por bloco (quantitativa 3) deve cair rápido.
- **Patches simples em camadas profundas:** campo receptivo efetivo grande = espaço de patches grande demais para k-means/FPS cobrir com K pequeno → predição: kernels por patch em blocos profundos ficam redundantes/de baixo rank, a não ser que K cresça muito ou se use whitening.
- **Protótipos (k-means/whitening):** forte suporte para SPiFiL-1 (k-means++ com whitening em vez de FPS puro) — hipótese testável: whitening ZCA + k-means ≥ FPS no mesmo K.
- **Camada a camada:** não testa.
- **Saturação:** κ cresce com K até 1600–4000 sem saturar em CIFAR; em datasets pequenos o limite deve vir do nº de amostras do SVM, não de K.

### Limitações
- Só uma camada; não diz nada sobre composição hierárquica.
- Imagens 32×32 (CIFAR/NORB/STL reduzido); campos receptivos 6–12 px. Nada em 224 px.
- Dimensão final 4K–16K (pooling em quadrantes), bem acima de 1280; comparação com a nossa cabeça não é direta.
- Hiperparâmetros escolhidos por validação cruzada no próprio CIFAR.

---

## 2. Chan, Jia, Gao, Lu, Zeng & Ma (IEEE TIP 2015) — PCANet: A Simple Deep Learning Baseline for Image Classification?

Fonte: https://arxiv.org/abs/1404.3606 (PDF lido).

### Princípio
Rede convolucional sem otimização numérica: estágio 1 = PCA dos patches k1×k2 (média do patch removida) → L1 filtros; estágio 2 = PCA dos patches dos L1 mapas → L2 filtros; saída = binarização (hashing) dos L2 mapas, conversão para inteiro, histogramas por bloco (opcionalmente SPP) e SVM linear. Nenhuma não linearidade entre estágios. Variantes: RandNet (filtros gaussianos aleatórios) e LDANet (filtros por LDA, supervisionados).

### Resultados-chave (números)
- Padrão: k1=k2=5, L1=L2=8. "Dois estágios são em geral suficientes; **arquitetura mais profunda não leva necessariamente a melhora**."
- Sem remoção da média do patch o desempenho "degrada significativamente".
- ReLU/|·| depois do estágio 1: **nenhuma melhora** (a invariância vem do hashing+histograma).
- 2 estágios (L1k1² + L2k2² variáveis) superam 1 estágio equivalente com L1L2 filtros de (k1+k2−1)² na maioria dos casos — fatoração de baixo rank reduz overfitting.
- Extended Yale B: **99,58%**; 60% de oclusão: PCANet-2 86,49% vs PCANet-1 84,60%; 80% de oclusão: PCANet-1 54,38 > PCANet-2 51,73 (o 2º estágio não ajuda sempre).
- FERET: média 97,25%, Dup-I 95,84%, Dup-II 94,02%. LFW (não supervisionado): 86,28%.
- MNIST: erro PCANet-2 0,66%, LDANet-2 0,62%, RandNet-2 0,63%, PCANet-1 0,94% (PCANet-1 com k1=13: 0,62%).
- CIFAR-10 (L1=40, L2=8, SPP 4×4/2×2/1×1, **PCA para 1280 dims**, SVM linear): **77,14%**; multi-escala (k=3 e k=5 concatenados) **78,67%** (+1,5 pp). Abaixo de k-means de Coates (79,6%) e ~11 pp abaixo do estado da arte supervisionado.
- RandNet < PCANet consistentemente; LDANet (supervisionado) não aproveita bem os rótulos.

### O que se aplica ao nosso caso
- PCANet é o parente mais próximo do SPiFiL "kernels derivados dos dados, sem gradiente, + SVM linear". O resultado CIFAR com **embedding PCA-1280 + SVM linear** é quase o nosso protocolo ORIG — referência de piso para "encoder sem backprop".
- Remoção da média do patch é essencial → checar no SPiFiL se patches são centrados antes do FPS/Fisher (A1-B).
- O ganho de 1,5 pp por multi-escala (k=3+5) sustenta testar concat multi-filtro/multiescala no embedding (Agente 3).
- "PCA = AE linear mais simples" → o AE local por bloco é uma generalização não linear de PCANet; a hipótese "AE local preserva/melhora" deve bater pelo menos o PCA do mesmo bloco.

### Predições para as nossas hipóteses
- **Profundidade ótima:** 2 estágios em dados alinhados (faces, dígitos, texturas) — predição: parasitas (alinhados, fundo preto) saturam em 2–3 blocos SPiFiL; Flowers/RS (variabilidade de pose/escala) exigem mais e é onde PCANet falha.
- **Patches simples em camadas profundas:** PCANet falha onde há variabilidade (CIFAR, sugere-se insuficiente para Pascal/ImageNet) → predição: SPiFiL puro em blocos profundos fica abaixo do backprop em Flowers/RS/AID; o ganho de SPiFiL-1..3 deve aparecer nesses domínios, não em parasitas.
- **Protótipos:** PCA (variância máxima) bate filtros aleatórios e LDA → diversidade/ortogonalidade dos kernels importa; testar rank efetivo dos kernels SPiFiL vs PCA dos mesmos patches.
- **Camada a camada:** cada estágio é ajustado sozinho, sem feedback; funciona até 2 estágios.
- **Saturação:** a dimensão do histograma cresce exponencialmente com estágios (L1·L2·2^L2·B) — é o gargalo declarado para ir além de 2 estágios.

### Limitações
- Só testado com imagens alinhadas e pequenas; os autores dizem que precisa de alinhamento de pose/escala para Pascal/ImageNet.
- Nenhuma não linearidade antes da saída; não há protótipos invariantes.
- Hiperparâmetros fixos (L=8 inspirado em Gabor); sem estudo sistemático de 3+ estágios.
- Feature final de alta dimensão (antes do PCA) e cara para SVM em dataset grande.

---

## 3. Belilovsky, Eickenberg & Oyallon (ICML 2019) — Greedy Layerwise Learning Can Scale to ImageNet

Fonte: https://proceedings.mlr.press/v97/belilovsky19a.html (PDF lido).

### Princípio
Treinar uma CNN bloco a bloco, cada novo bloco (conv 3×3 + ReLU) resolvendo um problema auxiliar **supervisionado** raso: `x_{j+1}=ρ W_j P_j x_j`, `z_{j+1}=C_j x_{j+1}`, com C_j = média espacial + linear (k=1) ou (k−1) convs de largura constante + média + linear (k=2,3). Blocos anteriores congelados, sem feedback entre camadas. Downsampling P = **invertível** (space-to-depth 2×2), escolhido em vez de max/avg pool por não perder informação.

### Resultados-chave (números)
- ImageNet, k=1 (8 camadas): **58,1% top-1 / 79,7% top-5** (ensemble das saídas por camada 59,3 / 80,8) > AlexNet 56,5 / 79,1. Top-5 cresce quase linearmente de ~23% para ~80% nas 5 primeiras camadas. Treino 87% → sugere overfit.
- ImageNet, k=2: 65,7 / 86,3; **k=3: 69,7 / 88,7** (ens. 71,6 / 89,8) ≈ VGG-13 (69,9 / 89,3).
- VGG-11 treinada k=3 camada a camada: **67,6% top-1** vs ponta a ponta **67,9%** (empate dentro de 0,3 pp).
- CIFAR-10: k=1 88,3% (≈ AlexNet 89%); k=2 90,4%; k=3 91,7% (ens. 92,8) vs VGG 92,5%.
- Separabilidade linear (CIFAR-10, 5 seeds, desvio médio 0,58 pp): **cresce monotonicamente com a profundidade** para k=1 e também para k>1 (sem ser otimizada). Com k=1 cresce mais rápido no início mas **satura num patamar mais baixo**; k maior cresce mais devagar e chega mais alto.
- Compressão integrada: poda de 128→64 filtros a cada camada nova → 87,6% vs 87,5% ponta a ponta (mesma arquitetura final).

### O que se aplica ao nosso caso
- Sustenta diretamente o pipeline "SPiFiL por bloco + AE local por bloco + congelar/proteger prefixo" (aprendizagem local, protect freeze/lowlr). Diferença crítica: o objetivo local de Belilovsky é **supervisionado**; o nosso AE local é reconstrução (label-free). O paper não garante que reconstrução produza separabilidade.
- Sondas lineares por bloco = protocolo pronto para a quantitativa 3 (benefício marginal por bloco) e para "por que mapas intermediários com covariância vão bem" (D) — mas como diagnóstico, nunca como seleção com teste.
- Downsampling invertível é uma alternativa ao max-pool do FLIM para preservar informação no stem em 224 px (custo: 4× canais).
- O ensemble das saídas por camada (+1,2 pp top-1) sustenta fusão intermediária no embedding final.

### Predições para as nossas hipóteses
- **Profundidade ótima:** com objetivo local raso a separabilidade satura cedo → predição: com AE local (ainda mais distante do κ do que um classificador linear), o κ ORIG por bloco satura em poucos blocos; acrescentar blocos só ajuda se o objetivo local "olhar mais longe" (análogo a k=2/3: decoder mais profundo, objetivo preditivo/LeJEPA local).
- **Patches simples em camadas profundas:** não testa patches, mas mostra que objetivos locais mais ricos (k>1) são necessários para camadas profundas → coerente com SPiFiL-2/3.
- **Aprendizagem camada a camada:** pode igualar ponta a ponta (VGG-11: −0,3 pp) → predição: protect=freeze + AE local não deve perder mais que ~1 pp κ para joint se o objetivo local for adequado; perda maior aponta para objetivo local fraco.
- **Fusão intermediária:** saídas de camadas intermediárias somam (+1,2 pp) → apoia multiescala/concat no 1280.
- **Herança/crescimento:** cada bloco novo sobre prefixo congelado = crescimento progressivo B0→B4 sem reinicializar o prefixo.

### Limitações
- Objetivo local supervisionado (usa rótulos em toda camada) — não é o nosso cenário label-free.
- Sem skip connections, sem BN no modelo base; arquiteturas VGG-like grandes (dezenas de M de params), ImageNet/CIFAR inteiros (não few-shot).
- k=1 tem 87% de treino no ImageNet → regularização não explorada; custo de memória menor, mas tempo total não é menor que ponta a ponta.
- Não explica teoricamente o crescimento rápido da separabilidade.

---

## 4. Tan & Le (ICML 2019) — EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Fonte: https://proceedings.mlr.press/v97/tan19a.html (PDF lido).

### Princípio
Escalar profundidade d, largura w e resolução r juntas com um coeficiente φ: `d=α^φ, w=β^φ, r=γ^φ`, sujeito a `α·β²·γ²≈2` (FLOPs ∝ d·w²·r², então cada φ dobra os FLOPs). α,β,γ achados por grid pequeno em φ=1 sobre o baseline e depois fixados. O baseline (B0) vem de uma busca multiobjetivo com recompensa `ACC×(FLOPs/T)^w`, w=−0,07 (otimiza FLOPs, não latência).

### Resultados-chave (números)
- Observação 1: escalar uma só dimensão melhora, mas o ganho **satura em ~80% top-1**. Observação 2: equilibrar as três é necessário.
- Coeficientes do B0: **α=1,2, β=1,1, γ=1,15**.
- **EfficientNet-B0: 5,3M params, 0,39B FLOPs, 76,3% top-1** (224 px) vs ResNet-50 26M / 4,1B / 76,0%.
- B1 7,8M / 0,70B / 78,8%; B2 9,2M / 1,0B / 79,8%; B3 12M / 1,8B / 81,1%; B4 19M / 4,2B / 82,6%; B7 66M / 37B / 84,4% (8,4× menor e 6,1× mais rápido que GPipe 557M).
- Mesmo baseline, mesma meta de FLOPs: compound > escala de uma dimensão em **até +2,5 pp**.
- MobileNetV2 0,3B 72,0% → depth ×4 76,8% (1,2B); width ×2 76,4% (1,1B); res ×2 74,8% (1,2B); **compound 77,4% (1,3B)**. ResNet-50: depth 78,1, width 77,7, res 77,5, **compound 78,8**.
- Transferência (fine-tuning de ImageNet): **Flowers 98,8%** (B7), Flowers 98,5% com B5 (28M); CIFAR-10 98,1% com B0 (4M); CIFAR-100 88,1% com B0.
- Dropout cresce de 0,2 (B0) a 0,5 (B7).

### O que se aplica ao nosso caso
- **Nossa família B0–B4 inteira (≤5,5M) cabe dentro de um EfficientNet-B0 (5,3M).** Estamos no regime "abaixo do B0", em que o paper não mediu; os coeficientes α,β,γ do paper **não devem ser herdados** como verdade.
- Resolução fixa em 224 (MISSION) → só α e β variam; a restrição vira `α·β²≈2` por passo de φ. O repositório já usa α∈{1,0…1,4}, β∈{1,05…1,3} com resolução 224 ([nosso] all_results_today.md:203) — coerente, mas sem γ a lei de saturação da Obs. 1 prediz saturação mais cedo.
- O objetivo do B0 (ACC×FLOPs^w) é a versão escalarizada do nosso κ×FLOPs; MO-DEHB faz a versão Pareto.
- Escalonamento a partir de **um bom baseline** é o que conta ("ter um bom baseline é crítico"): justifica escolher a semente Bk por validação antes de crescer.

### Predições para as nossas hipóteses
- **Compound scaling:** crescimento B0→B4 por width+depth juntos deve vencer só-width ou só-depth no mesmo FLOPs (até ~2,5 pp no ImageNet; no nosso regime o tamanho do efeito é desconhecido).
- **Saturação:** escala de uma dimensão satura rápido → predição: com resolução fixa, κ satura antes de 5,5M em datasets pequenos/alinhados; resultados não monotônicos (pequenos dominando grandes, ex.: [nosso] coconut c00001 f200 com 22k params κ 0,813 e ham10000 c00001 f200 com 20k κ 0,534, all_results_today.md:31) são compatíveis com saturação + overfit, não contradizem o paper.
- **Profundidade ótima:** "redes largas e rasas têm dificuldade em capturar features de alto nível" → em Flowers/RS a profundidade deve importar mais que largura.
- **Patches simples / protótipos / camada a camada:** não trata.

### Limitações
- Treino supervisionado ponta a ponta em ImageNet (1,28M imagens); nada sobre encoders sem backprop nem datasets pequenos sem fine-tuning.
- α,β,γ achados para um baseline específico (MBConv+SE) e para escalar **para cima**; ao encolher a regra não vale (TinyNet, abaixo).
- FLOPs ≠ latência/memória (ativações crescem ~O(s); Dollár 2021, abaixo).
- Grid de α,β,γ só em φ=1; não reotimizado em cada φ.

---

## 5. Awad, Sharma, Müller, Thomas & Hutter (IJCAI 2023) — MO-DEHB: Evolutionary-based Hyperband for Multi-Objective Optimization

Fonte: https://arxiv.org/abs/2305.04502 (PDF lido).

### Princípio
Estende DEHB (Hyperband + evolução diferencial, uma subpopulação por fidelidade) para múltiplos objetivos com duas mudanças: (1) **promoção** entre fidelidades em cada bracket de successive halving (top 1/η) passa a ordenar por **non-dominated sorting (NDS)** e desempatar dentro da última frente por **crowding distance (variante NSGA-II)** ou **EpsNet** (escolhe iterativamente o ponto mais distante dos já escolhidos); (2) **seleção** DE: alvo × filho comparados pela frente NDS da população global; se na mesma frente, sai o de menor contribuição de hipervolume. Mutação rand1 + crossover binomial; pais de fidelidade inferior quando a subpopulação é pequena.

### Resultados-chave (números)
- 15 benchmarks: NAS (NAS-Bench-101/1Shot1/201), NAS+HPO (Flowers e Fashion-MNIST cruas: acurácia × tamanho do modelo; JAHS-Bench-201 surrogates: acurácia × latência), fairness (Adult: acurácia × DSP).
- Protocolo: **10 seeds**; 24 h de relógio nas bases cruas, 4 h nas surrogate; limite de ⌈20+80·√#HPs⌉ avaliações; métrica log10(HV − HV_emp).
- Ganham na média de rank por HV ao fim, mas: JAHS-cifar10 MO-DEHB_NSGA-II −8,103±0,286 vs SMAC −7,807±0,601; JAHS-fashion −8,430±0,342 vs RS −7,515±1,044; Adult −1,932±0,002 (ambas variantes) vs RS −1,919±0,004.
- **Flowers (raw, o benchmark mais parecido com o nosso): MO-DEHB_NSGA-II −1,091±0,019 vs RS −1,066±0,013**; no texto, AGE-MOEA supera MO-DEHB_NSGA-II no Flowers, e EpsNet é melhor no início e pior no fim.
- NAS-201-ImageNet: **RS competitivo até o fim**; diferenças nos NAS-Bench-101/1Shot1 são da ordem de 0,01–0,03 em log, dentro de 1 desvio-padrão.
- Início lento em NAS-201-Cifar100 e JAHS-Colorectal (só fica competitivo depois).

### O que se aplica ao nosso caso
- É o otimizador base da nossa busca κ×FLOPs. A promoção por NDS sem considerar o "porte" mistura B0 e B4 na mesma frente; a MISSION pede cinco subpopulações B0–B4 com Pareto só entre comparáveis — isso é uma modificação nossa, não do paper.
- O DEHB opera em vetor contínuo [0,1]^D (igual ao nosso genoma atual, all_results_today.md:93). A MISSION pede genoma hierárquico com reparos — também fora do paper.
- **Promoção multi-fidelidade favorece quem aprende rápido**: o ranking em fidelidade baixa decide quem sobe. Com SPiFiL (pesos bons sem treino), modelos pequenos/rasos convergem mais cedo → risco de "vantagem indevida dos pequenos nas promoções" (MISSION A1-C). O paper não trata correlação entre fidelidades.
- Protocolo de 10 seeds e HV contra frente empírica combinada é o padrão a imitar para a comparação final do Agente 4 (n=1 não basta).

### Predições para as nossas hipóteses
- **Saturação/κ×FLOPs:** MO-DEHB mantém diversidade na frente via crowding/EpsNet → deve expor a região de saturação (frente plana acima de certo FLOPs) se ela existir.
- **Ganho sobre RS é pequeno e dependente do problema** (Flowers e NAS-201-ImageNet) → predição: na nossa escala (poucos candidatos por dataset), MO-DEHB vs busca aleatória estratificada deve ser comparado com várias seeds antes de atribuir ganho ao otimizador.
- **Promoção:** se ranking em fidelidade 5/15 tiver baixa correlação de Spearman com 100/200, a promoção por NDS seleciona errado — hipótese diretamente testável com os dados de fidelidade existentes (quantitativa 4).
- Profundidade, patches, protótipos, camada a camada, compound: não trata.

### Limitações
- Avaliação com poucos candidatos por benchmark e diferenças frequentemente dentro do desvio-padrão; ganho declarado "em todos os 15" não se sustenta na Tabela 1 linha a linha (RS/SMAC/QNParEGO/AGE-MOEA vencem em alguns).
- Na Tabela 1 do arXiv, a linha **Flower é idêntica, número a número, à linha NAS1shot1-cifar100** (inclusive a MO-DEHB_NSGA-II −1,091±0,019) — provável erro de cópia; tratar o resultado Flowers com cautela.
- Versão síncrona (paralelismo limitado; versão assíncrona fica como trabalho futuro).
- Só 2 objetivos por benchmark; sem genoma hierárquico/condicional; sem herança de pesos entre gerações (cada configuração é treinada do zero).

---

## 6. Literatura adicional (curta) que sustenta decisões

### 6.1 Compound scaling em CNNs pequenas
- **Han et al., "Model Rubik's Cube: Twisting Resolution, Depth and Width for TinyNets", NeurIPS 2020** — https://arxiv.org/abs/2010.14819 (abstract lido). Ao **encolher** a partir do EfficientNet-B0, "**resolução e profundidade são mais importantes que largura**"; inverter a fórmula do EfficientNet não serve para redes pequenas. TinyNet-E: 59,9% top-1 com ~24M FLOPs (+1,9 pp sobre MobileNetV3 no mesmo custo).
  → Decisão: em B0 (≤300k) não reduzir profundidade para economizar params; preferir manter blocos e estreitar canais. Com resolução fixa em 224, a profundidade é a dimensão a proteger nos portes pequenos. **Predição:** em B0–B1, crescimento "depth-first" ≥ "width-first" no mesmo FLOPs.
- **Bello et al., "Revisiting ResNets: Improved Training and Scaling Strategies", NeurIPS 2021** (venue de memória; a página arXiv não a lista) — https://arxiv.org/abs/2103.07579 (abstract lido). "Escalar a profundidade em regimes em que pode haver overfitting; largura é preferível caso contrário"; aumentar resolução mais devagar que o EfficientNet recomenda.
  → Decisão: a regra de escala depende do regime de treino; com datasets pequenos e treino longo a profundidade pode overfitar → medir κ val por porte e não assumir α>β.
- **Dollár, Singh & Girshick, "Fast and Accurate Model Scaling", CVPR 2021** — https://arxiv.org/abs/2103.06877 (abstract lido). Estratégias de mesmo FLOPs diferem em ativações e tempo real: as usuais aumentam ativações ~O(s); "fast compound scaling" (principalmente largura, pouca profundidade/resolução) mantém ativações ~O(√s) com acurácia similar.
  → Decisão: registrar memória/ativações e latência além de FLOPs (o objetivo FLOPs sozinho esconde custo em GPU), como a MISSION já pede.

### 6.2 Second-order pooling (B-CNN / MPN-COV / iSQRT-COV / Compact Bilinear)
- **Lin, RoyChowdhury & Maji, Bilinear CNNs for Fine-grained Visual Recognition, ICCV 2015** (venue de memória; arXiv revisado em 2017) — https://arxiv.org/abs/1504.07889 (abstract lido). Produto externo de features de duas CNNs com pooling: CUB-200 84,1%, NABirds 79,4%, FGVC Aircraft 86,9%, Stanford Cars 91,3%; 30 fps em Titan X; a dimensão pode cair uma ordem de grandeza sem perda significativa.
- **Gao, Beijbom, Zhang & Darrell, Compact Bilinear Pooling, CVPR 2016** — https://arxiv.org/abs/1511.06062 (abstract lido). Bilinear tem de centenas de milhares a milhões de dimensões; duas aproximações (via análise de kernel) mantêm o poder discriminativo com "poucos milhares" de dimensões e são treináveis ponta a ponta.
- **Li, Xie, Wang & Zuo, MPN-COV: Is Second-order Information Helpful for Large-scale Visual Recognition?, ICCV 2017** — https://arxiv.org/abs/1703.08050 (abstract lido). Pooling de covariância com **normalização por potência de matriz (raiz quadrada)**; o problema central é estimar covariância "de amostra pequena de features de alta dimensão". ImageNet: +4 pp AlexNet, +3 pp VGG-M, +2,5 pp VGG-16; ResNet-50+MPN-COV > ResNet-101 ≈ ResNet-152.
- **Li, Xie, Wang & Gao, iSQRT-COV, CVPR 2018** — https://arxiv.org/abs/1712.01034 (abstract lido). Raiz quadrada da matriz por iteração de Newton–Schulz (pré-normalização, iteração acoplada, pós-compensação), só multiplicações de matriz → muito mais rápido que EIG/SVD na GPU, converge em menos épocas; estado da arte em 3 bases fine-grained.

→ O que sustenta no nosso caso:
1. [nosso] A leitura COV (PCA top-64 por bloco → sqrtm via eigh fp64 → triângulo superior, signed-sqrt+L2, StandardScaler) sobe muito sobre ORIG, ex. eggs H-LEM c00003 f200 0,580 → 0,958; larvae 0,858 → 0,967 (all_results_today.md:32, 68). É exatamente o pipeline MPN-COV (normalização por raiz de matriz) + redução de dimensão de B-CNN/compacto. A literatura dá um mecanismo plausível (estatística de 2ª ordem + normalização de potência), mas **COV é outra régua** (scaler, múltiplos blocos) e não entra na seleção (R2).
2. Decisão para o Agente 3: se 2ª ordem entrar no backbone, precisa estar **dentro** da cabeça 1280 (dimensão fixa, ex. compact bilinear / covariância de k canais projetada para 1280) e medida em ORIG — nunca o CovReadout externo como substituto silencioso.
3. Custo: eigh fp64 está ~1 h por fit no athena (all_results_today.md:290); iSQRT-COV (Newton–Schulz, só matmul, BF16/FP32) é a alternativa documentada para trazer isso ao custo de treino; validar contra eigh em FP32 (regra HPC da MISSION).
4. MPN-COV aponta o risco de covariância com poucas amostras espaciais e muitos canais → em mapas profundos (poucas posições, muitos canais) a covariância fica mal condicionada; o PCA top-k antes do sqrtm é a mitigação já usada — manter k ≤ nº de posições espaciais.
5. **Predição:** a vantagem COV−ORIG é maior em parasitas (textura/forma fina, fundo preto, poucas classes) que em RS/AID (contexto/cena); e a integração de 2ª ordem dentro da cabeça 1280 deve recuperar parte (não necessariamente toda) da diferença COV−ORIG.

---

## 7. Síntese: predições consolidadas (para research_hypotheses.md)

| # | Hipótese | Predição da literatura | Fonte | Controle × tratamento sugerido |
|---|---|---|---|---|
| P1 | Profundidade ótima | parasitas (alinhados) saturam em 2–3 blocos; Flowers/RS pedem mais | PCANet (2 estágios), Coates (1 camada), EfficientNet (largo-raso perde alto nível) | mesmo porte, depth 2/3/4/6 com width compensada; κ val ORIG |
| P2 | Patches simples insuficientes em camadas profundas | kernels por patch em blocos profundos ficam redundantes/baixo rank; falham onde há variabilidade de pose/escala | Coates (campo grande), PCANet (CIFAR), Belilovsky (k=1 satura) | rank efetivo/redundância por bloco + κ por sonda (diagnóstico) |
| P3 | Protótipos com whitening | whitening ZCA + k-means(++) ≥ FPS no mesmo K; remover média do patch é obrigatório | Coates, PCANet | SPiFiL-0 vs SPiFiL-1 (k-means+ZCA), mesmo K e arquitetura |
| P4 | Aprendizagem local preserva | freeze+objetivo local ≈ ponta a ponta (−0,3 pp em VGG-11), mas só com objetivo local "mais profundo" | Belilovsky | noae vs AE local vs AE+LeJEPA, protect freeze/lowlr/full |
| P5 | Fusão intermediária | saídas por camada somam (+1,2 pp); multiescala +1,5 pp | Belilovsky, PCANet | cabeça 1280 GAP vs concat multiescala, mesmo FLOPs |
| P6 | Compound scaling | width+depth juntos > uma dimensão; nos portes pequenos, profundidade > largura | EfficientNet, TinyNet, Bello | crescimento só-width × só-depth × compound no mesmo FLOPs |
| P7 | Saturação antes de 5,5M | com resolução fixa e dados pequenos, κ satura antes de B4; não monotonicidade esperada | EfficientNet Obs. 1, Bello | curva κ val × params B0..B4, ≥3 seeds |
| P8 | Promoção multi-fidelidade enviesada | ranking em fidelidade baixa favorece pequenos/rápidos | MO-DEHB (SH por NDS) | Spearman κ(f5/f15) × κ(f100/f200) por porte |
| P9 | 2ª ordem na cabeça | covariância normalizada por raiz explica parte do salto COV−ORIG | MPN-COV, iSQRT-COV, B-CNN | cabeça 1280 GAP vs cabeça 1280 com cov compacta, régua ORIG |
| P10 | Otimizador | ganho do MO-DEHB sobre busca aleatória é pequeno e dependente do problema | MO-DEHB Tab. 1 | MO-DEHB vs RS estratificado, mesmas avaliações, ≥3 seeds |

Todas são predições; nenhuma está comprovada no nosso regime (≤5,5M, 224 px, SVM linear sobre encoder sem fine-tuning supervisionado, datasets pequenos).
