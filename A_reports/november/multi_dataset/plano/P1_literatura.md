# P1: como as leis de escala foram descobertas, e o que o nosso plano deve copiar

Escrito em 07/11/2026. Só leitura de artigos e do código; nada foi treinado nem lançado.

**Resposta curta:** a literatura ensina o método, não os números.
Nenhum artigo lido ajusta lei para SSL de imagem em CNN pequena, com poucos dados e leitura por SVM. Cada pedaço existe em algum lugar, mas o conjunto é novo.

Cinco coisas mudam o plano:

1. **B0–B3 é o produto, não o experimento.** Se profundidade, largura e resolução crescem juntas, ninguém consegue dizer qual delas ajudou (Kricheli p3-5). Para achar a regra, cada orçamento precisa de várias formas diferentes de rede.
2. **As 6 frações valem menos do que parecem.** Em `remote_sensing` só 2 delas têm 25 imagens por classe, e 1% de `Larvae_subgroup` são **3 imagens** (seção 7).
3. **O ruído manda.** κ=0,90 com ±0,02 é ±20% do erro que sobra. As melhores leis acertam cerca de 4% (Choshen p4). Só o sorteio da validação já dá ±0,04 em `Larvae_subgroup` [conta].
4. **Temos dois eixos de dados, não um:** imagens no SSL e rótulos no SVM. Nos YAMLs `pcts` eles andam juntos; `svm_train_pct` os separa sem treino novo (`eval/svm.py:743`).
5. **Perda baixa não garante κ alto.** Com o mesmo erro de reconstrução, a acurácia foi de 50% a quase 90% (Balestriero p2). A perda não pode ser o eixo da lei.

**Marcas:**
- **pN:** página do PDF do arXiv, lida no texto.
- **[resumo]:** veio só do abstract.
- **[web]:** veio de extrato de página web, não do PDF.
- **[não lido]:** não consegui ler.
- **[não achado]:** procurei e o texto não traz.
- **[conta]:** conta nossa, não do artigo.
- **[hip]:** hipótese nossa, sem medida.
- **Nós vemos:** limite que o artigo não diz.

---

## 1. Três tipos de "lei"

| Tipo | O que faz | Quem | Como se prova |
|---|---|---|---|
| **Receita de alocação** | Diz como gastar o orçamento. Não prevê erro | EfficientNet, V2, RegNet, Bello, Dollár | Ganha de outra regra ao mesmo custo |
| **Lei preditiva** | Prevê o erro de um ponto que não foi treinado, e diz o erro da previsão | Rosenfeld, Hoffmann, Alabdulmohsin, Hoiem | Ponto retido, erro em número |
| **Explicação** | Diz de onde vem o expoente | Bahri, Sharma, Hutter | Teoria mais teste controlado |

O usuário pediu uma lei "de montagem", no estilo EfficientNet. Pelo critério acima, isso é receita. Para virar lei, ela precisa prever κ em dataset que não entrou no ajuste.

---

## 2. Fichas dos artigos pedidos

### 2.1 Regras arquiteturais

**EfficientNet** (Tan e Le 2019, 1905.11946)
- **Desenho:** ImageNet, operador fixo. Varia um eixo por vez sobre o B0 (w de 1 a 5, d de 1 a 8, r de 1 a 2,5), depois a largura sob 4 bases (p4). Regularização não fica fixa: dropout sobe de 0,2 a 0,5 (p7).
- **Modelos:** a grade é só "small grid search", sem tamanho, faixa ou passo (p2, p5): não dado. Família de 8 modelos (p6). Sementes [não achado].
- **Equação:** d=α^φ, w=β^φ, r=γ^φ, com α·β²·γ²≈2 e α=1,2, β=1,1, γ=1,15 (p5).
- **Ajuste:** nenhuma curva. A restrição é conta de custo, porque FLOPs crescem com d·w²·r². O medido são três números, em φ=1, num modelo, num dataset (p5). O produto real é 1,92 [conta].
- **Validação:** aplicado a MobileNet e ResNet-50 (p6), ganho de até 2,5% (p8). Nunca prevê erro.
- **Limites:** "the effectiveness of model scaling heavily depends on the baseline network" (p2); busca em modelo grande seria melhor, mas cara (p5).
- **Nós vemos:** sem réplica. B0 tem 5,3M parâmetros (p6), acima do nosso teto, então α, β, γ não se herdam. φ dos modelos não aparece no texto; B5–B7 fecham com φ≈5, 6, 7 e B1–B3 não fecham com inteiros [conta, resoluções de Bello p14].

**EfficientNetV2** (Tan e Le 2021, 2104.00298)
- **Desenho:** busca de arquitetura ciente do tempo de treino, depois escala S→M/L (p4). ImageNet, ImageNet21k, 4 de transferência.
- **Modelos:** até 1000 na busca, ~10 épocas, imagem reduzida (p4). Família S/M/L/XL. Réplicas de 3 a 5 em algumas tabelas (p5, p8, p9).
- **Equação:** recompensa A·S^w·P^v, w=−0,07, v=−0,05 (p4). Coeficientes novos de escala [não achado].
- **Ajuste:** w e v "empirically determined" (p4).
- **Validação:** mesma receita, V2-M 85,1 contra B7 85,0, com −17% de parâmetros e −37% de FLOPs (p8).
- **Limites:** a regra antiga vira heurística: teto de 480 px e mais camadas nos estágios finais (p4). "Equally scaling up every stage is sub-optimal" (p3). Imagem de 128 px prefere augmentation fraca, e 300 px, forte (p5).
- **Nós vemos:** a escala não uniforme não tem ablação isolada. O menor V2 (7,4M, p9) também passa de 5M.

**RegNet** (Radosavovic 2020, 2003.13678; base em 2019, 1905.13214)
- **Desenho:** ImageNet, bloco X, r=224, 360–400 MF, 10 épocas (p4). Não procura uma rede: refina um espaço de 16 para 6 dimensões (p6), olhando a curva de erro da população (EDF).
- **Modelos:** n=500 por espaço (p4); 100 redes de 25 épocas por regime (p7); melhor de 25, repetida 5 vezes (p8). Em 2019: 25 mil por espaço em CIFAR-10 (p3).
- **Equação:** u_j=w_0+w_a·j; w_j=w_0·w_m^⌊s_j⌉ (p5-6). Achado: ~20 blocos, w_m≈2,5, b=1 (p7).
- **Ajuste:** grade em w_0, w_a, w_m minimizando o log-ratio médio (p6). Intervalo por bootstrap, 10⁴ repetições (p3).
- **Validação:** a ordem dos espaços se mantém em 800 MF, 50 épocas, 5 estágios e outros blocos (p6). Em 2019: com 10 amostras a EDF é "quite noisy"; com 100, D≈0,1 (p6). Igualar o custo muda o veredito: "much of the original gap was due to mis-matched complexity distributions" (p5).
- **Limites:** "a fixed resolution of 224×224 is best, even at higher flops" (p7), o oposto do EfficientNet.
- **Nós vemos:** "higher flops" é só 800 MF. Dez épocas valem para população, não para modelo individual.

**Revisiting ResNets** (Bello 2021, 2103.07579)
- **Desenho:** ResNet+D+SE em ImageNet. Largura ×[0,25…2], profundidade 26–400, resolução 128–448 (p5), a 10, 100 e 350 épocas (p6).
- **Modelos:** total não dado; a grade cheia teria 175 [conta]. Tabelas com 2 runs (p5). Sementes das curvas [não achado].
- **Equação:** nenhuma. Erro segue potência de FLOPs em regime baixo e a tendência quebra depois (p5). Expoente [não achado].
- **Ajuste:** inspeção da fronteira velocidade × acurácia (p6).
- **Validação:** ResNet-RS 1,7 a 2,7 vezes mais rápida (p7); EfficientNet-RS só reduz resolução e melhora a fronteira (p8).
- **Limites:** "We therefore do not recommend generating scaling rules exclusively in a small scale regime" (p7). A 350 épocas a profundidade vence; a 10, a largura (p6). Resolução rende menos acima de 320 px (p5).
- **Nós vemos:** épocas vêm misturadas com regularização, que muda entre regimes (p17).

**Fast and Accurate Model Scaling** (Dollár 2021, 2103.06877)
- **Desenho:** ImageNet, bases EfficientNet-B0 e RegNet, escala até 32× (p5-8), mesma receita, 100–400 épocas (p6).
- **Modelos:** total não dado. 3 repetições na tabela principal (p6).
- **Equação:** e_d=(1−a)/2, e_w=a, e_r=(1−a)/2; d′=s^e_d·d, w′=√s^e_w·w, r′=√s^e_r·r. Custo: f=s·dw²r²; p=s^((1+a)/2)·dw²; ativações=s^((2−a)/2)·dwr² (p5). Por camada: f=w²r²k², p=k²w², a=wr² (p3).
- **Ajuste:** sem ajuste; testa a=1/3, 2/3, 4/5, 1 e escolhe a=0,8 (p5, p7).
- **Validação:** tempo correlaciona com ativações, r≥0,99; com FLOPs, 0,81; com parâmetros, 0,56 (p4). "The exact value of α < 1 does not greatly influence the error" (p7).
- **Limites:** só sobe de escala; "may be task dependent" (p8).
- **Nós vemos:** só ImageNet. Se a curva for plana, grade fina em a não compensa.

### 2.2 Leis por dados e tamanho

**Hestness 2017** (1712.00409)
- **Desenho:** 4 domínios, fatias aleatórias cobrindo 2 a 3 ordens de grandeza em passos de ~2×, validação fixa, melhor modelo por fatia (p2, p4-5).
- **Modelos:** número de tamanhos não dado. ImageNet: 1 a 800 imagens por classe, ResNets de 89 mil a 121M (p8).
- **Equação:** ε=2,24·m^−0,31; s(m)=2,76·m^0,57, com s em milhões de parâmetros e m em imagens por classe (p9). Expoentes gerais de −0,07 a −0,35 (p2).
- **Ajuste:** [não achado].
- **Validação:** só dentro da amostra, erro relativo abaixo de 0,6% (p6).
- **Limites:** três regiões: pouco dado (abaixo de ~25 imagens por classe é chute, p8), potência e piso (p10-11). O piso não foi alcançado (p11).
- **Nós vemos:** em NMT, modelo fixo dá −0,36 e o envelope, −0,128 (p6). Logo curva por modelo fixo e curva do melhor modelo são coisas diferentes.

**Rosenfeld 2019** (1909.12673, ICLR 2020; é o "2020" do pedido)
- **Desenho:** 6 datasets de imagem e 3 de texto, hiperparâmetros fixos, grade de modelo × dados, subamostra por classe (p3-4).
- **Modelos:** 42 a 49 configurações por dataset, até 6 constantes (p6).
- **Equação:** ε̃=a·n^−α+b·m^−β+c∞; ε̂=ε0·‖ε̃/(ε̃−iη)‖, onde ε0 é o chute (p6). ImageNet: α=0,75, β=0,61 (p16).
- **Ajuste:** mínimos quadrados na divergência relativa, início aleatório (p6, p8, p16).
- **Validação:** 10-fold, erro médio abaixo de 1% (p6). Ajustando com 1/16 do modelo e 1/8 dos dados, prevê o resto no ImageNet com μ=−4,5% e σ=4,7% (p8).
- **Limites:** só hiperparâmetros constantes (p10). Precisa de "at least two measurements in each scaling dimension" e pelo menos tantos pontos quanto constantes (p21).
- **Nós vemos:** c∞ quase zero em 4 de 6 datasets de imagem (p16): o piso não foi identificado.

**Rosenfeld 2021, poda** (2006.10621; tese em 2108.07686)
- **Desenho:** poda iterativa em ResNets, variando d, l, w, n (p2-3).
- **Modelos:** 152 configurações com 4.301 pontos, e 15 com 274 (p3); 5 constantes (p6).
- **Equação:** invariante m*=l^φ·w^ψ·d (p5). Valores [não achado].
- **Ajuste:** mínimos quadrados no erro relativo (p4-5).
- **Validação:** μ<2%, σ<6% (p5). Com 40 redes, o desvio nas medidas é de um ponto percentual (p7).
- **Limites:** extrapolar é sensível a erro sistemático (p19).
- **Nós vemos:** "40" foi medido com a forma já conhecida. A tese não muda a forma, mas mostra um platô sem ruído que não é erro de Bayes (p81): nosso piso ajustado não é "o limite do dataset".

**Kaplan 2020** (2001.08361)
- **Desenho:** Transformers de texto, N de 768 a 1,5B, D de 22M a 23B (p7).
- **Modelos:** não dado.
- **Equação:** L(N)=(8,8e13/N)^0,076; L(D)=(5,4e13/D)^0,095 (p4); forma conjunta na p5.
- **Ajuste:** [não achado].
- **Validação:** 6 ordens de grandeza; erro fora da amostra [não achado]. Forma quase não importa: razão largura/profundidade varia 40 vezes com pouco efeito (p8).
- **Limites:** sem teoria (p22); não estudou aumento de dados (p23).
- **Nós vemos:** é a perda de texto. Hoffmann culpa o cronograma fixo (Hoffmann p3). Tay mostra forma mudando o desempenho final com perda parecida (seção 5). Copie o experimento, não a conclusão.

**Hoffmann 2022** (2203.15556) e **Besiroglu 2024** (2404.10102)
- **Desenho:** três métodos: curvas de treino, IsoFLOP (9 orçamentos, varia o tamanho, parábola por orçamento) e paramétrico (p5-6).
- **Modelos:** mais de 400, de 70M a 16B (p1).
- **Equação:** L̂=E+A/N^α+B/D^β; E=1,69, A=406,4, B=410,7, α=0,34, β=0,28 (p6, p25). Expoentes do ótimo a≈b≈0,5 nos 3 métodos (p8).
- **Ajuste:** L-BFGS, Huber com δ=10⁻³ no log, grade de inícios (p6, p25).
- **Validação:** 3 métodos concordam; bootstrap (p8); o modelo Chinchilla ganha do Gopher (p9).
- **Limites:** poucos runs no topo, menos de 1 época (p15).
- **Besiroglu:** os dados da figura dão outra lei. O intervalo real é "roughly 50-fold wider" (p5); um intervalo tão apertado exigiria ~600 mil runs (p5). Causa: otimizador parou cedo porque a perda estava em média, não em soma (p4-5).
- **Nós vemos:** IsoFLOP é o desenho natural da nossa lei arquitetural. δ=10⁻³ não serve para nós, porque nosso resíduo em log é ~0,2 [conta].

### 2.3 Leis em visão e o molde de forma

**Scaling ViT** (Zhai 2021, 2106.04560)
- **Desenho:** ViT supervisionado, 11 formas de 5M a 1,8B, JFT de 30M a 3B (p7, p11). Sonda few-shot: regressão linear com L2 em feature congelada, 5 e 10 exemplos (p11-12).
- **Modelos:** total não dado; ~260 linhas nas tabelas.
- **Equação:** E=a(C+d)^−b+c (p3), com C em dias de TPU. 10-shot: 0,12+0,63(C+0,52)^−0,32 (p2). b vai de 0,20 a 0,77 conforme o dataset (p12).
- **Ajuste:** sobre a fronteira de Pareto (p2); método [não achado].
- **Validação:** só "fairly accurately" (p2); fora da amostra [não achado].
- **Limites:** o menor modelo sai da lei: as curvas do Ti/16 se sobrepõem para todo tamanho de dado (p2). L/16 em 30M piora com mais passos (p14).
- **Nós vemos:** é a lei com sonda congelada mais próxima da nossa, mas o menor dado (30M) é cerca de 800 vezes o nosso maior dataset [conta]. Ela mistura modelo e passos em C.

**Revisiting Neural Scaling Laws** (Alabdulmohsin 2022, 2209.06640)
- **Desenho:** JFT-300M few-shot, NMT, LM, BIG-Bench; x é exemplos vistos, um eixo por vez (p6-7).
- **Modelos:** 6 arquiteturas de 32M a 494M em visão (p6); ~90 tarefas (p2). Pontos por curva não dado.
- **Equação:** M1: ε=βx^c. M2: ε−ε∞=βx^c. M3: ε=β(x⁻¹+γ)^(−c). M4: (ε−ε∞)/(ε0−ε)^α=βx^c, com ε0 no chute (p3-4).
- **Ajuste:** perda quadrática em log, descida por blocos (p5).
- **Validação:** **é o protocolo a copiar.** Ajusta em [0,τ], testa em (τ,2τ], τ=x_max/2 (p6). M4 vence em mais de 70% das tarefas de visão (p7).
- **Limites:** curvas com expoentes de −0,24 a −0,40 "overlap with each other significantly"; o melhor ajuste escolhe −0,24 e a melhor extrapolação, −0,40. "Scaling law parameters should be validated by" extrapolação (p5).
- **Nós vemos:** só extrapola 2×. Nosso ruído é várias vezes o erro de M4 [conta].

**Getting ViT in Shape** (Alabdulmohsin 2023, 2305.13035): **molde da lei arquitetural**
- **Desenho:** ViT supervisionado em JFT-3B. Estrela: parte de um centro grande e varia um eixo por vez com os outros folgados (largura, profundidade, MLP; 6 valores cada, 3 durações, p5-6, p17). Depois grade 4×4×4 pequena para achar um ponto na fronteira. Depois sobe tudo junto (p5).
- **Modelos:** "only 115" experimentos (p2), contra "over 400" de Hoffmann.
- **Equação:** f_k=α_k·x_k^(−a_k)+(β_k·x_k^(b_k)+ξ_k)·t^(−c)+ε_k. Ótimo x*_k ∝ t^(s_k), com s_k=c/(a_k+b_k) (p4). Medido: s_depth≈0,45, s_width≈0,22, s_MLP≈0,6; c≈0,65±0,06, igual entre eixos (p6).
- **Ajuste:** "minimizing the relative error" (p4). Otimizador, início e sementes [não achado].
- **Validação:** centro da estrela nunca entra no ajuste (p6). SoViT-400m iguala g/14 com menos da metade do custo de inferência (p1, p8). Erro numérico da predição [não achado].
- **Limites:** "We also assume that data is unlimited so that there is no risk of overfitting" (p4). Resolução fixa em 224 (p17). Ótimo precisa cair no interior da grade (p17). Em modelo pequeno, a forma muda com o domínio (p4, p8).
- **Nós vemos:** sem intervalo de s_k. Variar a forma em +33% ou +200% de profundidade quase não muda o resultado (Fig. 6): ótimo largo. Temos 7 constantes por eixo e dado limitado, então a eq. não cabe em 4 modelos.

**Explaining Neural Scaling Laws** (Bahri 2021, 2102.06701)
- **Desenho:** teoria mais teste em random features, teacher-student, WRN e CNN em CIFAR, SVHN e MNIST, linear sobre EfficientNet-B5 congelada (p2, p6, p17, p32).
- **Modelos:** não dado; 8 a 50 repetições por ponto (p17).
- **Equação:** quatro regimes. Limitado por variância: L−L∞ ∝ 1/x. Limitado por resolução: x^(−4/d) ou mais íngreme (p2, p6, p9).
- **Ajuste:** [não achado].
- **Validação:** medido α≈1 no primeiro regime e 0,26 a 0,62 no segundo (p2). Fora da amostra [não achado].
- **Limites:** trocar o dataset e somar ruído na entrada movem o expoente (p3, p9). Número de classes não move (p9-10). Em features fixas com rótulo real, o expoente cai para 0,05–0,25 (p33).
- **Nós vemos:** nosso SVM é "feature fixa, rótulo real". Espere expoente baixo [hip], e não explique diferenças entre datasets pelo número de classes.

### 2.4 Outros que mudam o desenho

| Artigo | O que traz | Número-chave |
|---|---|---|
| **Sharma e Kaplan 2020** (2004.10802) | α≈4/d, d medido na última camada oculta | MNIST sai 37% fora; CNN só até ~70 mil parâmetros (p20) |
| **Caballero 2022** (2210.14891) | BNSL: lei com quebras, 3 constantes por quebra | Melhor em 69,44% das tarefas de visão (p5). Quebra além dos dados não dá para prever (p11). Com parâmetros no eixo, "considerably larger than 7 points" (p18) |
| **Hutter 2021** (2102.04074) | O expoente vem da cauda de casos raros | Modelo de brinquedo: "totally unrealistic" (p13) |
| **Nakkiran 2019** (1912.02292) | Mais dado ou modelo pode piorar perto do limiar | "having 4× more train samples does not help" (p9); com parada ótima, não viram piora (p10) |
| **Sorscher 2022** (2206.14486) | Poda de dados pode passar da potência | Exige boa métrica; ganho pequeno no ImageNet (p32) |
| **Choshen 2025** (2410.11840) | 485 modelos, mais de 1.000 leis | "5 models is a safe bet" (p7); "4% is the best ARE typically obtained" e até 20% ainda distingue escolhas (p4). ARE = erro relativo absoluto médio |
| **Li 2025** (2502.18969) | Levantamento de 51 artigos, checklist | Pontos "from 4 to several hundred, but most used fewer than 50" (p5). Log-linearizar "generally not advised" (p7) |
| **Ivgi 2022** (2202.06387) | 8 escalas × 5 sementes, sobre pré-treino | Prevê 2 escalas acima com erro ≤2,5% na maioria (p6). Com menos de 10 mil exemplos, R²<0,85 (p5) |
| **Kricheli 2026** (2605.08541) | Desenho colinear não identifica expoentes | Desenho não colinear vence em 97,3% dos casos, e em 68,5% a orçamento igual (p1, p8). Rodapé diz NeurIPS 2026, não confirmei |

---

## 3. Dados pequenos e extrapolação de curvas

Curva de aprendizado é o erro contra o número de exemplos.

| Artigo | Desenho | Forma | Pontos | Erro fora da amostra | Limite |
|---|---|---|---|---|---|
| **Hoiem 2021** (2010.11029) | ResNets em Cifar100, Places, Caltech. 25 a 400 por classe (p4, p11) | e(n)=α+η·n^γ, resumo por e_N e β_N (p3) | 4 bastam com erro ≫ ruído (p5, p12) | RMSE 0,85 ponto em n=400; γ fixo em −0,5 dá 1,21 (p5) | "α, η, and γ cannot be meaningfully compared" (p3). Abaixo de 25 por classe, a curva desvia (p11). n=1600 é "the limit of accurate extrapolation" de n≤400 (p5) |
| **Figueroa 2012** [web] | SVM linear em dado biomédico, 100 repetições | 1−a−b·x^c, peso maior no fim | ≥5 | RMSE médio 0,29–1,52% | IC "may sometimes be optimistic" |
| **Cho 2015** (1511.06348) | GoogLeNet em CT, 5 a 200 por classe | 100+b1·x^b2, assíntota fixa | 6 | 98% previsto, 97,25% medido, 5× (p6) | Um ponto só. Preprint |
| **Mahmood 2022** (2207.01725) | CIFAR, ImageNet, VOC. Ajuste em 10/20/50% | Potência, arctan, log, raiz (p3) | r≤10 | CIFAR10 potência: 39 → 6 ao subir de 10% para 50%; arctan: 8 → 0,4 (p5) | Bom ajuste não é boa decisão: arctan pede ~5× dados (p5-6) |
| **Jain 2023** (2303.01598) | 16 datasets, ajuste em 5 pontos | Log quadrático por partes (p3) | 5 + meta-modelo | Sobre-estima dados em 12× (p1); forma por partes baixa para 1,9× | Sinal oposto ao de Mahmood |
| **Bornschein 2020** (ICML) | MNIST a ImageNet, 30 sementes | SNR=√(Δ²/Var[Δ]) (p7) | Nº de tamanhos [não achado] | Nenhum par cruzou fora da incerteza (p6) | "only an hypothesis", só além do limiar de interpolação (p2). Tamanho ideal quase não depende de n (p5) |
| **Brigato 2020** (2003.12843) | CNN 27k–398k contra ResNet-20 | Sem lei | 10 runs | CNN ganha em N=10, perde em N=1280 (p5): cruza | Máximo no teste por run (p3) |
| **Konz 2024** (2401.08865) | 11 datasets binários, N de 500 a 1750 | log L ≲ −(1/d)log N+log K_F (p5) | Não dado | Extrapolação [não achado] | É cota, não previsão: não tire γ de d |
| **Rodrigo González 2026** (2606.06725), preprint | U-Net de 32,9k a 6,1M, 15 modelos, 3 tamanhos, 2 datasets (p4-6) | Kaplan com αD=0,249 global, nível por rede (p4-6) | 3 | Spearman 0,950; só ordem, "relative but not absolute" (p6, p12) | "sensitive to noise" com 3 tamanhos (p12). Ajusta na perda de teste, o que nossa regra proíbe |
| **LCDB 1.1** (2505.15657) | 32 aprendizes, 265 datasets tabulares (p4-5) | POW4, MMF4, WBL4 (p7) | Não dado | Só interpolação (p9) | 14,9% das curvas são mal comportadas (p7). Peaking "cannot be modeled" (p25) |
| **Viering e Loog 2021** (2103.10948) | Revisão | "no universal model" (p1); piso é "a necessity" (p7) | Não dado | Não se aplica | Curvas cruzam em 15 de 36 estudos (p5) |
| **Mohr e van Rijn 2022** (2201.12150) | Revisão | Melhor ajuste não é melhor preditor (p19) | Não dado | Não se aplica | Falta benchmark de extrapolação (p54) |

**Curva por época** (Domhan 2015, LC-PFN 2310.20447): resolvem outro problema, parar treino ruim. Não servem para 6 frações. **Klein 2017** [não lido]: OpenReview bloqueou e o PDF deu 404.

**Resumo para nós:**
- Forma: piso + potência, ajustada no erro.
- Com poucos pontos, fixe ou compartilhe γ. Hoiem: desvio de β cai de 0,72 para 0,2 (p5).
- Não extrapole mais que 2 a 4 vezes.
- Comece acima de ~25 imagens por classe.
- Falha quando: horizonte longo, poucas imagens por classe, n≈d, curva que sobe e desce, diferença entre redes menor que o ruído.

---

## 4. Escala em self-supervised e representação

**Só um artigo tem lei ajustada (Cherti), e ele usa pares imagem-texto.** Em SSL só de imagem existem tendências, descritas a olho.

| Artigo | Desenho | Modelos | Equação | Sonda | Achado | Limite para nós |
|---|---|---|---|---|---|---|
| **Goyal 2019** (1905.01235) | Jigsaw e Colorization; AlexNet e ResNet-50; 1M a 100M imagens (p3) | 2 redes | Só tendência: "increases log-linearly" (p8), "saturates (log-linearly)" (p3) | **SVM linear**, grade de 26 C, 3-fold (p3, p14) | Gap entre redes cresce com dados; dificuldade do pretexto ajuda, mas não sempre (p4, p22) | Menor ponto é 27× nosso maior dataset. Passos não ficam fixos (p13) |
| **SEER, Goyal 2021** (2103.01988) | SwAV, RegNet, 1B imagens (p4) | 6 RegNets (p4) | Tendência | Sonda linear e 1%/10% (p5-8) | "the number of updates matters more than seeing the same images multiple times" (p8) | Fixar passos ao subamostrar |
| **SEER 10B, Goyal 2022** (2202.08360) | Até 10B parâmetros (p4) | 7 tamanhos (p32) | "log-linear" (p14) | Várias | Curvas por tarefa não são monótonas (p32) | Média esconde dataset |
| **Kolesnikov 2019** (1901.09005) | 6 CNNs, 4 pretextos, largura 4 a 16 (p3-5) | Total não dado | Tendência | Logística (p11) | "neither is the ranking of architectures consistent across different methods" (p5). Largura ajuda muito em umas e nada em outras. Dimensão: "from 31 % to 43 %" (p8) | **Lei vale para uma perda.** Trocar a perda pode trocar o ranking |
| **SimCLRv2** (2006.10029) | ResNet de 24M a 795M, 1%/10% de rótulos (p4-5) | 13 (p5) | "improves label efficiency by 10×" com 10× o tamanho (p6) | Fine-tuning | Ganho maior com menos rótulos (p4-5) | É fine-tuning, não sonda |
| **Cole 2021** (2105.05837) | SimCLR, 50k a 1M, 3 domínios (p3) | ResNet-50 | "fewer than 500k" (p8) | Linear | 2 a 4 vezes menos resolução perde ~15% e ~34% (p7). Domínio próprio vale mais que volume (p4) | Menor ponto é maior que todos os nossos. LR muda abaixo de 50k (p16) |
| **Xie 2022** (2206.04664) | MIM, SwinV2 49M–1B, 10% a 100% do IN1K (p3) | Até 63 | Regressão: val loss × transferência (p8-9) | Fine-tuning | Modelo grande com pouco dado piora: H fica abaixo de L em 20% (p5). Val loss prevê melhor que treino (p8) | Sonda linear [não achado]. Modelos gigantes |
| **Cherti 2022** (2212.07143) | CLIP, 29 modelos, 80M a 2B, 3B a 34B vistos (p5, p36) | 29 | **E=β·C^α** (p7), sem piso | Softmax em feature cacheada, 10-shot, 25-shot e completo (p8-9) | Mais rótulos mexem em β mais que em α (p8, leitura nossa) | "density of sampling the scales space is low" (p12). α muda com os pontos usados (p3, p24) |
| **DINOv2** (2304.07193) | ViT-g 1,1B, 142M curadas, menores destilados (p4-7) | 6 pontos (p9) | Sem lei | k-NN e linear (p8) | Curadoria bate volume (p8) | Destilar não é nosso caso |
| **DINOv3** (2508.10104) | 7B, 1,7B imagens curadas (p8, p30) | 1 professor, 9 alunos | Sem lei | Linear (p63) | "does not necessarily translate into higher model quality" (p8). Segmentação cai depois de ~200k iterações, classificação sobe (p11) | Global e local divergem |
| **AIM, AIMv2, Web-SSL** (2401.08541, 2411.14402, 2504.01017) | Modelos de 0,3B a 7B | Até 28 | Sem lei; AIM "defer further investigations" (p6) | Sonda atentiva e linear | Perda e sonda em "strong correlation" (AIM p9) sem r dado. Melhor camada não é a última (AIM p7, p13) | Escala fora da nossa |
| **Newell 2020** (2003.14323) | SSL sintético, utilidade U(n)=n̂/n−1 (p2) | Não dado | Definição | Rótulos variados | "utility approaches zero" com rótulos (p8, p11) | Reportar U(n) |
| **Asano 2020** (1904.13132) | AlexNet, N=1, 10, 1000 imagens (p6) | 1 | Tendência | Por camada | "for conv1 and conv2, a single image is enough" (p8). Augmentation vale pelo menos 10 pontos (p7) | **Imagens pagam nas camadas fundas.** Medir por camada |
| **Cao 2021** (2103.13559) | ResNet-18/50, a partir de 2.040 imagens | Não dado | Tendência | Linear e fine-tuning | "Large models suffer more from training on small data" (p7) | Resultado muda com a sonda (p6) |
| **Low-data SSL** (2404.17202) | ResNet-50, ViT-L; 50k a 300k | Curva de 4 pontos (p19) | Limiar lido | Linear | Limite "between 50k and 65k" (p19), mas MedPix de 32k contradiz (p22) | Baseline sem SSL é obrigatório |

**O que escala e o que satura (leitura conjunta):**
1. Dados: ganho perto de log, sempre a olho.
2. Dados saturam quando o modelo é pequeno ou o pretexto é fácil (Goyal p3-4, AIMv2 p4).
3. Tamanho rende mais com mais dados e menos rótulos (Goyal p4, SimCLRv2 p6). Com pouco dado pode piorar (Xie p5, Cao p7).
4. Passos pesam tanto quanto imagens, e muitas curvas misturam os dois (SEER p8, Cole p16).
5. Domínio e composição podem vencer volume (Cole p4, DINOv2 p8).
6. Camadas rasas e tarefas fáceis saturam primeiro (Asano p8).

**CNN pequena + dado pequeno + SSL:** como conjunto, não achei. O menor modelo com contagem dada é 21M (DINOv3), 4 vezes o nosso teto. SVM congelado em SSL só em Goyal, com 1M ou mais de imagens.

---

## 5. O que muda ao medir por SVM em vez de perda

**Não achei nenhum artigo que ajuste lei de escala medindo com SVM em feature congelada de rede pré-treinada.** Procurei "SVM" e "support vector" nos 84 textos. Os mais perto: Goyal (SVM, sem equação), Figueroa e LCDB (SVM como aprendiz, sem rede antes), Zhai e Cherti (sonda linear, outra origem).

A ideia central: leis clássicas têm uma etapa. A nossa tem duas: **treino → feature → leitor**. O leitor tem curva, piso e regime próprios.

| Efeito | Fonte | Como controlar |
|---|---|---|
| **O expoente é do par rede + leitor.** ResNet18 vai de −0,69 a −1,06 só trocando o leitor | Prato p4 | Leitor fixo em toda a curva. Não comparar expoente entre leitores |
| **O piso não é Bayes e muda com rótulos** | Abnar p8; tese de Rosenfeld p81 | Ajustar piso por dataset, não chamá-lo de limite |
| **O leitor troca de regime com n.** Com d=1280, todo ponto vira vetor de suporte até n≈35 [conta, teoria binária e isotrópica] | Muthukumar p14-15 | Tratar n pequeno à parte. Reportar vetores de suporte |
| **Whitening depende de n.** `PCA()` em `pw_whiten` dá no máximo min(n,d) dimensões: 1% de `Larvae_subgroup` lê em 3 dimensões | `eval/svm.py:365-373` | Reportar `dim`. Ajuste feito só no treino, sem vazamento. A variante `_pwlw` existe para n≪d (`eval/svm.py:713`) |
| **Mesma perda, κ diferente.** Em Balestriero, um dos braços usa sinal supervisionado (p7-8): mostra que há folga, não que SSL a alcance | Balestriero p2; Saunshi p7; Tay p5 | Perda só como covariável, termo a termo |
| **Reconstrução aprende primeiro o que tem mais variância, não o que serve.** Top 90% da variância dá 45%; bottom 20%, 55% em TinyImagenet | Balestriero p1, p4 | Separar recon de SIGReg/VICReg na análise |
| **Val loss pode prever, mas não em toda parte.** Xie: correlações fortes em fine-tuning. Balestriero: reconstrução igual, acurácia diferente | Xie p8-9; Balestriero p2 | Testar nos nossos runs, por dataset |
| **Escolher pela perda falha com pouco dado.** Correlação 0,60 em flowers102 sem correção; λ^0,4 não é expoente de dados e foi fixado olhando acurácia | LeJEPA p15 | Recalcular Spearman nos nossos runs, fora da amostra |
| **Camada e pooling são escolhas escondidas.** Em tarefa distante, a melhor camada não é a última | Abnar p12; Guillotine p6-7; AIM p7 | Fixar antes, ou escolher na validação e declarar |
| **C importa.** Goyal escolhe C por 3-fold em 26 valores; Kornblith, 26 valores. Nós usamos C=100 fixo. Viering: hiperparâmetro fixo dá curva subótima | Goyal p14; Kornblith p21; Viering p3; `core/svm.py:143` | Com 3 imagens não há validação cruzada. Manter C fixo e medir sensibilidade em poucas células, pelo argumento `C` do `fit_svm` |
| **SVM linear é o mais bem comportado.** Curvas ruins: 1,4–5,7%, contra 42–65% do sigmoide | LCDB p19 | Manter linear. Padronizar dentro do treino de cada fração |
| **Vazamento em pré-processamento muda 8 a 16% das curvas** | LCDB p5, p16 | Ajustar tudo no treino da fração. O código já faz isso no `pw_whiten` |
| **κ tem degrau, teto e p_e móvel.** O degrau é 1/(N_val·(1−p_e)). Com classes balanceadas, 1−κ é reta no erro. Com desbalanceio, p_e muda com a predição | Schaeffer p4 [conta para κ] | Reportar erro e κ. `compute_metrics` já dá `acc` e `f1` |
| **Faixa estreita esconde tendência.** Só 39% das tarefas tiveram lei limpa em LLM | Lourie p1, p4 | Incluir modelos pequenos. Ler no eixo log(1−κ) |
| **Melhor-de-N infla** | Abnar p9; Prato p3 | Checkpoint pela validação, nunca pelo teste |
| **Posto sem rótulo é alarme, não medida.** RankMe: "only a necessary condition". SIGReg e VICReg empurram o posto por construção [hip] | RankMe p5-6; LiDAR p10 | Usar para detectar colapso |
| **SVM e logística dão quase o mesmo** (r=0,998) | Kornblith p21 | Nossas conclusões devem valer para sonda linear em geral |

**Dois sinais a favor:** o número de vetores de suporte já é calculado (`core/svm.py:121`), embora no caminho `growth_stages` eu não tenha visto ir para o CSV. E validação e teste já são fixos entre frações, medidos em `splits_incremental`.

---

## 6. Como separar lei de aprendizado de lei arquitetural

Em palavras simples:
- **Lei de aprendizado:** mesma rede, mais imagens. Quanto o erro cai?
- **Lei arquitetural:** mesmas imagens, mesmo custo, outra forma de rede. Qual forma erra menos?
- **Interação:** a melhor forma muda quando há mais imagens?

As duas primeiras são perguntas separadas. A terceira é a única que liga as duas, e precisa de teste próprio.

**Como os autores fazem:**

| Truque | Quem | O que separa | Problema para nós |
|---|---|---|---|
| **Gargalo:** um eixo enorme, mede o outro | Bahri p2, Sharma p16, SoViT p4 | Modelo de dado | Dado infinito não existe aqui: maior treino tem 26 mil imagens |
| **Grade cheia:** cada termo da soma é um eixo | Rosenfeld p6, Hoffmann p6 | Modelo e dado, com poucas constantes | 42 a 49 configurações por dataset |
| **Estrela:** um eixo por vez, com centro retido | SoViT p5-6, Tan p4 | Cada eixo de forma | Interação entre eixos fica de fora: Tan Obs. 2 (p4) diz que ela existe |
| **IsoFLOP:** custo fixo, varia a forma | Hoffmann p5-6 | Forma a custo igual | Precisa de pelo menos 3 formas por orçamento |
| **Fronteira:** só o melhor de cada custo | Hestness p4-5, Zhai p2, Cherti p22 | **Nada: mistura tudo** | Viés de melhor-de-N |
| **Escala composta:** tudo cresce junto | Tan p5 | **Nada: é um raio só** | Kricheli p3-5 |

**Por que B0–B3 sozinho não serve** [analogia nossa, Kricheli prova para N e D]: 4 modelos numa linha reta medem um efeito só, o da família inteira. Pôr mais modelos na mesma linha não resolve. Conserto do artigo: pelo menos duas razões bem separadas (p5). IsoFLOP não sofre disso (p2, p9).

**A mesma armadilha nos dados.** Nos YAMLs `pcts`, encoder e SVM usam a mesma fração (`eval/svm.py:743`): imagens de SSL e rótulos andam juntos. É um raio. `svm_train_pct` dá outro.

**Onde os autores discordam:**

| Tema | Lados |
|---|---|
| Profundidade | Tan: sobe junto. RegNet: fica em ~20 blocos. Bello: sobe se houver overfitting. Dollár: prioriza largura |
| Resolução | Tan: agressiva. RegNet: fixa em 224. Bello: rende menos acima de 320. V2: teto de 480 |
| Custo | Tan: FLOPs. Dollár: ativações. Dehghani: "no single cost indicator is sufficient" (p1), e parâmetro é o pior (p9) |
| Regime | RegNet confia em 10 épocas (p4). Bello rejeita (p7) |
| Expoente | Hestness: arquitetura só desloca (p11-12). Choshen e Bahri: muda (p5; p20-21) |
| Hiperparâmetros | Fixar (Rosenfeld p4, Li p30) ou ajustar (Ivgi p1, p6) |
| Sementes | Ivgi usa 5. Choshen prefere mais tamanhos (p1, p7) |
| Sinal do erro | Mahmood subestima dados, Jain sobre-estima |

**A nossa grade da fila** (`fila.md`, 36 YAMLs): 3 larguras, 3 cabeças, 3 profundidades, 200 px, split 1, 50%. É uma estrela útil para triagem, mas sem resolução, sem repetição e com a dimensão da leitura mudando junto. Não dá lei sozinha.

---

## 7. Quantos pontos

Não existe número mágico (Li p5, Choshen p7). Mas dá para tirar pisos.

| O que estimar | Constantes | Mínimo | Recomendado | Fonte |
|---|---|---|---|---|
| Curva e(n), γ fixo ou compartilhado | 2 | 3 pontos | 4 | Hoiem p5 |
| Curva e(n), γ livre | 3 | 4 pontos | 5 a 6, e 1 retido | Hoiem p5, Figueroa [web] |
| Curva até o chute (Rosenfeld, M4) | 4 | 5 pontos | 6 ou mais, com pontos baixos | Rosenfeld p21, Alabdulmohsin 2022 p4 |
| Superfície e(N,n) | 5 | 2 por eixo, e pelo menos 5 pontos | ~40 configurações | Rosenfeld p21, Rosenfeld 2021 p7 |
| Tamanhos de modelo | - | 3 | 5 | Choshen p4, p7 |
| Forma ótima num orçamento | 3 (parábola) | 3 formas | 5, com ótimo no interior | Hoffmann p5-6, SoViT p17 |
| Eixo estilo SoViT | 7 | - | 6 valores × 3 durações por eixo | SoViT p4, p6, p17 |
| Sementes por ponto | - | 3 | 5, se o orçamento couber | Ivgi p6, Choshen p7 |
| Comparar populações | - | 100 | 100 a 1000 | Radosavovic 2019 p6 |
| Achar uma quebra | +3 por quebra | Pontos além da quebra | Mais de 7 no eixo de parâmetros | Caballero p10, p18 |

Regra de Kricheli que vale guardar: **mais ruído pede mais pontos, não mais espalhamento** (p31).

**As 6 frações na realidade.** Contagem real do split 1, em `splits_incremental`. "Úteis" são as frações com 25 imagens por classe ou mais (Hoiem p11, medido em outra tarefa: é ponto de partida, não lei). Ruído é o desvio de κ só pelo sorteio da validação, com κ=0,90 e classes balanceadas [conta].

| Dataset | K | 1% | 5% | 10% | 25% | 50% | 75% | 100% | Val | Úteis (de 6) | Ruído κ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Protozoan | 7 | 268 | 1.315 | 2.622 | 6.546 | 13.086 | 19.627 | 26.162 | 5.607 | 6 | 0,004 |
| Eggs | 9 | 96 | 452 | 897 | 2.228 | 4.448 | 6.670 | 8.885 | 1.903 | 5 | 0,007 |
| aid | 30 | 84 | 364 | 713 | 1.761 | 3.498 | 5.259 | 6.996 | 1.502 | 4 | 0,008 |
| protozoan_cysts | 7 | 72 | 339 | 674 | 1.678 | 3.351 | 5.027 | 6.698 | 1.435 | 5 | 0,009 |
| Protozoan_subgroup | 7 | 56 | 283 | 562 | 1.380 | 2.766 | 4.144 | 5.532 | 1.190 | 5 | 0,009 |
| helminth_eggs | 9 | 40 | 184 | 362 | 897 | 1.791 | 2.688 | 3.580 | 766 | 4 | 0,012 |
| Eggs_subgroup | 9 | 33 | 147 | 276 | 681 | 1.377 | 2.090 | 2.782 | 593 | 4 | 0,013 |
| flowers | 5 | 22 | 99 | 195 | 482 | 963 | 1.444 | 1.922 | 412 | 4 | 0,017 |
| helminth_larvae | 2 | 26 | 124 | 247 | 615 | 1.230 | 1.845 | 2.460 | 527 | 5 | 0,019 |
| remote_sensing | 14 | 16 | 41 | 75 | 175 | 344 | 516 | 682 | 159 | 2 | 0,025 |
| Larvae | 2 | 12 | 57 | 113 | 281 | 560 | 839 | 1.118 | 240 | 5 | 0,028 |
| Larvae_subgroup | 2 | 3 | 27 | 51 | 136 | 246 | 378 | 519 | 107 | 4 | 0,042 |

Notas:
- A 1%, faltam classes: Protozoan_subgroup 6 de 7, Eggs_subgroup 7 de 9, remote_sensing 11 de 14 (pelo prefixo do nome).
- K=9 em Eggs e 7 em Protozoan vêm do brief; o prefixo do nome dá 16 e 12. Com 12, Protozoan perde a fração de 1%.
- Frações não são exatas: 1% de 519 seria 5, e são 3. O eixo x deve ser a contagem real.
- Subconjuntos são aninhados: cada fração contém a anterior. A fração de 10% já existe no disco.
- O ruído de validação é o mesmo para quem usa a mesma validação. Por isso comparar **em pares** dentro do split reduz muito esse ruído.

**Quanto o ruído custa no expoente** [conta]. Reta em log(1−κ), sem piso, com desvio 0,2 no log, que é ±0,02 em κ=0,90. Com piso, piora.

| Frações usadas | 1 run | 3 repetições | 5 repetições |
|---|---|---|---|
| 1, 5, 25, 50, 75, 100 | 0,050 | 0,029 | 0,022 |
| 5, 25, 50, 75, 100 | 0,084 | 0,048 | 0,037 |
| 10, 25, 50, 75, 100 | 0,108 | 0,062 | 0,048 |
| 25, 50, 75, 100 | 0,192 | 0,111 | 0,086 |

Expoentes esperados vão de 0,05 a 0,7 (Bahri p33, Hoiem via Viering p7). Com 4 frações e 1 run, o erro é do tamanho do sinal.
Acrescentar a fração de 10% vale quase tanto quanto triplicar os runs (0,108 contra 0,111). Compartilhar γ entre as redes do mesmo dataset dá 4 redes × 3 splits = 12 curvas, e leva a 0,055 [conta].

**Piso do desenho por dataset de descoberta** [conta]:
- 5 tamanhos (B0–B3 mais um intermediário), porque 4 fica abaixo do "safe bet" de Choshen.
- Aprendizado: 5 tamanhos × 4 frações úteis ou mais = 20 configurações.
- Arquitetura: 4 orçamentos × 5 formas = 20. O centro já está na lista acima, então são 16 novas.
- Soma: ~36 configurações, próximo do piso de Rosenfeld (42 a 49). Com 3 splits, ~108 runs por dataset.
- Eixo rótulos: sem treino novo, só SVM.

---

## 8. O que a literatura não tem

Lacunas reais, para não fingir apoio:
- Lei ajustada em SSL só de imagem [não achado].
- Lei ajustada com leitura por SVM após pré-treino [não achado].
- Eixos imagens × rótulos varridos em separado com sonda congelada [não achado].
- κ como variável de lei: nada nos artigos de curva e nas leis; nos demais não varri sistematicamente.
- Lei com inicialização por dados ou treino camada a camada (L_method §2).
- Forma ótima sob dado limitado: SoViT supõe dado infinito (p4). Trocar compute por n é adaptação nossa.
- Resolução como eixo em lei ajustada: SoViT fixa 224 (p17), Rosenfeld e Hoffmann não têm.

**Atenção:** "sem rótulo no treino" vale para o SSL e o backprop. A verba do SPiFiL é por classe (L_method §3.2), então a inicialização parece consumir rótulos. Isso deve ser conferido, porque muda a conta de rótulos nas frações baixas.

---

## 9. Fontes e procedência

**Como foi lido.** Seis leitores, um por bloco, sobre o texto extraído dos PDFs, com página. Depois conferi mais de 100 trechos direto nas fontes, priorizando os que sustentam as recomendações. Bateram.
- Não confirmei "K=3 ideal" de Kricheli (usei só "duas razões", p5) nem os números de Ciga (2109.01721), e por isso ficaram de fora.
- Números citados só uma vez, em tabela, vieram do leitor e não foram todos reconferidos.
- Figueroa veio de extrato web. Klein 2017 não foi lido.
- Nada acima veio só do resumo. As duas frases que só achei lá (Goyal, pretextos pouco difíceis; AIM, "no sign of saturation") ficaram de fora.
- "Rosenfeld 2020" do pedido é o artigo da ICLR 2020 (arXiv 2019). Incluí também o da poda (arXiv 2020, ICML 2021).

**Em cache** (`<scratchpad>/law/txt/`): 1712.00409, 1905.11946, 1905.13214, 1909.12673, 2001.08361, 2003.13678, 2004.10802, 2102.06701, 2103.06877, 2103.07579, 2104.00298, 2106.04560, 2203.15556, 2206.14486, 2209.06640, 2210.14891, 2305.13035, 2404.10102, 2511.08544.

**Baixados agora** (`<scratchpad>/plit/txt/`):
- SSL: 1901.09005, 1904.13132, 1905.01235, 2003.14323, 2006.10029, 2011.13377, 2103.01988, 2103.13559, 2105.05837, 2112.10740, 2202.08360, 2206.04664, 2212.07143, 2304.07193, 2307.03275, 2401.08541, 2404.17202, 2411.14402, 2504.01017, 2508.10104.
- Leitura e métrica: 1805.08974, 2005.08054, 2110.02095, 2110.06990, 2109.10686, 2202.14037, 2206.13378, 2210.02885, 2304.15004, 2312.04000, 2402.11337, 2507.00885.
- Curvas e dados pequenos: 1511.06348, 2003.12843, 2010.11029, 2103.10948, 2201.12150, 2207.01725, 2303.01598, 2310.20447, 2401.08865, 2403.06311, 2505.15657, 2510.11553, 2511.13883, 2606.06725; Domhan 2015 (IJCAI), Bornschein 2020 (ICML).
- Ajuste e teoria: 1912.02292, 2006.10621, 2102.04074, 2108.07686, 2110.12894, 2202.06387, 2405.14005, 2408.16947, 2410.11840, 2502.18969, 2504.04338, 2506.04598, 2605.08541.

**Do repositório:** `core/svm.py:121,143`, `eval/svm.py:365-373,743`, `core/data/parasite_dataset.py`, `experiments/autoencoder/*_pcts*.yaml`, `*_ssl100_svm*.yaml`, `fila.md`, `agentes/L_method.md`, `agentes/R2.md`. Grafo consultado primeiro. Ele não indexa a chave de YAML `svm_train_pct`, e só para ela usei grep.

---

## 10. Recomendações para o plano

**R1. Três perguntas, três desenhos.** Aprendizado: forma fixa, n varia. Arquitetura: n e custo fixos, forma varia. Interação: a forma ótima em duas frações distantes.
*Rosenfeld p6; Hoffmann p5-6; SoViT p4.*

**R2. Descobrir fora do raio; B0–B3 só no fim.** Em cada orçamento, pelo menos 3 formas, de preferência 5, com o ótimo no interior da grade. Escala composta só depois de escolher a regra.
*Kricheli p2-5, p9; Hoffmann p5-6; SoViT p17; Tan p5.*

**R3. Congelar a receita como regra, não como valor.** LR, regularização e augmentation seguem uma regra escrita antes, igual para toda a curva. Regra pode depender de resolução, porque 128 e 300 px pedem augmentation diferente.
*Li p30; Ivgi p1, p6; Rosenfeld p10; Kornblith p3-4; EfficientNetV2 p5.*

**R4. Variável resposta: e=1−κ de validação. Eixo x: contagem real.** O chute vale e=1 em todo dataset, então ε0=1 sem depender de K [conta]. Reportar `acc` e `f1` junto.
*Rosenfeld p6; Alabdulmohsin 2022 p4; Schaeffer p4.*

**R5. Curva por dataset com piso, e comparar por (e_N, β_N).** Forma E+A·n^−γ, só com 25 imagens por classe ou mais. Frações abaixo disso saem do ajuste e são relatadas à parte. γ compartilhado entre as redes do dataset. Nunca comparar α, η, γ entre curvas. Acrescentar a fração de 10%, que já existe.
*Hoiem p3, p5, p11; Viering p7; Rodrigo González p5-6 (preprint).*

**R6. Ajuste honesto e escrito antes.** Mínimos quadrados não lineares com grade de inícios, sem log-linearizar com piso, perda em soma, δ de Huber ligado ao ruído. Intervalo por bootstrap hierárquico (sorteia escala, depois semente). Preencher o checklist de Li antes de rodar.
*Li p2, p7, p15, p30; Hoffmann p6, p25; Besiroglu p4-5; Ivgi p3.*

**R7. Validar por extrapolação e por família, com previsão escrita antes.** Ajustar até 50%, prever 75% e 100%. Ajustar até um tamanho, prever o seguinte. Tirar uma família inteira, porque irmãs dividem imagens (Eggs: 498 de 599 do teste do subgrupo estão no treino ou validação da base). Horizonte de no máximo 2 a 4 vezes.
*Alabdulmohsin 2022 p5-6; Rosenfeld p8; Ivgi p6; Hoiem p5; Mahmood p5; Mohr p19; R2.md.*

**R8. Ruído antes de lei.** Medir o ruído primeiro. Comparar em pares dentro do mesmo split. Meta de erro relativa ao ruído, não aos 4% da literatura. Preferir mais tamanhos a mais sementes até ter 5 tamanhos, depois sementes. Reportar SNR de cada comparação.
*Choshen p1, p4, p7; Ivgi p6; Bornschein p7; Kricheli p31.*

**R9. Orçamento em três contas, resultado em faixa.** Registrar FLOPs, parâmetros e ativações por run. Hoje FLOPs não são registrados (`fila.md`). ≤5M não limita resolução. Igualar custo antes de comparar. Entregar a faixa de formas boas, não um ponto, porque o ótimo tende a ser largo.
*Dollár p3-5, p7; Dehghani p1, p9; Radosavovic 2019 p5; SoViT Fig. 6.*

**R10. Confirmar no regime final.** Regra achada em poucas épocas ou numa fração só é hipótese. Rodar um subconjunto até o fim do treino, em mais de uma fração. Declarar se passos ou épocas ficam fixos ao subamostrar.
*Bello p6-7; SEER p8; Cole p16; Nakkiran p7.*

**R11. Separar imagens de SSL e rótulos de SVM, e uma lei por perda.** Grade n_ssl × n_svm com `svm_train_pct`: reusa encoders já treinados e custa só SVM, enfileirado depois do treino. Curvas por camada. Baseline SPiFiL sem SSL, e aleatório só como controle. Trocar a perda reabre a lei.
*Kolesnikov p5; Newell p8, p11; SimCLRv2 p6; Cherti p8; Asano p8; Kricheli p3-5.*

**R12. Leitor fixo, e perda só como covariável.** Fixar pooling, camada, dimensão da leitura e C ao longo da curva; c=1280 fixo ao variar largura. Plotar κ contra n/d antes de ajustar. Reportar `dim` e vetores de suporte. Não usar perda total como eixo: separar termos e medir correlação com κ fora da amostra. Posto só como alarme de colapso.
*Prato p4; Muthukumar p14; Nakkiran p17; Goyal p14; LCDB p5, p19; Abnar p12; Balestriero p2; Saunshi p7; Xie p8; LeJEPA p15; RankMe p5.*

---

## 11. Equações candidatas

Notação: e=1−κ na validação. n=imagens reais. n_s=imagens no SSL. n_l=rótulos no SVM. N=tamanho. C=orçamento em FLOPs. Ordem sugerida: E1, E3, E6, E7, E5.

**E1. Curva por braço (da literatura).**
e(n) = E + A·n^(−γ)
3 constantes; γ compartilhada no dataset. Ler por e_N e β_N=2A·γ·N^(−γ). *Hoiem p3; Hestness p9; Viering p7.*

**E2. Curva até o chute (adaptada: ε0=1 vem do κ).**
(e−E)/(1−e)^a = B·n^(−c)
4 constantes. Só se as frações baixas entrarem. *Alabdulmohsin 2022 p4; alternativa: envelope de Rosenfeld p6.*

**E3. Dois eixos de dados (hipótese nossa).**
e(n_s, n_l) = E + A·n_s^(−a) + B·n_l^(−b)
Variante com interação, se a soma falhar: E + (A + B·n_l^(−b))·n_s^(−a). *Forma aditiva de Rosenfeld p6 e Hoffmann p6; ninguém mediu para SSL e SVM.*

**E4. Tamanho × dados (da literatura).**
e(N, n) = E + A·N^(−α) + B·n^(−β)
5 constantes por dataset. *Hoffmann p6; Rosenfeld p6.*

**E5. Tamanho ótimo por dados (adaptada).**
e(x, n) = E + A·x^(−a) + B·x^b·n^(−c), logo x* = K·n^s, com s=c/(a+b)
Troca compute por n: é nossa, SoViT supõe dado infinito (p4). Forma empírica irmã: s(m)=2,76·m^0,57 (Hestness p9). Se não houver ótimo interno, usar **o menor modelo a até 0,01 do platô**.

**E6. Forma sob orçamento (adaptada).**
log e(ρ; C) = log e*(C) + q·(log ρ − log ρ*(C))², com ρ*(C) = K·C^s
3 constantes por orçamento e por razão de forma ρ (ex.: d/w, r/w). Com duas razões e termo cruzado, 6 constantes. *Hoffmann p5-6; interação de Tan p4.*

**E7. Regra de montagem com uma constante (da literatura).**
d ∝ s^((1−a)/2), w ∝ s^(a/2), r ∝ s^((1−a)/4), com FLOPs ∝ s
Uma constante a, em vez das duas livres de α·β²·γ²≈2. Melhor para poucos datasets. *Dollár p5; Tan p5.*
Variante [hip]: d cresce com log da resolução, d = d0 + round(log2(r/r0)), porque o campo de visão é RF(d)=6·2^d−5 (L_method §3.1). Isso quebra a forma d=α^φ de Tan.
Restrições exatas, não medidas: parâmetros ≤5M, memória dentro do teto, N_ℓ ≥ max(3·w_ℓ, k²·w_{ℓ−1}+1) (L_method §3.2).

**E8. Multiplicador do SPiFiL (controle, adaptada).**
e_R(m·n) = e_S(n). Com E e γ iguais: m = (A_R/A_S)^(1/γ)
Quantas vezes mais dado o aleatório precisa para empatar com o SPiFiL. *Hoiem p7 usa a mesma ideia: e(n) com augmentation ≈ e(1,8n) sem ela; Hernandez, via L_method §3.3.*
