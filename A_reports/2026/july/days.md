# Meeting 28 julho.
- Carta resposta disponivel para avaliacao
- Correcoes feitas no artigo sibgrapi 2026[AZUL]

----
- experimento com relu -> Softmax para 5% e 75% feito!
- experimento com softplus -> Softmax para 5% e 75% feito!
- compreensao do problema relu com e sem backprop.
----



# Exploracao Modelo FLIM (15 de julho de 2026)

- Mateus Oliveira task: classificacao.
- Foco no journal:
    - Para isso precisamos rodar os resultados em varios datasets.
    - Auto machine learning para achar a melhor a arquitetura
    - Avaliacao durante o treinamento MLP + Sigmoid:
        - Freeze
        - Unfreeze
    - o que aconteceria se ela fosse um sigmoid e nao RELU?
    - possibilidade de usar GeLu

## Datasets a serem testados

Sugestao lista de datsets:
* https://unidata.pro/blog/best-ml-classification-datasets/

### Datasets selecionados
1. flower
2. food101
3. face expression 2013
4. parasito
5. dermotologico ISIC <<< 
6. Coqueiros
7. remoting sensing classification
8. medMNIST

### Outra possibilidade de datasets
NWPU-RESISC45, AID**, EuroSAT, and the SATIN metadataset (Explorar ver qual vai entrar nos experimentos.)

## Espaco de cores

O Leornado conseguiu conseguiu um espaco de cores melhor que o LAB, chjamado OK-LAB, avaliar no FLIM.
- ta na lib IFT?

## Bases

[hawk] Rodar base 5% e 75%, solicitacao de uma base pequena e uma base grande. relacionado a (sigmoid experiment)

dataset AID enviado no grupo.
* https://captain-whu.github.io/AID/

## Frases Soltas

- Adaptacao desses modelos pre-treinados para outros cases.
- Decoder adaptivo, dynamic tree
- Modelo weak supervised.

Modelo montado. (FLIM init)
entrada (3×H×W, LAB)
   │
   ├─ Encoder FLIM (3 blocos conv)         ← inicializado com pesos FLIM pré-treinados
   │     conv1: Conv2d(3→24, 5×5) + ReLU + MaxPool(3×3, s2)
   │     conv2: Conv2d(24→32, 5×5) + ReLU + MaxPool(3×3, s2)
   │     conv3: Conv2d(32→48, 5×5) + ReLU + MaxPool(3×3, s2)
   │  → mapa de features 48 canais
   │
   └─ TwoLayerSigmoidHead
         AdaptiveAvgPool2d(1) → flatten (48)
         Linear(48 → 24) → Sigmoid
         Linear(24 → C)  → Softmax
   → probabilidades por classe (C)