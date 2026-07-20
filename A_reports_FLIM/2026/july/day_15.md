# Consagrar o FLIM

- Mateus Oliveira task: classificacao.
- Foco no journal:
    - Para isso precisamos rodar os resultados em varios datasets.
    - Auto machine learning para achar a melhor a arquitetura
    - Avaliacao durante o treinamento MLP + Sigmoid:
        - Sem backprop
        - Com backprop
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
5. patologia Felipe
6. dermotologico ISIC
7. Coqueiros
8. remoting sensing classification
9. medMNIST

### Outra possibilidade de datasets
NWPU-RESISC45, AID, EuroSAT, and the SATIN metadataset (Explorar ver qual vai entrar nos experimentos.)

## Espaco de cores

O Leornado conseguiu conseguiu um espaco de cores melhor que o LAB, chjamado OK-LAB, avaliar no FLIM.
- ta na lib IFT?

## Bases

[hawk] Rodar base 5% e 75%, solicitacao de uma base pequena e uma base grande.

dataset AID enviado no grupo.
* https://captain-whu.github.io/AID/

## Frases Soltas

- Adaptacao desses modelos pre-treinados para outros cases.
- Decoder adaptivo, dynamic tree
- Modelo weak supervised.
