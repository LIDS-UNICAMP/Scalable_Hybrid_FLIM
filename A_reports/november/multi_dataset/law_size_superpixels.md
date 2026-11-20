# Lei tamanho da imagem x superpixels (só L1 SPiFiL, sem treino)

**Resposta:** não existe lei única. O tamanho da imagem quase não muda o melhor número de superpixels (R² 0,01). Quem muda é o dataset. Em quase todos, imagem menor e poucos superpixels ganham.

Medida: L1 de 64 filtros 5x5, sem backprop, GeM + TTA + PCA-whitening, SVM oficial, 50%. 1.146 linhas, 0 erros, todo SVM convergiu. Leitura direta no teste, a pedido; escolha pela validação ao lado.

## n recomendado
Qualidade = média de κ e F1 (macro). Vale para todo tamanho, porque o tamanho quase não move n. Curva por dataset: quadrática em log n + tamanho.

| dataset | n (val) [IC 95%] | n (teste) | só κ / só F1 | ganho de acertar n: κ / F1 | R² κ / F1 | melhor tam, n (val) | κ / F1 teste | melhor pelo teste: κ |
|---|---|---|---|---|---|---|---|---|
| Eggs | 12 [12; 12] | 12 | 12 / 12 | 0,174 / 0,185 | 0,54 / 0,48 | 64, 12 | 0,641 / 0,685 | 0,641 |
| Eggs_subgroup | 12 [12; 12] | 12 | 12 / 12 | 0,128 / 0,156 | 0,55 / 0,49 | 128, 25 | 0,695 / 0,720 | 0,695 |
| Larvae | 60 [45; 82] | 67 | 68 / 65 | 0,336 / 0,184 | 0,27 / 0,26 | 128, 100 | 0,616 / 0,808 | 0,616 |
| Larvae_subgroup | 106 [55; 186] | 127 | 128 / 126 | 0,441 / 0,251 | 0,40 / 0,40 | 128, 100 | 0,676 / 0,838 | 0,688 |
| Protozoan (1 split) | 12 [12; 20] | 12 | 12 / 12 | 0,220 / 0,092 | 0,78 / 0,77 | 64, 12 | 0,407 / 0,340 | 0,407 |
| Protozoan_subgroup | 12 [12; 16] | 12 | 12 / 17 | 0,077 / 0,135 | 0,64 / 0,26 | 64, 25 | 0,674 / 0,648 | 0,674 |
| aid | 12 [12; 400] | 36 | 35 / 37 | 0,022 / 0,021 | 0,41 / 0,34 | 128, 200 | 0,586 / 0,601 | 0,586 |
| flowers | 12 [12; 400] | 12 | 12 / 12 | 0,038 / 0,030 | 0,35 / 0,36 | 128, 12 | 0,529 / 0,622 | 0,550 |
| helminth_eggs | 12 [12; 12] | 12 | 12 / 12 | 0,039 / 0,035 | 0,78 / 0,60 | 64, 25 | 0,880 / 0,880 | 0,890 |
| helminth_larvae | 394 [290; 627] | 313 | 313 / 313 | 0,084 / 0,042 | 0,38 / 0,38 | 256, 400 (ampliado) | 0,935 / 0,968 | 0,936 |
| protozoan_cysts | 75 [56; 142] | 67 | 67 / 67 | 0,056 / 0,073 | 0,71 / 0,58 | 128, 200 | 0,824 / 0,800 | 0,840 |
| remote_sensing | 12 [12; 765] | 12 | 12 / 12 | 0,054 / 0,069 | 0,09 / 0,10 | 256, 12 | 0,425 / 0,421 | 0,449 |

12 é a ponta da grade: nesses datasets o melhor n pode ser menor ainda. A "melhor célula" só olha 64, 128, 256 e 640, os tamanhos com 3 splits.

**κ e F1 concordam:** discordam em 1 degrau ou mais em só 2 de 95 pares (dataset, tamanho): Protozoan_subgroup a 448 e aid a 640. Só o tamanho do efeito muda: em Larvae errar n custa quase o dobro em κ.

## Equação (conjunto, x = log2 n, s = log2(S/200), a = base do dataset)
- κ = a + 0,0396x − 0,0035x² + 0,0007xs − 0,0275s − 0,0032s². R² 0,82; sem contar a base do dataset, 0,13.
- F1 = a + 0,0150x − 0,0016x² + 0,0009xs − 0,0197s − 0,0031s². R² 0,89; sem a base, 0,12.
- log2 n* = 4,01 + 0,18 log2 S (κ teste): R² 0,013, inclinação [−0,02; 0,37]. Na validação: 4,86 + 0,07 log2 S, R² 0,001, [−0,12; 0,29].

Pixels por superpixel fixo pediria inclinação 2, e o intervalo fica longe. Logo, n não cresce com a imagem. O dataset explica 52 a 57% de n*.

**Tamanho pesa mais:** κ médio dos 12 é 0,643 a 64 px, 0,645 a 128 e 0,569 a 640. Lembre: isso é só uma camada, que enxerga 7 px.

**Dataset deixado de fora:** a regra do conjunto acerta n a 1 degrau em só 23% dos casos e perde 0,049 de qualidade, igual a um n fixo (0,050). Ela não generaliza.

**Regra prática:** 2 classes, n = 100; resto, n = 12. Perde 0,022, contra 0,029 de n = 50 fixo e 0,050 de n = 12 fixo. Foi montada depois de ver os dados e não passou por dataset deixado de fora. protozoan_cysts é o pior caso (0,054).

**O que o SPiFiL vale:** no n recomendado, contra filtros aleatórios, +0,058 de κ [0,005; 0,122] na média dos 12 (amostra: n = 100 aleatório nos 8 tamanhos, só split 1). Ganha muito em Protozoan (+0,28), Larvae (+0,23) e Protozoan_subgroup (+0,17). Perde um pouco em aid (−0,04) e helminth_eggs (−0,03).

**Contaminação:** escolher pelo teste infla κ em 0,007 na média, até 0,024 (remote_sensing). A coluna de validação é a honesta.

## O que não ficou provado
- **Por que 2 classes pedem mais n:** com n = 12 ou 25 o banco tem só 12 a 50 filtros, não 64. Parte do ganho é largura, não superpixel. Tirar esses bancos muda pouco a inclinação (0,15), mas não separa os dois efeitos por dataset.
- **Objeto/quadro:** R² 0,01 com 7 datasets. Não explica n, mas 7 pontos e medida sem máscara não bastam para descartar.
- **Cobertura:** 3 splits só em 64, 128, 256 e 640. Em 96, 200, 320, 448, só split 1. Protozoan só tem split 1, sem 320. 896 e 1280 de Larvae não rodaram.
- **n abaixo de 12** não foi medido, e n = 800 só existe em split 1 até 320 px, fora da conta. n = 1600 só em 3 datasets.
- **Vale para o modelo treinado?** Aqui não há backprop nem camadas fundas: "imagem pequena ganha" pode virar com L3 e L4.
- **IC largos:** aid, flowers e remote_sensing têm curva quase plana, e neles n importa pouco.

Reprodução: `experiments/autoencoder/disc_md_l1fast*.yaml`, `eval/l1_fast.py`, CSV em `results/svm_disc_md_l1fast_*.csv`, pesos em `/exps/moliveira/spifil_growth/disc_md_l1fast/`.

## Curvas por dataset (07/11)

`superpixels_regressao_kappa.png` e `superpixels_regressao_f1.png`: validação, quadrática em log n mais tamanho, um painel por dataset, pontos sem o efeito do tamanho. Esse ajuste não tem o termo n × tamanho e usa κ ou F1, não a média dos dois; por isso difere um pouco da tabela acima (Larvae 65, Larvae_subgroup 97, helminth_larvae 414, protozoan_cysts 84).

aid sai com n = 1600, mas é artefato: a curva é quase plana, 0,57 a 0,585, com R² 0,30. flowers (R² 0,18) e remote_sensing (0,09) também não têm curva clara. Quem tem curva nítida: Eggs, Eggs_subgroup, Protozoan, Protozoan_subgroup, helminth_eggs, helminth_larvae, protozoan_cysts.

helminth_eggs, helminth_larvae e protozoan_cysts rodaram com remoção de filtros repetidos ligada (`dedup=1`); as outras 9 bases, desligada. Não comparar o nível de κ entre esses dois grupos.
