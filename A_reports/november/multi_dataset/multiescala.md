# Filtros multiescala contra aleatório, sem backprop

Pergunta: filtros tirados de Gaussiana, DoG e LoG ganham do filtro sorteado? Mesma rede, mesma largura real, mesma normalização, mesma leitura. Só muda o conteúdo do kernel. Sem rótulo, máscara ou bbox na seleção; o teste não é lido.

Auditoria: `multiescala_auditoria.md`. Código: `flim/scale_space.py`, `eval/spf_grade.py` (`amostra.multiescala`), `eval/ms_tabela.py`. YAMLs: `experiments/autoencoder/disc_md_ms_*`. CSVs: `results/svm_disc_md_ms_*`; registro das escalas em `results/ms_escala/`.

## Critérios escritos antes de ver os dados

Escritos em 07/11, antes de a primeira linha `ms` existir.

- **Δκ:** κ do método menos κ do melhor aleatório (Xavier ou `random_z`), na mesma base, split, sorteio, largura e leitura. Média dos 5 sorteios em cada split; intervalo t de 95% pareado pelos 3 splits.
- **Leitura que decide:** 3 camadas concatenadas (`L3 cat`, 192-d), fixada para todos. `L3 L` e `L1 L` são só consulta.
- **Triagem nas 5 primeiras bases:** Δ > 0 em pelo menos 3 de 5. Não prova robustez.
- **Bom, com 12 bases:** Δ ≥ +0,03 com intervalo acima de 0 em pelo menos 7 de 12, e nenhuma base com Δ < −0,03.
- **Aleatório congelado antes:** roda primeiro, e `eval.ms_tabela congela` grava o hash dos CSVs antes de o método começar. A tabela confere esse hash a cada volta.
- **Decisão só por validação.** Não há escolha de célula por base: largura, profundidade, leitura e kernel são fixos.

Cuidados da auditoria: as 12 bases não são independentes (3 famílias de parasitos, mais aid, flowers e remote_sensing), e os 3 splits são reamostragens, não folds. Com 3 splits o intervalo é largo; por isso "7 de 12" é exigente de propósito.

## Protocolo fixo

64 filtros, 3 camadas, kernel 5, pool 2, 200 px, 50 superpixels, 50% dos dados, 500 imagens de treino sorteadas sem olhar classe, sorteios 42 a 46, 3 splits. Leitura GeM + whitening + SVM. Modelo na GPU, embedding copiado para a CPU, SVM na CPU, 2 threads por processo.

Rodada A, 5 bases (flowers, Protozoan_subgroup, Eggs_subgroup, Larvae_subgroup, aid), depois as outras 7:

| Braço | Pontos | Conteúdo do kernel |
|---|---|---|
| Xavier, `random_z` | não há | direção sorteada |
| geodésico cru (`crus@geodesic`) | centro geodésico | patch cru |
| S1 (`crus@s1`) | centro geodésico | patch da imagem suavizada, σ de 1,6 a 3,2 sorteado por ponto |
| S2 (`crus@s2`) | centro geodésico | patch do mapa DoG, mesmos σ |
| S3 (`crus@s3`) | extremos DoG | patch cru |
| S4 (`crus@s4`) | os de S3 | patch do mapa DoG na oitava e nível do extremo |
| S5 (`crus@s5`) | os de S3 | região de lado 5·σ/1,6, reduzida a 5×5 |
| S6 (`trio@s6`) | centro geodésico | regiões de 5, 10 e 20 px da mesma posição, as três entram juntas |
| S7 (`crus@s7`) | extremos LoG | patch cru |
| S8 (`quota@s8`) | geodésico, SIFT filtrado, S3, S7, S5, S6 | mesma cota de cada um, 64 no total |

Rodada B, patch cru em tudo; só muda onde o patch nasce: geodésico, S3 (DoG) e S7 (LoG) vêm da rodada A, e entram centroide (`crus@centroid`), SIFT filtrado (`crus@sift_nms`), objeto sem rótulo (`crus@obj`) e FPS (`fps@geodesic`).

Escolhas que o pedido deixou abertas:

- **Espaço de escalas:** σ₀ 1,6, 3 níveis por oitava, 4 oitavas a 200 px, sem a oitava −1. Contraste 0,02/3 (o do SIFT da grade), razão de bordas 10. Extremos em até 50 por imagem, com o mesmo NMS de `sift_nms`, para ter tantos candidatos quanto superpixels.
- **LoG:** limiar do DoG dividido por (k − 1), porque D ≈ (k − 1)·σ²∇²L.
- **Normalização:** cada patch é padronizado no próprio mapa (média e desvio das sementes medoide nesse mapa) e depois entra na mesma dobra de todos. Sem isso o DoG, de média perto de 0, viraria uma constante. Para patch cru isso não muda nada.
- **Camadas 2 e 3:** a pirâmide é a do mapa daquela camada, com σ nos pixels do mapa. A oitava é cortada ao que o mapa comporta.

## Retomar

Sessões `disc_md_ms_q` (fila) e `disc_md_ms_tab` (tabela), que fecham sozinhas; logs `/exps/moliveira/queues/logs/ms.*`; `ms.status` diz o que terminou. Relançar um YAML retoma: cada job pula as células com linha `ok`. Refazer a tabela: `python -m eval.ms_tabela`.

## Resultados

<!-- gerado:inicio -->
Atualizada 07/11 22:29 UTC. Só validação; o teste não é lido. 6675 leituras prontas.

- `ctl`: congelado em 07/11 16:53 UTC, 15 CSVs, hash confere.
- `ctl2`: congelado em 07/11 22:25 UTC, 21 CSVs, hash confere.

### Aleatório

| base | aleatório | κ | acurácia | acurácia balanceada | macro-F1 | desvio entre sementes | leituras | init s | inferência s | GPU MiB | RAM MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Eggs | random_z | 0,571 | 0,837 | 0,632 | 0,645 | 0,012 | 15 | 9,1 | 8,4 | 1134 | 3732 |
| Eggs | xavier | 0,582 | 0,840 | 0,644 | 0,655 | 0,030 | 15 | 8,8 | 8,3 | 1134 | 3731 |
| Eggs_subgroup | random_z | 0,631 | 0,795 | 0,656 | 0,662 | 0,023 | 15 | 9,9 | 3,0 | 1134 | 2952 |
| Eggs_subgroup | xavier | 0,656 | 0,809 | 0,670 | 0,680 | 0,037 | 15 | 10,3 | 3,1 | 1134 | 2951 |
| Larvae | random_z | 0,321 | 0,817 | 0,667 | 0,660 | 0,075 | 15 | 7,3 | 1,0 | 1134 | 3081 |
| Larvae | xavier | 0,370 | 0,830 | 0,690 | 0,685 | 0,073 | 15 | 8,6 | 1,1 | 1134 | 3081 |
| Larvae_subgroup | random_z | 0,319 | 0,784 | 0,664 | 0,659 | 0,080 | 15 | 5,7 | 0,6 | 1022 | 2178 |
| Larvae_subgroup | xavier | 0,312 | 0,783 | 0,657 | 0,655 | 0,132 | 15 | 6,5 | 0,6 | 1022 | 2177 |
| Protozoan | random_z | 0,481 | 0,829 | 0,444 | 0,500 | 0,041 | 15 | 11,8 | 27,9 | 1134 | 5434 |
| Protozoan | xavier | 0,508 | 0,834 | 0,468 | 0,525 | 0,030 | 15 | 11,1 | 26,9 | 1134 | 5389 |
| Protozoan_subgroup | random_z | 0,634 | 0,771 | 0,571 | 0,578 | 0,020 | 15 | 8,2 | 5,3 | 1134 | 3317 |
| Protozoan_subgroup | xavier | 0,652 | 0,783 | 0,592 | 0,600 | 0,027 | 15 | 10,2 | 5,7 | 1134 | 3312 |
| aid | random_z | 0,626 | 0,639 | 0,641 | 0,637 | 0,012 | 15 | 8,8 | 7,1 | 1134 | 4547 |
| aid | xavier | 0,627 | 0,640 | 0,642 | 0,638 | 0,009 | 15 | 7,7 | 6,5 | 1134 | 4545 |
| flowers | random_z | 0,532 | 0,627 | 0,628 | 0,628 | 0,019 | 15 | 10,4 | 2,2 | 1134 | 2791 |
| flowers | xavier | 0,536 | 0,630 | 0,632 | 0,631 | 0,019 | 15 | 11,3 | 2,1 | 1134 | 2790 |
| helminth_eggs | random_z | 0,862 | 0,925 | 0,871 | 0,882 | 0,014 | 15 | 7,4 | 3,0 | 1134 | 2965 |
| helminth_eggs | xavier | 0,864 | 0,925 | 0,873 | 0,880 | 0,019 | 15 | 6,2 | 2,8 | 1134 | 2963 |
| helminth_larvae | random_z | 0,923 | 0,983 | 0,957 | 0,962 | 0,027 | 15 | 11,3 | 2,6 | 1134 | 2770 |
| helminth_larvae | xavier | 0,915 | 0,981 | 0,956 | 0,957 | 0,033 | 15 | 11,3 | 2,6 | 1134 | 2769 |
| protozoan_cysts | random_z | 0,798 | 0,878 | 0,816 | 0,808 | 0,012 | 15 | 9,9 | 6,4 | 1134 | 3349 |
| protozoan_cysts | xavier | 0,802 | 0,880 | 0,813 | 0,803 | 0,014 | 15 | 11,4 | 6,9 | 1134 | 3345 |
| remote_sensing | random_z | 0,454 | 0,506 | 0,437 | 0,436 | 0,021 | 15 | 5,3 | 0,6 | 1066 | 3715 |
| remote_sensing | xavier | 0,470 | 0,521 | 0,447 | 0,446 | 0,029 | 15 | 4,8 | 0,6 | 1066 | 3715 |

### Matriz método × dataset: L3 `cat`, leitura que decide

Δκ contra o melhor aleatório. `*` passa (Δ ≥ +0,03 com intervalo acima de 0), `!` regressão (Δ < −0,03), `(n)` splits medidos quando faltam. As contagens só usam base com os 3 splits.

| método | flowers | Protozoan_subgroup | Eggs_subgroup | Larvae_subgroup | aid | bases | Δ > 0 | boas | regressões | média | mediana | variância | pior | triagem 3/5 | 7/12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geodésico cru | +0,001 | +0,086* | +0,025 | +0,265* | -0,049! | 5 | 4 | 2 | 1 | +0,066 | +0,025 | 0,0148 | -0,049 | passa | aberto |
| S1 Gauss + geodésico | -0,008 | +0,090* | +0,013 | +0,263* | -0,056! | 5 | 3 | 2 | 1 | +0,060 | +0,013 | 0,0156 | -0,056 | passa | aberto |
| S2 DoG + geodésico | +0,008 | +0,031* | +0,009 | +0,167 | -0,061! | 5 | 4 | 1 | 1 | +0,031 | +0,009 | 0,0070 | -0,061 | passa | aberto |
| S3 extremo DoG + cru | +0,009 | +0,087* | +0,023 | +0,213* | -0,059! | 5 | 4 | 2 | 1 | +0,055 | +0,023 | 0,0105 | -0,059 | passa | aberto |
| S4 extremo DoG + DoG | -0,010 | -0,034! | -0,048! | +0,084 | -0,111! | 5 | 1 | 0 | 3 | -0,024 | -0,034 | 0,0050 | -0,111 | não | aberto |
| S5 região ∝ escala | +0,016 | +0,111* | +0,022 | +0,290* | -0,039! | 5 | 4 | 2 | 1 | +0,080 | +0,022 | 0,0167 | -0,039 | passa | aberto |
| S6 multiescala cru | +0,008 | +0,074* | +0,021 | +0,257* | -0,048! | 5 | 4 | 2 | 1 | +0,062 | +0,021 | 0,0137 | -0,048 | passa | aberto |
| S7 extremo LoG + cru | +0,002 | +0,085* | +0,030 | +0,275* | -0,056! | 5 | 4 | 2 | 1 | +0,067 | +0,030 | 0,0161 | -0,056 | passa | aberto |
| S8 fusão | +0,013 | +0,094* | +0,029 | +0,294* | -0,043! | 5 | 4 | 2 | 1 | +0,077 | +0,029 | 0,0170 | -0,043 | passa | aberto |
| centroide | +0,008 | +0,097* | +0,030 | +0,289* | -0,041! | 5 | 4 | 2 | 1 | +0,077 | +0,030 | 0,0166 | -0,041 | passa | aberto |
| SIFT filtrado | -0,001 | +0,088* | +0,012 | +0,241* | -0,039! | 5 | 3 | 2 | 1 | +0,060 | +0,012 | 0,0123 | -0,039 | passa | aberto |
| objeto sem rótulo | +0,020 | +0,083* | +0,029 | +0,333* | -0,039! | 5 | 4 | 2 | 1 | +0,085 | +0,029 | 0,0210 | -0,039 | passa | aberto |
| FPS | +0,003 | +0,087* | +0,060* | +0,269* | +0,015 | 5 | 5 | 3 | 0 | +0,087 | +0,060 | 0,0115 | +0,003 | passa | aberto |

### Tabela pareada

| base | método | κ | acurácia | acurácia balanceada | macro-F1 | desvio entre sementes | leituras | init s | inferência s | GPU MiB | RAM MiB | melhor aleatório | κ dele | Δκ | IC 95% pareado | splits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flowers | geodésico cru | 0,537 | 0,630 | 0,631 | 0,629 | 0,014 | 15 | 77,1 | 3,9 | 1134 | 2690 | xavier | 0,536 | +0,001 | [-0,075; +0,077] | 3 |
| flowers | S1 Gauss + geodésico | 0,528 | 0,624 | 0,624 | 0,623 | 0,020 | 15 | 56,7 | 3,8 | 1461 | 2858 | xavier | 0,536 | -0,008 | [-0,109; +0,094] | 3 |
| flowers | S2 DoG + geodésico | 0,543 | 0,636 | 0,637 | 0,636 | 0,018 | 15 | 49,3 | 3,6 | 1619 | 4145 | xavier | 0,536 | +0,008 | [-0,072; +0,087] | 3 |
| flowers | S3 extremo DoG + cru | 0,545 | 0,637 | 0,638 | 0,636 | 0,023 | 15 | 60,3 | 3,6 | 1215 | 2702 | xavier | 0,536 | +0,009 | [-0,104; +0,121] | 3 |
| flowers | S4 extremo DoG + DoG | 0,526 | 0,622 | 0,623 | 0,622 | 0,024 | 15 | 58,0 | 3,7 | 1703 | 2858 | xavier | 0,536 | -0,010 | [-0,065; +0,045] | 3 |
| flowers | S5 região ∝ escala | 0,552 | 0,642 | 0,644 | 0,643 | 0,025 | 15 | 51,7 | 3,3 | 1264 | 4145 | xavier | 0,536 | +0,016 | [-0,060; +0,091] | 3 |
| flowers | S6 multiescala cru | 0,544 | 0,637 | 0,637 | 0,636 | 0,021 | 15 | 41,0 | 4,2 | 1254 | 2702 | xavier | 0,536 | +0,008 | [-0,020; +0,037] | 3 |
| flowers | S7 extremo LoG + cru | 0,538 | 0,632 | 0,632 | 0,631 | 0,020 | 15 | 36,3 | 3,5 | 1216 | 2858 | xavier | 0,536 | +0,002 | [-0,097; +0,101] | 3 |
| flowers | S8 fusão | 0,549 | 0,640 | 0,641 | 0,640 | 0,020 | 15 | 109,5 | 3,7 | 1329 | 4548 | xavier | 0,536 | +0,013 | [-0,060; +0,087] | 3 |
| flowers | centroide | 0,544 | 0,636 | 0,638 | 0,637 | 0,012 | 15 | 80,0 | 2,7 | 1134 | 2492 | xavier | 0,536 | +0,008 | [-0,065; +0,081] | 3 |
| flowers | SIFT filtrado | 0,535 | 0,629 | 0,629 | 0,628 | 0,020 | 15 | 85,6 | 2,4 | 1134 | 2492 | xavier | 0,536 | -0,001 | [-0,078; +0,075] | 3 |
| flowers | objeto sem rótulo | 0,556 | 0,646 | 0,646 | 0,645 | 0,019 | 15 | 60,3 | 2,4 | 1134 | 2492 | xavier | 0,536 | +0,020 | [-0,042; +0,083] | 3 |
| flowers | FPS | 0,539 | 0,632 | 0,634 | 0,633 | 0,023 | 15 | 86,1 | 2,6 | 1134 | 2498 | xavier | 0,536 | +0,003 | [-0,076; +0,082] | 3 |
| Protozoan_subgroup | geodésico cru | 0,738 | 0,838 | 0,620 | 0,638 | 0,019 | 15 | 45,9 | 9,9 | 1134 | 2938 | xavier | 0,652 | +0,086 | [+0,053; +0,119] | 3 |
| Protozoan_subgroup | S1 Gauss + geodésico | 0,742 | 0,841 | 0,617 | 0,637 | 0,017 | 15 | 53,9 | 11,4 | 1461 | 3151 | xavier | 0,652 | +0,090 | [+0,051; +0,130] | 3 |
| Protozoan_subgroup | S2 DoG + geodésico | 0,683 | 0,803 | 0,611 | 0,618 | 0,014 | 15 | 49,4 | 10,8 | 1619 | 3930 | xavier | 0,652 | +0,031 | [+0,001; +0,061] | 3 |
| Protozoan_subgroup | S3 extremo DoG + cru | 0,739 | 0,837 | 0,648 | 0,660 | 0,019 | 15 | 29,1 | 9,9 | 1209 | 2938 | xavier | 0,652 | +0,087 | [+0,074; +0,099] | 3 |
| Protozoan_subgroup | S4 extremo DoG + DoG | 0,618 | 0,767 | 0,549 | 0,563 | 0,021 | 15 | 50,7 | 11,1 | 1697 | 3151 | xavier | 0,652 | -0,034 | [-0,062; -0,005] | 3 |
| Protozoan_subgroup | S5 região ∝ escala | 0,763 | 0,852 | 0,660 | 0,671 | 0,021 | 15 | 49,3 | 10,7 | 1258 | 3930 | xavier | 0,652 | +0,111 | [+0,088; +0,133] | 3 |
| Protozoan_subgroup | S6 multiescala cru | 0,726 | 0,830 | 0,621 | 0,636 | 0,026 | 15 | 29,5 | 9,4 | 1254 | 2938 | xavier | 0,652 | +0,074 | [+0,041; +0,107] | 3 |
| Protozoan_subgroup | S7 extremo LoG + cru | 0,737 | 0,837 | 0,635 | 0,649 | 0,021 | 15 | 30,9 | 10,6 | 1209 | 3151 | xavier | 0,652 | +0,085 | [+0,049; +0,121] | 3 |
| Protozoan_subgroup | S8 fusão | 0,746 | 0,842 | 0,661 | 0,675 | 0,018 | 15 | 75,8 | 11,1 | 1304 | 4185 | xavier | 0,652 | +0,094 | [+0,072; +0,115] | 3 |
| Protozoan_subgroup | centroide | 0,750 | 0,845 | 0,618 | 0,635 | 0,023 | 15 | 52,1 | 7,7 | 1134 | 2753 | xavier | 0,652 | +0,097 | [+0,046; +0,148] | 3 |
| Protozoan_subgroup | SIFT filtrado | 0,740 | 0,838 | 0,649 | 0,661 | 0,018 | 15 | 58,4 | 7,5 | 1134 | 2753 | xavier | 0,652 | +0,088 | [+0,054; +0,121] | 3 |
| Protozoan_subgroup | objeto sem rótulo | 0,735 | 0,835 | 0,633 | 0,645 | 0,019 | 15 | 55,7 | 7,9 | 1134 | 2753 | xavier | 0,652 | +0,083 | [+0,069; +0,097] | 3 |
| Protozoan_subgroup | FPS | 0,739 | 0,837 | 0,634 | 0,644 | 0,032 | 15 | 81,0 | 7,6 | 1134 | 2757 | xavier | 0,652 | +0,087 | [+0,048; +0,126] | 3 |
| Eggs_subgroup | geodésico cru | 0,682 | 0,825 | 0,669 | 0,666 | 0,022 | 15 | 73,1 | 5,3 | 1134 | 2731 | xavier | 0,656 | +0,025 | [+0,004; +0,047] | 3 |
| Eggs_subgroup | S1 Gauss + geodésico | 0,669 | 0,820 | 0,644 | 0,648 | 0,012 | 15 | 64,7 | 4,4 | 1461 | 2912 | xavier | 0,656 | +0,013 | [-0,024; +0,050] | 3 |
| Eggs_subgroup | S2 DoG + geodésico | 0,665 | 0,816 | 0,661 | 0,661 | 0,026 | 15 | 69,9 | 4,4 | 1619 | 3749 | xavier | 0,656 | +0,009 | [-0,033; +0,050] | 3 |
| Eggs_subgroup | S3 extremo DoG + cru | 0,680 | 0,824 | 0,662 | 0,666 | 0,019 | 15 | 41,9 | 4,9 | 1208 | 2731 | xavier | 0,656 | +0,023 | [-0,030; +0,076] | 3 |
| Eggs_subgroup | S4 extremo DoG + DoG | 0,609 | 0,787 | 0,593 | 0,603 | 0,024 | 15 | 50,0 | 4,5 | 1697 | 2912 | xavier | 0,656 | -0,048 | [-0,094; -0,002] | 3 |
| Eggs_subgroup | S5 região ∝ escala | 0,678 | 0,822 | 0,680 | 0,678 | 0,020 | 15 | 62,6 | 4,6 | 1258 | 3749 | xavier | 0,656 | +0,022 | [-0,016; +0,060] | 3 |
| Eggs_subgroup | S6 multiescala cru | 0,677 | 0,821 | 0,666 | 0,665 | 0,023 | 15 | 40,4 | 5,3 | 1254 | 2731 | xavier | 0,656 | +0,021 | [-0,009; +0,051] | 3 |
| Eggs_subgroup | S7 extremo LoG + cru | 0,687 | 0,828 | 0,674 | 0,673 | 0,023 | 15 | 40,0 | 4,4 | 1210 | 2912 | xavier | 0,656 | +0,030 | [-0,008; +0,069] | 3 |
| Eggs_subgroup | S8 fusão | 0,686 | 0,825 | 0,686 | 0,676 | 0,021 | 15 | 133,3 | 5,1 | 1306 | 4015 | xavier | 0,656 | +0,029 | [+0,005; +0,054] | 3 |
| Eggs_subgroup | centroide | 0,686 | 0,826 | 0,676 | 0,668 | 0,021 | 15 | 42,8 | 3,7 | 1134 | 2551 | xavier | 0,656 | +0,030 | [+0,004; +0,055] | 3 |
| Eggs_subgroup | SIFT filtrado | 0,668 | 0,816 | 0,668 | 0,663 | 0,019 | 15 | 46,3 | 3,9 | 1134 | 2551 | xavier | 0,656 | +0,012 | [-0,027; +0,050] | 3 |
| Eggs_subgroup | objeto sem rótulo | 0,685 | 0,826 | 0,682 | 0,680 | 0,020 | 15 | 33,3 | 4,0 | 1134 | 2551 | xavier | 0,656 | +0,029 | [-0,008; +0,065] | 3 |
| Eggs_subgroup | FPS | 0,716 | 0,843 | 0,721 | 0,722 | 0,021 | 15 | 53,1 | 4,2 | 1134 | 2557 | xavier | 0,656 | +0,060 | [+0,034; +0,085] | 3 |
| Larvae_subgroup | geodésico cru | 0,585 | 0,870 | 0,796 | 0,792 | 0,068 | 15 | 36,2 | 1,2 | 1022 | 2028 | random_z | 0,319 | +0,265 | [+0,178; +0,353] | 3 |
| Larvae_subgroup | S1 Gauss + geodésico | 0,582 | 0,868 | 0,796 | 0,791 | 0,081 | 15 | 40,7 | 0,9 | 1349 | 2216 | random_z | 0,319 | +0,263 | [+0,099; +0,427] | 3 |
| Larvae_subgroup | S2 DoG + geodésico | 0,486 | 0,842 | 0,746 | 0,743 | 0,071 | 15 | 29,3 | 0,9 | 1507 | 2485 | random_z | 0,319 | +0,167 | [-0,031; +0,365] | 3 |
| Larvae_subgroup | S3 extremo DoG + cru | 0,533 | 0,853 | 0,771 | 0,766 | 0,077 | 15 | 17,9 | 1,1 | 1094 | 2028 | random_z | 0,319 | +0,213 | [+0,030; +0,396] | 3 |
| Larvae_subgroup | S4 extremo DoG + DoG | 0,404 | 0,818 | 0,702 | 0,701 | 0,102 | 15 | 22,1 | 0,8 | 1583 | 2216 | random_z | 0,319 | +0,084 | [-0,112; +0,280] | 3 |
| Larvae_subgroup | S5 região ∝ escala | 0,609 | 0,873 | 0,820 | 0,804 | 0,069 | 15 | 24,4 | 0,9 | 1144 | 2485 | random_z | 0,319 | +0,290 | [+0,049; +0,531] | 3 |
| Larvae_subgroup | S6 multiescala cru | 0,576 | 0,868 | 0,791 | 0,788 | 0,076 | 15 | 25,8 | 1,0 | 1146 | 2057 | random_z | 0,319 | +0,257 | [+0,060; +0,454] | 3 |
| Larvae_subgroup | S7 extremo LoG + cru | 0,594 | 0,873 | 0,806 | 0,797 | 0,077 | 15 | 14,9 | 0,8 | 1095 | 2216 | random_z | 0,319 | +0,275 | [+0,086; +0,464] | 3 |
| Larvae_subgroup | S8 fusão | 0,613 | 0,881 | 0,804 | 0,806 | 0,073 | 15 | 64,6 | 0,9 | 1188 | 2584 | random_z | 0,319 | +0,294 | [+0,114; +0,474] | 3 |
| Larvae_subgroup | centroide | 0,608 | 0,874 | 0,816 | 0,804 | 0,074 | 15 | 45,4 | 0,9 | 1022 | 1878 | random_z | 0,319 | +0,289 | [+0,143; +0,434] | 3 |
| Larvae_subgroup | SIFT filtrado | 0,560 | 0,859 | 0,787 | 0,780 | 0,081 | 15 | 42,1 | 0,8 | 1022 | 1878 | random_z | 0,319 | +0,241 | [+0,033; +0,448] | 3 |
| Larvae_subgroup | objeto sem rótulo | 0,652 | 0,891 | 0,826 | 0,826 | 0,066 | 15 | 36,8 | 0,7 | 1022 | 1890 | random_z | 0,319 | +0,333 | [+0,044; +0,621] | 3 |
| Larvae_subgroup | FPS | 0,588 | 0,869 | 0,800 | 0,794 | 0,064 | 15 | 51,2 | 0,7 | 1022 | 1892 | random_z | 0,319 | +0,269 | [+0,079; +0,458] | 3 |
| aid | geodésico cru | 0,578 | 0,593 | 0,598 | 0,594 | 0,011 | 15 | 84,7 | 13,2 | 1134 | 3117 | xavier | 0,627 | -0,049 | [-0,055; -0,042] | 3 |
| aid | S1 Gauss + geodésico | 0,571 | 0,585 | 0,591 | 0,585 | 0,013 | 15 | 45,2 | 12,1 | 1461 | 3304 | xavier | 0,627 | -0,056 | [-0,068; -0,044] | 3 |
| aid | S2 DoG + geodésico | 0,566 | 0,581 | 0,585 | 0,580 | 0,009 | 15 | 78,1 | 14,3 | 1619 | 4858 | xavier | 0,627 | -0,061 | [-0,067; -0,055] | 3 |
| aid | S3 extremo DoG + cru | 0,568 | 0,583 | 0,587 | 0,582 | 0,013 | 15 | 71,1 | 13,8 | 1215 | 3143 | xavier | 0,627 | -0,059 | [-0,072; -0,046] | 3 |
| aid | S4 extremo DoG + DoG | 0,516 | 0,532 | 0,535 | 0,530 | 0,010 | 15 | 51,1 | 11,8 | 1704 | 3304 | xavier | 0,627 | -0,111 | [-0,121; -0,101] | 3 |
| aid | S5 região ∝ escala | 0,588 | 0,602 | 0,606 | 0,601 | 0,010 | 15 | 81,8 | 14,2 | 1265 | 4858 | xavier | 0,627 | -0,039 | [-0,066; -0,012] | 3 |
| aid | S6 multiescala cru | 0,579 | 0,593 | 0,597 | 0,592 | 0,010 | 15 | 42,9 | 13,2 | 1254 | 3143 | xavier | 0,627 | -0,048 | [-0,063; -0,033] | 3 |
| aid | S7 extremo LoG + cru | 0,571 | 0,585 | 0,590 | 0,585 | 0,012 | 15 | 30,5 | 11,1 | 1216 | 3304 | xavier | 0,627 | -0,056 | [-0,070; -0,043] | 3 |
| aid | S8 fusão | 0,584 | 0,598 | 0,603 | 0,598 | 0,014 | 15 | 168,7 | 14,2 | 1331 | 5328 | xavier | 0,627 | -0,043 | [-0,053; -0,033] | 3 |
| aid | centroide | 0,586 | 0,600 | 0,605 | 0,600 | 0,011 | 15 | 82,3 | 11,1 | 1134 | 2936 | xavier | 0,627 | -0,041 | [-0,049; -0,033] | 3 |
| aid | SIFT filtrado | 0,588 | 0,602 | 0,607 | 0,602 | 0,008 | 15 | 80,9 | 11,0 | 1134 | 2943 | xavier | 0,627 | -0,039 | [-0,049; -0,029] | 3 |
| aid | objeto sem rótulo | 0,588 | 0,602 | 0,606 | 0,601 | 0,010 | 15 | 81,3 | 10,9 | 1134 | 2943 | xavier | 0,627 | -0,039 | [-0,047; -0,031] | 3 |
| aid | FPS | 0,642 | 0,655 | 0,657 | 0,653 | 0,010 | 15 | 81,4 | 10,0 | 1134 | 2955 | xavier | 0,627 | +0,015 | [+0,001; +0,029] | 3 |

### Comparações diretas

| comparação | base | Δκ (primeiro − segundo) | IC 95% pareado |
|---|---|---|---|
| Gauss contra cru, mesmos pontos | flowers | -0,008 | [-0,037; +0,021] |
| Gauss contra cru, mesmos pontos | Protozoan_subgroup | +0,004 | [-0,023; +0,031] |
| Gauss contra cru, mesmos pontos | Eggs_subgroup | -0,013 | [-0,044; +0,019] |
| Gauss contra cru, mesmos pontos | Larvae_subgroup | -0,002 | [-0,117; +0,112] |
| Gauss contra cru, mesmos pontos | aid | -0,008 | [-0,020; +0,005] |
| DoG contra cru, mesmos pontos | flowers | +0,007 | [-0,002; +0,015] |
| DoG contra cru, mesmos pontos | Protozoan_subgroup | -0,055 | [-0,101; -0,009] |
| DoG contra cru, mesmos pontos | Eggs_subgroup | -0,017 | [-0,072; +0,039] |
| DoG contra cru, mesmos pontos | Larvae_subgroup | -0,099 | [-0,269; +0,072] |
| DoG contra cru, mesmos pontos | aid | -0,013 | [-0,024; -0,001] |
| mapa DoG contra cru, mesmos extremos | flowers | -0,019 | [-0,079; +0,041] |
| mapa DoG contra cru, mesmos extremos | Protozoan_subgroup | -0,120 | [-0,138; -0,102] |
| mapa DoG contra cru, mesmos extremos | Eggs_subgroup | -0,071 | [-0,093; -0,049] |
| mapa DoG contra cru, mesmos extremos | Larvae_subgroup | -0,129 | [-0,405; +0,147] |
| mapa DoG contra cru, mesmos extremos | aid | -0,052 | [-0,060; -0,045] |
| região ∝ escala contra tamanho fixo, mesmos extremos | flowers | +0,007 | [-0,030; +0,044] |
| região ∝ escala contra tamanho fixo, mesmos extremos | Protozoan_subgroup | +0,024 | [+0,010; +0,038] |
| região ∝ escala contra tamanho fixo, mesmos extremos | Eggs_subgroup | -0,001 | [-0,018; +0,016] |
| região ∝ escala contra tamanho fixo, mesmos extremos | Larvae_subgroup | +0,077 | [+0,017; +0,137] |
| região ∝ escala contra tamanho fixo, mesmos extremos | aid | +0,020 | [-0,002; +0,042] |
| várias escalas contra uma, mesmos pontos | flowers | +0,008 | [-0,045; +0,060] |
| várias escalas contra uma, mesmos pontos | Protozoan_subgroup | -0,012 | [-0,013; -0,012] |
| várias escalas contra uma, mesmos pontos | Eggs_subgroup | -0,004 | [-0,029; +0,020] |
| várias escalas contra uma, mesmos pontos | Larvae_subgroup | -0,008 | [-0,135; +0,119] |
| várias escalas contra uma, mesmos pontos | aid | +0,000 | [-0,021; +0,022] |
| LoG contra DoG, patch cru | flowers | -0,007 | [-0,031; +0,018] |
| LoG contra DoG, patch cru | Protozoan_subgroup | -0,002 | [-0,028; +0,024] |
| LoG contra DoG, patch cru | Eggs_subgroup | +0,007 | [-0,015; +0,029] |
| LoG contra DoG, patch cru | Larvae_subgroup | +0,062 | [-0,028; +0,151] |
| LoG contra DoG, patch cru | aid | +0,003 | [-0,012; +0,018] |
| extremo DoG contra centro geodésico, patch cru | flowers | +0,008 | [-0,034; +0,050] |
| extremo DoG contra centro geodésico, patch cru | Protozoan_subgroup | +0,000 | [-0,030; +0,031] |
| extremo DoG contra centro geodésico, patch cru | Eggs_subgroup | -0,002 | [-0,050; +0,046] |
| extremo DoG contra centro geodésico, patch cru | Larvae_subgroup | -0,052 | [-0,162; +0,058] |
| extremo DoG contra centro geodésico, patch cru | aid | -0,011 | [-0,019; -0,002] |
| fusão contra geodésico | flowers | +0,012 | [-0,003; +0,028] |
| fusão contra geodésico | Protozoan_subgroup | +0,007 | [-0,028; +0,042] |
| fusão contra geodésico | Eggs_subgroup | +0,004 | [-0,004; +0,011] |
| fusão contra geodésico | Larvae_subgroup | +0,028 | [-0,102; +0,159] |
| fusão contra geodésico | aid | +0,006 | [-0,011; +0,022] |

### Matriz em L3 `L`, só para consulta

| método | flowers | Protozoan_subgroup | Eggs_subgroup | Larvae_subgroup | aid | bases | Δ > 0 | boas | regressões | média | mediana | variância | pior | triagem 3/5 | 7/12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geodésico cru | +0,000 | +0,086* | +0,064* | +0,219* | -0,035! | 5 | 4 | 3 | 1 | +0,067 | +0,064 | 0,0096 | -0,035 | passa | aberto |
| S1 Gauss + geodésico | -0,016 | +0,059* | +0,059 | +0,144 | -0,041! | 5 | 3 | 1 | 1 | +0,041 | +0,059 | 0,0053 | -0,041 | passa | aberto |
| S2 DoG + geodésico | -0,029 | +0,047* | +0,057* | +0,130 | -0,060! | 5 | 3 | 2 | 1 | +0,029 | +0,047 | 0,0056 | -0,060 | passa | aberto |
| S3 extremo DoG + cru | +0,004 | +0,095* | +0,050 | +0,217* | -0,046! | 5 | 4 | 2 | 1 | +0,064 | +0,050 | 0,0101 | -0,046 | passa | aberto |
| S4 extremo DoG + DoG | -0,084! | -0,014 | -0,010 | -0,031! | -0,116! | 5 | 0 | 0 | 3 | -0,051 | -0,031 | 0,0022 | -0,116 | não | aberto |
| S5 região ∝ escala | +0,008 | +0,121* | +0,063* | +0,232* | -0,025 | 5 | 4 | 3 | 0 | +0,080 | +0,063 | 0,0103 | -0,025 | passa | aberto |
| S6 multiescala cru | -0,016 | +0,064 | +0,059* | +0,208* | -0,037! | 5 | 3 | 2 | 1 | +0,056 | +0,059 | 0,0092 | -0,037 | passa | aberto |
| S7 extremo LoG + cru | -0,001 | +0,101* | +0,063* | +0,234* | -0,042! | 5 | 3 | 3 | 1 | +0,071 | +0,063 | 0,0114 | -0,042 | passa | aberto |
| S8 fusão | +0,007 | +0,109* | +0,062* | +0,212* | -0,037! | 5 | 4 | 3 | 1 | +0,071 | +0,062 | 0,0093 | -0,037 | passa | aberto |
| centroide | +0,003 | +0,088* | +0,062* | +0,183* | -0,030 | 5 | 4 | 3 | 0 | +0,061 | +0,062 | 0,0068 | -0,030 | passa | aberto |
| SIFT filtrado | +0,002 | +0,099* | +0,061* | +0,219* | -0,028 | 5 | 4 | 3 | 0 | +0,070 | +0,061 | 0,0094 | -0,028 | passa | aberto |
| objeto sem rótulo | +0,022 | +0,109* | +0,070* | +0,291* | -0,034! | 5 | 4 | 3 | 1 | +0,092 | +0,070 | 0,0153 | -0,034 | passa | aberto |
| FPS | +0,015 | +0,119* | +0,094* | +0,231 | +0,023 | 5 | 5 | 2 | 0 | +0,096 | +0,094 | 0,0076 | +0,015 | passa | aberto |

### Matriz em L1 `L`, só para consulta

| método | flowers | Protozoan_subgroup | Eggs_subgroup | Larvae_subgroup | aid | bases | Δ > 0 | boas | regressões | média | mediana | variância | pior | triagem 3/5 | 7/12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| geodésico cru | -0,036! | +0,027 | -0,002 | +0,196* | -0,016 | 5 | 2 | 1 | 1 | +0,034 | -0,002 | 0,0088 | -0,036 | não | aberto |
| S1 Gauss + geodésico | -0,063! | +0,019 | -0,027 | +0,047 | -0,040! | 5 | 2 | 0 | 2 | -0,013 | -0,027 | 0,0020 | -0,063 | não | aberto |
| S2 DoG + geodésico | -0,017 | -0,036! | -0,032! | -0,055! | -0,014 | 5 | 0 | 0 | 3 | -0,031 | -0,032 | 0,0003 | -0,055 | não | aberto |
| S3 extremo DoG + cru | -0,035! | +0,030 | -0,009 | +0,128* | -0,022 | 5 | 2 | 1 | 1 | +0,018 | -0,009 | 0,0043 | -0,035 | não | aberto |
| S4 extremo DoG + DoG | -0,033! | -0,154! | -0,125! | -0,284! | -0,067! | 5 | 0 | 0 | 5 | -0,133 | -0,125 | 0,0094 | -0,284 | não | aberto |
| S5 região ∝ escala | -0,009 | +0,059* | +0,007 | +0,108 | -0,009 | 5 | 3 | 1 | 0 | +0,031 | +0,007 | 0,0026 | -0,009 | passa | aberto |
| S6 multiescala cru | -0,016 | +0,030* | -0,004 | +0,096* | -0,007 | 5 | 2 | 2 | 0 | +0,020 | -0,004 | 0,0021 | -0,016 | não | aberto |
| S7 extremo LoG + cru | -0,041! | +0,024 | +0,004 | +0,097 | -0,026 | 5 | 3 | 0 | 1 | +0,012 | +0,004 | 0,0029 | -0,041 | passa | aberto |
| S8 fusão | -0,015 | +0,022 | +0,010 | +0,112* | -0,011 | 5 | 3 | 1 | 0 | +0,024 | +0,010 | 0,0027 | -0,015 | passa | aberto |
| centroide | -0,020 | +0,049* | +0,011 | +0,130* | -0,014 | 5 | 3 | 2 | 0 | +0,031 | +0,011 | 0,0038 | -0,020 | passa | aberto |
| SIFT filtrado | -0,017 | +0,023 | +0,017 | +0,108* | +0,002 | 5 | 4 | 1 | 0 | +0,027 | +0,017 | 0,0023 | -0,017 | passa | aberto |
| objeto sem rótulo | -0,013 | +0,027 | +0,008 | +0,123 | -0,006 | 5 | 3 | 0 | 0 | +0,028 | +0,008 | 0,0031 | -0,013 | passa | aberto |
| FPS | -0,008 | +0,053* | +0,038 | +0,115* | +0,031* | 5 | 4 | 3 | 0 | +0,046 | +0,038 | 0,0020 | -0,008 | passa | aberto |

### Registro das escalas

| tipo | oitava | nível | σ | resolução (em 200 px) | ms/imagem | extremos/imagem | após contraste e borda/imagem | após NMS/imagem |
|---|---|---|---|---|---|---|---|---|
| dog | 0 | 1 | 2,02 | 200×200 | 0,11 | 11975,5 | 16,9 | 6,3 |
| dog | 0 | 2 | 2,54 | 200×200 | 0,11 | 10564,9 | 10,9 | 4,2 |
| dog | 0 | 3 | 3,20 | 200×200 | 0,11 | 9086,6 | 8,5 | 3,6 |
| dog | 1 | 1 | 4,03 | 100×100 | 0,04 | 1715,0 | 5,4 | 2,4 |
| dog | 1 | 2 | 5,08 | 100×100 | 0,04 | 1216,1 | 3,1 | 1,8 |
| dog | 1 | 3 | 6,40 | 100×100 | 0,04 | 557,2 | 2,1 | 1,3 |
| dog | 2 | 1 | 8,06 | 50×50 | 0,03 | 3,9 | 1,3 | 0,7 |
| dog | 2 | 2 | 10,16 | 50×50 | 0,03 | 1,6 | 0,7 | 0,4 |
| dog | 2 | 3 | 12,80 | 50×50 | 0,03 | 1,2 | 0,5 | 0,4 |
| dog | 3 | 1 | 16,13 | 25×25 | 0,03 | 0,9 | 0,3 | 0,2 |
| dog | 3 | 2 | 20,32 | 25×25 | 0,03 | 0,4 | 0,2 | 0,2 |
| dog | 3 | 3 | 25,60 | 25×25 | 0,03 | 0,2 | 0,1 | 0,1 |
| gauss | 0 | 0 | 1,60 | 200×200 | 0,08 | 0,0 | 0,0 | 0,0 |
| gauss | 0 | 1 | 2,02 | 200×200 | 0,05 | 0,0 | 0,0 | 0,0 |
| gauss | 0 | 2 | 2,54 | 200×200 | 0,04 | 0,0 | 0,0 | 0,0 |
| gauss | 0 | 3 | 3,20 | 200×200 | 0,04 | 0,0 | 0,0 | 0,0 |
| gauss | 0 | 4 | 4,03 | 200×200 | 0,06 | 0,0 | 0,0 | 0,0 |
| gauss | 0 | 5 | 5,08 | 200×200 | 0,07 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 0 | 3,20 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 1 | 4,03 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 2 | 5,08 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 3 | 6,40 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 4 | 8,06 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 1 | 5 | 10,16 | 100×100 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 0 | 6,40 | 50×50 | 0,02 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 1 | 8,06 | 50×50 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 2 | 10,16 | 50×50 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 3 | 12,80 | 50×50 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 4 | 16,13 | 50×50 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 2 | 5 | 20,32 | 50×50 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 0 | 12,80 | 25×25 | 0,02 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 1 | 16,13 | 25×25 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 2 | 20,32 | 25×25 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 3 | 25,60 | 25×25 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 4 | 32,25 | 25×25 | 0,03 | 0,0 | 0,0 | 0,0 |
| gauss | 3 | 5 | 40,64 | 25×25 | 0,03 | 0,0 | 0,0 | 0,0 |
| log | 0 | 1 | 2,02 | 200×200 | 0,09 | 12947,9 | 22,9 | 6,6 |
| log | 0 | 2 | 2,54 | 200×200 | 0,09 | 11794,3 | 15,7 | 5,0 |
| log | 0 | 3 | 3,20 | 200×200 | 0,09 | 10460,3 | 10,8 | 3,8 |
| log | 1 | 1 | 4,03 | 100×100 | 0,04 | 2051,0 | 7,7 | 2,4 |
| log | 1 | 2 | 5,08 | 100×100 | 0,04 | 1634,7 | 4,6 | 2,1 |
| log | 1 | 3 | 6,40 | 100×100 | 0,04 | 1135,2 | 2,8 | 1,5 |
| log | 2 | 1 | 8,06 | 50×50 | 0,03 | 68,4 | 1,8 | 0,8 |
| log | 2 | 2 | 10,16 | 50×50 | 0,03 | 2,5 | 1,0 | 0,6 |
| log | 2 | 3 | 12,80 | 50×50 | 0,03 | 1,4 | 0,6 | 0,4 |
| log | 3 | 1 | 16,13 | 25×25 | 0,03 | 1,1 | 0,4 | 0,3 |
| log | 3 | 2 | 20,32 | 25×25 | 0,03 | 0,7 | 0,3 | 0,2 |
| log | 3 | 3 | 25,60 | 25×25 | 0,03 | 0,3 | 0,2 | 0,2 |

### Onde as sementes caem

| base | ponto | sementes/imagem | baixa variância (sem máscara) | em objeto [oráculo] | área do objeto |
|---|---|---|---|---|---|
| Eggs | centroid | 50,0 | 0,491 | 0,505 | 0,199 |
| Eggs | geodesic | 50,0 | 0,525 | 0,480 | 0,199 |
| Eggs | obj | 31,4 | 0,000 | 0,526 | 0,199 |
| Eggs | s3 | 24,0 | 0,420 | 0,465 | 0,199 |
| Eggs | s7 | 28,3 | 0,398 | 0,447 | 0,199 |
| Eggs | sift_nms | 34,2 | 0,252 | 0,447 | 0,199 |
| Eggs_subgroup | centroid | 50,0 | 0,518 | 0,528 | 0,202 |
| Eggs_subgroup | geodesic | 50,0 | 0,557 | 0,494 | 0,202 |
| Eggs_subgroup | obj | 30,8 | 0,000 | 0,552 | 0,202 |
| Eggs_subgroup | s3 | 22,6 | 0,440 | 0,516 | 0,202 |
| Eggs_subgroup | s7 | 26,2 | 0,410 | 0,494 | 0,202 |
| Eggs_subgroup | sift_nms | 31,7 | 0,269 | 0,475 | 0,202 |
| Larvae | centroid | 50,0 | 0,456 | 0,256 | 0,061 |
| Larvae | geodesic | 50,0 | 0,476 | 0,254 | 0,061 |
| Larvae | obj | 35,7 | 0,000 | 0,335 | 0,061 |
| Larvae | s3 | 13,9 | 0,328 | 0,356 | 0,061 |
| Larvae | s7 | 17,2 | 0,314 | 0,325 | 0,061 |
| Larvae | sift_nms | 24,6 | 0,167 | 0,296 | 0,061 |
| Larvae_subgroup | centroid | 50,0 | 0,456 | 0,250 | 0,063 |
| Larvae_subgroup | geodesic | 50,0 | 0,479 | 0,247 | 0,063 |
| Larvae_subgroup | obj | 36,3 | 0,000 | 0,321 | 0,063 |
| Larvae_subgroup | s3 | 15,9 | 0,300 | 0,330 | 0,063 |
| Larvae_subgroup | s7 | 19,1 | 0,270 | 0,312 | 0,063 |
| Larvae_subgroup | sift_nms | 26,4 | 0,142 | 0,294 | 0,063 |
| Protozoan | centroid | 50,0 | 0,413 | 0,398 | 0,121 |
| Protozoan | geodesic | 50,0 | 0,480 | 0,343 | 0,121 |
| Protozoan | obj | 32,3 | 0,000 | 0,411 | 0,121 |
| Protozoan | s3 | 24,7 | 0,258 | 0,395 | 0,121 |
| Protozoan | s7 | 26,9 | 0,265 | 0,381 | 0,121 |
| Protozoan | sift_nms | 25,6 | 0,232 | 0,388 | 0,121 |
| Protozoan_subgroup | centroid | 50,0 | 0,423 | 0,405 | 0,120 |
| Protozoan_subgroup | geodesic | 50,0 | 0,491 | 0,348 | 0,120 |
| Protozoan_subgroup | obj | 31,8 | 0,000 | 0,430 | 0,120 |
| Protozoan_subgroup | s3 | 23,5 | 0,238 | 0,426 | 0,120 |
| Protozoan_subgroup | s7 | 25,9 | 0,242 | 0,411 | 0,120 |
| Protozoan_subgroup | sift_nms | 25,4 | 0,201 | 0,409 | 0,120 |
| aid | centroid | 50,0 | 0,417 | - | - |
| aid | geodesic | 50,0 | 0,472 | - | - |
| aid | obj | 35,8 | 0,000 | - | - |
| aid | s3 | 42,5 | 0,378 | - | - |
| aid | s7 | 44,8 | 0,321 | - | - |
| aid | sift_nms | 46,3 | 0,141 | - | - |
| flowers | centroid | 50,0 | 0,431 | - | - |
| flowers | geodesic | 50,0 | 0,496 | - | - |
| flowers | obj | 37,5 | 0,000 | - | - |
| flowers | s3 | 41,5 | 0,381 | - | - |
| flowers | s7 | 44,2 | 0,348 | - | - |
| flowers | sift_nms | 45,7 | 0,231 | - | - |
| helminth_eggs | centroid | 50,0 | 0,912 | - | - |
| helminth_eggs | geodesic | 50,0 | 0,904 | - | - |
| helminth_eggs | obj | 14,5 | 0,000 | - | - |
| helminth_eggs | s3 | 2,7 | 0,790 | - | - |
| helminth_eggs | s7 | 3,5 | 0,721 | - | - |
| helminth_eggs | sift_nms | 4,0 | 0,575 | - | - |
| helminth_larvae | centroid | 50,0 | 0,923 | - | - |
| helminth_larvae | geodesic | 50,0 | 0,923 | - | - |
| helminth_larvae | obj | 14,8 | 0,000 | - | - |
| helminth_larvae | s3 | 2,1 | 0,738 | - | - |
| helminth_larvae | s7 | 2,6 | 0,676 | - | - |
| helminth_larvae | sift_nms | 3,0 | 0,479 | - | - |
| protozoan_cysts | centroid | 50,0 | 0,929 | - | - |
| protozoan_cysts | geodesic | 50,0 | 0,914 | - | - |
| protozoan_cysts | obj | 13,2 | 0,000 | - | - |
| protozoan_cysts | s3 | 2,6 | 0,824 | - | - |
| protozoan_cysts | s7 | 3,5 | 0,763 | - | - |
| protozoan_cysts | sift_nms | 3,8 | 0,741 | - | - |
| remote_sensing | centroid | 50,0 | 0,437 | - | - |
| remote_sensing | geodesic | 50,0 | 0,489 | - | - |
| remote_sensing | obj | 34,6 | 0,000 | - | - |
| remote_sensing | s3 | 41,6 | 0,370 | - | - |
| remote_sensing | s7 | 44,2 | 0,314 | - | - |
| remote_sensing | sift_nms | 46,6 | 0,128 | - | - |
<!-- gerado:fim -->
