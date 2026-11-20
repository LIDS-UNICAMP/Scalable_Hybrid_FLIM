# Curvas 1–100%: resultados até agora (09/11, ~22:50 UTC)

Bases impuras (Protozoan, Eggs, Larvae). κ de validação. As curvas da receita com backprop ainda não têm nenhum treino completo (0 de 54).

## DINOv2-giant e I-JEPA ViT-g (congelados + SVM; média de 3 splits; `results/benchmark/{dinov2-giant,ijepa_vitg16_22k}.csv`)

| base | modelo | 1% | 5% | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|---|---|
| Eggs | DINOv2 | 0,822 | 0,941 | 0,955 | 0,976 | 0,980 | 0,985 | 0,985 |
| Eggs | I-JEPA | 0,634 | 0,855 | 0,902 | 0,949 | 0,970 | 0,975 | 0,975 |
| Larvae | DINOv2 | 0,858 | 0,915 | 0,963 | 0,974 | 0,968 | 0,963 | 0,973 |
| Larvae | I-JEPA | 0,617 | 0,768 | 0,748 | 0,835 | 0,850 | 0,882 | 0,872 |
| Protozoan | DINOv2 | 0,757 | 0,884 | 0,908 | 0,934 | 0,939 | 0,947 | 0,949 |
| Protozoan | I-JEPA | 0,597 | 0,797 | 0,860 | 0,899 | 0,921 | 0,930 | 0,934 |

## Receita SPiFiL sem backprop (referência, 50%, 3 splits, L6 `cat`)
Protozoan 0,834; Eggs 0,866; Larvae 0,855 (melhor aleatório 0,592 / 0,630 / 0,473).

## Receita SPiFiL COM backprop: avaliação parcial por camada (Protozoan, split 1; `results/svm_hyb_curva_receita_parcial_L{1,2,3}.csv`)
Encoder até a camada indicada, GeM da última camada disponível, SVM oficial.

| treino | até L1 | até L2 | até L3 |
|---|---|---|---|
| 1% | 0,287 | **0,412** | 0,396 |
| 5% | 0,511 | **0,613** | 0,532 |
| 25% | 0,666 | — | — |
| 50% | 0,666 | — | — |

- L2 melhora sobre L1 (+0,10 a +0,13). L3 PIORA sobre L2 (−0,016 e −0,081): um primeiro sinal de que, com backprop, camadas a mais podem atrapalhar (regra de parada).
- Ainda faltam L4 a L6; o resultado final é na camada 6.

## Andamento
54 treinos da receita, 1 por GPU, ~2 h por camada (~12 h por treino) → ~7 dias no ritmo atual; depois, o controle aleatório (mais ~7 dias). Opções para acelerar estão com o usuário. A árvore-espelho para o avaliador é mantida pela sessão `espelho_sync`.

## Atualização (10/11, ~03:30 UTC)
Protozoan, split 1, parcial por camada (κ val): 1%: L1 0,287 → L2 0,412 → L3 0,396 → **L4 0,228**; 5%: L1 0,511 → L2 0,613 → L3 0,532 → **L4 0,479**. Cada camada a partir da L3 piora.
Larvae no jaci, L6 completo (κ val / test; época do ckpt escolhido por val/loss): s1 p1 −0,150/−0,157 (96); s1 p5 0,261/0,260 (158); s1 p25 0,510/0,510 (68); s1 p50 0,209/0,170 (5); s2 p1 −0,015/−0,035 (64); s2 p5 0,111/0,064 (199); s2 p25 0,445/0,384 (188). CSVs no jaci: results/svm_hyb_curva_receita_{val,test}_gemttapwr_jaci_cur_s{1,2}.csv.
Leitura: a receita com backprop fica muito abaixo da receita sem backprop (Larvae 0,855; Protozoan 0,834). A escolha do checkpoint por val/loss (reconstrução) se descola do κ (s1 p50 escolheu a época 5).
Protozoan s1 1%, L6 completo (ckpt por val/loss): κ val 0,241, F1 0,297 (`results/svm_hyb_curva_receita_parcial_L6.csv`). Sequência L1→L6: 0,287 → 0,412 → 0,396 → 0,228 → ? → 0,241. Melhor camada: L2.
Protozoan s1 5%, L6 completo: κ val 0,499, F1 0,530 (`results/svm_hyb_curva_receita_parcial_L6_p5.csv`). Sequência L1→L6: 0,511 → 0,613 → 0,532 → 0,479 → ? → 0,499. Melhor camada: L2. Referências a 5%: DINOv2 0,884, I-JEPA 0,797.
