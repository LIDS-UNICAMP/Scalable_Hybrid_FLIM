# eggs — treino/val/teste por % e por classe (split1)

val e teste são fixos (os da busca, split1); só o treino da SVM muda com a %. ADAPTAÇÃO: o descritor splits_incremental não casa com o split da busca → amostra estratificada semeada de p% do treino de cada classe (mín. 1 por classe).

| classe | tr 1% | tr 5% | tr 10% | tr 25% | tr 50% | tr 75% | tr 100% | val | teste | % do treino |
|---|---|---|---|---|---|---|---|---|---|---|
| H.nana | 4 | 18 | 35 | 88 | 176 | 263 | 351 | 75 | 75 | 4.0% |
| H.diminuta | 1 | 3 | 6 | 15 | 30 | 44 | 59 | 12 | 12 | 0.7% |
| Ancilostomideo | 2 | 10 | 20 | 50 | 100 | 150 | 200 | 43 | 43 | 2.3% |
| E.vermicularis | 1 | 4 | 7 | 18 | 36 | 55 | 73 | 15 | 15 | 0.8% |
| A.lumbricoides | 6 | 29 | 58 | 146 | 292 | 439 | 585 | 125 | 125 | 6.6% |
| T.trichiura | 3 | 15 | 30 | 76 | 152 | 229 | 305 | 65 | 65 | 3.4% |
| S.mansoni | 2 | 9 | 18 | 44 | 89 | 134 | 178 | 38 | 38 | 2.0% |
| Taenia parasites | 3 | 13 | 26 | 66 | 132 | 199 | 265 | 57 | 57 | 3.0% |
| impurities | 69 | 343 | 687 | 1717 | 3434 | 5152 | 6869 | 1473 | 1473 | 77.3% |
| TOTAL | 91 | 444 | 887 | 2220 | 4441 | 6665 | 8885 | 1903 | 1903 | 100% |
