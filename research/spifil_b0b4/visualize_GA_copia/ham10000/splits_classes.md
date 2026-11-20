# ham10000 — treino/val/teste por % e por classe (split1)

val e teste são fixos (os da busca, split1); só o treino da SVM muda com a %. ADAPTAÇÃO: o descritor splits_incremental não casa com o split da busca → amostra estratificada semeada de p% do treino de cada classe (mín. 1 por classe).

| classe | tr 1% | tr 5% | tr 25% | tr 50% | tr 75% | tr 100% | val | teste | % do treino |
|---|---|---|---|---|---|---|---|---|---|
| MEL | 4 | 22 | 112 | 224 | 337 | 449 | 100 | 564 | 10.9% |
| NV | 28 | 139 | 694 | 1388 | 2082 | 2776 | 580 | 3349 | 67.2% |
| BCC | 2 | 11 | 54 | 108 | 163 | 217 | 50 | 247 | 5.3% |
| AKIEC | 1 | 7 | 33 | 66 | 98 | 131 | 30 | 166 | 3.2% |
| BKL | 5 | 23 | 114 | 228 | 343 | 457 | 98 | 544 | 11.1% |
| DF | 1 | 3 | 13 | 26 | 39 | 52 | 10 | 53 | 1.3% |
| VASC | 1 | 2 | 12 | 24 | 35 | 47 | 10 | 85 | 1.1% |
| TOTAL | 42 | 207 | 1032 | 2064 | 3097 | 4129 | 878 | 5008 | 100% |
