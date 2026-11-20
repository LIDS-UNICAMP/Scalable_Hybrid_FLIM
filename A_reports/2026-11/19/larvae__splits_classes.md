# larvae — treino/val/teste por % e por classe (split1)

val e teste são fixos (os da busca, split1); só o treino da SVM muda com a %. ADAPTAÇÃO: o descritor splits_incremental não casa com o split da busca → amostra estratificada semeada de p% do treino de cada classe (mín. 1 por classe).

| classe | tr 1% | tr 5% | tr 10% | tr 25% | tr 50% | tr 75% | tr 100% | val | teste | % do treino |
|---|---|---|---|---|---|---|---|---|---|---|
| S.stercoralis | 2 | 9 | 17 | 43 | 86 | 130 | 173 | 37 | 37 | 15.5% |
| Impurities | 9 | 47 | 94 | 236 | 472 | 709 | 945 | 203 | 203 | 84.5% |
| TOTAL | 11 | 56 | 111 | 279 | 558 | 839 | 1118 | 240 | 240 | 100% |
