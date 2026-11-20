# protozoan_raw — treino/val/teste por % e por classe (split1)

val e teste são fixos (os da busca, split1); só o treino da SVM muda com a %.

| classe | tr 1% | tr 5% | tr 10% | tr 25% | tr 50% | tr 75% | tr 100% | val | teste | % do treino |
|---|---|---|---|---|---|---|---|---|---|---|
| E.coli | 7 | 31 | 61 | 153 | 305 | 457 | 609 | 130 | 130 | 2.3% |
| E.histolytica | 5 | 24 | 47 | 116 | 231 | 346 | 461 | 99 | 99 | 1.8% |
| E.nana | 13 | 63 | 125 | 313 | 625 | 937 | 1249 | 267 | 267 | 4.8% |
| Giardia | 14 | 68 | 136 | 338 | 676 | 1014 | 1351 | 290 | 290 | 5.2% |
| I.butschlii | 24 | 116 | 231 | 577 | 1154 | 1731 | 2307 | 495 | 495 | 8.8% |
| B.hominis parasites | 3 | 11 | 22 | 55 | 109 | 163 | 217 | 46 | 46 | 0.8% |
| Impurities | 202 | 1002 | 2000 | 4994 | 9985 | 14978 | 19965 | 4280 | 4279 | 76.3% |
| TOTAL | 268 | 1315 | 2622 | 6546 | 13085 | 19626 | 26159 | 5607 | 5606 | 100% |
