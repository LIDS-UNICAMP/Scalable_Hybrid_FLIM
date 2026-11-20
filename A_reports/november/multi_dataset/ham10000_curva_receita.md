# HAM10000: curva da receita SPiFiL sem backprop (11/11/2026)

Receita fps sem saliência (a melhor no HAM) + TTA, leitura L6 cat_tta, 3 splits. Régua: random_z do mesmo tamanho.
Correções aplicadas: (1) a 100% o split do HAM tem validação == treino (core/data/splits.py:9), então o ponto de 100% é medido no TESTE; (2) o cache de fontes não separava porcentagens, e os splits 2 e 3 de 5% a 100% foram refeitos (`svm_disc_md_receita_tta_ham_fix_*`). O split 1 e o pct1 já estavam limpos.

| treino | SPiFiL + TTA | aleatório + TTA | medido em |
|---|---|---|---|
| 1% | 0,269 | 0,236 | validação |
| 5% | 0,364 | 0,315 | validação |
| 25% | 0,472 | 0,390 | validação |
| 50% | 0,515 | 0,423 | validação |
| 75% | 0,528 | 0,442 | validação |
| 100% | 0,545 | 0,457 | **teste** |

A saliência não ajuda (fpsw + TTA a 100%: 0,528). O ganho sobre o aleatório cresce até ~0,09 a partir de 25%. A curva quase achata entre 50% e 100%.
CSVs: `results/svm_disc_md_receita_tta_ham_ham10000_*` (split 1 e pct1), `results/svm_disc_md_receita_tta_ham_fix_*` (splits 2–3, 5–75%), `results/svm_disc_md_receita_tta_ham100_test_*` (100%, teste).
