# TTA na receita SPiFiL sem backprop (10/11/2026)

TTA = média do GeM sobre 8 vistas (rot90 × flip) com o mesmo encoder congelado e o mesmo PCA-whitening do treino (`amostra.tta`, `eval/spf_grade.py:1318-1321`). Só na receita; DINOv2, I-JEPA e os outros benchmarks continuam sem TTA (regra do usuário). κ e F1 de validação, 50%, 3 splits, L6 `cat`. YAML `experiments/autoencoder/disc_md_receita_tta.yaml` (athena) e `_jaci` (parasitos).

| base | receita κ (F1) | receita + TTA κ (F1) | Δκ | aleatório sem TTA |
|---|---|---|---|---|
| aid (fps) | 0,771 (0,779) | 0,844 (0,850) | +0,073 (3/3 splits) | 0,635 |
| remote_sensing (fps) | 0,492 (0,459) | 0,539 (0,524) | +0,047 (2/3; s3 empata) | 0,487 |
| flowers (fpsw) | 0,655 (0,726) | 0,660 (0,730) | +0,005 (empate) | 0,549 |
| Eggs (fpsw) | 0,866 (0,887) | 0,919 (0,928) | +0,053 (3/3) | 0,560 |
| Protozoan (fpsw) | 0,834 (0,830) | 0,887 (0,864) | +0,053 (3/3) | 0,485 |
| Larvae (fpsw) | 0,855 (0,928) | 0,878 (0,939) | +0,023 (1/3 sobe; 2 empatam) | 0,325 |

Origem: `/exps/moliveira/results/svm_disc_md_receita_tta_<base>_s0200_split{1,2,3}_pct50_sh?.csv`, receita na linha 84 e receita + TTA na linha 88.
Parasitos: CSVs no jaci, copiados para `/exps/moliveira/recebido_jaci/exps/results/` (rsync nº 29); receita na linha 84 e receita + TTA na linha 88 de `..._<base>_s0200_split{1,2,3}_pct50_sh1.csv`.
Leitura: o TTA ajuda onde a orientação não importa: cenas aéreas e parasitos no microscópio (+0,05 a +0,07). Na flowers não muda. Com TTA, o fps sem saliência alcança a receita nos parasitos (Protozoan 0,911, Eggs 0,910, Larvae 0,868).
