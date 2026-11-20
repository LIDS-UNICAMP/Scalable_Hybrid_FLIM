
## lejepa/pequeno · remote_sensing (TESTE, leitura oficial cat+TTA+SVM) — 11/11 23:44
- Modelo 40-80-152-152-152-312 k3 (0,98 M), 200 px, LeJEPA lr 5e-5 bf16, ckpt = menor val_ssl_loss (sem rótulo), 849–1000 épocas.
- 50%: κ 0,476 → 0,563 (+0,087; 3/3 splits sobem); F1 0,481 → 0,568.
- 100%: κ 0,529 → 0,586 (+0,057; 3/3); F1 0,516 → 0,588.
- Fonte: results/svm_ssl_pequeno_lejepa_test_catttapw_athena.csv (stages L0 vs L6).
- Leitura: o SSL (LeJEPA) melhora o SPiFiL pequeno no remote_sensing em todos os splits; ainda melhorava no teto de 1000 épocas.
- Benchmarks no mesmo teste (results/benchmark/*.csv, média 3 splits, κ 50% / 100%): AIMv2-3B 0,743/0,746; DINOv2-giant 0,710/0,719; I-JEPA ViT-g 0,648/0,698; ConvNeXtV2-pico (8,6 M) 0,629/0,655. SPiFiL+LeJEPA (0,98 M): 0,563/0,586.

## dino/pequeno · remote_sensing (TESTE, cat+TTA+SVM) — 12/11 05:39
- 50%: κ 0,476 → 0,454 (−0,022); 100%: κ 0,529 → 0,510 (−0,019). Piora em 2 de 3 splits nos dois tamanhos.
- Fonte: results/svm_ssl_pequeno_dino_test_catttapw_athena.csv. Leitura: o DINOv2 (EMA, Sinkhorn, KoLeo) não serve para o SPiFiL de 1 M; a sonda exagerava a perda (−0,15), mas a leitura oficial confirma que piora.
- Comparação no mesmo teste (50%): LeJEPA 0,563, SimCLR 0,566, DINOv2 0,454, sem SSL 0,476.

## lejepa/8,2 M · Protozoan split 1, 50% (TESTE, cat+TTA+SVM) — 12/11 02:59 (jaci)
- κ 0,814 → 0,875 (+0,060); F1 0,768 → 0,845 (+0,077); acc 0,767 → 0,849.
- Ckpt = época 45 = mínimo de val_loss (0,6265); ckpt_monitor probe/val_ssl_loss; sem rótulo (o máx. da sonda foi na época 40).
- 51 épocas; a sonda caiu nas épocas 1–9 (0,47) e recuperou a partir da 20. Fonte: jaci results/svm_ssl_8M_lejepa_test_catttapw_jaci.csv.

## barlow (oficial Meta)/pequeno · aid, remote_sensing, flowers (TESTE, cat+TTA+SVM, 3 splits) — 12/11
- aid (1,97 M): 50% 0,739 → 0,699 (−0,040); 100% 0,783 → 0,763 (−0,020).
- remote_sensing: 50% 0,476 → 0,476 (0,000); 100% 0,529 → 0,518 (−0,011).
- flowers: 50% 0,594 → 0,591 (−0,003); 100% 0,611 → 0,597 (−0,014).
- Leitura: o Barlow Twins fiel ao main.py (lote 64, AdamW) não melhora o SPiFiL pequeno; empata ou piora. Fonte: results/svm_ssl_pequeno_barlow_test_catttapw_athena.csv.

## barlow lote 64 · Protozoan (TESTE, cat+TTA+SVM, 3 splits; jaci) — 12/11 12:58
- 50%: κ 0,789 → 0,362 (−0,43); 100%: κ 0,797 → 0,601 (−0,20). Colapso; ckpts nas épocas 0–26 pela menor val_ssl_loss.
- Fonte: jaci results/svm_ssl_pequeno_barlow_test_catttapw_jaci.csv. Motivo da troca para o lote oficial 2048 (GradCache).
- Larvae (splits 1–2): 50% κ 0,788 → 0,787; 100% (s1) 0,816 → 0,775. Eggs (só s1): 50% 0,860 → 0,818; 100% 0,883 → 0,858. Resumo Barlow-64 nos parasitos: Protozoan colapsa, Larvae neutro, Eggs piora.

## lejepa/pequeno · Eggs (TESTE, cat+TTA+SVM, 3 splits; jaci) — 12/11 13:49
- 50%: κ 0,855 → 0,878 (+0,023); 100%: κ 0,877 → 0,888 (+0,011). Ganha em 6 de 6 splits. F1 50% 0,902 → 0,919.
- Ckpts (menor val_ssl_loss): épocas 51/65/77 (50%), 28/54/36 (100%). Fonte: jaci results/svm_ssl_pequeno_lejepa_test_catttapw_jaci.csv.

## lejepa/pequeno · ham10000 (TESTE PARCIAL, ckpt atual; jaci) — 12/11 15:03
- 50% (3 splits): κ 0,465 → 0,493 (+0,028); 100% (2 splits): 0,502 → 0,543 (+0,040). Ganha em 5 de 5 splits com ckpt. Treino ainda rodando; eval final virá.
- Fonte: jaci results/svm_ssl_pequeno_lejepa_test_catttapw_jaci_ham10000_parcial.csv.

## Testes athena 12/11 ~18:00 (cat+TTA+SVM, 3 splits salvo indicado)
- CosFace: remote_sensing 50% 0,476 → 0,591 (+0,115, MELHOR SSL no rs); 100% 0,529 → 0,563. flowers 50% 0,594 → 0,580; 100% 0,611 → 0,664 (+0,053).
- DINOv2: flowers 50% 0,594 → 0,551; 100% 0,611 → 0,587 (piora, como no rs).
- LeJEPA+VICReg: remote_sensing 50% +0,031, 100% −0,005; coconut_trees (parcial) −0,045 a −0,048.

## lejepa/pequeno · Larvae (TESTE, 3 splits; jaci) — 12/11 15:12
- 50%: κ 0,767 → 0,663 (−0,104); 100%: 0,782 → 0,742 (−0,040). Ganha só em 1 de 6 splits. Base de teste pequena (κ em degraus). Primeira base em que o LeJEPA piora.
- SimCLR aid (encoder antigo 4,34 M; só split 1): 50% 0,688 → 0,850 (+0,162); 100% 0,734 → 0,878 (+0,144). MAIOR ganho de SSL até agora.
- LeJEPA+Barlow (reg): remote_sensing 50% 0,476 → 0,527; 100% 0,529 → 0,574.
- LeJEPA+VICReg coconut_trees (3 splits): 50% 0,818 → 0,776; 100% 0,822 → 0,751 (piora).
- ham10000 FINAL (3 splits): 50% κ 0,465 → 0,493 (+0,028); 100% 0,502 → 0,536 (+0,034); ganha 6 de 6. F1 100% 0,555 → 0,584. Fonte: jaci results/svm_ssl_pequeno_lejepa_test_catttapw_jaci.csv.
