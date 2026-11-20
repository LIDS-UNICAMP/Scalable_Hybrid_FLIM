# remote_sensing — κ de TESTE, 100% dos dados (15/11 22:40)

Réguas diferentes; ver notas. Fonte externa: presentation_results_cluster_november/results_grids.md (athena), disc-wt = Scalable_Hybrid_FLIM_discovery.

| # | técnica | params | κ teste | n | régua | Δ vs c00004 |
|---|---|---|---|---|---|---|
| 1 | AIMv2-3B (FM congelado) | 2,72 bi | 0,746 | 3 | C=100 embedding | +0,079 |
| 2 | DINOv2-g (FM) | 1,136 bi | 0,719 | 3 | C=100 embedding | +0,052 |
| 3 | ResNet-50 a1 (FM) | 23,5 mi | 0,704 | 3 | C=100 embedding | +0,037 |
| 4 | I-JEPA ViT-g/16 (FM) | 1,011 bi | 0,698 | 3 | C=100 embedding | +0,031 |
| **5** | **Hybrid SPiFiL MO-DEHB c00004 huge (nosso)** | **1,93 mi** | **0,667** | 1 | C=0,1 cov_head | — |
| 6 | ConvNeXtV2-pico (FM) | 8,55 mi | 0,655 | 3 | C=100 embedding | −0,012 |
| 7 | LEMONADE c00098 small (nosso) | 0,53 mi | 0,637 | 1 | C=0,1 cov_head | −0,030 |
| 8 | Hybrid MO-DEHB c00010 tiny (nosso) | 0,25 mi | 0,627 | 1 | C=0,1 cov_head | −0,040 |
| 9 | SPiFiL pequeno + LeJEPA | 0,98 mi | 0,586 | 3 | C=100 GeM+TTA+PW | −0,081 |
| 10 | SPiFiL pequeno + LeJEPA-Barlow | 0,98 mi | 0,574 | 3 | C=100 GeM+TTA+PW | −0,093 |
| 11 | SPiFiL pequeno + CosFace | 0,98 mi | 0,563 | 3 | C=100 GeM+TTA+PW | −0,104 |
| 12 | SPiFiL pequeno + SimCLR | 0,98 mi | 0,554 | 1 | C=100 GeM+TTA+PW | −0,113 |
| 13 | SPiFiL pequeno + Barlow 2048 | 0,98 mi | 0,550 | 3 | C=100 GeM+TTA | −0,117 |
| 14 | SPiFiL pequeno sem SSL (L0) | 0,98 mi | 0,529 | 3 | C=100 GeM+TTA+PW | −0,138 |
| 15 | Hybrid MO-DEHB c00001 tiny (nosso pior) | 0,02 mi | 0,524 | 1 | C=0,1 cov_head | −0,143 |
| 16 | SPiFiL pequeno + LeJEPA-VICReg | 0,98 mi | 0,524 | 3 | C=100 GeM+TTA+PW | −0,143 |
| 17 | SPiFiL pequeno + Barlow 64 | 0,98 mi | 0,518 | 3 | C=100 GeM+TTA+PW | −0,149 |
| 18 | SPiFiL pequeno + DINOv2 sem iBOT | 0,98 mi | 0,510 | 3 | C=100 GeM+TTA+PW | −0,157 |

Busca Hybrid MO-DEHB (nossa): 5 candidatos em f200, melhor 0,667 (c00004), pior 0,524 (c00001).
Notas: os nossos têm n=1 (split1); os externos, n=3. A régua cov_head também deu o maior salto nos nossos (0,51→0,61 em média), então parte do ganho pode ser a leitura: o justo é reavaliar o c00004 com a régua deles (C=100 GeM+TTA+PW) e o SPiFiL pequeno com cov_head.
