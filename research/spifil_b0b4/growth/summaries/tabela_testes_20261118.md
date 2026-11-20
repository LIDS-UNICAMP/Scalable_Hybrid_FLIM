# Tabela de TESTE consolidada — 18/11/2026 (gerada 17:48; teste nunca usado para seleção)

Régua ORIG: SVM linear C=100, sem scaler. Nas cadeias v3, o teste vem como so2 / tta8 / gap. f15 = checkpoint f15, que venceu o f45 na validação. FALTA = sem linha em test_orig.jsonl.

## Cadeias de crescimento (seed 42)

| base | cadeia | faixa | nó (fid) | κ val | κ teste (so2 / tta8 / gap ou ORIG) |
|---|---|---|---|---|---|
| remote_sensing | fix4 (gpu3) | B0 | seed00 (f0) membro | 0.4940 | 0.5018 |
| remote_sensing | fix4 (gpu3) | B0 | seed01 (f0) membro | 0.5714 | 0.5344 |
| remote_sensing | fix4 (gpu3) | B1 | B1g0c00 (f45) membro+melhor | 0.5924 | 0.5770 |
| remote_sensing | fix4 (gpu3) | B2 | seed02 (f0) membro | 0.5024 | 0.5837 |
| remote_sensing | fix4 (gpu3) | B2 | B2g1c02 (f45) membro | 0.5931 | 0.5288 |
| remote_sensing | fix4 (gpu3) | B2 | B2g1c02 (f15) melhor | 0.6203 | 0.5918 |
| remote_sensing | fix4 (gpu3) | B3 | B3g0c00 (f15) melhor | 0.5931 | 0.5288 |
| remote_sensing | fix5_nossl (gpu3) | B0 | seed00 (f0) membro | 0.4940 | 0.5018 |
| remote_sensing | fix5_nossl (gpu3) | B0 | seed01 (f0) membro | 0.5714 | 0.5344 |
| remote_sensing | fix5_nossl (gpu3) | B2 | seed02 (f0) membro | 0.5024 | 0.5837 |
| remote_sensing | fix5_nossl (gpu3) | B3 | B3g0c00 (f45) membro | 0.5172 | 0.5711 |
| remote_sensing | fix5_nossl (gpu3) | B1 | B1g0c00 (f15) melhor | 0.5714 | 0.5344 |
| remote_sensing | fix5_nossl (gpu3) | B3 | B3g0c07 (f15) melhor | 0.5223 | 0.6325 |
| flowers | fix5_nossl (gpu3) | B0 | seed00 (f0) membro | 0.6009 | 0.6377 |
| flowers | fix5_nossl (gpu3) | B0 | seed01 (f0) membro | 0.6471 | 0.6351 |
| flowers | fix5_nossl (gpu3) | B1 | B1g0c00 (f45) membro | 0.6775 | 0.6502 |
| flowers | fix5_nossl (gpu3) | B1 | B1g1c00 (f45) membro+melhor | 0.6805 | 0.6803 |
| flowers | fix5_nossl (gpu3) | B1 | B1g1c02 (f45) membro | 0.6504 | 0.6685 |
| flowers | fix5_nossl (gpu3) | B2 | seed02 (f0) membro | 0.6562 | 0.6593 |
| ham10000 | fix5_nossl (gpu3) | B0 | seed00 (f0) membro | 0.4786 | 0.4980 |
| ham10000 | fix5_nossl (gpu3) | B0 | seed01 (f0) membro | 0.4336 | 0.4826 |
| ham10000 | fix5_nossl (gpu3) | B1 | B1g0c02 (f45) membro+melhor | 0.4957 | 0.5133 |
| ham10000 | fix5_nossl (gpu3) | B1 | B1g0c03 (f45) membro | 0.4719 | 0.4833 |
| ham10000 | fix5_nossl (gpu3) | B4 | seed02 (f0) membro | 0.4775 | 0.5476 |
| aid | fix5_nossl (gpu3) | B0 | seed00 (f0) membro | 0.6360 | 0.6581 |
| aid | fix5_nossl (gpu3) | B0 | seed01 (f0) membro | 0.6863 | 0.6767 |
| aid | fix5_nossl (gpu3) | B1 | B1g0c02 (f45) membro | 0.6870 | 0.6822 |
| aid | fix5_nossl (gpu3) | B1 | B1g0c04 (f45) membro | 0.6973 | FALTA |
| aid | fix5_nossl (gpu3) | B4 | seed02 (f0) membro | 0.7463 | 0.7270 |
| aid | fix5_nossl (gpu3) | B1 | B1g0c04 (f15) melhor | 0.6981 | 0.6732 |
| coconut_trees | fix5_nossl (gpu3) | B0 | seed00 (f0) membro | 0.6542 | 0.6515 |
| coconut_trees | fix5_nossl (gpu3) | B0 | seed01 (f0) membro | 0.6871 | 0.6881 |
| coconut_trees | fix5_nossl (gpu3) | B2 | seed02 (f0) membro | 0.6516 | 0.6920 |
| coconut_trees | fix5_nossl (gpu3) | B1 | B1g0c02 (f15) melhor | 0.6941 | 0.7018 |
| flowers | fix4 (gpu2) | B0 | seed00 (f0) membro | 0.6009 | 0.6377 |
| flowers | fix4 (gpu2) | B0 | seed01 (f0) membro | 0.6471 | 0.6351 |
| flowers | fix4 (gpu2) | B2 | seed02 (f0) membro | 0.6562 | 0.6593 |
| flowers | fix4 (gpu2) | B3 | B3g0c00 (f15) melhor | 0.6719 | 0.6684 |
| flowers | v3/superpixels_prune | B0 | seed00 (f0) membro | 0.6101 | 0.6377 / 0.6195 / 0.6377 |
| flowers | v3/superpixels_prune | B0 | seed01 (f0) membro | 0.7291 | 0.7958 / 0.7988 / 0.6351 |
| flowers | v3/superpixels_prune | B1 | B1g0c02 (f45) membro | 0.7415 | 0.7779 / 0.7898 / 0.6714 |
| flowers | v3/superpixels_prune | B1 | B1g0c06 (f45) membro | 0.7171 | 0.7167 / 0.7381 / 0.6282 |
| flowers | v3/superpixels_prune | B1 | B1g1c01 (f45) membro | 0.7108 | 0.6894 / 0.6710 / 0.6377 |
| flowers | v3/superpixels_prune | B1 | B1g1c05 (f45) membro | 0.7350 | 0.7375 / 0.6976 / 0.6527 |
| flowers | v3/superpixels_prune | B1 | B1g1c07 (f15) membro | 0.7082 | 0.6860 / 0.6859 / 0.5763 |
| flowers | v3/superpixels_prune | B2 | seed02 (f0) membro | 0.6899 | 0.7107 / 0.7233 / 0.6593 |
| flowers | v3/superpixels_prune | B1 | B1g0c00 (f15) melhor | 0.7564 | 0.7442 / 0.7499 / 0.6503 |
| flowers | v3/superpixels_prune | B2 | B2g0c00 (f15) melhor | 0.7350 | 0.7375 / 0.6976 / 0.6527 |
| remote_sensing | v3/superpixels_prune | B0 | seed00 (f0) membro | 0.4790 | 0.4695 / 0.4904 / 0.5018 |
| remote_sensing | v3/superpixels_prune | B0 | seed01 (f0) membro | 0.5493 | 0.5776 / 0.5642 / 0.5344 |
| remote_sensing | v3/superpixels_prune | B1 | B1g0c03 (f45) membro | 0.5293 | 0.5683 / 0.5961 / 0.4500 |
| remote_sensing | v3/superpixels_prune | B1 | B1g0c07 (f45) membro | 0.5219 | 0.5776 / 0.5909 / 0.5271 |
| remote_sensing | v3/superpixels_prune | B1 | B1g1c00 (f45) membro | 0.5503 | 0.5836 / 0.5635 / 0.5219 |
| remote_sensing | v3/superpixels_prune | B1 | B1g1c03 (f45) membro | 0.5347 | 0.5702 / 0.5702 / 0.5145 |
| remote_sensing | v3/superpixels_prune | B1 | B1g1c05 (f45) membro | 0.5122 | 0.5110 / 0.4903 / 0.4985 |
| remote_sensing | v3/superpixels_prune | B2 | seed02 (f0) membro | 0.5553 | 0.6397 / 0.6397 / 0.5837 |
| remote_sensing | v3/superpixels_prune | B2 | B2g1c02 (f45) membro | 0.5306 | 0.5769 / 0.5697 / 0.4570 |
| remote_sensing | v3/superpixels_prune | B2 | B2g1c05 (f45) membro | 0.5440 | 0.5559 / 0.5563 / 0.1290 |
| remote_sensing | v3/superpixels_prune | B1 | B1g0c07 (f15) melhor | 0.5913 | FALTA |
| remote_sensing | v3/superpixels_prune | B2 | B2g1c02 (f15) melhor | 0.5505 | FALTA |
| remote_sensing | v3/superpixels_prune | B3 | B3g0c03 (f15) melhor | 0.5635 | FALTA |
| remote_sensing | v3_fixed/ae_so2 | B0 | seed00 (f0) membro | 0.4790 | 0.4695 / 0.4904 / 0.5018 |
| remote_sensing | v3_fixed/ae_so2 | B0 | seed01 (f0) membro | 0.5493 | 0.5776 / 0.5642 / 0.5344 |
| remote_sensing | v3_fixed/ae_so2 | B1 | B1g0c03 (f15) membro | 0.5493 | 0.5683 / 0.5479 / 0.4898 |
| remote_sensing | v3_fixed/ae_so2 | B1 | B1g0c07 (f45) membro | 0.5292 | 0.5419 / 0.5557 / 0.4188 |
| remote_sensing | v3_fixed/ae_so2 | B1 | B1g1c01 (f45) membro+melhor | 0.5771 | 0.5286 / 0.5424 / 0.4942 |
| remote_sensing | v3_fixed/ae_so2 | B2 | seed02 (f0) membro | 0.5553 | 0.6397 / 0.6397 / 0.5837 |
| remote_sensing | v3_fixed/ae_so2 | B2 | B2g0c01 (f45) membro | 0.5504 | 0.6044 / 0.5977 / 0.4766 |
| remote_sensing | v3_fixed/ae_so2 | B2 | B2g0c07 (f15) membro | 0.5500 | FALTA |
| remote_sensing | v3_fixed/ae_so2 | B2 | B2g1c03 (f45) membro+melhor | 0.5838 | 0.5205 / 0.5293 / 0.4874 |
| remote_sensing | v3_fixed/ae_so2 | B2 | B2g1c04 (f45) membro | 0.5577 | 0.5768 / 0.5694 / 0.4766 |
| remote_sensing | v3_fixed/ae_so2 | B3 | B3g0c02 (f15) melhor | 0.5553 | FALTA |

## Híbrido de outubro (seed 42)

| base | cabeça | dim | κ val | κ teste |
|---|---|---|---|---|
| flowers | gap1280 | 1280 | 0.5928 | 0.6202 |
| flowers | so2_4096_pca | 4096 | 0.7110 | 0.7502 |
| flowers | so2_4096_pca_tta8 | 4096 | 0.6940 | 0.6929 |
| remote_sensing | gap1280 | 1280 | 0.4465 | 0.4614 |
| remote_sensing | so2_4096_pca | 4096 | 0.4724 | 0.5415 |
| remote_sensing | so2_4096_pca_tta8 | 4096 | 0.4799 | 0.5778 |

## Outros testes

PNAS-CS: só existe remote_sensing (2772 linhas em ga_search/modehb/reports/test_svm_C100.jsonl). Candidatos f200 e phase0:

| cand | papel | fid | params | κ val | κ teste ORIG | κ teste COV |
|---|---|---|---|---|---|---|
| c00009 (melhor val) | search | 200 | 1.93M | 0.6051 | 0.3330 | 0.6190 |
| c00007 | search | 200 | 0.25M | 0.5847 | 0.4966 | 0.6480 |
| c00008 | search | 200 | 1.38M | 0.5842 | 0.6121 | 0.6710 |
| c00010 | search | 200 | 3.09M | 0.5300 | 0.3586 | 0.5067 |
| c00052 | search | 200 | 0.35M | 0.5200 | 0.3548 | 0.5396 |
| c00011 | search | 200 | 5.20M | 0.5146 | 0.3601 | 0.5234 |
| c00003 | phase0_ae | 45 | 0.47M | 0.5077 | 0.4297 | 0.5267 |
| c00001 | phase0_noae | 45 | 0.47M | 0.4790 | 0.4502 | 0.5408 |

A2 flowers: test_read=false nas 51 linhas de results.jsonl, então não há teste.

## Pendentes

- Rodando na GPU 0 (fila sequencial, timeout de 1 h por cadeia): v3/superpixels_prune remote_sensing, f15 B1g0c07 / B2g1c02 / B3g0c03, desde 17:45. Depois vem v3_fixed/ae_so2 remote_sensing (nada meu nela; só esvazia a fila).
- Novos membros e melhores que surgiram durante a coleta (as cadeias ainda rodam; o canal de teste de cada cadeia mede no fim da faixa): aid B1g0c04 f45; v3_fixed B2g0c07 f15 e B3g0c02 f15.
- flowers v3 B1g1c07 (nova versão): meu processo estourou o timeout (rc=124). O canal da própria cadeia também o mediu.
