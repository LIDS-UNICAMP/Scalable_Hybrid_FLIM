# Técnica 2 — Graph Clustering (famílias estruturais)

Representações: WL subtree (GraKeL, 3 iterações) e embedding estrutural (cosseno). Concordância: Spearman(WL, embedding) = 0.57; ARI(espectral WL vs Ward embedding) = 0.24; ARI(espectral vs HDBSCAN) = 0.09 (HDBSCAN: 29 clusters, 344 ruído).
Validação com GED aproximada normalizada (custo/(nós+arestas dos dois); timeout 2.0 s por par, logo é um LIMITE SUPERIOR) (40 pares): Spearman(GED, WL) = 0.08715639365794166, Spearman(GED, embedding) = 0.4269431028666076.
k pela silhueta = 2; k usado na descrição = **6** (config k_report). Silhuetas: 2:0.27, 3:0.26, 4:0.20, 5:0.18, 6:0.15, 7:0.14, 8:0.14, 9:0.15, 10:0.14, 12:0.12. Silhueta ≤ 0,3 em todo k: as famílias são **fracamente separadas** (o espaço do GA é quase contínuo); os clusters descrevem tendências, não grupos nítidos.

| cluster | n | medoid | top_flowers | top_remote_sensing | top_ham10000 | top_aid | top_coconut_trees | ops | distingue | z_kappa | focus |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 137 | modehb_spifil/remote_sensing/c00258/f15 | 10 | 14 | 19 | 8 | 1 | bn, pw1, maxpools2, add, dbn, avgpools2 | edge:pw1>bn, op:dw5, edge:dw5>pw1, op:silu | -0.50 | coconut_trees:c00015, remote_sensing:c00015 |
| 1 | 37 | modehb_lemonade_seeded/remote_sensing/c00007/f15 | 0 | 30 | 0 | 0 | 0 | conv3, relu, maxpools2, bn, add, silu | edge:conv3>relu, edge:relu>conv3, op:conv3, op:relu | 0.03 | remote_sensing:c00002 |
| 2 | 21 | modehb_hybrid/aid/c00009/f15 | 0 | 10 | 0 | 0 | 0 | maxpools2, relu, pw1, add, dbn, gelu | edge:pw1>gelu, edge:maxpools2>dw3, edge:dw3>pw1, edge:gelu>add | 0.14 |  |
| 3 | 178 | modehb_lemonade/remote_sensing/c00042/f15 | 15 | 57 | 19 | 20 | 1 | pw1, conv3, conv5, cat, maxpool, bn | edge:avgpools2>pw1, edge:avgpools2>maxpool, edge:maxpool>pw1, edge:conv5>cat | 0.15 | growth:B1g0c06, growth_v3_jaci:B1g0c01 |
| 4 | 22 | modehb_lemonade/coconut_trees/c00003/f200 | 8 | 8 | 1 | 0 | 2 | relu, zca, conv3, maxpools2, conv7, conv5 | edge:conv3>zca, edge:zca>relu, op:zca, edge:maxpools2>conv3 | 0.46 |  |
| 5 | 194 | modehb_lemonade_C01_final/remote_sensing/c00063/f15 | 20 | 74 | 25 | 5 | 2 | pw1, maxpools2, conv5, conv3, bn, cat | edge:maxpools2>maxpool, edge:maxpools2>pw1, edge:maxpools2>conv5, op:conv5 | 0.14 | flowers:c00064 |

## As melhores de bases diferentes pertencem às mesmas famílias?

Qui-quadrado de independência (base × cluster, só top30%): p = 1.906387821584063e-10. p pequeno = a distribuição de famílias DEPENDE da base (cada base favorece famílias diferentes).

| par | JS |
|---|---|
| flowers|coconut_trees | 0.038 |
| flowers|ham10000 | 0.055 |
| ham10000|aid | 0.088 |
| flowers|remote_sensing | 0.147 |
| ham10000|coconut_trees | 0.157 |
| flowers|aid | 0.157 |
| remote_sensing|ham10000 | 0.161 |
| remote_sensing|coconut_trees | 0.225 |
| remote_sensing|aid | 0.236 |
| aid|coconut_trees | 0.293 |

JS = divergência de Jensen–Shannon (bits) entre as distribuições de famílias das top30%: 0 = mesmas famílias, 1 = disjuntas.

Figuras: clusters_mds.png/.svg, clusters_similaridade.png, clusters_base_x_familia.png
