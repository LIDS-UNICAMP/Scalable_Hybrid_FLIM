# Heatmaps — eggs

- **Nosso:** MO-DEHB hybrid c00002 f200 (small) — `/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/modehb_hybrid/eggs/state/seed42/candidates/c00002/f200/encoder.pt` — 562,463 params — κ teste (SVM original) 0.846 (recalculado aqui: 0.846). Sem TTA (a mesma leitura do κ).
- **FM:** DINOv2 ViT-g/14 — `facebook/dinov2-giant` — 1,136,480,768 params — κ teste 0.988 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 518 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (1903 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Parasitas:** o nosso modelo vê `pre/orig` (objeto recortado, nomes renomeados; mapeamento oficial `splits/split1_modehb_mapping.csv`, mesmo split 1); o FM vê `raw/orig` (campo inteiro). Por isso a coluna imagem traz os dois nomes.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000420 | H.nana | H.nana ✓ | H.nana ✓ | img_01_H_nana.png |
| 2 | 000002_00001578 / 000002_00000023 | H.diminuta | H.diminuta ✓ | H.diminuta ✓ | img_02_H_diminuta.png |
| 3 | 000003_00001946 / 000003_00000075 | Ancilostomideo | Ancilostomideo ✓ | Ancilostomideo ✓ | img_03_Ancilostomideo.png |
| 4 | 000004_00002772 / 000004_00000066 | E.vermicularis | impurities ✗ | E.vermicularis ✓ | img_04_E_vermicularis.png |
| 5 | 000005_00004039 / 000005_00000784 | A.lumbricoides | A.lumbricoides ✓ | A.lumbricoides ✓ | img_05_A_lumbricoides.png |
| 6 | 000006_00007531 / 000006_00000390 | T.trichiura | T.trichiura ✓ | T.trichiura ✓ | img_06_T_trichiura.png |
| 7 | 000007_00008995 / 000007_00000059 | S.mansoni | S.mansoni ✓ | S.mansoni ✓ | img_07_S_mansoni.png |
| 8 | 000008_00011144 / 000008_00000037 | Taenia parasites | Taenia parasites ✓ | Taenia parasites ✓ | img_08_Taenia_parasites.png |
| 9 | 000011_00002556 / 000009_00001686 | impurities | impurities ✓ | impurities ✓ | img_09_impurities.png |
| 10 | 000001_00000194 | H.nana | H.nana ✓ | H.nana ✓ | img_10_H_nana.png |
