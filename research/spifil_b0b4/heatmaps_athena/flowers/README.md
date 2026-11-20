# Heatmaps — flowers

- **Nosso:** MO-DEHB hybrid c00001 f200 (tiny) — `/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/modehb_hybrid_C01_final/flowers/state/seed42/candidates/c00001/f200/encoder.pt` — 21,079 params — κ teste (SVM original) 0.649 (recalculado aqui: 0.649). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 0.979 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (412 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000451 | daisy | daisy ✓ | daisy ✓ | img_01_daisy.png |
| 2 | 000002_00000218 | dandelion | dandelion ✓ | dandelion ✓ | img_02_dandelion.png |
| 3 | 000003_00000207 | rose | rose ✓ | rose ✓ | img_03_rose.png |
| 4 | 000004_00000329 | sunflower | sunflower ✓ | sunflower ✓ | img_04_sunflower.png |
| 5 | 000005_00000404 | tulip | tulip ✓ | tulip ✓ | img_05_tulip.png |
| 6 | 000001_00000181 | daisy | daisy ✓ | daisy ✓ | img_06_daisy.png |
| 7 | 000002_00000175 | dandelion | dandelion ✓ | dandelion ✓ | img_07_dandelion.png |
| 8 | 000003_00000308 | rose | rose ✓ | rose ✓ | img_08_rose.png |
| 9 | 000004_00000002 | sunflower | daisy ✗ | rose ✗ | img_09_sunflower.png |
| 10 | 000005_00000397 | tulip | tulip ✓ | tulip ✓ | img_10_tulip.png |
