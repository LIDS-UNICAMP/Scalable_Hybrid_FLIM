# Heatmaps — ham10000

- **Nosso:** MO-DEHB hybrid c00001 f200 — `/exps/moliveira/heatmaps_src/artifacts/modehb_hybrid/ham10000/state/seed42/candidates/c00001/f200/encoder.pt` — 19,550 params — κ teste (SVM original) 0.534 (recalculado aqui: 0.534). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 0.725 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (5008 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_ISIC_0033496 | MEL | MEL ✓ | MEL ✓ | img_01_MEL.png |
| 2 | 000002_ISIC_0031135 | NV | NV ✓ | NV ✓ | img_02_NV.png |
| 3 | 000003_ISIC_0032696 | BCC | BCC ✓ | BCC ✓ | img_03_BCC.png |
| 4 | 000004_ISIC_0025368 | AKIEC | BCC ✗ | AKIEC ✓ | img_04_AKIEC.png |
| 5 | 000005_ISIC_0031125 | BKL | NV ✗ | BKL ✓ | img_05_BKL.png |
| 6 | 000006_ISIC_0027118 | DF | NV ✗ | DF ✓ | img_06_DF.png |
| 7 | 000007_ISIC_0027888 | VASC | VASC ✓ | VASC ✓ | img_07_VASC.png |
| 8 | 000001_ISIC_0030360 | MEL | BKL ✗ | MEL ✓ | img_08_MEL.png |
| 9 | 000002_ISIC_0032772 | NV | NV ✓ | NV ✓ | img_09_NV.png |
| 10 | 000003_ISIC_0033366 | BCC | NV ✗ | MEL ✗ | img_10_BCC.png |
