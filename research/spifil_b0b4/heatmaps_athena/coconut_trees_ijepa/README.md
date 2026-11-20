# Heatmaps — coconut_trees_ijepa

- **Nosso:** MO-DEHB hybrid c00001 f200 (tiny) — `/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/modehb_hybrid/coconut_trees/state/seed42/candidates/c00001/f200/encoder.pt` — 22,368 params — κ teste (SVM original) 0.813 (recalculado aqui: 0.813). Sem TTA (a mesma leitura do κ).
- **FM:** I-JEPA ViT-g/16 — `facebook/ijepa_vitg16_22k` — 1,011,368,576 params — κ teste 0.786 (benchmark, split 1, 100%). Por que este FM: segundo melhor FM da base no benchmark; versao extra porque refazer o SVM do melhor (ConvNeXtV2-pico, pasta coconut_trees/) leva ~8 h. Entrada 224 px, crop_pct 0.9, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (2038 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000306 | coco | coco ✓ | coco ✓ | img_01_coco.png |
| 2 | 000002_00003212 | non_coco | non_coco ✓ | non_coco ✓ | img_02_non_coco.png |
| 3 | 000001_00002432 | coco | coco ✓ | coco ✓ | img_03_coco.png |
| 4 | 000002_00000880 | non_coco | non_coco ✓ | non_coco ✓ | img_04_non_coco.png |
| 5 | 000001_00004412 | coco | coco ✓ | coco ✓ | img_05_coco.png |
| 6 | 000002_00003132 | non_coco | non_coco ✓ | non_coco ✓ | img_06_non_coco.png |
| 7 | 000001_00005538 | coco | coco ✓ | coco ✓ | img_07_coco.png |
| 8 | 000002_00000230 | non_coco | non_coco ✓ | non_coco ✓ | img_08_non_coco.png |
| 9 | 000001_00007564 | coco | coco ✓ | coco ✓ | img_09_coco.png |
| 10 | 000002_00002806 | non_coco | non_coco ✓ | non_coco ✓ | img_10_non_coco.png |
