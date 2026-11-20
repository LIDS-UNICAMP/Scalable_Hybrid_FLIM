# Heatmaps — remote_sensing

- **Nosso:** PNAS-CS P1 c00008 f100 — `/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/pnas_cs/remote_sensing/state/seed42/candidates/c00008/f100/encoder.pt` — 1,383,536 params — κ teste (SVM original) 0.633 (recalculado aqui: 0.633). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 0.753 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (159 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000003 | Agriculture | River ✗ | Agriculture ✓ | img_01_Agriculture.png |
| 2 | 000002_00000038 | Airport | Airport ✓ | Grassland ✗ | img_02_Airport.png |
| 3 | 000004_00000082 | City | City ✓ | City ✓ | img_03_City.png |
| 4 | 000005_00000027 | Desert | Desert ✓ | Desert ✓ | img_04_Desert.png |
| 5 | 000007_00000093 | Grassland | Grassland ✓ | Forest ✗ | img_05_Grassland.png |
| 6 | 000008_00000004 | Highway | Highway ✓ | Highway ✓ | img_06_Highway.png |
| 7 | 000010_00000042 | Mountain | Mountain ✓ | Mountain ✓ | img_07_Mountain.png |
| 8 | 000012_00000065 | Port | City ✗ | Port ✓ | img_08_Port.png |
| 9 | 000013_00000010 | Railway | Mountain ✗ | Railway ✓ | img_09_Railway.png |
| 10 | 000014_00000016 | River | River ✓ | River ✓ | img_10_River.png |
