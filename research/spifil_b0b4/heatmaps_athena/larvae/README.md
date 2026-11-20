# Heatmaps — larvae

- **Nosso:** LEMONADE c00003 f200 (medium) — `/dados/home/moliveira/Scalable_Hybrid_FLIM/artifacts/modehb_lemonade/larvae/state/seed42/candidates/c00003/f200/encoder.pt` — 1,341,376 params — κ teste (SVM original) 0.858 (recalculado aqui: 0.858). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 1.000 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (240 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Parasitas:** o nosso modelo vê `pre/orig` (objeto recortado, nomes renomeados; mapeamento oficial `splits/split1_modehb_mapping.csv`, mesmo split 1); o FM vê `raw/orig` (campo inteiro). Por isso a coluna imagem traz os dois nomes.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000202 | S.stercoralis | S.stercoralis ✓ | S.stercoralis ✓ | img_01_S_stercoralis.png |
| 2 | 000002_00000566 / 000002_00000319 | Impurities | Impurities ✓ | Impurities ✓ | img_02_Impurities.png |
| 3 | 000001_00000126 | S.stercoralis | S.stercoralis ✓ | S.stercoralis ✓ | img_03_S_stercoralis.png |
| 4 | 000002_00001087 / 000002_00000840 | Impurities | Impurities ✓ | Impurities ✓ | img_04_Impurities.png |
| 5 | 000001_00000181 | S.stercoralis | S.stercoralis ✓ | S.stercoralis ✓ | img_05_S_stercoralis.png |
| 6 | 000002_00001251 / 000002_00001004 | Impurities | Impurities ✓ | Impurities ✓ | img_06_Impurities.png |
| 7 | 000001_00000030 | S.stercoralis | S.stercoralis ✓ | S.stercoralis ✓ | img_07_S_stercoralis.png |
| 8 | 000002_00000795 / 000002_00000548 | Impurities | Impurities ✓ | Impurities ✓ | img_08_Impurities.png |
| 9 | 000001_00000029 | S.stercoralis | Impurities ✗ | S.stercoralis ✓ | img_09_S_stercoralis.png |
| 10 | 000002_00001083 / 000002_00000836 | Impurities | Impurities ✓ | Impurities ✓ | img_10_Impurities.png |
