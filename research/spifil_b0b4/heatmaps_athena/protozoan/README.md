# Heatmaps — protozoan

- **Nosso:** MO-DEHB hybrid c00001 f45 — `/exps/moliveira/heatmaps_src/artifacts/modehb_hybrid/protozoan/state/seed42/candidates/c00001/f45/encoder.pt` — 19,790 params — κ teste (SVM original) 0.798 (recalculado aqui: 0.797). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 0.968 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (5606 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **Parasitas:** o nosso modelo vê `pre/orig` (objeto recortado, nomes renomeados; mapeamento oficial `splits/split1_modehb_mapping.csv`, mesmo split 1); o FM vê `raw/orig` (campo inteiro). Por isso a coluna imagem traz os dois nomes.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000001_00000215 | E.coli | E.coli ✓ | E.coli ✓ | img_01_E_coli.png |
| 2 | 000002_00002159 / 000002_00000249 | E.histolytica | Impurities ✗ | E.histolytica ✓ | img_02_E_histolytica.png |
| 3 | 000003_00004626 / 000003_00001383 | E.nana | E.nana ✓ | E.nana ✓ | img_03_E_nana.png |
| 4 | 000004_00013340 / 000004_00001423 | Giardia | Giardia ✓ | Giardia ✓ | img_04_Giardia.png |
| 5 | 000005_00021724 / 000005_00000923 | I.butschlii | Impurities ✗ | I.butschlii ✓ | img_05_I_butschlii.png |
| 6 | 000006_00034045 / 000006_00000257 | B.hominis parasites | Impurities ✗ | B.hominis parasites ✓ | img_06_B_hominis_parasites.png |
| 7 | 000011_00030260 / 000007_00021718 | Impurities | Impurities ✓ | Impurities ✓ | img_07_Impurities.png |
| 8 | 000001_00000011 | E.coli | E.coli ✓ | E.coli ✓ | img_08_E_coli.png |
| 9 | 000002_00002529 / 000002_00000619 | E.histolytica | E.histolytica ✓ | E.histolytica ✓ | img_09_E_histolytica.png |
| 10 | 000003_00003595 / 000003_00000352 | E.nana | E.nana ✓ | E.nana ✓ | img_10_E_nana.png |
