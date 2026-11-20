# Heatmaps — aid

- **Nosso:** MO-DEHB hybrid c00001 f45 — `/exps/moliveira/heatmaps_src/artifacts/modehb_hybrid_C01_2049/aid/state/seed42/candidates/c00001/f45/encoder.pt` — 23,408 params — κ teste (SVM original) 0.675 (recalculado aqui: 0.665). Sem TTA (a mesma leitura do κ).
- **FM:** AIMv2-3B/14 (448) — `apple/aimv2-3B-patch14-448` — 2,723,017,728 params — κ teste 0.979 (benchmark, split 1, 100%). Por que este FM: melhor FM da base no benchmark. Entrada 448 px, crop_pct 1.0, sem TTA, fp16 nos mapas.
- **Split:** teste do split 1. **Imagens:** 10 da interseção dos dois conjuntos de teste (1502 imagens), estratificadas por classe com seed 42 (≤10 classes: rodadas de 1 imagem por classe; >10 classes: 10 classes sorteadas, 1 imagem cada), escolhidas sem olhar predição nem κ. As mesmas imagens para os dois modelos.
- **Heatmap:** mapa de evidência do SVM linear do teste (`core.svm.fit_svm`, C=100, sem scaler, ovo, ajustado no treino do split 1). w_c = soma das linhas ovo a favor da classe predita menos as linhas contra; mapa = Σ_k w_c,k · f_k(x,y). Nosso: f = saída da conv 1×1 do head antes do pooling (pesos dos blocos de pooling somados por canal). ViT: f = tokens de patch após a norma final (AIMv2 e I-JEPA: o SVM usa a média desses tokens; DINOv2: o SVM usa o CLS, os patches estão no mesmo espaço). ConvNeXt: f = último estágio com a LayerNorm do pooler por posição. Cada mapa normalizado 0–1 por imagem, redimensionado (bilinear) e sobreposto (jet, alpha 0,45) na imagem como cada modelo a vê (nosso: 224 px; FM: o recorte do timm). ✓/✗ = predição do SVM certa/errada.
- **κ recalculado ≠ tabela:** o checkpoint veio da Jaci (torch 2.7) e aqui roda torch 2.2; a diferença vem das features, o protocolo do SVM é o mesmo.
- **Script:** `/dados/home/moliveira/presentation_results_cluster_november/figs/heatmaps/make_heatmaps.py` (passos ours_list → fm → ours → plot).

| # | imagem (benchmark / nossa) | classe | nosso | FM | arquivo |
|---|---|---|---|---|---|
| 1 | 000005_00000251 | Bridge | Bridge ✓ | Bridge ✓ | img_01_Bridge.png |
| 2 | 000009_00000392 | DenseResidential | DenseResidential ✓ | DenseResidential ✓ | img_02_DenseResidential.png |
| 3 | 000010_00000079 | Desert | Desert ✓ | Desert ✓ | img_03_Desert.png |
| 4 | 000011_00000179 | Farmland | Farmland ✓ | Farmland ✓ | img_04_Farmland.png |
| 5 | 000016_00000086 | Mountain | Mountain ✓ | Mountain ✓ | img_05_Mountain.png |
| 6 | 000017_00000175 | Park | Bridge ✗ | Park ✓ | img_06_Park.png |
| 7 | 000018_00000223 | Parking | DenseResidential ✗ | Parking ✓ | img_07_Parking.png |
| 8 | 000022_00000187 | RailwayStation | Industrial ✗ | RailwayStation ✓ | img_08_RailwayStation.png |
| 9 | 000025_00000023 | School | Viaduct ✗ | School ✓ | img_09_School.png |
| 10 | 000028_00000197 | Stadium | Stadium ✓ | Pond ✗ | img_10_Stadium.png |
