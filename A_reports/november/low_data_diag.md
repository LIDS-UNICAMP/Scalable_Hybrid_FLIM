# Diagnóstico 1–5% — disc_bp01w64c1280 vs I-JEPA (A17, 02/11)

## 1. Imagens (iguais nos splits 1/2/3; JSON `splits_incremental/split{s}/data_descriptor_perc{p}.json`, config.py:193-201; lido em core/data/parasite_dataset.py:57,96)
| ds | 1% train/val/test | 5% train/val/test | 50% train |
|---|---|---|---|
| eggs (9 cl) | 24 / 2531 / 2557 | 124 / 2431 / 2557 | 1276 |
| larvae (2 cl) | 17 / 1740 / 1757 | 87 / 1670 / 1757 | 878 |
| protozoan (7 cl) | 46 / 4736 / 4786 | 235 / 4547 / 4786 | 2390 |
- O AE treina só no `train`; o SVM é ajustado nesse mesmo train (eggs 1%: ~2–3 por classe) e testado em ~2.5k.
- SPiFiL: 1 imagem por classe do artigo (flim/spifil.py:525-531, sem a impureza): eggs 8, larvae 2, protozoan 6. É o mesmo número em qualquer pct → SPiFiL não é o que encolhe.

## 2. Passos (config.yaml: batch_size 32, max_epochs 200, EarlyStopping patience 50 em probe/svm_kappa; epoch do best_kappa.ckpt)
- 1%: 1 batch/época (24, 17, 46→2) × ~199 épocas ≈ 200 passos (protozoan ≈ 400). Best quase sempre na época 199 = ainda melhorando.
- 50%: 40 / 28 / 75 batches × ~190 (L4 ~66–158) ≈ 5–14k passos.
- → 1% treina **~30–40× menos passos**, e o ckpt é selecionado por κ do SVM no train minúsculo (seleção ruidosa).

## 3. I-JEPA `pretrained_pct`
eval/svm_variants/svm_ijepa.py:275-278,286: encoder congelado `facebook/ijepa_vith14_1k` (ImageNet-1k, ViT-H/14); `pretrained_pct` só define o train do SVM. Nada é pré-treinado no parasito. → Mesmos rótulos (justo no SVM), mas o I-JEPA tem ~1,2M imagens de pré-treino vs nossas 17–46. A comparação a 1–5% mede "prior de ImageNet vs nenhum prior".

## 4. Por camada a 1/5%
**Não dá com o que está aqui**: todos os `results/svm_disc_bp01w64c1280_*.csv` só têm pct=50; os `*_pcts_*` estão em Jaci (A_reports/november/jaci/). Sem números por camada.

## 5. Diagnóstico
O teto a 1–5% é de dados, não de arquitetura: o AE vê 17–46 imagens e ~200 passos, e o SVM tem 2–3 rótulos/classe contra um I-JEPA com prior de ImageNet. O protozoan a 5% já empata (0.728 vs 0.725), com 235 imagens → o limiar fica em ~100–200 imagens de treino.
Causas mais prováveis e o teste barato de cada uma:
1. **Poucos passos (≈200) / best na última época** → fixar ~8k passos em 1% (max_epochs 8000 ou repetir o dataset) num YAML irmão; se o κ sobe, era subtreino.
2. **SVM com poucos rótulos + seleção de ckpt ruidosa** → avaliar o *mesmo* ckpt de 1% com o SVM ajustado no train de 50% (só os rótulos crescem); e o inverso (ckpt de 50% + SVM de 1%). Isso separa o encoder do classificador. Testar também `SPiFiL só (época 0)` vs best.
3. **Overfit do backprop/AE em 17–46 imagens** → comparar a época 0 (só SPiFiL, sem backprop) com o best a 1%, e `layerwise_backprop: 0` vs 0.1; usar o AE só nas imagens não rotuladas do val (sem rótulo: protocolo SSL ok) para ver se mais imagens sem rótulo fecham o gap.
(SPiFiL é descartado: usa sempre 2–8 imagens, independente do pct.)
