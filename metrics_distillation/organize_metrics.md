# Organize Metrics — SVM Results

Resultados SVM @ 100% dados de treino (média ± std sobre splits de validação cruzada).
Métricas: F1 macro, Cohen's Kappa (κ), Acurácia (Acc).

---

<table>
<thead>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="3">Eggs</th>
    <th colspan="3">Larvae</th>
    <th colspan="3">Protozoan</th>
  </tr>
  <tr>
    <th>F1</th><th>κ</th><th>Acc</th>
    <th>F1</th><th>κ</th><th>Acc</th>
    <th>F1</th><th>κ</th><th>Acc</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>I-JEPA</td>
    <td><b><span style="font-size:1.2em">0.96 ± 0.01</span></b></td><td><b><span style="font-size:1.2em">0.96 ± 0.01</span></b></td><td><b><span style="font-size:1.2em">0.96 ± 0.01</span></b></td>
    <td><b><span style="font-size:1.2em">0.97 ± 0.00</span></b></td><td><b><span style="font-size:1.2em">0.95 ± 0.01</span></b></td><td><b><span style="font-size:1.2em">0.98 ± 0.01</span></b></td>
    <td>0.88 ± 0.00</td><td><b><span style="font-size:1.2em">0.89 ± 0.00</span></b></td><td>0.89 ± 0.01</td>
  </tr>
  <tr>
    <td>FLIM</td>
    <td>0.94 ± 0.01</td><td>0.88 ± 0.02</td><td>0.93 ± 0.01</td>
    <td>0.97 ± 0.00</td><td>0.87 ± 0.01</td><td>0.97 ± 0.00</td>
    <td><b><span style="font-size:1.2em">0.91 ± 0.02</span></b></td><td>0.85 ± 0.03</td><td><b><span style="font-size:1.2em">0.91 ± 0.02</span></b></td>
  </tr>
  <tr>
    <td>Distillation FLIM</td>
    <td>0.17 ± 0.04</td><td>0.07 ± 0.04</td><td>0.17 ± 0.03</td>
    <td>0.92 ± 0.01</td><td>0.85 ± 0.02</td><td>0.91 ± 0.02</td>
    <td>0.25 ± 0.01</td><td>0.25 ± 0.01</td><td>0.28 ± 0.00</td>
  </tr>
  <tr>
    <td>FLIM + readCONV (next_layers 1×1)</td>
    <td>0.91 ± 0.01</td><td>0.91 ± 0.01</td><td>0.92 ± 0.02</td>
    <td>0.96 ± 0.01</td><td>0.92 ± 0.01</td><td>0.96 ± 0.01</td>
    <td>0.85 ± 0.00</td><td>0.86 ± 0.01</td><td>0.86 ± 0.01</td>
  </tr>
  <tr>
    <td>FLIM + readCONV (3×3 BN2d)</td>
    <td>0.91 ± 0.01</td><td>0.90 ± 0.01</td><td>0.91 ± 0.02</td>
    <td>0.97 ± 0.01</td><td>0.94 ± 0.01</td><td>0.97 ± 0.01</td>
    <td>0.82 ± 0.02</td><td>0.83 ± 0.01</td><td>0.83 ± 0.02</td>
  </tr>
  <tr>
    <td>Distillation FLIM (1×1 BN2d)</td>
    <td>0.09 ± 0.00</td><td>0.00 ± 0.00</td><td>0.11 ± 0.00</td>
    <td>0.47 ± 0.00</td><td>0.00 ± 0.00</td><td>0.50 ± 0.00</td>
    <td>0.11 ± 0.00</td><td>0.00 ± 0.00</td><td>0.14 ± 0.00</td>
  </tr>
  <tr>
    <td>FLIM + readCONV (1×1 BN2d, one layer)</td>
    <td>0.42 ± 0.06</td><td>0.30 ± 0.05</td><td>0.74 ± 0.01</td>
    <td>0.53 ± 0.34</td><td>0.20 ± 0.51</td><td>0.53 ± 0.34</td>
    <td>0.31 ± 0.06</td><td>0.20 ± 0.04</td><td>0.44 ± 0.06</td>
  </tr>
  <tr>
    <td>FLIM + readCONV (2L 1×1 BN2d)</td>
    <td>0.84 ± 0.01</td><td>0.82 ± 0.01</td><td>0.86 ± 0.01</td>
    <td>0.94 ± 0.00</td><td>0.88 ± 0.01</td><td>0.95 ± 0.00</td>
    <td>0.80 ± 0.03</td><td>0.79 ± 0.02</td><td>0.82 ± 0.03</td>
  </tr>
  <tr>
    <td>Distil 2L 1×1 FLIM init (com norm ImageNet)</td>
    <td>0.80 ± 0.07</td><td>0.77 ± 0.07</td><td>0.84 ± 0.05</td>
    <td>0.79 ± 0.27</td><td>0.63 ± 0.44</td><td>0.84 ± 0.17</td>
    <td>0.67 ± 0.04</td><td>0.56 ± 0.07</td><td>0.70 ± 0.05</td>
  </tr>
  <tr>
    <td>FLIM frozen + conv 1×1 BN2d (proj 1280)</td>
    <td>0.77 ± 0.02</td><td>0.79 ± 0.02</td><td>0.75 ± 0.02</td>
    <td>0.95 ± 0.01</td><td>0.91 ± 0.02</td><td>0.95 ± 0.01</td>
    <td>0.74 ± 0.03</td><td>0.75 ± 0.01</td><td>0.69 ± 0.03</td>
  </tr>
  <tr>
    <td>FLIM init + 1×1 BN2d, sem norm (encoder 48)</td>
    <td>0.27 ± 0.17</td><td>0.24 ± 0.23</td><td>0.28 ± 0.16</td>
    <td>0.92 ± 0.01</td><td>0.84 ± 0.01</td><td>0.92 ± 0.01</td>
    <td>0.17 ± 0.05</td><td>0.07 ± 0.06</td><td>0.19 ± 0.04</td>
  </tr>
  <tr>
    <td>FLIM init + 1×1 BN2d, sem norm (proj 1280, best ckpt)</td>
    <td>0.55 ± 0.11</td><td>0.44 ± 0.13</td><td>0.77 ± 0.13</td>
    <td>0.92 ± 0.02</td><td>0.85 ± 0.04</td><td>0.93 ± 0.02</td>
    <td>0.45 ± 0.14</td><td>0.33 ± 0.15</td><td>0.58 ± 0.10</td>
  </tr>
  <tr>
    <td>FLIM init + 1×1 BN2d, sem norm (proj 1280, last ckpt)</td>
    <td>0.53 ± 0.09</td><td>0.42 ± 0.10</td><td>0.75 ± 0.03</td>
    <td>0.93 ± 0.01</td><td>0.86 ± 0.03</td><td>0.93 ± 0.01</td>
    <td>0.29 ± 0.05</td><td>0.16 ± 0.03</td><td>0.47 ± 0.04</td>
  </tr>
  <tr>
    <td>FLIM init + 2L 1×1, sem norm (encoder 48)</td>
    <td>0.57 ± 0.21</td><td>0.57 ± 0.21</td><td>0.57 ± 0.20</td>
    <td>0.95 ± 0.00</td><td>0.90 ± 0.00</td><td>0.95 ± 0.01</td>
    <td>0.63 ± 0.08</td><td>0.59 ± 0.08</td><td>0.60 ± 0.08</td>
  </tr>
  <tr>
    <td>FLIM init + 2L 1×1, sem norm (proj 1280)</td>
    <td>0.86 ± 0.04</td><td>0.84 ± 0.05</td><td>0.89 ± 0.02</td>
    <td>0.96 ± 0.01</td><td>0.92 ± 0.01</td><td>0.96 ± 0.00</td>
    <td>0.77 ± 0.05</td><td>0.74 ± 0.07</td><td>0.80 ± 0.03</td>
  </tr>
  <tr>
    <td>LeJEPA</td>
    <td>0.43 ± 0.06</td><td>0.31 ± 0.06</td><td>0.49 ± 0.12</td>
    <td>0.53 ± 0.27</td><td>0.27 ± 0.29</td><td>0.60 ± 0.18</td>
    <td>0.26 ± 0.06</td><td>0.10 ± 0.06</td><td>0.35 ± 0.10</td>
  </tr>
</tbody>
</table>

---

## Notas

- **I-JEPA**: ViT-H/14 pré-treinado (`facebook/ijepa_vith14_1k`), ~632M parâmetros, frozen.
- **FLIM**: encoder CNN ~60K parâmetros construído via superpixels sem backpropagação.
- **Distillation FLIM**: FLIM treinado com KD (conv readout durante treino), conv **descartada** no eval. Embedding: `(B, 48)` bruto do encoder FLIM.
- **FLIM + readCONV (next_layers 1×1)**: FLIM treinado com KD, proj head mantida no eval (4× convs 1×1: 48→128→256→512→1280, GAP). Embedding: `(B, 1280)`.
- **FLIM + readCONV (3×3 BN2d)**: FLIM treinado com KD, proj head mantida no eval (1× conv 3×3 + BN2d: 48→1280, GAP). Embedding: `(B, 1280)`. Arquitetura `OneLayerConvDistillationProjectionHead`.
- **Distillation FLIM (1×1 BN2d)**: FLIM treinado com KD (conv 1×1 BN2d one-layer durante treino), conv **descartada** no eval. Embedding: `(B, 48)` bruto. Colapso total (κ=0.00 em todos os datasets).
- **FLIM + readCONV (1×1 BN2d, one layer)**: FLIM treinado com KD, proj head 1×1 BN2d mantida no eval (1× conv 1×1: 48→1280, BN2d, GELU, GAP). Embedding: `(B, 1280)`. Alta variância em Larvae (std ≈ 0.34), modelo instável.
- **FLIM + readCONV (2L 1×1 BN2d)**: FLIM treinado com KD, proj head 2-layer mantida no eval (48→256→1280, BN2d, GELU, GAP). Inicialização `trunc_normal`. Embedding: `(B, 1280)`.
- **Distil 2L 1×1 FLIM init (com norm ImageNet)**: Igual ao anterior mas inicialização `flim` (pesos do algoritmo FLIM). Treinado **com** normalização ImageNet. Alta variância em Larvae.
- **FLIM frozen + conv 1×1 BN2d**: encoder FLIM **congelado** + proj head 1×1 BN2d treinada separadamente com KD. Embedding: `(B, 1280)`. Média sobre 6 splits (2 checkpoints × 3 folds).
- **FLIM init + 1×1 BN2d, sem norm**: Inicialização `flim`, treinado **sem** normalização ImageNet. Duas variantes: encoder48 (embedding `(B, 48)`) e proj1280 (embedding `(B, 1280)`). Resultados avaliados em best checkpoint e last checkpoint do treino.
- **FLIM init + 2L 1×1, sem norm**: Mesma configuração 2L porém sem normalização ImageNet. Duas variantes: encoder48 e proj1280.
- **LeJEPA**: treinado do zero nos datasets de parasitas, inicialização `trunc_normal`.
- Avaliação via SVM sobre embeddings brutos `student_emb (B, 48)` do encoder FLIM.
