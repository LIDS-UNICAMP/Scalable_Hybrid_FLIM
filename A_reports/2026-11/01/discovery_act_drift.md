# A8: quanto o backprop afasta as ativações do SPiFiL cru (01/11/2026)

**Resposta curta:** o backprop mexe bastante no meio da rede (L2/L3) e quase nada nas pontas (L1 e L4 1×1).
O que ele faz não é "levar ativação para o fundo" no sentido absoluto. Ele **apaga o contraste objeto × fundo
que o SPiFiL trazia**: na L3, o objeto tinha ~2,9× a energia do fundo e passa a ter ~1×.
Não há lei clara "mais deriva = mais κ". E, no espaço dos kernels, o SPiFiL anda **mais** que o aleatório, não menos.

## Como medi

- **RAW (cru):** camadas `<k` do ckpt treinado de L(k−1) + conv k crua de `L<k>_grow`. **TRAINED:** `L<k>/checkpoints/best_kappa.ckpt`.
  **FINAL:** conv1..k do ckpt mais fundo (com `layerwise_backprop` as camadas velhas continuam andando).
- 32 imagens de **validação** por ponto (fatia fixa); CPU, 4 threads, sem SVM. Pontos `<ds>_split{1,2,3}_pct50`.
- Por pixel: cosseno entre o vetor de canais cru e o treinado; mudança relativa `‖A_tr−A_raw‖/‖A_raw‖`;
  razão de energia `média ‖a‖²` dentro / fora do objeto.
- **Máscara:** a do dataset (`masks/`) e a sem rótulo do `recon_bg_weight` (LAB > 0,04 da mediana da borda).
  **As duas saem idênticas**: as imagens já vêm com fundo constante, então só reporto a GT.
  O objeto ocupa só ~2–4 % da imagem, por isso "fora" é quase tudo.
- Por ponto: `artifacts/spifil_growth/<família>/<ponto>/act_drift/metrics.csv` e `act_maps.png`.
- Scripts: `scratchpad/a8/act_drift.py` (por ponto) e `scratchpad/a8/summary.py` (junta tudo).

## 1. Deriva por camada (TRAINED, média de 9 pontos)

`rel` = mudança relativa da ativação (0 = igual, 1 = do tamanho do próprio sinal). `E in/out` = energia dentro ÷ fora.

| família | camada | cos dentro | cos fora | rel dentro | rel fora | E in/out cru → treinado | kernel cos | kernel rel |
|---|---|---|---|---|---|---|---|---|
| bp01w64c1280 | L1 | 0,90 | 0,99 | 0,33 | 0,18 | 0,61 → 0,61 | 1,00 | 0,12 |
| | L2 | 0,84 | 0,89 | 0,54 | 0,49 | **2,02 → 1,19** | 0,59 | 0,91 |
| | L3 | 0,71 | 0,73 | 0,74 | 0,90 | **2,86 → 0,99** | 0,54 | 1,17 |
| | L4 1×1 | 0,93 | 0,96 | 0,45 | 0,41 | 0,88 → 0,66 | 0,97 | 0,24 |
| bp01w64 | L1 | 0,90 | 0,99 | 0,33 | 0,18 | 0,61 → 0,62 | 1,00 | 0,12 |
| | L2 | 0,85 | 0,90 | 0,53 | 0,48 | **2,02 → 1,19** | 0,59 | 0,93 |
| | L3 | 0,71 | 0,74 | 0,75 | 0,88 | **2,95 → 0,96** | 0,58 | 1,07 |
| r1 (SPiFiL, 24/32/48) | L2 | 0,80 | 0,83 | 0,63 | 0,73 | 2,11 → 0,79 | 0,51 | 1,07 |
| | L3 | 0,72 | 0,75 | 0,80 | 1,05 | 2,27 → 0,84 | 0,50 | 1,22 |
| r1_rand (aleatório, 24/32/48) | L2 | 0,79 | 0,68 | 0,75 | 1,26 | 2,72 → 1,43 | 0,83 | 0,55 |
| | L3 | 0,88 | 0,85 | 0,60 | 0,81 | 2,10 → 1,43 | 0,82 | 0,59 |

**FINAL vs TRAINED (bp 0,1):** as camadas velhas andam mais um pouco depois de congeladas "pela metade":
no c1280, `rel` da L2 vai 0,49 → 0,58 e o da L3 0,89 → 0,95. No r1/r1_rand (sem backprop nas velhas), FINAL = TRAINED.

## 2. L4 1×1×1280: filtros SPiFiL × preenchimento aleatório

| canais | cos dentro | rel dentro | rel fora | E in/out cru → treinado | kernel cos | kernel rel |
|---|---|---|---|---|---|---|
| SPiFiL (100–400) | 0,89 | 0,56 | 0,41 | 0,84 → **0,42** | 0,97 | 0,24 |
| aleatórios (resto) | 0,93 | 0,42 | 0,41 | 0,94 → 0,73 | 0,97 | 0,24 |

Os kernels dos dois grupos andam igual; quem perde mais o objeto são os canais SPiFiL.

## 3. Deriva × κ de validação (Spearman, deriva do modelo FINAL, média das camadas)

| recorte | n | ρ(rel ativação, κ) | ρ(rel kernel, κ) |
|---|---|---|---|
| todos | 36 | −0,21 (p 0,21) | +0,08 (p 0,63) |
| bp01w64c1280 | 9 | +0,52 (p 0,15) | **−0,80 (p 0,01)** |
| bp01w64 | 9 | **−0,77 (p 0,02)** | +0,27 (p 0,49) |
| r1 | 9 | −0,45 (p 0,22) | +0,27 (p 0,49) |
| r1_rand | 9 | +0,25 (p 0,52) | +0,13 (p 0,73) |
| eggs | 12 | **+0,65 (p 0,02)** | −0,24 (p 0,46) |
| protozoan | 12 | **−0,79 (p 0,002)** | +0,58 (p 0,05) |

O sinal troca de família para família e de dataset para dataset. Não existe lei única; o que pesa é o dataset
(larvae deriva ~1,5× mais que eggs/protozoan em todas as famílias).

## 4. Eixo da escala do backprop (g20, só os pontos já terminados, FINAL)

| ponto | bp 0,1 | bp 0,3 | bp 1,0 |
|---|---|---|---|
| eggs s1: rel ativação | 0,40 | 0,43 | 0,56 |
| eggs s1: rel kernel | 0,63 | 0,69 | 0,84 |
| larvae s1: rel ativação | 0,79 | 0,79 | 0,92 |
| larvae s1: rel kernel | 0,48 | 0,39 | 0,51 |

Mais backprop → mais deriva (claro em eggs; quase plano até 0,3 em larvae). A L1 é a que mais sente:
`rel` 0,21 → 0,45 em eggs. **κ ainda não existe** (`results/svm_disc_g20_*_val_gem.csv` não saiu),
então a lei "deriva × bp × κ" fica para quando esses CSVs aparecerem: é só rodar o `summary.py` com as famílias g20.

## Ressalvas

- r1 × r1_rand não é par perfeito: o r1_rand treinou com `recon_bg_weight 0.1`, o r1 com 1,0.
- κ por estágio só existe para L3/L4 nos CSVs `_val`; L1/L2 ficam sem κ no `metrics.csv`.
- O κ do `run_metadata.json` dos g20 (≈0,51) não é κ de SVM; não usei.
