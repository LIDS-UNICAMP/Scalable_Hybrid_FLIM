# Plano: curvas de aprendizado (Protozoan, Eggs, Larvae — bases IMPURAS), athena

**Resumo:** 2 métodos (híbrido SPiFiL com backprop + seu controle aleatório; DINOv2/I-JEPA congelados), 3 bases, splits 1–3. DINOv2 e I-JEPA já estão quase prontos (só falta 10%). O híbrido roda do zero.
Métrica: κ de **validação**; teste sai em coluna/CSV separado. Nada de `_subgroup`, `helminth_*` ou `protozoan_cysts`.
Repositório: `Scalable_Hybrid_FLIM_discovery` (ele lê `datasets/organized/<base>`, que é o impuro).

## Bloqueio: 10% não passa no schema
`core/constants.py` tem `PERCENTAGES = [1, 5, 25, 50, 75, 100]`. O launcher recusa `10`. Os arquivos `perc10.json` existem.
Para ter 10% é preciso **uma linha** no `.py` (pôr 10 na lista). Não editei. Os YAMLs abaixo estão com 6 pcts; com a linha, é só acrescentar 10 .

## 1% é possível em todas as bases
Toda classe tem ≥1 imagem de treino a 1% nos 3 splits: Larvae 12 imagens (2 + 10), Eggs 96 (menor classe com 1), Protozoan 268 (menor com 3). Nenhum pct é inviável.

(O SPiFiL sem backprop saiu do plano por pedido do usuário: o braço SPiFiL é o híbrido com backprop.)

## Método 2 — híbrido oficial disc_bp01w64c1280_bg1
- Treino: `experiments/autoencoder/hyb_curva_parasitos.yaml` — **54 pontos** (3 bases × 3 splits × 6 pcts), cópia de `hyb_ham10000_oficial.yaml`.
- O runner growth **não** avalia (`evaluate.SVM: false`). κ/F1 por pct vêm de um eval depois do treino, como os `eval_svm_disc_bp01w64c1280_bg1_pcts_*`:
  - `experiments/autoencoder/eval_svm_hyb_curva_parasitos_val_gemttapwr.yaml` (54 jobs, validação — a métrica da curva)
  - `experiments/autoencoder/eval_svm_hyb_curva_parasitos_test_gemttapwr.yaml` (54 jobs, teste, separado)
  - Leitura: o decoder é descartado; só o encoder, GeM na última camada (1280-d) + TTA + whitening (`gem_tta_pw`, a convenção dos `*_bg1_pcts_*_gemttapwr`), estágio L4 → SVM oficial.
- Controle aleatório (mesma rede, mesmo treino, `random_layer_classic: true`, como o `_rand` do HAM): `hyb_curva_parasitos_rand.yaml` (54 pontos) + `eval_svm_hyb_curva_parasitos_rand_{val,test}_gemttapwr.yaml`.
  - Dry-run deles hoje diz "0 best_kappa.ckpt" — esperado, os pesos ainda não existem.
- Reuso: **não há nada**. Os pcts antigos do bg1 são de `helminth-*`/`protozoan-cysts` (puro). Não misturar.
- Memória/tempo: `GPU_CAP_MIB=6000`, `max_per_gpu: 4` (24 GB ≤ 25 GB) → 16 juntos. 200 épocas × 4 estágios; 1% leva minutos, 100% de Protozoan (~27 mil imagens) leva muitas horas. Estimativa grosseira: 1–2 dias de parede.

## Método 3 — DINOv2 e I-JEPA congelados + mesmo SVM
- Embeddings existem: `/exps/moliveira/benchmark/emb/dinov2-giant/` e `/exps/moliveira/benchmark/emb/ijepa_vitg16_22k/` (`Larvae.npz`, `Eggs.npz`, `Protozoan.npz`). Não há `ijepa_vith14`.
- Os números citados são do **dinov2-giant** a 50% (val κ médio: Larvae 0,968, Eggs 0,980, Protozoan 0,939).
- `results/benchmark/dinov2-giant.csv` e `ijepa_vitg16_22k.csv` já têm 1, 5, 25, 50, 75, 100 × 3 splits, com `val_kappa` e `test_kappa`. **Reusar.**
- Falta só 10% (CPU, sob o `svm.lock` existente). `common.svm` não olha a lista `PERCENTAGES`, então roda sem editar nada.

## Comandos (rodar de dentro de `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`)

Híbrido (treino, depois os dois evals, um por vez). Para o controle, o mesmo comando trocando `hyb_curva_parasitos` por `hyb_curva_parasitos_rand` (rodar depois, ou junto se couber: 4 jobs/GPU no total):
```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 PYTHONPATH=/exps/moliveira/queues/gpucap GPU_CAP_MIB=6000 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/hyb_curva_parasitos.yaml && for s in val test; do LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 PYTHONPATH=/exps/moliveira/queues/gpucap GPU_CAP_MIB=6000 /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -m experiments.ray.launch experiments/autoencoder/eval_svm_hyb_curva_parasitos_${s}_gemttapwr.yaml; done
```

DINOv2/I-JEPA, só 10% (CPU):
```
cd /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery && LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python -c "from benchmark.common import svm; [svm(m, d, 10) for m in ('facebook/dinov2-giant', 'facebook/ijepa_vitg16_22k') for d in ('Protozoan', 'Eggs', 'Larvae')]"
```

## Script do gráfico final (especificação)
- Entradas: `results/svm_hyb_curva_parasitos_val_gemttapwr.csv` e `..._rand_val_gemttapwr.csv` (estágio L4; aleatório tracejado), `results/benchmark/dinov2-giant.csv` e `ijepa_vitg16_22k.csv` (`val_kappa`).
- Filtrar só `Protozoan`, `Eggs`, `Larvae`.
- Figura: 3 painéis lado a lado (uma base cada), x = % de treino rotulado em escala log (ticks 1, 5, 10, 25, 50, 75, 100), y = κ de validação.
- Uma linha por método (híbrido SPiFiL, controle aleatório, DINOv2-giant, I-JEPA vitg16), ponto = média dos 3 splits, faixa = ± desvio-padrão.
- Teste: a mesma figura com `test_kappa`, em arquivo separado. Saída em `A_reports/november/curvas/`.
