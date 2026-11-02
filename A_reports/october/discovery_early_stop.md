# Critério de parada antecipada — análise retrospectiva (A4, 2026-11-02)

**Resposta curta:** quase todo o tempo de treino vem do cosine de 200 épocas, não da espera do early stop. Uma patience menor quase não economiza nas L1–L3, porque nelas a loss de validação melhora até o fim. A economia grande (~70%) exige `max_epochs 60` com o cosine reescalado. Isso precisa de **um** run de validação no κ antes de virar regra.

## 1. O que o treino faz hoje (corrige a premissa do pedido)

- Em **todas** as famílias disc_* e g20 o config tem `svm_probe: false` (por exemplo, `artifacts/spifil_growth/disc_g20_e_bp10/eggs_split1_pct50/L4/config.yaml`).
- Com `svm_probe` desligado, `monitor_spec()` (`methods/autoencoder/autoencoder_flim_module.py:587-590`) troca o monitor do ModelCheckpoint **e** do EarlyStopping para `probe/val_recon_loss`, em modo `min`. O yaml diz `probe/svm_kappa`, mas essa chave nunca é logada.
- Consequências:
  - **Não existe κ por época em nenhum run, e a sonda SVM custa zero.** Logo, `check_val_every_n_epoch` para aliviar a sonda não se aplica.
  - A regra real hoje é **patience 50 na val recon loss**, com `min_delta 0` e `max_epochs 200`.
  - O `best_kappa.ckpt` é a época de **menor val recon loss**. Conferido: o e_k3 s1 tem ckpt na época 105, que é o argmin; o c1280 s2 tem ckpt na 145, também o argmin.
  - O "confundidor da sonda" que eu levantei no quadro (o embed_mode escolhendo o ckpt) **não existe**. O ckpt nunca dependeu da leitura.
- **Sem κ por época, a perda de κ não pode ser simulada.** Uso dois substitutos:
  - **mesmo ckpt %:** fração de estágios em que a regra escolhe a mesma época de hoje. Nesses casos o κ é idêntico, ou seja, a perda é 0.
  - **ΔLoss relativa:** (val loss do ckpt da regra − val loss do ckpt de hoje) / val loss de hoje.
  - Para comparação: re-treinar o mesmo config move o κ em ~0,011 por split (flatten) e ~0,032 (avg). Re-treinar o pipeline inteiro move a média de 3 splits em ~0,018.

## 2. Estágios (752 estágios terminados, 43 famílias)

Fontes:
- curvas por época do W&B (`probe/val_recon_loss`), via cache `scratchpad/a4/curve_cache.json`;
- tempo de cada run: `_runtime` do W&B dividido pelo número de épocas.

Script: `scratchpad/a4/es_sim.py`. Dados por estágio: `scratchpad/a4/es_stages.json`.

| camada | n | época do best (mediana) | épocas rodadas (mediana) | s/época (mediana, com GPU compartilhada) |
|---|---|---|---|---|
| L1 | 219 | 198 | 200 | 12,1 |
| L2 | 213 | 198 | 200 | 13,3 |
| L3 | 207 | 193 | 200 | 13,4 |
| L4 | 98 | **137** | 188 | 12,2 |

- **L1–L3:** o best fica no fim, porque a lr do cosine só chega perto de zero nas últimas épocas e a loss ainda cai 1e-5 a 1e-4. O early stop quase nunca dispara.
- **L4:** o best vem antes (~137), e o early stop já corta parte do treino.

## 3. Regras simuladas (todas as camadas; por camada no item 4)

| regra (sobre a val recon loss) | tempo poupado | mesmo ckpt de hoje | ΔLoss rel., mediana | ΔLoss rel., p90 | \|época do ckpt − best\|, p90 |
|---|---|---|---|---|---|
| hoje: patience 50 | 0% | 100% | 0 | 0 | 0 |
| patience 30 | 2,0% | 98% | 0 | 0 | 0 |
| **patience 20** | **4,0%** | **92%** | 0 | 0 | 0 |
| patience 15 | 7,0% | 82% | 0 | 3,2e-5 | 31 |
| patience 10 | 18,8% | 58% | 0 | 3,5e-4 | 102 |
| patience 20 + min_delta 1e-5 | 10,1% | 40% | 2,8e-6 | 3,3e-5 | 38 |
| patience 20 + min_delta 5e-5 | 36,6% | 3% | 1,2e-4 | 2,3e-4 | 105 |
| max_epochs 100 (truncado, mesmo cosine) | 48,5% | 4% | 3,0e-4 | 4,1e-4 | 102 |
| max_epochs 60 (truncado, mesmo cosine) | 69,1% | 0% | 7,5e-4 | 1,1e-3 | 142 |
| val loss melhora <1% em 10 épocas | 91,4% | 0% | 3,0e-3 | 4,4e-3 | 185 |
| val loss melhora <0,1% em 10 épocas | 85,2% | 0% | 1,8e-3 | 2,3e-3 | 177 |

- A regra (b), "min_delta 0,005", foi traduzida para a loss. Na escala do κ ela não existe, porque não há κ por época. Na loss, um delta absoluto de 0,005 é ~1% de 0,51 e equivale à linha "melhora <1% em 10 épocas": para tudo por volta da época 10–20.
- **Truncar** (`max_epochs` 60/100 mantendo o cosine de 200) é pessimista. Um run de verdade com `max_epochs 60` reescala o cosine e chega à lr baixa na época 60. O ΔLoss real deve ser menor que o da tabela. **Só um run diz o κ.**

## 4. Por camada (patience; tempo poupado / mesmo ckpt)

| regra | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| patience 30 | 0% / 100% | 0% / 100% | 1,2% / 98% | 10,3% / 93% |
| patience 20 | 0% / 100% | 0% / 100% | 3,7% / 87% | **18,9% / 71%** (ΔLoss p90 4,7e-5) |
| patience 15 | 1,0% / 98% | 0,1% / 100% | 8,3% / 68% | 28,8% / 42% |
| patience 10 | 7,2% / 81% | 3,1% / 92% | 29,9% / 24% | 50,8% / 10% |

- **Patience por camada não ajuda nas L1–L3:** ali não há o que cortar. **Na L4, patience 20 corta ~19%** e mantém o mesmo ckpt em 71% dos estágios.

## 5. Recomendação

1. **Regra que cumpre o critério pedido, com o substituto disponível: patience 20 em todas as camadas.**
   - Economia: 4% no geral e ~19% na L4.
   - Em 92% dos estágios o ckpt é idêntico ao de hoje (perda de κ = 0).
   - Nos outros 8%, a ΔLoss relativa fica com p90 ≤ 5e-5, muito abaixo de qualquer variação já medida, e o κ não pode ter perda real distinguível do ruído.
   - **É segura, mas economiza pouco.**
2. **A alavanca de verdade é `max_epochs 60` com o cosine reescalado.** Ela economizaria ~69% e é a única regra que encurta as L1–L3.
   - **Não dá para garantir o κ retrospectivamente.** Não há κ por época, e o truncamento não reproduz um cosine de 60.
   - **Teste proposto, de custo 1 família:** bp01w64c1280 com `max_epochs: 60`, nos 3 datasets, a 50%, 3 splits.
   - **Critério de aceitação:** gem L4 com média de 3 splits dentro de ~0,018 (variação do pipeline) da base de 200 épocas, ou seja, eggs 0,899, larvae 0,910 e protozoan 0,875.
   - Indício a favor: na 1ª análise (Oct-30), 200 contra 1.000 épocas mudaram a loss mínima só 1,6e-4, e a loss chega a 1e-3 do mínimo nas épocas 6–36.
3. **Custo da sonda:** zero (desligada). `check_val_every_n_epoch` só economizaria o passe de val (forward da val inteira por época), que não é medido em separado. Para quantificar: comparar s/época com `check_val_every_n_epoch: 5` num smoke de 20 épocas.
