# Revisão de código + investigação de resultados — FLIM classification

**Data:** 2026-07-22 · **git:** `bdbadc2` · **Escopo:** pipeline `sigmoid2l_` (FLIM + cabeça 2-layer sigmoid) e a comparação de três cabeças (SVM / MLP-ReLU / MLP-Sigmoid) sobre o mesmo encoder FLIM.

Revisão organizada em quatro frentes: **A1** verificação do pipeline, **A2** proveniência dos números, **B1** o paradoxo ReLU > Sigmoid, **B2** a curva do MLP. Cada afirmação vem com `arquivo:linha` ou um número.

---

## Sumário executivo

1. **O pipeline é, no essencial, o que você descreve** — arquitetura, init FLIM, LAB/`ift_lab`/`imagenet_norm=True`, cobertura 3×3×{1%,75%}, frozen **e** unfrozen, métricas f1/kappa/acc. Cinco desvios importam (§A1).
2. **A proveniência dos três braços confere com o disco** (§A2). Ressalva crítica confirmada: `unified_eval --model svm --init flim` avalia **0 runs** — não é a fonte do braço SVM+FLIM.
3. **O paradoxo se dissolve: a teoria está mal-aplicada.** A Sigmoid aqui é uma **camada oculta (48→24)**, não o plano de decisão — as duas cabeças decidem por `Linear→Softmax`. E `imagenet_norm=True` sobre LAB[0,1] **infla as features FLIM ~3×**, saturando **62% das unidades sigmoides** e reduzindo o gradiente ~4×. ReLU não tem teto de saturação e por isso vence (§B1).
4. **A curva "cai e recupera" é real só para o MLP unfrozen** (eggs, protozoan). Para o MLP frozen é refutada; para larvae não existe dip. O mecanismo da recuperação é o **fine-tuning do encoder**, não o MLP em si (§B2).

---

## A1 — Verificação do pipeline: **CONFIRMADO com 5 desvios**

Checklist do que você afirmou ter montado:

| Propriedade afirmada | Veredito | Evidência |
|:--|:--|:--|
| Entrada 3×200×200, LAB, `loader="ift_lab"`, `imagenet_norm=True` | ✅ CONFIRMADO | loader default `"ift_lab"` `src/data_modules/datasets/dataset.py:124`, hard-coded no módulo `src/modules/classification_flim_module.py:300`; LAB via pyift `dataset.py:41-47,175-176`; `image_size=200` `parasite_data_module_lejepa_splited.py:110`; `imagenet_norm=True` default `parasite_data_module_lejepa_splited.py:119` + `classification_flim_module.py:301` |
| Encoder FLIM, 3 blocos Conv+ReLU+MaxPool, init FLIM | ✅ CONFIRMADO | `build_encoder_from_arch` `src/models/models.py:110-149`; pesos FLIM obrigatórios `classification_flim_module.py:125-129`; `ENCODER_INITS=("flim",)` `:87` |
| conv1 3→24, conv2 24→32, conv3 32→48, todas 5×5, MaxPool 3×3 s2 | ✅ CONFIRMADO (nominal) | `ch24_32_48_a0.5_f5/eggs/train1/architecture.json` (kernel `[5,5,0]`, pool `[3,3,0]` s2) |
| protozoan 24→30→48 | ✅ CONFIRMADO (via override) | canais re-lidos dos bias files: conv2=30 em `ch24_32_48_a0.5_f5/protozoan/train1/models/conv2-bias.txt`; `override_arch_channels` `classification_flim_module.py:131-146`. **O arch nominal é 24→32→48** (`models.py:537-560`); os 30 vêm do peso real |
| Head: AvgPool(1)→flatten(48)→Linear(48,24)→Sigmoid→Linear(24,C)→Softmax | ✅ CONFIRMADO | `TwoLayerSigmoidHead` `src/models/models.py:346-378`; `hidden_dim=48//2=24` `:365` |
| 3 datasets × 3 splits × {1%,75%}, frozen **e** unfrozen | ✅ CONFIRMADO | grids `classification_flim_ray.py:59-60`; `--percentages 1 75` `:17-22`; freeze `classification_flim_module.py:148-150`, `--freeze-encoder` `:260`; **36 runs reais no disco** (18 unfrozen + 18 frozen `_frozen`) em `artifacts/classification_flim/`, sem lacunas |
| Métricas f1, kappa, acc | ✅ CONFIRMADO | `compute_metrics` → `{kappa,acc,f1}` `src/metrics/classification.py:47-56` |

### Os 5 desvios a registrar

1. **A loss é `NLLLoss` sobre `log(softmax)`, não CrossEntropy sobre logits.** `F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)` `classification_flim_module.py:172-176`. É o contraponto intencional do Softmax que já está **dentro** do head. Matematicamente equivale a CE.
2. **As métricas reportadas pelo módulo são de VALIDAÇÃO, não de teste.** Só há `on_validation_epoch_end` (`val/kappa|acc|f1`) `classification_flim_module.py:213-221`; **não existe `test_step`**, e `test_dataloader` (existe, `parasite_data_module_lejepa_splited.py:265`) nunca é chamado pelo treino. Só `best_val_kappa` é persistido no metadata. → é por isso que o braço Sigmoid precisou de um script de inferência à parte (§A2).
3. **`imagenet_norm=True` foi usado apesar de comentário no código dizendo que está errado** para entrada LAB[0,1]/FLIM (`src/data_modules/datasets/lejepa_dataset.py:42`). Nenhum run sobrescreveu. **Este desvio é o motor mecânico do parad), ver §B1.**
4. **protozoan** — o encoder realizado é 24→**30**→48 porque os canais são re-derivados dos bias files; casa com sua descrição só *depois* desse override.
5. **18 diretórios stub obsoletos** `sigmoid2l_frozenclasshead_*` (esquema de nome antigo, Jul 17, `checkpoints/` vazios, sem `run_metadata.json`) convivem com os 36 runs válidos. Não contam para cobertura; o `run_manifest.csv` atual lista só os 18 frozen (última invocação do driver).

---

## A2 — Proveniência dos três braços: **confere com o disco**

| Braço | Fonte | Cobertura real | Como foi gerado | Schema |
|:--|:--|:--|:--|:--|
| **SVM + FLIM** (frozen) | `data/reports_felipe/svm/report_svm_{ds}_split{N}_perc{p}.csv` (**54 CSVs**, 1 linha cada, frozen) → agregado `artifacts/normalized/svm_flim_aggregated.csv` (18 linhas) | 3 ds × 3 splits × **6 pcts** (1/5/25/50/75/100), sem lacunas | Raw **pré-computado pelo Felipe** (Mar/2025). Agregação por `scripts/normalize_reports.py::normalize_felipe_svm` `:80-148` (regex `:73-75`, filtra frozen `:102`, `cistos→protozoan` `:59-64`, média±std ddof=1 `:126-146`) | `test_accuracy/f1_weighted/cohen_kappa` (val_* = N/A) |
| **FLIM + MLP (ReLU)** (frozen+unfrozen) | `data/reports_felipe/flim_mlp/report_flim_mlp_{ds}_split{N}_perc{p}.csv` (**54 CSVs, 2 linhas cada** = frozen+unfrozen) | 3 ds × 3 splits × **6 pcts** × 2 modos, sem lacunas | **Pré-computado pelo Felipe** (Mar/2025). Nenhum script do repo regenera; `normalize_reports.py` **não** lê `flim_mlp/`. Números da comparação foram agregados ad-hoc | 17 colunas (inclui val_*, precision/recall) |
| **FLIM + MLP (Sigmoid)** (frozen+unfrozen) | checkpoints `artifacts/classification_flim/sigmoid2l_classhead_*` → `results/sigmoid2l_test_results.csv` (**34 linhas**) | 3 ds × 3 splits × **{1,75} só** × 2 modos = 36 esperados; **faltam 2**: larvae split1/split2 pct1 **unfrozen** (checkpoints vazios → pulados em `eval_sigmoid2l_test.py:64-66`) | **Gerado fresco** por `scripts/eval_sigmoid2l_test.py` (inferência no split de **teste** sobre `best_kappa.ckpt`, pois o módulo não tem test_step) | 9 colunas `test_accuracy/f1_weighted/cohen_kappa` |

**Ressalva crítica — CONFIRMADA (não refutada).** Os números "SVM + FLIM" **NÃO** vêm de `python -m src.evaluate.unified_eval --model svm --init flim`. Esse módulo avalia encoders **SSL LeJEPA** resolvidos via W&B (`src/evaluate/unified_eval.py:19,68-73`, `resolve_ssl_runs` em `wandb_resolver.py:211`), carregando `module.model.encoder` de checkpoints SSL `:186` — nunca `data/reports_felipe/svm/` nem `load_FLIM_encoder`. O filtro `--init flim` casa `initialization_type` `wandb_resolver.py:262`; os checkpoints locais são `trunc_normal` → interseção vazia → **0 runs**. O comando tmux impresso no `auto_report` (linhas 65-71) está portanto **mal-rotulado** como fonte; a fonte real são os CSVs do Felipe.

**Cobertura assimétrica a registrar:** SVM e ReLU têm a curva completa 1/5/25/50/75/100 em disco; a comparação só mostra 1% e 75%. O braço Sigmoid **só existe** em 1% e 75%.

---

## B1 — O paradoxo ReLU > Sigmoid: **a teoria está mal-aplicada**

**Sua tese (apêndice):** ReLU preserva magnitude ("quão longe") e deveria contaminar a decisão; Sigmoid satura ("de que lado") e deveria dar o ideal um-plano-por-classe → previsão: **Sigmoid > ReLU**. **Observado: ReLU vence.**

**Por que a previsão falha — a Sigmoid não é o classificador aqui.**

O controle decisivo (eggs 75%, **as mesmas features FLIM congeladas**, três cabeças):

| Cabeça sobre as mesmas features frozen | Test acc |
|:--|--:|
| SVM linear | **0.926** |
| MLP ReLU (frozen) | ~0.818 |
| Sigmoid head (frozen) | **0.137** |

Um classificador *linear* atinge 0.926 → as features são ricamente separáveis. A Sigmoid frozen colapsa para 0.137. **As features são ótimas; a camada oculta Sigmoid é o que quebra.**

### Causas, ranqueadas

1. **(RAIZ) A Sigmoid é gargalo oculto, não plano de decisão.** `TwoLayerSigmoidHead.forward` `models.py:376-380`: `...→Linear(48,24)→Sigmoid→Linear(24,C)→Softmax`. A Sigmoid está na **camada oculta**; a **decisão** é `Linear(24,C)→Softmax`. O braço ReLU (`MLPHead` `models.py:304-326`) decide com logits crus + `CrossEntropyLoss` (`src/evaluate/mlp.py:158`). Ou seja, **as duas cabeças decidem por Linear+Softmax** — "de que lado vs. quão longe" não se aplica a nenhuma das decisões. A teoria raciocina sobre a sigmoid como se fosse o classificador de saída, mas aqui ela é uma camada interna de squashing. **Inversão de papéis:** uma Sigmoid oculta *comprime* a feature de 48-d em `[0,1]^24`, descartando a magnitude que o Linear-Softmax a jusante ainda precisa; ReLU *preserva* essa magnitude.

2. **(MECANISMO, quantificado) `imagenet_norm` infla as features → a Sigmoid oculta satura → gradiente evapora.** Passando 32 imagens `ift_lab` reais pelo encoder eggs/train1 + head:

   | | feat 48-d (média / std / max\|·\|) | pré-sigmoid \|z\| max | frac \|z\|>4 | **unidades saturadas** | **média s(1−s)** |
   |:--|:--|:--|:--|:--|:--|
   | `imagenet_norm=False` | 3.7 / 3.6 / 13.3 | 6.2 | 0.049 | 4.2% | **0.138** |
   | `imagenet_norm=True` (usado) | **11.5 / 11.3 / 40.3** | 11.7 | **0.701** | **61.6%** | **0.033** |

   O `Normalize(mean≈0.45,std≈0.22)` sobre LAB[0,1] — marcado errado em `lejepa_dataset.py:42` — **triplica** a magnitude (3.7→11.5), satura **62% das 24 unidades** e derruba o gradiente local médio de 0.138 para **0.033** (~4× de atenuação, antes de treinar). ReLU não tem teto (gradiente 1 para qualquer pré-ativação positiva) e passa incólume. Treino e eval-Sigmoid ambos nesse regime (`classification_flim_ray.py:336-337`, `eval_sigmoid2l_test.py:74-75,88`).

3. **(EXPLICA a catástrofe frozen e a recuperação unfrozen) freeze vs. unfreeze.** Frozen (`classification_flim_module.py:148-150`): o encoder não pode reescalar → head preso a features que saturam a sigmoid → gradiente evapora → **eggs 0.137**. Unfrozen: o encoder encolhe a saída de volta ao regime linear da sigmoid → **0.137→0.386** (eggs), 0.721 (protozoan), 0.969 (larvae). O tamanho do salto frozen→unfrozen é a assinatura de saturação sendo aliviada por reescalonamento — não de features ruins. ReLU também melhora com unfreeze (0.818→0.955), mas bem menos, pois nunca foi limitado por saturação.

4. **(SECUNDÁRIO) largura 24 + caixa `[0,1]^24` + init default.** `hidden_dim=24` `models.py:369`, init `nn.Linear` default (Kaiming-uniform, sem Xavier). Mesmo sem saturar, separar C classes de um vetor confinado ao hipercubo unitário é geometria pior para o Linear-Softmax que o cone ilimitado da ReLU — por isso a sigmoid-unfrozen (0.386), embora recuperada, ainda fica atrás da ReLU (0.955).

5. **(DESCARTADO como diferenciador) Softmax na saída + NLLLoss.** Sigmoid: softmax interno + `nll_loss(log(probs))` `classification_flim_module.py:172-176`; ReLU: logits + CE. **Mesmo objetivo matemático.** A única diferença é numérica menor (log após softmax+clamp vs. log-softmax fundido) — insuficiente para explicar 0.137 vs 0.818. Otimizador idêntico (AdamW lr=5e-4 wd=5e-2) também não diferencia.

6. **(INTERAÇÃO) regime 1%.** A 1% tudo é fraco (SVM 0.174 melhor; ReLU-frozen 0.069; sigmoid-unfrozen 0.122): com dezenas de imagens o encoder não fine-tuna de forma confiável, a fuga por reescalonamento da sigmoid fica indisponível e a forte regularização do SVM vence. Efeito de escassez de dados sobreposto à história arquitetural.

### Corolário do apêndice confirmado
Sua intuição "se o FLIM separa tão bem, um SVM direto já classifica bem" **está certa**: SVM linear sobre features FLIM frozen dá 0.926 (eggs), 0.963 (larvae), 0.901 (protozoan) a 75%. O problema nunca foi separabilidade — foi a camada sigmoide oculta destruindo features boas.

### Dois testes para validar o diagnóstico
- **(a)** Rodar o braço Sigmoid com `--no-imagenet-norm` → a acurácia **frozen** deve subir bruscamente.
- **(b)** Trocar a ativação oculta por ReLU/tanh-escalado (ou ir a logits), mantendo o resto → o gap frozen para o SVM deve encolher em direção ao ~0.82 da ReLU.

---

## B2 — A curva do MLP ("cai um pouco e recupera"): **PARCIALMENTE CONFIRMADO**

Test accuracy, média de 3 splits (SVM sempre frozen):

| dataset | pct | SVM_frozen | MLP_frozen | MLP_unfrozen |
|:--|--:|--:|--:|--:|
| eggs | 1 | 0.174 | 0.069 | 0.047 |
| eggs | 5 | 0.751 | 0.666 | 0.757 |
| eggs | 25 | 0.887 | 0.738 | 0.901 |
| eggs | 50 | 0.910 | 0.788 | 0.940 |
| eggs | 75 | 0.926 | 0.818 | 0.955 |
| eggs | 100 | 0.935 | 0.819 | 0.971 |
| protozoan | 1 | 0.208 | 0.050 | 0.067 |
| protozoan | 5 | 0.765 | 0.640 | 0.780 |
| protozoan | 25 | 0.862 | 0.705 | 0.894 |
| protozoan | 50 | 0.888 | 0.752 | 0.928 |
| protozoan | 75 | 0.901 | 0.770 | 0.946 |
| protozoan | 100 | 0.907 | 0.781 | 0.960 |
| larvae | 1 | 0.390 | 0.858 | 0.888 |
| larvae | 5 | 0.921 | 0.926 | 0.968 |
| larvae | 25 | 0.957 | 0.944 | 0.985 |
| larvae | 50 | 0.961 | 0.961 | 0.988 |
| larvae | 75 | 0.963 | 0.960 | 0.989 |
| larvae | 100 | 0.971 | 0.965 | 0.993 |

**Interpretação (a) — SVM → MLP-frozen (só adicionar a cabeça treinável nas mesmas features):** **REFUTADO** para eggs/protozoan. O MLP-frozen fica **abaixo do SVM em todos os pcts** (eggs Δ ≈ −0.10 a −0.15; protozoan Δ ≈ −0.13 a −0.16); nunca recupera. Larvae ~empate de 5% pra cima. Ou seja, trocar SVM por MLP-frozen é perda líquida — cai e **não** recupera.

**Interpretação (b) — trajetória do MLP unfrozen vs SVM ao longo do %:** **CONFIRMADO** para eggs e protozoan — dip a 1%, cruza o SVM e depois abre vantagem crescente. Pontos de cruzamento (primeiro pct onde MLP_unfrozen ≥ SVM):

| dataset | accuracy | cohen_kappa | f1_weighted |
|:--|:--|:--|:--|
| eggs | 5% | 25% | 25% |
| protozoan | 5% | 25% | 25% |
| larvae | 1% | 1% | 1% |

- **eggs acc:** dip −0.127 (1%), cruza em **5%** (+0.005), chega a +0.036 (100%). ✅
- **protozoan acc:** dip −0.141 (1%), cruza em **5%** (+0.015), chega a +0.053 (100%). ✅
- **larvae:** sem dip — o MLP unfrozen vence em todo pct **porque o SVM colapsa a 1%** (acc 0.390 ± **0.428**, std enorme), não porque o MLP seja especialmente bom lá.

**Sensibilidade de métrica:** em kappa/F1 a recuperação é mais lenta (cruza em 25%, não 5%), pois essas métricas penalizam mais o desbalanceamento de classe em pct baixo.

**Caveat de variância a 1%:** todos os pontos a 1% têm std alto (larvae SVM ±0.428; protozoan MLP_unfrozen ±0.092). Conclusões a 1% não são confiáveis.

**Sigmoid (só 1% e 75%):** não sustenta a narrativa de recuperação — a 75% fica em 0.266 (protozoan frozen) / 0.721 (unfrozen), muito abaixo do SVM 0.901.

**Veredito B2:** a história "cai e recupera" vale **apenas para o FLIM+MLP fine-tunado (unfrozen) em eggs e protozoan** (dip a 1%, ultrapassa por 5% em acc / 25% em kappa-F1). É **refutada para o MLP frozen** e **ausente em larvae**. O mecanismo da recuperação é **descongelar o encoder**, não adicionar o MLP.

---

## Conclusão integrada

O pipeline é o que você descreveu, com desvios que — não por acaso — são exatamente o que explica os resultados "sem sentido":

- A **Sigmoid oculta + `imagenet_norm` sobre LAB** produzem saturação, colapsando o braço que a teoria previa vencedor. A teoria não está errada; ela descreve a **função de decisão**, e aqui a sigmoid **não é a decisão** — ambas as cabeças decidem por Linear→Softmax.
- A superioridade da **ReLU** vem de preservar a magnitude das features FLIM (ricas, como o SVM linear a 0.926 comprova) para o softmax final, sem teto de saturação.
- A "recuperação" da curva é efeito de **fine-tuning do encoder**, isolável do MLP.

**Próximos passos sugeridos:** (1) reexecutar o braço Sigmoid com `--no-imagenet-norm` e medir o ganho frozen; (2) mover a ativação oculta para ReLU (ou remover a sigmoid oculta) mantendo o resto; (3) mostrar a curva completa 1/5/25/50/75/100 também para o braço Sigmoid, para simetria; (4) regenerar os 2 checkpoints faltantes (larvae split1/split2 pct1 unfrozen).
