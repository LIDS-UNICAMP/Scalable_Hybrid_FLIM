# Por que o κ do estágio 1 deu 0.35

**É possível resolver? Sim, e a conta fecha inteira.** O número de referência foi reproduzido a
partir dos mesmos pesos FLIM que já estão em disco. Em **45 células** (3 datasets × 3 splits ×
5 percentages ≥ 5%), o erro médio absoluto de κ contra o CSV externo é **0.0076**, com máximo
**0.0322**. **Os pesos nunca foram o problema** — isso está provado diretamente (§4). O que divergiu
foi o protocolo de avaliação, em três pontos empilhados (§3).

Gráfico e dados: `artifacts/plots/comparacao_flim_protocolo_original/`.

**Estado das modificações:** nada foi alterado em `src/`, `scripts/` ou `configs/`, e nenhum CSV ou
relatório registrado foi tocado. Foram criados apenas scripts de diagnóstico descartáveis em
`tools/` e a pasta de comparação em `artifacts/`, ambos somente-leitura sobre o repositório.

---

## 1. O que foi observado

| # | Sintoma relatado | Veredito |
|---|---|---|
| S1 | κ do estágio 1 ≈ 0.35 em protozoa @75%, esperado > 0.80 | **Confirmado.** Causa achada e medida. |
| S2 | W&B mistura estágio 1 e estágio 2 na mesma chave | **Parcialmente.** As runs já são separadas por nome, tag e config; só o nome da métrica colide. |
| S3 | O "SVM" do W&B talvez não seja o do avaliador oficial | **Falso.** É o mesmo, com os mesmos parâmetros. |
| S4 | FLIM puro registrado > 0.70, avaliação atual < 0.50 | **Confirmado**, e pela mesma causa de S1. |

Protozoa @75%, encoder FLIM congelado, por split:

| medição | split 1 | split 2 | split 3 |
|---|---|---|---|
| baseline do estágio 1 (val, como roda hoje) | 0.3426 | 0.2770 | 0.3399 |
| protocolo oficial no teste, mesma configuração | 0.2975 | 0.3143 | 0.3622 |
| protocolo oficial no teste, entrada e solver corrigidos | 0.7280 | 0.7033 | 0.7312 |
| **protocolo original completo (achatado)** | **0.8548** | **0.8127** | **0.8231** |
| referência externa (CSV) | 0.8545 | 0.8177 | 0.8391 |

---

## 2. Em palavras simples

O κ é medido treinando um SVM linear sobre o embedding do encoder. Esse SVM é resolvido por um
algoritmo iterativo, e o código dá a ele um orçamento de **10.000 iterações**
(`src/utils/evaluate.py:271`, e a cópia em `src/modules/autoencoder_flim_module.py:347`).

Em protozoa @75% com o embedding atual, o problema precisa de **34,8 milhões de iterações**. O SVM
para no meio do caminho, num ponto arbitrário, e devolve um separador ruim. **O κ que sai não mede o
encoder — mede onde o contador parou.**

Por que ficou tão difícil? As imagens são convertidas para o espaço de cor **LAB** e depois recebem a
normalização do **ImageNet**, calculada para RGB. O próprio repositório documenta que isso é errado:

> `src/data_modules/datasets/lejepa_dataset.py:42` — *"ImageNet RGB Normalize is wrong for FLIM-init
> (input is LAB[0,1])"*

Só que o padrão continua ligado (`src/modules/autoencoder_flim_module.py:541`,
`scripts/autoencoder_flim_ray.py:833`).

O dano é mais fundo que "escala errada". Os kernels FLIM carregam os vieses produzidos pelos
marcadores, e o viés é o que define onde o ReLU corta. Como convolução é linear,
`W·(x−μ)/σ + b = (1/σ)(W·x − W·μ + σ·b)` — o viés efetivo deixa de ser `b`, e os cortes calibrados
pelos marcadores param de cair onde deveriam.

**E ninguém viu o aviso porque ele foi jogado fora.** O `scikit-learn` emite `ConvergenceWarning`
nessa situação, mas o script pai captura o `stderr` do processo filho e só o imprime quando o
processo falha (`scripts/autoencoder_flim_ray.py:514-522`). As 36 runs terminaram bem, então os
avisos foram descartados.

### O efeito é errático, não só ruim

Um SVM truncado não dá um número menor — dá um número instável. Em larvae @75% split 3, mudar
**apenas a ordem das linhas de treino** move o κ de −0.0474 para +0.8220.

| | κ hoje | κ corrigido |
|---|---|---|
| larvae @75% split 2 | 0.0135 | **0.8101** |
| larvae @75% split 3 | −0.0474 | **0.9086** |

Esses dois "colapsos de split" nunca existiram. Não há nada de errado com o encoder de larvae.

---

## 3. A cadeia causal

O pipeline original usava o mapa `conv3` **achatado** sobre LAB[0,1]. **Qual `max_iter` ele usou não
é recuperável** — os CSVs externos não registram nenhum hiperparâmetro (§10). O que se sabe é que ele
**convergiu**: o expoente log-log de `training_time` versus `n_train` nesses CSVs é **1.63–2.02**, e
um SVM convergido é O(n²), enquanto um truncado achataria para ~1.0.

E nessa dimensionalidade o teto quase não importa: medimos que o achatado precisa de 45–69 mil
iterações (≈5× um teto de 10.000) e que **mesmo truncado em 10.000 ele entrega κ 0.8530**.

Depois, o repositório migrou as features para **GAP 48-d**
(`A_reports/2026/july/details_report/lejepa_view_pooling_handoff_2026-07-31.md`) e **manteve o
`max_iter=10000`** — que em 48-d fica **3.486× curto**. Somou-se a isso a `Normalize` do ImageNet
sobre imagens LAB. As três coisas empilhadas levam κ de 0.85 a 0.32.

### As três diferenças, e quanto cada uma vale

Protozoa @75%, teste, média dos 3 splits:

| configuração | κ |
|---|---|
| como o repositório roda hoje | **0.3247** |
| só deixar o SVM convergir | 0.5725 |
| só desligar a normalização | 0.6574 |
| as duas, ainda com GAP 48-d | 0.7209 |
| **as duas + `conv3` achatado — o protocolo original** | **0.8302** |
| alvo do CSV externo | 0.8371 |

**A normalização e a convergência não são aditivas.** Corrigida a normalização, o solver quase
converge sozinho: `n_iter` cai de 34,8 milhões para 254 mil (137× menos), e o teto de 10.000 quase
deixa de importar — a mediana de ganho por convergir depois disso é **+0.0013**. O teto era em boa
parte *sintoma* do mau condicionamento, não causa independente.

**A troca de features é o que fecha a conta.** Sem ela o teto é 0.72; com ela chega em 0.83.

### O que exatamente foi mudado para reproduzir

| # | dimensão | repositório hoje | reprodução |
|---|---|---|---|
| 1 | features | `AdaptiveAvgPool2d(1)` + flatten → **48-d** (`src/utils/evaluate.py:244,247-252`) | `conv3.flatten(start_dim=1)` → **27.648-d** |
| 2 | entrada | `_build_test(200, imagenet_norm=True)` | `imagenet_norm=False` — LAB[0,1] cru |
| 3 | solver | `max_iter=10000` | `max_iter=-1` (convergido) |

**Não mudou:** hiperparâmetros do SVC (copiados literalmente de `src/utils/evaluate.py:270-278` —
`C=1e2, degree=3, gamma="auto", coef0=0, ovo, linear`, **sem scaler**), pesos, encoder, loader
`ift_lab`, splits, `extract_features`, `compute_metrics`, fit no train / score no test.

### A representação intermediária foi testada e descartada

Um pooling `AdaptiveAvgPool2d(2)` (grid 2×2, 192-d) parece resolver protozoan, mas **não generaliza**:

| representação | protozoan | eggs | larvae |
|---|---|---|---|
| alvo (CSV externo) | 0.8371 | 0.8696 | 0.8353 |
| GAP 48-d + LAB[0,1] + convergido | 0.7209 | 0.7518 | 0.8664 |
| grid 2×2 (192-d) + LAB[0,1] + convergido | 0.8155 | 0.8397 | 0.8625 |
| **`conv3` achatado (27.648-d) + LAB[0,1] + convergido** | **0.8302** | **0.8719** | **0.8380** |

O grid 2×2 erra eggs por −0.030 e larvae por +0.027 — é uma interpolação entre 48-d e 27.648-d, não
a representação original.

**Confirmação independente pelo tempo.** Cronometrando um `SVC` com as formas reais (n_train=3584,
n_test=4786, 7 classes, nSV calibrado para reproduzir a acurácia registrada):

| d | fit (s) | predict (s) |
|---|---|---|
| 48 | 1,58 | 0,64 |
| 192 | 0,93 | 0,82 |
| 12288 | 66,4 | 128,7 |
| 27648 | ~165 | ~220 |

O CSV externo registra fit 69–78 s e predict 79–89 s — implicando **d ≈ 8.100–14.500**, na casa de
10⁴. Nem 48-d nem 192-d são compatíveis (erram por 44–124×). A leitura alternativa de que
`test_time` incluiria a extração de features foi descartada por medição: o `test_time` das pastas
`flim_mlp`/`ae_mlp`/`scratch` é **constante em 4,55 s** no mesmo test set, enquanto o do SVM varia de
0,96 a 94,88 s com o `percentage`.

---

## 4. Os pesos são os mesmos — prova direta

A pasta de entrega traz os mapas de ativação intermediários das próprias imagens-marcador
(`data/to_mateus/model/ch24_32_48_a0.5_f5/<ds>/train{K}/features/layer{0..3}/*.mimg`). Propagando
`layer{n-1}.mimg` pela `conv{n}` carregada e comparando com `layer{n}.mimg`, em **3 datasets × 3
splits × 16 imagens-marcador**:

- erro relativo máximo **≤ 1.2e-06** na conv1, **≤ 2.4e-04** nas conv2/conv3 — ruído de acumulação
  em float32
- `layer0.mimg` é **bit-exato** igual ao `ift_lab_loader(PNG)` do repositório (`max_abs = 0.000e+00`)

**Onde estão e o que é lido:**

```
data/to_mateus/model/ch24_32_48_a0.5_f5/{eggs,larvae,protozoan}/train{1,2,3}/models/
  conv{1,2,3}-kernels.npy   ← lido
  conv{1,2,3}-bias.txt      ← lido
  conv{n}-mean.txt/stdev.txt← NÃO lido (já embutidos nos kernels e no bias)
  labels{1,2,3}.txt         ← não lido (proveniência kernel→classe)
```

`architecture.json`: para eggs e larvae, da própria pasta (24/32/48). Para **protozoan**, de
`ch24_30_48_a0.5_f5/protozoan/train{K}/` (24/30/48) — porque o que está ao lado dos pesos declara 32
e **mente**: `conv2-kernels.npy` tem shape (600, **30**). O código já faz esse pareamento em
`scripts/autoencoder_flim_ray.py:106-117`.

Cadeia de carga: `_arch_json()` / `_flim_weights_path()` (`scripts/autoencoder_flim_ray.py:218-224`)
→ `load_FLIM_encoder()` (`src/models/models.py:555-602`), que transfere **apenas kernel e bias** de
cada uma das 3 camadas.

**A marker-norm já está embutida nos pesos.** Verificado numericamente:
`‖K[:,k]·stdev‖₂ = 1.000000` para todos os kernels das 3 camadas, e `bias_k = −Σ_p K[p,k]·mean[p]`
com erro ≤ 8.9e-06. Portanto `Conv2d(x;K)+b ≡ w_unitário·(x−μ)/σ`. **Aplicá-la de novo seria aplicar
duas vezes** — o que contradiz `analysis_flim_distill/REPORT_why_protozoan_collapses.md:98` e
`INVESTIGATION_flim_init_distillation.md:146-151`.

**O mapeamento `train{K}` ↔ `split{K}` está correto**, por prova combinatória: as imagens-marcador de
`train{K}` estão todas no `train` de `split{K}` e nenhuma no seu `test`; para j≠K, 1–5 marcadoras
vazam para o teste. Os arquivos de split entregues com os pesos são conjuntos idênticos aos que o
`DatasetParasite` lê (162 listas verificadas, 0 divergências).

**Não existe outra árvore de pesos:** 54 arquivos de kernel em 18 diretórios → **9 conjuntos
distintos por md5**; `~/scalable_FLIM_self_supervised` é cópia byte-idêntica.

---

## 5. Lista do que está errado

### A. Quebram os números do experimento

1. **`imagenet_norm=True` sobre imagens LAB.** `src/modules/autoencoder_flim_module.py:541`,
   `scripts/autoencoder_flim_ray.py:833`. Custa −0.35 de κ e descalibra o viés do FLIM.
2. **`max_iter=10000` sem scaler — o solver nunca converge.** `src/utils/evaluate.py:271`,
   `src/modules/autoencoder_flim_module.py:347`. `fit_status_=1` em **18 de 18 células** do protocolo
   oficial.
3. **Features GAP 48-d onde o protocolo original é `conv3` achatado.**
   `src/utils/evaluate.py:244,247-252`. Custa −0.12, e é o que torna o item 2 catastrófico.

### B. Contaminam resultados já registrados

4. **Quatro cópias do mesmo `SVC(max_iter=10000)` sem scaler:** `src/utils/evaluate.py:271`,
   `src/evaluate/svm_distillation.py:226`, `src/evaluate/svm_distillation_conv.py:213`,
   `src/evaluate/svm_ijepa.py:134`.
5. **~26 CSVs afetados — 2.010 linhas, 1.339 (67%) em pct ≥ 25.** Assinaturas visíveis sem re-rodar:
   **163 linhas com acurácia = exatamente 1/nº de classes** e **380 não-monotonicidades em
   `percentage`** em 18 arquivos. Ex.: `artifacts/SVM/svm_results.csv:109`, larvae 50→75%:
   0.7602 → **−0.0143**.
6. **Os cinco testes de Wilcoxon leem o mesmo CSV agregado sem filtrar `percentage`.** 12 das 18
   células pareadas vêm de pct ≥ 25. Alcança os 21 p-valores de
   `statistics/tools/reports_sibgrapi_camera_ready.md` §2.5 e os dois veredictos `EQUIVALENTES`.
7. **Assimetria de protocolo.** `svm_distill_with_projection.py:216-219` e `svm_real_flim.py:167-173`
   usam `StandardScaler` + `max_iter=20000`; o resto usa 10.000 sem scaler. Os braços não são
   pareados — e **os 20.000 também não bastam**: `artifacts/real_FLIM/` mostra protozoan caindo
   monotonicamente de pct25 a pct100 (0.4682 → 0.4207 → 0.3539 → 0.2773).

### C. Por que ninguém viu

8. **O `stderr` do filho é descartado quando ele termina bem.**
   `scripts/autoencoder_flim_ray.py:514-522`. O `ConvergenceWarning` foi emitido e perdido.
9. **`fit_status_` e `n_iter_` nunca são gravados**, em lugar nenhum do repositório.

### D. Instrumentação e W&B

10. `val/svm_kappa` é a mesma chave nos dois estágios — `:405-407`.
11. `baseline/*` é redundante: bit-idêntico à curva do estágio 1 em **18/18 células**.
12. `val/svm_kappa_delta` não tem consumidor nenhum.
13. **72 diretórios de run W&B para 36 runs lógicas** — 3 runs homônimas por célula de estágio 1.
14. `EarlyStopping(strict=False)` (`:650-652`): monitor errado **não levanta erro**, roda 1000 épocas
    em silêncio.
15. O eixo x reinicia entre estágios (estágio 1 vai a 453, estágio 2 recomeça em 0).

### E. Conceitual

16. O κ do estágio 1 é logado por época sendo **constante por construção** — o encoder está
    congelado. A oscilação visível era ruído do solver truncado.
17. `acc` e `f1` do repositório são **macro**; os dos CSVs externos são crua/*weighted*
    (`src/metrics/classification.py:57-61`). Só **κ** é comparável entre as duas famílias.

### F. Latentes

18. Em `pct100`, `train` e `validation` são o mesmo arquivo (4782 imagens idênticas). Não afeta a
    grade atual (5/75).
19. O `MaxPool2d` usa `padding=0` (`src/models/models.py:144`) → 99/49/24; o FLIM original usa
    `padding=1` → 100/50/25. **Medido: `padding=1` piora o κ** (|Δ| médio 0.0062 → 0.0082 sobre as 9
    células). Não vale mexer.
20. O `architecture.json` do protozoan em `ch24_32_48` declara 32 canais na layer2; os pesos têm 30.
21. `_ensure_pil_rgb` comentado em `src/data_modules/datasets/parasite_lejepa.py:108` — a
    equivalência entre treino e avaliador hoje é por acidente, não por contrato.

### G. Documentação errada

22. `README.md` (diff não commitado): diz que os dois callbacks olham `val/svm_kappa` (no estágio 1 é
    `val/recon_loss`), diz 18 runs (são 36), dá convenção de nome que o código não usa, e os comandos
    omitem `--stage`, que é obrigatório.
23. `svm_probe_curves.md` cita as chaves antigas.
24. `analysis_flim_distill/REPORT_why_protozoan_collapses.md:98` afirma que falta aplicar
    marker-norm. **Está errado** — ver §4.

---

## 6. O que **não** está errado

Verificado, para não virar alvo:

- **O congelamento do encoder.** `requires_grad=False` em todo o encoder
  (`src/models/models.py:672-676`), otimizador filtrando (`:477-479`), e o encoder é
  `Conv2d + ReLU + MaxPool` puro — **sem BatchNorm**. Confirmado por três caminhos independentes.
- **O probe é um SVM de verdade**, idêntico ao avaliador oficial (`:346-354` = `evaluate.py:270-278`).
  A suspeita de KNN disfarçado vale para `src/modules/lejepa_line_module.py:265`, não para este.
- **Os pesos, o mapeamento de split, e a inexistência de árvore alternativa** (§4).
- **O pipeline de dados**: mesmo dataset, mesmo loader `ift_lab`, transform determinístico dos dois
  lados (`V_train=V_eval=1`), mesma prevalência de classes até a terceira casa.
- **Val vs teste custa ~0.01**: medido nos pares do próprio pipeline externo, Δ = −0.009 / +0.005 /
  +0.011. A hipótese de "val pequeno demais" está errada por ~40× (val N=1198, não dezenas).
- **O tratamento dos rótulos.** O `+1`/`−1` do avaliador é cosmético; o probe usa 0-indexed
  consistentemente e dá o mesmo κ.

---

## 7. A anomalia do pct1 — célula degenerada da referência

Em pct1 nós ficamos **acima** do alvo, em todos os datasets e em qualquer representação. É a única
região da curva onde isso acontece, e é do lado da referência.

| dataset | nosso κ (médio) | alvo | Δ |
|---|---|---|---|
| protozoan | 0.4599 | 0.1447 | +0.3152 |
| eggs | 0.2539 | 0.1278 | +0.1261 |
| larvae | 0.3456 | 0.0284 | +0.3172 |

Provas de que o run externo daquele ponto é degenerado:

1. **larvae split2 registra κ = 0.0000, acc = 0.1269, f1w = 0.0286** — os três reproduzidos
   exatamente por um **preditor constante na classe minoritária**.
2. **`training_time` = 0.00 s** em protozoan e larvae, nos 3 splits, enquanto pct5 com 235 amostras
   custa 0,55 s.
3. Em protozoan, a concordância esperada implícita (invertendo a definição de κ) é
   **p_e = 0.071–0.079**. As referências são 0.597 (prever só a majoritária), 0.143 (uniforme sobre 7
   classes) e **0.067 (uniforme sobre as 6 minoritárias, nunca prevendo a classe dominante)**. O run
   externo está colado nesse último caso; o nosso dá p_e = 0.38–0.42.
4. A acurácia registrada (0.26 / 0.175 / 0.19) fica **abaixo** da baseline de majoritária (0.597) —
   pior que trivial.

Causas descartadas por medição: lista de amostras (descritores idênticos), classe sem representante
(todas representadas), escolha de feature (GAP e achatado dão o mesmo excesso), e um **sweep de 144
combinações** (C × scaler × class_weight) que não produziu nenhum preditor constante nem κ ≤ 0 — com
n=17–46 em 27.648-d, `C` e `class_weight` têm efeito **exatamente zero**.

Em protozoan, `max_iter=1` reproduz a faixa do alvo (0.199–0.226 contra 0.112–0.197), o que casa com
`training_time = 0.00`. Em larvae nem isso colapsa — o mecanismo lá continua sem explicação.

**Conclusão: a curva externa é confiável de pct5 em diante.** O pct1 não deve ser usado como
referência.

---

## 8. O que precisaria mudar

Em ordem de impacto. O risco é sobre **mexer**, não sobre deixar como está.

### 8.1 Desligar a normalização ImageNet nas runs com encoder FLIM
**Onde:** `scripts/autoencoder_flim_ray.py:833`, `src/modules/autoencoder_flim_module.py:541`.
**Impacto:** +0.33 de κ.
**Risco: médio, e não pode ser inversão global.** O braço de destilação **precisa** de
`imagenet_norm=True`, porque o teacher I-JEPA foi treinado com ela
(`analysis_flim_distill/REPORT_why_protozoan_collapses.md:44-52` mede −0.20 ao desligar lá). No
autoencoder não há teacher.
**Zero código:** a flag `--no-imagenet-norm` já existe e está ligada ponta a ponta. Cadeia
verificada: CLI (`:833`) → filho (`:506-507`) → arg (`:541`) → datamodule (`:595`) → transform
(`parasite_data_module_lejepa_splited.py:167`). O alvo da reconstrução é tratado nos dois casos
(`:218-220`), então `recon_loss` continua comparável.
**Armadilha:** `_run_name` (`:240-242`) **não codifica a flag**. Com `--skip-existing`, as 36 runs
antigas seriam puladas e a execução terminaria em segundos parecendo sucesso. Use `--run-prefix`
(`:840`).

### 8.2 Trocar as features de volta para o mapa achatado, ou declarar a mudança
**Onde:** `src/utils/evaluate.py:244,247-252`.
**Impacto:** +0.11 de κ. É o que fecha a conta com a referência.
**Risco: alto.** É a decisão de fundo. Ou o repositório volta ao achatado e os números voltam a ser
comparáveis com a referência externa, ou fica no pooled e **para de comparar com aquela curva** —
mas não pode fazer as duas coisas. Custo: o fit passa de ~1 s para ~90 s por célula.

### 8.3 Fazer o SVM convergir
**Onde:** `src/utils/evaluate.py:271` e as três cópias.
**Impacto:** +0.25 sozinho; **+0.001 se 8.1 já tiver sido feito**; quase nada se 8.2 for feito.
**Risco: alto.** O avaliador oficial é o contrato que produziu todos os números registrados. Mudar
`max_iter` torna os novos **não comparáveis** com os CSVs publicados. Ver §5.B.

### 8.4 Tornar o probe determinístico
**Onde:** `src/modules/autoencoder_flim_module.py:294-302`, que consome um loader com `shuffle=True`
(`parasite_data_module_lejepa_splited.py:248`).
**Impacto:** nenhum na média; elimina amplitude mediana de **0.099** entre épocas.
**Risco: baixo.** Resolve-se sozinho com 8.1+8.3 (SVM convergido é invariante à ordem).

### 8.5 Separar os estágios no W&B com uma chave única
**Onde:** `:405-408` (prefixo `stage1/`/`stage2/`) **e** `:643` (monitor) **e** `:806`
(`ckpt_monitor`).
**Risco: médio, com armadilha silenciosa.** `EarlyStopping(strict=False)` (`:650-652`) com monitor
inexistente **não levanta erro** — roda 1000 épocas. O monitor tem que ser derivado da mesma variável
de estágio, nunca digitado duas vezes.
As runs **já** são separadas por nome, tag e config — dá para filtrar o dashboard hoje, sem código.

### 8.6 Uma única métrica de κ
**Onde:** remover `on_fit_start` (`:304-332`), o replay do baseline e o delta (`:412-419`),
`baseline_flim_svm` (`:811`), o fallback (`:801-804`), e `scripts/autoencoder_flim_ray.py:710,713-715`.
**Risco: baixo.** Nenhum CSV do repositório tem coluna com essas chaves.

### 8.7 Registrar `fit_status_` e `n_iter_` junto com o κ
**Risco: nenhum.** É o que impede o problema de voltar sem ser notado.

### 8.8 Parar de descartar o `stderr` do filho
**Onde:** `scripts/autoencoder_flim_ray.py:514-522`. **Risco: nenhum.**

### 8.9 Corrigir README, `svm_probe_curves.md` e as notas de marker-norm
**Risco: nenhum.** Ver §5.G.

---

## 9. Como reproduzir

```bash
# Fatorial GAP vs achatado (protocolo oficial, fit no train / score no test)
OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE CUDA_VISIBLE_DEVICES=0 \
  conda run -n scalable_FLIM --no-capture-output \
  python tools/diag_flim_gap_vs_flatten.py --dataset protozoan --splits 1 2 3 \
    --percentage 75 --features A B --norms N1 N2 --solvers S1 S2

# Grade completa do probe: 18 células x 4 configurações, com fit_status_ e n_iter_
CUDA_VISIBLE_DEVICES=0 conda run -n scalable_FLIM \
  python tools/diag_probe_grid_sweep.py --cells all

# Prova de identidade dos pesos pelos .mimg + inventário + matriz 3x3
CUDA_VISIBLE_DEVICES=1 conda run -n scalable_FLIM --no-capture-output \
  python tools/diag_f2_weight_identity.py --part A --dataset protozoan --splits 1 2 3

# O gráfico da comparação
conda run -n scalable_FLIM python artifacts/plots/comparacao_flim_protocolo_original/build_comparacao_csv.py
conda run -n scalable_FLIM python artifacts/plots/comparacao_flim_protocolo_original/plot_comparacao.py
```

**Aviso operacional:** rode sempre com `OMP_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE`. Sem isso cada
processo mantém ~95 threads OpenMP em spin-wait enquanto o libsvm ajusta numa thread só — a máquina
chegou a load 245 e um fit de 100 s passou de 15 min sem terminar.

`tools/diag_flim_official_eval.py --fidelity-check` compara a réplica contra o `train_svm` oficial:
**Δκ = 0.00e+00**.

---

## 10. O que ficou em aberto

- **A configuração exata da baseline externa `SVM_FLIM` não é recuperável.** O pipeline não está
  versionado e os CSVs **não têm nenhuma coluna de hiperparâmetro** — sem `C`, `kernel`, `max_iter`,
  `n_features` nem semente. O que se sabe, e por quê:

  | item | status | base |
  |---|---|---|
  | `max_iter` | **desconhecido** | nada registra |
  | convergiu? | **sim** | expoente log-log `training_time` vs `n_train` = 1.63–2.02 (O(n²)) |
  | dimensão | **inferida, ordem 10⁴** | fit 69–78 s / predict 79–89 s cronometrados → d ≈ 8.100–14.500 |
  | achatou? | **inferido** | `.mimg` da layer3 = mapa 25×25×48; e a reprodução com flatten bate 45 células com MAE(κ) = 0.0076 |
  | normalização | **inferida** | só a reprodução sem ImageNet-norm bate |

  O achatamento é sustentado **pela consequência, não pela leitura**: achatando, as células batem.
  Evidência forte, mas indireta. Uma linha do código original encerraria a dúvida.
- Como consequência, **não se sabe se a baseline externa sofre da mesma truncagem** dos outros braços
  do repositório. Se ela for imune e os concorrentes truncados, a vantagem do FLIM está inflada; se
  ambos truncados, a direção do viés é indeterminada.
- **As medições de contaminação usaram o encoder FLIM puro**, não os checkpoints SSL que geraram cada
  CSV. Quantificam o **mecanismo**; não corrigem linha por linha.
- **O mecanismo do colapso de larvae em pct1** continua sem explicação (§7).
- **O achatado com `padding=1` (30.000-d) só foi medido em @75%.** Não sabemos se o padrão se mantém
  nos outros percentages.
- **A coluna `f1` da reprodução está vazia** — o repositório calcula macro e o CSV externo registra
  weighted. Para preencher seria preciso re-rodar as 54 células computando a variante weighted.

O repositório já havia registrado o mecanismo:
`A_reports/2026/july/details_report/why_svm_beats_mlp_2026-07-29.md:154` chama a convergência do
caminho FLIM de "LACUNA". O que não tinha sido feito era ligá-lo às conclusões estatísticas.
