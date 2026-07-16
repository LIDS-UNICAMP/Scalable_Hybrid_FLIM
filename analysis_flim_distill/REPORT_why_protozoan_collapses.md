# Por que o FLIM "sozinho funciona" mas "com teacher vai mal" — e por que o protozoan é o pior caso

*Investigação 2026-06-07. Pergunta do usuário: não faz sentido o FLIM performar sozinho e degradar quando misturado a um teacher. Resposta curta: **não é o teacher que erode o FLIM**. A afirmação anterior ("distillation erode os filtros profundos → κ desaba") juntou números de dois regimes diferentes e atribuiu a causa errada. A causa real é um **conflito de normalização de input que é impossível de satisfazer dos dois lados ao mesmo tempo** no pipeline atual.*

---

## 1. O fato que derruba a narrativa "teacher erode FLIM"

Comparação na MESMA régua (encoder 48-dim, SVM κ), protozoan, de `distill_destroys_flim.json` — tudo no **mesmo input** (ImageNet-norm sobre LAB, o pipeline padrão):

| Condição | κ | leitura |
|---|---|---|
| FLIM **não-treinado**, input correto (marker-norm / LAB[0,1]) | **0.685** | o "prior FLIM" honesto |
| FLIM **não-treinado**, input do pipeline (ImageNet-norm sobre LAB) | **0.330** | **−0.355 só pelo input, SEM teacher nenhum** |
| FLIM **destilado** c/ teacher, head 126k | 0.522 | **+0.19 vs não-treinado no mesmo input** |
| FLIM **destilado** c/ teacher, head 400k | **0.625** | **+0.295 vs não-treinado no mesmo input** |
| random (trunc) destilado c/ teacher, 400k | 0.230 | sem prior FLIM → colapsa |

**Lendo maçã-com-maçã (mesmo input):** o teacher leva o encoder FLIM de **0.330 → 0.625**. O teacher **ajuda**, não erode. O que derruba o FLIM antes de qualquer treino é o input descalibrado (−0.355), maior que toda a lacuna restante. E o prior importa: o random com o MESMO teacher e MESMA head só chega a 0.230.

> A queda "FLIM sozinho 0.85 → destilado baixo" é majoritariamente o **bug de input** + comparação de réguas trocadas, **não** o teacher comendo os filtros.

---

## 2. Onde minha afirmação anterior errou

Eu disse: *"conv3 cai pra cosine 0.19 → a head de 1 camada não protege → κ desaba pra 0.12–0.27 nas runs one-layer."* Isso costura **dois regimes distintos**:

- **conv3 cosine ≈0.19** vem das runs **COM** ImageNet-norm (`drift_init_vs_126k`, protozoan). Nessas runs o κ é **0.52** (saudável!), não colapsado.
- **κ 0.07–0.24** vem das runs **`no_imagenet_norm`** (o "fix"), que são runs **diferentes**.

Ou seja: no regime com norm, o conv3 deriva muito (cos 0.19) **e mesmo assim** κ=0.52. **Drift de filtro ≠ destruição.** O encoder reescrever o conv3 é ele *adaptando*, e o downstream melhora. A causal "erosão de filtro → colapso de κ" não se sustenta dentro de um mesmo experimento.

---

## 3. A causa real do colapso do protozoan: conflito de normalização de duas pontas

O `--no-imagenet-norm` (o fix do bug) tem efeito **oposto** por dataset:

```
RULER encoder48 — mean Δκ (NEW[no-norm] − OLD[imagenet-norm]), 3 splits:
  eggs     : +0.151   (fix ajuda)
  larvae   : +0.128   (fix ajuda)
  protozoan: −0.202   (fix DESTRÓI)   ← pct75: −0.47, pct100: −0.43
```

Protozoan no one-layer: imagenet-norm κ≈0.51 → no-imagenet-norm κ≈0.07 (pct100). Colapso, e **monotônico com pct** (quanto mais dados, pior) — assinatura clássica de "ajustar mais forte um *alvo corrompido*".

### Por quê — o teacher também perde a normalização

`prepare_teacher_input` ([distillation.py:371](../src/models/distillation.py#L371)) **só faz resize bilinear** da view do student; **não re-normaliza**. O docstring assume que "the student views are already ImageNet-normalised". E o I-JEPA ViT-H/14 ([ijepa_encoder.py:239-241](../src/models/ijepa_encoder.py#L239)) **exige** input ImageNet-normalizado.

Consequência: **student e teacher consomem o MESMO tensor**. Logo as duas exigências são **incompatíveis**:

| flag | FLIM kernels (querem LAB[0,1]) | I-JEPA teacher (quer ImageNet-norm RGB) |
|---|---|---|
| `imagenet_norm` (padrão) | ❌ descalibrado (−0.355 no probe) | ✅ alvo de qualidade |
| `no_imagenet_norm` (fix) | ✅ na faixa calibrada | ❌ recebe LAB[0,1]-como-RGB → **embeddings degradados** |

Não dá pra satisfazer os dois com o pipeline atual. Você só está **escolhendo qual lado matar de fome**.

### Por que o protozoan é justamente o que quebra dos dois lados

1. **É o mais sensível ao input do student** no probe não-treinado: 0.685 → 0.330 (−0.355), a maior queda dos três (eggs −0.19; larvae robusto, ~0). O sinal discriminativo do protozoan vive nos canais/escala LAB que o ImageNet-norm mais corrompe → ele *precisa* do `no_imagenet_norm`.
2. **É o mais dependente da qualidade do alvo do teacher**: 7 classes, baseline mais baixa, menos margem, sinal nos conv2/conv3 profundos (conv2 tem 30 canais, não 32 — específico do protozoan). Quando o teacher é alimentado com lixo (`no_imagenet_norm`), o alvo degrada e o protozoan é o que tem menos folga para absorver isso.

Resultado: **eggs/larvae** — o ganho no student supera a perda no teacher → fix ajuda. **Protozoan** — perde dos dois lados: com norm o student está faminto; sem norm o teacher está faminto. Não existe, no pipeline atual, uma configuração em que o protozoan ganhe nas duas pontas. É por isso que ele parece "ir mal com o teacher".

---

## 4. Resposta direta à pergunta

> "FLIM sozinho performa, mas misturado com o teacher vai mal — por quê?"

**Ele não vai mal *por causa* do teacher.** Medido na mesma régua e no mesmo input, a destilação com o teacher **melhora** o encoder FLIM (protozoan 0.330→0.625). A lacuna entre "FLIM standalone" e "FLIM destilado" vem de:

1. **Bug/conflito de normalização** (a maior parte): os kernels FLIM são calibrados para LAB[0,1] + marker-norm, mas o pipeline entrega ImageNet-norm sobre LAB. Isso sozinho derruba o protozoan −0.355 **antes de qualquer treino**.
2. **Esse conflito é de duas pontas e irreconciliável** hoje, porque o teacher reusa o tensor do student. Corrigir o student (`no_imagenet_norm`) quebra o teacher — e o protozoan é o único que perde de qualquer jeito.
3. Réguas trocadas (48d vs 1280d) inflaram artificialmente parte da "queda" em análises anteriores.

O "erode os filtros / conv3 cos 0.19" é **drift de adaptação, não destruição** — no mesmo experimento o κ está em 0.52–0.62, não colapsado.

---

## 5. O fix correto (testável)

Desacoplar as duas normalizações. **Tirar o ImageNet-norm do transform do student** (mantém FLIM em LAB[0,1]) **e aplicar o ImageNet-norm DENTRO do `prepare_teacher_input`**, só no ramo do teacher:

```python
# prepare_teacher_input: depois do resize p/ 224, ANTES do I-JEPA
x = F.interpolate(student_batch, size=(224,224), mode="bilinear", align_corners=False)
x = (x - mean_imagenet) / std_imagenet   # normaliza SÓ para o teacher
return x
```

Assim os dois lados ficam na escala certa simultaneamente. Predição: o protozoan deixa de colapsar no `no_imagenet_norm` (o teacher volta a dar alvo bom) **e** mantém o ganho do student. Re-rodar grid pequeno (protozoan+eggs, split1, pct∈{50,100}, 126k+400k) para confirmar.

Opcional (lado FLIM): aplicar a marker-norm por-camada (`conv{n}-mean/stdev.txt`, que existem mas nunca são carregados) entre as convs, como o treino FLIM original fez.

---

## Fontes
- `analysis_flim_distill/distill_destroys_flim.json` (probe não-treinado + drift de filtros, split1 pct100)
- `analysis_flim_distill/aggregate_nonorm_compare.py` (OLD vs NEW, 3 splits)
- `results/svm_nonorm_1x1_encoder48_results.csv`, `svm_old_1x1_encoder48_results.csv`
- Código: [distillation.py:371](../src/models/distillation.py#L371) (`prepare_teacher_input` só faz resize), [ijepa_encoder.py:239](../src/models/ijepa_encoder.py#L239) (teacher exige ImageNet-norm), [lejepa_dataset.py:30-56](../src/data_modules/datasets/lejepa_dataset.py#L30) (flag `imagenet_norm`)
</content>
</invoke>
