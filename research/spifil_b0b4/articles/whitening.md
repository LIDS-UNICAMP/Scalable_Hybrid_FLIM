# Whitening (ZCA / DBN / IterNorm) — regras de implementação fiéis aos artigos

> Fontes lidas (ar5iv, 17/11/2026): DBN arXiv:1804.08450; IterNorm arXiv:1904.03441.
> Código comparado: `ga_search/modehb/ops_norm_act.py` (`zca_matrix`, `ZCAWhitening`, `_InvSqrtPSD`, `DecorrelatedBatchNorm`, `make_norm`) e o uso em `ga_search/modehb/run_search.py` (`_fit_zca` L1062–1084; ordem SPiFiL → SoftHebb → ZCA fit → AE local, L933–950).

---

## 1. Referência

| Sigla | Artigo | Onde |
|---|---|---|
| DBN | Huang, Yang, Lang, Deng. *Decorrelated Batch Normalization*. CVPR 2018 | arXiv:1804.08450 |
| IterNorm | Huang, Zhou, Zhu, Liu, Shao. *Iterative Normalization: Beyond Standardization towards Efficient Whitening*. CVPR 2019 | arXiv:1904.03441 |
| ZCA estatístico (contexto) | Coates & Ng 2011 — origem do `zca` do repo (`flim/spifil.py:340`); não faz parte desta lista, citado só para separar os dois mecanismos | — |

## 2. Método

**DBN (Alg. 1, Sec. 3.3; Apêndice A.1, Eqs. A.1–A.6).** Por mini-batch, por grupo de k_G canais (em CNN cada posição espacial é amostra: m·h·w, Sec. 3.3):
μ = (1/m)Σx_j; Σ = (1/m)Σ(x_j − μ)(x_j − μ)ᵀ + εI; Σ = DΛDᵀ; Σ^{-1/2} = DΛ^{-1/2}Dᵀ (Eq. 4); x̂ = Σ^{-1/2}(x − μ).
ZCA, não PCA: PCA sofre "stochastic axis swapping" (Sec. 3.1; Fig. 2, DBN-PCA treina mal).
Backward (Sec. 3.2, Eq. 5): ∂L/∂Σ = D{(Kᵀ ⊙ (Dᵀ∂L/∂D)) + (∂L/∂Λ)_diag}Dᵀ, K_ij = 1/(σ_i − σ_j), i ≠ j.
Afim γ, β por canal após o whitening (Sec. 3.3; "often marginally improve").

**IterNorm (Alg. 1; Eqs. 3–5, 11–13).** Mesmo Σ (+εI), mas sem eigen:
Σ_N = Σ/tr(Σ) (Eq. 4); P₀ = I; P_k = ½(3P_{k−1} − P_{k−1}³Σ_N), k = 1..T (Eq. 3); Σ^{-1/2} = P_T/√tr(Σ) (Eq. 5).
Backward por autograd das iterações (Alg. 2; Eqs. 22–24). Só matmul: estável e rápido em GPU.

## 3. Receita oficial

| Item | DBN | IterNorm |
|---|---|---|
| Tamanho de grupo | **k_G = 16** em CNN (Sec. 4.2.1); ablação {1, 8, 16, 32, 64, 128}: 1 e 128 convergem mais devagar (Sec. 4.1) | padrão = whitening completo; ablação {256, 64, 32, 1} (Sec. 5.1); **recomendado 64** (usado em 5.2/5.3). Nota do paper: DBN "is unstable for a group size of 32 or above" |
| Iterações | — (eigen exato) | **T = 5** ("unless otherwise stated", Sec. 5.1); ablação T ∈ {0,1,3,5,7} |
| ε | "small positive number" — **sem valor** no texto | "a small number" — **sem valor** no texto |
| Running stats | μ_E e **Σ^{-1/2}_E** (não Σ) por média móvel: μ_E ← (1−λ)μ_E + λμ; Σ^{-1/2}_E ← (1−λ)Σ^{-1/2}_E + λ·DΛ^{-1/2}Dᵀ; init μ_E = 0, Σ^{-1/2}_E = I (Alg. 1 l. 10–11) | idem (Eq. 8), init μ̂ = 0, Σ̂^{-1/2} = I |
| Momentum λ | não informado | não informado |
| Eval | x̂ = Σ^{-1/2}_E (x − μ_E), depois γ, β | idem |
| Onde colocar | substitui BN (VGG-A); em ResNet só **um** DBN antes do 1º bloco residual (DBN-L1) | ResNet: **IterNorm-L1** (só o 1º BN); "IterNorm-full" sem ganho significativo; **L1 + DF** = também após o último average pooling |
| Treino | CIFAR VGG: bs 256, m 0.9, wd 5e-4; ImageNet: lr 0.1 ÷10 em 30/60, 90 ép., bs 256, wd 1e-4 | CIFAR VGG: SGD bs 256, lr 0.1 ÷5 em 60/120, 160 ép.; ImageNet ResNet-18: lr 0.1 ÷10 em 30/60/90, 100 ép., wd 1e-4 |
| Amostras vs dimensão | grupos existem porque o batch não estima bem Σ cheia; usar k_G < m (Sec. 3.4) | SND (Eqs. 9–10): whitening completo aumenta a perturbação estocástica; grupos menores e batches maiores reduzem SND (Sec. 4.1) |

## 4. Regras (checklist)

Grupos
- [ ] DBN: k_G = 16. IterNorm: k_G = 64. Nunca 1 (vira BN) nem "tudo" em DBN (instável ≥ 32).
- [ ] Amostras efetivas por grupo = N·H·W do batch; exigir N·H·W ≫ k_G (ao menos 10×) — em blocos profundos com mapa 6×6 e batch 8 isto falha.
- [ ] Último grupo menor é aceitável (paper não proíbe; registrar).

ε
- [ ] Somar εI à covariância antes da decomposição (DBN Alg. 1 l. 5; IterNorm Eq. 13). Valor não fixado pelos papers ⇒ declarar o nosso (ex.: 1e-5) como escolha, não como "do paper".
- [ ] IterNorm: tr(Σ) > 0 garantido por εI (necessário para Σ_N).

Running stats
- [ ] Média móvel de **μ** e da **matriz de whitening Σ^{-1/2}** (não de Σ). Init I.
- [ ] λ não é dado ⇒ usar 0.1 (convenção BN) e declarar.
- [ ] Atualizar running stats só em `train()`; nunca com dados de val/teste.

Eval
- [ ] `eval()` usa μ_E e Σ^{-1/2}_E; nada recalculado no batch de avaliação.
- [ ] γ, β aplicados após o whitening também em eval.

Forma
- [ ] ZCA (rotação de volta, D·Λ^{-1/2}·Dᵀ), nunca PCA.
- [ ] fp32 ou mais nas decomposições; IterNorm aceita fp32 com T = 5.
- [ ] Gradiente flui pela matriz de whitening (DBN: Eq. 5; IterNorm: autograd das T iterações).

Posição
- [ ] Começar por L1 (primeira normalização) e opcionalmente DF (após o pooling final) — é a configuração que os dois papers recomendam em ResNets.

## 5. Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| `zca` ≠ DBN | DBN/IterNorm: whitening **por batch**, recalculado a cada passo, com gradiente | `ZCAWhitening` (L187–255): ZCA estatístico (Coates), ajustado **uma vez** no treino (`_fit_zca`, 8 batches), buffers congelados, sem gradiente | Não chamar `zca` de DBN em relatórios. É outro mecanismo, válido como pré-processamento fixo |
| ZCA fica velho | DBN recalcula Σ por batch | ordem: SPiFiL → SoftHebb → **ZCA fit** → AE local treina a conv → LeJEPA treina tudo (run_search.py L933–950) ⇒ W/μ da ZCA ficam ajustados para pesos que mudaram | Re-ajustar a ZCA (`_fit_zca`) **após** o AE local e após o SSL (recalibração, análogo a recomputar stats de BN); ou trocar por DBN/IterNorm |
| ε do ZCA | — (não é DBN) | relativo: ε·mean(S), ε = 1e-2, piso 1e-8 (`zca_matrix` L121–134) | Ok para ZCA estatístico (escolha do repo/flim). Registrar como não-DBN |
| Covariância do DBN | (1/m)·X_cX_cᵀ + εI | `/ max(M−1,1)` + εI (L286) | Trocar para `/M` (Alg. 1) — diferença ínfima com M = N·H·W, mas é fidelidade barata |
| ε duplicado | εI uma vez | εI na cov **e** `clamp_min(eps)` nos autovalores (`_InvSqrtPSD`, L151) | Inofensivo (autovalores já ≥ ε); documentar que o clamp é só guarda numérica |
| Backward | Eq. 5 (K = 1/(σ_i − σ_j)) | Daleckii–Krein com L_ij = f'(s_i) para autovalores próximos (L156–168) | Fiel (idêntico para autovalores distintos) e mais robusto (canais mortos SPiFiL). Manter |
| Running stats | média móvel de μ e Σ^{-1/2} | `running_mean` e `running_W{i}` com momentum 0.1 (L289–302) | Fiel. Declarar λ = 0.1 como escolha nossa |
| Tamanho de grupo | DBN 16; IterNorm 64 | `group_size = min(16, max_group_size=16, C)` (L271) | Fiel ao DBN. O teto 16 impede IterNorm-64 ⇒ ao implementar IterNorm, teto próprio (64) |
| IterNorm | NS T = 5, trace-norm, grupo 64 | **não implementado** | Implementar `IterNorm(C, group_size=64, T=5, eps, momentum=0.1, affine=True)` só com matmul: elimina o eigh por passo (o comentário de `safe_eigh` L90–95 registra falha cusolver e GPU a 0% com ida à CPU) |
| `batch_zca` / `batch_dbn` | DBN substitui BN (não empilha) | `Sequential(BN, ZCA/DBN)` (L335–339) | Fora dos papers: BN antes do DBN é redundante (DBN já centra; a escala do BN é desfeita pelo whitening). Marcar como variante nossa; braço fiel = `dbn` sozinho |
| Posição | ResNet: só L1 (+ DF opcional) | gene `norm` por camada: DBN pode aparecer em qualquer bloco (`make_norm`, ConvBlock L426) | Braço fiel: DBN/IterNorm só no 1º bloco (L1) e opcionalmente após o pooling final da cabeça (DF), BN nos demais |
| Bias da conv | BN/DBN dispensam bias | `bias = norm_name == "none"` (L425) | Fiel. Mas blocos SPiFiL trazem bias −(K·μ_patch) e recebem a norma **depois** (run_search L927–931): o DBN apaga esse bias (centragem). Registrar |
| Amostras por grupo | k_G < m; SND | não verificado | Logar N·H·W/k_G por camada; recusar DBN quando < 10 |
| Dados do ajuste | só treino | `_fit_zca` usa `train_loader` (ok); `_log_init_diag` usa `val_x[:16]` (só diagnóstico) | Ok. Nunca usar val no ajuste |

## 6. Adaptação (sempre com init SPiFiL)

1. **SPiFiL primeiro, whitening depois:** a conv é inicializada por SPiFiL no prefixo congelado; o whitening entra **depois** da conv (como DBN substitui BN). Os γ, β começam em 1, 0 — e o whitening muda a escala da saída SPiFiL. Para preservar a função no passo 0 (princípio "backprop soma, nunca desfaz"), inicializar γ_c = desvio do canal c e β_c = média do canal c **medidos no treino com a saída SPiFiL** é uma adaptação nossa (não está nos papers) — declarar.
2. **ZCA estatístico (`zca`) como pré-processamento SPiFiL:** manter (é o mesmo do `flim/spifil.py:340`), mas re-ajustar após cada fase que muda pesos (AE, SSL). Sem isso a ZCA descreve uma rede que já não existe.
3. **Braço fiel DBN:** `norm=dbn` só na camada 1 (k_G = 16, ε = 1e-5 declarado, λ = 0.1), BN nas demais; eval com running stats.
4. **Braço fiel IterNorm:** IterNorm-L1 (+ DF após o pooling da cabeça, o que decorrelaciona o embedding lido pela SVM — relevante para o SVM C = 100 sem scaler do `eval_backbone_head`), k_G = 64, T = 5.
5. **Controles:** mesmo backbone SPiFiL com BN; mesma posição; mesmo nº de épocas; κ de validação na seleção, teste só relatório.
