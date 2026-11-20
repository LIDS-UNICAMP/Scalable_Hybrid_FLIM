# SoftHebb — regras de implementação fiéis ao artigo

> Fontes lidas (17/11/2026): paper arXiv:2209.11883 (ar5iv); código oficial github.com/NeuromorphicComputing/SoftHebb (`hebbconv.py`, `presets.json`, `README.md`).
> Código comparado: `ga_search/modehb/softhebb.py` (`soft_wta`, `softhebb_update`, `fit_softhebb`); chamada em `ga_search/modehb/run_search.py` L933–939; defaults em run_search.py L67 (`softhebb=dict(epochs=1, lr=0.01, temperature=1.0)`).
> Atenção: `MISSION.md` cita arXiv:2107.05747 (Moraitis et al. 2022, a regra original em MLP). Este documento segue o pedido: Journé et al., ICLR 2023 (arXiv:2209.11883), a versão convolucional profunda.

---

## 1. Referência

Journé, Rodriguez, Guo, Moraitis. *Hebbian Deep Learning Without Feedback*. ICLR 2023. arXiv:2209.11883. Código: NeuromorphicComputing/SoftHebb.

## 2. Método

Neurônio k (filtro), entrada local x (patch C·kh·kw), u_k = w_k·x (conv sem bias).

- **Soft-WTA (Sec. 3, Eq. 1):** y_k = e^{u_k/τ} / Σ_l e^{u_l/τ}.
- **Plasticidade (Sec. 3, Eq. 2):** Δw_ik = η·y_k·(x_i − u_k·w_ik) — só termos locais; ponto fixo ‖w_k‖ = 1.
- **Anti-Hebbiano (Sec. 3):** nega a atualização de todos os neurônios exceto o vencedor (argmax u). Código oficial (`HebbSoftKrotovConv2d.get_wta`, modo 0): `wta = -wta`; `wta[vencedor] = -wta[vencedor]`.
- **Atualização no código oficial (`delta_weight`):** `delta = yx - yu.view(-1,1,1,1) * weight`; normalizada `delta /= |delta|.amax() + 1e-30`.
- **lr adaptativo (Apêndice A.2, Eq. 5):** η_i = η·(r_i − 1)^q, r_i = ‖w_i‖; código: `lr * (|norm − 1| + nc) ** power_lr` ⇒ o lr vai a zero quando o filtro chega ao raio 1.
- **Triangle (Apêndice A.3.1, Eqs. 6–7):** Triangle(u_j) = RePU(u_j − ū), ū = média sobre canais na posição, RePU(u) = u^p para u > 0.

## 3. Receita oficial (CIFAR-10, Table 3 = preset `4SoftHebbCnnCIFAR`)

| Camada | kernel | canais | 1/τ | τ (nosso parâmetro) | η | q | pooling | ativação |
|---|---|---|---|---|---|---|---|---|
| 1 | 5 (pad 2) | 96 | 1 | 1.0 | 0.08 | 0.5 | Max 4 (s2) | Triangle p = 0.7 |
| 2 | 3 (pad 1) | 384 | 0.65 | 1.54 | 0.005 | 0.5 | Max 4 (s2) | Triangle p = 1.4 |
| 3 | 3 (pad 1) | 1536 | 0.25 | 4.0 | 0.01 | 0.5 | Avg 2 (s2) | Triangle p = 1.0 |

- **Ordem por camada (A.3):** BatchNorm → conv SoftHebb (stride 1, padded; preset `b0` = sem bias; `padding_mode: reflect`) → pooling → ativação.
- **BatchNorm (Sec. 4):** "standard initial parameters (γ=1, β=0), which we did not train" — standardização por canal antes de cada conv Hebbiana, também no forward.
- **Largura:** 96 no 1º bloco (48 no ImageNet), fator f_w = 4 por camada; a τ ótima depende de f_w (A.4).
- **Treino não supervisionado:** **1 época** por camada, camadas treinadas **em sequência** e congeladas ("Each layer was fully trained and frozen before the next one", Sec. 4; simultâneo dá resultado parecido, A.6.1). **Batch 10** (CIFAR-10, MNIST), 20 (STL-10, ImageNet 10%). ~**5000 iterações** de atualização ótimas no CIFAR-10. Reprodução: `--dataset-unsup CIFAR10_1` = `{"nb_epoch": 1, "batch_size": 10}`.
- **Init (A.1.2):** normal, σ = R·√(π/(2N)), N = nº de sinapses; **R > 2.5** faz todos os neurônios aprenderem (neurônios longe do raio 1 têm η grande).
- **Entrada (A.1.1):** dados normalizados em z-score ("normalized into standard scores"), não [0,1].
- **Whitening:** o paper **não** usa ZCA nos resultados reportados (nenhuma menção no texto). O código oficial tem ZCA como opção de dataset (`"zca_whitened": false` no default CIFAR10/STL10/ImageNette; presets `zca`, `1zca` etc. existem). A dependência efetiva do paper é da **standardização por BN não treinado** antes de cada camada + entrada em z-score.
- **Leitura (Sec. 4, A.6.2):** classificador linear com backprop sobre features congeladas, dropout 0.5, bs 64, lr 1e-3 com halving em 20/35/50/60/70/80/90% do treino, 50 épocas (CIFAR).

## 4. Regras (checklist)

Regra de aprendizado
- [ ] u = conv(x, w) **sem bias**.
- [ ] y = softmax(u/τ) sobre filtros; anti-Hebbiano: −y para não-vencedores, +y para o vencedor.
- [ ] Δw_k = Σ_{b,posições} y_k·(x − u_k·w_k); normalizar por max|Δ| (código oficial).
- [ ] η_k = η·(|‖w_k‖ − 1| + 1e-10)^q, q = 0.5.
- [ ] Sem gradiente, sem otimizador; pesos mudam só pela regra.

Temperatura
- [ ] Por camada: 1/τ = 1, 0.65, 0.25 (τ = 1, 1.54, 4) — temperatura sobe com a profundidade. Ajustar se a largura (f_w) for diferente (A.4).

Dependência de normalização / whitening
- [ ] BN não treinado (γ = 1, β = 0) **antes** de cada conv Hebbiana, no aprendizado **e** no forward.
- [ ] Entrada da rede em z-score.
- [ ] ZCA é opcional (opção do repo oficial, sem ablação no paper) — tratar como braço separado.

Épocas e lr
- [ ] 1 época, batch 10, ~5000 atualizações por camada (CIFAR-10). Se o dataset for pequeno, o que importa preservar é o **nº de atualizações**, não o nº de épocas.
- [ ] η por camada: 0.08, 0.005, 0.01.

Protocolo
- [ ] Camadas em sequência; cada uma congelada antes da próxima.
- [ ] Triangle como ativação de saída (entrada da próxima camada).
- [ ] Raio inicial R > 2.5 (senão parte dos neurônios não aprende: η ∝ |r − 1|^q).

## 5. Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Regra Δw, anti-Hebbiano, normalização max | Eq. 2 + código oficial | `soft_wta` + `softhebb_update` (L57–153) | **Fiel** (mesma expressão `yx − yu·w`, mesmo sinal anti-Hebbiano, mesma divisão por max) |
| lr adaptativo | η(|r−1| + nc)^q, q = 0.5 | `lr * (|radius−1| + 1e-10) ** 0.5` (L141–143) | Fiel |
| Temperatura | τ = 1, 1.54, 4 por camada | `temperature = 1.0` fixo para todas (run_search L67) | Tornar por camada: τ_i ∈ {1.0, 1.54, 4.0} para i = 0, 1, ≥2 (ou buscar no grid [0.1–100] do A.4 com a nossa largura) |
| lr | 0.08, 0.005, 0.01 | `lr = 0.01` fixo | Por camada: 0.08, 0.005, 0.01 |
| BN antes da conv | BN(γ = 1, β = 0, não treinado) no forward e no aprendizado | `standardize_input=False` por default e não passado (run_search L938) ⇒ aprende sobre a saída crua do prefixo; e o forward do modelo também não tem esse BN | Passar `standardize_input=True` **e** inserir `BatchNorm2d(affine=False)` antes da conv no bloco (senão a regra aprende numa escala e o forward roda em outra) |
| Ordem do bloco | BN → conv → pool → act | `ConvBlock`: conv → norm → act → pool (ops_norm_act L452–455) | Braço fiel: `BN(affine=False) → conv → pool → Triangle`; registrar como variante |
| Ativação | Triangle p = 0.7/1.4/1 | gene `activation` (relu, gelu, …); `input_triangle=False` | Braço fiel: activation = triangle(p por camada) |
| Nº de atualizações | ~5000 (bs 10, 1 época, 50k imgs) | 1 época com `batch_size` 32–64 (run_search L62/L132) em treinos de parasitas pequenos ⇒ dezenas de passos | `epochs = ceil(5000·10 / N_train)` com bs 10 (ou alvo de atualizações fixo); logar `softhebb/steps` |
| Init / raio | normal com R > 2.5 | **SPiFiL**: normas por filtro 0.36–1.19, média 0.54 (medido, `flim/spifil.py` L930–932) ⇒ η ∝ |r − 1|^0.5 ≈ 0.68 na média, mas ≈ 0 para filtros perto de 1 (esses ficam congelados) | Ver Adaptação: normalizar direção SPiFiL e fixar raio R (ex.: 3) antes do SoftHebb |
| Bias | conv sem bias | SPiFiL traz bias = −(K·μ_patch); SoftHebb ignora o bias (L89–90) e muda K ⇒ bias inconsistente após a regra | Recalcular bias = −(K_novo·μ_patch) a partir das médias dos patches SPiFiL, ou zerar o bias e confiar no BN de entrada (fiel) |
| Largura | 96 → 384 → 1536 (f_w = 4) | larguras do genoma (frequentemente 16–128) | Competição WTA com poucos neurônios é outro regime: registrar largura e, no braço fiel, usar f_w = 4 quando o orçamento Bk permitir |
| Sequencial | camada treinada e congelada antes da próxima | `fit_softhebb` por camada com prefixo congelado (`input_transform=prefix`) | Fiel |
| Depois do SoftHebb | só classificador linear | AE local + SSL treinam a camada depois (protocolo 3 fases, softhebb.py L28–35) | Fora do paper (híbrido nosso). Ok, mas medir κ logo após o SoftHebb (sem AE) para separar efeitos |
| Grupos | — (só conv densa) | per-group (vira Oja com 1 filtro/grupo, L92–98) | Extensão nossa; registrar |
| Padding | `reflect` | o da conv (zeros) | Detalhe; usar reflect no braço fiel |
| ZCA | opcional no repo; não usado nos resultados | `norm=zca` é ajustado **depois** do SoftHebb (`_fit_zca`, L940) e não antecede a conv | Se quiser o braço "SoftHebb + ZCA" do repo oficial, a ZCA tem de estar **na entrada** da conv Hebbiana (dados/prefixo), não na saída |

## 6. Adaptação (sempre com init SPiFiL)

Protocolo do repo (softhebb.py L28–35): SPiFiL → SoftHebb → AE local, sequencial. Para que o SoftHebb **some** ao SPiFiL e não desfaça:

1. **Direção do SPiFiL, raio do paper.** Antes do SoftHebb: ŵ_k = w_k/‖w_k‖ (direção escolhida pelo superpixel/patch) e w_k ← R·ŵ_k com R único (paper: R > 2.5; sugerido 3). Assim todos os filtros têm o mesmo η inicial e a regra só refina a direção enquanto o raio converge a 1 (Eq. 5). Sem isso, filtros SPiFiL com norma ≈ 1 ficam congelados e os de norma 0.36 mudam muito — um viés que não vem de dados.
   - Variante conservadora ("protect"): R = 1 + δ pequeno (ex.: 1.25) ⇒ η inicial ≈ 0.5·η, refino curto.
2. **Entrada standardizada:** BN(affine=False) antes da conv, no aprendizado e no forward (Sec. 4). SPiFiL também z-scoreia patches, então o BN de entrada é coerente com o SPiFiL.
3. **Bias:** após o SoftHebb, recomputar b_k = −w_k·μ_patch_k (regra SPiFiL) ou b = 0 com BN de entrada (fiel ao paper, preset `b0`). Escolher um e registrar.
4. **Hiperparâmetros por profundidade:** τ = 1 / 1.54 / 4 e η = 0.08 / 0.005 / 0.01 para camadas 1 / 2 / ≥3; q = 0.5; bs 10; ~5000 atualizações.
5. **Escala de volta:** o SoftHebb termina com ‖w‖ ≈ 1; se a camada seguinte/AE esperava a escala SPiFiL, reescalar para a norma original filtro a filtro (como o `random_layer` faz, `flim/spifil.py` L930–940) — só a direção muda.
6. **Medições obrigatórias:** antes/depois do SoftHebb: cos(w_SPiFiL, w_SoftHebb) por filtro, `winner_entropy`, `dead_fraction`, κ de validação com a SVM original (C = 100). Ablação: SPiFiL × SPiFiL+SoftHebb × aleatório(R)+SoftHebb (o braço do paper).
