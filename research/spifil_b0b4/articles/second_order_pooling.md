# Pooling de 2ª ordem — regras de implementação fiéis aos artigos

> Escopo: cabeça/leitura de covariância (bilinear) sobre o último mapa (ou vários mapas) do backbone SPiFiL.
> Fontes lidas (ar5iv, 17/11/2026): B-CNN arXiv:1504.07889; MPN-COV arXiv:1703.08050; iSQRT-COV arXiv:1712.01034;
> Compact Bilinear arXiv:1511.06062; complemento usado para a leitura sem treino: Improved B-CNN (Lin & Maji, BMVC 2017, arXiv:1707.06772).
> Código comparado: `ga_search/modehb/pooling_head.py` (`EmbeddingHead`, `CovReadout`), `research/spifil_b0b4/readout_x/eval_backbone_head.py` (`so2_vector`, `xb_vector`, `PixelMoments`).

---

## 1. Referência

| Sigla | Artigo | Onde |
|---|---|---|
| B-CNN | Lin, RoyChowdhury, Maji. *Bilinear CNN Models for Fine-grained Visual Recognition*. ICCV 2015 | arXiv:1504.07889 |
| MPN-COV | Li, Xie, Wang, Zuo. *Is Second-order Information Helpful for Large-scale Visual Recognition?* ICCV 2017 | arXiv:1703.08050 |
| iSQRT-COV | Li, Xie, Wang, Gao. *Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization*. CVPR 2018 | arXiv:1712.01034 |
| CBP / TS | Gao, Beijbom, Zhang, Darrell. *Compact Bilinear Pooling*. CVPR 2016 | arXiv:1511.06062 |
| iB-CNN (apoio) | Lin, Maji. *Improved Bilinear Pooling with CNNs*. BMVC 2017 | arXiv:1707.06772 |

## 2. Método

Mapa X ∈ ℝ^{d×n} (d canais, n = h·w posições).

- **B-CNN (Sec. III-A, Eqs. 1–2):** Φ(I) = Σ_l f_A(l)ᵀ f_B(l) — produto externo somado nas posições (2º momento **não centrado**, sem ordem espacial). Simétrico (f_A = f_B) ou assimétrico (dois streams). Normalização (Sec. III-A2): y = sign(x)·√|x|, z = y/‖y‖₂. Gradiente (Eq. 3): x = AᵀB ⇒ dℓ/dA = B(dℓ/dx)ᵀ, dℓ/dB = A(dℓ/dx). Dimensão 512×512 = 262k.
- **MPN-COV (Sec. 3.1, Eqs. 1–4):** P = X Ī Xᵀ, Ī = (1/N)(I − (1/N)11ᵀ) (covariância **centrada**, divisor N). Q = P^α = U F(Λ) Uᵀ via EIG, α = 1/2 em todos os experimentos. Pós-normalizações M-ℓ2 / M-Fro (Eq. 5) **pioram** (Table 2: MPN 38.51 vs MPN+M-ℓ2 39.62 vs MPN+M-Fro 39.93 top-1 err). Backward via EIG (Eqs. 7–11, K_ij = 1/(λ_i − λ_j)). Sem ε: autovalores < eps(λ₁) são truncados a zero (Sec. 5.1). Saída = triângulo superior: d(d+1)/2 = 32 896 para d = 256.
- **iSQRT-COV (Sec. 3, Eqs. 2–4):** mesmo Σ = X Ī Xᵀ (Sec. 3.1). Pré-normalização A = Σ/tr(Σ) (Eq. 3; Frobenius como alternativa). Newton–Schulz acoplado (Eq. 2): Y₀ = A, Z₀ = I; Y_k = ½Y_{k−1}(3I − Z_{k−1}Y_{k−1}); Z_k = ½(3I − Z_{k−1}Y_{k−1})Z_{k−1}. Pós-compensação (Eq. 4): C = √tr(Σ)·Y_N. Só matmul → GPU-friendly, backprop por autograd das iterações.
- **Compact Bilinear (Sec. 3, Alg. 1–2):** aproxima o kernel polinomial ⟨x,y⟩² com φ(x) de dimensão d ≪ c², C(X) = Σ_s φ(x_s) (Eq. 4). RM (Alg. 1): φ = (1/√d)(W₁x)∘(W₂x), W ∈ {±1}^{d×c} fixos. TS (Alg. 2): count sketch Ψ(x,h,s), φ = FFT⁻¹(FFT(Ψ₁)∘FFT(Ψ₂)); h, s aleatórios **fixos**. Custo TS O(hw(c + d log d)) (Table 2).
- **iB-CNN (Sec. 2.1, 3.2–3.5):** matrix-sqrt **seguida** de signed-sqrt elemento a elemento + ℓ2 é a melhor combinação; ε = 1 na diagonal (Eq. 1); Denman–Beavers com 5 iterações ≈ SVD (Table 3); backward por Lyapunov ligeiramente melhor que por SVD.

## 3. Receita oficial

| Item | B-CNN | MPN-COV | iSQRT-COV | CBP (TS) |
|---|---|---|---|---|
| Redução antes da cov | não (512 canais VGG) | conv 1×1 → **256** canais (Sec. 5.1) | conv 1×1 → **256** + ReLU após a última soma da ResNet (Sec. 4.1) | não (c = 512) |
| Estatística | Σ_l x xᵀ (não centrado) | X Ī Xᵀ (centrado, /N) | X Ī Xᵀ (centrado, /N) | Σ_s φ(x_s) |
| Normalização matricial | — | EIG, α = ½, truncar λ < eps(λ₁) | trace-pré + NS N = 5 + trace-pós | — |
| Pós (elemento) | signed-sqrt + ℓ2 | nenhuma (M-ℓ2/M-Fro pioram) | nenhuma | signed-sqrt + ℓ2 (Sec. 4.1.1) |
| Dimensão final | 262 144 | 32 896 | 32 896 | d = 8192 fixo (rec. 2000–8000, Sec. 4.2) |
| Treino | 1) só classificador (convexo), 2) fine-tune ponta a ponta η = 0.001, 45–100 épocas (Sec. IV-A1) | do zero, SGD m 0.9, lr 10^-1.2 → 10^-5 exp. (AlexNet/VGG, ≤ 20 ép.); ResNet-50 90 ép., lr 10^-1.2 ÷10 no platô; init N(0, var 0.01) (Sec. 5.1) | do zero: ResNet 60 ép., bs 256, wd 1e-4, lr 10^-1.1 ÷10 em 30/45; fine-tune FGVC: lr base 10^-2.1, 5× na FC, 50–100 ép. (Sec. 4.3–4.4) | fine-tune a partir da regressão logística treinada, lr 1e-3 const., wd 5e-4 (Sec. 4.1.2) |
| Sem treino (cabeça fixa) | SVM linear C_svm = 1 sobre features ℓ2 (Table II: 80.1/78.0/83.9 vs 84.1/86.6/91.3 com FT) | não reportado | não reportado | TS d = 8192 fixo ≈ bilinear completo sem FT (Table 4) |
| Ablações-chave | — | α ∈ [0.4, 0.9] quase igual; α = ½ melhor (Fig. 3) | N = 5–7 ≥ MPN-COV; N > 7 piora (Fig. 2); trace ≥ Frob (Table 3); **sem pós-compensação não converge** (Table 4) | RM > TS para d baixo; TS > RM para d > 2000; aprender W só ajuda em d muito baixo |

Apoio iB-CNN (Table 1, sem FT, VGG-D): signed-sqrt 80.1/82.9/77.7 → matrix-sqrt + signed-sqrt 82.8/86.7/80.9 (Birds/Cars/Aircraft).

## 4. Regras (checklist)

Redução 1×1 antes da covariância
- [ ] Reduzir os canais para d pequeno ANTES da covariância (MPN-COV/iSQRT: d = 256). Para a cabeça ≥ 1280 do MISSION: d(d+1)/2 ≥ 1280 ⇒ **d = 51 (1326)**; d = 55 (1540 ≥ 1536); d = 64 (2080 ≥ 2048); d = 72 (2628 ≥ 2560).
- [ ] Com treino: conv 1×1 + ReLU (iSQRT Sec. 4.1). Sem treino: projeção linear fixa ajustada só no treino (PCA) — ver Adaptação.

Covariância
- [ ] Centrar por imagem sobre as posições, divisor n (Ī de MPN-COV Eq. 1 / iSQRT Sec. 3.1). B-CNN usa 2º momento não centrado — não misturar sem registrar.
- [ ] fp32 no mínimo; fp64 se eigh.

Normalização matrix-sqrt (Newton–Schulz)
- [ ] Pré: A = Σ / tr(Σ) (Eq. 3). Garantir tr(Σ) > 0 (canal todo morto ⇒ pular / ε).
- [ ] NS acoplado Eq. 2, **N = 5** (N > 7 piora; N = 3 ≈ +0.3% erro).
- [ ] Pós: C = √tr(Σ)·Y_N (Eq. 4) — **obrigatório** quando há treino (sem ela "fails to converge", Table 4).
- [ ] Vetorizar o triângulo superior (d(d+1)/2).
- [ ] Com eigh exato: equivalência — sqrtm(Σ) = √tr·sqrtm(Σ/tr), logo pré+pós são no-op matemático; só importam para a convergência de NS.
- [ ] Autovalores negativos/ínfimos: MPN-COV trunca λ < eps(λ₁) a 0; iB-CNN soma ε·I (ε = 1 na escala deles).

Pós-normalização
- [ ] Treinado (MPN/iSQRT): nada após a sqrt (M-ℓ2/M-Fro pioram, MPN Table 2).
- [ ] Cabeça fixa + classificador linear (B-CNN/iB-CNN): signed-sqrt + ℓ2 **depois** da matrix-sqrt (iB-CNN Sec. 3.2).

Dimensão final
- [ ] Bilinear completo: d(d+1)/2 com d da redução. Compacto: TS com d livre (recomendado 2000–8000; 8192 no paper) → **d = embed_dim exato** (1280/1536/2048/2560) cabe direto em `EMBED_DIMS`.

Treinar junto
- [ ] Ordem B-CNN: primeiro só a última camada, depois ponta a ponta (lr 1e-3). iSQRT fine-tune: lr base 10^-2.1, 5× nas camadas novas.
- [ ] Backprop pela sqrt: usar NS (iSQRT) — o gradiente por EIG tem K_ij = 1/(λ_i − λ_j) (MPN Eq. 10), instável com autovalores repetidos (canais mortos, rank-deficiente).
- [ ] TS: hashes fixos; só os sinais s podem receber gradiente (Sec. 3.2.1) — paper usa W fixo.

Sem treinar (cabeça fixa)
- [ ] Projeção ajustada só no treino; matrix-sqrt (eigh exato ou NS N = 5) + signed-sqrt + ℓ2 + SVM linear (B-CNN Table II; iB-CNN Table 1).
- [ ] TS/RM com W aleatório fixo é válido sem treino (CBP Table 4).

## 5. Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Cabeça treinável de 2ª ordem | MPN/iSQRT: 1×1 → 256 + cov + sqrt, treinado ponta a ponta | `EmbeddingHead` (pooling_head.py:79) só tem 1ª ordem / momentos **por canal** (`moments` = μ, σ, skew, kurt; L62–74): nenhuma covariância entre canais | Adicionar `pooling="isqrt_cov"` (1×1 → d = 51/55/64/72, Σ centrado, NS N = 5, trace pré/pós, triu) e `pooling="tsketch"` (TS d = embed_dim). Relaxar o `assert out_dim == nominal_dim` (L116) para `≥` (MISSION exige ≥ 1280, não igualdade) no `isqrt_cov` |
| Custo da cabeça | 1×1 → 256 | 1×1 → 1280–2560 canais em resolução cheia (L104; doc L153: 16 GB) | `isqrt_cov` com d = 51: conv in×51 + cov HW·51² ≈ 18–21× menos FLOPs que 1×1→1280 (in_ch = 128–256); registrar no orçamento Bk |
| Redução antes da cov (CovReadout) | 1×1 aprendida (+ReLU) | PCA top-k = 64 por bloco, só treino (`CovReadout.fit`, L355–380) | Fiel como versão **sem treino** (projeção linear fixa = 1×1 sem ReLU). Manter linear (ReLU numa projeção fixa só descarta energia). Documentar como adaptação |
| Normalização matricial | NS N = 5 (iSQRT) ou EIG α = ½ (MPN) | eigh fp64 com `clamp(min=0)` (L404; eval_backbone_head L101–105) | Fiel ao MPN-COV (α = ½). Opcional `sqrtm="ns5"` em GPU: elimina o gargalo cusolver (doc L289–297: 75% do tempo de fitness) — **variante nova registrada**, não bitwise igual |
| Truncamento / ε | MPN: λ < eps(λ₁) → 0; iB-CNN: + ε·I | `clamp(min=0)`; ruído de ~1e-17 amplificado por signed-sqrt + StandardScaler (doc `_eigh_host` L240–244) | Adicionar ε relativo: Σ + ε·(tr Σ / k)·I (ε ~ 1e-3) ou truncar λ < 1e-12·λ₁ antes da sqrt: remove as coordenadas só-ruído que mudam κ entre cpu/gpu |
| Pós-normalização | MPN/iSQRT: nada; B-CNN/iB-CNN (fixo): signed-sqrt + ℓ2 | sqrtm → signed-sqrt + ℓ2 por grupo (L383–385, L401–408) e no vetor inteiro (eval_backbone_head L96–98) | **Fiel a iB-CNN** (fixo + linear). Se a cabeça for treinada (isqrt_cov), NÃO aplicar signed-sqrt/ℓ2 dentro da perda SSL; aplicar só na leitura SVM |
| Centragem | Ī (cov centrada /n) | `p - p.mean(1)` ; `/ p.shape[1]` (L392–393) | Fiel (MPN/iSQRT) |
| Dimensão | 32 896 (d = 256) | CovReadout k = 64 → 2080/bloco × blocos + head; so2: k = 48 → 1272 + 8 zeros; k = 88 → 4092 + 4 zeros | Zeros de preenchimento (`pad_to`) não carregam informação; preferir k tal que d(d+1)/2 + 2k ≥ D e não preencher (ou declarar D efetivo). k = 88 > posições do último mapa ⇒ Σ rank-deficiente (MPN usa n = 196 < d = 256, aceitável, mas com truncamento) |
| μ e σ concatenados | MPN/iSQRT só a cov; B-CNN só 2º momento | so2 = [μ, σ, triu(sqrtm Σ)] (eval_backbone_head L113–125) | Extensão fora dos 4 artigos (estilo embedding gaussiano). σ é redundante com diag(Σ); registrar como adaptação; ablação so2 sem σ |
| Cross-cov entre blocos | B-CNN assimétrico f_A f_Bᵀ (Eq. 1), soma nas posições, signed-sqrt + ℓ2 | `xb_vector` centrado, reamostrado por adaptive_avg_pool, sem sqrt (L135–152) | Fiel ao B-CNN assimétrico (centrado é desvio menor). Sem matrix-sqrt possível (não simétrico) — ok |
| Compact bilinear | TS d = 8192 fixo, signed-sqrt + ℓ2 | inexistente | Implementar TS com d = embed_dim exato (cabe na grade 1280–2560), h, s fixos por semente |
| Treino conjunto | B-CNN 2 etapas; lr 1e-3; iSQRT 5× nas camadas novas | head treinada só pela perda SSL/AE com `lr_new 1e-3` / `lr_old 1e-4` | Etapa 1: cabeça com PCA-init congelada (= leitura fixa); etapa 2: liberar 1×1 com lr_new e backbone com lr_old (≈ 10× menor, compatível com "5×") |
| eigh com gradiente | MPN Eq. 10 (K); iB-CNN: Lyapunov > SVD | CovReadout é só `no_grad` | Se backprop pela cov: NS (iSQRT), nunca eigh |

## 6. Adaptação (sempre com init SPiFiL)

1. **Backbone SPiFiL intocado.** A 2ª ordem vive só na cabeça: o principal é "backprop soma ao SPiFiL, nunca desfaz" — a cabeça é adicionada sobre o mapa SPiFiL; nada nela reinicializa o encoder.
2. **Inicialização da redução 1×1 por dados (análogo SPiFiL para a cabeça):** W_1×1 ← top-d autovetores da PCA dos pixels de treino do último bloco (exatamente o `CovReadout.fit`/`PixelMoments.pca`), bias = −Wμ. Com isso a cabeça treinável **começa idêntica** à leitura fixa (morfismo: κ do pai = κ do filho no passo 0), e o treino só soma.
3. **Cabeça fixa (sem treino), receita fiel:** PCA-treino (k) → Σ centrada /n → sqrtm (eigh fp64 ou NS N = 5 com trace pré/pós) → triu → signed-sqrt + ℓ2 → SVM original (`core.svm.fit_svm`, C = 100, nunca mudar). Isso é iB-CNN (Sec. 3.2) com a redução do MPN-COV.
4. **Cabeça treinada:** `isqrt_cov` (d = 51 para ≥ 1280) com NS N = 5 e pós-compensação obrigatória; perda SSL (LeJEPA/SIGReg) sobre o vetor triu; leitura SVM com signed-sqrt + ℓ2.
5. **Dimensão exata da grade:** usar TS (CBP Alg. 2) com d = 1280/1536/2048/2560 e signed-sqrt + ℓ2; h, s fixos semeados (sem treino) — ou RM com W aprendido se d < 2000 (CBP Sec. 4.2).
6. **Controles obrigatórios:** projeção aleatória ortonormal (já existe: `so2_*_rnd`) como controle da PCA; mesma dimensão D nos dois braços; κ de val para seleção, teste só relatório.
