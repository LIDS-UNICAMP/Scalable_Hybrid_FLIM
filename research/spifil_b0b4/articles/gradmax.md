# GradMax — regras de implementação × nossa cadeia

## Referência
Evci, U.; van Merriënboer, B.; Unterthiner, T.; Vladymyrov, M.; Pedregosa, F. *GradMax: Growing Neural Networks using Gradient Information*. ICLR 2022, arXiv:2201.05125.
Lido em https://ar5iv.labs.arxiv.org/html/2201.05125 (17/11/2026). Seções usadas: Fig. 1, §3 (Eqs. 5, 8–12), §4.1–4.2, Fig. 5, Fig. 6, Ap. B (convolução, Eqs. 15, 20, 21), Ap. D.

## Operação
- **Fan-in dos neurônios novos = 0.** Fig. 1: "GradMax sets incoming weights to zero … in order to keep the output unchanged". Com `W_ℓ^new = 0` e `f(0) = 0`, temos `h_ℓ^new = 0`, e então `W_{ℓ+1}^new h_ℓ^new = 0` vale "independent of the training data" (§3). É a preservação exata da função.
- **Fan-out escolhido para maximizar o gradiente.** O problema (Eq. 5) maximiza `‖E[∂L/∂W^new]‖²_F` com `‖W^new‖_F ≤ c`. Com o fan-in zero, o gradiente que importa é o do fan-in (Eqs. 8–11), e a forma reduzida (Eq. 12) tem solução fechada: as colunas de `W_{ℓ+1}^new` são os **k primeiros vetores singulares à esquerda** de `E_D[∂L/∂z_{ℓ+1} h_{ℓ−1}ᵀ]`, escalados por `c/‖(σ₁…σ_k)‖`.
- **Escala**: o §3 normaliza à norma média dos neurônios existentes, citando Dai et al. Já o §4.2 diz que a escala é "simply set" a 0,5. O artigo não reconcilia as duas versões.
- **Convolução (Ap. B)**: usa-se uma conv auxiliar `W_ℓ^aux = 0` de tamanho `(h_ℓ + h_{ℓ+1} − 1) × (w_ℓ + w_{ℓ+1} − 1)`, com `∂L/∂W^aux = ∂L/∂z_{ℓ+1} * h_{ℓ−1}` (Eqs. 15 e 20). Aplica-se im2col, depois SVD (Eq. 21), e os vetores singulares são remodelados em filtros. No MobileNet-v1, a depthwise intermediária é identidade.
- **Ruído e simetria**: não há ruído. Os vetores singulares são ortogonais, então não há simetria.
- **Linhas de base (§4)**: *Random*, *Firefly* (com fan-out pequeno ε = 10⁻⁴, porque "Firefly requires non-zero outgoing weights"), *NeST* (que "modifies the output of existing neurons due to non-zero initialization on both sides") e *splitting* (cria cópias e "can't be used for growing new layers").
- Fig. 5 (centro): ε maior, que **não** preserva a saída, piora a acurácia final.

### Onde e quando crescer
- O cronograma é **fixo**. O artigo declara que não trata de quando nem de onde crescer, e sugere os valores singulares como guia futuro.
- Professor–aluno (MLP): de 20:5:10 até o tamanho do professor em 5 crescimentos, a cada 200 passos. Variante grande: de 100:25:10, +5 neurônios a cada 500 passos, 3500 passos no total.
- Imagens: semente com **0,25× a largura alvo**. Crescem **todas** as camadas, proporcionalmente à largura alvo (na ResNet, só a 1ª conv de cada bloco).
- ImageNet (Fig. 5, esquerda): **crescer cedo** e a cada 5000 passos foi o melhor. Ap. D: "growing early in the training works best".
- **Estagnação**: GradMax **não** exige platô. O artigo contrasta com o splitting, "most effective when the network is very close to convergence". O alinhamento entre o gradiente de minibatch e o completo é alto no início e cai depois (Fig. 6).

## Receita
- CIFAR-10/100: SGD com momento 0,9, decaimento cosseno, 200 épocas, batch 128, lr 0,1 (WRN-28-1) ou 0,05 (VGG-11). O crescimento acontece **durante** esse treino único.
- Professor–aluno: GD com lr constante (melhor 0,1). GradMaxOpt usa Adam.
- ImageNet: batch 512, com SVD no mesmo batch do treino.
- Neurônios novos: **sem tratamento especial** de lr ou otimizador.
- Aceitação e seleção: **nenhuma**. O crescimento é determinístico pelo cronograma. A comparação é com a semente e o alvo treinados do zero.

## Regras (checklist "como no GradMax")
- [ ] Fan-in novo = 0. Ativação com f(0) = 0.
- [ ] Fan-out novo = top-k vetores singulares à esquerda de `E[∂L/∂z_{ℓ+1} h_{ℓ−1}ᵀ]` (conv: conv auxiliar + im2col), com escala c.
- [ ] Cronograma fixo: crescer cedo, em vários passos pequenos, todas as camadas proporcionalmente ao alvo, a partir de 0,25× a largura.
- [ ] Um treino só (SGD + cosseno), sem lr especial para os novos.
- [ ] Sem gate e sem interpolação.

## Divergências

| Item | Artigo | Nosso | Ação |
|---|---|---|---|
| Lado zerado | **Fan-in = 0** | **Fan-out = 0** (fan-in SPiFiL) | Irreconciliável: fan-in ≠ 0 com função preservada exige fan-out = 0. Manter o nosso e declarar o "GradMax transposto". |
| Fan-out | SVD do gradiente (Eq. 12) | Zero | Com fan-in SPiFiL não há como ter fan-out ≠ 0 e preservar a função. Usar o **objetivo** do GradMax na **seleção** (ver Adaptação). |
| Escolha dos neurônios novos | Ótima pelo gradiente (Eq. 5) | SPiFiL sem critério de gradiente; quantidade definida pelo plano | Selecionar os candidatos SPiFiL pela norma do gradiente do seu fan-out zero. |
| Onde e quando | Cronograma fixo, cedo, todas as camadas proporcionalmente | Por faixa: um salto grande (fator ≤ 2,5 por op, widens encadeados) depois do pai treinado | "Como no GradMax": vários crescimentos pequenos **dentro** de um único treino SSL, todas as camadas proporcionalmente, começando cedo. |
| Pai treinado / estagnação | Não exige; cedo é melhor | O pai é uma semente treinada (faixa B_k) | Divergência de regime: o nosso é crescimento **pós-convergência**, o oposto do recomendado aqui. Registrar. |
| Treino | Um SGD cosseno contínuo; novos sem tratamento especial | AE local → LeJEPA curto (lr 1e-3 fixo); herdados a 0,01× | "Como no GradMax": `protect: full`, um SSL só com cosseno. O lowlr é regra nossa (declarar). |
| ε / escala | Escala que preserva a saída é melhor (Fig. 5) | Preservação exata, verificada em eval() fp32 (`finalize` → `measure_preservation`) | **Alinhado** em princípio. |
| Aceitação | Nenhuma | WiSE + κ ≥ pai | Nossa. Para fidelidade, `wise_alphas: [1.0]` sem gate. |

## Adaptação para SPiFiL ("GradMax-SPiFiL", transposto)
Com o fan-out zero, o gradiente no instante do crescimento cai sobre o **fan-out**:
`∂L/∂W_{ℓ+1}^{new} = E[∂L/∂z_{ℓ+1} · h_ℓ^{new ᵀ}]`, com `h_ℓ^{new} = f(W_ℓ^{SPiFiL} * h_{ℓ−1})`.
1. Gerar um **pool** de candidatos SPiFiL maior que o necessário (por exemplo, 4× k) sobre o prefixo do pai, como já faz `NewFilterInit`.
2. Para cada candidato i, medir `s_i = ‖E_batch[∂L/∂z_{ℓ+1} · h_i ᵀ]‖_F`. L é a **perda que vamos treinar** (LeJEPA+SIGReg), calculada no pai em modo de treino com o mesmo batch e as mesmas vistas. Nada de rótulo nem de teste.
3. Ficar com os k de maior `s_i`, de preferência por greedy com ortogonalização (projetar fora os já escolhidos), o análogo dos vetores singulares ortogonais.
4. O fan-out continua 0, a função fica exata e os filtros são SPiFiL. Do GradMax fica o **critério** (máximo gradiente inicial), não a inicialização.
5. Cronograma fiel, como braço separado: a partir da semente B0, crescer de 0,25 em 0,25 até a faixa alvo **dentro** de um único LeJEPA, todas as camadas proporcionalmente, em intervalos fixos de passos, crescendo cedo.
6. Os valores `s_i` também servem de critério de **onde** crescer: a camada com maior Σ s_i, o que o próprio artigo sugere como trabalho futuro.
