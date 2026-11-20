# Net2Net — regras de implementação (fiel ao artigo) × nossa cadeia

## Referência
Chen, T.; Goodfellow, I.; Shlens, J. *Net2Net: Accelerating Learning via Knowledge Transfer*. ICLR 2016, arXiv:1511.05641.
Lido em https://ar5iv.labs.arxiv.org/html/1511.05641 (17/11/2026). Seções usadas: §1, §2.3 (Net2WiderNet, Algoritmo 1), §2.4 (Net2DeeperNet, Fig. 3), §3.1–3.4 (experimentos Inception-BN/ImageNet, Fig. 4), Apêndice A (hiperparâmetros).

## Operação

### Net2WiderNet (§2.3, Algoritmo 1)
- Mapa aleatório `g`: `g(j) = j` para `j ≤ n`; para `j > n`, `g(j)` é sorteado **com reposição** em `{1..n}`. Cada unidade nova é uma **cópia** de uma unidade existente.
- Pesos de entrada (fan-in): `U(i)_{k,j} = W(i)_{k,g(j)}` (coluna copiada).
- Pesos de saída (fan-out): `U(i+1)_{j,h} = W(i+1)_{g(j),h} / |{x : g(x) = g(j)}|`. A saída é **dividida pelo número de réplicas**, e é isso que preserva a função.
- BN: escala e deslocamento seguem o mesmo mapa `g` dos pesos. Em concatenação, o mapa leva em conta os deslocamentos (§2.3).
- Simetria (§2.3): as réplicas são idênticas. Para quebrar a simetria, o artigo diz que "one should add a small amount of noise to all but the first copy of each column of weights". Com dropout, a versão exata é aceitável. A magnitude do ruído não é especificada.

### Net2DeeperNet (§2.4, Fig. 3)
- `h(i) = φ(W(i)ᵀ h(i−1))` vira duas camadas: a nova matriz `U(i)` é **identidade**. Em conv, usam-se filtros identidade.
- Condição: `φ(I φ(v)) = φ(v)` vale para ReLU, não vale para a sigmoide, e o maxout exige uma quase-identidade.
- BN: "we must set the output scale and output bias of the normalization layer to undo the normalization". Isso exige passes de forward em dados de treino para estimar as estatísticas.
- Ruído: não é necessário, mas um pouco de ruído quebra a simetria mais rápido (§2.4).

### Onde e quando crescer
- Não há critério de onde nem de quando. O artigo apresenta o fluxo de trabalho do experimentador (§1): partir do **melhor modelo anterior** (o professor treinado) e treinar o novo "for a shorter period of time".
- Não há teste de estagnação. O professor é treinado antes (§3.2) e não há verificação explícita de convergência.

## Receita (o que se treina depois, e quanto)
- Treina-se a **rede inteira** (professor copiado mais as unidades novas), sem congelar nada e sem lr diferente por elemento.
- Hiperparâmetros (Ap. A): os do professor. "The teacher network does not need to have any of its hyperparameters changed". O **lr inicial do aluno é cerca de 1/10 do lr inicial do professor**.
- Otimizador: o mesmo do professor. Na linha de base Inception-BN foi RMSProp (Ap. B).
- Duração: não há número de épocas. A Fig. 4 mede minibatches: a linha de base aleatória precisa de cerca de 2×10⁶ minibatches a mais para empatar.
- Aceitação e seleção: **nenhuma**. O aluno é simplesmente treinado. Não há volta ao pai nem interpolação.
- Resultado (§3.2): a acurácia final é a mesma do treino do zero, só que alcançada mais rápido. Em §3.4 a exploração de desenho chega a 78,5% top-1.

## Regras (checklist "como no Net2Net")
- [ ] Unidade nova = réplica de uma unidade existente escolhida por `g` sorteado com reposição.
- [ ] Fan-out das réplicas dividido pela contagem `|{x: g(x)=g(j)}|`, aplicado também ao original.
- [ ] BN remapeado por `g`.
- [ ] Ruído pequeno em todas as cópias menos a primeira (ou dropout ligado).
- [ ] Deeper: camada identidade + ReLU. BN ajustado com estatísticas de dados reais para desfazer a normalização.
- [ ] Treino contínuo da rede inteira com o otimizador do pai e lr inicial = lr_pai/10.
- [ ] Sem gate de aceitação e sem interpolação com o pai.

## Divergências

| Item | Artigo | Nosso (`morphisms.py`, `growth_chain.py`) | Ação |
|---|---|---|---|
| Fan-in do canal novo | Cópia de um filtro existente (`g`) | Filtro SPiFiL ajustado sobre o prefixo do pai (`NewFilterInit`, growth_chain.py:642) | **Manter**: regra do projeto. Divergência irreconciliável. |
| Fan-out do canal novo | Cópia dividida pelo nº de réplicas (o original também é dividido) | Colunas do consumidor = 0 (`_remap_conv`, morphisms.py:109). O fan-out herdado fica intacto | **Manter**: dividir só preserva a função quando as réplicas são idênticas, e o filtro SPiFiL não é réplica. Dividir também alteraria o filtro herdado (o backprop "desfaria"). |
| Quebra de simetria | Ruído pequeno nas cópias | Não há simetria, porque cada filtro SPiFiL é distinto | Nenhuma. Registrar que o ruído é dispensável. |
| Deeper | Identidade + ReLU; BN calibrado nos dados | `deepen_identity` (morphisms.py:429): centro = I; BN com rm=0, rv=1, γ=√(1+eps), exato só em eval() | Para ficar fiel, calibrar o BN com estatísticas de dados (forward no treino) em vez de rm=0/rv=1. **Mas a identidade não é SPiFiL**: ver Adaptação. |
| Fallback aleatório | — (o artigo usa random só na linha de base) | `NewFilterInit` cai em init PyTorch para conv não densa, inception intercalado e exceção, e as linhas excedentes (`n < len(new_rows)`) ficam aleatórias (growth_chain.py:664-704) | **Corrigir (regra nossa)**: falha do SPiFiL deve rejeitar o plano, nunca preencher com aleatório. |
| O que treina | Rede inteira, mesmo lr (1/10) | Herdados a 0,01×lr (`PROTECT_SCALE["lowlr"]`, growth_chain.py:66), novos a lr cheio | "Como no Net2Net": `protect: full` com lr = lr_pai/10 em tudo. Hoje o lr é 1e-3 fixo = lr_new do pai (`ga_search/config/hmodehb_b0b4/base.yaml:103`), sem o /10. |
| Treino | Otimizador e cronograma do pai, contínuo | AE local (Adam 1e-3) → LeJEPA+SIGReg (AdamW 1e-3 fixo, wd 0), 15/45 épocas divididas 1:2 | Para ficar fiel: sem AE local, só o SSL do pai (LeJEPA, mesmo cronograma) com lr/10. |
| Aceitação | Nenhuma | WiSE α∈{0,.25,.5,.75,1} pela ORIG-val + κ_filho ≥ κ_pai (`decide`/`accept`, growth_chain.py:1329-1350) | O WiSE e o gate são nossos. Para ser fiel, rodar o braço `wise_alphas: [1.0]` sem gate (só registro) como ablação. |
| Quando | Depois do pai treinado ("melhor modelo anterior") | Por faixa (B_k → B_{k+1}), sem checar convergência do pai | Registrar a fidelidade de treino do pai (as épocas até a semente). Ver o gate de estagnação em `firefly_splitting.md`. |

## Adaptação para SPiFiL ("Net2Net-SPiFiL")
1. **Wider**: o fan-in é sempre SPiFiL e o fan-out é 0. Do Net2Net fica a **preservação exata da função** (FPI) e o **treino contínuo com lr/10**, não a cópia. Escrever nos relatórios: "Net2Net em função (FPI), não em operação".
2. **Deeper**: como a identidade não é um filtro SPiFiL, há duas saídas compatíveis com a regra:
   (a) usar `deepen_identity` só como operador **estrutural**, desde que o usuário aceite a identidade como "filtro herdado-equivalente", porque reproduz o mapa do pai;
   (b) preferido: `deepen_identity` seguido de `widen` da camada inserida com k filtros **SPiFiL** de fan-out 0, o que dá mais capacidade sem filtro aleatório. Calibrar o BN da camada inserida com dados (§2.4).
3. Receita fiel: `protect: full`, `lr = lr_pai/10`, `ssl = <SSL do pai>`, sem AE local, `wise_alphas: [1.0]`, aceitação só registrada. Para respeitar "o backprop soma, não desfaz", a variante intermediária é `protect: lowlr` com lr/10 nos novos. Isso deixa de ser fiel e precisa ser declarado.
4. Corrigir o fallback: SPiFiL que falha ou entrega menos filtros do que o pedido leva o plano a `infeasible_spifil`, nunca a init PyTorch.
