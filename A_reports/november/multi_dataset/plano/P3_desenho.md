# P3: desenho do experimento estatístico

07/11/2026. Só plano: nada foi lançado, treinado ou commitado. Simulações em CPU, `nice -n 10`, scripts em `<scratchpad>/p3/`.
Lê junto com `P1_literatura.md`, `P2_datasets.md` e `P4_arquitetura.md`, que saíram enquanto eu escrevia. Papéis dos datasets vêm de P2 e a família B0 a B3 vem de P4. Onde discordo, está na seção 7.

**Resposta curta:** recomendo o orçamento **médio: 655 cadeias, 468 GPU-horas, 5 a 7 dias nas 4 GPUs.** Uma "cadeia" é um modelo treinado do começo ao fim, estágio por estágio.

O que a simulação e os dados reais dizem, em ordem de importância:

1. **O expoente de dados `b` sai fácil, desde que as frações baixas entrem.** Com 4 frações e 3 splits ele cai em ±0,1 em 91% das vezes. Só com as frações de 25% para cima ele não sai (largura 1,72 contra 0,23).
2. **O expoente de tamanho `a` não sai por dataset, em nenhum desenho que caiba.** Compartilhado, só sai se o tamanho cobrir mais de 2 décadas. B0 a B3 cobre 1 década, então precisa de 3 tamanhos extras.
3. **α, β, γ saem grosseiros.** Dá para dizer "a largura leva a maior parte", não dá para separar β=1,21 de β=1,35. A regra final deve ser "ajusta, pega os 3 melhores por orçamento e **mede de novo**": isso corta a perda pela metade quando o modelo da superfície está errado.
4. **A lei de aprendizado já aparece nos dados reais:** 12 datasets, 4 modelos congelados, `b` entre 0,47 e 0,70. Isso é a curva de rótulos com encoder fixo. Falta medir no nosso regime.

**Três consertos antes de qualquer run novo** (seção 0): tirar `best_by_probe`, resolver a cabeça com `fill_random`, e **parar os braços `c2560`: eles morrem na L4** (reproduzi o erro em CPU).

**Marcas:** [medido] saída de comando ou log. [sim] simulação, com premissas minhas. [conta] aritmética. [hip] hipótese.

---

## 0. O que o pedido proíbe e o código ainda faz

A regra nova: todo modelo nasce do SPiFiL, treina sem rótulo, com reconstrução + perda de `losses/`, e rótulo só no SVM.

| Regra | O que acontece hoje | Evidência | O que este desenho faz |
|---|---|---|---|
| Sem rótulo no treino | `best_by_probe: true` escolhe o checkpoint pelo κ de validação, e esse peso alimenta o próximo `grow` | `autoencoder_flim_module.py:741-742`; `ckpt_monitor: probe/lin_select_kappa` no `run_metadata.json` | `best_by_probe: false`, monitor `val_recon_loss` (`:745-746`, só YAML) |
| Idem | A sonda também é ruim: 6 réplicas exatas (mesma semente, mesmo split) têm `val_recon` igual até 1e-5, e κ da sonda com desvio 0,02 a 0,04 na L2 e L3 | [medido] `p3/replicas.py`, `sc` w1 e w2, helminth_eggs | Desliga a sonda; mantém em 1 braço de F0 para o monitor por época |
| Idem | A sonda gasta 30% a 53% da época (mediana 42%) | [medido] `probe_s/epoch_s`, 29 estágios de `sc` | Entra na conta como economia |
| Rótulo só no SVM | `one_per_class` usa rótulo para sortear imagens, e o alocador divide filtros por classe | `flim/spifil.py:534-545`, `:929` | **Pergunta ao usuário** (seção 7). P4 §8 já declara isso |
| Nunca aleatório | `fill_random_last: true`: em helminth_eggs, 450 de 1280 filtros da cabeça nascem do SPiFiL; com 2 classes, 100 de 1280 | [medido] log do grow; `flim/spifil.py:971-983` | Vira fator: braços H0, H1, H2 de P4 §8 |
| Leitura abaixo de 2000-d | `c=2560` passa do limite e a L4 morre | [medido] `ValueError epoch_probe gem: 2560 canais passam de 2000 dimensoes`, reproduzido com a `architecture.json` real; guarda em `:486-488` | c = 1280 fixa, como em P4 |
| Perda como fator | **Nenhum YAML usa `reg_loss`.** Os λ de R8 são previsão por gradiente, em 1 checkpoint | [medido] grep em `experiments/autoencoder/`: 0 arquivos | F1 mede pela primeira vez |

---

## 1. Hipóteses

Duas perguntas separadas e uma ponte entre elas (P1 §6):
- **Lei de aprendizado:** mesma rede, mais imagens. Quanto o erro cai?
- **Lei arquitetural:** mesmas imagens, mesmo custo, outra forma de rede. Qual forma erra menos?
- **Ponte:** a melhor forma muda quando há mais imagens?

\* marca as 7 primárias, corrigidas por Holm. O resto é exploratório: relata, não decide. Os nomes começam com H para não confundir com os modelos B0 a B3 nem com os relatórios P1, B1, B2: HA aprendizado, HF forma, HX ponte, HP perda, HS SPiFiL.

| # | Hipótese | Teste | Passa se |
|---|---|---|---|
| **Lei de aprendizado** | | | |
| HA1\* | e(D)=E+B·D^−b descreve cada curva | Resíduo contra ruído medido em F0 | Resíduo ≤ 1,5σ em ≥ 5 de 7 datasets; b entre 0,3 e 1,0 |
| HA2\* | O expoente é parecido entre famílias | Individual contra família contra único, por AICc e por deixar 1 família de fora | Compartilhado empata (ΔAICc < 2) e o erro de deixar família de fora é ≤ 1,3× o do individual |
| HA3 | A curva prevê o que não viu | Ajusta até 50%, prevê 75% e 100% | \|Δκ\| mediano ≤ 0,015 |
| HA4 | Tamanho muda E e B, não b | b de B1 contra b de B3, mesmo dataset | Diferença dentro do intervalo |
| HA5 | Imagens de SSL e rótulos são eixos diferentes | Grade n_ssl × n_svm por `svm_train_pct`, só CPU | b_ssl e b_rótulo com intervalos separados |
| HA6 | A curva mede dados, não passos | 5% com os mesmos passos de 100%, em 2 datasets | Diferença ≤ 0,02; senão, passos entram na lei |
| **Lei arquitetural** | | | |
| HF1\* | A forma importa a custo igual | Melhor menos pior forma medida em B2 e B3, pareado | ≥ 0,02 em ≥ 3 dos 5 decisores. Senão: superfície plana, regra vira "menor modelo suficiente" |
| HF2\* | Crescer junto vence crescer 1 eixo | Melhor forma contra melhor eixo único, a custo igual | ≥ +0,01 agrupado, ou mesmo κ com metade do custo |
| HF3\* | A divisão do orçamento muda por dataset | Dá a cada dataset a divisão de outra família | Perde ≥ 0,005 agrupado. Senão: coeficiente individual é enfeite, fica o compartilhado |
| HF4 | A resolução satura no tamanho nativo | r\* contra lado nativo, em Eggs, Larvae, flowers | r\* ≤ nativo em 3 de 3 |
| HF5 | Profundidade segue log da resolução, ou não paga | d\* contra r; braços `sc_*_d*` | d\* cresce com r, ou d=3 fica a 0,01 do melhor |
| **Ponte** | | | |
| HX1 | A melhor forma vale a 5% e a 50% | Top 2 de B2 remedidos a 5% | Mesma ordem, ou diferença ≤ 0,01 |
| **Perda** | | | |
| HP1\* | Alguma perda com λ ≤ médio vence só reconstrução | Pareado em 5 decisores × 3 splits | ≥ +0,01 agrupado, intervalo sem 0 |
| HP2 | O λ escolhido em B0 serve em B3 | Melhor λ de B0 remedido em B3 | Perde ≤ 0,005 |
| HP3 | Dá para escolher perda sem rótulo | Dimensão efetiva de z contra κ do SVM, entre braços | Spearman ≥ 0,7 |
| **SPiFiL** | | | |
| HS1\* | SPiFiL vence aleatório na mesma forma | Pareado, amostra de controle | ≥ +0,01, intervalo sem 0, mesmo sinal em ≥ 2 de 3 famílias |
| HS2 | Cabeça só SPiFiL não perde | H1 e H2 contra H0, em B0 e B3 | Perde ≤ 0,01; aí H0 sai |

**Critério de lei e critério de meta são coisas diferentes.** A lei pode prever bem e o modelo ainda ficar longe do benchmark. A meta de κ está na seção 5.

---

## 2. Variáveis

**Papéis (P2, opção A por padrão).** Família inteira no mesmo papel. Conta como 1 unidade: a descoberta tem **3 unidades**, não 7.

| Papel | Datasets |
|---|---|
| Descoberta, r livre | Eggs, Larvae, flowers |
| Descoberta, r preso em 200 px | helminth_eggs, Eggs_subgroup, helminth_larvae, Larvae_subgroup |
| Confirmação | Protozoan, protozoan_cysts, Protozoan_subgroup |
| Externa | aid, remote_sensing (opção B de P2: coconut_trees, bloodmnist) |

**Decisores** (P2 §7, os que enxergam diferença): helminth_eggs, Eggs, flowers, Eggs_subgroup, helminth_larvae. Larvae e Larvae_subgroup só dão direção.
**Representantes**, um por família, para triagem: helminth_eggs, helminth_larvae, flowers.

| Tipo | Variável | Níveis |
|---|---|---|
| **Independente: dados** | Imagens de treino D | Contagem real, em 1, 5, 10, 25, 50, 75, 100%. A de 10% já existe no disco |
| | Rótulos no SVM | 1, 5, 25% com encoder a 100% (`svm_train_pct`, sem treino) |
| **Independente: forma** | d, convoluções espaciais | 3, 4, 5, 6 |
| | w, multiplicador de largura | 0,5 a 4; ¼, ½ e 6 só no bloco de `a` |
| | r, lado da imagem | r livre: 128 a min(640, nativo). r preso: 200 |
| **Independente: perda** | R(z) | Só reconstrução, Weak-SIGReg, Strong-SIGReg, VICReg, VICRegL, Barlow, CorInfoMax, UR-JEPA |
| | λ, médio e baixo | {0,01; 0,001}. VICRegL e CorInfoMax: {0,001; 0,0001}, porque R8 §2.3-2.4 mostra gradiente alto já em 0,001. São **15 braços** |
| **Independente: início** | Cabeça | H0, H1, H2 (P4 §8) |
| | Controle | SPiFiL sempre; `random_layer: true` só na amostra |
| **Bloco** | Dataset, família, split, semente | Seção 4.4 |
| **Fixa** | | **50 superpixels DISF**, sem máscara, `raw/`, c=1280 (múltiplo de C), kernels 5,5,5,3,1, pool nas 3 primeiras, 60 épocas por estágio, lote efetivo 32, lr 5e-4, backprop 0,1, leitura `gem_tta_pw` abaixo de 2000-d, val e teste fixos (conferido nos 12) |
| **Resposta principal** | κ de validação do SVM, como e=1−κ | CSV do eval |
| **Resposta secundária** | F1, acurácia | Mesmo CSV: colunas `f1`, `acc` |
| **Custo** | FLOPs e parâmetros | Fórmula de P4 §3 sobre a largura **real** (`architecture.json`), offline |
| | Tempo; memória | `elapsed_s`, `epoch_s`; pico exige P4 §12 item 7 |
| **Covariável registrada** | | Largura real, fração SPiFiL da cabeça, dimensão efetiva de z, canais mortos, `reg_ratio`, `val_recon_loss` |

- **Largura nominal engana.** 32/48/64 viram 27/45/63 com 9 classes [medido, `channels` no `run_metadata.json`]. Ajuste sempre pela real.
- **Teste:** fora de todo ajuste. Abre uma vez, no fim, depois da lei congelada.
- **Cuidado com lote.** A 640 px, com w alto, o lote cai para 8 ou 16 (agente B2). As perdas estimam estatística dentro do lote, então r e perda se misturam. Regra: r ≥ 448 só com lote ≥ 16.

---

## 3. Equações e como comparar

### 3.1 Lei de aprendizado

e(D) = E + B·D^(−b)

- E é o piso, B é a escala, b é a velocidade. O acaso é e=1 em todo dataset (P1 R4).
- Ajuste em e, nunca em log, com D em imagens reais.
- Peso pelo ruído medido: σ(p) = σ₅₀·(50/p)^0,35 [medido, benchmark: mediana 0,033 a 1% e 0,0085 a 50%].

Quatro versões, da mais livre à mais presa:

| Versão | Constantes | Quando ganha |
|---|---|---|
| Individual: E, B, b por dataset | 3K | Datasets diferem de verdade |
| Por família: E, B por dataset, b por família | 2K + F | Irmãos parecidos |
| Compartilhada: b único | 2K + 1 | Lei universal |
| **Hierárquica:** b_i = μ + família + δ_i, com encolhimento | entre as duas | Padrão para reportar |

Com split como efeito aleatório, porque frações aninhadas fazem 18 pontos valerem 3 curvas (P2 §6):

e_is(D) = E_i + B_i·D^(−b_i) + u_is + ε, com u_is ~ N(0, σ²_split)

Sem dependência nova: `scipy` e `statsmodels` estão no env; `numpyro` e `pymc` não [medido]. O encolhimento é empírico (média ponderada do dataset com o grupo).

### 3.2 Lei conjunta

e(N, D) = E + A·N^(−a) + B·D^(−b)

São 5 constantes. Variantes: a e b compartilhados; e, se a soma falhar, o termo de interação de P1 E5, e = E + A·x^(−a) + B·x^c·D^(−b), que dá o tamanho ótimo x\* ∝ D^s.

### 3.3 Lei arquitetural

**Superfície por dataset, 7 coeficientes** (5 com r preso):

e(d,w,r) = c₀ + c_d·ln d + c_w·ln w + c_r·ln r + q_d·(ln d)² + q_w·(ln w)² + q_r·(ln r)²

**α·β²·γ²≈2 não vale aqui.** Conferi independente de P4, com a mesma conclusão [conta, `p3/arch_cost.py`]:

| Eixo | Quanto o custo cresce | EfficientNet supõe |
|---|---|---|
| r | r^2,0 | r² |
| w | w^1,81 a w^1,92 | w² |
| d (camada no fim, na grade pequena) | d^0,11 a d^0,13 | d¹ |

Aplicar 1,2 / 1,1 / 1,15 aqui dá 1,6× / 2,6× / 4,2× de custo em φ=1, 2, 3, não 2× / 4× / 8×. A restrição real é **β^1,8·γ² ≈ 2, com d fora da conta** (P4 §3). Quem aprova é a fórmula exata de MACs.

**Como sai cada coeficiente.** α, β, γ são a inclinação de ln d\*, ln w\*, ln r\* contra φ = log₂(F/F₀), ao longo da fronteira. Com poucos datasets, use **uma constante só** (P1 E7):

π = parcela do orçamento que vai para largura, com β = 2^(π/1,8) e γ = 2^((1−π)/2)

O prior de P4 é π=0,5, que dá β=1,21 e γ=1,19. Com r preso, π=1 e β=1,47. Comparar π entre datasets é a forma mais honesta de testar HF3.

### 3.4 Como decidir entre as versões

Escrito antes de ver dado:

1. **AICc** compara versões nos mesmos pontos. Mais constantes só ganham com ΔAICc ≥ 2.
2. **Deixar 1 família de fora** (3 voltas) mede transferência. É o juiz: se discordar do AICc, ele vence.
3. **Previsão fora da amostra:** ajusta até 50% e prevê 75% e 100%; ajusta até B2 e prevê B3.
4. **Intervalo:** perfil de verossimilhança e bootstrap paramétrico. Bootstrap de split não serve, pois 3 splits dão só 10 reamostras (L_data §1).
5. Otimizador com grade de inícios, perda em soma e parada apertada (Besiroglu, via P1 R6).

**Aviso sobre poder** [sim, `sim_learn.py`]. Com expoentes realmente diferentes (τ=0,13), o AICc escolhe "individual" em 71% das vezes com 3 unidades de baixo ruído, e em só 26% com ruído alto. Ou seja: **"compartilhada venceu" pode ser falta de poder, não prova de lei.** Sempre reportar τ com intervalo.

### 3.5 Piloto real: a lei de aprendizado já aparece

Curvas congeladas do benchmark, 12 datasets × 6 frações × 3 splits, **só colunas `val_*`** [medido, `p3/bench_fit.py`]. Não é o nosso regime: é encoder fixo e mais rótulos. Serve para testar o método.

| Modelo | b mediano (quartis) | AICc individual / família / único | Prever 100% com ≤50%: erro médio (máx) | Prever 100% com ≤25%: b próprio / b de outras famílias |
|---|---|---|---|---|
| AIMv2 | 0,70 (0,51 a 0,74) | −1721 / −1736 / **−1741** | 0,009 (0,036) | 0,011 / 0,012 |
| ConvNeXt V2 | 0,47 (0,43 a 0,67) | −1418 / −1433 / **−1440** | 0,007 (0,012) | 0,015 / 0,014 |
| DINOv2 | 0,66 (0,62 a 0,77) | −1778 / **−1795** / −1773 | 0,002 (0,006) | 0,007 / 0,010 |
| I-JEPA | 0,49 (0,42 a 0,51) | −1550 / **−1569** / −1563 | 0,011 (0,071) | 0,012 / 0,016 |

Leitura: em 4 de 4, família ou único vence o individual. O b emprestado de outras famílias prevê quase tão bem quanto o próprio. É o que a meta do usuário sugere: **lei de aprendizado consistente entre datasets.**

---

## 4. Como cortar treinos

### 4.1 Quantos pontos identificam cada expoente

Premissas [sim]: b=0,6, E=0,02, ruído calibrado no benchmark. "Faixa" é onde caem 95% das estimativas.

**Expoente de dados b, um dataset** (`sim_learn.py`, `sim_cells.py`, `sim_cells2.py`):

| Frações | Splits | Faixa, ruído baixo (σ₅₀=0,0085) | Largura | Faixa, ruído alto (σ₅₀=0,022) |
|---|---|---|---|---|
| 5, 25, 100 | 1 | 0,37 a 1,56 | 1,19 | 0,22 a 2,00 |
| 5, 25, 100 | 3 | 0,42 a 1,04 | 0,62 | 0,33 a 2,00 |
| 1, 5, 25, 100 | 1 | 0,46 a 0,83 | 0,37 | 0,37 a 1,46 |
| **1, 5, 25, 100** | **3** | **0,50 a 0,73** | **0,23** | 0,43 a 0,98 |
| 1, 5, 25, 50, 75, 100 | 3 | 0,50 a 0,73 | 0,23 | 0,45 a 0,96 |
| 1, 5, 10, 25, 50, 75, 100 | 3 | 0,51 a 0,70 | 0,19 | 0,45 a 0,90 |
| Só células L de Eggs: 25 a 100 | 3 | **0,28 a 2,00** | **1,72** | 0,05 a 2,00 |
| Células L + a nova de 10% | 3 | 0,38 a 1,27 | 0,89 | 0,27 a 2,00 |
| Só células L de flowers: 5 a 100 | 3 | 0,42 a 1,01 | 0,59 | 0,34 a 2,00 |

- **Piso: 4 frações, com alguma abaixo de 5%, e 3 splits.** Passar de 4 para 6 não estreita b. Só melhora a previsão de 100%: erro 0,0075 → 0,0042.
- **O que manda é o alcance em D, não o número de pontos.**
- **Em dados reais a regra se confirma:** usando só células L, 9 de 28 curvas batem na borda (b=2,00). Com as 6 frações, nenhuma, e o maior b é 1,20 [medido].
- **Nos 7 datasets juntos, com o ruído de P2:** só L dá largura 0,65 a 0,74. Com L e R, peso pelo ruído, 0,14 a 0,28.

**Expoente de tamanho a** (`sim_joint.py`, `sim_joint3.py`, ruído 0,01). Erro de â, faixa de 95%:

| Desenho por dataset | Alcance em N | Runs | Um dataset, a=0,3 | Compartilhado, a=0,3 | Compartilhado, a=0,15 |
|---|---|---|---|---|---|
| B0 a B3 × 6 frações × 3 splits | 1 década | 72 | −0,23 a +0,95 | K=3: −0,15 a +0,90 | K=3: −0,08 a +1,08 |
| 4 tamanhos × 6 frações × 3 splits | 2,2 décadas | 72 | −0,16 a +0,33 | K=4: −0,08 a +0,16 | K=4: −0,06 a +0,28 |
| 7 tamanhos × 3 frações × 3 splits | 2,2 décadas | 63 | −0,17 a +0,34 | K=3: −0,12 a +0,20 | K=3: −0,07 a +0,43 |
| 6 tamanhos × 6 frações × 3 splits | 2,2 décadas | 108 | −0,15 a +0,26 | K=4: −0,09 a +0,14 | K=4: −0,06 a +0,24 |

- **Por dataset, `a` não sai.** P(erro ≤ 0,1) nunca passa de 0,72.
- **Alcance vale mais que pontos.** Mesmos 72 runs, com 2,2 décadas em vez de 1, levam o erro de +0,90 (K=3) a +0,16 (K=4). Comparando no mesmo K=6: +0,66 contra +0,15.
- **Se a≈0,15** (L_data: dobrar N tira ~10% do erro), o melhor resultado é uma cota: "a < 0,4". Não publique um número.
- **`b` não sofre:** nos desenhos com frações baixas, erro de ±0,1.

**Forma sob orçamento** (`sim_arch2.py`, ruído 0,01, 1 split). A superfície "certa" tem a mesma forma do ajuste. A "errada" tem joelho em r, queda brusca em d e interação forte: é a que protege de otimismo.

| Desenho | Pontos | Custo relativo | Perda média, certa | P(perda ≤ 0,01) | Perda média, errada | P(perda ≤ 0,01) |
|---|---|---|---|---|---|---|
| Estrela, um eixo por vez | 7 | 13 | 0,0059 | 0,80 | 0,0102 | 0,58 |
| Fatorial fracionado 3^(3−1) | 9 | 47 | 0,0036 | 0,90 | 0,0089 | 0,63 |
| **Latin hypercube** | **12** | 72 | 0,0025 | 0,94 | **0,0053** | **0,79** |
| Central composto | 15 | 76 | 0,0024 | 0,94 | 0,0064 | 0,73 |
| Latin hypercube | 16 | 117 | 0,0022 | 0,95 | 0,0064 | 0,74 |
| Latin hypercube | 24 | 139 | 0,0014 | 0,98 | 0,0058 | 0,77 |
| Fatorial completo | 27 | 134 | 0,0012 | 0,98 | 0,0057 | 0,77 |

Mais pontos não resolvem a superfície errada: **a perda trava em ~0,006, e o que trava é o modelo, não o ruído.** O que resolve é medir de novo (`sim_ladder.py`, escada de P4, r livre, 3 splits no remedido):

| Plano | Runs por dataset | Perda, certa | Perda, errada | P(≤ 0,01), errada |
|---|---|---|---|---|
| LHS 6 | 6 | 0,0101 | 0,0222 | 0,46 (6 pontos para 7 coeficientes: falha) |
| LHS 8 | 8 | 0,0031 | 0,0061 | 0,75 |
| LHS 16 | 16 | 0,0015 | 0,0057 | 0,77 |
| LHS 8 + top 3 remedidos | 35 | 0,0023 | 0,0039 | 0,85 |
| LHS 12 + top 2 remedidos | 30 | 0,0024 | 0,0047 | 0,81 |
| **LHS 16 + top 3 remedidos** | 43 | 0,0016 | **0,0036** | **0,86** |

- **Com r preso, quase não há o que amostrar.** São só 4 a 6 candidatos por orçamento. Remedir o top 3 já é medir quase tudo: perda 0,0007 [sim].
- **Ruído 0,02 custa caro:** LHS 16 + top 3 fica em 0,0055, P=0,78. Por isso F0 mede o ruído primeiro.
- **α, β, γ:** erro de p90 em ln, por dobra de custo: 0,04 em α, 0,06 a 0,12 em β, 0,07 a 0,10 em γ. Na superfície errada, 0,10 a 0,16. Separa β=1,21 de 1,35 no limite, e não separa diferenças menores. **Não cite os valores de α, β, γ da simulação:** eles vêm das premissas.
- **Quanto a regra por dataset ganha da melhor config comum:** perda 0,002 a 0,006 contra 0,0065 a 0,0105, ou seja ~0,004 de κ. Fica abaixo do que 3 splits enxergam (P2: 0,021 no melhor caso). Confirma o aviso de L_method §5: a regra ganha em custo, não em κ.

### 4.2 Técnica por técnica

| Técnica | Decisão | Por quê |
|---|---|---|
| Fatorial fracionado | Não | Pior que LHS no mesmo custo quando o modelo erra (0,0089 contra 0,0053) |
| **Latin hypercube** | **Sim, em F2** | Cobre o espaço sem depender da forma. Precisa de ≥ 8 pontos, para ter mais pontos que coeficientes |
| **Busca sucessiva** | **Sim, em F1** | 15 braços: 59 cadeias-equivalentes contra 180 de "tudo com 3 splits". Perda 0,0012 contra 0,0015 de "tudo com 1 split", no mesmo custo [sim, ρ=0,7, ruído 0,01, 4 datasets, `sim_sha.py`] |
| Top-k remedido | **Sim, em F2** | Tabela acima |
| Frações só na fronteira | **Sim, em F3** | 6 frações em todo ponto do LHS custariam ×6 |
| Eixo de rótulos | **Sim, grátis** | `svm_train_pct` reusa encoder pronto; só CPU |
| Sonda desligada | **Sim** | −42% do tempo e tira rótulo do treino |
| Épocas curtas, 15 por estágio | Só no degrau 1 de F1 | ρ ainda não medido. Os braços que passam têm 15 e 60 épocas: F1 mede ρ de graça. Se ρ < 0,5, volta a 30 épocas |
| Proxy L3 no lugar de L4 | Reserva | ρ = 0,69, 0,72, 0,69 (n=28, 16, 11), melhor final em 2º ou 3º no proxy [medido, `p3/proxy.py`; leitura flatten, split antigo] |
| Perda de reconstrução como proxy | **Não** | Igual até 1e-5 entre réplicas e no piso (R8); P1 também: "perda baixa não garante κ" |
| Sonda por época como proxy | **Não** | Desvio 0,02 a 0,04 em réplica exata |
| `check_val_every_n_epoch: 5` | Sim, não contado | ~−20% [conta], só YAML. É folga |
| Bifurcar de estágio pronto | Não contado | Exige código (L_method §4) |

### 4.3 O funil

| Fase | Pergunta | O que roda | Depende de |
|---|---|---|---|
| **F0** | Quanto é o ruído? | B0, representantes, 3 sementes × 3 splits. Mais 6 SVMs nas réplicas de `sc` (CPU) | Nada |
| **F1** | Qual perda e λ? | Degrau 1: 15 braços × 3 representantes, 15 épocas. Degrau 2: top 5, cheio, + Eggs. Degrau 3: top 2 + reconstrução, 3 splits, 5 decisores | Roda junto de F0 |
| **F2** | Qual forma por orçamento? | LHS por dataset, 50%, 1 split, melhor perda. Depois top k por orçamento, 3 splits. Braços de cabeça | F1 |
| **F3** | Lei de aprendizado | Fronteira × frações. Ordem: 100, 5, 25, 1, 10, 75. Bloco extra de `a`, controle de passos, ponte a 5% | F2 |
| **Congela** | | Lei escrita, com previsão de cada dataset (P2) | F3 |
| **F4** | Confirmação | Protozoan, previsão antes de treinar | Congela |
| **F5** | Externa | Só depois de F4. Teste abre aqui, uma vez | F4 |

- A amostra de controle aleatório e os braços de regra rodam junto de F3.
- **Controle aleatório reduzido a uma amostra:** 18 cadeias no médio, contra 108 na fila `sc`. Com 18 pares e desvio 0,02, a faixa da diferença é ±0,010 [conta].

### 4.4 Sementes

Pense em três dados diferentes: qual prova caiu para o aluno (split), o humor do aluno no dia (semente) e o lápis que falhou (GPU).

| Fonte | O que varia | Tamanho conhecido | Quantas |
|---|---|---|---|
| **Split** | Quem está em treino, validação, teste | 0,007 a 0,056 (P2 §7). É o maior ruído | **3**, todos os que existem, em todo ponto que decide |
| **Semente** | Sorteio de imagens do `grow` e ordem do SGD. Hoje é 42 em tudo. Mudar `seed_everything`, `model.seed` e `runner_args.seed` juntos | Dois treinos iguais diferem 0,0001 a 0,0035 em Eggs (L_data §2.1), mas sem variar o sorteio do SPiFiL | **3 só em F0**, nos representantes. Depois 1 por split |
| **GPU** | Não-determinismo | Grátis: 6 réplicas em `sc` | Só SVM |

Regra: **tamanhos antes de sementes** (P1 R8). O LHS usa só o split 1 e o top k usa 3 splits: o valor que vai para a tabela vem do remedido, nunca do ponto que foi escolhido, para não pagar o viés de "melhor de N" (L_data §4.5: ~+0,01 só por sorte).

### 4.5 Vazão real

| Medida | Valor | Fonte |
|---|---|---|
| Cadeia helminth_eggs, 4 estágios, 60 épocas, com sonda | 1,3 a 1,9 h por run | [medido] `elapsed_s`, `sc`, 11 a 13 runs juntos |
| Velocidade por run a 200 px, w1, sem sonda | 213 a 332 img/s | [medido] `epoch_s − probe_s`, ~3 runs por GPU |
| w2 | 110 a 200 img/s | [medido], `sc_w2_*` |
| w4 (só L1 até agora) | 91 img/s | [medido] 01:25, `sc_w4_d6` |
| 320 px | 106 a 142 img/s | [medido] flowers, `x1`, `x2`, `x3` |
| 640 px | 46 (aid), 29 (remote_sensing), 1 run por GPU | [medido] `x1`, sem cache; agente B1: 35 |
| SVM | 21 a 45 s; Protozoan 260 a 520 s; benchmark 30 a 100 s | [medido] logs de eval, `bench.status` |
| CPU | crispim ~54 núcleos, moliveira ~38, de 96; load 70 a 105 | [medido] `ps`, 01:07 e 01:25 |
| GPU | 19 a 35 GB por placa, 100% de uso | [medido] `nvidia-smi` |

**Modelo de custo** [conta, `p3/plan2.py`]:

GPU-h da cadeia = (d+1) · [60 · (n/cap + 1,5/slots) + 60/slots] / 3600, com cap = 640 · (200/r)² / w^0,75 img/s por GPU

- n = treino + validação. Até 3 runs por GPU, limitado pelos 35 GB. Perda: ×1,2 em média (duas vistas custam um forward a mais, **não medido**).
- **Calibração:** helminth_eggs B0 dá 0,32 no modelo contra 0,24 a 0,34 medido. A 640 px o modelo subestima: 5,5 contra 7,2 em aid, 0,69 contra 1,1 em remote_sensing, ambos sem cache. Por isso a margem ×1,5.
- **Teto de runs:** 12 no total, para não apertar o crispim. Com cache, 13 a 17 runs já rodam com load abaixo de 105. O gargalo hoje é GPU (100%), não CPU.
- SVM um por vez, sob flock, `nice -n 10`, depois do treino, em paralelo com a fase seguinte.

GPU-h por cadeia a 50%:

| Dataset | B0 | B1 | B2 | B3 |
|---|---|---|---|---|
| Eggs | 0,85 | 1,71 | 2,64 | 4,86 |
| helminth_eggs | 0,38 | 0,60 | 0,77 | 1,18 |
| Eggs_subgroup | 0,30 | 0,48 | 0,61 | 0,93 |
| Larvae | 0,16 | 0,28 | 0,39 | 0,72 |
| helminth_larvae | 0,28 | 0,43 | 0,55 | 0,83 |
| Larvae_subgroup | 0,10 | 0,15 | 0,17 | 0,24 |
| flowers | 0,23 | 0,42 | 0,63 | 1,15 |
| Protozoan | 2,39 | 3,93 | 5,16 | 8,07 |
| protozoan_cysts | 0,65 | 1,06 | 1,37 | 2,13 |
| Protozoan_subgroup | 0,55 | 0,89 | 1,15 | 1,77 |
| aid | 0,68 | 1,36 | 2,09 | 3,85 |
| remote_sensing | 0,12 | 0,20 | 0,27 | 0,49 |

### 4.6 Três orçamentos

| Fase | Pequeno | Médio | Completo |
|---|---|---|---|
| F0: sementes | 5 por representante | 9 | 9 |
| F1: degrau 2 / finalistas | top 4 / 2 | top 5 / 3 | 15 cheios / 4 |
| F2: LHS r livre / r preso / top k | 8 / 6 / 2 | 12 / 8 / 2 | 16 / 12 / 3 |
| F3: fronteira × frações | B1, 3 splits, 5 frações | + B3 em 3 datasets, + B0 com 1 split | B0 a B3, 7 datasets, 3 splits |
| Bloco de `a` | Não | w ¼, ½, 4 em 2 datasets | w ¼, ½, 4, 6 em 3 datasets |
| Controle aleatório | B0, 3 representantes | B0 e B3 | + externa |
| 200 épocas | Não | B1, 3 representantes | B1 e B3, 5 decisores, 3 splits |

Cadeias e GPU-h [conta, `plan2.py`]:

| Bloco | Pequeno | Médio | Completo |
|---|---|---|---|
| F0 ruído | 15 / 4 | 27 / 7 | 27 / 7 |
| F1 perda | 81 / 18 | 95 / 24 | 145 / 45 |
| F2 forma | 150 / 145 | 170 / 160 | 249 / 234 |
| Cabeça H1, H2 | 6 / 2 | 12 / 8 | 36 / 24 |
| F3 frações | 75 / 40 | 174 / 111 | 504 / 326 |
| Bloco de `a` | — | 54 / 22 | 108 / 61 |
| Passos e ponte | — | 24 / 8 | 51 / 22 |
| Controle aleatório | 9 / 3 | 18 / 12 | 30 / 28 |
| Regra comum e regra EfficientNet | — | 18 / 12 | 90 / 107 |
| 200 épocas | — | 3 / 5 | 30 / 124 |
| F4 confirmação | 12 / 13 | 27 / 62 | 144 / 347 |
| F5 externa | 12 / 12 | 33 / 38 | 120 / 138 |
| **Total** | **360 / 236** | **655 / 468** | **1.534 / 1.462** |
| **Dias, 4 GPUs** | **2,5** | **4,9** | **15,2** |
| **Dias, com margem ×1,5** | **3,7** | **7,3** | **22,8** |
| SVM, um por vez | 9 h | 16 h | 38 h |

Supõe as 4 GPUs livres: as filas `sc` e `base` têm que acabar antes. Opção B de P2 soma ~80 GPU-h. Baselines com EfficientNet treinado do zero ficam à parte (P4 §13: ~250 h de slot).

| O que cada um responde | Pequeno | Médio | Completo |
|---|---|---|---|
| Lei de aprendizado, b | Sim, em B1 | Sim, em 3 tamanhos | Sim, 4 tamanhos, 7 datasets |
| b muda com tamanho? (HA4) | Não | Sim, 3 representantes | Sim |
| Expoente a | **Não** | Faixa larga | ±0,2 se a≈0,3; cota se a≈0,15 |
| Perda e λ | Sim, com menos certeza | Sim | Sim, sem proxy |
| Fronteira e π | Grosseira | Sim | Melhor nos 5 decisores |
| SPiFiL contra aleatório | ±0,015 | ±0,010 | ±0,008 |
| 60 contra 200 épocas | **Não** | Sinal | Sim |
| Regra contra EfficientNet | **Não** | Sim, em B2 | Sim, B1 a B3 |
| Confirmação e externa | 2 + 2 datasets | 3 + 2 datasets | 3 + 2, várias frações |

**Por que o médio:** é o menor que responde às 7 primárias, e o completo gasta 3× mais para melhorar sobretudo `a`, que dificilmente sai de qualquer jeito. Se o tempo apertar, corte nesta ordem: 75%, depois B3 em F3, depois top k de 2 para 1 em Eggs.

---

## 5. Critérios escritos antes

Vale o que está aqui. Mudar depois de ver dado exige dizer no relatório.

**Parada por fase:**

| Fase | Para quando | E faz |
|---|---|---|
| F0 | Desvio entre sementes ≤ 0,01 | 1 semente por split no resto |
| | Entre 0,01 e 0,02 | 2 sementes só no top k e em B1 de F3 (+~35% no total [conta]) |
| | > 0,02 | **Pausa.** O sorteio de imagens do SPiFiL manda (L_data, hipótese g). Vira fator antes de seguir |
| F1 | Braço colapsa | Sai na hora: NaN, `val_recon_loss` sobe > 5%, ou dimensão efetiva de z abaixo da de reconstrução (R8 §2.5) |
| | Melhor braço − reconstrução < 0,005, intervalo com 0 | Fica a mais simples. Não insistir |
| | ρ(15 épocas, 60 épocas) < 0,5 | Repete o degrau 1 com 30 épocas |
| F2 | Faixa entre as formas ≤ 2 desvios | Superfície plana: pula B2 e B3 de F3 e declara "menor modelo suficiente" |
| | Top 1 igual antes e depois de remedir | Fronteira aceita |
| | Top 1 mudou em > 2 orçamentos | 1 rodada extra de top 3, só nos decisores. Máximo 1 rodada |
| F3 | Largura de b compartilhado ≤ 0,2 depois de 100, 5, 25, 1 | Pula 10 e 75, exceto em 2 datasets, para a previsão fora da amostra |
| Geral | GPU-h gasto > 1,3× do orçamento | Para, reporta o que tem |

**Sucesso e falha da lei:**

| Lei | Sucesso | Rebaixa, sem matar | Falha |
|---|---|---|---|
| **Aprendizado** | HA1 e HA2 passam; HA3 com erro mediano ≤ 0,015 e máximo ≤ 0,03 em F4 | Vale por família: "lei por domínio" | Erro mediano > 0,03 em F4, ou b fora do intervalo em 2 de 3 famílias |
| **Arquitetural** | Em F4, regra a ≤ 0,01 do melhor medido em ≥ 80% de dataset × orçamento; **e** HF2 passa | HF1 falha: "menor modelo suficiente". HF3 falha: regra compartilhada | Regra perde da melhor config comum **e** não gasta 2× menos |
| **Perda** | HP1 passa | Vence em 1 família: perda por família | Nenhuma vence: reconstrução sozinha |
| **SPiFiL** | HS1 passa | Só em modelos pequenos | Intervalo com 0 em todas as famílias: cai a parte SPiFiL da lei |
| **Externa** | Previsão escrita antes acerta em ≥ 3 de 4 células, intervalo de 95% | 2 de 4 | ≤ 1 de 4 |

**Como medir "a 0,01":** 3 splits não enxergam 0,01 em nenhum dataset sozinho (P2 §7). Então o critério é **agrupado, pareado por split**, e por dataset só se reporta a diferença. Nos 5 decisores são 15 pares: com desvio 0,02, a faixa é ±0,011 [conta]. Numa família de 3 datasets são 9 pares (±0,015). Em flowers sozinho são 3, e não decide nada.

**Meta de κ, separada:** reportar a diferença real para o melhor do benchmark, por dataset e por fração, com leitura igual dos dois lados (P4 §10). Não é critério da lei.

---

## 6. O que fazer com a grade `sc`

**Resposta curta: usar como piloto e referência, não como dado da lei final, e cortar 80% da fila.**

Estado às 01:38 [medido]: 2 cadeias OK, 1 erro, o resto em andamento. Só helminth_eggs passou da L1. Sem SVM.

**Por que ela não entra no ajuste final.** É outro regime:

| Problema | Efeito |
|---|---|
| Sem perda de SSL | É o nível λ=0 |
| `best_by_probe: true` | Usa rótulo; sonda varia 0,02 a 0,04 entre réplicas |
| `fill_random_last` com 1 imagem por classe | Cabeça de 8% a 35% SPiFiL, exceto `aid` e `remote_sensing` |
| Só 200 px | Sem eixo de resolução; `aid` e `remote_sensing` encolhidos 3× |
| 1 split, 1 semente | Sem ruído, sem par |
| 3 dos 6 datasets fora da descoberta | Ler `Protozoan_subgroup`, `aid`, `remote_sensing` contamina confirmação e externa (P2) |

| Parte | Decisão | Motivo |
|---|---|---|
| **`c2560`: `sc_w1_c2560` e os `scbig_*`** | **Cortar agora** | Morrem na L4: `sc_w1_c2560` já morreu assim. `scbig_w2` e `scbig_w4` rodam com a mesma chave, e só as larguras mudam no YAML. Estão em L2 e L1, gastando GPU à toa |
| Braços `_rand` | Cortar 16 de 18; ficam `w1_c1280`, `w4_c1280` | Amostra, pedido do usuário |
| `Protozoan_subgroup`, `aid`, `remote_sensing` | Cortar, ou **não abrir** | P2. Se já rodaram, ficam lacrados |
| `spf` em helminth_eggs, flowers, Larvae_subgroup | **Manter** | Referência λ=0, efeito de largura e profundidade |
| 6 réplicas w1, L3 | **Usar já**: 6 SVMs, CPU | Mede ruído de GPU no SVM, de graça |
| Estágios L2, L3, L4 | SVM em cada um | Mede ρ do proxy no regime novo |
| `elapsed_s`, memória | Usar | Vazão de w4 e d6 melhora esta conta |

- **Economia:** fila inteira ~570 h de slot (2,0 dias em 12 slots), cortada ~109 h (0,4 dia) [conta].
- **Na lei:** só como bloco "reconstrução", com indicador de regime, e nunca misturada sem esse indicador.
- **Para que ela serve em F2:** se w2 ou w4 não mexerem κ em helminth_eggs, o LHS de r preso encolhe.
- **Observação:** P2 pede trocar Larvae_subgroup por helminth_larvae nas próximas grades. O desenho já faz isso.

---

## 7. Divergências, perguntas, limites

**Onde discordo de P1, P2 e P4:**

| Tema | Eles | Eu | Por quê |
|---|---|---|---|
| Quais frações entram no ajuste | P2: só células L. P1 R5: só ≥ 25 imagens por classe | **L e R entram, com peso pelo ruído.** Reportar b com e sem R. X fica fora | Só L: largura 1,72 contra 0,23, e 9 de 28 curvas reais na borda. Se os dois b discordarem, a lei vale só acima de ~5 imagens por classe, e aí vale o intervalo largo |
| Fração de 10% | Não listada | Entra | Já existe. É L em Eggs e helminth_eggs (mínimo 6 por classe). Ajuda, mas sozinha não resolve: 0,89 |
| `best_by_probe` | P4 §8: exemplo com `true`, declarando o rótulo | `false` | Regra nova, sonda ruidosa, 42% do tempo |
| Quais perdas | P4: VICRegL e UR-JEPA só se ajudarem; CorInfoMax fora | As 7 no degrau 1 | Pedido do coordenador. Custa 15 de 60 épocas |
| Alcance em N | B0 a B3, 0,2M a 1,9M | + w ¼, ½, 4 | `a` não sai com 1 década |

**Perguntas para o usuário:**

1. **`one_per_class` e o alocador usam rótulo de C imagens.** Isso conta como "rótulo no treino"? Se contar, precisa de código no worktree (o `spifil` não pode mudar) e de um braço em F0. Sugiro aceitar e declarar, como P4.
2. **"Nunca aleatório" vale para a cabeça?** Se valer à risca, H0 sai, e a escolha fica entre H1 (c = candidatos, menor) e H2. Isso muda c=1280 e a leitura.
3. **Checkpoint por `val_recon_loss` serve?** Ela é quase plana, então o efeito é quase "última época". F0 compara os dois em 1 braço.
4. **Orçamento:** médio, com corte de `sc`?

**Limites:**

- **As simulações têm premissas minhas.** Tamanho dos efeitos, formato das superfícies e ruído só são calibrados onde está dito. Valem para comparar desenhos, não como previsão de κ.
- **`b` vem de encoder congelado.** Nada garante o mesmo número no SSL. A SSL muda a curva: cada ponto novo treina o encoder de novo.
- **ρ do proxy** vem de leitura flatten e split antigo.
- **Custo de perda com 2 vistas não foi medido.** w4 só em L1. 448 px não foi medido. r=640 subestimado até 1,6×.
- **`σ_run` é desconhecido** no regime novo. As simulações cobrem 0,005 a 0,02. Os dados antigos sugerem ~0,003 em Eggs (L_data §2.1) e 0,02 no HAL §4.15.
- **3 famílias na descoberta** derrubam uma regra, não cravam um expoente (P2). τ de família mal se estima.
- **Larvae a 1%** e outras células X ficam fora; a contagem de cadeias não desconta isso (é teto).
- **Sem GPU, sem SVM rodado por mim.** Nenhum número de teste entrou: li só colunas `val_*` e descartei `test_*` na carga.

**Busca:** grafo primeiro (`graphify query`, achou módulos e `measure_flops`). Ele não indexa YAML nem linhas internas, então usei grep só para `one_per_class`, `best_by_probe`, `fill_random` e `reg_loss`.

**Arquivos em `<scratchpad>/p3/`:**
- Vazão: `timing.py`, `timing2.py`, `timing3.py`, `timing4.py`, `svmtime.py`.
- Ruído e proxy: `replicas.py`, `proxy.py`, `guard.py` (erro de c2560).
- Dados: `nest.py` (frações aninhadas), `bench_pilot.py`, `bench_fit.py`.
- Simulações: `sim_learn.py`, `sim_cells.py`, `sim_cells2.py`, `sim_joint.py`, `sim_joint2.py`, `sim_joint3.py`, `sim_arch.py`, `sim_arch2.py`, `sim_refine.py`, `sim_ladder.py`, `sim_sha.py`.
- Custo: `arch_cost.py`, `cost.py`, `plan.py`, `plan2.py`. Saídas em `*.out`.
