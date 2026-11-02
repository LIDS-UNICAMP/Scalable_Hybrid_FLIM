# Discovery híbrido SPiFiL — log de decisões

Branch `discovery_hybrid_spifil`. CNN, entrada LAB, treino só com decoder (sem rótulo), um eixo por candidato, crescer camada a camada.

## Alvo (I-JEPA a 50%, κ teste, média de 3 splits — report_results_only_spifil.md:209)

| dataset | I-JEPA | melhor SPiFiL | fonte | gap |
|---|---|---|---|---|
| eggs | 0.9397 | 0.9094 (flim_L3) | :206 | 0.030 |
| larvae | 0.9433 | 0.9164 (flim_avg_L3) | :347 | 0.027 |
| protozoan | 0.8636 | 0.8587 (bp3, grid4 backprop) | :195 | 0.005 |

Aceite: κ teste ≥ I-JEPA nos 3 datasets a 50% (média dos splits rodados). Teste só no comparativo final.

## Painel / plano

- Decisão (crescer, aceitar eixo): κ de **validação** do SVM oficial (`core/svm.py:143` + `core/metrics.py:46`). Hoje `eval/svm.py:694` fixa `_loader("test")` → ver pergunta Q1.
- Crescimento: adiciona Lk+1 enquanto κ val melhora (> +0.002); para na 1ª queda.
- Rodada de busca: 50%, split 1, 3 datasets. Confirmação (candidato que bate o melhor): splits 1-3.
- Parada imediata: NaN/Inf na loss ou OOM. κ val abaixo do melhor em 2 checagens seguidas → para o candidato.
- GPU 3 apenas, `max_per_gpu` 2.

## Eixos (ordem; convolução primeiro)

1. Convolução — dilation 2 em L4–L5 sem pooling (R6 h2: L4–L5 pioram no flatten; dilation aumenta campo receptivo sem perder resolução). Precisa de código: dilation fixo `[1,1]` em flim/spifil.py:624-651.
2. Kernel / largura — 7x7 em L1, ou 32/48/64 filtros (R6 h1 já fixou 5x5 24/32/48 como melhor conhecido).
3. Objetivo sem rótulo — MAE por superpixel no lugar da BCE pura (R6 h3: BCE satura em ~0,468).
4. Fine-tune ponta a ponta só com reconstrução após layerwise (R6 h4: backprop grid4 é o melhor em protozoan).
5. Embedding multi-escala concat (R6 h5).
6. BN (runs layerwise5_bn em andamento) e 7. geo_aug (layerwise5_aug em andamento) — reusar o resultado, não rodar de novo.
8. Profundidade adaptativa por pct — fora do escopo (só 50%).

## Entradas

### 2026-10-29 — plano inicial
- R1 (baseline reproduzível): runner growth layerwise, 3 camadas, 5x5, 24/32/48, DISF/50, paper_classes, sem trava N≤D, flatten. É o flim_L3 (melhor eggs) dentro do runner novo; mede-se κ val e teste para ancorar.
- R2 (1º eixo): R1 + L4 com dilation 2 e sem pooling, cresce enquanto κ val sobe.
- Porquê: o flim_L3 é o melhor SPiFiL reproduzível neste runner; convolução é o eixo 1 pedido; dilation é a hipótese R6 mais barata que mexe só em L4+.

### 2026-10-29 — respostas do usuário ao plano (substituem as leituras padrão acima)
- **Critério de decisão:** métricas de validação **sem classificação**, só do autoencoder (perda de reconstrução em val, `val_recon_loss`). O SVM no teste roda só de vez em quando, como checagem. Por isso não se mexe em `eval/svm*` nem se liga `_svm_probe`. Q1 fechada.
  - Crescer Lk+1 enquanto a val_recon_loss da nova camada fica abaixo da anterior. Comparar losses só dentro da mesma função de perda: quando o eixo 3 trocar a loss, a âncora passa a ser o SVM teste de checagem.
- **Épocas:** `max_epochs` 200 por camada (antes 1000). A BCE satura em ~0,468.
- **Splits:** os 3 splits em toda rodada (não só o split 1).
- **BN e geo_aug:** aproveitar os runs layerwise5_bn e layerwise5_aug do usuário (eixos 6 e 7), sem rodar de novo.
- **GPU:** 3 (a máquina não tem GPU 4).

### 2026-10-29 — usuário: SVM na validação para decidir e extrair as leis
- **Decisão:** κ do SVM oficial medido na **validação**, com o mesmo `fit_svm` e `compute_metrics` de sempre. A chave `eval_set: val` (default `test`) entra em `eval/svm.py` com aval do usuário. Para cada camada e cada eixo vale a regra: κ val sobe = fica. A val_recon_loss do autoencoder vira sinal auxiliar (escolha do best dentro da camada).
- **Leis de escala:** saem da tabela κ val por camada × eixo (largura, kernel, dilation, profundidade, loss) × dataset. O teste fica só para conferir de vez em quando e para o comparativo final com o I-JEPA.

### 2026-10-30 — usuário: modelo leve
- **Limite:** o modelo é lightweight, com **no máximo 1M de parâmetros** (encoder mais qualquer projetora ou preditora usada para o embedding).
- **Projetora/preditora:** nunca pode ter mais parâmetros que o encoder.
- **Decoder:** fica fora da conta. Ele só serve para treinar e não vai para o SVM. Se o usuário quiser que o decoder também conte, rever.
- **Referência:** o encoder do R1 (5x5, 24/32/48) tem ~59k parâmetros. O A2 deve checar largura × kernel contra o teto.

### 2026-10-30 — usuário: mais camadas e teto de 5M
- **Profundidade:** sem limite de 3 camadas. O runner já aceita N camadas quando o YAML traz `layer_channels` e `layer_pools` com N itens. O limite de 5 era só o default de `LAYERWISE_SPECS`.
- **Teto de parâmetros:** sobe de 1M para **5M** (encoder mais projetora). A projetora continua sem poder ser maior que o encoder, e o decoder segue fora da conta.
- **R2 (A0) reescrito antes de rodar:** passou de "L4 com dilation 2" para um crescimento profundo de 8 camadas.
  - Canais: 24/32/48/48/64/64/64/64.
  - Kernels: 5,5,5,5 nas L1–L4 e 3 nas L5–L8.
  - Pool: 2 nas L1–L4 e nenhum nas L5–L8.
  - Dilation: 1 nas L1–L4 e 2,2,4,4 nas L5–L8.
  - Encoder com ~256k parâmetros.
  - Hipótese: no histórico as L4–L5 pioravam porque, sem pool, o campo receptivo não crescia. A dilation faz ele crescer sem encolher o mapa. O SVM em cada Lk mostra até onde vale crescer.

### 2026-10-30 — usuário: REGRA QUE PREVALECE — o SPiFiL tem que contribuir no treino
- Toda camada nasce dos kernels gerados pelo SPiFiL, nunca de inicialização aleatória. O treino com o decoder ajusta a partir deles.
- **Prova obrigatória:** o melhor candidato de cada frente ganha um controle idêntico com `random_layer_classic: true` (mesma arquitetura, filtros aleatórios). O SPiFiL só "contribui" se ganhar do controle no κ val. Esse controle entra na fila da GPU 3 após o R1.
- Um ganho de κ que venha de os filtros se afastarem dos kernels SPiFiL (fine-tune longo, lr alta) não vale. Por isso cada frente mede a similaridade entre os pesos treinados e os do grow.

### 2026-10-30 — usuário: encoder pequeno, profundo, com SPiFiL + backprop juntos
- **Norte:** encoder pequeno e profundo, sempre nascendo do SPiFiL e sempre testando backprop junto com o SPiFiL.
- **Largura:** a escada de largura do A2 (w96–w256, até 1,7M) foi cancelada. O w64 termina e entra só como ponto da lei de largura.
- **A2 muda de frente:** passa a cuidar de "SPiFiL + backprop". Cria a chave `layerwise_backprop: <escala de lr>`, com que as camadas antigas treinam com lr reduzido em vez de congelar. Hoje isso existe só no método ijepa_grow (growth.py:255-258).
- **Variante:** um fine-tune ponta a ponta só com reconstrução depois da última camada.
- **Por quê:** no histórico, o SPiFiL + backprop (grid4) foi o melhor SPiFiL a partir de 25% em eggs e protozoan (report:108-109). O SPiFiL dá o ponto de partida e o backprop ajusta o conjunto. A similaridade de cosseno aos kernels do grow vigia se o SPiFiL continua presente.

### 2026-10-30 — usuário: o SPiFiL pode ser só o inicializador (substitui a regra "não se afastar")
- **A regra principal vira "o SPiFiL contribui no treino":** ele pode ser apenas um bom inicializador. Os filtros podem se afastar dos kernels do grow; a deriva deixa de invalidar um ganho e passa a ser só informação.
- **A prova de contribuição** é a estabilidade frente a um controle `random_layer_classic: true` com a mesma arquitetura, em ≥3 seeds:
  - (1) a variância do κ val entre seeds e splits. O SPiFiL é determinístico (sem RNG), então não tem variância de inicialização;
  - (2) as épocas até a loss estabilizar;
  - (3) a oscilação da curva de val_recon_loss;
  - (4) o κ médio, que pode empatar.
- **`new_layer_lr_scale`:** deixa de ser obrigatório. Vira só um eixo opcional.

## 2026-10-30: achado de ativação no fundo
- Observação: no SPiFiL cru, a L2 ativa só no objeto. Depois do backprop, filtros passam a ativar no fundo (eggs s1, cos 0,78).
- Leitura do usuário: os gradientes estão aprendendo errado; o modelo deveria olhar só para o elemento.
- Hipótese: a recon por pixel é dominada pelo fundo e puxa os filtros para ele.
- Decisão: avisar A1, A2, A3 e A5. Próximo candidato (uma mudança): loss ponderada pelo foreground, sem rótulo.

## 2026-10-30: mask50 interrompido, fg01 lançado (A1-2)
- Medida (R1 eggs, foreground sem rótulo, IoU 1.0 com masks/): densidade de ativação dentro/fora do objeto, cru → treinado: L2 1.27× → 1.03×, L3 1.17× → 0.84×.
- Hipótese (SPiFiL): os filtros do SPiFiL vêm do objeto; a recon BCE uniforme, com ~97% de pixels de fundo, apaga esse foco.
- Mudança única: `recon_bg_weight: 0.1` (o peso do fundo na BCE cai de 1 para 0.1). YAML `growth_disc_a1_fg01.yaml`.
- O mask50 estava no ponto 1/9 e testa máscara aleatória, que não é direcionada. Foi interrompido para liberar a GPU 3, e os parciais foram mantidos.
- Critério: κ do SVM na val ≥ R1 e razão dentro/fora maior que no R1.

## 2026-10-30: `recon_bg_weight: 0.1` vira a base dos runs ainda não lançados
- Pedido do usuário: começar os treinos já levando o achado do fundo em conta.
- Aplicado em R2 (8 camadas), nos controles rand / rand_s1 / rand_s2 e no bp01 (A2).
- Pareamento: a base de comparação passa a ser o fg01. O R2 difere do fg01 só na profundidade, o controle rand difere do fg01 só na init, e o bp01 difere do fg01 só no backprop das camadas antigas. Continua uma mudança por candidato.
- O R1 (BCE uniforme) fica como referência antiga. Se o fg01 perder κ na val para o R1, a base volta para 1.0 e esses runs são refeitos.

## 2026-10-30 14:05: SVM desativado (pedido do usuário)
- Load de CPU em 245 com 96 núcleos e GPUs a 0–10%: o gargalo é a CPU. O SVM flatten do A5 levou até 4,6 h por célula.
- Parado o raw_vs_trained do A5; as 8 linhas já feitas ficam em results.jsonl.
- Nenhum eval_svm até nova ordem. Só treino na GPU. A medida de ativação dentro/fora (só forward) continua liberada.
- Pendentes para quando religar: val do resto do R1, w64, fg01, popz.

## 2026-10-30 18:24: tudo sai da GPU 3 e vai para as GPUs 0, 1 e 2 (pedido do usuário)
- A GPU 3 chegou a 47,8/49 GB, e um SVM do R1 caiu por OOM.
- Paradas propositais: as sessões disc_*, svmq_* e watch_*, mais as esperas do A2 e do A3.
- Relançado com skip: state, que pula os pontos prontos. Todos os YAMLs de treino e de eval agora têm gpu_ids [0, 1, 2] e max_per_gpu 1.
- Cadeias em tmux: disc_a1_fg01; disc_a2 (w64 → bp01); disc_a3 (popz → sp200); disc_r2; disc_rand (rand → s1 → s2).
- Cada treino, ao terminar, abre o svmq_<nome>. O SVM roda sob uma trava única (flock), um por vez. Monitor: tmux watch_disc.

## 2026-10-31: SPiFiL como inicializador, veredito pela curva (A4)
- 21 estágios (eggs/larvae L1–L3, protozoan L1). Pareado: SPiFiL vs aleatório com a mesma forma e a mesma norma por filtro, os dois com fundo 0.1.
- O aleatório chega a uma loss mais baixa em 16 de 21; o SPiFiL ganha em 4 (≤4e-5); 1 empate. O aleatório estabiliza antes em 20 de 21. A oscilação é igual.
- Conclusão: na otimização, o SPiFiL não é um inicializador melhor. A hipótese "o SPiFiL deixa o treino estável" caiu pela curva. Resta testar pelo κ: média e variância entre seeds (rand, rand_s1, rand_s2) e splits.

## 2026-10-31: R2 (8 camadas) interrompido
- κ val eggs (s1/s2): L3 0.898/0.898 → L4 0.871/0.891 → L5 0.863/0.862 → L6 0.877/0.835 → L7 0.839/0.844 → L8 0.808/0.819. A profundidade custa κ a partir da L4.
- Curva do larvae (A4): na L4 a distância val − treino salta ~3x a do eggs, e o early stop dispara cedo. Prevê a mesma queda.
- As L1–L3 do R2 eram cópia do fg01 (mesmo config). Continuar seria só gastar GPU. Interrompido às ~05:50; os parciais ficam.
- Ruído de re-run do κ ≈ 0.01 por split (fg01 × R2 com config idêntico). A partir de agora, decisão só pela média dos 3 splits.
- Ressalva: depois da L3 o mapa fica pequeno (12x12 → 11x11 com dilatação), então parte da queda pode ser o flatten menor e não só sobreajuste. A profundidade com mapa grande (pool só no fim) segue em aberto.

## 2026-10-31: 1º κ do controle aleatório. O rand vence o SPiFiL na L3 de eggs
- Pareados (mesmo config, fundo 0.1; o rand tem a mesma norma por filtro do SPiFiL, seed 42). κ val L3 eggs s1/s2/s3: rand 0.898/0.920/0.908 (média 0.909); fg01 (SPiFiL) 0.894/0.898/0.887 (média 0.893). Δ = +0.016 para o rand.
- s2/s3: Δ +0.021/+0.022, acima do ruído de re-run desses splits (≤0.0003). s1: dentro do ruído (até 0.016).
- O SPiFiL varia menos entre splits (dp 0.006 contra 0.011), mas em torno de uma média mais baixa; com n=3 isso não é significativo.
- Leitura: até aqui o SPiFiL não contribui como inicializador, nem na curva nem no κ. Faltam as seeds s1/s2 do rand e os outros datasets.
- Erro operacional: o relançamento das 18:24 NÃO pulou os pontos prontos (0 pulados). O fg01/w64/popz retreinaram o eggs; o ckpt de topo foi substituído, mas o run antigo (flim-ssl/oza2fp2k) continua no disco. O κ antigo do fg01 eggs s1 L3 (0.910) é desse ckpt antigo.

## 2026-10-31: κ val L3 completo do fg01 e parcial do rand. O SPiFiL é o inicializador estável
- fg01 (SPiFiL, fundo 0.1), média dos 3 splits: eggs 0.892, larvae 0.835, protozoan 0.855. R1: 0.897 / 0.829 / 0.848. Empate dentro do ruído.
- rand seed 42: eggs 0.897/0.919/0.907 (0.908); larvae 0.847/0.786/0.790 (0.808, dp 0.034); protozoan s1 0.815 (SPiFiL 0.877).
- Perfil: o rand ganha ~0.02 em eggs, mas tem quedas grandes e esporádicas (larvae s2/s3 −0.044/−0.048, protozoan s1 −0.062). O SPiFiL fica estável (dp ≤0.006 em eggs e larvae). A loss não vê nada disso.
- Hipótese (SPiFiL como inicializador): ele não melhora a otimização, mas evita mínimos ruins para a classificação. Faltam as seeds 1 e 2 do rand para descartar o azar da seed 42; os SVMs delas estão na frente da fila.
- popz (L3): 0.886 / 0.820 / 0.844, abaixo do fg01. O z-score populacional não ajuda.

## 2026-10-31: larvae L3, só SPiFiL + fundo 0.1 fica estável entre splits (A4)
- Queda de κ val do s1 para a média do s2/s3: R1 −0.026; w64 −0.054; rand seed 42 + fundo 0.1 −0.059; fg01 (SPiFiL + fundo 0.1) −0.005.
- Leitura: nenhum dos dois fatores basta sozinho; a combinação é a única estável.
- Ressalva: a medida dentro/fora em eggs s1 NÃO mostrou o fg01 mantendo o foco no objeto (L2 0.93, L3 0.94). Então o mecanismo "o fundo 0.1 segura os filtros no objeto" não está provado; o efeito pode vir de outro lugar. Medir dentro/fora no larvae s2/s3 (fg01 × R1 × rand) separa as duas leituras.
- Falta o controle de seed do rand no s2/s3 (seeds 1 e 2 treinando).

## 2026-10-31: o mecanismo "foco no objeto" caiu (A4, larvae L3)
- Razão de ativação dentro/fora treinada, s1/s2/s3: fg01 1.46/1.63/1.16; R1 1.30/1.37/1.32; rand 1.37/1.21/1.36. Spearman(razão, κ) nos 9 pontos = 0.20 (p=0.60).
- A estabilidade do fg01 no larvae é real no κ, mas não vem de manter o foco médio no objeto. A razão média de ativação não serve de proxy do κ; deixamos de usá-la para decidir.
- O que se mantém: o SPiFiL cru é 2.2–3.1x mais focado no objeto que o aleatório cru (1.3–2.0x), e o treino puxa todos para ~1.2–1.6.

## 2026-10-31: novo candidato w64fg = fg01 + largura 32/48/64
- Motivo: o w64 (BCE uniforme) foi o 1º ganho com o mesmo sinal nos 3 splits em protozoan L3: 0.868/0.860/0.851 (média 0.860, contra 0.848 do R1 e 0.864 do I-JEPA). Em eggs e larvae empata.
- Hipótese: a largura dá capacidade em protozoan (7 classes, onde a profundidade também somava), e o fundo 0.1 traz a estabilidade vista em larvae. Uma mudança contra o fg01: a largura. 117.744 parâmetros.
- 3 datasets × 3 splits, GPUs 0-2, SVM só na L3 depois do treino.

## 2026-10-31: correção. A estabilidade do fg01 em larvae era sorte
- O sp200 (SPiFiL + fundo 0.1, só muda os superpixels) em larvae L3: s1 0.890 (o maior de larvae), s2 0.776. Oscila 0.114, mais que o rand.
- Larvae é binário, com 223 casos na classe minoritária. O κ de um único treino varia ±0.05 entre candidatos parecidos.
- Regra nova: em larvae, Δκ < ~0.05 num split só é inconclusivo. Para decidir, usar a média de ≥2 seeds por split. A entrada "só SPiFiL + fundo 0.1 é estável" fica retirada.
- De pé: protozoan s1, o SPiFiL vence 2 seeds do rand por 0.06–0.08. O w64 em protozoan dá +0.011 nos 3 splits. O w64fg segue, motivado só pelo protozoan.

## 2026-10-31: split 1 fechado com 3 seeds do rand (L3)
- eggs: rand 0.897 (3 seeds) × SPiFiL 0.894. Empate.
- larvae: rand 0.843 × SPiFiL 0.839. Empate.
- protozoan: rand 0.815/0.794/0.832 (0.814 ± 0.019) × fg01 0.877, ou +0.063 (~3.4 dp). Até o pior SPiFiL (R1, 0.855) passa a melhor seed.
- Conclusão parcial: o SPiFiL contribui como inicializador em protozoan (7 classes, o mais difícil) e é neutro em eggs e larvae. Os splits 2 e 3 das seeds estão treinando.

## 2026-10-31: bp01 vence o fg01 em eggs L3 nos 3 splits
- bp01 (fg01 + camadas antigas treinando com lr 0.1×): 0.908/0.906/0.894 (média 0.902). fg01: 0.894/0.898/0.887 (0.893). Δ +0.010 com o mesmo sinal nos 3 splits. É o melhor eggs até agora.
- A loss da L3 do bp01 já era mais baixa em 6 de 6 splits. Regra: no mesmo encoder com mais liberdade de backprop, loss menor acompanha κ maior. Entre inicializadores diferentes isso não vale.
- Apoia a orientação do usuário de explorar o backprop junto com o SPiFiL. Próximo: bp01 em larvae e protozoan; se confirmar, combinar bp01 + w64fg (uma mudança de cada vez).

## 2026-10-31: bp01 é o melhor nos 3 datasets; lançado bp01w64
- bp01 L3 (média dos 3 splits): eggs 0.902 (+0.010 contra o fg01, 3/3), larvae 0.829 (empate), protozoan 0.866 (+0.010, 3/3; o maior protozoan até agora).
- Novo candidato bp01w64 = bp01 + largura 32/48/64 (uma mudança). Motivo: o w64 deu +0.011 em protozoan com BCE uniforme.
- O SVM das seeds 1 e 2 roda já nos ckpts de L3 prontos (só a L3; uma L3 existe só quando o ponto terminou), sem esperar o growth inteiro.

## 2026-10-31: SPiFiL × rand com seeds, eggs fechado e larvae s2
- eggs (3 seeds × 3 splits): o rand tem média ~0.008 maior (≤1 dp por split). Entre seeds ele varia 0.010–0.016; o SPiFiL repetido varia ≤0.0003 no s2/s3 (no s1, até 0.016). Empate na média, e o SPiFiL é mais reprodutível.
- larvae s2: o rand_s1 também cai (0.794 ≈ 0.787 da seed 0). fg01 0.831, ou +0.041 (~8 dp). Ressalva: o sp200 também cai ali (0.776), então o efeito é do fg01, não de toda variante SPiFiL.
- protozoan s1: +0.063 (3.4 dp).
- Regra de treino: onde o aleatório vai mal, o SPiFiL ajuda; onde vai bem, é neutro e mais reprodutível. Faltam larvae s3 e protozoan s2/s3 com seeds.

## 2026-10-31: larvae s2 com 3 seeds do rand
- rand L3: 0.787 / 0.794 / 0.800 (média 0.794). fg01 0.831, ou +0.037 (~6 dp entre seeds).
- Placar nas 6 combinações dataset×split com 3 seeds: o SPiFiL vence as 2 em que o rand fica abaixo de 0.82 (protozoan s1, larvae s2) e empata as 4 em que o rand fica acima de 0.84 (eggs s1–s3, larvae s1). A regra de treino se mantém.

## 2026-10-31: larvae s3 com 2 seeds do rand
- rand L3: 0.791 (seed 0) / 0.840 (seed 1). fg01 0.838. Diferença na média +0.023 (~0.7 dp): inconclusivo, a queda da seed 0 foi azar de seed. No larvae s2, ao contrário, as 3 seeds caem juntas.
- Placar com ≥2 seeds (7 células): o SPiFiL vence 2 (protozoan s1 +0.063, larvae s2 +0.037), empata 5 e não perde nenhuma.
- Evidência mais estável: o κ do rand varia 0.007–0.035 entre seeds (mediana ~0.016); o SPiFiL repetido varia ≤0.0003–0.008. O SPiFiL serve como inicializador que estabiliza o treino. Falta protozoan s2/s3 com seeds.

## 2026-10-31: protozoan s2/s3 com 2 seeds do rand; SPiFiL contribui
- protozoan L3 (conferido nos CSVs): rand s2 0.802/0.804, s3 0.797/0.818; fg01 0.836 / 0.853. Vantagem do SPiFiL nos 3 splits: +0.063 / +0.033 / +0.045. Média: 0.855 contra 0.808.
- Placar L3: protozoan vence 3/3, larvae vence s2 e empata s1/s3, eggs empata 3/3. O rand não vence nenhuma célula fora do ruído.
- Estabilidade: o rand varia 0.002–0.035 entre seeds (mediana ~0.015); o SPiFiL repetido varia ≤0.008.
- Conclusão: o critério "SPiFiL tem que contribuir" está cumprido. O SPiFiL ganha onde o aleatório vai mal e deixa o treino reprodutível. Falta só o rand_s2 em larvae s3 e protozoan s2/s3.

## 2026-10-31: candidato bp01c1280 = bp01 + L4 1x1x1280 (pedido do usuário)
- Mudança: uma camada 4 de 1280 filtros 1x1 sobre a L3 (24x24, sem pooling). O resto é igual ao bp01 (backprop 0.1, fundo 0.1).
- Hipótese SPiFiL: um filtro 1x1 do SPiFiL é o descritor de 48-d de um superpixel da L3, ou seja, um protótipo local. Com 1280 protótipos, cada imagem vira um "saco de palavras visuais" de 1280 dimensões, e o SVM linear separa melhor num espaço maior (larvae: −11 pontos para o I-JEPA).
- Limite medido: o SPiFiL dá 1 filtro por semente. Com 50 sp por imagem e 1 imagem por classe, o larvae dá 100 filtros. Com 600 sp, dá 964. Escolha do usuário: o SPiFiL dá o que conseguir, e o resto até 1280 é aleatório na norma média dos filtros SPiFiL, com bias 0 (novo `fill_random_last`, só na última camada; L1–L3 intactas).
- Leitura: média por canal (avgpool2d), também no κ de treino. Achatado daria 737k-d, inviável para a CPU. Efeito colateral: a época do ckpt de L1–L3 é escolhida pela média, não pelo achatado. Por isso o SVM avalia L3 e L4 na mesma leitura, e a L3 dessa família é a referência da camada 1x1.
- Params do encoder ~120k (≤5M).

## 2026-10-31: controle aleatório fechado (L3, 3 seeds × 3 splits × 3 datasets)
- protozoan: SPiFiL 0.855 contra rand 0.809 (+0.046, vence 3/3 por 3 a 9 dp). larvae: +0.013 (vence s2 por 5.7 dp, empata s1/s3). eggs: −0.008 (empate 3/3, ≤1.3 dp). Geral: +0.017. O rand não supera o SPiFiL em nenhuma célula fora do ruído.
- Estabilidade: o rand varia 0.012–0.022 entre seeds, em média (até 0.039); o SPiFiL repetido varia ≤0.008.
- Regra: o ganho do SPiFiL cresce quanto pior o aleatório vai no dataset. Critério "SPiFiL tem que contribuir": cumprido.
- bp01c1280: as L1–L3 repetem o bp01, mas não vou cortar. O ckpt de L1–L3 é escolhido pelo κ da média, e a L3 da própria família é o controle limpo da camada 1x1. Começar da L3 do bp01 exigiria uma chave nova no runner.

## 2026-10-31: w64fg não soma; L4 1x1 do c1280 baixa a loss
- w64fg L3: eggs 0.914/0.885/0.892 (0.897), larvae 0.853/0.800/0.845 (0.833), protozoan 0.879/0.861 (s3 pendente). Empata com w64, fg01 e bp01 dentro do ruído: largura e fundo 0.1 não somam.
- c1280 L4 (~120 épocas): é a primeira camada além da L3 que baixa a val loss (−8e-5 / −7e-5 contra a L3). A distância val − treino sobe só +1e-4 (no R2, +5e-4). Sem early stop, memória estável. O veredito sai no κ avgpool L4 × L3.

## 2026-11-01: bp01w64 no topo em eggs e larvae (L3, κ val)
- eggs 0.929/0.892/0.897 (0.906; o s1 é o maior κ de eggs medido), larvae 0.864/0.822/0.849 (0.845). Ganha +0.004 sobre o bp01 em eggs e +0.009 sobre o fg01 em larvae: dentro do ruído, mas no topo dos dois. Protozoan pendente.
- Em eggs, a ordem da loss de L3 (fg01 → bp01 → bp01w64) bate com a do κ. Em larvae não bate: a curva continua sem prever κ em geral.
- Base nova provisória: bp01w64 (backprop 0.1 + fundo 0.1 + largura 32/48/64). Fecha quando sair o protozoan.

## 2026-11-01: bp01w64 é a nova base; c1280 L4 sobreajusta em larvae
- bp01w64 L3 (κ val, média de 3 splits): eggs 0.906 (1º), larvae 0.845 (1º), protozoan 0.865 (empate no 1º; s1 0.888 é recorde). Único candidato no topo dos 3 datasets. Contra o rand: +0.005 / +0.022 / +0.056. Decisão: bp01w64 vira a base.
- c1280 L4 em larvae: early stop 3/3 (best 96/98/138), distância val − treino +3–4e-4, loss igual à da L3. É a assinatura de sobreajuste do R2. Previsão para o κ avgpool: larvae L4 < L3 nos 3 splits; eggs s2 <, s1 ≈, s3 ≥. O veredito sai no SVM.

## 2026-11-01: L4 1x1x1280 sobe o κ; a leitura (embed) pesa ~0.09
- c1280 κ val avgpool, L3 → L4 (conferido no CSV): eggs 0.825→0.869, 0.828→0.870, 0.804→0.864 (+0.04 a +0.06); larvae 0.915→0.934, 0.909→0.909, 0.874→0.892 (média L4 0.912). A camada 1x1 ajuda nos 6, mesmo com early stop. Retirada a regra "early stop ⇒ κ cai": ela só vale quando o vetor encolhe (flatten R2).
- Mesma L3 (pesos = bp01), só trocando a leitura: larvae avgpool ~0.90 contra flatten ~0.82 (+0.09); eggs avgpool ~0.82 contra flatten ~0.90 (−0.08). O larvae L4 avgpool (0.912) passa todo flatten medido (0.864) e fica a ~3 pontos do I-JEPA (0.943, teste). Hipótese: em larvae a posição atrapalha; em eggs a forma no mapa conta.
- Decisões: (1) a leitura vira um eixo por dataset, escolhido na validação; famílias só se comparam com a mesma leitura. (2) Lançado bp01w64c1280 = bp01w64 + L4 1x1x1280 (a mesma mudança na base nova). (3) SVM avgpool da L3 do bp01w64 (só eval, mesma L3), para separar o efeito da leitura do efeito da camada.

## 2026-11-01: tabela leitura × camada (κ val, média dos 3 splits)
| dataset | bp01 L3 flat | bp01 L3 avg | c1280 L4 avg | bp01w64 L3 flat | bp01w64 L3 avg |
|---|---|---|---|---|---|
| eggs | 0.902 | 0.819 | 0.868 | 0.906 | 0.845 |
| larvae | 0.829 | 0.899 | 0.912 | 0.845 | 0.904 |
| protozoan | 0.866 | 0.750 | 0.828 | 0.865 | 0.796 (2 splits) |
- O efeito da leitura se repete no bp01w64 (avg: eggs −0.061, larvae +0.059, protozoan −0.069); não é coisa do c1280.
- A L4 1x1 ajuda dentro do avg (+0.05 / +0.01 / +0.08), mas só passa a L3 flat em larvae.
- Escolha por dataset (validação): eggs = L3 flat (bp01w64 0.906); larvae = L4 avg (c1280 0.912); protozoan = L3 flat (bp01 0.866). Próximo: bp01w64c1280 L4 avg em larvae.

## 2026-11-01: avgpool é ~3x mais ruidoso; bp01w64c1280 empata; lançado bp01w64c96 (flatten)
- Ruído: as L1–L3 do bp01w64c1280 retreinam o bp01w64. A L3 avg das duas cópias difere em |Δκ| mediana 0.032 (máx 0.050) por split; no flatten, ~0.011. A média dos 3 splits fica estável (Δ ≤ 0.011). Regra: com avgpool, só comparar a média dos 3 splits.
- bp01w64c1280 L4 avg: eggs 0.880, larvae 0.917, protozoan 0.837. Contra o c1280: +0.012/+0.005/+0.009, dentro do ruído. A L4 1x1 sobe ~0.04 sobre a L3 avg nas duas famílias.
- Melhor por dataset (val): eggs bp01w64 L3 flat 0.906; larvae L4 1x1×1280 avg 0.912–0.917; protozoan bp01/bp01w64 L3 flat 0.866/0.865.
- Lançado bp01w64c96 = bp01w64 + L4 1x1×96 lida achatada (55k-d). Hipótese: a 1x1 recombina os 64 canais em protótipos SPiFiL sem descartar a posição, que é o que eggs e protozoan usam. Uma mudança contra o bp01w64 (L3 flat) e contra o c1280 (largura 96, leitura flat).

## 2026-11-01: bp01w64c96 não ajuda (hipótese refutada)
- L4 1x1×96 flatten, κ val médio, contra a L3 flatten do bp01w64: eggs 0.895 contra 0.906 (−0.011), larvae 0.843 contra 0.845, protozoan 0.858 contra 0.865 (−0.008). Média −0.007, dentro do ruído.
- Refutada: "o 1x1 recombina os canais sem perder a posição". Junto com o c1280: o 1x1 só ajuda quando a leitura descarta a posição (avg, +0.04). No flatten, não acrescenta. A curva mais limpa de todas não virou κ (a loss não prevê o κ).
- A escolha por dataset não muda: eggs bp01w64 L3 flat 0.906; larvae L4 1x1×1280 avg 0.912–0.917; protozoan bp01/bp01w64 L3 flat 0.866/0.865.

## 2026-11-01: varredura de porcentagens (lei de escala) com bp01w64c1280
- Pedido do usuário: passar pelas outras porcentagens e identificar leis.
- Uma família só, bp01w64c1280, em 1/5/25/75/100% (50% já pronto), 3 datasets × 3 splits = 45 pontos. Ela entrega as duas leituras vencedoras: L3 flatten (eggs, protozoan) e L4 1x1×1280 avg (larvae).
- Ressalva: o ckpt da L3 é escolhido pelo κ avg. Por isso o SVM flatten da L3 também roda em 50%, para checar contra o bp01w64 (0.906/0.845/0.865).
- 200 épocas por camada, como em 50% (o corte para 60 não foi validado no κ, e a lei precisa da mesma receita). max_per_gpu 2 nas GPUs 0–2 (L4 ~3 GB).
- SVM ao fim do treino, sob o flock: primeiro o avg (L3, L4), depois o flatten (L3, incluindo 50%).

## 2026-11-01: grade g20, 20 tentativas para alcançar o I-JEPA (pedido do usuário)
- Alvo (I-JEPA, teste): eggs 0.940, larvae 0.943, protozoan 0.864. Melhor val atual: 0.906 / 0.917 / 0.866.
- Cada experimento muda 1 knob da base: bp01w64 L3 flatten para eggs/protozoan, bp01w64c1280 L4 avg para larvae. Tudo parte do SPiFiL e treina sem rótulo (decoder).
- Foco onde falta mais: 8 eggs, 8 larvae, 4 protozoan. Cada um roda 1 dataset × 3 splits, em fila numa GPU fixa (5 por GPU, GPUs 0–3).
- Cuidado com o job do crispim (lejepa, ~18 GB por GPU): o A6 vigia a memória e mata um g20 nosso se uma GPU ficar com < 4 GB livres. Pico na largada: 29–35 de 49 GB.
- Leis buscadas: backprop 0 → 0.1 → 0.3 → 1.0; largura 48 → 64 → 96 → 128; largura da 1x1 1280 → 2560; sementes 50 → 100.
- Em paralelo: o A5 busca ideias na literatura (pooling, objetivos SSL para CNN, leis de escala, FLIM).

| id | dataset | base | mudança | GPU | hipótese SPiFiL |
|---|---|---|---|---|---|
| e_bp03 | helminth-eggs | flat | {'layerwise_backprop': 0.3} | GPU 0 | camadas antigas acompanham mais o filtro SPiFiL novo |
| e_bp10 | helminth-eggs | flat | {'layerwise_backprop': 1.0} | GPU 1 | SPiFiL so como init, encoder todo livre |
| e_w96 | helminth-eggs | flat | {'layer_channels': [48, 64, 96]} | GPU 2 | mais prototipos SPiFiL por classe |
| e_w128 | helminth-eggs | flat | {'layer_channels': [64, 96, 128]} | GPU 3 | lei da largura, 2o ponto |
| e_aug | helminth-eggs | flat | {'data.init_args.geo_aug': True} | GPU 0 | invariancia a rotacao/flip sobre o SPiFiL |
| e_bn | helminth-eggs | flat | {'model.init_args.batch_norm': True} | GPU 1 | BN estabiliza a escala dos filtros SPiFiL |
| e_bg0 | helminth-eggs | flat | {'model.init_args.recon_bg_weight': 0.0} | GPU 2 | fundo nao reconstruido: foco total no ovo |
| e_k3 | helminth-eggs | flat | {'kernel_size': 3} | GPU 3 | patch SPiFiL menor, mais local |
| l_bp03 | helminth-larvae | avg | {'layerwise_backprop': 0.3} | GPU 0 | camadas antigas acompanham mais o filtro SPiFiL novo |
| l_bp10 | helminth-larvae | avg | {'layerwise_backprop': 1.0} | GPU 1 | SPiFiL so como init, encoder todo livre |
| l_w96 | helminth-larvae | avg | {'layer_channels': [48, 64, 96, 1280]} | GPU 2 | base mais larga sob a 1x1 |
| l_aug | helminth-larvae | avg | {'data.init_args.geo_aug': True} | GPU 3 | larva em qualquer orientacao: invariancia ajuda |
| l_bn | helminth-larvae | avg | {'model.init_args.batch_norm': True} | GPU 0 | BN estabiliza a escala dos filtros SPiFiL |
| l_c2560 | helminth-larvae | avg | {'layer_channels': [32, 48, 64, 2560]} | GPU 1 | lei da largura da 1x1, 2o ponto |
| l_bg0 | helminth-larvae | avg | {'model.init_args.recon_bg_weight': 0.0} | GPU 2 | fundo nao reconstruido: foco total na larva |
| l_sp100 | helminth-larvae | avg | {'n_superpixels': 100} | GPU 3 | mais sementes: mais filtros SPiFiL, menos aleatorios na 1x1 |
| p_bp03 | protozoan-cysts | flat | {'layerwise_backprop': 0.3} | GPU 0 | camadas antigas acompanham mais o filtro SPiFiL novo |
| p_w96 | protozoan-cysts | flat | {'layer_channels': [48, 64, 96]} | GPU 1 | largura ajudou protozoan (w64) |
| p_aug | protozoan-cysts | flat | {'data.init_args.geo_aug': True} | GPU 2 | invariancia a rotacao/flip sobre o SPiFiL |
| p_bn | protozoan-cysts | flat | {'model.init_args.batch_norm': True} | GPU 3 | BN estabiliza a escala dos filtros SPiFiL |

## 2026-11-01: leituras novas só de eval (sugestão do A5, literatura)
- O A5 (relatório em A_reports/october/discovery_literature_ideas.md) aponta que o ganho mais barato está na leitura, coerente com o achado de ~0.09. Novos modos em `_encode_pooled` (eval/svm.py), um só SVM:
  - `spp`: pirâmide avg 1x1+2x2+4x4 (C×21-d). Guarda posição grossa sem achatar tudo (DSP, arXiv 1504.05277: +2 a +6).
  - `l2flat`: L2 por pixel + raiz com sinal antes do flatten. Tira a escala de ativação de cada pixel (DSP Tab. 2: +0.4 a +3.6).
  - `gem`: GeM p=3, entre avg e max (arXiv 1711.02512).
- Fila svmq_readouts, 1 por vez: bp01w64 L3 (spp, gem, l2flat) e bp01w64c1280 L4 (spp, gem), sem treino novo.
- Ressalva: o I-JEPA é ViT (tokens), então essas leituras não se aplicam 1:1 a ele. A comparação continua contra o κ de teste dele, e a escolha da leitura sai só da validação.
- Próximas apostas de treino (A5): autoencoder mascarado (loss só no tapado, SparK) e SIGReg (core/losses/real_sigreg.py já existe). Só depois das leituras.

## 2026-11-01: regra do usuário, nada de flatten e leitura < 2000-d
- Pedido: "evite avaliar com flatten. so avalie se a camada for menor que 2000". Vale para o SVM e para o κ de treino (escolha do ckpt).
- Parados e relançados os 12 g20 de eggs/protozoan (estavam no L1, com κ de treino flatten): agora o κ de treino é avg. O SVM é avg (L3) e, onde 64×21 < 2000, também spp (1344-d). w96/w128 ficam só em avg/gem (spp daria 2016/2688).
- Abandonado: l_c2560 (L4 de 2560-d não pode ser avaliado). Também cortados da fila de leituras o l2flat (37k-d) e o spp da L4 1280 (26.880-d). O eval "pcts_flat" virou spp (mesmo arquivo, nome/CSV _pcts_spp).
- Primeiro dado do spp (bp01w64 L3 eggs): 0.901/0.870, contra flatten 0.929/0.892 e avg ~0.845. O spp fica entre os dois e é a melhor leitura < 2000-d até agora em eggs.
- Nova referência: as comparações passam a ser feitas só entre leituras < 2000-d (avg, gem, spp). O flatten antigo fica como histórico.

## 2026-11-01: melhores leituras < 2000-d (κ val, média de 3 splits)
- eggs: spp L3 0.878 ≈ L4 avg 0.880; ambos acima do avg L3 0.845 (+0.033). Alvo I-JEPA 0.940 (−0.06).
- larvae: L4 avg 0.917 segue o melhor. O spp L3 dá 0.872: a posição atrapalha em larvae mesmo em pirâmide. Alvo 0.943 (−0.026).
- protozoan: spp L3 só no s1 (0.847), acima da L4 avg (0.834) e do avg L3 (0.811). Alvo 0.864. Faltam s2/s3.
- Sem o flatten, a distância para o I-JEPA cresceu em eggs. A próxima onda tem que atacar a leitura de eggs dentro de < 2000-d. GeM pendente.

## 2026-11-01: GeM na L4 1x1×1280 é a melhor leitura < 2000-d; g20 de eggs/protozoan rebaseados
- κ val (média de 3 splits), L4 gem (1280-d): eggs 0.899, larvae 0.910, protozoan 0.875 (0.876/0.875/0.873). Contra a L4 avg: 0.880/0.917/0.837; contra o spp L3: 0.878/0.872/0.829.
- protozoan 0.875 é o maior κ medido ali, acima do antigo flatten (0.866) e do I-JEPA (0.864, teste). eggs fica a 0.007 do antigo flatten com 22x menos dimensões. Na L3 (64 canais) o GeM perde para o avg: ele só funciona sobre muitos canais.
- Hipótese SPiFiL: a L4 tem 100–400 protótipos SPiFiL + aleatórios. O GeM (p=3) dá peso aos picos, isto é, a "este protótipo apareceu forte em algum lugar", parecido com detectar a presença de uma parte típica do parasita.
- Decisão: os 12 g20 de eggs/protozoan (estavam no L1) foram parados e relançados sobre a base L4 1x1×1280, com κ de treino e SVM em gem (L4) + avg (L3/L4). w96/w128 viram [48,64,96,1280]/[64,96,128,1280]; k3 vira layer_kernels [3,3,3,1]. Os 7 g20 de larvae seguem; o eval gem L4 deles entra na fila quando terminarem. A varredura de porcentagens também ganha eval gem L4 (1–100%).
- Distância ao I-JEPA (50%): eggs −0.041, larvae −0.026, protozoan +0.011.
- Crispim: o lejepa dele parou às 13:10 (4 ranks juntos, sem OOM do kernel, folga ≥ 7.7 GB). Causa desconhecida; usuário avisado.

## 2026-11-01: teste pedido pelo usuário; pesquisa de técnicas de convolução
- O usuário pediu o teste com os melhores modelos. Fila svmq_test, 1 por vez: bp01w64c1280 L4 com GeM e com avg, 3 datasets × 3 splits, eval_set test. A leitura de cada dataset já foi escolhida antes, na validação (eggs/protozoan GeM, larvae avg). O teste não muda essa escolha.
- Pesquisa (A5): técnicas de convolução compatíveis com SPiFiL + crescimento + decoder (SE, ECA, CBAM, CoordAtt, SimAM, GRN, blocos residuais/invertidos, BlurPool, dilatada/MixConv, GroupNorm...), com init próxima da identidade para não estragar os filtros SPiFiL.
- Implementação (A7): opção `se_reduction` (squeeze-and-excitation), desligada por padrão e com init quase identidade.

## 2026-11-01: técnicas de convolução (A5) → GRN, leitura densa, SE
- Ranking do A5 (A_reports/october/discovery_conv_techniques.md): 1) GRN do ConvNeXt V2 depois da L4 1x1, feito para MAE em CNN, que começa como identidade e custa 2·C params (no artigo, 83.7→84.6, acima de SE e CBAM); 2) ECA; 3) leitura densa GeM L3+L4; 4–5) BlurPool/SoftPool (exigem recrescer do L1); 6) CoordAtt; 7) GCNet; 8) SimAM.
- Descartados por incompatibilidade com o SPiFiL (um banco de filtros cheio por camada): depthwise/MobileNetV2/ConvNeXt block, SKNet/ASPP/Inception/MixConv, Weight Standardization. GroupNorm/LayerNorm não começam como identidade.
- Regra para todos: init exatamente identidade. Uma porta sigmoid em 0.5, ou mesmo 0.95 acumulada, desloca a calibração dos próximos filtros SPiFiL.
- Feito: leitura `gem_dense` (GeM da penúltima + da última camada, 1344-d < 2000) em eval/svm.py. Eval na fila depois do teste.
- A7: SE (`se_reduction`) implementado, desligado por padrão. Pedido o ajuste para identidade exata (2·σ, fc2 = 0) e depois o GRN (`grn`) na última camada.

## 2026-11-01 17:10: teto de 20 GB por GPU (ordem do usuário) e fila de espera
- Medido antes: GPU0 28.0, GPU1 20.9, GPU2 25.6, GPU3 20.5 GB, tudo nosso (o crispim saiu às 13:10). Parte disso são workers Ray ociosos (~0.26 GB cada, ~20 na GPU0).
- Pausados os 6 menos prioritários: p_aug, p_bp03, p_w96, p_bn (o protozoan já passa o I-JEPA na val) e l_bn, l_w96 (mais pesados; o BN caiu no artigo do ConvNeXt V2). O split 1 de l_bn/l_w96 estava pronto, então eles voltam com splits [2,3]. Depois: 15.6 / 16.2 / 18.0 / 17.2 GB.
- Fila em scratchpad/gpu_queue.txt (ordem: l_bn, l_w96, p_aug, p_bp03, p_w96, p_bn). O A6 virou o guardião do teto (alvo ≤19 GB): lança da fila quando sobra ≥4 GB, pausa e re-enfileira o menos prioritário se passar de 19.5 GB.
- Nova investigação (pedido do usuário, A8): o quanto o backprop muda os mapas de ativação em relação ao SPiFiL cru, dentro × fora da máscara, nos modelos top (bp01w64c1280, bp01w64) e no controle aleatório. Resultados em artifacts/.../<ponto>/act_drift/, mais a lei "drift × escala do backprop × κ". Roda em CPU, sem GPU.

## 2026-11-01: TESTE dos melhores modelos (bp01w64c1280 L4, pedido do usuário)
| κ teste (média de 3 splits) | eggs | larvae | protozoan |
|---|---|---|---|
| L4 GeM | 0.891 (0.913/0.893/0.868) | 0.939 (0.951/0.929/0.936) | 0.886 (0.876/0.898/0.884) |
| L4 avg | 0.880 (0.895/0.869/0.876) | 0.928 (0.938/0.902/0.943) | s2 refeito (o A7 quebrou o arquivo por 5 s) |
| I-JEPA (teste, 50%) | 0.940 | 0.943 | 0.864 |
- Leitura escolhida na validação: eggs/protozoan GeM, larvae avg. No teste: eggs −0.049, larvae −0.015, protozoan +0.022, ou seja, protozoan passa o I-JEPA também no teste. Em larvae, o GeM dá 0.939 no teste (−0.004), mas a escolha da val foi avg; não troco a leitura olhando o teste.
- SE e GRN (A7, commit e288dd0) entram na frente da fila: e_grn, l_grn, e_se, l_se (base bp01w64c1280, 1 mudança cada).

## 2026-11-01: quanto o backprop muda os mapas do SPiFiL (A8, A_reports/october/discovery_act_drift.md)
- A mudança é pequena nas pontas e grande no meio. Cosseno dos kernels: L1 ~1.00, L4 1x1 0.97, L2/L3 ~0.55. Mudança relativa da ativação (c1280): L1 0.19, L2 0.49, L3 0.89, L4 0.41.
- Achado principal: o treino APAGA o contraste objeto × fundo que o SPiFiL trazia. Energia dentro/fora da máscara: L2 2.0 → 1.2, L3 2.9 → 1.0. Na L4, os canais SPiFiL perdem mais (0.84 → 0.42) que os aleatórios (0.94 → 0.73). Isso confirma, com números, o achado do usuário de que "os gradientes aprendem errado".
- Nos kernels, o SPiFiL anda ~2x MAIS que o aleatório (rel L2/L3 1.07/1.22 contra 0.55/0.59). Não há lei "mais deriva = mais κ": o sinal troca por família e dataset (ρ geral −0.21, p 0.21).
- Mais backprop = mais deriva (g20 s1: eggs 0.40/0.43/0.56; larvae 0.79/0.79/0.92 para bp 0.1/0.3/1.0). A lei com κ fica para quando saírem os SVMs do g20.
- Máscara sem rótulo = máscara do dataset (fundo constante); o objeto ocupa só 2–4% da imagem. Artefatos em artifacts/spifil_growth/<fam>/<ponto>/act_drift/ (metrics.csv, act_maps.png).
- Próxima hipótese: ancorar o treino no SPiFiL (L2-SP: penalizar a distância dos kernels ao valor SPiFiL) para preservar o contraste e ver se o κ sobe.

## 2026-11-01: âncora L2-SP no SPiFiL (A7) e 4 candidatos na fila
- `spifil_anchor` (λ): soma `λ·Σ ||W_k − W0_k||²/||W0_k||²` à loss. W0 = pesos no início do run, ou seja, o kernel SPiFiL cru (L<k>_grow) para a camada nova e o ckpt anterior para as antigas. Começa em 0; desligado, nada muda.
- Escolha de λ: o A7 calibrou na L4 (drift 0.033, λ 1.8/8.8/17.6 para 10/50/100% da BCE). Mas o A8 mostra drift relativo ~1 nas L2/L3 sem âncora, onde o contraste some. Lá, λ 0.3 já vale ~50% da BCE, e λ 3 trava forte. Grade: λ ∈ {0.3, 3} em eggs e larvae (e_anc03, e_anc3, l_anc03, l_anc3), base bp01w64c1280.
- Hipótese (do usuário): o SPiFiL traz conhecimento a priori (contraste objeto/fundo). Prender os kernels perto dele preserva esse contraste e sobe o κ. Na fila logo depois dos GRN.

## 2026-11-01: 1º κ da g20, l_sp100 (100 superpixels) empata
- larvae κ val: gem L4 0.914/0.908/0.887 (0.903, contra 0.910 da base), avg L4 0.902 (contra 0.917, −0.016, na borda do ruído), avg L3 0.897 (contra 0.900).
- Lei das sementes SPiFiL (50 → 100): plana em larvae. Ter mais filtros SPiFiL e menos aleatórios na 1x1 não ajuda.

## 2026-11-01: l_aug (geo_aug) é o melhor larvae medido
- κ val gem L4: 0.924/0.919/0.915 (0.919) contra a base 0.910 (+0.010, ≥ base nos 3 splits, mas dentro do ruído ~0.012). Passa o avg L4 da base (0.917). O avg L4 do mesmo modelo cai para 0.901 (−0.016): o aug favorece o GeM e prejudica o avg.
- O geo_aug zera a distância val − treino na L4: a camada 1x1 não decora mais. Distância ao I-JEPA em larvae: −0.024 (val contra teste).
- Hipótese SPiFiL: o SPiFiL dá filtros de uma orientação (1 imagem por classe). A rotação/espelho força o treino a tornar os protótipos invariantes, e o GeM capta "a parte apareceu, em qualquer orientação". Aguardando e_aug para ver se o padrão se repete em eggs.

## 2026-11-01: confundidor da sonda de treino (A4), conferido
- O best_kappa.ckpt é escolhido pela sonda de treino na leitura do run. Conferido nos config.yaml: larvae g20 e a base usam avgpool2d, então em larvae NÃO há confundidor. A queda comum de −0.016 no avg dos 3 l_* contra a base indica que o 0.917 da base foi um sorteio alto (ruído do avg ~0.018 na média de 3 splits).
- Em eggs/protozoan há confundidor: os e_*/p_* usam sonda gem e a base usou avg. Criados e_base e p_base (bp01w64c1280 sem mudança, sonda gem) no topo da fila, como referência justa. Até lá, eggs/protozoan só se comparam g20 × g20.
- Larvae, gem L4 (g20 × g20, justo): aug 0.919 > bp10 0.914 > sp100 0.903.

## 2026-11-01: a base de larvae (0.917 avg) foi um sorteio alto; o pipeline não é bit-reprodutível
- Avg L4 larvae, por split: base 0.935/0.913/0.904; média das 3 g20 0.898/0.902/0.902. Excesso concentrado no s1 (+0.037). A referência realista do avg L4 em larvae é ~0.90.
- Setup conferido: os config.yaml da base e da g20 são idênticos, fora o próprio knob. Mas o pipeline não repete bit a bit: o L3 best de larvae s1 cai na época 154 na base e 184 no l_bp10, com pesos diferentes. Como os kernels SPiFiL da L4 são estimados sobre features diferentes, saem diferentes (max|Δ| ~0.9–1.07).
- Regra: a variação do pipeline inteiro por split (não determinismo da GPU + escolha da melhor época, propagados pelo SPiFiL) é maior que o ruído do SVM. Comparar contra a média do grupo, ou contra referências repetidas (e_base/p_base), nunca contra um único sorteio da base.

## 2026-11-01: larvae g20 num platô (4 eixos prontos)
- gem L4 κ val contra a média do grupo (0.913): aug 0.919 (+0.008), bg0 0.917 (+0.005), bp10 0.914 (0.000), sp100 0.903 (−0.014). Avg L4: todos entre 0.895 e 0.902 (±0.005).
- Nenhum eixo passa do ruído. Em larvae, os botões de treino mexem menos que a variação do pipeline inteiro: platô de ~0.91–0.92 (val). O teste da base deu 0.939 com gem, contra o I-JEPA 0.943.
- Leitura: o que falta em larvae não vem de treino de 1 knob. As próximas apostas são a leitura (gem_dense, na fila de SVM) e as técnicas que mudam a representação (âncora, GRN, SE).

## 2026-11-01: lei do kernel em eggs, k3 perde; lançado e_k7
- e_k3 (kernel 3x3 em L1–L3): gem L4 0.900/0.866/0.859 (0.875) contra 0.899 da base (−0.024, nos 3 splits). Avg L4 0.861 (−0.019), avg L3 0.827 (−0.015). Passa da variação do pipeline (~0.018), e o confundidor da sonda (gem) favorece o k3, então a perda real é ≥0.024. Primeiro eixo da g20 acima do ruído.
- Hipótese SPiFiL: o patch do superpixel recortado pelo filtro carrega a forma local do ovo; com 3x3 ele fica pequeno demais. Próximo ponto da lei: e_k7 (layer_kernels [7,7,7,1]), na fila logo depois de e_base/p_base.

## 2026-11-01: eggs, backprop 0.3 empata
- e_bp03 (layerwise_backprop 0.3): gem L4 0.914/0.920/0.881 (0.905) contra 0.899 da base (+0.006, dentro do ruído; parte pode ser o viés da sonda gem). Contra o e_k3, na mesma rodada g20, fica +0.030.
- Hipótese SPiFiL: com mais backprop, as camadas antigas acompanham melhor o filtro SPiFiL novo. Em eggs, a lei do backprop é plana de 0.1 a 0.3; falta o ponto bp 1.0 (e_bp10).

## 2026-11-01: eggs, aug neutro; platô em 7 eixos
- e_aug (flip H/V + rotação 90): gem L4 0.903, avg L4 0.881, contra a base 0.899/0.880. Empate. Não repete o "gem sobe, avg cai" de larvae.
- Hipótese SPiFiL: invariância a rotação/flip sobre os filtros SPiFiL. Em eggs não muda nada, porque os ovos já são ~isotrópicos.
- Placar dos 7 eixos com κ (larvae: aug, bg0, bp10, sp100; eggs: bp03, aug, k3): só o k3 passa do ruído, e para pior (−0.029). O platô é ~0.90 em eggs e ~0.91 em larvae (val). Treino de 1 knob não fecha o gap com o I-JEPA em eggs; as apostas restantes são a leitura (gem_dense) e as mudanças de representação (âncora, GRN, SE, k7).

## 2026-11-01: lei do backprop fechada (eggs 3 pontos, larvae 2)
- gem L4 κ val. eggs: bp 0.1 → 0.3 → 1.0 dá 0.899 → 0.905 → 0.908 (e_bp10 0.923/0.900/0.903). larvae: 0.1 → 1.0 dá 0.910 → 0.914. Avg L4 do e_bp10 0.883, igual à base.
- Lei: sobe de forma monotônica, mas fraca (+0.010 eggs, +0.004 larvae no total), e cada passo fica dentro do ruído. O ganho só aparece no gem.
- Hipótese SPiFiL: o SPiFiL vale como inicializador; soltar o encoder inteiro não apaga o que ele deu e ajuda um pouco. Junto com o A8 (mais backprop = mais drift), mostra que o drift dos mapas não custa κ.

## 2026-11-01: eggs, bn e bg0 neutros (6 eixos)
- gem L4 κ val, g20 × g20: bp10 0.908, bp03 0.905, aug 0.903, bn 0.897, bg0 0.892, k3 0.875. Fora o k3 (−0.026), os 5 eixos ficam numa faixa de 0.016, dentro da variação do pipeline.
- Hipóteses SPiFiL: BN estabiliza a escala dos filtros SPiFiL (não muda nada); fundo fora da reconstrução foca o ovo (não muda nada). 0 NaN, nenhuma divergência.

## 2026-11-02: orçamento de 60 épocas (pedido do usuário: treino muito lento)
- Premissa corrigida pelo A4: nas disc_*/g20 o svm_probe está desligado, então o ckpt e o EarlyStopping olham probe/val_recon_loss (patience 50, max 200). Não existe κ por época; o "confundidor da sonda gem" não existe.
- Simulação em 752 estágios (A_reports/october/discovery_early_stop.md): nas L1–L3 o best cai na época 193–198, porque o cosine ainda baixa a loss no fim. Patience 20 poupa só 4%; max_epochs 60 poupa ~69%.
- Decisão: max_epochs 60 (model e trainer, cosine reescalado) para tudo o que ainda não começou. Referência nova: disc_g20_base_e60 (bp01w64c1280, 3 datasets), no topo da fila. Os eixos da fila sem split pronto viraram *_e60 (e_se, l_se, p_aug, p_bp03, p_w96, p_bn) e são comparados contra base_e60. l_bn, l_w96, l_bp03 e l_grn ficam em 200, porque já têm splits prontos.
- Critério de aceite do e60: gem L4 a ≤0.018 da base de 200 (0.899/0.910/0.875). Se passar, vira o padrão. Hipótese SPiFiL: o SPiFiL já entrega filtros bons na partida, então o backprop precisa de poucas épocas para refinar.
- Ideias da literatura (rodada 2, A_reports/october/discovery_literature_round2.md): RankMe por camada como parada sem rótulo (calibrar contra κ antes de usar); successive halving nas fronteiras de camada para a grade; TTA e FixRes no SVM (só eval); EMA/SWA; whitening ZCA e k-means dos patches antes do SPiFiL; PCA para parte dos filtros; ortogonalidade; loss local por camada.

## 2026-11-02: gem_dense empata; máscara no treino não sobe κ
- gem_dense (GeM da L3 + L4 juntas, 1344-d) L4 κ val: eggs 0.905, larvae 0.904, protozoan 0.871, contra gem L4 0.899/0.910/0.875. Empate dentro do ruído: juntar a camada anterior na leitura não ajuda.
- Peso do fundo na recon (máscara do dataset): 1.0 (R1) → 0.1 (fg01) → 0 (bg0), todos empatam (L3 e L4). O A8 mostrou que, mesmo com o fundo a 0.1, o treino apaga o contraste objeto/fundo do SPiFiL. Pesar a loss pela máscara não segura esse contraste. A aposta para isso é a âncora L2-SP.
- Lançado: SVM avg das L1/L2 do bp01w64c1280 (eval_svm_disc_bp01w64c1280_val_l12.yaml), para a curva por camada.

## 2026-11-02: o SVM precisa do fundo? leitura só no objeto (pedido do usuário)
- Pergunta do usuário: ao treinar com fundo preto, os kernels aprendem o fundo. Será que o SVM precisa "saber" que a convolução está passando por fundo?
- Mudança única, só na eval: `avgpool2d_fg` e `gem_fg` (eval/svm.py) fazem o mesmo pooling, mas só nas células do mapa que tocam o objeto. O fg é sem rótulo, o mesmo do recon_bg_weight: pixel a mais de 0.04 da mediana da borda. Smoke test em eggs: o objeto ocupa 1.4–7% dos pixels e o vetor muda 25–40%.
- Leitura do resultado: fg > todo, o fundo atrapalha o SVM; fg < todo, o contraste com o fundo é informação; empate, o fundo é indiferente.
- Rodando no bp01w64c1280 (val e teste; L3/L4 avg_fg e L4 gem_fg), em fila sob flock, um por vez.

## 2026-11-02: foco em eggs (pedido do usuário)
- Teste 50% (gem L4): larvae 0.939 vs I-JEPA 0.943 (−0.004); protozoan 0.886 vs 0.864 (+0.022); eggs 0.891 vs 0.940 (−0.049). Eggs é o único longe.
- Ressalva da escolha da leitura: em larvae, a val não separa gem de avg (0.910 vs 0.917, dentro do ruído). Regra única daqui em diante: gem L4 nos 3 datasets.
- O que já sabemos de eggs: bp, aug, bn e bg0 são neutros; k3 é pior (−0.026); SPiFiL empata com aleatório na L3. Ou seja, em eggs o SPiFiL ainda não está dando vantagem. São 8 classes, 400 sementes SPiFiL (50 por imagem, 1 imagem por classe), e o ovo ocupa 1.4–7% da imagem (~25–50 px).
- Hipótese principal: falta campo receptivo. O RF das L1–L3 (5x5, pool 2) é ~40 px, menor ou igual ao ovo, então os filtros SPiFiL não enxergam o ovo inteiro (casca + interior).
- Leis lançadas (eggs, 3 splits, e60, contra a base_e60; uma mudança cada):
  - profundidade: e_d5 (+L 96), e_d6 (+L 96, 128);
  - campo receptivo sem parâmetro: e_dil2 (L3 com dilation 2);
  - sementes SPiFiL: e_sp25, e_sp100.
  - Já em curso: kernel (k3 pronto, k7 rodando), largura (w96, w128), âncora, GRN, SE e leitura só no objeto (fg).
- Fila reordenada: eggs primeiro, larvae/protozoan depois. A4 diagnostica o erro de eggs por classe (confusão, tamanho e textura) na val.

## 2026-11-02: o SVM não precisa do fundo (gem_fg empata) + auditoria da receita (A12)
- gem_fg (pooling só no objeto) L4 κ val: 0.899/0.900/0.871, contra gem 0.899/0.910/0.875. Empate. O SVM não precisa "ver" o fundo; o resumo da imagem inteira e o do objeto carregam a mesma informação. Junto com o fundo 1.0/0.1/0 empatando no treino, o fundo é indiferente nos dois lados.
- Auditoria da receita (A_reports/october/discovery_recipe.md):
  - O pacote SPiFiL não mudou; o que é novo mora no wrapper flim/spifil.py e no runner.
  - A L1 não fica congelada: com fresh_decoder, só a camada nova treina, e na L1 essa camada é a conv1 (muda 11% em eggs s1).
  - best_val_svm_kappa no run_metadata.json guarda a val recon loss, não um κ.
  - Dataset novo dá KeyError em flim/spifil.py:461 (lê json[dataset] sem .get). Precisa da entrada no scripts/spifil_paper_seeds.json e de paper_classes: false.
  - O SPiFiL usa rótulo (Fisher + filtros por classe), e nenhuma chave desliga isso fora o random_layer_classic.
  - A val loss também pesa o fundo pela borda, então o fundo entra na escolha do ckpt.
  - A classe impureza não ganha filtro SPiFiL, mas o SVM classifica as 9 classes.
  - L4: 400/100/300 filtros SPiFiL (eggs/larvae/protozoan), o resto aleatório. Encoder com 200.944 params (188.468 em protozoan).
- Receita sem fundo recomendada (a medir): base_e60 + impurities: true + paper_classes: false + recon_bg_weight 1.0 + leitura gem. bg1 já empatou na L3; nomask_e60 e bg1_e60 estão na fila.
- Hipótese nova para eggs (aguarda a confusão do A4): a impureza é a maior classe (3.344 imagens) e é a única sem filtro SPiFiL.

## 2026-11-02: eggs erra ovo × impureza, e a impureza não tem semente SPiFiL
- Diagnóstico do A4 (A_reports/october/discovery_eggs_diag.md, val, bp01w64c1280): 218 de 220 erros do gem L4 são ovo ↔ impureza; só 2 são ovo ↔ ovo. Impureza → ovo: 137 (Ascaris 33, H.nana 25, Taenia 18). Ovo → impureza: 81. Na L3 avg o padrão é o mesmo, e mais forte (327/332).
- A impureza cobre toda a faixa de tamanho dos ovos (575–1.862 px), e a textura e o contraste médios dela ficam no meio dos ovos. O que separa é forma fina e detalhe interno.
- Causa provável: `paper_classes: true` tira a impureza das sementes do SPiFiL em eggs (classe 9) e em protozoan (classe 7) (flim/spifil.py:466). Larvae mantém a impureza, e larvae é o dataset que já empata com o I-JEPA. Conferido: com paper_classes false, eggs pega 9 imagens/9 classes e protozoan 7/7.
- Lançado (uma mudança cada, contra a base_e60), no topo da fila:
  - e_imp_e60 (paper_classes false): a impureza ganha semente SPiFiL. Ressalva: as 2 imagens do artigo viram sorteio.
  - e_res49_e60 (layer_pools [2,2,1,1]): L3/L4 a 49×49, para o detalhe interno.
  - p_imp_e60: o mesmo teste em protozoan.
- Teto aproximado (A4): cortar metade das 137 impurezas tomadas por ovo leva eggs de ~0.90 a ~0.93.

## 2026-11-02: l_grn empata em larvae
- l_grn (GRN na L4 1x1): gem L4 κ val 0.911, contra 0.910 da base e 0.913 do grupo. Empate.
- Hipótese SPiFiL: o GRN faz os 1280 canais (400 SPiFiL + 880 aleatórios) competirem e realça os raros. Em larvae, não muda nada.
- Larvae, 5 eixos: aug 0.919, bg0 0.917, bp10 0.914, grn 0.911, sp100 0.903. Faixa de 0.016, platô em ~0.91 (val). No teste, larvae já está a −0.004 do I-JEPA.

## 2026-11-02: lei da largura em eggs é plana
- gem L4 κ val: 32/48/64 0.899; 48/64/96 0.901 (0.907/0.910/0.887); 64/96/128 0.900 (0.914/0.901/0.884). Plana.
- Hipótese SPiFiL: mais protótipos SPiFiL por classe dariam mais nuance. Não deram, e isso é coerente com o diagnóstico: o erro é ovo × impureza, e a impureza não tem semente nenhuma, qualquer que seja a largura.
- Placar com 13 eixos (eggs + larvae): só o k3 passa do ruído, e para baixo. Os botões de treino e de largura estão esgotados. Apostas abertas: semente de impureza (e_imp), resolução (e_res49), campo receptivo (k7, d5, d6, dil2).

## 2026-11-02: prender os pesos no SPiFiL piora (l_anc3); e_grn empata
- l_anc3 (âncora L2-SP com λ=3 nos kernels SPiFiL, larvae): gem L4 κ val 0.893/0.865/0.836 (0.865). Fica −0.048 contra o grupo larvae, nos 3 splits; é a maior perda da g20 (~3x o ruído).
- Hipótese testada (do usuário, via A8): segurar os kernels perto do SPiFiL preserva o contraste objeto/fundo e sobe o κ. Refutada para λ forte. Junto com a lei do backprop (mais liberdade, κ um pouco maior), o SPiFiL vale como ponto de partida, não como restrição. O drift dos mapas é o treino melhorando, não estragando.
- Falta o λ=0.3 (e_anc03, l_anc03_e60) para fechar a lei da âncora.
- e_grn (eggs): gem L4 0.898, empate. O GRN é neutro nos dois datasets.

## 2026-11-02: a âncora forte perde também em eggs (lei da âncora)
- e_anc3 (L2-SP λ=3 nos kernels SPiFiL, eggs): gem L4 κ val 0.851/0.824/0.812 (0.829). Fica −0.069 contra o grupo eggs, nos 3 splits; é a maior perda da g20 (~4x a variação do pipeline).
- Com o l_anc3 (−0.048), a âncora forte perde nos 2 datasets. Lei combinada: quanto mais o treino pode se afastar do SPiFiL, melhor (âncora λ=3 −0.05 a −0.07; bp 0.1 → 1.0 +0.004 a +0.010). Regra da receita: SPiFiL é ponto de partida, não alvo; nada de âncora.
- Falta o λ=0.3 (e_anc03, l_anc03_e60), para ver se a perda cresce com λ.

## 2026-11-02: gate das 60 épocas: rejeitado em eggs, aceito em larvae; semente de impureza não ajuda
- base_e60 gem L4 κ val: eggs 0.893/0.873/0.859 (0.875), contra 0.899 com 200 ep: −0.024, passa do limite de 0.018. REJEITA. Larvae 0.904/0.902/0.904 (0.903) contra 0.910: −0.007. ACEITA. Protozoan pendente.
- Leitura: em eggs o treino longo importa, porque o detalhe fino (ovo × impureza) é aprendido no fim do cosine. Em larvae, 60 épocas bastam.
- e_imp_e60 contra base_e60 (mesmo orçamento): 0.847 contra 0.875 (−0.028). Na confusão val, impureza → ovo foi de 137 a 190. A semente de impureza piora. Hipótese: 1 imagem não representa a impureza, que é heterogênea (575–1.862 px, todas as texturas), e os filtros dela confundem mais do que separam. Fica fora da receita.
- Consequência: as famílias e60 de eggs (res49, d5, dil2, sp100, sp25, se, d6) se comparam com 0.875, não com 0.899. Quem ganhar a 60 é confirmado a 200.
- Lançado no topo: e_base_e100 (eggs, 100 épocas), o ponto do meio da lei do orçamento em eggs (60 → 0.875, 200 → 0.899).

## 2026-11-02: e_dil2 empata (campo receptivo por dilation não ajuda eggs)
- e_dil2_e60 (L3 com dilation 2, +16 px de RF): gem L4 κ val 0.877/0.854/0.871 (0.868), contra base_e60 de eggs 0.875 (mesmas 60 épocas): −0.007, empate.
- Hipótese SPiFiL: com um RF maior, o filtro SPiFiL da L3 enxergaria o ovo inteiro. Não confirmada por dilation. Faltam a profundidade (e_d5, e_d6) e a resolução (e_res49) para fechar a pergunta do RF/resolução.

## 2026-11-02: e_d5 não ganha em eggs (profundidade)
- e_d5_e60 (+1 camada SPiFiL 5x5 de 96 antes da 1x1, gem na L5): κ val 0.881/0.847/0.842 (0.857), contra base_e60 de eggs 0.875: −0.018, na borda do ruído, para baixo.
- Campo receptivo em eggs a 60 ép.: k3 perde (a 200 ép.), dil2 −0.007, d5 −0.018. Diminuir o campo perde, aumentar não ganha: o campo atual (~40 px) já basta.
- Ressalva: a 60 épocas, quem é mais fundo é penalizado (mais camadas treinadas com orçamento curto, e o gate mostrou que eggs precisa de treino longo). O d6 fecha a curva; se d5/d6 ficarem abaixo, a profundidade fica fora da receita de eggs.
- Restam a resolução (e_res49) e o kernel 7 (e_k7, a 200 ép.).

## 2026-11-02: TTA no SVM (só eval)
- A fila de eggs acabou. Para eggs andar sem GPU, a 1ª ideia da literatura r2: `gem_tta` (eval/svm.py) tira a média do gem L4 nas 8 rotações/flips da imagem (Shanmugam et al., ICCV 2021). Mudança única na leitura; o treino é o mesmo.
- Hipótese: o ovo × impureza depende de detalhe fino e orientação; a média sobre as 8 vistas tira o ruído de orientação do resumo. Rodando no bp01w64c1280 (val e teste), sob flock.

## 2026-11-02: lei da âncora fechada em eggs: qualquer âncora derruba
- e_anc03 (L2-SP λ=0.3, eggs, 200 ép.): gem L4 κ val 0.839/0.817/0.783 (0.813). Lei em eggs: λ=0 0.899; λ=0.3 0.813; λ=3 0.829. Larvae λ=3 −0.048.
- Não é dose-resposta: qualquer âncora tira ~0.07–0.09 em eggs. É a regra mais forte da g20: o SPiFiL é inicialização, não restrição. Segurar o contraste objeto × fundo do SPiFiL (A8) não vira κ.
- l_anc03_e60 sai da fila: a lei já está decidida, e a GPU vai para eixos que ainda podem ganhar.

## 2026-11-02: eggs num platô de construção; aposta no orçamento de treino
- 60 ép., gem L4 κ val contra base_e60 0.875: e_res49 (L4 a 49×49) 0.859/0.826/0.856 (0.847, −0.028); e_sp100 0.883/0.877/0.844 (0.868, −0.007, empate).
- Placar das 5 hipóteses de construção em eggs: nenhuma ganha. Empatam sp100 e dil2; perdem d5 (−0.018), res49 (−0.028) e imp (−0.028). Somando os 8 eixos a 200 ép., o erro ovo ↔ impureza não sai por resolução, campo receptivo, profundidade, largura, sementes nem âncora.
- O único botão que mexeu em eggs foi o orçamento: 60 ép. 0.875 → 200 ép. 0.899 (+0.024). E o best das L1–L3 cai no fim do cosine (época ~195/200), ou seja, ainda subindo. Lançado e_base_e400 (eggs, 400 ép.) no topo da fila; com o e_base_e100 já rodando, a lei do orçamento fica com 4 pontos (60/100/200/400).
- Hipótese SPiFiL: o SPiFiL entrega bons filtros de partida, mas o detalhe fino ovo × impureza é aprendido devagar pelo backprop. Mais orçamento, mais detalhe.

## 2026-11-02: TTA na leitura é o primeiro ganho acima do ruído em eggs desde o GeM
- gem_tta (média do gem L4 nas 8 rotações/flips; mesmo modelo bp01w64c1280, sem treino novo), κ val:
  - eggs 0.899 → 0.912 (+0.014; splits 0.909→0.943, 0.892→0.906, 0.894→0.887);
  - larvae 0.910 → 0.946 (+0.036; splits 0.896→0.959, 0.918→0.923, 0.915→0.955);
  - protozoan pendente.
- Leitura: o encoder não é invariante a rotação (o aug no treino não mudou o κ), mas o resumo médio das 8 vistas é. O ovo/larva vem em qualquer orientação, e o SVM ganha com um vetor que não depende disso. Entra na receita como leitura padrão (custo: 8× a inferência).
- O teste do gem_tta está na fila (só para o relatório). A confusão ovo ↔ impureza com gem_tta está rodando (A4).
- Anomalia leve (A4): no mask50_e60, a val loss oscila porque a máscara é sorteada a cada avaliação, então o ckpt é escolhido por uma máscara de sorte. Não corrigi no meio da família; o κ do mask50 deve ser lido com essa ressalva.

## 2026-11-02: o gem_tta corta justamente as impurezas tomadas por ovo
- Confusão val de eggs com gem_tta (A4; κ 0.941/0.911/0.890): erros 220 → 185 (−16%). Impureza → ovo 137 → 111, o maior corte; ovo → impureza 81 → 73; recall da impureza 0.945 → 0.956. Ancylostoma 0.883 → 0.928; Schistosoma 0.903 → 0.925; Enterobius parado em 0.849.
- Leitura: uma impureza que parece ovo numa pose deixa de parecer nas outras 7; a média sobre as orientações separa ovo × impureza melhor que qualquer botão de construção (~25 eixos sem ganho).
- Lançado (só eval): gem_tta sobre e_aug, l_aug (aug no treino + TTA, redundantes?), e_bp10 (melhor eggs pela val) e e_base (a referência repetida).

## 2026-11-02: TTA confirmado no teste: larvae e protozoan passam o I-JEPA
- gem_tta val (A4): eggs 0.912 (+0.014), larvae 0.946 (+0.036), protozoan 0.917 (+0.043); sobe em 8 de 9 splits. O ganho cresce quanto menos o objeto tem orientação canônica (protozoan é o que mais ganha).
- Teste (só reporte; a leitura foi escolhida pela val), bp01w64c1280 L4:
  - gem → gem_tta: eggs 0.891 → 0.908 (0.929/0.900/0.894); larvae 0.939 → 0.952 (0.958/0.945/0.954); protozoan 0.886 → 0.921 (0.913/0.933/0.919).
  - Contra o I-JEPA (teste 50%): eggs −0.032, larvae +0.009, protozoan +0.058.
- Ressalva de justiça: o I-JEPA foi medido com uma vista só. Para comparar de igual para igual, o I-JEPA também deveria ser avaliado com TTA.
- e_se_e60 (SE, eggs): gem L4 0.871, contra 0.875 do base_e60. Empate.

## 2026-11-02: lei do kernel fechada em eggs; ruído do pipeline medido
- e_base (o mesmo pipeline da base, retreinado): gem L4 κ val 0.907/0.915/0.884 (0.902), contra 0.899 da base original. Por split: −0.002/+0.022/−0.010; na média, +0.003. É a 1ª medida direta da variação do pipeline (por split até ±0.02, na média de 3 splits ~0.003). Referência de eggs a 200 ép.: ~0.900.
- e_k7 (kernel 7 em L1–L3): 0.914/0.905/0.886 (0.902). Empata.
- Lei do kernel em eggs: k3 0.875 → k5 ~0.900 → k7 0.902. Do 3 para o 5 ganha, depois satura. Junto com dil2 e d5: o RF do k5 (~40 px) basta. Receita: k5.
- Correção de método (A4): famílias de 200 ép. se comparam com o e_base; as de 60 ép., com o base_e60.

## 2026-11-02: aug no treino + TTA na leitura somam: maior eggs medido
- e_aug (flip/rot90 no treino) + gem_tta: κ val L4 0.938/0.916/0.906 (0.920). Contra e_aug com gem (0.903): +0.017. Contra a base com gem_tta (0.912): +0.008 (s1 −0.005, s2 +0.010, s3 +0.019).
- Leitura: o aug sozinho não mexia o κ (gem), mas junto do TTA soma. Treinar com as 8 poses deixa as 8 vistas do TTA mais coerentes entre si. Eggs val a −0.020 do I-JEPA (teste), contra −0.041 com gem.
- Receita candidata: geo_aug no treino + gem_tta na leitura. Confirmar com l_aug (na fila) e com p_aug_e60 (treinando).
- e_sp25_e60: 0.881, contra 0.875 do base_e60 (empate). Lei das sementes plana: sp25 0.881, sp50 0.875, sp100 0.868.
- Teste do e_aug/l_aug com gem_tta (só reporte) enfileirado depois do svmq_tta2.

## 2026-11-02: aug + TTA somam também em larvae; receita candidata para os 3
- l_aug + gem_tta: κ val L4 0.949/0.954/0.955 (0.953). Contra a base com gem_tta (0.946): +0.007. Contra l_aug com gem (0.919): +0.033. É o mais estável medido (desvio 0.003); o aug tira a oscilação do s2. Fica 0.010 acima do I-JEPA (teste).
- Receita candidata para os 3 datasets: bp01w64c1280 + geo_aug (treino) + gem_tta (leitura), 200 épocas.
- Falta protozoan a 200 ép.: o g20_p_aug original foi pausado sem split pronto. Lançado disc_g20_p_aug200 (o mesmo YAML, família nova para não misturar com os parciais), no topo da fila, com eval gem e gem_tta.

## 2026-11-02: SE neutro nos 2 datasets
- l_se_e60 (SE, larvae, 60 ép.): gem L4 κ val 0.915/0.922/0.890 (0.909), contra base_e60 de larvae 0.903: +0.006, empate. Com e_se_e60 (0.871 contra 0.875), o SE é neutro em eggs e em larvae; fica fora da receita.

## 2026-11-02: e_bp10 + TTA é o maior eggs; lançado bp10 + aug
- e_bp10 (backprop 1.0) + gem_tta: κ val L4 0.935/0.920/0.918 (0.925). Contra a base com gem_tta (0.912): +0.012. Contra e_aug + TTA (0.920): +0.005. É o mais estável de eggs; fica a −0.016 do I-JEPA (teste).
- Leitura: com o gem, o bp10 ficava no ruído (+0.009). Com o TTA, a vantagem dele aparece. Soltar o encoder inteiro deixa as vistas mais ricas.
- Lançado disc_g20_e_bp10_aug (bp 1.0 + geo_aug, eggs, 200 ép.; uma mudança contra e_aug e contra e_bp10), no topo da fila, com eval gem e gem_tta. Hipótese SPiFiL: SPiFiL como partida, rede inteira livre e treinada nas 8 poses dá o resumo mais invariante.

## 2026-11-02: lei da profundidade fechada em eggs (cai); o ganho do bp10 + TTA é real
- e_base + gem_tta (a base repetida): κ val 0.928/0.916/0.881 (0.908), contra 0.912 da base original. Referência com TTA ~0.910. O e_bp10 + TTA (0.925) fica +0.015 acima, mais que a variação de repetição: ganho real.
- e_d6_e60 (6 camadas, gem na L6): 0.843/0.845/0.865 (0.851), contra base_e60 0.875 (−0.024). Lei da profundidade em eggs (60 ép.): 4 camadas 0.875 → 5 0.857 → 6 0.851. Cai a cada camada. Ressalva: a 60 ép., o mais fundo treina menos por camada.
- Síntese para eggs: ajudam o backprop livre e a invariância a rotação (TTA + aug). Não ajudam profundidade, resolução, campo receptivo (kernel > 5, dilation), largura, sementes, âncora, SE, GRN, BN.

## 2026-11-02: receita bp10 + aug + TTA nos 3 datasets
- Para fechar a receita nos 3 datasets com a mesma configuração: lançados disc_g20_l_bp10_aug e disc_g20_p_bp10_aug (bp 1.0 + geo_aug, 200 ép.), no topo da fila, cada um com eval gem/gem_tta em val e teste. O e_bp10_aug (eggs) já treina.
- Eval só: gem_tta sobre o l_bp10 (larvae, 200 ép., já treinado).

## 2026-11-02: gate das 60 épocas fechado nos 3 datasets
- base_e60 gem L4 κ val: eggs 0.875 contra 0.899 (−0.024, REJEITA); larvae 0.903 contra 0.910 (−0.007, ACEITA); protozoan 0.874/0.859/0.852 (0.861) contra 0.875 (−0.014, ACEITA na borda).
- Regra da busca: triar com 60 ép. em larvae e protozoan (~69% menos tempo) e confirmar os melhores com 200; em eggs, usar 200, ou comparar as e60 só entre si. O orçamento necessário segue a dificuldade do erro restante: eggs > protozoan > larvae.
- A receita final (bp10 + aug + TTA) está a 200 ép. nos 3, para ficar comparável ao melhor de cada dataset.

## 2026-11-02: lei do orçamento em eggs: 60 ≈ 100 < 200
- e_base_e100 (eggs, 100 ép.): gem L4 κ val 0.890/0.871/0.858 (0.873). Lei: 60 ép. 0.875; 100 ép. 0.873; 200 ép. ~0.900 (base 0.899, e_base 0.902). De 60 a 100 não muda; o ganho (+0.027) aparece só entre 100 e 200.
- Hipótese (A4): o que conta não é o número de épocas, e sim quanto tempo o cosine passa na cauda de lr baixa (ajuste fino). O e400 (rodando) dá o próximo ponto. Se a hipótese se sustentar, o teste barato é cosine de 100 ép. + 50 ép. em eta_min (precisa de uma opção nova no scheduler; não implementado ainda).
- Para eggs, 100 épocas não economizam nada: ficam as 200.

## 2026-11-02: o que soma com o TTA depende do dataset
- l_bp10 + gem_tta: κ val 0.915/0.917/0.948 (0.927), contra a base com TTA 0.946 (−0.019; dentro do ruído de larvae, mas sem ganho).
- Com TTA, eggs ganha com bp10 (+0.012 a +0.015) e com aug (+0.008); larvae ganha só com aug (+0.007, o mais estável).
- Receita comum: geo_aug + gem_tta. O bp10 é específico de eggs.
- Candidatos por dataset: eggs bp10 + aug + TTA (e_bp10_aug rodando); larvae aug + TTA (l_aug, 0.953); protozoan, falta o p_aug200 + TTA.
- O l_bp10_aug segue rodando: ele mostra se uma receita única nos 3 datasets custa algo em larvae.

## 2026-11-02: e_aug_wh: ZCA nos patches antes do SPiFiL
- Mudança: opção `whiten` no `grow` (default off). Com ela, os patches candidatos passam por ZCA, W = U diag(1/sqrt(S+eps)) U^T com eps = 1e-2·mean(S), fitada nos próprios patches de treino (Coates & Ng 2011; Thiry et al., ICLR 2021). O ranking de Fisher, a alocação e a seleção rodam no espaço branco, e o filtro gravado é W·f, que dá a mesma resposta no espaço original.
- Hipótese: o whitening tira a correlação entre pixels vizinhos. Assim o Fisher escolhe filtros de forma e borda, e não de brilho médio, o que pode separar ovo de impureza.
- Experimento: growth_disc_g20_e_aug_wh = e_aug + `whiten: true` (uma mudança só). Referência: e_aug com gem_tta, κ val 0.920.

## 2026-11-02: PCA-whitening do vetor final antes do SVM (só eval)
- Sufixo `_pw` no embed_mode (eval/svm.py): PCA(whiten=True) fitada nas features de TREINO, e depois L2, nos 2 lados, antes do fit_svm único (Jégou & Chum, ECCV 2012; par clássico do GeM em retrieval).
- Rodando (val): bp01w64c1280 gem_pw e gem_tta_pw; e_aug e l_aug gem_tta_pw. Referências: gem 0.899/0.910/0.875; gem_tta 0.912/0.946/0.917; e_aug+tta 0.920; l_aug+tta 0.953.
- Hipótese: os 1280 canais (400 SPiFiL + 880 aleatórios) são correlacionados; descorrelacionar e igualar a escala deixa o SVM linear usar as direções raras que separam ovo × impureza.

## 2026-11-02: semente de impureza também não ajuda protozoan
- p_imp_e60: gem L4 κ val 0.872/0.847/0.857 (0.859), contra base_e60 de protozoan 0.861 (empate). Nos 2 datasets: eggs −0.028, protozoan −0.002. A semente de impureza (paper_classes false) não ajuda; para datasets novos, paper_classes false é neutro ou um custo pequeno.

## 2026-11-02: BN perde em larvae
- l_bn (batch_norm, 200 ép.): gem L4 κ val 0.894/0.894/0.873 (0.887), contra a base 0.910: −0.023 nos 3 splits, acima da variação do pipeline. Em eggs é neutro (0.897). Fica fora da receita.
- Placar larvae (gem): aug 0.919, bg0 0.917, bp10 0.914, grn 0.911, sp100 0.903, bn 0.887, anc3 0.865.

## 2026-11-02: 4 opções parecidas com o whitening na seleção do SPiFiL (eggs)
- Diagnóstico: 99% dos erros de eggs são ovo ↔ impureza (detalhe fino). Hipótese geral: o problema está em como o SPiFiL escolhe os filtros. Cada família abaixo é e_aug + UMA opção (default off; desligada, o caminho é bit a bit o de antes). Referência: e_aug com gem_tta, κ val 0.920 (test 0.917).
- Como no `whiten`, as opções valem para TODAS as camadas crescidas (L1..L4), não só para a L1.
- e_aug_pn (`patch_norm: true`, Coates & Ng 2011): cada patch menos a sua média e dividido pelo seu desvio, antes do Fisher. Hipótese: sem o brilho e o contraste de cada patch, o Fisher escolhe pela forma. Na conv, o filtro é P·f / desvio médio (P = I − 11ᵀ/D).
- e_aug_div (`select_diverse: 0.9`): NMS de filtros. Na ordem do Fisher, um candidato com |cos| > 0.9 com outro melhor da mesma classe vai para o fim do rank. Hipótese: o SPiFiL gasta filtros em quase duplicatas. No smoke (L1, split 1), 71 de 400 candidatos são quase duplicatas.
- e_aug_knn (`knn_smooth: 5`): cada candidato vira a média dos seus 5 vizinhos da mesma classe (kmeans-lite). Hipótese: um centroide local é menos ruidoso que um patch só.
- e_aug_ipc3 (`images_per_class: 3`): 3 imagens por classe (a do artigo + 2 sorteadas). Hipótese: mais candidatos para o Fisher, e N sai de 400 para 1200, acima de D na L2 (800) e igual a D na L3 (1200). Hoje N fica abaixo de D nas duas.
- Evals: só val_gem, val_gemtta e test_gemtta (fila de SVM cheia).

## 2026-11-02: o _pw sem shrinkage colapsou; refeito com shrinkage
- gem_tta_pw (PCA(whiten=True) puro): eggs κ 0.343/0.239/0.444, larvae ~0 (prevê uma classe). Com ~n_treino componentes, branquear as direções de variância ~0 amplifica ruído. Resultado inválido; os CSVs _pw ficam como histórico.
- Correção: escala 1/sqrt(var + 0.1·mean(var)) (shrinkage, o mesmo eps relativo do ZCA do SPiFiL). Relançado como _pwr nas 4 leituras.

## 2026-11-02: whitening do vetor (com shrinkage) é o maior ganho isolado em eggs
- gem_tta_pwr (gem_tta + PCA-whitening com shrinkage + L2, PCA fitado no treino), bp01w64c1280, κ val L4:
  - eggs 0.946/0.942/0.938 (0.942), +0.030 sobre gem_tta; 1º eggs da val acima do I-JEPA (0.940, teste).
  - larvae 0.958/0.944/0.980 (0.961), +0.015.
  - protozoan pendente.
- Só pela leitura, com o mesmo encoder: eggs 0.880 (avg) → 0.891 (gem) → 0.912 (tta) → 0.942 (tta + whitening).
- Hipótese: os 1280 canais (400 SPiFiL + 880 aleatórios) são muito correlacionados, e o SVM linear via só as direções de maior variância. O whitening iguala as direções, e as raras (as que separam ovo × impureza) passam a contar.
- Enfileirado: o teste do bp01w64c1280, o e_bp10 (val e teste), e o e_aug/l_aug (teste) com gem_tta_pwr.

## 2026-11-02: whitening e TTA são aditivos e cobrem erros diferentes
- Ganho sobre gem, κ val (pwr / TTA / os dois): eggs +0.029/+0.013/+0.043; larvae +0.010/+0.036/+0.051; protozoan ~+0.008/+0.043/+0.052 (s3 do pwr pendente). gem_pwr de eggs sozinho dá 0.928.
- Leitura: em eggs o whitening domina (detalhe fino ovo × impureza); em larvae/protozoan domina o TTA (objetos sem orientação canônica). Receita de leitura: gem + TTA + whitening com shrinkage.

## 2026-11-02: lei de escala lançada (autorizada pelo usuário)
- Backbone: disc_bp01w64c1280 sem aug. Orçamento (gate e60): eggs 200 ép. (ref m=1 = disc_bp01w64c1280); larvae+protozoan 60 ép. (ref m=1 = disc_g20_base_e60). YAMLs separados: `growth_disc_sl_<eixo><valor>_{e200,e60}.yaml`.
- Leitura: gem_tta_pw (gem + TTA 8 vistas + PCA-whitening com shrinkage), na última camada. Por família: val gem, val gemttapwr, test gemttapwr (teste só reporta). Ref e60 ganhou val/test gemttapwr.
- Eixos, na ordem da fila: largura m ∈ {1/2, 1/4, 1/8, 2}; largura da 1x1 ∈ {160, 320, 640, 1920}; profundidade d3 (2 SPiFiL + 1x1, pools 2/2/1; e200 e e60) e d5 (4 SPiFiL + 1x1, só e200; e60 já é e_d5_e60).
- m=2 daria 1x1 2560 > 2000-d: limitada a 1920 (anotado).
- N esperado (convs com bias, entrada 3 canais; a conferir no architecture.json); fração SPiFiL da 1x1 em eggs (400 sementes):

| família | larguras | N | SPiFiL 1x1 eggs |
|---|---|---|---|
| w0125 | 4/6/8/160 | 3 558 | 100% |
| w025 | 8/12/16/320 | 13 276 | 100% |
| w050 | 16/24/32/640 | 51 192 | 62% |
| ref m=1 | 32/48/64/1280 | 200 944 | 31% |
| w200 | 64/96/128/1920 | 713 568 | 21% |
| c160 | 32/48/64/160 | 128 144 | 100% |
| c320 | 32/48/64/320 | 138 544 | 100% |
| c640 | 32/48/64/640 | 159 344 | 62% |
| c1920 | 32/48/64/1920 | 242 544 | 21% |
| d3 | 32/48/1280 | 103 600 | 31% |
| d5 | 32/48/64/96/1280 | 395 600 | 31% |

- Hipótese SPiFiL: a fração SPiFiL cresce quando o modelo encolhe; os modelos pequenos são "mais SPiFiL". Se o κ cair pouco nos pequenos, parte disso é o SPiFiL ocupando a 1x1 inteira, não só o N; reportar e controlar pela fração.

## 2026-11-02: com whitening, o aug para de somar em eggs; receita de eggs = bp10 + gem_tta_pw
- e_bp10 + gem_tta_pwr: κ val 0.951/0.948/0.947 (0.949), +0.007 sobre a base com pwr; o mais estável.
- e_aug + gem_tta_pwr: 0.938/0.939/0.936 (0.937), contra a base com pwr 0.942 (empate). Com o whitening, o aug deixa de somar (no gem_tta ele dava +0.008).
- Receita de eggs: bp 1.0 + leitura final, sem aug. O e_bp10_aug segue rodando para confirmar.

## 2026-11-02: lei do backprop em larvae tem pico em 0.3
- l_bp03 (200 ép.): gem L4 κ val 0.939/0.928/0.934 (0.933), +0.023 sobre a base, o maior larvae com gem e acima da variação do pipeline. Lei em larvae: bp 0.1 0.910 → 0.3 0.933 → 1.0 0.914 (pico em 0.3); em eggs sobe até 1.0.
- Enfileirado gem_tta_pw (val e teste) do l_bp03 e do p_bp03_e60.

## 2026-11-02: gate das 60 épocas com a leitura final: só larvae aceita
- base_e60 gem_tta_pw κ val: larvae 0.956 contra 0.961 (−0.005, ACEITA); protozoan 0.914/0.906/0.902 (0.907) contra 0.926 (−0.019, REJEITA na borda); eggs pendente (com gem já rejeitava).
- Regra: 60 épocas só em larvae. Os pontos _e60 da lei de escala de protozoan valem para a FORMA da lei (comparados com base_e60 no mesmo orçamento), não para o nível absoluto.

## 2026-11-02: a leitura final absorve os ganhos de treino
- l_bp03 + gem_tta_pwr: κ val 0.958/0.925/0.969 (0.951), contra a base com pwr 0.961 (empate). O +0.023 do bp03 no gem some com o whitening.
- Com gem_tta_pw, aug (eggs/larvae) e bp03 (larvae) empatam; só o bp10 em eggs sobrevive (+0.007). A leitura forte absorve o que o treino ajustava. Receita final: base bp01w64c1280 + gem_tta_pw nos 3; em eggs, bp 1.0.
- bg1_e60 (fundo peso 1): eggs gem 0.873 contra base_e60 0.875 (empate). A receita sem o fg do fundo não perde em eggs; faltam larvae (s1 0.920) e protozoan.

## 2026-11-02: a receita sem fundo empata (bg1 e nomask)
- 60 ép., gem L4 κ val contra base_e60 (0.875/0.903/0.861):
  - bg1 (fundo peso 1): 0.873/0.900/0.856, empate nos 3;
  - nomask (SPiFiL sem máscara, impurities true): eggs 0.882, larvae 0.903 (muito variável), empate; protozoan pendente.
- Leitura: o peso do fundo (0/0.1/1) e a máscara de objeto do SPiFiL não mudam o κ. A receita roda em dataset sem fundo/sem máscara sem custo medido. Receita sem fundo: impurities true + paper_classes false + recon_bg_weight 1.0 + gem_tta_pw (a combinação ainda não foi medida junta).

## 2026-11-02: validação do backbone nos 3 datasets (Jaci)
- Pedido do usuário: validar as leis e se o backbone principal funciona nos 3 datasets.
- disc_final_rep: bp01w64c1280 repetido (2º sorteio do pipeline), 3 datasets, 200 ép., eval gem e gem_tta_pw (val e teste). Mede se o resultado de teste (0.933/0.968/0.928) se repete.
- disc_final_nobg: receita sem fundo completa (impurities true + paper_classes false + recon_bg_weight 1.0), 3 datasets, 200 ép. As 3 mudanças só tinham sido medidas sozinhas (todas empatam a 60 ép.).
- Rodam em Jaci (max_per_gpu 2, ~7 GB cada), junto com a lei de escala (disc_sl_*).

## 2026-11-02: branqueamento dos mapas no treino (disc_nd_e200, Athena)
- Mudança: `feat_whiten: true` sobre bp01w64c1280, 3 datasets, 200 ép. ZCA de canal (Network Deconvolution, Ye et al. 2020, só canal, sem im2col; DBN, Huang et al. 2018) no fim de cada bloco = entrada da conv seguinte. A entrada da conv1 (LAB) fica crua: no layerwise o SPiFiL da conv1 recorta o LAB cru. W sem gradiente (backward do eigh explode).
- Hipótese SPiFiL: descorrelacionar os canais antes de cada conv faz a seleção SPiFiL e o backprop verem features não redundantes, o que o whitening do vetor final fazia na leitura.
- Base de referência 200 ép.: gem 0.899/0.910/0.875, gem_tta_pw 0.942/0.961/0.926.

## 2026-11-02: curva 1-100% (teste, leitura final) e a hipótese do whitening refutada
- gem_tta_pw (teste, média dos splits): eggs 0.268/0.676/0.889/0.933/0.960/0.967; larvae 0.606/0.821/0.930/0.968/0.979/0.981; protozoan 0.486/0.728/0.898/0.928/0.942/0.954 (pct 1/5/25/50/75/100; protozoan 100% sem o s3).
- I-JEPA (teste): eggs 0.661/0.864/0.917/0.940/0.945/0.957; larvae 0.822/0.881/0.919/0.943/0.941/0.950; protozoan 0.586/0.725/0.831/0.864/0.882/0.892.
- Passamos o I-JEPA com >=25% em larvae/protozoan e com >=75% em eggs; perdemos com 1-5%.
- Hipótese "o whitening colapsa com poucas amostras e causa a perda em 1%": REFUTADA. gem_tta sem whitening é pior em todos os pontos (Δ −0.004 a −0.082; parcial, protozoan com 2 splits). A perda em 1-5% vem do encoder com poucos dados, não da leitura.

## 2026-11-02: eggs empata com o I-JEPA no teste (e_bp10 + leitura final)
- e_bp10 (bp 1.0) + gem_tta_pw, teste 50% (Jaci): 0.9434/0.9396/0.9401 (0.941) contra I-JEPA 0.940. Empate (+0.001). Val era 0.949.
- Com isso, no teste a 50%: eggs 0.941 (empate), larvae 0.968 (+0.025), protozoan 0.928 (+0.064). O objetivo original (κ >= I-JEPA nos 3 a 50%) está atingido.
