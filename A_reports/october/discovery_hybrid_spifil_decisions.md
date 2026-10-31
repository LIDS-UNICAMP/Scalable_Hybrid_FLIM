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
