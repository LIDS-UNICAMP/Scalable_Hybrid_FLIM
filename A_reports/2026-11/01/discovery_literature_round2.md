# [A5b] Literatura, rodada 2: parar cedo e ideias novas (2026-11-02)

## A resposta, em uma frase

**Parar cedo:** meça o "posto efetivo" (RankMe) dos embeddings a cada época e pare a camada quando ele parar de subir. Na grade, corte as configs ruins nas fronteiras de camada (successive halving). **Platô:** as ideias mais baratas são só de eval: TTA com as 8 rotações/flips e testar numa resolução maior. Depois vem a média dos pesos (EMA/SWA), que é nativa do Lightning, e por último o whitening dos patches antes do SPiFiL.

Onde estamos: o protozoan **já passou** do alvo (0,875 contra 0,86). Faltam ~0,04 em ovos e ~0,03 em larvas. Muitas mudanças já caíram dentro do ruído, então qualquer ganho abaixo de ~0,01 vai precisar de várias seeds/splits para ser acreditado.

Já estavam na rodada 1 (`discovery_literature_ideas.md`), então ficam fora daqui: SPP, L2/raiz com sinal, GeM, concatenar camadas, pooling de covariância, scaler/whitening **no SVM**, AE mascarado (SparK/FCMAE), SIGReg e VICRegL.

---

## (1) Critérios rápidos de parada

### Ideia em palavras simples
Treinar uma camada é como encher um balde. Não precisamos esperar a água parar de pingar (a loss parar de cair). Basta ver que o nível útil (a diversidade dos embeddings) parou de subir. A nossa memória confirma o problema: em 12 de 34 runs de larvae, o κ teve o pico na época ≤2, mas o EarlyStopping olhava a `val/loss`, que continuou caindo depois disso. Ou seja, a loss é o termômetro errado.

### Comparação

| Critério | O que mede | Custo por checagem | Encaixe no Lightning | Serve para nós? |
|---|---|---|---|---|
| **RankMe** (Garrido et al., ICML 2023, https://arxiv.org/abs/2210.02885) | Quantas direções o embedding usa de verdade: `exp(entropia(σ/‖σ‖₁))` dos valores singulares da matriz N×1280. Ele cai quando há colapso e sobe quando a representação fica rica. Os autores mostram que escolher hiperparâmetros por ele perde quase nada contra escolher com rótulo. | 1 `torch.linalg.svdvals` de ~2048×1280: menos de 1 s na GPU | Callback que loga `val/rankme` → `EarlyStopping(monitor=...)` | **Sim (1ª escolha).** Sem rótulo, barato. Ressalva: foi validado em joint-embedding, não em AE, então é preciso calibrar (ver abaixo). |
| **α-ReQ** (Agrawal et al., NeurIPS 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract.html) | A inclinação α da lei de potência dos autovalores da covariância (λᵢ ∝ i^−α). Há faixas de α (perto de 1) com boa leitura linear. É condição necessária, não suficiente. | Mesma SVD, de graça junto com o RankMe | Logar `val/alpha` no mesmo callback | Como 2º sinal ou como alarme (α muito alto = colapso). Não use sozinho para parar. |
| **LiDAR** (Thilak et al., ICLR 2024, https://arxiv.org/abs/2312.04000) | O posto da matriz LDA "imagem vs. suas versões aumentadas". Bate o RankMe na escolha de hiperparâmetros (testado em I-JEPA, VICReg...). | Precisa de K vistas por imagem (dá para usar as 8 flips/rot90): ~8× mais forwards | Callback igual, com as vistas | Bom, mas mais caro. Fica como plano B se o RankMe não correlacionar. |
| **Platô da loss** | A loss parou de cair | Zero | `EarlyStopping(monitor="val/log_loss", min_delta=0.005, patience=3)`. Logar log(loss) faz o `min_delta`, que é absoluto, virar ~0,5% relativo. | Só como rede de segurança: já vimos que ela anda descolada do κ. |
| **Successive halving / ASHA / Hyperband** (Li et al. 2018, https://arxiv.org/abs/1810.05934; Hyperband https://arxiv.org/abs/1603.06560) | Roda todas as configs com orçamento pequeno, mantém o melhor 1/η e dá mais orçamento só para elas | O custo do proxy em cada degrau | Ray Tune `ASHAScheduler(time_attr=..., grace_period, reduction_factor=3)`, ou na mão nas fronteiras de camada | **Sim (2ª escolha), para a grade.** O growth já tem degraus naturais: as camadas. |
| **Extrapolar a curva** (LC-PFN, Adriaensen et al., NeurIPS 2023, https://arxiv.org/abs/2310.20447; Domhan et al., IJCAI 2015) | Prevê o valor final a partir do começo da curva. Para quando P(passar o melhor) < 5%. Relatam 2–6× de aceleração. | ms (`pip install lcpfn`) | Callback que chama o LC-PFN no histórico | Não agora. Com poucas épocas e κ ruidoso, a previsão fica fraca. |
| **Orçamento fixo por camada (greedy)** (Belilovsky et al., ICML 2019, https://arxiv.org/abs/1812.11446) | Cada camada resolve um problema auxiliar curto e é congelada. Isso escala até ImageNet. | Zero | `max_epochs` por camada no YAML | Sim, como experimento barato: `max_epochs ∈ {2, 5, atual}`. O pico na época ≤2 sugere que 2–5 já basta. |

### Recomendação concreta (as 2 melhores)

**R1: parada por camada com RankMe (sem rótulo).**
- A cada época de validação, pegue até 2048 imagens fixas (treino+val, nunca teste) e passe pelo encoder até a camada atual com o mesmo pooling da leitura (GeM/avg → ≤1280-d). Calcule `rankme` e `alpha`.
- Atenção: o RankMe tem teto em min(N, D). Com N < 1280 ele satura, por isso as ~2048 imagens.
- **Regra:** `min_epochs=2`. Pare a camada quando o `log(rankme)` não subir ≥0,01 (~1%) por **2 épocas seguidas**. Pare também se o RankMe cair >10% abaixo do pico (colapso). Salve o checkpoint pelo **máximo de `val/rankme`**, não pela `val/loss`. Mantenha a `val/log_loss` com `patience=3` como segurança: no Lightning, dois `EarlyStopping` funcionam como "OU", o que dá a parada mais rápida.
- **Calibração antes de adotar (1 run só):** num growth de larvae, salve o ckpt de **cada época** de 2 camadas. Depois do treino, rode o SVM em cada ckpt, um por vez (regra "SVM só após o treino"), e meça o Spearman ρ entre RankMe e κ. Com ρ ≥ 0,6 a regra entra no YAML. Se der menos, troque para LiDAR com as 8 vistas.

```python
class RankMe(pl.Callback):            # loga val/rankme e val/alpha
    def __init__(self, loader): self.loader = loader
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, m):
        z = torch.cat([m.embed(x.to(m.device)) for x, *_ in self.loader]).float()
        s = torch.linalg.svdvals(z - z.mean(0))
        p = s / s.sum() + 1e-7
        m.log("val/log_rankme", -(p * p.log()).sum())           # log do RankMe
        lam = (s ** 2)[5:len(s) // 2]                             # faixa do meio do espectro
        x = torch.arange(6, 6 + len(lam), device=s.device, dtype=s.dtype).log()
        A = torch.stack([x, torch.ones_like(x)], 1)
        m.log("val/alpha", -torch.linalg.lstsq(A, lam.log()[:, None]).solution[0, 0])
# EarlyStopping(monitor="val/log_rankme", mode="max", min_delta=0.01, patience=2)
# EarlyStopping(monitor="val/log_loss",   mode="min", min_delta=0.005, patience=3)
# ModelCheckpoint(monitor="val/log_rankme", mode="max")
```
(`m.embed` = forward até a camada atual + pooling da leitura. As chaves `early_stop: rankme`, `es_patience`, `es_min_delta` entram em `RUNNER_ARGS` do schema, sem flag de argparse. O RankMe original não centraliza z; centralizar muda pouco e evita que a média vire uma "direção".)

**R2: successive halving nas fronteiras de camada, para a grade.**
- Todas as configs da grade crescem até a **camada 3**. Ordene pelo proxy e fique com o melhor **1/3**. Cresça até a camada 5, fique de novo com 1/3 e só essas vão até a 8. Isso é η=3, com o 1º degrau na camada 3.
- O proxy pode ser o RankMe (sem rótulo) ou o κ do SVM em val. Se for o SVM, rode um job por vez no degrau, nunca em paralelo, por causa da CPU.
- Economia: com 27 configs e 8 camadas, o custo cai de 27×8=216 para 27×3 + 9×2 + 3×3 = 108 camadas-treino, ~2× mais barato. Ganha mais ainda com R1 junto.
- Risco: uma config que só melhora tarde morre cedo. Mitigação: `grace_period` = 3 camadas, nunca menos.

---

## (2) Ideias novas para quebrar o platô, ranqueadas por ganho/custo

| # | Ideia | Por quê, em 1 linha | Referência | Mudança mínima (1 botão) | Custo |
|---|---|---|---|---|---|
| 1 | **TTA diedral no SVM:** média do embedding sobre as 8 rotações/flips. Variante: as 8 vistas como amostras extras de treino do SVM | Ovos, larvas e cistos não têm "lado de cima", então tirar a média das 8 vistas apaga o ruído de orientação de graça | Shanmugam et al., *Better Aggregation in Test-Time Augmentation*, ICCV 2021, https://arxiv.org/abs/2011.11156 | eval YAML: `tta: d4` (média de 8 forwards antes do pool→SVM) | só eval, ~8× forward; sem retreino |
| 2 | **Resolução maior no teste** (FixRes): extrair as features com a entrada 1,15–1,3× maior | Com o crop/aug de treino, os objetos ficam "maiores" no treino que no teste, e subir a resolução do teste corrige isso. Com GeM/avg global, a rede aceita qualquer tamanho | Touvron et al., *Fixing the train-test resolution discrepancy*, NeurIPS 2019, https://arxiv.org/abs/1906.06423 | eval YAML: `test_scale: [1.0, 1.15, 1.3]` e escolher pela val | só eval |
| 3 | **Média dos pesos (EMA/SWA) dentro da camada** | O κ oscila e tem o pico cedo. A média dos pesos das últimas épocas cai num ponto mais plano e menos ruidoso, e é nativa (`StochasticWeightAveraging`, `torch.optim.swa_utils.AveragedModel`) | Izmailov et al., *Averaging Weights Leads to Wider Optima*, UAI 2018, https://arxiv.org/abs/1803.05407 | `swa_lrs` / `ema_decay: 0.999` em `runner_args` (callback nativo do Lightning) | baixo |
| 4 | **Whitening ZCA dos patches antes do SPiFiL recortar os filtros** | Sem whitening, os filtros recortados copiam a cor/brilho dominante (o fundo). Com ZCA, eles viram bordas e texturas, e é o ingrediente-chave nos métodos de patches | Thiry, Arbel, Belilovsky, Oyallon, *The Unreasonable Effectiveness of Patches in Deep Convolutional Kernels Methods*, ICLR 2021, https://arxiv.org/abs/2101.07528 | `spifil_whiten_eps: 1e-3` (estimar média/cov dos patches da camada e aplicar W=(C+εI)^−½ nos patches **e** na entrada da conv, ou dobrar W no kernel) | médio-baixo, só na init |
| 5 | **k-means sobre o pool de patches whitened para escolher os filtros** | Em vez de 1 filtro por superpixel, que repete o padrão comum, o k-means força diversidade: os k filtros cobrem o espaço de patches | Coates & Ng, *An Analysis of Single-Layer Networks in Unsupervised Feature Learning*, AISTATS 2011, https://proceedings.mlr.press/v15/coates11a.html; Thiry et al. 2021 (acima) | `spifil_select: kmeans` (sortear 10× mais patches candidatos e usar os k centróides normalizados como kernels) | médio-baixo, só na init |
| 6 | **Filtros PCA misturados aos SPiFiL** | Os primeiros componentes principais dos patches são a base ótima para reconstruir, que é justamente o objetivo do AE. Pôr ~10–25% dos kernels como PCA garante cobertura e o resto continua data-driven | Chan et al., *PCANet: A Simple Deep Learning Baseline for Image Classification?*, IEEE TIP 2015, https://arxiv.org/abs/1404.3606 | `pca_filters_frac: 0.25` | baixo, só na init |
| 7 | **Regularização de ortogonalidade nos kernels durante o fine-tune** | Com backprop, os filtros SPiFiL tendem a ficar parecidos uns com os outros (o posto cai). Um termo ‖WᵀW−I‖ mantém a diversidade, conversa com o RankMe e ajuda com pouco dado | Bansal, Chen, Wang, *Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?*, NeurIPS 2018, https://arxiv.org/abs/1810.09102 | `ortho_lambda: 1e-4` (soma na loss do training_step) | baixo |
| 8 | **Objetivo auxiliar local mais forte por camada** (decoder/cabeça auxiliar com 2–3 camadas, ou loss local contrastiva com sobreposição) | No treino greedy, a qualidade de cada camada depende da força do problema auxiliar. Belilovsky viu que aux de 2–3 camadas gera camadas melhores. LoCo/GIM trocam a reconstrução por InfoNCE local. O nosso backprop ×0,1 nas anteriores já é a "sobreposição" do LoCo | Belilovsky et al., ICML 2019, https://arxiv.org/abs/1812.11446; Xiong, Ren, Urtasun, *LoCo*, NeurIPS 2020, https://arxiv.org/abs/2008.01342; Löwe et al., *Greedy InfoMax*, NeurIPS 2019, https://arxiv.org/abs/1905.11786 | `aux_depth: 2` (decoder do estágio com 2 convs em vez de 1). A troca para InfoNCE local é o passo caro, e não é 1 botão | médio (aux) / alto (InfoNCE) |

### Por que essa ordem
- **1–2** não treinam nada. Rodam sobre os checkpoints *best* que já existem, num job de eval sequencial, e dão resposta em horas. São o melhor ganho/custo, mesmo que o ganho seja pequeno. Ressalva da 1: o treino já teve flip/rot90, então as features podem já ser quase invariantes e o TTA render ~0,005–0,01. A gente só sabe testando.
- **3** ataca o problema que já vimos (κ ruidoso com pico cedo) e é feature nativa do Lightning.
- **4–6** mexem só na init do SPiFiL, que é o diferencial do método. Teste a 4 sozinha primeiro, porque a 5 depende dela.
- **7** é barata mas depende do λ. Faça depois da R1 (RankMe), que mostra na hora se o posto parou de cair.
- **8** é a mais cara e a mais incerta. Entra só se 1–7 não renderem.

### Ordem sugerida
1. Um job de eval com `tta: d4` e `test_scale` sobre os ckpts best (ovos e larvas), um modo por vez.
2. Uma run de calibração da R1 (RankMe vs κ por época).
3. A grade seguinte (ideias 3, 4, 6, 7 como eixos) já com R1 + R2 ligados, porque é isso que torna a grade barata.

Fontes extras consultadas: Ray Tune ASHA (parâmetros `grace_period`/`reduction_factor`); LiDAR, ICLR 2024 (https://iclr.cc/virtual/2024/poster/18199); RankMe, PMLR (https://proceedings.mlr.press/v202/garrido23a.html); SparK (https://arxiv.org/abs/2301.03580) e ConvNeXt V2 (https://arxiv.org/abs/2301.00808), que já estavam na rodada 1.
