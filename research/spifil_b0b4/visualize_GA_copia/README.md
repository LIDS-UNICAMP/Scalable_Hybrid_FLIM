# visualize_GA — gráficos do GA (K1–K6)

**Rodar tudo** (reconstrói `_parts/`, junta em `models.csv`, gera as figuras de cada base):

```
E=/dados/home/moliveira/miniforge3/envs/scalable_FLIM; LD_LIBRARY_PATH=$E/lib $E/bin/python plot_ga.py
```

Uma base só: `... $E/bin/python <base>/plot.py` (mesmo caminho, só a base da pasta; roda de qualquer diretório).
Sem o `LD_LIBRARY_PATH` o PIL falha com `GLIBCXX_3.4.29`. Leva ~2,5 min (≈30 s dos `build_*.py`, o resto é sobretudo o MDS do K3).

Tudo vive em `plot_ga.py`, em seções: dados/merge, estilo, K1, K2, K3, K4, K5. As bases ficam na constante `DATASETS` no topo:
flowers, remote_sensing, ham10000, aid, coconut_trees e os parasitos eggs, larvae, protozoan (sem fundo) e
eggs_raw, larvae_raw, protozoan_raw (com fundo). Uma pasta por base, cada uma com o seu `plot.py`.

**Parasitos, com e sem fundo.** eggs/larvae/protozoan usam as imagens de fundo preto (pre/orig) das buscas
(`/exps/moliveira/results/benchmark/modehb_pre/run_modehb_pre.py`); os `_raw` usam o fundo original.
O título diz isso: "eggs (sem fundo)", "eggs_raw (com fundo)".

**Três versões de cada figura**, na mesma pasta: `<nome>.png` (leitura melhorada, regra abaixo), `<nome>_semTTA.png`
(nossos em `so2_4096_pca`, AE outubro no GAP `oct_gap`) e `<nome>_comTTA.png` (nossos em `so2_4096_pca_tta8`, AE outubro em `oct_gemttapwr`).
Nas versões sem/com TTA, quem não tem aquela leitura cai em `gap1280`: ponto apagado, fora do "melhor nosso". FMs e aleatório são iguais nas três.
O título diz a versão. O "melhor nosso" é sempre pelo maior κ de TESTE.

## Dados (comum a todas as figuras)

- `_parts/build_*.py` geram `_parts/<x>.csv`; `plot_ga.py` junta em `models.csv` e `noncomparable.csv`. A escrita é atômica (`.tmp` + `os.replace`): quem lê ao mesmo tempo nunca pega arquivo vazio. O `models.csv` é lido uma vez só por execução.
- Só 100% do treino e só κ de **teste** (SVM C=100). κ de val nunca entra.
- **Leitura (head) de cada modelo = a melhorada**, nesta prioridade: `so2_4096_pca_tta8` (2ª ordem + TTA8, ponto cheio) >
  `so2_4096_pca` (2ª ordem, só contorno) > `gap1280` (só GAP: ponto bem apagado, e fica fora da disputa do "melhor nosso").
  FMs: leitura padrão do benchmark, sem TTA, no mesmo split da busca do GA (`modehb_pre` nos parasitos sem fundo,
  `modehb_raw` nos `_raw` = split1_modehb, `fm_default` nas outras bases). Aleatório: `receita_L6_cat` (sem TTA, regra do projeto).
  AE outubro dos parasitos: só GeM+TTA+pw (`oct_gemttapwr`). Duplicatas de split viram média; depois média e desvio entre splits.
- Ficam fora: `*_subgroup`, `helminth_*`, `protozoan_cysts`, pct < 100, heads `xb_*`.
- **FMs**: estrelas com nome (AIMv2, DINOv2, I-JEPA, ConvNeXtV2, ResNet-50).
- **Aleatório puro** (family `aleatório puro (random_z)`, `_parts/random.csv`): a receita SPiFiL (6 camadas × 256 filtros, kernel 5) com os filtros sorteados, sem saliência, SoftHebb nem seleção, lida na L6 (head `receita_L6_cat`, sem TTA, como os nossos). X preto, com o κ na legenda. O CSV não traz params: x = 8,2e6, o valor do comentário "receita 8,2 M (6x256 k5)" em `Scalable_Hybrid_FLIM_discovery/experiments/autoencoder/disc_md_teste_aid_receita_tta_athena.yaml:1` (o "8,2 mi" do slide 18). Só existe em ham10000 e aid; nas outras bases simplesmente não aparece.
- **AE outubro** (family "AE outubro (SPiFiL camada a camada + AE)"): nos parasitos sem fundo vem de `_parts/build_oct.py`
  (GeM+TTA+pw: `Scalable_Hybrid_FLIM/results/svm_disc_bp01w64c1280_pcts_test_gemttapwr.csv`; GAP: `..._pcts_test_avg.csv`;
  só a L4, 100%, splits 1–3; protozoan GeM só tem 2). O GAP da L4 dá 0,918/0,961/0,875 (eggs/larvae/protozoan); os 0,905/0,957/0,843
  citados antes saem da média das linhas L3 e L4 juntas, que são duas leituras diferentes.
  O split é o `new_split_parasito`, com val = treino: só o κ de teste vale. params 200.944 (188.468 no protozoan),
  `A_reports/october/report_results_only_spifil_kappa.md:29`. Os HYB_oct_* dos parasitos saem do `growth.csv` (não contar duas vezes);
  em flowers/remote_sensing ficam os OCT_* do `growth.csv`. Sempre losango grande de borda preta, κ com setinha, entrada própria na legenda;
  é referência: sem aresta no K2, fora do "melhor nosso"; no K4/faixas entra em 100%; no K5 no grupo "AE outubro".
- **Melhor nosso**: seta no modelo nosso de maior κ de teste; embaixo, em letra menor, o nº de parâmetros (coluna `params`), ex. "(23,7 mil parâmetros)", ou "(parâmetros: sem dado)".
- Todas as figuras: grade fina cinza-clara atrás dos pontos (κ a cada 0,1; x nos ticks principais) e legenda logo abaixo do quadro, com "família, n=…" (n = nº de modelos no gráfico, explicado no título da legenda).

## K1 — `<base>/ga_kappa_params_by_method.png`

κ × nº de parâmetros (log), cor = família, barra = desvio entre splits. Eixo x comum às 5 bases.
Modelos sem `params` não cabem no eixo log e ficam fora (todos os SSL, os AE outubro de flowers/remote_sensing e alguns de crescimento em aid); o script imprime a contagem.

## K2 — `<base>/ga_containment_graph.png`

Linhagens pai→filho no plano params × κ. Promoções de fidelidade (f5→f15→…) colapsam: cada candidato é um ponto, o da maior fidelidade com κ e params.
A seta pai→filho só existe quando a coluna `parent` do filho é o pai (vinda de `growth_state`, `inherit.parent_cand_id`, `resumed_from` ou `parent_id`) e os dois têm κ e params.
Linha cheia = crescimento B0→B4, tracejada = herança, pontilhada = mutação; cor = família do filho; na legenda das linhas, n = nº de ligações.
Pontos em linhagem são maiores; o preenchimento segue a leitura (cheio/contorno/apagado). `edges.csv` guarda todas as arestas (inclusive promoções).
Em coconut_trees não há aresta desenhável (as 4 saem de `seed01_f0`, sem κ de teste) e a figura diz isso.

## K3 — `<base>/ga_kappa_params_similarity.png`

κ × params, cor = parentesco da config: mesma cor = config parecida.
1. Vetor de config: união das chaves do `config_vector` (ausente = 0; número vira número; texto vira one-hot; chaves de rastreio fora).
2. z-score por chave; sim = (cos + 1)/2.
3. 1 − sim → MDS 1-D (`random_state=0`) → 0–1 → colormap cíclico `twilight` (as duas pontas têm cor parecida).
4. O MDS é por base: a mesma cor em bases diferentes não quer dizer a mesma config.
`x` cinza = modelo nosso sem `config_vector`. O aleatório fica fora do MDS.
`similarity.csv` (dataset, model_a, model_b, sim; triângulo de cima): rodar uma base só troca apenas as linhas dela.
A seta "melhor nosso" só aponta modelo com params (em aid, o melhor sem params é `jaci_v3ae_aid_B1g0c02_f15`, κ=0,741).

## K4 — `<base>/ga_kappa_spifil_fraction.png` e `..._faixas.png`

- Figura 1: x = % do modelo que é SPiFiL (em 0% e 100% os pontos são espalhados ±1,5 só para não se sobreporem).
- Figura 2 (faixas): 0%, 1–25%, 25–50%, 50–75%, 75–99%, 100%; cada faixa dividida por família, com boxplot claro. Uma chave sob o eixo mostra quais grupos são de cada faixa; linhas verticais finas separam as faixas; "n=" sobre cada grupo = nº de modelos no grupo. O título da legenda explica que, dentro da faixa, os pontos só foram espalhados para não ficarem um em cima do outro.
- A legenda usa o máximo de colunas (3, 2 ou 1) que cabe na largura da figura.
- Linhas sem `spifil_frac` ficam fora (o script imprime a contagem).
- `spifil_frac=1.0` do GA vale para o encoder; a cabeça 1×1 aleatória não foi descontada.

## K5 — `<base>/ga_kappa_ssl_tecnica.png`

κ de teste por técnica de treino com backprop, no mesmo estilo da figura de faixas (FMs e aleatório à esquerda).
**Cor = épocas de treino com backprop** (viridis, escala log, colorbar abaixo do quadro). Branco = 0 épocas; cinza = sem registro.

| família | técnica (eixo x) | épocas (cor) |
|---|---|---|
| GA híbrido, MO-DEHB base, LEMONADE, v3/v4, PNAS-CS | campo `ssl` do genoma, decodificado com `CATS["ssl"] = [lejepa, ae, simclr]` de `_parts/build_ga.py`/`build_pnas.py`. O GA sempre treina AE local por camada antes do SSL global (`Scalable_Hybrid_FLIM/ga_search/modehb/run_search.py`): "AE+LeJEPA", "AE+AE", "AE+SimCLR" | `fidelity` = épocas deste candidato (AE local + SSL global; na promoção f5→f15 o total acumulado). Não conta o que um filho LEMONADE herdou do pai |
| PNAS-CS fase 0, variante `noae` (lida do `search_state.json`, porque o vetor não guarda) | "LeJEPA (sem AE local)" | `fidelity` |
| crescimento B0–B4 | pelo nome do run, que é o braço do `growth_state.json` ("ssl": none/lejepa): `nossl` → "AE local, sem SSL"; `ae_so2` e `jaci_v3ae_*` → "AE local (so2)"; `superpixels_prune` → "AE+LeJEPA (superpixels_prune)"; `fix4`/`gpu2` → "AE+LeJEPA"; semente f0 (`fid`=0) → "SPiFiL (sem backprop)" | AE+LeJEPA: `ae_epochs + ssl_epochs`; braços sem SSL: só `ae_epochs`; f0: 0 |
| SSL pequeno (sslp_*) | "SPiFiL pequeno + LeJEPA/SimCLR/Barlow/DINO/CosFace" (pela family) | sem registro |
| AE outubro | "AE outubro (SPiFiL camada a camada + AE)" | sem registro |
| SPiFiL (sem backprop), inclusive `__spifil_init` | "SPiFiL (sem backprop)" | 0 |

## K6 — `<base>/ga_curva_pct.png`

κ de **teste** × % do treino, uma figura por base, só na versão melhorada. Estilo dos plots de outubro do Scalable_Hybrid_FLIM
(`A_reports/october/plots_pct_vs_ijepa/merge_plots/kappa.png`, feito por `src/evaluate/eval_plotter.py` + `configs/eval/plots.yaml`):
eixo x categórico só com os % presentes (1, 5, 10, 25, 50, 75, 100), fundo cinza-claro com grade, título em negrito,
eixo y "Cohen's Kappa (κ)", legenda embaixo do gráfico com os parâmetros entre parênteses (ex.: "I-JEPA (1,01B)").
Ponto = média entre splits (duplicatas de split viram média antes); **barra com tampinha = ± desvio**, só onde há mais de um split.

- **FMs** (traço-ponto, uma cor e um marcador por modelo: AIMv2 azul, DINOv2 roxo, I-JEPA laranja, ResNet-50 azul-claro,
  ConvNeXtV2 rosa): curvas do benchmark no `models.csv`, family FM, head `fm_default`, 3 splits → com barra.
  FMs de `benchmark/all11` (11 bases): nos `_raw`, `fm_default` = descritor incremental real. Params do `models.csv`.
- **Melhor nosso** (vermelho, ●) e o **SPiFiL sem backprop** (verde, ■), linha cheia: ids
  `curva_<base>_<best|spifil_sem_bp>_p<pct>` em
  `/dados/home/moliveira/exps/jaci_results_mirror/readout_x_results/curvas_pct.jsonl` e
  `/dados/home/moliveira/Scalable_Hybrid_FLIM/research/spifil_b0b4/readout_x/results/curvas_pct.jsonl` (lidos se existirem; o que faltar não é desenhado).
  **Só split1** (os splits 2 e 3 vazam o treino SSL): sem barra, e a legenda diz "n=1 (split1)" — nunca se inventa desvio.
  Leitura **ponto a ponto** (id, pct, split), em todas as bases: a de maior κ de **val** entre `so2_4096_pca`, `so2_4096_pca_tta8`
  e `so2_4096_pca_tta2h` (se existir); desenha o κ de teste dessa leitura (pedido explícito: é a única escolha pela val).
  Sem κ de val, ou com `val_is_train=true`: `so2_4096_pca` (2ª ordem SEM TTA: a TTA8 quebra protozoan e os FMs não usam TTA),
  contado como "sem val"; TTA8 só se não houver a 2ª ordem. A 2ª linha do rótulo diz quem venceu,
  ex. "leitura escolhida pela val: 2ª ordem em 7/7 pontos". Id sem nenhuma dessas três leituras: o head que vier
  (ex.: `gem_tta_pw` = GeM+TTA+pw).
  Família = campo `family` do jsonl; se faltar, a do modelo do `ckpt` no `models.csv` (`<run>/<base>/<cand>/f<fid>`, ou
  `<base>_growth-<host>-<run>_<nó>_f<fid>` para nós de crescimento); senão, a do melhor nosso a 100%.
  Params: campo `params` do jsonl; senão, o do modelo do `ckpt` no `models.csv`; o SPiFiL sem backprop com os mesmos `blocks_channels`
  do melhor herda os params dele; sem nenhum desses, o rótulo fica sem parênteses.
- Parasitos sem fundo (eggs, larvae, protozoan; os `_raw` usam o descritor real e não têm nota): nota sob o eixo x, "<100%: amostra estratificada de p% do treino do split da busca
  (adaptação; não é o descritor incremental)".
- **AE outubro (SPiFiL camada a camada + AE), GeM+TTA+pw** (marrom, ◆) em eggs/larvae/protozoan: `Scalable_Hybrid_FLIM/results/svm_disc_bp01w64c1280_pcts_test_gemttapwr.csv`
  (L4, `status` ok, 3 splits → com barra; protozoan 100% só tem 2; não há 10%). Fica junto do "best" do curvas_pct; se o best tiver `fonte`
  apontando para esse mesmo CSV (hoje: protozoan), desenha uma curva só, vermelha, "melhor nosso = AE outubro (SPiFiL camada a camada + AE), GeM+TTA+pw".
- Base sem nenhuma curva nossa: só os FMs.

## marcados.csv

Coluna `versao` (melhorada/semTTA/comTTA). Todos os pontos marcados de cada figura (seta "melhor nosso", estrelas dos FMs, losangos do AE outubro, X do aleatório), uma linha por (figura, ponto):
dataset, figura, model, family, rotulo_legenda, head (leitura), kappa_desenhado, params, ckpt, host.
ckpt vem do jsonl de origem (`source`): campo `ckpt`; no `test_svm_C100.jsonl` do GA, `<run>/<dataset>/state/seed42/candidates/<cand>/f<fid>/encoder.pt`;
FM = nome do modelo no CSV do benchmark; AE outubro = coluna `ckpt` do CSV; aleatório = o YAML da rodada. host = campo `host` do jsonl, senão "jaci" se o arquivo é do espelho da Jaci.

Modelo sem método em nenhum arquivo não é desenhado: vai para `sem_metodo.csv` (model, family, dataset, source).
Os κ das estrelas (FMs) aparecem numa coluna à direita delas, com setinha.
