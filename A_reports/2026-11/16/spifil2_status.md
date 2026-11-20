# SPiFiL-2 (inicialização invariante entre vistas): status do Agente 2c

> Data: 16/11/2026. Implementação isolada, só CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `nice 15`), feita numa cópia própria do repositório. Nada da árvore viva nem de produção foi alterado, e nenhum processo foi tocado.
> **Status científico:** proposta **não testada**. Os testes provam que o código está correto (contrato, geometria, determinismo), **não** provam que o método melhora o κ. A validação científica fica para a matriz §9 do desenho (κ_val ORIG, critério H-L6 antes de selecionar em produção).

## Entregáveis

| Arquivo | md5 |
|---|---|
| `research/spifil_b0b4/agent2/spifil2.patch` (2 arquivos novos, 1388 linhas) | ver `spifil2.patch.md5` |
| → `ga_search/abstract_init/spifil2.py` (novo) | `cce768b235b5435ebf7c31f8faeb8851` |
| → `tests/abstract_init/test_spifil2.py` (novo) | `18b2818fbb33afbe5e698416e7184a90` |

`git apply --check` passa na árvore viva (sem aplicar).

**Dependência dura:** a interface do Agente 2a, `research/spifil_b0b4/agent2/interface/base.py` (md5 `ead8195416ede86726f5bcece852e155`) e `__init__abstract_init.py` (md5 `263cfe18da0e96db60ca462f0fe6e27e`), instalada como `ga_search/abstract_init/{base.py,__init__.py}`. Ela **não** está no meu patch, que deve ser aplicado depois do patch do 2a.

## O que foi implementado (desenho §5.2)

- **Interface.** `Spifil2Invariant(KernelInitializer)`, registrado como `"spifil2"` (`base.get_initializer("spifil2", variant=..., **overrides)`).
  - `fit(prefix, train_data, block_spec, context)` devolve `(weights, bias, provenance)`.
  - `init_block` vem da base e monta o `SpifilOpsBlock`.
  - A config vem de `context.config["spifil2"]` (ou das chaves de `DEFAULT_CONFIG`).
- **Vistas com θ.** `make_views_with_theta(x, V, cfg, seed|generator)` replica `ssl_trainers.make_views` (mesma ordem de sorteios) e devolve também θ (N,2,3) e os parâmetros fotométricos.
  - Teste de igualdade **bit a bit** contra `make_views`: config padrão, config custom e `generator`.
  - Extensão registrada `rot90`: sorteada **depois** dos demais sorteios, então não altera a sequência quando está desligada. θ' = θ·R, e o teste confirma que reamostrar a imagem com θ' reproduz a vista.
- **Correspondências.** `correspondences(θa, θb, grade)`:
  - u_img = A_a·u_a + t_a; u_b = A_b⁻¹(u_img − t_b);
  - margem de 1 célula, célula mais próxima, erro registrado;
  - descarte opcional de imagens com fundo demais (`max_bg_frac`, parasitas pre).
- **Candidatos (TREINO).**
  - Patches densos do prefixo congelado, holdout separado para a escala, ZCA (eps = 0,01·média dos autovalores) e `patch_norm` opcional.
  - K = 3·C_out por k-means esférico (k-means++ semeado) ou FPS em patches.
  - Dobra exata w = c·Z·P, b = −w·μ, mais `unit_response_std`.
  - Gancho `candidate_fn(X, K, gen) → W (K', D)` para plugar protótipos do SPiFiL-1 ou patches do SPiFiL-0.
- **Estatísticas entre vistas, por candidato:**
  - estabilidade = Pearson casado por posição (acumulado em fp64 sobre todos os pares);
  - repetibilidade = top-m picos mapeados com tolerância de 1 célula;
  - concordância SwAV = CE trocada entre o código Sinkhorn (equipartição, ε = 0,05, 3 iterações, log-domínio) de uma vista e o softmax(·/τ) da outra;
  - massa sob equipartição.
- **Seleção.**
  - score = λ_stab·z(stab) + λ_rep·z(rep) + λ_agree·z(agree) + λ_mass·z(log massa) − λ_dc·z(DC) [+ λ_fisher·z(Fisher GAP), só em `s2_sup`];
  - seleção gulosa com bônus de diversidade λ_div·(1 − max|cos|), NMS |cos| ≤ 0,9 no espaço branqueado e piso de estabilidade;
  - quando faltam candidatos, o piso e depois o NMS são relaxados, e as contagens vão para a proveniência (`relaxations`).
- **Refino opcional estilo VICRegL ou SwAV local** (`refine.steps > 0`).
  - Otimiza uma **cópia temporária** (w, b) sobre as features em cache do prefixo congelado. Nenhum parâmetro persistente é criado e o backbone não é tocado.
  - Depois do refino, recentra e reescala. A proveniência registra a loss, o cos com o init e a estabilidade depois do refino.
- **Tipos de bloco.**
  - `conv` (k = 1/3/5/7) e `pointwise`;
  - `depthwise` (m ≥ 1, por canal);
  - `dwsep` (dw por canal, depois pw ajustado na saída linear do dw);
  - `inception` (por ramo; o ramo pool sobre maxpool 3/1/1);
  - `residual`: skip `identity`, ou `project` = PCA 1×1 label-free, com linhas zero se C_out > C_in (registrado).
- **Augmentations por domínio = variante registrada** (`AUG_POLICIES`): `aug_default` (= produção), `none`, `rs`, `aid`, `flowers` (só hflip, canal 0), `ham10000`, `parasites_raw`, `parasites_pre`. São hipóteses do desenho, não verdades.
- **Braços registrados** (`VARIANTS`): `s2`, `s2_noinv` (**controle sem nenhum termo de invariância**, porque não supomos que invariância é boa), `s2_aug_domain`, `s2_nounitstd`, `s2_patches`, `s2_dcpen`, `s2_sup`, `s2_ref_vicregl`, `s2_ref_swav`.
- **Rótulos.** Label-free por padrão. Não lê `y` nem máscara: `labels_used = False`, `mask_used = False`. O modo supervisionado só existe via `s2_sup` e é recusado se o `TrainData` for label_free (I6).
- **Contratos.**
  - I1: `prefix_fingerprint` antes e depois, prefixo em eval sem grad, estado restaurado.
  - I5: `check_train_only`.
  - I7/I8: `InfeasibleCandidate("budget: ...")` para K > N, memória do cache acima de `max_gpu_gb` ou tempo.
- **Proveniência.** `make_provenance` da base mais:
  - `method_config` e `method_config_sha256`;
  - `variant` e a política de aug completa;
  - por banco: N, D, N/D, K, erank dos patches, n_pares, índices escolhidos, stab/rep/agree/massa/score dos escolhidos e a média do banco, fração DC (escolhidos × banco), redundância, refino;
  - `images_sha256`, `split_fingerprint`, erro médio do casamento.

  Tudo é serializável em JSON (testado).

## Testes (`tests/abstract_init/test_spifil2.py`): 35 passam em cerca de 5 s (CPU)

| Grupo | Testes |
|---|---|
| Shapes (I2) | 10 specs (conv k1/k3/k5, pointwise, depthwise m=2 e m=1, dwsep, inception, residual identity e project): chaves e shapes iguais a `SpifilOpsBlock(effective_spec)`, `load_state_dict(strict)`, forward. Mais o prefixo real conv+pool (grade 16×16) com I1. Mais `init_block` via registro da base (`validate_init`, pool max). |
| Determinismo (I4) | Mesma seed dá resultado bit a bit igual, seed diferente dá resultado diferente; idem com refino `vicregl_local` e `swav_local`. O refino não deixa parâmetros extras. |
| Só treino (I5) | `split_key` ∈ {val, test, validation} é recusado; o loader é chamado uma única vez, só com a seed. |
| Label-free (I6) | Rótulos permutados dão saída idêntica; `s2_sup` usa rótulos e é recusado com dados label_free. |
| θ correto | Vistas bit a bit iguais a `make_views` (3 seeds × 2 configs). θ reproduz a vista (com e sem rot90). Identidade e hflip exatos. **Ponto marcado** (gaussiana) recuperado a ≤ 1 célula (erro médio ≤ 0,75) com e sem rot90, na grade da imagem e numa grade 4× reduzida. Sinkhorn com equipartição. |
| Caso sintético estável × instável | Banco controlado {DC, ∂x, ∂y, Laplaciano} com vistas flip + jitter DC: **escolhe o Laplaciano** (o estável, stab > 0,8 e > max(instáveis) + 0,25). Só com flips: escolhe {DC, Laplaciano} e **rejeita os orientados** ∂x/∂y. Com k-means real, S2 escolhe filtros mais estáveis que a média do banco **e** que o controle `s2_noinv`. |
| Outros | NMS rejeita duplicata; o piso de estabilidade só é relaxado se faltar candidato; orçamento estourado dá `InfeasibleCandidate("budget")`; registros. |

**Suíte inteira da cópia** (`pytest tests/`, CPU): ver a seção "Suíte completa" no fim.

**Smoke de custo** (CPU, 4 threads): 128 imagens 96×96, prefixo conv(3→32)+pool (grade 48×48), bloco conv k3 com 48 canais (N ≈ 30k, D = 288, K = 144, cerca de 223k pares). `s2` leva 4,0 s e `s2_ref_vicregl` (30 passos) leva 4,6 s.

## Riscos e observações (para o gate e para a matriz §9)

1. **Viés para baixa frequência e DC.** Mesmo com o Pearson baixo, a repetibilidade (top-m dentro da imagem) e a concordância Sinkhorn **favorecem filtros DC/suaves**, porque um deslocamento DC por vista não muda o ranking de picos. No caso sintético com 2 filtros sob jitter, o DC entrou como 2º filtro. Isso confirma o risco do desenho §5.2. Mitigações disponíveis: `lam_dc` (variante `s2_dcpen`), a medição `dc_fraction` (escolhidos × banco) e a H-L6 antes de usar o critério para selecionar.
2. **Filtros orientados são "instáveis" sob flips** (o sinal troca), por construção. Com a política padrão (hflip + vflip), o S2 tende a preferir filtros simétricos. É exatamente por isso que existem as políticas por domínio e o controle `s2_noinv`.
3. **Zero-padding** na borda faz o filtro sem DC responder ao jitter nas células de borda (Laplaciano com stab 0,86, e não 1). A margem de correspondência é de 1 célula; ampliá-la é uma opção futura.
4. **Ruído puro** (smoke) dá estabilidade baixa em todo o banco: o piso de 0,2 foi relaxado em 46 de 48 escolhas, o que está registrado em `relaxations`. Em dados reais, o piso pode precisar ser calibrado por domínio.
5. **Refino.** Com lr = 1e-3 e pesos em escala `unit_response_std`, o refino quase não move os filtros (cos ≈ 0,9997 em 30 passos). O lr e o número de passos precisam de ablação. A loss oscila entre lotes.
6. **Aproximações registradas:**
   - `feature_grid_uniform`: a grade do prefixo cobre [-1, 1] uniformemente; é exata sem pooling e aproximada com pooling;
   - `views_pair_0_1`: só o par (0, 1) entra nas estatísticas;
   - `unit_response_std`;
   - `inception_no_cross_branch_diversity`;
   - `skip_pca_label_free`.

   O `rot90` é discreto (não há rotação contínua).
7. **Não implementado nesta versão:** casamento por feature (VICRegL), V > 2 vistas, diversidade entre ramos do inception, e estratificação por classe na leitura das imagens (lê as primeiras `max_images` do loader semeado).

## Suíte completa

`CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 15 python -m pytest -q tests/ -x` na cópia (com o `base.py` e o `__init__.py` do 2a instalados): **2340 passed, 7 skipped, 0 failed** em 79,8 s. Isso inclui os 35 testes do SPiFiL-2 e toda a suíte `tests/modehb`. Os 7 skips já existiam e não vêm deste patch.
