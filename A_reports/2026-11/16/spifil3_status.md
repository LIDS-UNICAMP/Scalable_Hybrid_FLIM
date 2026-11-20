# SPiFiL-3 (Agente 2d): status da implementação

Data: 16/11/2026. Referências: MISSION.md e prep_literatura_e_desenho.md §5.3. Rodou só em CPU (`CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `nice -n 15`). Não toquei na árvore viva nem em processos.

## Entrega

| Arquivo | Conteúdo |
|---|---|
| `research/spifil_b0b4/agent2/spifil3.patch` | Cria só 2 arquivos novos: `ga_search/abstract_init/spifil3.py` (708 linhas) e `tests/abstract_init/test_spifil3.py` (277 linhas) |
| `research/spifil_b0b4/agent2/spifil3.patch.md5` | `f00e929bff4e66deb18b48089d24499f` |
| Cópia de trabalho | `/tmp/claude-1014/-dados-home-moliveira-Scalable-Hybrid-FLIM/05a95f35-17a2-4976-a2bd-9c4d1e4ff9eb/scratchpad/a2d_spifil3` (git local; o baseline é o HEAD `0de1838` da árvore viva mais a interface) |

**Dependência.** O patch pressupõe a interface publicada em `research/spifil_b0b4/agent2/interface/`:

- `base.py` (md5 `ead8195416ede86726f5bcece852e155`) instalado como `ga_search/abstract_init/base.py`;
- `__init__abstract_init.py` (md5 `263cfe18da0e96db60ca462f0fe6e27e`) instalado como `ga_search/abstract_init/__init__.py`.

Na cópia, os dois fazem parte do baseline e não entram no patch. `git apply --check` passa na árvore viva, mas sem a interface o import falha.

## O que foi implementado (`Spifil3Predictive`, registro `spifil3`; controle `Spifil3Rand`, registro `spifil3_rand`)

**Interface.** As classes são subclasses de `base.KernelInitializer`, com `fit(prefix, train_data, block_spec, context) -> (weights, bias, prov)`. O `init_block` é herdado, de modo que o runner usa o mesmo caminho dos outros métodos. A config é montada em camadas, em ordem crescente de precedência:

1. `DEFAULT_CONFIG`;
2. kwargs do construtor;
3. `context.config` (plano ou em `["spifil3"]`). Chaves desconhecidas dão erro.

**Prefixo congelado (I1).** O fit e o init_block:

- põem o prefixo em `eval()` com `requires_grad=False`;
- calculam features só com `no_grad`;
- restauram o estado original no fim;
- verificam o fingerprint com `check_prefix_unchanged`.

Nenhum parâmetro do prefixo entra em otimizador.

**Estágio 1: latente preditivo** (label-free, só treino, sobre F = prefix(x) padronizado por canal).

- **Encoder de contexto f_θ:** conv k×k (C_in→d), GELU, conv 3×3. Variantes: `encoder=local` (1×1, RF = k) e `linear`.
- **Encoder-alvo f_ξ:** EMA de f_θ com momentum de 0,996 a 1, alvo com stop-grad e layer-norm por posição.
- **Preditor h_φ:** 2 convs 3×3, com token-alvo e embedding sincos 2D somados nos alvos.
- **Cabeça de reconstrução** 1×1 (d→C_in).
- **Duas tarefas somadas:**
  - context→target à la I-JEPA, ‖h_φ(f_θ(F_ctx)) − sg(f_ξ(F))‖² nos alvos;
  - masked feature reconstruction, ‖rec(h_φ(…)) − F‖² nos alvos.
- **Máscaras:** M = 4 blocos-alvo (2 se a grade for menor que 14×14), escala 0,15–0,2, aspecto 0,75–1,5, lado mínimo 2. O contexto é um bloco de escala 0,85–1,0 menos os alvos.
- **Anti-colapso:** EMA com stop-grad, piso de variância VICReg (γ = 1), covariância e o monitor erank(Z)/d ≥ 0,2. Abaixo desse limite o fit lança `InfeasibleCandidate("ssl_collapse")`.
- **Orçamento:** `steps` é configurável, com padrão de 300 (o desenho fala em até 2000). Há também `batch_size`, `max_images`, `max_patches` e `max_seconds`; estourar o tempo gera `InfeasibleCandidate("budget: max_seconds")`.
- **Descarte:** f_θ, h_φ, o reconstrutor e os tokens são temporários. Saem de escopo no fim do fit; os weakrefs ficam em `last_aux_refs` só para teste. Só `conv.*` (e `skip.*`) é devolvido.
- **Padding:** as convs temporárias usam padding `replicate`. Com isso, entrada constante dá latente constante e o colapso é detectável; com zero-padding, a borda criava variação artificial.

**Estágio 2: protótipos.**

1. Z = f_ξ(F) em N posições amostradas, com N ≤ `max_patches`.
2. ZCA (eps = 0,01·média dos autovalores).
3. k-means esférico com K = 3·C_out, inicialização k-means++ semeada.
4. Seleção de C_out protótipos por massa + FPS (a_rep = 0,5).
5. Atribuição T = cos(z̃, μ_j)/τ com τ = 0,1, centrada.

**Estágio 3: latente→kernel explícito.**

- **Regressão:** ridge do patch `unfold(F, k)` (D = C_in·k², mesmo suporte k×k, channel-major), padronizado, para T.
- **Escolha de λ:** grade {1e-4, …, 1}, por MSE em imagens de **treino** separadas (holdout de 20%; com uma única imagem, o holdout passa a ser por posições).
- **Kernel:** coluna unitária no espaço z, dobrada como weight = (w/σ_X).reshape(C_out, C_in, k, k) e b = −Σ μ_X·w/σ_X.
- **Diagnóstico R²:** sai sempre na proveniência, com `r2_per_kernel` (holdout), `r2_fit_per_kernel`, mediana/min/max, `r2_low` (mediana < 0,3) e as opções registradas.

**Controle S3-rand** (`Spifil3Rand`, ou `latent="random"`). É o mesmo pipeline com f_ξ na mesma inicialização semeada, sem treino (0 passos).

**Escopo de blocos.** Suportados:

- `conv` com k ímpar;
- `pointwise`;
- skip `identity`;
- skip `project`, com 1×1 por PCA label-free e linhas extras aleatórias semeadas quando C_out > C_in, registrado em `approximations`.

`depthwise`, `dwsep` e `inception` lançam `InfeasibleCandidate("spifil3: unsupported …")` de forma determinística. É uma lacuna conhecida.

**Proveniência.** Usa `base.make_provenance` com:

- `variant` (S3 ou S3-rand), a config efetiva e o seu sha256;
- `prefix_state_sha256` e o sha256 das imagens lidas;
- N, D, N/D e a grade;
- curva de loss subamostrada (jepa/rec/var/cov) e saúde do latente (erank_Z, erank/d, std mínima);
- K e as massas dos protótipos selecionados;
- λ e o MSE por λ, R² por kernel;
- diagnóstico do banco (normas, redundância, erank), mais ativação e % de canais mortos quando há `diag_batch`;
- ambiente (torch, numpy, python, git sha).

Também registra `labels_used=False`, `label_source="none"` e `mask_used=False`. O fit nunca lê `y` nem máscara.

## Testes (`tests/abstract_init/test_spifil3.py`, 20 testes, cerca de 10 s em CPU)

| Pedido | Teste |
|---|---|
| shapes | `test_shapes_and_assembly` com 6 casos: conv k = 3, 5 e 1, pointwise, residual identity e residual project. Verifica os shapes (C_out, C_in, k, k), fp32 finito, `assemble_block` estrito e o forward. |
| determinismo | `test_determinism`: a mesma seed dá pesos, R² e hash de config idênticos bit a bit; outra seed dá pesos diferentes. |
| prefixo intocado | `test_prefix_untouched`: pesos iguais antes e depois, fingerprint igual ao registrado, nenhum `p.grad`, prefixo em eval e congelado durante o fit e estado restaurado depois. |
| preditor fora do módulo final | `test_predictor_not_in_final_module`: só `conv.weight`/`conv.bias`; o bloco montado não tem predictor/encoders/reconstructor/tokens; os weakrefs dos temporários estão mortos após o gc; o objeto do método não guarda nenhum `nn.Module`. |
| R² > 0 com estrutura | `test_r2_positive_on_structured_synthetic`, com listras e bolhas: R² mediano no holdout > 0 e todos os R² de ajuste > 0. Complementos: `test_ridge_recovers_linear_map` (mapa linear exato recuperado, R² > 0,999, cos > 0,999) e `test_kernel_fold_matches_zspace_response` (a dobra no weight/bias reproduz a conv). |
| S3-rand roda | `test_s3_rand_runs_and_differs`: 0 passos, shapes e R² presentes, kernels diferentes do S3, e `Spifil3Rand` igual a `latent="random"` bit a bit. |
| extras | label-free invariante a permutação de rótulos; `split_key="val"` recusado; entrada constante dá `ssl_collapse`; bloco depthwise dá `InfeasibleCandidate`; máscaras disjuntas e não vazias; `diag_batch`; registro e `validate_init`; caminho do runner `init_block` com genoma. |

Resultados:

- **Suíte só do SPiFiL-3:** `20 passed in 10.09s`.
- **Suíte completa da cópia** (`tests/`, com modehb + abstract_init): `2325 passed, 7 skipped, 38 warnings in 85.62s`, exit 0. Log em `…/scratchpad/a2d_pytest_full.txt`.

## Smoke em escala parecida com a real (CPU, sintético; não é resultado científico)

Configuração: prefixo conv3→64 + pool, grade 28×28, C_in = 64, k = 3, C_out = 48, 64 imagens, N ≈ 20k (N/D = 34,7), 300 passos.

| Braço | Tempo | λ | R² holdout (mediana / min / max) | erank/d | Loss total |
|---|---|---|---|---|---|
| S3 | 29 s | 1e-3 | 0,634 / 0,214 / 0,944 | 0,67 | 1,80 → 0,51 |
| S3-rand | 0,4 s | 1e-3 | 0,647 / 0,272 / 0,897 | 0,78 | — |

Nos dados sintéticos, S3 ≈ S3-rand em R². Isso não testa a H-L7, que pede κ_val pós-treino em dados reais. Ainda assim, mostra por que o controle S3-rand é obrigatório.

## Pendências e riscos

- Faltam `depthwise`, `dwsep`, `inception` e o S3-sup (k-means por classe). Hoje todos dão InfeasibleCandidate determinístico, e a versão entregue é a label-free.
- `steps` = 300 por padrão é menor que os 2000 do desenho. Ajustar por `context.config` na matriz.
- O λ é global, um para todos os kernels (`approximations: global_lambda`).
- Em CUDA não há teste de tolerância. Só o determinismo em CPU foi verificado.
- Nada foi rodado em dados reais nem em GPU. O E2E (R9) continua com o runner (Agentes 2/5).
