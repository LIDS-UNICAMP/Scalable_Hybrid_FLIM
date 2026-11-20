# Leis de construcao — P0 e piloto P1

Repo: /dados/home/moliveira/Scalable_Hybrid_FLIM_discovery (athena). Codigo: eval/spf_grade.py (arquivo untracked no git).

## Auditoria (antes da mudanca)

- `cells()` (:1330-1343): cada largura de `amostra.widths` vira um plano uniforme `(width,)*depth`; braco FREE recebe `(0,)*depth`. `depth` e `pool` sao globais do `amostra`.
- `chain()` (:1201-1305): itera `for n, target in enumerate(plan)`. A camada n tira patches (`patch_pool`) do mapa de saida do `encoder` atual, que e a rede ja construida ate n-1: o espaco de patches da camada n e a ativacao real da camada anterior. `cin` nao e passado a mao: o patch tem `len(widths[-1])*k^2` dimensoes, e `spifil.grown_arch(arch, kernel, len(u), pools(spec)[n])` registra a largura REAL `len(u)` como saida; a proxima camada le dela. Com `entre_camadas`, o ref e branqueado (`map_whitener`) e o peso e dobrado de volta (`folded`).
- Linhas: `width` = len(u) real, `widths` = realizado acumulado, `width_req`/`short` (so em CLUSTER, i.e. com saliency.pesos/agrupamento).
- Falta de candidatos: `fps`/`fpsw` usam `min(width, len(z))` — nunca ha copia; realizado = disponivel. Controles aleatorios sempre entregam `width`.
- Identidade/dedup/resume (`run()` :1388-1407): chave `(arm, plan, kernel, seed)` lida do CSV; completa quando tem `expected` linhas `ok` (`expected` depende de `depth`). Gemeos: se o metodo entregou `widths` != plano, os `twin_controls` rodam com as larguras realizadas.
- `pools(spec)` (:1100): lista de `amostra.pool` ou escalar repetido. MaxPool 3 sem padding com stride p; p=1 = sem pool. Conv `same`.

## Mudancas

- `LEI` colunas (:157); `LAW_VERSION`, `p2`, `law_channels`, `conv_cost`, `law_name` (:1107-1151).
- `short_why` na linha (:1267): `realized=available (N candidatos)` quando len(u) < pedido.
- `cells()` (:1339): com `amostra.lei`, plano = `law_channels(lei, kernel)`.
- `main()` (:1368-1374): `amostra.leis: [...]` — cada entrada e UMA arquitetura; roda `run()` com `depth=lei.depth`, `pool` = pool da receita cortado/estendido com 1 ate `depth`, `widths: [0]` dummy, CSV `<csv>_lei<law_name>.csv`. A lei entra na identidade porque cada lei tem CSV proprio; dentro dele a chave (arm, plan, kernel, seed) e a de sempre.
- `run()` (:1410-1414): base ganha `lei` (ex. `A.v1`), `lei_args`, `depth_req`, `pools`, `params`, `macs`, `maps`, `rf`, calculados antes de rodar. Requested = `plan`/`width_req`; realized = `widths`/`width`; profundidade realizada = maior `layer`.
- Sem `leis`, nada muda (68 YAMLs spf_grade conferidos: `cells()` identico ao antigo).

## As leis (v1)

Todas as larguras: potencia de 2 por arredondamento em log2, `2**round(log2 x)` (meio exato -> par, regra do `round` do Python), presa em [32, 1024].

- **A geometrica** `{lei: A, w0, r, wmin=32, depth}`: w_i = p2(max(w0·r^i, wmin)), i = 0..D-1.
- **B posicao relativa** `{lei: B, w0, wL, f, depth}`: t = i/(D-1); log2 w = log2 w0 + (log2 wL − log2 w0)·f(t); f: lin = t, fast = √t (cai cedo), late = t² (cai tarde).
- **C orcamento** `{lei: C, P, depth, forma: {A ou B sem depth}}`: s ∈ {2^(j/8), j=−40..40}; w_i = p2(s·forma_continua_i); escolhe s que minimiza |Σ c_{i−1}c_i k² − P| (c_0 = 3, sem bias). Empate -> menor s.
- **D crescimento por dados** (so interface, `NotImplementedError`): a camada cresce enquanto criterios sem rotulo pedirem — posto efetivo do banco de patches (erank), cobertura (fracao de patches com cos > τ a algum filtro), redundancia (fracao de pares filtro-filtro com cos > COPY_COS) e ganho marginal (queda da energia residual ao somar o proximo bloco de filtros); para quando o ganho marginal < ε ou a redundancia > ρ.

Pseudocodigo:
```
law_channels(spec, k, s=1):
  A: [p2(s*max(w0*r**i, wmin)) for i < D]
  B: [p2(s*2**(a+(b-a)*f(i/(D-1)))) for i < D]
  C: argmin_{j} |params(law_channels(forma|D, k, 2**(j/8)), k) - P|
conv_cost(w, k, size, pool): por camada params += cin*c*k²; macs += cin*c*k²*side²;
  rf += (k-1)*jump; se pool>1: side=(side-3)//p+1, rf += 2*jump, jump*=p
```

Unit-checks: A(256, 0.5, D4) = 256-128-64-32 OK; A(512, 0.5, D5) = 512-256-128-64-32 OK.

Nota de arredondamento: r = 0.71 da 256·0.71 = 181.8, log2 = 7.506 -> 256; a lei vira "metade a cada 2 camadas". Em C a escala fracionaria + p2 desloca a forma (ex. early-heavy vira 1024-256-...), e o erro de orcamento residual e grande (passos de 2x por camada).

## Orcamento P

Receita uniforme 256 × 6, k5, entrada RGB/LAB 3 canais: P = Σ = 3·256·25 + 5·256²·25 = **8.211.200** params de conv; 2·MACs = 44,2 GFLOP por imagem 200 px. Ancora pareada: **existe** em disc_md_sol_wsal (widths [64, 256], depth 6, mesmos pools, k5, fps e fpsw@mapa_w0.5_1, random_z, orth) — mesmo protocolo; split 1, sorteio 42 estao la.

## Arquiteturas do P1 (200 px, k5, pools 2-2-2-2[-1-1])

Mapas: 99-49-24-11(-11-11), sempre ≥ 3 px (final 11). Depth 4 usa pools 2-2-2-2 (corte explicito da lista da receita). FLOPs = 2·MACs por imagem.

| # | lei | canais | D | params | GFLOP | RF |
|---|---|---|---|---|---|---|
| 1 | A w0 256 r .5 | 256-128-64-32 | 4 | 1,09M | 18,6 | 91 |
| 2 | A w0 256 r .5 | 256-128-64-32-32-32 | 6 | 1,15M | 18,6 | 219 |
| 3 | A w0 256 r .71 | 256-256-128-128 | 4 | 2,89M | 38,1 | 91 |
| 4 | A w0 256 r .71 | 256-256-128-128-64-64 | 6 | 3,19M | 38,1 | 219 |
| 5 | A w0 512 r .5 | 512-256-128-64 | 4 | 4,34M | 71,5 | 91 |
| 6 | A w0 512 r .5 | 512-256-128-64-32-32 | 6 | 4,42M | 71,5 | 219 |
| 7 | A w0 512 r .71 | 512-512-256-256 | 4 | 11,51M | 149,2 | 91 |
| 8 | A w0 512 r .71 | 512-512-256-256-128-128 | 6 | 12,74M | 149,5 | 219 |
| 9 | B lin 512→32 | 512-256-64-32 | 4 | 3,78M | 69,3 | 91 |
| 10 | B lin 512→32 | 512-256-128-128-64-32 | 6 | 4,80M | 71,8 | 219 |
| 11 | B fast 512→32 | 512-128-64-32 | 4 | 1,93M | 36,2 | 91 |
| 12 | B fast 512→32 | 512-128-64-64-32-32 | 6 | 2,06M | 36,3 | 219 |
| 13 | B late 512→32 | 512-512-128-32 | 4 | 8,33M | 139,5 | 91 |
| 14 | B late 512→32 | 512-512-256-256-64-32 | 6 | 11,97M | 149,3 | 219 |
| 15 | C early-heavy (A 512 r .5) | 1024-256-256-64-64-64 | 6 | 8,88M (+8,2%) | 143,0 | 219 |
| 16 | C gradual (B lin 512→32) | 512-256-256-128-64-32 | 6 | 6,03M (−26,6%) | 76,2 | 219 |
| 17 | C deep-preserving (B late 512→128) | 256-256-256-256-128-64 | 6 | 5,96M (−27,4%) | 43,7 | 219 |
| ref | uniforme 256 (sol_wsal) | 256×6 | 6 | 8,21M | 44,2 | 219 |

Sem duplicatas. 17 arquiteturas × 4 bracos (fps, fpsw@mapa_w0.5_1, random_z, orth) = 68 celulas por base, + gemeos se o metodo vier curto.

Leituras: L e cat (padrao da receita). Variante `_eq` NAO entrou: `amostra.eq` so reduz `ms` a dimensao de `gem` dentro de `leituras`; a receita nao usa `leituras`/`ms`, e ligar `ms` custa 2 forwards extras por camada.

## YAMLs

- experiments/autoencoder/disc_md_leis_p1_jaci.yaml (Jaci, fontes /dados/home/moliveira/exps/fontes_filtros/leis_p1, gpu [0] × 8).
- experiments/autoencoder/disc_md_leis_p1.yaml (athena, /exps/moliveira, gpu [0,1,2,3] × 1; GPU_CAP_MIB=4200 no lancamento).
- Dry-run (`python -m experiments.ray.launch <yaml> --dry-run`): ambos passam, 10 jobs cada (5 bases × split 1 × 2 shards).

Teste ponta a ponta (Larvae, 20 imagens, A w0 64 r .5 D4 = 64-32-32-32, fps + random_z, GPU_CAP_MIB=4000): 28 linhas ok, colunas lei/params/maps preenchidas; relancar nao refaz nada (resume).

## Agendamento (nao lancado)

- Jaci: disc_md_leis_p1_jaci.yaml entra depois dos experimentos que ja estao rodando e antes do benchmark.
- Athena: a gemea disc_md_leis_p1.yaml entra em seguida, com teto de 15 GB nossos por GPU.
