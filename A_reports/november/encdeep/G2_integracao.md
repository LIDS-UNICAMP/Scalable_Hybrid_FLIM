# G2 — integração encdeep (A0)

08/11/2026, athena, worktree `Scalable_Hybrid_FLIM_discovery`. Nada foi enfileirado: a inserção na fila é do G3.

## Resumo

O caminho encdeep está ligado ao `spf_grade` de forma opt-in. Ele só liga quando existe `runner_args.amostra.encdeep`.

Sem a chave, a célula minúscula de referência (remote_sensing, 3×32, fps + random_z, sorteio 42, 20 imagens de banco) dá as 20 linhas do CSV iguais, campo a campo, ao resultado de antes. O cabeçalho também é idêntico. A comparação exclui só as colunas de tempo e memória (`select_s`, `extract_s`, `fit_s`, `gpu_mib`, `rss_mib`).

Os checks do §14 passam, e o smoke na GPU (3 braços × 5 camadas × 256) rodou sem erro.

## Mudanças

| Arquivo | Linhas | O quê |
|---|---|---|
| `eval/encdeep.py` (novo) | 1–220 | `ends`, `geometry`, `geo_log`, `no_autograd`, `layer_rows`, `deep_mechanism`/`learn`, `trunk_hash`, `persist`, `val_ids`, `save_errors` |
| `eval/spf_grade.py` | 130 | import de `check_encdeep`; o schema não importa `eval.alt_encoders` |
| | 962–973 | `score_full(..., keep=False)`: com `keep`, devolve `pred` e `class_recall` |
| | 1228–1243 | kernel, dilatação e pool por camada (`ks`, `ds`, `ps`), `LayerSpec(dilation)`, `~nw` sem fold, log da geometria |
| | 1273–1283 | hook depois de `u = F.normalize(u)`: `with no_autograd(): u, b_v, cols = mech(...)`, seguido de `bias += b_v` |
| | 1287–1295 | `grown_arch(..., ps[n], ds[n])`, `trunk_hash` com assert de tronco intacto, `salvar` |
| | 1311–1352 | `deploy_params`; `erros` só nas leituras `L*`; `L_eq<D>` (só se a largura for > D) e `cat_eq<D>` |
| | 1394–1407 | `cells`: sufixos encdeep |
| | 1451–1510 | `run`: `check_encdeep`, `expected += cat_eq`, colunas `encdeep.FIELDS`, CSV lateral `_erros.csv` |
| `experiments/ray/schema.py` | 118–173, 233–234 | `ENCDEEP_MECHS/KEYS/NEEDS` e `check_encdeep`, chamado para `source: spf_grade` |

`encdeep` só é importado de forma tardia, dentro de `if "encdeep" in spec`. Isso corrige o ciclo encdeep → local_plasticity → alt_encoders → spf_grade, que derrubou um job (corrigido junto com o coordenador). Depois da correção, `import eval.spf_grade, eval.alt_encoders` funciona, e `python -m eval.alt_encoders` / `-m eval.spf_grade` passam da fase de imports.

## Chaves (`amostra.encdeep`)

`kernels` (lista com `depth` itens), `dilacao`, `camadas` (padrão [4,5]), `geometrias` (g0/g1/g2), `mecanismos` (`z0|hebb|clapp|clapp2|sfa|sfa0|fu`, com `~nw` opcional), `vistas` {l_jitter, ab_jitter}, `hebb`, `clapp`, `sfa`, `fu` (subchaves fechadas), `salvar`, `erros`, `dim_eq` (≤ 256).

Chave, subchave ou mecanismo desconhecido falha no launcher e também no `run`.

O nome do braço segue o padrão `fps~g1~hebb`. Os controles recebem só a geometria (`random_z~g1`).

Como cada mecanismo fica:

- **hebb**: aprende Q no espaço v e depois normaliza Q.
- **clapp / sfa**: devolvem (W, b_v) sem normalizar. No sfa, isso preserva o RMS por bloco.
- **Dobra**: igual à de sempre, K = (W/σ)·G e b = b_v − K·m.
- **Negativo do CLAPP**: a vista b da imagem vizinha no lote.

## Checks (`checks_G2.py`, CPU)

1. **Opção desligada**: CSV idêntico; `cells`/`deep_mechanism` inalterados; o schema aceita o YAML e recusa chave, mecanismo, subchave, kernels e `dim_eq` inválidos.
2. **Sem rótulo e sem autograd**: a análise do código (AST) não acha rótulo nem máscara no caminho; o hook recebe só `images["train"][0]`; backward/grad são recusados dentro do guard.
3. **Tronco intacto**: L1–L3 ficam iguais (state_dict e hash) nos 7 mecanismos. A inferência não muda pesos.
4. **Patches e bordas**: `extract_patches` = unfold = conv com dil 1/2. O whitening explícito bate com a conv dobrada com erro máximo ≤ 1,2e-6, bordas incluídas.
5. **Exportação**: só `conv*` e `architecture.json`. Os parâmetros do preditor/SFA são contados como auxiliares e ficam fora.
6. **Geometria a 200 px**: os mapas são 99-49-24-11-11 em G0 e 99-49-24-24-24 em G1/G2. O RF de L5 é 107 / 75 / 91, igual à tabela do G0 e ao mapa real.

## YAMLs (`experiments/autoencoder/`, todos `_jaci` também)

| YAML | Bases | Braços |
|---|---|---|
| `disc_md_encdeep_geo_off` / `_on` | aid, remote_sensing / flowers, coconut_trees (fpsw@mapa_w0.5_1) | {g0,g1,g2} × {z0, hebb} |
| `disc_md_encdeep_mech_off` / `_on` | idem | g1 **provisória**: z0, hebb, clapp, clapp2, sfa, sfa0, z0~nw, hebb~nw (H0–H3) + random_z, orth |

Configuração comum a todos:

- Split 1, sorteio 42, 200 px, 256 × 5, kernels 5-5-5-3-3.
- `entre_camadas`, `dim_eq 256`, `erros`, `salvar`.
- Fontes em `fontes_filtros/encdeep_{off,on}`.
- CSV em `results_sol/encdeep/`.

Dry-run: as 8 configurações geram 4 jobs cada. O dry-run de uma configuração antiga (`pf_hebb_d3_off`) continua OK.

Hiperparâmetros:

| Mecanismo | Valores |
|---|---|
| hebb | 300 passos, lote 8, lr 0,005, inv_temp 0,65, init_radius 1,414 |
| clapp | 300 passos, lote 8, lr 0,01, lr_p 1e-4 |
| sfa | 100 passos, lote 8 |

## Custo medido (smoke, remote_sensing 344/159, GPU 3, GPU_CAP 4000)

Rodaram 3 células (hebb, clapp2, sfa), com 78 linhas `ok`, em 12 min de relógio. Isso inclui montar as fontes de 344 imagens.

- **Pico de GPU (torch)**: 2,6–2,7 GB. RSS: 4,4 GB.
- **Tempo do mecanismo, L4 + L5**: hebb 45 + 32 s, clapp2 86 + 46 s, sfa 24 + 27 s.
- **Por célula**: cerca de 3 min.

Valores de diagnóstico na L4:

| Mecanismo | Valores |
|---|---|
| hebb | drift 0,11, dead 0 |
| clapp2 | drift 0,034, hinge ativo 0,34, P = 1.048.576 parâmetros auxiliares |
| sfa | posto 256, δ mediano 0,14 |

O `trunk_hash` é o mesmo nos três braços (9d66b51a…). Parâmetros implantados: 3,89 M até L4 e 4,48 M até L5. O CSV de erros tem 15 × 159 linhas.

Em aid e coconut, a extração e o SVM crescem com o tamanho do treino (10× / 14×). Para a fila, a estimativa é ≤ 3 GB por item.

Arquivos do smoke: `/exps/moliveira/eval/encdeep_smoke/`.
