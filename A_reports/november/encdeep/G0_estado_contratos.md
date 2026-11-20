# G0 — estado real e contratos (encdeep: L4/L5 sem backprop e sem rótulo)

08/11/2026, athena, worktree `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery` (branch `discovery`).
Grafo: `graphify update .` rodado antes; nós citados com `source_location`. Grep só para YAML, fila e
SPiFiL externo (o grafo não indexa esses arquivos). Nada foi implementado, lançado, parado ou editado.
O prompt completo está em `prompt_encdeep.md`. A última linha dele pede: **entrar na fila na frente
dos benchmarks, para rodar logo em seguida**.

## Resumo em uma frase

O tronco já monta L4 e L5 por SPiFiL/fps com whitening dobrado. Porém **nenhum checkpoint L1–L3 fica
salvo**: os pesos vão para um `TemporaryDirectory`. A forma mais simples é ligar uma chave opt-in
dentro de `chain()`, entre a escolha das direções `u` e a dobra.

---

## 1. Tronco (`eval/spf_grade.py`)

| Peça | Onde | Contrato |
|---|---|---|
| `cells(spec)` | L1357 | Célula = (braço+`~tails`, plan=(largura,)*depth ou `law_channels`, kernel, sorteio). Um único `kernel` vale para **todas** as camadas. |
| `tails(spec)` | L1340 | Sufixos `~variante` (norm, ativação, `~w<λ>`). Atenção: em `chain` (L1202), **todo tag que começa com `w` é lido como lambda do whitening**. |
| `law_channels` | L1137 | Lei A/B/C → larguras em potência de 2, entre 32 e 1024. Para encdeep basta `widths: [256]`, depth 5. |
| `chain(...)` | L1196–1328 | Monta a rede camada a camada sem gradiente, ver abaixo. |
| `pools(spec)` | L1123 | Escalar ou lista por camada. A receita vigente (`disc_md_pf_res320_off.yaml`) usa k5, 256, `pool: [2,2,2,2,1,1]`, `entre_camadas: true`, `fps`, controles `random_z, orth`. |
| `conv_cost` | L1155 | Mapa e campo receptivo com conv `same` e MaxPool 3 sem padding: `side=(side-3)//p+1`. |
| `run` / `main` | L1404 / L1384 | Retomada por CSV com a chave (arm, plan, kernel, seed). O dispatcher monta isto em `experiments/ray/runners/eval.py:_plan_l1_fast` L471 (`work_dir = artifacts_dir/<base>_split<s>_pct<p>`). |

**Ordem dentro de `chain` para a camada n** (L1226–1273):

1. `ref = patch_pool(lab, ref_seeds, encoder, layer).feats` (L1230). As sementes medoide do banco vêm de `sources`/`stored_sources` (L513/L550). `patch_pool` (`eval/l1_sem_rotulo.py` L113) roda o encoder L1..L(n−1) e projeta as sementes na grade pelo `spifil.rescale` (proporcional, gera duplicatas; `flim/spifil.py` L476). Depois `extract_patches` do SPiFiL faz o unfold canal-primeiro, com `dilation` já suportada (`SPiFiL/src/spifil/patches.py` L72–87).
2. **Whitening primeiro**: `fold = map_whitener(ref, span², shrink)` (L979). É uma ZCA por canal, com shrinkage 0,1, estimada no pixel central de cada patch das sementes. Fica em `fold=(shift m, gain G)`. Depois vem `whitened` (L1110), v' = G(p − m), por posição.
3. **N0 depois**: `mean, stdev` de `ref` já branqueado (L1234), por dimensão C·k². Com tags, passa por `renorm` (L987).
4. Escolha de `u` pelo método ou controle (L1249–1257). Os candidatos ficam em `Ctx` (L628): `keep` sem janela chapada nem repetida, até `CLUSTER_ROWS=5000`. Depois `u = F.normalize(u)` (L1260).
5. **Dobra**: `weight = u/stdev`, `bias = -(weight·mean)` (L1263). Em seguida `folded(weight, bias, shift, gain)` (L1117) dá K = (u/σ)·G e b = −K·m − (u/σ)·μ.
   - Na notação do §6.1: A = diag(1/σ)·(I_k² ⊗ G), m_total = m ⊕ G⁻¹μ. Q = linhas de `u`, b_v = 0.
6. `spifil.write_weights` + `grown_arch(arch, kernel, len(u), pool)` (L1268–1270), `build(...)` (L1273), e a leitura (L1280–1327).

**Dilatação**: `grown_arch` (`flim/spifil.py` L701) e `flim/arch.py` L96–105 aceitam `dilation_rate`, e `extract_patches` também. Mas `chain` **não passa dilatação** em `LayerSpec` (L1229) nem em `grown_arch`. G2 precisa expor isso nas duas chamadas ao mesmo tempo.

**Kernel por camada**: hoje não existe. Para L1–L3 em k5 e L4/L5 em k3, G2 precisa de `encdeep.kernels` aplicado a `span`, `LayerSpec` e `grown_arch`.

**Checkpoint L3: não existe.**
- `chain` grava cada `L{n}` em `tempfile.TemporaryDirectory(dir=work_dir)` (L1225), e isso é apagado no fim da célula. Os `artifacts_dir` (por exemplo `/exps/moliveira/spifil_growth/disc_md_pf_res320_off/aid_split1_pct50`) estão vazios.
- O que persiste é o `.pt` de `amostra.fontes`: `<fontes>/<base>_split<s>_d<draw>.pt`, com LAB do banco, sementes, corte e diagnóstico, ~0,5 GB por arquivo. Ele não guarda pesos nem hash.
- L1–L3 de `fps` são reconstruídos de forma determinística a partir do `.pt` mais (arm, seed). O fps roda em numpy CPU, o eigh do `map_whitener` em double CPU.
- **Proposta G2**: com `encdeep`, copiar `L{n}` para `work_dir/encdeep/<arm>_k<k>_s<seed>/L{n}/`, escrever o sha256 dos `conv*-weights` / `conv*-bias` numa coluna `trunk_hash` e o fold (m, G, μ, σ) em `.pt` ao lado.
- Check do §14: reconstruir duas vezes e comparar os hashes.
- Um módulo novo recebe o tronco congelado como: `encoder` (nn.Module L1..L(n−1), `.eval()`), `fold`/`mean`/`stdev` da camada n, `images["train"]` (uint8, `to_float`) e o `LayerSpec(kernel, dilation)`.

**Hook opt-in (ponto único de integração, G2, dono A0)**: em `chain`, logo após L1260 e antes de L1263.

```python
if mech := deep_mechanism(spec, n, tags):          # None => caminho atual, byte a byte
    u, b_v, info2 = mech(u=u, encoder=encoder, fold=fold, mean=mean, stdev=stdev,
                         layer=layer, train=images["train"][0], to_float=to_float, seed=seed + n)
```

- Depois: `bias = -(weight·mean) + b_v`.
- O tag do braço é `~hebb`, `~clapp`, `~sfa` ou `~fu`. **Nenhum pode começar com `w`**.
- O braço zero continua sendo o `fps` sem tag, no mesmo CSV e na mesma geometria.
- `deep_mechanism` fica num arquivo novo do A0, `eval/encdeep.py`. Ele despacha para os três módulos abaixo e recusa chave desconhecida em `amostra.encdeep`.

## 2. SoftHebb (`eval/alt_encoders.py`)

- `HebbBlock` (L181): BN sem afim, conv sem viés, triângulo, MaxPool 3.
- `hebb_step(block, h, lr, beta, bias, norm)` (L197): competição softmax com anti-Hebb, Δw = conv2d_weight(z, y) − (y·u)·w, normalização `neuron` ou `tensor` (`delta_norm`), lr·|‖w‖−1|^½.
- `learn_softhebb` (L216): `layer(key,i)` repete o último valor da lista. Para L4/L5, `lr` = 0,005 e `inv_temp` = 0,65.
- `run_hebb` (L479): `calib` = passos só de BN, `calib_steps`, durations 0/300; `spifil_init` (L433) usa o `METHODS` da grade.
- O caminho atual **não é o tronco da grade**: usa BN por canal no lugar de ZCA+N0, triângulo no lugar de ReLU com viés, k5 e pool 2 fixos. As `pf_hebb_d3/d4_*` medem isso, não L4/L5 do `spf_grade`.
- **Reuso sem duplicar**: N0 tem σ por (canal, offset), então v = A(p−m) não é um operador de mapa. Mas `hebb_step` funciona tal como está sobre patches desdobrados.
  - Montar um bloco 1×1 com `bn=Identity` e `conv.weight` = Q (C_out×d×1×1).
  - Entrar com `h = v_rows[:, :, None, None]`, onde `v_rows` = (unfold com dilatação real do mapa cru com padding zero → whitened → N0).
  - Passar `bias=b_v` na competição. Q começa em `u`, ou seja, as linhas fps de norma 1.
  - Equivalência de borda: o patch cru de zeros dá A(0−m), que é igual a `b − QAm` da conv dobrada.
- Memória: 8 imagens × 576 posições × 2304 × 4 B ≈ 42 MB por lote.
- Mesma ativação nos quatro braços H0–H3: ReLU com viés da grade. A competição usa u+b_v.

## 3. Leitura e SVM

- `readouts` (L1047) e `features` (`eval/l1_amostras.py` L196): GeM p=3 (`gem` L1015) nos mapas pós-ReLU/pool de cada camada.
- `pca_fit` (L1103): só no treino. `MAX_DIM=2000` (L142): acima disso, PCA para 1999.
- `read()` (L1295) emite `L` (a camada n isolada), `cat` (últimas camadas abaixo de 2000-d; com 256×5 = 1280, entram todas) e `<nome>_b` (sem `pw_whiten`, quando há `entre_camadas`).
- `_eq`: `L_ms_eq` / `cat_ms_eq` (só com `ms`+`eq`) e `vlad<K>_eq`.
- A ablação "cat sem L4/L5" já existe: é a linha `cat` da camada 3.
- O §12 pede dimensão comum ≤256: falta uma leitura `cat_eq<D>` (PCA no treino), em G2.
- `score_full` (L961): `pw_whiten` (`eval/svm.py` L365, PCA+shrink 0,1+L2) → `fit_svm` (`core/svm.py` L143: SVC linear, C=100, OVO, sem teto de iterações) → `compute_metrics` (`core/metrics.py` L46: kappa, acc balanceada, f1).
- `pred` é calculado (L965) e **descartado**; não há recall por classe nem IDs.
- **Dá para salvar (§11)**: opt-in `encdeep.erros: true` → `score_full` devolve `pred` e o `decision_function` OVO (sem tratar como probabilidade). G2 grava `<csv>_erros/<arm>_s<seed>_L<n>.npz` com IDs de `ParasiteDataset(set_name="validation", ...).samples` (mesma ordem de `load_images`, `eval/l1_fast.py` L73), y, pred, scores e recall por classe.
- Regra "um SVM, uma métrica": não duplicar, só estender `score_full`.

## 4. Filas

**Athena** — o consumidor é um laço bash inline na tmux `fila_athena`. Não há script nem lock.
- Pega a primeira linha não-`#` de `/exps/moliveira/queues/fila_athena.txt`. Só lança se mx ≤ 10800 MiB e load < 86.
- Apaga a linha com `sed -i` (primeiro casamento exato) e anota em `fila_athena.txt.feitos`.
- Abre a tmux `<nome>` com `LD_LIBRARY_PATH=<env>/lib`, `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`, `GPU_CAP_MIB=4200`, `PYTHONPATH=/exps/moliveira/queues/gpucap`, `OMP/MKL/OPENBLAS=2` e `nice -n 10 python -m experiments.ray.launch <yaml>`. Log em `queues/logs/sol.<nome>.log`. Dorme 600 s depois de lançar, 300 s se não lançou.
- Formato da linha: caminho relativo ao repo (`experiments/autoencoder/X.yaml`).
- Fila hoje: `disc_md_pf_hebb_rand_d4.yaml`, `disc_md_pf_res448_off.yaml`.
- **Não há linha de benchmark na fila**: o benchmark roda à parte em `bench_athena`.
- **Protocolo seguro**: uma única escrita atômica logo após um lançamento (durante o sleep de 600 s). Usar `echo ... >>` se não houver benchmark; se houver, `sed -i '/bench/i <linha>'` acima do primeiro. Nunca reescrever o arquivo inteiro, porque o consumidor também faz `sed -i`.
- O pedido "na frente dos benchmarks, próximos a rodar" se cumpre inserindo **antes de `pf_res448_off`**, ou depois dele se essa ordem já estiver autorizada. Essa decisão é do A0.

**Jaci** — `fila3` é um script (`~/exps/tmp/fila3.txt`, sessão `spf_jaci_fila3`, log `~/exps/queues/logs/spf_fila3.log`) com a lista `PEND` de itens que viram `disc_md_<item>_jaci.yaml`.
- O par remoto é a sessão `spifil_explore`. A inserção é **por mensagem a ele** com o nome do YAML `_jaci`. Não se edita o script remoto. O rsync é do usuário.

**Células relacionadas**:

| Célula | Estado | Resultado |
|---|---|---|
| pf_hebb_d3_on, d4_off, d4_on | rodando athena | `results_sol/pool_foco/svm_disc_md_pf_hebb_*` (parcial) |
| pf_hebb_d3_off | rodando jaci | `~/exps/results_sol/pool_foco/` (jaci) |
| pf_hebb_rand_d3 | pronto | pool_foco, 4 CSV |
| pf_hebb_rand_d4 | próximo na fila athena | — |
| sol_hebb3_conf | pronto (5 bases × 3 splits) | `results_sol/alternative_encoders/svm_disc_md_sol_hebb3_conf_*` |
| leitura_a2 | **parcial**: split 1, sem remote_sensing; fila3 tem gate | `results_sol/pooling/` (não pool_foco). gem/gem_p2/ms, eq, w 64/256, depth 3, pool 2 |
| pf_res320_off | pronto | pool_foco `_s0320_..._sh{0,1}` |
| pf_res320_conf | rodando jaci | espelho `recebido_jaci/.../pool_foco` |
| pf_res448_off | último na fila athena | — |

Todas as `pf_hebb_*` estão no caminho `alt_encoders` (BN, triângulo), em L3/L4. **Nenhuma é a extensão L4/L5 do tronco** `spf_grade`: são dependência histórica, não duplicata.

## 5. Donos de arquivo (G1, sem escrita compartilhada)

`flim/` não tem módulo de plasticidade (16 arquivos; nada de hebb, clapp ou sfa). Os módulos propostos são novos e ficam em `flim/`, porque são aprendizado de camada e não leitura:

| Dono | Arquivo novo | Assinatura |
|---|---|---|
| H1 | `flim/local_plasticity.py` | `learn_layer_hebb(v_batches: Iterable[Tensor(N,d)], Q0: Tensor(C,d), b_v: Tensor(C), spec: dict) -> tuple[Tensor, dict]`. Importa `hebb_step` de `eval.alt_encoders`, sem copiar. |
| C1 | `flim/local_predictive.py` | `learn_layer_clapp(pairs: Iterable[(Xa_rows, Xb_rows, q)], W0: Tensor(C,d), b0: Tensor(C), grid: tuple[int,int], spec: dict) -> tuple[Tensor, Tensor, dict]`, mais `pool_op(grid, out_size) -> (U, Ut)` explícito, C0 = (1,1) e C1 = (2,2). |
| S1 | `flim/slow_features.py` | `fit_sfa(u_a: Tensor(M,C), u_b: Tensor(M,C), W0: Tensor(C,d), b0: Tensor(C), spec: dict) -> tuple[Tensor, Tensor, dict]` (W_novo = W0·R_all; b_novo = (b0−μ)·R_all). F-U opcional no mesmo dono: `fit_fu(A, B, W0, spec) -> (W, dict)`. |
| A0 | `eval/encdeep.py` (novo) + edições em `eval/spf_grade.py` (hook, kernels e dilatação por camada, persistência L{n}, `cat_eq`, `erros`) | `deep_mechanism(spec, n, tags) -> Callable or None`; `view_pairs(train, to_float, spec, seed)` (vistas fotométricas leves via `core/data/train_aug.TrainAug`); `layer_rows(encoder, x, fold, mean, stdev, layer) -> v_rows`. |

- Os três módulos recebem **tensores já em v-space** e não importam `spf_grade`. Assim os checks do §14 rodam em CPU com tensores minúsculos.
- Workers não tocam `spf_grade.py`, `schema.py`, `core/svm.py` nem a fila.
- Restrição: código que jobs em andamento importam (`spf_grade`, `alt_encoders`) só muda em G2 e opt-in. Os jobs atuais já carregaram o módulo, mas uma `launch` futura da fila relê o arquivo. Desligado tem que dar igual byte a byte.

## 6. Viabilidade (conv `same`, MaxPool 3 sem padding, pools 2-2-2-2-1)

| Entrada | Geometria | Mapa de entrada da L4 (= saída L3) | L4 conv → pós-pool | L5 | RF L3 / L4 / L5 |
|---|---|---|---|---|---|
| 200 | receita k5 | 24×24 | 24 → 11 | 11 | 43 / 91 / 155 |
| 200 | G0 (k3 em L4/L5) | 24 | 24 → 11 | 11 | 43 / 75 / 107 |
| 200 | G1 (sem pool após L4) | 24 | 24 → 24 | 24 | 43 / 59 / 75 |
| 200 | G2 (G1 + dil 2 na L5) | 24 | 24 | 24 | 43 / 59 / 91 |
| 320 | receita k5 / G0 | 39 | 39 → 19 | 19 | iguais |
| 320 | G1 / G2 | 39 | 39 | 39 | 43 / 59 / 75 (91) |

- Atenção: **G1 tem campo receptivo menor que G0**, porque o pool aumenta o RF. G2 recupera só até 91. Isso é ganho de resolução, não de contexto, e precisa ir no relato.
- 2×2 adaptativo (C1) cabe em todos os casos: o pré-pool da L4 é 24 ou 39, e L5 é 11 ou maior. Os bins de 11 não são iguais (5/6 com sobreposição), então Uᵀ precisa pesar pela área real.
- Patches 3×3×256: d = 2304, 9,2 KB/linha fp32. Denso por imagem a 200: 576 × 2304 × 4 = 5,3 MB (a 320: 14 MB).
  - 2000 imagens dariam 10,6 GB → **streaming obrigatório**.
  - Gram fp64: 40,5 MiB. Com k5 (d=6400): 312 MiB.
  - Candidatos fps: ≤ 500 × 50 sementes → duplicatas na grade 24×24, que o `Ctx.keep` remove. Sobra o bastante para 256.
  - P do C1: 1024² fp32 = 4 MiB.
- Teto: `GPU_CAP_MIB=4200` por job na fila athena. Lote de patch precisa caber nisso.

## 7. Chaves YAML propostas (`runner_args.amostra.encdeep`)

```yaml
amostra:
  depth: 5
  widths: [256]
  kernels: [5]
  pool: [2, 2, 2, 2, 1]
  entre_camadas: true
  methods: [fps]
  controls: [random_z, orth]
  encdeep:
    kernels: [5, 5, 5, 3, 3]       # por camada; sobrescreve `kernels` só nestas posições
    dilacao: [1, 1, 1, 1, 1]       # G2: [1,1,1,1,2]
    camadas: [4]                   # depois [4, 5]
    mecanismos: [hebb]             # vira tag ~hebb; o braço sem tag é o zero
    hebb: {steps: 300, batch: 8, images: 2000, lr: 0.005, inv_temp: 0.65, delta_norm: tensor}
    clapp: {pares: ..., lr: ..., lr_p: ..., saida: [1, 1]}   # C1: [2, 2]
    sfa: {lentos: 128, residuo: 128, tol: 1e-6, piso_rms: 1e-3}
    salvar: true                   # L{n} + trunk_hash
    erros: true                    # npz de IDs/pred/scores
    dim_eq: 256                    # leitura cat_eq / L_eq
```

- YAML: `experiments/autoencoder/disc_md_encdeep_<braço>.yaml` (+ `_jaci`), CSV em `/exps/moliveira/results_sol/encdeep/`, `fontes` em diretório novo `/exps/moliveira/fontes_filtros/encdeep_<...>`.
- **O que o schema valida**: `experiments/ray/schema.py` L102–113 só checa as chaves de `runner_args` (`amostra` está na lista), e `_check_eval_args` (L135) exige `artifacts_dir, results_csv, l1_sizes, l1_superpixels, l1_grow, amostra` para `spf_grade` (L170).
- **O conteúdo de `amostra` não é validado**, e `spf_grade` ignora chave desconhecida em silêncio. O §15 proíbe isso, então G2 precisa de uma validação mínima de `amostra.encdeep` em `eval/encdeep.py` (ou no schema), que falhe com chave ou mecanismo desconhecido.
- Preflight e dry-run escrevem `jobs/*.yaml` em `output.work_dir` (`eval.py` L497): usar um work_dir novo.

## Bloqueios e itens inviáveis como escritos

1. **"Recuperar checkpoint L3 existente" (§5.1)**: não existe. Dá para reconstruir de forma determinística a partir do `.pt` de fontes, e G2 passa a persistir com hash.
2. **"Aproveitar SoftHebb L3/L4 prontos" (§6.1)**: os prontos (`sol_hebb3_conf`, `pf_hebb_*`) são outro tronco (BN, triângulo, sem ZCA). Não são o fatorial H0–H3 no tronco `spf_grade`, então o fatorial L4 também precisa rodar, não só L5.
3. **Kernel 3×3 em L4/L5** muda a geometria em relação à receita k5. Os controles k3 estáticos são novos; a receita k5 histórica fica só como referência.
4. **Dilatação e kernel por camada** não existem em `chain`; são mudança mínima de G2.
5. **leitura_a2** está incompleta (sem remote_sensing, sem splits 2 e 3). M1 precisa esperá-la ou declarar uma receita nova.
6. **Fila athena sem lock**: a inserção é por janela de tempo, não por exclusão mútua.
