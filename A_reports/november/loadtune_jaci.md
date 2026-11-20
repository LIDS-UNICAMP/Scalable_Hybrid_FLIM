# loadtune no jaci: como alimentar a GPU (13/11/2026)

**Resposta:** 4 workers, prefetch 2, `pin_memory` ligado, workers persistentes, e **até 4 runs juntos** na GPU.
Quem limita o treino é a GPU, não a leitura das imagens. O único botão que faz diferença é o `pin_memory` (+27%).

## Como medi

- Ferramenta: `loadtune` 0.3.0 (`profile_session`), instalada no env `scalable_FLIM` do jaci.
- O que roda: o `ParasiteDataModule` de verdade (cache de 200 px, `no_crop`, split 1, 50%, batch 32) e o passo de verdade do
  `AutoEncoderFlimModule` da rede 2 (32-64-128-256-384-2048): AE (`_recon_loss`) e SimCLR (2 vistas `parasite_aug` na GPU).
  L1 = só a 1ª camada; L6 = as 6 camadas, tudo treinando. Pesos: L1 do `g2_ae` do jaci, L6 do `q1/ae/aid` do athena.
- Cada medida: 20 passos de aquecimento + 300 medidos. "Espera" = fração do tempo que a GPU ficou esperando dado.
- `non_blocking` foi ligado junto com o `pin_memory` (sem memória fixa ele não faz nada).
- Batch, lr e arquitetura: intocados.
- Script e dados brutos no jaci: `/dados/home/moliveira/exps/tmp/loadtune/` (`lt_bench.py`, `r1/r2/r3.jsonl`).

## O que deu (1 run sozinho)

aid, AE, L1 — grade completa (30 combinações):

| pin_memory | workers 4…24, prefetch 2/4/8 | espera |
|---|---|---|
| desligado | ~1.490 img/s em todas | ~15% |
| **ligado** | ~1.840–1.900 img/s em todas (4 workers é o melhor) | 2–4% |

Mais workers ou prefetch maior não ajudam: com 4 workers a fila já está cheia.

Outros casos (pin ligado, 4 workers, prefetch 2):

| base (nº imagens) | método | camada | img/s | espera | GPU |
|---|---|---|---|---|---|
| aid (10.000) | SimCLR | L1 | 4.114 | 3% | 88% |
| aid | AE | L6 | 1.204 | 1% | 97% |
| aid | SimCLR | L6 | 1.945 | 1% | 95% |
| Protozoan (37.376) | AE | L1 | 1.920 | 0,4% | 99% |
| Protozoan | SimCLR | L6 | 1.981 | 0,4% | 96% |
| ham10000 (10.015) | AE | L1 | 1.866 | 3% | 99% |
| Eggs (12.691) | AE | L1 | 1.887 | 2% | 94% |

Em todos: sem pin cai ~20–25%; 8 ou 16 workers dão o mesmo que 4 (dentro do ruído).
Base grande ou pequena dá o mesmo: as imagens já estão no cache de 200 px, então cada imagem custa igual.
A base pequena só espera um pouquinho mais (1–3%) porque vira a época mais vezes.

## Vários runs juntos (SimCLR L1 aid, o caso que mais pede dado)

| runs | img/s somando todos | espera | CPU da máquina | RAM nossa |
|---|---|---|---|---|
| 1 | 4.114 | 3% | 4% | ~4 GB |
| 3 | ~4.650 | 2% | 6% | ~10 GB |
| 4 | ~4.700 | 2% | 7% | ~15 GB |

A GPU já fica em 90–100% com 1 run. Juntar runs não deixa o total muito maior, mas também não engasga:
a espera continua em 2%. Cada run usa só 1–2 GB de GPU, então 4 cabem com folga no teto de 88 GB.
Com 4 runs, as pausas de cada um (validação, sonda) são preenchidas pelos outros.

4 runs misturados (o que a fila faz de verdade), pin ligado, 4 workers:

| run | img/s (sozinho → com 3 vizinhos) | espera |
|---|---|---|
| aid AE L1 | ~1.900 → 715 | 1% |
| Protozoan SimCLR L1 | ~4.100 → 752 | 0,2% |
| Eggs AE L6 | ~1.200 → 488 | 0,5% |
| ham10000 SimCLR L6 | ~1.950 → 589 | 1% |

Ninguém passa fome de dado; sem pin a espera sobe para 6–9%. 8, 16 ou 24 workers dão o mesmo ou um pouco menos que 4.

## Escolha (gravada em `jaci:/dados/home/moliveira/exps/queues/loadtune/jaci.json`)

| chave | valor | por quê |
|---|---|---|
| num_workers | 4 | já enche a fila; mais não ajuda |
| prefetch_factor | 2 | 4 e 8 dão o mesmo |
| pin_memory | true | +27%; é o único que importa |
| persistent_workers | true | não recria workers a cada época |
| runs_por_gpu | 4 | espera fica em 2% e a memória sobra |
| omp_threads | 8 | o valor de hoje (`cpus_per_experiment`); não foi medido |

Vale igual para AE, SimCLR e VICReg (VICReg = 2 vistas, como o SimCLR) e para as redes 1, 3 e 4:
o lado dos dados é o mesmo (mesma base, mesmo batch, mesmos 200 px); só muda a conta na GPU,
e a conta mais leve (L1) já não passava fome.

`non_blocking`: ligado na medição, mas fica fora do JSON (a chave viraria `--data.init_args.non_blocking`, que não existe).
Nem precisa: o Lightning já copia para a GPU com `non_blocking=True` (`lightning/fabric/utilities/apply_func.py:103`);
basta o `pin_memory` ligado para a cópia ficar assíncrona.
