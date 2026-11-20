# loadtune no athena (GPUs 2 e 3), rede 2

**Resposta curta:** use `num_workers=4`, `prefetch_factor=2`, `pin_memory=True` e `persistent_workers=True`, com **2 runs por GPU**. Nas bases pequenas (remote_sensing, Larvae, flowers) use `num_workers=2`. Isso está gravado em `/exps/moliveira/queues/loadtune/athena.json`.

## O que eu medi

- **Ferramenta:** `loadtune` 0.3.0 (`profile_session`). O passo medido é o de treino de verdade: `ParasiteDataModule` com batch 32, 200 px, `no_crop`, cache `/exps/moliveira/image_cache`, e o `training_step` do `AutoEncoderFlimModule` com `parasite_aug` na GPU, seguido de backward e AdamW. A cópia para a GPU usa a mesma função do Lightning, com `non_blocking=True`.
- **Pesos:** encoder do teste em `/exps/moliveira/tmp/q1/<metodo>/aid_split1_pct50/L1_grow` e `L6_grow`, na fase L6 congelada.
- **Grade:** `num_workers` {2, 4, 8, 12, 16} × `prefetch_factor` {2, 4, 8} × `pin_memory` {não, sim}, sempre com `persistent_workers=True`. Testei 1 run e 2 runs por GPU (4 runs ao mesmo tempo nas duas GPUs), e também 3 por GPU.
- **Bases:** food101 (grande), Protozoan (grande) e remote_sensing (pequena). Todas as bases estão no cache no **mesmo formato**: 3×200×200 uint8, 0,12 MB por imagem. Por isso carregar uma imagem custa o mesmo em qualquer base. O que muda entre elas é só o tamanho, ou seja, quantos passos cabem em cada época.

**Ruído.** Até ~04:05, dois testes `run_growth` da rede 4 rodavam nas GPUs 2 e 3. Além disso, a GPU 2 esquenta até 88–90 °C e reduz o clock sozinha (throttle térmico). Nessas condições, o img/s chegou a variar 2× para a mesma config. Para contornar:
- rodei cada config duas vezes, na ordem direta e na inversa, para cancelar a deriva;
- usei como medida principal a **espera por dados**, que não depende da GPU;
- repeti os casos-chave depois que os testes acabaram (linhas "limpo").

## Tabela (img/s por run · espera por dados por passo)

| caso | ruído? | nw=2 pin | nw=4 pin | nw=8 pin | nw=4 **sem** pin |
|---|---|---|---|---|---|
| food101 AE L1, 1 run | limpo | 866 · 1,2% | 866 · 1,3% | 817 · 1,6% | 678 · 13,8% |
| food101 SimCLR L1, 1 run | limpo | 1082 · 1,8% | 1034 · 1,4% | 1013 · 1,4% | 778 · 16,3% |
| remote_sensing AE L1, 1 run | limpo | 741 · 15,3% | 724 · 17,6% | 665 · 19,4% | 621 · 20,7% |
| remote_sensing SimCLR L1, 1 run | limpo | 779 · 18,1% | 807 · 18,2% | 659 · 23,1% | 559 · 19,8% |
| Protozoan AE L1, 1 run | parte com ruído | 827 · 1,7% | 804 · 1,9% | 796 · 1,5% | 633 · 16,4% |
| food101 AE L6, 1 run | com ruído | 443 · 0,9% | 456 · 0,8% | 454 · 1,0% | 386 · 11,2% |
| food101 SimCLR L6, 1 run | com ruído | 1447 · 1,2% | 1500 · 1,2% | 1194 · 1,5% | 672 · 19,4% |
| food101 VICReg L1, 1 run | com ruído | 421 · 0,6% | 282 · 0,3% | 322 · 0,8% | 387 · 11,1% |
| food101 AE L1, **2 por GPU** (4 runs) | limpo | 448 · 0,4% | 424 · 0,4% | 409 · 0,4% | — |
| food101 SimCLR L1, **2 por GPU** (4 runs) | limpo | 872 · 1,4% | 854 · 1,0% | 850 · 1,0% | — |
| food101 AE L1, **3 por GPU** | limpo | 300 · 0,7% | — | — | — |
| food101 SimCLR L1, **3 por GPU** | limpo | 666 · 2,1% | — | — | — |

Com `num_workers` de 12 e 16 o resultado foi igual ou pior que com 8 em todos os casos. Já `prefetch_factor` 4 ou 8 nunca ganhou do 2 além do ruído. Esses números estão nos dados brutos.

## O que isso quer dizer

1. **Carregar dados é barato.** Cada imagem já está pronta no cache. Um worker sozinho entrega uns 500 img/s, então 2 workers já seguram 1 run, e 4 dão folga.
2. **Mais workers não ajuda.** De 8 para cima, o img/s cai um pouco. Cada worker a mais custa ~1 GB de RAM: um run com 16 workers ocupa 18 GB.
3. **`pin_memory=True` é o único botão que importa.** Ele deixa a memória pronta para a cópia rápida até a GPU. Sem ele, a espera por dados sobe de ~1% para 11–16% e o run fica 20–30% mais lento.
4. **`prefetch_factor`** (quantos lotes cada worker deixa prontos na fila) não muda nada, então fica no 2.
5. **Bases pequenas esperam no começo de cada época.** O remote_sensing tem 344 imagens de treino, ou seja, 10 passos por época. A cada época a fila esvazia e é preciso esperar o primeiro lote, o que dá 15–18% de espera. Pôr mais workers piora isso, então nessas bases fico com 2.
6. **2 runs por GPU vale a pena.** No AE, a GPU já fica cheia com 1 run: 2 runs dão o mesmo total (~900 img/s por GPU). No SimCLR, 1 run deixa a GPU folgada: 2 runs somam ~1730 img/s contra ~1060 com 1 run, **+60%**. Com 3 por GPU nada engasga (a espera fica em 2% ou menos), mas o ganho é pequeno: +16% no SimCLR e 0% no AE. Cada run também fica 1,5× mais lento, e a L6 (2048 canais, expansor 8192 do VICReg) gasta muito mais memória. Por isso fico com **2**.
7. **CPU:** cada run gasta ~1,5–2,5 núcleos ao todo (processo principal mais workers). Com 4 runs dá uns 10 núcleos, bem abaixo do teto de ~80% dos núcleos livres. `omp_threads` ficou em 8, o mesmo valor usado nas medições.

## Escolha (gravada no JSON)

| item | num_workers | prefetch_factor | pin_memory | persistent_workers | runs_por_gpu | omp_threads |
|---|---|---|---|---|---|---|
| default, g1–g4, todos os métodos | 4 | 2 | True | True | 2 | 8 |
| `gN/<método>/{remote_sensing, Larvae, flowers}` | 2 | 2 | True | True | 2 | 8 |

- `non_blocking` não entrou no JSON porque o Lightning já copia com `non_blocking=True`.
- **g1, g3, g4 não foram medidos direto, só por analogia.** O custo dos dados é igual para qualquer rede (mesmo cache, mesmo lote). A L1 de g1 e de g4 é igual à de g2 (32 canais, k5), e a L1 de g3 é mais pesada (64 canais), então pede ainda menos dados por segundo. As camadas fundas pedem menos ainda.
- Dados brutos e o script: `/exps/moliveira/queues/loadtune/athena_bruto/`.
