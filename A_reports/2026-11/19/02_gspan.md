# Técnica 1 — gSpan: subgrafos frequentes

gSpan oficial (betterenvi/gSpan) por base no top-f por κ de **validação**; cada padrão é recontado nos grafos **direcionados** originais (monomorfismo). freq_top = fração das top da base que contém o padrão; *balanceada* = média das bases (cada base pesa igual). enrichment = freq_top − freq_demais (balanceadas). dz_κ = diferença de κ_val padronizado (z na base) entre quem contém e quem não contém. q = Fisher exato unilateral (top > demais), corrigido por Benjamini–Hochberg.

**Aviso:** frequência alta ≠ bloco bom. O que sugere bloco bom é enrichment > 0 com q baixo E dz_κ > 0 em várias bases.

## Top 10% (n top por base: {'aid': 8, 'coconut_trees': 5, 'flowers': 20, 'ham10000': 23, 'remote_sensing': 62})

1552 padrões fechados (de 3327 minerados; 2–8 nós). Uma entrada por candidato (a fidelidade de maior κ_val).

### Mais frequentes (balanceado)

| pattern | kind | freq_top_balanced | freq_rest_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.836 | 0.608 | 0.228 | 5 | 0.436 | 0.000 |
| conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.680 | 0.683 | -0.003 | 5 | 0.298 | 0.774 |
| conv3→cat; conv5→cat; pw1→cat | ramificado | 0.680 | 0.697 | -0.017 | 5 | 0.224 | 0.820 |
| conv3→cat; pw1→cat | ramificado | 0.680 | 0.708 | -0.028 | 5 | 0.151 | 0.820 |
| pw1 → cat | par | 0.680 | 0.721 | -0.041 | 5 | -0.074 | 0.820 |
| skip1 → add | residual | 0.679 | 0.752 | -0.073 | 5 | 0.125 | 0.774 |
| add → maxpools2 | par | 0.607 | 0.529 | 0.078 | 4 | 0.208 | 0.625 |
| maxpools2 → conv5 | par | 0.586 | 0.513 | 0.073 | 4 | 0.185 | 0.027 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.570 | 0.428 | 0.142 | 4 | 0.364 | 0.027 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.570 | 0.438 | 0.132 | 4 | 0.276 | 0.027 |
| maxpools2 → pw1 → cat | sequência | 0.570 | 0.447 | 0.123 | 4 | 0.176 | 0.027 |
| maxpools2 → skip1 → add | residual | 0.559 | 0.474 | 0.086 | 4 | 0.340 | 0.063 |
| skip1 → add → maxpools2 | residual | 0.545 | 0.472 | 0.074 | 4 | 0.324 | 0.683 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.498 | 0.481 | 0.016 | 3 | 0.259 | 0.497 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat | ramificado | 0.498 | 0.486 | 0.012 | 3 | 0.236 | 0.497 |
| cat→bn; conv3→cat; pw1→cat | ramificado | 0.498 | 0.499 | -0.001 | 3 | 0.149 | 0.497 |
| pw1 → cat → bn | sequência | 0.498 | 0.510 | -0.013 | 3 | 0.033 | 0.497 |
| skip1→add; maxpools2→skip1; maxpools2→conv3 | residual | 0.392 | 0.288 | 0.105 | 2 | 0.453 | 0.027 |
| cat→bn; pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.372 | 0.260 | 0.112 | 2 | 0.413 | 0.368 |
| maxpools2 → pw1 → cat → bn | sequência | 0.372 | 0.267 | 0.105 | 2 | 0.342 | 0.430 |

### Enriquecidos nas melhores (enrichment>0 e dz_κ>0), ordenados por nº de bases ≥50%

| pattern | kind | freq_top_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.836 | 0.228 | 5 | 0.436 | 0.000 |
| add → maxpools2 | par | 0.607 | 0.078 | 4 | 0.208 | 0.625 |
| maxpools2 → conv5 | par | 0.586 | 0.073 | 4 | 0.185 | 0.027 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.570 | 0.142 | 4 | 0.364 | 0.027 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.570 | 0.132 | 4 | 0.276 | 0.027 |
| maxpools2 → pw1 → cat | sequência | 0.570 | 0.123 | 4 | 0.176 | 0.027 |
| maxpools2 → skip1 → add | residual | 0.559 | 0.086 | 4 | 0.340 | 0.063 |
| skip1 → add → maxpools2 | residual | 0.545 | 0.074 | 4 | 0.324 | 0.683 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.498 | 0.016 | 3 | 0.259 | 0.497 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat | ramificado | 0.498 | 0.012 | 3 | 0.236 | 0.497 |
| skip1→add; maxpools2→skip1; maxpools2→conv3 | residual | 0.392 | 0.105 | 2 | 0.453 | 0.027 |
| cat→bn; pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.372 | 0.112 | 2 | 0.413 | 0.368 |
| maxpools2 → pw1 → cat → bn | sequência | 0.372 | 0.105 | 2 | 0.342 | 0.430 |
| selu → add | par | 0.400 | 0.056 | 1 | 0.142 | 0.495 |
| selu→add; skip1→add | residual | 0.394 | 0.119 | 1 | 0.220 | 0.158 |
| skip1→add; maxpools2→skip1; maxpools2→conv5 | residual | 0.392 | 0.123 | 1 | 0.299 | 0.027 |
| selu → maxpools2 | par | 0.390 | 0.092 | 1 | 0.020 | 0.028 |
| zca → silu | par | 0.377 | 0.097 | 1 | 0.152 | 0.246 |
| add→maxpools2; selu→add; skip1→add | residual | 0.359 | 0.188 | 1 | 0.416 | 0.065 |
| selu → add → maxpools2 | sequência | 0.359 | 0.141 | 1 | 0.251 | 0.218 |

Exemplos de arquiteturas (top) que contêm os 5 primeiros:

- `maxpools2 → conv3`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`
- `add → maxpools2`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00043/f45`, `modehb_hybrid/aid/c00044/f15`
- `maxpools2 → conv5`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00044/f15`
- `pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00044/f15`
- `pw1→cat; maxpools2→pw1; maxpools2→conv3`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00044/f15`

## Top 20% (n top por base: {'aid': 17, 'coconut_trees': 5, 'flowers': 34, 'ham10000': 42, 'remote_sensing': 124})

1057 padrões fechados (de 2491 minerados; 2–8 nós). Uma entrada por candidato (a fidelidade de maior κ_val).

### Mais frequentes (balanceado)

| pattern | kind | freq_top_balanced | freq_rest_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.778 | 0.595 | 0.184 | 5 | 0.436 | 0.002 |
| conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.692 | 0.680 | 0.012 | 5 | 0.298 | 0.514 |
| conv3→cat; conv5→cat; pw1→cat | ramificado | 0.692 | 0.696 | -0.004 | 5 | 0.224 | 0.547 |
| conv3→cat; pw1→cat | ramificado | 0.692 | 0.707 | -0.016 | 5 | 0.151 | 0.547 |
| pw1 → cat | par | 0.692 | 0.723 | -0.031 | 5 | -0.074 | 0.638 |
| skip1 → add | residual | 0.691 | 0.748 | -0.057 | 5 | 0.125 | 0.362 |
| add → maxpools2 | par | 0.558 | 0.531 | 0.027 | 5 | 0.208 | 0.718 |
| maxpools2 → conv5 | par | 0.543 | 0.522 | 0.021 | 3 | 0.185 | 0.274 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.534 | 0.423 | 0.111 | 3 | 0.364 | 0.062 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.534 | 0.434 | 0.100 | 3 | 0.276 | 0.062 |
| maxpools2 → pw1 → cat | sequência | 0.534 | 0.445 | 0.089 | 3 | 0.176 | 0.062 |
| skip1 → add → maxpools2 | residual | 0.517 | 0.468 | 0.049 | 3 | 0.324 | 0.514 |
| maxpools2 → skip1 → add | residual | 0.514 | 0.465 | 0.049 | 3 | 0.340 | 0.036 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.508 | 0.475 | 0.032 | 3 | 0.259 | 0.144 |
| cat→bn; conv3→cat; pw1→cat | ramificado | 0.508 | 0.495 | 0.013 | 3 | 0.149 | 0.144 |
| pw1 → cat → bn | sequência | 0.508 | 0.509 | -0.001 | 3 | 0.033 | 0.144 |
| bn → zca | par | 0.438 | 0.517 | -0.079 | 2 | -0.151 | 0.718 |
| zca → silu | par | 0.384 | 0.272 | 0.111 | 1 | 0.152 | 0.460 |
| selu → maxpools2 | par | 0.374 | 0.288 | 0.086 | 1 | 0.020 | 0.144 |
| selu → add | par | 0.362 | 0.349 | 0.014 | 1 | 0.142 | 0.421 |

### Enriquecidos nas melhores (enrichment>0 e dz_κ>0), ordenados por nº de bases ≥50%

| pattern | kind | freq_top_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.778 | 0.184 | 5 | 0.436 | 0.002 |
| conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.692 | 0.012 | 5 | 0.298 | 0.514 |
| add → maxpools2 | par | 0.558 | 0.027 | 5 | 0.208 | 0.718 |
| maxpools2 → conv5 | par | 0.543 | 0.021 | 3 | 0.185 | 0.274 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.534 | 0.111 | 3 | 0.364 | 0.062 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.534 | 0.100 | 3 | 0.276 | 0.062 |
| maxpools2 → pw1 → cat | sequência | 0.534 | 0.089 | 3 | 0.176 | 0.062 |
| skip1 → add → maxpools2 | residual | 0.517 | 0.049 | 3 | 0.324 | 0.514 |
| maxpools2 → skip1 → add | residual | 0.514 | 0.049 | 3 | 0.340 | 0.036 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.508 | 0.032 | 3 | 0.259 | 0.144 |
| cat→bn; conv3→cat; pw1→cat | ramificado | 0.508 | 0.013 | 3 | 0.149 | 0.144 |
| zca → silu | par | 0.384 | 0.111 | 1 | 0.152 | 0.460 |
| selu → maxpools2 | par | 0.374 | 0.086 | 1 | 0.020 | 0.144 |
| selu → add | par | 0.362 | 0.014 | 1 | 0.142 | 0.421 |
| silu → maxpools2 | par | 0.342 | 0.129 | 1 | 0.086 | 0.144 |
| cat→bn; pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.339 | 0.078 | 1 | 0.413 | 0.478 |
| maxpools2 → pw1 → cat → bn | sequência | 0.339 | 0.070 | 1 | 0.342 | 0.514 |
| dbn → selu | par | 0.368 | 0.090 | 0 | 0.016 | 0.122 |
| maxpools2 → skip1 → add → maxpools2 | residual | 0.357 | 0.057 | 0 | 0.269 | 0.120 |
| selu→add; skip1→add | residual | 0.349 | 0.083 | 0 | 0.220 | 0.053 |

Exemplos de arquiteturas (top) que contêm os 5 primeiros:

- `maxpools2 → conv3`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00023/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`
- `conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`
- `add → maxpools2`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00043/f45`, `modehb_hybrid/aid/c00044/f15`
- `maxpools2 → conv5`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00044/f15`
- `pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1`: `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`, `modehb_hybrid/aid/c00044/f15`

## Top 30% (n top por base: {'aid': 33, 'coconut_trees': 6, 'flowers': 53, 'ham10000': 64, 'remote_sensing': 193})

3371 padrões fechados (de 6736 minerados; 2–8 nós). Uma entrada por candidato (a fidelidade de maior κ_val).

### Mais frequentes (balanceado)

| pattern | kind | freq_top_balanced | freq_rest_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.767 | 0.527 | 0.240 | 5 | 0.436 | 0.002 |
| skip1 → add | residual | 0.730 | 0.730 | -0.000 | 5 | 0.125 | 0.308 |
| conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.726 | 0.644 | 0.082 | 5 | 0.298 | 0.016 |
| conv3→cat; conv5→cat; pw1→cat | ramificado | 0.726 | 0.664 | 0.062 | 5 | 0.224 | 0.026 |
| conv3→cat; pw1→cat | ramificado | 0.726 | 0.677 | 0.049 | 5 | 0.151 | 0.026 |
| pw1 → cat | par | 0.726 | 0.699 | 0.027 | 5 | -0.074 | 0.048 |
| maxpools2 → conv5 | par | 0.555 | 0.505 | 0.050 | 4 | 0.185 | 0.305 |
| add → maxpools2 | par | 0.547 | 0.535 | 0.012 | 4 | 0.208 | 0.402 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.545 | 0.431 | 0.114 | 4 | 0.259 | 0.010 |
| cat→bn; conv3→cat; pw1→cat | ramificado | 0.545 | 0.455 | 0.090 | 4 | 0.149 | 0.010 |
| pw1 → cat → bn | sequência | 0.545 | 0.474 | 0.071 | 4 | 0.033 | 0.010 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.536 | 0.377 | 0.159 | 3 | 0.364 | 0.017 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.536 | 0.392 | 0.145 | 3 | 0.276 | 0.017 |
| maxpools2 → pw1 → cat | sequência | 0.536 | 0.407 | 0.129 | 3 | 0.176 | 0.017 |
| skip1 → add → maxpools2 | residual | 0.511 | 0.454 | 0.057 | 3 | 0.324 | 0.135 |
| maxpools2 → skip1 → add | residual | 0.475 | 0.450 | 0.025 | 3 | 0.340 | 0.005 |
| bn → zca | par | 0.457 | 0.505 | -0.048 | 2 | -0.151 | 0.254 |
| silu → add | par | 0.398 | 0.405 | -0.006 | 1 | -0.058 | 0.126 |
| zca → silu | par | 0.388 | 0.243 | 0.144 | 1 | 0.152 | 0.083 |
| selu → add | par | 0.384 | 0.344 | 0.039 | 1 | 0.142 | 0.459 |

### Enriquecidos nas melhores (enrichment>0 e dz_κ>0), ordenados por nº de bases ≥50%

| pattern | kind | freq_top_balanced | enrichment | n_datasets_ge50 | dz_kappa_mean | fisher_min_q |
|---|---|---|---|---|---|---|
| maxpools2 → conv3 | par | 0.767 | 0.240 | 5 | 0.436 | 0.002 |
| conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.726 | 0.082 | 5 | 0.298 | 0.016 |
| conv3→cat; conv5→cat; pw1→cat | ramificado | 0.726 | 0.062 | 5 | 0.224 | 0.026 |
| conv3→cat; pw1→cat | ramificado | 0.726 | 0.049 | 5 | 0.151 | 0.026 |
| maxpools2 → conv5 | par | 0.555 | 0.050 | 4 | 0.185 | 0.305 |
| add → maxpools2 | par | 0.547 | 0.012 | 4 | 0.208 | 0.402 |
| cat→bn; conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat | ramificado | 0.545 | 0.114 | 4 | 0.259 | 0.010 |
| cat→bn; conv3→cat; pw1→cat | ramificado | 0.545 | 0.090 | 4 | 0.149 | 0.010 |
| pw1 → cat → bn | sequência | 0.545 | 0.071 | 4 | 0.033 | 0.010 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.536 | 0.159 | 3 | 0.364 | 0.017 |
| pw1→cat; maxpools2→pw1; maxpools2→conv3 | ramificado | 0.536 | 0.145 | 3 | 0.276 | 0.017 |
| maxpools2 → pw1 → cat | sequência | 0.536 | 0.129 | 3 | 0.176 | 0.017 |
| skip1 → add → maxpools2 | residual | 0.511 | 0.057 | 3 | 0.324 | 0.135 |
| maxpools2 → skip1 → add | residual | 0.475 | 0.025 | 3 | 0.340 | 0.005 |
| zca → silu | par | 0.388 | 0.144 | 1 | 0.152 | 0.083 |
| selu → add | par | 0.384 | 0.039 | 1 | 0.142 | 0.459 |
| avgpools2 → skip1 → add | residual | 0.368 | 0.035 | 1 | 0.077 | 0.112 |
| silu→add; skip1→add | residual | 0.363 | 0.059 | 1 | 0.104 | 0.043 |
| cat→bn; pw1→cat; maxpools2→pw1; maxpools2→conv3; maxpools2→conv5; maxpools2→maxpool; maxpool→pw1 | ramificado | 0.354 | 0.134 | 1 | 0.413 | 0.084 |
| maxpools2 → pw1 → cat → bn | sequência | 0.354 | 0.122 | 1 | 0.342 | 0.112 |

Exemplos de arquiteturas (top) que contêm os 5 primeiros:

- `maxpools2 → conv3`: `modehb_hybrid/aid/c00015/f5`, `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00023/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`
- `conv3→cat; conv5→cat; pw1→cat; maxpool→pw1; pw1→cat`: `modehb_hybrid/aid/c00015/f5`, `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`
- `conv3→cat; conv5→cat; pw1→cat`: `modehb_hybrid/aid/c00015/f5`, `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`
- `conv3→cat; pw1→cat`: `modehb_hybrid/aid/c00015/f5`, `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00028/f5`, `modehb_hybrid/aid/c00034/f15`
- `maxpools2 → conv5`: `modehb_hybrid/aid/c00015/f5`, `modehb_hybrid/aid/c00020/f5`, `modehb_hybrid/aid/c00025/f45`, `modehb_hybrid/aid/c00034/f15`, `modehb_hybrid/aid/c00038/f45`

## Operadores isolados (presença por grafo, top30 vs demais) — NÃO são subgrafos

| op | top30_aid | top30_coconut_trees | top30_flowers | top30_ham10000 | top30_remote_sensing | rest_aid | rest_coconut_trees | rest_flowers | rest_ham10000 | rest_remote_sensing |
|---|---|---|---|---|---|---|---|---|---|---|
| add | 0.91 | 0.67 | 0.74 | 0.91 | 0.70 | 0.87 | 0.77 | 0.85 | 0.83 | 0.69 |
| avgpools2 | 0.91 | 0.33 | 0.55 | 0.69 | 0.41 | 0.77 | 0.45 | 0.72 | 0.72 | 0.57 |
| bn | 0.91 | 0.67 | 0.77 | 0.94 | 0.77 | 0.98 | 0.91 | 0.89 | 0.94 | 0.88 |
| cat | 0.82 | 0.67 | 0.66 | 0.78 | 0.70 | 0.94 | 0.50 | 0.76 | 0.78 | 0.52 |
| conv3 | 0.94 | 1.00 | 0.91 | 0.92 | 0.98 | 0.98 | 0.86 | 0.91 | 0.98 | 0.71 |
| conv5 | 0.91 | 0.67 | 0.77 | 0.94 | 0.75 | 0.98 | 0.73 | 0.83 | 0.87 | 0.83 |
| conv7 | 0.12 | 0.33 | 0.34 | 0.27 | 0.20 | 0.48 | 0.23 | 0.41 | 0.54 | 0.38 |
| dbn | 0.52 | 0.67 | 0.74 | 0.70 | 0.66 | 0.77 | 0.68 | 0.72 | 0.74 | 0.67 |
| denseoutspifilopsblock | 0.03 | AUSENTE | AUSENTE | 0.02 | AUSENTE | 0.00 | AUSENTE | AUSENTE | 0.00 | AUSENTE |
| dw3 | 0.21 | 0.00 | 0.25 | 0.39 | 0.24 | 0.29 | 0.23 | 0.35 | 0.17 | 0.16 |
| dw5 | 0.21 | 0.17 | 0.25 | 0.44 | 0.14 | 0.44 | 0.32 | 0.28 | 0.41 | 0.29 |
| dw7 | 0.27 | 0.33 | 0.15 | 0.14 | 0.30 | 0.31 | 0.18 | 0.31 | 0.26 | 0.28 |
| gatedskip | 0.06 | AUSENTE | AUSENTE | 0.02 | AUSENTE | 0.00 | AUSENTE | AUSENTE | 0.02 | AUSENTE |
| gelu | 0.42 | 0.33 | 0.62 | 0.45 | 0.62 | 0.73 | 0.73 | 0.72 | 0.70 | 0.59 |
| maxpool | 0.82 | 0.67 | 0.66 | 0.78 | 0.70 | 0.90 | 0.41 | 0.67 | 0.76 | 0.48 |
| maxpools2 | 1.00 | 1.00 | 0.98 | 0.97 | 0.93 | 0.98 | 0.86 | 0.93 | 0.93 | 0.83 |
| pw1 | 1.00 | 0.67 | 0.81 | 0.98 | 0.78 | 1.00 | 0.86 | 0.93 | 0.93 | 0.74 |
| relu | 0.36 | 0.50 | 0.64 | 0.55 | 0.61 | 0.62 | 0.73 | 0.69 | 0.65 | 0.67 |
| selu | 0.76 | 0.50 | 0.64 | 0.81 | 0.60 | 0.79 | 0.45 | 0.54 | 0.74 | 0.48 |
| silu | 0.88 | 0.50 | 0.51 | 0.69 | 0.46 | 0.62 | 0.32 | 0.61 | 0.57 | 0.59 |
| skip1 | 0.76 | 0.67 | 0.72 | 0.86 | 0.65 | 0.77 | 0.77 | 0.76 | 0.80 | 0.55 |
| zca | 0.85 | 0.83 | 0.79 | 0.78 | 0.64 | 0.81 | 0.68 | 0.72 | 0.74 | 0.55 |

