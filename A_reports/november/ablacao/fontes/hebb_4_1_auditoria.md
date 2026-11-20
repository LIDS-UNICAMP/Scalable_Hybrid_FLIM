# 4.1 — Auditoria do SoftHebb (`eval/alt_encoders.py`) contra a referência

**Resposta curta:** a regra de atualização está certa — bate com a do Journé et al. 2023 até 5,6e-17 num teste com numpy. O que está diferente é a *receita* em volta dela: temperatura 4 contra 1/0,65, o passo normalizado por neurônio e não pelo tensor inteiro, a norma inicial perto de 1,4 contra ~25, e a arquitetura fixa. A calibração da BN **não** explica o resultado de A contra C a 300 passos: A, C e D usam os mesmos 300 lotes de calibração.

Repositório: `athena:/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`. Referência: código oficial `NeuromorphicComputing/SoftHebb` (branch main), baixado com curl (`hebbconv.py`, `utils.py`, `train.py`, `engine.py`, `model.py`, `layer.py`, `presets.json`, preset `3SoftHebbCnnCIFAR`). Não li o PDF do arXiv; as equações abaixo foram conferidas no código.

## 0. O que A, C e D são de verdade (`eval/alt_encoders.py:501`)

```
regime = "C" if steps else "A" if arm != "random" else "D"
```

- **A** = init SPiFiL (fps ou crus@geodesic), 0 passos de Hebb.
- **D** = init aleatório (kaiming), 0 passos → **aleatório fixo**, e não "aleatório + SoftHebb".
- **C** = qualquer init com 30, 90 ou 300 passos de Hebb.

Portanto, "aleatório com SoftHebb" é `init=random, regime=C`. A frase "D ficou abaixo do FPS fixo em 9 de 12" diz, na verdade, que o aleatório fixo perde para o FPS fixo, o que é esperado.

O "0 passos" não é uma rede sem calibração. Em `alt_encoders.py:490`, quando `steps == 0`, a função `learn_softhebb` roda mesmo assim `durations[-1] = 300` passos com `lr=0`. A BN (`momentum=None`, média cumulativa, `:184`) fica em modo train durante esses passos (`:217`, `hebb_step` chama `block.bn(h)` em `:198`). Assim, A e D calibram a BN com os **mesmos 300 lotes de 8** que C a 300 passos. C30 e C90 calibram com só 240 e 720 imagens.

Kappa médio (melhor camada; média de split × sorteio × largura; sem o braço restrito):

| base | fps A | fps C300 | rand D | rand C300 | geo A | geo C300 |
|---|---|---|---|---|---|---|
| Eggs | .752 | .740 | .732 | .726 | .692 | .747 |
| Larvae | .670 | .616 | .622 | .616 | .700 | .641 |
| aid | .676 | .662 | .695 | .664 | .608 | .652 |
| flowers | .592 | .599 | .594 | .589 | .548 | .588 |
| remote_sensing | .487 | .474 | .506 | .457 | .478 | .479 |

O padrão é este: o Hebb **puxa todos os pontos de partida para o mesmo lugar**. Ele ajuda o geodésico, que parte de mais baixo, e atrapalha fps e random, que partem de mais alto.

## 1. Comparação item a item

| Item | Nosso código | Referência | Rótulo |
|---|---|---|---|
| Regra | `hebb_step` `:195-208`: `delta = conv2d_weight(z, y) − Σ(y·u)·w`, ou seja Δw = y(x − u w) | `hebbconv.py` `delta_weight`: `yx − yu·w` | igual (a) |
| Anti-Hebb nos perdedores | `y = −softmax`; o vencedor troca de sinal (`:203-204`) | `HebbSoftKrotovConv2d.get_wta`, `mode==0`: `wta=-wta; wta[argmax]=-wta[argmax]` | igual (a) |
| Eixo da competição | softmax em `dim=1` (canais) em cada posição, sobre `u` cru, antes do triângulo | `activation(..., dim=0)` sobre `pre_x_flat` (canais × posições), `pre_triangle=False` | igual (a) |
| Softmax estável | `torch.softmax` (estável) | `torch.softmax(t_invert*x)` | igual (a) |
| Temperatura | `inv_temp: 4.0` (inversa) em **todas** as camadas (YAML) | `t_invert` 1 (L1), 0,65 (L2) e, no preset de 4 camadas, 0,25 | **difere (b)** — 4× a 16× mais "duro" |
| Normalização do Δw | `amax` **por neurônio** (`:206`) | `nc = abs(delta_weight).amax()` **global**, no tensor inteiro (`hebbconv.py` `delta_weight`) | **difere (b)** |
| lr adaptativo | `lr·sqrt(|‖w‖−1| + 1e-10)` por neurônio (`:207`) | `lr·(|‖w‖−1| + 1e-10)^power_lr`, com `lp0.5` no preset | igual (a) |
| lr e agenda | 0,05 constante, igual em todas as camadas | 0,08 (L1) e 0,005 (L2); a agenda é só o fator adaptativo | **difere (b)** |
| Init | kaiming normal, ‖w‖ ≈ √2 ≈ 1,4 (`:186`); a init SPiFiL também é normalizada para √2 (`:436`) | `normal` com range `(r²/n)^{1/2}`, r = 25 → ‖w‖ ≈ 25 (`train.py:62-80`) | **difere (b)** — partimos quase convergidos: lr efetivo ≈ 0,64·lr, que cai a zero quando ‖w‖ → 1 |
| Escala da entrada | BN sem afim antes de cada conv | `batchnorm2d` sem afim antes de cada conv | igual (a) |
| Δw sobre lote e posições | soma, sem divisão; depois normaliza pelo max | soma, sem divisão; depois normaliza pelo max | igual (a), a menos do eixo do max |
| Ordem dos blocos | BN → conv → triângulo (potência 1) → MaxPool(3, s2) (`:189-192`) | BN → conv → triângulo (0,7 / 1,4) → AvgPool(4,2,1) na L1, MaxPool(4,2,1) depois | **difere (b)** |
| Padding | zeros | `padding_mode='reflect'` (default do preset) | difere (b), menor |
| Largura e kernel | 64 ou 256 em todas as camadas, k5 em todas | 96 → 384 (→ 1536), k5 → k3 | difere (b) |
| BN no treino e na avaliação | treino em train mode com `momentum=None` (cumulativa); depois `block.eval()` (`:256`) | o bloco treinado em train mode (momentum 0,1); os outros em `eval()` (`model.py:202-223`) | igual na essência (a) |
| Camadas sequenciais | sim: uma camada por vez, as anteriores congeladas (`:216-257`) | sim: `train_unsup(blocks=[i])`, um bloco por vez | igual (a) |
| Volume de treino | 300 passos × 8 = 2.400 imagens por camada (≈1,2 época de 2.000) | 1 época de CIFAR (50k, lote 10 → 5.000 passos) | difere (b) |
| Encoder congelado na extração | `embed` é `@no_grad` e os blocos estão em eval | — | (c) — a referência usa um classificador próprio, não SVM |
| Leitura | GeM por camada → SVM | camada linear com dropout por cima | (c) |

## 2. Checagens (`/exps/moliveira/reports/rodada2/hebb_checks.py`, CPU)

Para rodar, da raiz do discovery: `PYTHONPATH=. CUDA_VISIBLE_DEVICES="" LD_LIBRARY_PATH=... python /exps/moliveira/reports/rodada2/hebb_checks.py`.

1. **Um passo contra numpy explícito** (3×2×6×6, 4 filtros 3×3): max |diff| = **5,6e-17** → a implementação é exatamente a fórmula declarada.
2. **lr = 0**: os pesos ficam **idênticos** (`torch.equal`), mas a `running_var` da BN **muda**.
3. **Pesos contra estatísticas**, com lr = 0 e pesos iguais: features após 1 passo de calibração vs após 300 diferem **1,6%**; 30 vs 300 diferem **0,3%**. Hebb a 300 passos vs só calibração a 300 passos: **26%**. Ou seja, quase todo o efeito vem do peso, não da BN.
4. Tudo finito. A norma por neurônio vai de ~1,4 para **1,00** (L1 e L2). Duas rodadas com a mesma semente dão resultados **idênticos**.
5. Na extração os blocos estão em eval, sem grad, e extrair duas vezes não muda nada (a BN não se mexe).

## 3. As diferenças explicam "FPS + SoftHebb < FPS fixo"?

- **Calibração da BN — não (c para o efeito).** A0 e C300 têm exatamente a mesma calibração (300 lotes, mesmo gerador). O efeito de 300 contra 30 lotes é de 0,3% nas features. A perda C < A vem do peso. O único confundidor é C30/C90 contra A, porque ali a calibração difere (240/720 contra 2.400 imagens) — é pequeno, mas existe.
- **Temperatura 4 e normalização por neurônio (b) — explicação provável.** Com β = 4 a competição fica quase dura. A normalização por neurônio dá a **todo** neurônio, inclusive quem quase nunca vence, um passo de tamanho máximo 1. Na referência, o max global faz o neurônio que perde sempre se mexer pouco. No nosso caso, os perdedores levam um anti-Hebb cheio e se afastam do que o SPiFiL escolheu. Isso combina com o "puxa todos para o mesmo lugar".
- **Norma inicial √2 (b) — plausível.** Ela torna o lr efetivo pequeno e logo zero, mas o pouco que se move, o max por neurônio amplifica. Na referência, a fase de r = 25 → 1 é onde os filtros se formam. Partindo do SPiFiL com ‖w‖ = √2, o Hebb só "desfaz" sem ter tempo de "formar".
- **Arquitetura (pool, triângulo, larguras, k) (b) — afeta o nível absoluto, pouco o A contra C**, porque A e C compartilham a mesma arquitetura.
- **Volume de treino (b) — fraco.** Mais passos (90 → 300) não revertem a queda do fps.

## 4. Piloto proposto (só proposta, nada lançado)

**Configuração fiel à referência** (novas subchaves de `amostra.hebb_init`): `inv_temp: [1.0, 0.65, 0.25]`, `lr: [0.08, 0.005, 0.005]`, `delta_norm: global` (amax do tensor), `triangle_power: [0.7, 1.4, 1.4]`, `pool: [avg_4_2_1, max_4_2_1, max_4_2_1]`, `padding_mode: reflect`. Para o init aleatório: `init_radius: 25`. Para os braços SPiFiL, manter √2 (é o banco de partida).

Uma chave para separar calibração de Hebb: `calib_steps` (lotes de BN com lr = 0, independente de `steps`).

**Ajuste do usuário:** o aleatório é régua pura e nunca recebe técnica. **H02 e H04 saíram.**

| id | braço | init | steps | lr | uso |
|---|---|---|---|---|---|
| H00 | aleatório fixo | random | 0 | 0 | só controle (= D atual) |
| H01 | FPS fixo | fps | 0, `calib_steps: 1` | 0 | régua sem calibração extra |
| H03 | FPS só calibração | fps | 0, `calib_steps: 300` | 0 | = A atual |
| H05 | FPS + SoftHebb fiel | fps | 300 | ref | o teste |
| H06 | geodésico + SoftHebb | crus@geodesic | 300 | atual | reaproveitar o hebb1 C300 (split 1, sorteio 42) |

Bases: Eggs, Larvae, aid, flowers, remote_sensing; split 1; sorteio 42; larguras 64 e 256; leitura `gem_pw`.

**Orçamento por braço e por camada** (imagem de 200 px; 3 camadas; lote 8):

| braço | atualizações de peso | imagens vistas | patches (L1 / L2 / L3) | passadas no pool de 2.000 |
|---|---|---|---|---|
| H00 | 0 | 2.400 (só BN, como hoje) | — | 1,2 |
| H01 | 0 | 8 | — | 0,004 |
| H03 | 0 | 2.400 | — | 1,2 |
| H05 / H06 | 300 por camada = 900 | 2.400 por camada | ≈ 9,6e7 / 2,4e7 / 5,8e6 | 1,2 por camada |

Referência para comparar: CIFAR, ~5.000 atualizações e ~5e7 patches na L1. Nosso orçamento de patches é igual ou maior; o de atualizações é 17× menor.

Total: 5 bases × 2 larguras × 3 braços novos (H01, H03, H05) = 30 células; H00 e H06 já existem no hebb1.

**YAML** (seguindo o padrão): `experiments/autoencoder/disc_md_sol_hebb3.yaml`, copiado de `disc_md_sol_hebb1.yaml`. Valores do YAML:

- `grid.datasets: [Eggs, Larvae, aid, flowers, remote_sensing]`
- `splits: [1]`
- `amostra.draws: [42]`
- `hebb_init.inits: [fps]`, `steps: [300]`, mais as subchaves acima e `calib_steps`.

O schema (`experiments/ray/schema.py:171`) já aceita `amostra` como bloco livre: **não precisa mexer no schema**. As chaves novas precisam ser lidas em `learn_softhebb`/`hebb_step` (`alt_encoders.py:195-260`) e no rótulo de config em `cells` (`:320-325`), com default igual ao comportamento de hoje. Isso fica para uma etapa futura.
