# growth_mem: estado da memória do B1 (RS/flowers)

Diff: `growth_mem.diff` (md5 em `growth_mem.diff.md5`). Aplicar com `cd /dados/home/moliveira/exps/growth/code && patch -p1 < growth_mem.diff`.

## Pico estimado por fase (RS B1, B0 = c00306: conv3 210ch + max, dwsep7 210ch batch_dbn + avg; 224 px, B=64, V=4 => 256 imgs)

| fase | antes | depois | como |
|---|---|---|---|
| grow SPiFiL do bloco novo (N=5600, D=27) | 2 -> 28+ GB em 4 s (log) | encode completo cortado; N×N só ~125 MB, e o scorer em blocos entra só se der OOM | `_SkipEncode` + `spifil_oom` (portados do rs_patched); cópia do prefixo liberada |
| AE local | baixo | igual (+ empty_cache) | — |
| SSL LeJEPA (fp32, proxy: bytes salvos p/ backward, ×256/4) | 65 GB (210ch), 86 GB (280ch) | 5,6 GB / 6,8 GB | checkpoint por bloco; BN/DBN com momentum 0 e buffers restaurados na recomputação |
| WiSE (5 alphas) | 2 state_dicts na GPU | em CPU (modelo de ~0,2 M params: impacto pequeno) | `wise_on_cpu` |
| embed SVM (eval, B=64) | DBN: ~5 cópias da ativação | ~2 cópias | DBN eval por grupo no buffer de saída, afim in-place sem autograd |

O proxy conta tensores salvos (inclusive storages compartilhados), então superestima o absoluto. Em produção (bf16) o observado foi ~29–36 GB. A razão sem/com ckpt (~10x) é o número que importa.

## Equivalência (CPU, bit a bit, torch.equal)
- DBN eval novo == antigo (C=16, 37, 210; com e sem grad).
- SSL com checkpoint: saída, gradientes e TODO o state_dict (running_mean/var, running_W, num_batches_tracked) idênticos.
- Blend WiSE com âncora em CPU == GPU/original; nomes de parâmetros (ElementProtect) inalterados.
- tests: test_growth_mem 8/8, test_growth_chain 26/26 (+ e2e), test_spifil_oom 22 + 1 skip.

## Opções DESLIGADAS (mudam ou podem mudar a ciência), em `mem:` do growth config
- `embed_batch_size: null`: micro-lote da extração da SVM (kernel cuDNN diferente -> arredondamento).
- `ssl_amp_bf16: false`: SSL em bf16.
Ligados: `grad_checkpoint`, `wise_on_cpu`, `empty_cache`. O resultado de cada nó grava `mem`.
O OOM agora loga `str(exc)[:300]`.

Não validado na GPU: a Jaci estava com 95/98 GB ocupados. Na GPU a recomputação usa os mesmos kernels; o não determinismo do backward do cuDNN existe com ou sem o patch.

## growth_mem2 (17/11): o pico NÃO está no grow do SPiFiL
- Repro em CPU do realize_plan do B1 (c00306, widen 0/1 -> 280, 56 imagens, N=5600 D=27) com TorchDispatchMode: nenhum tensor > 0,3 GB; o grow termina em 3,3 s. Depois do print "[grow] N=5600 D=27" nada mais imprime até o AE local do bloco 0 (224 px, batch 64, fp32).
- A conta fecha com os 4,50 GiB pedidos: 64 × W × 224² × 4 B = 4,50 GiB para W ≈ 376. Isso é UMA ativação do bloco 0 alargado no AE (conv -> BN -> ReLU -> max).
- AE bloco 0, W=376, medido em CPU (pico RSS, escalado para B=64): ~23 GB sem checkpoint, ~22 GB com checkpoint. O checkpoint do bloco no AE (growth_mem2) é bit a bit igual (grads e state iguais), mas ganha pouco: num bloco único, a recomputação refaz as mesmas ativações.
- CORREÇÃO da tabela acima: os 5,6/6,8 GB do SSL com checkpoint contam só o que fica salvo para o backward, não a recomputação de um bloco. No SSL (256 imagens) a recomputação do bloco 0 com W ~ 376 custa ~4 × 19 GB. Em fp32 com batch 64 × 4 vistas, um bloco 0 largo em 224 px não cabe em 29 GB sem mudar a ciência.
- O que resta, tudo mudando a ciência (decisão do coordenador): train.amp_dtype=bf16, batch ou vistas menores no AE/SSL, ou limitar a largura do bloco 0 alargado.

## growth_mem3 (17/11): SSL global, bloco 0 em resolução cheia
- Pilha do athena: backward do SSL, pede 9,00 GiB = 256 × 376 × 224² × 2 B (bf16). É um tensor do bloco 0 alargado (conv3 -> BN -> SELU -> MaxPool3/2) durante a recomputação do checkpoint. A cabeça (1×1 210->640 em 56²) custa ~2 GB, não é ela.
- Na recomputação do bloco 0 ficavam juntos: saída da conv, saída do BN, saída da SELU, índices int64 do MaxPool e a saída do pool, ou seja ~4,7 tensores cheios.
- Correção bit a bit: `_ActPool` (ativação sem parâmetros + skip identidade + pool) em micro-lotes de amostras (`ACTPOOL_CHUNK=16`). Salva só y (saída do BN) e recomputa act/pool por fatia no backward. O grad sai com o mesmo layout que o autograd daria, o que importa para a soma do afim do DBN.
- CPU, genoma real (c00306, widen 0/1 -> 376), 224 px, fp32, escalado para 256 imagens:

| | pico |
|---|---|
| sem checkpoint | 112 GB |
| growth_mem (checkpoint) | 90 GB |
| growth_mem3 (checkpoint + _ActPool, fatia = 1/16 do lote) | 70 GB |

  Em bf16 no athena (~48 GB pedidos com growth_mem) a estimativa fica em ~37 GB.
- Bit a bit (torch.equal): embedding, todos os gradientes e o state_dict. Testado em 64 px com o genoma real e em 224 px com 16 imagens, e em `test_actpool_fused_bitwise` (fatias 1, 3 e o lote inteiro).
