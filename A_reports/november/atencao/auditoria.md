# Atenção sem parâmetros no SPiFiL: auditoria (10/11/2026)

Só leitura do código; nada rodado.

## Onde inserir (`eval/spf_grade.py`)
- Ponto único: anexar `Attn()` ao `Sequential conv{k}` depois do build (1330-1333), no mesmo molde do laço `ACTS` que troca o ReLU. Depois do ReLU; antes ou depois do pool.
- Modo (b), construção: `patch_pool(lab, seeds, encoder, layer)` (1270/1282) já roda o encoder montado, então a camada seguinte recorta kernels do mapa atendido sem código extra. O whitening `fold` passa a ser estimado sobre esse mapa (1274/1284/1317).
- Modo (a), só inferência: constrói sem atenção e liga a atenção numa cópia, na leitura (1336-1343).
- Só no mapa final: `readouts.maps` (1084) / `features`.

## Custo (200 px, pools 2-2-2-2-1-1, 256 canais, fp32)
| camada | lado | N tokens | mapa/imagem | matriz N×N |
|---|---|---|---|---|
| L1 | 100² | 10.000 | 10 MB | 400 MB, ~50 GFLOP (inviável cheia) |
| L2 | 50² | 2.500 | 2,5 MB | 25 MB |
| L3 | 25² | 625 | 0,6 MB | 1,6 MB |
| L4-L6 | 12² | 144 | 0,15 MB | desprezível |
Atenção espacial, SimAM e multiescala: O(N·C), quase grátis. Self-attention e non-local em L1/L2: só com janela 8×8 ou chaves resumidas (pool 4×).

## Plano
- Chave `amostra.atencao: {mecanismo, camadas, modo: a|b, alpha, janela}`, validada no schema como `check_encdeep` (schema.py:133).
- `ATTN = {nome: fn}` ao lado de `ACTS` (1036); só operações do torch.
- Modo (a): linhas pareadas `L_att`/`cat_att`, como no TTA (1341), e ajuste do `expected` (1481-1486). Modo (b): braço separado (sufixo em `tails`, 1406).
- random_z recebe a mesma atenção, pelo mesmo encoder.

## Riscos
- Modo (a) no meio da rede: o whitening da conv seguinte foi estimado sem atenção, e o resultado sai enviesado contra a atenção. Modo (a) só na última camada ou no final; no meio da rede, usar o modo (b).
- O assert `trunk_hash` do encdeep (1324) falha se a atenção mudar o tronco.
- SPP já perdeu ~0,05 (`ablacao_completa.md:118`): fica fora da triagem.
- A softmax de XXᵀ/√d satura em mapa pós-ReLU: normalizar os canais antes.

## Recursos
O teto é 25 GB nossos por GPU no athena; a curva com backprop usa ~22 GB por GPU, então sobram ~3 GB. Cabem já: espacial, SimAM e self-attention/non-local de L3 até o final. Esperam: L1/L2 cheias, o modo (b) inteiro e o jaci.
