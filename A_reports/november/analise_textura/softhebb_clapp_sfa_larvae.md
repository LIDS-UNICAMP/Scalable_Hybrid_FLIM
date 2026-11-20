# SoftHebb, CLAPP e SFA na Larvae: antes e depois (10/11/2026)

Material didático para a apresentação. Energia = fração da ativação (média dos canais) dentro da máscara real, 30 imagens de validação, sem o preenchimento.

| técnica | onde atua | κ sem → com | energia na larva antes → depois | figura |
|---|---|---|---|---|
| SoftHebb (receita dos autores, rede de 3 camadas, 256) | L1–L3 | 0,679 → 0,732 (15 runs; `ablacao/12_sol_hebb3_conf.md:10`) | L1 31→36%, L2 32→36%, L3 28→30% | `softhebb_antes_depois_larvae.png` |
| CLAPP (L4/L5 da receita) | L4, L5 | cat 0,771 → 0,771; L4 0,812→0,755; L5 0,785→0,816 (1 célula, split 1) | L4 23,8→23,4%, L5 15,5→15,4% | `clapp_antes_depois_larvae.png` |
| SFA (L4/L5 da receita, 5×256) | L4, L5 | cat 0,848 → 0,818 (split 1) | L4 33,1→27,6%, L5 31,8→26,9% | `sfa_antes_depois_larvae.png` |

Leitura: o SoftHebb concentra a ativação no objeto e sobe o κ; o CLAPP é neutro (os pesos mudam 2,6%); o SFA guarda o que não muda com a cor (o fundo liso) e piora.
Equações com arquivo:linha: SoftHebb `eval/alt_encoders.py:201-211`; CLAPP `flim/local_predictive.py:63-69,87-89`; SFA `flim/slow_features.py:61-91`.
CSVs das células novas no scratchpad da sessão (`clapp/res.csv`, `sfa/larvae.csv`).
