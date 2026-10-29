# Busca real no tmux

Lançada na branch `gpt_sol_G_A`, sessão `gpt_sol_G_A_ga`, GPU física 2
(RTX A6000). Configuração copiada de `ga_search/config/progressive_nas.yaml`
para `results/nas_search/run.yaml`: protozoan-cysts, split 1, treino 5%, seed 42,
população 6, até 5 gerações, profundidade máxima 10 e cinco épocas A/B.
Os candidatos são executados sequencialmente nessa GPU.

```bash
tmux attach -t gpt_sol_G_A_ga
tail -f results/nas_search/console.log
```

`Ctrl-b d` desanexa sem encerrar a busca. O launcher está em
`results/nas_search/launch.sh`; ele verifica a branch, fixa CUDA_VISIBLE_DEVICES=2
e usa a env scalable_FLIM. Resultados/checkpoints ficam em `results/nas_search`;
`launch.json` registra GPU, configuração, data de lançamento e log.

A primeira tentativa foi preservada em `results/nas_search_failed_cuda_init`.
Ela não chegou ao treino: a leitura dos contadores CUDA acontecia antes da
inicialização do allocator em um encoder vazio. `flim/progressive.py` agora
chama `torch.cuda.init()` antes de zerar os contadores.

O teste `test_first_cuda_fit_initializes_memory_counters` passou na GPU 2,
executando ajuste SPiFiL real em um subprocesso sem CUDA previamente
inicializada. A suíte SPiFiL em CPU também foi repetida; o novo teste CUDA é
ignorado quando a GPU não está disponível.

Após a correção, a busca foi relançada em um diretório novo, preservando o
histórico anterior. Foram obtidos 3.143 patches dos sete doadores reais de
treino. Candidatos acima do limite de MACs foram rejeitados, e um candidato
viável permaneceu ativo na GPU. O treino completo e suas métricas ainda estão
em execução; este registro não declara seu resultado final.

Para retomar após interrupção, na mesma branch e com os mesmos arquivos:

```bash
bash results/nas_search/launch.sh
```

O script já passa `--resume`; não execute outra cópia enquanto o processo
anterior estiver ativo.
