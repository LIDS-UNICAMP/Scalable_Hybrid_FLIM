# pf_res320_conf (Jaci) — Largura, profundidade, kernel e tamanho da entrada

Índice: [`ablacao_completa.md`](../multi_dataset/ablacao_completa.md). Só validação; κ = Cohen, F1 = macro, acc = acurácia balanceada. Os splits se sobrepõem: a família é a unidade independente.

- **Experimento / YAML:** `disc_md_pf_res320_conf_jaci.yaml` (`svm_disc_md_pf_res320_conf`, runner `eval`)
- **Alternativa:** entrada 320 px, sem saliência (`fps`)
- **O que testa:** confirmação do piloto `pf_res320_off` (split 1, Athena) nos splits 2 e 3
- **Como:** receita `sol_wsal` (L6 256, entre_camadas, `cat`), 50 superpixels em toda resolução; âncora 200 px pareada = célula lei A w0 256 r1.0 de `disc_md_pf_leis_off`; controles aleatórios puros; splits 2 e 3 × sorteios 42 e 43; 8 jobs
- **Bases:** aid, rs
- **Resultados:** — (sem número ainda)
- **Estado:** na fila (Jaci, na frente da fila; entra quando a GPU baixar de 77 GB; imagens de 320 px enviadas, 3,3 GB)
- **Decisão:** —

## Fontes
- comentário do topo de `experiments/autoencoder/disc_md_pf_res320_conf_jaci.yaml`
- histórico da conversa (sessão 990a33ea, mensagens do assistente), 08/11 21:12–21:13 UTC
- [`pool_foco.md`](fontes/pool_foco.md)
