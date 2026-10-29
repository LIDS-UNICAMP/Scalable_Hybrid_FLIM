# Protozoan: grade progressiva multirresolução

72 buscas: porcentagens 1/5/25/50/75/100, splits 1/2/3 e resoluções 128/160/200/256. Configurações `protozoan_r*.yaml` e `ray_protozoan_r*.yaml`; manifesto reproduzível `manifest_protozoan.json`.

Somente treino e validação. Holdout estratificado fixo de 20% construído exclusivamente do `train` nativo de 100% de cada split, seed 42+split. Validação: 953 imagens por split. Pool restante: 3829 imagens, com subconjuntos percentuais aninhados. Os arquivos originais são preservados; não há acesso a imagens de teste. Este protocolo difere do antigo experimento Protozoan50%.

Batch16; embedding1280; SPiFiL baseado em imagem com450superpixels, um doador por classe do treino; conv3/5 convencionais sequenciais, canais7/14/21/28, pooling1/2, BN opcional, ReLU/ReLU6/SiLU, dropout0/.05/.1/.2/.3; profundidade1–10. Cinco épocas faseA e cinco faseB; LR.001/refine.0001, L1. GA: população6, até5gerações, elitismo1, mutação.8, paciência3. Jminimizado comlambda.004, cobertura mínima.8; HDBSCAN min_cluster_size15/min_samples5. Limites:2M parâmetros e500M MACs.

Execução pelo launcher Ray existente, uma busca por vez nesta fila, GPU2 compartilhada,16threadsCPU e16workerspreload/DBCV. Eggs eLarvae continuam. W&B online no projeto ophira-ai/progressive_nas_ga; grupo por resolução.

Preparar novamente (arquivos divergentes são recusados):

```bash
python -c "from ga_search.prepare_multires_grid import prepare; prepare(datasets=('protozoan-cysts',), manifest_name='manifest_protozoan.json')"
```

Executar/retomar todas as resoluções no ambiente configurado:

```bash
bash results/nas_multires/launch_protozoan.sh
```

O script verifica a branch gpt_sol_G_A, percorre as quatro configurações Ray e retoma estados compatíveis. Logs do driver: results/nas_multires/protozoan_driver/. Monitor dos três datasets: reports/nas_multires_live.md.
