# Exploração auto-supervisionada: contexto comum para os agentes (11/11/2026)

Pedido do usuário: treinar o melhor modelo **pequeno** (1 a 5 milhões de parâmetros) de cada base, **inicializado com SPiFiL**, com quatro famílias de auto-supervisão e comparar no mesmo protocolo:
1. DINOv2 (EMA, multi-crop, centralização/Sinkhorn, KoLeo; iBOT só se houver tokens de patch);
2. CosFace sem rótulo (NT-Xent com margem aditiva no positivo, s = 1/τ);
3. SimCLR (NT-Xent);
4. LeJEPA com regularizações trocáveis.

Escopo:
- Bases: aid, flowers, remote_sensing e coconut_trees, a 50% e a 100%.
- Treino no JACI primeiro; o athena só coordena e avalia.

## Onde está tudo
- Repositório de trabalho: `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, worktree do repositório principal na branch `discovery` (não é a main).
  - É aqui que estão o SPiFiL (`eval/spf_grade.py`), os modelos pequenos e o LeJEPA já ligado.
  - O caminho `/dados/home/moliveira/Scalable_Hybrid_FLIM` é a main e NÃO tem `eval/spf_grade.py`.
- Jaci tem uma cópia sem git do mesmo diretório, em `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery`, sincronizada por rsync a partir do athena.
  - O athena é a fonte única de `eval/spf_grade.py`: antes de copiar, compare com a cópia do jaci e mescle o que for diferente.
  - Os caminhos de dados e resultados no jaci ficam em `/dados/home/moliveira/exps/...`.
  - Cada rsync é registrado em `/exps/moliveira/queues/rsync_contador.log`.
- Fila do jaci: é operada pela sessão `jaci_shared`, que eu (o orquestrador) aciono. A GPU do jaci tem 98 GB, e a regra é ficar em 88–91 GB, deixando ≥4 GB para o root.
  - Ordem atual: LeJEPA na receita de 8,2 M (`lejepa_ordem_jaci.txt`), depois a grade ≤5 M e as explorações.
  - Os agentes NÃO lançam nada no jaci: só preparam os YAMLs e me dizem o que mandar.

## O que já se sabe (não refaça)
- **Modelos pequenos SPiFiL** (`A_reports/november/multi_dataset/restricao_5M.md`, auditoria em `auditoria_restr5m_spifil.md`; YAMLs `experiments/autoencoder/disc_md_restr5m_*_jaci.yaml`). Melhores de ~1 M com 3 splits:
  - remote_sensing: L4 RegNet 40-80-152-152-152-312 k3, κ 0,546;
  - coconut_trees: L6 RF 6×104 k3-3-3-5-5-5, κ 0,760;
  - aid: o melhor é de ~4,5 M, L3 ResNet 56-112-216-432-432-432 k3, κ 0,836;
  - flowers: EfficientNet 3×160 k5 a 104 px, κ 0,615. As outras leis em flowers ainda estão rodando.
- **Receita inteira + backprop**:
  - O encoder SPiFiL é montado por `spf_grade` com `amostra.salvar_encoder` (formato FLIM) e carregado no runner growth com `init_flim_dir` + `layerwise_from_flim`.
  - Treino em `methods/autoencoder/autoencoder_flim_module.py`. Opções existentes: `lejepa` (perda λ·SIGReg + (1−λ)·invariância), `head: mlp`, `num_slices`, multi-crop `global_views`, `ckpt_last` (sem rótulo: a última época), sonda κ por época só gravada, e `epochs_by_dataset` no growth.
  - O autoencoder (reconstrução) derrubou o κ (Eggs 0,77 → 0,53).
  - O LeJEPA fiel ao artigo (lr 5e-4) também derruba a sonda nas primeiras épocas (Protozoan 0,642 → 0,405 → 0,471).
  - Hipótese aberta: o lr de "treino do zero" estraga a receita, e o fine-tuning pede um lr bem menor.
- **Avaliação padrão**: SVM linear (C=100) sobre GeM por camada concatenado (cat) + PCA-whitening, com TTA de 8 vistas (gem_cat_tta / catttapw), em validação.
  - No HAM a 100% a validação é igual ao treino (`core/data/splits.py:9`). Confirme se isso também vale para as 4 bases daqui.
- **Losses em `core/losses/`**: `invariance_loss.py`, `real_sigreg.py`, `simple_sigreg.py`. Outras (StrongSIGReg, weak_sigreg, ur_jepa_reg, vicreg, vicregl, barlow_twins, corinfomax) podem estar em `methods/` ou em outro lugar: procure.

## Regras
- Sem rótulo no treino. Sem rótulo na escolha do checkpoint (última época). Rótulo só na avaliação.
- O aleatório (régua) é random_z do mesmo tamanho, sem nenhuma técnica aplicada.
- YAML no padrão do repositório (`experiments/autoencoder/*.yaml`, validado por `experiments/ray/schema.py`). Nada de flag de argparse, nada de `.sh`, nada de commit.
- Código novo só quando não existir. Opt-in, com o caminho antigo idêntico. Confira os imports antes e depois.
- Nada se perde: sem `rsync --delete`, sem sobrescrever pesos.
- Python: `LD_LIBRARY_PATH=/dados/home/moliveira/miniforge3/envs/scalable_FLIM/lib /dados/home/moliveira/miniforge3/envs/scalable_FLIM/bin/python`.
