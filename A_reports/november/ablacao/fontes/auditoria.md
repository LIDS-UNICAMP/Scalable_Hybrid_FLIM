# Auditoria da rodada 2 (08/11/2026)

## Leitura e controles aleatórios (athena)
- GeM: concluída. `eval/spf_grade.py:1010-1011`, p=3 por padrão; `gem_p2`, `gem_p4`, `gem_p6` rodaram no `sol_p5` (`/exps/moliveira/results_sol/pooling/`, 3 splits, draws 42/43, larguras 64/256). A0 e A1 já existem: reaproveitar.
- Multiescala: parcial. `ms` = GeM nas escalas 1, 0,75 e 0,5 (`:160`, `:1067-1069`), 3 vezes a dimensão de `L`; `cat_ms` empilha camadas a partir da última até 2000-d (`:1229-1238`). Dimensões medianas: L 155, L_ms 219, cat 219, cat_ms 576. Falta uma versão com dimensão igual à referência (A2 controlado). PCA só quando ≥ 2000-d, ajustada no treino.
- Última camada contra `cat`: ambas em todos os CSVs do `sol_p5`.
- Aleatórios: random_z, xavier e orth com o mesmo protocolo do `fps` no `sol_p5`; tnorm, he e whrand só em outras grades.
- Métricas: `val_kappa`, `val_f1` (macro a confirmar), `val_acc`; NÃO há acurácia balanceada nem métrica por classe (só impureza contra parasito). Lacuna para as entregas.
- `sol_p5` sem remote_sensing.
- κ de validação (média ± desvio de 3 splits; largura e camada pelo melhor κ médio, portanto otimista):

| base | L fps / aleatório | cat fps / aleatório | cat_ms fps / aleatório |
|---|---|---|---|
| Eggs | .764±.014 / .728±.025 | .804±.007 / .771±.024 | .816±.015 / .786±.023 |
| Protozoan | .774±.018 / .696±.023 | .813±.013 / .728±.006 | .850±.010 / .765±.007 |
| Larvae | .586±.022 / .486±.064 | .698±.035 / .496±.068 | .685±.089 / .458±.043 |
| aid | .673±.007 / .668±.015 | .704±.005 / .715±.008 | .729±.010 / .723±.008 |
| flowers | .596±.020 / .582±.035 | .614±.043 / .614±.029 | .610±.041 / .631±.026 |

## Diversidade funcional, SoftHebb e Scattering (athena)
- Diversidade: implementada em `functional()`, `eval/spf_grade.py:878-905`.
  - Variantes: f2 = B1 (FPS em 1 − correlação); f3 = B2 (guloso de baixa redundância); f4 = cobertura (facility location); f5 = B3 (distância visual + funcional); f6 = f3 + estabilidade.
  - Mesmos candidatos e mesma largura para todos.
  - Resposta: relu(u·z(x)), com o z-score comum das janelas de treino; as colunas são centradas e com norma 1 (`:866`, `:873`). Os candidatos que nunca disparam saem.
  - **A grade principal do `sol_p3` não tem CSV no athena** (conferir no jaci). Só existe coconut_trees, split 1, 64 filtros: fps 0,743 contra f2 0,732. f3 a f6 não têm resultado.
  - Faltam os diagnósticos de cobertura, fração inativa e estabilidade.
- SoftHebb `sol_hebb1`: completo (12 bases × 3 splits × draws 42/43).
  - Regimes: A = início SPiFiL com 0 passos (só calibra a BN); C = SoftHebb por 30, 90 ou 300 passos; D = aleatório com 0 passos.
  - Já existem: atualização curta (30 passos), restrita (bola de raio 0,5·|w0|, `alt_encoders.py:233-235`), treino sequencial por camada, e filtros salvos antes e depois (`*_init.npy`, `*_wins.npy`).
  - Não existem: atualizar uma camada escolhida, nem recuperar filtros inativos ou redundantes.
  - Homeostase (`sol_hebb2`, E04 e E05): só piloto, 6 bases, split 1.
  - fps L3, 256 filtros: A 0,732; C30 0,708; C90 0,703; C300 0,704; restrita 0,721.
  - Com 300 passos: cos_init 0,68 a 0,80; o erank cai de 97–121 para 55–72.
- Scattering `sol_p6`: completo.
  - Morlet via FFT, ordens 0 a 2, J de 2 a 4, L = 8; dimensões 243, 651 e 1251.
  - Melhor: J4 com gem_pw, 0,719, contra SPiFiL fps 0,732.
  - Não existe concatenação FPS + Scattering (D2 ausente).

## Jaci (`/dados/home/moliveira/exps/reports/auditoria_rodada2_jaci.md`)
- **`sol_p3` completo no jaci**: 36 de 36 CSVs em `results_sol/filter_diversity`, 6 bases × 3 splits × draws 42/43, com random_z, xavier e orth. Os braços f2 a f6 correspondem a B1 a B3; no athena só existe o coconut. Precisa do rsync de volta.
- `ms_a2` e `ms_b2` (pontos em várias escalas): completos, 3 splits × 5 sorteios.
- `sol_hebb2` (homeostase): rodando, parcial; no jaci não há controle sem homeostase (o hebb1 está no athena).
- GeM p, `ms` e Scattering no jaci: só os irmãos de food101, ainda na fila.
- Não existem em lugar nenhum: SoftHebb restrito por camada, recuperação de filtros, FPS + Scattering.

## Seção 3 (diversidade funcional): resultado com o que já existia (`sol_p3`, jaci; relatório em `/dados/home/moliveira/exps/reports/rodada2_sol_p3.md`)
Diferença pareada f − fps em 72 células (6 bases × 2 larguras × 3 splits × 2 sorteios), L3 `cat`:
- B1 (f2): −0,001 ± 0,050
- B2 (f3): +0,001 ± 0,057
- cobertura (f4): −0,023 ± 0,044
- B3 (f5): −0,004 ± 0,050
- f6: −0,051 ± 0,090

F1: f3 +0,007 e f5 +0,004 (f5 vence em F1 em 5 de 6 bases com 256). Sinais locais: aid com f3 +0,017 e +0,028, desvio baixo; Eggs_subgroup com 64 (f2 +0,050). Larvae_subgroup com f6 desaba (−0,17 e −0,24).
Diagnóstico: f3 reduz a redundância (cos_mean 0,101 contra 0,137 do fps, com 64) sem perder erank; f4 e f6 perdem 30 a 45% do erank e são os que mais perdem κ.
**Decisão: nenhuma variante de diversidade funcional melhora o FPS de forma consistente com o mesmo orçamento. Linha encerrada como receita geral; f3 em aid fica como sinal exploratório.** Não precisa rodar de novo.

## Integridade dos CSVs (athena: 1024 + 309 arquivos, 144.398 linhas de SVM, 35 famílias)
- Métricas (`core/metrics.py:46-76`, `eval/spf_grade.py:955-964`):
  - `val_kappa` = Cohen não ponderado;
  - `val_f1` = F1 macro;
  - **`val_acc` = acurácia balanceada** (macro; corrige a nota acima);
  - `val_accuracy` = taxa de acerto simples.
  - Não há métrica por classe. O teste nunca é carregado (`spf_grade.py:72`); só a família l1fast tem colunas de teste, e nenhuma análise as lê.
- Duplicatas: 896 grupos (971 linhas), todos repetições de shard da mesma célula, com valores iguais (diferença máxima de 6,7e-6). Deduplicar pela chave completa (com candidates, images, bg_src, bg_sel, k, widths, image_size, n_superpixels).
- Erros: 1 linha (amostra_banco, bandwidth); 3 CSVs vazios de coconut.
- Splits faltando: l1fast random (só o split 1), porclasse Protozoan_subgroup (split 2 em 2 braços), sol_p2 flowers (split 3, ainda rodando).
- Protocolo: pct 50 e imagem de 200 px em quase tudo; superpixels 50 (400 no cl_*); as leituras variam por família, então só comparar dentro da mesma família.
- **Splits sobrepostos:** os 3 splits são reembaralhamentos do mesmo conjunto. Os treinos compartilham cerca de 70% das imagens (Jaccard 0,54) e as validações compartilham de 4 a 13%. Os 3 splits NÃO são repetições independentes: a incerteza deve ser lida como variabilidade de reamostragem, sem testes que assumam independência; a família de bases é a unidade independente.
