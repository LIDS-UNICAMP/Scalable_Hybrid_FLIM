# Auditoria da rodada 2 no jaci (só leitura)

Data: 2026-11-08, 10:20 (-03). Fonte: YAMLs em `experiments/autoencoder/*_jaci.yaml` do worktree do jaci, logs em
`/dados/home/moliveira/exps/queues/logs/` e CSVs em `/dados/home/moliveira/exps/results*/`. Só validação; o teste não é lido.

Legenda de estado: **completa** = rc 0 e todos os jobs OK; **parcial** = rodando ou com CSVs incompletos; **na fila** = YAML
pronto, ainda não lançado; **não existe** = nenhum YAML no jaci.

| Linha | YAML no jaci | CSVs | Splits × draws | Estado | Controles (mesmo protocolo) |
|---|---|---|---|---|---|
| (a) GeM p=3 (`gem`) contra p=2 (`gem_p2`), p=4, p=6 | só `disc_md_sol_p5_extf_jaci` (food101) | nenhum | 3 × 2 | na fila (fila2) | random_z, xavier, orth (+ gêmeos) |
| (a) `ms`: GeM em 1, 0,75 e 0,5 da imagem (leitura) | só `disc_md_sol_p5_extf_jaci` (food101) | nenhum | 3 × 2 | na fila | random_z, xavier, orth |
| (a) multiescala dos pontos (scale space, s1 a s8; não é leitura) | `disc_md_ms_a2_jaci`, `disc_md_ms_b2_jaci` | 63 + 21 em `results/` | 3 × 5 | completa (rc 0) | xavier, random_z no `ms_ctl2` do athena (congelado); no jaci só o `ms_ctl_extf` (food101, na fila) |
| (a) última camada contra `cat` | todos os `spf_grade` (sol_wsal, spfcombo_w/w2, spfdeep_esp, sol_p1/p3/n6, hz, r3/r4) | readouts `L` (por camada) e `cat` em todos | conforme cada YAML | completa nos terminados | os controles de cada YAML |
| (b) diversidade funcional | `disc_md_sol_p3_jaci` (6 bases) | 36 em `results_sol/filter_diversity/` | 3 × 2 | completa (rc 0, 36/36) | random_z, xavier, orth (+ gêmeos) |
| (b) B1, B2, B3 | nenhum braço com esses nomes; o sol_p3 tem f2, f3, f4, f5, f6, kmed | — | — | não existe com esse nome | — |
| (c) SoftHebb com homeostase (hebb2: E04 ganho pela frequência, E05 só inativos) | `disc_md_sol_hebb2_jaci` (12 bases; inits random e fps; 64 e 256; 3 camadas; 300 passos) | 8 de 36 bases×split escritos em `results_sol/alternative_encoders/` (328 linhas ok, split 1) | 3 × 2 | parcial (0/36 jobs fechados; rodando desde 09:16) | nenhum dentro do YAML; o par sem homeostase é o hebb1, que não existe no jaci |
| (c) variante restrita ou por camada | não há chave para isso no hebb2 (homeo só na competição, igual nas 3 camadas) | — | — | não existe | — |
| (d) Scattering (sol_p6) | só `disc_md_sol_p6_extf_jaci` (food101): J em {2, 3, 4}, L 8, leituras avg_pw e gem_pw, regime A (fixo, só o 1º sorteio); junto, PCANet e SoftHebb | nenhum | 3 × 2 | na fila | nenhum no YAML (o aleatório D é o xavier/random_z do ms_ctl) |
| (d) concatenação FPS + Scattering | não há chave para isso em nenhum YAML do jaci | — | — | não existe | — |

Observações:
1. No jaci, GeM p, `ms` e o Scattering só existem como irmãos de food101 (`_extf`), ainda na fila. As versões de 12 bases rodam no athena.
2. "Última camada contra `cat`" não precisa de experimento novo: todo CSV do `spf_grade` tem `readout = L` por camada e `cat`.
3. hebb2 escreve as colunas `encoder`, `regime`, `config` (por exemplo `fps_w256_he04`), `init`, `width`, `layer` e `readout` (`gem_pw`, `cat_gem_pw`), além das medidas de geometria (cos com o init, erank, procrustes).
