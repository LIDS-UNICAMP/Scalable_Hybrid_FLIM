# Ponto de partida do SSL: o SPiFiL pequeno ANTES de qualquer treino (11/11/2026)

**Resposta curta:** o SPiFiL já ganha do aleatório do mesmo tamanho em todas as 8 bases, de +0,04 (remote_sensing) a +0,32 (Larvae).
É essa a régua que o SSL precisa superar.

## Como foi medido
- Os números vêm de resultados que já existem: os CSVs restr5m no jaci (`/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery/results/`).
- Leitura: `cat_tta` na última camada (L6), na validação, a 50%.
- κ e F1 são a média dos splits que já têm resultado.
- O aleatório é o `random_z` de forma idêntica, rodado na mesma célula.
- Todos os modelos recebem a imagem em 200 px, com pools 2-2-2-2-1-1.
- O EfficientNet de 104 px e o dwr ficaram de fora, porque mudam a resolução.

## Tabela (50%, val, cat_tta L6)

| base | forma (`w`, `k`) | params | método | splits | κ SPiFiL | F1 | κ random_z | F1 rz | ganho κ |
|---|---|---|---|---|---|---|---|---|---|
| aid | 216×6, k3-3-3-5-5-5 | 4,34 M | fps | 3 | 0,827 | 0,834 | 0,762 | 0,771 | +0,065 |
| remote_sensing | 40-80-152-152-152-312, k3 | 0,98 M | fps | 3 | 0,546 | 0,533 | 0,508 | 0,483 | +0,038 |
| coconut_trees | 104×6, k3-3-3-5-5-5 | 1,01 M | fps | 3 | 0,760 | 0,880 | 0,710 | 0,855 | +0,051 |
| flowers ⚠ | 104×6, k3-3-3-5-5-5 | 1,01 M | fpsw@mapa_w0.5_1 | 1 | 0,665 | 0,736 | 0,528 | 0,624 | +0,137 |
| Eggs ⚠ | 56-112-216-216-216-440, k3 | 1,97 M | fpsw@mapa_w0.5_1 | 1 | 0,911 | 0,910 | 0,735 | 0,806 | +0,176 |
| Protozoan ⚠ | 128×6, k5 | 2,06 M | fpsw@mapa_w0.5_1 | 1 | 0,867 | 0,851 | 0,611 | 0,609 | +0,256 |
| Larvae ⚠ | 104×6, k3-3-3-5-5-5 | 1,01 M | fpsw@mapa_w0.5_1 | 1 | 0,933 | 0,967 | 0,614 | 0,807 | +0,319 |
| ham10000 ⚠ | 80-168-328-328-328-656, k3 | 4,49 M | fpsw@mapa_w0.5_1 | 1 | 0,506 | 0,549 | 0,415 | 0,443 | +0,092 |

⚠ = **escolha provisória**: só o split 1 existe. As outras leis dessas bases ainda estão rodando no restr5m.

### Como cada forma foi escolhida
A regra: pegar o melhor modelo ≤5 M com entrada de 200 px. Se houver um menor a até 0,01 dele, fica o menor. A comparação usa o split 1, o único que todas as leis já têm.
- **Eggs:** o melhor é 80-168-328-328-328-656 (4,49 M, κ 0,921). O de 1,97 M ficou a 0,00999 dele, então fica o de 1,97 M.
- **Protozoan:** o melhor é 6×192 k5 (4,62 M, κ 0,873), e o 6×128 k5 (2,06 M) ficou a 0,006 dele. O 6×88 k5 (0,98 M) não entrou na comparação, porque só tem o split 2.
- **Larvae:** a RF 6×104 (1,01 M) é a melhor de todas, com folga de 0,046.
- **ham10000:** nenhuma forma menor fica a até 0,01 do melhor (a 312×6 fica a 0,0115), então fica o de 4,49 M.
- **Método no ham10000:** o pedido era fps. Só que, a 200 px, o restr5m só mediu o fpsw. Por isso o YAML monta os dois braços, cada um no seu diretório, e a tabela mostra o fpsw.
- **flowers:** a escolha continua provisória, como já tinha sido combinado.

## Citações (CSV:linha, todos em `/dados/home/moliveira/Scalable_Hybrid_FLIM_discovery/results/` no jaci)
Uma linha por split. A coluna `source_csv_line` de `resultados.csv` traz o caminho completo.
- aid fps: `svm_disc_md_restr5m_k3_cena_aid_s0200_split{1,2,3}_pct50_sh1_leidepth6_k3-3-3-5-5-5_leiW_w216-216-216-216-216-216.csv:44`; random_z: o mesmo arquivo com `_sh0`, também na linha 44.
- remote_sensing fps: `svm_disc_md_restr5m_k3_cena_remote_sensing_s0200_split{1,2,3}_pct50_sh1_leidepth6_leiW_w40-80-152-152-152-312.csv:44`; random_z: `_sh0`, linha 44.
- coconut_trees fps: `svm_disc_md_restr5m_k3_cena_coconut_trees_s0200_split{1,2,3}_pct50_sh1_leidepth6_k3-3-3-5-5-5_leiW_w104-104-104-104-104-104.csv:44`; random_z: `_sh0`, linha 44.
- flowers fpsw: `svm_disc_md_restr5m_k3_obj_flowers_s0200_split1_pct50_sh1_leidepth6_k3-3-3-5-5-5_leiW_w104-104-104-104-104-104.csv:44`; random_z: `_sh0`, linha 44.
- Eggs fpsw: `svm_disc_md_restr5m_k3_obj_Eggs_s0200_split1_pct50_sh1_leidepth6_leiW_w56-112-216-216-216-440.csv:44`; random_z: `_sh0`, linha 44.
- Protozoan fpsw: `svm_disc_md_restr5m_k5_obj_Protozoan_s0200_split1_pct50_sh1.csv:89`; random_z: `svm_disc_md_restr5m_k5_obj_Protozoan_s0200_split1_pct50_sh0.csv:132`.
- Larvae fpsw: `svm_disc_md_restr5m_k3_obj_Larvae_s0200_split1_pct50_sh1_leidepth6_k3-3-3-5-5-5_leiW_w104-104-104-104-104-104.csv:44`; random_z: `_sh0`, linha 44.
- ham10000 fpsw: `svm_disc_md_restr5m_k3_obj_ham10000_s0200_split1_pct50_sh1_leidepth6_leiW_w80-168-328-328-328-656.csv:44`; random_z: `_sh0`, linha 44.

## 100%: ainda sem número
- Nenhum desses modelos foi medido a 100%.
- Quem vai dar esse número é o estágio L0 de cada run SSL, ou seja, as linhas do CSV dos YAMLs `experiments/autoencoder/hyb_ssl_encoder_<base>_jaci.yaml`.
- Nas 7 bases fora o HAM, a validação a 100% é a mesma da de 50%, separada do treino (conferido no `data_descriptor_perc100.json` do split 1).
- **ham10000 a 100%: a validação é igual ao treino** (5007 de 5007 imagens repetidas). Ali o κ mede o treino, e não pode ser comparado com o de 50%.

## Encoders
- Os encoders são salvos em `/dados/home/moliveira/exps/encoders/ssl_pequeno/<base>/<braço>/<base>_split<s>_pct<p>`. Os braços são o método e `random_z`; no ham10000, também `fps`.
- Todos esperam entrada de 200 px.
